"""Gymnasium environment wrapping the hospital discrete-event simulator.

The environment is deliberately thin: sim.py remains the source of hospital
physics; this file exposes an RL state/action/reward interface around it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import json
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim import Config, Hospital, arrival_rate, surge_multiplier
from rl_policy import RLAction, RLPolicy
from hospital_forecaster import HospitalDemandForecaster


SCENARIOS = {
    "normal": [],
    "flu": [(48.0, 96.0, 1.6)],
    "mass_casualty": [(48.0, 54.0, 4.0)],
}


@dataclass
class RewardWeights:
    """Scalarisation used by PPO; all weights are explicit and tunable."""

    critical_queue: float = 4.0
    critical_wait: float = 0.75
    ed_queue: float = 0.50
    boarding: float = 2.0
    boarding_gt4: float = 3.0
    lwbs_event: float = 10.0
    elective_cancel_event: float = 5.0
    ot_block_hour: float = 2.0
    surge_bed_hour: float = 0.25
    early_discharge: float = 1.0


class HospitalRLEnv(gym.Env):
    """RL environment for adaptive hospital resource allocation.

    Action space (Discrete):
        Nine interpretable intervention bundles defined in rl_policy.ACTION_TABLE.

    Observation is a compact vector of normalized operational state variables.
    """

    metadata = {"render_modes": []}

    # Names are public so saved traces/experiments can remain interpretable.
    BASE_OBS_NAMES = [
        "ed_utilization",
        "ed_queue",
        "ed_mean_wait",
        "critical_queue",
        "critical_mean_wait",
        "critical_oldest_wait",
        "boarding",
        "boarding_gt4h",
        "ward_utilization",
        "ward_queue",
        "icu_utilization",
        "icu_queue",
        "ot_utilization",
        "ct_utilization",
        "ct_down",
        "recent_arrivals",
        "recent_discharges",
        "recent_lwbs",
        "recent_elective_cancels",
        "elective_queue",
        "recent_ot_block_h",
        "surge_beds_active",
        "demand_multiplier",
        "time_sin",
        "time_cos",
        "dow_sin",
        "dow_cos",
    ]

    FORECAST_MEAN_NAMES = [
        *(f"forecast_arrivals_t+{h}" for h in range(1, 7)),
        *(f"forecast_critical_arrivals_t+{h}" for h in range(1, 7)),
    ]
    FORECAST_STD_NAMES = [
        *(f"forecast_arrivals_std_t+{h}" for h in range(1, 7)),
        *(f"forecast_critical_arrivals_std_t+{h}" for h in range(1, 7)),
    ]
    OBS_NAMES = BASE_OBS_NAMES

    def __init__(
        self,
        *,
        duration_h: float = 168.0,
        warmup_h: float = 24.0,
        decision_interval_h: float = 1.0,
        scenario_mode: str = "mixed",
        fixed_scenario: Optional[str] = None,
        base_config: Optional[dict[str, Any]] = None,
        reward_weights: Optional[RewardWeights] = None,
        trace: bool = False,
        forecaster_path: Optional[str] = None,
        forecaster: Optional[HospitalDemandForecaster] = None,
        include_forecast_uncertainty: bool = True,
    ) -> None:
        super().__init__()

        if duration_h <= warmup_h:
            raise ValueError("duration_h must be greater than warmup_h")
        if decision_interval_h <= 0:
            raise ValueError("decision_interval_h must be positive")
        if scenario_mode not in {"mixed", "normal", "stress"}:
            raise ValueError("scenario_mode must be mixed, normal, or stress")
        if fixed_scenario is not None and fixed_scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {fixed_scenario!r}")

        self.duration_h = float(duration_h)
        self.warmup_h = float(warmup_h)
        self.decision_interval_h = float(decision_interval_h)
        self.scenario_mode = scenario_mode
        self.fixed_scenario = fixed_scenario
        self.base_config = dict(base_config or {})
        self.reward_weights = reward_weights or RewardWeights()
        self.collect_trace = bool(trace)
        self.include_forecast_uncertainty = bool(include_forecast_uncertainty)

        if forecaster is not None and forecaster_path is not None:
            raise ValueError("provide either forecaster or forecaster_path, not both")
        self.forecaster = forecaster
        if self.forecaster is None and forecaster_path is not None:
            self.forecaster = HospitalDemandForecaster.load(forecaster_path)
        self.predictive = self.forecaster is not None

        self.obs_names = list(self.BASE_OBS_NAMES)
        if self.predictive:
            if self.forecaster.config.horizon_h != 6:
                raise ValueError(
                    "This environment expects a 6-hour forecaster horizon; "
                    f"loaded horizon={self.forecaster.config.horizon_h}"
                )
            self.obs_names.extend(self.FORECAST_MEAN_NAMES)
            if self.include_forecast_uncertainty:
                self.obs_names.extend(self.FORECAST_STD_NAMES)

        from rl_policy import ACTION_TABLE
        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.obs_names),),
            dtype=np.float32,
        )

        self.hospital: Optional[Hospital] = None
        self.policy: Optional[RLPolicy] = None
        self.scenario_name: Optional[str] = None
        self.elapsed_decisions = 0
        self.trace_rows: list[dict[str, Any]] = []
        self._prev_event_index = 0
        self._cached_forecast: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Scenario and simulator construction
    # ------------------------------------------------------------------
    def _choose_scenario(self, options: dict[str, Any]) -> str:
        requested = options.get("scenario") if options else None
        if requested is not None:
            if requested not in SCENARIOS:
                raise ValueError(f"unknown scenario {requested!r}")
            return requested

        if self.fixed_scenario is not None:
            return self.fixed_scenario

        if self.scenario_mode == "normal":
            return "normal"
        if self.scenario_mode == "stress":
            return str(self.np_random.choice(["flu", "mass_casualty"]))
        return str(self.np_random.choice(list(SCENARIOS)))

    def _make_config(self, seed: int, scenario: str) -> Config:
        kwargs = dict(self.base_config)
        kwargs["seed"] = int(seed)
        kwargs["surges"] = list(SCENARIOS[scenario])
        return Config(**kwargs)

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _queue_stats(self):
        assert self.hospital is not None
        now = self.hospital.env.now
        waiting = list(self.hospital.ed.waiting)
        waits = np.asarray([now - item[2] for item in waiting], dtype=float)
        critical_waits = np.asarray(
            [now - item[2] for item in waiting if item[0].acuity <= 2],
            dtype=float,
        )
        return waits, critical_waits

    def _boarding_gt4(self) -> int:
        assert self.hospital is not None
        now = self.hospital.env.now
        by_id = {p.pid: p for p in self.hospital.patients}
        return sum(
            1
            for pid in self.hospital.boarding
            if pid in by_id
            and math.isfinite(by_id[pid].t_ed_done)
            and now - by_id[pid].t_ed_done > 4.0
        )

    def _recent_events(self, start_t: float, end_t: float) -> dict[str, float]:
        assert self.hospital is not None
        events = [
            e for e in self.hospital.events
            if start_t < e.get("t", -1.0) <= end_t
        ]
        out = {
            "arrivals": 0.0,
            "discharges": 0.0,
            "lwbs": 0.0,
            "elective_cancels": 0.0,
            "ot_block_h": 0.0,
            "early_discharges": 0.0,
        }
        for e in events:
            typ = e.get("type")
            if typ == "arrival":
                out["arrivals"] += 1.0
            elif typ in {"discharge_ed", "inpatient_discharge"}:
                out["discharges"] += 1.0
                if e.get("early"):
                    out["early_discharges"] += 1.0
            elif typ == "lwbs":
                out["lwbs"] += 1.0
            elif typ == "elective_cancelled":
                out["elective_cancels"] += 1.0
            elif typ == "surgery_end":
                out["ot_block_h"] += float(e.get("ot_block_h", 0.0) or 0.0)
        return out

    def _state_dict(self) -> dict[str, float]:
        assert self.hospital is not None
        h = self.hospital
        waits, critical_waits = self._queue_stats()
        ed = h.ed
        ward = h.ward
        icu = h.icu
        ot = h.ot
        ct = h.ct

        recent = self._recent_events(
            max(self.warmup_h, h.env.now - self.decision_interval_h),
            h.env.now,
        )

        elective_queue = sum(
            1
            for c in h.electives
            if c.t_arrive >= self.warmup_h
            and not c.cancelled
            and not math.isfinite(c.t_inpt_bed)
        )

        max_ed_wait = float(waits.max()) if waits.size else 0.0
        mean_ed_wait = float(waits.mean()) if waits.size else 0.0
        mean_critical_wait = (
            float(critical_waits.mean()) if critical_waits.size else 0.0
        )
        oldest_critical_wait = (
            float(critical_waits.max()) if critical_waits.size else 0.0
        )

        hour = h.env.now % 24.0
        dow = int(h.env.now // 24.0) % 7
        daily_phase = 2.0 * math.pi * hour / 24.0
        dow_phase = 2.0 * math.pi * dow / 7.0

        dmult = surge_multiplier(h.cfg, h.env.now)

        # Normalizers are deliberately explicit rather than data-dependent.
        state = {
            "ed_utilization": self._clip01(ed.used / max(1, ed.cap)),
            "ed_queue": self._clip01(len(ed.waiting) / 30.0),
            "ed_mean_wait": self._clip01(mean_ed_wait / 12.0),
            "critical_queue": self._clip01(len(critical_waits) / 15.0),
            "critical_mean_wait": self._clip01(mean_critical_wait / 12.0),
            "critical_oldest_wait": self._clip01(oldest_critical_wait / 24.0),
            "boarding": self._clip01(len(h.boarding) / 20.0),
            "boarding_gt4h": self._clip01(self._boarding_gt4() / 15.0),
            "ward_utilization": self._clip01(ward.used / max(1, ward.cap)),
            "ward_queue": self._clip01(len(ward.waiting) / 20.0),
            "icu_utilization": self._clip01(icu.used / max(1, icu.cap)),
            "icu_queue": self._clip01(len(icu.waiting) / 10.0),
            "ot_utilization": self._clip01(ot.used / max(1, ot.cap)),
            "ct_utilization": self._clip01(ct.used / max(1, ct.cap)),
            "ct_down": self._clip01(ct.down / max(1, h.cfg.ct_scanners)),
            "recent_arrivals": self._clip01(recent["arrivals"] / 15.0),
            "recent_discharges": self._clip01(recent["discharges"] / 15.0),
            "recent_lwbs": self._clip01(recent["lwbs"] / 10.0),
            "recent_elective_cancels": self._clip01(recent["elective_cancels"] / 5.0),
            "elective_queue": self._clip01(elective_queue / 20.0),
            "recent_ot_block_h": self._clip01(recent["ot_block_h"] / 6.0),
            "surge_beds_active": self._clip01(ed.surge / max(1, h.cfg.max_surge_beds)),
            "demand_multiplier": self._clip01(dmult / 4.0),
            "time_sin": 0.5 + 0.5 * math.sin(daily_phase),
            "time_cos": 0.5 + 0.5 * math.cos(daily_phase),
            "dow_sin": 0.5 + 0.5 * math.sin(dow_phase),
            "dow_cos": 0.5 + 0.5 * math.cos(dow_phase),
        }
        return state

    def _forecast_vector(self, refresh: bool = False) -> dict[str, float]:
        if not self.predictive:
            return {}

        if self._cached_forecast and not refresh:
            return dict(self._cached_forecast)

        assert self.hospital is not None
        assert self.forecaster is not None
        mean, std = self.forecaster.predict_from_hospital(
            self.hospital,
            include_uncertainty=self.include_forecast_uncertainty,
        )

        out: dict[str, float] = {}
        H = self.forecaster.config.horizon_h

        for h in range(H):
            out[f"forecast_arrivals_t+{h + 1}"] = self._clip01(
                float(mean[2 * h]) / 20.0
            )
            out[f"forecast_critical_arrivals_t+{h + 1}"] = self._clip01(
                float(mean[2 * h + 1]) / 10.0
            )

        if self.include_forecast_uncertainty:
            for h in range(H):
                out[f"forecast_arrivals_std_t+{h + 1}"] = self._clip01(
                    float(std[2 * h]) / 10.0
                )
                out[f"forecast_critical_arrivals_std_t+{h + 1}"] = self._clip01(
                    float(std[2 * h + 1]) / 5.0
                )

        self._cached_forecast = dict(out)
        return out

    def _state_vector(self) -> np.ndarray:
        s = self._state_dict()
        f = self._forecast_vector()
        merged = {**s, **f}
        return np.asarray([merged[name] for name in self.obs_names], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _reward(self, start_t: float, end_t: float) -> tuple[float, dict[str, float]]:
        assert self.hospital is not None
        h = self.hospital
        w = self.reward_weights

        waits, critical_waits = self._queue_stats()
        critical_queue = float(len(critical_waits))
        boarding = float(len(h.boarding))
        boarding_gt4 = float(self._boarding_gt4())
        ed_queue = float(len(h.ed.waiting))
        critical_wait = float(critical_waits.mean()) if critical_waits.size else 0.0

        dt = max(1e-9, end_t - start_t)
        events = self._recent_events(start_t, end_t)
        surge_bed_hours = float(h.ed.surge) * dt

        components = {
            "critical_queue_cost": w.critical_queue * critical_queue * dt,
            "critical_wait_cost": w.critical_wait * critical_wait * dt,
            "ed_queue_cost": w.ed_queue * ed_queue * dt,
            "boarding_cost": w.boarding * boarding * dt,
            "boarding_gt4_cost": w.boarding_gt4 * boarding_gt4 * dt,
            "lwbs_cost": w.lwbs_event * events["lwbs"],
            "elective_cancel_cost": w.elective_cancel_event * events["elective_cancels"],
            "ot_block_cost": w.ot_block_hour * events["ot_block_h"],
            "surge_cost": w.surge_bed_hour * surge_bed_hours,
            "early_discharge_cost": w.early_discharge * events["early_discharges"],
        }
        total_cost = float(sum(components.values()))

        # Scale rewards to a numerically comfortable PPO range without changing
        # the ordering induced by the scalarised objective.
        reward = -total_cost / 10.0
        components["surge_bed_hours"] = surge_bed_hours
        components["reward"] = reward
        return reward, components

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        options = options or {}
        episode_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31 - 1))
        self.scenario_name = self._choose_scenario(options)

        cfg = self._make_config(episode_seed, self.scenario_name)
        self.policy = RLPolicy()
        self.hospital = Hospital(cfg, self.policy, alerts=False, rtls=False)

        # Neutral actions during warm-up. This generates the state needed by
        # the policy without allowing training actions to influence burn-in.
        self.policy.set_action(0)
        self.hospital.run(self.warmup_h)

        self.elapsed_decisions = 0
        self._prev_event_index = len(self.hospital.events)
        self.trace_rows = []
        self._cached_forecast = {}

        obs = self._state_vector()
        info = {
            "scenario": self.scenario_name,
            "seed": episode_seed,
            "time_h": self.hospital.env.now,
            "observation_names": list(self.obs_names),
        }
        return obs, info

    def step(self, action):
        if self.hospital is None or self.policy is None:
            raise RuntimeError("reset() must be called before step()")

        action_decoded = self.policy.set_action(action)
        start_t = float(self.hospital.env.now)
        end_t = min(self.duration_h, start_t + self.decision_interval_h)

        if end_t <= start_t:
            raise RuntimeError("episode already reached the configured horizon")

        self.hospital.run(end_t)
        reward, reward_components = self._reward(start_t, end_t)
        self._cached_forecast = {}
        obs = self._state_vector()
        terminated = bool(self.hospital.env.now >= self.duration_h - 1e-9)
        truncated = False

        metrics = self.hospital.metrics(warmup=self.warmup_h) if terminated else None
        forecast_info = self._forecast_vector()
        info = {
            "scenario": self.scenario_name,
            "time_h": float(self.hospital.env.now),
            "action": action_decoded.as_array(),
            "reward_components": reward_components,
            "metrics": metrics,
            "ed_surge": int(self.hospital.ed.surge),
            "boarding": int(len(self.hospital.boarding)),
            "predictive": self.predictive,
            "forecast_features": forecast_info,
        }

        if self.collect_trace:
            row = {
                "scenario": self.scenario_name,
                "time_h": float(self.hospital.env.now),
                "reward": float(reward),
                **{f"action_{i}": int(v) for i, v in enumerate(action_decoded.as_array())},
                **{k: float(v) for k, v in reward_components.items()},
                **self._state_dict(),
                **self._forecast_vector(),
            }
            self.trace_rows.append(row)

        self.elapsed_decisions += 1
        return obs, float(reward), terminated, truncated, info

    def get_trace(self):
        import pandas as pd
        return pd.DataFrame(self.trace_rows)

    def get_final_metrics(self) -> dict[str, float]:
        if self.hospital is None:
            raise RuntimeError("environment has not been reset")
        return self.hospital.metrics(warmup=self.warmup_h)


# ---------------------------------------------------------------------------
# Configuration serialization helper
# ---------------------------------------------------------------------------
def save_reward_config(weights: RewardWeights, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(weights), indent=2))
