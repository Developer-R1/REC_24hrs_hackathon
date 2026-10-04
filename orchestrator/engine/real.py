import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE / "sim_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import rl_policy                                              
from hospital_forecaster import HospitalDemandForecaster      
from hospital_rl_env import HospitalRLEnv                     
from rl_policy import RLPolicy                                
from sim import arrival_rate as sim_arrival_rate              
log = logging.getLogger(__name__)

ASSETS = HERE / "assets"
MODEL_PATH = Path(os.getenv("RL_MODEL_PATH", ASSETS / "best_model.zip"))
FORECASTER_PATH = Path(os.getenv("FORECASTER_PATH", ASSETS / "hospital_demand_forecaster.joblib"))
MODE = os.getenv("CONTROLLER_MODE", "hybrid").lower()          # hybrid | rl | rules
SIM_HOURS_PER_SEC = float(os.getenv("SIM_HOURS_PER_SEC", "0.05"))  # 1 simulated hour = 20 real seconds
MAX_CATCHUP_H = 1.0
SEED = int(os.getenv("SIM_SEED", "42"))
ACTION_KEYS = ("ed_surge_beds", "ward_expedite", "icu_expedite", "elective_protect")
METRIC_KEYS = ("ed_wait_mean_h", "ed_wait_p90_h", "critical_wait_mean_h", "lwbs_pct", "boarding_mean_h",
               "boarding_gt4h_pct", "elective_cancel_pct", "elective_delay_h", "ot_blocked_h_per_day",
               "icu_util_pct", "ward_util_pct")


class RealHospitalEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.model_note = "no trained model: rules only"
        self.forecaster = None
        self.forecaster_note = "not loaded"
        self._fc_hour, self._fc = None, None
        self._table = self._probe_action_table()
        self._load_forecaster()
        self.env = HospitalRLEnv(duration_h=1.0e6, warmup_h=24.0, decision_interval_h=1.0,
                                 scenario_mode="normal", trace=False,
                                 forecaster=self.forecaster, include_forecast_uncertainty=False)
        self._load_model()
        self.reset()

    # ------------------------------------------------------------------ loading
    def _probe_action_table(self):
        """Asks rl_policy.py what each action index means, so no table is copied by hand."""
        n = len(getattr(rl_policy, "ACTION_TABLE", range(9)))
        table = {}
        try:
            for i in range(n):
                p = RLPolicy()
                p.set_action(i)
                table[i] = {k: int(getattr(p.action, k)) for k in ACTION_KEYS}
        except Exception as exc:
            raise RuntimeError("rl_policy.py does not decode action indexes. Use the FINAL version "
                               f"that defines ACTION_TABLE. Original error: {exc}")
        return table

    def _load_forecaster(self):
        if not FORECASTER_PATH.exists():
            self.forecaster_note = f"missing file {FORECASTER_PATH.name}"
            return
        try:
            self.forecaster = HospitalDemandForecaster.load(FORECASTER_PATH)
            self.forecaster_note = "loaded"
        except Exception as exc:
            log.warning("forecaster failed to load", exc_info=True)
            self.forecaster_note = f"failed to load: {exc}"

    def _load_model(self):
        if MODE == "rules":
            self.model_note = "disabled by CONTROLLER_MODE=rules"
            return
        if not MODEL_PATH.exists():
            self.model_note = f"missing file {MODEL_PATH.name}"
            return
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(MODEL_PATH), device="cpu",
                             custom_objects={"lr_schedule": lambda _: 3e-4, "clip_range": lambda _: 0.2})
            need, have = int(model.observation_space.shape[0]), len(self.env.obs_names)
            if need != have:
                self.model_note = f"observation mismatch: model expects {need} inputs, environment gives {have}"
                return
            self.model, self.model_note = model, f"PPO loaded ({have} inputs)"
        except Exception as exc:
            log.warning("model failed to load", exc_info=True)
            self.model_note = f"failed to load: {exc}"

    # ------------------------------------------------------------------ lifecycle
    def reset(self):
        with self.lock:
            self.env.reset(seed=SEED, options={"scenario": "normal"})
            self.h, self.policy = self.env.hospital, self.env.policy
            self.action_idx, self.action = 0, dict(self._table[0])
            self._last_real = time.time()
            self._inj_pid = 5_000_000
            self._fc_hour = None

    def _advance(self):
        now = time.time()
        dt = min((now - self._last_real) * SIM_HOURS_PER_SEC, MAX_CATCHUP_H)
        self._last_real = now
        if dt > 1e-6:
            self.h.run(self.h.env.now + dt)

    # ------------------------------------------------------------------ actions
    def get_action(self):
        with self.lock:
            return {**self.action, "index": self.action_idx}

    def _index_for(self, a):
        vec = [int(a.get(k, self.action[k])) for k in ACTION_KEYS]
        return min(self._table, key=lambda i: sum(abs(self._table[i][k] - v) for k, v in zip(ACTION_KEYS, vec)))

    def apply_action(self, action):
        with self.lock:
            self._advance()
            idx = action.get("index")
            idx = int(idx) if idx is not None and int(idx) in self._table else self._index_for(action)
            self.policy.set_action(idx)
            self.action_idx, self.action = idx, dict(self._table[idx])
            return {**self.action, "index": idx}

    # ------------------------------------------------------------------ state and metrics
    def _count_events(self, types, window=1.0):
        t0, n = self.h.env.now - window, 0
        for e in reversed(self.h.events):
            if e["t"] <= t0:
                break
            if e["type"] in types:
                n += 1
        return n

    def _state(self):
        h, t = self.h, float(self.h.env.now)
        ed, ward, icu, ot, ct = h.ed, h.ward, h.icu, h.ot, h.ct
        waits = [t - w[2] for w in ed.waiting]
        crit = sum(1 for w in ed.waiting if w[0].acuity <= 2)
        occ = lambda u: min(1.0, u.used / max(1, u.cap))
        free = sum(max(0, u.limit - u.used) for u in (ed, ward, icu))
        staffed = sum(max(1, u.limit) for u in (ed, ward, icu))
        elective_q = sum(1 for c in h.electives if c.t_arrive >= self.env.warmup_h
                         and not c.cancelled and not math.isfinite(c.t_inpt_bed))
        return {
            "time": round(t, 2),
            "ed_occupancy": round(occ(ed), 3),
            "ed_queue": len(ed.waiting),
            "critical_queue": crit,
            "ed_mean_wait": round(sum(waits) / len(waits), 2) if waits else 0.0,
            "ward_occupancy": round(occ(ward), 3),
            "ward_queue": len(ward.waiting),
            "icu_occupancy": round(occ(icu), 3),
            "icu_queue": len(icu.waiting),
            "ot_occupancy": round(occ(ot), 3),
            "ot_queue": len(ot.waiting),
            "ct_occupancy": round(occ(ct), 3),
            "ct_queue": len(ct.waiting),
            "nurse_availability": round(free / staffed, 3),   # free staffed beds / staffed beds (proxy)
            "arrival_rate": round(sim_arrival_rate(h.cfg, t), 2),
            "discharge_rate": float(self._count_events({"discharge_ed", "inpatient_discharge"})),
            "elective_queue": elective_q,
            "hour": int(t % 24),
        }

    def get_state(self):
        with self.lock:
            self._advance()
            return self._state()

    def get_metrics(self):
        """Rolling window: the last 24 simulated hours (never before the warm-up ends)."""
        with self.lock:
            self._advance()
            start = max(self.env.warmup_h, self.h.env.now - 24.0)
            m = self.h.metrics(warmup=start)
            return {k: round(float(m[k]), 2) for k in METRIC_KEYS}

    def get_forecast(self):
        if not self.forecaster:
            return None
        hour = int(self.h.env.now)
        if self._fc_hour != hour:
            mean, _ = self.forecaster.predict_from_hospital(self.h, include_uncertainty=False)
            self._fc = {"arrivals": [round(float(mean[2 * i]), 1) for i in range(6)],
                        "critical": [round(float(mean[2 * i + 1]), 1) for i in range(6)], "horizon_h": 6}
            self._fc_hour = hour
        return self._fc

    def get_extras(self):
        with self.lock:
            self._advance()
            h = self.h
            try:
                forecast = self.get_forecast()
            except Exception as exc:
                log.warning("forecast failed: %s", exc)
                forecast = None
            return {"boarding": len(h.boarding), "boarding_gt4h": self.env._boarding_gt4(),
                    "ed_surge_active": h.ed.surge, "ct_scanners_down": h.ct.down,
                    "forecast": forecast, "controller": "rl" if self.model else "rules",
                    "controller_mode": MODE, "controller_note": self.model_note}

    def describe(self):
        with self.lock:
            return {"mode": MODE, "model": self.model_note, "forecaster": self.forecaster_note,
                    "obs_inputs": len(self.env.obs_names), "actions": self._table,
                    "sim_hour": round(self.h.env.now, 2)}

    # ------------------------------------------------------------------ scenarios
    def inject(self, kind, n=30):
        """Mass-casualty event: n patients reach the ED right now, about 30% critical (ESI 1-2)."""
        if kind != "mass_casualty":
            raise ValueError(f"unknown scenario {kind!r}; supported: mass_casualty")
        with self.lock:
            self._advance()
            h = self.h
            for _ in range(max(1, min(int(n), 200))):
                want_critical = h.rng.random() < 0.3
                for _try in range(60):
                    self._inj_pid += 1
                    p = h._new_patient(self._inj_pid)
                    if (p.acuity <= 2) == want_critical:
                        break
                h.patients.append(p)
                h._arr_this_hour += 1
                h.env.process(h._ed_flow(p))

    def expedite(self, unit, n=1):
        """Asks the simulator to discharge up to n eligible patients early. Returns how many were expedited."""
        with self.lock:
            self._advance()
            return len(self.h.expedite(unit, n))
    

    def _rule_target(self, s):
        """TriageFirst thresholds (boarding 3 on / 1 off, expedite at 4) plus ED-queue triggers added for responsiveness."""
        b, q = len(self.h.boarding), s["ed_queue"]
        if self.action["ed_surge_beds"] == 0:
            surge = 2 if (b >= 3 or q >= 8) else 0
        else:
            surge = 2 if (b > 1 or q >= 4) else 0
        exp = 1 if b >= 4 else 0
        return {"ed_surge_beds": surge, "ward_expedite": exp, "icu_expedite": exp, "elective_protect": 1}

    def recommend(self):
        """Returns (action, reasons, risk, source), or None when nothing should change."""
        with self.lock:
            self._advance()
            s = self._state()
            rule = self._rule_target(s)
            idx, source = None, "rules"
            if self.model is not None and MODE != "rules":
                try:
                    self.env._cached_forecast = {}
                    obs = self.env._state_vector()
                    raw, _ = self.model.predict(obs, deterministic=True)
                    idx = int(np.asarray(raw).reshape(-1)[0])
                    if idx not in self._table:
                        raise ValueError(f"model returned unknown action {idx}")
                    source = "rl"
                except Exception:
                    log.warning("model prediction failed; using rules", exc_info=True)
                    idx = None
            if idx is not None and MODE == "hybrid":
                merged = {k: max(self._table[idx][k], rule[k]) for k in ACTION_KEYS}
                new_idx = self._index_for(merged)
                if new_idx != idx:
                    idx, source = new_idx, "rl+rules"
            if idx is None:
                idx, source = self._index_for(rule), "rules"
            if idx == self.action_idx:
                return None
            new, old = dict(self._table[idx]), self.action
            return ({**new, "index": idx}, self._reasons(old, new, s, source), self._risk(old, new), source)

    @staticmethod
    def _risk(old, new):
        if any(new[k] > old[k] for k in ("ward_expedite", "icu_expedite", "elective_protect")):
            return "high"
        return "medium" if new["ed_surge_beds"] != old["ed_surge_beds"] else "low"

    def _reasons(self, old, new, s, source):
        b = len(self.h.boarding)
        r = []
        if new["ed_surge_beds"] != old["ed_surge_beds"]:
            r.append(f"Set ED surge beds to {new['ed_surge_beds']}: ED is {s['ed_occupancy']:.0%} full, "
                     f"{s['ed_queue']} waiting, {b} admitted patients boarding")
        for key, unit in (("ward_expedite", "ward"), ("icu_expedite", "ICU")):
            if new[key] != old[key]:
                r.append(f"Expedite one eligible {unit} discharge per hour: {b} boarders (threshold 4)" if new[key]
                         else f"Stop expediting {unit} discharges: boarding has eased to {b}")
        if new["elective_protect"] != old["elective_protect"]:
            r.append("Protect post-op beds: defer electives when downstream beds are nearly full" if new["elective_protect"]
                     else "Resume the baseline elective admission rule")
        try:
            fc = self.get_forecast()
            if fc:
                r.append(f"Forecast: about {fc['arrivals'][0]} arrivals next hour, {sum(fc['arrivals'][:3]):.0f} over 3 hours")
        except Exception:
            pass
        r.append({"rl": "Source: trained PPO policy", "rl+rules": "Source: PPO policy, raised by the safety rules",
                  "rules": "Source: TriageFirst rules"}[source])
        return r