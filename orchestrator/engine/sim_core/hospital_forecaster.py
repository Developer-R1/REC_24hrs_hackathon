from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sim import Config, Hospital
from policies import TriageFirst


def _json_default(obj: Any):
    """Convert common NumPy scalars/arrays to JSON-native objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ForecasterConfig:
    lookback_h: int = 24
    horizon_h: int = 6
    n_estimators: int = 300
    min_samples_leaf: int = 3
    max_features: float = 0.75
    random_state: int = 2026


class HospitalDemandForecaster:
    """Multi-horizon ensemble forecaster for ED demand."""

    VERSION = "1.0"

    def __init__(self, config: ForecasterConfig | None = None) -> None:
        self.config = config or ForecasterConfig()
        if self.config.lookback_h < 6:
            raise ValueError("lookback_h must be at least 6")
        if self.config.horizon_h < 1:
            raise ValueError("horizon_h must be positive")

        self.model: ExtraTreesRegressor | None = None
        self.feature_names_: list[str] = []
        self.target_names_: list[str] = []
        self.metadata_: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    @property
    def n_outputs(self) -> int:
        return 2 * self.config.horizon_h

    def _feature_names(self) -> list[str]:
        L = self.config.lookback_h
        names = []
        names += [f"arrivals_lag_{k}h" for k in range(L, 0, -1)]
        names += [f"critical_lag_{k}h" for k in range(L, 0, -1)]
        for w in (3, 6, 12, 24):
            names += [f"arrivals_mean_{w}h", f"critical_mean_{w}h"]
        for w in (6, 24):
            names += [f"arrivals_std_{w}h", f"critical_std_{w}h"]
        names += [
            "arrivals_trend_6h",
            "arrivals_trend_24h",
            "critical_trend_6h",
            "critical_trend_24h",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
        return names

    def _target_names(self) -> list[str]:
        names = []
        for h in range(1, self.config.horizon_h + 1):
            names.append(f"arrivals_t+{h}")
            names.append(f"critical_arrivals_t+{h}")
        return names

    @staticmethod
    def _safe_std(x: np.ndarray) -> float:
        return float(np.std(x, ddof=1)) if x.size >= 2 else 0.0

    @staticmethod
    def _slope(x: np.ndarray) -> float:
        if x.size < 2:
            return 0.0
        t = np.arange(x.size, dtype=float)
        return float(np.polyfit(t, x, 1)[0])

    def _make_features_from_history(
        self,
        arrivals: np.ndarray,
        critical: np.ndarray,
        origin_t: float,
    ) -> np.ndarray:
        L = self.config.lookback_h
        if arrivals.size < L or critical.size < L:
            raise ValueError("insufficient history for forecast features")

        a = arrivals[-L:].astype(float)
        c = critical[-L:].astype(float)
        feat: list[float] = []
        feat.extend(a.tolist())
        feat.extend(c.tolist())

        for w in (3, 6, 12, 24):
            aw = a[-w:]
            cw = c[-w:]
            feat.extend([float(np.mean(aw)), float(np.mean(cw))])
        for w in (6, 24):
            aw = a[-w:]
            cw = c[-w:]
            feat.extend([self._safe_std(aw), self._safe_std(cw)])

        feat.extend([
            self._slope(a[-6:]),
            self._slope(a),
            self._slope(c[-6:]),
            self._slope(c),
        ])

        hour = origin_t % 24.0
        dow = int(origin_t // 24.0) % 7
        hphase = 2.0 * math.pi * hour / 24.0
        dphase = 2.0 * math.pi * dow / 7.0
        feat.extend([
            math.sin(hphase),
            math.cos(hphase),
            math.sin(dphase),
            math.cos(dphase),
        ])
        return np.asarray(feat, dtype=np.float32)

    # ------------------------------------------------------------------
    # Event aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def hourly_counts_from_events(
        events: Iterable[dict[str, Any]],
        max_hour: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        arrivals = np.zeros(max_hour + 1, dtype=np.float32)
        critical = np.zeros(max_hour + 1, dtype=np.float32)
        for e in events:
            if e.get("type") != "arrival":
                continue
            t = float(e.get("t", -1.0))
            hour = int(math.floor(t + 1e-9))
            if 0 <= hour <= max_hour:
                arrivals[hour] += 1.0
                if int(e.get("acuity", 99)) <= 2:
                    critical[hour] += 1.0
        return arrivals, critical

    def dataset_from_hospital(
        self,
        hospital: Hospital,
        *,
        warmup_h: float = 24.0,
        duration_h: float | None = None,
        episode_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        end_t = int(math.floor(duration_h if duration_h is not None else hospital.env.now))
        H = self.config.horizon_h
        L = self.config.lookback_h
        arrivals, critical = self.hourly_counts_from_events(hospital.events, end_t + H)

        rows_x, rows_y, rows_episode = [], [], []
        first_origin = max(L, int(math.ceil(warmup_h)))
        last_origin = end_t - H
        for origin in range(first_origin, last_origin + 1):
            # History includes hours origin-L ... origin-1. The current hour
            # has not been allowed to leak into the feature vector.
            x = self._make_features_from_history(
                arrivals[origin - L:origin],
                critical[origin - L:origin],
                float(origin),
            )
            target = []
            for h in range(1, H + 1):
                target.extend([
                    float(arrivals[origin + h - 1]),
                    float(critical[origin + h - 1]),
                ])
            rows_x.append(x)
            rows_y.append(np.asarray(target, dtype=np.float32))
            rows_episode.append(episode_id)
        return (
            np.asarray(rows_x, dtype=np.float32),
            np.asarray(rows_y, dtype=np.float32),
            np.asarray(rows_episode, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "HospitalDemandForecaster":
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X and y must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X/y row counts differ")
        if y.shape[1] != self.n_outputs:
            raise ValueError(
                f"expected {self.n_outputs} targets, got {y.shape[1]}"
            )

        self.feature_names_ = self._feature_names()
        self.target_names_ = self._target_names()

        model = ExtraTreesRegressor(
            n_estimators=self.config.n_estimators,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        self.model = model
        self.metadata_ = dict(metadata or {})
        self.metadata_.update({
            "version": self.VERSION,
            "config": asdict(self.config),
            "feature_names": self.feature_names_,
            "target_names": self.target_names_,
        })
        return self

    def _require_model(self) -> ExtraTreesRegressor:
        if self.model is None:
            raise RuntimeError("forecaster has not been fitted/loaded")
        return self.model

    def predict(
        self,
        X: np.ndarray,
        *,
        include_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict means and, optionally, tree-dispersion uncertainty.

        The uncertainty calculation is substantially more expensive than the
        ensemble mean because it explicitly evaluates every individual tree.
        For RL evaluation this is normally unnecessary, so callers can set
        ``include_uncertainty=False``.
        """
        model = self._require_model()
        pred = np.asarray(model.predict(X), dtype=np.float32)
        pred = np.maximum(pred, 0.0)

        if not include_uncertainty:
            return pred, np.zeros_like(pred, dtype=np.float32)

        # Ensemble dispersion across trees. This is a model-disagreement
        # measure, not a calibrated statistical prediction interval.
        tree_preds = np.stack(
            [np.asarray(tree.predict(X), dtype=np.float32)
             for tree in model.estimators_],
            axis=0,
        )
        std = np.std(tree_preds, axis=0, ddof=1).astype(np.float32)
        return pred, np.maximum(std, 0.0)

    def predict_from_counts(
        self,
        arrivals: np.ndarray,
        critical: np.ndarray,
        origin_t: float,
        *,
        include_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = self._make_features_from_history(arrivals, critical, origin_t)[None, :]
        mean, std = self.predict(X, include_uncertainty=include_uncertainty)
        return mean[0], std[0]

    def predict_from_hospital(
        self,
        hospital: Hospital,
        *,
        include_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        origin = float(hospital.env.now)
        L = self.config.lookback_h
        end_hour = int(math.floor(origin + 1e-9))
        arrivals, critical = self.hourly_counts_from_events(
            hospital.events, end_hour
        )
        if end_hour < L:
            raise RuntimeError("hospital does not contain enough forecast history")
        return self.predict_from_counts(
            arrivals[end_hour - L:end_hour],
            critical[end_hour - L:end_hour],
            origin,
            include_uncertainty=include_uncertainty,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist the forecaster using a stable, module-independent format.

        We intentionally do not pickle ``self``.  When this file is executed
        with IPython ``%run``, classes can otherwise be recorded as
        ``__main__.HospitalDemandForecaster`` and become impossible to load
        from ``train_rl.py``.  The state-dict format stores only plain Python
        metadata plus the fitted sklearn estimator.
        """
        model = self._require_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "hospital_demand_forecaster_state_v2",
            "config": asdict(self.config),
            "feature_names": list(self.feature_names_),
            "target_names": list(self.target_names_),
            "metadata": dict(self.metadata_),
            "model": model,
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "HospitalDemandForecaster":
        """Load both the new stable format and legacy ``%run`` artifacts.

        Legacy artifacts may contain references such as
        ``__main__.HospitalDemandForecaster``.  The temporary aliases below
        let those older files be recovered without requiring a custom
        unpickler.  New saves never create that problem.
        """
        path = Path(path)

        # Compatibility for forecasters produced by older versions executed
        # through ``%run hospital_forecaster.py``.
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            setattr(main_mod, "HospitalDemandForecaster", cls)
            setattr(main_mod, "ForecasterConfig", ForecasterConfig)

        obj = joblib.load(path)

        # Preferred, stable state-dict format.
        if isinstance(obj, dict) and obj.get("format") == "hospital_demand_forecaster_state_v2":
            config = ForecasterConfig(**obj["config"])
            out = cls(config)
            out.model = obj["model"]
            out.feature_names_ = list(obj.get("feature_names", []))
            out.target_names_ = list(obj.get("target_names", []))
            out.metadata_ = dict(obj.get("metadata", {}))
            return out

        # Legacy object format.  The aliases above make this branch usable
        # even when the old object was serialized from ``__main__``.
        if isinstance(obj, cls):
            return obj

        raise TypeError(
            f"{path} does not contain a HospitalDemandForecaster "
            "or a supported state-dict artifact"
        )


# ---------------------------------------------------------------------------
# Simulation-data generation
# ---------------------------------------------------------------------------
def random_training_surges(rng: np.random.Generator) -> list[tuple[float, float, float]]:
    """Generate varied synthetic demand regimes for forecasting training."""
    surges: list[tuple[float, float, float]] = []
    n = int(rng.integers(0, 3))
    for _ in range(n):
        start = float(rng.integers(24, 120))
        duration = float(rng.integers(6, 49))
        end = min(160.0, start + duration)
        multiplier = float(rng.uniform(1.15, 3.5))
        surges.append((start, end, multiplier))
    return surges


def generate_forecaster_dataset(
    *,
    episodes: int = 180,
    duration_h: int = 168,
    warmup_h: int = 24,
    seed: int = 2026,
    config: ForecasterConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cfg = config or ForecasterConfig()
    master = np.random.default_rng(seed)
    forecaster = HospitalDemandForecaster(cfg)

    Xs, Ys, Eids = [], [], []
    manifests: list[dict[str, Any]] = []

    for episode in range(episodes):
        sim_seed = int(master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sim_seed + 991)
        base_rate = float(rng.uniform(3.0, 8.0))
        surges = random_training_surges(rng)
        sim_cfg = Config(
            seed=sim_seed,
            base_arrivals_per_hour=base_rate,
            surges=surges,
        )
        # Arrival demand is independent of policy. TriageFirst simply gives
        # the downstream simulation a stable operational reference.
        hospital = Hospital(sim_cfg, TriageFirst(), alerts=False, rtls=False).run(duration_h)
        X, y, eid = forecaster.dataset_from_hospital(
            hospital,
            warmup_h=warmup_h,
            duration_h=duration_h,
            episode_id=episode,
        )
        Xs.append(X)
        Ys.append(y)
        Eids.append(eid)
        manifests.append({
            "episode": episode,
            "seed": sim_seed,
            "base_arrivals_per_hour": base_rate,
            "surges": surges,
        })

    return (
        np.vstack(Xs),
        np.vstack(Ys),
        np.concatenate(Eids),
        manifests,
    )


def evaluate_predictions(
    model: HospitalDemandForecaster,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pred, std = model.predict(X)
    H = model.config.horizon_h
    rows = []
    for h in range(1, H + 1):
        for col, label in [(2 * (h - 1), "arrivals"), (2 * (h - 1) + 1, "critical_arrivals")]:
            err = y[:, col] - pred[:, col]
            rows.append({
                "horizon_h": h,
                "target": label,
                "mae": mean_absolute_error(y[:, col], pred[:, col]),
                "rmse": math.sqrt(mean_squared_error(y[:, col], pred[:, col])),
                "mean_ensemble_std": float(np.mean(std[:, col])),
            })
    per_h = pd.DataFrame(rows)
    summary = {
        "mae_all": float(mean_absolute_error(y, pred)),
        "rmse_all": float(math.sqrt(mean_squared_error(y, pred))),
    }
    return per_h, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hospital demand forecaster")
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--duration", type=int, default=168)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="rl_runs/forecaster")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = ForecasterConfig(
        lookback_h=args.lookback,
        horizon_h=args.horizon,
        n_estimators=args.estimators,
        random_state=args.seed,
    )

    print("Generating simulation data for demand forecasting...")
    X, y, episode_ids, manifests = generate_forecaster_dataset(
        episodes=args.episodes,
        duration_h=args.duration,
        warmup_h=args.warmup,
        seed=args.seed,
        config=config,
    )

    # Split by episode, never by row, to prevent temporal leakage across the
    # same simulated trajectory.
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(args.seed + 1)
    rng.shuffle(episodes)
    n = len(episodes)
    n_train = max(1, int(0.70 * n))
    n_val = max(1, int(0.15 * n))
    # Convert NumPy integer scalars to native Python int immediately.
    # This matters because these episode IDs are later written to JSON, and
    # json.dumps() does not serialize np.int64 values by default.
    train_e = {int(e) for e in episodes[:n_train]}
    val_e = {int(e) for e in episodes[n_train:n_train + n_val]}
    test_e = {int(e) for e in episodes[n_train + n_val:]}

    train_mask = np.array([e in train_e for e in episode_ids])
    val_mask = np.array([e in val_e for e in episode_ids])
    test_mask = np.array([e in test_e for e in episode_ids])

    model = HospitalDemandForecaster(config).fit(
        X[train_mask],
        y[train_mask],
        metadata={
            "episodes": args.episodes,
            "duration_h": args.duration,
            "warmup_h": args.warmup,
            "split": {"train": sorted(train_e), "val": sorted(val_e), "test": sorted(test_e)},
            "data_generation": "stochastic hospital DES with randomized synthetic demand surges",
        },
    )

    val_table, val_summary = evaluate_predictions(model, X[val_mask], y[val_mask])
    test_table, test_summary = evaluate_predictions(model, X[test_mask], y[test_mask])
    val_table.to_csv(out / "validation_metrics.csv", index=False)
    test_table.to_csv(out / "test_metrics.csv", index=False)

    # Store data in compact numpy format for reproducibility.
    np.savez_compressed(
        out / "forecast_dataset.npz",
        X=X,
        y=y,
        episode_ids=episode_ids,
    )
    _write_json(out / "simulation_manifest.json", manifests)
    _write_json(out / "model_metadata.json", model.metadata_)
    _write_json(out / "validation_summary.json", val_summary)
    _write_json(out / "test_summary.json", test_summary)
    model.save(out / "hospital_demand_forecaster.joblib")

    print("\nValidation:", val_summary)
    print("Test      :", test_summary)
    print("\nPer-horizon test metrics:")
    print(test_table.round(4).to_string(index=False))
    print(f"\nForecaster saved to: {(out / 'hospital_demand_forecaster.joblib').resolve()}")


if __name__ == "__main__":
    main()
