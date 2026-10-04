"""
Hospital Digital Twin — discrete-event simulation core.

Design principles
-----------------
1. The hospital dynamics live here; managerial decisions live in policies.py.
2. Policies never own the stochastic random-number generators. This gives
   common-random-number comparisons: the same seed produces the same demand,
   LOS, diagnostics and equipment-failure realizations.
3. Capacity, staffing, queues and downstream blocking are explicit state.
4. The model is deliberately a research prototype, not a calibrated clinical
   model. Population parameters must be calibrated before real-world use.

Run:
    from sim import Config, Hospital
    from policies import TriageFirst
    h = Hospital(Config(seed=1), TriageFirst()).run(168)
    print(h.metrics(warmup=24))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import simpy

# ---------------------------- population assumptions -----------------------
ACUITY_P = [0.03, 0.12, 0.35, 0.35, 0.15]  # ESI 1..5
ED_TREAT_H = {1: 3.0, 2: 2.5, 3: 2.0, 4: 1.2, 5: 0.7}
P_ADMIT = {1: 0.90, 2: 0.60, 3: 0.30, 4: 0.05, 5: 0.0}
P_ICU = {1: 0.60, 2: 0.20, 3: 0.03, 4: 0.0, 5: 0.0}
P_CT = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.1, 5: 0.0}
PATIENCE_H = {1: math.inf, 2: math.inf, 3: 10.0, 4: 5.0, 5: 3.0}
NURSE_RATIO = {"ed": 4, "ward": 6, "icu": 2}

ADMIT_FRAC = {
    "ward": sum(p * P_ADMIT[a] * (1 - P_ICU[a])
                for a, p in zip(range(1, 6), ACUITY_P)),
    "icu": sum(p * P_ADMIT[a] * P_ICU[a]
               for a, p in zip(range(1, 6), ACUITY_P)),
}


@dataclass
class Config:
    seed: int = 0
    ed_beds: int = 12
    ward_beds: int = 56
    icu_beds: int = 10
    ot_rooms: int = 2
    ct_scanners: int = 2
    max_surge_beds: int = 4
    base_arrivals_per_hour: float = 5.0
    ward_los_mean_h: float = 48.0
    icu_los_mean_h: float = 48.0
    los_sigma: float = 0.6
    clean_h: float = 0.5
    electives_per_weekday: int = 4
    elective_los_mean_h: float = 36.0
    ct_failures: bool = True
    surges: list[tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        ints = ("ed_beds", "ward_beds", "icu_beds", "ot_rooms",
                "ct_scanners", "max_surge_beds", "electives_per_weekday")
        for name in ints:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.base_arrivals_per_hour < 0 or self.los_sigma < 0:
            raise ValueError("arrival rate and los_sigma must be non-negative")
        if self.clean_h < 0 or self.elective_los_mean_h <= 0:
            raise ValueError("invalid housekeeping/elective LOS")
        for start, end, mult in self.surges:
            if end <= start or mult < 0:
                raise ValueError(f"invalid surge: {(start, end, mult)}")


def surge_multiplier(cfg: Config, t: float) -> float:
    mult = 1.0
    for start, end, factor in cfg.surges:
        if start <= t < end:
            mult *= factor
    return mult


def arrival_rate(cfg: Config, t: float) -> float:
    hour = t % 24.0
    dow = int(t // 24.0) % 7
    daily = 1.0 + 0.4 * math.sin(2 * math.pi * (hour - 9) / 24)
    weekly = 1.1 if dow in (0, 6) else 1.0
    return cfg.base_arrivals_per_hour * daily * weekly * surge_multiplier(cfg, t)


@dataclass
class Patient:
    pid: int
    kind: str
    acuity: int
    admit: bool = False
    dest: Optional[str] = None
    ed_treat_h: float = 0.0
    los_h: float = 0.0
    needs_ct: bool = False
    ct_h: float = 0.0
    patience_h: float = math.inf
    surgery_h: float = 0.0
    t_arrive: float = 0.0
    t_ed_bed: float = math.nan
    t_ed_done: float = math.nan
    t_inpt_bed: float = math.nan
    lwbs: bool = False
    cancelled: bool = False
    ot_block_h: float = 0.0
    ot_delay_h: float = 0.0
    expedite: Optional[simpy.Event] = None


class Unit:
    """Capacity-limited pool whose queue ordering is delegated to a policy."""

    def __init__(self, env, name: str, base: int, score: Callable,
                 clean_h: float = 0.0, max_extra: int = 0):
        self.env, self.name, self.base = env, name, base
        self.surge, self.down = 0, 0
        self.staffed: Optional[int] = None
        self.clean_h, self._score = clean_h, score
        self.state = ["U"] * (base + max_extra)  # U free, O occupied, H cleaning
        self.owner = [None] * (base + max_extra)
        self.waiting = []

    @property
    def cap(self):
        return max(0, self.base + self.surge - self.down)

    @property
    def used(self):
        return sum(s != "U" for s in self.state[:self.cap])

    @property
    def limit(self):
        return self.cap if self.staffed is None else min(self.cap, self.staffed)

    @property
    def free_slots(self):
        return max(0, self.limit - self.used)

    def _grab(self, p):
        for i in range(self.cap):
            if self.state[i] == "U":
                self.state[i], self.owner[i] = "O", p.pid
                return i
        raise RuntimeError(f"{self.name}: no physical free slot")

    def request(self, p):
        ev = self.env.event()
        self.waiting.append((p, ev, self.env.now))
        self.dispatch()
        return ev

    def try_acquire(self, p):
        if self.free_slots and not self.waiting:
            return self._grab(p)
        return None

    def cancel(self, ev):
        before = len(self.waiting)
        self.waiting = [x for x in self.waiting if x[1] is not ev]
        if len(self.waiting) != before:
            self.dispatch()

    def dispatch(self):
        while self.waiting and self.free_slots:
            now = self.env.now
            k = min(
                range(len(self.waiting)),
                key=lambda i: (
                    self._score(self.name, self.waiting[i][0],
                                now - self.waiting[i][2], now), i)
            )
            p, ev, _ = self.waiting.pop(k)
            if not ev.triggered:
                ev.succeed(self._grab(p))

    def discharge_bed(self, i):
        if self.clean_h:
            self.state[i] = "H"
            self.env.process(self._clean(i))
        else:
            self.state[i], self.owner[i] = "U", None
            self.dispatch()

    def _clean(self, i):
        yield self.env.timeout(self.clean_h)
        self.state[i], self.owner[i] = "U", None
        self.dispatch()


class _NullAlerts:
    active = []

    def evaluate(self, hospital):
        pass


class _NullRTLS:
    def step(self, hospital):
        pass


class Hospital:
    def __init__(self, cfg: Config, policy, alerts=True, rtls=True):
        self.cfg, self.policy = cfg, policy

        # Independent streams: demand/electives/faults.
        s_a, s_e, s_f = np.random.SeedSequence(cfg.seed).spawn(3)
        self.rng = np.random.default_rng(s_a)
        self.rng_e = np.random.default_rng(s_e)
        self.rng_f = np.random.default_rng(s_f)

        self.env = simpy.Environment()
        sc = self._score
        self.ed = Unit(self.env, "ed", cfg.ed_beds, sc, 0.1, cfg.max_surge_beds)
        self.ward = Unit(self.env, "ward", cfg.ward_beds, sc, cfg.clean_h)
        self.icu = Unit(self.env, "icu", cfg.icu_beds, sc, cfg.clean_h)
        self.ot = Unit(self.env, "ot", cfg.ot_rooms, sc, 0.5)
        self.ct = Unit(self.env, "ct", cfg.ct_scanners, sc, 0.1)
        self.units = {u.name: u for u in
                      (self.ed, self.ward, self.icu, self.ot, self.ct)}

        self.patients, self.electives = [], []
        self.boarding = set()
        self.inpatients = {"ward": {}, "icu": {}}
        self.events, self.samples, self.arr_hist = [], [], []
        self._arr_this_hour = 0
        self.nurse_hours = 0.0
        self.unstaffed_lost_bed_h = 0.0
        self.early_discharges = 0
        self._last_sample_t = None

        # Optional extensions: the simulator remains runnable without them.
        if alerts:
            try:
                from escalation import EscalationEngine
                self.alerts = EscalationEngine()
            except ImportError:
                self.alerts = _NullAlerts()
        else:
            self.alerts = None
        if rtls:
            try:
                from rtls import RTLS
                self.rtls = RTLS(self.rng_f)
            except ImportError:
                self.rtls = _NullRTLS()
        else:
            self.rtls = None

        self.policy.attach(self)
        for process in (self._arrivals(), self._elective_schedule(),
                        self._shifts(), self._control(), self._monitor()):
            self.env.process(process)
        if cfg.ct_failures:
            self.env.process(self._faults())

    def _score(self, unit, p, waited, now):
        return self.policy.score(unit, p, waited, now)

    def pool(self, dest):
        if dest not in ("ward", "icu"):
            raise ValueError(f"unknown destination {dest!r}")
        return self.ward if dest == "ward" else self.icu

    def log(self, kind, p=None, **kw):
        e = {"t": round(self.env.now, 3), "type": kind}
        if p is not None:
            e.update(pid=p.pid, acuity=p.acuity, kind=p.kind)
        e.update(kw)
        self.events.append(e)

    def set_policy(self, policy):
        self.policy = policy
        policy.attach(self)
        self.apply_staffing()

    def apply_staffing(self):
        """Apply policy-controlled physical capacity and staffing."""
        for name in ("ed", "ward", "icu"):
            u = self.units[name]

            if name == "ed":
                desired_surge = int(self.policy.surge_beds(self, "ed"))
                desired_surge = max(0, min(desired_surge, self.cfg.max_surge_beds))

                # Never shrink the physical ED below an occupied or cleaning
                # surge slot. Because Unit.cap is a contiguous prefix, retain
                # enough surge slots to include the highest occupied surge slot.
                required_surge = 0
                for i in range(u.base, u.base + self.cfg.max_surge_beds):
                    if u.state[i] != "U":
                        required_surge = max(required_surge, i - u.base + 1)

                u.surge = max(desired_surge, required_surge)

            requested_staff = int(self.policy.staffed_beds(self, name))
            u.staffed = max(0, min(requested_staff, u.cap))
            u.dispatch()

    # ------------------------------ demand
    def _arrivals(self):
        c = self.cfg
        # Exact thinning requires an upper bound on lambda(t).
        # Bound = product of every factor >= 1 (factors < 1 can only lower
        # the rate); the plain product under-bounds disjoint surge windows.
        surge_bound = math.prod(max(1.0, m) for _, _, m in c.surges)
        lam_max = c.base_arrivals_per_hour * 1.4 * 1.1 * surge_bound
        if lam_max <= 0:
            yield self.env.timeout(1.0)
            return
        pid = 0
        while True:
            yield self.env.timeout(self.rng.exponential(1 / lam_max))
            lam = arrival_rate(c, self.env.now)
            if self.rng.random() <= lam / lam_max:
                pid += 1
                self._arr_this_hour += 1
                p = self._new_patient(pid)
                self.patients.append(p)
                self.env.process(self._ed_flow(p))

    def _new_patient(self, pid):
        r, c = self.rng, self.cfg
        ac = int(r.choice([1, 2, 3, 4, 5], p=ACUITY_P))
        admit = r.random() < P_ADMIT[ac]
        dest = ("icu" if r.random() < P_ICU[ac] else "ward") if admit else None
        mean = c.icu_los_mean_h if dest == "icu" else c.ward_los_mean_h
        mu = math.log(mean) - c.los_sigma ** 2 / 2
        treat = float(r.gamma(4, ED_TREAT_H[ac] / 4))
        los = float(r.lognormal(mu, c.los_sigma)) if admit else 0.0
        needs_ct = r.random() < P_CT[ac]
        ct_h = float(r.gamma(4, 0.4 / 4))
        patience = (float(r.exponential(PATIENCE_H[ac]))
                    if math.isfinite(PATIENCE_H[ac]) else math.inf)
        return Patient(pid, "ed", ac, admit, dest, treat, los,
                       needs_ct, ct_h, patience, t_arrive=self.env.now,
                       expedite=self.env.event())

    def _elective_schedule(self):
        c = self.cfg
        slots = [8.0 + 1.5 * k for k in range(c.electives_per_weekday)]
        day, pid = 0, 100000
        while True:
            if day % 7 not in (0, 6):
                for slot in slots:
                    yield self.env.timeout(max(0, day * 24 + slot - self.env.now))
                    pid += 1
                    r = self.rng_e
                    dest = "icu" if r.random() < 0.15 else "ward"
                    mu = math.log(c.elective_los_mean_h) - 0.18
                    case = Patient(
                        pid, "elective", 3, True, dest,
                        surgery_h=float(r.lognormal(math.log(2.2) - 0.06, 0.35)),
                        los_h=float(r.lognormal(mu, 0.6)),
                        t_arrive=self.env.now, expedite=self.env.event())
                    self.env.process(self._elective_flow(case))
            day += 1
            yield self.env.timeout(max(0, day * 24 - self.env.now))

    # ------------------------------ pathways
    def _ed_flow(self, p):
        env = self.env
        self.log("arrival", p)
        ev = self.ed.request(p)
        if math.isfinite(p.patience_h):
            res = yield ev | env.timeout(p.patience_h)
            if ev not in res and not ev.triggered:
                self.ed.cancel(ev)
                p.lwbs = True
                self.log("lwbs", p, waited_h=round(p.patience_h, 2))
                return
        else:
            yield ev

        bed = ev.value
        p.t_ed_bed = env.now
        self.log("ed_bed_assigned", p, unit="ed", bed=bed)

        if p.needs_ct:
            cev = self.ct.request(p)
            yield cev
            yield env.timeout(p.ct_h)
            self.ct.discharge_bed(cev.value)

        yield env.timeout(p.ed_treat_h)
        p.t_ed_done = env.now

        if not p.admit:
            self.ed.discharge_bed(bed)
            self.log("discharge_ed", p)
            return

        pool = self.pool(p.dest)
        self.boarding.add(p.pid)
        iev = pool.request(p)
        yield iev
        self.boarding.discard(p.pid)
        ib = iev.value
        p.t_inpt_bed = env.now
        self.ed.discharge_bed(bed)
        self.log("admitted", p, unit=p.dest, bed=ib,
                 boarding_h=round(env.now - p.t_ed_done, 3))
        yield from self._inpatient_stay(p, pool, ib)

    def _inpatient_stay(self, p, pool, ib):
        self.inpatients[p.dest][p.pid] = (p, self.env.now,
                                          self.cfg.icu_los_mean_h
                                          if p.dest == "icu"
                                          else self.cfg.ward_los_mean_h)
        yield self.env.timeout(p.los_h) | p.expedite
        self.inpatients[p.dest].pop(p.pid, None)
        pool.discharge_bed(ib)
        self.log("inpatient_discharge", p, unit=p.dest, bed=ib,
                 early=p.expedite.triggered)

    def _elective_flow(self, case):
        env = self.env
        self.electives.append(case)
        self.log("elective_scheduled", case, dest=case.dest)
        defers = 0
        while self.policy.elective_decision(self, case) == "defer":
            defers += 1
            if defers > 3:
                case.cancelled = True
                self.log("elective_cancelled", case,
                         reason="no safe post-op bed after 3 deferrals")
                return
            yield env.timeout(1.0)

        case.ot_delay_h = env.now - case.t_arrive
        oev = self.ot.request(case)
        res = yield oev | env.timeout(6.0)
        if oev not in res and not oev.triggered:
            self.ot.cancel(oev)
            case.cancelled = True
            self.log("elective_cancelled", case, reason="no OT room within 6h")
            return

        room = oev.value
        case.ot_delay_h = env.now - case.t_arrive
        pool = self.pool(case.dest)
        bed = pool.try_acquire(case) if self.policy.reserve_postop else None
        self.log("surgery_start", case, unit="ot", room=room,
                 reserved=bed is not None)
        yield env.timeout(case.surgery_h)

        if bed is None:
            t0 = env.now
            bev = pool.request(case)
            yield bev
            bed = bev.value
            case.ot_block_h = env.now - t0

        self.ot.discharge_bed(room)
        case.t_inpt_bed = env.now
        self.log("surgery_end", case, unit=case.dest, bed=bed,
                 ot_block_h=round(case.ot_block_h, 2))
        yield from self._inpatient_stay(case, pool, bed)

    # ------------------------------ policy levers
    def expedite(self, unit_name, n):
        now, out = self.env.now, []
        # Key on (ratio, pid): equal ratios are common (several patients get
        # beds at the same instant) and Patient objects are not orderable.
        ranked = sorted(
            (((now - t0) / mean, p.pid, p)
             for p, t0, mean in self.inpatients[unit_name].values()
             if p.acuity >= 3 and now - t0 >= 0.7 * mean
             and not p.expedite.triggered),
            key=lambda x: (x[0], x[1]))
        for _, _, p in ranked[::-1][:max(0, int(n))]:
            p.expedite.succeed()
            self.early_discharges += 1
            out.append(p.pid)
        return out

    # ------------------------------ background
    def _shifts(self):
        while True:
            self.apply_staffing()
            yield self.env.timeout(8.0)

    def _control(self):
        while True:
            yield self.env.timeout(0.5)
            self.policy.control(self)
            # Apply congestion-responsive surge/staffing on the same control
            # timescale rather than waiting for the 8-hour shift cycle.
            self.apply_staffing()
            if self.alerts:
                self.alerts.evaluate(self)
            if self.rtls:
                self.rtls.step(self)

    def _faults(self):
        while True:
            yield self.env.timeout(self.rng_f.exponential(72.0))
            dur = float(self.rng_f.uniform(2, 4))
            self.ct.down += 1
            self.log("device_down", unit="ct", expected_hours=round(dur, 1))
            yield self.env.timeout(dur)
            self.ct.down -= 1
            self.log("device_up", unit="ct")
            self.ct.dispatch()

    def _monitor(self):
        while True:
            s = self.snapshot()
            self.samples.append(s)
            self.arr_hist.append(self._arr_this_hour)
            self._arr_this_hour = 0
            for name in ("ed", "ward", "icu"):
                u = self.units[name]
                self.nurse_hours += math.ceil(u.limit / NURSE_RATIO[name])
                unstaffed_free = max(0, u.cap - max(u.used, u.limit))
                self.unstaffed_lost_bed_h += min(len(u.waiting), unstaffed_free)
            yield self.env.timeout(1.0)

    # ------------------------------ views/metrics
    def snapshot(self, beds=False):
        s = {
            "t": round(self.env.now, 2),
            "boarding": len(self.boarding),
            "nurse_hours": self.nurse_hours,
            "early_discharges": self.early_discharges,
            "open_alerts": len(self.alerts.active) if self.alerts else 0,
        }
        for name, u in self.units.items():
            s[name] = {
                "used": u.used, "cap": u.cap, "staffed": u.limit,
                "waiting": len(u.waiting), "surge": u.surge, "down": u.down
            }
            if beds and name in ("ed", "ward", "icu"):
                s[name]["beds"] = list(u.state[:u.cap])
        return s

    def run(self, until):
        if until < self.env.now:
            raise ValueError("until must not move simulation backwards")
        self.env.run(until=until)
        return self

    def metrics(self, warmup=24.0):
        now = self.env.now
        if not 0 <= warmup <= now:
            raise ValueError(f"warmup must be in [0, {now}]")

        eds = [p for p in self.patients if p.t_arrive >= warmup]
        seen = [p for p in eds if not p.lwbs]
        waits = np.array([
            ((p.t_ed_bed if not math.isnan(p.t_ed_bed) else now) - p.t_arrive)
            for p in seen
        ])
        crit = np.array([
            ((p.t_ed_bed if not math.isnan(p.t_ed_bed) else now) - p.t_arrive)
            for p in seen if p.acuity <= 2
        ])
        board = np.array([
            p.t_inpt_bed - p.t_ed_done for p in eds
            if p.admit and not math.isnan(p.t_inpt_bed)
        ])
        els = [c for c in self.electives if c.t_arrive >= warmup]
        done = [c for c in els if not c.cancelled and not math.isnan(c.t_inpt_bed)]
        samples = [s for s in self.samples if s["t"] >= warmup]
        days = max(1e-9, (now - warmup) / 24)

        def stat(a, fn):
            return float(fn(a)) if len(a) else 0.0

        nurse_h = 0.0
        lost_h = 0.0
        for s in samples:
            for name in ("ed", "ward", "icu"):
                u = s[name]
                nurse_h += math.ceil(u["staffed"] / NURSE_RATIO[name])
                unstaffed_free = max(0, u["cap"] - u["staffed"])
                lost_h += min(u["waiting"], unstaffed_free)

        denom_h = max(1e-9, now - warmup)
        return {
            "patients": len(eds),
            "ed_wait_mean_h": stat(waits, np.mean),
            "ed_wait_p90_h": stat(waits, lambda x: np.percentile(x, 90)),
            "critical_wait_mean_h": stat(crit, np.mean),
            "lwbs_pct": 100 * sum(p.lwbs for p in eds) / max(1, len(eds)),
            "boarding_mean_h": stat(board, np.mean),
            "boarding_gt4h_pct": stat(board, lambda x: 100 * np.mean(x > 4)),
            "elective_cancel_pct": 100 * sum(c.cancelled for c in els) / max(1, len(els)),
            "elective_delay_h": stat([c.ot_delay_h for c in done], np.mean),
            "ot_blocked_h_per_day": sum(c.ot_block_h for c in els) / days,
            "nurse_hours_per_day": nurse_h / days,
            "lost_bed_h_per_day": lost_h / days,
            "early_discharges": self.early_discharges,
            "icu_util_pct": stat(
                [s["icu"]["used"] / max(1, s["icu"]["cap"]) for s in samples],
                lambda x: 100 * np.mean(x)),
            "ward_util_pct": stat(
                [s["ward"]["used"] / max(1, s["ward"]["cap"]) for s in samples],
                lambda x: 100 * np.mean(x)),
        }
