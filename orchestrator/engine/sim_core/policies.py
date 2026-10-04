"""
Policy layer for the Hospital Digital Twin.

Policies are deliberately deterministic functions of observable hospital state.
They do not sample randomness. That is crucial for fair common-random-number
comparisons in simulation experiments.

Lower score => earlier service.

Included policies
-----------------
FCFS:
    Pure first-come-first-served; tie-break via queue order supplied by
    Unit.dispatch().

TriageFirst:
    Acuity has priority, with an ageing term so lower-acuity patients cannot be
    starved indefinitely. Also bundles ED surge, elective gating and downstream
    expediting (see the validation suite for single-mechanism variants).

Hooks exposed to the simulator:
    score(), surge_beds(), staffed_beds(), elective_decision(), control(),
    reserve_postop
"""
from __future__ import annotations


class BasePolicy:
    name = "base"
    reserve_postop = False
    h = None

    def attach(self, hospital):
        self.h = hospital

    def score(self, unit, patient, waited, now):
        raise NotImplementedError

    def surge_beds(self, hospital, unit):
        """Request no physical surge capacity by default."""
        return 0

    def staffed_beds(self, hospital, unit):
        """Default staffing: staff the fixed/base physical capacity only."""
        return hospital.units[unit].base

    def elective_decision(self, hospital, case):
        """Start unless the downstream bed pool is already saturated."""
        pool = hospital.pool(case.dest)
        pressure = (pool.used + len(pool.waiting)) / max(1, pool.limit)
        return "defer" if pressure >= 0.98 else "start"

    def control(self, hospital):
        pass


class FCFS(BasePolicy):
    name = "fcfs"

    def score(self, unit, patient, waited, now):
        # Longer wait => more negative => served first under min().
        return -waited


class TriageFirst(BasePolicy):
    name = "triage"

    AGEING_RATE_H = 6.0       # hours of waiting per acuity level gained
    AGEING_CAP = 4.0          # max acuity levels gained by waiting
    SURGE_LEVEL = 2           # ED surge beds requested when active
    SURGE_ON_BOARDING = 3     # activate when boarding >= this
    SURGE_OFF_BOARDING = 1    # release when boarding <= this (hysteresis)
    EXPEDITE_BOARDING = 4     # expedite downstream when boarding >= this
    ELECTIVE_RESERVE = {"icu": 0.08, "ward": 0.05}

    def score(self, unit, patient, waited, now):
        # ESI 1 is most urgent. Ageing prevents indefinite starvation.
        ageing = min(waited / self.AGEING_RATE_H, self.AGEING_CAP)
        return (patient.acuity - ageing, -waited)

    def surge_beds(self, hospital, unit):
        """Request physical ED surge capacity; never mutates the Unit."""
        if unit != "ed":
            return 0

        boarding = len(hospital.boarding)
        level = min(self.SURGE_LEVEL, hospital.cfg.max_surge_beds)

        if hospital.units["ed"].surge == 0:
            return level if boarding >= self.SURGE_ON_BOARDING else 0
        return level if boarding > self.SURGE_OFF_BOARDING else 0

    def staffed_beds(self, hospital, unit):
        """Staff the physical capacity actually activated (ED only)."""
        u = hospital.units[unit]
        return u.base + u.surge if unit == "ed" else u.base

    def elective_decision(self, hospital, case):
        pool = hospital.pool(case.dest)
        pressure = (pool.used + len(pool.waiting)) / max(1, pool.limit)
        reserve = self.ELECTIVE_RESERVE.get(case.dest, 0.05)
        return "defer" if pressure >= 1.0 - reserve else "start"

    def control(self, hospital):
        if len(hospital.boarding) >= self.EXPEDITE_BOARDING:
            hospital.expedite("ward", 1)
            hospital.expedite("icu", 1)


POLICIES = {"fcfs": FCFS, "triage": TriageFirst}


def make_policy(name: str):
    try:
        return POLICIES[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown policy {name!r}; choose {sorted(POLICIES)}") from exc
