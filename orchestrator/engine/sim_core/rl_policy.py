"""
RL-controlled hospital policy with a compact categorical action space.

The RL agent retains the validated TriageFirst patient-priority rule and chooses
one operational intervention bundle per decision epoch.

Action index
------------
0  no intervention
1  activate 1 ED surge bed
2  activate 2 ED surge beds
3  expedite 1 ward patient
4  expedite 1 ICU patient
5  expedite 1 ward + 1 ICU patient
6  protect elective capacity
7  surge 2 + expedite ward + ICU
8  surge 2 + expedite ward + ICU + protect electives

This reduces the combinatorial MultiDiscrete action space to nine interpretable
actions and prevents PPO from trivially selecting independent binary switches
that remain permanently "on".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np

from policies import BasePolicy, TriageFirst


ACTION_TABLE = (
    {"name": "none", "ed_surge_beds": 0, "ward_expedite": 0,
     "icu_expedite": 0, "elective_protect": 0},
    {"name": "surge_1", "ed_surge_beds": 1, "ward_expedite": 0,
     "icu_expedite": 0, "elective_protect": 0},
    {"name": "surge_2", "ed_surge_beds": 2, "ward_expedite": 0,
     "icu_expedite": 0, "elective_protect": 0},
    {"name": "expedite_ward", "ed_surge_beds": 0, "ward_expedite": 1,
     "icu_expedite": 0, "elective_protect": 0},
    {"name": "expedite_icu", "ed_surge_beds": 0, "ward_expedite": 0,
     "icu_expedite": 1, "elective_protect": 0},
    {"name": "expedite_both", "ed_surge_beds": 0, "ward_expedite": 1,
     "icu_expedite": 1, "elective_protect": 0},
    {"name": "protect_elective", "ed_surge_beds": 0, "ward_expedite": 0,
     "icu_expedite": 0, "elective_protect": 1},
    {"name": "surge2_expedite_both", "ed_surge_beds": 2, "ward_expedite": 1,
     "icu_expedite": 1, "elective_protect": 0},
    {"name": "full_package", "ed_surge_beds": 2, "ward_expedite": 1,
     "icu_expedite": 1, "elective_protect": 1},
)


@dataclass(frozen=True)
class RLAction:
    index: int = 0
    name: str = "none"
    ed_surge_beds: int = 0
    ward_expedite: int = 0
    icu_expedite: int = 0
    elective_protect: int = 0

    @classmethod
    def from_index(cls, action: Union[int, np.integer, Iterable[int]]) -> "RLAction":
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.size != 1:
            raise ValueError(f"expected one categorical action, got {a.size}")
        idx = int(np.clip(a[0], 0, len(ACTION_TABLE) - 1))
        spec = ACTION_TABLE[idx]
        return cls(index=idx, **spec)

    def as_array(self) -> np.ndarray:
        return np.asarray([self.index], dtype=np.int64)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "ed_surge_beds": self.ed_surge_beds,
            "ward_expedite": self.ward_expedite,
            "icu_expedite": self.icu_expedite,
            "elective_protect": self.elective_protect,
        }


class RLPolicy(TriageFirst):
    """Validated triage priority with RL-controlled operational bundles."""

    name = "rl_triage_control"

    def __init__(self) -> None:
        super().__init__()
        self.action = RLAction()
        self.action_version = 0
        self._applied_control_version = -1

    def set_action(self, action) -> RLAction:
        self.action = RLAction.from_index(action)
        self.action_version += 1
        return self.action

    def surge_beds(self, hospital, unit):
        if unit != "ed":
            return 0
        return min(self.action.ed_surge_beds, hospital.cfg.max_surge_beds)

    def staffed_beds(self, hospital, unit):
        u = hospital.units[unit]
        return u.base + u.surge if unit == "ed" else u.base

    def elective_decision(self, hospital, case):
        if self.action.elective_protect:
            return TriageFirst.elective_decision(self, hospital, case)
        return BasePolicy.elective_decision(self, hospital, case)

    def control(self, hospital):
        """Apply expedite choices once for each agent decision epoch."""
        if self._applied_control_version == self.action_version:
            return

        if self.action.ward_expedite:
            hospital.expedite("ward", 1)
        if self.action.icu_expedite:
            hospital.expedite("icu", 1)

        self._applied_control_version = self.action_version
