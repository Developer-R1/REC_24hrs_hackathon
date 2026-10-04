import os

from django.utils import timezone

from .models import StaffProfile

SHIFT_H = 12
DOCTORS_ON, NURSES_ON = 3, 6
SIM_HOURS_PER_SEC = float(os.getenv("SIM_HOURS_PER_SEC", "0.05"))   # keep equal to engine/real.py


def slot_info(t):
    k = int(t // SHIFT_H)
    start = (k * SHIFT_H) % 24
    return {"slot": k, "label": f"{start:02d}:00 to {(start + SHIFT_H) % 24:02d}:00",
            "hours_left": round(SHIFT_H - (t % SHIFT_H), 1)}


def _on_duty(role, n_on, t):
    people = list(StaffProfile.objects.filter(role=role, user__is_active=True).order_by("seq", "id"))
    if not people:
        return []
    k, n_on = int(t // SHIFT_H), min(n_on, len(people))
    return [people[(k * n_on + i) % len(people)] for i in range(n_on)]


def doctors_on(t):
    return _on_duty("doctor", DOCTORS_ON, t)


def nurses_on(t):
    return _on_duty("nurse", NURSES_ON, t)


def owner_of(enc_id, docs, nurses):
    """Each patient belongs to one on-duty nurse, and to the doctor whose pair of nurses includes her."""
    if not docs or not nurses:
        return None, None
    nurse = nurses[enc_id % len(nurses)]
    return docs[min(nurses.index(nurse) // 2, len(docs) - 1)], nurse


def progress(enc):
    """Recovery from 0 to 1, advancing with simulated time."""
    hours = (timezone.now() - enc.started_at).total_seconds() * SIM_HOURS_PER_SEC
    return min(1.0, enc.progress0 + hours / enc.expected_h)