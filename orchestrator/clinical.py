from django.utils import timezone

from .models import Asset, Encounter, Suggestion
from .roster import progress

MAX_PENDING = 2
MIN_PROGRESS = 0.75


def run_cycle(state):
    """If a severe patient is waiting and no ED or ICU bed is free, suggest discharging the most recovered ward/ICU patients."""
    free = {}
    for kind in Asset.objects.filter(asset_type="bed", status="free").values_list("location__kind", flat=True):
        free[kind] = free.get(kind, 0) + 1
    active = state["critical_queue"] > 0 and (free.get("ed", 0) == 0 or free.get("icu", 0) == 0)
    pending = Suggestion.objects.filter(status="pending")
    if not active:
        pending.update(status="expired", decided_at=timezone.now(), decided_by="System")
        return
    room = MAX_PENDING - pending.count()
    if room <= 0:
        return
    seen = set(Suggestion.objects.exclude(status="expired").values_list("encounter_id", flat=True))
    cands = []
    for e in Encounter.objects.filter(status="in-progress", bed__location__kind__in=["ward", "icu"]).select_related("bed__location"):
        pr = progress(e)
        if e.acuity >= 3 and pr >= MIN_PROGRESS and e.id not in seen:
            cands.append((pr, e))
    cands.sort(key=lambda x: -x[0])
    which = "ED" if free.get("ed", 0) == 0 else "ICU"
    for pr, e in cands[:room]:
        Suggestion.objects.create(kind="discharge", encounter=e, reasons=[
            f"{state['critical_queue']} critical patient(s) waiting",
            f"No free {which} bed",
            f"Recovery at {pr:.0%} of expected stay, acuity {e.acuity}",
            f"Discharging frees a bed in {e.bed.location.name}"])