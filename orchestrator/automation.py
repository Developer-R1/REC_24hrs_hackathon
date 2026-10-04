import logging
import random
import threading
import time

from django.db import close_old_connections
from django.db.migrations import state
from django.db.migrations import state
from django.utils import timezone

from .engine import get_engine
from .models import Asset, AuditLog, Encounter, Location, PingLog, Proposal

log = logging.getLogger(__name__)

TICK = 3              # seconds between automation cycles
COOLDOWN = 25         # seconds after a decision before a new proposal
ESCALATE_AFTER = 60   # seconds per escalation level
EXPIRE_AFTER = 300
ROLES = ["Administrator", "MD"]

LAYOUT = [  # reader code, name, kind, grid x, grid z
    ("R-ED", "Emergency dept", "ed", 0, 0), ("R-CT", "CT suite", "ct", 1, 0),
    ("R-OT", "Operating theatre", "ot", 2, 0), ("R-WARD", "General ward", "ward", 0, 1),
    ("R-ICU", "ICU", "icu", 1, 1), ("R-STORE", "Equipment store", "store", 2, 1),
]
MOVES = {"ventilator": ["store", "icu", "ed", "ot"],
         "wheelchair": ["ed", "ct", "ward", "store"],
         "pump": ["ward", "icu", "ed", "store"]}
_started = False


def audit(actor, message):
    AuditLog.objects.create(actor=actor, message=message[:250])


def seed():
    if Location.objects.exists():
        return
    locs = {c: Location.objects.create(code=c, name=n, kind=k, gx=x, gz=z) for c, n, k, x, z in LAYOUT}

    def make(prefix, atype, label, count, homes):
        for i in range(1, count + 1):
            Asset.objects.create(tag_id=f"{prefix}-{i:02d}", name=f"{label} {i:02d}", asset_type=atype,
                                 location=locs[homes[(i - 1) % len(homes)]], last_seen=timezone.now())

    make("BED", "bed", "Bed", 18, ["R-ED", "R-WARD", "R-ICU"])
    make("VEN", "ventilator", "Ventilator", 6, ["R-STORE", "R-ICU", "R-ED"])
    make("WCH", "wheelchair", "Wheelchair", 5, ["R-ED", "R-WARD", "R-STORE", "R-CT", "R-ED"])
    make("PMP", "pump", "Infusion pump", 5, ["R-WARD", "R-ICU"])
    audit("System", "Seeded locations and RFID-tagged assets")


def ingest_ping(tag_id, reader_code):
    """The one entry point for RFID events: simulator and real readers both call this."""
    asset = Asset.objects.filter(tag_id=tag_id).first()
    loc = Location.objects.filter(code=reader_code).first()
    if not asset or not loc:
        return None
    moved = asset.location_id != loc.id
    asset.location, asset.last_seen = loc, timezone.now()
    asset.save(update_fields=["location", "last_seen"])
    if moved:
        PingLog.objects.create(asset=asset, location=loc)
    return asset, moved


def admit_to_bed(bed, patient_ref=None):
    Encounter.objects.filter(bed=bed, status="in-progress").update(status="finished", ended_at=timezone.now())
    bed.status = "in_use"
    bed.save(update_fields=["status"])
    return Encounter.objects.create(patient_ref=patient_ref or f"P-{random.randint(1000, 9999)}", bed=bed)


def discharge_bed(bed):
    Encounter.objects.filter(bed=bed, status="in-progress").update(status="finished", ended_at=timezone.now())
    bed.status = "cleaning"
    bed.save(update_fields=["status"])


def sync_beds(state):
    # ref_by_bed = dict(Encounter.objects.filter(status="in-progress").values_list("bed_id", "patient_ref"))
    # for kind, key in (("ed", "ed_occupancy"), ("ward", "ward_occupancy"), ("icu", "icu_occupancy")):
    #     beds = list(Asset.objects.filter(asset_type="bed", location__kind=kind))
    #     target = round(min(state[key], 1.0) * len(beds))
    #     for b in beds:
    #         if b.status == "cleaning" and random.random() < 0.5:
    #             b.status = "free"
    #             b.save(update_fields=["status"])
    #     used = [b for b in beds if b.status == "in_use"]
    #     for b in used:
    #         if b.id not in ref_by_bed:
    #             Encounter.objects.create(patient_ref=f"P-{random.randint(1000, 9999)}", bed=b)
    #     free = [b for b in beds if b.status == "free"]
    #     if len(used) < target:
    #         for b in free[:target - len(used)]:
    #             admit_to_bed(b)
    #     elif len(used) > target:
    #         # discharge simulated patients (P-xxxx) before any admitted through the HL7 feed
    #         used.sort(key=lambda b: not ref_by_bed.get(b.id, "P-").startswith("P-"))
    #         for b in used[:len(used) - target]:
    #             discharge_bed(b)
    # old = Encounter.objects.filter(status="finished").order_by("-id").values_list("id", flat=True)[300:301]
    # if old:
    #     Encounter.objects.filter(status="finished", id__lte=old[0]).delete()

    admits = discharges = 0
    ref_by_bed = dict(Encounter.objects.filter(status="in-progress").values_list("bed_id", "patient_ref"))
    for kind, key in (("ed", "ed_occupancy"), ("ward", "ward_occupancy"), ("icu", "icu_occupancy")):
        beds = list(Asset.objects.filter(asset_type="bed", location__kind=kind).select_related("location"))
        target = round(min(state[key], 1.0) * len(beds))
        for b in beds:
            if b.status == "cleaning" and random.random() < 0.5:
                b.status = "free"
                b.save(update_fields=["status"])
        used = [b for b in beds if b.status == "in_use"]
        for b in used:
            if b.id not in ref_by_bed:
                ref = f"P-{random.randint(1000, 9999)}"
                Encounter.objects.create(patient_ref=ref, bed=b)
                ref_by_bed[b.id] = ref
        free = [b for b in beds if b.status == "free"]
        if len(used) < target:
            for b in free[:target - len(used)]:
                _, ok = send_hl7("A01", f"P-{random.randint(1000, 9999)}", b, log=False)
                admits += ok
        elif len(used) > target:
            # discharge simulated patients (P-xxxx) before any admitted by hand
            used.sort(key=lambda b: not ref_by_bed.get(b.id, "P-").startswith("P-"))
            for b in used[:len(used) - target]:
                _, ok = send_hl7("A03", ref_by_bed.get(b.id, "P-0000"), b, log=False)
                discharges += ok
    if admits or discharges:
        audit("HL7 feed (auto)", f"{admits} admit(s), {discharges} discharge(s) received")
    old = Encounter.objects.filter(status="finished").order_by("-id").values_list("id", flat=True)[300:301]
    if old:
        Encounter.objects.filter(status="finished", id__lte=old[0]).delete()


def simulate_rfid(state):
    """Stands in for real tags and readers. Every tag pings every cycle."""
    by_kind = {l.kind: l for l in Location.objects.all()}
    for a in Asset.objects.select_related("location"):
        reader = a.location.code if a.location else "R-STORE"
        if a.asset_type != "bed" and random.random() < 0.06:
            kind = random.choice(MOVES[a.asset_type])
            reader = by_kind[kind].code
            new_status = "free" if kind == "store" else "in_use"
            if a.status != new_status:
                a.status = new_status
                a.save(update_fields=["status"])
        ingest_ping(a.tag_id, reader)
    sync_beds(state)
    cut = PingLog.objects.order_by("-id").values_list("id", flat=True)[300:301]
    if cut:
        PingLog.objects.filter(id__lte=cut[0]).delete()


def surge_score(s):
    arrivals = min(s["arrival_rate"] / 12, 1)
    queues = min((s["ed_queue"] + s["ct_queue"] + 2 * s["critical_queue"]) / 20, 1)
    score = 0.35 * min(s["ed_occupancy"], 1) + 0.25 * arrivals + 0.25 * (1 - s["nurse_availability"]) + 0.15 * queues
    return int(round(100 * score))


# def propose_action(s, cur):
#     """Rule-based policy. Replace the body with the RL model call when your teammate is ready,
#     and keep the rules as the fallback. Returns (action, reasons, risk) or None."""
#     a, why = dict(cur), []
#     occ, q = s["ed_occupancy"], s["ed_queue"]
#     if occ > 0.9 or q > 5:
#         want = min(10, 2 + q // 2)
#         if s["nurse_availability"] < 0.25:
#             want = min(want, 2)
#             why.append("Nurse availability is low, so surge beds are capped at 2")
#         a["ed_surge_beds"] = want
#         why.append(f"ED is {occ:.0%} full with {q} waiting: open {want} surge beds")
#     elif occ < 0.7 and q == 0 and cur["ed_surge_beds"] > 0:
#         a["ed_surge_beds"] = 0
#         why.append("ED pressure has eased: close surge beds")
#     icu = s["icu_occupancy"]
#     if icu > 0.85 or s["critical_queue"] > 0:
#         a["elective_action"], a["icu_reserve"] = "defer", 3
#         why.append(f"ICU is {icu:.0%} full: defer elective surgery and hold 3 ICU beds")
#     elif icu < 0.7 and s["elective_queue"] > 0:
#         a["elective_action"], a["icu_reserve"] = "start", 2
#         why.append("ICU has room: resume elective surgery")
#     if s["ct_queue"] >= 3 and occ > 0.8:
#         a["ct_priority"] = 0.8
#         why.append(f"CT queue is {s['ct_queue']}: give the ED 80% CT priority")
#     elif s["ct_queue"] == 0 and cur["ct_priority"] != 0.5:
#         a["ct_priority"] = 0.5
#         why.append("CT queue is clear: rebalance CT priority")
#     if a == cur:
#         return None
#     jump = a["ed_surge_beds"] - cur["ed_surge_beds"]
#     deferring = a["elective_action"] == "defer" and cur["elective_action"] != "defer"
#     risk = "high" if (deferring or jump >= 4) else ("medium" if jump else "low")
#     return a, why, risk





def maybe_propose(engine, state):
    if Proposal.objects.filter(status="pending").exists():
        return
    last = Proposal.objects.exclude(status="pending").order_by("-decided_at").first()
    if last and last.decided_at:
        wait = COOLDOWN * (4 if last.status == "declined" else 1)   # don't re-nag right after a decline
        if (timezone.now() - last.decided_at).total_seconds() < wait:
            return
    rec = engine.recommend()
    if not rec:
        return
    action, reasons, risk, source = rec
    p = Proposal.objects.create(action=action, reasons=reasons, risk=risk, source=source)
    audit(f"AI ({source})", f"Proposal #{p.id} created, {risk} risk")


def escalate():
    now = timezone.now()
    for p in Proposal.objects.filter(status="pending"):
        age = (now - p.created_at).total_seconds()
        if age > EXPIRE_AFTER:
            p.status, p.decided_at = "expired", now
            p.save()
            audit("System", f"Proposal #{p.id} expired")
            continue
        level = min(int(age // ESCALATE_AFTER), len(ROLES) - 1)
        if level > p.escalation_level:
            p.escalation_level = level
            p.save(update_fields=["escalation_level"])
            audit("System", f"Proposal #{p.id} escalated to {ROLES[level]}")


def _run():
    from .clinical import run_cycle
    time.sleep(3)
    while True:
        try:
            close_old_connections()
            seed()
            engine = get_engine()
            state = engine.get_state()
            simulate_rfid(state)
            maybe_propose(engine, state)
            run_cycle(state)
            escalate()
        except Exception:
            log.warning("autopilot cycle failed", exc_info=True)
        time.sleep(TICK)


def start():
    global _started
    if not _started:
        _started = True
        threading.Thread(target=_run, daemon=True, name="careflow-autopilot").start()


def send_hl7(event, patient, bed, log=True):
    """Builds a real HL7 v2 ADT message and runs it through the same parser as POST /api/hl7."""
    from .hl7 import process   # imported here to avoid a circular import
    ts = timezone.now().strftime("%Y%m%d%H%M%S")
    unit = bed.location.code.replace("R-", "") if bed.location else "ED"
    msg = "\r".join([
        f"MSH|^~\\&|HIS|CITYHOSP|CAREFLOW|CITYHOSP|{ts}||ADT^{event}|{random.randint(10000, 99999)}|P|2.5",
        f"PID|1||{patient}||SYNTHETIC^PATIENT",
        f"PV1|1|I|{unit}^{bed.tag_id}",
    ])
    return process(msg, actor="HL7 feed (auto)", log=log)