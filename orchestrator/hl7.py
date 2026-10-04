from django.http import HttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from .automation import admit_to_bed, audit, discharge_bed
from .models import Asset, Encounter


def parse(raw):
    """Split an HL7 v2 message into segments, e.g. segs['PID'] is the PID line split on '|'."""
    segs = {}
    for line in raw.replace("\r\n", "\r").replace("\n", "\r").split("\r"):
        line = line.strip()
        if line:
            parts = line.split("|")
            segs.setdefault(parts[0], parts)
    return segs


def ack(code, event, ctrl, text=""):
    ts = timezone.now().strftime("%Y%m%d%H%M%S")
    return "\r".join([f"MSH|^~\\&|CAREFLOW|HOSPITAL|HIS|HOSPITAL|{ts}||ACK^{event}|{ctrl}|P|2.5",
                      f"MSA|{code}|{ctrl}|{text}"])


def apply_adt(event, patient, bed_tag):
    bed = Asset.objects.filter(tag_id=bed_tag, asset_type="bed").first() if bed_tag else None
    open_enc = Encounter.objects.filter(patient_ref=patient, status="in-progress").select_related("bed").first()
    if event in ("A01", "A04"):
        if not bed:
            raise ValueError(f"unknown bed {bed_tag}")
        admit_to_bed(bed, patient)
        return f"admitted {patient} to {bed.name}"
    if event == "A02":
        if not bed:
            raise ValueError(f"unknown bed {bed_tag}")
        if open_enc and open_enc.bed:
            discharge_bed(open_enc.bed)
        admit_to_bed(bed, patient)
        return f"transferred {patient} to {bed.name}"
    if event == "A03":
        target = open_enc.bed if open_enc and open_enc.bed else bed
        if not target:
            raise ValueError("no bed to discharge")
        discharge_bed(target)
        return f"discharged {patient} from {target.name}, bed needs cleaning"
    raise ValueError(f"event {event} is not supported")


def process(raw, actor="HL7 feed", log=True):
    """Parse and apply one ADT message. Returns (ack_text, ok). Used by the web endpoint and the auto feed."""
    event, ctrl = "A00", "0"
    try:
        segs = parse(raw)
        mtype = segs["MSH"][8].split("^")
        ctrl = segs["MSH"][9]
        if mtype[0] != "ADT":
            raise ValueError("only ADT messages are supported")
        event = mtype[1]
        patient = segs["PID"][3].split("^")[0]
        bed_tag = None
        if "PV1" in segs and len(segs["PV1"]) > 3 and segs["PV1"][3]:
            loc = segs["PV1"][3].split("^")
            bed_tag = loc[1] if len(loc) > 1 else loc[0]
        result = apply_adt(event, patient, bed_tag)
    except (KeyError, IndexError, ValueError) as exc:
        return ack("AE", event, ctrl, str(exc)), False
    if log:
        audit(actor, f"ADT^{event}: {result}")
    return ack("AA", event, ctrl, result), True


@csrf_exempt
def receive(request):
    if request.method != "POST":
        return HttpResponse("POST an HL7 v2 ADT message as plain text", status=405, content_type="text/plain")
    text, ok = process(request.body.decode("utf-8", "ignore"))
    return HttpResponse(text, content_type="text/plain", status=200 if ok else 400)