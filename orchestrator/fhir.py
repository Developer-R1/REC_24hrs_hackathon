from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .models import Asset, Encounter, Location

V2_STATUS = "http://terminology.hl7.org/CodeSystem/v2-0116"
LOC_TYPE = "http://terminology.hl7.org/CodeSystem/location-physical-type"
BED_STATUS = {"free": ("U", "Unoccupied"), "in_use": ("O", "Occupied"), "cleaning": ("H", "Housekeeping")}


def room_to_fhir(loc):
    return {"resourceType": "Location", "id": loc.code, "status": "active", "name": loc.name,
            "physicalType": {"coding": [{"system": LOC_TYPE, "code": "wa", "display": "Ward"}]}}


def bed_to_fhir(a):
    code, disp = BED_STATUS.get(a.status, ("U", "Unoccupied"))
    out = {"resourceType": "Location", "id": a.tag_id, "status": "active", "name": a.name,
           "identifier": [{"system": "urn:careflow:rfid", "value": a.tag_id}],
           "operationalStatus": {"system": V2_STATUS, "code": code, "display": disp},
           "physicalType": {"coding": [{"system": LOC_TYPE, "code": "bd", "display": "Bed"}]}}
    if a.location:
        out["partOf"] = {"reference": f"Location/{a.location.code}", "display": a.location.name}
    return out


def device_to_fhir(a):
    out = {"resourceType": "Device", "id": a.tag_id, "status": "active",
           "identifier": [{"system": "urn:careflow:rfid", "value": a.tag_id}],
           "deviceName": [{"name": a.name, "type": "user-friendly-name"}],
           "type": {"text": a.asset_type},
           "note": [{"text": f"availability: {a.status}"}]}
    if a.location:
        out["location"] = {"reference": f"Location/{a.location.code}", "display": a.location.name}
    if a.last_seen:
        out["meta"] = {"lastUpdated": a.last_seen.isoformat()}
    return out


def encounter_to_fhir(e):
    area = e.bed.location.kind if e.bed and e.bed.location else "ward"
    out = {"resourceType": "Encounter", "id": str(e.id), "status": e.status,
           "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                     "code": "EMER" if area == "ed" else "IMP"},
           "subject": {"reference": f"Patient/{e.patient_ref}"},
           "period": {"start": e.started_at.isoformat()}}
    if e.ended_at:
        out["period"]["end"] = e.ended_at.isoformat()
    if e.bed:
        out["location"] = [{"location": {"reference": f"Location/{e.bed.tag_id}", "display": e.bed.name},
                            "status": "active" if e.status == "in-progress" else "completed"}]
    return out


def _bundle(resources):
    return {"resourceType": "Bundle", "type": "searchset", "total": len(resources),
            "entry": [{"resource": r} for r in resources]}


def _missing(kind, rid):
    return JsonResponse({"resourceType": "OperationOutcome", "issue": [
        {"severity": "error", "code": "not-found", "diagnostics": f"{kind}/{rid} not found"}]}, status=404)


@require_GET
def metadata(request):
    kinds = ("Location", "Device", "Encounter", "Patient")
    return JsonResponse({"resourceType": "CapabilityStatement", "status": "active", "kind": "instance",
                         "fhirVersion": "4.0.1", "format": ["json"], "software": {"name": "CareFlow AI"},
                         "rest": [{"mode": "server", "resource": [
                             {"type": k, "interaction": [{"code": "read"}, {"code": "search-type"}]} for k in kinds]}]})


@require_GET
def location_list(request):
    rooms = [room_to_fhir(l) for l in Location.objects.all()]
    beds = Asset.objects.filter(asset_type="bed").select_related("location")
    flag = request.GET.get("operational-status")
    if flag:
        wanted = set(flag.split(","))
        beds = [b for b in beds if BED_STATUS[b.status][0] in wanted]
        rooms = []
    return JsonResponse(_bundle(rooms + [bed_to_fhir(b) for b in beds]))


@require_GET
def location_read(request, rid):
    loc = Location.objects.filter(code=rid).first()
    if loc:
        return JsonResponse(room_to_fhir(loc))
    bed = Asset.objects.filter(tag_id=rid, asset_type="bed").select_related("location").first()
    return JsonResponse(bed_to_fhir(bed)) if bed else _missing("Location", rid)


@require_GET
def device_list(request):
    qs = Asset.objects.exclude(asset_type="bed").select_related("location")
    if request.GET.get("type"):
        qs = qs.filter(asset_type=request.GET["type"])
    if request.GET.get("location"):
        qs = qs.filter(location__code=request.GET["location"])
    return JsonResponse(_bundle([device_to_fhir(a) for a in qs]))


@require_GET
def device_read(request, rid):
    a = Asset.objects.filter(tag_id=rid).exclude(asset_type="bed").select_related("location").first()
    return JsonResponse(device_to_fhir(a)) if a else _missing("Device", rid)


@require_GET
def encounter_list(request):
    qs = Encounter.objects.select_related("bed__location").order_by("-id")
    if request.GET.get("status"):
        qs = qs.filter(status=request.GET["status"])
    return JsonResponse(_bundle([encounter_to_fhir(e) for e in qs[:100]]))


@require_GET
def encounter_read(request, rid):
    e = Encounter.objects.select_related("bed__location").filter(pk=rid if rid.isdigit() else 0).first()
    return JsonResponse(encounter_to_fhir(e)) if e else _missing("Encounter", rid)


@require_GET
def patient_read(request, rid):
    if not Encounter.objects.filter(patient_ref=rid).exists():
        return _missing("Patient", rid)
    return JsonResponse({"resourceType": "Patient", "id": rid,
                         "identifier": [{"system": "urn:careflow:synthetic", "value": rid}],
                         "name": [{"text": "Synthetic patient"}]})