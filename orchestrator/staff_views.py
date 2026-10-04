from datetime import timedelta

from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .automation import audit, send_hl7
from .engine import get_engine
from .models import Asset, Encounter, NurseTask, StaffProfile, Suggestion
from .roster import doctors_on, nurses_on, owner_of, progress, slot_info
from .security import mask_ref, roles


def _ctx():
    t = get_engine().get_state()["time"]
    return t, doctors_on(t), nurses_on(t)


def _hm(dt):
    return timezone.localtime(dt).strftime("%H:%M:%S")


def _bed_info(e):
    loc = e.bed.location.name if e.bed and e.bed.location else "-"
    return (e.bed.name if e.bed else "-"), loc


def _free_equipment(kind):
    a = Asset.objects.filter(asset_type=kind, status="free").select_related("location").first()
    return f"{a.name} ({a.location.name})" if a and a.location else ""


def _active_encounters():
    return Encounter.objects.filter(status="in-progress", bed__isnull=False).select_related("bed__location")


class DoctorBoard(APIView):
    permission_classes = [roles("doctor")]

    def get(self, request):
        me = request.user.profile
        t, docs, nurses = _ctx()
        base = {"name": me.full_name, "shift": slot_info(t)}
        if me not in docs:
            return Response({**base, "on_duty": False})
        i = docs.index(me)
        team = nurses[2 * i: 2 * i + 2]
        load = {n.id: NurseTask.objects.filter(assigned_to=n, status__in=["todo", "doing"]).count() for n in team}
        suggested = min(team, key=lambda n: load[n.id]) if team else None
        mine, by_nurse = [], {n.id: [] for n in team}
        for e in _active_encounters():
            d, n = owner_of(e.id, docs, nurses)
            if d == me:
                bed, unit = _bed_info(e)
                row = {"id": e.id, "ref": e.patient_ref, "bed": bed, "unit": unit, "acuity": e.acuity,
                       "progress": int(progress(e) * 100), "nurse": n.full_name, "since": _hm(e.started_at)}
                mine.append(row)
                if n.id in by_nurse:
                    by_nurse[n.id].append(row)
        ids = {r["id"] for r in mine}
        sugg = [{"id": s.id, "patient": s.encounter.patient_ref, "bed": _bed_info(s.encounter)[0],
                 "unit": _bed_info(s.encounter)[1], "acuity": s.encounter.acuity,
                 "progress": int(progress(s.encounter) * 100), "reasons": s.reasons,
                 "suggested_nurse": suggested.id if suggested else None}
                for s in Suggestion.objects.filter(status="pending", encounter_id__in=ids).select_related("encounter__bed__location")]
        cards = [{"id": n.id, "name": n.full_name, "open_tasks": load[n.id],
                  "patients": [{"bed": p["bed"], "unit": p["unit"], "ref": p["ref"]} for p in by_nurse[n.id]],
                  "tasks": [{"title": k.title, "status": k.status, "to": k.to_loc}
                            for k in NurseTask.objects.filter(assigned_to=n, status__in=["todo", "doing"])[:4]]} for n in team]
        recent = [{"title": k.title, "nurse": k.assigned_to.full_name, "status": k.status, "at": _hm(k.created_at)}
                  for k in NurseTask.objects.filter(assigned_to__in=team).select_related("assigned_to").order_by("-id")[:8]]
        st = get_engine().get_state()
        free = {}
        for kind in Asset.objects.filter(asset_type="bed", status="free").values_list("location__kind", flat=True):
            free[kind] = free.get(kind, 0) + 1
        return Response({**base, "on_duty": True, "nurses": cards, "patients": mine, "suggestions": sugg, "recent": recent,
                         "situation": {"critical_queue": st["critical_queue"], "ed_queue": st["ed_queue"],
                                       "free_ed": free.get("ed", 0), "free_icu": free.get("icu", 0)}})


class DoctorDecide(APIView):
    permission_classes = [roles("doctor")]

    def post(self, request, pk):
        me = request.user.profile
        t, docs, nurses = _ctx()
        if me not in docs:
            return Response({"error": "You are off shift."}, status=status.HTTP_403_FORBIDDEN)
        s = Suggestion.objects.filter(pk=pk, status="pending").select_related("encounter__bed__location").first()
        if not s:
            return Response({"error": "Suggestion is no longer pending."}, status=status.HTTP_404_NOT_FOUND)
        if owner_of(s.encounter_id, docs, nurses)[0] != me:
            return Response({"error": "Not your patient."}, status=status.HTTP_403_FORBIDDEN)
        decision = request.data.get("decision")
        if decision not in ("accept", "decline"):
            return Response({"error": "decision must be accept or decline"}, status=status.HTTP_400_BAD_REQUEST)
        s.decided_at, s.decided_by = timezone.now(), me.full_name
        if decision == "decline":
            s.status = "declined"
            s.save()
            audit(me.full_name, f"Declined suggestion #{s.id}")
            return Response({"status": "declined"})
        i = docs.index(me)
        team = nurses[2 * i: 2 * i + 2]
        try:
            want = int(request.data.get("nurse_id") or 0)
        except (TypeError, ValueError):
            want = 0
        chosen = next((n for n in team if n.id == want), team[0] if team else None)
        if not chosen:
            return Response({"error": "No nurse available on your team."}, status=status.HTTP_409_CONFLICT)
        e = s.encounter
        bed, unit = _bed_info(e)
        NurseTask.objects.create(kind="discharge", title=f"Discharge preparation, {bed}",
                                 detail="Patient has recovered. Complete the paperwork, then move the patient out.",
                                 encounter=e, from_loc=f"{bed}, {unit}", to_loc="Discharge area",
                                 equipment=_free_equipment("wheelchair"), assigned_to=chosen,
                                 assigned_by=me.full_name, priority=2)
        other = next((n for n in team if n != chosen), None)
        if other:
            NurseTask.objects.create(kind="prepare_bed", title=f"Prepare {bed} for a critical patient",
                                     detail="Clean and set up the bed as soon as it is vacated.", encounter=e,
                                     from_loc="", to_loc=f"{bed}, {unit}", equipment=_free_equipment("pump"),
                                     assigned_to=other, assigned_by=me.full_name, priority=1)
        try:
            unit_kind = "icu" if e.bed and e.bed.location and e.bed.location.kind == "icu" else "ward"
            get_engine().expedite(unit_kind, 1)
        except Exception:
            pass
        s.status = "accepted"
        s.save()
        audit(me.full_name, f"Accepted suggestion #{s.id}, work assigned to {chosen.full_name}")
        return Response({"status": "accepted", "assigned_to": chosen.full_name})


class NurseBoard(APIView):
    permission_classes = [roles("nurse")]

    def get(self, request):
        me = request.user.profile
        t, docs, nurses = _ctx()
        base = {"name": me.full_name, "shift": slot_info(t)}
        if me not in nurses:
            return Response({**base, "on_duty": False})
        doc = docs[min(nurses.index(me) // 2, len(docs) - 1)] if docs else None
        mine = []
        for e in _active_encounters():
            if owner_of(e.id, docs, nurses)[1] == me:
                mine.append(e)
        patients = [{"ref": mask_ref(e.patient_ref), "bed": _bed_info(e)[0], "unit": _bed_info(e)[1],
                     "acuity": e.acuity, "progress": int(progress(e) * 100)} for e in mine]
        cutoff = timezone.now() - timedelta(minutes=15)
        hint = lambda a: "Ventilator, infusion pump" if a <= 2 else ("Infusion pump" if a == 3 else "Wheelchair")
        incoming = [{"at": _hm(e.started_at), "ref": mask_ref(e.patient_ref), "to": f"{_bed_info(e)[0]}, {_bed_info(e)[1]}",
                     "acuity": e.acuity, "equipment": hint(e.acuity)}
                    for e in sorted(mine, key=lambda x: x.started_at, reverse=True) if e.started_at >= cutoff][:8]
        tasks = [{"id": k.id, "title": k.title, "detail": k.detail, "from": k.from_loc, "to": k.to_loc,
                  "equipment": k.equipment, "priority": k.priority, "status": k.status,
                  "patient": mask_ref(k.encounter.patient_ref) if k.encounter else "", "by": k.assigned_by}
                 for k in NurseTask.objects.filter(assigned_to=me).select_related("encounter").order_by("status", "-priority", "-id")[:12]]
        near = {p["unit"] for p in patients}
        gear = Asset.objects.exclude(asset_type="bed").filter(status="free").select_related("location")
        equipment = sorted(({"name": a.name, "type": a.asset_type, "where": a.location.name if a.location else "-"} for a in gear),
                           key=lambda g: g["where"] not in near)[:10]
        return Response({**base, "on_duty": True, "doctor": doc.full_name if doc else "-", "patients": patients,
                         "incoming": incoming, "tasks": tasks, "equipment": equipment})


class NurseTaskStatus(APIView):
    permission_classes = [roles("nurse")]

    def post(self, request, pk):
        me = request.user.profile
        task = NurseTask.objects.filter(pk=pk, assigned_to=me).select_related("encounter__bed__location").first()
        new = request.data.get("status")
        if not task or new not in ("doing", "done"):
            return Response({"error": "invalid request"}, status=status.HTTP_400_BAD_REQUEST)
        task.status = new
        if new == "done":
            task.done_at = timezone.now()
            e = task.encounter
            if task.kind == "discharge" and e and e.bed and e.status == "in-progress":
                try:
                    send_hl7("A03", e.patient_ref, e.bed)     # registers the discharge through the HL7 path
                except Exception:
                    pass
        task.save()
        audit(me.full_name, f"Task #{task.id} marked {new}")
        return Response({"status": new})


class MdOverview(APIView):
    permission_classes = [roles("md")]

    def get(self, request):
        t, docs, nurses = _ctx()
        pair = {}
        for i, d in enumerate(docs):
            pair[d.id] = [n.full_name for n in nurses[2 * i: 2 * i + 2]]
        people = lambda role, on: [{"name": p.full_name, "on": p in on, "nurses": pair.get(p.id, [])}
                                   for p in StaffProfile.objects.filter(role=role).order_by("seq", "id")]
        sugg = [{"id": s.id, "patient": s.encounter.patient_ref, "status": s.status, "by": s.decided_by, "at": _hm(s.created_at)}
                for s in Suggestion.objects.select_related("encounter").order_by("-id")[:8]]
        tasks = [{"title": k.title, "nurse": k.assigned_to.full_name, "status": k.status, "by": k.assigned_by}
                 for k in NurseTask.objects.select_related("assigned_to").order_by("-id")[:8]]
        return Response({"shift": slot_info(t), "doctors": people("doctor", docs), "nurses": people("nurse", nurses),
                         "suggestions": sugg, "tasks": tasks})