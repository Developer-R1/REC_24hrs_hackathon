from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .automation import ROLES, TICK, audit, ingest_ping, surge_score
from .engine import get_engine
from .models import Asset, AuditLog, Location, PingLog, Proposal
from .security import ApiKeyPermission, mask_text, role_of, roles
from .serializers import ActionSerializer


class StateView(APIView):
    permission_classes = [roles("admin", "md")]

    def get(self, request):
        return Response(get_engine().get_state())


class MetricsView(APIView):
    permission_classes = [roles("admin", "md")]

    def get(self, request):
        return Response(get_engine().get_metrics())


class ActionView(APIView):
    def get_permissions(self):
        return [(roles("md") if self.request.method == "POST" else roles("admin", "md"))()]

    def get(self, request):
        return Response(get_engine().get_action())

    def post(self, request):
        serializer = ActionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        applied = get_engine().apply_action(serializer.validated_data)
        audit(f"{request.user.profile.full_name} (MD)", f"Manual action override: {applied}")
        return Response({"status": "applied", "action": applied})


class ResetView(APIView):
    permission_classes = [roles("md")]

    def post(self, request):
        get_engine().reset()
        audit(f"{request.user.profile.full_name} (MD)", "Simulation reset")
        return Response({"status": "reset"})


class LiveView(APIView):
    """One call that feeds the operations dashboard (Administrator and MD)."""
    permission_classes = [roles("admin", "md")]

    def get(self, request):
        engine, now, role = get_engine(), timezone.now(), role_of(request.user)
        state = engine.get_state()
        extras = engine.get_extras()
        assets = [{"tag": a.tag_id, "name": a.name, "type": a.asset_type, "status": a.status,
                   "loc": a.location.code if a.location else None,
                   "seen": int((now - a.last_seen).total_seconds()) if a.last_seen else None}
                  for a in Asset.objects.select_related("location")]
        pending = [{"id": p.id, "action": p.action, "reasons": p.reasons, "risk": p.risk, "source": p.source,
                    "role": ROLES[min(p.escalation_level, len(ROLES) - 1)]}
                   for p in Proposal.objects.filter(status="pending")]
        feed = [{"at": timezone.localtime(p.at).strftime("%H:%M:%S"), "asset": p.asset.name, "to": p.location.name}
                for p in PingLog.objects.select_related("asset", "location").order_by("-id")[:12]]
        level = "full" if role == "md" else "partial"          # masking: the Administrator never sees full patient IDs
        audit_rows = [{"at": timezone.localtime(a.at).strftime("%H:%M:%S"), "actor": a.actor, "msg": mask_text(a.message, level)}
                      for a in AuditLog.objects.order_by("-id")[:10]]
        return Response({
            "state": state, "metrics": engine.get_metrics(), "action": engine.get_action(), "extras": extras,
            "surge_score": surge_score(state), "assets": assets, "proposals": pending,
            "locations": list(Location.objects.values("code", "name", "kind", "gx", "gz")),
            "feed": feed, "audit": audit_rows, "tick": TICK,
        })


class PingView(APIView):
    """RFID readers post here with the X-API-Key header."""
    authentication_classes = []
    permission_classes = [ApiKeyPermission]

    def post(self, request):
        res = ingest_ping(request.data.get("tag_id"), request.data.get("reader_id"))
        if not res:
            return Response({"error": "unknown tag or reader"}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"status": "ok", "moved": res[1]})


class DecideView(APIView):
    """Administrator or MD approves or declines an AI proposal. The approver is taken from the session, never from the request."""
    permission_classes = [roles("admin", "md")]

    def post(self, request, pk):
        p = Proposal.objects.filter(pk=pk, status="pending").first()
        if not p:
            return Response({"error": "proposal is not pending"}, status=status.HTTP_404_NOT_FOUND)
        decision = request.data.get("decision")
        if decision not in ("approve", "decline"):
            return Response({"error": "decision must be approve or decline"}, status=status.HTTP_400_BAD_REQUEST)
        who = f"{request.user.profile.full_name} ({request.user.profile.role})"
        if decision == "approve":
            get_engine().apply_action(p.action)
        p.status = "approved" if decision == "approve" else "declined"
        p.decided_by, p.decided_at = who, timezone.now()
        p.save()
        audit(who, f"Proposal #{p.id} {p.status}")
        return Response({"status": p.status})


class ScenarioView(APIView):
    permission_classes = [roles("admin", "md")]

    def post(self, request):
        kind = request.data.get("kind", "mass_casualty")
        try:
            n = int(request.data.get("n", 30))
            get_engine().inject(kind, n)
        except (ValueError, TypeError) as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        audit("Scenario", f"{kind}: {n} patients arriving at the ED")
        return Response({"status": "injected", "kind": kind, "n": n})