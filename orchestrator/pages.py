import json
from pathlib import Path

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.middleware.csrf import get_token
from django.shortcuts import redirect
from django.utils.html import escape
from django.views.decorators.http import require_POST

from .security import role_of

PAGES = Path(settings.BASE_DIR) / "orchestrator" / "templates"
HOME = {"md": "/md/", "admin": "/ops/", "doctor": "/doctor/", "nurse": "/nurse/"}
MAX_FAILS, LOCK_SECONDS = 5, 300


def _page(name, allowed):
    def view(request):
        role = role_of(request.user)
        if role is None:
            return redirect("/login/")
        if role not in allowed:
            return redirect(HOME[role])
        html = (PAGES / name).read_text(encoding="utf-8")
        html = (html.replace("__CSRF__", get_token(request)).replace("__ROLE__", role)
                    .replace("__NAME__", escape(request.user.profile.full_name)))
        return HttpResponse(html)
    return view


md_page = _page("dashboard.html", {"md"})
admin_page = _page("dashboard.html", {"admin"})
doctor_page = _page("doctor.html", {"doctor"})
nurse_page = _page("nurse.html", {"nurse"})


def home(request):
    role = role_of(request.user)
    return redirect(HOME[role] if role else "/login/")


def _ip(request):
    fwd = request.META.get("HTTP_X_FORWARDED_FOR")
    return fwd.split(",")[0].strip() if fwd else request.META.get("REMOTE_ADDR", "")


def login_page(request):
    from .automation import audit
    if request.method == "POST":
        try:
            data = json.loads(request.body or b"{}")
        except ValueError:
            data = {}
        username, password = str(data.get("username", "")).strip()[:150], str(data.get("password", ""))
        key = f"fails:{username.lower()}:{_ip(request)}"
        fails = cache.get(key, 0)
        if fails >= MAX_FAILS:
            return JsonResponse({"error": "Too many failed attempts. Try again in 5 minutes."}, status=429)
        user = authenticate(request, username=username, password=password)
        if user is None or role_of(user) is None:
            cache.set(key, fails + 1, LOCK_SECONDS)
            audit("Security", f"Failed login attempt for '{username[:30]}'")
            return JsonResponse({"error": "Invalid username or password."}, status=400)
        cache.delete(key)
        login(request, user)
        audit("Security", f"{user.profile.full_name} logged in")
        return JsonResponse({"next": HOME[user.profile.role]})
    role = role_of(request.user)
    if role:
        return redirect(HOME[role])
    return HttpResponse((PAGES / "login.html").read_text(encoding="utf-8").replace("__CSRF__", get_token(request)))


@require_POST
def logout_view(request):
    from .automation import audit
    if role_of(request.user):
        audit("Security", f"{request.user.profile.full_name} logged out")
    logout(request)
    return JsonResponse({"status": "ok"})