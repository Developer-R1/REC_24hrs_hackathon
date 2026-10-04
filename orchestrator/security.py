import hmac
import os
import re

from django.http import JsonResponse
from functools import wraps
from rest_framework.permissions import BasePermission


def role_of(user):
    """The user's role, or None when logged out, deactivated, or without a staff profile."""
    if not getattr(user, "is_authenticated", False) or not user.is_active:
        return None
    try:
        return user.profile.role
    except Exception:
        return None


def roles(*allowed):
    class _Allowed(BasePermission):
        def has_permission(self, request, view):
            return role_of(request.user) in allowed
    return _Allowed


def key_ok(request):
    key = os.getenv("EuTaxo_API_KEY", "")
    sent = request.META.get("HTTP_X_API_KEY", "")
    return bool(key) and hmac.compare_digest(key.encode(), sent.encode())


class ApiKeyPermission(BasePermission):
    def has_permission(self, request, view):
        return key_ok(request)


def guard(roles_allowed=("md", "admin"), allow_key=True):
    """For plain Django views (FHIR, HL7): allow logged-in staff of these roles, or a valid API key."""
    def deco(view):
        @wraps(view)
        def inner(request, *args, **kwargs):
            if role_of(request.user) in roles_allowed or (allow_key and key_ok(request)):
                return view(request, *args, **kwargs)
            return JsonResponse({"error": "forbidden"}, status=403)
        return inner
    return deco


# ---------------------------------------------------------------- masking
_REF = re.compile(r"\b(?:P-\d+|MRN\d+)\b")


def mask_ref(ref, level="partial"):
    if level == "full":
        return ref
    if level == "hidden" or len(ref) <= 4:
        return "****"
    return ref[:2] + "*" * (len(ref) - 4) + ref[-2:]


def mask_text(text, level="partial"):
    return _REF.sub(lambda m: mask_ref(m.group(0), level), text)


class SecurityHeadersMiddleware:
    """Stops browsers and proxies from caching sensitive pages, and adds basic protective headers."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        r = self.get_response(request)
        r["Cache-Control"] = "no-store, private"
        r["Pragma"] = "no-cache"
        r["X-Content-Type-Options"] = "nosniff"
        r["Referrer-Policy"] = "no-referrer"
        return r