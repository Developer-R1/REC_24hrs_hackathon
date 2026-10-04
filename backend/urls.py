# from pathlib import Path

# from django.conf import settings
# from django.contrib import admin
# from django.http import HttpResponse
# from django.urls import path, include


# def dashboard(request):
#     page = Path(settings.BASE_DIR) / "orchestrator" / "templates" / "dashboard.html"
#     return HttpResponse(page.read_text(encoding="utf-8"))


# urlpatterns = [
#     path("admin/", admin.site.urls),
#     path("api/", include("orchestrator.urls")),
#     path("fhir/", include("orchestrator.fhir_urls")),
#     path("", dashboard),
# ]


import os

from django.contrib import admin
from django.urls import include, path

from orchestrator import pages

urlpatterns = [
    path(os.environ.get("ADMIN_URL", "admin/"), admin.site.urls),   # set ADMIN_URL=some-secret-path/ in production
    path("api/", include("orchestrator.urls")),
    path("fhir/", include("orchestrator.fhir_urls")),
    path("login/", pages.login_page),
    path("logout/", pages.logout_view),
    path("md/", pages.md_page),
    path("ops/", pages.admin_page),
    path("doctor/", pages.doctor_page),
    path("nurse/", pages.nurse_page),
    path("", pages.home),
]