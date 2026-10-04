from django.urls import path

from . import fhir
from .security import guard

staff = guard(("md", "admin"), True)
md_only = guard(("md",), True)

urlpatterns = [
    path("metadata", staff(fhir.metadata)),
    path("Location", staff(fhir.location_list)),
    path("Location/<str:rid>", staff(fhir.location_read)),
    path("Device", staff(fhir.device_list)),
    path("Device/<str:rid>", staff(fhir.device_read)),
    path("Encounter", md_only(fhir.encounter_list)),
    path("Encounter/<str:rid>", md_only(fhir.encounter_read)),
    path("Patient/<str:rid>", md_only(fhir.patient_read)),
]