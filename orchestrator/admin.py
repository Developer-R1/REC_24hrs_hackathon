from django.contrib import admin

from .models import AuditLog, Encounter, NurseTask, StaffProfile, Suggestion


@admin.register(StaffProfile)
class StaffProfileAdmin(admin.ModelAdmin):
    list_display = ("full_name", "role", "seq", "user")
    list_filter = ("role",)


admin.site.register([AuditLog, Encounter, NurseTask, Suggestion])
