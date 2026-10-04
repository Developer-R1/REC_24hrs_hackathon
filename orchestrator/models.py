import random

from django.conf import settings
from django.db import models

class Location(models.Model):
    code = models.CharField(max_length=20, unique=True)   
    name = models.CharField(max_length=60)
    kind = models.CharField(max_length=10)                
    gx = models.IntegerField(default=0)                   
    gz = models.IntegerField(default=0)


class Asset(models.Model):
    tag_id = models.CharField(max_length=30, unique=True)  # RFID tag
    name = models.CharField(max_length=60)
    asset_type = models.CharField(max_length=12)           # bed, ventilator, wheelchair, pump
    location = models.ForeignKey(Location, null=True, on_delete=models.SET_NULL, related_name="assets")
    status = models.CharField(max_length=10, default="free")  # free, in_use, cleaning
    last_seen = models.DateTimeField(null=True, blank=True)


class PingLog(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    at = models.DateTimeField(auto_now_add=True)


class Proposal(models.Model):
    action = models.JSONField()
    reasons = models.JSONField(default=list)
    risk = models.CharField(max_length=10, default="low")
    source = models.CharField(max_length=10, default="rules")   # rules or rl
    status = models.CharField(max_length=10, default="pending") # pending, approved, declined, expired
    escalation_level = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    decided_by = models.CharField(max_length=40, blank=True)
    decided_at = models.DateTimeField(null=True, blank=True)


class AuditLog(models.Model):
    at = models.DateTimeField(auto_now_add=True)
    actor = models.CharField(max_length=40)
    message = models.CharField(max_length=255)


def _acuity():
    return random.choices([1, 2, 3, 4, 5], weights=[5, 12, 40, 30, 13])[0]


def _progress0():
    return round(random.uniform(0.0, 0.95), 2)


class Encounter(models.Model):
    patient_ref = models.CharField(max_length=30)          
    bed = models.ForeignKey(Asset, null=True, on_delete=models.SET_NULL, related_name="encounters")
    status = models.CharField(max_length=12, default="in-progress")   
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    acuity = models.IntegerField(default=_acuity)           
    progress0 = models.FloatField(default=_progress0)      
    expected_h = models.FloatField(default=48.0)         


class StaffProfile(models.Model):
    ROLES = [("md", "MD"), ("admin", "Administrator"), ("doctor", "Doctor"), ("nurse", "Nurse")]
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=10, choices=ROLES)
    full_name = models.CharField(max_length=80)
    seq = models.IntegerField(default=0)                    
    def __str__(self):
        return f"{self.full_name} ({self.role})"


class Suggestion(models.Model):
    kind = models.CharField(max_length=12, default="discharge")
    encounter = models.ForeignKey(Encounter, on_delete=models.CASCADE)
    reasons = models.JSONField(default=list)
    status = models.CharField(max_length=10, default="pending")   
    created_at = models.DateTimeField(auto_now_add=True)
    decided_at = models.DateTimeField(null=True, blank=True)
    decided_by = models.CharField(max_length=60, blank=True)


class NurseTask(models.Model):
    kind = models.CharField(max_length=12)                 
    title = models.CharField(max_length=120)
    detail = models.CharField(max_length=240, blank=True)
    encounter = models.ForeignKey(Encounter, null=True, on_delete=models.SET_NULL)
    from_loc = models.CharField(max_length=80, blank=True)
    to_loc = models.CharField(max_length=80, blank=True)
    equipment = models.CharField(max_length=80, blank=True)
    assigned_to = models.ForeignKey(StaffProfile, on_delete=models.CASCADE, related_name="tasks")
    assigned_by = models.CharField(max_length=80, blank=True)
    priority = models.IntegerField(default=2)
    status = models.CharField(max_length=8, default="todo")       
    created_at = models.DateTimeField(auto_now_add=True)
    done_at = models.DateTimeField(null=True, blank=True)  
