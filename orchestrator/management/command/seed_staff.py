import secrets

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from orchestrator.models import StaffProfile

PEOPLE = [
    ("md", "md.menon", "Dr. Rajesh Menon", 0),
    ("admin", "admin.natarajan", "Priya Natarajan", 0),
    ("doctor", "dr.rao", "Dr. Anita Rao", 0), ("doctor", "dr.shah", "Dr. Vikram Shah", 1),
    ("doctor", "dr.iyer", "Dr. Meera Iyer", 2), ("doctor", "dr.nair", "Dr. Arjun Nair", 3),
    ("doctor", "dr.khan", "Dr. Sana Khan", 4),
    ("nurse", "nr.lakshmi", "Lakshmi Devi", 0), ("nurse", "nr.joseph", "Joseph Thomas", 1),
    ("nurse", "nr.kavya", "Kavya Reddy", 2), ("nurse", "nr.farah", "Farah Ali", 3),
    ("nurse", "nr.ramesh", "Ramesh Kumar", 4), ("nurse", "nr.divya", "Divya Menon", 5),
    ("nurse", "nr.suresh", "Suresh Pillai", 6), ("nurse", "nr.anjali", "Anjali Das", 7),
    ("nurse", "nr.mary", "Mary George", 8), ("nurse", "nr.imran", "Imran Sheikh", 9),
]


class Command(BaseCommand):
    help = "Creates the staff accounts with random passwords, printed once."

    def handle(self, *args, **options):
        User = get_user_model()
        for role, username, name, seq in PEOPLE:
            if User.objects.filter(username=username).exists():
                self.stdout.write(f"skip {username} (already exists)")
                continue
            pw = secrets.token_urlsafe(9)
            user = User.objects.create_user(username=username, password=pw)
            StaffProfile.objects.create(user=user, role=role, full_name=name, seq=seq)
            self.stdout.write(f"{role:7} {username:18} {pw}")