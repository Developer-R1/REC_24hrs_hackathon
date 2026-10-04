import os
import sys

from django.apps import AppConfig


class OrchestratorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "orchestrator"

    def ready(self):
        cmd = sys.argv[1] if len(sys.argv) > 1 else ""
        if cmd in ("migrate", "makemigrations", "collectstatic", "shell", "test", "check"):
            return
        if cmd == "runserver" and os.environ.get("RUN_MAIN") != "true":
            return  # skip the autoreloader's parent process
        from .automation import start
        start()