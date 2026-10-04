import threading

_engine = None
_lock = threading.Lock()


def get_engine():
    """Creates the engine once, even if the autopilot thread and a web request ask at the same moment."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from .real import RealHospitalEngine
                _engine = RealHospitalEngine()
    return _engine