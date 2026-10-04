from django.urls import path
from django.views.decorators.csrf import csrf_exempt

from .hl7 import receive as hl7_receive
from .security import guard
from .staff_views import DoctorBoard, DoctorDecide, MdOverview, NurseBoard, NurseTaskStatus
from .views import (ActionView, DecideView, LiveView, MetricsView, PingView,
                    ResetView, ScenarioView, StateView)

urlpatterns = [
    path("state", StateView.as_view()),
    path("action", ActionView.as_view()),
    path("metrics", MetricsView.as_view()),
    path("reset", ResetView.as_view()),
    path("live", LiveView.as_view()),
    path("scenario", ScenarioView.as_view()),
    path("proposals/<int:pk>/decide", DecideView.as_view()),
    path("rtls/ping", PingView.as_view()),
    path("hl7", csrf_exempt(guard(roles_allowed=(), allow_key=True)(hl7_receive))),   # registration feed: API key only
    path("doctor/board", DoctorBoard.as_view()),
    path("doctor/suggestions/<int:pk>/decide", DoctorDecide.as_view()),
    path("nurse/board", NurseBoard.as_view()),
    path("nurse/tasks/<int:pk>/status", NurseTaskStatus.as_view()),
    path("md/overview", MdOverview.as_view()),
]