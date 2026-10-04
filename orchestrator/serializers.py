from rest_framework import serializers


class ActionSerializer(serializers.Serializer):
    ed_surge_beds = serializers.IntegerField(min_value=0, max_value=2)
    ward_expedite = serializers.IntegerField(min_value=0, max_value=1)
    icu_expedite = serializers.IntegerField(min_value=0, max_value=1)
    elective_protect = serializers.IntegerField(min_value=0, max_value=1)