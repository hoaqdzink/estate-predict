import json

from functools import partial

from django.http import JsonResponse as _JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from . import services
from .forms import HouseForm

JsonResponse = partial(_JsonResponse)


def _model_error(request, exc):
    return render(request, "valuation/error.html", {"message": str(exc)}, status=503)


@require_http_methods(["GET", "POST"])
def index(request):
    try:
        city = request.POST.get("city") or request.GET.get("city") or services.CITIES[0]
        district_choices = services.districts(city if city in services.CITIES else services.CITIES[0])
    except services.ModelNotReady as exc:
        return _model_error(request, exc)
    submitted = request.POST if request.method == "POST" else (request.GET if "area_m2" in request.GET else None)
    form = HouseForm(submitted, initial={"city": city, "house_type": services.HOUSE_TYPES[0]})
    form.fields["district"].widget.choices = [(d, d) for d in district_choices]
    result = comps = None
    if submitted is not None and form.is_valid():
        d = form.cleaned_data
        result = services.predict(d["city"], d["district"], d["house_type"], d["area_m2"], d["road_width_m"],
                                  d["floors"], d["width_m"], d["has_parking"])
        comps = services.comparables(d["district"], d["house_type"], d["area_m2"])
        contrib = result["explanation"]["contributions"]
        top = max(abs(c["shap_log1p"]) for c in contrib) or 1
        for c in contrib:
            c["bar"] = round(abs(c["shap_log1p"]) / top * 50, 1)     # nửa thanh: 50% mỗi phía trục
        wmax = max(abs(m["weight"]) for m in result["stacking"]["base_models"]) or 1
        for m in result["stacking"]["base_models"]:
            m["wbar"] = round(max(m["weight"], 0) / wmax * 100)
        iv = result["interval_95"]
        result["range_pos"] = round((result["price_billion_vnd"] - iv["low"]) / (iv["high"] - iv["low"]) * 100, 1)
        icons = {"n_schools": "bi-mortarboard", "n_health": "bi-hospital", "n_markets": "bi-basket",
                 "n_parks": "bi-tree", "n_transit": "bi-bus-front"}
        result["amenities"] = [{"name": services.AMENITY_NAMES[k], "value": v, "icon": icons[k]}
                               for k, v in result["location"]["amenities_2km"].items()]
    return render(request, "valuation/index.html", {"form": form, "result": result, "comps": comps, "page": "index"})


@require_GET
def model_page(request):
    try:
        summary = services.model_summary()
    except services.ModelNotReady as exc:
        return _model_error(request, exc)
    return render(request, "valuation/model.html", {"s": summary, "page": "model"})


@require_GET
def api_docs(request):
    return render(request, "valuation/api.html", {"page": "api", "cities": services.CITIES,
                                                  "house_types": services.HOUSE_TYPES})


# ---------------------------------------------------------------------------
# API JSON
# ---------------------------------------------------------------------------
@require_GET
def api_districts(request):
    city = request.GET.get("city", "")
    if city not in services.CITIES:
        return JsonResponse({"error": f"city phải là một trong {services.CITIES}"}, status=400)
    try:
        return JsonResponse({"city": city, "districts": services.districts(city)})
    except services.ModelNotReady as exc:
        return JsonResponse({"error": str(exc)}, status=503)


@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Nội dung yêu cầu phải là JSON hợp lệ"}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"error": "Nội dung yêu cầu phải là một đối tượng JSON"}, status=400)
    try:
        form = HouseForm(payload)
        form.fields["district"].widget.choices = []
        if not form.is_valid():
            return JsonResponse({"error": "Dữ liệu không hợp lệ",
                                 "fields": {k: [str(e) for e in v] for k, v in form.errors.items()}}, status=400)
        d = form.cleaned_data
        explain = str(request.GET.get("explain", "1")) != "0"
        result = services.predict(d["city"], d["district"], d["house_type"], d["area_m2"], d["road_width_m"],
                                  d["floors"], d["width_m"], d["has_parking"], explain=explain)
    except services.ModelNotReady as exc:
        return JsonResponse({"error": str(exc)}, status=503)
    return JsonResponse({"input": {k: v for k, v in d.items()}, **result})
