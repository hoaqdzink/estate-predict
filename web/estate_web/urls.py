from django.urls import include, path

urlpatterns = [path("", include("valuation.urls"))]
# TẠM THỜI: hiện traceback để chẩn đoán lỗi trên Vercel, sẽ gỡ sau
import sys, traceback
from django.http import HttpResponse
def handler500(request):
    return HttpResponse("".join(traceback.format_exception(*sys.exc_info())), content_type="text/plain", status=500)
