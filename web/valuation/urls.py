from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("mo-hinh/", views.model_page, name="model"),
    path("api/", views.api_docs, name="api_docs"),
    path("api/districts/", views.api_districts, name="api_districts"),
    path("api/predict/", views.api_predict, name="api_predict"),
]
