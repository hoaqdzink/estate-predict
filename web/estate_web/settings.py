"""Cấu hình Django cho web định giá bất động sản.

Không dùng cơ sở dữ liệu: mô hình được nạp từ models/learnt.pkl.
Biến môi trường:
  DJANGO_SECRET_KEY    khóa bí mật (bắt buộc khi DJANGO_DEBUG=0)
  DJANGO_DEBUG         1 (mặc định) hoặc 0
  DJANGO_ALLOWED_HOSTS danh sách tên miền, cách nhau bằng dấu phẩy
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY") or ("dev-only-insecure-key" if DEBUG else None)
if not SECRET_KEY:
    raise RuntimeError("Cần đặt biến môi trường DJANGO_SECRET_KEY khi DJANGO_DEBUG=0")
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "valuation",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "estate_web.urls"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
}]
WSGI_APPLICATION = "estate_web.wsgi.application"
DATABASES = {}

LANGUAGE_CODE = "vi"
TIME_ZONE = "Asia/Ho_Chi_Minh"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Đường dẫn tới các tệp do notebook tạo ra
MODEL_PATH = PROJECT_ROOT / "models" / "learnt.pkl"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.csv"
