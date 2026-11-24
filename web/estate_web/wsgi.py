import ctypes
import os
import sys
from pathlib import Path
from django.core.wsgi import get_wsgi_application
WEB_DIR = Path(__file__).resolve().parent.parent
# Vercel chạy hàm từ thư mục gốc repo, nên thêm web/ vào sys.path để import được estate_web và valuation
sys.path.insert(0, str(WEB_DIR))
# LightGBM cần libgomp.so.1, môi trường Vercel (Amazon Linux 2023) không có sẵn.
# Nạp bản đi kèm repo trước, để lightgbm dùng lại thư viện đã nạp này.
_libgomp = WEB_DIR / "lib" / "libgomp.so.1"
if sys.platform == "linux" and _libgomp.exists():
    try:
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)  # máy đã có sẵn thì dùng bản của máy
    except OSError:
        ctypes.CDLL(str(_libgomp), mode=ctypes.RTLD_GLOBAL)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "estate_web.settings")
application = get_wsgi_application()
