import os
import sys
from pathlib import Path
from django.core.wsgi import get_wsgi_application
# Vercel chạy hàm từ thư mục gốc repo, nên thêm web/ vào sys.path để import được estate_web và valuation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "estate_web.settings")
application = get_wsgi_application()
