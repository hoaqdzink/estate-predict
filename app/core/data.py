"""Nạp dữ liệu, mô hình, kết quả trung gian và cấu hình từ thanh bên."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "vietnam_housing_dirty.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "clean.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = ROOT / "models" / "learnt.pkl"
REPORTS = ROOT / "reports"
ARTIFACTS_PATH = REPORTS / "artifacts.pkl"
FIG_DIR = REPORTS / "figures"
REQUIRED = [MODEL_PATH, FEATURES_PATH, CLEAN_PATH, ARTIFACTS_PATH, REPORTS / "cleaning_log.csv"]

AMENITY_COLS = ["n_schools", "n_health", "n_markets", "n_parks", "n_transit"]
FEATURE_LABELS = {
    "area_m2": "Diện tích", "width_m": "Chiều ngang", "length_m": "Chiều dài", "road_width_m": "Độ rộng đường",
    "floors": "Số tầng", "bedrooms": "Số phòng ngủ", "width_length_ratio": "Tỷ lệ ngang/dài",
    "dist_center_km": "Khoảng cách tới trung tâm", "lat": "Vĩ độ", "lon": "Kinh độ",
    "n_schools": "Trường học xung quanh", "n_health": "Y tế xung quanh", "n_markets": "Chợ, siêu thị xung quanh",
    "n_parks": "Công viên xung quanh", "n_transit": "Trạm giao thông xung quanh",
    "has_dining_room": "Phòng ăn", "has_kitchen": "Nhà bếp", "has_terrace": "Sân thượng", "has_parking": "Chỗ đỗ ô tô",
    "city": "Thành phố", "district": "Quận/huyện", "house_type": "Loại nhà", "price_billion_vnd": "Giá (tỷ)",
}


def missing_files():
    return [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]


@st.cache_resource
def learnt():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def artifacts():
    return joblib.load(ARTIFACTS_PATH)


@st.cache_data
def raw():
    return pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False)


@st.cache_data
def clean():
    return pd.read_csv(CLEAN_PATH)


@st.cache_data
def features():
    return pd.read_csv(FEATURES_PATH)


@st.cache_data
def cleaning_log():
    return pd.read_csv(REPORTS / "cleaning_log.csv").rename(columns={"buoc": "Bước", "so_dong": "Số bản ghi"})


# ---------------------------------------------------------------------------
# Cấu hình từ thanh bên (lưu trong session_state, dùng chung cho mọi trang)
# ---------------------------------------------------------------------------
DEFAULTS = {"cfg_city": "Tất cả", "cfg_segments": (5.0, 15.0), "cfg_iqr_k": 1.5, "cfg_maha_p": 0.975,
            "cfg_level": 95, "cfg_boot": 1000, "cfg_seed": 7}


def cfg(key):
    return st.session_state.get(key, DEFAULTS[key])


def filter_city(df):
    city = cfg("cfg_city")
    return df if city == "Tất cả" else df[df["city"] == city]


def segment_labels():
    lo, hi = cfg("cfg_segments")
    return [f"Dưới {vn(lo, 0)} tỷ", f"{vn(lo, 0)}–{vn(hi, 0)} tỷ", f"Trên {vn(hi, 0)} tỷ"]


def segment(price):
    lo, hi = cfg("cfg_segments")
    return pd.cut(pd.Series(price), [0, lo, hi, np.inf], right=False, labels=segment_labels())


def vn(x, d=2):
    """Định dạng số kiểu Việt Nam: 1.234,56"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")
