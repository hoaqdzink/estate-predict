"""Nạp mô hình learnt.pkl và dự báo giá. Không phụ thuộc Streamlit."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

AMENITY_COLS = ["n_schools", "n_health", "n_markets", "n_parks", "n_transit"]
AMENITY_NAMES = {"n_schools": "Trường học", "n_health": "Y tế", "n_markets": "Chợ, siêu thị",
                 "n_parks": "Công viên", "n_transit": "Trạm giao thông"}
FEATURE_LABELS = {
    "area_m2": "Diện tích", "width_m": "Chiều ngang", "road_width_m": "Độ rộng đường", "floors": "Số tầng",
    "dist_center_km": "Khoảng cách tới trung tâm", "lat": "Vĩ độ", "has_parking": "Chỗ đỗ ô tô",
    "district": "Quận/huyện", "house_type": "Loại nhà", **{c: f"{n} xung quanh" for c, n in AMENITY_NAMES.items()},
}
MODEL_LABELS = {"Ridge": "Ridge", "Lasso": "Lasso", "ElasticNet": "ElasticNet",
                "Gradient_Boosting": "Gradient Boosting", "LightGBM": "LightGBM"}
HOUSE_TYPES = ["Nhà trong hẻm", "Nhà mặt tiền"]
CITIES = ["Hà Nội", "Hồ Chí Minh"]


class ModelNotReady(RuntimeError):
    """Thiếu tệp kết quả do notebook tạo ra."""


@lru_cache(maxsize=1)
def learnt():
    if not settings.MODEL_PATH.exists():
        raise ModelNotReady(f"Không tìm thấy {settings.MODEL_PATH.name}. Hãy chạy notebook trước.")
    return joblib.load(settings.MODEL_PATH)


@lru_cache(maxsize=1)
def features():
    if not settings.FEATURES_PATH.exists():
        raise ModelNotReady(f"Không tìm thấy {settings.FEATURES_PATH.name}. Hãy chạy notebook trước.")
    return pd.read_csv(settings.FEATURES_PATH)


def coords():
    return learnt()["district_coords"]


def districts(city):
    c = coords()
    return sorted(c.loc[c["city"] == city, "district"].tolist())


def _parts():
    pipe = learnt()["model"].regressor_
    prep, stack = pipe.named_steps["prep"], pipe.named_steps["model"]
    return prep, stack


def _original_feature(tcol):
    group, name = tcol.split("__", 1)
    if group == "cat":
        return next(f for f in learnt()["features"]["cat"] if name.startswith(f + "_"))
    return name


@lru_cache(maxsize=1)
def _explainers():
    import shap
    prep, stack = _parts()
    bg = prep.transform(build_input(features().sample(300, random_state=0)))
    out = []
    for est in stack.estimators_:
        if hasattr(est, "coef_"):
            out.append(("linear", est, bg.mean(axis=0), float(est.predict(bg).mean())))
        else:
            ex = shap.TreeExplainer(est, data=bg, feature_perturbation="interventional")
            out.append(("tree", ex, None, float(np.ravel(ex.expected_value)[0])))
    return out


def build_input(df):
    f = learnt()["features"]
    X = df[f["num"] + f["bool"] + f["cat"]].copy()
    X[f["bool"]] = X[f["bool"]].astype(float)
    return X.astype({c: float for c in f["num"]})


def predict(city, district, house_type, area_m2, road_width_m, floors, width_m=None, has_parking=False,
            explain=True):
    """Dự báo giá một căn nhà. Trả về dict có thể chuyển thành JSON."""
    c = coords()
    loc = c[(c["city"] == city) & (c["district"] == district)]
    if loc.empty:
        raise ValueError(f"Quận/huyện '{district}' không thuộc {city}")
    loc = loc.iloc[0]
    X = build_input(pd.DataFrame([{
        "area_m2": area_m2, "width_m": width_m, "road_width_m": road_width_m, "floors": floors,
        "dist_center_km": loc["dist_center_km"], "lat": loc["lat"], **{k: loc[k] for k in AMENITY_COLS},
        "has_parking": float(bool(has_parking)), "district": district, "house_type": house_type}]))
    L = learnt()
    price = float(L["model"].predict(X)[0])
    iv = L["interval"]
    lo, hi = np.expm1(np.log1p(price) + iv["q_lo"]), np.expm1(np.log1p(price) + iv["q_hi"])

    prep, stack = _parts()
    Xt = prep.transform(X)
    base_log = np.array([est.predict(Xt)[0] for est in stack.estimators_])
    w, b = stack.final_estimator_.coef_, float(stack.final_estimator_.intercept_)
    result = {
        "price_billion_vnd": round(price, 3),
        "interval_95": {"low": round(float(lo), 3), "high": round(float(hi), 3),
                        "test_coverage": round(float(iv["test_coverage"]), 4)},
        "price_per_m2_million_vnd": round(price * 1000 / area_m2, 1),
        "location": {"dist_center_km": round(float(loc["dist_center_km"]), 2),
                     "lat": float(loc["lat"]), "lon": float(loc["lon"]),
                     "amenities_2km": {k: int(loc[k]) for k in AMENITY_COLS}},
        "stacking": {"intercept": round(b, 4), "final_log1p": round(b + float(base_log @ w), 4),
                     "base_models": [{"name": MODEL_LABELS[n], "pred_log1p": round(float(p), 4),
                                      "pred_billion_vnd": round(float(np.expm1(p)), 3), "weight": round(float(wi), 4)}
                                     for (n, _), p, wi in zip(stack.estimators, base_log, w)]},
        "warnings": _warnings(area_m2, road_width_m, floors, price),
        "model": L["model_name"],
    }
    if explain:
        result["explanation"] = _explain(Xt, w, b, prep)
    return result


def _explain(Xt, w, b, prep):
    """SHAP của Stacking = tổng có trọng số SHAP của 5 mô hình con, gom theo thuộc tính gốc."""
    phi, base = np.zeros(Xt.shape[1]), b
    for wi, (kind, obj, mean, ev) in zip(w, _explainers()):
        v = obj.coef_ * (Xt[0] - mean) if kind == "linear" else np.ravel(obj.shap_values(Xt))
        phi += wi * v
        base += wi * ev
    s = pd.Series(phi, index=prep.get_feature_names_out()).groupby(_original_feature).sum()
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    return {"base_price_billion_vnd": round(float(np.expm1(base)), 3),
            "contributions": [{"feature": f, "label": FEATURE_LABELS.get(f, f), "shap_log1p": round(float(v), 4),
                               "effect_percent": round(float((np.exp(v) - 1) * 100), 1)} for f, v in s.items()]}


def _warnings(area, road, floors, price):
    d = features().describe()
    out = [f"Giá trị {name} nằm ngoài miền dữ liệu huấn luyện ({d.loc['min', col]:g}–{d.loc['max', col]:g}); kết quả kém tin cậy."
           for name, v, col in [("diện tích", area, "area_m2"), ("độ rộng đường", road, "road_width_m"),
                                ("số tầng", floors, "floors")] if not d.loc["min", col] <= v <= d.loc["max", col]]
    if price > 15:
        out.append("Mô hình có xu hướng định giá thấp nhà trên 15 tỷ (phân khúc ít mẫu); giá thực tế có thể cao hơn.")
    return out


def comparables(district, house_type, area, limit=10):
    f = features()
    comps = f[(f["district"] == district) & (f["house_type"] == house_type)
              & f["area_m2"].between(area * 0.7, area * 1.3)].copy()
    comps["ppm2"] = comps["price_billion_vnd"] * 1000 / comps["area_m2"]
    median = float(comps["price_billion_vnd"].median()) if len(comps) else None
    rows = comps.sort_values("area_m2").head(limit)[["area_m2", "road_width_m", "floors", "price_billion_vnd", "ppm2"]]
    return {"count": int(len(comps)), "median_price": median, "rows": rows.to_dict("records")}


def model_summary():
    L = learnt()
    metrics = pd.read_csv(settings.METRICS_PATH).to_dict("records") if settings.METRICS_PATH.exists() else []
    _, stack = _parts()
    weights = [{"name": MODEL_LABELS[n], "weight": float(w)} for (n, _), w in zip(stack.estimators, stack.final_estimator_.coef_)]
    return {"name": L["model_name"], "trained_on": L["trained_on"], "test_metrics": L["test_metrics"],
            "interval": L["interval"], "features": L["features"], "metrics": metrics, "weights": weights,
            "intercept": float(stack.final_estimator_.intercept_)}
