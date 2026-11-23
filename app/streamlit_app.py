"""Ứng dụng trình bày và định giá bất động sản (Đề tài 06).

Chạy: streamlit run app/streamlit_app.py
Yêu cầu: đã thực thi notebooks/01_cleaning.ipynb để tạo models/learnt.pkl và các tệp trong reports/.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "vietnam_housing_dirty.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "clean.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = ROOT / "models" / "learnt.pkl"
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"

st.set_page_config(page_title="Định giá bất động sản đô thị", layout="wide")


# ---------------------------------------------------------------------------
# Nạp dữ liệu và mô hình
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(path, **kw):
    return pd.read_csv(path, **kw)


required = [MODEL_PATH, FEATURES_PATH, CLEAN_PATH, REPORTS / "metrics.csv", REPORTS / "cv_results.csv",
            REPORTS / "cleaning_log.csv"]
missing = [p.relative_to(ROOT) for p in required if not p.exists()]
if missing:
    st.error("Thiếu các tệp kết quả: " + ", ".join(map(str, missing))
             + ". Hãy thực thi toàn bộ notebook `notebooks/01_cleaning.ipynb` trước.")
    st.stop()

learnt = load_model()
features = load_csv(FEATURES_PATH)
clean = load_csv(CLEAN_PATH)
metrics = load_csv(REPORTS / "metrics.csv")
cv_results = load_csv(REPORTS / "cv_results.csv")
cleaning_log = load_csv(REPORTS / "cleaning_log.csv")
coords = learnt["district_coords"]
tm = learnt["test_metrics"]
AMENITY_COLS = ["n_schools", "n_health", "n_markets", "n_parks", "n_transit"]
AMENITY_NAMES = {"n_schools": "Trường học", "n_health": "Bệnh viện, phòng khám", "n_markets": "Chợ, siêu thị",
                 "n_parks": "Công viên", "n_transit": "Trạm xe buýt, ga"}


# ---------------------------------------------------------------------------
# Cấu trúc mô hình và giải thích từng dự báo
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "area_m2": "Diện tích", "width_m": "Chiều ngang", "length_m": "Chiều dài", "road_width_m": "Độ rộng đường",
    "floors": "Số tầng", "bedrooms": "Số phòng ngủ", "width_length_ratio": "Tỷ lệ ngang/dài",
    "dist_center_km": "Khoảng cách tới trung tâm", "lat": "Vĩ độ", "lon": "Kinh độ",
    "n_schools": "Trường học xung quanh", "n_health": "Y tế xung quanh", "n_markets": "Chợ, siêu thị xung quanh",
    "n_parks": "Công viên xung quanh", "n_transit": "Trạm giao thông xung quanh",
    "has_dining_room": "Phòng ăn", "has_kitchen": "Nhà bếp", "has_terrace": "Sân thượng", "has_parking": "Chỗ đỗ ô tô",
    "city": "Thành phố", "district": "Quận/huyện", "house_type": "Loại nhà",
}
MODEL_LABELS = {"Ridge": "Ridge", "Lasso": "Lasso", "ElasticNet": "ElasticNet",
                "Gradient_Boosting": "Gradient Boosting", "LightGBM": "LightGBM"}
INPUT_COLS = learnt["features"]["num"] + learnt["features"]["bool"] + learnt["features"]["cat"]
PIPE = learnt["model"].regressor_
PREP, STACK = PIPE.named_steps["prep"], PIPE.named_steps["model"]
BASE_NAMES = [n for n, _ in STACK.estimators]
META_W, META_B = STACK.final_estimator_.coef_, float(STACK.final_estimator_.intercept_)
TCOLS = list(PREP.get_feature_names_out())


def original_feature(tcol):
    """'num__area_m2' -> 'area_m2'; 'cat__district_Quận 1' -> 'district'."""
    group, name = tcol.split("__", 1)
    if group == "cat":
        return next(f for f in learnt["features"]["cat"] if name.startswith(f + "_"))
    return name


def to_model_input(df):
    X = df[INPUT_COLS].copy()
    X[learnt["features"]["bool"]] = X[learnt["features"]["bool"]].astype(float)
    return X.astype({c: float for c in learnt["features"]["num"]})


@st.cache_resource
def build_explainers():
    """Nền so sánh: 300 bản ghi của tập dữ liệu, đã qua tiền xử lý."""
    import shap
    bg = PREP.transform(to_model_input(features.sample(300, random_state=0)))
    explainers = []
    for est in STACK.estimators_:
        if hasattr(est, "coef_"):        # mô hình tuyến tính: SHAP = hệ số × (x − trung bình)
            explainers.append(("linear", est, bg.mean(axis=0), float(est.predict(bg).mean())))
        else:                            # mô hình cây: TreeExplainer, cùng nền so sánh
            ex = shap.TreeExplainer(est, data=bg, feature_perturbation="interventional")
            explainers.append(("tree", ex, None, float(np.ravel(ex.expected_value)[0])))
    return explainers


def decompose(X):
    """Tách một dự báo thành từng bước tính của mô hình Stacking (thang log1p)."""
    Xt = PREP.transform(X)
    base_log = np.array([est.predict(Xt)[0] for est in STACK.estimators_])
    final_log = META_B + float(base_log @ META_W)
    return Xt, base_log, final_log


def explain(Xt):
    """Giá trị SHAP của cả khối Stacking = tổng có trọng số SHAP của từng mô hình con."""
    phi, base = np.zeros(Xt.shape[1]), META_B
    for w, (kind, obj, mean, ev) in zip(META_W, build_explainers()):
        p = obj.coef_ * (Xt[0] - mean) if kind == "linear" else np.ravel(obj.shap_values(Xt))
        phi += w * p
        base += w * ev
    grouped = pd.Series(phi, index=TCOLS).groupby(original_feature).sum()
    return base, grouped.sort_values(key=np.abs, ascending=False)


def shap_waterfall(base, contrib, values, top=8):
    """Biểu đồ thác nước trên đơn vị tỷ VNĐ: cộng dần từng đóng góp trên thang log rồi đổi về giá."""
    head = contrib.iloc[:top]
    rest = contrib.iloc[top:].sum()
    items = list(head.items()) + ([("Các thuộc tính khác", rest)] if len(contrib) > top else [])
    cum, rows = base, []
    for name, v in items:
        before, cum = cum, cum + v
        label = FEATURE_LABELS.get(name, name)
        if name in values:
            label += f" = {values[name]}"
        rows.append((label, np.expm1(before), np.expm1(cum) - np.expm1(before), v))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.6))
    for i, (label, start, delta, _) in enumerate(rows):
        ax.barh(i, delta, left=start, color="#C44E52" if delta < 0 else "#55A868")
        ax.text(start + delta + (0.05 if delta >= 0 else -0.05), i, f"{delta:+.2f}",
                va="center", ha="left" if delta >= 0 else "right", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(np.expm1(base), color="gray", linestyle=":", label=f"Giá tham chiếu {np.expm1(base):.2f} tỷ")
    ax.axvline(np.expm1(cum), color="#4C72B0", linestyle="--", label=f"Giá dự báo {np.expm1(cum):.2f} tỷ")
    ax.set_xlabel("Giá (tỷ VNĐ)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    table = pd.DataFrame([{"Thuộc tính": r[0], "Đóng góp (tỷ)": r[2], "Ảnh hưởng (%)": (np.exp(r[3]) - 1) * 100}
                          for r in rows])
    return fig, table


def stacking_table(base_log):
    t = pd.DataFrame({"Mô hình con": [MODEL_LABELS[n] for n in BASE_NAMES],
                      "Dự báo (log1p)": base_log, "Dự báo (tỷ)": np.expm1(base_log),
                      "Trọng số meta": META_W, "Đóng góp (log1p)": base_log * META_W})
    return t


def show_figure(name, caption):
    path = FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chưa có biểu đồ `{name}`. Hãy thực thi lại notebook.")


# ---------------------------------------------------------------------------
# Thanh điều hướng
# ---------------------------------------------------------------------------
PAGES = ["1. Ý tưởng", "2. Dẫn dắt: dữ liệu và tiền xử lý", "3. Phản biện: các lựa chọn và hạn chế",
         "4. Minh chứng: kết quả mô hình", "5. Mô hình huấn luyện (learnt.pkl)", "6. Định giá trực tuyến"]
st.sidebar.title("Đề tài 06")
st.sidebar.caption("Dự báo giá bất động sản đô thị\n\nUrban Real Estate Valuation & Hedonic Pricing")
page = st.sidebar.radio("Nội dung", PAGES)
st.sidebar.divider()
st.sidebar.metric("Mô hình", learnt["model_name"])
st.sidebar.metric("R² trên tập kiểm thử", f"{tm['R²']:.3f}")
st.sidebar.metric("MAPE trên tập kiểm thử", f"{tm['MAPE (%)']:.1f}%")


# ---------------------------------------------------------------------------
# 1. Ý tưởng
# ---------------------------------------------------------------------------
if page == PAGES[0]:
    st.title("Định giá bất động sản đô thị bằng học máy")
    st.markdown("""
**Bài toán:** ước lượng giá chuyển nhượng một căn nhà tại Hà Nội và TP.HCM từ các đặc điểm của căn nhà:
diện tích, kích thước, vị trí, tiện ích xung quanh, loại nhà, số tầng, số phòng và tiện ích trong nhà.

**Ý nghĩa thực tế:** môi giới và người mua thường định giá dựa trên kinh nghiệm và so sánh thủ công với vài
căn lân cận. Một mô hình định giá tự động giúp đưa ra mức giá tham khảo nhanh, nhất quán, kèm theo mức độ
bất định của ước lượng.

**Cơ sở lý thuyết, định giá hedonic (Hedonic Pricing):** giá của một bất động sản được xem là tổng giá trị
của các thuộc tính cấu thành nó. Mỗi thuộc tính (thêm 1 m² diện tích, ở mặt tiền thay vì trong hẻm, gần
trung tâm hơn 1 km) đóng góp một phần vào giá. Mô hình hồi quy ước lượng các phần đóng góp này từ dữ liệu.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tin rao thu thập", "3.500")
    c2.metric("Sau làm sạch", f"{len(clean):,}".replace(",", "."))
    c3.metric("Dùng cho mô hình", f"{len(features):,}".replace(",", "."))
    c4.metric("Số thuộc tính đầu vào", sum(len(v) for v in learnt["features"].values()))

    st.subheader("Quy trình thực hiện")
    st.markdown("""
| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Thu thập và làm sạch | Dữ liệu tin rao thô, 36 loại lỗi: số dạng chữ, sai đơn vị, Unicode, trùng lặp... | Tập dữ liệu sạch |
| Bước 1. Tiền xử lý | Mất cân bằng dữ liệu, phân phối giá, Box-Cox/Log1p, tọa độ và khoảng cách tới trung tâm, tiện ích xung quanh (OpenStreetMap), ngoại lai IQR/Mahalanobis | Tập đặc trưng |
| Bước 2. Mô hình hóa | Trích chọn thuộc tính, grid search cross-validation cho Ridge, Lasso, ElasticNet, Gradient Boosting, LightGBM, Stacking | `learnt.pkl` |
| Bước 3. Đánh giá | RMSE, MAE, R², MAPE trên tập kiểm thử độc lập; sai số theo phân khúc; giải thích SHAP | Bảng so sánh |
| Bước 4. Ứng dụng | Mô hình huấn luyện được và cách dùng để dự báo; định giá trực tuyến kèm khoảng dự báo 95% và giải thích từng dự báo | Ứng dụng này |
""")


# ---------------------------------------------------------------------------
# 2. Dẫn dắt
# ---------------------------------------------------------------------------
elif page == PAGES[1]:
    st.title("Dẫn dắt: từ dữ liệu thô tới dữ liệu cho mô hình")

    st.header("2.1 Dữ liệu thô")
    st.markdown("Dữ liệu được thu thập ở dạng thô: giá, diện tích, số tầng được ghi bằng nhiều cách khác nhau; "
                "tên thành phố, quận viết không thống nhất; có dòng trùng, dòng rỗng, dòng lệch cột.")
    raw = load_csv(RAW_PATH, dtype=str, keep_default_na=False)
    cols = ["city", "district", "house_type", "area_m2", "floors", "bedrooms", "price_billion_vnd"]
    st.dataframe(raw[cols].sample(12, random_state=7), width="stretch")
    c1, c2 = st.columns(2)
    c1.metric("Số cách viết tên thành phố", raw["city"].nunique())
    c2.metric("Số cách viết tên quận/huyện", raw["district"].nunique())

    st.header("2.2 Làm sạch dữ liệu")
    log = cleaning_log.rename(columns={"buoc": "Bước", "so_dong": "Số bản ghi"})
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(log["Bước"], log["Số bản ghi"], color="#4C72B0")
    ax.invert_yaxis()
    ax.bar_label(ax.containers[0], padding=3)
    ax.set_xlabel("Số bản ghi")
    st.pyplot(fig)
    st.markdown("Sau làm sạch: thành phố còn 2 giá trị, quận/huyện còn 39 giá trị chuẩn, mọi thuộc tính số "
                "đều đúng kiểu. Mỗi bước được trình bày chi tiết kèm bảng trước/sau trong notebook.")

    st.header("2.3 Mất cân bằng dữ liệu")
    seg = pd.cut(features["price_billion_vnd"], [0, 5, 15, np.inf], right=False,
                 labels=["Dưới 5 tỷ", "5–15 tỷ", "Trên 15 tỷ"]).value_counts().sort_index()
    c1, c2, c3 = st.columns(3)
    for col, (name, s) in zip([c1, c2, c3], [("Thành phố", features["city"].value_counts()),
                                              ("Loại nhà", features["house_type"].value_counts()),
                                              ("Phân khúc giá", seg)]):
        col.markdown(f"**{name}**")
        col.bar_chart((s / s.sum() * 100).rename("Tỷ lệ (%)"))
    st.markdown("Phân khúc trên 15 tỷ chỉ chiếm khoảng 13% dữ liệu; nhiều quận/huyện ngoại thành có dưới 20 bản ghi. "
                "**Hướng xử lý:** biến đổi log của giá, chia tập huấn luyện/kiểm thử phân tầng theo phân khúc giá, "
                "gộp quận ít mẫu, và đánh giá sai số riêng từng phân khúc.")

    st.header("2.4 Phân phối giá và biến đổi Log1p")
    price = features["price_billion_vnd"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(price, bins=50, color="#4C72B0", edgecolor="white")
    axes[0].set_title(f"Giá gốc (skewness = {price.skew():.2f})")
    axes[0].set_xlabel("Tỷ VNĐ")
    axes[1].hist(np.log1p(price), bins=50, color="#55A868", edgecolor="white")
    axes[1].set_title(f"Log1p(giá) (skewness = {np.log1p(price).skew():.2f})")
    plt.tight_layout()
    st.pyplot(fig)

    st.header("2.5 Vị trí: khoảng cách tới trung tâm")
    st.markdown("Mỗi quận/huyện được gán tọa độ tâm (OpenStreetMap) và khoảng cách tới Hồ Hoàn Kiếm (Hà Nội) "
                "hoặc chợ Bến Thành (TP.HCM).")
    city = st.selectbox("Thành phố", ["Hà Nội", "Hồ Chí Minh"], key="lead_city")
    g = features[features["city"] == city]
    ppm2 = (g["price_billion_vnd"] * 1000 / g["area_m2"]).groupby(g["dist_center_km"].round(1)).median()
    st.line_chart(ppm2.rename("Giá/m² trung vị (triệu)"), x_label="Khoảng cách tới trung tâm (km)")
    st.map(coords[coords["city"] == city].rename(columns={"lon": "longitude", "lat": "latitude"}), size=300)

    st.header("2.6 Tiện ích xung quanh")
    st.markdown("Số trường học, cơ sở y tế, chợ/siêu thị, công viên và trạm giao thông trong bán kính 2 km quanh tâm "
                "mỗi quận/huyện, đếm từ bản đồ OpenStreetMap (bản trích xuất ngày 19/11/2026).")
    g = features[features["city"] == city].assign(ppm2=lambda d: d["price_billion_vnd"] * 1000 / d["area_m2"])
    tbl = g.groupby("district").agg(**{AMENITY_NAMES[c]: (c, "first") for c in AMENITY_COLS},
                                     **{"Giá/m² trung vị (triệu)": ("ppm2", "median"), "Số bản ghi": ("ppm2", "size")})
    st.dataframe(tbl.sort_values("Giá/m² trung vị (triệu)", ascending=False).round(0), width="stretch")
    st.markdown("Quận nhiều tiện ích có giá/m² cao hơn. Tuy nhiên số tiện ích tương quan chặt với khoảng cách tới "
                "trung tâm, nên khi đã có khoảng cách, tiện ích chỉ bổ sung thêm rất ít thông tin cho mô hình.")


# ---------------------------------------------------------------------------
# 3. Phản biện
# ---------------------------------------------------------------------------
elif page == PAGES[2]:
    st.title("Phản biện: các lựa chọn đã cân nhắc và hạn chế")
    st.markdown("Mỗi quyết định trong quy trình được so sánh với phương án thay thế và kiểm chứng bằng dữ liệu.")

    with st.expander("Box-Cox hay Log1p cho biến mục tiêu?", expanded=True):
        st.markdown("""
- **Theo hình dạng phân phối**, Box-Cox tốt hơn: độ lệch gần 0 so với 0,57 của Log1p.
- **Theo sai số dự đoán**, Log1p tốt hơn: Ridge với Box-Cox có CV RMSE ≈ 5,16 tỷ, với Log1p ≈ 3,65 tỷ.
- **Nguyên nhân:** λ Box-Cox ước lượng âm (≈ −0,17) nên phép biến đổi ngược không ổn định khi mô hình tuyến tính ngoại suy.
- **Kết luận:** chọn Log1p. Tiêu chí cuối cùng là sai số trên đơn vị gốc, không phải hình dạng phân phối.
""")
    with st.expander("IQR hay Mahalanobis để loại ngoại lai?"):
        st.markdown("""
- IQR (trên log giá/m², riêng từng thành phố) đánh dấu 126 bản ghi; Mahalanobis đánh dấu 155; chỉ 51 bản ghi trùng nhau.
- Mahalanobis giả định phân phối chuẩn nhiều chiều, không phù hợp với số tầng, số phòng (biến rời rạc), nên đánh dấu nhầm nhiều nhà lớn nhiều phòng nhưng hợp lệ, và không đánh giá được 13% bản ghi thiếu dữ liệu.
- **Kết luận:** loại theo IQR; Mahalanobis dùng để kiểm tra chéo.
""")
    with st.expander("Có nên bỏ bớt thuộc tính?"):
        st.markdown("""
- Độ quan trọng hoán vị loại 8 thuộc tính yếu: `city`, `lon`, `length_m`, `width_length_ratio`, `bedrooms` và ba tiện ích trong nhà. Thông tin của chúng đã có ở thuộc tính khác hoặc quá nhiễu.
- Bỏ các thuộc tính này giúp LightGBM giảm sai số, Ridge tăng nhẹ; sai số trung bình không tăng.
- **Kết luận:** dùng tập 14 thuộc tính đã chọn: mô hình gọn hơn, không kém đi.
""")
    with st.expander("Tiện ích xung quanh có giúp mô hình dự báo tốt hơn?"):
        st.markdown("""
- Thêm 5 thuộc tính tiện ích gần như **không thay đổi** sai số cross-validation (chênh lệch dưới 0,01 tỷ).
- Nguyên nhân: tiện ích được đếm theo quận và tương quan chặt với khoảng cách tới trung tâm, nên mang thông tin trùng lặp.
- Vẫn giữ lại vì đáp ứng yêu cầu đề tài, có ý nghĩa diễn giải (SHAP xếp `n_markets` ngang khoảng cách tới trung tâm), và không làm mô hình kém đi.
- Muốn tiện ích có giá trị dự báo riêng, cần tọa độ của từng căn nhà.
""")
    with st.expander("Vì sao không có tình trạng pháp lý?"):
        st.markdown("""
- Đề tài yêu cầu định giá dựa trên cả tình trạng pháp lý, nhưng bộ dữ liệu gốc không thu thập thuộc tính này.
- **Không giả lập**, vì giá trị giả lập buộc phải gán sẵn quy luật (ví dụ nhà có sổ đắt hơn X%); mô hình khi đó chỉ học lại giả định của người làm.
- Ứng dụng ghi chú rằng giá dự báo áp dụng cho nhà có giấy tờ pháp lý đầy đủ.
""")
    with st.expander("Stacking có đáng với độ phức tạp?"):
        st.markdown(f"""
- Stacking có CV RMSE thấp nhất nhưng chỉ hơn LightGBM chưa tới 1%, nhỏ hơn độ dao động giữa các fold.
- Trên tập kiểm thử, Gradient Boosting có RMSE và MAE thấp hơn Stacking khoảng 0,015 tỷ, còn Stacking có MAPE thấp nhất.
- **Không đổi mô hình theo tập kiểm thử**, vì như vậy tập kiểm thử mất tính độc lập. Mô hình được chọn theo tiêu chí đặt trước (CV RMSE): **{learnt['model_name']}**.
""")
    with st.expander("Loại ngoại lai trước khi chia tập có gây rò rỉ dữ liệu?"):
        st.markdown("""
- Loại ngoại lai là bước loại bản ghi nhập sai, không học tham số nào dùng cho dự đoán.
- Mọi bước học từ dữ liệu (điền thiếu, chuẩn hóa, mã hóa, biến đổi biến mục tiêu, chọn tham số) đều nằm trong pipeline và chỉ học trên tập huấn luyện.
""")
    with st.expander("Khoảng tin cậy hay khoảng dự báo?"):
        st.markdown("""
- **Khoảng tin cậy** chứa *giá trung bình* của các căn có cùng đặc điểm.
- **Khoảng dự báo** chứa *giá của một căn cụ thể*, rộng hơn khoảng tin cậy.
- Môi giới cần biết giá của căn đang định giá, nên ứng dụng dùng **khoảng dự báo 95%** (phương pháp cross-conformal).
""")
    st.subheader("Hạn chế")
    st.markdown("""
- **Dữ liệu:** giá rao bán, không phải giá giao dịch; thu thập trước 01/2022; không có tình trạng pháp lý; vị trí chỉ ở cấp quận/huyện.
- **Dữ liệu được bổ sung lỗi nhân tạo** để thực hành làm sạch; các cột `url`, `title`, `contact`... là giả lập.
- **Mô hình:** sai số tương đối khoảng 25%; định giá thấp nhà trên 15 tỷ do phân khúc này ít mẫu.
- **Tiện ích xung quanh** chỉ tính được ở cấp quận/huyện; dữ liệu OpenStreetMap ở ngoại thành chưa đầy đủ.
""")


# ---------------------------------------------------------------------------
# 4. Minh chứng
# ---------------------------------------------------------------------------
elif page == PAGES[3]:
    st.title("Minh chứng: kết quả và giải thích")

    st.header("4.1 So sánh mô hình")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{tm['RMSE']:.2f} tỷ")
    c2.metric("MAE", f"{tm['MAE']:.2f} tỷ")
    c3.metric("R²", f"{tm['R²']:.3f}")
    c4.metric("MAPE", f"{tm['MAPE (%)']:.1f}%")
    st.markdown(f"Kết quả của mô hình **{learnt['model_name']}** trên tập kiểm thử độc lập (564 bản ghi).")
    show = metrics[["Mô hình", "RMSE", "MAE", "R²", "MAPE (%)", "CV RMSE"]]
    st.dataframe(show.style.format(precision=3).highlight_min(subset=["RMSE", "MAE", "MAPE (%)", "CV RMSE"],
                                                               color="#d4edda")
                 .highlight_max(subset=["R²"], color="#d4edda"), width="stretch", hide_index=True)
    show_figure("model_comparison.png", "So sánh bốn thang đo trên tập kiểm thử")
    st.markdown("""
- Nhóm mô hình cây (Gradient Boosting, LightGBM, Stacking) tốt hơn nhóm tuyến tính: học được quan hệ phi tuyến và tương tác.
- RMSE kiểm thử xấp xỉ CV RMSE: mô hình không quá khớp.
""")

    st.header("4.2 Giá thực tế và giá dự đoán")
    show_figure("pred_vs_actual.png", "Thực tế và dự đoán, phần dư")
    st.markdown("Sai số tăng theo mức giá: căn càng đắt, sai số tuyệt đối càng lớn. Đây là lý do dùng thang log.")

    st.header("4.3 Sai số theo phân khúc giá")
    show_figure("error_by_segment.png", "MAPE theo phân khúc giá")
    st.markdown("Sai số tương đối khá đồng đều giữa các phân khúc, nhưng phân khúc trên 15 tỷ bị **định giá thấp** "
                "(bias dương): hệ quả của mất cân bằng dữ liệu.")

    st.header("4.4 Giải thích mô hình (SHAP)")
    c1, c2 = st.columns(2)
    with c1:
        show_figure("shap_importance.png", "Mức đóng góp trung bình của thuộc tính")
    with c2:
        show_figure("shap_beeswarm.png", "Chiều hướng ảnh hưởng")
    st.markdown("Diện tích, độ rộng đường, loại nhà và số tầng quyết định phần lớn giá, tiếp theo là nhóm vị trí "
                "(tiện ích xung quanh, khoảng cách tới trung tâm). Kết quả trùng với độ quan trọng hoán vị và phù hợp "
                "lý thuyết hedonic: quy mô, khả năng tiếp cận, vị trí.")

    st.header("4.5 Khoảng dự báo 95%")
    show_figure("prediction_interval.png", "Khoảng dự báo trên tập kiểm thử")
    cov = learnt["interval"]["test_coverage"]
    st.markdown(f"Tỷ lệ giá thực tế nằm trong khoảng dự báo trên tập kiểm thử: **{cov:.1%}** (mục tiêu 95%).")

    with st.expander("Kết quả cross-validation trên tập huấn luyện"):
        st.dataframe(cv_results.round(4), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 5. Mô hình huấn luyện
# ---------------------------------------------------------------------------
elif page == PAGES[4]:
    st.title("Mô hình huấn luyện được và cách dùng để dự báo")

    st.header("5.1 Mô hình là gì")
    st.markdown(f"""
Mô hình được chọn là **{learnt['model_name']} Ensemble**, huấn luyện trên {str(f"{learnt['trained_on']:,}").replace(',', '.')} bản ghi.
Mô hình gồm ba tầng xếp chồng:

1. **Tiền xử lý:** điền giá trị thiếu (trung vị cho biến số, giá trị phổ biến nhất cho biến phân loại), chuẩn hóa
   biến số, mã hóa one-hot cho quận/huyện và loại nhà. {len(INPUT_COLS)} thuộc tính đầu vào trở thành
   {len(TCOLS)} cột số.
2. **Năm mô hình con** dự báo độc lập trên thang log1p của giá:
   - *Ridge, Lasso, ElasticNet*: hồi quy tuyến tính có chính quy hóa; mỗi cột có một hệ số, dự báo là tổng có trọng số.
   - *Gradient Boosting, LightGBM*: tổng của hàng trăm cây quyết định nhỏ; mỗi cây sửa phần sai số còn lại của các cây trước.
3. **Mô hình meta** (RidgeCV) kết hợp năm dự báo con thành một dự báo bằng tổng có trọng số. Trọng số được học từ
   dự báo ngoài fold, nên mô hình meta biết mô hình con nào đáng tin hơn.

Cuối cùng, dự báo được đổi từ thang log1p về tỷ VNĐ: $\\text{{giá}} = e^{{\\hat{{y}}}} - 1$.
""")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF3FA", fontname="Helvetica", fontsize=11];
  input [label="Thông số căn nhà\\n(14 thuộc tính)", fillcolor="#FFF4E0"];
  prep [label="Tiền xử lý\\nđiền thiếu, chuẩn hóa,\\none-hot (42 cột)"];
  r [label="Ridge"]; l [label="Lasso"]; e [label="ElasticNet"]; g [label="Gradient Boosting"]; b [label="LightGBM"];
  meta [label="Mô hình meta RidgeCV\\ntổng có trọng số"];
  inv [label="Đổi về tỷ VNĐ\\ne^y − 1"];
  out [label="Giá dự báo\\n+ khoảng 95%", fillcolor="#E3F4E7"];
  input -> prep; prep -> {r l e g b}; {r l e g b} -> meta; meta -> inv; inv -> out;
}""")

    st.subheader("Trọng số của mô hình meta")
    w = pd.DataFrame({"Mô hình con": [MODEL_LABELS[n] for n in BASE_NAMES], "Trọng số": META_W})
    params = {
        "Ridge": ["alpha"], "Lasso": ["alpha"], "ElasticNet": ["alpha", "l1_ratio"],
        "Gradient_Boosting": ["n_estimators", "learning_rate", "max_depth", "subsample"],
        "LightGBM": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"],
    }
    w["Tham số (từ grid search)"] = [", ".join(f"{k}={est.get_params()[k]}" for k in params[n])
                                     for n, est in zip(BASE_NAMES, STACK.estimators_)]
    c1, c2 = st.columns([3, 2])
    c1.dataframe(w.round(3), hide_index=True, width="stretch")
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.barh(w["Mô hình con"], w["Trọng số"], color=["#C44E52" if v < 0 else "#4C72B0" for v in w["Trọng số"]])
    ax.axvline(0, color="gray")
    ax.set_title(f"Hệ số chặn của mô hình meta: {META_B:.3f}", fontsize=9)
    plt.tight_layout()
    c2.pyplot(fig)
    st.markdown("Hai mô hình cây nhận khoảng 4/5 tổng trọng số: mô hình meta tin vào các mô hình học được quan hệ phi "
                "tuyến. Trọng số của Lasso gần 0 (hơi âm) vì Lasso gần như trùng thông tin với Ridge và ElasticNet.")

    st.header("5.2 Nội dung tệp `models/learnt.pkl`")
    st.markdown("Tệp được lưu bằng `joblib` ở cuối mục 6.7 của notebook, và được cập nhật tham số khoảng dự báo ở mục 7.5.")
    describe = {
        "model": "Pipeline đầy đủ: tiền xử lý → Stacking → đổi về tỷ VNĐ. Gọi `predict()` trực tiếp trên dữ liệu thô",
        "model_name": "Tên mô hình được chọn",
        "features": "Danh sách thuộc tính đầu vào theo nhóm (số, nhị phân, phân loại)",
        "district_coords": "Tọa độ, khoảng cách tới trung tâm và số tiện ích xung quanh của 39 quận/huyện",
        "city_center": "Tọa độ mốc trung tâm của hai thành phố",
        "target_transform": "Phép biến đổi biến mục tiêu đã dùng",
        "cv_rmse": "RMSE cross-validation trên tập huấn luyện",
        "trained_on": "Số bản ghi huấn luyện",
        "interval": "Tham số khoảng dự báo 95% (phân vị phần dư q_lo, q_hi) và độ phủ trên tập kiểm thử",
        "test_metrics": "RMSE, MAE, R², MAPE trên tập kiểm thử",
    }
    st.dataframe(pd.DataFrame([{"Khóa": k, "Kiểu": type(v).__name__, "Nội dung": describe.get(k, "")}
                               for k, v in learnt.items()]), hide_index=True, width="stretch")

    st.header("5.3 Đọc mô hình lên để dự báo")
    st.code('''import joblib
import numpy as np
import pandas as pd

learnt = joblib.load("models/learnt.pkl")
coords = learnt["district_coords"]

# Thông số căn nhà; vị trí và tiện ích lấy từ bảng quận/huyện
loc = coords[(coords.city == "Hà Nội") & (coords.district == "Cầu Giấy")].iloc[0]
house = pd.DataFrame([{
    "area_m2": 50, "width_m": 4.5, "road_width_m": 3, "floors": 5, "has_parking": 0.0,
    "district": "Cầu Giấy", "house_type": "Nhà trong hẻm",
    **loc[["dist_center_km", "lat", "n_schools", "n_health", "n_markets", "n_parks", "n_transit"]],
}])
cols = learnt["features"]["num"] + learnt["features"]["bool"] + learnt["features"]["cat"]

price = learnt["model"].predict(house[cols])[0]                  # tỷ VNĐ
lo = np.expm1(np.log1p(price) + learnt["interval"]["q_lo"])      # cận dưới 95%
hi = np.expm1(np.log1p(price) + learnt["interval"]["q_hi"])      # cận trên 95%
print(f"{price:.2f} tỷ, khoảng 95%: {lo:.2f} – {hi:.2f} tỷ")''', language="python")

    st.header("5.4 Minh họa từng bước tính trên một căn nhà thật")
    st.markdown("Chọn một căn trong dữ liệu để xem mô hình tính ra giá như thế nào qua từng tầng.")
    idx = st.number_input("Số thứ tự căn nhà trong `features.csv`", 0, len(features) - 1, 0)
    row = features.iloc[[int(idx)]]
    X = to_model_input(row)
    Xt, base_log, final_log = decompose(X)

    st.markdown("**Bước 1. Thông số đầu vào**")
    st.dataframe(X, hide_index=True, width="stretch")
    st.markdown("**Bước 2. Sau tiền xử lý** (chỉ hiện các cột khác 0)")
    vec = pd.Series(Xt[0], index=TCOLS)
    st.dataframe(vec[vec != 0].round(3).to_frame("giá trị").T, width="stretch")
    st.markdown("**Bước 3. Dự báo của từng mô hình con và kết hợp ở mô hình meta**")
    st.dataframe(stacking_table(base_log).round(3), hide_index=True, width="stretch")
    terms = " + ".join(f"{w_:.3f}×{p:.3f}" for w_, p in zip(META_W, base_log))
    st.markdown(f"Mô hình meta: $\\hat{{y}}$ = {META_B:.3f} + {terms} = **{final_log:.4f}**")
    st.markdown(f"**Bước 4. Đổi về tỷ VNĐ:** $e^{{{final_log:.4f}}} - 1$ = **{np.expm1(final_log):.2f} tỷ**. "
                f"Giá rao bán thực tế của căn này: **{row['price_billion_vnd'].iloc[0]:.2f} tỷ**.")
    direct = float(learnt["model"].predict(X)[0])
    st.caption(f"Kiểm tra: gọi trực tiếp `learnt['model'].predict()` cho {direct:.4f} tỷ, trùng với kết quả tính tay.")


# ---------------------------------------------------------------------------
# 6. Định giá trực tuyến
# ---------------------------------------------------------------------------
else:
    st.title("Định giá trực tuyến")
    st.markdown(f"Nhập thông số căn nhà để nhận giá dự báo và khoảng dự báo 95%. "
                f"Mô hình: **{learnt['model_name']}**, sai số tương đối trung bình khoảng {tm['MAPE (%)']:.0f}%.")

    rng = features.describe()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Vị trí")
        city = st.selectbox("Thành phố", ["Hà Nội", "Hồ Chí Minh"])
        district = st.selectbox("Quận/huyện", sorted(coords.loc[coords["city"] == city, "district"]))
        house_type = st.radio("Loại nhà", ["Nhà trong hẻm", "Nhà mặt tiền"], horizontal=True)
        road = st.number_input("Độ rộng đường/hẻm trước nhà (m)", 0.5, 60.0, 4.0, 0.5)
    with c2:
        st.subheader("Kích thước")
        area = st.number_input("Diện tích (m²)", 10.0, 1000.0, 50.0, 1.0)
        width = st.number_input("Chiều ngang (m), để trống nếu không rõ", 1.0, 50.0, value=None, step=0.1)
        length = st.number_input("Chiều dài (m), để trống nếu không rõ", 1.0, 100.0, value=None, step=0.5)
        floors = st.number_input("Số tầng", 1, 15, 4)
        bedrooms = st.number_input("Số phòng ngủ", 1, 20, 3)
    with c3:
        st.subheader("Tiện ích")
        dining = st.checkbox("Phòng ăn")
        kitchen = st.checkbox("Nhà bếp", value=True)
        terrace = st.checkbox("Sân thượng")
        parking = st.checkbox("Chỗ đỗ ô tô")

    errors = []
    if width is not None and length is not None and width > length:
        errors.append("Chiều ngang lớn hơn chiều dài; hãy kiểm tra lại (hoặc đảo hai giá trị).")
    if width is not None and length is not None and not 0.5 <= width * length / area <= 1.5:
        errors.append(f"Ngang × dài = {width * length:.0f} m² lệch nhiều so với diện tích {area:.0f} m².")

    if st.button("Định giá", type="primary"):
        for e in errors:
            st.warning(e)
        row = coords[(coords["city"] == city) & (coords["district"] == district)].iloc[0]
        X = pd.DataFrame([{
            "area_m2": area, "width_m": width, "length_m": length, "road_width_m": road,
            "floors": floors, "bedrooms": bedrooms,
            "width_length_ratio": (width / length) if width and length else np.nan,
            "dist_center_km": row["dist_center_km"], "lat": row["lat"], "lon": row["lon"],
            **{c: row[c] for c in AMENITY_COLS},
            "has_dining_room": float(dining), "has_kitchen": float(kitchen),
            "has_terrace": float(terrace), "has_parking": float(parking),
            "city": city, "district": district, "house_type": house_type,
        }])
        X = to_model_input(X)
        pred = float(learnt["model"].predict(X)[0])
        iv = learnt["interval"]
        lo, hi = np.expm1(np.log1p(pred) + iv["q_lo"]), np.expm1(np.log1p(pred) + iv["q_hi"])

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Giá dự báo", f"{pred:.2f} tỷ")
        m2.metric("Khoảng dự báo 95%", f"{lo:.2f} – {hi:.2f} tỷ")
        m3.metric("Đơn giá ước tính", f"{pred * 1000 / area:.0f} triệu/m²")
        st.caption(f"Khoảng dự báo 95%: với các căn có đặc điểm tương tự, khoảng 95% có giá nằm trong khoảng này "
                   f"(độ phủ kiểm chứng trên tập kiểm thử: {iv['test_coverage']:.1%}).")
        st.info("Giá dự báo áp dụng cho nhà có **giấy tờ pháp lý đầy đủ** (sổ đỏ/sổ hồng). Dữ liệu huấn luyện "
                "không có thông tin pháp lý; nhà chưa có sổ, giấy tay hoặc đang tranh chấp thường có giá thấp hơn.")

        st.subheader(f"Vị trí và tiện ích xung quanh: {district}")
        a = st.columns(6)
        a[0].metric("Cách trung tâm", f"{row['dist_center_km']:.1f} km")
        for col, c in zip(a[1:], AMENITY_COLS):
            col.metric(AMENITY_NAMES[c], int(row[c]))
        st.caption("Số tiện ích trong bán kính 2 km quanh tâm quận/huyện (OpenStreetMap).")

        out_of_range = [name for name, v, col in [("diện tích", area, "area_m2"), ("độ rộng đường", road, "road_width_m"),
                                                  ("số tầng", floors, "floors"), ("số phòng ngủ", bedrooms, "bedrooms")]
                        if not rng.loc["min", col] <= v <= rng.loc["max", col]]
        if out_of_range:
            st.warning("Giá trị " + ", ".join(out_of_range) + " nằm ngoài miền dữ liệu huấn luyện; "
                       "kết quả dự báo kém tin cậy.")
        if pred > 15:
            st.info("Mô hình có xu hướng định giá thấp nhà trên 15 tỷ (phân khúc ít mẫu); giá thực tế có thể cao hơn.")

        st.subheader("Mô hình tính ra giá này như thế nào")
        Xt, base_log, final_log = decompose(X)
        st.markdown("**1. Năm mô hình con dự báo, mô hình meta kết hợp bằng tổng có trọng số** (thang log1p của giá)")
        st.dataframe(stacking_table(base_log).round(3), hide_index=True, width="stretch")
        st.markdown(f"Hệ số chặn {META_B:.3f} + tổng đóng góp {float(base_log @ META_W):.3f} = **{final_log:.4f}** → "
                    f"$e^{{{final_log:.4f}}} - 1$ = **{np.expm1(final_log):.2f} tỷ**.")

        st.markdown("**2. Mỗi thuộc tính đẩy giá lên hoặc xuống bao nhiêu** (giá trị SHAP)")
        base, contrib = explain(Xt)
        shown = {"area_m2": f"{area:g} m²", "road_width_m": f"{road:g} m", "floors": floors,
                 "width_m": "không rõ" if width is None else f"{width:g} m", "house_type": house_type,
                 "district": district, "dist_center_km": f"{row['dist_center_km']:.1f} km",
                 "has_parking": "có" if parking else "không",
                 **{c: int(row[c]) for c in AMENITY_COLS}}
        fig, table = shap_waterfall(base, contrib, shown)
        c1, c2 = st.columns([3, 2])
        c1.pyplot(fig)
        c2.dataframe(table.round(2), hide_index=True, width="stretch")
        top = table.iloc[0]
        st.markdown(f"""
Mô hình bắt đầu từ **giá tham chiếu {np.expm1(base):.2f} tỷ** (mức giá của một căn "trung bình" trong dữ liệu),
rồi cộng dần ảnh hưởng của từng thuộc tính. Thuộc tính tác động mạnh nhất là **{top['Thuộc tính']}**
({top['Ảnh hưởng (%)']:+.0f}% so với căn trung bình). Màu xanh làm tăng giá, màu đỏ làm giảm giá.
Tổng các đóng góp bằng đúng giá dự báo.
""")
        with st.expander("Cách tính giá trị SHAP cho mô hình Stacking"):
            st.markdown("""
- Dự báo của Stacking là tổng có trọng số của năm mô hình con, nên giá trị SHAP của Stacking bằng tổng có trọng số
  giá trị SHAP của từng mô hình con (tính chất tuyến tính của giá trị Shapley).
- Mô hình tuyến tính: SHAP của một cột = hệ số × (giá trị của căn − giá trị trung bình).
- Mô hình cây: tính chính xác bằng `TreeExplainer`.
- Cả hai cùng so với một nền chung gồm 300 căn trong dữ liệu. Các cột one-hot của quận/huyện và loại nhà được cộng
  lại theo thuộc tính gốc.
- Giá trị SHAP nằm trên thang log1p; biểu đồ cộng dần các giá trị này rồi đổi về tỷ VNĐ, nên phần đóng góp bằng tiền
  phụ thuộc thứ tự cộng, còn cột "Ảnh hưởng (%)" thì không.
""")

        st.subheader("Các căn tương tự trong dữ liệu")
        comps = features[(features["district"] == district) & (features["house_type"] == house_type)
                         & features["area_m2"].between(area * 0.7, area * 1.3)].copy()
        if len(comps):
            comps["giá/m² (triệu)"] = (comps["price_billion_vnd"] * 1000 / comps["area_m2"]).round(0)
            st.dataframe(comps[["area_m2", "road_width_m", "floors", "bedrooms", "price_billion_vnd", "giá/m² (triệu)"]]
                         .sort_values("area_m2").head(10), width="stretch", hide_index=True)
            st.caption(f"{len(comps)} căn cùng quận, cùng loại nhà, diện tích ±30%. "
                       f"Giá trung vị: {comps['price_billion_vnd'].median():.2f} tỷ.")
        else:
            st.caption("Không có căn tương tự (cùng quận, cùng loại nhà, diện tích ±30%) trong dữ liệu.")
