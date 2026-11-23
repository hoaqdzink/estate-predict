"""Ứng dụng trình bày và định giá bất động sản (Đề tài 06).

Chạy: streamlit run app/streamlit_app.py
Yêu cầu: đã thực thi notebooks/01_cleaning.ipynb để tạo models/learnt.pkl và reports/artifacts.pkl.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from core import data, ui  # noqa: E402

st.set_page_config(page_title="Khai phá dữ liệu bất động sản đô thị", page_icon="🏠", layout="wide")
ui.setup()

missing = data.missing_files()
if missing:
    st.error("Thiếu các tệp kết quả: " + ", ".join(missing)
             + ". Hãy thực thi toàn bộ notebook `notebooks/01_cleaning.ipynb` trước (khoảng 3 phút).")
    st.stop()

V = "views/"
pages = {
    "Giới thiệu": [st.Page(V + "p00_overview.py", title="0. Tổng quan", icon=":material/home:", default=True)],
    "Chuẩn bị dữ liệu (KDD 1–3)": [
        st.Page(V + "p01_data.py", title="1. Dữ liệu & mất cân bằng", icon=":material/database:"),
        st.Page(V + "p01b_hand.py", title="1b. Ví dụ giải tay", icon=":material/calculate:"),
        st.Page(V + "p02_preprocess.py", title="2. Tiền xử lý", icon=":material/cleaning_services:"),
    ],
    "Khai phá dữ liệu (KDD 4)": [
        st.Page(V + "p03_selection.py", title="3. Chọn mẫu & Grid Search", icon=":material/tune:"),
        st.Page(V + "p04_model.py", title="4. Stacking & learnt.pkl", icon=":material/account_tree:"),
        st.Page(V + "p05_interval.py", title="5. Khoảng dự báo (bổ sung)", icon=":material/straighten:"),
    ],
    "Diễn giải & đánh giá (KDD 5)": [
        st.Page(V + "p06_compare.py", title="6. So sánh mô hình", icon=":material/leaderboard:"),
        st.Page(V + "p07_reliability.py", title="7. Đánh giá & độ tin cậy", icon=":material/verified:"),
        st.Page(V + "p08_explain.py", title="8. Giải thích kết quả (SHAP)", icon=":material/psychology:"),
    ],
    "Tri thức & ứng dụng": [
        st.Page(V + "p09_valuation.py", title="9. Định giá trực tuyến", icon=":material/payments:"),
        st.Page(V + "p10_conclusion.py", title="10. Tri thức & kết luận", icon=":material/flag:"),
    ],
}
nav = st.navigation(pages, expanded=True)


def reset_config():
    """Chạy trước khi trang được vẽ lại, nên được phép đặt lại giá trị của các ô cấu hình."""
    for k, v in data.DEFAULTS.items():
        st.session_state[k] = v


with st.sidebar:
    L = data.learnt()
    tm = L["test_metrics"]
    st.markdown(f"""<div class="side-card"><div class="k">Đề tài 06 · {L['model_name']}</div>
<div class="row"><span>R² kiểm thử</span><b>{data.vn(tm['R²'], 3)}</b></div>
<div class="row"><span>RMSE</span><b>{data.vn(tm['RMSE'])} tỷ</b></div>
<div class="row"><span>MAPE</span><b>{data.vn(tm['MAPE (%)'], 1)}%</b></div>
<div class="row"><span>Độ phủ khoảng 95%</span><b>{data.vn(L['interval']['test_coverage'] * 100, 1)}%</b></div></div>""",
                unsafe_allow_html=True)
    with st.expander("⚙️ Cấu hình phân tích", expanded=False):
        for k, v in data.DEFAULTS.items():
            st.session_state.setdefault(k, v)
        st.selectbox("Thành phố (trang dữ liệu, tiền xử lý)", ["Tất cả", "Hà Nội", "Hồ Chí Minh"], key="cfg_city")
        st.slider("Ranh giới phân khúc giá (tỷ)", 1.0, 40.0, step=1.0, key="cfg_segments")
        st.slider("Hệ số IQR (k)", 1.0, 3.0, step=0.1, key="cfg_iqr_k")
        st.select_slider("Mức Mahalanobis (phân vị χ²)", [0.95, 0.975, 0.99, 0.999], key="cfg_maha_p")
        st.slider("Mức khoảng dự báo (%)", 50, 99, key="cfg_level")
        st.select_slider("Số lần lặp bootstrap", [200, 500, 1000, 2000], key="cfg_boot")
        st.number_input("Hạt giống ngẫu nhiên (ví dụ giải tay)", 0, 9999, key="cfg_seed")
        st.button("Khôi phục mặc định", width="stretch", on_click=reset_config)
        st.caption("Cấu hình chỉ thay đổi các phân tích trình bày; mô hình đã huấn luyện cố định.")

nav.run()
