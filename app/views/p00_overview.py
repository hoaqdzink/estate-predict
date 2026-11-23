import streamlit as st

from core import data, ui

L, A = data.learnt(), data.artifacts()
tm = L["test_metrics"]
ui.hero("Khai phá dữ liệu · Đề tài 06", "Khai phá dữ liệu giá bất động sản đô thị",
        "Urban Real Estate Valuation & Hedonic Pricing: áp dụng quy trình khai phá tri thức (KDD) với tác vụ "
        "hồi quy để tìm ra các yếu tố quyết định giá nhà ở Hà Nội, TP.HCM và dự báo giá chuyển nhượng.",
        ["Tác vụ: hồi quy dữ liệu", "CLO: L.O.3.1 · L.O.3.3 · L.O.4.1 · L.O.4.2", "Quy trình KDD 5 giai đoạn",
         "3.500 tin rao, 36 loại lỗi", "7 mô hình hồi quy"])

m = st.columns(4)
m[0].metric("Bản ghi cho mô hình", data.vn(len(data.features()), 0), "từ 3.510 dòng thô", delta_color="off")
m[1].metric("RMSE kiểm thử", f"{data.vn(tm['RMSE'])} tỷ", f"MAE {data.vn(tm['MAE'])} tỷ", delta_color="off")
m[2].metric("R² kiểm thử", data.vn(tm["R²"], 3), f"MAPE {data.vn(tm['MAPE (%)'], 1)}%", delta_color="off")
m[3].metric("Độ phủ khoảng 95%", f"{data.vn(L['interval']['test_coverage'] * 100, 1)}%", f"mô hình {L['model_name']}", delta_color="off")

st.write("")
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.subheader("Thông tin đề tài")
    ui.info_grid([
        ("Tác vụ khai phá chính", "Hồi quy dữ liệu (Data Regression)"),
        ("Chuẩn đầu ra", "L.O.3.1, L.O.3.3, L.O.4.1, L.O.4.2"),
        ("Tập dữ liệu", "Tin rao bán nhà Hà Nội, TP.HCM (web scraping)"),
        ("Biến mục tiêu", "Giá rao bán (tỷ VNĐ)"),
        ("Quy trình", "Khai phá tri thức từ dữ liệu (KDD)"),
        ("Tri thức cần tìm", "Thuộc tính nào quyết định giá, mức độ ảnh hưởng, độ tin cậy của dự báo"),
    ])
with c2:
    st.subheader("Bài toán")
    ui.box("idea", """
**Bài toán:** từ hàng nghìn tin rao bán nhà thu thập tự động (nhiều lỗi, không đồng nhất), khai phá ra **tri thức về giá**: thuộc tính nào quyết định giá, ảnh hưởng bao nhiêu, và dùng tri thức đó để ước lượng giá một căn nhà mới kèm mức độ tin cậy.
**Tác vụ khai phá: hồi quy.** Biến mục tiêu là giá (biến số liên tục), nên đây là bài toán hồi quy chứ không phải phân lớp hay gom cụm.
**Cơ sở lý thuyết, định giá hedonic:** giá một bất động sản là tổng giá trị các thuộc tính cấu thành nó (thêm 1 m², mặt tiền thay vì trong hẻm, gần trung tâm hơn 1 km...). Mô hình hồi quy ước lượng phần đóng góp của từng thuộc tính từ dữ liệu.
""")

st.subheader("Quy trình khai phá tri thức (KDD)")
ui.steps([
    ("Lựa chọn dữ liệu", "3.510 dòng tin rao thô, 21 thuộc tính (trang 1)"),
    ("Tiền xử lý", "Làm sạch 36 loại lỗi, loại trùng, ngoại lai IQR: 2.819 bản ghi (trang 2)"),
    ("Biến đổi", "Log1p, tọa độ, khoảng cách tới trung tâm, tiện ích, trích chọn thuộc tính (trang 2, 3)"),
    ("Khai phá dữ liệu", "Hồi quy: Ridge, Lasso, ElasticNet, GBR, LightGBM, Stacking; Grid Search CV (trang 3, 4)"),
    ("Diễn giải & đánh giá", "RMSE, MAE, R², MAPE, bootstrap, SHAP, khoảng dự báo (trang 5–8)"),
    ("Tri thức", "Các yếu tố quyết định giá; công cụ định giá (trang 9, 10)"),
])
st.caption("KDD (Knowledge Discovery in Databases): quy trình khai phá tri thức gồm lựa chọn, tiền xử lý, biến đổi, khai phá và diễn giải/đánh giá. "
           "Khai phá dữ liệu là bước cốt lõi, nhưng phần lớn công sức nằm ở các bước chuẩn bị dữ liệu trước đó.")

st.subheader("Đối chiếu yêu cầu")
t1, t2, t3 = st.tabs(["Đề tài 06", "Yêu cầu trình bày", "Hướng dẫn 5 mục"])
with t1:
    st.markdown("""
| Yêu cầu của đề tài | Trang |
|---|---|
| Bước 1. Phân phối lệch phải, biến đổi Box-Cox / Log1p | 2. Tiền xử lý |
| Bước 1. Geocoding, tọa độ, khoảng cách tới trung tâm; tiện ích xung quanh | 2. Tiền xử lý |
| Bước 1. Loại ngoại lai bằng Mahalanobis / IQR | 2. Tiền xử lý |
| Bước 2. Ridge, Lasso, ElasticNet, Gradient Boosting, LightGBM, Stacking | 3, 4 |
| Bước 3. RMSE, MAE, R², MAPE trên tập kiểm thử độc lập | 6, 7 |
| Bước 4. Trang định giá trực tuyến kèm khoảng 95% | 5, 9 |
| Tình trạng pháp lý | Không có trong dữ liệu, ghi nhận là hạn chế (trang 10) |
""")
with t2:
    st.markdown("""
| Yêu cầu | Trang |
|---|---|
| Dữ liệu giả lập, mất cân bằng, chưa tiền xử lý (thiếu, nhiễu, ngoại lai) | 1, 2 |
| Chọn mẫu phản ánh đặc điểm bài toán, grid search cross-validation, trích chọn thuộc tính | 3 |
| Lưu kết quả học được (`learnt.pkl`) | 4 |
| So sánh với các mô hình khác | 6 |
| Đánh giá kết quả đạt được | 7, 8, 10 |
| Mạch trình bày: Ý tưởng → Dẫn dắt → Phản biện → Minh chứng | Mọi trang |
""")
with t3:
    st.markdown("""
| Mục | Trang |
|---|---|
| 1. Dữ liệu đủ lớn, chất lượng, phản ánh đặc điểm bài toán | 1, 2 |
| 2. Ảnh hưởng của thuộc tính lên mô hình | 3 (trích chọn), 8 (SHAP) |
| 3. Mô hình huấn luyện được là gì, cách đọc lên để dự báo | 4 |
| 4. Độ tin cậy của mô hình | 5, 7 |
| 5. Giải thích kết quả từ cách mô hình hoạt động | 1b, 4, 8, 9 |
""")

ui.box("lead", """
Mỗi trang trình bày theo cùng một mạch: **Ý tưởng** (vì sao làm bước này) → **Dẫn dắt** (làm như thế nào, số liệu từng bước) → **Phản biện** (lựa chọn thay thế, giới hạn) → **Minh chứng** (kết quả kiểm chứng).
Thanh bên trái có **Cấu hình phân tích**: thay đổi thành phố, ranh giới phân khúc giá, hệ số IQR, mức Mahalanobis, mức khoảng dự báo, số lần bootstrap. Các bảng và biểu đồ liên quan được tính lại ngay; mô hình đã huấn luyện không thay đổi.
""")
