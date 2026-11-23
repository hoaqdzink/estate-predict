import numpy as np
import streamlit as st
from scipy import stats

from core import data, ui

ui.header("Tri thức khai phá", "10. Tri thức khai phá được & kết luận",
          "Các quy luật về giá rút ra từ dữ liệu, kết quả mô hình, các lựa chọn đã phản biện, hạn chế và hướng phát triển")
L = data.learnt()
tm = L["test_metrics"]

f = data.features()
A = data.artifacts()
med = f.groupby(["city", "house_type"])["price_billion_vnd"].median()
ppm2 = f["price_billion_vnd"] * 1000 / f["area_m2"]
rho = {c: stats.spearmanr(g["dist_center_km"], ppm2[g.index])[0] for c, g in f.groupby("city")}
road = f[["road_width_m", "price_billion_vnd"]].corr().iloc[0, 1]
shap_top = A["shap"]["values"].abs().mean().sort_values(ascending=False).index[:4]
hi_share = (f["price_billion_vnd"] >= 15).mean()

st.subheader("Tri thức khai phá được từ dữ liệu")
ui.box("idea", "Mục tiêu của khai phá dữ liệu là tìm ra **tri thức có ích, chưa biết trước và kiểm chứng được** từ dữ liệu. Bảng dưới tóm tắt các quy luật về giá rút ra từ 2.819 tin rao, kèm bằng chứng và trang trình bày chi tiết.")
st.markdown(f"""
| # | Tri thức (quy luật về giá) | Bằng chứng | Trang |
|---|---|---|---|
| 1 | **Loại nhà là yếu tố phân hóa giá mạnh nhất:** nhà mặt tiền có giá trung vị gấp khoảng {data.vn(med['Hà Nội', 'Nhà mặt tiền'] / med['Hà Nội', 'Nhà trong hẻm'], 1)} lần nhà trong hẻm ở Hà Nội, {data.vn(med['Hồ Chí Minh', 'Nhà mặt tiền'] / med['Hồ Chí Minh', 'Nhà trong hẻm'], 1)} lần ở TP.HCM | Giá trung vị theo nhóm; SHAP | 2, 8 |
| 2 | **Độ rộng đường trước nhà tương quan với giá mạnh nhất** trong các biến số (r ≈ {data.vn(road)}) | Ma trận tương quan; độ quan trọng hoán vị | 3, 8 |
| 3 | **Giá/m² giảm khi xa trung tâm**, rõ ở TP.HCM (ρ ≈ {data.vn(rho['Hồ Chí Minh'])}) hơn ở Hà Nội (ρ ≈ {data.vn(rho['Hà Nội'])}), vì Hà Nội có nhiều khu giá cao không sát trung tâm | Tương quan Spearman | 2 |
| 4 | **Khu vực nhiều tiện ích xung quanh có giá cao hơn**, nhưng thông tin này phần lớn trùng với khoảng cách tới trung tâm khi chỉ có dữ liệu cấp quận | Spearman; so sánh mô hình có/không có tiện ích | 2, 3 |
| 5 | **Bốn thuộc tính quyết định phần lớn giá:** {", ".join(data.FEATURE_LABELS[x].lower() for x in shap_top)}; phù hợp lý thuyết hedonic (quy mô, khả năng tiếp cận, vị trí) | SHAP và độ quan trọng hoán vị cho cùng kết quả | 3, 8 |
| 6 | **Quan hệ giữa thuộc tính và giá là phi tuyến** (giá tăng chậm dần theo diện tích; tác động của diện tích phụ thuộc loại nhà): mô hình cây vượt nhóm tuyến tính | So sánh mô hình; đồ thị phụ thuộc SHAP | 6, 8 |
| 7 | **Thị trường lệch về phân khúc giá thấp và trung bình:** nhà từ 15 tỷ chỉ chiếm {data.vn(hi_share * 100, 0)}%, và đây là vùng mô hình định giá thấp | Phân tích mất cân bằng; sai số theo phân khúc | 1, 7 |
| 8 | **Tiện ích trong nhà gần như không mang thông tin về giá:** bếp, sân thượng, phòng ăn bị loại khi trích chọn (tin rao thường bỏ trống các mục này); số phòng ngủ cũng bị loại vì trùng thông tin với diện tích và số tầng | Trích chọn thuộc tính | 3 |
""")

st.subheader("Kết quả của bước khai phá (mô hình hồi quy)")
st.markdown(f"""
| Nội dung | Kết quả |
|---|---|
| Dữ liệu | 3.510 dòng thô (36 loại lỗi) → 2.945 bản ghi sạch → 2.819 bản ghi cho mô hình |
| Thuộc tính | 14 thuộc tính sau trích chọn, gồm khoảng cách tới trung tâm và 5 nhóm tiện ích xung quanh |
| Mô hình | **{L['model_name']} Ensemble** (Ridge, Lasso, ElasticNet, Gradient Boosting, LightGBM; meta RidgeCV) |
| Tập kiểm thử độc lập | RMSE {data.vn(tm['RMSE'])} tỷ · MAE {data.vn(tm['MAE'])} tỷ · R² {data.vn(tm['R²'], 3)} · MAPE {data.vn(tm['MAPE (%)'], 1)}% |
| Khoảng dự báo 95% | Độ phủ {data.vn(L['interval']['test_coverage'] * 100, 1)}% trên tập kiểm thử |
| Thuộc tính quan trọng nhất | Diện tích, độ rộng đường, loại nhà, số tầng, nhóm vị trí |
""")
ui.box("evidence", "Mô hình hồi quy giải thích được khoảng 3/4 biến thiên của giá rao bán, tổng quát hóa ổn định (không quá khớp), và kết quả giải thích được bằng lý thuyết định giá hedonic. Mô hình phù hợp để **ước lượng sơ bộ** và đưa ra khoảng giá tham khảo cho môi giới.")

st.subheader("Các lựa chọn đã phản biện")
Q = [
    ("Box-Cox hay Log1p?", "Box-Cox cho phân phối đẹp hơn nhưng sai số dự báo lớn hơn (λ âm làm phép biến đổi ngược không ổn định). Chọn **Log1p**."),
    ("IQR hay Mahalanobis?", "Mahalanobis giả định phân phối chuẩn nhiều chiều, đánh dấu nhầm nhà nhiều phòng hợp lệ và bỏ qua 13% bản ghi thiếu dữ liệu. Chọn **IQR trên log(giá/m²)**, Mahalanobis để kiểm tra chéo."),
    ("Có nên cân bằng lại dữ liệu bằng lấy mẫu?", "Không. Với hồi quy, nhân bản nhà đắt gây quá khớp. Thay bằng chia phân tầng, biến đổi log, gộp quận ít mẫu, đánh giá theo phân khúc."),
    ("Bỏ bớt thuộc tính?", "Bỏ 8 thuộc tính không phân biệt được với nhiễu; sai số trung bình không tăng, mô hình gọn hơn."),
    ("Tiện ích xung quanh có giúp dự báo?", "Gần như không đổi sai số vì trùng thông tin với khoảng cách tới trung tâm (chỉ có ở cấp quận). Giữ lại vì đề tài yêu cầu và có ý nghĩa diễn giải."),
    ("Stacking có đáng với độ phức tạp?", "Stacking tốt hơn LightGBM có ý nghĩa thống kê (bootstrap), ngang Gradient Boosting, và tốt hơn nhóm tuyến tính rõ ở MAE, MAPE. Mức cải thiện nhỏ so với độ phức tạp; giữ Stacking theo tiêu chí chọn đặt trước, không chọn lại theo tập kiểm thử."),
    ("Loại ngoại lai trước khi chia tập có rò rỉ?", "Không: đó là loại bản ghi nhập sai, không học tham số. Mọi bước học (điền thiếu, chuẩn hóa, mã hóa, chọn tham số) nằm trong pipeline, chỉ học trên tập huấn luyện."),
    ("Khoảng tin cậy hay khoảng dự báo?", "Khoảng dự báo, vì môi giới cần giá của một căn cụ thể, không phải giá trung bình."),
    ("Vì sao không có tình trạng pháp lý?", "Dữ liệu gốc không thu thập thuộc tính này. Không giả lập, vì mô hình khi đó chỉ học lại giả định của người làm."),
]
for q, a in Q:
    with st.expander(q):
        st.markdown(a)

st.subheader("Hạn chế")
ui.box("critique", """
- **Dữ liệu:** giá rao bán, không phải giá giao dịch; thu thập trước 01/2022; không có tình trạng pháp lý; vị trí chỉ ở cấp quận/huyện; lỗi trong dữ liệu được bổ sung nhân tạo; các cột `url`, `title`, `contact`... là giả lập.
- **Mô hình:** sai số tương đối khoảng 25%; định giá thấp phân khúc giá cao do ít mẫu; khoảng dự báo ở phân khúc này có độ phủ thấp hơn danh nghĩa.
- **Tiện ích xung quanh:** tính theo tâm quận nên không phân biệt được các căn trong cùng quận; dữ liệu OpenStreetMap ở ngoại thành chưa đầy đủ.
""")
st.subheader("Hướng phát triển")
st.markdown("""
- Thu thập dữ liệu có **tọa độ từng căn** và **tình trạng pháp lý**; khi đó tiện ích xung quanh có giá trị dự báo riêng.
- Khai thác văn bản tiêu đề và mô tả tin rao (hướng nhà, nội thất, "chính chủ", "cần bán gấp").
- Bổ sung dữ liệu phân khúc giá cao; thử trọng số mẫu hoặc mô hình riêng cho nhà mặt tiền.
- Conformal theo nhóm để khoảng dự báo đạt độ phủ đúng ở từng phân khúc.
- Cập nhật dữ liệu định kỳ để phản ánh mặt bằng giá hiện tại.
""")
