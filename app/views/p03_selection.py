import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import data, ui

ui.header("KDD 3–4 · Chọn mẫu, thuộc tính, tham số", "3. Chọn mẫu, thuộc tính & Grid Search CV",
          "Chia dữ liệu phản ánh đặc điểm bài toán, chọn thuộc tính có đóng góp, dò tham số bằng cross-validation")
A = data.artifacts()
t1, t2, t3 = st.tabs(["3.1 Chọn mẫu", "3.2 Trích chọn thuộc tính", "3.3 Grid Search CV"])

with t1:
    ui.box("idea", f"""
Tập kiểm thử phải giống phân bố thực tế để đánh giá trung thực. Vì phân khúc giá cao ít mẫu (trang 1), chia ngẫu nhiên đơn thuần có thể làm tập kiểm thử thiếu nhà đắt. Dữ liệu được chia **80% huấn luyện ({data.vn(A['split']['train'], 0)} bản ghi), 20% kiểm thử ({data.vn(A['split']['test'], 0)} bản ghi), phân tầng theo phân khúc giá** (dưới 5 tỷ, 5–15 tỷ, trên 15 tỷ), `random_state = {A['split']['random_state']}`.
""")
    seg = pd.DataFrame({"Huấn luyện": pd.Series(A["oof"]["segment"]).value_counts(normalize=True),
                        "Kiểm thử": pd.Series(A["test"]["segment"]).value_counts(normalize=True)}) * 100
    seg = seg.reindex(["Dưới 5 tỷ", "5–15 tỷ", "Trên 15 tỷ"])
    c1, c2 = st.columns([2, 3])
    c1.dataframe(seg.round(1), width="stretch")
    folds = pd.DataFrame([{"Lần lặp": f"Fold {i + 1}", "Phần": f"Phần {j + 1}", "Vai trò": "Kiểm định" if i == j else "Huấn luyện"}
                          for i in range(5) for j in range(5)])
    c2.altair_chart(alt.Chart(folds, title="Cross-validation 5 fold trên tập huấn luyện").mark_rect(stroke="white", strokeWidth=3, cornerRadius=4).encode(
        x=alt.X("Phần:N", title=None, axis=alt.Axis(labelAngle=0, orient="top"), scale=alt.Scale(paddingInner=0)),
        y=alt.Y("Lần lặp:N", title=None, scale=alt.Scale(paddingInner=0)),
        color=alt.Color("Vai trò:N", scale=alt.Scale(domain=["Huấn luyện", "Kiểm định"], range=["#BFD3FA", "#EA7A2B"])),
        tooltip=["Lần lặp", "Phần", "Vai trò"]).properties(height=190), use_container_width=True)
    ui.box("critique", """
- **Tập kiểm thử tách ra ngay từ đầu** và không được dùng để chọn thuộc tính, tham số hay mô hình. Mọi lựa chọn dựa trên cross-validation 5 fold trong tập huấn luyện.
- Cross-validation dùng `KFold` xáo trộn thay vì phân tầng theo fold: mỗi fold khoảng 450 bản ghi, đủ lớn để tỷ lệ phân khúc gần với tổng thể.
""")
    ui.box("evidence", "Tỷ lệ ba phân khúc giá trong tập huấn luyện và kiểm thử gần như bằng nhau (chênh dưới 0,2 điểm %).")

with t2:
    ui.box("idea", """
Không phải thuộc tính nào cũng có ích: thuộc tính nhiễu làm mô hình phức tạp hơn mà không chính xác hơn. Hai phương pháp độc lập được dùng:
- **Lasso (chuẩn L1):** đưa hệ số của cột ít đóng góp về đúng 0;
- **Độ quan trọng hoán vị:** xáo trộn ngẫu nhiên một thuộc tính và đo RMSE của LightGBM tăng bao nhiêu trên tập kiểm định (25% của tập huấn luyện).
""")
    imp = A["perm_importance"].reset_index().rename(columns={"index": "Thuộc tính"})
    imp["Tên"] = imp["Thuộc tính"].map(data.FEATURE_LABELS)
    imp["Kết luận"] = np.where(imp["Thuộc tính"].isin(A["weak_features"]), "Loại", "Giữ")
    imp["lo"] = imp["Mức tăng RMSE (tỷ)"] - imp["Độ lệch chuẩn"]
    imp["hi"] = imp["Mức tăng RMSE (tỷ)"] + imp["Độ lệch chuẩn"]
    base = alt.Chart(imp, title="Độ quan trọng hoán vị (mức tăng RMSE khi xáo trộn, ± 1 độ lệch chuẩn)")
    st.altair_chart(base.mark_bar().encode(
        y=alt.Y("Tên:N", sort="-x", title=None), x=alt.X("Mức tăng RMSE (tỷ):Q"),
        color=alt.Color("Kết luận:N", scale=alt.Scale(domain=["Giữ", "Loại"], range=["#2563EB", "#CBD2DE"])),
        tooltip=["Tên", alt.Tooltip("Mức tăng RMSE (tỷ):Q", format=".4f"), alt.Tooltip("Độ lệch chuẩn:Q", format=".4f")])
        + base.mark_rule().encode(y=alt.Y("Tên:N", sort="-x"), x="lo:Q", x2="hi:Q"), use_container_width=True)
    st.markdown("**Quy tắc loại:** độ quan trọng không lớn hơn độ lệch chuẩn của chính nó, tức là không phân biệt được với nhiễu.")
    coef = A["lasso_coef"]
    nz = coef[coef != 0].rename("Hệ số").reset_index().rename(columns={"index": "Cột"})
    c1, c2 = st.columns([3, 2])
    c1.altair_chart(alt.Chart(nz, title=f"Hệ số Lasso khác 0 ({len(nz)}/{len(coef)} cột sau mã hóa)").mark_bar().encode(
        y=alt.Y("Cột:N", sort="-x", title=None), x="Hệ số:Q",
        color=alt.condition("datum['Hệ số'] > 0", alt.value("#16A34A"), alt.value("#DC2626"))), use_container_width=True)
    c2.markdown("**So sánh tập thuộc tính** (CV 5 fold, tỷ VNĐ)")
    c2.dataframe(A["feature_compare"][["Mô hình", "CV RMSE", "CV MAE", "CV R²"]].round(4), hide_index=True, width="stretch")
    c2.markdown("**Đóng góp của tiện ích xung quanh**")
    c2.dataframe(A["amenity_compare"][["Mô hình", "CV RMSE", "CV MAE", "CV R²"]].round(4), hide_index=True, width="stretch")
    ui.box("critique", """
- Bỏ 8 thuộc tính yếu giúp LightGBM giảm sai số nhưng Ridge tăng nhẹ; sai số trung bình không tăng nên chọn tập gọn hơn (14 thuộc tính).
- `city` có độ quan trọng bằng 0 vì thông tin thành phố đã nằm trong tọa độ và khoảng cách. `bedrooms` bị loại vì tương quan với diện tích và số tầng.
- Thêm 5 thuộc tính tiện ích xung quanh gần như **không** đổi sai số: chúng trùng thông tin với khoảng cách tới trung tâm. Vẫn giữ vì đề tài yêu cầu, có ý nghĩa diễn giải và không làm mô hình kém đi.
""")
    ui.box("evidence", "Bốn thuộc tính quan trọng nhất: **diện tích, độ rộng đường, loại nhà, khoảng cách tới trung tâm**. Kết quả trùng với SHAP ở trang 8: hai phương pháp độc lập cho cùng kết luận.")

with t3:
    ui.box("idea", "Mỗi mô hình có tham số điều khiển độ phức tạp (mức phạt, số cây, độ sâu...). Grid search thử **mọi tổ hợp** trong lưới tham số, đánh giá mỗi tổ hợp bằng cross-validation 5 fold, và chọn tổ hợp có **RMSE trung bình thấp nhất** trên đơn vị tỷ VNĐ.")
    sr = A["search_rows"].copy()
    sr["Tham số tốt nhất"] = sr["Tham số tốt nhất"].astype(str)
    st.dataframe(sr.round(4), hide_index=True, width="stretch")
    name = st.selectbox("Xem chi tiết lưới tham số", list(A["search_cv"]))
    cv = A["search_cv"][name].copy()
    pcols = [c for c in cv.columns if c.startswith("param_")]
    nice = {c: c.replace("param_regressor__model__", "") for c in pcols}
    cv = cv.rename(columns=nice)
    cv["CV RMSE"] = -cv["mean_test_score"]
    cv["Độ lệch chuẩn"] = cv["std_test_score"]
    params = list(nice.values())
    if len(params) == 1:
        cv[params[0]] = cv[params[0]].astype(float)
        st.altair_chart(alt.Chart(cv, title=f"{name}: CV RMSE theo {params[0]}").mark_line(point=True).encode(
            x=alt.X(f"{params[0]}:Q", scale=alt.Scale(type="log")), y=alt.Y("CV RMSE:Q", scale=alt.Scale(zero=False)),
            tooltip=[params[0], alt.Tooltip("CV RMSE:Q", format=".4f")]), use_container_width=True)
    elif len(params) == 2:
        st.altair_chart(alt.Chart(cv, title=f"{name}: CV RMSE theo hai tham số").mark_rect().encode(
            x=f"{params[0]}:O", y=f"{params[1]}:O", color=alt.Color("CV RMSE:Q", scale=alt.Scale(scheme="blues", reverse=True)),
            tooltip=params + [alt.Tooltip("CV RMSE:Q", format=".4f")]), use_container_width=True)
    else:
        par = st.selectbox("Tham số", params)
        st.altair_chart(alt.Chart(cv, title=f"{name}: CV RMSE theo {par} (mỗi điểm là một tổ hợp)").mark_circle(size=60).encode(
            x=f"{par}:O", y=alt.Y("CV RMSE:Q", scale=alt.Scale(zero=False)), color=alt.Color(f"{params[0]}:N"),
            tooltip=params + [alt.Tooltip("CV RMSE:Q", format=".4f")]), use_container_width=True)
    st.dataframe(cv[params + ["CV RMSE", "Độ lệch chuẩn", "rank_test_score"]].sort_values("rank_test_score")
                 .rename(columns={"rank_test_score": "Hạng"}).head(10).round(4), hide_index=True, width="stretch")
    ui.box("critique", """
- Lưới tham số cần **đủ rộng**: nếu tham số tốt nhất nằm ở biên lưới, giá trị tối ưu có thể nằm ngoài. Với Ridge và Lasso, tham số tốt nhất nằm giữa lưới.
- Chênh lệch RMSE giữa các tổ hợp tốt thường nhỏ hơn độ lệch chuẩn giữa các fold: nhiều tổ hợp tốt ngang nhau, nên không cần lưới dày hơn.
- Grid search tốn thời gian theo cấp số nhân với số tham số (LightGBM: 16 tổ hợp × 5 fold = 80 lần huấn luyện).
""")
