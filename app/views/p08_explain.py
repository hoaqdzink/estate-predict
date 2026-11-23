import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import data, model, ui

ui.header("KDD 5 · Diễn giải", "8. Giải thích kết quả (SHAP)",
          "Mỗi thuộc tính đẩy giá lên hay xuống bao nhiêu, trên toàn tập kiểm thử và cho từng căn nhà")
A, L = data.artifacts(), data.learnt()
S, X = A["shap"]["values"], A["test"]["X"]

ui.box("idea", """
**Giá trị SHAP** (SHapley Additive exPlanations, dựa trên giá trị Shapley của lý thuyết trò chơi) chia một dự báo thành **tổng đóng góp của từng thuộc tính**: dự báo = giá tham chiếu + Σ đóng góp. Đóng góp dương làm tăng giá, âm làm giảm giá. Vì mô hình học trên thang log1p, đóng góp +0,1 tương ứng với giá tăng khoảng 10%.
Cách này nối **ý tưởng định giá hedonic** (giá là tổng giá trị các thuộc tính) với **cách mô hình hoạt động** thực tế.
""")

t1, t2, t3 = st.tabs(["8.1 Toàn tập kiểm thử", "8.2 Quan hệ thuộc tính – giá", "8.3 Từng căn nhà"])
with t1:
    imp = S.abs().mean().sort_values(ascending=False).rename("Trung bình |SHAP|").reset_index().rename(columns={"index": "f"})
    imp["Thuộc tính"] = imp["f"].map(data.FEATURE_LABELS)
    imp["Ảnh hưởng (%)"] = (np.exp(imp["Trung bình |SHAP|"]) - 1) * 100
    c1, c2 = st.columns([5, 4])
    c1.altair_chart(alt.Chart(imp, title="Mức đóng góp trung bình (thành phần LightGBM)").mark_bar().encode(
        y=alt.Y("Thuộc tính:N", sort="-x", title=None), x="Trung bình |SHAP|:Q",
        tooltip=["Thuộc tính", alt.Tooltip("Trung bình |SHAP|:Q", format=".4f"), alt.Tooltip("≈ ảnh hưởng (%):Q", format=".1f")]),
        use_container_width=True)
    c2.dataframe(imp[["Thuộc tính", "Trung bình |SHAP|", "Ảnh hưởng (%)"]].round(3), hide_index=True, width="stretch", height=420)

    long = S.reset_index().melt(id_vars="index", var_name="f", value_name="SHAP")
    vals = X.copy()
    vals["house_type"] = (vals["house_type"] == "Nhà mặt tiền").astype(float)
    vals["city"] = (vals["city"] == "Hồ Chí Minh").astype(float)
    num = vals.drop(columns=["district"]).apply(pd.to_numeric, errors="coerce").rank(pct=True)
    long["Giá trị (thứ hạng)"] = [num.at[i, f] if f in num.columns else np.nan for i, f in zip(long["index"], long["f"])]
    long["Thuộc tính"] = long["f"].map(data.FEATURE_LABELS)
    top = imp["Thuộc tính"].head(10).tolist()
    st.altair_chart(alt.Chart(long[long["Thuộc tính"].isin(top)], title="Chiều hướng ảnh hưởng (đỏ: giá trị cao, xanh: giá trị thấp, xám: quận/huyện)").mark_circle(size=12, opacity=0.6).encode(
        x=alt.X("SHAP:Q", title="Giá trị SHAP (thang log1p)"),
        y=alt.Y("Thuộc tính:N", sort=top, title=None), yOffset="jitter:Q",
        color=alt.Color("Giá trị (thứ hạng):Q", scale=alt.Scale(scheme="redblue", reverse=True))
    ).transform_calculate(jitter="random()").properties(height=420), use_container_width=True)

with t2:
    numeric = [f for f in ["area_m2", "road_width_m", "dist_center_km", "floors", "n_markets", "n_schools", "n_transit", "width_m"] if f in S.columns]
    f = st.selectbox("Thuộc tính", numeric, format_func=data.FEATURE_LABELS.get)
    dep = pd.DataFrame({"Giá trị": X[f], "SHAP": S[f], "Loại nhà": X["house_type"], "Quận": X["district"]})
    st.altair_chart(alt.Chart(dep, title=f"Ảnh hưởng của {data.FEATURE_LABELS[f].lower()} lên giá").mark_circle(size=22, opacity=0.6).encode(
        x=alt.X("Giá trị:Q", title=data.FEATURE_LABELS[f]), y=alt.Y("SHAP:Q", title="Giá trị SHAP (thang log1p)"),
        color="Loại nhà:N", tooltip=["Quận", "Giá trị", alt.Tooltip("SHAP:Q", format=".3f")]).interactive()
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q"), use_container_width=True)
    st.caption("Mỗi điểm là một căn trong tập kiểm thử. Đường cong cho thấy quan hệ phi tuyến mà mô hình học được, ví dụ diện tích tăng thì giá tăng nhưng chậm dần.")

with t3:
    feat = data.features()
    i = st.number_input("Chọn căn trong tập kiểm thử (0 – 563)", 0, len(X) - 1, 0)
    row = X.iloc[[int(i)]]
    Xin = model.to_model_input(row)
    Xt, base_log, final_log = model.decompose(Xin)
    base, contrib = model.explain(Xt)
    values = {"area_m2": f"{row['area_m2'].iloc[0]:g} m²", "road_width_m": f"{row['road_width_m'].iloc[0]:g} m",
              "house_type": row["house_type"].iloc[0], "district": row["district"].iloc[0],
              "dist_center_km": f"{row['dist_center_km'].iloc[0]:.1f} km"}
    fig, table = model.waterfall(base, contrib, values)
    c1, c2 = st.columns([3, 2])
    c1.pyplot(fig)
    c2.dataframe(table.round(2), hide_index=True, width="stretch")
    c = st.columns(3)
    c[0].metric("Giá tham chiếu", f"{data.vn(np.expm1(base))} tỷ")
    c[1].metric("Giá dự báo", f"{data.vn(np.expm1(final_log))} tỷ")
    c[2].metric("Giá thực tế", f"{data.vn(A['test']['y'][int(i)])} tỷ")
    ui.box("evidence", f"Giá tham chiếu + tổng đóng góp = {data.vn(base + contrib.sum(), 4)} trên thang log1p, đúng bằng dự báo của Stacking ({data.vn(final_log, 4)}). Phần giải thích khớp chính xác với cách mô hình tính ra giá.")

ui.box("critique", """
- Biểu đồ 8.1 và 8.2 giải thích **thành phần LightGBM** (trọng số lớn nhất trong Stacking); 8.3 giải thích **toàn bộ Stacking** bằng cách cộng có trọng số SHAP của năm mô hình con (tính chất tuyến tính của giá trị Shapley).
- SHAP mô tả **mô hình phản ứng thế nào**, không phải quan hệ nhân quả. Khoảng cách tới trung tâm và số tiện ích tương quan với nhau, nên mô hình chia ảnh hưởng của vị trí cho cả hai.
- Đóng góp bằng tỷ VNĐ trong biểu đồ thác nước phụ thuộc thứ tự cộng (do đổi từ thang log); cột "Ảnh hưởng (%)" thì không.
""")
ui.box("evidence", "Thứ tự quan trọng theo SHAP (diện tích, độ rộng đường, loại nhà, số tầng, nhóm vị trí) trùng với độ quan trọng hoán vị ở trang 3, và phù hợp lý thuyết hedonic: **quy mô, khả năng tiếp cận, vị trí** quyết định giá.")
