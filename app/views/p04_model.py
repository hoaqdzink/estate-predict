import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import data, model, ui

ui.header("KDD 4 · Mô hình khai phá được", "4. Mô hình Stacking & tệp learnt.pkl",
          "Mô hình huấn luyện được là gì, lưu những gì, và cách đọc lên để dự báo")
L, P = data.learnt(), model.parts()
feat = data.features()

ui.box("idea", f"""
Không có mô hình đơn lẻ nào tốt nhất ở mọi vùng dữ liệu: mô hình tuyến tính ổn định khi ngoại suy, mô hình cây học được quan hệ phi tuyến. **Stacking** kết hợp cả hai: năm mô hình con dự báo độc lập, rồi một **mô hình meta** học cách kết hợp các dự báo đó.
Mô hình được chọn theo CV RMSE thấp nhất: **{L['model_name']}**, huấn luyện trên {data.vn(L['trained_on'], 0)} bản ghi với {len(P['input_cols'])} thuộc tính đầu vào.
""")

t1, t2, t3, t4 = st.tabs(["4.1 Cấu trúc mô hình", "4.2 Nội dung learnt.pkl", "4.3 Đọc lên để dự báo", "4.4 Minh họa từng bước"])
with t1:
    st.graphviz_chart(f"""
digraph {{
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF3FA", fontname="Helvetica", fontsize=10];
  input [label="Thông số căn nhà\\n({len(P['input_cols'])} thuộc tính)", fillcolor="#FFF4E0"];
  prep [label="Tiền xử lý\\nđiền thiếu (trung vị / phổ biến nhất)\\nchuẩn hóa, one-hot\\n({len(P['tcols'])} cột)"];
  r [label="Ridge"]; l [label="Lasso"]; e [label="ElasticNet"]; g [label="Gradient Boosting"]; b [label="LightGBM"];
  meta [label="Mô hình meta RidgeCV\\nŷ = b + Σ wᵢ·ŷᵢ"];
  inv [label="Đổi về tỷ VNĐ\\ngiá = e^ŷ − 1"];
  out [label="Giá dự báo\\n+ khoảng dự báo", fillcolor="#E3F4E7"];
  input -> prep; prep -> {{r l e g b}}; {{r l e g b}} -> meta; meta -> inv; inv -> out;
}}""", width="stretch")
    params = {"Ridge": ["alpha"], "Lasso": ["alpha"], "ElasticNet": ["alpha", "l1_ratio"],
              "Gradient_Boosting": ["n_estimators", "learning_rate", "max_depth", "subsample"],
              "LightGBM": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"]}
    w = pd.DataFrame({"Mô hình con": [model.MODEL_LABELS[n] for n in P["base_names"]], "Trọng số meta": P["w"],
                      "Tham số (grid search)": [", ".join(f"{k}={est.get_params()[k]}" for k in params[n])
                                                for n, est in zip(P["base_names"], P["stack"].estimators_)]})
    c1, c2 = st.columns([3, 2])
    c1.dataframe(w.round(3), hide_index=True, width="stretch")
    c1.caption(f"Hệ số chặn của mô hình meta: {data.vn(P['b'], 4)}")
    c2.altair_chart(alt.Chart(w, title="Trọng số mô hình meta").mark_bar().encode(
        y=alt.Y("Mô hình con:N", sort="-x", title=None), x="Trọng số meta:Q",
        color=alt.condition("datum['Trọng số meta'] > 0", alt.value("#2563EB"), alt.value("#DC2626"))),
        use_container_width=True)
    st.markdown("""
| Tầng | Cách hoạt động |
|---|---|
| **Ridge, Lasso, ElasticNet** | Mỗi cột có một hệ số; dự báo là tổng có trọng số các cột (giống ví dụ giải tay ở trang 1b, có thêm mức phạt) |
| **Gradient Boosting, LightGBM** | Tổng của hàng trăm cây quyết định nhỏ; mỗi cây học phần sai số còn lại của các cây trước |
| **Mô hình meta** | Học trọng số kết hợp từ dự báo **ngoài fold** của các mô hình con, nên không bị đánh lừa bởi dự báo trên chính dữ liệu đã học |
""")
    ui.box("critique", "Hai mô hình cây nhận khoảng 4/5 tổng trọng số. Lasso có trọng số gần 0 (hơi âm) vì gần như trùng thông tin với Ridge và ElasticNet: mô hình meta tự giảm vai trò của mô hình dư thừa. Đổi lại, Stacking chậm hơn và khó giải thích hơn một mô hình đơn; trang 6 kiểm tra mức cải thiện có đáng không.")

with t2:
    describe = {
        "model": "Pipeline đầy đủ: tiền xử lý → Stacking → đổi về tỷ VNĐ. Gọi `predict()` trực tiếp trên dữ liệu thô",
        "model_name": "Tên mô hình được chọn", "features": "Danh sách thuộc tính đầu vào theo nhóm",
        "district_coords": "Tọa độ, khoảng cách tới trung tâm, số tiện ích xung quanh của 39 quận/huyện",
        "city_center": "Tọa độ mốc trung tâm hai thành phố", "target_transform": "Phép biến đổi biến mục tiêu",
        "cv_rmse": "RMSE cross-validation", "trained_on": "Số bản ghi huấn luyện",
        "interval": "Tham số khoảng dự báo 95% và độ phủ trên tập kiểm thử", "test_metrics": "Chỉ số trên tập kiểm thử",
    }
    st.dataframe(pd.DataFrame([{"Khóa": k, "Kiểu": type(v).__name__, "Nội dung": describe.get(k, "")}
                               for k, v in L.items()]), hide_index=True, width="stretch")
    st.markdown(f"Tệp `models/learnt.pkl` ({data.vn(data.MODEL_PATH.stat().st_size / 1024, 0)} KB) được lưu bằng `joblib` ở mục 6.7 của notebook.")
    st.json({"features": L["features"], "interval": L["interval"], "test_metrics": L["test_metrics"]}, expanded=False)

with t3:
    st.code('''import joblib
import numpy as np
import pandas as pd

learnt = joblib.load("models/learnt.pkl")
coords = learnt["district_coords"]

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
    loc = L["district_coords"].query("city == 'Hà Nội' and district == 'Cầu Giấy'").iloc[0]
    house = pd.DataFrame([{"area_m2": 50, "width_m": 4.5, "road_width_m": 3, "floors": 5, "has_parking": 0.0,
                           "district": "Cầu Giấy", "house_type": "Nhà trong hẻm",
                           **loc[["dist_center_km", "lat", *data.AMENITY_COLS]].to_dict()}])
    price = float(L["model"].predict(model.to_model_input(house))[0])
    lo, hi = np.expm1(np.log1p(price) + L["interval"]["q_lo"]), np.expm1(np.log1p(price) + L["interval"]["q_hi"])
    st.success(f"Kết quả chạy đoạn code trên: **{data.vn(price)} tỷ**, khoảng 95%: **{data.vn(lo)} – {data.vn(hi)} tỷ**")

with t4:
    idx = st.number_input("Số thứ tự căn nhà trong `features.csv`", 0, len(feat) - 1, 0)
    row = feat.iloc[[int(idx)]]
    X = model.to_model_input(row)
    Xt, base_log, final_log = model.decompose(X)
    st.markdown("**Bước 1. Thông số đầu vào**")
    st.dataframe(X, hide_index=True, width="stretch")
    st.markdown("**Bước 2. Sau tiền xử lý** (các cột khác 0)")
    vec = pd.Series(Xt[0], index=P["tcols"])
    st.dataframe(vec[vec != 0].round(3).to_frame("giá trị").T, width="stretch")
    st.markdown("**Bước 3. Năm mô hình con dự báo, mô hình meta kết hợp**")
    st.dataframe(model.stacking_table(base_log).round(4), hide_index=True, width="stretch")
    terms = " + ".join(f"({w_:.3f})({p:.3f})" for w_, p in zip(P["w"], base_log))
    st.latex(rf"\hat{{y}} = {P['b']:.3f} + {terms} = {final_log:.4f}")
    st.markdown("**Bước 4. Đổi về tỷ VNĐ**")
    st.latex(rf"\text{{giá}} = e^{{{final_log:.4f}}} - 1 = {np.expm1(final_log):.3f}\ \text{{tỷ}}")
    direct = float(L["model"].predict(X)[0])
    ui.box("evidence", f"Gọi trực tiếp `learnt['model'].predict()` cho **{data.vn(direct, 4)} tỷ**, trùng với kết quả tính tay từng tầng. Giá rao bán thực tế của căn này: **{data.vn(row['price_billion_vnd'].iloc[0])} tỷ**.")
