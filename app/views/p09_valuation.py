import numpy as np
import pandas as pd
import streamlit as st

from core import data, model, ui

ui.header("Ứng dụng tri thức", "9. Định giá trực tuyến",
          "Nhập thông số căn nhà để nhận giá dự báo, khoảng dự báo và giải thích cách mô hình tính ra giá")
L = data.learnt()
coords, feat = L["district_coords"], data.features()
level = data.cfg("cfg_level")
NAMES = {"n_schools": "Trường học", "n_health": "Y tế", "n_markets": "Chợ, siêu thị", "n_parks": "Công viên",
         "n_transit": "Trạm giao thông"}

city = st.radio("Thành phố", ["Hà Nội", "Hồ Chí Minh"], horizontal=True)
with st.form("valuation"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Vị trí**")
        district = st.selectbox("Quận/huyện", sorted(coords.loc[coords["city"] == city, "district"]))
        house_type = st.radio("Loại nhà", ["Nhà trong hẻm", "Nhà mặt tiền"], horizontal=True)
        road = st.number_input("Độ rộng đường/hẻm trước nhà (m)", 0.5, 60.0, 4.0, 0.5)
    with c2:
        st.markdown("**Kích thước**")
        area = st.number_input("Diện tích (m²)", 10.0, 1000.0, 50.0, 1.0)
        width = st.number_input("Chiều ngang (m), để trống nếu không rõ", 1.0, 50.0, value=None, step=0.1)
        length = st.number_input("Chiều dài (m), để trống nếu không rõ", 1.0, 100.0, value=None, step=0.5)
        floors = st.number_input("Số tầng", 1, 15, 4)
    with c3:
        st.markdown("**Tiện ích trong nhà**")
        parking = st.checkbox("Chỗ đỗ ô tô")
        st.caption("Các tiện ích khác (bếp, sân thượng, phòng ăn) và số phòng ngủ bị loại ở bước trích chọn thuộc tính vì không cải thiện dự báo (trang 3).")
    ok = st.form_submit_button("Định giá", type="primary", width="stretch")

if district not in set(coords.loc[coords["city"] == city, "district"]):
    st.warning("Quận/huyện không thuộc thành phố đã chọn; hãy chọn lại quận/huyện.")
    st.stop()
row = coords[(coords["city"] == city) & (coords["district"] == district)].iloc[0]
if not ok:
    st.info("Nhập thông số rồi bấm **Định giá**.")
    st.stop()

if width is not None and length is not None:
    if width > length:
        st.warning("Chiều ngang lớn hơn chiều dài; hãy kiểm tra lại.")
    if not 0.5 <= width * length / area <= 1.5:
        st.warning(f"Ngang × dài = {data.vn(width * length, 0)} m² lệch nhiều so với diện tích {data.vn(area, 0)} m².")

X = model.to_model_input(pd.DataFrame([{
    "area_m2": area, "width_m": width, "length_m": length, "road_width_m": road, "floors": floors,
    "dist_center_km": row["dist_center_km"], "lat": row["lat"], "lon": row["lon"],
    **{c: row[c] for c in data.AMENITY_COLS}, "has_parking": float(parking),
    "city": city, "district": district, "house_type": house_type}]))
pred = float(L["model"].predict(X)[0])
lo, hi = model.interval(np.array([pred]), level)

st.divider()
m = st.columns(4)
m[0].metric("Giá dự báo", f"{data.vn(pred)} tỷ")
m[1].metric(f"Khoảng dự báo {level}%", f"{data.vn(lo[0])} – {data.vn(hi[0])} tỷ")
m[2].metric("Đơn giá ước tính", f"{data.vn(pred * 1000 / area, 0)} triệu/m²")
m[3].metric("Cách trung tâm", f"{data.vn(row['dist_center_km'], 1)} km")
st.caption(f"Khoảng dự báo {level}%: với các căn có đặc điểm tương tự, khoảng {level}% có giá nằm trong khoảng này. Đổi mức trong Cấu hình phân tích.")

rng = feat.describe()
out = [n for n, v, c in [("diện tích", area, "area_m2"), ("độ rộng đường", road, "road_width_m"), ("số tầng", floors, "floors")]
       if not rng.loc["min", c] <= v <= rng.loc["max", c]]
if out:
    st.warning("Giá trị " + ", ".join(out) + " nằm ngoài miền dữ liệu huấn luyện; kết quả kém tin cậy.")
if pred > data.cfg("cfg_segments")[1]:
    st.info("Mô hình có xu hướng định giá thấp phân khúc giá cao (ít mẫu); giá thực tế có thể cao hơn (trang 7).")
st.info("Giá dự báo áp dụng cho nhà có **giấy tờ pháp lý đầy đủ**. Dữ liệu huấn luyện không có thông tin pháp lý; nhà chưa có sổ, giấy tay hoặc đang tranh chấp thường có giá thấp hơn.")

t1, t2, t3 = st.tabs(["Mô hình tính ra giá như thế nào", "Vị trí & tiện ích xung quanh", "Các căn tương tự"])
with t1:
    Xt, base_log, final_log = model.decompose(X)
    st.markdown("**1. Năm mô hình con dự báo, mô hình meta kết hợp** (thang log1p)")
    st.dataframe(model.stacking_table(base_log).round(3), hide_index=True, width="stretch")
    P = model.parts()
    st.latex(rf"\hat y = {P['b']:.3f} + {float(base_log @ P['w']):.3f} = {final_log:.4f} \;\Rightarrow\; e^{{{final_log:.4f}}} - 1 = {np.expm1(final_log):.2f}\ \text{{tỷ}}")
    st.markdown("**2. Mỗi thuộc tính đẩy giá lên hoặc xuống bao nhiêu** (SHAP)")
    base, contrib = model.explain(Xt)
    shown = {"area_m2": f"{area:g} m²", "road_width_m": f"{road:g} m", "floors": floors, "house_type": house_type,
             "width_m": "không rõ" if width is None else f"{width:g} m", "district": district,
             "dist_center_km": f"{row['dist_center_km']:.1f} km", "has_parking": "có" if parking else "không",
             **{c: int(row[c]) for c in data.AMENITY_COLS}}
    fig, table = model.waterfall(base, contrib, shown)
    c1, c2 = st.columns([3, 2])
    c1.pyplot(fig)
    c2.dataframe(table.round(2), hide_index=True, width="stretch")
    top = table.iloc[0]
    ui.box("evidence", f"Bắt đầu từ **giá tham chiếu {data.vn(np.expm1(base))} tỷ** (mức giá của căn \"trung bình\"), mô hình cộng dần ảnh hưởng của từng thuộc tính. Tác động mạnh nhất: **{top['Thuộc tính']}** ({data.vn(top['Ảnh hưởng (%)'], 0)}% so với căn trung bình). Tổng các đóng góp bằng đúng giá dự báo.")
with t2:
    a = st.columns(5)
    for col, c in zip(a, data.AMENITY_COLS):
        col.metric(NAMES[c], int(row[c]))
    st.caption("Số tiện ích trong bán kính 2 km quanh tâm quận/huyện (OpenStreetMap, 19/11/2026).")
    st.map(pd.DataFrame({"latitude": [row["lat"]], "longitude": [row["lon"]]}), size=600, zoom=12)
with t3:
    comps = feat[(feat["district"] == district) & (feat["house_type"] == house_type)
                 & feat["area_m2"].between(area * 0.7, area * 1.3)].copy()
    if len(comps):
        comps["Giá/m² (triệu)"] = (comps["price_billion_vnd"] * 1000 / comps["area_m2"]).round(0)
        st.dataframe(comps[["area_m2", "road_width_m", "floors", "price_billion_vnd", "Giá/m² (triệu)"]]
                     .rename(columns=data.FEATURE_LABELS).sort_values("Diện tích").head(15), hide_index=True, width="stretch")
        st.caption(f"{len(comps)} căn cùng quận, cùng loại nhà, diện tích ±30%. Giá trung vị: {data.vn(comps['price_billion_vnd'].median())} tỷ.")
    else:
        st.caption("Không có căn tương tự (cùng quận, cùng loại nhà, diện tích ±30%) trong dữ liệu.")
