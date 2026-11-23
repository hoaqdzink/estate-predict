import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import data, model, ui

ui.header("KDD 5 · Đánh giá", "7. Đánh giá & độ tin cậy",
          "Mô hình được chọn trên tập kiểm thử độc lập: sai số, khoảng tin cậy của chỉ số, sai số theo nhóm")
ui.config_note()
A, L = data.artifacts(), data.learnt()
best = L["model_name"]
y, X = A["test"]["y"], A["test"]["X"]
p = A["test"]["preds"][best]
B = data.cfg("cfg_boot")

ui.box("idea", f"""
Một chỉ số duy nhất (ví dụ R²) chưa đủ để tin một mô hình. Độ tin cậy được đánh giá theo bốn câu hỏi:
- Sai số trên dữ liệu **chưa từng thấy** là bao nhiêu, và **chắc chắn đến mức nào** (bootstrap {B} lần)?
- Kết quả có **ổn định** giữa các lần chia dữ liệu (5 fold) không?
- Mô hình **sai có hệ thống** ở nhóm nào (phân khúc giá, thành phố, loại nhà)?
- Khi nào **không nên** tin dự báo?
""")

st.subheader("7.1 Chỉ số trên tập kiểm thử và khoảng tin cậy 95% (bootstrap)")
point = model.metrics(y, p)
bs = model.bootstrap_metrics(y, p, B)
c = st.columns(4)
for col, k in zip(c, ["RMSE", "MAE", "R²", "MAPE (%)"]):
    lo, hi = np.quantile(bs[k], [0.025, 0.975])
    unit = " tỷ" if k in ("RMSE", "MAE") else ("%" if k == "MAPE (%)" else "")
    col.metric(k, f"{data.vn(point[k], 3 if k == 'R²' else 2)}{unit}", f"95%: {data.vn(lo, 3 if k == 'R²' else 2)} – {data.vn(hi, 3 if k == 'R²' else 2)}", delta_color="off")
st.caption(f"Khoảng tin cậy được ước lượng bằng cách lấy mẫu lại có hoàn lại {len(y)} căn của tập kiểm thử {B} lần.")
folds = A["cv_folds"][best]
f = pd.DataFrame({"Fold": range(1, 6), "RMSE": folds["RMSE"], "MAE": folds["MAE"], "R²": folds["R2"]})
c1, c2 = st.columns([2, 3])
c1.markdown("**Kết quả từng fold (CV trên tập huấn luyện)**")
c1.dataframe(f.round(3), hide_index=True, width="stretch")
c1.caption(f"Hệ số biến thiên của RMSE giữa các fold: {data.vn(f['RMSE'].std() / f['RMSE'].mean() * 100, 1)}%")
c2.altair_chart(alt.Chart(bs, title="Phân phối R² qua bootstrap").mark_bar().encode(
    x=alt.X("R²:Q", bin=alt.Bin(maxbins=40)), y="count()"), use_container_width=True)

st.subheader("7.2 Giá thực tế và giá dự báo")
D = X[["city", "district", "house_type", "area_m2"]].copy()
D["Thực tế"], D["Dự báo"] = y, p
D["Phần dư"] = y - p
D["Sai số (%)"] = np.abs(D["Phần dư"]) / y * 100
lim = float(max(y.max(), p.max()) * 1.05)
diag = alt.Chart(pd.DataFrame({"x": [0, lim]})).mark_line(strokeDash=[4, 4], color="#DC2626").encode(x="x:Q", y="x:Q")
tip = ["district", "house_type", "area_m2", alt.Tooltip("Thực tế:Q", format=".2f"), alt.Tooltip("Dự báo:Q", format=".2f"),
       alt.Tooltip("Sai số (%):Q", format=".0f")]
c1, c2 = st.columns(2)
c1.altair_chart((diag + alt.Chart(D, title="Thực tế và dự báo").mark_circle(size=18, opacity=0.6).encode(
    x=alt.X("Thực tế:Q", title="Giá thực tế (tỷ)"), y=alt.Y("Dự báo:Q", title="Giá dự báo (tỷ)"),
    color=alt.Color("house_type:N", title="Loại nhà"), tooltip=tip)).interactive(), use_container_width=True)
c2.altair_chart(alt.Chart(D, title="Phần dư theo giá dự báo").mark_circle(size=18, opacity=0.6).encode(
    x=alt.X("Dự báo:Q", title="Giá dự báo (tỷ)"), y=alt.Y("Phần dư:Q", title="Thực tế − dự báo (tỷ)"),
    color=alt.Color("city:N", title="Thành phố"), tooltip=tip).interactive()
    + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#DC2626").encode(y="y:Q"), use_container_width=True)
st.caption("Phần dư tỏa rộng khi giá tăng: sai số tuyệt đối tỷ lệ với mức giá. Đây là lý do mô hình học trên thang log.")

st.subheader("7.3 Sai số theo nhóm")
D["Phân khúc"] = data.segment(y).astype(str).values
group = st.radio("Nhóm theo", ["Phân khúc", "city", "house_type", "district"], horizontal=True,
                 format_func=lambda x: {"Phân khúc": "Phân khúc giá", "city": "Thành phố", "house_type": "Loại nhà", "district": "Quận/huyện"}[x])
rows = []
for g, d in D.groupby(group):
    m = model.metrics(d["Thực tế"].to_numpy(), d["Dự báo"].to_numpy())
    rows.append({"Nhóm": g, "Số bản ghi": len(d), **m, "Bias (tỷ)": d["Phần dư"].mean()})
G = pd.DataFrame(rows).drop(columns="R²")
if group == "district":
    G = G[G["Số bản ghi"] >= 10]
G = G.sort_values("Nhóm" if group == "Phân khúc" else "Số bản ghi", ascending=group == "Phân khúc")
c1, c2 = st.columns([3, 2])
c1.dataframe(G.round(3), hide_index=True, width="stretch")
c2.altair_chart(alt.Chart(G, title="Bias = trung bình (thực tế − dự báo)").mark_bar().encode(
    y=alt.Y("Nhóm:N", sort=None, title=None), x="Bias (tỷ):Q",
    color=alt.condition("datum['Bias (tỷ)'] > 0", alt.value("#DC2626"), alt.value("#2563EB"))), use_container_width=True)
st.caption("Bias dương (đỏ): mô hình định giá thấp hơn thực tế. Ranh giới phân khúc thay đổi được trong Cấu hình phân tích.")

st.subheader("7.4 Khi nào nên và không nên tin dự báo")
seg_hi = data.segment_labels()[2]
hi_mask = D["Phân khúc"] == seg_hi
lo_iv, hi_iv = model.interval(p, 95)
ins = (y >= lo_iv) & (y <= hi_iv)
st.markdown(f"""
| Tình huống | Mức tin cậy | Căn cứ |
|---|---|---|
| Nhà trong hẻm, giá phổ biến, quận nhiều mẫu | **Cao** | MAPE khoảng {data.vn(model.metrics(y[~hi_mask.to_numpy()], p[~hi_mask.to_numpy()])['MAPE (%)'], 0)}% ngoài phân khúc cao; độ phủ khoảng 95% đạt {data.vn(ins[~hi_mask.to_numpy()].mean() * 100, 0)}% |
| Phân khúc {seg_hi} | **Thấp hơn** | Bias {data.vn(D.loc[hi_mask, 'Phần dư'].mean(), 1)} tỷ (định giá thấp); độ phủ khoảng 95% chỉ {data.vn(ins[hi_mask.to_numpy()].mean() * 100, 0)}% |
| Nhà mặt tiền | **Trung bình** | Sai số tương đối lớn hơn nhà trong hẻm (bảng 7.3, nhóm theo loại nhà) |
| Quận/huyện ít mẫu (dưới 10 bản ghi) | **Thấp** | Được gộp thành nhóm chung khi mã hóa; chỉ dựa vào tọa độ và tiện ích |
| Giá trị nhập ngoài miền dữ liệu huấn luyện | **Không đảm bảo** | Mô hình cây không ngoại suy; ứng dụng cảnh báo khi nhập (trang 9) |
""")
ui.box("critique", """
- Bootstrap chỉ đo độ bất định do **mẫu kiểm thử** có hạn (564 căn); không đo sai lệch do dữ liệu khác thị trường (giá rao bán thay vì giá giao dịch, dữ liệu trước 01/2022).
- Mô hình bị kéo về mức giá phổ biến (định giá cao nhà rẻ, định giá thấp nhà đắt): hệ quả của mất cân bằng dữ liệu và của việc tối ưu sai số bình phương trung bình.
- R² khoảng 0,76: phần biến thiên còn lại đến từ thông tin không có trong dữ liệu (pháp lý, vị trí cụ thể trong quận, hướng nhà, chất lượng xây dựng).
""")
ui.box("evidence", f"""
- RMSE kiểm thử {data.vn(point['RMSE'])} tỷ nằm trong dải RMSE của 5 fold ({data.vn(f['RMSE'].min())}–{data.vn(f['RMSE'].max())} tỷ): mô hình **không quá khớp** và tổng quát hóa ổn định.
- Khoảng tin cậy 95% của R² là [{data.vn(np.quantile(bs['R²'], 0.025), 3)}; {data.vn(np.quantile(bs['R²'], 0.975), 3)}]: kết quả không phụ thuộc vào một vài căn cụ thể trong tập kiểm thử.
""")
