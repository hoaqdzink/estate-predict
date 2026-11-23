import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import data, model, ui

ui.header("KDD 5 · Độ bất định của tri thức", "5. Khoảng dự báo (bổ sung)",
          "Định lượng mức bất định của từng dự báo bằng phương pháp cross-conformal")
ui.config_note()
A, L = data.artifacts(), data.learnt()
level = data.cfg("cfg_level")
y, seg = A["test"]["y"], A["test"]["segment"]
pred = A["test"]["preds"][L["model_name"]]

ui.box("idea", """
Một con số giá dự báo chưa đủ: môi giới cần biết giá thật có thể nằm trong khoảng nào. Khoảng dự báo được xây dựng **không cần giả định phân phối** của sai số:
- Dự báo **ngoài fold** cho tập huấn luyện (mỗi căn được dự báo bởi mô hình không học căn đó).
- Tính phần dư trên thang log: r = ln(1 + giá thật) − ln(1 + giá dự báo).
- Lấy phân vị α/2 và 1 − α/2 của phần dư làm biên; khoảng của căn mới là giá dự báo nhân với e^(biên).
""")
q_lo, q_hi = model.interval_quantiles(level)
lo, hi = model.interval(pred, level)
inside = (y >= lo) & (y <= hi)
c = st.columns(4)
c[0].metric("Mức danh nghĩa", f"{level}%")
c[1].metric("Độ phủ thực tế (kiểm thử)", f"{data.vn(inside.mean() * 100, 1)}%", f"{data.vn((inside.mean() * 100 - level), 1)} điểm %")
c[2].metric("Biên dưới", f"{data.vn((np.exp(q_lo) - 1) * 100, 0)}%", help="So với giá dự báo")
c[3].metric("Biên trên", f"+{data.vn((np.exp(q_hi) - 1) * 100, 0)}%", help="So với giá dự báo")

st.subheader("Hiệu chỉnh: độ phủ thực tế so với mức danh nghĩa")
levels = np.arange(50, 100)
cal = pd.DataFrame({"Mức danh nghĩa (%)": levels,
                    "Độ phủ thực tế (%)": [(((y >= model.interval(pred, l_)[0]) & (y <= model.interval(pred, l_)[1])).mean() * 100) for l_ in levels]})
line = alt.Chart(pd.DataFrame({"x": [50, 99]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="x:Q")
st.altair_chart(line + alt.Chart(cal).mark_line(point=True).encode(
    x=alt.X("Mức danh nghĩa (%):Q", scale=alt.Scale(domain=[50, 100])), y=alt.Y("Độ phủ thực tế (%):Q", scale=alt.Scale(domain=[40, 100])),
    tooltip=["Mức danh nghĩa (%)", alt.Tooltip("Độ phủ thực tế (%):Q", format=".1f")]), use_container_width=True)
st.caption("Đường nét đứt: hiệu chỉnh hoàn hảo. Điểm nằm sát đường này nghĩa là khoảng dự báo đáng tin ở mọi mức.")

st.subheader("Độ phủ theo phân khúc giá")
abs_r = np.abs(A["oof"]["y"] - A["oof"]["pred"])
q_abs = np.quantile(abs_r, level / 100)
inside_abs = (y >= pred - q_abs) & (y <= pred + q_abs)
by = pd.DataFrame({"Phân khúc": seg, "log": inside, "tuyệt đối": inside_abs,
                   "rộng_log": (hi - lo) / pred, "rộng_abs": 2 * q_abs / pred})
tab = by.groupby("Phân khúc").agg(**{"Số bản ghi": ("log", "size"),
                                     "Độ phủ, thang log (%)": ("log", lambda s: s.mean() * 100),
                                     "Độ phủ, sai số tuyệt đối (%)": ("tuyệt đối", lambda s: s.mean() * 100),
                                     "Độ rộng TB, thang log (% giá)": ("rộng_log", lambda s: s.mean() * 100),
                                     "Độ rộng TB, tuyệt đối (% giá)": ("rộng_abs", lambda s: s.mean() * 100)})
st.dataframe(tab.reindex(["Dưới 5 tỷ", "5–15 tỷ", "Trên 15 tỷ"]).round(1), width="stretch")

st.subheader("Khoảng dự báo trên tập kiểm thử")
o = np.argsort(pred)
D = pd.DataFrame({"thứ tự": np.arange(len(o)), "dự báo": pred[o], "cận dưới": lo[o], "cận trên": hi[o], "thực tế": y[o],
                  "trong khoảng": np.where(inside[o], "Trong khoảng", "Ngoài khoảng")})
band = alt.Chart(D).mark_area(opacity=0.25, color="#2563EB").encode(x="thứ tự:Q", y="cận dưới:Q", y2="cận trên:Q")
st.altair_chart((band + alt.Chart(D).mark_line(color="#2563EB").encode(x="thứ tự:Q", y="dự báo:Q")
                 + alt.Chart(D).mark_circle(size=14).encode(
                     x=alt.X("thứ tự:Q", title="Căn nhà (sắp xếp theo giá dự báo)"), y=alt.Y("thực tế:Q", title="Giá (tỷ VNĐ)"),
                     color=alt.Color("trong khoảng:N", scale=alt.Scale(range=["#DC2626", "black"])),
                     tooltip=[alt.Tooltip("thực tế:Q", format=".2f"), alt.Tooltip("dự báo:Q", format=".2f"),
                              alt.Tooltip("cận dưới:Q", format=".2f"), alt.Tooltip("cận trên:Q", format=".2f")])
                 ).interactive(), use_container_width=True)

ui.box("critique", f"""
- **Khoảng tin cậy hay khoảng dự báo?** Đề tài dùng cụm từ "khoảng tin cậy 95%". Khoảng tin cậy chứa *giá trung bình* của các căn có cùng đặc điểm; khoảng dự báo chứa *giá của một căn cụ thể* và rộng hơn. Môi giới cần giá của căn đang định giá, nên ứng dụng dùng **khoảng dự báo**.
- **Thang log hay sai số tuyệt đối?** Khoảng tính trên sai số tuyệt đối có cùng độ rộng (± {data.vn(q_abs, 1)} tỷ) cho căn 2 tỷ và căn 30 tỷ: quá rộng với nhà rẻ, quá hẹp với nhà đắt. Tính trên thang log cho độ rộng **tỷ lệ với giá**, phù hợp thực tế định giá (bảng trên).
- **Phân khúc trên 15 tỷ có độ phủ thấp hơn** vì mô hình định giá thấp nhà đắt (trang 7). Hướng cải thiện: conformal theo nhóm, ước lượng phân vị riêng cho từng phân khúc.
- Khoảng khá rộng là phản ánh trung thực mức sai số của mô hình (MAPE ≈ {data.vn(L['test_metrics']['MAPE (%)'], 0)}%), không phải lỗi của phương pháp.
""")
ui.box("evidence", f"Ở mức {level}%, độ phủ thực tế trên tập kiểm thử độc lập là **{data.vn(inside.mean() * 100, 1)}%**; đường hiệu chỉnh bám sát đường lý tưởng ở mọi mức từ 50% đến 99%.")
