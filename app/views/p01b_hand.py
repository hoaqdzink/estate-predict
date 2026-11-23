import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

from core import data, ui

ui.header("KDD 4 · Minh họa thuật toán khai phá", "1b. Ví dụ giải tay trên 12 căn nhà",
          "Tính tay từng bước của hồi quy tuyến tính, Ridge, IQR và khoảng cách Haversine, đối chiếu với thư viện")
ui.config_note()

ui.box("idea", """
Trước khi dùng mô hình phức tạp, cần hiểu mô hình học từ dữ liệu như thế nào. Trang này lấy **12 căn nhà thật** trong dữ liệu và tính tay toàn bộ quá trình: lập ma trận, giải hệ phương trình chuẩn để tìm hệ số, dự báo một căn mới, đo sai số. Kết quả tính tay được đối chiếu với `scikit-learn` để chứng minh hai cách cho cùng một đáp án.
Mô hình minh họa: **ln(1 + giá) = β₀ + β₁·diện tích + β₂·độ rộng đường + β₃·mặt tiền**.
""")

# ---------------- Chọn mẫu ----------------
f = data.features().dropna(subset=["area_m2", "road_width_m", "house_type"])
f = f[f["area_m2"] <= 150]
seed = data.cfg("cfg_seed")
s = pd.concat([f[f["house_type"] == "Nhà trong hẻm"].sample(8, random_state=seed),
               f[f["house_type"] == "Nhà mặt tiền"].sample(4, random_state=seed)]).sample(frac=1, random_state=seed)
s = s.reset_index(drop=True)
D = pd.DataFrame({"Căn": range(1, 13), "Quận": s["district"], "Diện tích (m²)": s["area_m2"],
                  "Đường (m)": s["road_width_m"], "Mặt tiền": (s["house_type"] == "Nhà mặt tiền").astype(int),
                  "Giá (tỷ)": s["price_billion_vnd"]})
D["y' = ln(1+giá)"] = np.log1p(D["Giá (tỷ)"])

st.subheader("Bước 1. Dữ liệu mẫu")
st.markdown("8 căn trong hẻm và 4 căn mặt tiền, chọn ngẫu nhiên theo hạt giống trong Cấu hình phân tích. "
            "Biến mục tiêu được biến đổi `y' = ln(1 + giá)` như mô hình chính.")
st.dataframe(D.round(4), hide_index=True, width="stretch")

X = np.column_stack([np.ones(12), D["Diện tích (m²)"], D["Đường (m)"], D["Mặt tiền"]])
y = D["y' = ln(1+giá)"].to_numpy()
names = ["β₀ (hệ số chặn)", "β₁ (diện tích)", "β₂ (độ rộng đường)", "β₃ (mặt tiền)"]

st.subheader("Bước 2. Lập ma trận và giải phương trình chuẩn")
st.latex(r"\hat{\beta} = (X^\top X)^{-1} X^\top y'")
st.markdown("Ma trận **X** có 12 dòng (12 căn) và 4 cột: cột 1 toàn số 1 (cho hệ số chặn), diện tích, độ rộng đường, mặt tiền.")
XtX, Xty = X.T @ X, X.T @ y
inv = np.linalg.inv(XtX)
beta = inv @ Xty
lab = ["1", "Diện tích", "Đường", "Mặt tiền"]
c1, c2 = st.columns(2)
c1.markdown("**XᵀX** (4×4)")
c1.dataframe(pd.DataFrame(XtX, index=lab, columns=lab).round(2), width="stretch")
c2.markdown("**(XᵀX)⁻¹** (4×4)")
c2.dataframe(pd.DataFrame(inv, index=lab, columns=lab).round(6), width="stretch")
c1, c2 = st.columns(2)
c1.markdown("**Xᵀy'** (4×1)")
c1.dataframe(pd.DataFrame({"Xᵀy'": Xty}, index=lab).round(4), width="stretch")
c2.markdown("**β̂ = (XᵀX)⁻¹ Xᵀy'**")
c2.dataframe(pd.DataFrame({"Hệ số": beta}, index=names).round(5), width="stretch")
st.latex(rf"\ln(1+\text{{giá}}) = {beta[0]:.4f} {beta[1]:+.4f}\cdot\text{{dt}} {beta[2]:+.4f}\cdot\text{{đường}} {beta[3]:+.4f}\cdot\text{{mặt tiền}}")
st.markdown(f"""
**Đọc hệ số** (vì biến mục tiêu ở thang log, hệ số đọc theo phần trăm):
- Thêm 1 m² diện tích: giá thay đổi khoảng **{data.vn((np.exp(beta[1]) - 1) * 100, 2)}%**.
- Đường rộng thêm 1 m: giá thay đổi khoảng **{data.vn((np.exp(beta[2]) - 1) * 100, 1)}%**.
- Nhà mặt tiền so với nhà trong hẻm cùng diện tích và độ rộng đường: giá chênh khoảng **{data.vn((np.exp(beta[3]) - 1) * 100, 0)}%**.
""")

st.subheader("Bước 3. Dự báo một căn mới")
c = st.columns(3)
a_new = c[0].number_input("Diện tích (m²)", 15.0, 200.0, 50.0, 5.0)
r_new = c[1].number_input("Độ rộng đường (m)", 1.0, 30.0, 4.0, 0.5)
m_new = c[2].selectbox("Loại nhà", ["Nhà trong hẻm", "Nhà mặt tiền"]) == "Nhà mặt tiền"
yl = beta[0] + beta[1] * a_new + beta[2] * r_new + beta[3] * m_new
st.latex(rf"y' = {beta[0]:.4f} + ({beta[1]:.4f})({a_new:g}) + ({beta[2]:.4f})({r_new:g}) + ({beta[3]:.4f})({int(m_new)}) = {yl:.4f}")
st.latex(rf"\text{{giá}} = e^{{{yl:.4f}}} - 1 = {np.expm1(yl):.3f}\ \text{{tỷ}}")

st.subheader("Bước 4. Đo sai số trên 12 căn")
yhat = X @ beta
E = pd.DataFrame({"Căn": D["Căn"], "Giá thật (tỷ)": D["Giá (tỷ)"], "Giá dự báo (tỷ)": np.expm1(yhat),
                  "Sai số (tỷ)": D["Giá (tỷ)"] - np.expm1(yhat)})
E["|Sai số| / giá thật (%)"] = (E["Sai số (tỷ)"].abs() / E["Giá thật (tỷ)"] * 100)
st.dataframe(E.round(3), hide_index=True, width="stretch")
sse, sst = float(((y - yhat) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
e = E["Sai số (tỷ)"].to_numpy()
c = st.columns(4)
c[0].metric("SSE (thang log)", data.vn(sse, 4))
c[1].metric("SST (thang log)", data.vn(sst, 4))
c[2].metric("R² = 1 − SSE/SST", data.vn(1 - sse / sst, 3))
c[3].metric("RMSE (tỷ)", data.vn(float(np.sqrt((e ** 2).mean())), 3))
st.latex(r"R^2 = 1 - \frac{\sum (y'_i - \hat{y}'_i)^2}{\sum (y'_i - \bar{y}')^2}, \qquad RMSE = \sqrt{\frac{1}{n}\sum(\text{giá}_i - \widehat{\text{giá}}_i)^2}")

st.subheader("Bước 5. Đối chiếu với scikit-learn")
sk = LinearRegression().fit(X[:, 1:], y)
cmp = pd.DataFrame({"Tính tay": beta, "scikit-learn": np.r_[sk.intercept_, sk.coef_]}, index=names)
cmp["Chênh lệch"] = (cmp["Tính tay"] - cmp["scikit-learn"]).abs()
st.dataframe(cmp.style.format({"Tính tay": "{:.6f}", "scikit-learn": "{:.6f}", "Chênh lệch": "{:.1e}"}), width="stretch")
ui.box("evidence", f"Hệ số tính tay và hệ số của `LinearRegression` chênh nhau tối đa **{cmp['Chênh lệch'].max():.1e}**, tức là sai số làm tròn của máy tính: hai cách cho cùng một mô hình.")

st.subheader("Bước 6. Ridge: chính quy hóa làm co hệ số")
st.markdown("Ridge thêm mức phạt λ vào phương trình chuẩn. Các biến được chuẩn hóa (trừ trung bình, chia độ lệch chuẩn) để mức phạt công bằng giữa các biến; hệ số chặn không bị phạt.")
st.latex(r"\hat{\beta}_{ridge} = (Z^\top Z + \lambda I)^{-1} Z^\top (y' - \bar{y}')")
Z = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
yc = y - y.mean()
lam = st.slider("λ (mức phạt)", 0.0, 50.0, 5.0, 0.5)
ridge = lambda l: np.linalg.solve(Z.T @ Z + l * np.eye(3), Z.T @ yc)
st.dataframe(pd.DataFrame({"λ = 0 (OLS)": ridge(0), f"λ = {lam:g}": ridge(lam)},
                          index=["Diện tích (chuẩn hóa)", "Đường (chuẩn hóa)", "Mặt tiền (chuẩn hóa)"]).round(4),
             width="stretch")
path = pd.DataFrame([{"λ": l, "Thuộc tính": n, "Hệ số": b}
                     for l in np.linspace(0, 50, 101) for n, b in zip(["Diện tích", "Đường", "Mặt tiền"], ridge(l))])
st.altair_chart(alt.Chart(path, title="Đường đi của hệ số Ridge theo λ").mark_line().encode(
    x="λ:Q", y="Hệ số:Q", color="Thuộc tính:N", tooltip=["λ", "Thuộc tính", alt.Tooltip("Hệ số:Q", format=".4f")])
    + alt.Chart(pd.DataFrame({"λ": [lam]})).mark_rule(strokeDash=[4, 4]).encode(x="λ:Q"), use_container_width=True)
ui.box("critique", """
λ càng lớn, các hệ số càng bị kéo về 0: mô hình bớt phụ thuộc vào vài mẫu cụ thể (giảm phương sai) nhưng có thể bỏ sót quan hệ thật (tăng độ chệch). Không có λ "đúng" cho mọi dữ liệu; mô hình chính chọn λ bằng **grid search cross-validation** (trang 3).
""")

st.subheader("Bước 7. Phát hiện ngoại lai bằng IQR (tính tay)")
k = data.cfg("cfg_iqr_k")
v = np.sort((D["Giá (tỷ)"] * 1000 / D["Diện tích (m²)"]).to_numpy())
q1, q3 = np.quantile(v, [0.25, 0.75])
iqr = q3 - q1
st.markdown(f"Giá/m² (triệu) của 12 căn, đã sắp xếp: `{', '.join(data.vn(x, 1) for x in v)}`")
st.markdown(f"""
- Q1 ở vị trí (n − 1)·0,25 = 2,75 (nội suy giữa giá trị thứ 3 và 4): **Q1 = {data.vn(q1, 2)}**
- Q3 ở vị trí (n − 1)·0,75 = 8,25 (nội suy giữa giá trị thứ 9 và 10): **Q3 = {data.vn(q3, 2)}**
- IQR = Q3 − Q1 = **{data.vn(iqr, 2)}**; với k = {data.vn(k, 1)}: khoảng hợp lệ **[{data.vn(q1 - k * iqr, 1)}; {data.vn(q3 + k * iqr, 1)}]** triệu/m²
- Số căn ngoài khoảng: **{int(((v < q1 - k * iqr) | (v > q3 + k * iqr)).sum())}**
""")

st.subheader("Bước 8. Khoảng cách tới trung tâm (Haversine, tính tay)")
L = data.learnt()
dc = L["district_coords"]
row0 = dc[(dc["district"] == s.loc[0, "district"]) & (dc["city"] == s.loc[0, "city"])].iloc[0]
lat0, lon0 = L["city_center"][row0["city"]]
p1, p2 = np.radians(row0["lat"]), np.radians(lat0)
dphi, dlmb = p2 - p1, np.radians(lon0 - row0["lon"])
a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
d = 2 * 6371 * np.arcsin(np.sqrt(a))
st.markdown(f"Căn 1 ở **{row0['district']}** (tâm quận {row0['lat']:.5f}, {row0['lon']:.5f}); mốc trung tâm {row0['city']}: ({lat0}, {lon0}).")
st.latex(rf"a = \sin^2\frac{{\Delta\varphi}}{{2}} + \cos\varphi_1\cos\varphi_2\sin^2\frac{{\Delta\lambda}}{{2}} = {a:.3e}")
st.latex(rf"d = 2 \cdot 6371 \cdot \arcsin\sqrt{{a}} = {d:.3f}\ \text{{km}}")
st.caption(f"Giá trị lưu trong mô hình: {row0['dist_center_km']:.3f} km.")

ui.box("critique", """
**Giới hạn của ví dụ:** 12 căn và 3 thuộc tính chỉ dùng để minh họa cơ chế. Hệ số thay đổi nhiều khi đổi mẫu (thử đổi hạt giống): đây chính là phương sai cao khi dữ liệu ít, lý do mô hình chính được huấn luyện trên 2.255 bản ghi với 14 thuộc tính và kiểm chứng bằng cross-validation.
""")
