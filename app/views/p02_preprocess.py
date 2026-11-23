import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from core import data, ui

ui.header("KDD 2–3 · Tiền xử lý & biến đổi", "2. Tiền xử lý",
          "Làm sạch, biến đổi phân phối giá, xử lý vị trí và tiện ích xung quanh, loại ngoại lai")
ui.config_note()
A, L = data.artifacts(), data.learnt()
clean = data.filter_city(data.clean())
coords = L["district_coords"]

t1, t2, t3, t4 = st.tabs(["2.1 Làm sạch", "2.2 Phân phối giá & biến đổi", "2.3 Vị trí & tiện ích", "2.4 Ngoại lai"])

# ---------------------------------------------------------------- 2.1
with t1:
    ui.box("idea", "Đưa dữ liệu thô về dạng nhất quán, đúng kiểu, không trùng lặp, **mà không làm mất bản ghi có thể sửa được**. Mỗi loại lỗi có một quy tắc sửa riêng; bản ghi chỉ bị loại khi không thể khôi phục (trùng lặp, không có giá).")
    log = data.cleaning_log()
    st.altair_chart(alt.Chart(log, title="Số bản ghi sau từng bước").mark_bar().encode(
        y=alt.Y("Bước:N", sort=None, title=None), x="Số bản ghi:Q", tooltip=["Bước", "Số bản ghi"])
        + alt.Chart(log).mark_text(align="left", dx=3).encode(y=alt.Y("Bước:N", sort=None), x="Số bản ghi:Q",
                                                                text="Số bản ghi:Q"), use_container_width=True)
    st.markdown("""
| Nhóm | Bước | Quy tắc chính |
|---|---|---|
| Lỗi cấu trúc tệp | 3.1–3.3 | Bỏ tiêu đề lặp, dòng rỗng; dịch lại các ô của dòng lệch cột |
| Chuẩn hóa văn bản | 3.4–3.8 | Sửa lỗi font, chuẩn hóa Unicode NFKC, bỏ HTML/emoji; so khớp tên quận với danh mục chuẩn (ngưỡng tương đồng 0,8) |
| Chuyển đổi kiểu | 3.9–3.10 | Hàm trích số cho giá (`5ty2`, `3 tỷ 400 triệu`, `$`...), diện tích (`4x12`, `ha`), tầng (`1 trệt 3 lầu` = 4), ngày |
| Giá trị bất thường | 3.11–3.13 | Ngoài miền hợp lệ thành NaN; diện tích thừa chữ số 0 đối chiếu với tiêu đề; đảo ngang/dài |
| Trùng lặp, hoàn thiện | 3.14–3.19 | Trùng y hệt, trùng url, tin đăng lại (cùng đặc điểm, giá lệch ≤ 6%, cùng điện thoại hoặc tiêu đề) |
""")
    st.markdown("**So sánh trước và sau làm sạch**")
    raw = data.raw()
    col = st.selectbox("Thuộc tính", ["price_billion_vnd", "area_m2", "floors", "bedrooms", "width_m", "district",
                                      "city", "house_type", "has_parking"])
    rr = raw[raw["order"] != "order"].drop_duplicates("order")[["order", col]].rename(columns={col: "Trước"})
    cc = data.clean()[["order", col]].rename(columns={col: "Sau"})
    cc["order"] = cc["order"].astype(str)
    m = rr.merge(cc, on="order")
    changed = m[m["Trước"].astype(str) != m["Sau"].astype(str).str.replace(r"\.0$", "", regex=True)]
    changed = changed[changed["Trước"].str.replace(r"\.0$", "", regex=True) != changed["Sau"].astype(str).str.replace(r"\.0$", "", regex=True)]
    st.caption(f"{data.vn(len(changed), 0)} bản ghi có giá trị `{col}` được sửa. Hiển thị 25 trường hợp khác nhau.")
    st.dataframe(changed.drop_duplicates("Trước").head(25), hide_index=True, width="stretch")
    ui.box("critique", """
- **Không điền giá trị thiếu ở bước này.** Điền bằng trung vị toàn tập sẽ làm thông tin tập kiểm thử lọt vào huấn luyện; việc điền được đặt trong pipeline và chỉ học trên tập huấn luyện (trang 4).
- **Ngày `03/07/2021` không xác định được** là 3/7 hay 7/3; quy ước ưu tiên ngày/tháng, chỉ đọc tháng/ngày khi ngày/tháng không hợp lệ.
- **Quy ước tự đặt:** `n lầu` = n tầng, `1 trệt n lầu` = n + 1 tầng; `hẻm xe hơi` = 4 m, `hẻm ba gác` = 2,5 m; giá USD theo 25.000 VNĐ/USD.
""")

# ---------------------------------------------------------------- 2.2
with t2:
    ui.box("idea", "Giá nhà lệch phải: nhiều căn giá vừa phải, ít căn rất đắt. Mô hình tuyến tính giả định sai số phân phối chuẩn và phương sai đồng nhất, nên cần biến đổi giá về dạng cân đối hơn trước khi học.")
    y = clean["price_billion_vnd"].to_numpy()
    y_log = np.log1p(y)
    y_bc, lam = stats.boxcox(y)
    tbl = pd.DataFrame({"Phương pháp": ["Giá gốc", "Log1p", f"Box-Cox (λ = {lam:.3f})"],
                        "Skewness": [stats.skew(v) for v in (y, y_log, y_bc)],
                        "Kurtosis": [stats.kurtosis(v) for v in (y, y_log, y_bc)],
                        "r của Q-Q plot": [stats.probplot(v)[1][2] for v in (y, y_log, y_bc)]})
    st.dataframe(tbl.round(4), hide_index=True, width="stretch")
    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    for k_, (name, v) in enumerate([("Giá gốc", y), ("Log1p", y_log), ("Box-Cox", y_bc)]):
        ax[0, k_].hist(v, bins=50, color="#2563EB", edgecolor="white")
        ax[0, k_].set_title(f"{name}: skewness {stats.skew(v):.2f}".replace(".", ","))
        stats.probplot(v, plot=ax[1, k_])
        ax[1, k_].set_title(f"Q-Q plot: {name}")
    plt.tight_layout()
    st.pyplot(fig)
    tc = A["target_compare"][["Mô hình", "CV RMSE", "CV MAE", "CV R²", "Độ lệch chuẩn RMSE"]]
    ui.box("critique", f"""
Theo **hình dạng phân phối**, Box-Cox tốt nhất (skewness gần 0, r cao nhất). Nhưng mục tiêu là **dự báo giá trên đơn vị tỷ VNĐ**, nên lựa chọn được kiểm chứng lại bằng sai số cross-validation (bảng dưới).
Box-Cox với Ridge cho sai số lớn và dao động mạnh giữa các fold: λ âm (≈ {data.vn(A['box_cox']['lambda'], 2)}) nên phép biến đổi ngược không ổn định khi mô hình tuyến tính ngoại suy. Log1p không có giới hạn này.
""")
    st.dataframe(tc.round(4), hide_index=True, width="stretch")
    ui.box("evidence", "Kết luận: dùng **Log1p** cho biến mục tiêu của mọi mô hình. Tiêu chí cuối cùng là sai số dự báo trên đơn vị gốc, không phải độ đẹp của phân phối.")

# ---------------------------------------------------------------- 2.3
with t3:
    ui.box("idea", """
Dữ liệu chỉ có vị trí ở cấp quận/huyện (39 giá trị, nhiều quận rất ít mẫu). Biến vị trí rời rạc khó học; thay vào đó, mỗi quận được gán **tọa độ tâm**, **khoảng cách tới trung tâm** (Hồ Hoàn Kiếm, chợ Bến Thành) và **số tiện ích xung quanh** trong bán kính 2 km. Đây là các biến số liên tục, tổng quát hóa được cho cả quận ít mẫu.
**Nguồn:** ranh giới hành chính và tiện ích trên OpenStreetMap (bản trích xuất Việt Nam ngày 19/11/2026).
""")
    show = coords.copy()
    show = show if data.cfg("cfg_city") == "Tất cả" else show[show["city"] == data.cfg("cfg_city")]
    c1, c2 = st.columns([2, 3])
    c1.map(show.rename(columns={"lon": "longitude", "lat": "latitude"}), size=250)
    c2.dataframe(show.drop(columns=["lat", "lon"]).rename(columns=data.FEATURE_LABELS)
                 .sort_values(["Thành phố", "Khoảng cách tới trung tâm"]).round(2), hide_index=True, width="stretch", height=380)
    ppm = clean.assign(ppm2=clean["price_billion_vnd"] * 1000 / clean["area_m2"]).merge(
        coords[["city", "district", "dist_center_km", *data.AMENITY_COLS]], on=["city", "district"], how="left")
    g = ppm.groupby(["city", "district"]).agg(ppm2=("ppm2", "median"), n=("ppm2", "size"),
                                              dist=("dist_center_km", "first"), markets=("n_markets", "first")).reset_index()
    st.altair_chart(alt.Chart(g[g["n"] >= 5], title="Giá/m² trung vị theo khoảng cách tới trung tâm (mỗi điểm là một quận)").mark_circle().encode(
        x=alt.X("dist:Q", title="Khoảng cách tới trung tâm (km)"), y=alt.Y("ppm2:Q", title="Giá/m² trung vị (triệu)"),
        size=alt.Size("n:Q", title="Số bản ghi"), color=alt.Color("city:N", title="Thành phố"),
        tooltip=["district", "city", alt.Tooltip("ppm2:Q", format=".0f"), alt.Tooltip("dist:Q", format=".1f"), "n", "markets"]),
        use_container_width=True)
    rows = []
    for city, gg in ppm.groupby("city"):
        for c in ["dist_center_km", *data.AMENITY_COLS]:
            r, p = stats.spearmanr(gg[c], gg["ppm2"], nan_policy="omit")
            rows.append({"Thuộc tính": data.FEATURE_LABELS[c], "Thành phố": city, "ρ": r})
    st.markdown("**Tương quan Spearman với giá/m²**")
    st.dataframe(pd.DataFrame(rows).pivot(index="Thuộc tính", columns="Thành phố", values="ρ").round(3), width="stretch")
    ui.box("critique", """
- Vị trí và tiện ích chỉ ở **cấp quận**: mọi căn trong cùng quận có cùng giá trị; không phân biệt được hai căn ở hai đầu quận lớn.
- Số tiện ích tương quan chặt với khoảng cách tới trung tâm, nên mang nhiều thông tin trùng lặp (kiểm chứng ở trang 3).
- Dữ liệu OpenStreetMap ở ngoại thành chưa đầy đủ; Quận 2, Quận 9 đã sáp nhập vào TP Thủ Đức nên tọa độ là xấp xỉ.
""")

# ---------------------------------------------------------------- 2.4
with t4:
    k, p = data.cfg("cfg_iqr_k"), data.cfg("cfg_maha_p")
    ui.box("idea", f"""
Ngoại lai ở đây là các bản ghi có tổ hợp giá, diện tích, số tầng, số phòng bất thường, nhiều khả năng do nhập sai. Hai phương pháp được so sánh với cấu hình hiện tại:
- **IQR** (đơn biến) trên log(giá/m²), riêng từng thành phố, hệ số k = {data.vn(k, 1)};
- **Mahalanobis** (đa biến) trên log(diện tích), log(giá/m²), số tầng, số phòng ngủ, ngưỡng χ²({p}; 4).
""")
    w = data.clean().copy()
    w["log_ppm2"] = np.log(w["price_billion_vnd"] * 1000 / w["area_m2"])
    w["log_area"] = np.log(w["area_m2"])
    iqr_f = pd.Series(False, index=w.index)
    for city, gg in w.groupby("city"):
        v = gg["log_ppm2"].dropna()
        q1, q3 = v.quantile([0.25, 0.75])
        iqr_f[v.index] = (v < q1 - k * (q3 - q1)) | (v > q3 + k * (q3 - q1))
    X = w[["log_area", "log_ppm2", "floors", "bedrooms"]].astype(float).dropna()
    diff = X.to_numpy() - X.to_numpy().mean(0)
    d2 = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(np.cov(X.to_numpy(), rowvar=False)), diff)
    thr = stats.chi2.ppf(p, 4)
    maha_f = pd.Series(False, index=w.index)
    maha_f[X.index] = d2 > thr
    w = data.filter_city(w.assign(IQR=iqr_f, Mahalanobis=maha_f))
    c = st.columns(4)
    c[0].metric("Ngoại lai IQR", data.vn(int(w["IQR"].sum()), 0), f"{data.vn(w['IQR'].mean() * 100, 1)}%", delta_color="off")
    c[1].metric("Ngoại lai Mahalanobis", data.vn(int(w["Mahalanobis"].sum()), 0), f"ngưỡng D² > {data.vn(thr, 2)}", delta_color="off")
    c[2].metric("Cả hai cùng đánh dấu", data.vn(int((w["IQR"] & w["Mahalanobis"]).sum()), 0))
    c[3].metric("Không đánh giá được (Mahalanobis)", data.vn(int(w[["floors", "bedrooms"]].isna().any(axis=1).sum()), 0))
    w["Phân loại"] = np.select([w["IQR"] & w["Mahalanobis"], w["IQR"], w["Mahalanobis"]],
                               ["Cả hai", "Chỉ IQR", "Chỉ Mahalanobis"], "Bình thường")
    w["Giá/m² (triệu)"] = np.exp(w["log_ppm2"])
    st.altair_chart(alt.Chart(w.dropna(subset=["log_area"]), title="Ngoại lai theo hai phương pháp").mark_circle(size=18).encode(
        x=alt.X("area_m2:Q", scale=alt.Scale(type="log"), title="Diện tích (m², thang log)"),
        y=alt.Y("Giá/m² (triệu):Q", scale=alt.Scale(type="log")),
        color=alt.Color("Phân loại:N", scale=alt.Scale(domain=["Bình thường", "Chỉ Mahalanobis", "Chỉ IQR", "Cả hai"],
                                                        range=["#BBBBBB", "#EA7A2B", "#DC2626", "#8C1C13"])),
        tooltip=["district", "area_m2", "floors", "bedrooms", "price_billion_vnd", alt.Tooltip("Giá/m² (triệu):Q", format=".0f")]
    ).interactive(), use_container_width=True)
    kind = st.radio("Xem bản ghi", ["Chỉ IQR", "Chỉ Mahalanobis", "Cả hai"], horizontal=True)
    st.dataframe(w.loc[w["Phân loại"] == kind, ["district", "house_type", "area_m2", "floors", "bedrooms",
                                                "price_billion_vnd", "Giá/m² (triệu)"]].round(1).head(20),
                 hide_index=True, width="stretch")
    ui.box("critique", """
- **Mahalanobis giả định phân phối chuẩn nhiều chiều**, không phù hợp với số tầng, số phòng ngủ (biến rời rạc, lệch). Nhiều bản ghi "chỉ Mahalanobis" là nhà nhiều phòng hoặc 1 tầng giá cao: hiếm nhưng hợp lý. Phương pháp cũng bỏ qua khoảng 13% bản ghi thiếu số tầng/phòng.
- **IQR trên log(giá/m²)** bắt đúng các lỗi nhập liệu điển hình: diện tích thừa chữ số 0 (giá/m² quá thấp) và sai giá (giá/m² quá cao). Dùng log vì giá/m² cũng lệch phải; nếu không, IQR đánh dấu nhầm nhiều căn cao cấp.
- **Loại trước khi chia tập** không gây rò rỉ: đây là bước loại bản ghi sai, không học tham số dùng cho dự báo.
""")
    ui.box("evidence", "Mô hình chính dùng **IQR với k = 1,5**, loại **126 bản ghi (4,3%)**, còn **2.819 bản ghi**. Mahalanobis được dùng để kiểm tra chéo: 51 bản ghi bị cả hai phương pháp đánh dấu là ngoại lai rõ ràng nhất.")
