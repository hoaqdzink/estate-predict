import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import data, ui

ui.header("KDD 1 · Lựa chọn dữ liệu", "1. Dữ liệu & mất cân bằng",
          "Dữ liệu thô mô phỏng thu thập từ trang rao vặt: nhiều lỗi, chưa tiền xử lý, phân bố không đồng đều")
ui.config_note()

raw = data.raw()
feat = data.filter_city(data.features())
NUM = ["area_m2", "width_m", "length_m", "road_width_m", "floors", "bedrooms", "price_billion_vnd"]

ui.box("idea", """
Dữ liệu bất động sản thu thập tự động hiếm khi sạch: giá ghi bằng chữ (`5ty2`, `Thỏa thuận`), diện tích có đơn vị lẫn lộn, tên quận viết nhiều kiểu, tin đăng lại nhiều lần. Bộ dữ liệu được xây dựng để phản ánh đúng các đặc điểm này, nhằm chứng minh quy trình xử lý hoạt động được trên dữ liệu thực tế.
**Cách tạo:** lấy 3.500 tin rao thật (Hà Nội, TP.HCM) từ tập dữ liệu công khai thu thập trên alonhadat.com.vn, sau đó bổ sung có chủ đích **36 loại lỗi** (khoảng 13.500 ô lỗi) và 6 cột mô phỏng dữ liệu crawl (`url`, `title`, `address`, `posted_date`, `contact`, `scraped_at`).
""")

c = st.columns(5)
c[0].metric("Số dòng thô", data.vn(len(raw), 0))
c[1].metric("Số cột", len(raw.columns))
c[2].metric("Cách viết thành phố", raw["city"].nunique())
c[3].metric("Cách viết quận/huyện", raw["district"].nunique())
bad = sum((~raw[col].str.fullmatch(r"-?\d+(\.\d+)?")).sum() for col in NUM)
c[4].metric("Ô số không hợp lệ", data.vn(bad, 0), help="Ô ở 7 cột số không phải là một số thuần")

t1, t2, t3, t4 = st.tabs(["Dữ liệu thô", "Tổng quan lỗi", "Mô tả thuộc tính", "Mất cân bằng"])
with t1:
    cols = st.multiselect("Cột hiển thị", list(raw.columns),
                          ["title", "city", "district", "house_type", "area_m2", "floors", "price_billion_vnd"])
    st.dataframe(raw[cols].sample(20, random_state=data.cfg("cfg_seed")), hide_index=True, width="stretch")
    st.caption("20 dòng ngẫu nhiên; đổi hạt giống trong Cấu hình phân tích để xem mẫu khác.")

with t2:
    rows = []
    for col in raw.columns:
        s = raw[col]
        invalid = s[~s.str.fullmatch(r"-?\d+(\.\d+)?")] if col in NUM else s.iloc[0:0]
        rows.append({"Cột": col, "Số cách viết": s.nunique(), "Ô trống": int((s.str.strip() == "").sum()),
                     "Ô không phải số": len(invalid) if col in NUM else None,
                     "Ví dụ giá trị lỗi": ", ".join(invalid.drop_duplicates().head(5)) if col in NUM else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("""
| Nhóm lỗi | Ví dụ |
|---|---|
| Lỗi cấu trúc tệp | dòng tiêu đề lặp, dòng rỗng, dòng thiếu ô làm lệch cột |
| Số ghi dạng văn bản | `5 tỷ 200 triệu`, `8ty7`, `5.200.000.000`, `$208,000`, `4x12`, `0.0045 ha`, `1 trệt 3 lầu` |
| Sai đơn vị, lỗi Excel | giá ghi theo triệu, chiều ngang theo cm, `5-Thg4` thay cho 4,5 |
| Văn bản không thống nhất | `HN`, `Sài Gòn`, `Q.1`, Unicode tổ hợp, mất dấu, lỗi font `Cáº§u Giáº¥y`, HTML, emoji |
| Giá trị vô lý | 0 tầng, 99 phòng ngủ, diện tích âm, giá 999 |
| Trùng lặp | dòng trùng y hệt, tin đăng lại với giá lệch 2–5% |
""")

with t3:
    st.markdown("""
| Thuộc tính | Kiểu | Ý nghĩa |
|---|---|---|
| `city`, `district` | Phân loại | Thành phố, quận/huyện |
| `house_type` | Phân loại | `Nhà mặt tiền` / `Nhà trong hẻm` |
| `area_m2`, `width_m`, `length_m` | Số | Diện tích (m²), chiều ngang, chiều dài (m) |
| `road_width_m` | Số | Độ rộng đường/hẻm trước nhà (m) |
| `floors`, `bedrooms` | Số nguyên | Số tầng, số phòng ngủ |
| `has_dining_room`, `has_kitchen`, `has_terrace`, `has_parking` | Nhị phân | Tiện ích trong nhà |
| `price_billion_vnd` | Số | **Giá rao bán (tỷ VNĐ), biến mục tiêu** |
| `url`, `title`, `address`, `posted_date`, `contact`, `scraped_at` | Văn bản | Thông tin tin rao, dùng để làm sạch và phát hiện trùng lặp |
""")

with t4:
    st.markdown(f"Phạm vi: **{data.cfg('cfg_city')}**, {data.vn(len(feat), 0)} bản ghi sau tiền xử lý.")
    seg = data.segment(feat["price_billion_vnd"].to_numpy())
    c1, c2, c3 = st.columns(3)
    for col, (title, s) in zip([c1, c2, c3], [("Thành phố", feat["city"]), ("Loại nhà", feat["house_type"]),
                                               ("Phân khúc giá", seg)]):
        vc = s.value_counts().rename_axis("nhóm").reset_index(name="số bản ghi")
        vc["tỷ lệ"] = vc["số bản ghi"] / vc["số bản ghi"].sum()
        ch = alt.Chart(vc, title=title).mark_bar().encode(
            x=alt.X("nhóm:N", sort=None, title=None), y=alt.Y("số bản ghi:Q"),
            tooltip=["nhóm", "số bản ghi", alt.Tooltip("tỷ lệ:Q", format=".1%")])
        col.altair_chart(ch + ch.mark_text(dy=-8).encode(text=alt.Text("tỷ lệ:Q", format=".0%")), use_container_width=True)
        col.metric("Tỷ lệ nhóm lớn / nhóm nhỏ", data.vn(vc["số bản ghi"].max() / max(vc["số bản ghi"].min(), 1), 1) + " lần")

    vd = feat["district"].value_counts().rename_axis("quận").reset_index(name="số bản ghi")
    vd["nhóm"] = np.where(vd["số bản ghi"] < 20, "dưới 20 bản ghi", "từ 20 bản ghi")
    st.altair_chart(alt.Chart(vd, title="Số bản ghi theo quận/huyện").mark_bar().encode(
        x=alt.X("quận:N", sort="-y", title=None), y="số bản ghi:Q",
        color=alt.Color("nhóm:N", scale=alt.Scale(range=["#DC2626", "#2563EB"])),
        tooltip=["quận", "số bản ghi"]), use_container_width=True)

    ct = pd.crosstab(seg.values, feat["house_type"].values, normalize="columns") * 100
    st.markdown("**Tỷ lệ phân khúc giá trong từng loại nhà (%)**")
    st.dataframe(ct.round(1), width="stretch")

    hi = data.segment_labels()[2]
    share = (seg == hi).mean()
    few = (vd["số bản ghi"] < 20).sum()
    ui.box("evidence", f"""
- Phân khúc **{hi}** chỉ chiếm **{data.vn(share * 100, 1)}%** bản ghi; phân phối giá lệch phải (skewness {data.vn(feat['price_billion_vnd'].skew())}).
- **{few}** quận/huyện có dưới 20 bản ghi, chủ yếu là ngoại thành.
- Nhà mặt tiền tập trung ở phân khúc giá cao hơn nhà trong hẻm: loại nhà và phân khúc giá có liên hệ chặt.
""")
    ui.box("critique", """
**Mất cân bằng ảnh hưởng thế nào tới hồi quy?** Mô hình học chủ yếu từ nhà giá phổ biến nên có xu hướng **định giá thấp** nhà đắt, và sai số trung bình trên toàn tập che giấu sai số lớn ở nhóm ít mẫu.
**Vì sao không cân bằng lại bằng lấy mẫu (oversampling)?** Với hồi quy, nhân bản các căn đắt làm mô hình quá khớp vào vài mẫu và làm lệch phân phối giá thực tế. Đề tài chọn cách xử lý không làm thay đổi dữ liệu:
- biến đổi Log1p cho giá (trang 2);
- chia tập huấn luyện/kiểm thử **phân tầng theo phân khúc giá** (trang 3);
- gộp quận/huyện có dưới 10 bản ghi thành nhóm chung khi mã hóa (trang 4);
- **đánh giá sai số riêng từng phân khúc** (trang 7).
**Dữ liệu thật thêm lỗi thay vì giả lập hoàn toàn:** phân phối giá, quan hệ giữa diện tích, vị trí và giá là của thị trường thật, nên kết luận của mô hình có ý nghĩa; chỉ phần lỗi là nhân tạo.
""")
