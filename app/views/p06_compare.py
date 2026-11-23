import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import data, model, ui

ui.header("KDD 5 · Đánh giá", "6. So sánh mô hình",
          "Bảy mô hình trên cùng tập thuộc tính: cross-validation trên tập huấn luyện và tập kiểm thử độc lập")
ui.config_note()
A, L = data.artifacts(), data.learnt()
y, preds = A["test"]["y"], A["test"]["preds"]
best = L["model_name"]
B = data.cfg("cfg_boot")

ui.box("idea", "So sánh công bằng: cùng tập thuộc tính, cùng biến đổi Log1p, cùng cách chia fold, cùng tập kiểm thử. Mô hình cơ sở là **Linear Regression** (không có tham số); các mô hình còn lại phải chứng minh tốt hơn nó.")

tt = A["test_table"].copy()
tt["Nhóm"] = np.where(tt["Mô hình"].isin(["Gradient Boosting", "LightGBM", "Stacking"]), "Mô hình cây / kết hợp", "Tuyến tính")
show = tt[["Mô hình", "Nhóm", "RMSE", "MAE", "R²", "MAPE (%)", "CV RMSE", "Chênh lệch RMSE so với CV (%)"]]
st.dataframe(show.style.format(precision=3, decimal=",", thousands=".")
             .highlight_min(subset=["RMSE", "MAE", "MAPE (%)", "CV RMSE"], color="#d4edda")
             .highlight_max(subset=["R²"], color="#d4edda"), hide_index=True, width="stretch")
base = tt.set_index("Mô hình").loc["Linear Regression"]
bm = tt.set_index("Mô hình").loc[best]
st.caption(f"So với Linear Regression, {best} giảm RMSE {data.vn((1 - bm['RMSE'] / base['RMSE']) * 100, 1)}%, "
           f"MAE {data.vn((1 - bm['MAE'] / base['MAE']) * 100, 1)}%, MAPE {data.vn(base['MAPE (%)'] - bm['MAPE (%)'], 1)} điểm %.")

metric = st.radio("Thang đo", ["RMSE", "MAE", "R²", "MAPE (%)"], horizontal=True)
st.altair_chart(alt.Chart(tt, title=f"{metric} trên tập kiểm thử").mark_bar().encode(
    x=alt.X(f"{metric}:Q"), y=alt.Y("Mô hình:N", sort="x" if metric != "R²" else "-x", title=None),
    color=alt.condition(alt.datum["Mô hình"] == best, alt.value("#EA7A2B"), alt.value("#93B4F7")),
    tooltip=["Mô hình", alt.Tooltip(f"{metric}:Q", format=".4f")]), use_container_width=True)

st.subheader("Độ ổn định giữa các fold")
names = list(preds)
folds = pd.DataFrame([{"Mô hình": n, "Fold": i + 1, "RMSE": v}
                      for n in names for i, v in enumerate(A["cv_folds"][n]["RMSE"])])
st.altair_chart(alt.Chart(folds, title="RMSE của 5 fold cross-validation").mark_boxplot(extent="min-max").encode(
    x=alt.X("RMSE:Q", scale=alt.Scale(zero=False)), y=alt.Y("Mô hình:N", title=None))
    + alt.Chart(folds).mark_circle(color="black", size=25).encode(x="RMSE:Q", y="Mô hình:N", tooltip=["Mô hình", "Fold", alt.Tooltip("RMSE:Q", format=".3f")]),
    use_container_width=True)
TREE, LIN = ["Stacking", "LightGBM", "Gradient Boosting"], ["Ridge", "Lasso", "ElasticNet", "Linear Regression"]
pairs = [(A["cv_folds"][t]["RMSE"] < A["cv_folds"][l]["RMSE"]) for t in TREE for l in LIN]
st.caption(f"Các mô hình được đánh giá trên cùng 5 fold nên có thể so từng fold: trong {len(pairs) * 5} lần so một mô hình cây với một mô hình tuyến tính trên cùng fold, mô hình cây có RMSE thấp hơn {sum(int(x.sum()) for x in pairs)} lần.")

st.subheader("Khác biệt có ý nghĩa không? So sánh cặp bằng bootstrap")
c1, c2, c3 = st.columns(3)
m_a = c1.selectbox("Mô hình A", names, index=names.index(best))
m_b = c2.selectbox("Mô hình B", names, index=names.index("LightGBM"))
k = c3.selectbox("Thang đo", ["RMSE", "MAE", "MAPE (%)"])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(y), size=(B, len(y)))
full = np.arange(len(y))[None]


def score(p, i):
    e = y[i] - p[i]
    return {"RMSE": np.sqrt(np.mean(e ** 2, axis=1)), "MAE": np.mean(np.abs(e), axis=1),
            "MAPE (%)": np.mean(np.abs(e) / y[i], axis=1) * 100}[k]


diff = score(preds[m_a], idx) - score(preds[m_b], idx)
lo, hi = np.quantile(diff, [0.025, 0.975])
c = st.columns(3)
c[0].metric(f"{k}(A) − {k}(B)", data.vn(float(score(preds[m_a], full)[0] - score(preds[m_b], full)[0]), 3))
c[1].metric("Khoảng 95% của chênh lệch", f"[{data.vn(lo, 3)}; {data.vn(hi, 3)}]")
c[2].metric("Tỷ lệ lần A tốt hơn B", f"{data.vn((diff < 0).mean() * 100, 1)}%")
st.altair_chart(alt.Chart(pd.DataFrame({"chênh lệch": diff}), title=f"Phân phối chênh lệch {k} qua {B} lần bootstrap").mark_bar().encode(
    x=alt.X("chênh lệch:Q", bin=alt.Bin(maxbins=50), title=f"{k}(A) − {k}(B)"), y="count()")
    + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#DC2626").encode(x="x:Q"), use_container_width=True)
sig = "**có ý nghĩa**" if (hi < 0 or lo > 0) else "**không có ý nghĩa thống kê**"
st.markdown(f"Khoảng 95% của chênh lệch {'không chứa' if (hi < 0 or lo > 0) else 'chứa'} 0: khác biệt giữa **{m_a}** và **{m_b}** {sig} ở mức 5%.")

wins = pd.Series(np.array(names)[np.argmin(np.stack([score(preds[n], idx) for n in names]), axis=0)]).value_counts(normalize=True) * 100
st.markdown(f"**Tỷ lệ mỗi mô hình có {k} thấp nhất qua các lần bootstrap**")
st.dataframe(wins.rename("Tỷ lệ (%)").round(1).to_frame().T, width="stretch")

ui.box("critique", f"""
- Trên tập kiểm thử, Gradient Boosting có RMSE thấp hơn {best} một chút, nhưng **không đổi mô hình theo tập kiểm thử**: nếu làm vậy, tập kiểm thử trở thành một phần của quá trình chọn mô hình và mất tính độc lập. {best} được chọn theo tiêu chí đặt trước (CV RMSE thấp nhất).
- **{best} và Gradient Boosting:** khoảng 95% của chênh lệch chứa 0 ở cả RMSE, MAE và MAPE; hai mô hình tốt ngang nhau.
- **{best} và LightGBM:** {best} tốt hơn có ý nghĩa thống kê ở cả ba thang đo; kết hợp các mô hình con giúp ích so với LightGBM đứng một mình.
- **{best} và nhóm tuyến tính:** tốt hơn có ý nghĩa ở MAE và MAPE, nhưng **không có ý nghĩa ở RMSE** với 564 căn kiểm thử. RMSE bị chi phối bởi vài căn rất đắt mà mọi mô hình đều dự báo sai nhiều, nên dao động mạnh giữa các lần lấy mẫu.
- Ridge, Lasso, ElasticNet gần như không khác Linear Regression: số thuộc tính ít so với số mẫu nên chính quy hóa ít tác dụng.
""")
ui.box("evidence", f"Thứ hạng giữa hai nhóm mô hình được giữ nguyên ở cả cross-validation và tập kiểm thử. RMSE kiểm thử xấp xỉ CV RMSE (chênh lệch từ {data.vn(tt['Chênh lệch RMSE so với CV (%)'].min(), 0)}% đến {data.vn(tt['Chênh lệch RMSE so với CV (%)'].max(), 0)}%): không có dấu hiệu quá khớp. Chọn cặp mô hình và thang đo ở trên để tự kiểm chứng các nhận định.")
