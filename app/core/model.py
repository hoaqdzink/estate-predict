"""Cấu trúc mô hình Stacking, giải thích dự báo (SHAP), khoảng dự báo, chỉ số đánh giá."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core import data

MODEL_LABELS = {"Ridge": "Ridge", "Lasso": "Lasso", "ElasticNet": "ElasticNet",
                "Gradient_Boosting": "Gradient Boosting", "LightGBM": "LightGBM"}


def parts():
    L = data.learnt()
    pipe = L["model"].regressor_
    prep, stack = pipe.named_steps["prep"], pipe.named_steps["model"]
    return {
        "prep": prep, "stack": stack, "base_names": [n for n, _ in stack.estimators],
        "w": stack.final_estimator_.coef_, "b": float(stack.final_estimator_.intercept_),
        "tcols": list(prep.get_feature_names_out()),
        "input_cols": L["features"]["num"] + L["features"]["bool"] + L["features"]["cat"],
    }


def original_feature(tcol):
    group, name = tcol.split("__", 1)
    if group == "cat":
        return next(f for f in data.learnt()["features"]["cat"] if name.startswith(f + "_"))
    return name


def to_model_input(df):
    f = data.learnt()["features"]
    X = df[f["num"] + f["bool"] + f["cat"]].copy()
    X[f["bool"]] = X[f["bool"]].astype(float)
    return X.astype({c: float for c in f["num"]})


@st.cache_resource
def explainers():
    """Nền so sánh chung: 100 bản ghi của tập dữ liệu sau tiền xử lý."""
    import shap
    p = parts()
    bg = p["prep"].transform(to_model_input(data.features().sample(100, random_state=0)))
    out = []
    for est in p["stack"].estimators_:
        if hasattr(est, "coef_"):
            out.append(("linear", est, bg.mean(axis=0), float(est.predict(bg).mean())))
        else:
            ex = shap.TreeExplainer(est, data=bg, feature_perturbation="interventional")
            out.append(("tree", ex, None, float(np.ravel(ex.expected_value)[0])))
    return out


def decompose(X):
    """Trả về (vector sau tiền xử lý, dự báo của 5 mô hình con, dự báo cuối) trên thang log1p."""
    p = parts()
    Xt = p["prep"].transform(X)
    base_log = np.array([est.predict(Xt)[0] for est in p["stack"].estimators_])
    return Xt, base_log, p["b"] + float(base_log @ p["w"])


def explain(Xt):
    """SHAP của Stacking = tổng có trọng số SHAP của từng mô hình con (thang log1p)."""
    p = parts()
    phi, base = np.zeros(Xt.shape[1]), p["b"]
    for w, (kind, obj, mean, ev) in zip(p["w"], explainers()):
        v = obj.coef_ * (Xt[0] - mean) if kind == "linear" else np.ravel(obj.shap_values(Xt))
        phi += w * v
        base += w * ev
    grouped = pd.Series(phi, index=p["tcols"]).groupby(original_feature).sum()
    return base, grouped.sort_values(key=np.abs, ascending=False)


def stacking_table(base_log):
    p = parts()
    return pd.DataFrame({"Mô hình con": [MODEL_LABELS[n] for n in p["base_names"]],
                         "Dự báo (log1p)": base_log, "Dự báo (tỷ)": np.expm1(base_log),
                         "Trọng số meta": p["w"], "Đóng góp (log1p)": base_log * p["w"]})


def waterfall(base, contrib, values, top=8):
    head, rest = contrib.iloc[:top], contrib.iloc[top:].sum()
    items = list(head.items()) + ([("Các thuộc tính khác", rest)] if len(contrib) > top else [])
    cum, rows = base, []
    for name, v in items:
        before, cum = cum, cum + v
        label = data.FEATURE_LABELS.get(name, name) + (f" = {values[name]}" if name in values else "")
        rows.append((label, np.expm1(before), np.expm1(cum) - np.expm1(before), v))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.6))
    for i, (label, start, delta, _) in enumerate(rows):
        ax.barh(i, delta, left=start, color="#DC2626" if delta < 0 else "#16A34A")
        ax.text(start + delta, i, f" {delta:+.2f} ".replace(".", ","), va="center",
                ha="left" if delta >= 0 else "right", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(np.expm1(base), color="gray", linestyle=":", label=f"Giá tham chiếu {data.vn(np.expm1(base))} tỷ")
    ax.axvline(np.expm1(cum), color="#2563EB", linestyle="--", label=f"Giá dự báo {data.vn(np.expm1(cum))} tỷ")
    ax.set_xlabel("Giá (tỷ VNĐ)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    table = pd.DataFrame([{"Thuộc tính": r[0], "Đóng góp (tỷ)": r[2], "Ảnh hưởng (%)": (np.exp(r[3]) - 1) * 100}
                          for r in rows])
    return fig, table


# ---------------------------------------------------------------------------
# Khoảng dự báo và chỉ số
# ---------------------------------------------------------------------------
def interval_quantiles(level):
    """Phân vị phần dư ngoài fold trên thang log1p cho mức tin cậy level (%)."""
    oof = data.artifacts()["oof"]
    resid = np.log1p(oof["y"]) - np.log1p(oof["pred"])
    a = (1 - level / 100) / 2
    return np.quantile(resid, [a, 1 - a])


def interval(pred, level):
    q_lo, q_hi = interval_quantiles(level)
    lp = np.log1p(pred)
    return np.expm1(lp + q_lo), np.expm1(lp + q_hi)


def metrics(y, p):
    e = y - p
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "R²": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            "MAPE (%)": float(np.mean(np.abs(e / y)) * 100)}


@st.cache_data
def bootstrap_metrics(y, p, n, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    return pd.DataFrame([metrics(y[i], p[i]) for i in idx])
