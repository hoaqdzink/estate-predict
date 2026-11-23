"""Thành phần giao diện dùng chung: tiêu đề trang, khung Ý tưởng / Dẫn dắt / Phản biện / Minh chứng."""
import re

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from core import data

CSS = """
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1240px;}
h1 {font-size: 2.05rem !important; font-weight: 700 !important; letter-spacing: -0.01em;}
h2, h3 {font-weight: 650 !important; letter-spacing: -0.005em;}
.page-tag {display:inline-block; padding:3px 12px; border-radius:999px; background:#E8EEFC; color:#1D4ED8;
           font-size:0.75rem; font-weight:600; letter-spacing:0.02em;}
.box {border-left:4px solid; border-radius:10px; padding:0.85rem 1.1rem; margin:0.8rem 0 1.1rem 0; font-size:0.95rem;
      line-height:1.6;}
.box b.title {display:block; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;}
.box.idea {background:#EFF4FF; border-color:#2563EB;} .box.idea b.title {color:#1D4ED8;}
.box.lead {background:#F5F6F8; border-color:#6B7280;} .box.lead b.title {color:#4B5563;}
.box.critique {background:#FFF5EB; border-color:#EA7A2B;} .box.critique b.title {color:#C2410C;}
.box.evidence {background:#ECFAF1; border-color:#16A34A;} .box.evidence b.title {color:#15803D;}
.box code {background:rgba(0,0,0,0.05); padding:1px 5px; border-radius:4px; font-size:0.85em;}
div[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #E5E9F2; border-radius:12px; padding:0.75rem 1rem;
                             box-shadow:0 1px 2px rgba(16,24,40,0.04);}
div[data-testid="stMetricValue"] {font-size:1.55rem; font-weight:650;}
div[data-testid="stMetricLabel"] p {font-size:0.82rem; color:#6B7280;}
div[data-testid="stMetricDelta"] svg {display:none;}
div[data-testid="stMetricDelta"] {font-size:0.8rem;}
/* Bảng Markdown */
div[data-testid="stMarkdownContainer"] table {border-collapse:separate; border-spacing:0; width:100%;
    border:1px solid #E5E9F2; border-radius:10px; overflow:hidden; font-size:0.92rem; margin:0.4rem 0 1rem 0;}
div[data-testid="stMarkdownContainer"] th {background:#F4F6FB; font-weight:600; color:#374151; text-align:left;}
div[data-testid="stMarkdownContainer"] th, div[data-testid="stMarkdownContainer"] td {padding:0.55rem 0.8rem;
    border:none; border-bottom:1px solid #EEF1F6;}
div[data-testid="stMarkdownContainer"] tr:last-child td {border-bottom:none;}
div[data-testid="stMarkdownContainer"] tbody tr:nth-child(even) {background:#FAFBFD;}
/* Thẻ tab */
button[data-baseweb="tab"] {font-weight:600;}
/* Khối giới thiệu */
.hero {background:linear-gradient(120deg,#1E3A8A 0%,#2563EB 55%,#3B82F6 100%); color:#fff; border-radius:16px;
       padding:1.8rem 2rem; margin-bottom:1.2rem;}
.hero h1 {color:#fff !important; margin:0.2rem 0 0.4rem 0; font-size:2.1rem !important;}
.hero p {color:#DBE6FF; margin:0; font-size:1.02rem;}
.hero .chips {margin-top:1rem;}
.hero .chip {display:inline-block; background:rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.25);
             color:#fff; border-radius:999px; padding:3px 12px; font-size:0.8rem; margin:0 6px 6px 0;}
.info-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px;}
.info-item {background:#F8FAFD; border:1px solid #E5E9F2; border-radius:12px; padding:0.7rem 0.9rem;}
.info-item .k {font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:#6B7280; font-weight:600;}
.info-item .v {font-size:0.95rem; color:#111827; margin-top:2px;}
.steps {display:flex; gap:8px; flex-wrap:wrap; margin:0.3rem 0 1rem 0;}
.step {flex:1 1 130px; background:#fff; border:1px solid #E5E9F2; border-radius:12px; padding:0.7rem 0.8rem;
       box-shadow:0 1px 2px rgba(16,24,40,0.04); position:relative;}
.step .n {display:inline-flex; width:24px; height:24px; border-radius:50%; background:#2563EB; color:#fff;
          font-size:0.78rem; font-weight:700; align-items:center; justify-content:center; margin-bottom:6px;}
.step .t {font-weight:650; font-size:0.9rem; color:#111827;}
.step .d {font-size:0.78rem; color:#6B7280; margin-top:2px; line-height:1.4;}
.side-card {background:#fff; border:1px solid #E5E9F2; border-radius:12px; padding:0.7rem 0.9rem; margin-bottom:0.6rem;}
.side-card .k {font-size:0.72rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;}
.side-card .row {display:flex; justify-content:space-between; font-size:0.9rem; padding:2px 0;}
.side-card .row b {color:#1D4ED8;}
</style>
"""
TITLES = {"idea": "Ý tưởng", "lead": "Dẫn dắt", "critique": "Phản biện", "evidence": "Minh chứng"}


PALETTE = ["#2563EB", "#EA7A2B", "#16A34A", "#DC2626", "#7C3AED", "#0891B2", "#CA8A04", "#DB2777"]
FONT = '"Source Sans Pro", "Source Sans 3", sans-serif'


def _altair_theme():
    axis = {"labelFont": FONT, "titleFont": FONT, "labelFontSize": 12, "titleFontSize": 12, "labelColor": "#4B5563",
            "titleColor": "#374151", "titleFontWeight": 600, "gridColor": "#EEF1F6", "domain": False,
            "tickColor": "#E5E9F2", "labelLimit": 260, "titlePadding": 10}
    return {"config": {
        "font": FONT, "background": "transparent", "view": {"stroke": None},
        "title": {"font": FONT, "fontSize": 15, "fontWeight": 600, "color": "#111827", "anchor": "start", "offset": 12},
        "axis": axis, "axisY": {**axis, "labelPadding": 6}, "axisX": axis,
        "legend": {"labelFont": FONT, "titleFont": FONT, "labelFontSize": 12, "titleFontSize": 12,
                   "titleColor": "#374151", "labelColor": "#4B5563", "orient": "top", "labelLimit": 260},
        "range": {"category": PALETTE},
        "bar": {"color": PALETTE[0], "cornerRadiusEnd": 3},
        "line": {"color": PALETTE[0], "strokeWidth": 2.5},
        "point": {"color": PALETTE[0], "filled": True, "size": 60},
        "circle": {"color": PALETTE[0]}, "area": {"color": PALETTE[0]},
        "boxplot": {"box": {"color": "#93B4F7"}, "median": {"color": "#1E3A8A"}, "rule": {"color": "#94A3B8"}},
        "scale": {"bandPaddingInner": 0.25},
        "rect": {"stroke": "white", "strokeWidth": 3},
    }}


def _decimals(col):
    v = col.dropna().astype(float)
    v = v[np.isfinite(v)]
    for d in range(0, 5):
        if np.allclose(v, v.round(d), atol=1e-9):
            return d
    return 3


def vn_style(df):
    """Định dạng bảng kiểu Việt Nam (1.234,56), giữ số chữ số thập phân đã làm tròn."""
    sty = df.style
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_float_dtype(df[c]):
            sty = sty.format(subset=[c], precision=_decimals(df[c]), decimal=",", thousands=".", na_rep="–")
        elif pd.api.types.is_integer_dtype(df[c]):
            sty = sty.format(subset=[c], thousands=".", na_rep="–")
    if pd.api.types.is_float_dtype(df.index):
        sty = sty.format_index(precision=_decimals(df.index.to_series()), decimal=",", thousands=".")
    return sty


def _patch_dataframe():
    if getattr(DeltaGenerator.dataframe, "_vn", False):
        return
    original = DeltaGenerator.dataframe

    def dataframe(self, data=None, *args, **kwargs):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame) and not data.columns.duplicated().any() and data.index.is_unique:
            data = vn_style(data)
        return original(self, data, *args, **kwargs)

    dataframe._vn = True
    DeltaGenerator.dataframe = dataframe
    st.dataframe = lambda *a, **k: dataframe(st._main, *a, **k)


def _patch_altair_chart():
    """Streamlit mới dùng width="stretch" thay cho use_container_width; giữ tương thích cả hai."""
    import inspect
    if getattr(DeltaGenerator.altair_chart, "_vn", False):
        return
    original = DeltaGenerator.altair_chart
    has_width = "width" in inspect.signature(original).parameters

    def altair_chart(self, chart, *args, use_container_width=None, **kwargs):
        if has_width and use_container_width is not None:
            kwargs.setdefault("width", "stretch" if use_container_width else "content")
        elif use_container_width is not None:
            kwargs["use_container_width"] = use_container_width
        return original(self, chart, *args, **kwargs)

    altair_chart._vn = True
    DeltaGenerator.altair_chart = altair_chart
    st.altair_chart = lambda *a, **k: altair_chart(st._main, *a, **k)


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        alt.theme.register("estate", enable=True)(_altair_theme)
    except AttributeError:  # altair < 5.5
        alt.themes.register("estate", _altair_theme)
        alt.themes.enable("estate")
    plt.rcParams.update({
        "figure.dpi": 110, "font.family": "sans-serif", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#CBD2DE", "axes.labelcolor": "#374151",
        "xtick.color": "#4B5563", "ytick.color": "#4B5563", "axes.grid": True, "grid.color": "#EEF1F6",
        "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.prop_cycle":
        plt.cycler(color=PALETTE), "figure.facecolor": "white", "axes.facecolor": "white",
    })
    _patch_dataframe()
    _patch_altair_chart()


def hero(tag, title, subtitle, chips=()):
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><span class="page-tag" style="background:rgba(255,255,255,0.18);color:#fff">{tag}</span>'
                f'<h1>{title}</h1><p>{subtitle}</p><div class="chips">{chips_html}</div></div>', unsafe_allow_html=True)


def info_grid(items):
    cells = "".join(f'<div class="info-item"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in items)
    st.markdown(f'<div class="info-grid">{cells}</div>', unsafe_allow_html=True)


def steps(items):
    cells = "".join(f'<div class="step"><div class="n">{i}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
                    for i, (t, d) in enumerate(items, 1))
    st.markdown(f'<div class="steps">{cells}</div>', unsafe_allow_html=True)


def header(tag, title, subtitle=""):
    st.markdown(f'<span class="page-tag">{tag}</span>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def _inline(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return re.sub(r"(?<![*\w])\*(?!\s)(.+?)\*(?![*\w])", r"<i>\1</i>", t)


def _to_html(text):
    out, items = [], []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(f"<li>{_inline(line[2:])}</li>")
            continue
        if items:
            out.append("<ul style='margin:0.2rem 0 0.2rem 1.1rem;padding:0'>" + "".join(items) + "</ul>")
            items = []
        if line:
            out.append(f"<p style='margin:0.2rem 0'>{_inline(line)}</p>")
    if items:
        out.append("<ul style='margin:0.2rem 0 0.2rem 1.1rem;padding:0'>" + "".join(items) + "</ul>")
    return "".join(out)


def box(kind, text):
    """Khung nội dung theo mạch trình bày (Markdown đơn giản: **đậm**, *nghiêng*, `mã`, danh sách '- ')."""
    st.markdown(f'<div class="box {kind}"><b class="title">{TITLES[kind]}</b>{_to_html(text)}</div>',
                unsafe_allow_html=True)


def figure(name, caption):
    path = data.FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chưa có biểu đồ `{name}`; hãy thực thi lại notebook.")


def config_note():
    st.caption("⚙️ Một số phân tích trên trang này thay đổi theo **Cấu hình phân tích** ở thanh bên trái. "
               "Mô hình đã huấn luyện không thay đổi.")
