"""Explainable-AI pneumonia dashboard for clinicians (Stanford AIMI styling).

Run from the repo root:
    streamlit run dashboard/app.py

Layout follows the team's design references:
  header — Stanford AIMI brand bar
  left   — worklist (TICICDS C1): Cardinal stripe = High likelihood, gray = everything else
  middle — summary card: risk band, score scale, validation (C2 / fig. 4)
           toolbar: finding / method / view / display options
           X-ray with heatmap | why: lung-zone contributions (fig. 3) + plain summary (AICare C)
           tabs: related findings (AICare B), cohort context (AICare D), performance, table view
  right  — patient info (C5), what the model can't see, evidence for trust

Colour rules:
  Cardinal red = brand + "High likelihood" (always with ▲ icon and the word HIGH).
  Orange / blue = evidence that raises / lowers likelihood (validated CVD-safe pair).
  No red/green on clinical values (TICICDS: clinicians read it as the model's reasons).
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import feedback as FB  # noqa: E402
from xai import dataset as D  # noqa: E402
from xai import evaluation as E  # noqa: E402
from xai import model as M  # noqa: E402
from xai import regions as R  # noqa: E402
from xai import reliability as Rel  # noqa: E402
from xai import saliency as S  # noqa: E402
from xai import summary as Sm  # noqa: E402

CACHE = Path(os.environ.get("PNEUMO_CACHE", HERE / "cache"))
SAMPLE = HERE.parent / "data" / "sample_cxr.png"
EXPLAIN_TARGETS = ["Pneumonia", "Consolidation", "Lung Opacity", "Infiltration"]
# model output name -> CheXpert reference label column
LABEL_FOR = {"Pneumonia": "Pneumonia", "Consolidation": "Consolidation",
             "Lung Opacity": "Lung Opacity", "Effusion": "Pleural Effusion",
             "Atelectasis": "Atelectasis", "Edema": "Edema"}
WORKLIST_MAX = 80

# ---------------------------------------------------------------- Stanford palette
CARDINAL, CARDINAL_DARK = "#8C1515", "#820000"
INK, INK_2, INK_3 = "#2E2D29", "#53565A", "#7A7870"     # Stanford black / cool grey / stone
LINE, SURFACE, CANVAS = "#E3E0D8", "#FFFFFF", "#F7F6F2"
SANDSTONE = "#D2C295"
C_RAISE, C_LOWER, C_NEUTRAL = "#D1660F", "#006CB8", "#A8A69E"  # validated pair (dataviz)
WARN_BG, WARN_EDGE = "#FFF6E0", "#B07D00"
FONT = "'Source Sans 3', 'Source Sans Pro', -apple-system, sans-serif"

st.set_page_config(page_title="Pneumonia XAI · Stanford AIMI", page_icon="🫁", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Source+Sans+3:wght@400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: {FONT}; }}
.stApp {{ background: {CANVAS}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 3.4rem; padding-bottom: 2.5rem; max-width: 1760px; }}

/* brand bar */
.brand {{ background: {CARDINAL}; color: #fff; border-radius: 10px; padding: .7rem 1.2rem;
          display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }}
.brand .title {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.35rem; font-weight: 700; letter-spacing: .2px; }}
.brand .sub {{ opacity: .85; font-size: .9rem; margin-left: .8rem; }}
.brand .chip {{ border: 1px solid rgba(255,255,255,.55); border-radius: 999px; padding: .12rem .65rem;
               font-size: .78rem; margin-left: .4rem; white-space: nowrap; }}

/* cards: every keyed container named card_* */
[class*="st-key-card_"] {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px;
                          padding: 1rem 1.1rem 1.05rem; box-shadow: 0 1px 2px rgba(46,45,41,.05); }}
.card-title {{ font-family: 'Source Serif 4', Georgia, serif; font-weight: 700; font-size: 1.06rem;
               color: {INK}; margin: 0 0 .35rem; }}
.eyebrow {{ text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 700; color: {INK_3}; }}
.muted {{ color: {INK_2}; font-size: .86rem; }}
.small {{ font-size: .8rem; color: {INK_3}; }}

/* risk badges: icon + word, never colour alone */
.badge {{ display: inline-flex; align-items: center; gap: .35rem; border-radius: 999px;
          padding: .18rem .7rem; font-weight: 700; font-size: .8rem; letter-spacing: .05em; }}
.badge.High {{ background: {CARDINAL}; color: #fff; }}
.badge.Medium {{ background: #fff; color: {INK}; border: 1.5px solid {INK}; }}
.badge.Low {{ background: #fff; color: {INK_3}; border: 1.5px solid {C_NEUTRAL}; }}
.headline {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.75rem; font-weight: 700;
             line-height: 1.15; color: {INK}; margin: .35rem 0 .3rem; }}
.headline.High {{ color: {CARDINAL}; }}

/* score scale */
.scale {{ position: relative; height: 10px; border-radius: 6px; overflow: hidden; display: flex; margin-top: .55rem; }}
.scale .seg {{ height: 100%; }}
.scale .low {{ background: #ECEAE4; }} .scale .med {{ background: #CFCBC0; }} .scale .high {{ background: {CARDINAL}; opacity: .85; }}
.scale-wrap {{ position: relative; }}
.marker {{ position: absolute; top: -4px; width: 4px; height: 18px; background: {INK}; border-radius: 2px;
           box-shadow: 0 0 0 2px #fff; transform: translateX(-2px); }}
.scale-labels {{ display: flex; justify-content: space-between; font-size: .72rem; color: {INK_3}; margin-top: .3rem; }}
.score-num {{ font-size: 2rem; font-weight: 700; color: {INK}; line-height: 1; }}

/* callouts */
.callout {{ border-left: 4px solid {WARN_EDGE}; background: {WARN_BG}; border-radius: 8px;
            padding: .55rem .75rem; margin: .45rem 0; font-size: .9rem; color: {INK}; }}
.callout.info {{ border-left-color: {C_NEUTRAL}; background: #F4F3EF; }}
.measured {{ background: #F4F1EA; border-radius: 8px; padding: .55rem .75rem; margin-top: .6rem;
             font-size: .86rem; color: {INK}; line-height: 1.45; }}
.kv {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .4rem; margin: .4rem 0 .5rem; }}
.kv div {{ background: {CANVAS}; border-radius: 8px; padding: .45rem .55rem; }}
.kv .k {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; color: {INK_3}; font-weight: 700; }}
.kv .v {{ font-size: 1.15rem; font-weight: 700; color: {INK}; }}
.bullets {{ margin: .2rem 0 0; padding-left: 1.1rem; }} .bullets li {{ margin-bottom: .35rem; line-height: 1.45; }}
.legend-dot {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin:0 .3rem 0 .6rem; vertical-align:-1px; }}

/* worklist cards (buttons keyed wl_*) */
[class*="st-key-wl_"] button {{ justify-content: flex-start; text-align: left; border-radius: 10px;
     border: 1px solid {LINE}; border-left: 6px solid {C_NEUTRAL}; background: {SURFACE}; padding: .5rem .75rem; }}
[class*="st-key-wl_"] button p {{ text-align: left; line-height: 1.35; }}
[class*="st-key-wl_"] button:hover {{ border-color: {INK_3}; border-left-color: inherit; }}
.wl-scroll {{ }}
</style>
""", unsafe_allow_html=True)


# ====================================================================== loading
@st.cache_resource(show_spinner="Loading models (first run downloads weights)…")
def models():
    return M.load_classifier(), M.load_segmenter(), M.load_autoencoder()


@st.cache_data(show_spinner=False)
def cohort(mtime: float = 0.0) -> tuple[pd.DataFrame | None, dict | None]:
    pq, mj = CACHE / "predictions.parquet", CACHE / "metrics.json"
    df = pd.read_parquet(pq) if pq.exists() else None
    metrics = json.loads(mj.read_text()) if mj.exists() else None
    return df, metrics


@st.cache_data(show_spinner=False)
def dataset_index() -> pd.DataFrame:
    return D.load_index() if D.available() else pd.DataFrame()


@st.cache_data(show_spinner="Analysing image…", max_entries=64)
def analyze(key: str, src: str | bytes, name: str = "", version: str = M.PIPELINE_VERSION):
    clf, seg, ae = models()
    img = M.prepare(M.read_image(src, name))
    zones = R.zone_masks(R.lung_masks(seg, img))
    return {"img": img, "probs": M.predict(clf, img.x), "zones": zones,
            "ood": Rel.ood_score(ae, img.x)}


@st.cache_data(show_spinner=False, max_entries=256)
def saliency_map(key: str, method: str, target: str, _prep: M.PreparedImage,
                 version: str = M.PIPELINE_VERSION) -> np.ndarray:
    clf = models()[0]
    with torch.enable_grad():
        return S.compute(method, clf, _prep, M.target_index(clf, target))


def esc(x) -> str:
    return html.escape(str(x))


def md_bold(text: str) -> str:
    """Escape text, then turn **x** / *x* (used by summary/limits strings) into HTML."""
    import re
    t = esc(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)


def card_title(text: str, eyebrow: str | None = None) -> None:
    eb = f"<div class='eyebrow'>{esc(eyebrow)}</div>" if eyebrow else ""
    st.markdown(f"{eb}<div class='card-title'>{esc(text)}</div>", unsafe_allow_html=True)


# ====================================================================== charts
def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13, family=FONT, color=INK), showlegend=False,
                      hoverlabel=dict(font_size=13, font_family=FONT, bgcolor="#fff", bordercolor=LINE))
    fig.update_xaxes(gridcolor="#EEECE6", zerolinecolor="#BDBAB0", linecolor=LINE, tickfont=dict(color=INK_3))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK_2))
    return fig


def zone_chart(contrib: dict[str, float]) -> go.Figure:
    """TICICDS fig. 3 style: reasons for decreased (left) / increased (right) likelihood."""
    order = [*R.ZONES, R.OUTSIDE][::-1]
    vals = [contrib[z] for z in order]
    fig = go.Figure(go.Bar(
        x=vals, y=order, orientation="h",
        marker=dict(color=[C_RAISE if v > 0 else C_LOWER for v in vals], cornerradius=4),
        text=[f"{v:+.0%}" if abs(v) >= 0.02 else "" for v in vals], textposition="outside",
        textfont=dict(color=INK_2), cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1%} of net attribution<extra></extra>"))
    lim = max(0.15, max(abs(v) for v in vals) * 1.3)
    fig.update_xaxes(range=[-lim, lim * 1.25], tickformat=".0%")
    fig = _layout(fig, 300)
    fig.update_layout(bargap=0.38)
    return fig


def findings_chart(probs: dict[str, float]) -> go.Figure:
    names = M.RELATED[::-1]
    vals = [probs[n] for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=[M.friendly(n) for n in names], orientation="h",
        marker=dict(color=C_LOWER, cornerradius=4),
        text=[f"{v:.2f}" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2), hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=INK_3, dash="dash", width=1.5))
    fig.add_annotation(x=0.5, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                       text="model threshold", font=dict(size=11, color=INK_3))
    fig.update_xaxes(range=[0, 1.08])
    fig = _layout(fig, 290)
    fig.update_layout(margin=dict(t=24), bargap=0.38)
    return fig


def reference_series(df: pd.DataFrame, target: str) -> tuple[pd.Series | None, str]:
    """Radiologist labels for `target`, or a clearly named proxy for pneumonia."""
    lab = f"label_{LABEL_FOR.get(target, '')}"
    if lab in df:
        return df[lab], ""
    if target == "Pneumonia" and {"label_Consolidation", "label_Lung Opacity"} <= set(df.columns):
        proxy = (df["label_Consolidation"] == 1) | (df["label_Lung Opacity"] == 1)
        return proxy.astype(int), " consolidation/opacity"
    return None, ""


def cohort_chart(df: pd.DataFrame, target: str, score: float, low: float, high: float) -> go.Figure:
    col = f"prob_{target}"
    ref, ref_name = reference_series(df, target)
    fig = go.Figure()
    if ref is not None:
        groups = [(f"Reference{ref_name}: present", ref == 1, C_RAISE),
                  (f"Reference{ref_name}: absent", ref == 0, C_LOWER),
                  ("Uncertain / not mentioned", ~ref.isin([0, 1]), C_NEUTRAL)]
    else:
        groups = [("All studies", pd.Series(True, index=df.index), C_LOWER)]
    for name, mask, color in groups:
        if mask.sum():
            fig.add_trace(go.Histogram(
                x=df.loc[mask, col], name=f"{name} (n={int(mask.sum())})",
                xbins=dict(start=0, end=1, size=0.04),
                marker=dict(color=color, line=dict(width=1.5, color="#fff")),
                hovertemplate="score %{x}<br>%{y} studies<extra>" + name + "</extra>"))
    for x, txt in ((low, "Medium"), (high, "High")):
        fig.add_vline(x=x, line=dict(color=INK_3, dash="dot", width=1))
        fig.add_annotation(x=x, y=1, yref="paper", text=f"{txt} ≥ {x:.2f}", showarrow=False,
                           xanchor="left", xshift=3, font=dict(size=11, color=INK_3))
    fig.add_vline(x=score, line=dict(width=3, color=INK))
    fig.add_annotation(x=score, y=1.0, yref="paper", yanchor="bottom", text="<b>this patient</b>",
                       showarrow=False, font=dict(color=INK))
    fig = _layout(fig, 330)
    fig.update_layout(barmode="stack", showlegend=True, margin=dict(t=28, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, font=dict(color=INK_2)))
    fig.update_xaxes(range=[0, 1], title=dict(text=f"Model {M.friendly(target).lower()} score",
                                              font=dict(color=INK_3)))
    fig.update_yaxes(title=dict(text="Studies", font=dict(color=INK_3)), gridcolor="#EEECE6")
    return fig


PLOT_CFG = {"displayModeBar": False}


# ====================================================================== dialog
@st.dialog("Validate prediction")
def validate_dialog(image_id: str, score: float, band: str):
    st.markdown(f"Model: **{band}** pneumonia score ({score:.2f})")
    roles = FB.ROLES
    role = st.selectbox("Your role", roles,
                        index=roles.index(st.session_state.get("role", roles[0])))
    agreement = st.radio("Do you agree with the prediction?", FB.AGREEMENT,
                         index=None, horizontal=True)
    st.markdown("**Which action(s) did you take?**")
    actions = [a for a in FB.ACTIONS if st.checkbox(a, key=f"act_{a}")]
    comment = st.text_area("Flag or comment (optional)",
                           placeholder="e.g. heatmap is on the ECG lead, not the lung; "
                                       "opacity is a known old scar")
    if st.button("Submit", type="primary", disabled=agreement is None, width="stretch"):
        FB.record({"image_id": image_id, "model_score": round(score, 4), "model_band": band,
                   "agreement": agreement, "actions": "; ".join(actions),
                   "comment": comment, "reviewer_role": role})
        st.session_state["role"] = role
        st.session_state["toast"] = "Validation saved"
        st.rerun()


# ====================================================================== data + sidebar
# keyed on the cache files' modification time, so re-running precompute shows up without a restart
_mt = max((f.stat().st_mtime for f in CACHE.glob("*") if f.is_file()), default=0.0)
df_cohort, metrics = cohort(_mt)
idx = dataset_index()
have_dataset = not idx.empty
reviews = FB.latest_by_image()

with st.sidebar:
    st.markdown("<div class='card-title'>Settings</div>", unsafe_allow_html=True)
    low, high = st.slider(
        "Risk bands (Low | Medium | High)", 0.0, 1.0, (0.40, 0.60), 0.01,
        help="Thresholds on the model score. Tune locally with your clinicians "
             "(Duke sepsis program: thresholds were set per unit).")
    st.divider()
    upload = st.file_uploader("Analyse your own image", type=["png", "jpg", "jpeg", "dcm"])
    st.divider()
    st.caption("Research prototype for the Stanford AIMI group 5 project. Not for clinical use. "
               "Model: torchxrayvision DenseNet-121 (8 public CXR datasets). "
               "Data: CheXlocalize (Stanford AIMI).")

if msg := st.session_state.pop("toast", None):
    st.toast(msg, icon="✅")


def band_of(p: float) -> str:
    return Sm.risk_band(p, low, high)


# ====================================================================== brand bar
n_studies = f"{len(df_cohort)} studies" if df_cohort is not None else "single-image mode"
st.markdown(
    f"<div class='brand'><div><span class='title'>Stanford AIMI</span>"
    f"<span class='sub'>Pneumonia XAI · chest X-ray decision support</span></div>"
    f"<div><span class='chip'>CheXlocalize · {esc(n_studies)}</span>"
    f"<span class='chip'>Research prototype, not for clinical use</span></div></div>",
    unsafe_allow_html=True)

if df_cohort is None:
    msg = ("Dataset found but not yet scored. Run `python dashboard/scripts/precompute.py`."
           if have_dataset else
           "**Single-image mode.** No CheXlocalize data in `data/chexlocalize/` (see "
           "`dashboard/README.md`). Showing the sample X-ray; upload your own in the sidebar.")
    st.info(msg)

left, mid, right = st.columns([1.0, 3.3, 1.35], gap="medium")

# ====================================================================== worklist
with left:
    with st.container(key="card_worklist"):
        card_title("Worklist")
        with st.popover("Filter studies", width="stretch"):
            splits = sorted(df_cohort["split"].unique()) if df_cohort is not None else []
            split_sel = st.multiselect("Split", splits, default=splits) if splits else []
            band_sel = st.multiselect("Risk band", ["High", "Medium", "Low"],
                                      default=["High", "Medium", "Low"])
            only_open = st.toggle("Only not yet validated", False)
            gt_only = st.toggle("Only with radiologist outline", False)

        worklist = pd.DataFrame()
        if df_cohort is not None:
            worklist = df_cohort.copy()
            worklist["band"] = worklist["prob_Pneumonia"].map(band_of)
            if split_sel:
                worklist = worklist[worklist["split"].isin(split_sel)]
            worklist = worklist[worklist["band"].isin(band_sel)]
            if only_open:
                worklist = worklist[~worklist["id"].isin(reviews)]
            if gt_only:
                worklist = worklist[worklist["has_gt"]]
            worklist = worklist.sort_values("prob_Pneumonia", ascending=False)
        elif have_dataset:
            worklist = idx.copy()
            worklist["band"] = None

        # selection (needed before rendering cards so the active one can be styled)
        if upload is not None:
            data = upload.getvalue()
            sel_key, src, src_name = "upload:" + hashlib.md5(data).hexdigest(), data, upload.name
            meta = {"id": upload.name, "split": None, "has_gt": False}
        elif not worklist.empty:
            if st.session_state.get("selected") not in set(worklist["id"]):
                st.session_state["selected"] = worklist["id"].iloc[0]
            row = worklist.set_index("id").loc[st.session_state["selected"]]
            meta = {"id": st.session_state["selected"], **row.to_dict()}
            sel_key, src, src_name = meta["id"], meta["image_path"], ""
        else:
            sel_key, src, src_name = "sample", str(SAMPLE), ""
            meta = {"id": "sample_cxr.png", "split": None, "has_gt": False}

        if worklist.empty:
            st.markdown("<div class='muted'>No studies loaded.</div>", unsafe_allow_html=True)
        else:
            counts = worklist["band"].value_counts()
            st.markdown(
                f"<div class='muted'>{len(worklist)} studies · highest score first<br>"
                f"<b style='color:{CARDINAL}'>▲ {counts.get('High', 0)} high</b> · "
                f"{counts.get('Medium', 0)} medium · {counts.get('Low', 0)} low</div>",
                unsafe_allow_html=True)
            shown = worklist.head(WORKLIST_MAX)
            # per-card CSS: Cardinal stripe for High, highlighted active card
            css = []
            for r in shown.itertuples():
                k = f".st-key-wl_{r.id} button"
                if r.band == "High":
                    css.append(f"{k} {{ border-left-color: {CARDINAL} !important; }}")
                if r.id == meta["id"] and upload is None:
                    css.append(f"{k} {{ background: #F4EFE6 !important; border-color: {INK_2} !important;"
                               f" box-shadow: 0 0 0 1px {INK_2} inset; }}")
            st.markdown(f"<style>{''.join(css)}</style>", unsafe_allow_html=True)
            with st.container(height=760, border=False):
                for r in shown.itertuples():
                    b = r.band or "Unscored"
                    glyph = {"High": "▲", "Medium": "◆", "Low": "●"}.get(b, "○")
                    done = "  ✓ validated" if r.id in reviews else ""
                    demo = []
                    if isinstance(r.age, (int, float)) and not np.isnan(r.age):
                        demo.append(f"{int(r.age)}y")
                    if isinstance(r.sex, str) and r.sex:
                        demo.append(r.sex[0])
                    who = f"{r.patient}" + (f" · {' '.join(demo)}" if demo else "")
                    label = f"**{glyph} {b.upper()}**{done}  \n{who}"
                    if st.button(label, key=f"wl_{r.id}", width="stretch"):
                        st.session_state["selected"] = r.id
                        st.rerun()
                if len(worklist) > WORKLIST_MAX:
                    st.caption(f"+ {len(worklist) - WORKLIST_MAX} more. Narrow with Filter.")

# ====================================================================== analysis
res = analyze(sel_key, src, src_name, M.PIPELINE_VERSION)
img, probs, zones = res["img"], res["probs"], res["zones"]
p_pna = probs[M.PNEUMONIA]
band = band_of(p_pna)
glyph = {"High": "▲", "Medium": "◆", "Low": "●"}[band]

with mid:
    # ---------------------------------------------------------- summary card (C2)
    with st.container(key="card_hero"):
        c1, c2, c3 = st.columns([2.5, 1.35, 1.05], gap="large", vertical_alignment="center")
        with c1:
            crumb = " · ".join(str(x) for x in (meta.get("patient", meta["id"]), meta.get("study"))
                               if isinstance(x, str) and x)
            st.markdown(
                f"<div class='eyebrow'>{esc(crumb)} · latest chest X-ray</div>"
                f"<div class='headline {band}'>{glyph} {Sm.HEADLINE[band]}</div>"
                f"<div class='muted'>{md_bold(Sm.GUIDANCE[band])}</div>", unsafe_allow_html=True)
            if df_cohort is not None and "label_Consolidation" in df_cohort:
                bt = E.band_table(df_cohort, "prob_Pneumonia", low, high).loc[band]
                if bt["n"]:
                    st.markdown(
                        f"<div class='measured'><b>What {band} meant on the {len(df_cohort)} reference films</b> "
                        f"({bt['n']} scored {band}): consolidation {E.fmt_ci(*bt['Consolidation'])} · "
                        f"consolidation or opacity {E.fmt_ci(*bt['Consolidation or opacity'])} · "
                        f"any abnormal finding {E.fmt_ci(*bt['Any abnormal finding'])}. "
                        f"<span class='small'>95% ranges; radiologist labels; no pneumonia labels "
                        f"available.</span></div>", unsafe_allow_html=True)
        with c2:
            pos = min(max(p_pna, 0), 1) * 100
            st.markdown(
                f"<div class='eyebrow'>Model score</div>"
                f"<div class='score-num'>{p_pna:.2f} <span class='badge {band}'>{glyph} {band.upper()}</span></div>"
                f"<div class='scale-wrap'><div class='scale'>"
                f"<div class='seg low' style='width:{low * 100:.1f}%'></div>"
                f"<div class='seg med' style='width:{(high - low) * 100:.1f}%'></div>"
                f"<div class='seg high' style='width:{(1 - high) * 100:.1f}%'></div></div>"
                f"<div class='marker' style='left:{pos:.1f}%'></div></div>"
                f"<div class='scale-labels'><span>0</span><span>Low</span><span>Medium</span>"
                f"<span>High</span><span>1</span></div>"
                f"<div class='small' style='margin-top:.3rem'>0.50 = model's operating threshold. "
                f"Not the probability of disease.</div>", unsafe_allow_html=True)
        with c3:
            rv = reviews.get(meta["id"])
            if rv:
                st.markdown(f"<div class='eyebrow'>Validation</div><div style='font-weight:700'>"
                            f"✓ {esc(rv['agreement'])}</div><div class='small'>{esc(rv['reviewer_role'])} · "
                            f"{esc(str(rv['timestamp'])[5:16].replace('T', ' '))}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='eyebrow'>Validation</div>"
                            "<div style='font-weight:700'>⚠ Not yet validated</div>",
                            unsafe_allow_html=True)
            if st.button("Validate", type="primary", width="stretch", key="validate_btn"):
                validate_dialog(meta["id"], p_pna, band)

    # ---------------------------------------------------------- toolbar
    with st.container(key="card_toolbar"):
        t1, t2 = st.columns([1, 2], gap="medium", vertical_alignment="bottom")
        target = t1.selectbox("Explain finding", EXPLAIN_TARGETS, format_func=M.friendly,
                              help="Switch to a related finding to check whether the model's "
                                   "evidence is consistent across findings.")
        method = t2.segmented_control("Explanation method", list(S.METHODS), default="gradcam",
                                      format_func=S.METHODS.get, key="method") or "gradcam"
        st.markdown(f"<div class='small'>{esc(S.METHOD_HELP[method])}</div>", unsafe_allow_html=True)

    with st.spinner(f"Computing {S.METHODS[method]}…"):
        sal = saliency_map(sel_key, method, target, img, M.PIPELINE_VERSION)
    contrib = R.zone_contributions(sal, zones)

    # shared off-lung check (Grad-CAM, pneumonia) for the summary + warnings
    gc = sal if method == "gradcam" and target == M.PNEUMONIA else \
        saliency_map(sel_key, "gradcam", M.PNEUMONIA, img, M.PIPELINE_VERSION)
    outside = R.zone_contributions(gc, zones)[R.OUTSIDE]
    cohort_out = df_cohort["gradcam_outside"].to_numpy() \
        if df_cohort is not None and "gradcam_outside" in df_cohort else None
    cohort_ood = df_cohort["ood"].to_numpy() if df_cohort is not None else None
    flags = Rel.per_image_flags(meta, res["ood"], cohort_ood, outside, cohort_out)
    outside_unusual = any("outside the" in f.text for f in flags)

    # ---------------------------------------------------------- image + summary | why
    v1, v2 = st.columns([1.12, 1], gap="medium")
    with v1:
        with st.container(key="card_image"):
            c_title, c_disp = st.columns([2, 1], vertical_alignment="center")
            with c_title:
                card_title(f"Evidence for {M.friendly(target).lower()}", eyebrow=S.METHODS[method])
            view = st.segmented_control("View", ["Heatmap", "Original", "Compare"],
                                        default="Heatmap", key="view",
                                        label_visibility="collapsed") or "Heatmap"
            with c_disp.popover("Display", width="stretch"):
                alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.55, 0.05)
                show_zones = st.toggle("Lung zones", True)
                show_gt = st.toggle("Radiologist outline", True,
                                    help="CheXlocalize masks for consolidation + airspace opacity. "
                                         "Not available in real clinical use; shown for evaluation.")
            gt = None
            if show_gt and meta.get("has_gt") and meta.get("split"):
                gt = D.gt_mask(meta["id"], meta["split"], img)
            ov = S.overlay(img.display, sal, alpha)
            base = np.repeat(img.display[..., None], 3, axis=2)
            if show_zones:
                ov, base = R.draw_zones(ov, zones), R.draw_zones(base, zones)
            if gt is not None:
                ov = S.draw_contours(ov, gt)
            if view == "Compare":
                a, b = st.columns(2)
                a.image(base, width="stretch")
                b.image(ov, width="stretch")
            else:
                st.image(ov if view == "Heatmap" else base, width="stretch")
            notes = ["Patient's right is on the viewer's left"]
            if gt is not None:
                notes.append(f"<span style='color:#0099B0;font-weight:700'>cyan</span> = radiologist "
                             f"outline (overlap IoU {S.iou(S.binarize(sal), gt):.2f})")
            if view != "Original":
                notes.append("brighter = stronger evidence")
            st.markdown(f"<div class='small'>{' · '.join(notes)}</div>", unsafe_allow_html=True)
            loc = (metrics or {}).get("localization", {})
            if method in loc and "_chance" in loc:
                h, iou_m = loc[method]["hit"], loc[method]["iou"]
                ch, lungs_b = loc["_chance"]["hit"], loc["_lungs"]["iou"]
                beats = iou_m["mean"] > lungs_b["mean"]
                st.markdown(
                    f"<div class='measured'><b>How reliable is this heatmap?</b> On {h['n']} reference films "
                    f"with a radiologist outline, the {S.METHODS[method]} peak for pneumonia fell inside "
                    f"the outline {E.fmt_ci(h['mean'], h['lo'], h['hi'])} of the time (a random point: "
                    f"{ch['mean']:.0%}). Overlap IoU {iou_m['mean']:.2f} vs {lungs_b['mean']:.2f} for "
                    f"simply highlighting both lungs, so it is "
                    f"<b>{'better' if beats else 'no better'} than that naive baseline</b>.</div>",
                    unsafe_allow_html=True)

        with st.container(key="card_summary"):
            card_title("Summary", eyebrow="Generated from the numbers on screen · no LLM")
            tband = band_of(probs[target])
            lines = Sm.build(target, probs[target], tband, contrib,
                             {k: probs[k] for k in M.RELATED if k != target}, S.METHODS[method],
                             outside_unusual=outside_unusual and target == M.PNEUMONIA
                             and method == "gradcam", signed=method != "gradcam")
            st.markdown("<ul class='bullets'>" + "".join(f"<li>{md_bold(l)}</li>" for l in lines)
                        + "</ul>", unsafe_allow_html=True)

    with v2, st.container(key="card_why"):
        card_title("Why the model said it", eyebrow="Lung-zone contributions")
        st.markdown(
            f"<div class='small'><span class='legend-dot' style='background:{C_RAISE};margin-left:0'></span>"
            f"raises the score<span class='legend-dot' style='background:{C_LOWER}'></span>"
            f"lowers the score</div>", unsafe_allow_html=True)
        st.plotly_chart(zone_chart(contrib), width="stretch", config=PLOT_CFG)
        if method == "gradcam":
            st.markdown("<div class='small'><b>Grad-CAM can only show evidence <i>for</i></b> a finding. "
                        "Switch to Integrated Gradients or Occlusion to see regions that lower the score."
                        "</div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>Zones come from automatic lung segmentation, which leaves "
                    "out lung behind the heart and diaphragm, so some real lung evidence counts "
                    "as 'outside'.</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid #E3E0D8;margin:.9rem 0 .7rem'>",
                    unsafe_allow_html=True)
        card_title("Other findings that raise the same score", eyebrow="Differential on imaging")
        if df_cohort is not None:
            cols = [f"prob_{n}" for n in M.RELATED if f"prob_{n}" in df_cohort]
            r = df_cohort[cols].corrwith(df_cohort["prob_Pneumonia"])
            st.markdown(f"<div class='small'>On the reference films the pneumonia score moves with each of "
                        f"these (correlation {r.min():.2f}–{r.max():.2f}), so it cannot tell them apart. "
                        f"Bars past the dashed line are above the model's threshold.</div>",
                        unsafe_allow_html=True)
        st.plotly_chart(findings_chart(probs), width="stretch", config=PLOT_CFG)

    # ---------------------------------------------------------- tabs
    with st.container(key="card_tabs"):
        tab2, tab3, tab4 = st.tabs(["Cohort context", "Model performance", "Table view"])
        with tab2:
            if df_cohort is not None and f"prob_{target}" in df_cohort:
                pct = (df_cohort[f"prob_{target}"] < probs[target]).mean()
                st.markdown(f"<div class='muted'>This patient scores higher than <b>{pct:.0%}</b> of the "
                            f"{len(df_cohort)} reference studies.</div>", unsafe_allow_html=True)
                st.plotly_chart(cohort_chart(df_cohort, target, probs[target], low, high),
                                width="stretch", config=PLOT_CFG)
            else:
                st.caption("Run the precompute script to see cohort context.")
        with tab3:
            if metrics:
                ci = lambda v: E.fmt_ci(v["auroc"], v["lo"], v["hi"], pct=False) if v.get("lo") is not None else (f"{v['auroc']:.2f}" if v.get("auroc") else "—")  # noqa: E731
                st.markdown("<div class='card-title' style='font-size:.98rem'>1 · Discrimination</div>"
                            "<div class='muted'>Each model output against its own radiologist label. "
                            "AUROC: 1.0 = perfect, 0.5 = chance; 95% bootstrap range.</div>",
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{"Model output": M.friendly(k) + (f" (vs {v['proxy']})" if v.get("proxy") else ""),
                                            "AUROC (95% range)": ci(v), "Positives": v["n_pos"],
                                            "Studies": v["n"]} for k, v in metrics["auroc"].items()]),
                             hide_index=True, width="stretch")
                if metrics.get("specificity"):
                    st.markdown("<div class='card-title' style='font-size:.98rem;margin-top:.6rem'>2 · Specificity of the "
                                "pneumonia score</div><div class='muted'>How well the <b>pneumonia</b> score "
                                "ranks films with each label. Similar values across conditions mean the score is "
                                "not specific.</div>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame([{"Radiologist label": k, "Pneumonia-score AUROC": ci(v),
                                                "Positives": v["n_pos"]}
                                               for k, v in sorted(metrics["specificity"].items(),
                                                                  key=lambda kv: -kv[1]["auroc"])]),
                                 hide_index=True, width="stretch")
                if df_cohort is not None and "label_Consolidation" in df_cohort:
                    bt = E.band_table(df_cohort, "prob_Pneumonia", low, high)
                    st.markdown("<div class='card-title' style='font-size:.98rem;margin-top:.6rem'>3 · What each "
                                "band contains</div><div class='muted'>Share of reference films in each band with "
                                "each radiologist label, at your current thresholds (95% Wilson range).</div>",
                                unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame([{"Band": b, "Films": int(r["n"]),
                                                **{k: E.fmt_ci(*r[k]) for k in ("Consolidation", "Consolidation or opacity",
                                                                                "Any abnormal finding")}}
                                               for b, r in bt.iterrows()]), hide_index=True, width="stretch")
                loc = metrics.get("localization", {})
                if "_chance" in loc:
                    st.markdown("<div class='card-title' style='font-size:.98rem;margin-top:.6rem'>4 · Heatmap "
                                "localization</div><div class='muted'>Pneumonia heatmaps vs radiologist outline "
                                "(consolidation + airspace opacity), with baselines a heatmap should beat.</div>",
                                unsafe_allow_html=True)
                    rows = [{"Method": S.METHODS[k], "Peak inside outline": E.fmt_ci(v["hit"]["mean"], v["hit"]["lo"], v["hit"]["hi"]),
                             "Mean IoU": E.fmt_ci(v["iou"]["mean"], v["iou"]["lo"], v["iou"]["hi"], pct=False),
                             "Films": v["hit"]["n"]} for k, v in loc.items() if not k.startswith("_")]
                    rows += [{"Method": "Baseline: random point", "Peak inside outline": f"{loc['_chance']['hit']['mean']:.0%}",
                              "Mean IoU": "—", "Films": loc["_chance"]["hit"]["n"]},
                             {"Method": "Baseline: highlight both lungs", "Peak inside outline": "—",
                              "Mean IoU": E.fmt_ci(loc["_lungs"]["iou"]["mean"], loc["_lungs"]["iou"]["lo"],
                                                   loc["_lungs"]["iou"]["hi"], pct=False),
                              "Films": loc["_lungs"]["iou"]["n"]}]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("For context: in Saporta et al. (2022) every saliency method, Grad-CAM "
                           "included, overlapped radiologist outlines less than a second radiologist did. "
                           f"Evaluated {metrics['generated']} on {metrics['n_images']} CheXlocalize validation "
                           f"films (model input: {metrics.get('input_mode', '?')}); not validated on local data.")
            else:
                st.caption("Run the precompute script to evaluate the model.")
        with tab4:
            a, b = st.columns(2)
            a.dataframe(pd.DataFrame({"Zone": list(contrib),
                                      "Net share": [f"{v:+.1%}" for v in contrib.values()]}),
                        hide_index=True, width="stretch")
            b.dataframe(pd.DataFrame({"Finding": [M.friendly(n) for n in M.RELATED],
                                      "Score": [round(probs[n], 3) for n in M.RELATED],
                                      "Above threshold": ["yes" if probs[n] >= .5 else "no" for n in M.RELATED]}),
                        hide_index=True, width="stretch")

# ====================================================================== right column
with right:
    with st.container(key="card_patient"):
        card_title("Patient")
        age, sex, view_pos = meta.get("age"), meta.get("sex"), meta.get("ap_pa")
        fmt = lambda v: esc(v) if isinstance(v, str) and v else "—"  # noqa: E731
        st.markdown(
            "<div class='kv'>"
            f"<div><div class='k'>Age</div><div class='v'>"
            f"{int(age) if isinstance(age, (int, float)) and not np.isnan(age) else '—'}</div></div>"
            f"<div><div class='k'>Sex</div><div class='v'>{fmt(sex[0].upper() if isinstance(sex, str) and sex else None)}</div></div>"
            f"<div><div class='k'>View</div><div class='v'>{fmt(view_pos)}</div></div></div>"
            f"<div class='small'>Study {esc(meta['id'])}</div>", unsafe_allow_html=True)
        labels = {k[6:]: v for k, v in meta.items() if k.startswith("label_") and v in (0, 1, -1)}
        if labels:
            with st.expander("Radiologist reference labels"):
                st.caption("For evaluation only. Not available to the model or in real use.")
                txt = {1: "**Present**", 0: "Absent", -1: "Uncertain"}
                st.markdown("\n".join(f"- {k}: {txt[int(v)]}" for k, v in labels.items()))

    with st.container(key="card_limits"):
        card_title("What the model can't see")
        for f in flags:
            st.markdown(f"<div class='callout {'info' if f.level != 'warn' else ''}'>"
                        f"{'⚠ ' if f.level == 'warn' else ''}{md_bold(f.text)}</div>",
                        unsafe_allow_html=True)
        with st.expander("Known limits of this model"):
            st.markdown("\n".join(f"- {t}" for t in Rel.STATIC_LIMITS))

    with st.container(key="card_trust"):
        card_title("Evidence for trust")
        if metrics:
            pa = metrics["auroc"].get("Pneumonia", {})
            if pa.get("auroc") is not None:
                st.markdown(
                    f"<div class='eyebrow'>Pneumonia score AUROC{' · proxy labels' if pa.get('proxy') else ''}</div>"
                    f"<div class='score-num'>{pa['auroc']:.2f}</div>"
                    + (f"<div class='small'>95% range {pa['lo']:.2f}–{pa['hi']:.2f}</div>" if pa.get("lo") else "")
                    + f"<div class='small' style='margin-top:.35rem'>{pa['n_pos']} positive / "
                    f"{pa['n'] - pa['n_pos']} negative studies"
                    + (f". No pneumonia labels in this dataset, so checked against "
                       f"<b>{esc(pa['proxy'])}</b>." if pa.get("proxy") else ".")
                    + "</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small' style='margin-top:.5rem'>{esc(metrics['model'])} · "
                        f"evaluated {esc(metrics['generated'])} · details in <b>Model performance</b>.</div>",
                        unsafe_allow_html=True)
        else:
            st.caption("Run the precompute script to evaluate the model on the reference set.")
        fb = FB.load()
        if not fb.empty:
            agree = (fb["agreement"] == "Agree").mean()
            st.markdown(f"<div class='small' style='margin-top:.4rem'>Clinician validations: "
                        f"<b>{len(fb)}</b> · agreement <b>{agree:.0%}</b></div>", unsafe_allow_html=True)
