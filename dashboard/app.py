"""Explainable-AI pneumonia dashboard for clinicians.

Run from the repo root:
    streamlit run dashboard/app.py

Layout follows the team's design references:
  left   — worklist (TICICDS C1): red = high likelihood, gray = everything else
  middle — prediction + validation (C2 / fig. 4), X-ray with heatmap, lung-zone
           contributions (fig. 3), related findings (AICare B), plain-language
           summary (AICare C) and cohort context (AICare D)
  right  — patient info (C5), what the model can't see, evidence for trust
"""
from __future__ import annotations

import hashlib
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

# Palette (dataviz reference instance). Status red is reserved for "High".
C_FOR, C_AGAINST, C_NEUTRAL = "#eb6834", "#2a78d6", "#8a8984"
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
# cohort: positive = orange / negative = blue, matching "raises" / "lowers" elsewhere
C_COHORT = [C_FOR, C_AGAINST, C_NEUTRAL]
C_HIGH = "#c62f2f"
BAND_STYLE = {"High": (C_HIGH, "▲"), "Medium": ("#52514e", "◆"), "Low": ("#8a8984", "●")}

st.set_page_config(page_title="Pneumonia XAI", page_icon="🫁", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1800px;}
.band-card {border-radius: 10px; padding: .9rem 1.1rem; border: 1px solid rgba(128,128,128,.25);}
.band-label {font-size: 1.65rem; font-weight: 700; line-height: 1.2;}
.muted {opacity: .7; font-size: .85rem;}
.sim-pill {display:inline-block; padding:.05rem .45rem; border-radius:999px;
           border:1px solid rgba(128,128,128,.4); font-size:.72rem; opacity:.8;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 10px;}
</style>""", unsafe_allow_html=True)


# ====================================================================== loading
@st.cache_resource(show_spinner="Loading models (first run downloads weights)…")
def models():
    return M.load_classifier(), M.load_segmenter(), M.load_autoencoder()


@st.cache_data(show_spinner=False)
def cohort() -> tuple[pd.DataFrame | None, dict | None]:
    pq, mj = CACHE / "predictions.parquet", CACHE / "metrics.json"
    df = pd.read_parquet(pq) if pq.exists() else None
    metrics = json.loads(mj.read_text()) if mj.exists() else None
    return df, metrics


@st.cache_data(show_spinner=False)
def dataset_index() -> pd.DataFrame:
    return D.load_index() if D.available() else pd.DataFrame()


@st.cache_data(show_spinner="Analysing image…", max_entries=64)
def analyze(key: str, src: str | bytes, name: str = ""):
    clf, seg, ae = models()
    img = M.prepare(M.read_image(src, name))
    zones = R.zone_masks(R.lung_masks(seg, img.x_seg))
    return {"img": img, "probs": M.predict(clf, img.x), "zones": zones,
            "ood": Rel.ood_score(ae, img.x)}


@st.cache_data(show_spinner=False, max_entries=256)
def saliency_map(key: str, method: str, target: str, _x: torch.Tensor) -> np.ndarray:
    clf = models()[0]
    with torch.enable_grad():
        return S.compute(method, clf, _x, M.target_index(clf, target))


# ====================================================================== charts
def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13), showlegend=False, hoverlabel=dict(font_size=13))
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zerolinecolor="rgba(128,128,128,.5)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def zone_chart(contrib: dict[str, float]) -> go.Figure:
    """TICICDS fig. 3 style: reasons for decreased (left) / increased (right) likelihood."""
    order = [*R.ZONES, R.OUTSIDE][::-1]
    vals = [contrib[z] for z in order]
    colors = [C_FOR if v > 0 else C_AGAINST for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=order, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[f"{v:+.0%}" if abs(v) >= 0.02 else "" for v in vals], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1%} of net attribution<extra></extra>"))
    lim = max(0.15, max(abs(v) for v in vals) * 1.3)
    fig.update_xaxes(range=[-lim, lim * 1.25], tickformat=".0%",
                     title=dict(text=f"<span style='color:{C_AGAINST}'>← lowers likelihood</span>"
                                     f"     <span style='color:{C_FOR}'>raises likelihood →</span>",
                                font=dict(size=12)))
    fig = _layout(fig, 320)
    fig.update_layout(bargap=0.35)
    return fig


def findings_chart(probs: dict[str, float]) -> go.Figure:
    names = M.RELATED[::-1]
    vals = [probs[n] for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=[M.friendly(n) for n in names], orientation="h",
        marker=dict(color=C_SERIES[0], cornerradius=4),
        text=[f"{v:.2f}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=C_NEUTRAL, dash="dash", width=1.5))
    fig.add_annotation(x=0.5, y=1.08, yref="paper", showarrow=False,
                       text="model threshold", font=dict(size=11, color=C_NEUTRAL))
    fig.update_xaxes(range=[0, 1.08])
    fig = _layout(fig, 300)
    fig.update_layout(margin=dict(t=30), bargap=0.35)
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
        groups = [(f"Reference{ref_name}: positive", ref == 1), (f"Reference{ref_name}: negative", ref == 0),
                  ("Uncertain / not mentioned", ~ref.isin([0, 1]))]
    else:
        groups = [("All images", pd.Series(True, index=df.index))]
    for (name, mask), color in zip(groups, C_COHORT):
        if mask.sum():
            fig.add_trace(go.Histogram(
                x=df.loc[mask, col], name=f"{name} (n={int(mask.sum())})",
                xbins=dict(start=0, end=1, size=0.04), marker=dict(color=color, line=dict(width=1, color="rgba(255,255,255,.6)")),
                hovertemplate="score %{x}<br>%{y} images<extra>" + name + "</extra>"))
    for x, txt in ((low, "Medium"), (high, "High")):
        fig.add_vline(x=x, line=dict(color=C_NEUTRAL, dash="dot", width=1))
        fig.add_annotation(x=x, y=1, yref="paper", text=f"{txt} ≥ {x:.2f}", showarrow=False,
                           xanchor="left", font=dict(size=11, color=C_NEUTRAL))
    fig.add_vline(x=score, line=dict(width=3, color=C_NEUTRAL))
    fig.add_annotation(x=score, y=1.0, yref="paper", yanchor="bottom", text="<b>this patient</b>",
                       showarrow=False)
    fig = _layout(fig, 330)
    fig.update_layout(barmode="stack", showlegend=True, margin=dict(t=28, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.32, x=0))
    fig.update_xaxes(range=[0, 1], title=f"Model {M.friendly(target).lower()} score")
    fig.update_yaxes(title="Images", gridcolor="rgba(128,128,128,.18)")
    return fig


# ====================================================================== dialog
@st.dialog("Validate prediction")
def validate_dialog(image_id: str, score: float, band: str, role: str):
    st.markdown(f"Model: **{band}** likelihood of pneumonia (score {score:.2f})")
    agreement = st.radio("Do you agree with the prediction?", FB.AGREEMENT,
                         index=None, horizontal=True)
    actions = [a for a in FB.ACTIONS if st.checkbox(a, key=f"act_{a}")]
    comment = st.text_area("Flag or comment (optional)",
                           placeholder="e.g. heatmap is on the ECG lead, not the lung; "
                                       "opacity is a known old scar")
    if st.button("Submit", type="primary", disabled=agreement is None, width="stretch"):
        FB.record({"image_id": image_id, "model_score": round(score, 4), "model_band": band,
                   "agreement": agreement, "actions": "; ".join(actions),
                   "comment": comment, "reviewer_role": role})
        st.session_state["toast"] = "Validation saved"
        st.rerun()


# ====================================================================== sidebar
df_cohort, metrics = cohort()
idx = dataset_index()
have_dataset = not idx.empty
reviews = FB.latest_by_image()

with st.sidebar:
    st.header("Settings")
    low, high = st.slider(
        "Risk bands (Low | Medium | High)", 0.0, 1.0, (0.40, 0.60), 0.01,
        help="Thresholds on the model score. Tune locally with your clinicians "
             "(Duke sepsis program: thresholds were set per unit).")
    role = st.selectbox("Your role", FB.ROLES)
    st.divider()
    st.subheader("Explanation")
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.55, 0.05)
    show_zones = st.toggle("Show lung zones", True)
    show_gt = st.toggle("Show radiologist outline (if available)", True,
                        help="CheXlocalize masks for consolidation + airspace opacity. "
                             "Not available in real clinical use; shown for evaluation.")
    st.divider()
    st.subheader("Worklist filters")
    splits = sorted(df_cohort["split"].unique()) if df_cohort is not None else []
    split_sel = st.multiselect("Split", splits, default=splits) if splits else []
    band_sel = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    only_open = st.toggle("Only not yet validated", False)
    gt_only = st.toggle("Only images with radiologist outline", False)
    st.divider()
    upload = st.file_uploader("Or analyse your own image", type=["png", "jpg", "jpeg", "dcm"])

if msg := st.session_state.pop("toast", None):
    st.toast(msg, icon="✅")

# ====================================================================== worklist data
def band_of(p: float) -> str:
    return Sm.risk_band(p, low, high)


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

# ====================================================================== selection
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

# ====================================================================== banners
if df_cohort is None:
    if have_dataset:
        st.info("Dataset found but not yet scored. Run `python dashboard/scripts/precompute.py` "
                "to fill the worklist with risk levels, cohort context and validation metrics.")
    else:
        st.info("**Single-image mode.** No CheXlocalize data found in `data/chexlocalize/`. "
                "See `dashboard/README.md` to add it, then run the precompute script. "
                "Showing the sample X-ray; upload your own in the sidebar.")

# ====================================================================== layout
left, mid, right = st.columns([1.05, 2.5, 1.45], gap="medium")

with left:
    st.subheader("Worklist")
    if worklist.empty:
        st.caption("No dataset loaded.")
    else:
        st.caption(f"{len(worklist)} studies · sorted by pneumonia score")
        with st.container(height=900, border=False):
            for r in worklist.head(80).itertuples():
                b = r.band
                icon = {"High": "🟥", None: "⬜"}.get(b, "⬜")
                done = " ✓" if r.id in reviews else ""
                demo = []
                if isinstance(r.age, (int, float)) and not np.isnan(r.age):
                    demo.append(f"{int(r.age)}y")
                if isinstance(r.sex, str):
                    demo.append(r.sex[0])
                label = f"{icon} **{(b or 'Unscored').upper()}**{done}  \n{r.patient}" + \
                    (f" · {' '.join(demo)}" if demo else "")
                if st.button(label, key=f"wl_{r.id}", width="stretch",
                             type="primary" if r.id == meta["id"] and upload is None else "secondary"):
                    st.session_state["selected"] = r.id
                    st.rerun()
            if len(worklist) > 80:
                st.caption(f"+ {len(worklist) - 80} more. Narrow with the filters.")

res = analyze(sel_key, src, src_name)
img, probs, zones = res["img"], res["probs"], res["zones"]
p_pna = probs[M.PNEUMONIA]
band = band_of(p_pna)
color, glyph = BAND_STYLE[band]

with mid:
    st.markdown(f"<span class='muted'>Patient › {meta.get('patient', meta['id'])} › "
                f"{meta.get('study', '')}</span>", unsafe_allow_html=True)

    # ---- prediction + validation (C2)
    with st.container(border=True):
        c1, c2 = st.columns([3, 1.2], vertical_alignment="center")
        with c1:
            st.markdown(
                f"<div class='muted'>Latest prediction · chest X-ray</div>"
                f"<div class='band-label' style='color:{color}'>{glyph} {band} likelihood of pneumonia</div>",
                unsafe_allow_html=True)
            st.markdown(Sm.GUIDANCE[band])
        with c2:
            rv = reviews.get(meta["id"])
            if rv:
                st.markdown(f"✓ **Validated**: {rv['agreement']}  \n"
                            f"<span class='muted'>{rv['reviewer_role']} · {rv['timestamp'][5:16].replace('T', ' ')}</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("⚠ **Not yet validated**")
            if st.button("Validate", type="primary", width="stretch"):
                validate_dialog(meta["id"], p_pna, band, role)
        with st.expander("Score details"):
            st.markdown(
                f"Raw model score: **{p_pna:.3f}**. Scores are calibrated so **0.50 = the model's "
                f"operating threshold** for pneumonia. It is **not** the probability that this patient "
                f"has pneumonia. Bands: Low < {low:.2f} ≤ Medium < {high:.2f} ≤ High.")

    # ---- image + heatmap
    with st.container(border=True):
        t1, t2 = st.columns([1.3, 2], vertical_alignment="bottom")
        target = t1.selectbox("Explain finding", EXPLAIN_TARGETS,
                              format_func=M.friendly,
                              help="Switch to a related finding to check whether the model's "
                                   "evidence is consistent across findings.")
        method = t2.segmented_control("Method", list(S.METHODS), default="gradcam",
                                      format_func=S.METHODS.get) or "gradcam"
        st.caption(S.METHOD_HELP[method])

        with st.spinner(f"Computing {S.METHODS[method]}…"):
            sal = saliency_map(sel_key, method, target, img.x)
        gt = None
        if show_gt and meta.get("has_gt") and meta.get("split"):
            gt = D.gt_mask(meta["id"], meta["split"], img.crop_box, img.orig_shape)
        ov = S.overlay(img.display, sal, alpha)
        base = np.repeat(img.display[..., None], 3, axis=2)
        if show_zones:
            ov, base = R.draw_zones(ov, zones), R.draw_zones(base, zones)
        if gt is not None:
            ov = S.draw_contours(ov, gt)
        i1, i2 = st.columns(2)
        i1.image(base, caption="Original (patient's right on viewer's left)", width="stretch")
        cap = f"{S.METHODS[method]} · evidence for {M.friendly(target).lower()}"
        if gt is not None:
            cap += " · cyan = radiologist outline"
            cap += f" (IoU {S.iou(S.binarize(sal), gt):.2f})"
        i2.image(ov, caption=cap, width="stretch")

    # ---- why: zones + related findings
    contrib = R.zone_contributions(sal, zones)
    with st.container(border=True):
        z1, z2 = st.columns(2)
        with z1:
            st.markdown("**Where the evidence is**: share of attribution by lung zone")
            st.caption("Lung zones come from an automatic segmentation that leaves out lung behind "
                       "the heart and diaphragm, so some real lung evidence counts as 'outside'.")
            st.plotly_chart(zone_chart(contrib), width="stretch",
                            config={"displayModeBar": False})
        with z2:
            st.markdown("**Related findings**: model scores")
            st.plotly_chart(findings_chart(probs), width="stretch",
                            config={"displayModeBar": False})
        with st.expander("Table view"):
            st.dataframe(pd.DataFrame({"Zone": list(contrib), "Net share": [f"{v:+.1%}" for v in contrib.values()]}),
                         hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame({"Finding": [M.friendly(n) for n in M.RELATED],
                                       "Score": [round(probs[n], 3) for n in M.RELATED],
                                       "Above threshold": ["yes" if probs[n] >= .5 else "no" for n in M.RELATED]}),
                         hide_index=True, width="stretch")

    # ---- off-lung attention (Grad-CAM, pneumonia), shared by summary + warnings
    gc = sal if method == "gradcam" and target == M.PNEUMONIA else \
        saliency_map(sel_key, "gradcam", M.PNEUMONIA, img.x)
    outside = R.zone_contributions(gc, zones)[R.OUTSIDE]
    cohort_out = df_cohort["gradcam_outside"].to_numpy() \
        if df_cohort is not None and "gradcam_outside" in df_cohort else None
    cohort_ood = df_cohort["ood"].to_numpy() if df_cohort is not None else None
    flags = Rel.per_image_flags(meta, res["ood"], cohort_ood, outside, cohort_out)
    outside_unusual = any("outside the" in f.text for f in flags)

    # ---- summary (AICare C)
    with st.container(border=True):
        st.markdown("**Summary** <span class='sim-pill'>generated from the numbers above · no LLM</span>",
                    unsafe_allow_html=True)
        tband = band_of(probs[target])
        for line in Sm.build(target, probs[target], tband, contrib,
                             {k: probs[k] for k in M.RELATED if k != target}, S.METHODS[method],
                             outside_unusual=outside_unusual and target == M.PNEUMONIA and method == "gradcam"):
            st.markdown(f"- {line}")

    # ---- cohort (AICare D)
    if df_cohort is not None and f"prob_{target}" in df_cohort:
        with st.container(border=True):
            pct = (df_cohort[f"prob_{target}"] < probs[target]).mean()
            st.markdown(f"**Cohort context**: this patient scores higher than **{pct:.0%}** "
                        f"of the {len(df_cohort)} reference studies")
            st.plotly_chart(cohort_chart(df_cohort, target, probs[target], low, high),
                            width="stretch", config={"displayModeBar": False})

with right:
    # ---- patient info (C5)
    with st.container(border=True):
        st.markdown("**Patient information**")
        a, b, c = st.columns(3)
        age = meta.get("age")
        a.metric("Age", f"{int(age)}" if isinstance(age, (int, float)) and not np.isnan(age) else "—")
        b.metric("Sex", meta.get("sex")[0].upper() if isinstance(meta.get("sex"), str) and meta.get("sex") else "—")
        c.metric("View", meta.get("ap_pa") if isinstance(meta.get("ap_pa"), str) else "—")
        st.caption(f"Study: {meta['id']}")
        labels = {k[6:]: v for k, v in meta.items() if k.startswith("label_") and v in (0, 1, -1)}
        if labels:
            with st.expander("Radiologist reference labels (evaluation only)"):
                st.caption("From the CheXpert report labels. Not available to the model or in real use.")
                txt = {1: "**Present**", 0: "Absent", -1: "Uncertain"}
                st.markdown("\n".join(f"- {k}: {txt[int(v)]}" for k, v in labels.items()))

    # ---- what the model can't see
    with st.container(border=True):
        st.markdown("**What the model can't see**")
        for f in flags:
            if f.level == "warn":
                st.warning(f.text, icon="⚠️")
            else:
                st.caption(f.text)
        with st.expander("Known limits of this model", expanded=False):
            st.markdown("\n".join(f"- {t}" for t in Rel.STATIC_LIMITS))

    # ---- evidence for trust
    with st.container(border=True):
        st.markdown("**Evidence for trust**")
        if metrics:
            pa = metrics["auroc"].get("Pneumonia", {})
            if pa.get("auroc") is not None:
                st.metric("Pneumonia score AUROC" + (" (proxy labels)" if pa.get("proxy") else ""),
                          f"{pa['auroc']:.2f}",
                          help="How well the score separates positive from negative studies. "
                               "1.0 = perfect, 0.5 = chance.")
                if pa.get("proxy"):
                    st.caption("This dataset has no pneumonia labels, so the pneumonia score is checked "
                               f"against the radiologist **{pa['proxy']}** instead.")
                st.caption(f"Based on {pa['n_pos']} positive / {pa['n'] - pa['n_pos']} negative studies"
                           + (". Few positives, so this estimate is uncertain." if pa['n_pos'] < 30 else "."))
            with st.expander("Performance details"):
                st.caption("Discrimination by finding (reference set)")
                st.dataframe(pd.DataFrame([{"Finding": M.friendly(k) + (" (proxy)" if v.get("proxy") else ""),
                                            "AUROC": v["auroc"],
                                            "Positives": v["n_pos"], "Studies": v["n"]}
                                           for k, v in metrics["auroc"].items()]),
                             hide_index=True, width="stretch")
                if metrics.get("localization"):
                    st.caption("Heatmap vs radiologist outline (consolidation + airspace opacity)")
                    st.dataframe(pd.DataFrame([{"Method": S.METHODS[k], "Mean IoU": v["mean_iou"],
                                                "Peak inside outline": v["hit_rate"], "Images": v["n"]}
                                               for k, v in metrics["localization"].items()]),
                                 hide_index=True, width="stretch")
                    st.caption("For context: in Saporta et al. (2022) all saliency methods, Grad-CAM "
                               "included, overlapped radiologist outlines less than a second radiologist did.")
            st.caption(f"Model: {metrics['model']} · evaluated {metrics['generated']} on "
                       f"{metrics['n_images']} studies (CheXlocalize). Not validated on local data.")
        else:
            st.caption("Run the precompute script to evaluate the model on the reference set.")
        fb = FB.load()
        if not fb.empty:
            agree = (fb["agreement"] == "Agree").mean()
            st.caption(f"Clinician validations so far: {len(fb)} · agreement {agree:.0%}")
