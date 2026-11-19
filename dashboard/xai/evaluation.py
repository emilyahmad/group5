"""Honest summary statistics: every number ships with its uncertainty.

Used by the precompute script (stored metrics) and the app (band tables that
change live with the user's thresholds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

Z = 1.96
PROXY_NAME = "consolidation or lung opacity"


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Proportion k/n with a 95% Wilson interval (well behaved for small n)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def auroc_ci(y, s, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC with a 95% percentile-bootstrap interval."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    npos = int(y.sum())
    out = {"n": int(len(y)), "n_pos": npos, "auroc": None, "lo": None, "hi": None}
    if not 0 < npos < len(y):
        return out
    out["auroc"] = round(float(roc_auc_score(y, s)), 3)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            stats.append(roc_auc_score(y[i], s[i]))
    out["lo"], out["hi"] = (round(float(v), 3) for v in np.percentile(stats, [2.5, 97.5]))
    return out


def mean_ci(x, n_boot: int = 2000, seed: int = 0) -> dict:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if not len(x):
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}


def proxy_label(df: pd.DataFrame) -> pd.Series | None:
    """'Consolidation or lung opacity present': the closest available stand-in for pneumonia."""
    if {"label_Consolidation", "label_Lung Opacity"} <= set(df.columns):
        return ((df["label_Consolidation"] == 1) | (df["label_Lung Opacity"] == 1)).astype(int)
    return None


def any_abnormal(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.startswith("label_")
            and c not in ("label_Support Devices", "label_No Finding")]
    return (df[cols] == 1).any(axis=1).astype(int)


def band_table(df: pd.DataFrame, score_col: str, low: float, high: float) -> pd.DataFrame:
    """What each risk band actually contained on the reference set, with 95% CIs."""
    bands = {"High": df[score_col] >= high,
             "Medium": (df[score_col] >= low) & (df[score_col] < high),
             "Low": df[score_col] < low}
    outcomes = {"Consolidation": (df["label_Consolidation"] == 1).astype(int) if "label_Consolidation" in df else None,
                "Consolidation or opacity": proxy_label(df),
                "Any abnormal finding": any_abnormal(df)}
    rows = []
    for band, m in bands.items():
        row = {"band": band, "n": int(m.sum())}
        for name, y in outcomes.items():
            if y is None:
                continue
            p, lo, hi = wilson(int(y[m].sum()), int(m.sum()))
            row[name] = (p, lo, hi)
        rows.append(row)
    return pd.DataFrame(rows).set_index("band")


def fmt_ci(p, lo, hi, pct: bool = True) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if pct:
        return f"{p:.0%} ({lo:.0%}–{hi:.0%})"
    return f"{p:.2f} ({lo:.2f}–{hi:.2f})"
