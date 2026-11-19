"""Deterministic plain-language summary (AICare panel C, without an LLM).

Every sentence is built only from numbers shown elsewhere on the screen, so
nothing here can cite evidence the model did not use.
"""
from __future__ import annotations

from .model import friendly
from .regions import OUTSIDE, describe_zone


def risk_band(p: float, low: float, high: float) -> str:
    return "High" if p >= high else ("Medium" if p >= low else "Low")


GUIDANCE = {
    "High": "Image findings are consistent with pneumonia. Correlate clinically "
            "(symptoms, SpO₂, temperature, WBC/CRP) and review the film yourself; "
            "consider discussing with the treating physician / radiology.",
    "Medium": "Equivocal image evidence. Review the film and the clinical picture; "
              "consider a lateral view, a repeat film or comparison with priors.",
    "Low": "Little image evidence of pneumonia. This does **not** rule out early or "
           "atypical pneumonia; clinical assessment takes precedence.",
}


def build(target: str, prob: float, band: str, zones: dict[str, float],
          related: dict[str, float], method_label: str,
          outside_unusual: bool = False) -> list[str]:
    lines = [f"The model rates the likelihood of **{friendly(target).lower()}** on this film as "
             f"**{band}** (score {prob:.2f}; 0.50 = model's operating threshold)."]

    pos = sorted(((z, v) for z, v in zones.items() if v > 0.05 and z != OUTSIDE), key=lambda kv: -kv[1])
    if pos:
        top = " and ".join(f"{describe_zone(z)} ({v:.0%})" for z, v in pos[:2])
        lines.append(f"Within the lungs, {method_label} places the most supporting evidence in the {top}.")
    else:
        lines.append(f"{method_label} finds no lung zone with a clear share of the evidence.")
    if outside_unusual:
        lines.append(f"An unusually large share ({zones.get(OUTSIDE, 0):.0%}) of the attention is "
                     "**outside the lung fields**; check for devices or labels before trusting the score.")
    neg = [(z, v) for z, v in zones.items() if v < -0.05]
    if neg:
        z, v = min(neg, key=lambda kv: kv[1])
        lines.append(f"The {describe_zone(z)} pushes the score down ({v:.0%}).")
    if band == "Low":
        lines.append("At low scores the heatmap shows the strongest of *weak* evidence; "
                     "diffuse or off-lung patterns are expected and not meaningful.")

    above = [f"{friendly(k).lower()} ({v:.2f})" for k, v in related.items() if v >= 0.5]
    if above:
        lines.append("Related findings above the model's threshold: " + ", ".join(above) + ".")
    else:
        lines.append("No related findings (consolidation, opacity, effusion, …) are above "
                     "the model's threshold.")
    return lines
