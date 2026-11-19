"""Deterministic plain-language text (AICare panel C, without an LLM).

Every sentence is built only from numbers shown elsewhere on the screen, so
nothing here can cite evidence the model did not use. Wording is deliberately
limited to what was measured on the reference set:
  * the "pneumonia" output behaves like a general airspace-abnormality score
    (it ranks oedema, effusion and atelectasis almost as well as consolidation),
    so no sentence claims the film shows pneumonia;
  * Low does not rule anything out.
"""
from __future__ import annotations

from .model import friendly
from .regions import OUTSIDE, describe_zone


def risk_band(p: float, low: float, high: float) -> str:
    return "High" if p >= high else ("Medium" if p >= low else "Low")


HEADLINE = {
    "High": "High pneumonia score",
    "Medium": "Medium pneumonia score",
    "Low": "Low pneumonia score",
}

GUIDANCE = {
    "High": "The film likely shows abnormal lung findings. This score is **not specific to pneumonia**: "
            "it rises just as much with oedema, effusion and atelectasis. Review the film yourself "
            "and correlate clinically (symptoms, SpO₂, temperature, WBC/CRP).",
    "Medium": "Mixed image evidence. Many films in this band have some abnormality. Review the film and "
              "the clinical picture; consider a lateral view, a repeat film or comparison with priors.",
    "Low": "Less image evidence of airspace disease. This does **not** rule out pneumonia: early, "
           "atypical or retrocardiac disease is easy to miss, and clinical assessment takes precedence.",
}


def build(target: str, prob: float, band: str, zones: dict[str, float],
          related: dict[str, float], method_label: str,
          outside_unusual: bool = False, signed: bool = True) -> list[str]:
    lines = [f"The model's **{friendly(target).lower()}** score for this film is **{prob:.2f}** "
             f"(**{band}** band; 0.50 = the model's operating threshold, not a probability)."]

    pos = sorted(((z, v) for z, v in zones.items() if v > 0.05 and z != OUTSIDE), key=lambda kv: -kv[1])
    if pos:
        top = " and ".join(f"{describe_zone(z)} ({v:.0%})" for z, v in pos[:2])
        lines.append(f"Within the lungs, {method_label} puts the most weight on the {top}. "
                     "This shows where the model looked, not a confirmed lesion.")
    else:
        lines.append(f"{method_label} gives no lung zone a clear share of the evidence.")
    if outside_unusual:
        lines.append(f"An unusually large share ({zones.get(OUTSIDE, 0):.0%}) of the attention is "
                     "**outside the lung fields**; check for devices or labels before trusting the score.")
    if signed:
        neg = [(z, v) for z, v in zones.items() if v < -0.05]
        if neg:
            z, v = min(neg, key=lambda kv: kv[1])
            lines.append(f"The {describe_zone(z)} pushes the score down ({v:.0%}).")
    if band == "Low":
        lines.append("At low scores the heatmap shows the strongest of *weak* evidence; "
                     "diffuse or off-lung patterns are expected and not meaningful.")

    above = [f"{friendly(k).lower()} ({v:.2f})" for k, v in related.items() if v >= 0.5]
    if above:
        lines.append("Other findings scored above threshold, any of which can raise this score: "
                     + ", ".join(above) + ".")
    else:
        lines.append("No related finding (consolidation, opacity, effusion, …) is above the "
                     "model's threshold.")
    return lines
