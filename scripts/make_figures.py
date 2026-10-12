#!/usr/bin/env python3
"""Regenerate figures from results/results.json in the site style.

    python scripts/make_figures.py            # writes figures/evaluation.svg

Hand-written SVG: paper card #F6F2E8, ink #231F1B, secondary #3D372F, rule
#B3A992; series blue #1F6FB5 (the app's model), purple #7A4E9E (the other
model), baselines in #8A8171; Archivo for text. Direct labels, no legend.
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
CARD, INK, SECONDARY, RULE = "#F6F2E8", "#231F1B", "#3D372F", "#B3A992"
BLUE, PURPLE, BASE = "#1F6FB5", "#7A4E9E", "#8A8171"
SPLITS = {"random": "Random edge split", "drug_holdout": "Per-drug split (controls for target popularity)"}
X0, X1 = 0.4, 1.0  # ROC AUC axis range


def rows(split, app):
    other = "uniform" if app == "degree_matched" else "degree_matched"
    names = {"degree_matched": "Random forest, degree-matched negatives",
             "uniform": "Random forest, uniform negatives"}
    out = [(names[app] + " (app)", split["models"][app], BLUE, True),
           (names[other], split["models"][other], PURPLE, True)]
    for key, name in (("l3_paths", "Length-3 paths"),
                      ("preferential_attachment", "Preferential attachment"),
                      ("chance", "Chance")):
        out.append((name, split["baselines"][key], BASE, False))
    return out


def evaluation_svg(results):
    app = results["app_model"]["train_negatives"]
    W, label_w, plot_w, value_w = 920, 330, 440, 130
    row_h, panel_gap, top = 30, 46, 74
    panels = list(SPLITS)
    n_rows = len(rows(results["splits"][panels[0]], app))
    panel_h = 34 + n_rows * row_h
    H = top + len(panels) * panel_h + (len(panels) - 1) * panel_gap + 64

    def x(v):
        return label_w + (v - X0) / (X1 - X0) * plot_w

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'role="img" aria-labelledby="t d" font-family="Archivo, sans-serif">',
         '<title id="t">ROC AUC on held-out drug-target pairs, by split and method</title>',
         f'<desc id="d">Dots are ROC AUC on held-out pairs; lines are 95% bootstrap intervals. '
         f'Source: results.json, commit {escape(results["git_commit"])}.</desc>',
         f'<rect width="{W}" height="{H}" fill="{CARD}"/>',
         f'<text x="24" y="34" font-size="20" font-weight="700" fill="{INK}">How well does the model '
         f'rank held-out interactions?</text>',
         f'<text x="24" y="56" font-size="14" fill="{SECONDARY}">ROC AUC on held-out pairs (1 known '
         f'interaction : 1 sampled non-interaction). Lines: 95% bootstrap interval.</text>']
    y = top
    for kind in panels:
        split = results["splits"][kind]
        p.append(f'<text x="24" y="{y + 18}" font-size="15" font-weight="700" fill="{INK}">'
                 f'{escape(SPLITS[kind])}</text>')
        y0 = y + 34
        for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            dash = "" if tick == 0.5 else ' stroke-dasharray="3 3"'
            p.append(f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{y0 - 6}" '
                     f'y2="{y0 + n_rows * row_h - 6}" stroke="{RULE}" stroke-width="1"{dash}/>')
        for i, (name, b, color, model) in enumerate(rows(split, app)):
            cy = y0 + i * row_h + 8
            lo, hi = b["roc_auc_ci95"]
            weight = "600" if model else "400"
            p.append(f'<text x="24" y="{cy + 5}" font-size="14" font-weight="{weight}" '
                     f'fill="{INK}">{escape(name)}</text>')
            p.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{cy}" y2="{cy}" stroke="{color}" '
                     f'stroke-width="3" stroke-linecap="round"/>')
            p.append(f'<circle cx="{x(b["roc_auc"]):.1f}" cy="{cy}" r="6" fill="{color}"/>')
            p.append(f'<text x="{label_w + plot_w + 16}" y="{cy + 5}" font-size="14" fill="{INK}">'
                     f'<tspan font-weight="700">{b["roc_auc"]:.3f}</tspan>'
                     f'<tspan fill="{SECONDARY}" font-size="12"> {lo:.3f}–{hi:.3f}</tspan></text>')
        y = y0 + n_rows * row_h + panel_gap - 34
    axis_y = y + 34 - panel_gap + 12
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        p.append(f'<text x="{x(tick):.1f}" y="{axis_y}" font-size="12" text-anchor="middle" '
                 f'fill="{SECONDARY}">{tick:.1f}</text>')
    p.append(f'<text x="{label_w + plot_w / 2:.1f}" y="{axis_y + 22}" font-size="13" '
             f'text-anchor="middle" fill="{SECONDARY}">ROC AUC (0.5 = chance, 1.0 = perfect ranking)'
             f'</text>')
    p.append("</svg>")
    return "\n".join(p) + "\n"


def main():
    results = json.loads((ROOT / "results" / "results.json").read_text())
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    (out / "evaluation.svg").write_text(evaluation_svg(results))
    print(f"wrote {out / 'evaluation.svg'}")


if __name__ == "__main__":
    main()
