#!/usr/bin/env python3
"""Regenerate figures from results/results.json in the site style.

    python scripts/make_figures.py            # writes figures/evaluation.svg and the
                                              # results block in README.md

Hand-written SVG: paper card #F6F2E8, ink #231F1B, secondary #3D372F, rule
#B3A992; series blue #1F6FB5 (the app's model), purple #7A4E9E (the other
model), baselines in #8A8171; Archivo for text. Direct labels, no legend.
"""

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent.parent
CARD, INK, SECONDARY, RULE = "#F6F2E8", "#231F1B", "#3D372F", "#B3A992"
BLUE, PURPLE, BASE = "#1F6FB5", "#7A4E9E", "#8A8171"
SPLITS = {
    "cold_drug": "Whole-drug holdout (unseen drugs)",
    "drug_holdout": "Per-drug split (controls for target popularity)",
    "random": "Random edge split",
}
X0, X1 = 0.4, 1.0  # ROC AUC axis range


def rows(split: dict[str, Any]) -> list[tuple[str, dict[str, Any], str, bool]]:
    out = [
        ("Attribute model (v3, the app)", split["attribute_model"], BLUE, True),
        ("Graph-only model (v2)", split["graph_model"], PURPLE, True),
    ]
    for key, name in (
        ("structural_nearest_neighbour", "Structural nearest neighbour"),
        ("l3_paths", "Length-3 paths"),
        ("preferential_attachment", "Preferential attachment"),
        ("chance", "Chance"),
    ):
        out.append((name, split["baselines"][key], BASE, False))
    return out


def evaluation_svg(results: dict[str, Any]) -> str:
    W, label_w, plot_w = 920, 330, 440
    row_h, panel_gap, top = 30, 46, 74
    panels = list(SPLITS)
    n_rows = len(rows(results["splits"][panels[0]]))
    panel_h = 34 + n_rows * row_h
    H = top + len(panels) * panel_h + (len(panels) - 1) * panel_gap + 64

    def x(v: float) -> float:
        return label_w + (v - X0) / (X1 - X0) * plot_w

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-labelledby="t d" font-family="Archivo, sans-serif">',
        '<title id="t">ROC AUC on held-out drug-target pairs, by split and method</title>',
        f'<desc id="d">Dots are ROC AUC on held-out pairs; lines are 95% bootstrap intervals. '
        f"Source: results.json, commit {escape(results['git_commit'])}.</desc>",
        f'<rect width="{W}" height="{H}" fill="{CARD}"/>',
        f'<text x="24" y="34" font-size="20" font-weight="700" fill="{INK}">'
        "How well does the model "
        f"rank held-out interactions?</text>",
        f'<text x="24" y="56" font-size="14" fill="{SECONDARY}">ROC AUC on held-out pairs (1 known '
        f"interaction : 1 sampled non-interaction). Lines: 95% bootstrap interval.</text>",
    ]
    y = top
    for kind in panels:
        split = results["splits"][kind]
        p.append(
            f'<text x="24" y="{y + 18}" font-size="15" font-weight="700" fill="{INK}">'
            f"{escape(SPLITS[kind])}</text>"
        )
        y0 = y + 34
        for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            dash = "" if tick == 0.5 else ' stroke-dasharray="3 3"'
            p.append(
                f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{y0 - 6}" '
                f'y2="{y0 + n_rows * row_h - 6}" stroke="{RULE}" stroke-width="1"{dash}/>'
            )
        for i, (name, b, color, model) in enumerate(rows(split)):
            cy = y0 + i * row_h + 8
            lo, hi = b["roc_auc_ci95"]
            weight = "600" if model else "400"
            p.append(
                f'<text x="24" y="{cy + 5}" font-size="14" font-weight="{weight}" '
                f'fill="{INK}">{escape(name)}</text>'
            )
            p.append(
                f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{cy}" y2="{cy}" stroke="{color}" '
                f'stroke-width="3" stroke-linecap="round"/>'
            )
            p.append(f'<circle cx="{x(b["roc_auc"]):.1f}" cy="{cy}" r="6" fill="{color}"/>')
            p.append(
                f'<text x="{label_w + plot_w + 16}" y="{cy + 5}" font-size="14" fill="{INK}">'
                f'<tspan font-weight="700">{b["roc_auc"]:.3f}</tspan>'
                f'<tspan fill="{SECONDARY}" font-size="12"> {lo:.3f}–{hi:.3f}</tspan></text>'
            )
        y = y0 + n_rows * row_h + panel_gap - 34
    axis_y = y + 34 - panel_gap + 12
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        p.append(
            f'<text x="{x(tick):.1f}" y="{axis_y}" font-size="12" text-anchor="middle" '
            f'fill="{SECONDARY}">{tick:.1f}</text>'
        )
    p.append(
        f'<text x="{label_w + plot_w / 2:.1f}" y="{axis_y + 22}" font-size="13" '
        f'text-anchor="middle" fill="{SECONDARY}">ROC AUC (0.5 = chance, 1.0 = perfect ranking)'
        f"</text>"
    )
    p.append("</svg>")
    return "\n".join(p) + "\n"


START, END = "<!-- results:start -->", "<!-- results:end -->"


def readme_block(results: dict[str, Any]) -> str:
    """Markdown results, every number read from the results files."""
    v2 = json.loads((ROOT / "results" / "results.json").read_text())
    abl = json.loads((ROOT / "results" / "results_attribute_ablation.json").read_text())
    sim = {
        r["bin"]: r
        for r in json.loads((ROOT / "results" / "results_cold_similarity.json").read_text())["rows"]
    }

    def cell(b: dict[str, Any]) -> str:
        lo, hi = b["roc_auc_ci95"]
        return f"{b['roc_auc']:.3f} ({lo:.3f}–{hi:.3f})"

    order = list(SPLITS)
    lines = [START]
    ds = v2["dataset"]
    lines += [
        f"DrugBank {v2['source']['version']} (exported {v2['source']['exported_on']}): "
        f"{ds['drugs_with_human_target']:,} drugs with a human target, {ds['human_targets']:,} "
        "human "
        f"targets, {ds['interactions']:,} interactions. Run {results.get('run_date', '')}, commit "
        f"`{results['git_commit']}`, seed {results['seed']}.",
        "",
        "ROC AUC on held-out pairs (one sampled non-interaction per known interaction), with 95% "
        "bootstrap intervals over test pairs:",
        "",
        "| Method | " + " | ".join(SPLITS[k] for k in order) + " |",
        "|---" * (len(order) + 1) + "|",
    ]
    for name, _, _, _ in rows(results["splits"][order[0]]):
        cells = [cell(b) for k in order for n, b, _, _ in rows(results["splits"][k]) if n == name]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    cold = results["splits"]["cold_drug"]["attribute_model"]
    no_cat = abl["variants"]["without categories and ATC"]["val_roc_auc"]
    lines += [
        "",
        f"On unseen drugs: {cold['roc_auc']:.2f} on the test set, "
        f"{cold['validation']['roc_auc']:.2f} on the smaller validation set, and about "
        f"{no_cat:.2f} (validation) without curated categories and ATC codes, the fairer "
        f"expectation for a genuinely new compound. {sim['identical (>= 0.999)']['test_drugs']} "
        f"of {sim['all test drugs']['test_drugs']:,} test drugs have a structurally identical "
        f"twin among training drugs; without them the score is "
        f"{sim['all except identical']['attribute_model']['roc_auc']:.2f}.",
    ]
    sug = results.get("app_suggestions")
    if sug:
        parts = []
        for k, who in (
            ("seen", "drugs with known targets"),
            ("cold", "drugs with no known targets"),
        ):
            t = sug[k]["default_threshold"]
            median = sug[k]["median_candidates_at_or_above"][f"{t:.2f}"]
            parts.append(f"{who} {t:.2f} (median {median:g} candidates)")
        lines += [
            "",
            "Default suggestion thresholds: "
            + "; ".join(parts)
            + f". Rule: {sug['default_threshold_rule']}.",
        ]
    lines += [
        "",
        "Produced by `python src/datasources/drugbank_loader.py`, "
        "`python src/datasources/drugbank_attributes.py`, `python train_attribute_model.py`, "
        "`python scripts/analyze_cold_holdout.py`, `python "
        "scripts/ablate_attributes_validation.py` "
        "and `python predict_links.py`; the v2 graph-only model by `python train_ml_models.py`; "
        "table and figure by `python scripts/make_figures.py`.",
        END,
    ]
    return "\n".join(lines)


def update_readme(results: dict[str, Any]) -> bool:
    fp = ROOT / "README.md"
    text = fp.read_text()
    if START not in text:
        return False
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    fp.write_text(head + readme_block(results) + tail)
    return True


def main() -> None:
    results = json.loads((ROOT / "results" / "results_attributes.json").read_text())
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    (out / "evaluation.svg").write_text(evaluation_svg(results))
    print(f"wrote {out / 'evaluation.svg'}")
    if update_readme(results):
        print("updated README.md results block")


if __name__ == "__main__":
    main()
