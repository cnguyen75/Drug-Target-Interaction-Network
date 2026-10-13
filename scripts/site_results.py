#!/usr/bin/env python3
"""Write a site-facing results file: results/results.json plus extra checks.

    python scripts/site_results.py OUT.json

Every number here is computed by this script or copied from
results/results.json (produced by train_ml_models.py and predict_links.py).
Nothing from DrugBank is copied row for row; only counts and medians.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from config import DTI_FP  # noqa: E402
from pipeline.explore import NO_EVIDENCE  # noqa: E402
from pipeline.link_prediction import DTILinkPredictor  # noqa: E402

COMMAND = "python scripts/site_results.py OUT.json"


def spread(model, drugs, threshold, preds):
    G = model.G
    at = np.array([int((np.fromiter(model.score_candidates(d).values(), float) >= threshold).sum())
                   for d in drugs])
    top1 = preds[preds["rank"] == 1]
    isolated = {t for t in model.target_nodes if G.degree(t) > 0 and model.train_graph.degree(t) == 0}
    ties = preds[preds["rank"] <= 5].groupby("drug_id")["score"].apply(
        lambda s: s.round(2).duplicated().any())
    return {
        "command": COMMAND,
        "threshold": threshold,
        "drugs": len(drugs),
        "drugs_with_no_candidate_at_threshold": int((at == 0).sum()),
        "drugs_with_at_most_10": int((at <= 10).sum()),
        "drugs_with_100_or_more": int((at >= 100).sum()),
        "max_candidates_for_one_drug": int(at.max()),
        "p90_candidates": float(np.percentile(at, 90)),
        "most_common_top_suggestion_drug_count": int(top1["target_id"].value_counts().iloc[0]),
        "share_of_top_suggestions_with_no_evidence": round(float(
            (top1["evidence"] == NO_EVIDENCE).mean()), 3),
        "share_of_top_suggestions_on_targets_isolated_in_training": round(float(
            top1["target_id"].isin(isolated).mean()), 3),
        "share_of_drugs_with_tied_scores_in_top_5": round(float(ties.mean()), 3),
    }


def main():
    out = Path(sys.argv[1])
    res = {name: json.loads((ROOT / "results" / f"{name}.json").read_text())
           for name in ("results", "results_attributes", "results_cold_similarity",
                        "results_attribute_ablation")}
    v3 = res["results_attributes"]
    dti = pd.read_parquet(DTI_FP)
    target_deg = dti["target_id"].value_counts()
    preds = pd.read_parquet(ROOT / "models" / "predictions.parquet")

    checks = {
        "target_popularity": {
            "command": COMMAND,
            "median_known_drugs_of_target_in_a_known_interaction": float(
                dti["target_id"].map(target_deg).median()),
            "median_known_drugs_of_a_target_drawn_uniformly": float(target_deg.median()),
            "share_of_targets_with_at_most_2_known_drugs": round(float((target_deg <= 2).mean()), 3),
            "note": "why uniform random negatives reward preferring popular targets",
        },
    }
    for name in ("seen", "cold"):
        model = DTILinkPredictor.load_model(ROOT / "models" / f"attribute_model_{name}.pkl")
        drugs = sorted(preds[preds["copy"] == name]["drug_id"].unique())
        checks[f"app_suggestions_spread_{name}"] = spread(
            model, drugs, v3["app_suggestions"][name]["default_threshold"],
            preds[preds["copy"] == name])
    site = {
        "about": ("Every quotable number for the drug-target track. 'repo_results' holds the "
                  "repository's committed results files (commands inside each); 'checks' are "
                  f"computed by `{COMMAND}`. The app's model is v3 (results_attributes); "
                  "results.json is the v2 graph-only model."),
        "repo_results": res,
        "checks": checks,
    }
    out.write_text(json.dumps(site, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
