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
from pipeline.explore import featured_mask  # noqa: E402
from pipeline.link_prediction import DTILinkPredictor  # noqa: E402

COMMAND = "python scripts/site_results.py OUT.json"


def main():
    out = Path(sys.argv[1])
    results = json.loads((ROOT / "results" / "results.json").read_text())
    p = DTILinkPredictor.load_model(ROOT / "models" / "link_prediction_model.pkl")
    G = p.G
    dti = pd.read_parquet(DTI_FP)
    target_deg = dti["target_id"].value_counts()
    threshold = results["app_suggestions"]["default_threshold"]

    featured = sorted(d for d in p.drugs_df[featured_mask(p.drugs_df)]["drug_id"] if G.degree(d) > 0)
    at_threshold = np.array([int((np.fromiter(p.score_candidates(d).values(), float) >= threshold).sum())
                             for d in featured])
    preds = pd.read_parquet(ROOT / "models" / "predictions.parquet")
    top1 = preds[preds["rank"] == 1]["target_id"].value_counts()
    ties = preds[preds["rank"] <= 5].groupby("drug_id")["score"].apply(
        lambda s: s.round(2).duplicated().any())

    checks = {
        "target_popularity": {
            "command": COMMAND,
            "median_known_drugs_of_target_in_a_known_interaction": float(
                dti["target_id"].map(target_deg).median()),
            "median_known_drugs_of_a_target_drawn_uniformly": float(target_deg.median()),
            "share_of_targets_with_at_most_2_known_drugs": round(float((target_deg <= 2).mean()), 3),
            "note": "why uniform random negatives reward preferring popular targets",
        },
        "app_suggestions_spread": {
            "command": COMMAND,
            "threshold": threshold,
            "featured_drugs_with_a_known_target": len(featured),
            "drugs_with_no_candidate_at_threshold": int((at_threshold == 0).sum()),
            "drugs_with_at_most_10": int((at_threshold <= 10).sum()),
            "drugs_with_100_or_more": int((at_threshold >= 100).sum()),
            "max_candidates_for_one_drug": int(at_threshold.max()),
            "p90_candidates": float(np.percentile(at_threshold, 90)),
            "most_common_top_suggestion_drug_count": int(top1.iloc[0]),
            "share_of_drugs_with_tied_scores_in_top_5": round(float(ties.mean()), 3),
            "distinct_calibrated_score_levels_in_top_10s": int(preds["score"].round(4).nunique()),
        },
    }
    site = {
        "about": ("Every quotable number for the drug-target track. 'repo_results' is "
                  "results/results.json from the repository (commands inside); 'checks' are computed "
                  f"by `{COMMAND}`."),
        "repo_results": results,
        "checks": checks,
    }
    out.write_text(json.dumps(site, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
