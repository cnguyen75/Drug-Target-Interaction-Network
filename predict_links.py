#!/usr/bin/env python3
"""Rank unrecorded targets for every featured drug with the app's model.

    python predict_links.py

Featured drugs are approved small molecules excluding nutraceuticals, salts and
inorganic entries (src/pipeline/explore.py). For each featured drug with at
least one known human target, every unlinked target is scored with the
calibrated model. Ranking is by score, then target specificity: a target with
fewer known drugs ranks above a very common one at the same score.

Writes models/predictions.parquet (top 10 per drug; derived from DrugBank, so
gitignored) and adds an "app_suggestions" block to results/results.json: how
many candidates clear each threshold for the median featured drug, and the
default threshold, chosen as the lowest of THRESHOLDS that leaves the median
drug with at most SHORTLIST candidates.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))
from pipeline.explore import evidence, featured_mask, rank_key
from pipeline.link_prediction import DTILinkPredictor

TOP_K = 10
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SHORTLIST = 10
RESULTS_FP = Path("results/results.json")


def main():
    model_fp = Path("models/link_prediction_model.pkl")
    if not model_fp.exists():
        sys.exit("Missing models/link_prediction_model.pkl. Run: python train_ml_models.py")
    p = DTILinkPredictor.load_model(model_fp)
    emb = pd.read_parquet("data/embeddings_full.parquet")
    genes = p.targets_df.set_index("target_id")["gene_symbol"].to_dict()

    def label(n):
        return genes.get(n) or p.G.nodes[n]["name"]

    drugs = p.drugs_df[featured_mask(p.drugs_df)]
    rows, counts = [], {t: [] for t in THRESHOLDS}
    for drug in sorted(d for d in drugs["drug_id"] if d in p.G and p.G.degree(d) > 0):
        scores = p.score_candidates(drug)
        values = np.fromiter(scores.values(), dtype=float)
        for t in THRESHOLDS:
            counts[t].append(int((values >= t).sum()))
        top = sorted(scores.items(), key=lambda x: rank_key(p.G, x))[:TOP_K]
        for rank, (target, score) in enumerate(top, 1):
            rows.append({"drug_id": drug, "target_id": target, "rank": rank,
                         "score": float(score), "target_known_drugs": p.G.degree(target),
                         "evidence": evidence(p.G, drug, target, emb, label=label)})
    out = pd.DataFrame(rows)
    out.to_parquet("models/predictions.parquet", index=False)

    medians = {t: float(np.median(c)) for t, c in counts.items()}
    default = next((t for t in THRESHOLDS if medians[t] <= SHORTLIST), THRESHOLDS[-1])
    block = {
        "command": "python predict_links.py",
        "featured_drugs_scored": int(out["drug_id"].nunique()),
        "median_candidates_at_or_above": {f"{t:.2f}": medians[t] for t in THRESHOLDS},
        "default_threshold": default,
        "default_threshold_rule": (f"lowest of {', '.join(f'{t:.2f}' for t in THRESHOLDS)} that "
                                   f"leaves the median featured drug with at most {SHORTLIST} "
                                   "candidates"),
        "ranking": "calibrated score, then fewer known drugs for the target (rarer first)",
    }
    results = json.loads(RESULTS_FP.read_text())
    results["app_suggestions"] = block
    RESULTS_FP.write_text(json.dumps(results, indent=2) + "\n")
    print(f"{out['drug_id'].nunique()} drugs, {len(out)} predicted links -> models/predictions.parquet")
    print(json.dumps(block, indent=2))


if __name__ == "__main__":
    main()
