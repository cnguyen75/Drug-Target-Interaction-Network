#!/usr/bin/env python3
"""Rank unrecorded targets for every featured drug with the app's model (v3).

    python predict_links.py

Featured drugs are approved small molecules excluding nutraceuticals, salts and
inorganic entries (src/pipeline/explore.py). Drugs with at least one known
human target are scored by the 'seen' copy of the attribute model; featured
drugs with no known target by the 'cold' copy. Every unlinked target is
scored with the calibrated model and ranked by score, then target
specificity (a target with fewer known drugs first).

Writes models/predictions.parquet (top 10 per drug; derived from DrugBank, so
gitignored) and an "app_suggestions" block in results/results_attributes.json:
for each copy, how many candidates clear each threshold for the median drug,
and the default threshold, chosen as the lowest of THRESHOLDS that leaves the
median drug with at most SHORTLIST candidates, or the highest threshold when
none does (recorded as rule_met: false).
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.append(str(Path(__file__).parent / "src"))
from pipeline.explore import evidence, featured_mask, rank_key
from pipeline.link_prediction import DTILinkPredictor


TOP_K = 10
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SHORTLIST = 10
RESULTS_FP = Path("results/results_attributes.json")


def main() -> None:
    copies = {}
    for name in ("seen", "cold"):
        fp = Path(f"models/attribute_model_{name}.pkl")
        if not fp.exists():
            sys.exit(f"Missing {fp}. Run: python train_attribute_model.py")
        copies[name] = DTILinkPredictor.load_model(fp)
    seen = copies["seen"]
    G = seen.G
    genes = seen.targets_df.set_index("target_id")["gene_symbol"].to_dict()

    def label(n: str) -> str:
        return genes.get(n) or G.nodes[n]["name"]

    featured = seen.drugs_df[featured_mask(seen.drugs_df)]["drug_id"]
    groups = {
        "seen": sorted(d for d in featured if G.degree(d) > 0),
        "cold": sorted(d for d in featured if G.degree(d) == 0),
    }

    rows, block = [], {}
    for name, drugs in groups.items():
        model = copies[name]
        counts = {t: [] for t in THRESHOLDS}
        for i, drug in enumerate(drugs, 1):
            scores = model.score_candidates(drug)
            values = np.fromiter(scores.values(), dtype=float)
            for t in THRESHOLDS:
                counts[t].append(int((values >= t).sum()))
            top = sorted(scores.items(), key=lambda x: rank_key(G, x))[:TOP_K]
            for rank, (target, score) in enumerate(top, 1):
                rows.append(
                    {
                        "drug_id": drug,
                        "target_id": target,
                        "rank": rank,
                        "copy": name,
                        "score": float(score),
                        "target_known_drugs": G.degree(target),
                        "evidence": evidence(G, drug, target, label=label, attr=model),
                    }
                )
            if i % 250 == 0:
                print(f"[{name}] {i}/{len(drugs)}")
        medians = {t: float(np.median(c)) for t, c in counts.items()}
        default = next((t for t in THRESHOLDS if medians[t] <= SHORTLIST), THRESHOLDS[-1])
        block[name] = {
            "drugs_scored": len(drugs),
            "median_candidates_at_or_above": {f"{t:.2f}": medians[t] for t in THRESHOLDS},
            "default_threshold": default,
            "rule_met": medians[default] <= SHORTLIST,
        }

    out = pd.DataFrame(rows)
    out.to_parquet("models/predictions.parquet", index=False)
    block.update(
        {
            "command": "python predict_links.py",
            "default_threshold_rule": (
                f"for each copy, the lowest of "
                f"{', '.join(f'{t:.2f}' for t in THRESHOLDS)} that leaves the "
                f"median featured drug with at most {SHORTLIST} candidates"
            ),
            "ranking": "calibrated score, then fewer known drugs for the target (rarer first)",
        }
    )
    results = json.loads(RESULTS_FP.read_text())
    results["app_suggestions"] = block
    RESULTS_FP.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"{out['drug_id'].nunique()} drugs, {len(out)} predicted links -> models/predictions.parquet"
    )
    print(json.dumps(block, indent=2))


if __name__ == "__main__":
    main()
