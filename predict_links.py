#!/usr/bin/env python3
"""Rank unrecorded targets for every featured drug with the trained model.

    python predict_links.py

Featured drugs are approved small molecules excluding nutraceuticals, salts and
inorganic entries (src/pipeline/explore.py). For each featured drug with at
least one known human target, every unlinked target is scored and the top 10
are kept with a one-line evidence note. Writes models/predictions.parquet,
which is derived from DrugBank and therefore gitignored.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))
from pipeline.explore import evidence, featured_mask
from pipeline.link_prediction import DTILinkPredictor

TOP_K = 10


def support_count(G, drug, target):
    """Drugs that bind the target and share at least one target with the drug."""
    mine = set(G.neighbors(drug))
    return sum(1 for o in G.neighbors(target) if o != drug and mine & set(G.neighbors(o)))


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
    rows = []
    for drug in sorted(d for d in drugs["drug_id"] if d in p.G and p.G.degree(d) > 0):
        scores = p.score_candidates(drug)
        top = sorted(scores.items(), key=lambda x: (-x[1], -support_count(p.G, drug, x[0]), x[0]))
        for rank, (target, score) in enumerate(top[:TOP_K], 1):
            rows.append({"drug_id": drug, "target_id": target, "rank": rank,
                         "score": float(score), "support": support_count(p.G, drug, target),
                         "evidence": evidence(p.G, drug, target, emb, label=label)})
    out = pd.DataFrame(rows)
    out.to_parquet("models/predictions.parquet", index=False)
    print(f"{out['drug_id'].nunique()} drugs, {len(out)} predicted links -> models/predictions.parquet")


if __name__ == "__main__":
    main()
