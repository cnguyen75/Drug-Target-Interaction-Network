#!/usr/bin/env python3
"""Does the whole-drug holdout result come from near-duplicate drugs?

    python scripts/analyze_cold_holdout.py

DrugBank lists salts and alternative forms of one molecule as separate
entries, so a "new" test drug can have a structural twin in the training
graph. This re-creates the whole-drug holdout exactly as
train_attribute_model.py does (same seed, so the same split, negatives and
models), then scores the test pairs in bins of how similar each test drug is
to its nearest drug in the message graph (max Tanimoto, Morgan radius 2,
2048 bits). The bins are fixed in advance: identical (>= 0.999), close
(0.7-0.999), moderate (0.4-0.7), distant (< 0.4), no structure.

Writes results/results_cold_similarity.json.
"""

import json
import os
from pathlib import Path
import sys


if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd
from rdkit import DataStructs


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
os.chdir(ROOT)
from config import DRUGS_FP, DTI_FP, TARGETS_FP  # noqa: E402
from datasources.drugbank_attributes import DRUG_ATTR_FP, TARGET_ATTR_FP  # noqa: E402
from pipeline.attributes import AttributeLinkPredictor  # noqa: E402
from pipeline.embeddings import embed_graph  # noqa: E402
from pipeline.link_prediction import DTILinkPredictor, _ranking_metrics, bootstrap_ci  # noqa: E402


SEED = 42
BINS = [
    ("identical (>= 0.999)", 0.999, 1.01),
    ("close (0.7 to 0.999)", 0.7, 0.999),
    ("moderate (0.4 to 0.7)", 0.4, 0.7),
    ("distant (< 0.4)", -0.01, 0.4),
]


def main():
    tables = (pd.read_parquet(DRUGS_FP), pd.read_parquet(TARGETS_FP), pd.read_parquet(DTI_FP))
    da, ta = pd.read_parquet(DRUG_ATTR_FP), pd.read_parquet(TARGET_ATTR_FP)
    graph = DTILinkPredictor(seed=SEED, train_negatives="degree_matched")
    attr = AttributeLinkPredictor(da, ta, seed=SEED)
    for p in (graph, attr):
        p.load_data(*tables)
        p.split_edges("cold_drug")
    emb = embed_graph(graph.train_graph, seed=SEED)
    for p in (graph, attr):
        p.set_embeddings(emb)
        p.fit()
        p.validate()
    for p in (graph, attr):
        p.evaluate_test()

    pairs = attr.evaluation_results["test_pairs"]
    y = np.asarray(attr.evaluation_results["test_labels"])
    scores = {
        "graph_model": np.asarray(graph.evaluation_results["test_scores"]),
        "attribute_model": np.asarray(attr.evaluation_results["test_scores"]),
        "structural_nearest_neighbour": np.array(
            [attr.attribute_features(d, t)[0] for d, t in pairs]
        ),
    }

    seen = [d for d in attr.drug_nodes if attr.train_graph.degree(d) > 0 and d in attr.fps]
    seen_fps = [attr.fps[d] for d in seen]
    nearest = {}
    for d in set(attr.test_drugs):
        nearest[d] = (
            max(DataStructs.BulkTanimotoSimilarity(attr.fps[d], seen_fps))
            if d in attr.fps
            else None
        )

    def summarise(mask, label):
        n_pos = int(y[mask].sum())
        drugs = len({pairs[i][0] for i in np.flatnonzero(mask)})
        row = {
            "bin": label,
            "test_drugs": drugs,
            "test_positives": n_pos,
            "test_negatives": int(mask.sum() - n_pos),
        }
        if n_pos and mask.sum() - n_pos:
            for name, s in scores.items():
                roc, ap = _ranking_metrics(y[mask], s[mask])
                ci = bootstrap_ci(y[mask], s[mask], seed=SEED)
                row[name] = {
                    "roc_auc": round(float(roc), 4),
                    "roc_auc_ci95": [round(v, 4) for v in ci["roc_auc_ci95"]],
                }
        return row

    sim = np.array([nearest[d] if nearest[d] is not None else np.nan for d, _ in pairs])
    rows = [summarise(np.ones(len(y), bool), "all test drugs")]
    for label, lo, hi in BINS:
        rows.append(summarise((sim >= lo) & (sim < hi), label))
    rows.append(summarise(np.isnan(sim), "no structure"))
    rows.append(summarise(~np.isnan(sim) & (sim < 0.999), "all except identical"))

    out = {
        "command": "python scripts/analyze_cold_holdout.py",
        "similarity": "max Tanimoto of the test drug to any drug with an edge in the message graph",
        "rows": rows,
    }
    Path("results/results_cold_similarity.json").write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        print(
            r["bin"],
            r["test_drugs"],
            r["test_positives"],
            {k: r[k]["roc_auc"] for k in scores if k in r},
        )


if __name__ == "__main__":
    main()
