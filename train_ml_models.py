#!/usr/bin/env python3
"""Train and evaluate the link predictor; train the models app.py loads.

    python src/datasources/drugbank_loader.py     # parse DrugBank (see DRUGBANK_PATH)
    python train_ml_models.py                     # this script

For each evaluation split ("random" and the stricter "drug_holdout"; see
src/pipeline/link_prediction.py), Node2Vec is trained on that split's message
graph only, then the random forest is fitted and evaluated with baselines and
bootstrap confidence intervals. A final model ("final" split: all edges, no
test set) is trained for the app's predictions.

Writes (all gitignored except results/results.json):
    results/results.json                  every reported number, with the commands
    models/link_prediction_model.pkl      final DTILinkPredictor, used by the app
    models/evaluation_results.pkl         per-split metrics, baselines, calibration
    models/drug_repurposing_model.pkl     DrugRepurposer on full-graph embeddings
    data/embeddings_full.parquet          Node2Vec on the full graph (similarity, layout)
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    # gensim seeds its initial vectors with hash(); fix it so runs are reproducible.
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))
from config import DRUGS_FP, DTI_FP, RAW_XML, TARGETS_FP
from pipeline.build_graph import build_bipartite_graph, graph_stats
from pipeline.drug_repurposing import DrugRepurposer
from pipeline.embeddings import embed_graph
from pipeline.link_prediction import DTILinkPredictor

SEED = 42
COMMAND = "python train_ml_models.py"
DATA_COMMAND = "python src/datasources/drugbank_loader.py"


def r4(x):
    return round(float(x), 4)


def metric_block(m):
    return {
        "roc_auc": r4(m["roc_auc"]),
        "roc_auc_ci95": [r4(v) for v in m["roc_auc_ci95"]],
        "average_precision": r4(m["ap_score"]),
        "average_precision_ci95": [r4(v) for v in m["ap_ci95"]],
    }


def run_split(kind, tables):
    t0 = time.time()
    p = DTILinkPredictor(seed=SEED)
    p.load_data(*tables)
    p.split_edges(kind)
    print(f"[{kind}] Node2Vec on the message graph...")
    p.set_embeddings(embed_graph(p.train_graph, seed=SEED))
    print(f"[{kind}] training random forest...")
    p.train_model()
    print(f"[{kind}] done in {time.time() - t0:.0f}s")
    return p


def split_summary(p):
    ev = p.evaluation_results
    mp = ev["model_performance"]
    out = {
        "description": mp["split_description"],
        "counts": {k: mp[k] for k in ("message_edges", "train_positives", "train_negatives",
                                      "test_positives", "test_negatives", "test_drugs")
                   if k in mp},
        "random_forest": {
            **metric_block({"roc_auc": mp["auc"], "roc_auc_ci95": mp["roc_auc_ci95"],
                            "ap_score": mp["ap_score"], "ap_ci95": mp["ap_ci95"]}),
            "pr_auc_trapezoidal": r4(mp["pr_auc"]),
            "brier_score": r4(mp["brier_score"]),
        },
        "baselines": {name: metric_block({"roc_auc": b["roc_auc"], "roc_auc_ci95": b["roc_auc_ci95"],
                                          "ap_score": b["ap_score"], "ap_ci95": b["ap_ci95"]})
                      for name, b in ev["baselines"].items()},
        "feature_importances": {k: r4(v) for k, v in ev["feature_importances"].items()},
        "bootstrap_resamples": mp["n_boot"],
    }
    return out


def main():
    for fp in (DRUGS_FP, TARGETS_FP, DTI_FP):
        if not fp.exists():
            sys.exit(f"Missing {fp}. Run: {DATA_COMMAND}")
    for d in ("models", "data", "results"):
        Path(d).mkdir(exist_ok=True)

    t0 = time.time()
    tables = (pd.read_parquet(DRUGS_FP), pd.read_parquet(TARGETS_FP), pd.read_parquet(DTI_FP))
    drugs_df, targets_df, dti_df = tables
    stats = graph_stats(build_bipartite_graph(*tables))

    splits = {kind: run_split(kind, tables) for kind in ("random", "drug_holdout")}
    pd.to_pickle({kind: {k: v for k, v in p.evaluation_results.items()
                         if k not in ("test_pairs",)}
                  for kind, p in splits.items()}, "models/evaluation_results.pkl")

    print("[final] training the app's model on all edges...")
    final = DTILinkPredictor(seed=SEED)
    final.load_data(*tables)
    final.split_edges("final")
    final.set_embeddings(embed_graph(final.train_graph, seed=SEED))
    final.train_model()
    final.save_model("models/link_prediction_model.pkl")

    print("Node2Vec on the full graph (similarity and layout)...")
    emb_full = embed_graph(build_bipartite_graph(*tables), seed=SEED)
    emb_full.to_parquet("data/embeddings_full.parquet")
    repurposer = DrugRepurposer()
    repurposer.load_data(drugs_df, targets_df, dti_df, emb_full)
    pd.to_pickle(repurposer, "models/drug_repurposing_model.pkl")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    results = {
        "commands": [DATA_COMMAND, COMMAND],
        "git_commit": commit,
        "python": platform.python_version(),
        "seed": SEED,
        "runtime_seconds": round(time.time() - t0, 1),
        "source": {
            "file": RAW_XML.name,
            "note": "DrugBank full database export; licensed data, not in the repo",
        },
        "dataset": {
            "drugs": len(drugs_df),
            "drugs_with_human_target": stats["drugs_with_edges"],
            "human_targets": stats["targets"],
            "interactions": stats["edges"],
            "approved_drugs": int(drugs_df["approved"].sum()),
        },
        "model": "RandomForestClassifier(n_estimators=100) on 12 graph and Node2Vec features",
        "metrics_note": ("ROC AUC and average precision on held-out pairs (1:1 positives to "
                         "negatives); 95% percentile bootstrap intervals over test pairs. "
                         "Chance and common neighbours tie every pair, so both score 0.5 by "
                         "construction."),
        "splits": {kind: split_summary(p) for kind, p in splits.items()},
    }
    Path("results/results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
