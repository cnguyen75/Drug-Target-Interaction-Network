#!/usr/bin/env python3
"""Train the link predictor and drug-repurposing model that app.py loads.

    PYTHONHASHSEED=0 python train_ml_models.py

Reads data_raw/*_full.parquet (from src/datasources/drugbank_loader.py) and writes:
    models/link_prediction_model.pkl     DTILinkPredictor (trained)
    models/drug_repurposing_model.pkl    DrugRepurposer
    models/evaluation_results.pkl        metrics, baselines, calibration (read by the app)
    data/embeddings_full.parquet         Node2Vec on the full graph (similarity, layout)
    data/embeddings_message.parquet      Node2Vec on the message graph (link-prediction features)
    results/results.json                 every reported number, with this command
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

import networkx as nx
import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))
from config import DRUGS_FP, DTI_FP, TARGETS_FP
from pipeline.drug_repurposing import DrugRepurposer
from pipeline.embeddings import embed_graph
from pipeline.link_prediction import DTILinkPredictor

SEED = 42
COMMAND = "PYTHONHASHSEED=0 python train_ml_models.py"


def main():
    for fp in (DRUGS_FP, TARGETS_FP, DTI_FP):
        if not fp.exists():
            sys.exit(f"Missing {fp}. Run: python src/datasources/drugbank_loader.py")

    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    drugs_df = pd.read_parquet(DRUGS_FP)
    targets_df = pd.read_parquet(TARGETS_FP)
    dti_df = pd.read_parquet(DTI_FP)

    t0 = time.time()
    predictor = DTILinkPredictor(seed=SEED)
    predictor.load_data(drugs_df, targets_df, dti_df)
    predictor.random_split_edges()

    print("Node2Vec on the message graph (link-prediction features)...")
    emb_message = embed_graph(predictor.train_graph, seed=SEED)
    emb_message.to_parquet("data/embeddings_message.parquet")
    predictor.set_embeddings(emb_message)

    print("Training random forest...")
    predictor.train_model()
    ev = predictor.evaluation_results
    calibration = predictor.evaluate_calibration(ev["test_edges"], ev["test_negatives"])
    baselines = predictor.compare_baselines(ev["test_edges"], ev["test_negatives"])
    predictor.save_model("models/link_prediction_model.pkl")

    pd.to_pickle({
        "model_performance": ev["model_performance"],
        "baselines": baselines,
        "calibration": calibration,
        "feature_importances": ev["feature_importances"],
    }, "models/evaluation_results.pkl")

    print("Node2Vec on the full graph (similarity and layout)...")
    full_graph = nx.Graph()
    full_graph.add_edges_from(zip(dti_df["drug_id"], dti_df["target_id"]))
    emb_full = embed_graph(full_graph, seed=SEED)
    emb_full.to_parquet("data/embeddings_full.parquet")

    repurposer = DrugRepurposer()
    repurposer.load_data(drugs_df, targets_df, dti_df, emb_full)
    pd.to_pickle(repurposer, "models/drug_repurposing_model.pkl")

    mp = ev["model_performance"]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    results = {
        "command": COMMAND,
        "data_command": "python src/datasources/drugbank_loader.py",
        "git_commit": commit,
        "python": platform.python_version(),
        "seed": SEED,
        "runtime_seconds": round(time.time() - t0, 1),
        "dataset": {
            "drugs": len(drugs_df),
            "drugs_with_human_target": int(dti_df["drug_id"].nunique()),
            "human_targets": len(targets_df),
            "interactions": len(dti_df),
        },
        "split": mp["split"],
        "counts": {k: mp[k] for k in ("message_edges", "train_positives", "train_negatives",
                                      "test_positives", "test_negatives")},
        "random_forest": {
            "roc_auc": round(mp["auc"], 4),
            "pr_auc_trapezoidal": round(mp["pr_auc"], 4),
            "average_precision": round(mp["ap_score"], 4),
            "brier_score": round(calibration["brier_score"], 4),
        },
        "baselines": {
            name: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}
            for name, b in baselines.items()
        },
        "feature_importances": {k: round(float(v), 4)
                                for k, v in ev["feature_importances"].items()},
    }
    Path("results/results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
