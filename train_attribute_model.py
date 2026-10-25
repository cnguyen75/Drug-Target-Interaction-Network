#!/usr/bin/env python3
"""Part D: does adding DrugBank attributes let the model score unseen drugs?

    python src/datasources/drugbank_attributes.py   # once: drug and target attributes
    python train_attribute_model.py

Compares, on identical splits and negatives:
    graph model      the current app model's design (12 graph/embedding
                     features, degree-matched training negatives)
    attribute model  the same plus 8 attribute features (src/pipeline/attributes.py)
on three splits: random edge split, per-drug split, and the whole-drug
holdout (test drugs have every edge removed). No hyperparameters are tuned;
validation scores are recorded, and each test set is scored once.

It then trains the two calibrated copies the app uses (v3):
    models/attribute_model_seen.pkl  per-drug protocol ("final"): drugs with
                                     known targets, and target queries
    models/attribute_model_cold.pkl  whole-drug protocol ("final_cold"): drugs
                                     with no known targets

Writes results/results_attributes.json. Does not touch results/results.json;
`python train_ml_models.py` reproduces the v2 graph-only model.
"""

from datetime import date
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd


sys.path.append(str(Path(__file__).parent / "src"))
from config import DRUGS_FP, DTI_FP, TARGETS_FP
from datasources.drugbank_attributes import DRUG_ATTR_FP, TARGET_ATTR_FP
from pipeline.attributes import ATTRIBUTE_FEATURES, AttributeLinkPredictor
from pipeline.embeddings import embed_graph
from pipeline.link_prediction import DTILinkPredictor, _ranking_metrics, bootstrap_ci


SEED = 42
SPLITS = ("random", "drug_holdout", "cold_drug")
COMMAND = "python train_attribute_model.py"


def r4(x):
    return round(float(x), 4)


def block(roc, ap, ci, val=None):
    out = {
        "roc_auc": r4(roc),
        "roc_auc_ci95": [r4(v) for v in ci["roc_auc_ci95"]],
        "average_precision": r4(ap),
        "average_precision_ci95": [r4(v) for v in ci["ap_ci95"]],
    }
    if val:
        out["validation"] = val
    return out


def model_block(p):
    mp = p.evaluation_results["model_performance"]
    return block(
        mp["auc"],
        mp["ap_score"],
        {"roc_auc_ci95": mp["roc_auc_ci95"], "ap_ci95": mp["ap_ci95"]},
        {"roc_auc": r4(mp["val_auc"]), "average_precision": r4(mp["val_ap_score"])},
    )


def main():
    t0 = time.time()
    for fp in (DRUGS_FP, TARGETS_FP, DTI_FP, DRUG_ATTR_FP, TARGET_ATTR_FP):
        if not fp.exists():
            sys.exit(f"Missing {fp}. Run the loader and src/datasources/drugbank_attributes.py")
    tables = (pd.read_parquet(DRUGS_FP), pd.read_parquet(TARGETS_FP), pd.read_parquet(DTI_FP))
    drug_attrs, target_attrs = pd.read_parquet(DRUG_ATTR_FP), pd.read_parquet(TARGET_ATTR_FP)
    current = json.loads(Path("results/results.json").read_text())

    out_splits = {}
    for kind in SPLITS:
        ts = time.time()
        graph = DTILinkPredictor(seed=SEED, train_negatives="degree_matched")
        attr = AttributeLinkPredictor(drug_attrs, target_attrs, seed=SEED)
        for p in (graph, attr):
            p.load_data(*tables)
            p.split_edges(kind)
        print(f"[{kind}] Node2Vec on the message graph...")
        emb = embed_graph(graph.train_graph, seed=SEED)
        for p in (graph, attr):
            p.set_embeddings(emb)
            p.fit()
            p.validate()
        assert graph.fit_neg == attr.fit_neg and graph.val_neg == attr.val_neg
        # Test sets: scored once each, after all fitting and validation.
        for p in (graph, attr):
            p.evaluate_test()
        pairs = graph.evaluation_results["test_pairs"]
        assert pairs == attr.evaluation_results["test_pairs"]
        y = graph.evaluation_results["test_labels"]

        # Extra baseline, no learning: structural nearest neighbour among the target's binders.
        nn = np.array([attr.attribute_features(d, t)[0] for d, t in pairs])
        nn_roc, nn_ap = _ranking_metrics(y, nn)
        baselines = {
            k: block(b["roc_auc"], b["ap_score"], b)
            for k, b in graph.evaluation_results["baselines"].items()
            if k in ("l3_paths", "preferential_attachment", "chance")
        }
        baselines["structural_nearest_neighbour"] = block(
            nn_roc, nn_ap, bootstrap_ci(y, nn, seed=SEED)
        )

        mp = graph.evaluation_results["model_performance"]
        entry = {
            "description": mp["split_description"],
            "counts": {
                k: mp[k]
                for k in (
                    "message_edges",
                    "fit_positives",
                    "val_positives",
                    "test_positives",
                    "test_negatives",
                    "test_drugs",
                )
                if k in mp
            },
            "graph_model": model_block(graph),
            "attribute_model": model_block(attr),
            "baselines": baselines,
            "attribute_model_feature_importances": {
                k: r4(v) for k, v in attr.evaluation_results["feature_importances"].items()
            },
            "test_drugs_with_structure": (
                r4(np.mean([d in attr.fps for d in attr.test_drugs])) if kind != "random" else None
            ),
        }
        if kind in current["splits"]:
            ref = current["splits"][kind]["models"]["degree_matched"]
            entry["graph_model_matches_results_json"] = (
                entry["graph_model"]["roc_auc"] == ref["roc_auc"]
                and entry["graph_model"]["average_precision"] == ref["average_precision"]
            )
        out_splits[kind] = entry
        print(f"[{kind}] done in {time.time() - ts:.0f}s")

    app_copies = {}
    for name, kind in (("seen", "final"), ("cold", "final_cold")):
        print(f"[app/{name}] training the app copy ({kind})...")
        p = AttributeLinkPredictor(drug_attrs, target_attrs, seed=SEED)
        p.load_data(*tables)
        p.split_edges(kind)
        p.set_embeddings(embed_graph(p.train_graph, seed=SEED))
        p.fit()
        p.validate()
        p.calibrate("isotonic")
        p.save_model(f"models/attribute_model_{name}.pkl")
        mp = p.evaluation_results["model_performance"]
        app_copies[name] = {
            "split": mp["split_description"],
            "validation": {
                "roc_auc": r4(mp["val_auc"]),
                "average_precision": r4(mp["val_ap_score"]),
                "positives": mp["val_positives"],
            },
            "calibration": "isotonic regression on the validation pairs",
        }

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    results = {
        "commands": ["python src/datasources/drugbank_attributes.py", COMMAND],
        "git_commit": commit,
        "run_date": date.today().isoformat(),
        "python": platform.python_version(),
        "seed": SEED,
        "runtime_seconds": round(time.time() - t0, 1),
        "attribute_features": ATTRIBUTE_FEATURES,
        "tuning": "none; both models use the app model's settings. Validation scores are reported.",
        "metrics_note": (
            "ROC AUC and average precision on held-out pairs, 1:1 with degree-matched "
            "negatives on the per-drug and whole-drug splits and uniform negatives on "
            "the random split; 95% bootstrap intervals over test pairs (1,000 resamples). "
            "Both models see the same splits and negatives."
        ),
        "splits": out_splits,
        "app_model": {
            "version": "v3 attribute model",
            "routing": (
                "drugs with known targets and target queries use the 'seen' copy; drugs "
                "with no known targets use the 'cold' copy"
            ),
            "copies": app_copies,
        },
    }
    Path("results/results_attributes.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {m: v[m]["roc_auc"] for m in ("graph_model", "attribute_model")}
                for k, v in out_splits.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
