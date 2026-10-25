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

from datetime import date
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


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
CALIBRATION = "isotonic"
COMMAND = "python train_ml_models.py"
DATA_COMMAND = "python src/datasources/drugbank_loader.py"


def source_meta() -> dict[str, str]:
    """DrugBank file name, version and export date recorded by the loader."""
    fp = DRUGS_FP.parent / "drugbank_meta.json"
    return json.loads(fp.read_text()) if fp.exists() else {"file": RAW_XML.name}


def r4(x: float) -> float:
    return round(float(x), 4)


def metric_block(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "roc_auc": r4(m["roc_auc"]),
        "roc_auc_ci95": [r4(v) for v in m["roc_auc_ci95"]],
        "average_precision": r4(m["ap_score"]),
        "average_precision_ci95": [r4(v) for v in m["ap_ci95"]],
    }


NEGATIVE_SCHEMES = ("uniform", "degree_matched")
NEGATIVE_LABELS = {
    "uniform": "uniform random negatives",
    "degree_matched": "degree-matched negatives (same drug, similar-degree target)",
}


def fit_variants(
    kind: str, tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> dict[str, DTILinkPredictor]:
    """Fit one model per training-negative scheme on the same split and embeddings."""
    t0 = time.time()
    variants, emb = {}, None
    for scheme in NEGATIVE_SCHEMES:
        p = DTILinkPredictor(seed=SEED, train_negatives=scheme)
        p.load_data(*tables)
        p.split_edges(kind)
        if emb is None:
            print(f"[{kind}] Node2Vec on the message graph...")
            emb = embed_graph(p.train_graph, seed=SEED)
        p.set_embeddings(emb)
        p.fit()
        p.validate()
        variants[scheme] = p
    print(f"[{kind}] fitted both variants in {time.time() - t0:.0f}s")
    return variants


def select_scheme(variants: dict[str, DTILinkPredictor]) -> tuple[str, dict[str, dict[str, Any]]]:
    """Pick the training-negative scheme on per-drug validation (ROC AUC, then AP)."""
    val = {s: p.evaluation_results["model_performance"] for s, p in variants.items()}
    best = max(val, key=lambda s: (round(val[s]["val_auc"], 4), round(val[s]["val_ap_score"], 4)))
    return best, {
        s: {
            "roc_auc": r4(v["val_auc"]),
            "average_precision": r4(v["val_ap_score"]),
            "positives": v["val_positives"],
            "negatives": v["val_negatives"],
        }
        for s, v in val.items()
    }


def model_block(p: DTILinkPredictor) -> dict[str, Any]:
    mp = p.evaluation_results["model_performance"]
    return {
        **metric_block(
            {
                "roc_auc": mp["auc"],
                "roc_auc_ci95": mp["roc_auc_ci95"],
                "ap_score": mp["ap_score"],
                "ap_ci95": mp["ap_ci95"],
            }
        ),
        "pr_auc_trapezoidal": r4(mp["pr_auc"]),
        "brier_score": r4(mp["brier_score"]),
        "validation": {"roc_auc": r4(mp["val_auc"]), "average_precision": r4(mp["val_ap_score"])},
        "feature_importances": {
            k: r4(v) for k, v in p.evaluation_results["feature_importances"].items()
        },
    }


def split_summary(variants: dict[str, DTILinkPredictor]) -> dict[str, Any]:
    ref = variants["uniform"]
    ev, mp = ref.evaluation_results, ref.evaluation_results["model_performance"]
    counts = {
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
    }
    return {
        "description": mp["split_description"],
        "counts": counts,
        "models": {s: model_block(p) for s, p in variants.items()},
        "baselines": {
            name: metric_block(
                {
                    "roc_auc": b["roc_auc"],
                    "roc_auc_ci95": b["roc_auc_ci95"],
                    "ap_score": b["ap_score"],
                    "ap_ci95": b["ap_ci95"],
                }
            )
            for name, b in ev["baselines"].items()
        },
        "bootstrap_resamples": mp["n_boot"],
    }


def main() -> None:
    for fp in (DRUGS_FP, TARGETS_FP, DTI_FP):
        if not fp.exists():
            sys.exit(f"Missing {fp}. Run: {DATA_COMMAND}")
    for d in ("models", "data", "results"):
        Path(d).mkdir(exist_ok=True)

    t0 = time.time()
    tables = (pd.read_parquet(DRUGS_FP), pd.read_parquet(TARGETS_FP), pd.read_parquet(DTI_FP))
    drugs_df, targets_df, dti_df = tables
    stats = graph_stats(build_bipartite_graph(*tables))

    splits = {kind: fit_variants(kind, tables) for kind in ("random", "drug_holdout")}

    # Choose the app's training negatives on per-drug validation, before any
    # test pair is scored. Then score every test set once.
    chosen, val_table = select_scheme(splits["drug_holdout"])
    print(f"Selected on per-drug validation: {chosen}")
    for variants in splits.values():
        for p in variants.values():
            p.evaluate_test()
    pd.to_pickle(
        {
            kind: {
                s: {k: v for k, v in p.evaluation_results.items() if k != "test_pairs"}
                for s, p in variants.items()
            }
            for kind, variants in splits.items()
        },
        "models/evaluation_results.pkl",
    )

    print(f"[final] training the app's model on all edges ({chosen})...")
    final = DTILinkPredictor(seed=SEED, train_negatives=chosen)
    final.load_data(*tables)
    final.split_edges("final")
    final.set_embeddings(embed_graph(final.train_graph, seed=SEED))
    final.fit()
    final.validate()
    final.calibrate(CALIBRATION)
    final.save_model("models/link_prediction_model.pkl")
    final_mp = final.evaluation_results["model_performance"]

    print("Node2Vec on the full graph (similarity and layout)...")
    emb_full = embed_graph(build_bipartite_graph(*tables), seed=SEED)
    emb_full.to_parquet("data/embeddings_full.parquet")
    repurposer = DrugRepurposer()
    repurposer.load_data(drugs_df, targets_df, dti_df, emb_full)
    pd.to_pickle(repurposer, "models/drug_repurposing_model.pkl")

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    results = {
        "commands": [DATA_COMMAND, COMMAND],
        "git_commit": commit,
        "python": platform.python_version(),
        "seed": SEED,
        "runtime_seconds": round(time.time() - t0, 1),
        "run_date": date.today().isoformat(),
        "source": {
            **source_meta(),
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
        "metrics_note": (
            "ROC AUC and average precision on held-out pairs (1:1 positives to "
            "negatives); 95% percentile bootstrap intervals over test pairs. Each "
            "model is fitted on 75% of the training positives; the other 25% are "
            "validation. "
            "Chance and common neighbours tie every pair, so both score 0.5 by "
            "construction."
        ),
        "splits": {kind: split_summary(v) for kind, v in splits.items()},
        "negative_schemes": NEGATIVE_LABELS,
        "app_model": {
            "train_negatives": chosen,
            "selected_by": (
                "higher ROC AUC on the per-drug split's validation drugs, "
                "measured before any test pair was scored"
            ),
            "per_drug_validation": val_table,
            "calibration": CALIBRATION,
            "calibration_note": (
                "isotonic regression fitted on the app model's validation pairs "
                "(1:1 known interactions to degree-matched non-interactions)"
            ),
            "validation": {
                "roc_auc": r4(final_mp["val_auc"]),
                "average_precision": r4(final_mp["val_ap_score"]),
                "positives": final_mp["val_positives"],
            },
            "split": final_mp["split_description"],
        },
    }
    Path("results/results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
