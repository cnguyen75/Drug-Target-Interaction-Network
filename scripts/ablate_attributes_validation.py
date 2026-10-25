#!/usr/bin/env python3
"""How much does the whole-drug result depend on curated, mechanism-like attributes?

    python scripts/ablate_attributes_validation.py

DrugBank categories (MeSH pharmacological actions such as "Adrenergic
beta-Antagonists") and ATC codes are assigned partly because a drug's targets
are already known, so a genuinely new compound would not have them. This
refits the attribute model on the whole-drug holdout without some feature
groups and scores the VALIDATION drugs only; the test set is not used.

Writes results/results_attribute_ablation.json.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
os.chdir(ROOT)
from config import DRUGS_FP, DTI_FP, TARGETS_FP  # noqa: E402
from datasources.drugbank_attributes import DRUG_ATTR_FP, TARGET_ATTR_FP  # noqa: E402
from pipeline.attributes import ATTRIBUTE_FEATURES, AttributeLinkPredictor  # noqa: E402
from pipeline.embeddings import embed_graph  # noqa: E402
from pipeline.link_prediction import FEATURE_NAMES, _ranking_metrics, bootstrap_ci  # noqa: E402


SEED = 42
ALL = FEATURE_NAMES + ATTRIBUTE_FEATURES
VARIANTS = {
    "all attribute features": [],
    "without categories": ["category_max_jaccard"],
    "without categories and ATC": ["category_max_jaccard", "atc_max_jaccard"],
    "structure only": [
        "category_max_jaccard",
        "atc_max_jaccard",
        "same_class_share",
        "same_superclass_share",
    ],
}


def main():
    tables = (pd.read_parquet(DRUGS_FP), pd.read_parquet(TARGETS_FP), pd.read_parquet(DTI_FP))
    p = AttributeLinkPredictor(
        pd.read_parquet(DRUG_ATTR_FP), pd.read_parquet(TARGET_ATTR_FP), seed=SEED
    )
    p.load_data(*tables)
    p.split_edges("cold_drug")
    p.set_embeddings(embed_graph(p.train_graph, seed=SEED))
    p.fit()  # draws the same fit negatives as train_attribute_model.py
    p.validate()  # and the same validation negatives

    X_fit = p._features(p.fit_edges + p.fit_neg)
    y_fit = np.array([1] * len(p.fit_edges) + [0] * len(p.fit_neg))
    X_val = p._features(p.val_edges + p.val_neg)
    y_val = p.val_labels

    rows = {}
    for name, drop in VARIANTS.items():
        keep = [i for i, f in enumerate(ALL) if f not in drop]
        scaler = StandardScaler().fit(X_fit[:, keep])
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
        rf.fit(scaler.transform(X_fit[:, keep]), y_fit)
        s = rf.predict_proba(scaler.transform(X_val[:, keep]))[:, 1]
        roc, ap = _ranking_metrics(y_val, s)
        ci = bootstrap_ci(y_val, s, seed=SEED)
        rows[name] = {
            "dropped": drop,
            "val_roc_auc": round(float(roc), 4),
            "val_roc_auc_ci95": [round(v, 4) for v in ci["roc_auc_ci95"]],
            "val_average_precision": round(float(ap), 4),
        }
        print(f"{name:45s} ROC AUC {roc:.4f} {rows[name]['val_roc_auc_ci95']}")
    out = {
        "command": "python scripts/ablate_attributes_validation.py",
        "split": "cold_drug (whole-drug holdout), validation drugs only; test set not used",
        "validation_positives": len(p.val_edges),
        "validation_drugs": len(p.val_drugs),
        "variants": rows,
    }
    Path("results/results_attribute_ablation.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
