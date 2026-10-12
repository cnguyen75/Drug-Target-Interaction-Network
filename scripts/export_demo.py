#!/usr/bin/env python3
"""Export a small replay file for a static demo page.

    python scripts/export_demo.py OUT.json                 # model outputs + public identifiers
    python scripts/export_demo.py OUT.json --include-known # also names known targets (see below)

For each drug in DEMO_DRUGS (well-known approved small molecules, chosen by
hand to span drug classes), the file holds the app model's top predicted
targets with calibrated scores, ties, each target's number of known drugs and
a short reason built from counts.

DrugBank's full database is licensed CC BY-NC 4.0 under an academic licence.
By default nothing row-for-row from DrugBank is exported: no known
drug-target pairs and no DrugBank text. Drug names come with DrugBank IDs
(both are in DrugBank's CC0 "Open Data" vocabulary); targets are identified
by UniProt accession and gene symbol. Known interactions appear only as
counts. --include-known adds each drug's known targets (UniProt accession and
gene symbol only); Colin approved this for the 23 demo drugs on 2026-10-12.
The attribution field must be shown wherever the demo appears.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from pipeline.explore import featured_mask, rank_key  # noqa: E402
from pipeline.link_prediction import DTILinkPredictor  # noqa: E402

DEMO_DRUGS = [
    "Imatinib", "Gefitinib", "Tamoxifen", "Methotrexate", "Metformin", "Atorvastatin",
    "Simvastatin", "Amlodipine", "Lisinopril", "Warfarin", "Clopidogrel", "Ibuprofen",
    "Celecoxib", "Acetylsalicylic acid", "Morphine", "Sertraline", "Fluoxetine", "Haloperidol",
    "Diazepam", "Sildenafil", "Omeprazole", "Loratadine", "Caffeine",
]
ATTRIBUTION = "Interaction data from DrugBank 5.1, used under CC BY-NC 4.0 for non-commercial purposes."
TOP_K = 5
MAX_BYTES = 500_000


def reason(G, drug, target):
    """Counts only: drugs sharing a target with `drug` that already bind `target`."""
    mine = set(G.neighbors(drug))
    n = sum(1 for o in G.neighbors(target) if o != drug and mine & set(G.neighbors(o)))
    if n:
        return {"kind": "shared_target_path", "supporting_drugs": n,
                "text": (f"1 drug that shares a target with this drug already binds it." if n == 1
                         else f"{n} drugs that share a target with this drug already bind it.")}
    return {"kind": "graph_position", "supporting_drugs": 0,
            "text": "No shared-target path; the score comes from degree and embedding position."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--include-known", action="store_true")
    args = ap.parse_args()

    results = json.loads((ROOT / "results" / "results.json").read_text())
    p = DTILinkPredictor.load_model(ROOT / "models" / "link_prediction_model.pkl")
    G = p.G
    drugs = p.drugs_df[featured_mask(p.drugs_df)]
    by_name = dict(zip(drugs["name"], drugs["drug_id"]))
    genes = p.targets_df.set_index("target_id")["gene_symbol"].to_dict()
    threshold = results["app_suggestions"]["default_threshold"]

    out_drugs, missing = [], []
    for name in DEMO_DRUGS:
        d = by_name.get(name)
        if d is None or G.degree(d) == 0:
            missing.append(name)
            continue
        scores = p.score_candidates(d)
        ranked = sorted(scores.items(), key=lambda x: rank_key(G, x))
        top = ranked[:TOP_K]
        shown = [f"{s:.2f}" for _, s in top]
        entry = {
            "id": d,
            "name": name,
            "known_target_count": G.degree(d),
            "candidates_at_or_above_default": int(sum(1 for _, s in scores.items() if s >= threshold)),
            "predicted": [{
                "rank": i + 1,
                "uniprot": t,
                "gene": genes.get(t) or None,
                "score": round(float(s), 4),
                "score_tied": shown.count(f"{s:.2f}") > 1,
                "target_known_drug_count": G.degree(t),
                "reason": reason(G, d, t),
            } for i, (t, s) in enumerate(top)],
        }
        if args.include_known:
            entry["known_targets"] = [{"uniprot": t, "gene": genes.get(t) or None}
                                      for t in sorted(G.neighbors(d))]
        out_drugs.append(entry)

    demo = {
        "schema_version": 1,
        "generated_on": date.today().isoformat(),
        "command": "python scripts/export_demo.py OUT.json"
                   + (" --include-known" if args.include_known else ""),
        "results_git_commit": results["git_commit"],
        "source": {k: results["source"][k] for k in ("file", "version", "exported_on")},
        "model": {
            "description": results["model"],
            "train_negatives": results["app_model"]["train_negatives"],
            "calibration": results["app_model"]["calibration"],
            "default_threshold": threshold,
        },
        "attribution": ATTRIBUTION,
        "contents": ("model outputs and public identifiers only" if not args.include_known
                     else "model outputs, public identifiers and each drug's known targets "
                          "(UniProt accession and gene symbol)"),
        "score_note": ("Calibrated on validation pairs with one non-interaction per interaction; a "
                       "ranking signal, not the chance an interaction is real. score_tied marks "
                       "scores that round to the same two decimals as another shown candidate."),
        "drugs": out_drugs,
        "not_found": missing,
    }
    text = json.dumps(demo, indent=1, ensure_ascii=False)
    if len(text.encode()) > MAX_BYTES:
        sys.exit(f"demo is {len(text.encode()):,} bytes, over {MAX_BYTES:,}")
    Path(args.out).write_text(text + "\n")
    print(f"{len(out_drugs)} drugs, {len(text.encode()):,} bytes -> {args.out}; not found: {missing}")


if __name__ == "__main__":
    main()
