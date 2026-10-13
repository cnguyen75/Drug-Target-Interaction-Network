#!/usr/bin/env python3
"""Export a small replay file for a static demo page.

    python scripts/export_demo.py OUT.json                 # model outputs + public identifiers
    python scripts/export_demo.py OUT.json --include-known # also names known targets (see below)

For each drug in DEMO_DRUGS (well-known approved small molecules, chosen by
hand to span drug classes) and DEMO_DRUGS_NO_TARGETS (a drug with no known
human targets), the file holds the v3 app model's top predicted targets with
calibrated scores, ties, each target's number of known drugs and the app's
one-line reason.

DrugBank's full database is licensed CC BY-NC 4.0 under an academic licence.
By default nothing row-for-row from DrugBank is exported: no known
drug-target pairs and no DrugBank text. Drug names come with DrugBank IDs
(both are in DrugBank's CC0 "Open Data" vocabulary); targets are identified
by UniProt accession and gene symbol. Known interactions appear only as
counts, and reasons give only their kind (their text names known binders).
--include-known adds each drug's known targets (UniProt accession and gene
symbol only) and the full reason text; Colin approved this for the demo drugs
on 2026-10-12.
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
from pipeline.explore import evidence, featured_mask, rank_key  # noqa: E402
from pipeline.link_prediction import DTILinkPredictor  # noqa: E402

DEMO_DRUGS = [
    "Imatinib", "Gefitinib", "Tamoxifen", "Methotrexate", "Metformin", "Atorvastatin",
    "Simvastatin", "Amlodipine", "Lisinopril", "Warfarin", "Clopidogrel", "Ibuprofen",
    "Celecoxib", "Acetylsalicylic acid", "Morphine", "Sertraline", "Fluoxetine", "Haloperidol",
    "Diazepam", "Sildenafil", "Omeprazole", "Loratadine", "Caffeine",
]
ATTRIBUTION = "Interaction data from DrugBank 5.1, used under CC BY-NC 4.0 for non-commercial purposes."
# A well-known approved drug with no human targets in DrugBank (its one recorded
# target is a yeast enzyme), to show the unseen-drug capability.
DEMO_DRUGS_NO_TARGETS = ["Fluconazole"]
TOP_K = 5
MAX_BYTES = 500_000


REASON_KINDS = (("Structurally similar", "structural_similarity"),
                ("Bound by", "shared_target_drugs"),
                ("Similar to", "protein_family"),
                ("known binders share", "chemical_class"))


def reason(G, drug, target, model, label, with_text):
    """The app's evidence line for this suggestion, with its kind (text only if allowed)."""
    text = evidence(G, drug, target, label=label, attr=model)
    kind = next((k for marker, k in REASON_KINDS if marker in text), "none")
    return {"kind": kind, "text": text} if with_text else {"kind": kind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--include-known", action="store_true")
    args = ap.parse_args()

    results = json.loads((ROOT / "results" / "results_attributes.json").read_text())
    copies = {k: DTILinkPredictor.load_model(ROOT / "models" / f"attribute_model_{k}.pkl")
              for k in ("seen", "cold")}
    G = copies["seen"].G
    drugs = copies["seen"].drugs_df[featured_mask(copies["seen"].drugs_df)]
    by_name = dict(zip(drugs["name"], drugs["drug_id"]))
    genes = copies["seen"].targets_df.set_index("target_id")["gene_symbol"].to_dict()

    def label(n):
        return genes.get(n) or G.nodes[n]["name"]

    thresholds = {k: results["app_suggestions"][k]["default_threshold"] for k in ("seen", "cold")}

    out_drugs, missing = [], []
    for name in DEMO_DRUGS + DEMO_DRUGS_NO_TARGETS:
        d = by_name.get(name)
        expected_cold = name in DEMO_DRUGS_NO_TARGETS
        if d is None or (G.degree(d) == 0) != expected_cold:
            missing.append(name)
            continue
        copy = "cold" if expected_cold else "seen"
        scores = copies[copy].score_candidates(d)
        top = sorted(scores.items(), key=lambda x: rank_key(G, x))[:TOP_K]
        shown = [f"{s:.2f}" for _, s in top]
        entry = {
            "id": d,
            "name": name,
            "model_copy": copy,
            "known_target_count": G.degree(d),
            "candidates_at_or_above_default": int(sum(1 for s in scores.values()
                                                      if s >= thresholds[copy])),
            "predicted": [{
                "rank": i + 1,
                "uniprot": t,
                "gene": genes.get(t) or None,
                "score": round(float(s), 4),
                "score_tied": shown.count(f"{s:.2f}") > 1,
                "target_known_drug_count": G.degree(t),
                "reason": reason(G, d, t, copies[copy], label, args.include_known),
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
        "source": json.loads((ROOT / "results" / "results.json").read_text())["source"],
        "model": {
            "description": ("v3: random forest on 12 graph and Node2Vec features plus 8 attribute "
                            "features (structure, protein family, categories, ATC, chemical class)"),
            "train_negatives": "degree_matched",
            "calibration": "isotonic",
            "copies": {"seen": "drugs with known targets", "cold": "drugs with no known targets"},
            "default_threshold": thresholds,
        },
        "attribution": ATTRIBUTION,
        "contents": ("model outputs and public identifiers only" if not args.include_known
                     else "model outputs, public identifiers and each drug's known targets "
                          "(UniProt accession and gene symbol)"),
        "score_note": ("Calibrated (per model copy) on validation pairs with one non-interaction per "
                       "interaction; a "
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
