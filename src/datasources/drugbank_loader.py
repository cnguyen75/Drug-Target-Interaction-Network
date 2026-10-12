"""DrugBank XML data loader and parser.

Usage:
    python src/datasources/drugbank_loader.py            # reads data_raw/drugbank.xml

Writes data_raw/drugs_full.parquet, targets_full.parquet and dti_full.parquet,
which app.py and scripts/train_ml_models.py read.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from lxml import etree

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DRUGS_FP, DTI_FP, RAW_XML, TARGETS_FP

NS_URI = "http://www.drugbank.ca"
NS = {"db": NS_URI}


def _text(elem, path):
    found = elem.find(path, NS)
    return found.text if found is not None and found.text else ""


def _parse_drug(drug_elem, drugs_data, targets_data, dti_data):
    """Append one top-level <drug> element's rows to the three output lists."""
    drug_id = drug_elem.find("db:drugbank-id[@primary='true']", NS)
    if drug_id is None or not (drug_id.text or "").strip():
        return
    drug_id = drug_id.text.strip()

    # "small molecule" or "biotech" is an attribute on <drug>, not a group.
    drug_type = drug_elem.get("type", "unknown")

    groups = [g.text for g in drug_elem.findall("db:groups/db:group", NS) if g.text]
    approved = "approved" in groups

    atc_codes = [a.get("code") for a in drug_elem.findall("db:atc-codes/db:atc-code", NS)
                 if a.get("code")]
    atc_l1 = atc_l2 = None
    if atc_codes and len(atc_codes[0]) >= 3:
        atc_l1 = atc_codes[0][0]
        atc_l2 = atc_codes[0][:3]

    smiles = None
    for prop in drug_elem.findall("db:calculated-properties/db:property", NS):
        if _text(prop, "db:kind") == "SMILES" and _text(prop, "db:value"):
            smiles = _text(prop, "db:value")
            break

    drugs_data.append({
        "drug_id": drug_id,
        "name": _text(drug_elem, "db:name"),
        "type": drug_type,
        "groups": ",".join(groups),
        "approved": approved,
        "atc_codes": ",".join(atc_codes) if atc_codes else None,
        "atc_l1": atc_l1,
        "atc_l2": atc_l2,
        "indications": _text(drug_elem, "db:indication"),
        "smiles": smiles,
    })

    for target_elem in drug_elem.findall("db:targets/db:target", NS):
        # Only human targets
        organism = _text(target_elem, "db:organism")
        if "Homo sapiens" not in organism:
            continue

        # Target ID (UniProt)
        polypeptide = target_elem.find("db:polypeptide", NS)
        if polypeptide is None or not polypeptide.get("id"):
            continue
        target_id = polypeptide.get("id")

        targets_data.append({
            "target_id": target_id,
            "name": _text(target_elem, "db:name") or _text(polypeptide, "db:name"),
            "gene_symbol": _text(polypeptide, "db:gene-name"),
            "organism": "Homo sapiens",
            "is_human": True,
            "protein_family": None,
        })

        actions = [a.text for a in target_elem.findall("db:actions/db:action", NS) if a.text]
        dti_data.append({
            "drug_id": drug_id,
            "target_id": target_id,
            "action": actions[0] if actions else "unknown",
        })


def parse_drugbank_xml(xml_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse DrugBank XML and extract drugs, targets, and interactions.

    Only top-level <drug> entries are read (nested <drug> elements inside
    pathways and interactions are references, not entries). The file is
    streamed so the full export does not have to fit in memory.

    Returns:
        Tuple of (drugs_df, targets_df, dti_df) DataFrames
    """
    print(f"Parsing DrugBank XML from: {xml_path}")

    drugs_data, targets_data, dti_data = [], [], []
    drug_tag = f"{{{NS_URI}}}drug"

    context = etree.iterparse(str(xml_path), events=("end",), tag=drug_tag)
    for _, elem in context:
        parent = elem.getparent()
        # Top-level entries are direct children of the root <drugbank> element.
        if parent is None or parent.getparent() is not None:
            continue
        _parse_drug(elem, drugs_data, targets_data, dti_data)
        elem.clear()
        while elem.getprevious() is not None:
            del parent[0]

    drugs_df = pd.DataFrame(drugs_data).drop_duplicates(subset=["drug_id"])
    targets_df = pd.DataFrame(targets_data).drop_duplicates(subset=["target_id"])
    dti_df = pd.DataFrame(dti_data).drop_duplicates(subset=["drug_id", "target_id"])

    print(f"Parsed {len(drugs_df)} drugs, {len(targets_df)} targets, {len(dti_df)} interactions")

    return drugs_df, targets_df, dti_df


def main():
    """Parse DrugBank XML and save to parquet files."""
    if not RAW_XML.exists():
        print(f"Error: DrugBank XML file not found at {RAW_XML}")
        print("Please place drugbank.xml in the data_raw/ directory")
        sys.exit(1)

    drugs_df, targets_df, dti_df = parse_drugbank_xml(RAW_XML)

    DRUGS_FP.parent.mkdir(exist_ok=True)
    drugs_df.to_parquet(DRUGS_FP, index=False)
    targets_df.to_parquet(TARGETS_FP, index=False)
    dti_df.to_parquet(DTI_FP, index=False)

    print("Saved data to:")
    print(f"  Drugs: {DRUGS_FP}")
    print(f"  Targets: {TARGETS_FP}")
    print(f"  DTI: {DTI_FP}")

    print("\nSummary:")
    print(f"  Drugs: {len(drugs_df)}")
    print(f"  Drugs with at least one human target: {dti_df['drug_id'].nunique()}")
    print(f"  Targets: {len(targets_df)}")
    print(f"  Interactions: {len(dti_df)}")
    print(f"  Approved drugs: {drugs_df['approved'].sum()}")


if __name__ == "__main__":
    main()
