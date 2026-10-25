"""Drug and target attributes from the DrugBank export (for the attribute model).

    python src/datasources/drugbank_attributes.py

Writes data_raw/drug_attributes.parquet (drug_id, categories, ChemOnt kingdom /
superclass / class / subclass / direct parent) and data_raw/target_attributes.parquet
(target_id, Pfam identifiers, GO identifiers). Both are DrugBank-derived and
gitignored. The tables the current model reads are not touched.
"""

import os
from pathlib import Path
import sys

from lxml import etree
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_XML, REPO_ROOT
from datasources.drugbank_loader import HUMAN_LABELS, NS, NS_URI, open_drugbank


DRUG_ATTR_FP = REPO_ROOT / "data_raw" / "drug_attributes.parquet"
TARGET_ATTR_FP = REPO_ROOT / "data_raw" / "target_attributes.parquet"
CHEMONT = ("kingdom", "superclass", "class", "subclass", "direct-parent")


def _is_human(target_elem, polypeptide):
    org = polypeptide.find("db:organism", NS)
    taxon = org.get("ncbi-taxonomy-id") if org is not None else None
    if taxon is not None:
        return taxon == "9606"
    return (target_elem.findtext("db:organism", default="", namespaces=NS) or "") in HUMAN_LABELS


def parse_attributes(xml_path: Path):
    drugs, targets = {}, {}
    drug_tag = f"{{{NS_URI}}}drug"
    with open_drugbank(xml_path) as stream:
        for _, elem in etree.iterparse(stream, events=("end",), tag=drug_tag, huge_tree=True):
            parent = elem.getparent()
            if parent is None or parent.getparent() is not None:
                continue
            pid = elem.find("db:drugbank-id[@primary='true']", NS)
            if pid is not None and (pid.text or "").strip():
                drug_id = pid.text.strip()
                cats = sorted(
                    {
                        c.text.strip()
                        for c in elem.findall("db:categories/db:category/db:category", NS)
                        if c.text
                    }
                )
                row = {"drug_id": drug_id, "categories": "|".join(cats)}
                for level in CHEMONT:
                    row[level.replace("-", "_")] = (
                        elem.findtext(f"db:classification/db:{level}", default="", namespaces=NS)
                        or ""
                    ).strip()
                drugs.setdefault(drug_id, row)
                for t in elem.findall("db:targets/db:target", NS):
                    pp = t.find("db:polypeptide", NS)
                    if (
                        pp is None
                        or not pp.get("id")
                        or not _is_human(t, pp)
                        or pp.get("id") in targets
                    ):
                        continue
                    pfams = sorted(
                        {
                            p.findtext("db:identifier", default="", namespaces=NS)
                            for p in pp.findall("db:pfams/db:pfam", NS)
                        }
                        - {""}
                    )
                    gos = sorted(
                        {
                            g.findtext("db:description", default="", namespaces=NS)
                            for g in pp.findall("db:go-classifiers/db:go-classifier", NS)
                        }
                        - {""}
                    )
                    targets[pp.get("id")] = {
                        "target_id": pp.get("id"),
                        "pfams": "|".join(pfams),
                        "go_terms": "|".join(gos),
                    }
            elem.clear()
            while elem.getprevious() is not None:
                del parent[0]
    return pd.DataFrame(list(drugs.values())), pd.DataFrame(list(targets.values()))


def main():
    if not RAW_XML.exists():
        sys.exit(f"DrugBank export not found at {RAW_XML}; set DRUGBANK_PATH")
    drugs, targets = parse_attributes(RAW_XML)
    drugs.to_parquet(DRUG_ATTR_FP, index=False)
    targets.to_parquet(TARGET_ATTR_FP, index=False)
    print(
        f"{len(drugs)} drugs, {len(targets)} targets -> {DRUG_ATTR_FP.name}, {TARGET_ATTR_FP.name}"
    )


if __name__ == "__main__":
    main()
