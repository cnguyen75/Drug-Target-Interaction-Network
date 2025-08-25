"""DrugBank XML data loader and parser."""

import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from lxml import etree

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DRUGS_FP, DTI_FP, RAW_XML, TARGETS_FP


def parse_drugbank_xml(xml_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse DrugBank XML and extract drugs, targets, and interactions.
    
    Args:
        xml_path: Path to DrugBank XML file
        
    Returns:
        Tuple of (drugs_df, targets_df, dti_df) DataFrames
    """
    print(f"Parsing DrugBank XML from: {xml_path}")
    
    # DrugBank namespace
    ns = {"db": "http://www.drugbank.ca"}
    
    # Parse XML
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    drugs_data = []
    targets_data = []
    dti_data = []
    
    # Process each drug
    for drug_elem in root.findall(".//db:drug", ns):
        try:
            # Extract drug information
            drug_id = drug_elem.find("db:drugbank-id[@primary='true']", ns)
            if drug_id is None:
                continue
            drug_id = drug_id.text
            
            name_elem = drug_elem.find("db:name", ns)
            name = name_elem.text if name_elem is not None else ""
            
            # Drug type
            type_elem = drug_elem.find("db:groups", ns)
            drug_type = "small molecule"  # default
            if type_elem is not None:
                group_elems = type_elem.findall("db:group", ns)
                if any(g.text == "biotech" for g in group_elems if g.text):
                    drug_type = "biologic"
            
            # Approval status
            approved = False
            if type_elem is not None:
                group_elems = type_elem.findall("db:group", ns)
                approved = any(g.text == "approved" for g in group_elems if g.text)
            
            # ATC codes
            atc_l1 = None
            atc_l2 = None
            atc_elem = drug_elem.find(".//db:atc-codes", ns)
            if atc_elem is not None:
                first_atc = atc_elem.find("db:atc-code", ns)
                if first_atc is not None:
                    atc_code = first_atc.get("code")
                    if atc_code and len(atc_code) >= 3:
                        atc_l1 = atc_code[0]
                        atc_l2 = atc_code[:3]
            
            # Indications
            indications = ""
            indication_elem = drug_elem.find("db:indication", ns)
            if indication_elem is not None:
                indications = indication_elem.text or ""
            
            # SMILES
            smiles = None
            calc_props = drug_elem.find(".//db:calculated-properties", ns)
            if calc_props is not None:
                for prop in calc_props.findall("db:property", ns):
                    kind = prop.find("db:kind", ns)
                    value = prop.find("db:value", ns)
                    if (kind is not None and kind.text == "SMILES" and 
                        value is not None and value.text):
                        smiles = value.text
                        break
            
            # Add drug to list
            drugs_data.append({
                "drug_id": drug_id,
                "name": name,
                "type": drug_type,
                "approved": approved,
                "atc_l1": atc_l1,
                "atc_l2": atc_l2,
                "indications": indications,
                "smiles": smiles
            })
            
            # Extract targets
            targets_elem = drug_elem.find(".//db:targets", ns)
            if targets_elem is not None:
                for target_elem in targets_elem.findall("db:target", ns):
                    try:
                        # Check organism (only human targets)
                        organism_elem = target_elem.find("db:organism", ns)
                        if (organism_elem is None or 
                            "Homo sapiens" not in (organism_elem.text or "")):
                            continue
                        
                        # Target ID (UniProt)
                        polypeptide = target_elem.find(".//db:polypeptide", ns)
                        if polypeptide is None:
                            continue
                        
                        target_id = polypeptide.get("id")
                        if not target_id:
                            continue
                        
                        # Gene symbol
                        gene_symbol = ""
                        gene_elem = polypeptide.find("db:gene-name", ns)
                        if gene_elem is not None:
                            gene_symbol = gene_elem.text or ""
                        
                        # Add target
                        targets_data.append({
                            "target_id": target_id,
                            "gene_symbol": gene_symbol,
                            "organism": "Homo sapiens",
                            "protein_family": None
                        })
                        
                        # Extract actions
                        actions_elem = target_elem.find(".//db:actions", ns)
                        action = "unknown"
                        if actions_elem is not None:
                            first_action = actions_elem.find("db:action", ns)
                            if first_action is not None:
                                action = first_action.text or "unknown"
                        
                        # Confidence based on approval status
                        confidence = 1.0 if approved else 0.7
                        
                        # Add interaction
                        dti_data.append({
                            "drug_id": drug_id,
                            "target_id": target_id,
                            "action": action,
                            "confidence": confidence
                        })
                        
                    except Exception as e:
                        print(f"Error processing target for drug {drug_id}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing drug: {e}")
            continue
    
    # Create DataFrames
    drugs_df = pd.DataFrame(drugs_data)
    targets_df = pd.DataFrame(targets_data)
    dti_df = pd.DataFrame(dti_data)
    
    # Remove duplicates
    drugs_df = drugs_df.drop_duplicates(subset=["drug_id"])
    targets_df = targets_df.drop_duplicates(subset=["target_id"])
    dti_df = dti_df.drop_duplicates(subset=["drug_id", "target_id"])
    
    print(f"Parsed {len(drugs_df)} drugs, {len(targets_df)} targets, {len(dti_df)} interactions")
    
    return drugs_df, targets_df, dti_df


def main():
    """Main function to parse DrugBank XML and save to parquet files."""
    if not RAW_XML.exists():
        print(f"Error: DrugBank XML file not found at {RAW_XML}")
        print("Please place drugbank.xml in the data_raw/ directory")
        sys.exit(1)
    
    # Parse XML
    drugs_df, targets_df, dti_df = parse_drugbank_xml(RAW_XML)
    
    # Save to parquet
    drugs_df.to_parquet(DRUGS_FP, index=False)
    targets_df.to_parquet(TARGETS_FP, index=False)
    dti_df.to_parquet(DTI_FP, index=False)
    
    print("Saved data to:")
    print(f"  Drugs: {DRUGS_FP}")
    print(f"  Targets: {TARGETS_FP}")
    print(f"  DTI: {DTI_FP}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Drugs: {len(drugs_df)}")
    print(f"  Targets: {len(targets_df)}")
    print(f"  Interactions: {len(dti_df)}")
    print(f"  Approved drugs: {drugs_df['approved'].sum()}")


if __name__ == "__main__":
    main()

