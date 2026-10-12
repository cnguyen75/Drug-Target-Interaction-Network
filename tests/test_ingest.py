"""Tests for data ingestion pipeline."""

import os
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from datasources.drugbank_loader import parse_drugbank_xml


class TestDrugBankIngest:
    """Test DrugBank XML ingestion."""
    
    def test_parse_tiny_xml(self):
        """Test parsing of tiny DrugBank XML fixture."""
        xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"
        
        drugs_df, targets_df, dti_df = parse_drugbank_xml(xml_path)
        
        # Check required columns exist
        required_drug_cols = ["drug_id", "name", "type", "approved", "atc_l1", 
                             "atc_l2", "indications", "smiles"]
        required_target_cols = ["target_id", "gene_symbol", "organism", "protein_family"]
        required_dti_cols = ["drug_id", "target_id", "action"]
        
        for col in required_drug_cols:
            assert col in drugs_df.columns, f"Missing drug column: {col}"
        
        for col in required_target_cols:
            assert col in targets_df.columns, f"Missing target column: {col}"
        
        for col in required_dti_cols:
            assert col in dti_df.columns, f"Missing DTI column: {col}"
    
    def test_approved_boolean_parsed(self):
        """Test that approved boolean is correctly parsed."""
        xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"
        
        drugs_df, _, _ = parse_drugbank_xml(xml_path)
        
        # Check approved drugs
        approved_drugs = drugs_df[drugs_df["approved"]]
        assert len(approved_drugs) == 2, "Should have 2 approved drugs"
        
        # Check unapproved drugs
        unapproved_drugs = drugs_df[~drugs_df["approved"]]
        assert len(unapproved_drugs) == 1, "Should have 1 unapproved drug"
        
        # Check specific drug IDs
        assert "DB00001" in approved_drugs["drug_id"].values
        assert "DB00002" in approved_drugs["drug_id"].values
        assert "DB00003" in unapproved_drugs["drug_id"].values
    
    def test_human_filtering(self):
        """Test that only human targets are kept."""
        xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"
        
        _, targets_df, _ = parse_drugbank_xml(xml_path)
        
        # All targets should be human
        assert all(targets_df["organism"] == "Homo sapiens"), "All targets should be human"
        
        # Should have 2 unique human targets
        assert len(targets_df) == 2, "Should have 2 human targets"
        assert "P12345" in targets_df["target_id"].values
        assert "P67890" in targets_df["target_id"].values
        
        # Mouse targets should be excluded (by label, and by taxonomy ID)
        assert "P99999" not in targets_df["target_id"].values
        assert "P88888" not in targets_df["target_id"].values
    
    def test_missing_ids_excluded(self):
        """Test that drugs/targets with missing IDs are excluded."""
        xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"
        
        drugs_df, targets_df, dti_df = parse_drugbank_xml(xml_path)
        
        # All drugs should have valid IDs
        assert not drugs_df["drug_id"].isna().any(), "No drug should have missing ID"
        assert not drugs_df["drug_id"].str.strip().eq("").any(), "No drug should have empty ID"
        
        # All targets should have valid IDs
        assert not targets_df["target_id"].isna().any(), "No target should have missing ID"
        assert not targets_df["target_id"].str.strip().eq("").any(), ("No target should have "
                                                                      "empty ID")
        
        # All DTI edges should have valid endpoints
        assert not dti_df["drug_id"].isna().any(), "No DTI should have missing drug ID"
        assert not dti_df["target_id"].isna().any(), "No DTI should have missing target ID"
    
    def test_deduplication(self):
        """Test that duplicates are removed."""
        xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"
        
        drugs_df, targets_df, dti_df = parse_drugbank_xml(xml_path)
        
        # Check no duplicate drugs
        assert len(drugs_df) == len(drugs_df.drop_duplicates(subset=["drug_id"]))
        
        # Check no duplicate targets
        assert len(targets_df) == len(targets_df.drop_duplicates(subset=["target_id"]))
        
        # Check no duplicate DTI edges
        assert len(dti_df) == len(dti_df.drop_duplicates(subset=["drug_id", "target_id"]))
    
    def test_atc_code_parsing(self):
        """Test ATC code parsing."""
        xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"
        
        drugs_df, _, _ = parse_drugbank_xml(xml_path)
        
        # Check ATC L1 parsing
        drug1 = drugs_df[drugs_df["drug_id"] == "DB00001"].iloc[0]
        assert drug1["atc_l1"] == "A"
        assert drug1["atc_l2"] == "A01"
        
        drug2 = drugs_df[drugs_df["drug_id"] == "DB00002"].iloc[0]
        assert drug2["atc_l1"] == "B"
        assert drug2["atc_l2"] == "B02"
    
    def test_smiles_parsing(self):
        """Test SMILES parsing."""
        xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"
        
        drugs_df, _, _ = parse_drugbank_xml(xml_path)
        
        # Check SMILES for drug with calculated properties
        drug1 = drugs_df[drugs_df["drug_id"] == "DB00001"].iloc[0]
        assert drug1["smiles"] == "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        
        # Check SMILES for drug without calculated properties
        drug2 = drugs_df[drugs_df["drug_id"] == "DB00002"].iloc[0]
        assert pd.isna(drug2["smiles"])

    def test_drug_type_from_attribute(self):
        """Test that drug type comes from the <drug type=...> attribute."""
        xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"

        drugs_df, _, _ = parse_drugbank_xml(xml_path)

        types = dict(zip(drugs_df["drug_id"], drugs_df["type"]))
        assert types["DB00001"] == "small molecule"
        assert types["DB00002"] == "biotech"

    def test_nested_drug_references_ignored(self):
        """Test that <drug> references nested inside an entry are not entries."""
        xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"

        drugs_df, _, dti_df = parse_drugbank_xml(xml_path)

        assert sorted(drugs_df["drug_id"]) == ["DB00001", "DB00002", "DB00003"]
        assert "Nested reference" not in drugs_df["name"].values
        assert len(dti_df) == 3


def test_attribute_parsing():
    """Categories, ChemOnt classification and Pfam families (attribute model inputs)."""
    from datasources.drugbank_attributes import parse_attributes

    xml_path = Path(__file__).parent / "fixtures" / "tiny_drugbank.xml"
    drugs, targets = parse_attributes(xml_path)
    a = drugs.set_index("drug_id").loc["DB00001"]
    assert a["categories"] == "Test Category One|Test Category Two"
    assert a["class"] == "Test class" and a["direct_parent"] == "Test parent"
    assert drugs.set_index("drug_id").loc["DB00002", "categories"] == ""
    t = targets.set_index("target_id")
    assert t.loc["P12345", "pfams"] == "PF00001"
    assert "P99999" not in t.index and "P88888" not in t.index
