"""Configuration and path management for the DTI Explorer."""

import os
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory."""
    current = Path(__file__).parent
    while current.parent != current:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (pyproject.toml)")


def ensure_data_dir() -> Path:
    """Ensure the data directory exists and return its path."""
    data_dir = get_repo_root() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# Repository root and data directory
REPO_ROOT = get_repo_root()
DATA_DIR = ensure_data_dir()

# Raw DrugBank export: the .xml.zip as downloaded (or an extracted .xml).
# It lives outside the repo; override with DRUGBANK_PATH.
RAW_XML = Path(os.environ.get(
    "DRUGBANK_PATH",
    Path.home() / "Documents" / "drugbank" / "drugbank_all_full_database.xml.zip",
)).expanduser()

# Parsed tables (parquet), written by src/datasources/drugbank_loader.py
DRUGS_FP = REPO_ROOT / "data_raw" / "drugs_full.parquet"
TARGETS_FP = REPO_ROOT / "data_raw" / "targets_full.parquet"
DTI_FP = REPO_ROOT / "data_raw" / "dti_full.parquet"

# Graph paths (gpickle)
G_BIP_FP = DATA_DIR / "graph_bipartite.gpickle"
G_DRUG_FP = DATA_DIR / "graph_drug_projection.gpickle"
G_TARG_FP = DATA_DIR / "graph_target_projection.gpickle"

# Embedding paths
EMB_ALL_FP = DATA_DIR / "embeddings_all.parquet"

