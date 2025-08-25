"""Configuration and path management for the DTI Explorer."""

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

# Raw data paths
RAW_XML = REPO_ROOT / "data_raw" / "drugbank.xml"

# Processed data paths (parquet)
DRUGS_FP = DATA_DIR / "drugs.parquet"
TARGETS_FP = DATA_DIR / "targets.parquet"
DTI_FP = DATA_DIR / "dti.parquet"

# Graph paths (gpickle)
G_BIP_FP = DATA_DIR / "graph_bipartite.gpickle"
G_DRUG_FP = DATA_DIR / "graph_drug_projection.gpickle"
G_TARG_FP = DATA_DIR / "graph_target_projection.gpickle"

# Embedding paths
EMB_ALL_FP = DATA_DIR / "embeddings_all.parquet"

