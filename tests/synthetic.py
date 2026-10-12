"""Small synthetic drug-target tables for tests (made up; not DrugBank data)."""

import numpy as np
import pandas as pd


def synthetic_tables(n_drugs=120, n_targets=60, n_blocks=4, n_edges=500, seed=0):
    """Block-structured random bipartite data: drugs mostly hit targets in their own block."""
    rng = np.random.default_rng(seed)
    drugs = [f"SD{i:04d}" for i in range(n_drugs)]
    targets = [f"ST{i:04d}" for i in range(n_targets)]
    db, tb = n_drugs // n_blocks, n_targets // n_blocks
    edges = set()
    while len(edges) < n_edges:
        b = rng.integers(n_blocks)
        d = drugs[b * db + rng.integers(db)]
        t = targets[b * tb + rng.integers(tb)] if rng.random() < 0.85 else targets[rng.integers(n_targets)]
        edges.add((d, t))
    drugs_df = pd.DataFrame({"drug_id": drugs, "name": [f"Synthetic drug {i}" for i in range(n_drugs)],
                             "type": "small molecule", "groups": "approved", "approved": True,
                             "atc_codes": None})
    targets_df = pd.DataFrame({"target_id": targets,
                               "name": [f"Synthetic target {i}" for i in range(n_targets)],
                               "gene_symbol": "", "organism": "Homo sapiens", "is_human": True})
    dti_df = pd.DataFrame(sorted(edges), columns=["drug_id", "target_id"]).assign(action="unknown")
    return drugs_df, targets_df, dti_df
