"""Build the drug-target bipartite graph from the parsed tables.

python -m src.pipeline.build_graph      # prints graph statistics
"""

import os
import sys

import networkx as nx
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def build_bipartite_graph(
    drugs_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    dti_df: pd.DataFrame,
    connected_only: bool = False,
) -> nx.Graph:
    """Bipartite graph: drug and target nodes (node_type, name), one edge per known interaction.

    Interactions whose drug or target is missing from the tables are dropped,
    so every node keeps its node_type attribute.
    """
    G = nx.Graph()
    G.add_nodes_from(
        (d, {"node_type": "drug", "name": n})
        for d, n in zip(drugs_df["drug_id"], drugs_df["name"], strict=False)
    )
    G.add_nodes_from(
        (t, {"node_type": "target", "name": n})
        for t, n in zip(targets_df["target_id"], targets_df["name"], strict=False)
    )
    drugs, targets = set(drugs_df["drug_id"]), set(targets_df["target_id"])
    G.add_edges_from(
        (d, t)
        for d, t in zip(dti_df["drug_id"], dti_df["target_id"], strict=False)
        if d in drugs and t in targets
    )
    if connected_only:
        G.remove_nodes_from([n for n in list(G.nodes()) if G.degree(n) == 0])
    return G


def most_connected(
    dti_df: pd.DataFrame, table: pd.DataFrame, id_col: str, n: int = 25
) -> pd.DataFrame:
    """Top n rows of `table` by number of distinct interaction partners, most connected first.

    Duplicate interaction rows are counted once and duplicate ids in `table`
    appear once, so a drug or target can never be listed twice.
    """
    other = "target_id" if id_col == "drug_id" else "drug_id"
    counts = (
        dti_df.drop_duplicates(subset=[id_col, other])[id_col].value_counts().rename("connections")
    )
    unique = table.drop_duplicates(subset=[id_col]).set_index(id_col)
    top = (
        counts[counts.index.isin(unique.index)]
        .sort_index()
        .sort_values(ascending=False, kind="stable")
        .head(n)
    )
    return unique.loc[top.index].assign(connections=top.values).reset_index()


def graph_stats(G: nx.Graph) -> dict[str, int]:
    drugs = [n for n, d in G.nodes(data=True) if d["node_type"] == "drug"]
    targets = [n for n, d in G.nodes(data=True) if d["node_type"] == "target"]
    return {
        "drugs": len(drugs),
        "targets": len(targets),
        "edges": G.number_of_edges(),
        "drugs_with_edges": sum(1 for n in drugs if G.degree(n) > 0),
        "targets_with_edges": sum(1 for n in targets if G.degree(n) > 0),
        "connected_components_with_edges": sum(1 for c in nx.connected_components(G) if len(c) > 1),
    }


def main() -> None:
    from config import DRUGS_FP, DTI_FP, TARGETS_FP

    G = build_bipartite_graph(
        pd.read_parquet(DRUGS_FP), pd.read_parquet(TARGETS_FP), pd.read_parquet(DTI_FP)
    )
    for k, v in graph_stats(G).items():
        print(f"{k}: {v:,}")


if __name__ == "__main__":
    main()
