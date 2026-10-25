"""Tests for graph construction (synthetic data only)."""

import os
import sys

import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))
from synthetic import synthetic_tables

from pipeline.build_graph import build_bipartite_graph, graph_stats


def test_bipartite_and_typed() -> None:
    drugs, targets, dti = synthetic_tables()
    G = build_bipartite_graph(drugs, targets, dti)
    assert G.number_of_edges() == len(dti)
    for a, b in G.edges():
        assert {G.nodes[a]["node_type"], G.nodes[b]["node_type"]} == {"drug", "target"}


def test_unknown_endpoints_dropped() -> None:
    drugs, targets, dti = synthetic_tables()
    extra = pd.DataFrame([{"drug_id": "NOPE", "target_id": targets["target_id"][0]}])
    G = build_bipartite_graph(drugs, targets, pd.concat([dti, extra]))
    assert "NOPE" not in G
    assert all("node_type" in d for _, d in G.nodes(data=True))


def test_connected_only_and_stats() -> None:
    drugs, targets, dti = synthetic_tables()
    G = build_bipartite_graph(drugs, targets, dti, connected_only=True)
    s = graph_stats(G)
    assert s["drugs"] == s["drugs_with_edges"] and s["targets"] == s["targets_with_edges"]


def test_most_connected_no_duplicates_and_ranked() -> None:
    from pipeline.build_graph import most_connected

    drugs, targets, dti = synthetic_tables()
    drugs = pd.concat([drugs, drugs.iloc[[0]]])
    dti = pd.concat([dti, dti.iloc[[0]]])
    top = most_connected(dti, drugs, "drug_id", n=10)
    assert top["drug_id"].is_unique
    assert list(top["connections"]) == sorted(top["connections"], reverse=True)
    expected = dti.drop_duplicates()["drug_id"].value_counts()
    assert all(expected[d] == c for d, c in zip(top["drug_id"], top["connections"], strict=False))
