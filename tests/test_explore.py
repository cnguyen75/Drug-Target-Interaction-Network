"""Tests for explore-view helpers (synthetic data only)."""

import os
import sys

import networkx as nx
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))
from synthetic import synthetic_tables

from pipeline.build_graph import build_bipartite_graph
from pipeline.explore import evidence, featured_mask, has_carbon, neighbourhood, radial_layout


def test_has_carbon() -> None:
    assert has_carbon("CCO") and has_carbon("c1ccccc1") and has_carbon("N[C@@H](C)C(O)=O")
    assert (
        not has_carbon("[Cu]") and not has_carbon("[Cl-].[Cl-].[Zn++]") and not has_carbon("[Ca++]")
    )


def test_featured_mask_excludes_ions_salts_nutraceuticals() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "organic",
                "approved": True,
                "type": "small molecule",
                "groups": "approved",
                "smiles": "CCO",
            },
            {
                "name": "ion",
                "approved": True,
                "type": "small molecule",
                "groups": "approved",
                "smiles": "[Cu]",
            },
            {
                "name": "salt",
                "approved": True,
                "type": "small molecule",
                "groups": "approved",
                "smiles": "[Zn++].CC([O-])=O",
            },
            {
                "name": "nutra",
                "approved": True,
                "type": "small molecule",
                "groups": "approved,nutraceutical",
                "smiles": "NCC(O)=O",
            },
            {
                "name": "biotech",
                "approved": True,
                "type": "biotech",
                "groups": "approved",
                "smiles": None,
            },
            {
                "name": "unapproved",
                "approved": False,
                "type": "small molecule",
                "groups": "experimental",
                "smiles": "CCO",
            },
        ]
    )
    assert list(df[featured_mask(df)]["name"]) == ["organic"]


def _graph() -> nx.Graph:
    return build_bipartite_graph(*synthetic_tables())


def test_neighbourhood_and_layout() -> None:
    G = _graph()
    drug = max((n for n, d in G.nodes(data=True) if d["node_type"] == "drug"), key=G.degree)
    unlinked = [
        t for t, d in G.nodes(data=True) if d["node_type"] == "target" and not G.has_edge(drug, t)
    ]
    scores = {t: i / len(unlinked) for i, t in enumerate(unlinked)}
    nb = neighbourhood(G, drug, scores, min_score=0.5)
    assert nb["known_total"] == G.degree(drug)
    assert all(s >= 0.5 for _, s in nb["predicted"]) and len(nb["predicted"]) <= 5
    assert all(not G.has_edge(drug, p) for p, _ in nb["predicted"])
    pos = radial_layout(nb)
    assert set(pos) >= {drug, *nb["known"], *nb["related"], *(p for p, _ in nb["predicted"])}


def test_evidence_mentions_l3_support() -> None:
    G = _graph()
    drug = max((n for n, d in G.nodes(data=True) if d["node_type"] == "drug"), key=G.degree)
    for t in G.nodes:
        if G.nodes[t]["node_type"] == "target" and not G.has_edge(drug, t):
            text = evidence(G, drug, t)
            assert text.endswith(".")
            if text.startswith("Bound by"):
                return
    raise AssertionError("expected at least one length-3 path in the synthetic graph")


def test_attribute_evidence_lines() -> None:
    from test_attributes import _tables

    from pipeline.attributes import AttributeLinkPredictor

    tables, da, ta = _tables()
    p = AttributeLinkPredictor(da, ta, seed=3)
    p.load_data(*tables)
    G = p.G
    seen = set()
    for d in sorted(n for n in G if G.nodes[n]["node_type"] == "drug")[:30]:
        for t in sorted(n for n in G if G.nodes[n]["node_type"] == "target"):
            if G.has_edge(d, t):
                continue
            text = evidence(G, d, t, attr=p)
            assert text.endswith(".")
            seen.add(text.split(" ")[0])
            # target-centred wording names the target rather than "it"
            assert "binds it" not in evidence(G, t, d, attr=p)
    assert "Structurally" in seen


def test_no_evidence_wording() -> None:
    from test_attributes import _tables

    from pipeline.attributes import AttributeLinkPredictor
    from pipeline.explore import NO_EVIDENCE

    tables, da, ta = _tables()
    p = AttributeLinkPredictor(da, ta, seed=3)
    p.load_data(*tables)
    G = p.G
    lonely = "LONELY"
    G.add_node(lonely, node_type="drug", name="Lonely")
    t = sorted(n for n in G if G.nodes[n]["node_type"] == "target")[0]
    assert evidence(G, lonely, t, attr=p) == NO_EVIDENCE
