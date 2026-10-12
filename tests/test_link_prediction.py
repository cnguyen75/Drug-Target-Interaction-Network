"""Tests for the evaluation protocol (synthetic data only)."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))
from pipeline.link_prediction import DTILinkPredictor, bootstrap_ci
from synthetic import synthetic_tables


def _predictor(split):
    p = DTILinkPredictor(seed=1)
    p.load_data(*synthetic_tables())
    p.split_edges(split)
    return p


def test_held_out_edges_not_in_message_graph():
    for split in ("random", "drug_holdout"):
        p = _predictor(split)
        for d, t in p.train_edges + p.test_edges:
            assert not p.train_graph.has_edge(d, t)
        assert not set(p.train_edges) & set(p.test_edges)


def test_drug_holdout_keeps_an_edge_per_test_drug():
    p = _predictor("drug_holdout")
    assert not set(p.test_drugs) & set(p.train_drugs)
    for d in p.test_drugs:
        assert p.train_graph.degree(d) >= 1
        assert any(e[0] == d for e in p.test_edges)


def test_degree_matched_negatives_same_drug_and_not_edges():
    p = _predictor("drug_holdout")
    negs = p.sample_negatives(p.test_edges)
    assert len(negs) == len(p.test_edges) == len(set(negs))
    for (d, t), (nd, nt) in zip(p.test_edges, negs):
        assert d == nd and not p.G.has_edge(nd, nt)


def test_uniform_negatives_disjoint_from_exclude():
    p = _predictor("random")
    train_neg = p.sample_negatives(p.train_edges)
    test_neg = p.sample_negatives(p.test_edges, exclude=set(train_neg))
    assert not set(train_neg) & set(test_neg)
    assert not any(p.G.has_edge(d, t) for d, t in test_neg)


def test_seeded_and_reproducible():
    a, b = _predictor("random"), _predictor("random")
    assert a.test_edges == b.test_edges


def test_train_and_baselines():
    p = _predictor("random")
    perf = p.train_model()
    base = p.evaluation_results["baselines"]
    assert 0 <= perf["auc"] <= 1
    lo, hi = perf["roc_auc_ci95"]
    assert lo <= perf["auc"] <= hi
    # Ties for every pair: chance and common neighbours score exactly 0.5.
    assert base["chance"]["roc_auc"] == 0.5
    assert base["common_neighbors"]["roc_auc"] == 0.5


def test_bootstrap_ci_brackets_point_estimate():
    y = [0, 1] * 50
    s = [0.2, 0.8] * 50
    ci = bootstrap_ci(y, s, n_boot=100)
    assert ci["roc_auc_ci95"] == [1.0, 1.0]


def test_score_candidates_matches_per_pair_features():
    from synthetic import synthetic_tables
    from pipeline.embeddings import embed_graph

    p = _predictor("final")
    p.set_embeddings(embed_graph(p.train_graph, dims=128, walks=2, walklen=10))
    p.train_model()
    drug = sorted(p.drug_nodes)[3]
    target = sorted(p.target_nodes)[5]
    fast = p.score_candidates(drug)
    for t in list(fast)[:15]:
        assert abs(fast[t] - p.predict_interaction(drug, t)) < 1e-9
    fast_t = p.score_candidates(target)
    for d in list(fast_t)[:15]:
        assert abs(fast_t[d] - p.predict_interaction(d, target)) < 1e-9
    assert not any(p.G.has_edge(drug, t) for t in fast)
