"""Tests for the whole-drug holdout and the attribute model (synthetic data only)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))
from pipeline.attributes import ATTRIBUTE_FEATURES, AttributeLinkPredictor
from pipeline.link_prediction import DTILinkPredictor
from synthetic import synthetic_tables

SMILES = ["CCO", "CCN", "c1ccccc1O", "c1ccccc1N", "CC(=O)O", "CCCC", None]


def _tables():
    drugs, targets, dti = synthetic_tables()
    drugs = drugs.assign(smiles=[SMILES[i % len(SMILES)] for i in range(len(drugs))],
                         atc_codes=["A01AB01" if i % 3 == 0 else None for i in range(len(drugs))])
    drug_attrs = pd.DataFrame({"drug_id": drugs["drug_id"],
                               "categories": ["Cat A|Cat B" if i % 2 else "Cat A" for i in range(len(drugs))],
                               "class": ["Class " + str(i % 4) for i in range(len(drugs))],
                               "superclass": ["Super " + str(i % 2) for i in range(len(drugs))]})
    target_attrs = pd.DataFrame({"target_id": targets["target_id"],
                                 "pfams": ["PF%05d" % (i % 6) for i in range(len(targets))]})
    return (drugs, targets, dti), drug_attrs, target_attrs


def test_cold_drug_split_removes_every_edge_of_test_and_train_drugs():
    p = DTILinkPredictor(seed=3, train_negatives="degree_matched")
    p.load_data(*synthetic_tables())
    p.split_edges("cold_drug")
    for d in set(p.test_drugs) | set(p.train_drugs):
        assert p.train_graph.degree(d) == 0
    assert {d for d, _ in p.test_edges} == set(p.test_drugs)
    assert not set(p.test_drugs) & set(p.train_drugs)
    assert {d for d, _ in p.val_edges} == p.val_drugs
    negs = p.sample_negatives(p.test_edges)
    assert all(d == nd for (d, _), (nd, _) in zip(p.test_edges, negs))


def test_attribute_model_same_split_and_negatives_as_graph_model():
    tables, da, ta = _tables()
    a = DTILinkPredictor(seed=3, train_negatives="degree_matched")
    b = AttributeLinkPredictor(da, ta, seed=3)
    for p in (a, b):
        p.load_data(*tables)
        p.split_edges("cold_drug")
        p.fit()
        p.validate()
    assert a.fit_neg == b.fit_neg and a.val_neg == b.val_neg
    assert b._features(b.fit_edges[:4]).shape[1] == 12 + len(ATTRIBUTE_FEATURES)


def test_attribute_features_use_message_graph_only():
    tables, da, ta = _tables()
    p = AttributeLinkPredictor(da, ta, seed=3)
    p.load_data(*tables)
    p.split_edges("cold_drug")
    d, t = p.test_edges[0]
    f = dict(zip(ATTRIBUTE_FEATURES, p.attribute_features(d, t)))
    assert d not in p._binders(t)          # its own (hidden) edge is not a binder
    assert -1.0 <= f["fp_max_binders"] <= 1.0
    # identical structure to a binder gives similarity 1
    binders = p._binders(t)
    twin = next((b for b in binders if p.fps.get(b) is not None and d in p.fps
                 and tables[0].set_index("drug_id").at[b, "smiles"]
                 == tables[0].set_index("drug_id").at[d, "smiles"]), None)
    if twin is not None:
        assert np.isclose(f["fp_max_binders"], 1.0)
