"""Attribute features: a link predictor that can score drugs with no known edges.

Each pair feature compares the drug with the target's known binders in the
message graph (so held-out edges never contribute), using attributes from the
DrugBank export rather than the graph:

    fp_max_binders        max Tanimoto similarity (Morgan, radius 2, 2048 bits)
                          between the drug and the target's binders
    fp_top3_binders       mean of the three highest such similarities
    fp_max_family         max Tanimoto to binders of other targets sharing a
                          Pfam family with the target
    category_max_jaccard  max Jaccard overlap of DrugBank categories with binders
    atc_max_jaccard       max Jaccard overlap of ATC level-4 codes with binders
    same_class_share      share of binders in the drug's ChemOnt class
    same_superclass_share share of binders in the drug's ChemOnt superclass
    has_structure         1 if the drug has a usable SMILES string

Missing values are -1 (no structure, no categories, no binders). These 8
features are appended to the 12 graph features, so for a drug with no edges
the graph half is uninformative and the attribute half carries the signal.
"""

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from pipeline.link_prediction import FEATURE_NAMES, DTILinkPredictor

RDLogger.DisableLog("rdApp.*")

ATTRIBUTE_FEATURES = [
    "fp_max_binders", "fp_top3_binders", "fp_max_family", "category_max_jaccard",
    "atc_max_jaccard", "same_class_share", "same_superclass_share", "has_structure",
]


def _split(s):
    return frozenset(x for x in (s if isinstance(s, str) else "").split("|") if x)


def _jaccard_max(a, others):
    if not a:
        return -1.0
    best = -1.0
    for b in others:
        if b:
            best = max(best, len(a & b) / len(a | b))
    return best


class AttributeLinkPredictor(DTILinkPredictor):
    """DTILinkPredictor with 8 attribute features appended (see module docstring)."""

    def __init__(self, drug_attrs, target_attrs, seed=42, train_negatives="degree_matched"):
        super().__init__(seed=seed, train_negatives=train_negatives)
        self.drug_attrs = drug_attrs.set_index("drug_id")
        self.target_attrs = target_attrs.set_index("target_id")

    def load_data(self, drugs_df, targets_df, dti_df, embeddings_df=None):
        super().load_data(drugs_df, targets_df, dti_df, embeddings_df)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.fps = {}
        for d, smi in zip(drugs_df["drug_id"], drugs_df["smiles"]):
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
            if mol is not None:
                self.fps[d] = gen.GetFingerprint(mol)
        atc = dict(zip(drugs_df["drug_id"], drugs_df["atc_codes"]))
        self.atc4 = {d: frozenset(c[:5] for c in (v if isinstance(v, str) else "").split(",")
                                  if len(c) >= 5)
                     for d, v in atc.items()}
        da = self.drug_attrs
        self.categories = {d: _split(c) for d, c in zip(da.index, da["categories"])}
        self.chem_class = dict(zip(da.index, da["class"]))
        self.chem_superclass = dict(zip(da.index, da["superclass"]))
        self.pfams = {t: _split(p) for t, p in zip(self.target_attrs.index, self.target_attrs["pfams"])}
        self.pfam_targets = {}
        for t, fams in self.pfams.items():
            for f in fams:
                self.pfam_targets.setdefault(f, set()).add(t)

    def split_edges(self, kind="random"):
        out = super().split_edges(kind)
        self._family_cache = {}
        return out

    def _binders(self, t):
        return list(self.train_graph.neighbors(t)) if t in self.train_graph else []

    def _family_binders(self, t):
        if t not in self._family_cache:
            related = set().union(*(self.pfam_targets.get(f, set()) for f in self.pfams.get(t, ()))) \
                if self.pfams.get(t) else set()
            related.discard(t)
            self._family_cache[t] = sorted({b for r in related for b in self._binders(r)})
        return self._family_cache[t]

    def _sims(self, d, others):
        fps = [self.fps[o] for o in others if o in self.fps and o != d]
        if d not in self.fps or not fps:
            return np.array([])
        return np.array(DataStructs.BulkTanimotoSimilarity(self.fps[d], fps))

    def attribute_features(self, d, t):
        binders = [b for b in self._binders(t) if b != d]
        sims = self._sims(d, binders)
        fam = self._sims(d, [b for b in self._family_binders(t) if b != d])
        cls, sup = self.chem_class.get(d, ""), self.chem_superclass.get(d, "")
        share = (lambda attr, val: -1.0 if not val or not binders
                 else float(np.mean([attr.get(b, "") == val for b in binders])))
        return np.array([
            sims.max() if sims.size else -1.0,
            np.sort(sims)[-3:].mean() if sims.size else -1.0,
            fam.max() if fam.size else -1.0,
            _jaccard_max(self.categories.get(d, frozenset()), [self.categories.get(b) for b in binders]),
            _jaccard_max(self.atc4.get(d, frozenset()), [self.atc4.get(b) for b in binders]),
            share(self.chem_class, cls),
            share(self.chem_superclass, sup),
            1.0 if d in self.fps else 0.0,
        ], dtype=float)

    def _features(self, pairs):
        base = super()._features(pairs)
        attrs = np.vstack([self.attribute_features(d, t) for d, t in pairs])
        return np.hstack([base, attrs])

    def evaluate_test(self):
        perf = super().evaluate_test()
        self.evaluation_results["feature_importances"] = dict(
            zip(FEATURE_NAMES + ATTRIBUTE_FEATURES, self.model.feature_importances_))
        return perf

    def score_candidates(self, node, candidates=None):
        raise NotImplementedError("the attribute model is evaluated only; the app uses the graph model")

    def predict_interaction(self, drug_id, target_id):
        raw = self._raw_scores([(drug_id, target_id)])
        return float(self._calibrated(raw)[0])
