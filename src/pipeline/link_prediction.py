"""Link prediction for drug-target interactions.

Two evaluation splits are supported (see split_edges):

random
    Known edges are shuffled with a fixed seed and split 60/20/20 into message
    edges (the graph every feature and embedding is computed on), train
    positives and test positives. Negatives are uniformly random drug-target
    pairs with no known interaction, drawn only from drugs and targets that
    have at least one interaction. This is a random edge split: DrugBank has
    no per-interaction dates, so nothing here is temporal.

drug_holdout
    The stricter split. A true cold-start split (whole drugs unseen) cannot be
    scored by this model: Node2Vec only embeds nodes it walked over, so a drug
    with no edges in the message graph has no embedding and zero degree, and
    every candidate target would be ranked by target popularity alone.
    Instead, 20% of drugs with at least two targets are test drugs and half of
    each test drug's edges (at least one) are hidden; another 20% of drugs
    supply training positives the same way. Each positive (d, t) gets a
    negative (d, t') for the same drug, with t' drawn from targets in the same
    degree bin as t, so the model cannot win by preferring popular targets or
    by recognising which drugs appear in the positive set.

In both splits, train and test positives are absent from the message graph,
so the model is trained on the same feature distribution it is tested on.
"""

import math
import pickle

import networkx as nx
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc as area_under_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "drug_degree", "target_degree", "jaccard", "preferential_attachment",
    "resource_allocation", "adamic_adar", "shortest_path", "common_neighbors",
    "total_neighbors", "embedding_dot", "drug_embedding_mean", "target_embedding_mean",
]

SPLIT_DESCRIPTIONS = {
    "random": "Random edge split 60/20/20 (message/train/test), uniform random negatives",
    "drug_holdout": ("Per-drug holdout: 20% of drugs with 2+ targets are test drugs with half "
                     "their edges hidden; negatives use the same drug and a degree-matched target"),
    "final": "All edges: 80% message graph, 20% training positives; no test set",
}


def _ranking_metrics(y, scores):
    return roc_auc_score(y, scores), average_precision_score(y, scores)


def bootstrap_ci(y, scores, n_boot=1000, seed=42, alpha=0.05):
    """Percentile bootstrap CIs for ROC AUC and AP, resampling test pairs with replacement."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    aucs, aps = [], []
    for _ in range(n_boot):
        idx = rng.integers(len(y), size=len(y))
        if y[idx].min() == y[idx].max():
            continue
        a, p = _ranking_metrics(y[idx], scores[idx])
        aucs.append(a)
        aps.append(p)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "roc_auc_ci95": [float(np.percentile(aucs, lo)), float(np.percentile(aucs, hi))],
        "ap_ci95": [float(np.percentile(aps, lo)), float(np.percentile(aps, hi))],
        "n_boot": len(aucs),
    }


class DTILinkPredictor:
    """Drug-Target Interaction Link Predictor using machine learning."""

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.scaler = StandardScaler()
        self.drug_nodes = set()
        self.target_nodes = set()
        self.split = None
        self.train_graph = None  # the message graph features are computed on
        self.message_edges = []
        self.train_edges = []
        self.test_edges = []
        self.embeddings_df = None

    def load_data(self, drugs_df, targets_df, dti_df, embeddings_df=None):
        """Load data for link prediction."""
        from pipeline.build_graph import build_bipartite_graph

        self.drugs_df = drugs_df
        self.targets_df = targets_df
        self.dti_df = dti_df
        self.set_embeddings(embeddings_df)
        self.G = build_bipartite_graph(drugs_df, targets_df, dti_df)
        self.drug_nodes = {n for n, d in self.G.nodes(data=True) if d["node_type"] == "drug"}
        self.target_nodes = {n for n, d in self.G.nodes(data=True) if d["node_type"] == "target"}
        print(f"Built graph: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")

    def set_embeddings(self, embeddings_df):
        """Use node embeddings indexed by node_id (or with a node_id column)."""
        if embeddings_df is not None and "node_id" in embeddings_df.columns:
            embeddings_df = embeddings_df.set_index("node_id")
        self.embeddings_df = embeddings_df

    def _sorted_edges(self):
        return sorted((a, b) if a in self.drug_nodes else (b, a) for a, b in self.G.edges())

    # ----- splits -------------------------------------------------------------

    def split_edges(self, kind="random"):
        """Split known edges into message / train / test sets (seeded). Returns (train, test)."""
        edges = self._sorted_edges()
        if kind == "random":
            edges = [edges[i] for i in self.rng.permutation(len(edges))]
            n_test = int(len(edges) * 0.2)
            n_train = int(len(edges) * 0.2)
            test, train = edges[:n_test], edges[n_test:n_test + n_train]
        elif kind == "final":
            edges = [edges[i] for i in self.rng.permutation(len(edges))]
            test, train = [], edges[:int(len(edges) * 0.2)]
        elif kind == "drug_holdout":
            train, test = self._drug_holdout(edges)
        else:
            raise ValueError(f"unknown split {kind!r}")

        held = set(train) | set(test)
        self.split = kind
        self.train_edges, self.test_edges = train, test
        self.message_edges = [e for e in edges if e not in held]
        self.train_graph = nx.Graph()
        self.train_graph.add_nodes_from(self.G.nodes(data=True))
        self.train_graph.add_edges_from(self.message_edges)

        print(f"{kind} split (seed {self.seed}): {len(self.message_edges)} message edges, "
              f"{len(train)} train edges, {len(test)} test edges")
        return train, test

    def _drug_holdout(self, edges, drug_frac=0.2, edge_frac=0.5):
        by_drug = {}
        for d, t in edges:
            by_drug.setdefault(d, []).append((d, t))
        eligible = sorted(d for d, es in by_drug.items() if len(es) >= 2)
        order = [eligible[i] for i in self.rng.permutation(len(eligible))]
        n = int(len(order) * drug_frac)
        self.test_drugs, self.train_drugs = order[:n], order[n:2 * n]

        def hide(drugs):
            hidden = []
            for d in drugs:
                es = by_drug[d]
                k = max(1, math.floor(len(es) * edge_frac))
                hidden += [es[i] for i in self.rng.permutation(len(es))[:k]]
            return hidden

        return hide(self.train_drugs), hide(self.test_drugs)

    # ----- negatives ----------------------------------------------------------

    def sample_negatives(self, positives, exclude=frozenset()):
        """One negative per positive, following the current split's protocol."""
        if self.split == "drug_holdout":
            return self._degree_matched_negatives(positives, exclude)
        return self._uniform_negatives(len(positives), exclude)

    def _uniform_negatives(self, n, exclude):
        drugs = sorted(d for d in self.drug_nodes if self.G.degree(d) > 0)
        targets = sorted(t for t in self.target_nodes if self.G.degree(t) > 0)
        pairs = set()
        while len(pairs) < n:
            for i, j in zip(self.rng.integers(len(drugs), size=n),
                            self.rng.integers(len(targets), size=n)):
                pair = (drugs[i], targets[j])
                if pair in pairs or pair in exclude or self.G.has_edge(*pair):
                    continue
                pairs.add(pair)
                if len(pairs) == n:
                    break
        return sorted(pairs)

    def _degree_bin(self, t):
        return int(math.log2(self.G.degree(t)))

    def _degree_matched_negatives(self, positives, exclude):
        bins = {}
        for t in sorted(self.target_nodes):
            if self.G.degree(t) > 0:
                bins.setdefault(self._degree_bin(t), []).append(t)
        used = set(exclude)
        negatives = []
        for d, t in positives:
            b = self._degree_bin(t)
            pool = bins[b]
            for _ in range(50):
                cand = pool[self.rng.integers(len(pool))]
                if (d, cand) not in used and not self.G.has_edge(d, cand):
                    break
            else:
                # Bin exhausted for this drug: widen to neighbouring bins.
                pool = [x for k in (b - 1, b, b + 1) for x in bins.get(k, [])
                        if (d, x) not in used and not self.G.has_edge(d, x)]
                cand = pool[self.rng.integers(len(pool))]
            used.add((d, cand))
            negatives.append((d, cand))
        return negatives

    # ----- features -----------------------------------------------------------

    def extract_features_safe(self, drug_id, target_id, use_train_graph=True):
        """Extract features for a drug-target pair from the message graph."""
        graph = self.train_graph if use_train_graph else self.G

        drug_degree = graph.degree(drug_id) if drug_id in graph else 0
        target_degree = graph.degree(target_id) if target_id in graph else 0

        drug_neighbors = set(graph.neighbors(drug_id)) if drug_id in graph else set()
        target_neighbors = set(graph.neighbors(target_id)) if target_id in graph else set()
        common = drug_neighbors & target_neighbors
        union = drug_neighbors | target_neighbors

        # In a bipartite graph a drug and a target never share a neighbour, so
        # these four are always 0. Kept so the feature set matches the original.
        jaccard = len(common) / len(union) if union else 0.0
        resource_allocation = sum(1 / graph.degree(n) for n in common)
        adamic_adar = sum(1 / np.log(graph.degree(n)) for n in common if graph.degree(n) > 1)
        common_neighbors = len(common)

        preferential_attachment = drug_degree * target_degree

        try:
            shortest_path = nx.shortest_path_length(graph, drug_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            shortest_path = -1

        drug_embedding = np.zeros(128)
        target_embedding = np.zeros(128)
        embedding_dot = 0.0
        if self.embeddings_df is not None:
            if drug_id in self.embeddings_df.index:
                drug_embedding = self.embeddings_df.loc[drug_id].values
            if target_id in self.embeddings_df.index:
                target_embedding = self.embeddings_df.loc[target_id].values
            embedding_dot = float(np.dot(drug_embedding, target_embedding))

        return np.array([
            drug_degree, target_degree, jaccard, preferential_attachment,
            resource_allocation, adamic_adar, shortest_path, common_neighbors,
            len(union), embedding_dot,
            drug_embedding.mean(), target_embedding.mean(),
        ], dtype=float)

    def _features(self, pairs):
        return np.vstack([self.extract_features_safe(d, t) for d, t in pairs])

    # ----- training and evaluation -------------------------------------------

    def train_model(self):
        """Fit on the current split's training pairs; evaluate on its test pairs if any."""
        if self.train_graph is None:
            self.split_edges("random")

        train_neg = self.sample_negatives(self.train_edges)
        X = self._features(self.train_edges + train_neg)
        y = np.array([1] * len(self.train_edges) + [0] * len(train_neg))
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.seed, n_jobs=-1)
        self.model.fit(self.scaler.fit_transform(X), y)

        perf = {
            "split": self.split,
            "split_description": SPLIT_DESCRIPTIONS[self.split],
            "message_edges": len(self.message_edges),
            "train_positives": len(self.train_edges),
            "train_negatives": len(train_neg),
        }
        self.evaluation_results = {"model_performance": perf}
        if not self.test_edges:
            return perf

        test_neg = self.sample_negatives(self.test_edges, exclude=set(train_neg))
        pairs = self.test_edges + test_neg
        y_test = np.array([1] * len(self.test_edges) + [0] * len(test_neg))
        proba = self.model.predict_proba(self.scaler.transform(self._features(pairs)))[:, 1]

        roc, ap = _ranking_metrics(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        perf.update({
            "auc": roc,
            "ap_score": ap,
            "pr_auc": area_under_curve(recall, precision),
            "brier_score": float(np.mean((proba - y_test) ** 2)),
            "test_positives": len(self.test_edges),
            "test_negatives": len(test_neg),
            **bootstrap_ci(y_test, proba, seed=self.seed),
        })
        if self.split == "drug_holdout":
            perf["test_drugs"] = len(self.test_drugs)

        self.evaluation_results.update({
            "baselines": self.compare_baselines(pairs, y_test),
            "calibration": {"fraction_of_positives": frac_pos,
                            "mean_predicted_value": mean_pred},
            "feature_importances": dict(zip(FEATURE_NAMES, self.model.feature_importances_)),
            "test_pairs": pairs,
            "test_labels": y_test,
            "test_scores": proba,
        })
        print(f"[{self.split}] ROC AUC {roc:.4f} {perf['roc_auc_ci95']}, AP {ap:.4f} {perf['ap_ci95']}")
        return perf

    def _l3_paths(self, drug_id, target_id):
        """Number of drug-target'-drug'-target paths in the message graph."""
        g = self.train_graph
        drug_targets = set(g.neighbors(drug_id))
        return sum(len(drug_targets & set(g.neighbors(d2))) for d2 in g.neighbors(target_id))

    def compare_baselines(self, pairs, y_test):
        """Unsupervised baselines scored on the message graph, with bootstrap CIs.

        Common neighbours is 0 for every drug-target pair in a bipartite graph
        (a drug's neighbours are targets, a target's are drugs), so it ties
        every pair and scores exactly like chance. It is reported as computed.
        Length-3 paths is the bipartite analogue of common neighbours.
        """
        g = self.train_graph
        scores = {
            "chance": [0.0] * len(pairs),
            "common_neighbors": [len(set(g.neighbors(d)) & set(g.neighbors(t))) for d, t in pairs],
            "preferential_attachment": [g.degree(d) * g.degree(t) for d, t in pairs],
            "l3_paths": [self._l3_paths(d, t) for d, t in pairs],
        }
        results = {}
        for name, s in scores.items():
            roc, ap = _ranking_metrics(y_test, s)
            results[name] = {"roc_auc": roc, "ap_score": ap, **bootstrap_ci(y_test, s, seed=self.seed)}
        return results

    def predict_interaction(self, drug_id, target_id):
        """Predict interaction probability for a drug-target pair."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        features = self.extract_features_safe(drug_id, target_id, use_train_graph=True)
        return self.model.predict_proba(self.scaler.transform(features.reshape(1, -1)))[0, 1]

    def score_pairs(self, pairs):
        """Interaction probabilities for many (drug, target) pairs."""
        return self.model.predict_proba(self.scaler.transform(self._features(pairs)))[:, 1]

    def score_candidates(self, node, candidates=None):
        """Scores for every unlinked partner of one node, as {partner_id: score}.

        For a drug the partners are targets; for a target, drugs. Partners
        already linked in the full graph are excluded. Features are computed
        exactly as extract_features_safe does, but with one BFS from `node`
        instead of one per pair. In a bipartite graph a drug and a target share
        no neighbours, so the common-neighbour features are 0 and the
        neighbourhood union is the sum of the two degrees.
        """
        is_drug = node in self.drug_nodes
        pool = self.target_nodes if is_drug else self.drug_nodes
        if candidates is None:
            candidates = [n for n in sorted(pool)
                          if self.G.degree(n) > 0 and not self.G.has_edge(node, n)]
        if not candidates:
            return {}
        g = self.train_graph
        dist = nx.single_source_shortest_path_length(g, node)
        k = g.degree(node)
        other_deg = np.array([g.degree(c) for c in candidates], dtype=float)
        sp = np.array([dist.get(c, -1) for c in candidates], dtype=float)

        emb = self.embeddings_df
        node_vec = np.zeros(128)
        cand_vecs = np.zeros((len(candidates), 128))
        if emb is not None:
            if node in emb.index:
                node_vec = emb.loc[node].values
            present = [i for i, c in enumerate(candidates) if c in emb.index]
            if present:
                cand_vecs[present] = emb.loc[[candidates[i] for i in present]].values
        dot = cand_vecs @ node_vec if emb is not None else np.zeros(len(candidates))

        n = len(candidates)
        drug_deg = np.full(n, k, dtype=float) if is_drug else other_deg
        target_deg = other_deg if is_drug else np.full(n, k, dtype=float)
        drug_mean = np.full(n, node_vec.mean()) if is_drug else cand_vecs.mean(axis=1)
        target_mean = cand_vecs.mean(axis=1) if is_drug else np.full(n, node_vec.mean())
        zeros = np.zeros(n)
        X = np.column_stack([drug_deg, target_deg, zeros, drug_deg * target_deg, zeros, zeros,
                             sp, zeros, drug_deg + target_deg, dot, drug_mean, target_mean])
        scores = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        return dict(zip(candidates, scores))

    def save_model(self, filepath):
        """Save the trained model."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        """Load a trained model saved with save_model."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
