"""Link prediction for drug-target interactions.

Evaluation protocol (see train_model):
  Known edges are shuffled with a fixed seed and split three ways:
    - message edges (60%): the graph every feature and embedding is computed on
    - train edges (20%): positive training examples, absent from the message graph
    - test edges (20%): positive test examples, absent from the message graph
  Train and test positives are therefore both unseen by the features, so the
  model is trained on the same feature distribution it is tested on. Negatives
  are random drug-target pairs with no known interaction, drawn only from drugs
  and targets that have at least one interaction, and train/test negatives do
  not overlap.

  DrugBank has no per-interaction dates, so this is a random split, not a
  temporal one.
"""

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


class DTILinkPredictor:
    """Drug-Target Interaction Link Predictor using machine learning."""

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.scaler = StandardScaler()
        self.drug_nodes = set()
        self.target_nodes = set()
        self.train_graph = None  # the message graph features are computed on
        self.test_edges = None
        self.embeddings_df = None

    def load_data(self, drugs_df, targets_df, dti_df, embeddings_df=None):
        """Load data for link prediction."""
        self.drugs_df = drugs_df
        self.targets_df = targets_df
        self.dti_df = dti_df
        self.set_embeddings(embeddings_df)

        self.G = nx.Graph()
        for drug_id, name in zip(drugs_df["drug_id"], drugs_df["name"]):
            self.G.add_node(drug_id, node_type="drug", name=name)
            self.drug_nodes.add(drug_id)
        for target_id, name in zip(targets_df["target_id"], targets_df["name"]):
            self.G.add_node(target_id, node_type="target", name=name)
            self.target_nodes.add(target_id)
        self.G.add_edges_from(zip(dti_df["drug_id"], dti_df["target_id"]))

        print(f"Built graph: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")

    def set_embeddings(self, embeddings_df):
        """Use node embeddings indexed by node_id (or with a node_id column)."""
        if embeddings_df is not None and "node_id" in embeddings_df.columns:
            embeddings_df = embeddings_df.set_index("node_id")
        self.embeddings_df = embeddings_df

    @staticmethod
    def _canonical(edge, drug_nodes):
        a, b = edge
        return (a, b) if a in drug_nodes else (b, a)

    def random_split_edges(self, train_ratio=0.2, test_ratio=0.2):
        """Randomly split known edges into message / train / test sets (seeded)."""
        edges = sorted(self._canonical(e, self.drug_nodes) for e in self.G.edges())
        order = self.rng.permutation(len(edges))
        edges = [edges[i] for i in order]

        n_test = int(len(edges) * test_ratio)
        n_train = int(len(edges) * train_ratio)
        test_edges = edges[:n_test]
        train_edges = edges[n_test:n_test + n_train]
        message_edges = edges[n_test + n_train:]

        self.train_graph = nx.Graph()
        self.train_graph.add_nodes_from(self.G.nodes(data=True))
        self.train_graph.add_edges_from(message_edges)
        self.message_edges = message_edges
        self.test_edges = test_edges

        print(f"Random split (seed {self.seed}): {len(message_edges)} message edges, "
              f"{len(train_edges)} train edges, {len(test_edges)} test edges")
        return train_edges, test_edges

    def sample_negatives(self, n, exclude=frozenset()):
        """Sample n distinct non-interacting pairs among drugs/targets with >=1 interaction."""
        drugs = sorted(d for d in self.drug_nodes if self.G.degree(d) > 0)
        targets = sorted(t for t in self.target_nodes if self.G.degree(t) > 0)
        pairs = set()
        while len(pairs) < n:
            ds = self.rng.integers(len(drugs), size=n)
            ts = self.rng.integers(len(targets), size=n)
            for i, j in zip(ds, ts):
                pair = (drugs[i], targets[j])
                if pair in pairs or pair in exclude or self.G.has_edge(*pair):
                    continue
                pairs.add(pair)
                if len(pairs) == n:
                    break
        return sorted(pairs)

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

    def train_model(self):
        """Train on the train split and evaluate on the held-out test split.

        Call random_split_edges() first if embeddings must be trained on the
        message graph; otherwise it is called here.
        """
        if self.train_graph is None:
            self.random_split_edges()
        train_edges = self._train_edges_from_split()
        test_edges = self.test_edges

        train_negatives = self.sample_negatives(len(train_edges))
        test_negatives = self.sample_negatives(len(test_edges), exclude=set(train_negatives))

        X = self._features(train_edges + train_negatives)
        y = np.array([1] * len(train_edges) + [0] * len(train_negatives))
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(n_estimators=100, random_state=self.seed, n_jobs=-1)
        self.model.fit(X_scaled, y)

        X_test = self._features(test_edges + test_negatives)
        y_test = np.array([1] * len(test_edges) + [0] * len(test_negatives))
        y_proba = self.model.predict_proba(self.scaler.transform(X_test))[:, 1]

        roc = roc_auc_score(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = area_under_curve(recall, precision)
        ap = average_precision_score(y_test, y_proba)

        print("Model Performance:")
        print(f"  ROC AUC: {roc:.4f}")
        print(f"  PR AUC (trapezoidal): {pr_auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print(f"  Test Set: {len(test_edges)} positive, {len(test_negatives)} negative pairs")

        self.evaluation_results = {
            "model_performance": {
                "auc": roc,
                "pr_auc": pr_auc,
                "ap_score": ap,
                "test_positives": len(test_edges),
                "test_negatives": len(test_negatives),
                "train_positives": len(train_edges),
                "train_negatives": len(train_negatives),
                "message_edges": len(self.message_edges),
                "split": f"random edge split 60/20/20, seed {self.seed}",
            },
            "feature_importances": dict(zip(FEATURE_NAMES, self.model.feature_importances_)),
            "test_edges": test_edges,
            "test_negatives": test_negatives,
            "test_scores": y_proba,
        }
        return roc, pr_auc, ap

    def _train_edges_from_split(self):
        message = set(self.message_edges)
        test = set(self.test_edges)
        all_edges = sorted(self._canonical(e, self.drug_nodes) for e in self.G.edges())
        return [e for e in all_edges if e not in message and e not in test]

    def evaluate_calibration(self, test_edges, test_negatives):
        """Evaluate model calibration on the test split."""
        y_test = np.array([1] * len(test_edges) + [0] * len(test_negatives))
        y_proba = self.model.predict_proba(
            self.scaler.transform(self._features(test_edges + test_negatives)))[:, 1]

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_proba, n_bins=10)
        return {
            "fraction_of_positives": fraction_of_positives,
            "mean_predicted_value": mean_predicted_value,
            "brier_score": float(np.mean((y_proba - y_test) ** 2)),
        }

    def _l3_paths(self, drug_id, target_id):
        """Number of drug-target'-drug'-target paths in the message graph."""
        g = self.train_graph
        if drug_id not in g or target_id not in g:
            return 0
        drug_targets = set(g.neighbors(drug_id))
        return sum(len(drug_targets & set(g.neighbors(d2))) for d2 in g.neighbors(target_id))

    def compare_baselines(self, test_edges, test_negatives):
        """Score the test pairs with unsupervised baselines on the message graph.

        Common neighbours, Jaccard and Adamic-Adar are 0 for every drug-target
        pair in a bipartite graph, so they cannot rank anything; they are
        reported as such rather than padded with random noise. Preferential
        attachment and length-3 path counts are the bipartite equivalents.
        """
        pairs = test_edges + test_negatives
        y_test = np.array([1] * len(test_edges) + [0] * len(test_negatives))
        g = self.train_graph

        scores = {
            "preferential_attachment": [g.degree(d) * g.degree(t) for d, t in pairs],
            "l3_paths": [self._l3_paths(d, t) for d, t in pairs],
        }
        if self.embeddings_df is not None:
            scores["embedding_dot"] = [self.extract_features_safe(d, t)[9] for d, t in pairs]

        results = {}
        for name, s in scores.items():
            results[name] = {
                "roc_auc": roc_auc_score(y_test, s),
                "ap_score": average_precision_score(y_test, s),
            }
        results["common_neighbors_jaccard_adamic_adar"] = {
            "roc_auc": None,
            "ap_score": None,
            "note": "always 0 for drug-target pairs in a bipartite graph",
        }
        return results

    def predict_interaction(self, drug_id, target_id):
        """Predict interaction probability for a drug-target pair."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        features = self.extract_features_safe(drug_id, target_id, use_train_graph=True)
        return self.model.predict_proba(self.scaler.transform(features.reshape(1, -1)))[0, 1]

    def save_model(self, filepath):
        """Save the trained model."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        """Load a trained model saved with save_model."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
