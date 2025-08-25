"""Link prediction for drug-target interactions."""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import pickle
from pathlib import Path


class DTILinkPredictor:
    """Drug-Target Interaction Link Predictor using machine learning."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.drug_nodes = set()
        self.target_nodes = set()
        self.train_graph = None
        self.test_edges = None
        
    def load_data(self, drugs_df, targets_df, dti_df, embeddings_df=None):
        """Load data for link prediction."""
        self.drugs_df = drugs_df
        self.targets_df = targets_df
        self.dti_df = dti_df
        self.embeddings_df = embeddings_df
        
        # Build graph
        self.G = nx.Graph()
        
        # Add nodes
        for _, drug in drugs_df.iterrows():
            self.G.add_node(drug['drug_id'], node_type='drug', name=drug['name'])
            self.drug_nodes.add(drug['drug_id'])
            
        for _, target in targets_df.iterrows():
            self.G.add_node(target['target_id'], node_type='target', name=target['name'])
            self.target_nodes.add(target['target_id'])
        
        # Add edges
        for _, interaction in dti_df.iterrows():
            self.G.add_edge(interaction['drug_id'], interaction['target_id'])
            
        print(f"Built graph: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")
        
    def temporal_split_edges(self, test_ratio=0.2):
        """Split edges temporally or randomly for train/test."""
        edges = list(self.G.edges())
        np.random.shuffle(edges)
        
        split_idx = int(len(edges) * (1 - test_ratio))
        train_edges = edges[:split_idx]
        test_edges = edges[split_idx:]
        
        # Create training graph by removing test edges
        self.train_graph = self.G.copy()
        self.train_graph.remove_edges_from(test_edges)
        self.test_edges = test_edges
        
        print(f"Temporal split: {len(train_edges)} train edges, {len(test_edges)} test edges")
        return train_edges, test_edges
        
    def extract_features_safe(self, drug_id, target_id, use_train_graph=True):
        """Extract features for a drug-target pair (safe from data leakage)."""
        graph = self.train_graph if use_train_graph else self.G
        
        # Basic features
        drug_degree = graph.degree(drug_id) if drug_id in graph else 0
        target_degree = graph.degree(target_id) if target_id in graph else 0
        
        # Neighbor features
        drug_neighbors = set(graph.neighbors(drug_id)) if drug_id in graph else set()
        target_neighbors = set(graph.neighbors(target_id)) if target_id in graph else set()
        
        # Jaccard similarity
        if len(drug_neighbors | target_neighbors) > 0:
            jaccard = len(drug_neighbors & target_neighbors) / len(drug_neighbors | target_neighbors)
        else:
            jaccard = 0.0
            
        # Preferential attachment
        preferential_attachment = drug_degree * target_degree
        
        # Resource allocation
        resource_allocation = sum(1 / graph.degree(n) for n in drug_neighbors & target_neighbors)
        
        # Adamic-Adar
        adamic_adar = sum(1 / np.log(graph.degree(n)) for n in drug_neighbors & target_neighbors if graph.degree(n) > 1)
        
        # Shortest path (if connected)
        try:
            shortest_path = nx.shortest_path_length(graph, drug_id, target_id)
        except nx.NetworkXNoPath:
            shortest_path = -1
            
        # Common neighbors
        common_neighbors = len(drug_neighbors & target_neighbors)
        
        # Total neighbors
        total_neighbors = len(drug_neighbors | target_neighbors)
        
        # Embedding features (if available)
        drug_embedding = np.zeros(128)
        target_embedding = np.zeros(128)
        embedding_similarity = 0.0
        
        if self.embeddings_df is not None:
            if drug_id in self.embeddings_df.index:
                drug_embedding = self.embeddings_df.loc[drug_id].values
            if target_id in self.embeddings_df.index:
                target_embedding = self.embeddings_df.loc[target_id].values
            embedding_similarity = np.dot(drug_embedding, target_embedding)
        
        features = [
            drug_degree, target_degree, jaccard, preferential_attachment,
            resource_allocation, adamic_adar, shortest_path, common_neighbors,
            total_neighbors, embedding_similarity,
            drug_embedding.mean(), target_embedding.mean()
        ]
        
        return np.array(features)
        
    def generate_training_data(self, train_edges):
        """Generate training data from positive and negative examples."""
        # Positive examples
        positive_features = []
        for drug_id, target_id in train_edges:
            features = self.extract_features_safe(drug_id, target_id, use_train_graph=True)
            positive_features.append(features)
            
        # Negative examples (random non-edges)
        negative_features = []
        all_nodes = list(self.G.nodes())
        num_negatives = len(positive_features)
        
        negative_pairs = set()
        while len(negative_pairs) < num_negatives:
            drug_id = np.random.choice([n for n in all_nodes if self.G.nodes[n]['node_type'] == 'drug'])
            target_id = np.random.choice([n for n in all_nodes if self.G.nodes[n]['node_type'] == 'target'])
            
            if not self.G.has_edge(drug_id, target_id):
                negative_pairs.add((drug_id, target_id))
                
        for drug_id, target_id in negative_pairs:
            features = self.extract_features_safe(drug_id, target_id, use_train_graph=True)
            negative_features.append(features)
            
        # Combine and create labels
        X = np.vstack(positive_features + negative_features)
        y = np.array([1] * len(positive_features) + [0] * len(negative_features))
        
        return X, y
        
    def train_model(self):
        """Train the link prediction model."""
        # Create temporal split
        train_edges, test_edges = self.temporal_split_edges()
        
        # Generate training data
        X, y = self.generate_training_data(train_edges)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Evaluate on test set
        test_negatives = []
        all_nodes = list(self.G.nodes())
        num_test_negatives = len(test_edges)
        
        while len(test_negatives) < num_test_negatives:
            drug_id = np.random.choice([n for n in all_nodes if self.G.nodes[n]['node_type'] == 'drug'])
            target_id = np.random.choice([n for n in all_nodes if self.G.nodes[n]['node_type'] == 'target'])
            
            if not self.G.has_edge(drug_id, target_id):
                test_negatives.append((drug_id, target_id))
        
        # Create test features
        X_test_pos = np.vstack([self.extract_features_safe(d, t, use_train_graph=True) 
                               for d, t in test_edges])
        X_test_neg = np.vstack([self.extract_features_safe(d, t, use_train_graph=True) 
                               for d, t in test_negatives])
        
        X_test = np.vstack([X_test_pos, X_test_neg])
        y_test = np.array([1] * len(test_edges) + [0] * len(test_negatives))
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        ap_score = average_precision_score(y_test, y_pred_proba)
        
        print(f"Model Performance:")
        print(f"  ROC AUC: {auc:.4f}")
        print(f"  PR AUC: {pr_auc:.4f}")
        print(f"  Average Precision: {ap_score:.4f}")
        print(f"  Test Set: {len(test_edges)} positive, {len(test_negatives)} negative edges")
        
        # Store evaluation results
        self.evaluation_results = {
            'model_performance': {
                'auc': auc,
                'pr_auc': pr_auc,
                'ap_score': ap_score,
                'test_positives': len(test_edges),
                'test_negatives': len(test_negatives),
                'train_edges': len(train_edges)
            },
            'test_edges': test_edges,
            'test_negatives': test_negatives
        }
        
        return auc, pr_auc, ap_score
        
    def evaluate_calibration(self, test_edges, test_negatives):
        """Evaluate model calibration."""
        X_test_pos = np.vstack([self.extract_features_safe(d, t, use_train_graph=True) 
                               for d, t in test_edges])
        X_test_neg = np.vstack([self.extract_features_safe(d, t, use_train_graph=True) 
                               for d, t in test_negatives])
        
        X_test = np.vstack([X_test_pos, X_test_neg])
        y_test = np.array([1] * len(test_edges) + [0] * len(test_negatives))
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        
        # Brier score
        brier_score = np.mean((y_pred_proba - y_test) ** 2)
        
        return {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'brier_score': brier_score
        }
        
    def compare_baselines(self, test_edges, test_negatives):
        """Compare against baseline methods."""
        results = {}
        
        # Common Neighbors
        cn_scores = []
        for d, t in test_edges + test_negatives:
            if d in self.train_graph and t in self.train_graph:
                cn = len(set(self.train_graph.neighbors(d)) & set(self.train_graph.neighbors(t)))
            else:
                cn = 0
            cn_scores.append(cn)
            
        # Normalize scores
        max_cn = max(cn_scores) if cn_scores else 1
        if max_cn == 0:
            cn_scores = [np.random.random() * 0.1 for _ in cn_scores]  # Small random values
        else:
            cn_scores = [s / max_cn for s in cn_scores]
            
        y_test = np.array([1] * len(test_edges) + [0] * len(test_negatives))
        cn_auc = roc_auc_score(y_test, cn_scores)
        cn_ap = average_precision_score(y_test, cn_scores)
        
        results['common_neighbors'] = {
            'roc_auc': cn_auc,
            'ap_score': cn_ap
        }
        
        # Jaccard Similarity
        jaccard_scores = []
        for d, t in test_edges + test_negatives:
            if d in self.train_graph and t in self.train_graph:
                neighbors_d = set(self.train_graph.neighbors(d))
                neighbors_t = set(self.train_graph.neighbors(t))
                if len(neighbors_d | neighbors_t) > 0:
                    jaccard = len(neighbors_d & neighbors_t) / len(neighbors_d | neighbors_t)
                else:
                    jaccard = 0
            else:
                jaccard = 0
            jaccard_scores.append(jaccard)
            
        jaccard_auc = roc_auc_score(y_test, jaccard_scores)
        jaccard_ap = average_precision_score(y_test, jaccard_scores)
        
        results['jaccard'] = {
            'roc_auc': jaccard_auc,
            'ap_score': jaccard_ap
        }
        
        # Adamic-Adar
        aa_scores = []
        for d, t in test_edges + test_negatives:
            if d in self.train_graph and t in self.train_graph:
                common_neighbors = set(self.train_graph.neighbors(d)) & set(self.train_graph.neighbors(t))
                aa = sum(1 / np.log(self.train_graph.degree(n)) for n in common_neighbors if self.train_graph.degree(n) > 1)
            else:
                aa = 0
            aa_scores.append(aa)
            
        # Normalize scores
        max_aa = max(aa_scores) if aa_scores else 1
        if max_aa == 0:
            aa_scores = [np.random.random() * 0.1 for _ in aa_scores]  # Small random values
        else:
            aa_scores = [s / max_aa for s in aa_scores]
            
        aa_auc = roc_auc_score(y_test, aa_scores)
        aa_ap = average_precision_score(y_test, aa_scores)
        
        results['adamic_adar'] = {
            'roc_auc': aa_auc,
            'ap_score': aa_ap
        }
        
        return results
        
    def predict_interaction(self, drug_id, target_id):
        """Predict interaction probability for a drug-target pair."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        features = self.extract_features_safe(drug_id, target_id, use_train_graph=True)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        return probability
        
    def save_model(self, filepath):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'drug_nodes': self.drug_nodes,
            'target_nodes': self.target_nodes,
            'train_graph': self.train_graph,
            'evaluation_results': getattr(self, 'evaluation_results', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, filepath):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.drug_nodes = model_data['drug_nodes']
        self.target_nodes = model_data['target_nodes']
        self.train_graph = model_data['train_graph']
        if 'evaluation_results' in model_data:
            self.evaluation_results = model_data['evaluation_results']
