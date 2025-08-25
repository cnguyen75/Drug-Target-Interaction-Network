"""Advanced analytics for drug discovery and network analysis."""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import community  # python-louvain


class DrugRepurposingAnalytics:
    """Advanced drug repurposing analytics with lazy loading."""
    
    def __init__(self, drugs_df, targets_df, dti_df, embeddings_df=None):
        self.drugs_df = drugs_df
        self.targets_df = targets_df
        self.dti_df = dti_df
        self.embeddings_df = embeddings_df
        self._embeddings_prepared = False
        
    def _prepare_embeddings(self):
        """Prepare embeddings for analysis (lazy loading)."""
        if self._embeddings_prepared:
            return
            
        if self.embeddings_df is None:
            # Create dummy embeddings based on target profiles
            print("Warning: No embeddings available. Creating dummy embeddings for analysis.")
            all_nodes = list(set(self.drugs_df['drug_id'].tolist() + self.targets_df['target_id'].tolist()))
            dummy_embeddings = np.random.rand(len(all_nodes), 128)
            self.embeddings_df = pd.DataFrame(dummy_embeddings, index=all_nodes)
            
        # Separate drug and target embeddings
        drug_nodes = set(self.drugs_df['drug_id'])
        target_nodes = set(self.targets_df['target_id'])
        
        self.drug_embeddings = self.embeddings_df[self.embeddings_df.index.isin(drug_nodes)]
        self.target_embeddings = self.embeddings_df[self.embeddings_df.index.isin(target_nodes)]
        
        self._embeddings_prepared = True
        
    def find_similar_drugs(self, drug_id, top_k=10, similarity_threshold=0.7):
        """Find similar drugs based on embeddings (optimized for performance)."""
        self._prepare_embeddings()
        
        if drug_id not in self.drugs_df['drug_id'].values:
            return []
            
        if self.drug_embeddings is not None and drug_id in self.drug_embeddings.index:
            # Use subset of top drugs by degree for faster computation
            drug_connections = self.dti_df['drug_id'].value_counts()
            subset_drugs = drug_connections.head(1000).index  # Top 1000 drugs
            
            subset_embeddings = self.drug_embeddings[self.drug_embeddings.index.isin(subset_drugs)]
            
            if drug_id in subset_embeddings.index:
                drug_embedding = subset_embeddings.loc[drug_id].values.reshape(1, -1)
                
                # Calculate similarities with subset
                similarities = cosine_similarity(drug_embedding, subset_embeddings.values)[0]
                
                # Create similarity pairs
                similarity_pairs = list(zip(subset_embeddings.index, similarities))
                
                # Filter and sort
                similar_drugs = []
                for similar_drug_id, similarity in similarity_pairs:
                    if similar_drug_id != drug_id and similarity >= similarity_threshold:
                        similar_drug_info = self.drugs_df[self.drugs_df['drug_id'] == similar_drug_id].iloc[0]
                        similar_drugs.append({
                            'drug_id': similar_drug_id,
                            'name': similar_drug_info['name'],
                            'similarity': similarity,
                            'type': similar_drug_info.get('type', 'Unknown'),
                            'approved': similar_drug_info.get('approved', False)
                        })
                        
                # Sort by similarity and return top_k
                similar_drugs.sort(key=lambda x: x['similarity'], reverse=True)
                return similar_drugs[:top_k]
                
        return []
        
    def repurposing_opportunities(self, drug_id, top_k=5):
        """Find repurposing opportunities for a drug."""
        self._prepare_embeddings()
        
        if drug_id not in self.drugs_df['drug_id'].values:
            return []
            
        current_targets = set(self.dti_df[self.dti_df['drug_id'] == drug_id]['target_id'])
        
        opportunities = []
        
        # Find targets that the drug doesn't currently interact with
        all_targets = set(self.targets_df['target_id'])
        potential_targets = all_targets - current_targets
        
        # Score potential targets based on similarity to current targets
        for target_id in list(potential_targets)[:50]:  # Limit for performance
            target_info = self.targets_df[self.targets_df['target_id'] == target_id].iloc[0]
            
            # Calculate similarity to current targets
            if self.target_embeddings is not None and target_id in self.target_embeddings.index:
                target_embedding = self.target_embeddings.loc[target_id].values.reshape(1, -1)
                
                # Find similarity to current targets
                similarities = []
                for current_target in current_targets:
                    if current_target in self.target_embeddings.index:
                        current_embedding = self.target_embeddings.loc[current_target].values.reshape(1, -1)
                        similarity = cosine_similarity(target_embedding, current_embedding)[0, 0]
                        similarities.append(similarity)
                        
                if similarities:
                    avg_similarity = np.mean(similarities)
                    opportunities.append({
                        'target_id': target_id,
                        'name': target_info['name'],
                        'similarity': avg_similarity,
                        'organism': target_info.get('organism', 'Unknown')
                    })
                    
        # Sort by similarity and return top_k
        opportunities.sort(key=lambda x: x['similarity'], reverse=True)
        return opportunities[:top_k]


class NetworkAnalytics:
    """Network analytics for community detection and centrality analysis."""
    
    def __init__(self, G, drugs_df, targets_df, dti_df):
        self.G = G
        self.drugs_df = drugs_df
        self.targets_df = targets_df
        self.dti_df = dti_df
        
    def centrality_analysis(self, drug_id=None):
        """Analyze network centrality metrics."""
        if drug_id:
            # Focused analysis on specific drug
            if drug_id in self.G:
                neighbors = list(self.G.neighbors(drug_id))
                degree_centrality = len(neighbors) / (len(self.G.nodes()) - 1) if len(self.G.nodes()) > 1 else 0
                
                # Calculate betweenness centrality (expensive, so limit scope)
                subgraph = self.G.subgraph(list(neighbors) + [drug_id])
                betweenness_centrality = nx.betweenness_centrality(subgraph, normalized=True).get(drug_id, 0)
                
                # Calculate closeness centrality (expensive, so limit scope)
                closeness_centrality = nx.closeness_centrality(subgraph).get(drug_id, 0)
                
                return {
                    'degree_centrality': degree_centrality,
                    'betweenness_centrality': betweenness_centrality,
                    'closeness_centrality': closeness_centrality,
                    'neighbor_count': len(neighbors)
                }
            else:
                return {
                    'degree_centrality': 0.0,
                    'betweenness_centrality': 0.0,
                    'closeness_centrality': 0.0,
                    'neighbor_count': 0
                }
        else:
            # General analysis
            centrality_metrics = {}
            
            # Get top drugs by degree
            drug_degrees = [(n, d) for n, d in self.G.degree() if self.G.nodes[n]['node_type'] == 'drug']
            drug_degrees.sort(key=lambda x: x[1], reverse=True)
            
            for drug_id, degree in drug_degrees[:10]:
                centrality_metrics[drug_id] = {
                    'degree_centrality': degree / (len(self.G.nodes()) - 1),
                    'degree': degree
                }
                
            return centrality_metrics
            
    def community_detection(self):
        """Detect communities in the network."""
        communities = {}
        
        try:
            # Try Louvain method first with higher resolution to get more communities
            partition = community.best_partition(self.G, resolution=1.0)
            
            # If too few communities, try with even higher resolution
            if len(set(partition.values())) < 5:
                partition = community.best_partition(self.G, resolution=1.5)
                
            # Group nodes by community
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = set()
                communities[community_id].add(node)
                
        except Exception as e:
            print(f"Louvain failed: {e}")
            try:
                # Fallback to Label Propagation
                communities_list = list(nx.community.label_propagation_communities(self.G))
                for i, community_nodes in enumerate(communities_list):
                    communities[i] = community_nodes
                    
            except Exception as e:
                print(f"Label Propagation failed: {e}")
                # Final fallback to connected components
                components = list(nx.connected_components(self.G))
                for i, component in enumerate(components):
                    if len(component) >= 10:  # Only keep significant components
                        communities[i] = component
                        
        # Filter communities to only include significant ones
        significant_communities = {}
        for community_id, members in communities.items():
            if len(members) >= 5:  # Reduced threshold to show more communities
                significant_communities[community_id] = members
                
        # Limit to top 20 largest communities if too many (to get more variety)
        if len(significant_communities) > 20:
            sorted_communities = sorted(significant_communities.items(), 
                                      key=lambda x: len(x[1]), reverse=True)
            significant_communities = {i: members for i, (_, members) in enumerate(sorted_communities[:20])}
            
        print(f"Detected {len(significant_communities)} communities")
        return significant_communities


class PredictiveInsights:
    """Predictive insights for drug combinations and interactions."""
    
    def __init__(self, drugs_df, targets_df, dti_df, link_predictor=None):
        self.drugs_df = drugs_df
        self.targets_df = targets_df
        self.dti_df = dti_df
        self.link_predictor = link_predictor
        
    def drug_combination_recommendations(self, drug_id, top_k=5):
        """Find drug combination recommendations."""
        if drug_id not in self.drugs_df['drug_id'].values:
            return []
            
        drug_targets = set(self.dti_df[self.dti_df['drug_id'] == drug_id]['target_id'])
        
        # Get subset of drugs for faster computation
        drug_connections = self.dti_df['drug_id'].value_counts()
        subset_drugs = drug_connections.head(500).index  # Top 500 drugs
        subset_drugs_df = self.drugs_df[self.drugs_df['drug_id'].isin(subset_drugs)].drop_duplicates(subset=['drug_id'])
        
        recommendations = []
        seen_drugs = set()
        
        for _, other_drug in subset_drugs_df.iterrows():
            if other_drug['drug_id'] != drug_id and other_drug['drug_id'] not in seen_drugs:
                other_targets = set(self.dti_df[self.dti_df['drug_id'] == other_drug['drug_id']]['target_id'])
                
                # Calculate complementarity (low overlap is good)
                overlap = len(drug_targets & other_targets)
                total_targets = len(drug_targets | other_targets)
                
                if total_targets > 0:
                    complementarity_score = 1 - (overlap / total_targets)
                    
                    if complementarity_score > 0.5:  # Only recommend if complementary
                        recommendations.append({
                            'drug_id': other_drug['drug_id'],
                            'name': other_drug['name'],
                            'complementarity_score': complementarity_score,
                            'overlap': overlap,
                            'total_targets': total_targets
                        })
                        seen_drugs.add(other_drug['drug_id'])
                        
        # Sort by complementarity and return top_k
        recommendations.sort(key=lambda x: x['complementarity_score'], reverse=True)
        return recommendations[:top_k]
        
    def drug_drug_interaction_warnings(self, drug_id, top_k=5):
        """Find potential drug-drug interaction warnings."""
        if drug_id not in self.drugs_df['drug_id'].values:
            return []
            
        drug_targets = set(self.dti_df[self.dti_df['drug_id'] == drug_id]['target_id'])
        
        # Get subset of drugs for faster computation
        drug_connections = self.dti_df['drug_id'].value_counts()
        subset_drugs = drug_connections.head(500).index  # Top 500 drugs
        subset_drugs_df = self.drugs_df[self.drugs_df['drug_id'].isin(subset_drugs)].drop_duplicates(subset=['drug_id'])
        
        warnings = []
        seen_drugs = set()
        
        for _, other_drug in subset_drugs_df.iterrows():
            if other_drug['drug_id'] != drug_id and other_drug['drug_id'] not in seen_drugs:
                other_targets = set(self.dti_df[self.dti_df['drug_id'] == other_drug['drug_id']]['target_id'])
                
                # Calculate interaction strength (high overlap is bad)
                overlap = len(drug_targets & other_targets)
                total_targets = len(drug_targets | other_targets)
                
                if total_targets > 0:
                    interaction_strength = overlap / total_targets
                    
                    if interaction_strength > 0.3:  # Only warn if significant overlap
                        # Determine warning level
                        if interaction_strength > 0.7:
                            warning_level = "High"
                        elif interaction_strength > 0.5:
                            warning_level = "Medium"
                        else:
                            warning_level = "Low"
                            
                        warnings.append({
                            'drug_id': other_drug['drug_id'],
                            'name': other_drug['name'],
                            'interaction_strength': interaction_strength,
                            'warning_level': warning_level,
                            'overlap': overlap,
                            'total_targets': total_targets
                        })
                        seen_drugs.add(other_drug['drug_id'])
                        
        # Sort by interaction strength and return top_k
        warnings.sort(key=lambda x: x['interaction_strength'], reverse=True)
        return warnings[:top_k]
