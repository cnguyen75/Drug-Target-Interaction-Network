"""Drug repurposing analysis using embeddings and similarity."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class DrugRepurposer:
    """Drug repurposing analysis using embeddings and network features."""
    
    def __init__(self):
        self.drug_embeddings = None
        self.target_embeddings = None
        self.drugs_df = None
        self.targets_df = None
        self.dti_df = None
        
    def load_data(self, drugs_df, targets_df, dti_df, embeddings_df=None):
        """Load data for drug repurposing."""
        self.drugs_df = drugs_df
        self.targets_df = targets_df
        self.dti_df = dti_df
        
        if embeddings_df is not None:
            # Check if node_id is already the index
            if embeddings_df.index.name == 'node_id':
                self.embeddings_df = embeddings_df
            else:
                # Set node_id as index if it's a column
                if 'node_id' in embeddings_df.columns:
                    self.embeddings_df = embeddings_df.set_index('node_id')
                else:
                    self.embeddings_df = embeddings_df
                    
            # Separate drug and target embeddings
            drug_nodes = set(drugs_df['drug_id'])
            target_nodes = set(targets_df['target_id'])
            
            self.drug_embeddings = self.embeddings_df[self.embeddings_df.index.isin(drug_nodes)]
            self.target_embeddings = self.embeddings_df[self.embeddings_df.index.isin(target_nodes)]
            
            print(f"Loaded embeddings: {len(self.drug_embeddings)} drugs, {len(self.target_embeddings)} targets")
        else:
            print("No embeddings provided. Drug repurposing will use network-based similarity.")
            
    def find_similar_drugs(self, drug_id, top_k=10, similarity_threshold=0.7):
        """Find similar drugs based on embeddings or network similarity."""
        if drug_id not in self.drugs_df['drug_id'].values:
            return []
            
        drug_info = self.drugs_df[self.drugs_df['drug_id'] == drug_id].iloc[0]
        
        if self.drug_embeddings is not None and drug_id in self.drug_embeddings.index:
            # Use embeddings for similarity
            drug_embedding = self.drug_embeddings.loc[drug_id].values.reshape(1, -1)
            
            # Calculate similarities with all other drugs
            similarities = cosine_similarity(drug_embedding, self.drug_embeddings.values)[0]
            
            # Create similarity pairs
            similarity_pairs = list(zip(self.drug_embeddings.index, similarities))
            
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
            
        else:
            # Use network-based similarity (target overlap)
            drug_targets = set(self.dti_df[self.dti_df['drug_id'] == drug_id]['target_id'])
            
            similar_drugs = []
            for _, other_drug in self.drugs_df.iterrows():
                if other_drug['drug_id'] != drug_id:
                    other_targets = set(self.dti_df[self.dti_df['drug_id'] == other_drug['drug_id']]['target_id'])
                    
                    if len(drug_targets | other_targets) > 0:
                        jaccard_similarity = len(drug_targets & other_targets) / len(drug_targets | other_targets)
                        
                        if jaccard_similarity >= similarity_threshold:
                            similar_drugs.append({
                                'drug_id': other_drug['drug_id'],
                                'name': other_drug['name'],
                                'similarity': jaccard_similarity,
                                'type': other_drug.get('type', 'Unknown'),
                                'approved': other_drug.get('approved', False)
                            })
                            
            # Sort by similarity and return top_k
            similar_drugs.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_drugs[:top_k]
            
    def repurposing_opportunities(self, drug_id, top_k=5):
        """Find repurposing opportunities for a drug."""
        if drug_id not in self.drugs_df['drug_id'].values:
            return []
            
        drug_info = self.drugs_df[self.drugs_df['drug_id'] == drug_id].iloc[0]
        current_targets = set(self.dti_df[self.dti_df['drug_id'] == drug_id]['target_id'])
        
        opportunities = []
        
        # Find targets that the drug doesn't currently interact with
        all_targets = set(self.targets_df['target_id'])
        potential_targets = all_targets - current_targets
        
        # Score potential targets based on similarity to current targets
        for target_id in list(potential_targets)[:100]:  # Limit for performance
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
        
    def save_model(self, filepath):
        """Save the repurposing model."""
        model_data = {
            'drug_embeddings': self.drug_embeddings,
            'target_embeddings': self.target_embeddings,
            'drugs_df': self.drugs_df,
            'targets_df': self.targets_df,
            'dti_df': self.dti_df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, filepath):
        """Load a repurposing model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.drug_embeddings = model_data['drug_embeddings']
        self.target_embeddings = model_data['target_embeddings']
        self.drugs_df = model_data['drugs_df']
        self.targets_df = model_data['targets_df']
        self.dti_df = model_data['dti_df']
