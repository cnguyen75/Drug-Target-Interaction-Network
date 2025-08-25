"""Callback functions for the DTI Explorer application."""
import dash
from dash import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

# Import analytics modules
from src.pipeline.advanced_analytics import DrugRepurposingAnalytics, NetworkAnalytics, PredictiveInsights

def register_callbacks(app, G, drugs_df, targets_df, dti_df, embeddings_df, link_predictor, drug_repurposer):
    """Register all callback functions with the Dash app."""
    
    # Initialize analytics objects
    drug_repurposing_analytics = DrugRepurposingAnalytics()
    network_analytics = NetworkAnalytics()
    predictive_insights = PredictiveInsights()
    
    # Load data for analytics
    drug_repurposing_analytics.load_data(drugs_df, targets_df, dti_df, embeddings_df)
    network_analytics.load_data(G, drugs_df, targets_df)
    predictive_insights.load_data(G, drugs_df, targets_df, dti_df)
    
    @app.callback(
        [Output("drug_search", "options"),
         Output("target_search", "options")],
        [Input("url", "pathname")]
    )
    def update_search_options(pathname):
        """Update search dropdown options."""
        # Drug options
        drug_options = [{"label": f"{row['name']} ({row['drug_id']})", "value": row['drug_id']} 
                       for _, row in drugs_df.head(1000).iterrows()]
        
        # Target options  
        target_options = [{"label": f"{row['name']} ({row['target_id']})", "value": row['target_id']} 
                         for _, row in targets_df.head(1000).iterrows()]
        
        return drug_options, target_options
    
    @app.callback(
        [Output("similarity_drug_dropdown", "options"),
         Output("prediction_drug_dropdown", "options"),
         Output("analytics_drug_dropdown", "options")],
        [Input("url", "pathname")]
    )
    def update_drug_dropdowns(pathname):
        """Update drug dropdown options for all tabs."""
        # Filter for approved drugs and exclude endogenous compounds
        approved_drugs = drugs_df[
            (drugs_df['groups'].str.contains('approved', case=False, na=False)) &
            (~drugs_df['name'].str.contains('endogenous|compound|acid|phosphate|atp|adp|amp', case=False, na=False))
        ]
        
        # Get drugs with interactions
        drugs_with_interactions = approved_drugs[
            approved_drugs['drug_id'].isin(dti_df['drug_id'].unique())
        ]
        
        # Limit to top 200 by connections
        drug_connection_counts = dti_df['drug_id'].value_counts()
        top_drugs = drugs_with_interactions[
            drugs_with_interactions['drug_id'].isin(drug_connection_counts.head(200).index)
        ]
        
        drug_options = [{"label": f"{row['name']} ({row['drug_id']})", "value": row['drug_id']} 
                       for _, row in top_drugs.iterrows()]
        
        return drug_options, drug_options, drug_options
    
    @app.callback(
        [Output("prediction_target_dropdown", "options")],
        [Input("url", "pathname")]
    )
    def update_target_dropdown(pathname):
        """Update target dropdown options."""
        target_options = [{"label": f"{row['name']} ({row['target_id']})", "value": row['target_id']} 
                         for _, row in targets_df.head(1000).iterrows()]
        return [target_options]
    
    # Add more callbacks here...
    # (This is a placeholder - the actual callbacks will be moved from app.py)
    
    return app
