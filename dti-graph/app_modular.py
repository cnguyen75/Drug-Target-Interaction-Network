#!/usr/bin/env python3
"""
Drug-Target Interaction Explorer Pro - Modular Edition
Advanced network analysis and machine learning insights for drug-target interactions.
"""

import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import modular components
from src.app.layout import create_main_layout
from src.app.callbacks import register_callbacks

# Import data loading and ML modules
from src.datasources.drugbank_loader import DrugBankLoader
from src.pipeline.link_prediction import DTILinkPredictor
from src.pipeline.drug_repurposing import DrugRepurposer

def load_data():
    """Load all data and models."""
    print("Loading data and models...")
    
    # Load data
    loader = DrugBankLoader()
    drugs_df, targets_df, dti_df = loader.load_data()
    
    # Load graph
    graph_path = Path("data/graph_full.pkl")
    if graph_path.exists():
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print("Graph file not found. Please run the pipeline first.")
        return None, None, None, None, None, None
    
    # Load embeddings
    embeddings_path = Path("data/embeddings_full.parquet")
    if embeddings_path.exists():
        embeddings_df = pd.read_parquet(embeddings_path)
        print(f"Loaded embeddings: {len(embeddings_df)} nodes")
    else:
        print("Embeddings file not found. Please run the pipeline first.")
        return None, None, None, None, None, None
    
    # Load ML models
    link_predictor = DTILinkPredictor()
    drug_repurposer = DrugRepurposer()
    
    model_path = Path("data/link_prediction_model.pkl")
    if model_path.exists():
        link_predictor.load_model(model_path)
        print("Link prediction model loaded!")
    else:
        print("Link prediction model not found. Please run train_ml_models.py first.")
    
    repurposing_path = Path("data/drug_repurposing_model.pkl")
    if repurposing_path.exists():
        drug_repurposer.load_model(repurposing_path)
        print("Drug repurposing model loaded!")
    else:
        print("Drug repurposing model not found. Please run train_ml_models.py first.")
    
    print("All pre-trained models loaded successfully!")
    return G, drugs_df, targets_df, dti_df, embeddings_df, link_predictor, drug_repurposer

def create_app():
    """Create and configure the Dash application."""
    # Load data
    data = load_data()
    if data is None:
        print("Failed to load data. Exiting.")
        return None
    
    G, drugs_df, targets_df, dti_df, embeddings_df, link_predictor, drug_repurposer = data
    
    # Create Dash app
    app = dash.Dash(__name__, 
                    external_stylesheets=[
                        'https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap'
                    ])
    
    # Configure app
    app.title = "DTI Explorer Pro"
    app.layout = create_main_layout()
    
    # Register callbacks
    register_callbacks(app, G, drugs_df, targets_df, dti_df, embeddings_df, link_predictor, drug_repurposer)
    
    return app

def main():
    """Main application entry point."""
    print("Starting DTI Explorer Pro - Modular Edition...")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    G, drugs_df, targets_df, dti_df, embeddings_df, link_predictor, drug_repurposer = data
    
    print("Loaded data:")
    print(f"  - Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  - Drugs: {len(drugs_df):,}")
    print(f"  - Targets: {len(targets_df):,}")
    
    # Create app
    app = create_app()
    if app is None:
        return
    
    print("Starting web server...")
    app.run_server(debug=True, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()
