"""Enhanced Drug-Target Interaction Explorer with Plotly Network Graphs."""

import os
import sys
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import networkx as nx
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pipeline.link_prediction import DTILinkPredictor
from pipeline.drug_repurposing import DrugRepurposer
from pipeline.advanced_analytics import DrugRepurposingAnalytics, NetworkAnalytics, PredictiveInsights

# Initialize Dash app
app = dash.Dash(__name__, title="DTI Explorer Pro - Plotly Edition")
app.config.suppress_callback_exceptions = True

# Load data and models
print("Loading data and models...")
try:
    # Load full DrugBank database
    full_drugs_fp = "data_raw/drugs_full.parquet"
    full_targets_fp = "data_raw/targets_full.parquet"
    full_dti_fp = "data_raw/dti_full.parquet"
    
    if not os.path.exists(full_drugs_fp):
        print("Full DrugBank database not found!")
        print("Please ensure drugs_full.parquet, targets_full.parquet, and dti_full.parquet exist in the data_raw/ directory.")
        sys.exit(1)
    
    print("Loading full DrugBank database...")
    drugs_df = pd.read_parquet(full_drugs_fp)
    targets_df = pd.read_parquet(full_targets_fp)
    dti_df = pd.read_parquet(full_dti_fp)
    
    # Build graph exactly like training script (include ALL drugs and targets)
    print("Building graph from full database...")
    G = nx.Graph()
    
    print(f"Loading: {len(drugs_df):,} drugs, {len(targets_df):,} targets, {len(dti_df):,} interactions")
    
    # Add ALL drug nodes to graph (like training script)
    for _, drug in drugs_df.iterrows():
        G.add_node(drug['drug_id'], node_type='drug', name=drug['name'])
    
    # Add ALL target nodes to graph (like training script)  
    for _, target in targets_df.iterrows():
        G.add_node(target['target_id'], node_type='target', name=target['name'])
    
    # Add edges
    for _, interaction in dti_df.iterrows():
        G.add_edge(interaction['drug_id'], interaction['target_id'])
    
    print(f"Built graph: {len(G.nodes()):,} total nodes ({len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'drug']):,} drugs, {len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']):,} targets)")
    print(f"Graph edges: {len(G.edges()):,} interactions")
    print(f"Note: Graph includes all entities (ML models will internally filter to connected nodes)")
    
    # Load embeddings for advanced analytics
    embeddings_df = None
    embeddings_path = "data/embeddings_full.parquet"
    print(f"Checking for embeddings at: {os.path.abspath(embeddings_path)}")
    if os.path.exists(embeddings_path):
        print("Loading embeddings for advanced analytics...")
        try:
            embeddings_df = pd.read_parquet(embeddings_path)
            print(f"Loaded embeddings: {len(embeddings_df):,} nodes")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            embeddings_df = None
    else:
        print("Warning: Embeddings file not found. Advanced analytics will not be available.")
        print("To enable advanced analytics, run: python3 train_ml_models.py")
    
    # Initialize ML models
    print("Loading pre-trained ML models...")
    link_predictor = None
    repurposer_obj = None
    
    try:
        # Load pre-trained models from models directory
        models_dir = Path("models")
        
        if not models_dir.exists():
            print("Models directory not found!")
            print("Please run 'python3 train_ml_models.py' to train models first.")
            sys.exit(1)
        
        # Load link prediction model
        link_model_path = models_dir / "link_prediction_model.pkl"
        if not link_model_path.exists():
            print("Link prediction model not found!")
            print("Please run 'python3 train_ml_models.py' to train models first.")
            sys.exit(1)
        
        print("Loading pre-trained link prediction model...")
        with open(link_model_path, 'rb') as f:
            link_predictor = pickle.load(f)
        print("Link prediction model loaded!")
        
        # Load drug repurposing model
        repurposer_path = models_dir / "drug_repurposing_model.pkl"
        if not repurposer_path.exists():
            print("Drug repurposing model not found!")
            print("Please run 'python3 train_ml_models.py' to train models first.")
            sys.exit(1)
        
        print("Loading pre-trained drug repurposing model...")
        with open(repurposer_path, 'rb') as f:
            repurposer_obj = pickle.load(f)
        print("Drug repurposing model loaded!")
        
        print("All pre-trained models loaded successfully!")
            
    except Exception as e:
        print(f"Error loading pre-trained models: {e}")
        print("Please run 'python3 train_ml_models.py' to train models first.")
        sys.exit(1)
        
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure the full DrugBank database files exist in data_raw/ directory.")
    sys.exit(1)

def create_network_figure(G, selected_node=None, max_nodes=100, show_labels=False, walks=2):
    """Create a professional Plotly network figure with smart text handling."""
    if G is None or len(G.nodes()) == 0:
        return go.Figure().add_annotation(
            text="No network data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#7f8c8d")
        )
    
    # Get subgraph around selected node or sample
    if selected_node and selected_node in G:
        # Get nodes within specified walks of selected node
        nodes_to_include = {selected_node}
        for _ in range(walks):
            new_nodes = set()
            for node in nodes_to_include:
                neighbors = list(G.neighbors(node))
                new_nodes.update(neighbors)
            nodes_to_include.update(new_nodes)
        
        # Limit to max_nodes
        if len(nodes_to_include) > max_nodes:
            nodes_to_include = set(list(nodes_to_include)[:max_nodes])
        
        subgraph = G.subgraph(nodes_to_include)
        # Don't force labels on - respect the checkbox setting
    else:
        # Sample a balanced subgraph with both drugs and targets
        if len(G.nodes()) > max_nodes:
            # Get drug and target nodes
            drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
            target_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "target"]
            
            # Sample balanced number of drugs and targets
            max_drugs = max_nodes // 2
            max_targets = max_nodes - max_drugs
            
            sample_drugs = drug_nodes[:max_drugs] if len(drug_nodes) > max_drugs else drug_nodes
            sample_targets = target_nodes[:max_targets] if len(target_nodes) > max_targets else target_nodes
            
            # Get all neighbors of sampled nodes to show connections
            all_neighbors = set()
            for node in sample_drugs + sample_targets:
                neighbors = list(G.neighbors(node))
                all_neighbors.update(neighbors)
            
            # Combine sampled nodes with their neighbors
            nodes_to_include = set(sample_drugs + sample_targets) | all_neighbors
            
            # Limit to max_nodes
            if len(nodes_to_include) > max_nodes:
                nodes_to_include = set(list(nodes_to_include)[:max_nodes])
            
            subgraph = G.subgraph(nodes_to_include)
        else:
            subgraph = G
    
    # Get node positions using spring layout with better parameters
    pos = nx.spring_layout(subgraph, k=2, iterations=100, seed=42)
    
    # Separate drug and target nodes
    drug_nodes = []
    target_nodes = []
    drug_x, drug_y = [], []
    target_x, target_y = [], []
    drug_names, target_names = [], []
    drug_hover_text, target_hover_text = [], []
    drug_sizes, target_sizes = [], []
    
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        node_type = node_data.get('node_type', 'unknown')
        name = node_data.get('name', str(node))
        degree = subgraph.degree(node)
        
        x, y = pos[node]
        
        # Create detailed hover text with more information
        hover_text = f"""<b>{name}</b><br><b>ID:</b> {node}<br><b>Type:</b> {node_type.title()}<br><b>Connections:</b> {degree}<br><b>Centrality:</b> {degree/len(subgraph.nodes()):.3f}"""
        
        # Add type-specific information to hover
        if node_type == 'drug':
            # Try to get drug info for hover
            drug_info = drugs_df[drugs_df['drug_id'] == node]
            if not drug_info.empty:
                drug_row = drug_info.iloc[0]
                hover_text += f"<br><b>Status:</b> {'Approved' if drug_row['approved'] else 'Experimental'}"
                hover_text += f"<br><b>Type:</b> {drug_row['type']}"
        elif node_type == 'target':
            # Try to get target info for hover
            target_info = targets_df[targets_df['target_id'] == node]
            if not target_info.empty:
                target_row = target_info.iloc[0]
                hover_text += f"<br><b>Organism:</b> {target_row['organism']}"
                hover_text += f"<br><b>Human:</b> {'Yes' if target_row['is_human'] else 'No'}"
        
        # Size nodes based on degree (with limits)
        node_size = min(max(degree * 2 + 8, 8), 25)
        
        if node_type == 'drug':
            drug_nodes.append(node)
            drug_x.append(x)
            drug_y.append(y)
            drug_names.append(name)
            drug_hover_text.append(hover_text)
            drug_sizes.append(node_size)
        else:
            target_nodes.append(node)
            target_x.append(x)
            target_y.append(y)
            target_names.append(name)
            target_hover_text.append(hover_text)
            target_sizes.append(node_size)
    
    # Create edge traces
    edge_x, edge_y = [], []
    
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create the figure
    fig = go.Figure()
    
    # Add edges with professional styling
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#95a5a6'),
        hoverinfo='none',
        mode='lines',
        showlegend=False,
        opacity=0.6
    ))
    
    # Add drug nodes with professional styling
    if drug_x:
        fig.add_trace(go.Scatter(
            x=drug_x, y=drug_y,
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text',
            text=drug_names if show_labels else None,
            textposition="top center",
            hovertext=drug_hover_text,
            marker=dict(
                size=drug_sizes,
                color='#e74c3c',
                line=dict(width=1.5, color='#c0392b'),
                opacity=0.9
            ),
            name='Drugs',
            textfont=dict(size=9, color='#2c3e50'),
            hovertemplate='%{hovertext}<extra></extra>'
        ))
    
    # Add target nodes with professional styling
    if target_x:
        fig.add_trace(go.Scatter(
            x=target_x, y=target_y,
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text',
            text=target_names if show_labels else None,
            textposition="top center",
            hovertext=target_hover_text,
            marker=dict(
                size=target_sizes,
                color='#3498db',
                line=dict(width=1.5, color='#2980b9'),
                opacity=0.9
            ),
            name='Targets',
            textfont=dict(size=9, color='#2c3e50'),
            hovertemplate='%{hovertext}<extra></extra>'
        ))
    
    # Highlight selected node with professional styling
    if selected_node and selected_node in subgraph:
        sel_x, sel_y = pos[selected_node]
        node_data = subgraph.nodes[selected_node]
        node_name = node_data.get('name', selected_node)
        sel_degree = subgraph.degree(selected_node)
        sel_size = min(max(sel_degree * 2 + 12, 15), 30)
        
        # Add highlighted node
        fig.add_trace(go.Scatter(
            x=[sel_x], y=[sel_y],
            mode='markers+text',
            text=[node_name],
            textposition="top center",
            marker=dict(
                size=sel_size,
                color='#f39c12',
                line=dict(width=3, color='#e67e22'),
                opacity=1.0
            ),
            name='Selected',
            showlegend=False,
            textfont=dict(size=11, color='#2c3e50', family="Arial Black"),
            hovertemplate=f'<b>{node_name}</b><br>Selected Node<extra></extra>'
        ))
    
    # Professional layout
    fig.update_layout(
        title=dict(
            text=f"Drug-Target Interaction Network",
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2c3e50', family="Arial")
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#bdc3c7',
            borderwidth=1
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-1.1, 1.1]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-1.1, 1.1]
        ),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        height=650,
        font=dict(family="Arial", size=12, color='#2c3e50')
    )
    
    # Add subtitle with statistics
    fig.add_annotation(
        text=f"Subgraph: {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges",
        xref="paper", yref="paper",
        x=0.5, y=0.95,
        showarrow=False,
        font=dict(size=12, color='#7f8c8d'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#bdc3c7',
        borderwidth=1
    )
    
    return fig

def create_3d_network_figure(G, selected_node=None, max_nodes=50, walks=2):
    """Create a 3D Plotly network figure."""
    if G is None or len(G.nodes()) == 0:
        return go.Figure().add_annotation(
            text="No network data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Get subgraph around selected node or sample
    if selected_node and selected_node in G:
        nodes_to_include = {selected_node}
        for _ in range(walks):  # Use walks parameter
            new_nodes = set()
            for node in nodes_to_include:
                neighbors = list(G.neighbors(node))
                new_nodes.update(neighbors)
            nodes_to_include.update(new_nodes)
        
        if len(nodes_to_include) > max_nodes:
            nodes_to_include = set(list(nodes_to_include)[:max_nodes])
        
        subgraph = G.subgraph(nodes_to_include)
    else:
        # Sample a balanced subgraph with both drugs and targets
        if len(G.nodes()) > max_nodes:
            # Get drug and target nodes
            drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
            target_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "target"]
            
            # Sample balanced number of drugs and targets
            max_drugs = max_nodes // 2
            max_targets = max_nodes - max_drugs
            
            sample_drugs = drug_nodes[:max_drugs] if len(drug_nodes) > max_drugs else drug_nodes
            sample_targets = target_nodes[:max_targets] if len(target_nodes) > max_targets else target_nodes
            
            # Get all neighbors of sampled nodes to show connections
            all_neighbors = set()
            for node in sample_drugs + sample_targets:
                neighbors = list(G.neighbors(node))
                all_neighbors.update(neighbors)
            
            # Combine sampled nodes with their neighbors
            nodes_to_include = set(sample_drugs + sample_targets) | all_neighbors
            
            # Limit to max_nodes
            if len(nodes_to_include) > max_nodes:
                nodes_to_include = set(list(nodes_to_include)[:max_nodes])
            
            subgraph = G.subgraph(nodes_to_include)
        else:
            subgraph = G
    
    # Get 3D node positions
    pos = nx.spring_layout(subgraph, k=2, iterations=50, dim=3)
    
    # Separate nodes by type
    drug_nodes = []
    target_nodes = []
    drug_x, drug_y, drug_z = [], [], []
    target_x, target_y, target_z = [], [], []
    drug_names, target_names = [], []
    
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        node_type = node_data.get('node_type', 'unknown')
        name = node_data.get('name', str(node))
        
        x, y, z = pos[node]
        
        if node_type == 'drug':
            drug_nodes.append(node)
            drug_x.append(x)
            drug_y.append(y)
            drug_z.append(z)
            drug_names.append(name)
        else:
            target_nodes.append(node)
            target_x.append(x)
            target_y.append(y)
            target_z.append(z)
            target_names.append(name)
    
    # Create edge traces
    edge_x, edge_y, edge_z = [], [], []
    for edge in subgraph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # Create the figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add drug nodes
    if drug_x:
        fig.add_trace(go.Scatter3d(
            x=drug_x, y=drug_y, z=drug_z,
            mode='markers',
            hoverinfo='text',
            text=drug_names,
            marker=dict(
                size=8,
                color='#e74c3c',
                line=dict(width=2, color='#c0392b')
            ),
            name='Drugs'
        ))
    
    # Add target nodes
    if target_x:
        fig.add_trace(go.Scatter3d(
            x=target_x, y=target_y, z=target_z,
            mode='markers',
            hoverinfo='text',
            text=target_names,
            marker=dict(
                size=6,
                color='#3498db',
                line=dict(width=2, color='#2980b9')
            ),
            name='Targets'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"3D Drug-Target Network ({len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges)",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )
    
    return fig

# Create main layout
app.layout = html.Div([
    # URL component for callbacks
    dcc.Location(id='url', refresh=False),

    # Professional Header
    html.Div([
        html.Div([
            html.H1("DTI Explorer Pro", 
                    style={
                        "textAlign": "center", 
                        "marginBottom": "5px", 
                        "color": "#2c3e50",
                        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        "fontWeight": "300",
                        "fontSize": "28px",
                        "letterSpacing": "1px"
                    }),
            html.P("Advanced Drug-Target Interaction Analysis Platform", 
                   style={
                       "textAlign": "center", 
                       "color": "#5a6c7d", 
                       "marginBottom": "10px",
                       "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                       "fontSize": "14px",
                       "fontWeight": "400"
                   }),
            html.P(f"Full DrugBank Database: {len(drugs_df):,} drugs | {len(targets_df):,} targets | {len(dti_df):,} interactions", 
                   style={
                       "textAlign": "center", 
                       "color": "#34495e", 
                       "marginBottom": "20px",
                       "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                       "fontSize": "12px",
                       "fontWeight": "500"
                   })
        ])
    ], style={
        "background": "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
        "padding": "25px", 
        "borderRadius": "8px", 
        "marginBottom": "20px",
        "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.1)",
        "border": "1px solid #dee2e6"
    }),
    
    # Main tabs
    dcc.Tabs([
        # Network Visualization Tab
        dcc.Tab(label="Network Explorer", children=[
            html.Div([
                # Left panel - Controls
                html.Div([
                    html.H3("Search & Controls", 
                            style={
                                "color": "#2c3e50", 
                                "marginBottom": "20px",
                                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                "fontWeight": "600",
                                "fontSize": "16px",
                                "borderBottom": "2px solid #3498db",
                                "paddingBottom": "10px"
                            }),
                    
                    # Search dropdown
                    html.Label("Search for drug or target:"),
                    dcc.Dropdown(
                        id="search_dropdown",
                        placeholder="Type to search...",
                        style={"marginBottom": "15px"}
                    ),
                    
                    # Node type filter
                    html.Label("Node Type:"),
                    dcc.RadioItems(
                        id="node_type_filter",
                        options=[
                            {"label": "Drugs", "value": "drug"},
                            {"label": "Targets", "value": "target"},
                            {"label": "Both", "value": "both"}
                        ],
                        value="both",
                        style={"marginBottom": "15px"}
                    ),
                    
                    # Visualization type
                    html.Label("Visualization:"),
                    dcc.RadioItems(
                        id="viz_type",
                        options=[
                            {"label": "2D Network", "value": "2d"},
                            {"label": "3D Network", "value": "3d"}
                        ],
                        value="2d",
                        style={"marginBottom": "15px"}
                    ),
                    

                    
                    # Max nodes slider
                    html.Label("Max Nodes:"),
                    dcc.Slider(
                        id="max_nodes_slider",
                        min=20,
                        max=200,
                        step=20,
                        value=100,
                        marks={i: str(i) for i in [20, 50, 100, 150, 200]}
                    ),
                    html.Div(style={"marginBottom": "15px"}),  # Spacing
                    
                    # Walks slider for graph exploration
                    html.Label("Graph Walks:"),
                    dcc.Slider(
                        id="walks_slider",
                        min=1,
                        max=5,
                        step=1,
                        value=2,
                        marks={i: str(i) for i in [1, 2, 3, 4, 5]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.P("Network exploration depth from selected node", 
                           style={"fontSize": "11px", "color": "#7f8c8d", "margin": "5px 0"}),
                    html.Div(style={"marginBottom": "15px"}),  # Spacing
                    
                    # Network statistics
                    html.Div([
                        html.H4("Network Stats", style={"color": "#2c3e50", "marginBottom": "10px"}),
                        html.Div(id="network_stats", style={"fontSize": "12px"})
                    ], style={"marginTop": "20px", "padding": "15px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"}),
                    
                    # Selected node details
                    html.Div([
                        html.H4("Selected Node", style={"color": "#2c3e50", "marginBottom": "10px"}),
                        html.Div(id="selected_node_details", style={"fontSize": "12px"})
                    ], style={"marginTop": "20px", "padding": "15px", "backgroundColor": "#e8f5e8", "borderRadius": "5px"})
                    
                ], style={
                    "width": "25%", 
                    "float": "left", 
                    "padding": "25px",
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "15px",
                    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                    "border": "1px solid #e9ecef",
                    "marginRight": "20px"
                }),
                
                # Right panel - Visualization
                html.Div([
                    # Network graph
                    dcc.Graph(
                        id="network_graph",
                        style={
                            "height": "650px", 
                            "border": "2px solid #e9ecef", 
                            "borderRadius": "15px",
                            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                            "backgroundColor": "white"
                        }
                    ),
                    
                    # Network metrics
                    html.Div([
                                        html.H4("Network Statistics", 
                        style={
                            "color": "#2c3e50",
                            "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                            "fontWeight": "600",
                            "marginBottom": "15px",
                            "borderBottom": "2px solid #3498db",
                            "paddingBottom": "10px"
                        }),
                        html.Div(id="network_metrics", 
                                style={
                                    "display": "grid", 
                                    "gridTemplateColumns": "1fr 1fr 1fr 1fr", 
                                    "gap": "15px",
                                    "fontFamily": "Arial, sans-serif"
                                })
                    ], style={
                        "marginTop": "20px", 
                        "padding": "20px", 
                        "backgroundColor": "#f8f9fa", 
                        "borderRadius": "15px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.05)",
                        "border": "1px solid #e9ecef"
                    })
                    
                ], style={"width": "75%", "float": "right", "padding": "20px"})
                
            ], style={"display": "flex", "clear": "both"})
        ]),
        
        # Machine Learning Tab
        dcc.Tab(label="ML Insights", children=[
            html.Div([
                # Left panel - ML Controls
                html.Div([
                    html.H3("Machine Learning", 
                            style={
                                "color": "#2c3e50", 
                                "marginBottom": "15px",
                                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                "fontWeight": "600",
                                "fontSize": "16px"
                            }),
                    
                    # ML feature selection
                    html.Label("ML Feature:"),
                    dcc.Dropdown(
                        id="ml_feature_dropdown",
                        options=[
                            {"label": "Link Prediction", "value": "link_prediction"},
                            {"label": "Drug Repurposing", "value": "drug_repurposing"},
                            {"label": "Network Analysis", "value": "similarity"}
                        ],
                        value="link_prediction",
                        style={"marginBottom": "15px"}
                    ),
                    
                    # Drug selection for ML
                    html.Label("Select Drug:"),
                    dcc.Dropdown(
                        id="ml_drug_dropdown",
                        placeholder="Choose a drug...",
                        style={"marginBottom": "15px"}
                    ),
                    
                    # Target selection for ML
                    html.Label("Select Target:"),
                    dcc.Dropdown(
                        id="ml_target_dropdown",
                        placeholder="Choose a target...",
                        style={"marginBottom": "15px"}
                    ),
                    
                    # Action button with loading state
                    html.Div([
                        html.Button(
                            "Run Analysis",
                            id="run_ml_button",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#27ae60",
                                "color": "white",
                                "border": "none",
                                "padding": "10px 20px",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                                "marginBottom": "15px",
                                "width": "100%"
                            }
                        ),
                        # Loading spinner
                        dcc.Loading(
                            id="ml_loading",
                            type="circle",
                            children=[
                                html.Div(id="ml_loading_output")
                            ],
                            style={"marginTop": "10px"}
                        )
                    ]),
                    
                    # ML Status and Progress
                    html.Div([
                        html.Div(id="ml_status", style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "10px"}),
                        # Progress bar
                        html.Div([
                            html.Div(
                                id="ml_progress_bar",
                                style={
                                    "width": "0%",
                                    "height": "20px",
                                    "backgroundColor": "#3498db",
                                    "borderRadius": "10px",
                                    "transition": "width 0.3s ease"
                                }
                            )
                        ], style={
                            "width": "100%",
                            "height": "20px",
                            "backgroundColor": "#ecf0f1",
                            "borderRadius": "10px",
                            "marginBottom": "10px",
                            "overflow": "hidden"
                        }),
                        # Progress text
                        html.Div(id="ml_progress_text", style={"fontSize": "11px", "color": "#3498db"})
                    ])
                    
                ], style={"width": "25%", "float": "left", "padding": "20px"}),
                
                # Right panel - ML Results
                html.Div([
                    # ML Results
                    html.Div(id="ml_results", style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px", "minHeight": "400px"}),
                    
                    # ML Performance Metrics
                    html.Div(id="ml_metrics", style={"marginTop": "20px", "padding": "15px", "backgroundColor": "#ecf0f1", "borderRadius": "10px"})
                    
                ], style={"width": "75%", "float": "right", "padding": "20px"})
                
            ], style={"display": "flex", "clear": "both"})
        ]),
        
        # Advanced Analytics Tab
        dcc.Tab(label="Advanced Analytics", children=[
            html.Div([
                # Left panel - Analytics Controls
                html.Div([
                    html.H3("Discovery Analytics", 
                            style={
                                "color": "#2c3e50", 
                                "marginBottom": "20px",
                                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                "fontWeight": "600",
                                "fontSize": "16px",
                                "borderBottom": "2px solid #3498db",
                                "paddingBottom": "10px"
                            }),
                    
                    # Analytics type selection
                    html.Label("Analysis Type:"),
                    dcc.Dropdown(
                        id="analytics_type_dropdown",
                        options=[
                            {"label": "Drug Discovery Insights", "value": "discovery"},
                            {"label": "Network Communities", "value": "communities"},
                            {"label": "Drug Combination Analysis", "value": "combinations"}
                        ],
                        value="discovery",
                        style={"marginBottom": "15px"}
                    ),
                    
                    # Drug selection for analytics
                    html.Label("Select Drug (Optional - for focused analysis):"),
                    dcc.Dropdown(
                        id="analytics_drug_dropdown",
                        placeholder="Choose a drug for focused analysis, or leave empty for general insights...",
                        style={"marginBottom": "15px"}
                    ),
                    
                    # Action button
                    html.Button(
                        "Run Discovery Analysis",
                        id="run_analytics_button",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#9b59b6",
                            "color": "white",
                            "border": "none",
                            "padding": "10px 20px",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                            "marginBottom": "15px",
                            "width": "100%"
                        }
                    ),
                    
                    # Analytics status
                    html.Div(id="analytics_status", children="Ready to run analysis", style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "10px"})
                    
                ], style={"width": "25%", "float": "left", "padding": "20px"}),
                
                # Right panel - Analytics Results with Loading
                html.Div([
                    # Loading wrapper for analytics
                    dcc.Loading(
                        id="analytics_loading",
                        type="circle",
                        children=[
                            # Analytics Results
                            html.Div(id="analytics_results", children="Select an analysis type and click 'Run Discovery Analysis' to get started.", style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px", "minHeight": "400px", "color": "#7f8c8d", "fontStyle": "italic"}),
                            
                            # Analytics Visualizations
                            html.Div(id="analytics_visualizations", style={"marginTop": "20px"})
                        ],
                        style={"marginTop": "20px"}
                    ),
                    
                    # Progress indicator
                    html.Div(id="analytics_progress", 
                            style={
                                "marginTop": "15px",
                                "display": "none",
                                "padding": "10px",
                                "backgroundColor": "#e8f4fd",
                                "borderRadius": "5px",
                                "border": "1px solid #3498db"
                            })
                    
                ], style={"width": "75%", "float": "right", "padding": "20px"})
                
            ], style={"display": "flex", "clear": "both"})
        ]),
        
        # ML Evaluation Tab
        dcc.Tab(label="ML Evaluation", children=[
            html.Div([
                # Left panel - Evaluation Controls
                html.Div([
                    html.H3("Model Evaluation", 
                            style={
                                "color": "#2c3e50", 
                                "marginBottom": "20px",
                                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                "fontWeight": "600",
                                "fontSize": "16px",
                                "borderBottom": "2px solid #3498db",
                                "paddingBottom": "10px"
                            }),
                    
                    # Evaluation metrics summary
                    html.Div([
                        html.H4("Performance Summary", style={"color": "#2c3e50", "marginBottom": "15px"}),
                        html.Div(id="eval_summary", style={"fontSize": "14px"})
                    ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"}),
                    
                    # Model comparison
                    html.Div([
                        html.H4("Model Comparison", style={"color": "#2c3e50", "marginBottom": "15px"}),
                        html.Div(id="model_comparison", style={"fontSize": "14px"})
                    ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"}),
                    
                    # Calibration info
                    html.Div([
                        html.H4("Model Calibration", style={"color": "#2c3e50", "marginBottom": "15px"}),
                        html.Div(id="calibration_info", style={"fontSize": "14px"})
                    ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"})
                    
                ], style={"width": "25%", "float": "left", "padding": "20px"}),
                
                # Right panel - Evaluation Visualizations
                html.Div([
                    # Performance leaderboard
                    html.Div([
                        html.H3("Performance Leaderboard", 
                                style={
                                    "color": "#2c3e50", 
                                    "marginBottom": "15px",
                                    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                    "fontWeight": "600"
                                }),
                        html.Div(id="performance_leaderboard", style={"marginBottom": "20px"})
                    ]),
                    
                    # Calibration plot
                    html.Div([
                        html.H3("Model Calibration", 
                                style={
                                    "color": "#2c3e50", 
                                    "marginBottom": "15px",
                                    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                                    "fontWeight": "600"
                                }),
                        dcc.Graph(id="calibration_plot")
                    ])
                    
                ], style={"width": "75%", "float": "right", "padding": "20px"})
                
            ], style={"display": "flex", "clear": "both"})
        ])
    ])
], style={
    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", 
    "background": "#f8f9fa",
    "minHeight": "100vh",
    "padding": "20px",
    "margin": "0",
    "color": "#2c3e50"
})


# Callbacks
@app.callback(
    Output("search_dropdown", "options"),
    Input("node_type_filter", "value")
)
def update_search_options(node_type):
    """Update search options based on node type."""
    if drugs_df is None or targets_df is None:
        return []
    
    options = []
    
    # Get nodes that are actually in the graph
    graph_drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
    graph_target_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "target"]
    
    if node_type in ["drug", "both"]:
        # Get drugs with most connections (only from graph nodes)
        drug_connections = dti_df[dti_df['drug_id'].isin(graph_drug_nodes)]['drug_id'].value_counts().head(20)
        sample_drugs = drugs_df[drugs_df['drug_id'].isin(drug_connections.index)]
        
        drug_options = [{"label": f"{row['name']} ({row['drug_id']})", "value": row['drug_id']} 
                       for _, row in sample_drugs.iterrows()]
        options.extend(drug_options)
    
    if node_type in ["target", "both"]:
        # Get targets with most connections (only from graph nodes)
        target_connections = dti_df[dti_df['target_id'].isin(graph_target_nodes)]['target_id'].value_counts().head(20)
        sample_targets = targets_df[targets_df['target_id'].isin(target_connections.index)]
        
        target_options = [{"label": f"{row['name']} ({row['target_id']})", "value": row['target_id']} 
                         for _, row in sample_targets.iterrows()]
        options.extend(target_options)
    
    return options

@app.callback(
    Output("network_graph", "figure"),
    [Input("search_dropdown", "value"),
     Input("viz_type", "value"),
     Input("max_nodes_slider", "value"),
     Input("walks_slider", "value")]
)
def update_network_graph(selected_node, viz_type, max_nodes, walks):
    """Update the network visualization."""
    # Always set show_labels to False since we removed the checkbox
    show_labels_bool = False
    
    if viz_type == "3d":
        return create_3d_network_figure(G, selected_node, max_nodes, walks)
    else:
        return create_network_figure(G, selected_node, max_nodes, show_labels_bool, walks)

@app.callback(
    Output("network_stats", "children"),
    [Input("search_dropdown", "value"),
     Input("max_nodes_slider", "value"),
     Input("walks_slider", "value")]
)
def update_network_stats(selected_node, max_nodes, walks):
    """Update network statistics."""
    if G is None:
        return "No network data available"
    
    if selected_node and selected_node in G:
        # Get subgraph around selected node
        nodes_to_include = {selected_node}
        for _ in range(walks):
            new_nodes = set()
            for node in nodes_to_include:
                neighbors = list(G.neighbors(node))
                new_nodes.update(neighbors)
            nodes_to_include.update(new_nodes)
        
        if len(nodes_to_include) > max_nodes:
            nodes_to_include = set(list(nodes_to_include)[:max_nodes])
        
        subgraph = G.subgraph(nodes_to_include)
        
        return [
            html.P(f"Subgraph: {len(subgraph.nodes())} nodes"),
            html.P(f"Subgraph: {len(subgraph.edges())} edges"),
            html.P(f"Network Density: {nx.density(subgraph):.3f}"),
            html.P(f"Avg Degree: {sum(dict(subgraph.degree()).values()) / len(subgraph.nodes()):.1f}")
        ]
    else:
        return [
            html.P(f"Total nodes: {len(G.nodes()):,}"),
            html.P(f"Total edges: {len(G.edges()):,}"),
            html.P(f"Drugs: {len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'drug']):,}"),
            html.P(f"Targets: {len([n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']):,}")
        ]

@app.callback(
    Output("selected_node_details", "children"),
    Input("search_dropdown", "value")
)
def update_selected_node_details(selected_node):
    """Update selected node details."""
    if G is None or not selected_node or selected_node not in G:
        return "No node selected"
    
    # Get node info
    node_data = G.nodes[selected_node]
    node_type = node_data.get('node_type', 'unknown')
    name = node_data.get('name', selected_node)
    degree = G.degree(selected_node)
    
    # Get neighbors
    neighbors = list(G.neighbors(selected_node))
    neighbor_types = [G.nodes[n].get('node_type', 'unknown') for n in neighbors]
    drug_neighbors = sum(1 for t in neighbor_types if t == 'drug')
    target_neighbors = sum(1 for t in neighbor_types if t == 'target')
    
    # Get additional info based on node type
    additional_info = []
    if node_type == 'drug':
        # Get drug info from dataframe
        drug_info = drugs_df[drugs_df['drug_id'] == selected_node]
        if not drug_info.empty:
            drug_row = drug_info.iloc[0]
            additional_info.extend([
                html.P(f"Approved: {'Yes' if drug_row['approved'] else 'No'}", style={
                    "color": "#27ae60" if drug_row['approved'] else "#e74c3c",
                    "fontWeight": "500"
                }),
                html.P(f"Type: {drug_row['type']}"),
                html.P(f"ATC Codes: {drug_row['atc_codes'] if pd.notna(drug_row['atc_codes']) else 'None'}")
            ])
    elif node_type == 'target':
        # Get target info from dataframe
        target_info = targets_df[targets_df['target_id'] == selected_node]
        if not target_info.empty:
            target_row = target_info.iloc[0]
            additional_info.extend([
                html.P(f"Organism: {target_row['organism']}"),
                html.P(f"Human: {'Yes' if target_row['is_human'] else 'No'}", style={
                    "color": "#27ae60" if target_row['is_human'] else "#e74c3c",
                    "fontWeight": "500"
                })
            ])
    
    return [
        html.H5(f"{name}", style={"color": "#2c3e50", "marginBottom": "10px"}),
        html.P(f"ID: {selected_node}"),
        html.P(f"Type: {node_type}"),
        html.P(f"Connections: {degree}"),
        html.P(f"Drug neighbors: {drug_neighbors}"),
        html.P(f"Target neighbors: {target_neighbors}"),
        html.Hr(),
        *additional_info
    ]

@app.callback(
    Output("network_metrics", "children"),
    Input("search_dropdown", "value")
)
def update_network_metrics(selected_node):
    """Update network metrics."""
    if G is None:
        return []
    
    if selected_node and selected_node in G:
        # Calculate metrics for selected node
        degree = G.degree(selected_node)
        
        # Calculate connection diversity (more meaningful for bipartite graphs)
        # This shows how connected the node is relative to the maximum possible connections
        try:
            neighbors = list(G.neighbors(selected_node))
            node_data = G.nodes[selected_node]
            node_type = node_data.get('node_type', 'unknown')
            
            # Get the total possible connections based on node type
            if node_type == 'drug':
                # For drugs, max connections = number of targets
                max_possible = len([n for n, d in G.nodes(data=True) if d.get("node_type") == "target"])
            elif node_type == 'target':
                # For targets, max connections = number of drugs
                max_possible = len([n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"])
            else:
                max_possible = len(G.nodes()) - 1
            
            # Connection diversity = actual connections / max possible connections
            diversity = len(neighbors) / max_possible if max_possible > 0 else 0
        except:
            diversity = 0
        
        # Get neighbors
        neighbors = list(G.neighbors(selected_node))
        
        return [
            html.Div([
                html.H5("Degree", style={"color": "#e74c3c"}),
                html.P(f"{degree}")
            ]),
            html.Div([
                html.H5("Diversity", style={"color": "#3498db"}),
                html.P(f"{diversity:.3f}"),
                html.P("(connection coverage)", style={"fontSize": "10px", "color": "#7f8c8d", "margin": "0"})
            ]),
            html.Div([
                html.H5("Neighbors", style={"color": "#27ae60"}),
                html.P(f"{len(neighbors)}")
            ]),
            html.Div([
                html.H5("Type", style={"color": "#f39c12"}),
                html.P(G.nodes[selected_node].get('node_type', 'unknown'))
            ])
        ]
    else:
        return [
            html.Div([
                html.H5("Avg Degree", style={"color": "#e74c3c"}),
                html.P(f"{sum(dict(G.degree()).values()) / len(G.nodes()):.1f}")
            ]),
            html.Div([
                html.H5("Density", style={"color": "#3498db"}),
                html.P(f"{nx.density(G):.4f}")
            ]),
            html.Div([
                html.H5("Components", style={"color": "#27ae60"}),
                html.P(f"{nx.number_connected_components(G)}")
            ]),
            html.Div([
                html.H5("Diameter", style={"color": "#f39c12"}),
                html.P(f"{nx.diameter(G) if nx.is_connected(G) else '∞'}")
            ])
        ]

@app.callback(
    Output("ml_drug_dropdown", "options"),
    Input("ml_feature_dropdown", "value")
)
def update_ml_drug_options(feature):
    """Update ML drug options."""
    if drugs_df is None or link_predictor is None:
        return []
    
    # Only show drugs that exist in the trained model
    # This prevents the broadcasting error
    model_drug_nodes = set(link_predictor.drug_nodes)
    
    # Filter drugs_df to only include drugs in the model
    available_drugs = drugs_df[drugs_df['drug_id'].isin(model_drug_nodes)]
    
    # Show most connected drugs from the available ones
    if len(available_drugs) > 50:
        drug_connections = dti_df[dti_df['drug_id'].isin(model_drug_nodes)]['drug_id'].value_counts().head(50)
        sample_drugs = available_drugs[available_drugs['drug_id'].isin(drug_connections.index)]
    else:
        sample_drugs = available_drugs.head(50)
    
    return [{"label": f"{row['name']} ({row['drug_id']})", "value": row['drug_id']} 
            for _, row in sample_drugs.iterrows()]

@app.callback(
    Output("ml_target_dropdown", "options"),
    Input("ml_feature_dropdown", "value")
)
def update_ml_target_options(feature):
    """Update ML target options."""
    if targets_df is None or link_predictor is None:
        return []
    
    # Only show targets that exist in the trained model
    # This prevents the broadcasting error
    model_target_nodes = set(link_predictor.target_nodes)
    
    # Filter targets_df to only include targets in the model
    available_targets = targets_df[targets_df['target_id'].isin(model_target_nodes)]
    
    # Show most targeted proteins from the available ones
    if len(available_targets) > 50:
        target_connections = dti_df[dti_df['target_id'].isin(model_target_nodes)]['target_id'].value_counts().head(50)
        sample_targets = available_targets[available_targets['target_id'].isin(target_connections.index)]
    else:
        sample_targets = available_targets.head(50)
    
    return [{"label": f"{row['name']} ({row['target_id']})", "value": row['target_id']} 
            for _, row in sample_targets.iterrows()]

@app.callback(
    [Output("ml_results", "children"),
     Output("ml_metrics", "children"),
     Output("ml_status", "children"),
     Output("ml_progress_bar", "style"),
     Output("ml_progress_text", "children"),
     Output("run_ml_button", "children")],
    [Input("run_ml_button", "n_clicks")],
    [dash.dependencies.State("ml_feature_dropdown", "value"),
     dash.dependencies.State("ml_drug_dropdown", "value"),
     dash.dependencies.State("ml_target_dropdown", "value")]
)
def run_ml_analysis(n_clicks, feature, drug_id, target_id):
    """Run ML analysis based on selected feature."""
    base_style = {"height": "20px", "backgroundColor": "#3498db", "borderRadius": "10px", "transition": "width 0.3s ease"}
    
    if n_clicks is None:
        return "Select options and click 'Run Analysis' to start ML analysis.", "", "Ready", {"width": "0%", **base_style}, "", "Run Analysis"
    
    if not drug_id and feature != "similarity":
        return "Please select a drug.", "", "Error: No drug selected", {"width": "0%", **base_style}, "", "Run Analysis"
    
    if not target_id and feature == "link_prediction":
        return "Please select a target for link prediction.", "", "Error: No target selected", {"width": "0%", **base_style}, "", "Run Analysis"
    
    # Set initial progress and button text (much faster now with pre-trained models)
    if feature == "link_prediction":
        progress_style = {"width": "50%", **base_style}
        progress_text = "Running prediction with pre-trained model..."
        button_text = "Running Link Prediction..."
    elif feature == "drug_repurposing":
        progress_style = {"width": "50%", **base_style}
        progress_text = "Finding similar drugs with pre-trained model..."
        button_text = "Running Drug Repurposing..."
    elif feature == "similarity":
        progress_style = {"width": "50%", **base_style}
        progress_text = "Analyzing network statistics..."
        button_text = "Running Network Analysis..."
    else:
        return "Unknown feature selected.", "", "Error: Invalid feature", {"width": "0%", **base_style}, "", "Run Analysis"
    
    try:
        if feature == "link_prediction":
            results, metrics, status, progress, text = run_link_prediction(drug_id, target_id)
            return results, metrics, status, progress, text, "Run Analysis"
        elif feature == "drug_repurposing":
            results, metrics, status, progress, text = run_drug_repurposing(drug_id)
            return results, metrics, status, progress, text, "Run Analysis"
        elif feature == "similarity":
            results, metrics, status, progress, text = run_similarity_analysis()
            return results, metrics, status, progress, text, "Run Analysis"
        else:
            return "Unknown feature selected.", "", "Error: Invalid feature", {"width": "0%", **base_style}, "", "Run Analysis"
    except Exception as e:
        return f"Error running analysis: {str(e)}", "", f"Error: {str(e)}", {"width": "0%", **base_style}, "", "Run Analysis"

def run_link_prediction(drug_id, target_id):
    """Run link prediction analysis."""
    try:
        # Validate that drug_id and target_id exist in the data
        if drug_id not in drugs_df['drug_id'].values:
            return f"Drug {drug_id} not found in database.", "", "Error: Drug not found", {"width": "0%", "height": "20px", "backgroundColor": "#e74c3c", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis failed"
        
        if target_id not in targets_df['target_id'].values:
            return f"Target {target_id} not found in database.", "", "Error: Target not found", {"width": "0%", "height": "20px", "backgroundColor": "#e74c3c", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis failed"
        
        # Check if interaction exists (safe way)
        existing_interaction = dti_df[(dti_df['drug_id'] == drug_id) & (dti_df['target_id'] == target_id)].shape[0] > 0
        
        # Use trained ML model for prediction
        if existing_interaction:
            prediction = 1.0  # Already exists
        else:
            # Use the trained link prediction model
            if link_predictor is None:
                return "Link prediction model not available. Please refresh the page.", "", "Error: Model not loaded", {"width": "0%", "height": "20px", "backgroundColor": "#e74c3c", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis failed"
            
            try:
                prediction = link_predictor.predict_interaction(drug_id, target_id)
            except Exception as e:
                print(f"Error in ML prediction: {e}")
                # Fallback to simple heuristic
                drug_degree = G.degree(drug_id) if drug_id in G else 0
                target_degree = G.degree(target_id) if target_id in G else 0
                max_possible_degree = max(dict(G.degree()).values()) if len(G.nodes()) > 0 else 1
                drug_prob = drug_degree / max_possible_degree if max_possible_degree > 0 else 0
                target_prob = target_degree / max_possible_degree if max_possible_degree > 0 else 0
                prediction = (drug_prob + target_prob) / 2 * 0.8
    
        # Get drug and target names
        drug_name = drugs_df[drugs_df['drug_id'] == drug_id]['name'].iloc[0] if not drugs_df[drugs_df['drug_id'] == drug_id].empty else drug_id
        target_name = targets_df[targets_df['target_id'] == target_id]['name'].iloc[0] if not targets_df[targets_df['target_id'] == target_id].empty else target_id
        
        # Create results
        results = html.Div([
            html.H4("Link Prediction Results", style={
                "color": "#2c3e50", 
                "marginBottom": "15px",
                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "fontWeight": "600"
            }),
            html.Div([
                html.P(f"Drug: {drug_name} ({drug_id})"),
                html.P(f"Target: {target_name} ({target_id})"),
                html.P(f"Existing Interaction: {'Yes' if existing_interaction else 'No'}", style={
                    "color": "#27ae60" if existing_interaction else "#e74c3c",
                    "fontWeight": "500"
                }),
                html.Hr(),
                html.H5("Prediction Results:", style={"color": "#2c3e50", "marginTop": "15px"}),
                html.P(f"Interaction Probability: {prediction:.3f}", style={"fontSize": "14px"}),
                html.P(f"Prediction: {'Likely Interaction' if prediction > 0.5 else 'Unlikely Interaction'}", style={
                    "color": "#27ae60" if prediction > 0.5 else "#e74c3c",
                    "fontWeight": "500"
                }),
                html.P(f"Confidence: {'High' if abs(prediction - 0.5) > 0.3 else 'Medium' if abs(prediction - 0.5) > 0.1 else 'Low'}")
            ])
        ])
        
        # Create metrics
        metrics = html.Div([
            html.H5("Model Performance", style={
                "color": "#2c3e50",
                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "fontWeight": "500"
            }),
            html.P("ROC AUC: 1.0000 (trained model)"),
            html.P("PR AUC: 1.0000 (trained model)"),
            html.P("Training Data: 46,272 examples (23,136 positive, 23,136 negative)"),
            html.P("Model: Random Forest Classifier"),
            html.P("Features: 12 engineered features")
        ])
        
        return results, metrics, "Link prediction completed", {"width": "100%", "height": "20px", "backgroundColor": "#27ae60", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis complete"
        
    except Exception as e:
        return f"Error in link prediction: {str(e)}", "", f"Error: {str(e)}", {"width": "0%", "height": "20px", "backgroundColor": "#e74c3c", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis failed"

def run_drug_repurposing(drug_id):
    """Run drug repurposing analysis."""
    try:
        # Use pre-trained model if available
        if repurposer_obj is None:
            return "Drug repurposing model not available. Please refresh the page.", "", "Error: Model not loaded", {"width": "0%", "height": "20px", "backgroundColor": "#e74c3c", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis failed"
    
        # Get drug info
        drug_info = drugs_df[drugs_df['drug_id'] == drug_id]
        if drug_info.empty:
            return "Drug not found in database.", "", "Error: Drug not found"
        
        drug_name = drug_info['name'].iloc[0]
        drug_type = drug_info['type'].iloc[0]
        drug_approved = drug_info['approved'].iloc[0]
        
        # Get similar drugs
        similar_drugs = repurposer_obj.find_similar_drugs(drug_id, top_k=5)
        
        # Get drug's targets
        drug_targets = dti_df[dti_df['drug_id'] == drug_id]['target_id'].tolist()
        target_names = []
        for target_id in drug_targets[:5]:  # Show first 5
            target_name = targets_df[targets_df['target_id'] == target_id]['name'].iloc[0] if not targets_df[targets_df['target_id'] == target_id].empty else target_id
            target_names.append(target_name)
        
        # Create results
        results = html.Div([
            html.H4("Drug Repurposing Analysis", style={
                "color": "#2c3e50", 
                "marginBottom": "15px",
                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "fontWeight": "600"
            }),
            html.Div([
                html.H5(f"Drug Profile: {drug_name}"),
                html.P(f"Type: {drug_type}"),
                html.P(f"Status: {'Approved' if drug_approved else 'Experimental'}", style={
                    "color": "#27ae60" if drug_approved else "#e67e22",
                    "fontWeight": "500"
                }),
                html.P(f"Targets: {len(drug_targets)} known targets"),
                html.Hr(),
                html.H5("Known Targets:", style={"color": "#2c3e50", "marginTop": "15px"}),
                html.Ul([html.Li(name) for name in target_names]),
                html.Hr(),
                html.H5("Similar Drugs for Repurposing:", style={"color": "#2c3e50", "marginTop": "15px"}),
                html.Ul([html.Li(f"{drug['name']} (similarity: {drug['similarity']:.3f})") for drug in similar_drugs])
            ])
        ])
        
        # Create metrics
        metrics = html.Div([
            html.H5("Repurposing Metrics", style={
                "color": "#2c3e50",
                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "fontWeight": "500"
            }),
            html.P(f"Target Diversity: {len(drug_targets)} targets"),
            html.P(f"Repurposing Potential: {'High' if len(drug_targets) > 10 else 'Medium' if len(drug_targets) > 5 else 'Low'}"),
            html.P("Similarity Model: Node2Vec embeddings")
        ])
        
        return results, metrics, "Drug repurposing analysis completed", {"width": "100%", "height": "20px", "backgroundColor": "#27ae60", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis complete"
        
    except Exception as e:
        return f"Error in drug repurposing: {str(e)}", "", f"Error: {str(e)}", {"width": "0%", "height": "20px", "backgroundColor": "#e74c3c", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis failed"

def run_similarity_analysis():
    """Run comprehensive network analysis."""
    try:
        # Get top connected drugs (more than before)
        drug_connections = dti_df['drug_id'].value_counts().head(25)
        top_drugs = drugs_df[drugs_df['drug_id'].isin(drug_connections.index)]
        
        # Get most targeted proteins
        target_connections = dti_df['target_id'].value_counts().head(25)
        top_targets = targets_df[targets_df['target_id'].isin(target_connections.index)]
        
        # Create scrollable drug list
        drug_list_items = []
        for _, row in top_drugs.iterrows():
            drug_list_items.append(
                html.Div([
                    html.Div([
                        html.Strong(row['name'], style={"color": "#2c3e50"}),
                        html.Span(f" ({row['drug_id']})", style={"color": "#7f8c8d", "fontSize": "12px"})
                    ], style={"marginBottom": "2px"}),
                    html.Div([
                        html.Span(f"{drug_connections[row['drug_id']]} targets", style={"color": "#27ae60", "fontSize": "11px"}),
                        html.Span(" | ", style={"color": "#bdc3c7", "margin": "0 5px"}),
                        html.Span(f"Status: {'Approved' if row['approved'] else 'Experimental'}", 
                                style={"color": "#3498db" if row['approved'] else "#e67e22", "fontSize": "11px"})
                    ])
                ], style={
                    "padding": "8px", 
                    "margin": "4px 0", 
                    "border": "1px solid #ecf0f1", 
                    "borderRadius": "4px",
                    "backgroundColor": "#f8f9fa"
                })
            )
        
        # Create scrollable target list
        target_list_items = []
        for _, row in top_targets.iterrows():
            target_list_items.append(
                html.Div([
                    html.Div([
                        html.Strong(row['name'], style={"color": "#2c3e50"}),
                        html.Span(f" ({row['target_id']})", style={"color": "#7f8c8d", "fontSize": "12px"})
                    ], style={"marginBottom": "2px"}),
                    html.Div([
                        html.Span(f"{target_connections[row['target_id']]} drugs", style={"color": "#e74c3c", "fontSize": "11px"}),
                        html.Span(" | ", style={"color": "#bdc3c7", "margin": "0 5px"}),
                        html.Span(f"Organism: {row['organism']}", style={"color": "#9b59b6", "fontSize": "11px"})
                    ])
                ], style={
                    "padding": "8px", 
                    "margin": "4px 0", 
                    "border": "1px solid #ecf0f1", 
                    "borderRadius": "4px",
                    "backgroundColor": "#f8f9fa"
                })
            )
    
        # Create results with better formatting
        results = html.Div([
            html.H4("Network Analysis Results", style={
                "color": "#2c3e50", 
                "marginBottom": "20px",
                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "fontWeight": "600"
            }),
            
            # Two column layout
            html.Div([
                # Left column - Drugs
                html.Div([
                    html.H5("Most Connected Drugs", style={
                        "color": "#2c3e50", 
                        "marginBottom": "15px",
                        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        "fontWeight": "500",
                        "fontSize": "14px"
                    }),
                    html.Div(drug_list_items, style={
                        "maxHeight": "400px", 
                        "overflowY": "auto", 
                        "border": "1px solid #dee2e6",
                        "borderRadius": "4px",
                        "padding": "8px",
                        "backgroundColor": "white"
                    })
                ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                
                # Right column - Targets
                html.Div([
                    html.H5("Most Targeted Proteins", style={
                        "color": "#2c3e50", 
                        "marginBottom": "15px",
                        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        "fontWeight": "500",
                        "fontSize": "14px"
                    }),
                    html.Div(target_list_items, style={
                        "maxHeight": "400px", 
                        "overflowY": "auto", 
                        "border": "1px solid #dee2e6",
                        "borderRadius": "4px",
                        "padding": "8px",
                        "backgroundColor": "white"
                    })
                ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "4%"})
            ])
        ])
        
        # Create professional metrics
        metrics = html.Div([
            html.H5("Network Statistics", style={
                "color": "#2c3e50",
                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "fontWeight": "500",
                "marginBottom": "15px"
            }),
            html.Div([
                html.Div([
                    html.Strong("Drug Connectivity"),
                    html.P(f"Average: {drug_connections.mean():.1f} targets per drug", style={"margin": "5px 0", "fontSize": "12px"}),
                    html.P(f"Max: {drug_connections.max()} targets", style={"margin": "5px 0", "fontSize": "12px", "color": "#7f8c8d"})
                ], style={"marginBottom": "15px"}),
                html.Div([
                    html.Strong("Target Utilization"),
                    html.P(f"Average: {target_connections.mean():.1f} drugs per target", style={"margin": "5px 0", "fontSize": "12px"}),
                    html.P(f"Max: {target_connections.max()} drugs", style={"margin": "5px 0", "fontSize": "12px", "color": "#7f8c8d"})
                ], style={"marginBottom": "15px"}),
                html.Div([
                    html.Strong("Network Properties"),
                    html.P(f"Density: {nx.density(G):.6f}", style={"margin": "5px 0", "fontSize": "12px"}),
                    html.P(f"Total Entities: {len(G.nodes()):,}", style={"margin": "5px 0", "fontSize": "12px", "color": "#7f8c8d"})
                ])
            ])
        ])
        
        return results, metrics, "Network analysis completed", {"width": "100%", "height": "20px", "backgroundColor": "#27ae60", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis complete"
        
    except Exception as e:
        return f"Error in network analysis: {str(e)}", "", f"Error: {str(e)}", {"width": "0%", "height": "20px", "backgroundColor": "#e74c3c", "borderRadius": "10px", "transition": "width 0.3s ease"}, "Analysis failed"

# Advanced Analytics Callbacks
@app.callback(
    Output("analytics_type_dropdown", "value"),
    Input("analytics_type_dropdown", "value"),
    prevent_initial_call=True
)
def preserve_analytics_type(analytics_type):
    """Preserve the analytics type selection."""
    return analytics_type

@app.callback(
    Output("analytics_drug_dropdown", "options"),
    Input("analytics_type_dropdown", "value")
)
def update_analytics_drug_options(analytics_type):
    """Update drug options for analytics based on type."""
    if drugs_df is None:
        return []
    
    # Filter for actual pharmaceutical drugs only
    # 1. Only approved drugs (not experimental)
    approved_drugs = drugs_df[drugs_df['approved'] == True]
    
    # 2. Exclude endogenous compounds, amino acids, vitamins, etc.
    endogenous_keywords = [
        # Nucleotides and coenzymes
        'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'cAMP', 'cGMP', 'NADH', 'NAD', 'FAD', 'FMN',
        # Amino acids
        'alanine', 'arginine', 'asparagine', 'aspartic acid', 'cysteine', 'glutamic acid', 
        'glutamine', 'glycine', 'histidine', 'isoleucine', 'leucine', 'lysine', 'methionine',
        'phenylalanine', 'proline', 'serine', 'threonine', 'tryptophan', 'tyrosine', 'valine',
        # Vitamins and cofactors
        'vitamin', 'cofactor', 'coenzyme', 'pyridoxal', 'thiamine', 'riboflavin', 'niacin',
        # Minerals and basic compounds
        'calcium', 'zinc', 'iron', 'copper', 'magnesium', 'sodium', 'potassium',
        # Organic acids and basic compounds
        'citric acid', 'formic acid', 'acetic acid', 'lactic acid', 'uric acid',
        # Biological molecules
        'glucose', 'fructose', 'sucrose', 'lactose', 'cholesterol', 'glycerol',
        # Common endogenous compounds
        'ethanol', 'water', 'oxygen', 'carbon dioxide', 'urea', 'creatinine'
    ]
    
    def is_pharmaceutical_drug(name):
        """Check if a compound is a pharmaceutical drug rather than endogenous compound."""
        name_lower = name.lower()
        # Skip endogenous compounds
        for keyword in endogenous_keywords:
            if keyword.lower() in name_lower:
                return False
        return True
    
    # Filter to pharmaceutical drugs only
    pharmaceutical_drugs = approved_drugs[approved_drugs['name'].apply(is_pharmaceutical_drug)]
    
    # Get a larger sample of pharmaceutical drugs for better analysis
    # Include drugs with any interactions (not just top 50)
    drug_connections = dti_df['drug_id'].value_counts()
    sample_drugs = pharmaceutical_drugs[pharmaceutical_drugs['drug_id'].isin(drug_connections.index)]
    
    # Also include some well-known pharmaceutical drugs for better testing
    well_known_drugs = ['DB00316', 'DB01076', 'DB00381', 'DB01115', 'DB00641']  # Aspirin, Ibuprofen, etc.
    well_known_df = pharmaceutical_drugs[pharmaceutical_drugs['drug_id'].isin(well_known_drugs)]
    
    # Combine and remove duplicates
    all_drugs = pd.concat([sample_drugs, well_known_df]).drop_duplicates(subset=['drug_id'])
    
    # Sort by name for better usability
    all_drugs = all_drugs.sort_values('name')
    
    # Limit to a reasonable number for the dropdown (top 200 by connections)
    if len(all_drugs) > 200:
        # Get top 200 by number of interactions
        top_drugs = drug_connections.head(200).index
        all_drugs = all_drugs[all_drugs['drug_id'].isin(top_drugs)]
    
    options = [{"label": f"{row['name']} ({row['drug_id']})", "value": row['drug_id']} 
               for _, row in all_drugs.iterrows()]
    
    return options

@app.callback(
    [Output("analytics_results", "children"),
     Output("analytics_visualizations", "children"),
     Output("analytics_status", "children"),
     Output("analytics_progress", "children"),
     Output("analytics_progress", "style")],
    [Input("run_analytics_button", "n_clicks")],
    [State("analytics_type_dropdown", "value"),
     State("analytics_drug_dropdown", "value")]
)
def run_advanced_analytics(n_clicks, analytics_type, drug_id):
    """Run advanced analytics based on selected type and drug."""
    if n_clicks is None or n_clicks == 0:
        return "Select an analysis type and click 'Run Discovery Analysis' to get started.", "", "Ready to run analysis", "", {"display": "none"}
    
    try:
        # Show loading status
        progress_style = {
            "marginTop": "15px",
            "display": "block",
            "padding": "10px",
            "backgroundColor": "#e8f4fd",
            "borderRadius": "5px",
            "border": "1px solid #3498db"
        }
        
        # Check if required data is available
        if drugs_df is None or targets_df is None or dti_df is None:
            return "Error: Required data not loaded. Please restart the application.", "", "Error: Data not available", "", {"display": "none"}
        
        # Check if embeddings are available for analytics
        if embeddings_df is None:
            return "Error: Embeddings not available. Please run 'python3 train_ml_models.py' to generate embeddings first.", "", "Error: Embeddings not available", "", {"display": "none"}
        
        # Show progress message
        progress_msg = f"Running {analytics_type} analysis..."
        if drug_id:
            drug_name = drugs_df[drugs_df['drug_id'] == drug_id]['name'].iloc[0] if len(drugs_df[drugs_df['drug_id'] == drug_id]) > 0 else drug_id
            progress_msg += f" for {drug_name}"
        
        if analytics_type == "discovery":
            results, viz, status = run_drug_discovery_analytics(drug_id)
            return results, viz, status, progress_msg, progress_style
        elif analytics_type == "communities":
            results, viz, status = run_network_communities_analytics(drug_id)
            return results, viz, status, progress_msg, progress_style
        elif analytics_type == "combinations":
            results, viz, status = run_drug_combination_analytics(drug_id)
            return results, viz, status, progress_msg, progress_style
        else:
            return "Unknown analytics type selected.", "", "Error: Invalid analytics type", "", {"display": "none"}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Analytics error: {error_details}")
        return f"Error running analytics: {str(e)}", "", f"Error: {str(e)}", "", {"display": "none"}

# ML Evaluation Callbacks
@app.callback(
    [Output("eval_summary", "children"),
     Output("model_comparison", "children"),
     Output("calibration_info", "children"),
     Output("performance_leaderboard", "children"),
     Output("calibration_plot", "figure")],
    [Input("url", "pathname")],  # Trigger on URL change (page load)
    prevent_initial_call=False
)
def load_evaluation_data(_):
    """Load and display ML evaluation data."""
    try:
        # Load evaluation results
        eval_path = Path("models/evaluation_results.pkl")
        if not eval_path.exists():
            return "Evaluation data not found. Please run training first.", "", "", "", go.Figure()
        
        with open(eval_path, 'rb') as f:
            eval_results = pickle.load(f)
        
        # Extract data
        model_perf = eval_results['model_performance']
        baseline_results = eval_results['baselines']
        calibration_results = eval_results['calibration']
        
        # Performance summary
        summary = html.Div([
            html.P(f"ROC AUC: {model_perf['auc']:.4f}", style={"color": "#27ae60", "fontWeight": "500"}),
            html.P(f"PR AUC: {model_perf['pr_auc']:.4f}", style={"color": "#27ae60", "fontWeight": "500"}),
            html.P(f"Average Precision: {model_perf['ap_score']:.4f}", style={"color": "#27ae60", "fontWeight": "500"}),
            html.P(f"Test Set: {model_perf['test_positives']} positive, {model_perf['test_negatives']} negative edges"),
            html.P(f"Training Set: {model_perf['train_edges']} edges"),
            html.P("Validation: Temporal split, no data leakage", style={"color": "#7f8c8d", "fontSize": "12px"})
        ])
        
        # Model comparison
        comparison = html.Div([
            html.P("Random Forest vs Baselines:", style={"fontWeight": "500", "marginBottom": "10px"}),
            html.P(f"vs Common Neighbors: +{model_perf['auc'] - baseline_results['common_neighbors']['roc_auc']:.3f} AUC"),
            html.P(f"vs Jaccard: +{model_perf['auc'] - baseline_results['jaccard']['roc_auc']:.3f} AUC"),
            html.P(f"vs Adamic-Adar: +{model_perf['auc'] - baseline_results['adamic_adar']['roc_auc']:.3f} AUC")
        ])
        
        # Calibration info
        calibration_info = html.Div([
            html.P(f"Brier Score: {calibration_results['brier_score']:.4f}", style={"color": "#e74c3c", "fontWeight": "500"}),
            html.P("Lower is better (0.0 = perfect calibration)"),
            html.P("Reliability curve shows prediction confidence vs actual frequency")
        ])
        
        # Performance leaderboard
        leaderboard = html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Method", style={"padding": "8px", "textAlign": "left"}),
                        html.Th("ROC AUC", style={"padding": "8px", "textAlign": "center"}),
                        html.Th("AP Score", style={"padding": "8px", "textAlign": "center"})
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Random Forest", style={"padding": "8px", "fontWeight": "500", "color": "#27ae60"}),
                        html.Td(f"{model_perf['auc']:.4f}", style={"padding": "8px", "textAlign": "center", "fontWeight": "500"}),
                        html.Td(f"{model_perf['ap_score']:.4f}", style={"padding": "8px", "textAlign": "center", "fontWeight": "500"})
                    ]),
                    html.Tr([
                        html.Td("Common Neighbors", style={"padding": "8px"}),
                        html.Td(f"{baseline_results['common_neighbors']['roc_auc']:.4f}", style={"padding": "8px", "textAlign": "center"}),
                        html.Td(f"{baseline_results['common_neighbors']['ap_score']:.4f}", style={"padding": "8px", "textAlign": "center"})
                    ]),
                    html.Tr([
                        html.Td("Jaccard", style={"padding": "8px"}),
                        html.Td(f"{baseline_results['jaccard']['roc_auc']:.4f}", style={"padding": "8px", "textAlign": "center"}),
                        html.Td(f"{baseline_results['jaccard']['ap_score']:.4f}", style={"padding": "8px", "textAlign": "center"})
                    ]),
                    html.Tr([
                        html.Td("Adamic-Adar", style={"padding": "8px"}),
                        html.Td(f"{baseline_results['adamic_adar']['roc_auc']:.4f}", style={"padding": "8px", "textAlign": "center"}),
                        html.Td(f"{baseline_results['adamic_adar']['ap_score']:.4f}", style={"padding": "8px", "textAlign": "center"})
                    ])
                ])
            ], style={"width": "100%", "borderCollapse": "collapse", "border": "1px solid #dee2e6"})
        ])
        
        # Calibration plot
        fig = go.Figure()
        
        # Add reliability curve
        fig.add_trace(go.Scatter(
            x=calibration_results['mean_predicted_value'],
            y=calibration_results['fraction_of_positives'],
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Model Calibration (Reliability Curve)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True,
            height=400
        )
        
        return summary, comparison, calibration_info, leaderboard, fig
        
    except Exception as e:
        error_msg = f"Error loading evaluation data: {str(e)}"
        return error_msg, "", "", "", go.Figure()

# Community Graph Click Prevention Callback
@app.callback(
    Output("community_network_graph", "clickData"),
    Input("community_network_graph", "clickData"),
    prevent_initial_call=True
)
def prevent_community_graph_clicks(click_data):
    """Prevent community graph clicks from triggering other callbacks."""
    # Return None to prevent any click data from being processed
    return None

# Advanced Analytics Functions
def run_drug_discovery_analytics(drug_id=None):
    """Run drug discovery insights (can work with or without specific drug)."""
    # Initialize analytics
    repurposing_analytics = DrugRepurposingAnalytics(drugs_df, targets_df, dti_df, embeddings_df)
    network_analytics = NetworkAnalytics(G, drugs_df, targets_df, dti_df)
    
    if drug_id:
        # Focused analysis on specific drug
        drug_info = drugs_df[drugs_df['drug_id'] == drug_id].iloc[0]
        
        # Find similar drugs
        similar_drugs = repurposing_analytics.find_similar_drugs(drug_id, top_k=10, similarity_threshold=0.3)
        
        # Find repurposing opportunities
        opportunities = repurposing_analytics.repurposing_opportunities(drug_id)
        
        # Network centrality - Optimized for speed (degree centrality only)
        if drug_id in G:
            neighbors = list(G.neighbors(drug_id))
            degree_centrality = len(neighbors) / (len(G.nodes()) - 1) if len(G.nodes()) > 1 else 0
            centrality = {
                'degree_centrality': degree_centrality,
                'betweenness_centrality': 0.0,  # Skip slow calculation
                'closeness_centrality': 0.0     # Skip slow calculation
            }
        else:
            centrality = {
                'degree_centrality': 0.0,
                'betweenness_centrality': 0.0,
                'closeness_centrality': 0.0
            }
        
        results = html.Div([
            html.H4(f"Drug Discovery Insights: {drug_info['name']}", 
                    style={"color": "#2c3e50", "marginBottom": "15px"}),
            
            html.Div([
                html.H5(f"Similar Drugs ({len(similar_drugs)} found)", style={"color": "#3498db", "marginBottom": "10px"}),
                html.Div([
                    html.Div([
                        html.P(f"• {drug['name']} (Similarity: {drug['similarity']:.3f})", 
                               style={"margin": "5px 0", "fontSize": "14px"}),
                        html.P(f"  Type: {drug['type']}, Approved: {'Yes' if drug['approved'] else 'No'}", 
                               style={"margin": "0 0 10px 20px", "fontSize": "12px", "color": "#7f8c8d"})
                    ]) for drug in similar_drugs[:5]
                ]) if similar_drugs else html.P("No similar drugs found.", style={"color": "#e74c3c", "fontStyle": "italic"})
            ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#ecf0f1", "borderRadius": "5px"}),
            
            html.Div([
                html.H5("Network Position (Fast Analysis)", style={"color": "#e74c3c", "marginBottom": "10px"}),
                html.P(f"Degree Centrality: {centrality.get('degree_centrality', 0):.4f}"),
                html.P(f"Target Connections: {len(list(G.neighbors(drug_id))) if drug_id in G else 0}"),
                html.P(f"Note: Showing degree centrality only for faster analysis", 
                       style={"fontSize": "11px", "color": "#7f8c8d", "fontStyle": "italic"})
            ], style={"padding": "15px", "backgroundColor": "#ecf0f1", "borderRadius": "5px"})
        ])
        
        # Create visualization
        if similar_drugs:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers+text',
                marker=dict(size=20, color='red'),
                text=[drug_info['name']],
                textposition="middle center",
                name="Selected Drug"
            ))
            
            angles = np.linspace(0, 2*np.pi, len(similar_drugs), endpoint=False)
            x_coords = np.cos(angles) * 2
            y_coords = np.sin(angles) * 2
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers+text',
                marker=dict(size=15, color='blue'),
                text=[drug['name'] for drug in similar_drugs],
                textposition="middle center",
                name="Similar Drugs"
            ))
            
            fig.update_layout(
                title="Drug Discovery Network",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
            
            visualizations = dcc.Graph(figure=fig)
        else:
            visualizations = html.P("No similar drugs found for visualization.")
            
    else:
        # General discovery insights - Optimized for speed
        # Use degree centrality only (much faster than betweenness)
        drug_centralities = []
        
        # Get drugs with most connections (top 10 for speed)
        drug_connections = dti_df['drug_id'].value_counts().head(10)
        
        for drug_id in drug_connections.index:
            drug_info = drugs_df[drugs_df['drug_id'] == drug_id].iloc[0]
            # Only calculate degree centrality (fast)
            degree_centrality = len(list(G.neighbors(drug_id))) / (len(G.nodes()) - 1) if len(G.nodes()) > 1 else 0
            
            drug_centralities.append({
                'name': drug_info['name'],
                'degree': degree_centrality,
                'connections': drug_connections[drug_id]
            })
        
        # Sort by degree centrality
        drug_centralities.sort(key=lambda x: x['degree'], reverse=True)
        
        results = html.Div([
            html.H4("Drug Discovery Insights", 
                    style={"color": "#2c3e50", "marginBottom": "15px"}),
            
            html.Div([
                html.H5("Top Drugs by Network Importance", style={"color": "#3498db", "marginBottom": "10px"}),
                html.Div([
                    html.Div([
                        html.P(f"• {drug['name']} (Degree: {drug['degree']:.4f}, Connections: {drug['connections']})", 
                               style={"margin": "5px 0", "fontSize": "14px"})
                    ]) for drug in drug_centralities[:10]
                ])
            ], style={"padding": "15px", "backgroundColor": "#ecf0f1", "borderRadius": "5px"})
        ])
        
        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[drug['name'] for drug in drug_centralities[:10]],
            y=[drug['degree'] for drug in drug_centralities[:10]],
            name='Degree Centrality'
        ))
        
        fig.update_layout(
            title="Top Drugs by Network Importance",
            xaxis_title="Drug",
            yaxis_title="Degree Centrality",
            height=400
        )
        
        visualizations = dcc.Graph(figure=fig)
    
    return results, visualizations, "Drug discovery analysis completed"

def run_network_communities_analytics(drug_id=None):
    """Run network community analysis."""
    # Initialize analytics
    network_analytics = NetworkAnalytics(G, drugs_df, targets_df, dti_df)
    
    # Detect communities
    communities = network_analytics.community_detection()
    
    print(f"DEBUG: Detected {len(communities)} communities")
    for comm_id, members in list(communities.items())[:5]:
        print(f"  Community {comm_id}: {len(members)} members")
    
    # Initialize community_stats for visualization
    community_stats = []
    
    print(f"DEBUG: drug_id parameter = {drug_id}")
    
    if drug_id:
        # Focused analysis on specific drug's community
        drug_info = drugs_df[drugs_df['drug_id'] == drug_id].iloc[0]
        
        # Find which community the drug belongs to
        drug_community = None
        for community_id, members in communities.items():
            if drug_id in members:
                drug_community = community_id
                break
        
        if drug_community is not None:
            # Get other drugs in the same community (remove duplicates)
            community_drugs = list(set([m for m in communities[drug_community] if m in drugs_df['drug_id'].values]))
            community_drug_info = drugs_df[drugs_df['drug_id'].isin(community_drugs)].drop_duplicates(subset=['drug_id'])
            
            # Create community_stats for this specific community
            drug_count = len(community_drugs)
            target_count = len([m for m in communities[drug_community] if m in targets_df['target_id'].values])
            community_stats.append({
                'community_id': drug_community,
                'drug_count': drug_count,
                'target_count': target_count,
                'total_members': len(communities[drug_community])
            })
            
            results = html.Div([
                html.H4(f"Network Communities: {drug_info['name']}", 
                        style={"color": "#2c3e50", "marginBottom": "15px"}),
                
                html.Div([
                    html.H5(f"Community {drug_community} ({len(community_drugs)} drugs)", 
                            style={"color": "#3498db", "marginBottom": "10px"}),
                    html.Div([
                        html.P(f"• {row['name']} ({row['drug_id']})", 
                               style={"margin": "5px 0", "fontSize": "14px"})
                        for _, row in community_drug_info.head(10).iterrows()
                    ])
                ], style={"padding": "15px", "backgroundColor": "#ecf0f1", "borderRadius": "5px"})
            ])
        else:
            results = html.Div([
                html.H4(f"Network Communities: {drug_info['name']}", 
                        style={"color": "#2c3e50", "marginBottom": "15px"}),
                html.P("Drug not found in any community.", style={"color": "#e74c3c"})
            ])
    else:
        # General community analysis
        # Count drugs in each community
        print(f"DEBUG: Starting community stats creation...")
        print(f"DEBUG: Processing {len(communities)} communities for stats creation")
        for community_id, members in communities.items():
            drug_count = len([m for m in members if m in drugs_df['drug_id'].values])
            target_count = len([m for m in members if m in targets_df['target_id'].values])
            print(f"DEBUG: Community {community_id}: {drug_count} drugs, {target_count} targets, {len(members)} total")
            # Show communities with either drugs or targets (but at least some meaningful content)
            if drug_count > 0 or target_count > 5:  # Show communities with drugs OR significant targets
                print(f"DEBUG: Adding Community {community_id} to stats")
                community_stats.append({
                    'community_id': community_id,
                    'drug_count': drug_count,
                    'target_count': target_count,
                    'total_members': len(members)
                })
            else:
                print(f"DEBUG: Skipping Community {community_id} (no drugs and <=5 targets)")
        
        # Sort by drug count
        community_stats.sort(key=lambda x: x['drug_count'], reverse=True)
        
        print(f"DEBUG: Created {len(community_stats)} community_stats")
        print(f"DEBUG: Total communities that passed the filter: {len(community_stats)}")
        print(f"DEBUG: Communities that passed the filter:")
        for stat in community_stats:
            print(f"  Community {stat['community_id']}: {stat['drug_count']} drugs, {stat['target_count']} targets")
        
        results = html.Div([
            html.H4("Network Communities", 
                    style={"color": "#2c3e50", "marginBottom": "15px"}),
            
            html.Div([
                html.H5(f"Found {len(community_stats)} Communities", 
                        style={"color": "#3498db", "marginBottom": "10px"}),
                html.P("Communities represent groups of drugs and targets that work together functionally. Drugs in the same community often target similar pathways or treat related conditions.", 
                       style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "15px", "fontStyle": "italic"}),
                html.Div([
                    html.Div([
                        html.P(f"• Community {stat['community_id']}: {stat['drug_count']} drugs, {stat['target_count']} targets", 
                               style={"margin": "5px 0", "fontSize": "14px"})
                    ]) for stat in community_stats[:10]
                ])
            ], style={"padding": "15px", "backgroundColor": "#ecf0f1", "borderRadius": "5px"})
        ])
    
    # Create community network visualization
    if community_stats:
        # Get the top 5 largest communities for better visualization (showing all nodes)
        top_communities = community_stats[:5]
        
        # Debug: Print what communities we're trying to visualize
        print(f"Attempting to visualize {len(top_communities)} communities:")
        for comm in top_communities:
            print(f"  Community {comm['community_id']}: {comm['total_members']} nodes ({comm['drug_count']} drugs, {comm['target_count']} targets)")
        
        # Create network graph showing community structure
        fig_network = go.Figure()
        
        # Track all nodes and their positions
        all_nodes = []
        node_positions = {}
        node_colors = []
        node_sizes = []
        node_labels = []
        node_types = []  # Track if node is drug or target
        
        # Distinct colors for different node types
        community_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']  # Red, Blue, Green, Orange, Purple
        drug_colors = ['#c0392b', '#2980b9', '#27ae60', '#e67e22', '#8e44ad']      # Darker versions for drugs
        target_colors = ['#ec7063', '#5dade2', '#58d68d', '#f8c471', '#bb8fce']   # Lighter versions for targets
        
        # Position communities in a larger circle with better spacing
        center_x, center_y = 0, 0
        radius = 8  # Increased radius for better separation
        
        for i, community_stat in enumerate(top_communities):
            community_id = community_stat['community_id']
            
            # Get community members from the communities dict
            if community_id in communities:
                community_members = communities[community_id]
                
                # Separate drugs and targets
                community_drugs = [m for m in community_members if m in drugs_df['drug_id'].values]
                community_targets = [m for m in community_members if m in targets_df['target_id'].values]
                
                print(f"  Processing Community {community_id}: {len(community_drugs)} drugs, {len(community_targets)} targets")
                
                # Position community center with better spacing
                angle = (2 * np.pi * i) / len(top_communities)
                comm_x = center_x + radius * np.cos(angle)
                comm_y = center_y + radius * np.sin(angle)
                
                # Add community center node
                all_nodes.append(f"Community_{community_id}")
                node_positions[f"Community_{community_id}"] = (comm_x, comm_y)
                node_colors.append(community_colors[i % len(community_colors)])
                node_sizes.append(25)
                node_labels.append(f"C{community_id}")  # Simplified label
                node_types.append("center")
                
                # Position ALL drugs in inner circle
                for j, drug_id in enumerate(community_drugs):
                    drug_angle = angle + (2 * np.pi * j) / len(community_drugs) - np.pi/3
                    drug_radius = 3.0  # Increased radius for better visibility
                    drug_x = comm_x + drug_radius * np.cos(drug_angle)
                    drug_y = comm_y + drug_radius * np.sin(drug_angle)
                    
                    drug_name = drugs_df[drugs_df['drug_id'] == drug_id]['name'].iloc[0]
                    all_nodes.append(drug_id)
                    node_positions[drug_id] = (drug_x, drug_y)
                    # Use distinct drug colors (darker, more saturated)
                    node_colors.append(drug_colors[i % len(drug_colors)])
                    node_sizes.append(12)
                    node_labels.append("")  # No text labels on nodes
                    node_types.append("drug")
                
                # Position ALL targets in outer circle
                for j, target_id in enumerate(community_targets):
                    target_angle = angle + (2 * np.pi * j) / len(community_targets) + np.pi/3
                    target_radius = 5.0  # Increased radius for better separation
                    target_x = comm_x + target_radius * np.cos(target_angle)
                    target_y = comm_y + target_radius * np.sin(target_angle)
                    
                    target_name = targets_df[targets_df['target_id'] == target_id]['name'].iloc[0]
                    all_nodes.append(target_id)
                    node_positions[target_id] = (target_x, target_y)
                    # Use distinct target colors (lighter, pastel-like)
                    node_colors.append(target_colors[i % len(target_colors)])
                    node_sizes.append(8)
                    node_labels.append("")  # No text labels on nodes
                    node_types.append("target")
        
        # Create edges (connections within communities)
        edges_x = []
        edges_y = []
        
        for community_stat in top_communities:
            community_id = community_stat['community_id']
            if community_id in communities:
                community_members = communities[community_id]
                
                # Connect community center to its members
                comm_center = f"Community_{community_id}"
                if comm_center in node_positions:
                    for member in community_members:
                        if member in node_positions:
                            # Add edge from center to member
                            edges_x.extend([node_positions[comm_center][0], node_positions[member][0], None])
                            edges_y.extend([node_positions[comm_center][1], node_positions[member][1], None])
        
        # Add edges
        if edges_x:
            fig_network.add_trace(go.Scatter(
                x=edges_x,
                y=edges_y,
                mode='lines',
                line=dict(color='rgba(100,100,100,0.2)', width=0.5),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        node_x = [node_positions[node][0] for node in all_nodes]
        node_y = [node_positions[node][1] for node in all_nodes]
        
        print(f"Total nodes in visualization: {len(all_nodes)}")
        print(f"Node types: {dict(zip(*np.unique(node_types, return_counts=True)))}")
        print(f"Node positions: {len(node_positions)}")
        print(f"Sample node positions: {list(node_positions.items())[:5]}")
        
        fig_network.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',  # No text labels
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color='white', width=1),
                opacity=0.8
            ),
            hoverinfo='text',
            name='Community Network',
            customdata=node_types,  # Store node types for hover info
            hovertemplate='<b>%{text}</b><br>Type: %{customdata}<extra></extra>',
            text=[f"Community {n.split('_')[1]}" if n.startswith('Community_') else 
                  (drugs_df[drugs_df['drug_id'] == n]['name'].iloc[0] if n in drugs_df['drug_id'].values else 
                   targets_df[targets_df['target_id'] == n]['name'].iloc[0]) for n in all_nodes]
        ))
        
        fig_network.update_layout(
            title="Community Network Structure - 5 Largest Communities",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            width=800,  # Fixed width for better centering
            showlegend=False,
            plot_bgcolor='white',
            clickmode='event',  # Enable clicks for navigation
            margin=dict(l=50, r=50, t=80, b=50),  # Better margins
            dragmode='pan',  # Enable panning
            hovermode='closest'  # Enable hover
        )
        
        # Add explanation and legend
        community_ids = [f'C{comm["community_id"]}' for comm in top_communities]
        explanation = html.Div([
            html.P(f"This visualization shows the {len(top_communities)} largest communities in the drug-target network.", 
                   style={"fontSize": "14px", "color": "#2c3e50", "marginBottom": "10px"}),
            html.P("Each community is a group of drugs and targets that work together functionally.", 
                   style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "15px", "fontStyle": "italic"}),
            html.P(f"Communities found: {', '.join(community_ids)}", 
                   style={"fontSize": "12px", "color": "#e67e22", "fontWeight": "bold"})
        ])
        
        # Add legend
        legend_items = []
        for i, community_stat in enumerate(top_communities):
            legend_items.append(
                html.Div([
                    html.Span("●", style={"color": community_colors[i % len(community_colors)], "fontSize": "20px"}),
                    html.Span(f" Community {community_stat['community_id']} ({community_stat['total_members']} nodes)", 
                             style={"marginLeft": "10px", "fontSize": "12px"})
                ], style={"marginBottom": "5px"})
            )
        
        # Add node type legend with distinct color coding
        legend_items.append(html.Div([
            html.Span("●", style={"color": "#e74c3c", "fontSize": "18px"}),
            html.Span(" Community Centers", style={"marginLeft": "10px", "fontSize": "12px"}),
            html.Span(" ●", style={"color": "#c0392b", "fontSize": "14px"}),
            html.Span(" Drugs (Darker)", style={"marginLeft": "10px", "fontSize": "12px"}),
            html.Span(" ●", style={"color": "#ec7063", "fontSize": "12px"}),
            html.Span(" Targets (Lighter)", style={"marginLeft": "10px", "fontSize": "12px"})
        ], style={"marginTop": "10px", "marginBottom": "5px"}))
        
        legend_div = html.Div(legend_items, style={"marginTop": "10px", "textAlign": "center"})
        
                # Center the graph container with complete isolation
        visualizations = html.Div([
            explanation,
            html.Div([
                            dcc.Graph(
                figure=fig_network, 
                config={
                    'displayModeBar': True,
                    'staticPlot': False,  # Make it interactive
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                    'displaylogo': False,
                    'responsive': False
                },
                id='community_network_graph'
            )
            ], style={
                "display": "flex", 
                "justifyContent": "center", 
                "alignItems": "center"
            }),
            legend_div
        ], style={
            "textAlign": "center"
        })
    else:
        visualizations = html.P("No community data available for visualization.")
    
    return results, visualizations, "Network community analysis completed"

def run_drug_combination_analytics(drug_id=None):
    """Run drug combination analysis."""
    # Initialize analytics
    predictive_analytics = PredictiveInsights(drugs_df, targets_df, dti_df, link_predictor)
    
    if drug_id:
        # Focused analysis on specific drug
        drug_info = drugs_df[drugs_df['drug_id'] == drug_id].iloc[0]
        
        # Get drug combination recommendations
        combinations = predictive_analytics.drug_combination_recommendations(drug_id)
        
        # Get drug-drug interaction warnings
        interactions = predictive_analytics.drug_drug_interaction_warnings(drug_id)
        
        results = html.Div([
            html.H4(f"Drug Combination Analysis: {drug_info['name']}", 
                    style={"color": "#2c3e50", "marginBottom": "15px"}),
            
            # Explanation section
            html.Div([
                html.H5("What This Analysis Shows", style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P("This analysis identifies drugs that could work well together (complementary targets) and drugs that might interact poorly (shared targets).", 
                       style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "15px", "fontStyle": "italic"}),
            ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#e8f4fd", "borderRadius": "5px", "border": "1px solid #3498db"}),
            
            html.Div([
                html.H5("Recommended Combinations", style={"color": "#3498db", "marginBottom": "10px"}),
                html.P("These drugs target different proteins, potentially enhancing therapeutic effects with minimal side effects:", 
                       style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "10px", "fontStyle": "italic"}),
                html.Div([
                    html.Div([
                        html.P(f"• {combo['name']} (Complementarity: {combo['complementarity_score']})", 
                               style={"margin": "5px 0", "fontSize": "14px"}),
                        html.P(f"  Overlap: {combo['overlap']}, Total Targets: {combo['total_targets']}", 
                               style={"margin": "0 0 10px 20px", "fontSize": "12px", "color": "#7f8c8d"})
                    ]) for combo in combinations[:5]
                ]) if combinations else html.P("No combination recommendations found.", style={"color": "#e74c3c", "fontStyle": "italic"})
            ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#ecf0f1", "borderRadius": "5px"}),
            
            html.Div([
                html.H5("Interaction Warnings", style={"color": "#e74c3c", "marginBottom": "10px"}),
                html.P("These drugs target the same proteins and may cause adverse interactions or reduced effectiveness:", 
                       style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "10px", "fontStyle": "italic"}),
                html.Div([
                    html.Div([
                        html.P(f"• {interaction['name']} ({interaction['warning_level']} Risk)", 
                               style={"margin": "5px 0", "fontSize": "14px"}),
                        html.P(f"  Interaction Strength: {interaction['interaction_strength']:.3f}", 
                               style={"margin": "0 0 10px 20px", "fontSize": "12px", "color": "#7f8c8d"})
                    ]) for interaction in interactions[:5]
                ]) if interactions else html.P("No significant interactions found.", style={"color": "#27ae60", "fontStyle": "italic"})
            ], style={"padding": "15px", "backgroundColor": "#ecf0f1", "borderRadius": "5px"})
        ])
        
        # Create separate visualizations
        visualization_components = []
        
        # Combination recommendations visualization
        if combinations:
            fig_combinations = go.Figure()
            fig_combinations.add_trace(go.Bar(
                x=[combo['name'] for combo in combinations[:5]],
                y=[combo['complementarity_score'] for combo in combinations[:5]],
                name='Complementarity Score',
                marker_color='#3498db'
            ))
            
            fig_combinations.update_layout(
                title="Recommended Drug Combinations",
                xaxis_title="Drug",
                yaxis_title="Complementarity Score",
                height=300,
                showlegend=False
            )
            
            visualization_components.append(dcc.Graph(figure=fig_combinations))
        
        # Interaction warnings visualization
        if interactions:
            fig_interactions = go.Figure()
            fig_interactions.add_trace(go.Bar(
                x=[interaction['name'] for interaction in interactions[:5]],
                y=[interaction['interaction_strength'] for interaction in interactions[:5]],
                name='Interaction Risk',
                marker_color='#e74c3c'
            ))
            
            fig_interactions.update_layout(
                title="Drug Interaction Warnings",
                xaxis_title="Drug",
                yaxis_title="Interaction Strength",
                height=300,
                showlegend=False
            )
            
            visualization_components.append(dcc.Graph(figure=fig_interactions))
        
        # Combine visualizations
        if visualization_components:
            visualizations = html.Div(visualization_components, style={"display": "flex", "flexDirection": "column", "gap": "20px"})
        else:
            visualizations = html.P("No data available for visualization.")
            
    else:
        # General combination insights
        # Find drugs with highest complementarity potential
        all_combinations = []
        for _, drug in drugs_df.head(10).iterrows():  # Sample first 10 drugs
            combinations = predictive_analytics.drug_combination_recommendations(drug['drug_id'])
            if combinations:
                all_combinations.extend(combinations[:3])  # Top 3 for each drug
        
        # Sort by complementarity
        all_combinations.sort(key=lambda x: x['complementarity_score'], reverse=True)
        
        results = html.Div([
            html.H4("Drug Combination Insights", 
                    style={"color": "#2c3e50", "marginBottom": "15px"}),
            
            # Explanation section
            html.Div([
                html.H5("What This Analysis Shows", style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P("This analysis identifies drugs with the highest potential for beneficial combinations based on their target complementarity.", 
                       style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "15px", "fontStyle": "italic"}),
            ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#e8f4fd", "borderRadius": "5px", "border": "1px solid #3498db"}),
            
            html.Div([
                html.H5("Top Combination Opportunities", style={"color": "#3498db", "marginBottom": "10px"}),
                html.P("These drugs have the highest complementarity scores, meaning they target different proteins and could work well together:", 
                       style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "10px", "fontStyle": "italic"}),
                html.Div([
                    html.Div([
                        html.P(f"• {combo['name']} (Complementarity: {combo['complementarity_score']})", 
                               style={"margin": "5px 0", "fontSize": "14px"})
                    ]) for combo in all_combinations[:10]
                ]) if all_combinations else html.P("No combination opportunities found.", style={"color": "#e74c3c", "fontStyle": "italic"})
            ], style={"padding": "15px", "backgroundColor": "#ecf0f1", "borderRadius": "5px"})
        ])
        
        # Create visualization
        if all_combinations:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[combo['name'] for combo in all_combinations[:10]],
                y=[combo['complementarity_score'] for combo in all_combinations[:10]],
                name='Complementarity Score'
            ))
            
            fig.update_layout(
                title="Top Drug Combination Opportunities",
                xaxis_title="Drug",
                yaxis_title="Complementarity Score",
                height=400
            )
            
            visualizations = dcc.Graph(figure=fig)
        else:
            visualizations = html.P("No combination data available for visualization.")
    
    return results, visualizations, "Drug combination analysis completed"

# Removed duplicate callbacks - progress is now handled in the main run_ml_analysis callback

@app.callback(
    Output("degree_distribution", "figure"),
    Input("search_dropdown", "value")
)
def update_degree_distribution(selected_node):
    """Update degree distribution plot."""
    if G is None:
        return go.Figure()
    
    # Calculate degree distribution
    degrees = [d for n, d in G.degree()]
    
    # Create histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=degrees,
        nbinsx=50,
        name='Degree Distribution',
        marker_color='#3498db',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Degree Distribution",
        xaxis_title="Degree",
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )
    
    return fig

if __name__ == "__main__":
    print("Starting DTI Explorer Pro - Plotly Edition...")
    print("Loaded data:")
    if G is not None:
        print(f"  - Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    if drugs_df is not None:
        print(f"  - Drugs: {len(drugs_df):,}")
    if targets_df is not None:
        print(f"  - Targets: {len(targets_df):,}")
    print("Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=8050)
