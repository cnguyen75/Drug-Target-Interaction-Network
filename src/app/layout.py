"""Layout components for the DTI Explorer application."""
import dash
from dash import html, dcc
import plotly.graph_objs as go

def create_main_layout():
    """Create the main application layout."""
    return html.Div([
        # Location component for URL routing
        dcc.Location(id='url', refresh=False),
        
        # Header
        html.Div([
            html.H1("Drug-Target Interaction Explorer Pro", 
                    style={
                        "color": "#2c3e50",
                        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        "fontWeight": "700",
                        "textAlign": "center",
                        "marginBottom": "10px",
                        "textShadow": "2px 2px 4px rgba(0,0,0,0.1)"
                    }),
            html.P("Advanced Network Analysis & Machine Learning Insights", 
                   style={
                       "color": "#7f8c8d",
                       "textAlign": "center",
                       "fontSize": "16px",
                       "fontStyle": "italic",
                       "marginBottom": "30px"
                   })
        ], style={
            "backgroundColor": "#ecf0f1",
            "padding": "20px",
            "borderRadius": "15px",
            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
            "marginBottom": "30px"
        }),
        
        # Main content area with tabs
        dcc.Tabs([
            # Network Visualization Tab
            dcc.Tab(label="Network Visualization", children=[
                html.Div([
                    # Left panel - Controls
                    html.Div([
                        # Search section
                        html.Div([
                            html.H4("Search & Explore", style={"color": "#2c3e50", "marginBottom": "15px"}),
                            
                            # Drug search
                            html.Label("Search Drug:"),
                            dcc.Dropdown(
                                id="drug_search",
                                placeholder="Type to search drugs...",
                                style={"marginBottom": "15px"}
                            ),
                            
                            # Target search
                            html.Label("Search Target:"),
                            dcc.Dropdown(
                                id="target_search",
                                placeholder="Type to search targets...",
                                style={"marginBottom": "15px"}
                            ),
                            
                            # Search button
                            html.Button(
                                "Search Network",
                                id="search_button",
                                style={
                                    "backgroundColor": "#3498db",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "10px 20px",
                                    "borderRadius": "5px",
                                    "cursor": "pointer",
                                    "fontWeight": "bold",
                                    "marginBottom": "20px"
                                }
                            )
                        ], style={"marginBottom": "30px"}),
                        
                        # Visualization controls
                        html.Div([
                            html.H4("Visualization Controls", style={"color": "#2c3e50", "marginBottom": "15px"}),
                            
                            # Graph type selector
                            html.Label("Graph Type:"),
                            dcc.RadioItems(
                                id="graph_type",
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
                                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"
                            })
                        ], style={"width": "75%", "float": "right"})
                    ], style={"display": "flex", "clear": "both"})
                ])
            ]),
            
            # Similarity Analysis Tab
            dcc.Tab(label="Similarity Analysis", children=[
                html.Div([
                    # Left panel - Controls
                    html.Div([
                        html.H4("Similarity Analysis", style={"color": "#2c3e50", "marginBottom": "20px"}),
                        
                        # Drug selection for similarity
                        html.Label("Select Drug:"),
                        dcc.Dropdown(
                            id="similarity_drug_dropdown",
                            placeholder="Choose a drug...",
                            style={"marginBottom": "20px"}
                        ),
                        
                        # Similarity type
                        html.Label("Similarity Type:"),
                        dcc.RadioItems(
                            id="similarity_type",
                            options=[
                                {"label": "Drug Similarity", "value": "drug"},
                                {"label": "Target Similarity", "value": "target"}
                            ],
                            value="drug",
                            style={"marginBottom": "20px"}
                        ),
                        
                        # Number of results
                        html.Label("Number of Results:"),
                        dcc.Slider(
                            id="similarity_count",
                            min=5,
                            max=20,
                            step=5,
                            value=10,
                            marks={i: str(i) for i in [5, 10, 15, 20]}
                        ),
                        
                        html.Button(
                            "Find Similar",
                            id="find_similar_button",
                            style={
                                "backgroundColor": "#27ae60",
                                "color": "white",
                                "border": "none",
                                "padding": "10px 20px",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                                "fontWeight": "bold",
                                "marginTop": "20px"
                            }
                        )
                    ], style={
                        "width": "25%", 
                        "float": "left", 
                        "padding": "25px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "15px",
                        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
                    }),
                    
                    # Right panel - Results
                    html.Div([
                        html.Div(id="similarity_results", style={"padding": "20px"})
                    ], style={
                        "width": "75%", 
                        "float": "right", 
                        "padding": "20px"
                    })
                ], style={"display": "flex", "clear": "both"})
            ]),
            
            # Link Prediction Tab
            dcc.Tab(label="Link Prediction", children=[
                html.Div([
                    # Left panel - Controls
                    html.Div([
                        html.H4("Predict Drug-Target Interactions", style={"color": "#2c3e50", "marginBottom": "20px"}),
                        
                        # Drug selection
                        html.Label("Select Drug:"),
                        dcc.Dropdown(
                            id="prediction_drug_dropdown",
                            placeholder="Choose a drug...",
                            style={"marginBottom": "20px"}
                        ),
                        
                        # Target selection
                        html.Label("Select Target:"),
                        dcc.Dropdown(
                            id="prediction_target_dropdown",
                            placeholder="Choose a target...",
                            style={"marginBottom": "20px"}
                        ),
                        
                        html.Button(
                            "Predict Interaction",
                            id="predict_button",
                            style={
                                "backgroundColor": "#e74c3c",
                                "color": "white",
                                "border": "none",
                                "padding": "10px 20px",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                                "fontWeight": "bold"
                            }
                        )
                    ], style={
                        "width": "25%", 
                        "float": "left", 
                        "padding": "25px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "15px",
                        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
                    }),
                    
                    # Right panel - Results
                    html.Div([
                        html.Div(id="prediction_results", style={"padding": "20px"})
                    ], style={
                        "width": "75%", 
                        "float": "right", 
                        "padding": "20px"
                    })
                ], style={"display": "flex", "clear": "both"})
            ]),
            
            # Advanced Analytics Tab
            dcc.Tab(label="Advanced Analytics", children=[
                html.Div([
                    # Left panel - Analytics Controls
                    html.Div([
                        html.H4("Discovery Analytics", style={"color": "#2c3e50", "marginBottom": "20px"}),
                        
                        # Analysis type selection
                        html.Label("Analysis Type:"),
                        dcc.Dropdown(
                            id="analytics_type",
                            options=[
                                {"label": "Drug Discovery Insights", "value": "discovery"},
                                {"label": "Network Communities", "value": "communities"},
                                {"label": "Drug Combination Analytics", "value": "combinations"}
                            ],
                            placeholder="Select analysis type...",
                            style={"marginBottom": "20px"}
                        ),
                        
                        # Drug selection (optional)
                        html.Label("Select Drug (Optional):"),
                        dcc.Dropdown(
                            id="analytics_drug_dropdown",
                            placeholder="Choose a drug for focused analysis...",
                            style={"marginBottom": "20px"}
                        ),
                        
                        html.Button(
                            "Run Discovery Analysis",
                            id="run_analytics_button",
                            style={
                                "backgroundColor": "#9b59b6",
                                "color": "white",
                                "border": "none",
                                "padding": "10px 20px",
                                "borderRadius": "5px",
                                "cursor": "pointer",
                                "fontWeight": "bold"
                            }
                        )
                    ], style={
                        "width": "25%", 
                        "float": "left", 
                        "padding": "25px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "15px",
                        "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
                    }),
                    
                    # Right panel - Analytics Results with Loading
                    html.Div([
                        dcc.Loading(
                            id="analytics_loading",
                            type="circle",
                            children=[
                                html.Div(id="analytics_results", children="Select an analysis type and click 'Run Discovery Analysis' to get started.", style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px", "minHeight": "400px", "color": "#7f8c8d", "fontStyle": "italic"}),
                                html.Div(id="analytics_visualizations", style={"marginTop": "20px"})
                            ],
                            style={"marginTop": "20px"}
                        ),
                        html.Div(id="analytics_progress", children="Ready to run analysis", style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "10px"})
                    ], style={"width": "75%", "float": "right", "padding": "20px"})
                ], style={"display": "flex", "clear": "both"})
            ]),
            
            # ML Evaluation Tab
            dcc.Tab(label="ML Evaluation", children=[
                html.Div([
                    html.H4("Machine Learning Model Evaluation", style={"color": "#2c3e50", "marginBottom": "20px"}),
                    html.Div(id="ml_evaluation_results", style={"padding": "20px"})
                ])
            ])
        ], style={
            "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
        })
    ], style={
        "padding": "20px",
        "backgroundColor": "#f5f6fa",
        "minHeight": "100vh"
    })
