"""Drug-target interaction network: explore, predictions, evaluation, methods.

    make run          # builds anything missing, then serves http://127.0.0.1:8050

Every number shown comes from results/results.json, the parsed DrugBank tables
or the trained model; nothing is typed in.
"""

import json
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlencode

import dash
from dash import Input, Output, State, dcc, html
import pandas as pd
import plotly.graph_objects as go


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import DRUGS_FP, DTI_FP, TARGETS_FP
from pipeline.build_graph import build_bipartite_graph
from pipeline.explore import (
    edge_table,
    evidence,
    featured_mask,
    neighbourhood,
    radial_layout,
    rank_key,
)
from pipeline.link_prediction import DTILinkPredictor


ROOT = Path(__file__).parent
RESULTS_FP = ROOT / "results" / "results.json"  # dataset, v2 graph-only model
ATTR_RESULTS_FP = ROOT / "results" / "results_attributes.json"  # v3 attribute model (the app's)
ABLATION_FP = ROOT / "results" / "results_attribute_ablation.json"
COLD_SIM_FP = ROOT / "results" / "results_cold_similarity.json"
MODEL_FPS = {
    "seen": ROOT / "models" / "attribute_model_seen.pkl",
    "cold": ROOT / "models" / "attribute_model_cold.pkl",
}
PREDICTIONS_FP = ROOT / "models" / "predictions.parquet"

INK, SECONDARY, RULE, CARD = "#231F1B", "#3D372F", "#B3A992", "#F6F2E8"
BLUE, ORANGE, PURPLE, BASELINE = "#1F6FB5", "#C0601F", "#7A4E9E", "#8A8171"

SPLIT_LABELS = {
    "cold_drug": "Whole-drug holdout (unseen drugs)",
    "drug_holdout": "Per-drug split (controls for target popularity)",
    "random": "Random edge split",
}
V3_ROWS = [
    ("attribute_model", "Attribute model (v3, the app)", "blue"),
    ("graph_model", "Graph-only model (v2)", "purple-bar"),
]
BASELINE_ROWS = [
    ("structural_nearest_neighbour", "Structural nearest neighbour"),
    ("l3_paths", "Length-3 paths"),
    ("preferential_attachment", "Preferential attachment"),
    ("chance", "Chance"),
]
DEFAULT_NODE_NAME = "Imatinib"
ATTRIBUTION = (
    "Interaction data from DrugBank 5.1, used under CC BY-NC 4.0 for non-commercial purposes."
)


# ----- data ------------------------------------------------------------------


def load():
    paths = (
        DRUGS_FP,
        TARGETS_FP,
        DTI_FP,
        RESULTS_FP,
        ATTR_RESULTS_FP,
        ABLATION_FP,
        COLD_SIM_FP,
        *MODEL_FPS.values(),
    )
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        sys.exit("Missing " + ", ".join(missing) + ". Run: make run")
    drugs = pd.read_parquet(DRUGS_FP)
    targets = pd.read_parquet(TARGETS_FP)
    dti = pd.read_parquet(DTI_FP)
    G = build_bipartite_graph(drugs, targets, dti)
    copies = {k: DTILinkPredictor.load_model(fp) for k, fp in MODEL_FPS.items()}
    results = {
        k: json.loads(fp.read_text())
        for k, fp in (
            ("v2", RESULTS_FP),
            ("v3", ATTR_RESULTS_FP),
            ("ablation", ABLATION_FP),
            ("cold_sim", COLD_SIM_FP),
        )
    }
    predictions = pd.read_parquet(PREDICTIONS_FP) if PREDICTIONS_FP.exists() else None
    return drugs, targets, G, copies, results, predictions


DRUGS, TARGETS, G, COPIES, ALL_RESULTS, PREDICTIONS = load()
RESULTS = ALL_RESULTS["v2"]  # dataset counts and the v2 graph-only evaluation
V3 = ALL_RESULTS["v3"]
DRUG_INFO = DRUGS.set_index("drug_id")
TARGET_INFO = TARGETS.set_index("target_id")
FEATURED = set(DRUGS.loc[featured_mask(DRUGS), "drug_id"])
SMALL_MOLECULES = set(DRUGS.loc[DRUGS["type"] == "small molecule", "drug_id"])
THRESHOLDS = {
    k: V3.get("app_suggestions", {}).get(k, {}).get("default_threshold", 0.9)
    for k in ("seen", "cold")
}
PHONE_WIDTH = 640


def small_molecule_or_target(n):
    """Default graph filter: hide drug entries that are not small molecules."""
    return not is_drug(n) or n in SMALL_MOLECULES


def is_drug(n):
    return G.nodes[n]["node_type"] == "drug"


def label(n):
    """Short label: drug name, or the target's gene symbol (full name in details)."""
    if is_drug(n):
        return G.nodes[n]["name"] or n
    gene = TARGET_INFO.at[n, "gene_symbol"] if n in TARGET_INFO.index else ""
    return gene or G.nodes[n]["name"] or n


def scorable_without_targets(n):
    """A drug with no known targets can still be scored if it has a structure."""
    return n in COPIES["cold"].fps


def search_options():
    opts = []
    for n, d in G.nodes(data=True):
        if G.degree(n) == 0 and not (d["node_type"] == "drug" and scorable_without_targets(n)):
            continue
        if d["node_type"] == "drug":
            tag = "drug" if G.degree(n) else "drug, no known targets"
            opts.append({"label": f"{d['name']} · {tag}", "value": n, "search": f"{d['name']} {n}"})
        else:
            gene = TARGET_INFO.at[n, "gene_symbol"]
            text = f"{gene} — {d['name']}" if gene else d["name"]
            opts.append({"label": f"{text} · target", "value": n, "search": f"{text} {n}"})
    opts.sort(key=lambda o: o["label"].lower())
    return opts


SEARCH_OPTIONS = search_options()
OPTION_BY_VALUE = {o["value"]: o for o in SEARCH_OPTIONS}
MAX_MATCHES = 30


def match_options(query, current=None):
    """Up to MAX_MATCHES options for a search, prefix matches first; always keeps `current`."""
    q = (query or "").strip().lower()
    out = []
    if q:
        starts = [o for o in SEARCH_OPTIONS if o["search"].lower().startswith(q)]
        contains = [o for o in SEARCH_OPTIONS if q in o["search"].lower() and o not in starts]
        out = (starts + contains)[:MAX_MATCHES]
    if current in OPTION_BY_VALUE and OPTION_BY_VALUE[current] not in out:
        out = [OPTION_BY_VALUE[current]] + out
    return [{"label": o["label"], "value": o["value"]} for o in out]


DEFAULT_NODE = next(
    (n for n in sorted(FEATURED) if G.nodes[n]["name"] == DEFAULT_NODE_NAME and G.degree(n) > 0),
    max(FEATURED, key=G.degree),
)


def copy_for(node):
    """'cold' for a drug with no known targets, else 'seen' (see methods)."""
    return "cold" if is_drug(node) and G.degree(node) == 0 else "seen"


def suggestions(node):
    """Scores for unlinked partners. A target's suggestions are featured drugs with known targets."""
    model = COPIES[copy_for(node)]
    if is_drug(node):
        return model.score_candidates(node)
    cands = [d for d in sorted(FEATURED) if G.degree(d) > 0 and not G.has_edge(node, d)]
    return model.score_candidates(node, cands)


# ----- small components ------------------------------------------------------


def mono(text=None, className="mono", **kw):
    return html.Span(text, className=className, **kw)


def fmt_int(n):
    return f"{n:,}"


def header():
    ds = RESULTS["dataset"]
    return html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            mono(
                                "drugbank · node2vec · chemical structure · random forest link prediction"
                            ),
                            html.H1("Drug–target interaction network"),
                        ],
                        className="title",
                    ),
                    html.Nav(id="nav", **{"aria-label": "Views"}),
                ],
                className="site-header",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Span(fmt_int(ds["drugs_with_human_target"]), className="stat"),
                            " ",
                            html.Span("drugs with a human target", className="secondary"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(fmt_int(ds["human_targets"]), className="stat"),
                            " ",
                            html.Span("human targets", className="secondary"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(fmt_int(ds["interactions"]), className="stat"),
                            " ",
                            html.Span("known interactions", className="secondary"),
                        ]
                    ),
                    mono(
                        f"DrugBank {RESULTS['source'].get('version', '')}, exported "
                        f"{RESULTS['source'].get('exported_on', '')} · counts from results.json",
                        className="mono stats-note",
                    ),
                ],
                className="stats",
            ),
        ],
        className="head-block",
    )


VIEWS = [
    ("explore", "/"),
    ("predictions", "/predictions"),
    ("evaluation", "/evaluation"),
    ("methods", "/methods"),
]


def nav(pathname):
    # The current view is a plain anchor so it can carry aria-current; dcc.Link cannot.
    return [
        html.A(name, href=href, className="nav-link active", **{"aria-current": "page"})
        if pathname == href
        else dcc.Link(name, href=href, className="nav-link")
        for name, href in VIEWS
    ]


# ----- explore view ----------------------------------------------------------


def key_item(svg, text):
    return html.Div([svg, html.Span(text)], className="key-item")


def explore_view(node):
    return html.Section(
        [
            html.Aside(
                [
                    html.Label(
                        [
                            "Find a drug or target",
                            dcc.Dropdown(
                                id="node-search",
                                options=match_options("", node),
                                value=node,
                                clearable=True,
                                placeholder="Type a drug, gene or protein name",
                                optionHeight=44,
                                className="search",
                            ),
                        ],
                        className="field",
                    ),
                    html.Div(
                        [
                            html.Div("Show", className="field-title"),
                            dcc.Checklist(
                                id="show",
                                value=["known", "predicted", "shared"],
                                className="checks",
                                options=[
                                    {"label": "Known interactions", "value": "known"},
                                    {"label": "Predicted links", "value": "predicted"},
                                    {"label": "Nodes that share a partner", "value": "shared"},
                                ],
                            ),
                            dcc.Checklist(
                                id="all-types",
                                value=[],
                                className="checks",
                                options=[{"label": "Show all entry types", "value": "all"}],
                            ),
                            html.Div(
                                "Off: drug entries that are not small molecules (biologics, cell "
                                "therapies) are left out of the graph but kept in the table.",
                                className="secondary small",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div("Minimum prediction score", className="field-title"),
                            dcc.Slider(
                                id="min-score",
                                min=0,
                                max=1,
                                step=0.05,
                                value=THRESHOLDS[copy_for(node)],
                                marks=None,
                                tooltip=None,
                                className="slider",
                            ),
                            mono(id="min-score-label"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div("Key", className="field-title"),
                            key_item(html.Span(className="glyph drug"), "Drug (circle)"),
                            key_item(html.Span(className="glyph target"), "Target (square)"),
                            key_item(html.Span(className="glyph line known"), "Known interaction"),
                            key_item(html.Span(className="glyph line predicted"), "Predicted link"),
                            key_item(html.Span(className="glyph line shared"), "Shares a partner"),
                        ],
                        className="key",
                    ),
                ],
                className="card controls",
            ),
            html.Div(
                [
                    html.Div([html.H2(id="nb-title"), mono(id="nb-meta")], className="card-head"),
                    dcc.Loading(
                        dcc.Graph(
                            id="nb-graph",
                            config={"displayModeBar": False, "scrollZoom": False},
                            responsive=True,
                            className="nb-graph",
                        ),
                        type="dot",
                        color=INK,
                    ),
                    html.Div(id="nb-table", className="nb-table"),
                    html.Div(
                        [
                            mono("click a node to recentre · drag to pan"),
                            html.Button(
                                id="more-link",
                                n_clicks=0,
                                className="link-button",
                                style={"display": "none"},
                            ),
                            html.Button(
                                "view as table",
                                id="table-toggle",
                                n_clicks=0,
                                className="link-button",
                            ),
                        ],
                        className="card-foot",
                    ),
                    dcc.Store(id="table-on", data=False),
                ],
                className="card network",
            ),
            html.Aside(
                dcc.Loading(html.Div(id="details"), type="dot", color=INK), className="card details"
            ),
        ],
        className="explore",
    )


def network_figure(nb, show):
    pos = radial_layout(nb)
    center = nb["center"]
    center_is_drug = is_drug(center)
    fig = go.Figure()

    def edges(kind, color, width, dash=None):
        xs, ys = [], []
        for e in nb["edges"]:
            if e["kind"] == kind:
                (x0, y0), (x1, y1) = pos[e["source"]], pos[e["target"]]
                xs += [x0, x1, None]
                ys += [y0, y1, None]
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    hoverinfo="skip",
                    line=dict(color=color, width=width, dash=dash),
                )
            )

    if "shared" in show:
        edges("shared", RULE, 1.5)
    if "known" in show:
        edges("known", INK, 2)
    if "predicted" in show:
        edges("predicted", PURPLE, 3, "dash")

    def outward(n):
        x, y = pos[n]
        vert = "top" if y > 0.35 else "bottom" if y < -0.35 else "middle"
        horiz = "right" if x > 0.35 else "left" if x < -0.35 else "center"
        return "top center" if (vert, horiz) == ("middle", "center") else f"{vert} {horiz}"

    def nodes(ids, drug, size, fill, line_color, line_width, texts, textpos=None):
        if not ids:
            return
        textpos = textpos or [outward(n) for n in ids]
        hover = [
            f"<b>{G.nodes[n]['name']}</b><br>{'drug' if drug else 'target'} · {n}"
            f"<br>{G.degree(n)} known {'targets' if drug else 'drugs'}"
            for n in ids
        ]
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in ids],
                y=[pos[n][1] for n in ids],
                mode="markers+text",
                cliponaxis=False,
                customdata=ids,
                hovertext=hover,
                hoverinfo="text",
                text=texts,
                textposition=textpos,
                textfont=dict(family="Archivo", size=12, color=INK),
                marker=dict(
                    symbol="circle" if drug else "square",
                    size=size,
                    color=fill,
                    line=dict(color=line_color, width=line_width),
                ),
            )
        )

    partner_is_drug = not center_is_drug
    partner_fill = BLUE if partner_is_drug else ORANGE
    same_fill = BLUE if center_is_drug else ORANGE

    if "shared" in show:
        nodes(
            nb["related"], center_is_drug, 14, same_fill, CARD, 2, [label(n) for n in nb["related"]]
        )
    if "known" in show:
        many = len(nb["known"]) > 12
        top_labels = set(sorted(nb["known"], key=lambda n: -G.degree(n))[:12])
        nodes(
            nb["known"],
            partner_is_drug,
            16,
            partner_fill,
            CARD,
            2,
            [label(n) if (not many or n in top_labels) else "" for n in nb["known"]],
        )
    if "predicted" in show and nb["predicted"]:
        ids = [p for p, _ in nb["predicted"]]
        nodes(
            ids,
            partner_is_drug,
            16,
            CARD,
            PURPLE,
            3,
            [
                f"{label(p)} · {score_text(s, [v for _, v in nb['predicted']])[0]}"
                for p, s in nb["predicted"]
            ],
        )
    nodes([center], center_is_drug, 30, same_fill, INK, 3, [""], "top center")
    fig.add_annotation(
        x=0,
        y=0.22,
        text=f"<b>{label(center)}</b>",
        showarrow=False,
        yanchor="bottom",
        font=dict(family="Archivo", size=15, color=INK),
        bgcolor=CARD,
    )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        dragmode="pan",
        margin=dict(l=70, r=80, t=24, b=24),
        font=dict(family="Archivo", color=INK),
        xaxis=dict(visible=False, range=[-2.75, 2.75]),
        yaxis=dict(visible=False, range=[-2.65, 2.65]),
        hoverlabel=dict(bgcolor=CARD, bordercolor=INK, font=dict(family="Archivo", color=INK)),
    )
    return fig


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, font=dict(family="Archivo", size=15, color=SECONDARY)
    )
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=8, r=8, t=8, b=8),
    )
    return fig


def score_text(score, shown):
    """Two decimals when they tell this candidate apart; '=' marks a tie."""
    text = f"{score:.2f}"
    tied = sum(f"{s:.2f}" == text for s in shown) > 1
    return ("=" + text, True) if tied else (text, False)


def score_row(name, full_name, score, href, reason, shown=()):
    return html.Div(
        [
            html.Div(
                [
                    dcc.Link(name, href=href, className="row-name"),
                    html.Span(
                        html.Span(
                            style={"width": f"{score * 100:.0f}%"}, className="bar-fill purple"
                        ),
                        className="bar-track",
                    ),
                    html.Span(score_text(score, shown)[0], className="row-score"),
                ],
                className="score-row",
            ),
            html.Div(full_name, className="full-name") if full_name else None,
            html.Div(reason, className="reason"),
        ],
        className="score-item",
    )


def details_panel(node, nb, scores, min_score):
    drug = is_drug(node)
    name = G.nodes[node]["name"]
    if drug:
        info = DRUG_INFO.loc[node]
        status = ", ".join(g for g in str(info["groups"]).split(",") if g) or "no status listed"
        meta = f"{node} · {status} · {info['type']}"
        note = (
            None
            if node in FEATURED
            else html.P(
                "Not featured (not an approved small molecule, or an ion, salt or supplement). "
                "Its predictions are shown but it is left out of the predictions view.",
                className="note",
            )
        )
        if G.degree(node) == 0:
            note = html.P(
                "No known human targets in DrugBank. Its suggestions come from the whole-drug "
                "model, which compares its structure, chemical class and categories with drugs "
                "that do have known targets.",
                className="note",
            )
    else:
        info = TARGET_INFO.loc[node]
        meta = f"UniProt {node}" + (f" · gene {info['gene_symbol']}" if info["gene_symbol"] else "")
        note = None
    ranked = [
        (p, s) for p, s in sorted(scores.items(), key=lambda x: rank_key(G, x)) if s >= min_score
    ]
    shown = [s for _, s in ranked[:5]]
    heading = "Suggested new targets" if drug else "Suggested drugs (approved small molecules)"
    if ranked:
        rows = [
            score_row(
                label(p),
                None if label(p) == G.nodes[p]["name"] else G.nodes[p]["name"],
                s,
                f"/?{urlencode({'node': p})}",
                evidence(G, node, p, label=label, attr=COPIES[copy_for(node)]),
                shown,
            )
            for p, s in ranked[:5]
        ]
        ties = any(score_text(s, shown)[1] for s in shown)
        more = html.Div(
            [
                mono(
                    f"{len(ranked)} candidate{'s' if len(ranked) != 1 else ''} at {min_score:.2f} or above"
                    + ("; top 5 shown" if len(ranked) > 5 else "")
                ),
                html.Div(
                    "= marks tied scores; the model cannot order those candidates.",
                    className="secondary small",
                )
                if ties
                else None,
            ]
        )
    else:
        rows = [
            html.P(
                f"No suggestions at {min_score:.2f} or above. Lower the minimum score to see "
                "weaker candidates.",
                className="empty",
            )
        ]
        more = None
    return html.Div(
        [
            html.Div(
                [
                    mono(f"selected {'drug' if drug else 'target'}"),
                    html.H2(name),
                    html.Div(meta, className="secondary small"),
                ]
            ),
            note,
            html.Div(
                [
                    html.Span(fmt_int(nb["known_total"]), className="stat-small"),
                    f" known {'targets' if drug else 'drugs'}",
                ],
                className="secondary",
            ),
            html.Div(
                [
                    html.Div(heading, className="field-title"),
                    html.Div(
                        "Ranked by calibrated score, then by how rare the target is. A score of 0.9 "
                        "means that, in validation, 90% of pairs scored that high were known "
                        "interactions in a sample with one non-interaction per interaction. Real "
                        "non-interactions are far more common, so read it as a ranking, not a chance "
                        "the interaction is real.",
                        className="secondary small",
                    ),
                ]
            ),
            html.Div(rows, className="score-list"),
            more,
        ],
        className="details-inner",
    )


# ----- predictions view ------------------------------------------------------


def prediction_table(rows_df):
    rows = [
        html.Tr(
            [
                html.Td(
                    dcc.Link(
                        DRUG_INFO.at[r.drug_id, "name"], href=f"/?{urlencode({'node': r.drug_id})}"
                    )
                ),
                html.Td(
                    [
                        html.Span(label(r.target_id), className="strong"),
                        html.Br(),
                        html.Span(TARGET_INFO.at[r.target_id, "name"], className="secondary small"),
                    ]
                ),
                html.Td(f"{r.target_known_drugs:,}", className="num known-drugs"),
                html.Td(f"{r.score:.2f}", className="num score"),
                html.Td(r.evidence, className="secondary small"),
            ]
        )
        for r in rows_df.itertuples()
    ]
    return html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Drug"),
                            html.Th("Suggested target"),
                            html.Th("Target's known drugs"),
                            html.Th("Score"),
                            html.Th("Support in the data"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            className="table pred-table",
        ),
        className="table-wrap",
    )


def predictions_view():
    if PREDICTIONS is None:
        return html.Section(
            html.Div(
                [
                    html.H2("Predicted links"),
                    html.P("No predictions yet. Run: python predict_links.py", className="empty"),
                ],
                className="card",
            )
        )
    first = PREDICTIONS[PREDICTIONS["rank"] == 1]

    def top(copy, n):
        return (
            first[first["copy"] == copy]
            .sort_values(["score", "target_known_drugs", "drug_id"], ascending=[False, True, True])
            .head(n)
        )

    cold, seen = top("cold", 20), top("seen", 40)
    n_cold = first[first["copy"] == "cold"]["drug_id"].nunique()
    n_seen = first[first["copy"] == "seen"]["drug_id"].nunique()
    return html.Section(
        html.Div(
            [
                html.Div(
                    [html.H2("Top suggestion for each drug, strongest first")],
                    className="card-head",
                ),
                html.P(
                    "Each featured drug's highest-scoring target it has no recorded interaction with. "
                    "Featured means an approved small molecule, excluding ions, salts and supplements. "
                    "Ranked by calibrated score, then by target specificity: at equal scores, a target "
                    "with fewer known drugs ranks above a very common one. The known-drugs column shows "
                    "how popular each target already is. These are hypotheses ranked by a model, not "
                    "findings.",
                    className="secondary",
                ),
                html.Div(
                    [
                        html.H3("Drugs with no known targets"),
                        mono(f"{len(cold)} of {n_cold:,} shown"),
                    ],
                    className="card-head",
                ),
                html.P(
                    "Scored by the whole-drug model from structure, chemical class and categories alone.",
                    className="secondary small",
                ),
                prediction_table(cold),
                html.Div(
                    [html.H3("Drugs with known targets"), mono(f"{len(seen)} of {n_seen:,} shown")],
                    className="card-head",
                ),
                prediction_table(seen),
            ],
            className="card",
        )
    )


# ----- evaluation view -------------------------------------------------------


def metric_bar(name, block, colour, key="roc_auc"):
    value = block[key]
    lo, hi = block[f"{key}_ci95"] if key == "roc_auc" else block["average_precision_ci95"]
    return html.Div(
        [
            html.Span(name, className="bar-label"),
            html.Span(
                [
                    html.Span(
                        style={"width": f"{value * 100:.1f}%"}, className="bar-fill " + colour
                    ),
                    html.Span(
                        style={
                            "left": f"{lo * 100:.1f}%",
                            "width": f"{max(hi - lo, 0.002) * 100:.1f}%",
                        },
                        className="ci",
                        title=f"95% interval {lo:.3f}–{hi:.3f}",
                    ),
                ],
                className="bar-track tall",
            ),
            html.Span(
                [
                    html.Strong(f"{value:.2f}"),
                    html.Span(f" {lo:.2f}–{hi:.2f}", className="mono ci-text"),
                ],
                className="bar-value",
            ),
        ],
        className="bar-row",
    )


def split_panel(kind, split):
    rows = [(name, split[key], colour) for key, name, colour in V3_ROWS]
    rows += [(name, split["baselines"][key], "grey") for key, name in BASELINE_ROWS]
    c = split["counts"]
    test_line = (
        f"{c['test_positives']:,} held-out interactions and {c['test_negatives']:,} "
        "sampled non-interactions"
    )
    if "test_drugs" in c:
        test_line += f" across {c['test_drugs']:,} test drugs"
    ap_rows = [
        html.Tr(
            [
                html.Td(name),
                html.Td(f"{b['average_precision']:.3f}", className="num"),
                html.Td("{:.3f}–{:.3f}".format(*b["average_precision_ci95"]), className="num mono"),
            ]
        )
        for name, b, _ in rows
    ]
    return html.Div(
        [
            html.H3(SPLIT_LABELS[kind]),
            html.P(split["description"] + ".", className="secondary small"),
            html.Div("ROC AUC on held-out pairs", className="field-title"),
            *[metric_bar(name, b, colour) for name, b, colour in rows],
            mono(test_line),
            html.Details(
                [
                    html.Summary("Average precision"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([html.Th("Method"), html.Th("AP"), html.Th("95% interval")])
                            ),
                            html.Tbody(ap_rows),
                        ],
                        className="table compact",
                    ),
                ]
            ),
        ],
        className="split",
    )


def headline():
    """The unseen-drug result and its cautious companions, from the results files."""
    cold = V3["splits"]["cold_drug"]
    test = cold["attribute_model"]["roc_auc"]
    val = cold["attribute_model"]["validation"]["roc_auc"]
    abl = ALL_RESULTS["ablation"]["variants"]["without categories and ATC"]["val_roc_auc"]
    sim = {r["bin"]: r for r in ALL_RESULTS["cold_sim"]["rows"]}
    ident = sim["identical (>= 0.999)"]["test_drugs"]
    total = sim["all test drugs"]["test_drugs"]
    no_ident = sim["all except identical"]["attribute_model"]["roc_auc"]
    graph = cold["graph_model"]["roc_auc"]
    return html.Div(
        [
            html.P(
                [
                    html.Strong(
                        f"For a drug the model has never seen, it ranks held-out interactions at "
                        f"ROC AUC {test:.2f}"
                    ),
                    f" (graph-only model: {graph:.2f}, chance: 0.50).",
                ],
                className="lead",
            ),
            html.Ul(
                [
                    html.Li(
                        f"{test:.2f} on the test set; {val:.2f} on the smaller validation set."
                    ),
                    html.Li(
                        f"About {abl:.2f} (validation) when curated DrugBank categories and ATC codes are "
                        "removed. Those are partly assigned because a drug's targets are already known, "
                        "so this is the fairer expectation for a genuinely new compound."
                    ),
                    html.Li(
                        f"Near-duplicates: {ident} of {total:,} test drugs have a structurally identical "
                        f"twin among training drugs; without them the score is {no_ident:.2f}."
                    ),
                ],
                className="headline-list",
            ),
        ],
        className="headline",
    )


def evaluation_view():
    splits = V3["splits"]
    return html.Section(
        html.Div(
            [
                html.Div(
                    [
                        html.H2("How good are the predictions?"),
                        mono(
                            f"from results_attributes.json · run {V3.get('run_date', '')} · "
                            f"commit {V3['git_commit']}"
                        ),
                    ],
                    className="card-head",
                ),
                headline(),
                html.Div(
                    [split_panel(k, splits[k]) for k in ("cold_drug", "drug_holdout", "random")],
                    className="splits three",
                ),
                html.P(
                    "The graph-only model needs a drug's existing interactions, so on unseen drugs it is "
                    "near chance; the attribute model compares the drug's structure, class and categories "
                    "with known binders, which works with or without interactions. Structural nearest "
                    "neighbour is the same structural comparison with no learning.",
                    className="why",
                ),
                html.Div(
                    [
                        html.H3("Read this before trusting it"),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.Strong("Unseen by the graph, not by DrugBank. "),
                                        "A held-out drug still has its DrugBank structure, class and categories. "
                                        "Categories and ATC codes partly encode known mechanism (see the cautious "
                                        "figure above).",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Not by date. "),
                                        "DrugBank has no per-interaction dates, so no split is temporal.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Negatives are unmeasured, not disproven. "),
                                        "They are pairs with no recorded interaction; some are real interactions "
                                        "nobody has tested. The random split's test negatives are uniform; the "
                                        "others use the same drug and a target of similar degree.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Balanced test sets. "),
                                        "One negative per positive, so average precision near 0.5 is chance. Real "
                                        "screening has far more non-interactions, and precision would be lower.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Scores are rankings. "),
                                        "Calibrated on balanced validation pairs; do not read a score as the chance "
                                        "an interaction is real.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Intervals cover test-pair sampling only. "),
                                        "1,000 bootstrap resamples of the test pairs; they do not cover a different "
                                        "split seed or a different DrugBank release.",
                                    ]
                                ),
                            ]
                        ),
                    ],
                    className="caveats",
                ),
            ],
            className="card",
        )
    )


# ----- methods view ----------------------------------------------------------

ATTRIBUTE_DESCRIPTIONS = [
    (
        "Structural similarity to binders",
        "highest Tanimoto similarity (Morgan fingerprints, radius 2, "
        "2,048 bits) between the drug and any drug already known to bind the target",
    ),
    ("Top-3 structural similarity", "mean of the three highest of those similarities"),
    (
        "Protein-family similarity",
        "highest similarity to a drug that binds another target sharing a "
        "Pfam family with this one",
    ),
    (
        "Category overlap",
        "highest Jaccard overlap of DrugBank categories with the target's binders",
    ),
    ("ATC overlap", "highest Jaccard overlap of ATC level-4 codes with the target's binders"),
    ("Same chemical class", "share of the target's binders in the drug's ChemOnt class"),
    ("Same chemical superclass", "share of the target's binders in the drug's ChemOnt superclass"),
    ("Has a structure", "1 if the drug has a usable SMILES string (90% of drugs with targets do)"),
]


def methods_view():
    ds = RESULTS["dataset"]
    r = RESULTS["splits"]["random"]["counts"]
    v2_val = RESULTS["app_model"]["per_drug_validation"]
    abl = ALL_RESULTS["ablation"]["variants"]
    copies = V3["app_model"]["copies"]
    sug = V3.get("app_suggestions", {})
    return html.Section(
        html.Div(
            [
                html.H2("Methods"),
                html.H3("Data"),
                html.P(
                    f"A DrugBank full-database export ({RESULTS['source']['file']}) is parsed into drugs, "
                    f"human targets and interactions. Of {ds['drugs']:,} drug entries, "
                    f"{ds['drugs_with_human_target']:,} have at least one human target; there are "
                    f"{ds['human_targets']:,} human targets and {ds['interactions']:,} interactions. Human "
                    "means NCBI taxonomy 9606 on the target's polypeptide. Drug structures (SMILES), "
                    "categories, ATC codes and ChemOnt classes, and target Pfam families, come from the "
                    "same export. DrugBank is licensed, so no data or derived table is in the repository."
                ),
                html.H3("Graph and embeddings"),
                html.P(
                    "Drugs and targets form a bipartite graph: edges only join a drug to a target. "
                    "Node2Vec (128 dimensions, 10 walks of length 40 per node, p = q = 1, window 10, "
                    "seed 42) embeds every node that has an edge, trained only on each split's message "
                    "graph so held-out edges cannot leak through the embeddings."
                ),
                html.H3("The app's model (v3): graph and attribute features"),
                html.P(
                    "A random forest (100 trees) scores a drug-target pair from 12 graph features (both "
                    "degrees, preferential attachment, shortest-path length, neighbourhood size, the "
                    "embedding dot product and each embedding's mean, plus four neighbourhood-overlap "
                    "features that are always 0 in a bipartite graph) and 8 attribute features. Each "
                    "attribute feature compares the drug with the drugs already known to bind the target, "
                    "using only interactions in the training graph:"
                ),
                html.Ul(
                    [html.Li([html.Strong(n + ". "), d + "."]) for n, d in ATTRIBUTE_DESCRIPTIONS]
                ),
                html.P(
                    "Missing values (no structure, no categories, no known binders) are coded -1. For a "
                    "drug with no interactions the graph features carry nothing, and the attribute "
                    "features do the work."
                ),
                html.H3("Ablation"),
                html.P(
                    "Categories and ATC codes are curated and partly assigned because a drug's targets are "
                    "already known, so they flatter the unseen-drug result. Refitting on the whole-drug "
                    "holdout without some features, scored on validation drugs only (the test set was not "
                    "used): "
                    + "; ".join(f"{name.lower()} {v['val_roc_auc']:.3f}" for name, v in abl.items())
                    + "."
                ),
                html.H3("Evaluation splits"),
                html.P(
                    f"Random edge split: known edges are shuffled (seed 42) and split 60/20/20 into a "
                    f"message graph ({r['message_edges']:,} edges) that features are computed on, training "
                    "positives and test positives; negatives are uniformly random unlinked pairs among "
                    "drugs and targets that have at least one interaction."
                ),
                html.P(
                    "Per-drug split (controls for target popularity): 20% of drugs with two or more "
                    "targets are test drugs and half of each one's edges are hidden; another 20% supply "
                    "training positives the same way. Every test positive is paired with a negative for "
                    "the same drug and a target in the same message-graph degree bin."
                ),
                html.P(
                    "Whole-drug holdout: 20% of drugs with a known target lose every edge and are tested; "
                    "another 20% are made edgeless the same way for training, so training looks like "
                    "testing. Negatives as in the per-drug split."
                ),
                html.P(
                    "In every split a quarter of the training drugs (or edges, for the random split) is "
                    "held back for validation. Choices are made on validation; each test set is scored "
                    "once."
                ),
                html.H3("Baselines"),
                html.P(
                    "Structural nearest neighbour scores a pair by the drug's highest structural "
                    "similarity to the target's known binders, with no learning. Length-3 paths counts "
                    "drug → target′ → drug′ → target routes. Preferential attachment multiplies the two "
                    "degrees. Chance gives every pair the same score. On the whole-drug holdout the two "
                    "graph baselines score every pair 0 and so equal chance."
                ),
                html.P(
                    "Common neighbours is not shown. In a bipartite graph a drug and a target never share "
                    "a neighbour, so it is 0 for every pair and scores exactly 0.5."
                ),
                html.H3("How the app scores a query"),
                html.P(
                    f"Two calibrated copies of the v3 model are used. Drugs with known targets, and target "
                    f"queries, use a copy trained with the per-drug protocol (half the edges of 40% of "
                    f"drugs hidden, no drug losing its last edge; validation ROC AUC "
                    f"{copies['seen']['validation']['roc_auc']:.3f}). Drugs with no known targets use a "
                    f"copy trained with the whole-drug protocol (every edge of 25% of drugs hidden; "
                    f"validation ROC AUC {copies['cold']['validation']['roc_auc']:.3f}). A single copy "
                    "would have to hide some drugs' edges entirely to learn the unseen-drug case, and "
                    "those drugs would then look edgeless when the app scored them. Each copy is "
                    "calibrated with isotonic regression on its validation pairs."
                ),
                html.P(threshold_paragraph(sug)),
                html.P(
                    "The one-line support under each suggestion is read from the data, not from the model, "
                    "in this order: structural similarity of at least 0.5 to a known binder of the target; "
                    "drugs that bind the target and share a target with the selected drug; similarity of "
                    "at least 0.5 to a binder of a target in the same protein family; known binders in the "
                    "same chemical class. Drug entries that are not small molecules are left out of the "
                    "graph by default and kept in the table."
                ),
                html.H3("The v2 graph-only model"),
                html.P(
                    "v2 is the same random forest on the 12 graph features only. It is kept and "
                    "reproducible with python train_ml_models.py, which writes results/results.json. Its "
                    "training negatives were chosen on per-drug validation: degree-matched pairs scored "
                    f"ROC AUC {v2_val['degree_matched']['roc_auc']:.3f} against "
                    f"{v2_val['uniform']['roc_auc']:.3f} for uniform random pairs, which reward preferring "
                    "popular targets. v3 uses the same degree-matched negatives."
                ),
                html.H3("What the first version got wrong"),
                html.Ul(
                    [
                        html.Li(
                            [
                                html.Strong("Leakage. "),
                                "Training positives were described by a graph that still contained the edge "
                                "being predicted, and Node2Vec was trained on the full graph, test edges "
                                "included.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Easy negatives. "),
                                "Negatives were drawn from every DrugBank drug, most of which have no human "
                                "target, so degree alone told them apart from real interactions.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Padded baselines. "),
                                "Common neighbours, Jaccard and Adamic-Adar were always 0, so the code replaced "
                                "them with small random numbers and reported the resulting ~0.50 as a baseline.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Placeholder embeddings. "),
                                "When no embeddings loaded, similarity results were computed from random "
                                "vectors and shown as real.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("A species filter that matched nothing. "),
                                'Targets were kept only if labelled "Homo sapiens"; this DrugBank release says '
                                '"Humans", so the filter dropped every interaction.',
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Labels. "),
                                'The interface showed a hard-coded "ROC AUC: 1.0000" and called a random split '
                                "temporal. An earlier figure of 0.93 ROC AUC could not be reproduced.",
                            ]
                        ),
                    ]
                ),
                html.H3("Reproduce"),
                html.Pre(
                    "\n".join(
                        ["make run", "", "# which runs, as needed:"]
                        + RESULTS["commands"]
                        + V3["commands"]
                        + ["python predict_links.py", "python app.py"]
                    ),
                    className="code",
                ),
            ],
            className="card prose",
        )
    )


def threshold_paragraph(sug):
    if not sug:
        return "Suggestions have not been generated yet. Run: python predict_links.py"
    parts = []
    for key, who in (("seen", "drugs with known targets"), ("cold", "drugs with no known targets")):
        b = sug[key]
        med = ", ".join(
            f"{float(t):.2f}: {v:g}" for t, v in b["median_candidates_at_or_above"].items()
        )
        unmet = (
            ""
            if b.get("rule_met", True)
            else "; no listed threshold meets the rule, so the highest is used and the list is "
            "longer than ten"
        )
        parts.append(
            f"For {who} the default minimum score is {b['default_threshold']:.2f} (median "
            f"candidates at or above each score: {med}{unmet})"
        )
    return (
        "Suggestions rank by calibrated score, then by how few known drugs the target has. "
        + "; ".join(parts)
        + f". Rule: {sug['default_threshold_rule']}. Scores that round to the "
        "same two decimals are marked with '=' because the model cannot order them."
    )


# ----- app -------------------------------------------------------------------

app = dash.Dash(
    __name__,
    title="Drug–target interaction network",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="viewport-width"),
        html.Div(
            [
                header(),
                html.Main(id="page"),
                html.Footer(ATTRIBUTION, className="site-footer mono"),
            ],
            className="wrap",
        ),
    ],
    className="page",
)


@app.callback(
    Output("page", "children"),
    Output("nav", "children"),
    Input("url", "pathname"),
    State("url", "search"),
)
def route(pathname, search):
    pathname = pathname or "/"
    if pathname == "/predictions":
        return predictions_view(), nav(pathname)
    if pathname == "/evaluation":
        return evaluation_view(), nav(pathname)
    if pathname == "/methods":
        return methods_view(), nav(pathname)
    node = parse_qs((search or "").lstrip("?")).get("node", [DEFAULT_NODE])[0]
    if node not in G or (
        G.degree(node) == 0 and not (is_drug(node) and scorable_without_targets(node))
    ):
        node = DEFAULT_NODE
    return explore_view(node), nav("/")


@app.callback(
    Output("node-search", "value"),
    Input("nb-graph", "clickData"),
    State("node-search", "value"),
    prevent_initial_call=True,
)
def recentre(click, current):
    if click and click["points"] and click["points"][0].get("customdata"):
        return click["points"][0]["customdata"]
    return current


app.clientside_callback(
    "function(path) { return window.innerWidth; }",
    Output("viewport-width", "data"),
    Input("url", "pathname"),
)


@app.callback(
    Output("table-on", "data"),
    Input("table-toggle", "n_clicks"),
    Input("more-link", "n_clicks"),
    State("table-on", "data"),
    prevent_initial_call=True,
)
def flip_table(_, __, on):
    return not on


@app.callback(
    Output("table-toggle", "children"),
    Output("nb-table", "style"),
    Output("nb-graph", "style"),
    Input("table-on", "data"),
)
def toggle_table(on):
    if on:
        return "view as network", {"display": "block"}, {"display": "none"}
    return "view as table", {"display": "none"}, {}


@app.callback(
    Output("node-search", "options"),
    Input("node-search", "search_value"),
    State("node-search", "value"),
    prevent_initial_call=True,
)
def search(query, current):
    """Server-side search: sending all 16,000 options made the page slow to load."""
    if not query:
        return dash.no_update
    return match_options(query, current)


@app.callback(
    Output("min-score", "value"), Input("node-search", "value"), prevent_initial_call=True
)
def threshold_for_node(node):
    """Each model copy has its own calibrated default threshold."""
    return THRESHOLDS[copy_for(node)] if node else THRESHOLDS["seen"]


@app.callback(Output("min-score-label", "children"), Input("min-score", "value"))
def min_score_label(v):
    return f"{v:.2f} and above"


@app.callback(
    Output("nb-graph", "figure"),
    Output("nb-title", "children"),
    Output("nb-meta", "children"),
    Output("details", "children"),
    Output("nb-table", "children"),
    Output("more-link", "children"),
    Output("more-link", "style"),
    Input("node-search", "value"),
    Input("show", "value"),
    Input("min-score", "value"),
    Input("all-types", "value"),
    Input("viewport-width", "data"),
)
def render(node, show, min_score, all_types, width):
    hidden = {"display": "none"}
    if not node:
        msg = "Search for a drug or target to see its neighbourhood."
        return (
            empty_figure(msg),
            "No selection",
            "",
            html.P(msg, className="empty"),
            html.P(msg, className="empty"),
            "",
            hidden,
        )
    phone = bool(width) and width <= PHONE_WIDTH
    keep = None if all_types else small_molecule_or_target
    scores = suggestions(node)
    nb = neighbourhood(
        G,
        node,
        scores,
        min_score,
        max_known=12 if phone else 24,
        max_related=5 if phone else 8,
        keep=keep,
    )
    full = neighbourhood(G, node, scores, min_score, max_known=10**6, max_related=8)
    drug = is_drug(node)
    partners = "targets" if drug else "drugs"
    meta = (
        (
            f"no known {partners}"
            if nb["known_total"] == 0
            else f"{nb['known_total']} known {partners}"
        )
        + (f", {nb['known_hidden']} other entry types hidden" if nb["known_hidden"] else "")
        + f" · {len(nb['predicted'])} predicted · {len(nb['related'])} related "
        + ("drugs" if drug else "targets")
    )
    not_drawn = nb["known_total"] - len(nb["known"])
    more_text, more_style = "", hidden
    if not_drawn:
        more_text, more_style = f"+{not_drawn} more, view as table", {}
    table = edge_table(G, full)
    table_el = html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(c.capitalize()) for c in table.columns])),
                html.Tbody(
                    [html.Tr([html.Td(v) for v in row]) for row in table.itertuples(index=False)]
                ),
            ],
            className="table compact",
        ),
        className="table-wrap",
    )
    return (
        network_figure(nb, show or []),
        f"{G.nodes[node]['name']} and its neighbourhood",
        meta,
        details_panel(node, nb, scores, min_score),
        table_el,
        more_text,
        more_style,
    )


if __name__ == "__main__":
    # Local only. Dash debug mode exposes an interactive debugger, so it is off
    # unless DASH_DEBUG=1 is set.
    app.run(
        debug=os.environ.get("DASH_DEBUG") == "1",
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 8050)),
    )
