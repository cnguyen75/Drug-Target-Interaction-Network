"""Drug-target interaction network: explore, predictions, evaluation, methods.

    make run          # builds anything missing, then serves http://127.0.0.1:8050

Every number shown comes from results/results.json, the parsed DrugBank tables
or the trained model; nothing is typed in.
"""

import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import DRUGS_FP, DTI_FP, TARGETS_FP
from pipeline.build_graph import build_bipartite_graph
from pipeline.explore import edge_table, evidence, featured_mask, neighbourhood, radial_layout
from pipeline.link_prediction import DTILinkPredictor

ROOT = Path(__file__).parent
RESULTS_FP = ROOT / "results" / "results.json"
MODEL_FP = ROOT / "models" / "link_prediction_model.pkl"
PREDICTIONS_FP = ROOT / "models" / "predictions.parquet"
EMBEDDINGS_FP = ROOT / "data" / "embeddings_full.parquet"

INK, SECONDARY, RULE, CARD = "#231F1B", "#3D372F", "#B3A992", "#F6F2E8"
BLUE, ORANGE, PURPLE, BASELINE = "#1F6FB5", "#C0601F", "#7A4E9E", "#8A8171"

SPLIT_LABELS = {
    "random": "Random edge split",
    "drug_holdout": "Per-drug split (controls for target popularity)",
}
MODEL_LABELS = {"uniform": "Random forest, uniform negatives",
                "degree_matched": "Random forest, degree-matched negatives"}
METHOD_LABELS = {
    "random_forest": "Random forest",
    "l3_paths": "Length-3 paths",
    "preferential_attachment": "Preferential attachment",
    "chance": "Chance",
}
NEGATIVE_WORDS = {"uniform": "uniform-negative", "degree_matched": "degree-matched"}
OTHER_SCHEME = {"uniform": "degree_matched", "degree_matched": "uniform"}
DEFAULT_NODE_NAME = "Imatinib"


# ----- data ------------------------------------------------------------------

def load():
    missing = [str(p.relative_to(ROOT)) for p in (DRUGS_FP, TARGETS_FP, DTI_FP, RESULTS_FP, MODEL_FP,
                                                   EMBEDDINGS_FP) if not p.exists()]
    if missing:
        sys.exit("Missing " + ", ".join(missing) + ". Run: make run")
    drugs = pd.read_parquet(DRUGS_FP)
    targets = pd.read_parquet(TARGETS_FP)
    dti = pd.read_parquet(DTI_FP)
    G = build_bipartite_graph(drugs, targets, dti)
    predictor = DTILinkPredictor.load_model(MODEL_FP)
    emb = pd.read_parquet(EMBEDDINGS_FP)
    if "node_id" in emb.columns:
        emb = emb.set_index("node_id")
    results = json.loads(RESULTS_FP.read_text())
    predictions = pd.read_parquet(PREDICTIONS_FP) if PREDICTIONS_FP.exists() else None
    return drugs, targets, G, predictor, emb, results, predictions


DRUGS, TARGETS, G, PREDICTOR, EMB, RESULTS, PREDICTIONS = load()
DRUG_INFO = DRUGS.set_index("drug_id")
TARGET_INFO = TARGETS.set_index("target_id")
FEATURED = set(DRUGS.loc[featured_mask(DRUGS), "drug_id"])
SMALL_MOLECULES = set(DRUGS.loc[DRUGS["type"] == "small molecule", "drug_id"])
APP_SCHEME = RESULTS["app_model"]["train_negatives"]
DEFAULT_THRESHOLD = RESULTS.get("app_suggestions", {}).get("default_threshold", 0.9)
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


def search_options():
    opts = []
    for n, d in G.nodes(data=True):
        if G.degree(n) == 0:
            continue
        if d["node_type"] == "drug":
            opts.append({"label": f"{d['name']} · drug", "value": n, "search": f"{d['name']} {n}"})
        else:
            gene = TARGET_INFO.at[n, "gene_symbol"]
            text = f"{gene} — {d['name']}" if gene else d["name"]
            opts.append({"label": f"{text} · target", "value": n, "search": f"{text} {n}"})
    opts.sort(key=lambda o: o["label"].lower())
    return opts


SEARCH_OPTIONS = search_options()
DEFAULT_NODE = next((n for n in sorted(FEATURED) if G.nodes[n]["name"] == DEFAULT_NODE_NAME
                     and G.degree(n) > 0), max(FEATURED, key=G.degree))


def suggestions(node):
    """Scores for unlinked partners. A target's suggestions are featured drugs only."""
    if is_drug(node):
        return PREDICTOR.score_candidates(node)
    cands = [d for d in sorted(FEATURED) if G.degree(d) > 0 and not G.has_edge(node, d)]
    return PREDICTOR.score_candidates(node, cands)


# ----- small components ------------------------------------------------------

def mono(text=None, className="mono", **kw):
    return html.Span(text, className=className, **kw)


def fmt_int(n):
    return f"{n:,}"


def header():
    ds = RESULTS["dataset"]
    return html.Div([
        html.Header([
            html.Div([
                mono("drugbank · node2vec · random forest link prediction"),
                html.H1("Drug–target interaction network"),
            ], className="title"),
            html.Nav(id="nav", **{"aria-label": "Views"}),
        ], className="site-header"),
        html.Section([
            html.Div([html.Span(fmt_int(ds["drugs_with_human_target"]), className="stat"),
                      " ", html.Span("drugs with a human target", className="secondary")]),
            html.Div([html.Span(fmt_int(ds["human_targets"]), className="stat"),
                      " ", html.Span("human targets", className="secondary")]),
            html.Div([html.Span(fmt_int(ds["interactions"]), className="stat"),
                      " ", html.Span("known interactions", className="secondary")]),
            mono(f"DrugBank {RESULTS['source'].get('version', '')}, exported "
                 f"{RESULTS['source'].get('exported_on', '')} · counts from results.json",
                 className="mono stats-note"),
        ], className="stats"),
    ], className="head-block")


VIEWS = [("explore", "/"), ("predictions", "/predictions"), ("evaluation", "/evaluation"),
         ("methods", "/methods")]


def nav(pathname):
    # The current view is a plain anchor so it can carry aria-current; dcc.Link cannot.
    return [html.A(name, href=href, className="nav-link active", **{"aria-current": "page"})
            if pathname == href else dcc.Link(name, href=href, className="nav-link")
            for name, href in VIEWS]


# ----- explore view ----------------------------------------------------------

def key_item(svg, text):
    return html.Div([svg, html.Span(text)], className="key-item")


def explore_view(node):
    return html.Section([
        html.Aside([
            html.Label([
                "Find a drug or target",
                dcc.Dropdown(id="node-search", options=SEARCH_OPTIONS, value=node, clearable=True,
                             placeholder="Type a drug, gene or protein name", optionHeight=44,
                             className="search"),
            ], className="field"),
            html.Div([
                html.Div("Show", className="field-title"),
                dcc.Checklist(id="show", value=["known", "predicted", "shared"], className="checks",
                              options=[{"label": "Known interactions", "value": "known"},
                                       {"label": "Predicted links", "value": "predicted"},
                                       {"label": "Nodes that share a partner", "value": "shared"}]),
                dcc.Checklist(id="all-types", value=[], className="checks",
                              options=[{"label": "Show all entry types", "value": "all"}]),
                html.Div("Off: drug entries that are not small molecules (biologics, cell "
                         "therapies) are left out of the graph but kept in the table.",
                         className="secondary small"),
            ]),
            html.Div([
                html.Div("Minimum prediction score", className="field-title"),
                dcc.Slider(id="min-score", min=0, max=1, step=0.05, value=DEFAULT_THRESHOLD,
                           marks=None, tooltip=None, className="slider"),
                mono(id="min-score-label"),
            ]),
            html.Div([
                html.Div("Key", className="field-title"),
                key_item(html.Span(className="glyph drug"), "Drug (circle)"),
                key_item(html.Span(className="glyph target"), "Target (square)"),
                key_item(html.Span(className="glyph line known"), "Known interaction"),
                key_item(html.Span(className="glyph line predicted"), "Predicted link"),
                key_item(html.Span(className="glyph line shared"), "Shares a partner"),
            ], className="key"),
        ], className="card controls"),
        html.Div([
            html.Div([html.H2(id="nb-title"), mono(id="nb-meta")], className="card-head"),
            dcc.Loading(dcc.Graph(id="nb-graph", config={"displayModeBar": False, "scrollZoom": False},
                                  responsive=True, className="nb-graph"),
                        type="dot", color=INK),
            html.Div(id="nb-table", className="nb-table"),
            html.Div([mono("click a node to recentre · drag to pan"),
                      html.Button(id="more-link", n_clicks=0, className="link-button",
                                  style={"display": "none"}),
                      html.Button("view as table", id="table-toggle", n_clicks=0,
                                  className="link-button")], className="card-foot"),
            dcc.Store(id="table-on", data=False),
        ], className="card network"),
        html.Aside(dcc.Loading(html.Div(id="details"), type="dot", color=INK),
                   className="card details"),
    ], className="explore")


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
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     line=dict(color=color, width=width, dash=dash)))

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
        hover = [f"<b>{G.nodes[n]['name']}</b><br>{'drug' if drug else 'target'} · {n}"
                 f"<br>{G.degree(n)} known {'targets' if drug else 'drugs'}" for n in ids]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ids], y=[pos[n][1] for n in ids], mode="markers+text", cliponaxis=False,
            customdata=ids, hovertext=hover, hoverinfo="text", text=texts, textposition=textpos,
            textfont=dict(family="Archivo", size=12, color=INK),
            marker=dict(symbol="circle" if drug else "square", size=size, color=fill,
                        line=dict(color=line_color, width=line_width))))

    partner_is_drug = not center_is_drug
    partner_fill = BLUE if partner_is_drug else ORANGE
    same_fill = BLUE if center_is_drug else ORANGE

    if "shared" in show:
        nodes(nb["related"], center_is_drug, 14, same_fill, CARD, 2,
              [label(n) for n in nb["related"]])
    if "known" in show:
        many = len(nb["known"]) > 12
        top_labels = set(sorted(nb["known"], key=lambda n: -G.degree(n))[:12])
        nodes(nb["known"], partner_is_drug, 16, partner_fill, CARD, 2,
              [label(n) if (not many or n in top_labels) else "" for n in nb["known"]])
    if "predicted" in show and nb["predicted"]:
        ids = [p for p, _ in nb["predicted"]]
        nodes(ids, partner_is_drug, 16, CARD, PURPLE, 3,
              [f"{label(p)} · {s:.2f}" for p, s in nb["predicted"]])
    nodes([center], center_is_drug, 30, same_fill, INK, 3, [""], "top center")
    fig.add_annotation(x=0, y=0.22, text=f"<b>{label(center)}</b>", showarrow=False, yanchor="bottom",
                       font=dict(family="Archivo", size=15, color=INK), bgcolor=CARD)

    fig.update_layout(
        showlegend=False, paper_bgcolor=CARD, plot_bgcolor=CARD, dragmode="pan",
        margin=dict(l=70, r=80, t=24, b=24), font=dict(family="Archivo", color=INK),
        xaxis=dict(visible=False, range=[-2.6, 2.6]),
        yaxis=dict(visible=False, range=[-2.5, 2.5]),
        hoverlabel=dict(bgcolor=CARD, bordercolor=INK, font=dict(family="Archivo", color=INK)),
    )
    return fig


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(family="Archivo", size=15,
                                                                color=SECONDARY))
    fig.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD, xaxis=dict(visible=False),
                      yaxis=dict(visible=False), margin=dict(l=8, r=8, t=8, b=8))
    return fig


def score_text(score, shown):
    """Two decimals when they tell this candidate apart; '=' marks a tie."""
    text = f"{score:.2f}"
    tied = sum(f"{s:.2f}" == text for s in shown) > 1
    return ("=" + text, True) if tied else (text, False)


def score_row(name, full_name, score, href, reason, shown=()):
    return html.Div([
        html.Div([
            dcc.Link(name, href=href, className="row-name"),
            html.Span(html.Span(style={"width": f"{score * 100:.0f}%"}, className="bar-fill purple"),
                      className="bar-track"),
            html.Span(score_text(score, shown)[0], className="row-score"),
        ], className="score-row"),
        html.Div(full_name, className="full-name") if full_name else None,
        html.Div(reason, className="reason"),
    ], className="score-item")


def details_panel(node, nb, scores, min_score):
    drug = is_drug(node)
    name = G.nodes[node]["name"]
    if drug:
        info = DRUG_INFO.loc[node]
        status = ", ".join(g for g in str(info["groups"]).split(",") if g) or "no status listed"
        meta = f"{node} · {status} · {info['type']}"
        note = None if node in FEATURED else html.P(
            "Not featured (not an approved small molecule, or an ion, salt or supplement). "
            "Its predictions are shown but it is left out of the predictions view.", className="note")
    else:
        info = TARGET_INFO.loc[node]
        meta = f"UniProt {node}" + (f" · gene {info['gene_symbol']}" if info["gene_symbol"] else "")
        note = None
    ranked = [(p, s) for p, s in sorted(scores.items(), key=lambda x: (-x[1], G.degree(x[0]), x[0]))
              if s >= min_score]
    shown = [s for _, s in ranked[:5]]
    heading = "Suggested new targets" if drug else "Suggested drugs (approved small molecules)"
    if ranked:
        rows = [score_row(label(p), None if label(p) == G.nodes[p]["name"] else G.nodes[p]["name"], s,
                          f"/?{urlencode({'node': p})}", evidence(G, node, p, EMB, label=label), shown)
                for p, s in ranked[:5]]
        ties = any(score_text(s, shown)[1] for s in shown)
        more = html.Div([
            mono(f"{len(ranked)} candidate{'s' if len(ranked) != 1 else ''} at {min_score:.2f} or above"
                 + ("; top 5 shown" if len(ranked) > 5 else "")),
            html.Div("= marks tied scores; the model cannot order those candidates.",
                     className="secondary small") if ties else None,
        ])
    else:
        rows = [html.P(f"No suggestions at {min_score:.2f} or above. Lower the minimum score to see "
                       "weaker candidates.", className="empty")]
        more = None
    return html.Div([
        html.Div([mono(f"selected {'drug' if drug else 'target'}"), html.H2(name),
                  html.Div(meta, className="secondary small")]),
        note,
        html.Div([html.Span(fmt_int(nb["known_total"]), className="stat-small"),
                  f" known {'targets' if drug else 'drugs'}"], className="secondary"),
        html.Div([
            html.Div(heading, className="field-title"),
            html.Div("Ranked by calibrated score, then by how rare the target is. A score of 0.9 "
                     "means that, in validation, 90% of pairs scored that high were known "
                     "interactions in a sample with one non-interaction per interaction. Real "
                     "non-interactions are far more common, so read it as a ranking, not a chance "
                     "the interaction is real.", className="secondary small"),
        ]),
        html.Div(rows, className="score-list"),
        more,
    ], className="details-inner")


# ----- predictions view ------------------------------------------------------

def predictions_view():
    if PREDICTIONS is None:
        return html.Section(html.Div([
            html.H2("Predicted links"),
            html.P("No predictions yet. Run: python predict_links.py", className="empty"),
        ], className="card"))
    top = PREDICTIONS[PREDICTIONS["rank"] == 1].sort_values(
        ["score", "target_known_drugs", "drug_id"], ascending=[False, True, True]).head(40)
    rows = [html.Tr([
        html.Td(dcc.Link(DRUG_INFO.at[r.drug_id, "name"], href=f"/?{urlencode({'node': r.drug_id})}")),
        html.Td([html.Span(label(r.target_id), className="strong"), html.Br(),
                 html.Span(TARGET_INFO.at[r.target_id, "name"], className="secondary small")]),
        html.Td(f"{r.target_known_drugs:,}", className="num known-drugs"),
        html.Td(f"{r.score:.2f}", className="num score"),
        html.Td(r.evidence, className="secondary small"),
    ]) for r in top.itertuples()]
    n_drugs = PREDICTIONS["drug_id"].nunique()
    return html.Section(html.Div([
        html.Div([html.H2("Top suggestion for each drug, strongest first"),
                  mono(f"{len(top)} of {n_drugs:,} featured drugs shown")], className="card-head"),
        html.P("Each featured drug's highest-scoring target it has no recorded interaction with. "
               "Featured means an approved small molecule, excluding ions, salts and supplements. "
               "Ranked by calibrated score, then by target specificity: at equal scores, a target "
               "with fewer known drugs ranks above a very common one. The known-drugs column shows "
               "how popular each target already is. These are hypotheses ranked by a model, not "
               "findings.", className="secondary"),
        html.Div(html.Table([
            html.Thead(html.Tr([html.Th("Drug"), html.Th("Suggested target"),
                                html.Th("Target's known drugs"), html.Th("Score"),
                                html.Th("Support in the data")])),
            html.Tbody(rows),
        ], className="table pred-table"), className="table-wrap"),
    ], className="card"))


# ----- evaluation view -------------------------------------------------------

def metric_bar(name, block, primary, key="roc_auc"):
    value = block[key]
    lo, hi = block[f"{key}_ci95"] if key == "roc_auc" else block["average_precision_ci95"]
    return html.Div([
        html.Span(name, className="bar-label"),
        html.Span([
            html.Span(style={"width": f"{value * 100:.1f}%"},
                      className="bar-fill " + ("blue" if primary else "grey")),
            html.Span(style={"left": f"{lo * 100:.1f}%", "width": f"{max(hi - lo, 0.002) * 100:.1f}%"},
                      className="ci", title=f"95% interval {lo:.3f}–{hi:.3f}"),
        ], className="bar-track tall"),
        html.Span([html.Strong(f"{value:.2f}"), html.Span(f" {lo:.2f}–{hi:.2f}", className="mono ci-text")],
                  className="bar-value"),
    ], className="bar-row")


def split_panel(kind, split):
    models = split["models"]
    rows = [(MODEL_LABELS[m] + (" (app)" if m == APP_SCHEME else ""), models[m], True)
            for m in ("uniform", "degree_matched")]
    rows += [(METHOD_LABELS[b], split["baselines"][b], False)
             for b in ("l3_paths", "preferential_attachment", "chance")]
    c = split["counts"]
    test_line = (f"{c['test_positives']:,} held-out interactions and {c['test_negatives']:,} "
                 "sampled non-interactions")
    if kind == "drug_holdout":
        test_line += f" across {c['test_drugs']:,} test drugs"
    ap_rows = [html.Tr([html.Td(name), html.Td(f"{b['average_precision']:.3f}", className="num"),
                        html.Td("{:.3f}–{:.3f}".format(*b["average_precision_ci95"]),
                                className="num mono")]) for name, b, _ in rows]
    return html.Div([
        html.H3(SPLIT_LABELS[kind]),
        html.P(split["description"] + ".", className="secondary small"),
        html.Div("ROC AUC on held-out pairs", className="field-title"),
        *[metric_bar(name, b, primary) for name, b, primary in rows],
        mono(test_line + " · bars show the score, the dark band its 95% interval"),
        html.Details([
            html.Summary("Average precision"),
            html.Table([html.Thead(html.Tr([html.Th("Method"), html.Th("AP"), html.Th("95% interval")])),
                        html.Tbody(ap_rows)], className="table compact"),
        ]),
    ], className="split")


def why_sentence():
    """Why the two splits disagree, written from results.json."""
    s = RESULTS["splits"]
    u_r, u_d = (s[k]["models"]["uniform"]["roc_auc"] for k in ("random", "drug_holdout"))
    d_r, d_d = (s[k]["models"]["degree_matched"]["roc_auc"] for k in ("random", "drug_holdout"))
    return (f"Each model does best where its test negatives look like its training negatives. The "
            f"random split's test negatives are uniform pairs, which mostly involve rarely studied "
            f"targets, so preferring popular targets pays: the uniform-negative model scores "
            f"{u_r:.2f} there but {u_d:.2f} on the per-drug split. The degree-matched model has "
            f"learned not to lean on popularity, so it scores {d_r:.2f} on the random split and "
            f"{d_d:.2f} on the per-drug split. The random split also strands one-target drugs with "
            f"no edges left, which the per-drug split avoids by keeping at least one per test drug.")


def evaluation_view():
    splits = RESULTS["splits"]
    app_m = RESULTS["app_model"]
    val = app_m["per_drug_validation"]
    app_rf = {k: v["models"][APP_SCHEME] for k, v in splits.items()}
    return html.Section(html.Div([
        html.Div([html.H2("How good are the predictions?"),
                  mono(f"from results.json · run {RESULTS.get('run_date', 'date not recorded')} · "
                       f"commit {RESULTS['git_commit']}")], className="card-head"),
        html.P(["Two random forests differ only in their training negatives: uniform random "
                "pairs, or pairs that use the same drug and a target of similar popularity. The "
                f"app uses the {NEGATIVE_WORDS[APP_SCHEME]} model, chosen on the per-drug split's "
                f"validation drugs (ROC AUC {val[APP_SCHEME]['roc_auc']:.3f} against "
                f"{val[OTHER_SCHEME[APP_SCHEME]]['roc_auc']:.3f}) before any test pair was scored."],
               className="secondary"),
        html.Div([split_panel(k, v) for k, v in splits.items()], className="splits"),
        html.P(why_sentence(), className="why"),
        html.Div([
            html.H3("Read this before trusting it"),
            html.Ul([
                html.Li([html.Strong("Not by date. "),
                         "DrugBank has no per-interaction dates, so neither split is temporal. Both "
                         "test drugs and targets the model has already seen in training."]),
                html.Li([html.Strong("No unseen drugs. "),
                         "Node2Vec only embeds nodes that have edges, so a drug with no known targets "
                         "cannot be scored. These numbers say nothing about brand-new compounds."]),
                html.Li([html.Strong("Negatives are unmeasured, not disproven. "),
                         "They are pairs with no recorded interaction; some are real interactions "
                         "nobody has tested. The random split's test negatives are uniform; the "
                         "per-drug split's use the same drug and a target of similar degree."]),
                html.Li([html.Strong("Balanced test sets. "),
                         "One negative per positive, so average precision near 0.5 is chance. Real "
                         "screening has far more non-interactions, and precision would be lower."]),
                html.Li([html.Strong("Scores are rankings. "),
                         f"Brier scores for the app's model are {app_rf['random']['brier_score']:.3f} "
                         f"(random) and {app_rf['drug_holdout']['brier_score']:.3f} (per-drug), on "
                         "balanced data and before calibration."]),
                html.Li([html.Strong("Intervals cover test-pair sampling only. "),
                         f"{splits['random']['bootstrap_resamples']:,} bootstrap resamples of the test "
                         "pairs; they do not cover a different split seed or a different DrugBank "
                         "release."]),
            ]),
        ], className="caveats"),
    ], className="card"))


# ----- methods view ----------------------------------------------------------

def app_model_paragraph():
    m = RESULTS["app_model"]
    val = m["per_drug_validation"]
    other = OTHER_SCHEME[m["train_negatives"]]
    return (f"The app uses the random forest trained with {RESULTS['negative_schemes'][m['train_negatives']]}. "
            f"On the per-drug split's validation drugs it scored ROC AUC "
            f"{val[m['train_negatives']]['roc_auc']:.3f}, against {val[other]['roc_auc']:.3f} for "
            f"{RESULTS['negative_schemes'][other]}; that comparison was made before any test pair "
            "was scored. Uniform negatives teach the model that popular targets are likely partners, "
            "which pushed the same few receptors to the top of every drug's list. The app's copy is "
            "retrained on all known edges (80% as the message graph, 20% as training positives, a "
            "quarter of those held back) and its scores are calibrated with isotonic regression on "
            f"that held-back set (validation ROC AUC {m['validation']['roc_auc']:.3f}).")


def suggestions_paragraph():
    a = RESULTS.get("app_suggestions")
    if not a:
        return "Suggestions have not been generated yet. Run: python predict_links.py"
    med = a["median_candidates_at_or_above"]
    counts = ", ".join(f"{float(t):.2f}: {v:g}" for t, v in med.items())
    return (f"For every drug, every target it has no recorded interaction with is scored, then "
            f"ranked by calibrated score and, at equal scores, by how few known drugs the target "
            f"has. The median featured drug has this many candidates at or above each score "
            f"({counts}). The default minimum score is {a['default_threshold']:.2f}: the "
            f"{a['default_threshold_rule']}. Scores that round to the same two decimals are marked "
            "with '=' because the model cannot order them.")



def methods_view():
    ds = RESULTS["dataset"]
    r = RESULTS["splits"]["random"]["counts"]
    return html.Section(html.Div([
        html.H2("Methods"),
        html.H3("Data"),
        html.P(f"A DrugBank full-database export ({RESULTS['source']['file']}) is parsed into drugs, "
               f"human targets and interactions. Of {ds['drugs']:,} drug entries, "
               f"{ds['drugs_with_human_target']:,} have at least one human target; there are "
               f"{ds['human_targets']:,} human targets and {ds['interactions']:,} interactions. Human "
               "means NCBI taxonomy 9606 on the target's polypeptide. DrugBank is licensed, so no "
               "data or derived table is in the repository."),
        html.H3("Graph and embeddings"),
        html.P("Drugs and targets form a bipartite graph: edges only join a drug to a target. "
               "Node2Vec (128 dimensions, 10 walks of length 40 per node, p = q = 1, window 10, "
               "seed 42) embeds every node that has an edge. For each evaluation, Node2Vec is trained "
               "only on that split's message graph, so held-out edges cannot leak through the "
               "embeddings."),
        html.H3("Model"),
        html.P("A random forest (100 trees) scores a drug-target pair from 12 features: both "
               "degrees, preferential attachment, shortest-path length, neighbourhood size, the "
               "embedding dot product and each embedding's mean, plus four neighbourhood-overlap "
               "features kept from the first version that are always 0 here (see below)."),
        html.H3("Evaluation splits"),
        html.P(f"Random edge split: known edges are shuffled (seed 42) and split 60/20/20 into a "
               f"message graph ({r['message_edges']:,} edges) that features are computed on, training "
               "positives and test positives; negatives are uniformly random unlinked pairs among "
               "drugs and targets that have at least one interaction."),
        html.P("Per-drug split (controls for target popularity): 20% of drugs with two or more "
               "targets are test drugs and half of each one's edges are hidden; another 20% supply "
               "training positives the same way. Every test positive is paired with a negative for "
               "the same drug and a target in the same degree bin (log2 of its degree), so "
               "preferring popular targets earns nothing."),
        html.P("In both splits a quarter of the training positives (by drug in the per-drug split) "
               "is held back as a validation set. Choices are made on validation; each test set is "
               "scored once, after the choices."),
        html.H3("Baselines"),
        html.P("Length-3 paths counts drug → target′ → drug′ → target routes. Preferential "
               "attachment multiplies the two degrees. Chance gives every pair the same score."),
        html.P("Common neighbours is not shown. In a bipartite graph a drug's neighbours are all "
               "targets and a target's are all drugs, so a drug and a target never share one: the "
               "count is 0 for every pair, it ties everything and scores exactly 0.5. Length-3 paths "
               "is its bipartite counterpart. The same holds for Jaccard, Adamic-Adar and resource "
               "allocation, which is why those four model features are always 0."),
        html.H3("Which model the app uses, and why"),
        html.P(app_model_paragraph()),
        html.H3("Suggestions and evidence"),
        html.P(suggestions_paragraph()),
        html.P("The one-line support under each suggestion is read from the data, not from the "
               "model: drugs that bind the suggested target and share a target with the selected "
               "drug, or, failing that, the drug's 25 nearest neighbours in the embedding that bind "
               "it. Drug entries that are not small molecules (biologics, cell therapies) are left "
               "out of the graph by default and kept in the table."),
        html.H3("What the first version got wrong"),
        html.Ul([
            html.Li([html.Strong("Leakage. "),
                     "Training positives were described by a graph that still contained the edge "
                     "being predicted, and Node2Vec was trained on the full graph, test edges "
                     "included."]),
            html.Li([html.Strong("Easy negatives. "),
                     "Negatives were drawn from every DrugBank drug, most of which have no human "
                     "target, so degree alone told them apart from real interactions."]),
            html.Li([html.Strong("Padded baselines. "),
                     "Common neighbours, Jaccard and Adamic-Adar were always 0, so the code replaced "
                     "them with small random numbers and reported the resulting ~0.50 as a baseline."]),
            html.Li([html.Strong("Placeholder embeddings. "),
                     "When no embeddings loaded, similarity results were computed from random "
                     "vectors and shown as real."]),
            html.Li([html.Strong("A species filter that matched nothing. "),
                     "Targets were kept only if labelled \"Homo sapiens\"; this DrugBank release says "
                     "\"Humans\", so the filter dropped every interaction."]),
            html.Li([html.Strong("Labels. "),
                     "The interface showed a hard-coded \"ROC AUC: 1.0000\" and called a random split "
                     "temporal. An earlier figure of 0.93 ROC AUC could not be reproduced."]),
        ]),
        html.H3("Reproduce"),
        html.Pre("\n".join(["make run", "", "# which runs, as needed:"] + RESULTS["commands"]
                           + ["python predict_links.py", "python app.py"]), className="code"),
    ], className="card prose"))


# ----- app -------------------------------------------------------------------

app = dash.Dash(__name__, title="Drug–target interaction network", update_title=None,
                suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="viewport-width"),
    html.Div([header(), html.Main(id="page")], className="wrap"),
], className="page")


@app.callback(Output("page", "children"), Output("nav", "children"), Input("url", "pathname"),
              State("url", "search"))
def route(pathname, search):
    pathname = pathname or "/"
    if pathname == "/predictions":
        return predictions_view(), nav(pathname)
    if pathname == "/evaluation":
        return evaluation_view(), nav(pathname)
    if pathname == "/methods":
        return methods_view(), nav(pathname)
    node = parse_qs((search or "").lstrip("?")).get("node", [DEFAULT_NODE])[0]
    if node not in G or G.degree(node) == 0:
        node = DEFAULT_NODE
    return explore_view(node), nav("/")


@app.callback(Output("node-search", "value"), Input("nb-graph", "clickData"),
              State("node-search", "value"), prevent_initial_call=True)
def recentre(click, current):
    if click and click["points"] and click["points"][0].get("customdata"):
        return click["points"][0]["customdata"]
    return current


app.clientside_callback(
    "function(path) { return window.innerWidth; }",
    Output("viewport-width", "data"), Input("url", "pathname"))


@app.callback(Output("table-on", "data"), Input("table-toggle", "n_clicks"),
              Input("more-link", "n_clicks"), State("table-on", "data"), prevent_initial_call=True)
def flip_table(_, __, on):
    return not on


@app.callback(Output("table-toggle", "children"), Output("nb-table", "style"),
              Output("nb-graph", "style"), Input("table-on", "data"))
def toggle_table(on):
    if on:
        return "view as network", {"display": "block"}, {"display": "none"}
    return "view as table", {"display": "none"}, {}


@app.callback(Output("min-score-label", "children"), Input("min-score", "value"))
def min_score_label(v):
    return f"{v:.2f} and above"


@app.callback(Output("nb-graph", "figure"), Output("nb-title", "children"), Output("nb-meta", "children"),
              Output("details", "children"), Output("nb-table", "children"),
              Output("more-link", "children"), Output("more-link", "style"),
              Input("node-search", "value"), Input("show", "value"), Input("min-score", "value"),
              Input("all-types", "value"), Input("viewport-width", "data"))
def render(node, show, min_score, all_types, width):
    hidden = {"display": "none"}
    if not node:
        msg = "Search for a drug or target to see its neighbourhood."
        return (empty_figure(msg), "No selection", "", html.P(msg, className="empty"),
                html.P(msg, className="empty"), "", hidden)
    phone = bool(width) and width <= PHONE_WIDTH
    keep = None if all_types else small_molecule_or_target
    scores = suggestions(node)
    nb = neighbourhood(G, node, scores, min_score, max_known=12 if phone else 24,
                       max_related=5 if phone else 8, keep=keep)
    full = neighbourhood(G, node, scores, min_score, max_known=10 ** 6, max_related=8)
    drug = is_drug(node)
    partners = "targets" if drug else "drugs"
    meta = (f"{nb['known_total']} known {partners}"
            + (f", {nb['known_hidden']} other entry types hidden" if nb["known_hidden"] else "")
            + f" · {len(nb['predicted'])} predicted · {len(nb['related'])} related "
            + ("drugs" if drug else "targets"))
    not_drawn = nb["known_total"] - len(nb["known"])
    more_text, more_style = "", hidden
    if not_drawn:
        more_text, more_style = f"+{not_drawn} more, view as table", {}
    table = edge_table(G, full)
    table_el = html.Div(html.Table([
        html.Thead(html.Tr([html.Th(c.capitalize()) for c in table.columns])),
        html.Tbody([html.Tr([html.Td(v) for v in row]) for row in table.itertuples(index=False)]),
    ], className="table compact"), className="table-wrap")
    return (network_figure(nb, show or []), f"{G.nodes[node]['name']} and its neighbourhood", meta,
            details_panel(node, nb, scores, min_score), table_el, more_text, more_style)


if __name__ == "__main__":
    # Local only. Dash debug mode exposes an interactive debugger, so it is off
    # unless DASH_DEBUG=1 is set.
    app.run(debug=os.environ.get("DASH_DEBUG") == "1", host="127.0.0.1",
            port=int(os.environ.get("PORT", 8050)))
