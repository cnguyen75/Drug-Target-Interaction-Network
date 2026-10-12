"""Data behind the explore view: featured drugs, neighbourhoods, layout and evidence text."""

import math
import re

import numpy as np
import pandas as pd

# Carbon atom in SMILES: aromatic c, or C not starting a two-letter element (Cl, Ca, Cu, ...).
_CARBON = re.compile(r"c|C(?![ladeflmnorsu])")


def has_carbon(smiles):
    """True if a SMILES string contains a carbon atom (bracket atoms reduced to their element)."""
    atoms = re.sub(r"\[\d*([A-Za-z][a-z]?)[^\]]*\]", r"\1", smiles)
    return bool(_CARBON.search(atoms))


def featured_mask(drugs_df):
    """Approved small molecules, excluding nutraceuticals, mixtures/salts and inorganic entries.

    DrugBank lists ions, salts and supplements (Copper, Zinc acetate, NADH,
    Glutamic acid, ...) as approved small molecules. They stay searchable but
    are not featured. A salt or mixture is a SMILES with more than one
    '.'-separated fragment; an inorganic entry has no carbon atom.
    """
    groups = drugs_df["groups"].fillna("").str.split(",")
    smiles = drugs_df["smiles"].fillna("")
    return (
        drugs_df["approved"].astype(bool)
        & (drugs_df["type"] == "small molecule")
        & ~groups.apply(lambda g: "nutraceutical" in g)
        & (smiles != "")
        & ~smiles.str.contains(".", regex=False)
        & smiles.apply(has_carbon)
    )


def _names(G, nodes, label=None):
    return [label(n) if label else (G.nodes[n].get("name") or n) for n in nodes]


def _join(names):
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def evidence(G, node, partner, emb=None, max_names=2, label=None):
    """One sentence of support in the data for a suggested link node-partner.

    Uses the full known graph. Paths of length 3 first (other drugs that bind
    the suggested target and share a target with the drug); otherwise the
    node's nearest neighbours in the full-graph embedding that already link to
    the partner; otherwise says no path exists. `label` maps a node id to the
    name shown (for example a target's gene symbol).
    """
    is_drug = G.nodes[node]["node_type"] == "drug"
    node_partners = set(G.neighbors(node))
    support = []
    for other in G.neighbors(partner):
        shared = len(node_partners & set(G.neighbors(other)))
        if other != node and shared:
            support.append((shared, other))
    name = label(node) if label else (G.nodes[node].get("name") or node)
    if support:
        support.sort(key=lambda x: (-x[0], x[1]))
        shown = _names(G, [o for _, o in support[:max_names]], label)
        if len(support) > max_names:
            who = f"{', '.join(shown)} and {len(support) - max_names} more"
        else:
            who = _join(shown)
        verb = "shares" if len(support) == 1 else "share"
        if is_drug:
            return f"Bound by {who}, which {verb} targets with {name}."
        return f"Binds {who}, which {verb} drugs with {name}."

    if emb is not None and node in emb.index:
        same_type = [n for n in emb.index if n in G and G.nodes[n]["node_type"] ==
                     G.nodes[node]["node_type"] and n != node]
        vecs = emb.loc[same_type].values
        v = emb.loc[node].values
        sims = vecs @ v / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(v) + 1e-12)
        nearest = [same_type[i] for i in np.argsort(-sims)[:25]]
        linked = [n for n in nearest if G.has_edge(n, partner)][:max_names]
        if linked:
            kind = "drugs" if is_drug else "targets"
            return f"Among its 25 nearest {kind} in the embedding, {_join(_names(G, linked, label))} link to it."
    return "No shared-target path in the data; the score comes from degree and embedding position."


def rank_key(G, item):
    """Suggestion order used everywhere: score, then rarer partner, then id."""
    partner, score = item
    return (-round(float(score), 6), G.degree(partner), partner)


def neighbourhood(G, node, scores, min_score, max_known=24, max_related=8, max_predicted=5,
                  keep=None):
    """Nodes and edges around `node` for the network figure and the table view.

    Known partners are ordered most-connected first and capped at max_known.
    `keep(n)` can hide nodes (for example non-small-molecule drug entries);
    hidden known partners are counted in known_hidden.
    """
    keep = keep or (lambda n: True)
    all_known = sorted(G.neighbors(node), key=lambda n: (-G.degree(n), n))
    known = [k for k in all_known if keep(k)]
    known_shown = known[:max_known]
    known_set = set(all_known)

    related = {}
    for k in known_shown:
        for other in G.neighbors(k):
            if other != node and keep(other):
                related.setdefault(other, set()).add(k)
    related_shown = sorted(related, key=lambda o: (-len(related[o]), -G.degree(o), o))[:max_related]

    predicted = [(p, s) for p, s in sorted(scores.items(), key=lambda x: rank_key(G, x))
                 if s >= min_score and p not in known_set][:max_predicted]

    edges = [{"source": node, "target": k, "kind": "known"} for k in known_shown]
    edges += [{"source": o, "target": k, "kind": "shared"}
              for o in related_shown for k in sorted(related[o])]
    edges += [{"source": node, "target": p, "kind": "predicted", "score": float(s)}
              for p, s in predicted]
    return {
        "center": node,
        "known": known_shown,
        "known_total": len(all_known),
        "known_hidden": len(all_known) - len(known),
        "related": related_shown,
        "related_shared": {o: sorted(related[o]) for o in related_shown},
        "predicted": predicted,
        "edges": edges,
    }


def radial_layout(nb):
    """Deterministic positions: known partners on an inner ring, predicted
    partners spread over the right side, related same-type nodes over the left
    side, ordered by where their shared partners sit."""
    pos = {nb["center"]: (0.0, 0.0)}
    known = nb["known"]
    angle = {}
    for i, k in enumerate(known):
        a = 2 * math.pi * i / max(len(known), 1) + math.pi / 2
        angle[k] = a
        pos[k] = (math.cos(a), math.sin(a))

    def spread(n, lo, hi):
        if n == 1:
            return [(lo + hi) / 2]
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    predicted = [p for p, _ in nb["predicted"]]
    for p, a in zip(predicted, spread(len(predicted), math.radians(55), math.radians(-55))):
        pos[p] = (2.0 * math.cos(a), 2.0 * math.sin(a))

    def mean_angle(o):
        shared = nb["related_shared"][o]
        a = math.atan2(sum(math.sin(angle[k]) for k in shared),
                       sum(math.cos(angle[k]) for k in shared))
        return a % (2 * math.pi)

    related = sorted(nb["related"], key=lambda o: (mean_angle(o), o))
    for o, a in zip(related, spread(len(related), math.radians(105), math.radians(255))):
        pos[o] = (2.2 * math.cos(a), 2.2 * math.sin(a))
    return pos


def edge_table(G, nb):
    """Rows for "view as table"."""
    rows = []
    for e in nb["edges"]:
        rows.append({
            "from": G.nodes[e["source"]].get("name") or e["source"],
            "to": G.nodes[e["target"]].get("name") or e["target"],
            "relation": {"known": "known interaction", "shared": "shares this partner",
                         "predicted": "predicted link"}[e["kind"]],
            "score": f"{e['score']:.2f}" if "score" in e else "",
        })
    return pd.DataFrame(rows, columns=["from", "to", "relation", "score"])
