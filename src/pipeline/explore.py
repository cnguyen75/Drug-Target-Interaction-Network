"""Data behind the explore view: featured drugs, neighbourhoods, layout and evidence text."""

from collections.abc import Callable, Iterable, Mapping
import math
import re
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import DataStructs


# Carbon atom in SMILES: aromatic c, or C not starting a two-letter element (Cl, Ca, Cu, ...).
_CARBON = re.compile(r"c|C(?![ladeflmnorsu])")

Label = Callable[[str], str]

if TYPE_CHECKING:
    from pipeline.attributes import AttributeLinkPredictor


def has_carbon(smiles: str) -> bool:
    """True if a SMILES string contains a carbon atom (bracket atoms reduced to their element)."""
    atoms = re.sub(r"\[\d*([A-Za-z][a-z]?)[^\]]*\]", r"\1", smiles)
    return bool(_CARBON.search(atoms))


def featured_mask(drugs_df: pd.DataFrame) -> pd.Series:
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


def _names(G: nx.Graph, nodes: Iterable[str], label: Label | None = None) -> list[str]:
    return [label(n) if label else (G.nodes[n].get("name") or n) for n in nodes]


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


STRUCTURE_CUTOFF = 0.5  # Tanimoto; fixed before looking at any suggestion
NO_EVIDENCE = "No supporting evidence above threshold; ranked on model score alone."


def _shared_partner_line(
    G: nx.Graph, node: str, partner: str, name: str, max_names: int, label: Label | None
) -> str | None:
    is_drug = G.nodes[node]["node_type"] == "drug"
    node_partners = set(G.neighbors(node))
    support = []
    for other in G.neighbors(partner):
        shared = len(node_partners & set(G.neighbors(other)))
        if other != node and shared:
            support.append((shared, other))
    if not support:
        return None
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


def _attribute_lines(
    G: nx.Graph,
    attr: "AttributeLinkPredictor",
    drug: str,
    target: str,
    node_is_drug: bool,
    label: Label | None,
) -> dict[str, str | None]:
    """Candidate evidence lines: {"structure", "family", "class"} (each a sentence or None)."""
    lab = label or (lambda n: G.nodes[n].get("name") or n)
    it = "it" if node_is_drug else lab(target)
    binders = [b for b in G.neighbors(target) if b != drug]
    lines = {"structure": None, "family": None, "class": None}
    if drug in attr.fps:
        with_fp = [b for b in binders if b in attr.fps]
        if with_fp:
            sims = DataStructs.BulkTanimotoSimilarity(
                attr.fps[drug], [attr.fps[b] for b in with_fp]
            )
            i = int(np.argmax(sims))
            if sims[i] >= STRUCTURE_CUTOFF:
                lines["structure"] = (
                    f"Structurally similar to {lab(with_fp[i])} "
                    f"(Tanimoto {sims[i]:.2f}), which binds {it}."
                )
    if drug in attr.fps and lines["structure"] is None:
        best = (0.0, None, None)
        for f in sorted(attr.pfams.get(target, ())):
            for t2 in sorted(attr.pfam_targets.get(f, ())):
                if t2 == target or t2 not in G:
                    continue
                cands = [b for b in G.neighbors(t2) if b != drug and b in attr.fps]
                if not cands:
                    continue
                sims = DataStructs.BulkTanimotoSimilarity(
                    attr.fps[drug], [attr.fps[b] for b in cands]
                )
                j = int(np.argmax(sims))
                if sims[j] > best[0]:
                    best = (sims[j], cands[j], t2)
        if best[0] >= STRUCTURE_CUTOFF:
            lines["family"] = (
                f"Similar to {lab(best[1])} (Tanimoto {best[0]:.2f}), which binds "
                f"{lab(best[2])} in the same protein family."
            )
    cls = attr.chem_class.get(drug, "")
    if cls and binders:
        n = sum(attr.chem_class.get(b, "") == cls for b in binders)
        if n:
            whose = "its" if node_is_drug else "the drug's"
            verb = "shares" if n == 1 else "share"
            lines["class"] = (
                f"{n} of {len(binders)} known binders {verb} {whose} chemical class ({cls})."
            )
    return lines


def evidence(
    G: nx.Graph,
    node: str,
    partner: str,
    emb: pd.DataFrame | None = None,
    max_names: int = 2,
    label: Label | None = None,
    attr: "AttributeLinkPredictor | None" = None,
) -> str:
    """One sentence of support in the data for a suggested link node-partner.

    Uses the full known graph. With `attr` (an AttributeLinkPredictor), the
    order is: structural similarity (Tanimoto >= 0.5) to a known binder of the
    target; drugs that bind the target and share a target with the drug;
    similarity to a binder of a target in the same Pfam family; binders in the
    same chemical class. Without `attr`, shared-target drugs, then embedding
    neighbours. Otherwise it says there is no such evidence. `label` maps a
    node id to the name shown (for example a target's gene symbol).
    """
    is_drug = G.nodes[node]["node_type"] == "drug"
    name = label(node) if label else (G.nodes[node].get("name") or node)
    drug, target = (node, partner) if is_drug else (partner, node)

    lines = {"structure": None, "family": None, "class": None}
    if attr is not None:
        lines = _attribute_lines(G, attr, drug, target, is_drug, label)
        if lines["structure"]:
            return lines["structure"]

    shared = _shared_partner_line(G, node, partner, name, max_names, label)
    if shared:
        return shared
    if lines["family"] or lines["class"]:
        return lines["family"] or lines["class"]

    if attr is None and emb is not None and node in emb.index:
        same_type = [
            n
            for n in emb.index
            if n in G and G.nodes[n]["node_type"] == G.nodes[node]["node_type"] and n != node
        ]
        vecs = emb.loc[same_type].values
        v = emb.loc[node].values
        sims = vecs @ v / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(v) + 1e-12)
        nearest = [same_type[i] for i in np.argsort(-sims)[:25]]
        linked = [n for n in nearest if G.has_edge(n, partner)][:max_names]
        if linked:
            kind = "drugs" if is_drug else "targets"
            return f"Among its 25 nearest {kind} in the embedding, {_join(_names(G, linked, label))} link to it."
    if attr is not None:
        return NO_EVIDENCE
    return "No shared-target path in the data; the score comes from degree and embedding position."


def rank_key(G: nx.Graph, item: tuple[str, float]) -> tuple[float, int, str]:
    """Suggestion order used everywhere: score, then rarer partner, then id."""
    partner, score = item
    return (-round(float(score), 6), G.degree(partner), partner)


def neighbourhood(
    G: nx.Graph,
    node: str,
    scores: Mapping[str, float],
    min_score: float,
    max_known: int = 24,
    max_related: int = 8,
    max_predicted: int = 5,
    keep: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
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

    predicted = [
        (p, s)
        for p, s in sorted(scores.items(), key=lambda x: rank_key(G, x))
        if s >= min_score and p not in known_set
    ][:max_predicted]

    edges = [{"source": node, "target": k, "kind": "known"} for k in known_shown]
    edges += [
        {"source": o, "target": k, "kind": "shared"}
        for o in related_shown
        for k in sorted(related[o])
    ]
    edges += [
        {"source": node, "target": p, "kind": "predicted", "score": float(s)} for p, s in predicted
    ]
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


def radial_layout(nb: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Deterministic positions: known partners on an inner ring, predicted
    partners spread over the right side, related same-type nodes over the left
    side, ordered by where their shared partners sit."""
    pos = {nb["center"]: (0.0, 0.0)}
    known = nb["known"]
    angle = {}
    for i, k in enumerate(known):
        a = 2 * math.pi * i / max(len(known), 1) + math.pi / 2
        angle[k] = a
        pos[k] = (1.3 * math.cos(a), 1.3 * math.sin(a))

    def spread(n: int, lo: float, hi: float) -> list[float]:
        if n == 1:
            return [(lo + hi) / 2]
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    predicted = [p for p, _ in nb["predicted"]]
    for p, a in zip(predicted, spread(len(predicted), math.radians(55), math.radians(-55))):
        pos[p] = (2.25 * math.cos(a), 2.25 * math.sin(a))

    def mean_angle(o: str) -> float:
        shared = nb["related_shared"][o]
        a = math.atan2(
            sum(math.sin(angle[k]) for k in shared), sum(math.cos(angle[k]) for k in shared)
        )
        return a % (2 * math.pi)

    related = sorted(nb["related"], key=lambda o: (mean_angle(o), o))
    for o, a in zip(related, spread(len(related), math.radians(105), math.radians(255))):
        pos[o] = (2.4 * math.cos(a), 2.4 * math.sin(a))
    return pos


def edge_table(G: nx.Graph, nb: dict[str, Any]) -> pd.DataFrame:
    """Rows for "view as table"."""
    rows = []
    for e in nb["edges"]:
        rows.append(
            {
                "from": G.nodes[e["source"]].get("name") or e["source"],
                "to": G.nodes[e["target"]].get("name") or e["target"],
                "relation": {
                    "known": "known interaction",
                    "shared": "shares this partner",
                    "predicted": "predicted link",
                }[e["kind"]],
                "score": f"{e['score']:.2f}" if "score" in e else "",
            }
        )
    return pd.DataFrame(rows, columns=["from", "to", "relation", "score"])
