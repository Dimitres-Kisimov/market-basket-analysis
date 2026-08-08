"""Category affinity network: co-purchase communities via greedy modularity.

The rule list answers "given X in the basket, what should I offer next?". A
CATEGORY MANAGER asks a different question: "which categories belong together
as a group -- for bundles, planogram adjacency and the promo calendar?" This
module answers it by turning the frequent 2-itemsets that Apriori already
mined into an undirected CATEGORY AFFINITY GRAPH and partitioning it into
communities:

* every frequent category pair with lift >= ``min_lift`` becomes an edge,
  weighted by its EXCESS co-purchase over independence (``lift - 1``);
* communities are found by greedy agglomerative modularity maximisation
  (Newman 2004): start with every category alone, repeatedly perform the
  merge that raises weighted modularity the most, stop when no merge helps;
* edges that end up BETWEEN two communities are reported as *bridges* --
  category pairs that link otherwise-separate assortment groups, i.e.
  cross-merchandising candidates rather than within-group bundles.

Everything reuses the shared pipeline (the mined itemsets and the same lift
definition as :mod:`basket.rules`) and is fully deterministic: no RNG, no
wall clock, lexicographic tie-breaks everywhere. Honesty notes, repeated in
the deliverables: the data is SYNTHETIC and seeded, the communities describe
observed co-purchase structure (correlation, not causation), and modularity
is a heuristic grouping -- a different objective could group differently.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AffinityEdge:
    """An undirected co-purchase edge between two categories."""

    item_a: str  # lexicographically first endpoint
    item_b: str
    support: float  # fraction of baskets containing both categories
    support_count: int  # absolute number of baskets behind the edge
    lift: float  # support / (P(a) * P(b)), same definition as basket.rules
    leverage: float  # support - P(a) * P(b)
    weight: float  # lift - 1.0: excess co-purchase used as the modularity weight

    @property
    def label(self) -> str:
        return f"{self.item_a} -- {self.item_b}"


@dataclass(frozen=True)
class Community:
    """One detected community with its internal-edge statistics."""

    community_id: int
    members: tuple[str, ...]  # sorted category names
    n_members: int
    n_internal_edges: int
    lift_mean: float  # mean lift over internal edges (0.0 when there are none)
    lift_max: float  # max lift over internal edges (0.0 when there are none)
    top_edge: AffinityEdge | None  # highest-lift internal edge, None for singletons

    @property
    def label(self) -> str:
        return " + ".join(self.members)


@dataclass
class AffinityReport:
    """The affinity-network read-out plus the settings behind it."""

    n_baskets: int
    min_support: float
    min_lift: float
    n_nodes: int
    n_edges: int
    modularity: float  # weighted modularity Q of the final partition
    communities: list[Community] = field(default_factory=list)
    edges: list[AffinityEdge] = field(default_factory=list)  # sorted by lift desc
    membership: dict[str, int] = field(default_factory=dict)  # category -> community_id
    bridges: list[AffinityEdge] = field(default_factory=list)  # cross-community edges

    @property
    def n_grouped(self) -> int:
        """Communities with at least two members."""
        return sum(1 for c in self.communities if c.n_members >= 2)

    @property
    def isolated(self) -> tuple[str, ...]:
        """Categories with no qualifying edge (their own singleton community)."""
        return tuple(c.members[0] for c in self.communities if c.n_members == 1)

    def plain_language(self) -> str:
        """A one-paragraph, jargon-light summary of the network."""
        lead = (
            f"The {self.n_nodes} categories are connected by {self.n_edges} "
            f"co-purchase edges (pair support >= {self.min_support:.0%}, lift >= "
            f"{self.min_lift:.2f}); greedy modularity grouping finds "
            f"{self.n_grouped} communities (weighted modularity "
            f"{self.modularity:.2f})."
        )
        if self.isolated:
            lead += (
                f" {len(self.isolated)} categories have no qualifying edge and stay "
                f"ungrouped: {', '.join(self.isolated)}."
            )
        if self.bridges:
            top = self.bridges[0]
            lead += (
                f" The strongest BRIDGE between communities is {top.label} "
                f"(lift {top.lift:.2f}) -- a cross-merchandising candidate, not a "
                "within-group bundle."
            )
        lead += (
            " Communities describe observed co-purchase structure on synthetic "
            "data -- correlation, not causation."
        )
        return lead


def build_affinity_edges(
    itemsets: Mapping[frozenset[str], float],
    n_baskets: int,
    min_lift: float = 1.10,
) -> list[AffinityEdge]:
    """Turn mined frequent pairs into affinity edges (lift filtered, lift ranked).

    ``itemsets`` is the output of :func:`basket.apriori.apriori` (or
    ``fpgrowth`` -- they are tested equal), so every pair here already clears
    the run's minimum support. Only pairs with ``lift >= min_lift`` become
    edges; ``min_lift`` must be at least 1.0 so every edge weight
    (``lift - 1``) is non-negative, which weighted modularity requires.
    """
    if n_baskets < 1:
        raise ValueError("n_baskets must be positive")
    if min_lift < 1.0:
        raise ValueError("min_lift must be at least 1.0 (edges must be positive affinity)")
    singles = {
        next(iter(itemset)): support
        for itemset, support in itemsets.items()
        if len(itemset) == 1
    }
    edges: list[AffinityEdge] = []
    for itemset, support in itemsets.items():
        if len(itemset) != 2:
            continue
        item_a, item_b = sorted(itemset)
        support_a = singles.get(item_a)
        support_b = singles.get(item_b)
        if support_a is None or support_b is None:
            # Cannot happen for complete downward-closed input; guard anyway.
            continue
        lift = support / (support_a * support_b)
        if lift < min_lift:
            continue
        edges.append(
            AffinityEdge(
                item_a=item_a,
                item_b=item_b,
                support=support,
                support_count=round(support * n_baskets),
                lift=lift,
                leverage=support - support_a * support_b,
                weight=lift - 1.0,
            )
        )
    edges.sort(key=lambda e: (-e.lift, e.item_a, e.item_b))
    return edges


def modularity(
    communities: Sequence[Collection[str]], edges: Sequence[AffinityEdge]
) -> float:
    """Weighted modularity Q of a partition (Newman & Girvan 2004).

    ``Q = sum_c [ W_c / m - (K_c / 2m)^2 ]`` where ``m`` is the total edge
    weight, ``W_c`` the weight inside community ``c`` (each edge once) and
    ``K_c`` the summed weighted degree of its nodes. Defined as 0.0 when the
    graph carries no weight. Communities must be disjoint and cover every
    edge endpoint.
    """
    node_community: dict[str, int] = {}
    for index, community in enumerate(communities):
        for node in community:
            if node in node_community:
                raise ValueError(f"node {node!r} appears in more than one community")
            node_community[node] = index
    total_weight = sum(edge.weight for edge in edges)
    if total_weight <= 0.0:
        return 0.0
    degree: dict[str, float] = defaultdict(float)
    internal: dict[int, float] = defaultdict(float)
    for edge in edges:
        if edge.item_a not in node_community or edge.item_b not in node_community:
            raise ValueError(f"edge {edge.label} has an endpoint outside the partition")
        degree[edge.item_a] += edge.weight
        degree[edge.item_b] += edge.weight
        community_a = node_community[edge.item_a]
        if community_a == node_community[edge.item_b]:
            internal[community_a] += edge.weight
    q = 0.0
    for index, community in enumerate(communities):
        k_c = sum(degree.get(node, 0.0) for node in community)
        q += internal.get(index, 0.0) / total_weight - (k_c / (2.0 * total_weight)) ** 2
    return q


def detect_communities(
    nodes: Collection[str],
    edges: Sequence[AffinityEdge],
    tol: float = 1e-12,
) -> list[tuple[str, ...]]:
    """Greedy agglomerative modularity maximisation (Newman 2004), deterministic.

    Start with every node in its own community; at each step evaluate every
    merge of two edge-connected communities and perform the one that increases
    modularity the most (candidates are scanned in lexicographic order, so
    equal gains resolve to the lexicographically first pair); stop when no
    merge improves modularity by more than ``tol``. Nodes without edges stay
    singleton communities. Returns communities as sorted member tuples,
    largest first, then lexicographic.
    """
    all_nodes = sorted(
        set(nodes) | {e.item_a for e in edges} | {e.item_b for e in edges}
    )
    communities: list[frozenset[str]] = [frozenset((node,)) for node in all_nodes]
    while True:
        ordered = sorted(communities, key=lambda c: tuple(sorted(c)))
        index_of = {node: i for i, community in enumerate(ordered) for node in community}
        current_q = modularity(ordered, edges)
        connected: set[tuple[int, int]] = set()
        for edge in edges:
            index_a, index_b = index_of[edge.item_a], index_of[edge.item_b]
            if index_a != index_b:
                connected.add((min(index_a, index_b), max(index_a, index_b)))
        best_pair: tuple[int, int] | None = None
        best_q = current_q
        for index_a, index_b in sorted(connected):
            trial = [c for i, c in enumerate(ordered) if i not in (index_a, index_b)]
            trial.append(ordered[index_a] | ordered[index_b])
            trial_q = modularity(trial, edges)
            if trial_q > best_q + tol:
                best_q = trial_q
                best_pair = (index_a, index_b)
        if best_pair is None:
            break
        index_a, index_b = best_pair
        merged = ordered[index_a] | ordered[index_b]
        communities = [c for i, c in enumerate(ordered) if i not in (index_a, index_b)]
        communities.append(merged)
    return sorted(
        (tuple(sorted(community)) for community in communities),
        key=lambda members: (-len(members), members),
    )


def affinity_network(
    itemsets: Mapping[frozenset[str], float],
    n_baskets: int,
    *,
    min_support: float = 0.02,
    min_lift: float = 1.10,
) -> AffinityReport:
    """Build the affinity graph from mined itemsets and detect its communities.

    ``min_support`` is reported, not applied here -- the itemsets already
    embody it (they come from the shared Apriori/FP-growth pass). Fully
    deterministic for a given input.
    """
    edges = build_affinity_edges(itemsets, n_baskets, min_lift=min_lift)
    nodes = sorted(
        item for itemset in itemsets if len(itemset) == 1 for item in itemset
    )
    community_sets = detect_communities(nodes, edges)
    q = modularity(community_sets, edges)
    membership = {
        node: community_id
        for community_id, members in enumerate(community_sets)
        for node in members
    }
    communities: list[Community] = []
    for community_id, members in enumerate(community_sets):
        internal = [
            edge
            for edge in edges
            if membership[edge.item_a] == community_id
            and membership[edge.item_b] == community_id
        ]
        lifts = [edge.lift for edge in internal]
        communities.append(
            Community(
                community_id=community_id,
                members=members,
                n_members=len(members),
                n_internal_edges=len(internal),
                lift_mean=sum(lifts) / len(lifts) if lifts else 0.0,
                lift_max=max(lifts) if lifts else 0.0,
                top_edge=internal[0] if internal else None,  # edges pre-sorted by lift
            )
        )
    bridges = [
        edge for edge in edges if membership[edge.item_a] != membership[edge.item_b]
    ]
    return AffinityReport(
        n_baskets=n_baskets,
        min_support=min_support,
        min_lift=min_lift,
        n_nodes=len(nodes),
        n_edges=len(edges),
        modularity=q,
        communities=communities,
        edges=edges,
        membership=membership,
        bridges=bridges,
    )


# --------------------------------------------------------------------------- #
# Deterministic outputs: CSV + a hand-drawn SVG (no wall-clock, no RNG).
# --------------------------------------------------------------------------- #

_CSV_HEADER = (
    "item_a",
    "item_b",
    "support",
    "support_count",
    "lift",
    "leverage",
    "weight",
    "community_a",
    "community_b",
    "edge_type",
)


def write_affinity_csv(report: AffinityReport, path: str) -> int:
    """Write the edge list (with community assignments) as deterministic CSV."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_CSV_HEADER)
        for edge in report.edges:
            community_a = report.membership[edge.item_a]
            community_b = report.membership[edge.item_b]
            writer.writerow(
                [
                    edge.item_a,
                    edge.item_b,
                    f"{edge.support:.4f}",
                    edge.support_count,
                    f"{edge.lift:.4f}",
                    f"{edge.leverage:.4f}",
                    f"{edge.weight:.4f}",
                    community_a,
                    community_b,
                    "internal" if community_a == community_b else "bridge",
                ]
            )
    return os.path.getsize(path)


# SVG palette (matches the deliverables' light surface / other charts).
_SVG_INK = "#0b0b0b"
_SVG_INK_SECONDARY = "#52514e"
_SVG_INK_MUTED = "#898781"
_SVG_SURFACE = "#fcfcfb"
_SVG_PANEL_BORDER = "#e1e0d9"
_SVG_BASELINE = "#c3c2b7"
_SVG_COMMUNITY_COLORS = ("#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834")


def _svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pretty(category: str) -> str:
    return category.replace("_", " ")


def _circle_positions(
    n: int, center_x: float, center_y: float, radius: float
) -> list[tuple[float, float, float]]:
    """(x, y, cos_angle) per node, evenly spaced clockwise starting at the top."""
    positions: list[tuple[float, float, float]] = []
    for k in range(n):
        angle = -math.pi / 2.0 + 2.0 * math.pi * k / n
        positions.append(
            (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
                math.cos(angle),
            )
        )
    return positions


def render_affinity_svg(report: AffinityReport) -> str:
    """Build a deterministic, self-contained node-link SVG of the network.

    One panel per multi-member community (nodes on a circle, edge width scaled
    by excess lift), isolated categories listed as text, then the top bridges
    between communities. No timestamps or randomness, so the bytes are
    identical on every run.
    """
    width = 900
    margin = 36
    panel_w = 402
    panel_h = 264
    panel_gap = 24
    columns = 2
    top_pad = 96

    panels = [c for c in report.communities if c.n_members >= 2]
    n_rows = (len(panels) + columns - 1) // columns
    panels_bottom = top_pad + n_rows * (panel_h + panel_gap)
    isolated_h = 22 if report.isolated else 0
    bridges_shown = report.bridges[:6]
    bridges_h = 30 + 18 * len(bridges_shown) if bridges_shown else 30
    height = panels_bottom + isolated_h + bridges_h + 78

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="{_SVG_SURFACE}"/>')
    parts.append(
        f'<text x="{margin}" y="40" font-size="20" font-weight="bold" '
        f'fill="{_SVG_INK}">Category affinity network: {report.n_grouped} co-purchase '
        f"communities</text>"
    )
    parts.append(
        f'<text x="{margin}" y="64" font-size="12.5" fill="{_SVG_INK_SECONDARY}">'
        f"{report.n_nodes} categories, {report.n_edges} edges (pair support &gt;= "
        f"{report.min_support:.0%}, lift &gt;= {report.min_lift:.2f}); greedy "
        f"modularity Q = {report.modularity:.2f}. Edge width = excess lift.</text>"
    )

    for panel_index, community in enumerate(panels):
        row, column = divmod(panel_index, columns)
        panel_x = margin + column * (panel_w + panel_gap)
        panel_y = top_pad + row * (panel_h + panel_gap)
        color = _SVG_COMMUNITY_COLORS[community.community_id % len(_SVG_COMMUNITY_COLORS)]
        parts.append(
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" '
            f'rx="8" fill="none" stroke="{_SVG_PANEL_BORDER}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{panel_x + 14}" y="{panel_y + 24}" font-size="12.5" '
            f'font-weight="bold" fill="{_SVG_INK}">Community '
            f"{community.community_id} &#183; {community.n_members} categories</text>"
        )
        parts.append(
            f'<text x="{panel_x + 14}" y="{panel_y + 42}" font-size="10.5" '
            f'fill="{_SVG_INK_MUTED}">{community.n_internal_edges} internal edges, '
            f"avg lift {community.lift_mean:.2f}, max {community.lift_max:.2f}</text>"
        )
        center_x = panel_x + panel_w / 2.0
        center_y = panel_y + 42 + (panel_h - 42) / 2.0
        radius = 62.0
        members = community.members
        positions = _circle_positions(len(members), center_x, center_y, radius)
        position_of = {member: positions[i] for i, member in enumerate(members)}
        internal = [
            e
            for e in report.edges
            if report.membership[e.item_a] == community.community_id
            and report.membership[e.item_b] == community.community_id
        ]
        for edge in internal:
            x1, y1, _ = position_of[edge.item_a]
            x2, y2, _ = position_of[edge.item_b]
            stroke_w = 1.0 + 2.0 * min(1.0, edge.weight)
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{stroke_w:.1f}" stroke-opacity="0.45"/>'
            )
        for member in members:
            x, y, cos_angle = position_of[member]
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            if cos_angle > 0.35:
                anchor, dx = "start", 9.0
            elif cos_angle < -0.35:
                anchor, dx = "end", -9.0
            else:
                anchor, dx = "middle", 0.0
            dy = 3.5 if abs(cos_angle) > 0.35 else (-9.0 if y < center_y else 15.0)
            parts.append(
                f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="10" '
                f'text-anchor="{anchor}" fill="{_SVG_INK}">'
                f"{_svg_escape(_pretty(member))}</text>"
            )

    cursor_y = panels_bottom
    if report.isolated:
        names = ", ".join(_pretty(item) for item in report.isolated)
        parts.append(
            f'<text x="{margin}" y="{cursor_y + 4}" font-size="11" '
            f'fill="{_SVG_INK_SECONDARY}">No qualifying edges (ungrouped): '
            f"{_svg_escape(names)}</text>"
        )
        cursor_y += isolated_h

    parts.append(
        f'<text x="{margin}" y="{cursor_y + 16}" font-size="12.5" font-weight="bold" '
        f'fill="{_SVG_INK}">Bridges between communities (cross-merchandising '
        f"candidates)</text>"
    )
    if bridges_shown:
        for i, edge in enumerate(bridges_shown):
            line_y = cursor_y + 36 + 18 * i
            community_a = report.membership[edge.item_a]
            community_b = report.membership[edge.item_b]
            parts.append(
                f'<text x="{margin}" y="{line_y}" font-size="11" '
                f'fill="{_SVG_INK_SECONDARY}">{_svg_escape(_pretty(edge.item_a))} '
                f"&#8212; {_svg_escape(_pretty(edge.item_b))} &#183; lift "
                f"{edge.lift:.2f} &#183; {edge.support:.1%} of orders "
                f"({edge.support_count}) &#183; communities {community_a} &#8596; "
                f"{community_b}</text>"
            )
        cursor_y += 36 + 18 * len(bridges_shown)
    else:
        parts.append(
            f'<text x="{margin}" y="{cursor_y + 36}" font-size="11" '
            f'fill="{_SVG_INK_SECONDARY}">None at these thresholds.</text>'
        )
        cursor_y += 36 + 6

    parts.append(
        f'<line x1="{margin}" y1="{cursor_y + 12}" x2="{width - margin}" '
        f'y2="{cursor_y + 12}" stroke="{_SVG_BASELINE}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{margin}" y="{cursor_y + 34}" font-size="10" '
        f'fill="{_SVG_INK_MUTED}">SYNTHETIC DATA: all figures come from a seeded '
        f"simulation. Communities are observed co-purchase structure -- correlation, "
        f"not causation; modularity is a heuristic grouping.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_affinity_svg(report: AffinityReport, path: str) -> int:
    """Write the hand-drawn network SVG; returns the file size in bytes."""
    svg = render_affinity_svg(report)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(svg)
        handle.write("\n")
    return os.path.getsize(path)
