import pytest

from basket.affinity import (
    AffinityEdge,
    affinity_network,
    build_affinity_edges,
    detect_communities,
    modularity,
    render_affinity_svg,
    write_affinity_csv,
    write_affinity_svg,
)
from basket.apriori import apriori
from basket.data import PLANTED_BUNDLES


def _edge(a: str, b: str, weight: float = 1.0) -> AffinityEdge:
    """A synthetic edge for graph-algorithm tests (metrics fields are dummies)."""
    item_a, item_b = sorted((a, b))
    return AffinityEdge(
        item_a=item_a,
        item_b=item_b,
        support=0.1,
        support_count=10,
        lift=1.0 + weight,
        leverage=0.05,
        weight=weight,
    )


def _two_triangles() -> tuple[list[str], list[AffinityEdge]]:
    """Two unit-weight triangles a-b-c and d-e-f joined by one bridge c-d."""
    nodes = ["a", "b", "c", "d", "e", "f"]
    edges = [
        _edge("a", "b"), _edge("a", "c"), _edge("b", "c"),
        _edge("d", "e"), _edge("d", "f"), _edge("e", "f"),
        _edge("c", "d"),
    ]
    return nodes, edges


def test_edge_metrics_hand_checked(tiny_baskets):
    """All edge metrics verified by hand on the five-basket example.

    milk 4/5, bread 4/5, butter 1/5; milk+bread 3/5 -> lift 0.9375 (excluded);
    milk+butter and bread+butter 1/5 -> lift 0.2 / (0.8 * 0.2) = 1.25.
    """
    itemsets = apriori(tiny_baskets, min_support=0.2, max_len=2)
    edges = build_affinity_edges(itemsets, len(tiny_baskets), min_lift=1.10)
    assert [(e.item_a, e.item_b) for e in edges] == [
        ("bread", "butter"), ("butter", "milk")
    ]
    for edge in edges:
        assert edge.support == pytest.approx(0.2)
        assert edge.support_count == 1
        assert edge.lift == pytest.approx(1.25)
        assert edge.leverage == pytest.approx(0.2 - 0.8 * 0.2)
        assert edge.weight == pytest.approx(0.25)


def test_edge_builder_validates_inputs(tiny_baskets):
    itemsets = apriori(tiny_baskets, min_support=0.2, max_len=2)
    with pytest.raises(ValueError):
        build_affinity_edges(itemsets, len(tiny_baskets), min_lift=0.9)
    with pytest.raises(ValueError):
        build_affinity_edges(itemsets, 0)


def test_modularity_hand_checked():
    """Q verified by hand on the two-triangle graph (m = 7, unit weights).

    Two-triangle partition: Q = 2 * (3/7 - (7/14)^2) = 5/14. All-in-one: 0.
    Singletons: -(4*4 + 2*9)/196 = -17/98.
    """
    nodes, edges = _two_triangles()
    triangles = [("a", "b", "c"), ("d", "e", "f")]
    assert modularity(triangles, edges) == pytest.approx(5 / 14)
    assert modularity([tuple(nodes)], edges) == pytest.approx(0.0)
    singletons = [(n,) for n in nodes]
    assert modularity(singletons, edges) == pytest.approx(-17 / 98)


def test_modularity_validates_partition():
    _, edges = _two_triangles()
    with pytest.raises(ValueError):
        modularity([("a", "b"), ("b", "c")], edges)  # overlapping communities
    with pytest.raises(ValueError):
        modularity([("a", "b", "c")], edges)  # edge endpoints outside partition


def test_modularity_empty_graph_is_zero():
    assert modularity([("a",), ("b",)], []) == 0.0


def test_detect_communities_recovers_two_triangles():
    nodes, edges = _two_triangles()
    communities = detect_communities(nodes, edges)
    assert communities == [("a", "b", "c"), ("d", "e", "f")]
    assert modularity(communities, edges) == pytest.approx(5 / 14)


def test_detect_communities_collapses_to_base_cases():
    # No edges: every node stays its own singleton community.
    assert detect_communities(["x", "y", "z"], []) == [("x",), ("y",), ("z",)]
    # A single edge: the pair merges (Q goes from -0.5 to 0.0).
    assert detect_communities(["x", "y"], [_edge("x", "y")]) == [("x", "y")]
    # Nodes only present as edge endpoints are picked up too.
    assert detect_communities([], [_edge("x", "y")]) == [("x", "y")]


def test_network_on_tiny_baskets_hand_checked(tiny_baskets):
    """The butter chain merges into one community of all three categories.

    Greedy trace by hand (m = 0.5): singletons Q = -0.375, merge bread+butter
    -> Q = -0.125, merge in milk -> Q = 0.0. No bridges remain.
    """
    itemsets = apriori(tiny_baskets, min_support=0.2, max_len=2)
    report = affinity_network(itemsets, len(tiny_baskets), min_support=0.2, min_lift=1.10)
    assert report.n_nodes == 3
    assert report.n_edges == 2
    assert [c.members for c in report.communities] == [("bread", "butter", "milk")]
    assert report.modularity == pytest.approx(0.0)
    assert report.bridges == []
    assert report.isolated == ()


def test_planted_bundles_land_in_one_community_each(baskets):
    """Ground truth: every planted bundle's members share a community."""
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    report = affinity_network(itemsets, len(baskets), min_lift=1.10)
    for bundle in PLANTED_BUNDLES:
        community_ids = {report.membership[item] for item in bundle}
        assert len(community_ids) == 1, f"bundle {bundle} split across {community_ids}"


def test_network_structure_is_consistent(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    report = affinity_network(itemsets, len(baskets), min_lift=1.10)
    # Membership is a partition of exactly the frequent single categories.
    frequent_singles = {next(iter(s)) for s in itemsets if len(s) == 1}
    assert set(report.membership) == frequent_singles
    members_union = [m for c in report.communities for m in c.members]
    assert sorted(members_union) == sorted(frequent_singles)
    assert len(members_union) == len(set(members_union))
    # Communities are ordered largest-first and ids match list positions.
    sizes = [c.n_members for c in report.communities]
    assert sizes == sorted(sizes, reverse=True)
    assert [c.community_id for c in report.communities] == list(range(len(sizes)))
    # Every edge is internal xor bridge, consistently with membership.
    bridge_set = {(e.item_a, e.item_b) for e in report.bridges}
    for edge in report.edges:
        is_bridge = report.membership[edge.item_a] != report.membership[edge.item_b]
        assert ((edge.item_a, edge.item_b) in bridge_set) == is_bridge
    # Per-community stats agree with the edge list.
    for community in report.communities:
        internal = [
            e for e in report.edges
            if report.membership[e.item_a] == community.community_id
            and report.membership[e.item_b] == community.community_id
        ]
        assert community.n_internal_edges == len(internal)
        if internal:
            assert community.lift_max == pytest.approx(max(e.lift for e in internal))
            assert community.top_edge == internal[0]
        else:
            assert community.top_edge is None


def test_network_is_deterministic(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    first = affinity_network(itemsets, len(baskets), min_lift=1.10)
    second = affinity_network(itemsets, len(baskets), min_lift=1.10)
    assert first.membership == second.membership
    assert first.edges == second.edges
    assert first.communities == second.communities
    assert first.modularity == second.modularity


def test_plain_language_reports_counts(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    report = affinity_network(itemsets, len(baskets), min_lift=1.10)
    text = report.plain_language()
    assert str(report.n_grouped) in text
    assert str(report.n_edges) in text
    assert "not causation" in text


def test_csv_and_svg_writers_deterministic(baskets, tmp_path):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    report = affinity_network(itemsets, len(baskets), min_lift=1.10)

    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    write_affinity_csv(report, str(csv_a))
    write_affinity_csv(report, str(csv_b))
    assert csv_a.read_bytes() == csv_b.read_bytes()
    lines = csv_a.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "item_a,item_b,support,support_count,lift,leverage,weight,"
        "community_a,community_b,edge_type"
    )
    assert len(lines) == 1 + report.n_edges
    assert sum(1 for line in lines if line.endswith(",bridge")) == len(report.bridges)

    svg = render_affinity_svg(report)
    assert svg.startswith("<svg")
    assert "Category affinity network" in svg
    # The atlas plate header and the design system's non-colour channels.
    assert "CATEGORY ATLAS" in svg and "PLATE 04 / 07" in svg
    assert "stroke-dasharray" in svg  # community hulls (and any bridges) are dashed
    assert render_affinity_svg(report) == svg  # pure function of the report
    svg_path = tmp_path / "n.svg"
    write_affinity_svg(report, str(svg_path))
    assert svg_path.read_text(encoding="utf-8").startswith("<svg")
