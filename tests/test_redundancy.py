import pytest

from basket.apriori import apriori
from basket.redundancy import (
    closed_itemsets,
    maximal_itemsets,
    render_redundancy_svg,
    rule_improvement,
    rule_redundancy,
    write_redundancy_csv,
    write_redundancy_svg,
)
from basket.rules import generate_rules


def _tiny_itemsets(tiny_baskets):
    return apriori(tiny_baskets, min_support=0.2, max_len=3)


def _tiny_rules(tiny_baskets):
    itemsets = _tiny_itemsets(tiny_baskets)
    return itemsets, generate_rules(itemsets, len(tiny_baskets))


def _find(rules, antecedent, consequent):
    key = (frozenset(antecedent), frozenset(consequent))
    return next(r for r in rules if (r.antecedent, r.consequent) == key)


def test_closed_itemsets_hand_checked(tiny_baskets):
    """Closure verified by hand on the five-basket example.

    Counts: milk 4, bread 4, butter 1; milk+bread 3, milk+butter 1,
    bread+butter 1; milk+bread+butter 1. Closed = no mined superset with the
    same count: {milk}, {bread} (their supersets drop to 3 or 1),
    {milk, bread} (triple drops to 1) and the triple itself. NOT closed:
    {butter}, {milk, butter}, {bread, butter} -- each has a superset at count 1.
    """
    itemsets = _tiny_itemsets(tiny_baskets)
    assert len(itemsets) == 7
    closed = closed_itemsets(itemsets, 5)
    assert closed == {
        frozenset({"milk"}),
        frozenset({"bread"}),
        frozenset({"milk", "bread"}),
        frozenset({"milk", "bread", "butter"}),
    }


def test_maximal_itemsets_hand_checked(tiny_baskets):
    """Only the triple has no frequent superset; maximal is a subset of closed."""
    itemsets = _tiny_itemsets(tiny_baskets)
    maximal = maximal_itemsets(itemsets)
    assert maximal == {frozenset({"milk", "bread", "butter"})}
    assert maximal <= closed_itemsets(itemsets, 5)


def test_closed_requires_positive_baskets(tiny_baskets):
    itemsets = _tiny_itemsets(tiny_baskets)
    with pytest.raises(ValueError):
        closed_itemsets(itemsets, 0)
    with pytest.raises(ValueError):
        rule_redundancy(itemsets, [], 0)


def test_improvement_hand_checked_vs_baseline(tiny_baskets):
    """butter -> milk: conf 1.0 vs baseline P(milk) = 0.8 -> improvement 0.2.

    A single-item antecedent has only the empty set as a proper subset, so the
    best alternative is the baseline itself.
    """
    itemsets, rules = _tiny_rules(tiny_baskets)
    rule = _find(rules, {"butter"}, {"milk"})
    improvement, best_antecedent, best_confidence = rule_improvement(rule, itemsets)
    assert improvement == pytest.approx(0.2)
    assert best_antecedent == frozenset()
    assert best_confidence == pytest.approx(0.8)


def test_improvement_hand_checked_tiebreak(tiny_baskets):
    """milk + bread -> butter: conf 1/3; the extra condition genuinely helps.

    Alternatives: baseline P(butter) = 0.2, bread -> butter = 0.25,
    milk -> butter = 0.25. The two singletons tie, so the lexicographically
    first ({bread}) wins deterministically; improvement = 1/3 - 1/4 = 1/12.
    """
    itemsets, rules = _tiny_rules(tiny_baskets)
    rule = _find(rules, {"milk", "bread"}, {"butter"})
    improvement, best_antecedent, best_confidence = rule_improvement(rule, itemsets)
    assert improvement == pytest.approx(1 / 12)
    assert best_antecedent == frozenset({"bread"})
    assert best_confidence == pytest.approx(0.25)


def test_report_hand_checked(tiny_baskets):
    """The full tiny-fixture verdict, traced by hand.

    Six rules survive the default filters; exactly two are redundant, each
    covered at equal confidence (improvement 0.0) by a one-item rule:
    bread + butter -> milk by butter -> milk, and butter + milk -> bread by
    butter -> bread (both conf 1.0). The other four have strictly positive
    improvement (0.4, 1/12, 0.2, 0.2).
    """
    itemsets, rules = _tiny_rules(tiny_baskets)
    assert len(rules) == 6
    report = rule_redundancy(itemsets, rules, len(tiny_baskets))
    assert report.n_itemsets == 7
    assert report.n_closed == 4
    assert report.n_maximal == 1
    assert report.n_rules == 6
    assert report.n_redundant == 2
    assert report.n_kept == 4
    # Verdicts keep the input (lift-ranked) rule order.
    assert [v.rule for v in report.verdicts] == list(rules)
    pruned = {v.label: v for v in report.pruned}
    assert set(pruned) == {"bread + butter -> milk", "butter + milk -> bread"}
    assert pruned["bread + butter -> milk"].best_alternative_label == "butter -> milk"
    assert pruned["butter + milk -> bread"].best_alternative_label == "butter -> bread"
    for verdict in report.pruned:
        assert verdict.improvement == pytest.approx(0.0)
        assert verdict.best_alternative_confidence == pytest.approx(1.0)
    kept_improvements = sorted(v.improvement for v in report.kept)
    assert kept_improvements == pytest.approx(sorted([0.4, 1 / 12, 0.2, 0.2]))


def test_every_pruned_rule_is_covered_by_a_kept_rule(baskets):
    """The pruning guarantee: acting on the kept rules loses nothing.

    For a shared consequent lift is proportional to confidence, so any
    nonempty covering rule also clears the run's filters and must appear in
    the rule list -- and a covering rule is itself never redundant here
    (its own baseline alternative would contradict lift > 1).
    """
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets))
    report = rule_redundancy(itemsets, rules, len(baskets))
    by_key = {(v.rule.antecedent, v.rule.consequent): v for v in report.verdicts}
    assert report.n_redundant > 0  # the big fixture does contain restatements
    for verdict in report.pruned:
        antecedent = verdict.best_alternative_antecedent
        assert antecedent  # never the bare baseline: kept rules have lift > 1
        assert antecedent < verdict.rule.antecedent
        cover = by_key[(antecedent, verdict.rule.consequent)]
        assert not cover.redundant
        assert cover.rule.confidence >= verdict.rule.confidence - 1e-12
        assert cover.rule.confidence == pytest.approx(verdict.best_alternative_confidence)


def test_single_antecedent_rules_are_never_redundant(baskets):
    """lift >= 1.10 means conf > P(consequent), the only simpler alternative."""
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets))
    report = rule_redundancy(itemsets, rules, len(baskets))
    for verdict in report.verdicts:
        if len(verdict.rule.antecedent) == 1:
            assert not verdict.redundant
            assert verdict.improvement > 0


def test_report_structure_is_consistent(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets))
    report = rule_redundancy(itemsets, rules, len(baskets))
    assert report.n_kept + report.n_redundant == report.n_rules == len(rules)
    assert all(v.improvement > report.tol for v in report.kept)
    assert all(v.improvement <= report.tol for v in report.pruned)
    assert report.maximal <= report.closed <= set(itemsets)
    # Ranking by improvement is a permutation of the kept verdicts.
    ranked = report.kept_by_improvement()
    assert sorted(v.label for v in ranked) == sorted(v.label for v in report.kept)
    assert all(
        ranked[i].improvement >= ranked[i + 1].improvement for i in range(len(ranked) - 1)
    )


def test_collapse_to_base_cases(tiny_baskets):
    # No rules at all: an empty but well-formed report.
    itemsets = _tiny_itemsets(tiny_baskets)
    report = rule_redundancy(itemsets, [], len(tiny_baskets))
    assert report.n_rules == report.n_redundant == report.n_kept == 0
    assert report.verdicts == []
    # Singletons only: everything is trivially closed and maximal.
    singles = {frozenset({"a"}): 0.5, frozenset({"b"}): 0.4}
    assert closed_itemsets(singles, 10) == set(singles)
    assert maximal_itemsets(singles) == set(singles)
    # All supports distinct: every itemset is closed.
    distinct = apriori(tiny_baskets, min_support=0.4, max_len=3)
    assert closed_itemsets(distinct, 5) == set(distinct)


def test_report_is_deterministic(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets))
    first = rule_redundancy(itemsets, rules, len(baskets))
    second = rule_redundancy(itemsets, rules, len(baskets))
    assert first.verdicts == second.verdicts
    assert first.closed == second.closed
    assert first.maximal == second.maximal
    assert first.plain_language() == second.plain_language()


def test_plain_language_reports_counts(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets))
    report = rule_redundancy(itemsets, rules, len(baskets))
    text = report.plain_language()
    assert str(report.n_rules) in text
    assert str(report.n_redundant) in text
    assert str(report.n_closed) in text
    assert "not causation" in text


def test_csv_and_svg_writers_deterministic(baskets, tmp_path):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets))
    report = rule_redundancy(itemsets, rules, len(baskets))

    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    write_redundancy_csv(report, str(csv_a))
    write_redundancy_csv(report, str(csv_b))
    assert csv_a.read_bytes() == csv_b.read_bytes()
    lines = csv_a.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "rank,rule,antecedent,consequent,support,support_count,confidence,lift,"
        "improvement,verdict,best_simpler_rule,best_simpler_confidence"
    )
    assert len(lines) == 1 + report.n_rules
    import csv as csv_module

    with open(csv_a, encoding="utf-8", newline="") as handle:
        rows = list(csv_module.DictReader(handle))
    assert sum(1 for row in rows if row["verdict"] == "redundant") == report.n_redundant
    assert sum(1 for row in rows if row["verdict"] == "kept") == report.n_kept

    svg = render_redundancy_svg(report)
    assert svg.startswith("<svg")
    assert "Minimal non-redundant rule set" in svg
    assert "CATEGORY ATLAS" in svg and "PLATE 02 / 07" in svg  # atlas plate header
    assert render_redundancy_svg(report) == svg  # pure function of the report
    svg_path = tmp_path / "r.svg"
    write_redundancy_svg(report, str(svg_path))
    assert svg_path.read_text(encoding="utf-8").startswith("<svg")
