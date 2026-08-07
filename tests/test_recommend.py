from basket.apriori import apriori
from basket.data import AVG_LINE_VALUE_EUR, CATEGORIES
from basket.recommend import (
    applicable_rules,
    cross_sell_recommendations,
    next_best_product,
    rank_cross_sell_categories,
)
from basket.rules import generate_rules


def _rules(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    return generate_rules(itemsets, len(baskets), min_confidence=0.30, min_lift=1.10)


def test_recommendations_sorted_by_lift_and_honest(baskets):
    rules = _rules(baskets)
    recommendations = cross_sell_recommendations(rules, AVG_LINE_VALUE_EUR, top_n=10)
    assert recommendations
    lifts = [r.rule.lift for r in recommendations]
    assert lifts == sorted(lifts, reverse=True)
    for recommendation in recommendations:
        assert not recommendation.rule.thin_support  # thin rules excluded by default
        assert recommendation.est_incremental_value_eur >= 0.0
        assert "ESTIMATE" in recommendation.headline
        assert "EUR" in recommendation.headline
        assert "not causation" in recommendation.headline


def test_next_best_product(baskets):
    rules = _rules(baskets)
    best_rule = next(rule for rule in rules if not rule.thin_support)
    current = set(best_rule.antecedent)
    recommendation = next_best_product(current, rules, AVG_LINE_VALUE_EUR)
    assert recommendation is not None
    assert recommendation.rule.antecedent <= frozenset(current)
    assert recommendation.rule.consequent.isdisjoint(current)
    # A basket that already contains every category has nothing left to add.
    assert next_best_product(set(CATEGORIES), rules) is None


def test_rank_cross_sell_categories(baskets):
    rules = _rules(baskets)
    best_rule = next(rule for rule in rules if not rule.thin_support)
    current = set(best_rule.antecedent)
    ranked = rank_cross_sell_categories(current, rules, top_k=5)
    assert 1 <= len(ranked) <= 5
    assert len(ranked) == len(set(ranked))  # de-duplicated
    for category in ranked:
        assert category not in current  # never suggests what is already there
    # The single next-best product is the head of the ranked list (same ranking).
    single = next_best_product(current, rules)
    assert single is not None
    assert ranked[0] in single.rule.consequent
    # An all-category basket has nothing to add.
    assert rank_cross_sell_categories(set(CATEGORIES), rules) == []


def test_applicable_rules_filters_and_thin_support():
    baskets = [["A", "B"]] * 5 + [["A"], ["B"]]
    itemsets = apriori(baskets, min_support=0.2, max_len=2)
    # thin_support_count=1 keeps rules; =100 flags them all as thin.
    kept = generate_rules(itemsets, len(baskets), min_confidence=0.3, min_lift=0.5, thin_support_count=1)
    thin = generate_rules(itemsets, len(baskets), min_confidence=0.3, min_lift=0.5, thin_support_count=100)
    assert applicable_rules(["A"], kept)  # A -> B fires
    assert not applicable_rules(["A"], thin)  # thin rules excluded by default
    assert applicable_rules(["A"], thin, include_thin_support=True)
    # Nothing to add once the consequent is already present.
    assert not applicable_rules(["A", "B"], kept)
