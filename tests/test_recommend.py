from basket.apriori import apriori
from basket.data import AVG_LINE_VALUE_EUR, CATEGORIES
from basket.recommend import cross_sell_recommendations, next_best_product
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
