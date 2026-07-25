import math

import pytest

from basket.apriori import apriori
from basket.rules import generate_rules


def _rule_for(rules, antecedent, consequent):
    for rule in rules:
        if rule.antecedent == frozenset(antecedent) and rule.consequent == frozenset(consequent):
            return rule
    raise AssertionError(f"rule {antecedent} -> {consequent} not found")


def test_metrics_hand_checked(tiny_baskets):
    itemsets = apriori(tiny_baskets, min_support=0.4, max_len=2)
    rules = generate_rules(
        itemsets, n_baskets=5, min_confidence=0.5, min_lift=0.5, thin_support_count=30
    )
    rule = _rule_for(rules, {"milk"}, {"bread"})
    # Hand-computed: support 3/5, confidence 3/4, lift 0.75/0.8 = 0.9375,
    # leverage 0.6 - 0.8*0.8 = -0.04, conviction (1-0.8)/(1-0.75) = 0.8.
    assert rule.support == pytest.approx(0.6)
    assert rule.confidence == pytest.approx(0.75)
    assert rule.lift == pytest.approx(0.9375)
    assert rule.leverage == pytest.approx(-0.04)
    assert rule.conviction == pytest.approx(0.8)
    assert rule.support_count == 3
    assert rule.thin_support  # 3 < 30


def test_conviction_infinite_when_confidence_is_one():
    baskets = [["A", "B"], ["A", "B"], ["B"]]
    itemsets = apriori(baskets, min_support=0.5, max_len=2)
    rules = generate_rules(itemsets, 3, min_confidence=0.9, min_lift=0.5, thin_support_count=1)
    rule = _rule_for(rules, {"A"}, {"B"})
    assert rule.confidence == pytest.approx(1.0)
    assert math.isinf(rule.conviction)


def test_filters_and_lift_ranking(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets), min_confidence=0.30, min_lift=1.10)
    assert rules, "expected rules on the synthetic data"
    for rule in rules:
        assert rule.confidence >= 0.30
        assert rule.lift >= 1.10
    lifts = [rule.lift for rule in rules]
    assert lifts == sorted(lifts, reverse=True)


def test_thin_support_flag(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    all_thin = generate_rules(
        itemsets, len(baskets), min_confidence=0.3, min_lift=1.1,
        thin_support_count=10**9,
    )
    assert all(rule.thin_support for rule in all_thin)
    none_thin = generate_rules(
        itemsets, len(baskets), min_confidence=0.3, min_lift=1.1, thin_support_count=0
    )
    assert not any(rule.thin_support for rule in none_thin)
    # Flag matches the raw count comparison.
    flagged = generate_rules(
        itemsets, len(baskets), min_confidence=0.3, min_lift=1.1, thin_support_count=100
    )
    for rule in flagged:
        assert rule.thin_support == (rule.support_count < 100)
