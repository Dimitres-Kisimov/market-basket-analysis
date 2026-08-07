"""Cross-sell recommendations derived from mined association rules.

HONESTY NOTES, stated once here and repeated in every deliverable:

* Lift is observational. "Buyers of X are N x more likely to buy Z" describes
  co-purchase frequency in the (synthetic) history -- correlation, not
  causation. A campaign built on these rules still needs an A/B test.
* The incremental-value figure is an ESTIMATE with a stated assumption: if a
  targeted buyer of the antecedent attaches the consequent at the rule's
  confidence instead of the baseline rate, the expected extra revenue per
  targeted order is (confidence - baseline) * average line value of the
  consequent. Real attach rates will differ.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from basket.rules import Rule, format_itemset

ASSUMPTION_NOTE = (
    "ESTIMATE - assumes targeted buyers attach the consequent at the rule's "
    "confidence instead of the baseline rate; observational lift, correlation "
    "is not causation."
)


@dataclass(frozen=True)
class Recommendation:
    rule: Rule
    est_incremental_value_eur: float  # per targeted order, ESTIMATE
    headline: str


def _estimated_incremental_value(
    rule: Rule, avg_line_value_eur: Mapping[str, float]
) -> float:
    """(confidence - baseline attach rate) * avg line value of the consequent."""
    uplift_probability = max(0.0, rule.confidence - rule.consequent_support)
    consequent_value = sum(avg_line_value_eur.get(item, 0.0) for item in rule.consequent)
    return uplift_probability * consequent_value


def _headline(rule: Rule, value_eur: float) -> str:
    return (
        f"Buyers of {format_itemset(rule.antecedent)} are {rule.lift:.1f}x more "
        f"likely to also buy {format_itemset(rule.consequent)} "
        f"(confidence {rule.confidence:.0%}, lift {rule.lift:.2f}); "
        f"est. incremental basket value EUR {value_eur:.2f} per targeted order "
        f"[{ASSUMPTION_NOTE}]"
    )


def cross_sell_recommendations(
    rules: Sequence[Rule],
    avg_line_value_eur: Mapping[str, float],
    top_n: int = 10,
    include_thin_support: bool = False,
) -> list[Recommendation]:
    """Turn ranked rules into a cross-sell action list, sorted by lift."""
    recommendations: list[Recommendation] = []
    for rule in sorted(rules, key=lambda r: -r.lift):
        if rule.thin_support and not include_thin_support:
            continue
        value = _estimated_incremental_value(rule, avg_line_value_eur)
        recommendations.append(
            Recommendation(
                rule=rule,
                est_incremental_value_eur=value,
                headline=_headline(rule, value),
            )
        )
        if len(recommendations) >= top_n:
            break
    return recommendations


def applicable_rules(
    current_basket: Iterable[str],
    rules: Iterable[Rule],
    include_thin_support: bool = False,
) -> list[Rule]:
    """Rules that fire on a basket in progress, ranked highest-lift first.

    A rule is applicable when its antecedent is already fully in the basket and
    its consequent is entirely absent (there is something new to suggest). Thin
    -support rules are excluded unless ``include_thin_support`` is set. The order
    matches :func:`basket.rules.generate_rules` (lift, then confidence, then a
    stable itemset tie-break), so callers get a deterministic ranking.
    """
    basket = frozenset(current_basket)
    matched = [
        rule
        for rule in rules
        if (include_thin_support or not rule.thin_support)
        and rule.antecedent <= basket
        and rule.consequent.isdisjoint(basket)
    ]
    matched.sort(
        key=lambda r: (-r.lift, -r.confidence, format_itemset(r.antecedent), format_itemset(r.consequent))
    )
    return matched


def rank_cross_sell_categories(
    current_basket: Iterable[str],
    rules: Iterable[Rule],
    top_k: int = 5,
    include_thin_support: bool = False,
) -> list[str]:
    """Top-``top_k`` distinct categories to cross-sell into a basket in progress.

    Walks the applicable rules highest-lift first and collects their consequent
    categories, de-duplicated and skipping anything already in the basket, until
    ``top_k`` are gathered. This is the multi-suggestion generalisation of
    :func:`next_best_product` and the ranking the recommender back-test scores.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    basket = frozenset(current_basket)
    ranked: list[str] = []
    for rule in applicable_rules(basket, rules, include_thin_support):
        for category in sorted(rule.consequent):
            if category not in basket and category not in ranked:
                ranked.append(category)
                if len(ranked) >= top_k:
                    return ranked
    return ranked


def next_best_product(
    current_basket: Iterable[str],
    rules: Sequence[Rule],
    avg_line_value_eur: Mapping[str, float] | None = None,
) -> Recommendation | None:
    """Best next cross-sell for a basket in progress.

    Picks the highest-lift non-thin rule whose antecedent is already fully in
    the basket and whose consequent is entirely absent. Returns None when no
    rule applies.
    """
    basket = frozenset(current_basket)
    applicable = applicable_rules(basket, rules)
    if not applicable:
        return None
    best = max(applicable, key=lambda r: (r.lift, r.confidence))
    value = (
        _estimated_incremental_value(best, avg_line_value_eur)
        if avg_line_value_eur is not None
        else 0.0
    )
    return Recommendation(
        rule=best, est_incremental_value_eur=value, headline=_headline(best, value)
    )
