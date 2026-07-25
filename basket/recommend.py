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
    applicable = [
        rule
        for rule in rules
        if not rule.thin_support
        and rule.antecedent <= basket
        and rule.consequent.isdisjoint(basket)
    ]
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
