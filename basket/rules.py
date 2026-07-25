"""Association rules with support, confidence, lift, leverage and conviction.

All metrics are computed from scratch. A note on interpretation that also
appears in the exported deliverables: these are OBSERVATIONAL co-purchase
statistics. High lift means two categories appear together more often than
independence would predict -- it does not prove that buying one *causes*
buying the other. Rules whose absolute support count is small are flagged
``thin_support`` because their metrics are unstable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

from basket.apriori import min_count_for_support


@dataclass(frozen=True)
class Rule:
    """An association rule ``antecedent -> consequent`` with its metrics."""

    antecedent: frozenset[str]
    consequent: frozenset[str]
    support: float  # fraction of baskets containing antecedent AND consequent
    confidence: float  # P(consequent | antecedent)
    lift: float  # confidence / P(consequent)
    leverage: float  # support - P(antecedent) * P(consequent)
    conviction: float  # (1 - P(consequent)) / (1 - confidence); inf if confidence == 1
    support_count: int  # absolute number of baskets behind the rule
    thin_support: bool  # True when support_count is below the stability threshold

    @property
    def consequent_support(self) -> float:
        """Baseline P(consequent), recovered from confidence / lift."""
        return self.confidence / self.lift


def format_itemset(itemset: frozenset[str]) -> str:
    return " + ".join(sorted(itemset))


def format_rule(rule: Rule) -> str:
    return f"{format_itemset(rule.antecedent)} -> {format_itemset(rule.consequent)}"


def generate_rules(
    itemsets: Mapping[frozenset[str], float],
    n_baskets: int,
    min_confidence: float = 0.30,
    min_lift: float = 1.10,
    thin_support_count: int = 30,
) -> list[Rule]:
    """Derive filtered, lift-ranked rules from mined frequent itemsets.

    ``itemsets`` maps itemset -> support fraction (the output of ``apriori`` or
    ``fpgrowth``). Every antecedent/consequent split of every itemset with two
    or more items is scored; rules below ``min_confidence`` or ``min_lift`` are
    dropped. Rules backed by fewer than ``thin_support_count`` baskets are kept
    but flagged ``thin_support=True`` so downstream consumers can exclude them.
    """
    if n_baskets < 1:
        raise ValueError("n_baskets must be positive")
    rules: list[Rule] = []
    for itemset, support in itemsets.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for split_size in range(1, len(items)):
            for antecedent_items in combinations(items, split_size):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                support_antecedent = itemsets.get(antecedent)
                support_consequent = itemsets.get(consequent)
                if support_antecedent is None or support_consequent is None:
                    # Cannot happen for complete downward-closed input; guard anyway.
                    continue
                confidence = support / support_antecedent
                lift = confidence / support_consequent
                leverage = support - support_antecedent * support_consequent
                if confidence >= 1.0:
                    conviction = math.inf
                else:
                    conviction = (1.0 - support_consequent) / (1.0 - confidence)
                if confidence < min_confidence or lift < min_lift:
                    continue
                support_count = round(support * n_baskets)
                rules.append(
                    Rule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=support,
                        confidence=confidence,
                        lift=lift,
                        leverage=leverage,
                        conviction=conviction,
                        support_count=support_count,
                        thin_support=support_count < thin_support_count,
                    )
                )
    rules.sort(
        key=lambda r: (-r.lift, -r.confidence, format_itemset(r.antecedent), format_itemset(r.consequent))
    )
    return rules


def thin_support_threshold_for(min_support: float, n_baskets: int) -> int:
    """A helper for callers that want the threshold Apriori itself used."""
    return min_count_for_support(min_support, n_baskets)
