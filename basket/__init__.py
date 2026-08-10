"""Market-basket analysis and cross-sell toolkit for a B2B distributor.

All algorithms (Apriori, FP-growth, association-rule metrics, k-means) are
implemented from scratch on numpy/pandas. All data is SYNTHETIC and seeded.
"""

from basket.apriori import apriori
from basket.data import (
    AVG_LINE_VALUE_EUR,
    CATEGORIES,
    PLANTED_BUNDLES,
    baskets_from_transactions,
    generate_transactions,
    planted_pairs,
)
from basket.fpgrowth import fpgrowth
from basket.recommend import cross_sell_recommendations, next_best_product
from basket.redundancy import (
    RedundancyReport,
    RuleVerdict,
    closed_itemsets,
    maximal_itemsets,
    rule_improvement,
    rule_redundancy,
)
from basket.rules import Rule, generate_rules
from basket.segment import kmeans, segment_customers
from basket.stability import (
    RuleStability,
    StabilityReport,
    bootstrap_splits,
    rule_stability,
    time_window_splits,
)

__version__ = "1.0.0"

__all__ = [
    "AVG_LINE_VALUE_EUR",
    "CATEGORIES",
    "PLANTED_BUNDLES",
    "RedundancyReport",
    "Rule",
    "RuleStability",
    "RuleVerdict",
    "StabilityReport",
    "apriori",
    "baskets_from_transactions",
    "bootstrap_splits",
    "closed_itemsets",
    "cross_sell_recommendations",
    "fpgrowth",
    "generate_rules",
    "generate_transactions",
    "kmeans",
    "maximal_itemsets",
    "next_best_product",
    "planted_pairs",
    "rule_improvement",
    "rule_redundancy",
    "rule_stability",
    "segment_customers",
    "time_window_splits",
]
