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
from basket.rules import Rule, generate_rules
from basket.segment import kmeans, segment_customers

__version__ = "1.0.0"

__all__ = [
    "AVG_LINE_VALUE_EUR",
    "CATEGORIES",
    "PLANTED_BUNDLES",
    "Rule",
    "apriori",
    "baskets_from_transactions",
    "cross_sell_recommendations",
    "fpgrowth",
    "generate_rules",
    "generate_transactions",
    "kmeans",
    "next_best_product",
    "planted_pairs",
    "segment_customers",
]
