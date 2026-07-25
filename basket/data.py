"""Seeded synthetic transaction generator for a B2B industrial distributor.

IMPORTANT: every number produced by this module is SYNTHETIC. The generator
simulates order lines for a fictional maintenance/construction-supplies
distributor. It is seeded and fully deterministic so that results in the
README and the exported deliverables can be reproduced exactly.

The generator *plants* a handful of product bundles with known co-purchase
structure (e.g. fasteners + power tools + work gloves). These planted
associations are exposed as ground truth (``PLANTED_BUNDLES`` /
``planted_pairs``) so the mining code can be tested against a known answer.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

#: Product categories carried by the fictional distributor.
CATEGORIES: tuple[str, ...] = (
    "abrasives",
    "adhesives_sealants",
    "cutting_tools",
    "electrical",
    "fasteners",
    "hand_tools",
    "janitorial",
    "lubricants",
    "pipe_fittings",
    "power_tools",
    "ppe_eyewear",
    "ppe_gloves",
    "storage_handling",
    "welding_supplies",
)

#: Average order-line value per category (EUR). Synthetic but plausible.
AVG_LINE_VALUE_EUR: dict[str, float] = {
    "abrasives": 55.0,
    "adhesives_sealants": 42.0,
    "cutting_tools": 130.0,
    "electrical": 110.0,
    "fasteners": 45.0,
    "hand_tools": 85.0,
    "janitorial": 35.0,
    "lubricants": 60.0,
    "pipe_fittings": 95.0,
    "power_tools": 260.0,
    "ppe_eyewear": 30.0,
    "ppe_gloves": 38.0,
    "storage_handling": 150.0,
    "welding_supplies": 180.0,
}

#: Ground-truth bundles injected into a share of baskets. The mining code is
#: expected to rediscover these as high-lift associations.
PLANTED_BUNDLES: tuple[tuple[str, ...], ...] = (
    ("fasteners", "power_tools", "ppe_gloves"),
    ("welding_supplies", "ppe_eyewear"),
    ("pipe_fittings", "adhesives_sealants"),
    ("cutting_tools", "lubricants"),
)

#: Per-basket probability that each bundle above is injected (mutually
#: exclusive draws; at most one bundle per basket).
_BUNDLE_INJECTION_PROB: tuple[float, ...] = (0.13, 0.10, 0.10, 0.09)

#: Customer archetypes and their category affinity weights. These drive both
#: basket fill items and the spend structure that k-means segmentation should
#: roughly recover.
_ARCHETYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "construction": {
        "fasteners": 3.0,
        "power_tools": 2.5,
        "hand_tools": 2.0,
        "ppe_gloves": 2.0,
        "storage_handling": 1.5,
        "abrasives": 1.0,
    },
    "maintenance": {
        "electrical": 3.0,
        "lubricants": 2.2,
        "pipe_fittings": 2.0,
        "adhesives_sealants": 1.8,
        "janitorial": 1.8,
        "hand_tools": 1.2,
    },
    "fabrication": {
        "welding_supplies": 3.0,
        "cutting_tools": 2.5,
        "abrasives": 2.2,
        "ppe_eyewear": 2.0,
        "lubricants": 1.2,
        "ppe_gloves": 1.0,
    },
}

_ARCHETYPE_SHARES: tuple[float, ...] = (0.40, 0.35, 0.25)

_BASE_WEIGHT = 0.3  # floor weight so every category can appear in any basket


def planted_pairs() -> set[frozenset[str]]:
    """All 2-item subsets of the planted bundles (ground truth for tests)."""
    return {frozenset(pair) for bundle in PLANTED_BUNDLES for pair in combinations(bundle, 2)}


def _archetype_probability_vector(archetype: str) -> np.ndarray:
    weights = np.full(len(CATEGORIES), _BASE_WEIGHT)
    for category, weight in _ARCHETYPE_WEIGHTS[archetype].items():
        weights[CATEGORIES.index(category)] = weight
    return weights / weights.sum()


def generate_transactions(
    n_baskets: int = 6000,
    n_customers: int = 420,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic synthetic order-line table.

    Returns a DataFrame with one row per order line and columns:
    ``basket_id, customer_id, archetype, category, amount_eur``.
    """
    if n_baskets < 1 or n_customers < 1:
        raise ValueError("n_baskets and n_customers must be positive")
    rng = np.random.default_rng(seed)
    archetype_names = tuple(_ARCHETYPE_WEIGHTS)
    probability_vectors = {a: _archetype_probability_vector(a) for a in archetype_names}
    customer_archetypes = rng.choice(
        len(archetype_names), size=n_customers, p=_ARCHETYPE_SHARES
    )

    rows: list[tuple[int, str, str, str, float]] = []
    for basket_id in range(n_baskets):
        customer = int(rng.integers(0, n_customers))
        archetype = archetype_names[customer_archetypes[customer]]

        items: set[str] = set()
        draw = rng.random()
        cumulative = 0.0
        for bundle, prob in zip(PLANTED_BUNDLES, _BUNDLE_INJECTION_PROB, strict=True):
            cumulative += prob
            if draw < cumulative:
                items.update(bundle)
                break

        n_fill = int(rng.integers(1, 6))  # 1..5 archetype-driven fill items
        fill = rng.choice(
            len(CATEGORIES), size=n_fill, replace=False, p=probability_vectors[archetype]
        )
        items.update(CATEGORIES[i] for i in fill)

        for category in sorted(items):
            amount = AVG_LINE_VALUE_EUR[category] * float(rng.lognormal(0.0, 0.35))
            rows.append((basket_id, f"C{customer:04d}", archetype, category, round(amount, 2)))

    return pd.DataFrame(
        rows, columns=["basket_id", "customer_id", "archetype", "category", "amount_eur"]
    )


def baskets_from_transactions(transactions: pd.DataFrame) -> list[frozenset[str]]:
    """Collapse the order-line table into one frozenset of categories per basket."""
    grouped = transactions.groupby("basket_id", sort=True)["category"]
    return [frozenset(categories) for _, categories in grouped]
