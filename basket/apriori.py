"""Apriori frequent-itemset mining, implemented from scratch.

Reference: Agrawal & Srikant, "Fast Algorithms for Mining Association Rules"
(VLDB 1994). No mining library is used; candidate generation, downward-closure
pruning and support counting are all implemented here on plain Python sets.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations


def min_count_for_support(min_support: float, n_baskets: int) -> int:
    """Smallest absolute count c such that c / n_baskets >= min_support.

    The tiny epsilon guards against float products like 0.02 * 6000 landing an
    ulp above the exact integer and inflating the ceiling by one.
    """
    return max(1, math.ceil(min_support * n_baskets - 1e-9))


def _generate_candidates(
    previous_frequent: set[frozenset[str]], k: int
) -> set[frozenset[str]]:
    """Join step + prune step (downward closure) of Apriori."""
    sorted_itemsets = sorted(tuple(sorted(itemset)) for itemset in previous_frequent)
    candidates: set[frozenset[str]] = set()
    for i, a in enumerate(sorted_itemsets):
        for b in sorted_itemsets[i + 1 :]:
            if a[: k - 2] != b[: k - 2]:
                continue  # prefix join: only merge itemsets sharing the first k-2 items
            candidate = frozenset(a) | frozenset(b)
            if len(candidate) != k:
                continue
            # Prune: every (k-1)-subset must itself be frequent.
            if all(
                frozenset(subset) in previous_frequent
                for subset in combinations(sorted(candidate), k - 1)
            ):
                candidates.add(candidate)
    return candidates


def apriori(
    baskets: Sequence[Iterable[str]],
    min_support: float = 0.02,
    max_len: int = 3,
) -> dict[frozenset[str], float]:
    """Mine frequent itemsets up to ``max_len`` items.

    Returns a mapping ``itemset -> support`` where support is the fraction of
    baskets containing the itemset (always computed as count / n_baskets, so
    results are exactly comparable with :func:`basket.fpgrowth.fpgrowth`).
    """
    if not 0.0 < min_support <= 1.0:
        raise ValueError("min_support must be in (0, 1]")
    basket_sets = [frozenset(b) for b in baskets]
    n = len(basket_sets)
    if n == 0:
        return {}
    min_count = min_count_for_support(min_support, n)

    item_counts = Counter(item for basket in basket_sets for item in basket)
    current: dict[frozenset[str], int] = {
        frozenset((item,)): count
        for item, count in item_counts.items()
        if count >= min_count
    }
    all_frequent: dict[frozenset[str], int] = dict(current)

    k = 2
    while current and k <= max_len:
        candidates = _generate_candidates(set(current), k)
        counts = dict.fromkeys(candidates, 0)
        for basket in basket_sets:
            if len(basket) < k:
                continue
            for candidate in candidates:
                if candidate <= basket:
                    counts[candidate] += 1
        current = {c: cnt for c, cnt in counts.items() if cnt >= min_count}
        all_frequent.update(current)
        k += 1

    return {itemset: count / n for itemset, count in all_frequent.items()}
