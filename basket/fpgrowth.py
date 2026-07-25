"""FP-growth frequent-itemset mining, implemented from scratch.

Reference: Han, Pei & Yin, "Mining Frequent Patterns without Candidate
Generation" (SIGMOD 2000). Builds an FP-tree with a header table, then mines
recursively over conditional pattern bases. Support values are returned as
count / n_baskets so the output is exactly comparable with
:func:`basket.apriori.apriori` (a test asserts equality).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from basket.apriori import min_count_for_support


class _FPNode:
    __slots__ = ("children", "count", "item", "parent")

    def __init__(self, item: str | None, parent: _FPNode | None):
        self.item = item
        self.parent = parent
        self.count = 0
        self.children: dict[str, _FPNode] = {}


def _build_tree(
    weighted_transactions: list[tuple[list[str], int]], min_count: int
) -> tuple[dict[str, list[_FPNode]], dict[str, int]]:
    """Build an FP-tree; return its header table and per-item frequent counts."""
    item_counts: dict[str, int] = defaultdict(int)
    for items, weight in weighted_transactions:
        for item in items:
            item_counts[item] += weight
    frequent = {item: c for item, c in item_counts.items() if c >= min_count}
    # Descending count, name as tie-break: the canonical FP-tree insertion order.
    rank = {
        item: position
        for position, item in enumerate(
            sorted(frequent, key=lambda it: (-frequent[it], it))
        )
    }

    root = _FPNode(None, None)
    header: dict[str, list[_FPNode]] = defaultdict(list)
    for items, weight in weighted_transactions:
        path = sorted((i for i in items if i in frequent), key=rank.__getitem__)
        node = root
        for item in path:
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, node)
                node.children[item] = child
                header[item].append(child)
            child.count += weight
            node = child
    return header, frequent


def _mine(
    weighted_transactions: list[tuple[list[str], int]],
    min_count: int,
    suffix: frozenset[str],
    results: dict[frozenset[str], int],
    max_len: int | None,
) -> None:
    header, frequent = _build_tree(weighted_transactions, min_count)
    # Mine items from least to most frequent (bottom of the tree upward).
    for item in sorted(frequent, key=lambda it: (frequent[it], it)):
        itemset = frozenset(suffix | {item})
        results[itemset] = frequent[item]
        if max_len is not None and len(itemset) >= max_len:
            continue
        conditional_base: list[tuple[list[str], int]] = []
        for node in header[item]:
            path: list[str] = []
            parent = node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                conditional_base.append((path, node.count))
        if conditional_base:
            _mine(conditional_base, min_count, itemset, results, max_len)


def fpgrowth(
    baskets: Sequence[Iterable[str]],
    min_support: float = 0.02,
    max_len: int | None = 3,
) -> dict[frozenset[str], float]:
    """Mine frequent itemsets with FP-growth; same contract as ``apriori``."""
    if not 0.0 < min_support <= 1.0:
        raise ValueError("min_support must be in (0, 1]")
    basket_sets = [frozenset(b) for b in baskets]
    n = len(basket_sets)
    if n == 0:
        return {}
    min_count = min_count_for_support(min_support, n)
    weighted = [(sorted(b), 1) for b in basket_sets]
    results: dict[frozenset[str], int] = {}
    _mine(weighted, min_count, frozenset(), results, max_len)
    return {itemset: count / n for itemset, count in results.items()}
