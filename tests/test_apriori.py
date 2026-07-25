from itertools import combinations

from basket.apriori import apriori
from basket.data import planted_pairs


def test_tiny_hand_checked_supports(tiny_baskets):
    result = apriori(tiny_baskets, min_support=0.4, max_len=3)
    # Hand-computed on the five baskets: milk 4/5, bread 4/5, {milk, bread} 3/5.
    assert result == {
        frozenset({"milk"}): 0.8,
        frozenset({"bread"}): 0.8,
        frozenset({"milk", "bread"}): 0.6,
    }
    # butter appears once (support 0.2) and must be pruned.
    assert frozenset({"butter"}) not in result


def test_downward_closure_holds(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    assert itemsets, "expected a non-empty result"
    for itemset, support in itemsets.items():
        assert 0.0 < support <= 1.0
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for subset in combinations(sorted(itemset), size):
                subset = frozenset(subset)
                assert subset in itemsets, f"missing subset {subset} of {itemset}"
                # Anti-monotone: a subset can never be rarer than its superset.
                assert itemsets[subset] >= support


def test_recovers_planted_pairs(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    for pair in planted_pairs():
        assert pair in itemsets, f"planted pair {set(pair)} not recovered"
        a, b = (frozenset({item}) for item in pair)
        lift = itemsets[pair] / (itemsets[a] * itemsets[b])
        assert lift >= 1.5, f"planted pair {set(pair)} has weak lift {lift:.2f}"
