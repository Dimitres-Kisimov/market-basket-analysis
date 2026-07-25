from basket.apriori import apriori
from basket.fpgrowth import fpgrowth


def test_tiny_hand_checked(tiny_baskets):
    result = fpgrowth(tiny_baskets, min_support=0.4, max_len=3)
    assert result == {
        frozenset({"milk"}): 0.8,
        frozenset({"bread"}): 0.8,
        frozenset({"milk", "bread"}): 0.6,
    }


def test_fpgrowth_equals_apriori(tiny_baskets, baskets):
    # Small hand-built instance at a low threshold (includes butter).
    assert fpgrowth(tiny_baskets, min_support=0.2, max_len=3) == apriori(
        tiny_baskets, min_support=0.2, max_len=3
    )
    # Full synthetic dataset: identical itemsets AND identical supports.
    fp = fpgrowth(baskets, min_support=0.02, max_len=3)
    ap = apriori(baskets, min_support=0.02, max_len=3)
    assert fp == ap
