import pandas as pd

from basket.data import (
    AVG_LINE_VALUE_EUR,
    CATEGORIES,
    PLANTED_BUNDLES,
    baskets_from_transactions,
    generate_transactions,
    planted_pairs,
)


def test_determinism_same_seed():
    first = generate_transactions(n_baskets=800, n_customers=120, seed=11)
    second = generate_transactions(n_baskets=800, n_customers=120, seed=11)
    pd.testing.assert_frame_equal(first, second)
    other = generate_transactions(n_baskets=800, n_customers=120, seed=12)
    assert not first.equals(other)


def test_schema_and_values(transactions):
    assert list(transactions.columns) == [
        "basket_id", "customer_id", "archetype", "category", "amount_eur",
    ]
    assert set(transactions["category"]).issubset(set(CATEGORIES))
    assert (transactions["amount_eur"] > 0).all()
    # Every basket id in range is present and no basket repeats a category.
    assert set(transactions["basket_id"]) == set(range(2500))
    duplicates = transactions.duplicated(subset=["basket_id", "category"])
    assert not duplicates.any()
    baskets = baskets_from_transactions(transactions)
    assert len(baskets) == 2500
    assert all(len(b) >= 1 for b in baskets)


def test_planted_ground_truth_exposed():
    assert len(PLANTED_BUNDLES) == 4
    for bundle in PLANTED_BUNDLES:
        assert len(bundle) >= 2
        assert set(bundle).issubset(set(CATEGORIES))
    # 3-item bundle contributes three pairs, each 2-item bundle one pair.
    assert len(planted_pairs()) == 6
    assert set(AVG_LINE_VALUE_EUR) == set(CATEGORIES)
