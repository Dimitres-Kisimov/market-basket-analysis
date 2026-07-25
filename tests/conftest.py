"""Shared fixtures: one moderately sized synthetic dataset per test session."""

import pytest

from basket.data import baskets_from_transactions, generate_transactions


@pytest.fixture(scope="session")
def transactions():
    return generate_transactions(n_baskets=2500, n_customers=300, seed=7)


@pytest.fixture(scope="session")
def baskets(transactions):
    return baskets_from_transactions(transactions)


@pytest.fixture()
def tiny_baskets():
    """Hand-checkable five-basket example used across metric tests."""
    return [
        ["milk", "bread"],
        ["milk", "bread", "butter"],
        ["milk"],
        ["bread"],
        ["milk", "bread"],
    ]
