import numpy as np

from basket.segment import kmeans, segment_customers


def _three_blobs(seed: int = 3):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
    points = np.vstack([c + rng.normal(0.0, 0.4, size=(40, 2)) for c in centers])
    return points


def test_kmeans_converges_and_assigns_every_point():
    points = _three_blobs()
    result = kmeans(points, 3, seed=1)
    assert result.converged
    assert result.labels.shape == (120,)
    assert set(result.labels) == {0, 1, 2}  # no empty cluster
    # Well-separated blobs of 40 points each must be recovered exactly.
    sizes = sorted(int((result.labels == j).sum()) for j in range(3))
    assert sizes == [40, 40, 40]
    assert result.inertia > 0.0


def test_kmeans_deterministic_given_seed():
    points = _three_blobs()
    first = kmeans(points, 3, seed=5)
    second = kmeans(points, 3, seed=5)
    assert np.array_equal(first.labels, second.labels)
    assert np.allclose(first.centroids, second.centroids)
    assert first.inertia == second.inertia


def test_segment_customers_profiles(transactions):
    segmentation = segment_customers(transactions, k=3, seed=0)
    n_customers = transactions["customer_id"].nunique()
    assert len(segmentation.assignments) == n_customers
    assert set(segmentation.assignments.unique()) == {0, 1, 2}
    assert len(segmentation.profiles) == 3
    assert sum(p.n_customers for p in segmentation.profiles) == n_customers
    assert abs(sum(p.share_of_customers for p in segmentation.profiles) - 1.0) < 1e-9
    # Profiles are ordered largest-first and carry sensible spend numbers.
    sizes = [p.n_customers for p in segmentation.profiles]
    assert sizes == sorted(sizes, reverse=True)
    for profile in segmentation.profiles:
        assert profile.avg_total_spend_eur > 0
        assert len(profile.top_categories) == 3
        for category, share in profile.top_categories:
            assert category in segmentation.spend.columns
            assert 0.0 < share <= 1.0
