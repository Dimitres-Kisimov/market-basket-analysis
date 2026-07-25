"""Customer segmentation: k-means from scratch (numpy) on category-spend shares.

Deliberately small and honest: k-means on spend-share vectors with k-means++
seeding, several restarts, and an empty-cluster repair step. The synthetic
generator plants three purchasing archetypes, so k=3 is the natural choice;
the segmentation is descriptive, not predictive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KMeansResult:
    labels: np.ndarray  # (n,) cluster index per row
    centroids: np.ndarray  # (k, d)
    inertia: float  # sum of squared distances to assigned centroid
    n_iter: int
    converged: bool


@dataclass
class SegmentProfile:
    segment: int
    n_customers: int
    share_of_customers: float
    avg_total_spend_eur: float
    top_categories: list[tuple[str, float]]  # (category, centroid spend share)


@dataclass
class Segmentation:
    assignments: pd.Series  # customer_id -> segment
    profiles: list[SegmentProfile]
    kmeans_result: KMeansResult
    spend: pd.DataFrame  # customers x categories, EUR


def _kmeans_plus_plus_init(
    points: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    n = points.shape[0]
    centroids = np.empty((k, points.shape[1]))
    centroids[0] = points[rng.integers(n)]
    closest_sq = ((points - centroids[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        total = closest_sq.sum()
        if total <= 0.0:
            centroids[j] = points[rng.integers(n)]
        else:
            centroids[j] = points[rng.choice(n, p=closest_sq / total)]
        closest_sq = np.minimum(closest_sq, ((points - centroids[j]) ** 2).sum(axis=1))
    return centroids


def _kmeans_single(
    points: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
) -> KMeansResult:
    centroids = _kmeans_plus_plus_init(points, k, rng)
    labels = np.zeros(points.shape[0], dtype=int)
    converged = False
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        distances_sq = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances_sq.argmin(axis=1)
        new_centroids = centroids.copy()
        for j in range(k):
            members = labels == j
            if members.any():
                new_centroids[j] = points[members].mean(axis=0)
            else:  # empty cluster: re-seed at the point farthest from its centroid
                farthest = distances_sq.min(axis=1).argmax()
                new_centroids[j] = points[farthest]
        shift = float(((new_centroids - centroids) ** 2).sum())
        centroids = new_centroids
        if shift <= tol:
            converged = True
            break
    distances_sq = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = distances_sq.argmin(axis=1)
    inertia = float(distances_sq[np.arange(points.shape[0]), labels].sum())
    return KMeansResult(labels, centroids, inertia, n_iter, converged)


def kmeans(
    points: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 8,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> KMeansResult:
    """Lloyd's k-means with k-means++ seeding and ``n_init`` restarts."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < k or k < 1:
        raise ValueError("need a 2-D array with at least k rows and k >= 1")
    rng = np.random.default_rng(seed)
    best: KMeansResult | None = None
    for _ in range(n_init):
        candidate = _kmeans_single(points, k, rng, max_iter, tol)
        if best is None or candidate.inertia < best.inertia:
            best = candidate
    assert best is not None
    return best


def customer_spend_matrix(transactions: pd.DataFrame) -> pd.DataFrame:
    """Customers x categories total spend (EUR), zero-filled, sorted both ways."""
    matrix = transactions.pivot_table(
        index="customer_id",
        columns="category",
        values="amount_eur",
        aggfunc="sum",
        fill_value=0.0,
    )
    return matrix.sort_index().sort_index(axis=1).astype(float)


def segment_customers(
    transactions: pd.DataFrame,
    k: int = 3,
    seed: int = 0,
    top_n_categories: int = 3,
) -> Segmentation:
    """Cluster customers on spend *shares* so big spenders don't dominate."""
    spend = customer_spend_matrix(transactions)
    shares = spend.div(spend.sum(axis=1), axis=0).to_numpy()
    result = kmeans(shares, k, seed=seed)

    # Relabel clusters by descending size for stable, readable reporting.
    order = np.argsort([-(result.labels == j).sum() for j in range(k)], kind="stable")
    relabel = {int(old): new for new, old in enumerate(order)}
    labels = np.array([relabel[int(label)] for label in result.labels])
    centroids = result.centroids[order]
    result = KMeansResult(labels, centroids, result.inertia, result.n_iter, result.converged)

    assignments = pd.Series(labels, index=spend.index, name="segment")
    total_spend = spend.sum(axis=1)
    profiles: list[SegmentProfile] = []
    for j in range(k):
        members = labels == j
        centroid = centroids[j]
        top_idx = np.argsort(-centroid)[:top_n_categories]
        profiles.append(
            SegmentProfile(
                segment=j,
                n_customers=int(members.sum()),
                share_of_customers=float(members.mean()),
                avg_total_spend_eur=float(total_spend.to_numpy()[members].mean()),
                top_categories=[
                    (str(spend.columns[i]), float(centroid[i])) for i in top_idx
                ],
            )
        )
    return Segmentation(assignments, profiles, result, spend)
