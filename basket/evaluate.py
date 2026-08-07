"""Recommender back-test: does the rule-based cross-sell actually predict?

The rules and the :func:`basket.recommend.next_best_product` engine look
convincing, but a high lift on the full data is not evidence that the
recommender predicts what a customer will *actually* buy next. This module
measures that directly with a standard offline, leave-one-out evaluation --
the predictive-accuracy counterpart to the rule-stability trust layer.

Protocol (fully deterministic):

1. Split the baskets into a TRAIN slice and a TEST slice by arrival order (the
   ``basket_id`` generation sequence is used as a documented proxy for time, the
   same convention as :mod:`basket.stability`); rules are mined on TRAIN only so
   no test basket informs its own recommendation.
2. For every TEST basket with at least two categories, hide exactly one category
   (a seeded random pick), feed the remaining categories to the recommender as a
   basket in progress, and ask for its top-``max_k`` next-best categories.
3. Score whether the hidden category is recovered: ``hit_rate@K`` (fraction of
   trials whose held-out category is in the top-K), ``MRR`` (mean reciprocal
   rank of the held-out category within the top-``max_k`` list) and ``coverage``
   (fraction of trials where the recommender returned anything at all).

The exact same trials are scored for a POPULARITY baseline -- always recommend
the globally most-frequent categories (from TRAIN) that are not already in the
basket. Reporting the rule recommender *and* the baseline side by side is the
honest way to show the rules add predictive value beyond base rates, rather than
asserting it.

Everything runs on SYNTHETIC, seeded data; every number is MEASURED. One honest
caveat: the synthetic generator is stationary, so an arrival-order split is not a
real distribution shift -- it validates the machinery and quantifies predictive
lift over popularity. On real, drifting order history the identical back-test
would report how well yesterday's rules predict tomorrow's baskets.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from basket.apriori import apriori
from basket.recommend import rank_cross_sell_categories
from basket.rules import generate_rules

Basket = frozenset


@dataclass(frozen=True)
class HoldoutTrial:
    """One leave-one-out test case: the visible basket and the hidden target."""

    query: frozenset[str]  # categories shown to the recommender
    held_out: str  # the single category hidden from it


@dataclass(frozen=True)
class RecommenderMetrics:
    """Scored accuracy for one recommender over the shared set of trials."""

    name: str
    n_trials: int
    hit_rate: dict[int, float]  # K -> fraction of trials with the target in top-K
    mrr: float  # mean reciprocal rank of the target within the top-max_k list
    coverage: float  # fraction of trials that produced any recommendation


@dataclass
class EvaluationReport:
    """A back-test read-out for the rule recommender vs a popularity baseline."""

    n_baskets: int
    n_train: int
    n_test: int
    n_trials: int
    train_fraction: float
    k_values: tuple[int, ...]
    seed: int
    min_support: float
    min_confidence: float
    min_lift: float
    n_rules: int
    rules: RecommenderMetrics
    popularity: RecommenderMetrics
    _extra: dict = field(default_factory=dict, repr=False)

    @property
    def max_k(self) -> int:
        return max(self.k_values)

    def hit_rate_ratio(self) -> dict[int, float]:
        """Rule hit_rate@K divided by popularity hit_rate@K (0.0 if baseline 0)."""
        return {
            k: (self.rules.hit_rate[k] / self.popularity.hit_rate[k])
            if self.popularity.hit_rate[k]
            else 0.0
            for k in self.k_values
        }

    def plain_language(self) -> str:
        """A one-paragraph, jargon-light summary of the back-test."""
        headline_k = self.k_values[len(self.k_values) // 2]
        ratio = self.hit_rate_ratio()[headline_k]
        return (
            f"Leave-one-out back-test on {self.n_trials} held-out test baskets "
            f"(mined on {self.n_train} training baskets, tested on the later "
            f"{self.n_test}). The rule recommender recovers the hidden category "
            f"in {self.rules.hit_rate[headline_k]:.0%} of cases within its top "
            f"{headline_k} (MRR {self.rules.mrr:.3f}), versus "
            f"{self.popularity.hit_rate[headline_k]:.0%} for a popularity baseline "
            f"(MRR {self.popularity.mrr:.3f}) -- {ratio:.2f}x the baseline hit-rate. "
            "Observational: this measures predictive fit on synthetic data, not "
            "the causal revenue an A/B test would."
        )


# --------------------------------------------------------------------------- #
# Building blocks (each pure and independently testable).
# --------------------------------------------------------------------------- #


def train_test_split_by_arrival(
    baskets: Sequence[Basket], train_fraction: float = 0.7
) -> tuple[list[frozenset[str]], list[frozenset[str]]]:
    """Split baskets into an earlier TRAIN slice and a later TEST slice.

    The split is contiguous in arrival order (a documented proxy for time), so no
    test basket can leak into the rules that recommend for it.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    n = len(baskets)
    cut = round(n * train_fraction)
    if cut < 1 or cut >= n:
        raise ValueError(f"train_fraction {train_fraction} leaves an empty split of {n}")
    pool = [frozenset(b) for b in baskets]
    return pool[:cut], pool[cut:]


def holdout_trials(
    test_baskets: Sequence[Basket], seed: int = 42, min_basket_size: int = 2
) -> list[HoldoutTrial]:
    """Build one leave-one-out trial per eligible test basket (deterministic).

    A basket qualifies when it has at least ``min_basket_size`` categories. For
    each, one category is hidden by a seeded random pick and the rest become the
    query. Baskets are processed in order, so the trials are reproducible.
    """
    if min_basket_size < 2:
        raise ValueError("min_basket_size must be at least 2")
    rng = np.random.default_rng(seed)
    trials: list[HoldoutTrial] = []
    for basket in test_baskets:
        items = sorted(basket)
        if len(items) < min_basket_size:
            continue
        held_out = items[int(rng.integers(0, len(items)))]
        query = frozenset(items) - {held_out}
        trials.append(HoldoutTrial(query=query, held_out=held_out))
    return trials


def popularity_ranking(train_baskets: Sequence[Basket]) -> list[str]:
    """Categories ranked by descending TRAIN frequency (name as tie-break)."""
    counts = Counter(item for basket in train_baskets for item in basket)
    return [item for item, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def popularity_predictions(
    ranking: Sequence[str], query: frozenset[str], max_k: int
) -> list[str]:
    """Top-``max_k`` popular categories not already in the query basket."""
    predictions: list[str] = []
    for category in ranking:
        if category not in query:
            predictions.append(category)
            if len(predictions) >= max_k:
                break
    return predictions


def score_recommender(
    name: str,
    predictions: Sequence[Sequence[str]],
    held_out_items: Sequence[str],
    k_values: Sequence[int],
) -> RecommenderMetrics:
    """Score ranked predictions against the held-out targets, trial by trial.

    ``predictions[i]`` is the ranked recommendation list for trial ``i`` and
    ``held_out_items[i]`` is that trial's hidden category. ``hit_rate@K`` counts
    the target appearing in the first ``K`` recommendations; ``MRR`` uses its rank
    within the top ``max(k_values)`` list (0 if absent); ``coverage`` is the
    fraction of trials with a non-empty list.
    """
    n = len(held_out_items)
    if n == 0:
        raise ValueError("need at least one trial to score")
    if len(predictions) != n:
        raise ValueError("predictions and held_out_items must align")
    ks = tuple(sorted({int(k) for k in k_values}))
    if not ks or ks[0] < 1:
        raise ValueError("k_values must be positive integers")
    max_k = ks[-1]
    hits = dict.fromkeys(ks, 0)
    reciprocal = 0.0
    covered = 0
    for preds, target in zip(predictions, held_out_items, strict=True):
        if preds:
            covered += 1
        for k in ks:
            if target in preds[:k]:
                hits[k] += 1
        for rank, category in enumerate(preds[:max_k], start=1):
            if category == target:
                reciprocal += 1.0 / rank
                break
    return RecommenderMetrics(
        name=name,
        n_trials=n,
        hit_rate={k: hits[k] / n for k in ks},
        mrr=reciprocal / n,
        coverage=covered / n,
    )


def evaluate_recommenders(
    baskets: Sequence[Basket] | None = None,
    *,
    train_fraction: float = 0.7,
    k_values: Sequence[int] = (1, 3, 5),
    min_support: float = 0.02,
    min_confidence: float = 0.30,
    min_lift: float = 1.10,
    seed: int = 42,
    train_baskets: Sequence[Basket] | None = None,
    trials: Sequence[HoldoutTrial] | None = None,
) -> EvaluationReport:
    """Back-test the rule recommender against a popularity baseline.

    Mines rules on the TRAIN slice (reusing :func:`basket.apriori.apriori` and
    :func:`basket.rules.generate_rules`), then scores both recommenders on the
    identical leave-one-out trials. Pass ``train_baskets`` and ``trials`` to
    evaluate a fixed, hand-built scenario; otherwise both are derived from
    ``baskets`` via the arrival-order split and seeded hold-out.
    """
    ks = tuple(sorted({int(k) for k in k_values}))
    if not ks or ks[0] < 1:
        raise ValueError("k_values must be positive integers")
    max_k = ks[-1]

    if train_baskets is None or trials is None:
        if baskets is None:
            raise ValueError("provide baskets, or both train_baskets and trials")
        train_baskets, test_baskets = train_test_split_by_arrival(baskets, train_fraction)
        trials = holdout_trials(test_baskets, seed=seed)
        n_test = len(test_baskets)
        n_baskets = len(baskets)
    else:
        train_baskets = [frozenset(b) for b in train_baskets]
        n_test = len(trials)
        n_baskets = len(train_baskets) + n_test
    trials = list(trials)
    if not trials:
        raise ValueError("no eligible test baskets (need at least two categories each)")

    itemsets = apriori(train_baskets, min_support=min_support, max_len=3)
    rules = generate_rules(
        itemsets, len(train_baskets), min_confidence=min_confidence, min_lift=min_lift
    )

    held = [trial.held_out for trial in trials]
    rule_predictions = [
        rank_cross_sell_categories(trial.query, rules, top_k=max_k) for trial in trials
    ]
    ranking = popularity_ranking(train_baskets)
    pop_predictions = [
        popularity_predictions(ranking, trial.query, max_k) for trial in trials
    ]

    rules_metrics = score_recommender("association rules", rule_predictions, held, ks)
    pop_metrics = score_recommender("popularity baseline", pop_predictions, held, ks)

    return EvaluationReport(
        n_baskets=n_baskets,
        n_train=len(train_baskets),
        n_test=n_test,
        n_trials=len(trials),
        train_fraction=train_fraction,
        k_values=ks,
        seed=seed,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        n_rules=len(rules),
        rules=rules_metrics,
        popularity=pop_metrics,
    )


# --------------------------------------------------------------------------- #
# Deterministic outputs: CSV + a hand-drawn SVG (no wall-clock, no RNG).
# --------------------------------------------------------------------------- #


def _metrics_row(report: EvaluationReport, metrics: RecommenderMetrics) -> list:
    row: list = [metrics.name, metrics.n_trials]
    row.extend(f"{metrics.hit_rate[k]:.4f}" for k in report.k_values)
    row.extend([f"{metrics.mrr:.4f}", f"{metrics.coverage:.4f}"])
    return row


def write_evaluation_csv(report: EvaluationReport, path: str) -> int:
    """Write the back-test metrics as UTF-8 CSV with LF terminators (deterministic)."""
    header = ["recommender", "n_trials"]
    header.extend(f"hit_rate_at_{k}" for k in report.k_values)
    header.extend(["mrr", "coverage"])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerow(_metrics_row(report, report.rules))
        writer.writerow(_metrics_row(report, report.popularity))
        ratio = report.hit_rate_ratio()
        ratio_row: list = ["hit_rate_ratio_rules_over_popularity", report.n_trials]
        ratio_row.extend(f"{ratio[k]:.4f}" for k in report.k_values)
        ratio_row.extend(
            [
                f"{(report.rules.mrr / report.popularity.mrr) if report.popularity.mrr else 0.0:.4f}",
                "",
            ]
        )
        writer.writerow(ratio_row)
    return os.path.getsize(path)


# SVG palette (matches the deliverables' light surface / stability chart).
_SVG_INK = "#0b0b0b"
_SVG_INK_SECONDARY = "#52514e"
_SVG_INK_MUTED = "#898781"
_SVG_SURFACE = "#fcfcfb"
_SVG_TRACK = "#f0efec"
_SVG_BASELINE = "#c3c2b7"
_SVG_RULES = "#2a78d6"  # blue: the rule recommender
_SVG_POP = "#898781"  # gray: popularity baseline


def _svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_evaluation_svg(report: EvaluationReport) -> str:
    """Build a deterministic, self-contained grouped-bar SVG of the back-test.

    One labelled pair of bars per metric (hit-rate@K for each K, then MRR): blue
    for the rule recommender, gray for the popularity baseline, each scaled to
    [0, 1]. No timestamps or randomness, so the bytes are identical every run.
    """
    # Metric rows: (label, rules value, popularity value, formatter).
    rows: list[tuple[str, float, float]] = []
    for k in report.k_values:
        rows.append((f"hit-rate@{k}", report.rules.hit_rate[k], report.popularity.hit_rate[k]))
    rows.append(("MRR", report.rules.mrr, report.popularity.mrr))

    width = 900
    margin_left = 36
    bar_x = 250
    bar_w = 380
    group_h = 54  # two bars + gap per metric
    top_pad = 116
    bottom_pad = 104
    height = top_pad + group_h * len(rows) + bottom_pad

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="{_SVG_SURFACE}"/>')
    parts.append(
        f'<text x="{margin_left}" y="40" font-size="20" font-weight="bold" '
        f'fill="{_SVG_INK}">Cross-sell recommender back-test</text>'
    )
    parts.append(
        f'<text x="{margin_left}" y="64" font-size="12.5" fill="{_SVG_INK_SECONDARY}">'
        f"Leave-one-out on {report.n_trials} held-out baskets: does the rule engine "
        f"recover a hidden category?</text>"
    )
    parts.append(
        f'<text x="{margin_left}" y="82" font-size="11" fill="{_SVG_INK_MUTED}">'
        f"Rules mined on {report.n_train} training baskets; scored against a "
        f"popularity baseline on the same trials. Synthetic seeded data.</text>"
    )
    # Legend.
    parts.append(
        f'<rect x="{margin_left}" y="94" width="12" height="12" rx="2" fill="{_SVG_RULES}"/>'
    )
    parts.append(
        f'<text x="{margin_left + 18}" y="104" font-size="11" '
        f'fill="{_SVG_INK_SECONDARY}">association rules</text>'
    )
    parts.append(
        f'<rect x="{margin_left + 160}" y="94" width="12" height="12" rx="2" fill="{_SVG_POP}"/>'
    )
    parts.append(
        f'<text x="{margin_left + 178}" y="104" font-size="11" '
        f'fill="{_SVG_INK_SECONDARY}">popularity baseline</text>'
    )

    for i, (label, rules_value, pop_value) in enumerate(rows):
        group_top = top_pad + i * group_h
        parts.append(
            f'<text x="{margin_left}" y="{group_top + 18}" font-size="12" '
            f'font-weight="bold" fill="{_SVG_INK}">{_svg_escape(label)}</text>'
        )
        for offset, (value, color) in enumerate(
            ((rules_value, _SVG_RULES), (pop_value, _SVG_POP))
        ):
            bar_y = group_top + offset * 20
            parts.append(
                f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="15" rx="3" '
                f'fill="{_SVG_TRACK}"/>'
            )
            value_w = round(bar_w * max(0.0, min(1.0, value)), 1)
            if value_w > 0:
                parts.append(
                    f'<rect x="{bar_x}" y="{bar_y}" width="{value_w}" height="15" rx="3" '
                    f'fill="{color}"/>'
                )
            parts.append(
                f'<text x="{bar_x + bar_w + 10}" y="{bar_y + 12}" font-size="10.5" '
                f'fill="{_SVG_INK_SECONDARY}">{value:.3f}</text>'
            )

    ratio = report.hit_rate_ratio()
    mid_k = report.k_values[len(report.k_values) // 2]
    caption_y = top_pad + group_h * len(rows) + 26
    parts.append(
        f'<line x1="{margin_left}" y1="{caption_y - 16}" x2="{width - margin_left}" '
        f'y2="{caption_y - 16}" stroke="{_SVG_BASELINE}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{margin_left}" y="{caption_y}" font-size="11" '
        f'fill="{_SVG_INK_SECONDARY}">Rule hit-rate@{mid_k} is {ratio[mid_k]:.2f}x the '
        f"popularity baseline; recommender coverage {report.rules.coverage:.0%}.</text>"
    )
    parts.append(
        f'<text x="{margin_left}" y="{caption_y + 22}" font-size="10" '
        f'fill="{_SVG_INK_MUTED}">SYNTHETIC DATA: all figures come from a seeded '
        f"simulation. Hit-rate is predictive fit, not the causal uplift of an A/B test.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_evaluation_svg(report: EvaluationReport, path: str) -> int:
    """Write the hand-drawn back-test SVG; returns the file size in bytes."""
    svg = render_evaluation_svg(report)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(svg)
        handle.write("\n")
    return os.path.getsize(path)
