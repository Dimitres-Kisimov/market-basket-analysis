"""Rule-stability / robustness check: do the top rules persist across splits?

This is a TRUST LAYER on top of the association rules. A single mining pass
produces confident-looking metrics, but some rules are backed by a real,
repeatable co-purchase pattern while others are artefacts of one particular
slice of the data. To tell them apart we re-run the EXACT SAME pipeline
(:func:`basket.apriori.apriori` -> :func:`basket.rules.generate_rules`, with
the same thresholds) on several subsets of the baskets and ask, for each of
the top-N rules by lift on the full data:

* in how many subsets does the rule reappear *above the same support,
  confidence and lift thresholds* (its ``stability_score``), and
* how much does its lift move across the subsets where it survives
  (``lift_mean`` / ``lift_std`` / ``lift_cv``)?

A rule that clears the thresholds in every subset is labelled ``stable``; one
that appears only sometimes is a candidate that should not be acted on without
more evidence.

Two split strategies are provided, both fully deterministic:

* :func:`time_window_splits` -- the baskets carry no timestamp, so we use
  basket arrival order (the ``basket_id`` generation sequence) as a *documented
  proxy for time* and cut it into ``k`` contiguous equal windows. This is the
  roadmap's "rule stability across time splits" check.
* :func:`bootstrap_splits` -- seeded resampling with replacement (``n_folds``
  folds, each the size of the input). An ordering-free statistical view of the
  same question.

Everything here runs on SYNTHETIC, seeded data and every number reported is
MEASURED, not assumed. One honest caveat worth stating up front: the synthetic
generator is stationary (it draws every basket from the same distribution), so
strong planted rules are *expected* to persist across time windows. The value
of this check on the synthetic data is (a) it validates the machinery and
(b) it still exposes borderline rules that flicker in and out near the
thresholds. On real order history -- where demand drifts and promotions come
and go -- the identical check would separate durable rules from seasonal or
one-off artefacts.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from basket import design
from basket.apriori import apriori
from basket.rules import Rule, format_itemset, generate_rules

Basket = frozenset


@dataclass(frozen=True)
class RuleStability:
    """Stability measurements for a single reference rule across the splits."""

    antecedent: frozenset[str]
    consequent: frozenset[str]
    reference_lift: float
    reference_confidence: float
    reference_support: float
    reference_support_count: int
    n_splits: int
    n_present: int  # splits where the rule cleared every threshold
    stability_score: float  # n_present / n_splits, in [0, 1]
    lift_mean: float  # mean lift over the splits where the rule is present
    lift_std: float  # population std of lift over present splits (0 if < 2)
    lift_cv: float  # lift_std / lift_mean (0 when lift_mean == 0)
    lift_min: float
    lift_max: float
    stable: bool  # stability_score >= stable_threshold

    @property
    def label(self) -> str:
        return f"{format_itemset(self.antecedent)} -> {format_itemset(self.consequent)}"


@dataclass
class StabilityReport:
    """A stability read-out for the top-N rules, plus the settings behind it."""

    method: str  # "time_window" or "bootstrap"
    n_splits: int
    top_n: int
    stable_threshold: float
    min_support: float
    min_confidence: float
    min_lift: float
    rules: list[RuleStability] = field(default_factory=list)

    @property
    def n_stable(self) -> int:
        return sum(1 for r in self.rules if r.stable)

    def plain_language(self) -> str:
        """A one-paragraph, jargon-light summary of the result."""
        method_phrase = (
            f"{self.n_splits} time windows"
            if self.method == "time_window"
            else f"{self.n_splits} bootstrap resamples"
        )
        n = len(self.rules)
        unstable = n - self.n_stable
        lead = (
            f"{self.n_stable} of the top {n} rules by lift are STABLE: they clear "
            f"support >= {self.min_support:.0%}, confidence >= "
            f"{self.min_confidence:.0%} and lift >= {self.min_lift:.2f} in every one "
            f"of {method_phrase}."
        )
        if unstable:
            lead += (
                f" The other {unstable} appear in only some splits, so treat them as "
                "candidates to validate, not established facts."
            )
        else:
            lead += " No top rule dropped out of any split."
        return lead


def time_window_splits(
    baskets: Sequence[Basket], k: int = 4
) -> list[list[frozenset[str]]]:
    """Cut ``baskets`` into ``k`` contiguous, near-equal windows by arrival order.

    The baskets have no timestamp, so their position in the list (the
    ``basket_id`` generation sequence) is used as a documented proxy for time.
    Windows are contiguous and cover the input exactly once.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    n = len(baskets)
    if n < k:
        raise ValueError(f"need at least k={k} baskets, got {n}")
    bounds = [round(i * n / k) for i in range(k + 1)]
    return [[frozenset(b) for b in baskets[bounds[i] : bounds[i + 1]]] for i in range(k)]


def bootstrap_splits(
    baskets: Sequence[Basket], n_folds: int = 5, seed: int = 42
) -> list[list[frozenset[str]]]:
    """Draw ``n_folds`` seeded bootstrap resamples (with replacement, full size).

    Deterministic given ``seed``: no wall-clock, no global RNG state.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    n = len(baskets)
    if n < 1:
        raise ValueError("need at least one basket")
    pool = [frozenset(b) for b in baskets]
    rng = np.random.default_rng(seed)
    folds: list[list[frozenset[str]]] = []
    for _ in range(n_folds):
        idx = rng.integers(0, n, size=n)
        folds.append([pool[i] for i in idx])
    return folds


def _rule_index(
    split: Sequence[frozenset[str]],
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> dict[tuple[frozenset[str], frozenset[str]], Rule]:
    """Mine one split with the SHARED pipeline; key rules by (antecedent, consequent)."""
    itemsets = apriori(split, min_support=min_support, max_len=3)
    rules = generate_rules(
        itemsets, len(split), min_confidence=min_confidence, min_lift=min_lift
    )
    return {(rule.antecedent, rule.consequent): rule for rule in rules}


def rule_stability(
    baskets: Sequence[Basket],
    reference_rules: Sequence[Rule],
    *,
    method: str = "time_window",
    n_splits: int = 4,
    top_n: int = 10,
    min_support: float = 0.02,
    min_confidence: float = 0.30,
    min_lift: float = 1.10,
    stable_threshold: float = 1.0,
    seed: int = 42,
    splits: Sequence[Sequence[frozenset[str]]] | None = None,
) -> StabilityReport:
    """Measure how well the top-``top_n`` reference rules persist across splits.

    ``reference_rules`` must already be ranked by lift (the output of
    :func:`basket.rules.generate_rules`); the first ``top_n`` are examined. Each
    split is mined with the identical thresholds, so a rule "appears" in a split
    only if it clears ``min_support``/``min_confidence``/``min_lift`` there.

    A rule is flagged ``stable`` when its ``stability_score`` (fraction of splits
    in which it appears) is at least ``stable_threshold`` (default 1.0 = every
    split).
    """
    if splits is None:
        if method == "time_window":
            splits = time_window_splits(baskets, k=n_splits)
        elif method == "bootstrap":
            splits = bootstrap_splits(baskets, n_folds=n_splits, seed=seed)
        else:
            raise ValueError("method must be 'time_window' or 'bootstrap'")
    n_effective = len(splits)
    if n_effective < 2:
        raise ValueError("need at least two splits")

    indices = [
        _rule_index(split, min_support, min_confidence, min_lift) for split in splits
    ]

    top = list(reference_rules[:top_n])
    results: list[RuleStability] = []
    for rule in top:
        key = (rule.antecedent, rule.consequent)
        lifts = [idx[key].lift for idx in indices if key in idx]
        n_present = len(lifts)
        score = n_present / n_effective
        if lifts:
            arr = np.asarray(lifts, dtype=float)
            lift_mean = float(arr.mean())
            lift_std = float(arr.std()) if n_present >= 2 else 0.0
            lift_min = float(arr.min())
            lift_max = float(arr.max())
            lift_cv = lift_std / lift_mean if lift_mean else 0.0
        else:
            lift_mean = lift_std = lift_min = lift_max = lift_cv = 0.0
        results.append(
            RuleStability(
                antecedent=rule.antecedent,
                consequent=rule.consequent,
                reference_lift=rule.lift,
                reference_confidence=rule.confidence,
                reference_support=rule.support,
                reference_support_count=rule.support_count,
                n_splits=n_effective,
                n_present=n_present,
                stability_score=score,
                lift_mean=lift_mean,
                lift_std=lift_std,
                lift_cv=lift_cv,
                lift_min=lift_min,
                lift_max=lift_max,
                stable=score >= stable_threshold,
            )
        )
    return StabilityReport(
        method=method,
        n_splits=n_effective,
        top_n=len(top),
        stable_threshold=stable_threshold,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        rules=results,
    )


# --------------------------------------------------------------------------- #
# Deterministic outputs: CSV + a hand-drawn SVG (no wall-clock, no RNG).
# --------------------------------------------------------------------------- #

_CSV_HEADER = (
    "rank",
    "rule",
    "antecedent",
    "consequent",
    "reference_lift",
    "reference_confidence",
    "reference_support",
    "reference_support_count",
    "n_splits",
    "n_present",
    "stability_score",
    "lift_mean",
    "lift_std",
    "lift_cv",
    "lift_min",
    "lift_max",
    "stable",
)


def write_stability_csv(report: StabilityReport, path: str) -> int:
    """Write the stability table as UTF-8 CSV with LF terminators (deterministic)."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_CSV_HEADER)
        for rank, item in enumerate(report.rules, start=1):
            writer.writerow(
                [
                    rank,
                    item.label,
                    format_itemset(item.antecedent),
                    format_itemset(item.consequent),
                    f"{item.reference_lift:.4f}",
                    f"{item.reference_confidence:.4f}",
                    f"{item.reference_support:.4f}",
                    item.reference_support_count,
                    item.n_splits,
                    item.n_present,
                    f"{item.stability_score:.4f}",
                    f"{item.lift_mean:.4f}",
                    f"{item.lift_std:.4f}",
                    f"{item.lift_cv:.4f}",
                    f"{item.lift_min:.4f}",
                    f"{item.lift_max:.4f}",
                    "yes" if item.stable else "no",
                ]
            )
    return os.path.getsize(path)


# Visual system: shared category-atlas tokens (see basket/design.py). The
# verdict is a STATUS (good vs needs-a-look), so the bars wear the reserved
# status colours -- never the categorical palette -- and the warning fill
# carries a tone-on-tone hatch so the verdict is never colour alone (each row
# also prints its n/n split fraction).
_svg_escape = design.svg_escape

_HATCH_ID = "atlas-hatch-warning"


def _hatch_defs() -> str:
    """45-degree tone-on-tone hatch for the window-specific (warning) bars."""
    return (
        f'<defs><pattern id="{_HATCH_ID}" patternUnits="userSpaceOnUse" '
        f'width="6" height="6" patternTransform="rotate(45)">'
        f'<rect width="6" height="6" fill="{design.STATUS_WARNING}"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" '
        f'stroke="{design.STATUS_WARNING_DEEP}" stroke-width="1.6"/>'
        f"</pattern></defs>"
    )


def _truncate(text: str, limit: int = 46) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_stability_svg(report: StabilityReport) -> str:
    """Build a deterministic, self-contained hand-drawn SVG bar chart.

    One row per top rule; bar length = stability_score (0..1); green = stable
    across every split, amber = window-specific. No timestamps or randomness,
    so the bytes are identical on every run.
    """
    width = 900
    margin_left = design.SVG_MARGIN
    bar_x = 430
    bar_w = 300
    row_h = 30
    rows = report.rules

    method_phrase = (
        f"{report.n_splits} time windows"
        if report.method == "time_window"
        else f"{report.n_splits} bootstrap resamples"
    )
    parts = design.svg_open(width, 10)  # height patched at the end
    parts.append(_hatch_defs())
    header_bottom = design.svg_plate_header(
        parts,
        width=width,
        plate="stability",
        title=f"Rule stability across {_svg_escape(method_phrase)}",
        subtitle=(
            f"{report.n_stable} of the top {len(rows)} rules by lift clear support/"
            f"confidence/lift in every split."
        ),
        note=(
            f"Thresholds: support &gt;= {report.min_support:.0%}, confidence &gt;= "
            f"{report.min_confidence:.0%}, lift &gt;= {report.min_lift:.2f}. "
            f"Synthetic seeded data."
        ),
    )
    top_pad = header_bottom + 16
    # Column headers.
    parts.append(
        f'<text x="{margin_left}" y="{top_pad - 8}" font-size="10.5" '
        f'font-weight="bold" fill="{design.INK_SECONDARY}">rule (antecedent -&gt; '
        f"consequent)</text>"
    )
    parts.append(
        f'<text x="{bar_x}" y="{top_pad - 8}" font-size="10.5" font-weight="bold" '
        f'fill="{design.INK_SECONDARY}">stability score</text>'
    )

    for i, item in enumerate(rows):
        row_top = top_pad + i * row_h
        text_y = row_top + 18
        fill = design.STATUS_GOOD if item.stable else f"url(#{_HATCH_ID})"
        label = _svg_escape(_truncate(item.label))
        parts.append(
            f'<text x="{margin_left}" y="{text_y}" font-size="11.5" '
            f'fill="{design.INK}">{label}</text>'
        )
        parts.append(
            f'<rect x="{bar_x}" y="{row_top + 5}" width="{bar_w}" height="16" '
            f'rx="3" fill="{design.NEUTRAL_FILL}"/>'
        )
        value_w = round(bar_w * item.stability_score, 1)
        if value_w > 0:
            parts.append(
                f'<rect x="{bar_x}" y="{row_top + 5}" width="{value_w}" height="16" '
                f'rx="3" fill="{fill}"/>'
            )
        parts.append(
            f'<text x="{bar_x + bar_w + 10}" y="{text_y}" font-size="11" '
            f'fill="{design.INK_SECONDARY}">{item.stability_score:.0%} '
            f"({item.n_present}/{item.n_splits}) &#183; lift "
            f"{item.reference_lift:.2f}</text>"
        )

    # Legend (status = icon + label, never colour alone) + disclaimer.
    legend_y = top_pad + row_h * len(rows) + 24
    parts.append(
        f'<rect x="{margin_left}" y="{legend_y - 11}" width="14" height="14" rx="3" '
        f'fill="{design.STATUS_GOOD}"/>'
    )
    parts.append(
        f'<text x="{margin_left + 7}" y="{legend_y}" font-size="10" '
        f'text-anchor="middle" fill="{design.SURFACE}">&#10003;</text>'
    )
    parts.append(
        f'<text x="{margin_left + 22}" y="{legend_y}" font-size="11" '
        f'fill="{design.INK_SECONDARY}">stable (every split)</text>'
    )
    parts.append(
        f'<rect x="{margin_left + 180}" y="{legend_y - 11}" width="14" height="14" '
        f'rx="3" fill="url(#{_HATCH_ID})"/>'
    )
    parts.append(
        f'<text x="{margin_left + 187}" y="{legend_y}" font-size="10" '
        f'font-weight="bold" text-anchor="middle" fill="{design.INK}">!</text>'
    )
    parts.append(
        f'<text x="{margin_left + 202}" y="{legend_y}" font-size="11" '
        f'fill="{design.INK_SECONDARY}">window-specific</text>'
    )
    height = design.svg_footer_caption(
        parts,
        width=width,
        y=legend_y + 16,
        text=(
            "SYNTHETIC DATA: all figures come from a seeded simulation. Lift is "
            "observational -- correlation is not causation."
        ),
    ) + 10
    return design.svg_close(parts, width=width, height=height)


def write_stability_svg(report: StabilityReport, path: str) -> int:
    """Write the hand-drawn SVG; returns the file size in bytes."""
    svg = render_stability_svg(report)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(svg)
        handle.write("\n")
    return os.path.getsize(path)
