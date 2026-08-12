"""Rule redundancy: closed/maximal itemsets and a minimal non-redundant rule set.

Association mining is repetitive by construction: if ``fasteners ->
power_tools`` holds, then ``fasteners + ppe_gloves -> power_tools`` usually
holds too, and a rule list ranked by lift happily shows both. This module is
the RIGOR LAYER that separates rules carrying new information from rules that
merely restate a simpler one, using two classic ideas:

* CLOSED and MAXIMAL frequent itemsets (Pasquier et al. 1999; Bayardo 1998).
  An itemset is *closed* when no mined superset has the same support -- when it
  is not implied "for free" by a larger pattern -- and *maximal* when no mined
  superset is frequent at all. The closed itemsets are a LOSSLESS summary of
  the frequent-itemset collection (every itemset's support is the support of
  its smallest closed superset); the maximal itemsets are the smaller, lossy
  skyline.
* CONFIDENCE IMPROVEMENT (Bayardo, Agrawal & Gunopulos 1999). The improvement
  of ``X -> Y`` is its confidence minus the best confidence of any simpler
  rule ``X' -> Y`` with ``X'`` a proper subset of ``X`` -- including the empty
  antecedent, whose "confidence" is the baseline ``P(Y)``. A rule with
  improvement <= 0 is REDUNDANT: dropping the extra condition loses nothing,
  because the simpler rule fires on strictly more baskets at least as
  confidently. Keeping only the rules with positive improvement yields a
  minimal non-redundant action list.

Improvement is computed from the mined supports directly, so every simpler
generalisation is considered -- whether or not it survived the run's own
confidence/lift filters. (For a shared consequent, lift is proportional to
confidence, so any nonempty covering rule in fact clears both filters too and
is present in the kept list; only the empty-antecedent baseline is not a rule.)

Everything reuses the shared pipeline -- the itemset collection from
Apriori/FP-growth and the rule list from :mod:`basket.rules` -- and is fully
deterministic: no RNG, no wall clock, and ties in the covering-rule search
resolve to the simplest, lexicographically first alternative.

Honesty notes, repeated in the deliverables: the data is SYNTHETIC and seeded;
redundancy is a statement about information content, not causality; and
closure is relative to the MINED collection (itemsets are mined up to
``max_len`` items, so "no superset" means "no superset within that cap").
"""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from basket import design
from basket.rules import Rule, format_itemset, format_rule


def _support_counts(
    itemsets: Mapping[frozenset[str], float], n_baskets: int
) -> dict[frozenset[str], int]:
    """Recover integer basket counts from support fractions (exact for count/n input)."""
    if n_baskets < 1:
        raise ValueError("n_baskets must be positive")
    return {itemset: round(support * n_baskets) for itemset, support in itemsets.items()}


def closed_itemsets(
    itemsets: Mapping[frozenset[str], float], n_baskets: int
) -> set[frozenset[str]]:
    """The closed itemsets: no proper superset IN THE COLLECTION has the same support.

    Supports are compared as absolute basket counts (recovered exactly from the
    count/n fractions that ``apriori``/``fpgrowth`` emit), so float noise cannot
    split or merge equivalence classes. Closure is relative to the mined
    collection: itemsets are mined up to ``max_len`` items, so "no superset"
    means "no superset within that cap".
    """
    counts = _support_counts(itemsets, n_baskets)
    return {
        itemset
        for itemset, count in counts.items()
        if not any(
            itemset < other and other_count == count
            for other, other_count in counts.items()
        )
    }


def maximal_itemsets(itemsets: Mapping[frozenset[str], float]) -> set[frozenset[str]]:
    """The maximal itemsets: no proper superset in the collection at all."""
    collection = list(itemsets)
    return {
        itemset
        for itemset in collection
        if not any(itemset < other for other in collection)
    }


def rule_improvement(
    rule: Rule, itemsets: Mapping[frozenset[str], float]
) -> tuple[float, frozenset[str], float]:
    """Confidence improvement of ``rule`` over every simpler rule with its consequent.

    Returns ``(improvement, best_antecedent, best_confidence)`` where
    ``improvement = confidence(rule) - best_confidence`` and
    ``best_antecedent`` is the proper subset of the rule's antecedent whose
    rule toward the same consequent is most confident. The empty antecedent
    (the baseline ``P(consequent)``, recovered from the rule itself) is always
    a candidate, so the result is total. Candidates are scanned smallest
    antecedent first, lexicographically within a size, and only a STRICTLY
    higher confidence replaces the incumbent -- so ties resolve to the
    simplest, alphabetically first covering rule, deterministically.

    Every needed support is a subset of the rule's own itemset, so for
    downward-closed input (the output of ``apriori``/``fpgrowth``) every
    nonempty candidate is evaluable; missing entries are skipped defensively.
    """
    best_antecedent: frozenset[str] = frozenset()
    best_confidence = rule.consequent_support  # empty antecedent: baseline P(Y)
    items = sorted(rule.antecedent)
    for size in range(1, len(items)):
        for subset_items in combinations(items, size):
            subset = frozenset(subset_items)
            support_subset = itemsets.get(subset)
            support_joint = itemsets.get(subset | rule.consequent)
            if support_subset is None or support_joint is None:
                continue  # cannot happen for downward-closed input; guard anyway
            confidence = support_joint / support_subset
            if confidence > best_confidence:
                best_confidence = confidence
                best_antecedent = subset
    return rule.confidence - best_confidence, best_antecedent, best_confidence


@dataclass(frozen=True)
class RuleVerdict:
    """One rule's redundancy verdict with the evidence behind it."""

    rule: Rule
    improvement: float  # confidence minus the best simpler-rule confidence
    best_alternative_antecedent: frozenset[str]  # empty = the baseline P(consequent)
    best_alternative_confidence: float
    redundant: bool  # improvement <= tol: a simpler rule says it at least as well

    @property
    def label(self) -> str:
        return format_rule(self.rule)

    @property
    def best_alternative_label(self) -> str:
        """The covering rule, or the baseline when no nonempty subset beats it."""
        consequent = format_itemset(self.rule.consequent)
        if not self.best_alternative_antecedent:
            return f"baseline P({consequent})"
        return f"{format_itemset(self.best_alternative_antecedent)} -> {consequent}"


@dataclass
class RedundancyReport:
    """The redundancy read-out for a rule list, plus the settings behind it."""

    n_baskets: int
    tol: float
    n_itemsets: int
    closed: set[frozenset[str]] = field(default_factory=set)
    maximal: set[frozenset[str]] = field(default_factory=set)
    verdicts: list[RuleVerdict] = field(default_factory=list)  # input (lift-rank) order

    @property
    def n_closed(self) -> int:
        return len(self.closed)

    @property
    def n_maximal(self) -> int:
        return len(self.maximal)

    @property
    def n_rules(self) -> int:
        return len(self.verdicts)

    @property
    def n_redundant(self) -> int:
        return sum(1 for v in self.verdicts if v.redundant)

    @property
    def n_kept(self) -> int:
        return self.n_rules - self.n_redundant

    @property
    def kept(self) -> list[RuleVerdict]:
        """Non-redundant verdicts in the input (lift-ranked) order."""
        return [v for v in self.verdicts if not v.redundant]

    @property
    def pruned(self) -> list[RuleVerdict]:
        """Redundant verdicts in the input (lift-ranked) order."""
        return [v for v in self.verdicts if v.redundant]

    def kept_by_improvement(self) -> list[RuleVerdict]:
        """Non-redundant verdicts ranked by what the extra condition buys."""
        return sorted(self.kept, key=lambda v: (-v.improvement, v.label))

    def plain_language(self) -> str:
        """A one-paragraph, jargon-light summary of the result."""
        lead = (
            f"{self.n_redundant} of the {self.n_rules} rules are REDUNDANT: a "
            f"simpler rule with the same consequent already predicts at least as "
            f"confidently, so the {self.n_kept} rules with positive confidence "
            f"improvement carry all of the list's information."
        )
        if self.n_redundant:
            top = self.pruned[0]  # highest-lift pruned rule (input is lift-ranked)
            lead += (
                f" Highest-lift pruned rule: {top.label} (lift "
                f"{top.rule.lift:.2f}) -- covered by {top.best_alternative_label} "
                f"at {top.best_alternative_confidence:.0%} confidence."
            )
        lead += (
            f" Of the {self.n_itemsets} frequent itemsets, {self.n_closed} are "
            f"closed (a lossless summary of all supports) and {self.n_maximal} "
            f"are maximal. Redundancy is about information content on synthetic "
            "data -- correlation, not causation."
        )
        return lead


def rule_redundancy(
    itemsets: Mapping[frozenset[str], float],
    rules: Sequence[Rule],
    n_baskets: int,
    *,
    tol: float = 1e-9,
) -> RedundancyReport:
    """Score every rule's confidence improvement and classify the itemsets.

    ``itemsets`` and ``rules`` are the outputs of the shared pipeline
    (``apriori``/``fpgrowth`` -> ``generate_rules``); the verdict list keeps
    the input rule order (ranked by lift). A rule is ``redundant`` when its
    improvement is at most ``tol`` -- the tolerance absorbs float noise in
    exact ties, which are the common redundant case. Fully deterministic.
    """
    closed = closed_itemsets(itemsets, n_baskets)
    maximal = maximal_itemsets(itemsets)
    verdicts: list[RuleVerdict] = []
    for rule in rules:
        improvement, best_antecedent, best_confidence = rule_improvement(rule, itemsets)
        verdicts.append(
            RuleVerdict(
                rule=rule,
                improvement=improvement,
                best_alternative_antecedent=best_antecedent,
                best_alternative_confidence=best_confidence,
                redundant=improvement <= tol,
            )
        )
    return RedundancyReport(
        n_baskets=n_baskets,
        tol=tol,
        n_itemsets=len(itemsets),
        closed=closed,
        maximal=maximal,
        verdicts=verdicts,
    )


# --------------------------------------------------------------------------- #
# Deterministic outputs: CSV + a hand-drawn SVG (no wall-clock, no RNG).
# --------------------------------------------------------------------------- #

_CSV_HEADER = (
    "rank",
    "rule",
    "antecedent",
    "consequent",
    "support",
    "support_count",
    "confidence",
    "lift",
    "improvement",
    "verdict",
    "best_simpler_rule",
    "best_simpler_confidence",
)


def write_redundancy_csv(report: RedundancyReport, path: str) -> int:
    """Write the verdict table as UTF-8 CSV with LF terminators (deterministic)."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_CSV_HEADER)
        for rank, verdict in enumerate(report.verdicts, start=1):
            rule = verdict.rule
            writer.writerow(
                [
                    rank,
                    verdict.label,
                    format_itemset(rule.antecedent),
                    format_itemset(rule.consequent),
                    f"{rule.support:.4f}",
                    rule.support_count,
                    f"{rule.confidence:.4f}",
                    f"{rule.lift:.4f}",
                    f"{verdict.improvement:.4f}",
                    "redundant" if verdict.redundant else "kept",
                    verdict.best_alternative_label,
                    f"{verdict.best_alternative_confidence:.4f}",
                ]
            )
    return os.path.getsize(path)


# Visual system: shared category-atlas tokens (see basket/design.py). The
# funnel stages are ordinal, so each group takes monotone lightness steps of
# the single blue hue (validated with the palette validator's --ordinal mode).
_svg_escape = design.svg_escape


def _truncate(text: str, limit: int = 46) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_redundancy_svg(report: RedundancyReport) -> str:
    """Build a deterministic, self-contained hand-drawn funnel SVG.

    Two bar groups (rules mined -> non-redundant; itemsets -> closed ->
    maximal; bar length = count, direct-labelled) followed by the
    highest-lift pruned rules, each naming the simpler rule that covers it.
    No timestamps or randomness, so the bytes are identical on every run.
    """
    width = 900
    margin = design.SVG_MARGIN
    bar_x = 430
    bar_w = 300
    row_h = 30
    group_gap = 14

    rule_rows = [
        ("mined at the run's thresholds", report.n_rules),
        ("non-redundant (improvement > 0)", report.n_kept),
    ]
    itemset_rows = [
        ("frequent (mined, up to 3 items)", report.n_itemsets),
        ("closed (lossless summary)", report.n_closed),
        ("maximal (skyline)", report.n_maximal),
    ]
    pruned_shown = report.pruned[:8]

    parts = design.svg_open(width, 10)  # height patched at the end
    top_pad = design.svg_plate_header(
        parts,
        width=width,
        plate="redundancy",
        title=(
            f"Minimal non-redundant rule set: {report.n_kept} of "
            f"{report.n_rules} rules carry all the information"
        ),
        subtitle=(
            "A rule is redundant when a simpler rule with the same consequent "
            "predicts at least as confidently (confidence improvement &lt;= 0)."
        ),
        note=(
            "Improvement is measured against every simpler generalisation, "
            "including the baseline P(consequent). Synthetic seeded data."
        ),
    )
    rules_top = top_pad + 18
    itemsets_top = rules_top + len(rule_rows) * row_h + group_gap + 24
    pruned_top = itemsets_top + len(itemset_rows) * row_h + group_gap + 24
    pruned_h = 18 * len(pruned_shown) if pruned_shown else 18

    def bar_group(title: str, rows: list[tuple[str, int]], group_top: int) -> None:
        parts.append(
            f'<text x="{margin}" y="{group_top - 8}" font-size="12" '
            f'font-weight="bold" fill="{design.INK}">{_svg_escape(title)}</text>'
        )
        group_max = max((count for _, count in rows), default=0)
        # Funnel stages are ordered, so they take an ordinal lightness ramp of
        # the single blue hue: light at the wide end, dark at the distilled end.
        ramp = design.member_ramp(design.SERIES_BLUE, len(rows))
        for i, (label, count) in enumerate(rows):
            row_top = group_top + i * row_h
            text_y = row_top + 18
            parts.append(
                f'<text x="{margin}" y="{text_y}" font-size="11.5" '
                f'fill="{design.INK}">{_svg_escape(label)}</text>'
            )
            parts.append(
                f'<rect x="{bar_x}" y="{row_top + 5}" width="{bar_w}" height="16" '
                f'rx="3" fill="{design.NEUTRAL_FILL}"/>'
            )
            value_w = round(bar_w * count / group_max, 1) if group_max else 0.0
            if value_w > 0:
                parts.append(
                    f'<rect x="{bar_x}" y="{row_top + 5}" width="{value_w}" '
                    f'height="16" rx="3" fill="{ramp[i]}"/>'
                )
            parts.append(
                f'<text x="{bar_x + bar_w + 10}" y="{text_y}" font-size="11" '
                f'fill="{design.INK_SECONDARY}">{count}</text>'
            )

    bar_group("Association rules", rule_rows, rules_top)
    bar_group("Frequent itemsets (closure of the collection)", itemset_rows, itemsets_top)

    parts.append(
        f'<text x="{margin}" y="{pruned_top - 8}" font-size="12" font-weight="bold" '
        f'fill="{design.INK}">Highest-lift pruned rules and the simpler rule that '
        f"covers each</text>"
    )
    if pruned_shown:
        for i, verdict in enumerate(pruned_shown):
            line_y = pruned_top + 12 + 18 * i
            parts.append(
                f'<text x="{margin}" y="{line_y}" font-size="11" '
                f'fill="{design.INK_SECONDARY}">{_svg_escape(_truncate(verdict.label))} '
                f"&#183; lift {verdict.rule.lift:.2f} &#183; conf "
                f"{verdict.rule.confidence:.0%} &#8212; covered by "
                f"{_svg_escape(verdict.best_alternative_label)} at "
                f"{verdict.best_alternative_confidence:.0%}</text>"
            )
    else:
        parts.append(
            f'<text x="{margin}" y="{pruned_top + 12}" font-size="11" '
            f'fill="{design.INK_SECONDARY}">None -- every rule clears positive '
            f"improvement at these thresholds.</text>"
        )

    height = design.svg_footer_caption(
        parts,
        width=width,
        y=pruned_top + pruned_h + 18,
        text=(
            "SYNTHETIC DATA: all figures come from a seeded simulation. Redundancy "
            "is about information content -- correlation is not causation."
        ),
    ) + 10
    return design.svg_close(parts, width=width, height=height)


def write_redundancy_svg(report: RedundancyReport, path: str) -> int:
    """Write the hand-drawn funnel SVG; returns the file size in bytes."""
    svg = render_redundancy_svg(report)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(svg)
        handle.write("\n")
    return os.path.getsize(path)
