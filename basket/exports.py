"""Executive deliverables: a PDF briefing (matplotlib) and an Excel workbook.

Headless-safe by construction: only the matplotlib object-oriented API is used
(``matplotlib.figure.Figure`` + ``PdfPages``), never ``pyplot``, so no GUI
backend is touched. Every page and sheet repeats the synthetic-data disclaimer
and the correlation-is-not-causation note.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from basket.apriori import apriori
from basket.data import AVG_LINE_VALUE_EUR, baskets_from_transactions, generate_transactions
from basket.recommend import ASSUMPTION_NOTE, Recommendation, cross_sell_recommendations
from basket.rules import Rule, format_itemset, format_rule, generate_rules
from basket.segment import Segmentation, segment_customers

DISCLAIMER = (
    "SYNTHETIC DATA: all figures come from a seeded simulation built for this "
    "portfolio project. No real customer or sales data was used."
)
CAUSATION_NOTE = (
    "Lift is observational: correlation is not causation. Validate any "
    "cross-sell action with a controlled test before scaling it."
)

# Chart chrome and palette (validated defaults; light surface).
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SURFACE = "#fcfcfb"
_NEUTRAL_FILL = "#f0efec"
_SERIES = ("#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834")
_DIVERGING = ("#2a78d6", "#f0efec", "#e34948")  # blue <-> gray(=lift 1) <-> red

_A4_LANDSCAPE = (11.69, 8.27)


@dataclass
class AnalysisResult:
    transactions: pd.DataFrame
    baskets: list[frozenset[str]]
    itemsets: dict[frozenset[str], float]
    rules: list[Rule]
    segmentation: Segmentation
    recommendations: list[Recommendation]
    n_baskets: int
    seed: int
    min_support: float
    min_confidence: float
    min_lift: float


def run_analysis(
    n_baskets: int = 6000,
    seed: int = 42,
    min_support: float = 0.02,
    min_confidence: float = 0.30,
    min_lift: float = 1.10,
    k_segments: int = 3,
) -> AnalysisResult:
    """Run the full pipeline: generate -> mine -> rules -> segment -> recommend."""
    transactions = generate_transactions(n_baskets=n_baskets, seed=seed)
    baskets = baskets_from_transactions(transactions)
    itemsets = apriori(baskets, min_support=min_support, max_len=3)
    rules = generate_rules(
        itemsets, len(baskets), min_confidence=min_confidence, min_lift=min_lift
    )
    segmentation = segment_customers(transactions, k=k_segments, seed=seed)
    recommendations = cross_sell_recommendations(rules, AVG_LINE_VALUE_EUR, top_n=10)
    return AnalysisResult(
        transactions=transactions,
        baskets=baskets,
        itemsets=itemsets,
        rules=rules,
        segmentation=segmentation,
        recommendations=recommendations,
        n_baskets=len(baskets),
        seed=seed,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )


def _pair_lift_matrix(
    baskets: list[frozenset[str]], categories: list[str]
) -> np.ndarray:
    """Full category-pair lift matrix (diagonal masked as NaN)."""
    n = len(baskets)
    index = {c: i for i, c in enumerate(categories)}
    presence = np.zeros((n, len(categories)))
    for row, basket in enumerate(baskets):
        for item in basket:
            if item in index:
                presence[row, index[item]] = 1.0
    joint = (presence.T @ presence) / n
    marginal = np.diag(joint).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = joint / np.outer(marginal, marginal)
    lift[~np.isfinite(lift)] = np.nan
    np.fill_diagonal(lift, np.nan)
    return lift


def _new_page(title: str, subtitle: str) -> Figure:
    fig = Figure(figsize=_A4_LANDSCAPE, facecolor=_SURFACE)
    fig.text(0.06, 0.94, title, fontsize=17, fontweight="bold", color=_INK)
    fig.text(0.06, 0.905, subtitle, fontsize=9.5, color=_INK_SECONDARY)
    fig.text(0.06, 0.035, DISCLAIMER, fontsize=7.5, color=_INK_MUTED)
    return fig


def _cover_page(pdf: PdfPages, result: AnalysisResult) -> None:
    fig = Figure(figsize=_A4_LANDSCAPE, facecolor=_SURFACE)
    fig.text(
        0.08, 0.80, "Cross-Sell Opportunity Analysis",
        fontsize=26, fontweight="bold", color=_INK,
    )
    fig.text(
        0.08, 0.745,
        "Market-basket mining for a B2B maintenance & construction supplies distributor",
        fontsize=13, color=_INK_SECONDARY,
    )
    fig.text(
        0.08, 0.705,
        f"Generated {date.today().isoformat()}  |  seed {result.seed}  |  "
        f"min support {result.min_support:.0%}  |  min confidence "
        f"{result.min_confidence:.0%}  |  min lift {result.min_lift:.2f}",
        fontsize=9, color=_INK_MUTED,
    )

    top = result.rules[0] if result.rules else None
    tiles = [
        (f"{result.n_baskets:,}", "orders analysed"),
        (f"{len(result.itemsets):,}", "frequent itemsets"),
        (f"{len(result.rules):,}", "rules kept"),
        (f"{top.lift:.1f}x" if top else "-", "strongest lift"),
    ]
    for i, (value, label) in enumerate(tiles):
        x = 0.08 + i * 0.22
        fig.text(x, 0.565, value, fontsize=24, fontweight="bold", color=_INK)
        fig.text(x, 0.525, label, fontsize=10, color=_INK_SECONDARY)

    if top is not None:
        fig.text(
            0.08, 0.43,
            f"Headline: {format_rule(top)}  (lift {top.lift:.2f}, "
            f"confidence {top.confidence:.0%}, {top.support_count} orders)",
            fontsize=12, color=_INK,
        )

    fig.text(
        0.08, 0.30, DISCLAIMER + "\n" + CAUSATION_NOTE,
        fontsize=10, color=_INK,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": _NEUTRAL_FILL,
            "edgecolor": _BASELINE,
        },
        va="top", wrap=True,
    )
    fig.text(0.08, 0.09, "Author: Dimitres Kisimov", fontsize=9, color=_INK_MUTED)
    pdf.savefig(fig)


def _rules_page(pdf: PdfPages, result: AnalysisResult, max_rows: int = 12) -> None:
    fig = _new_page(
        "Top association rules (ranked by lift)",
        f"Filtered at confidence >= {result.min_confidence:.0%} and lift >= "
        f"{result.min_lift:.2f}. Lift is observational -- correlation is not causation.",
    )
    ax = fig.add_axes((0.05, 0.10, 0.90, 0.76))
    ax.axis("off")
    header = ["Rule", "Support", "Orders", "Confidence", "Lift", "Conviction", "Note"]
    body = []
    for rule in result.rules[:max_rows]:
        conviction = "inf" if rule.conviction == float("inf") else f"{rule.conviction:.2f}"
        body.append(
            [
                format_rule(rule),
                f"{rule.support:.1%}",
                f"{rule.support_count}",
                f"{rule.confidence:.1%}",
                f"{rule.lift:.2f}",
                conviction,
                "thin support" if rule.thin_support else "",
            ]
        )
    table = ax.table(
        cellText=body,
        colLabels=header,
        loc="upper center",
        cellLoc="center",
        colWidths=[0.40, 0.09, 0.08, 0.11, 0.08, 0.11, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(_GRID)
        if row == 0:
            cell.set_text_props(fontweight="bold", color=_INK)
            cell.set_facecolor(_NEUTRAL_FILL)
        else:
            cell.set_text_props(color=_INK_SECONDARY)
            cell.set_facecolor(_SURFACE)
    pdf.savefig(fig)


def _heatmap_page(pdf: PdfPages, result: AnalysisResult) -> None:
    categories = sorted({item for basket in result.baskets for item in basket})
    lift = _pair_lift_matrix(result.baskets, categories)
    finite = lift[np.isfinite(lift)]
    vmin = min(0.85, float(finite.min())) if finite.size else 0.5
    vmax = max(1.15, float(finite.max())) if finite.size else 1.5

    fig = _new_page(
        "Category-pair lift heatmap",
        "Gray = independence (lift 1.0), red = bought together more than chance, "
        "blue = less. Diagonal masked.",
    )
    ax = fig.add_axes((0.16, 0.16, 0.62, 0.68))
    cmap = LinearSegmentedColormap.from_list("lift_diverging", _DIVERGING)
    cmap.set_bad(_SURFACE)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    image = ax.imshow(lift, cmap=cmap, norm=norm)

    labels = [c.replace("_", " ") for c in categories]
    ax.set_xticks(range(len(categories)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(categories)), labels, fontsize=7)
    ax.tick_params(colors=_INK_MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_color(_BASELINE)

    # Selective direct labels: only clearly non-independent pairs get a number.
    for i in range(len(categories)):
        for j in range(len(categories)):
            value = lift[i, j]
            if np.isfinite(value) and (value >= 1.5 or value <= 0.6):
                position = norm(value)
                ax.text(
                    j, i, f"{value:.1f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if (position >= 0.75 or position <= 0.25) else _INK,
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("lift", color=_INK_SECONDARY, fontsize=9)
    colorbar.ax.tick_params(colors=_INK_MUTED, labelsize=7)
    colorbar.outline.set_edgecolor(_BASELINE)
    pdf.savefig(fig)


def _segments_page(pdf: PdfPages, result: AnalysisResult) -> None:
    segmentation = result.segmentation
    k = len(segmentation.profiles)
    fig = _new_page(
        "Customer segments (k-means on spend shares)",
        "Descriptive clustering of per-customer category spend mix; "
        "bar length = share of segment spend (centroid).",
    )
    spend_columns = list(segmentation.spend.columns)
    slot_width = 0.90 / k
    for j, profile in enumerate(segmentation.profiles):
        # Wide gutters so category labels never collide with a neighbour's bars.
        ax = fig.add_axes((0.115 + j * slot_width, 0.18, slot_width - 0.105, 0.55))
        centroid = segmentation.kmeans_result.centroids[j]
        top_idx = np.argsort(-centroid)[:6][::-1]
        names = [str(spend_columns[i]).replace("_", " ") for i in top_idx]
        shares = centroid[top_idx]
        bars = ax.barh(names, shares, color=_SERIES[j % len(_SERIES)], height=0.55)
        for bar, share in zip(bars, shares, strict=True):
            ax.text(
                bar.get_width() + max(shares) * 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{share:.0%}", va="center", fontsize=7.5, color=_INK_SECONDARY,
            )
        ax.set_xlim(0, max(shares) * 1.25)
        ax.set_title(
            f"Segment {profile.segment}\n{profile.n_customers} customers "
            f"({profile.share_of_customers:.0%}), avg spend "
            f"EUR {profile.avg_total_spend_eur:,.0f}",
            fontsize=9, color=_INK, loc="left",
        )
        ax.tick_params(colors=_INK_MUTED, labelsize=7.5, length=0)
        ax.set_xticks([])
        for name, spine in ax.spines.items():
            spine.set_visible(name == "left")
            spine.set_color(_BASELINE)
    pdf.savefig(fig)


def export_pdf(result: AnalysisResult, path: str) -> None:
    """Write the four-page executive PDF."""
    with PdfPages(path) as pdf:
        _cover_page(pdf, result)
        _rules_page(pdf, result)
        _heatmap_page(pdf, result)
        _segments_page(pdf, result)


def _style_header(sheet: Worksheet, row: int, n_columns: int) -> None:
    fill = PatternFill("solid", fgColor="F0EFEC")
    for column in range(1, n_columns + 1):
        cell = sheet.cell(row=row, column=column)
        cell.font = Font(bold=True)
        cell.fill = fill
        cell.alignment = Alignment(horizontal="left")


def _set_widths(sheet: Worksheet, widths: list[int]) -> None:
    for i, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(i)].width = width


def export_excel(result: AnalysisResult, path: str) -> None:
    """Write the analyst workbook: Rules, Itemsets, Segments, Recommendations."""
    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "Rules"
    sheet.append([DISCLAIMER])
    sheet.append([CAUSATION_NOTE])
    sheet.append([])
    header_row = 4
    sheet.append(
        [
            "Rule", "Antecedent", "Consequent", "Support", "Orders",
            "Confidence", "Lift", "Leverage", "Conviction", "Thin support",
        ]
    )
    for rule in result.rules:
        sheet.append(
            [
                format_rule(rule),
                format_itemset(rule.antecedent),
                format_itemset(rule.consequent),
                round(rule.support, 4),
                rule.support_count,
                round(rule.confidence, 4),
                round(rule.lift, 3),
                round(rule.leverage, 4),
                "inf" if rule.conviction == float("inf") else round(rule.conviction, 3),
                "YES" if rule.thin_support else "",
            ]
        )
    _style_header(sheet, header_row, 10)
    _set_widths(sheet, [46, 30, 22, 10, 8, 11, 8, 10, 11, 12])
    sheet.freeze_panes = f"A{header_row + 1}"

    sheet = workbook.create_sheet("Itemsets")
    sheet.append([DISCLAIMER])
    sheet.append([])
    sheet.append(["Itemset", "Size", "Support", "Orders"])
    ordered = sorted(result.itemsets.items(), key=lambda kv: (-kv[1], format_itemset(kv[0])))
    for itemset, support in ordered:
        sheet.append(
            [
                format_itemset(itemset),
                len(itemset),
                round(support, 4),
                round(support * result.n_baskets),
            ]
        )
    _style_header(sheet, 3, 4)
    _set_widths(sheet, [46, 7, 10, 9])
    sheet.freeze_panes = "A4"

    sheet = workbook.create_sheet("Segments")
    sheet.append([DISCLAIMER])
    sheet.append([])
    sheet.append(
        ["Segment", "Customers", "Share", "Avg total spend EUR", "Top categories (centroid share)"]
    )
    for profile in result.segmentation.profiles:
        top = ", ".join(f"{name} {share:.0%}" for name, share in profile.top_categories)
        sheet.append(
            [
                profile.segment,
                profile.n_customers,
                round(profile.share_of_customers, 3),
                round(profile.avg_total_spend_eur, 2),
                top,
            ]
        )
    _style_header(sheet, 3, 5)
    assignments_header = 5 + len(result.segmentation.profiles)
    sheet.append([])
    sheet.append(["Customer", "Segment", "Total spend EUR"])
    totals = result.segmentation.spend.sum(axis=1)
    for customer_id, segment in result.segmentation.assignments.items():
        sheet.append([customer_id, int(segment), round(float(totals[customer_id]), 2)])
    _style_header(sheet, assignments_header, 3)
    _set_widths(sheet, [12, 11, 18, 20, 60])

    sheet = workbook.create_sheet("Recommendations")
    sheet.append([DISCLAIMER])
    sheet.append([ASSUMPTION_NOTE])
    sheet.append([])
    sheet.append(
        ["Rank", "Buyers of", "Recommend", "Lift", "Confidence",
         "Est. incremental EUR/order", "Headline"]
    )
    for rank, recommendation in enumerate(result.recommendations, start=1):
        rule = recommendation.rule
        sheet.append(
            [
                rank,
                format_itemset(rule.antecedent),
                format_itemset(rule.consequent),
                round(rule.lift, 3),
                round(rule.confidence, 4),
                round(recommendation.est_incremental_value_eur, 2),
                recommendation.headline,
            ]
        )
    _style_header(sheet, 4, 7)
    _set_widths(sheet, [6, 30, 22, 8, 11, 24, 110])
    sheet.freeze_panes = "A5"

    workbook.save(path)


def write_deliverables(
    output_dir: str = "deliverables",
    n_baskets: int = 6000,
    seed: int = 42,
    min_support: float = 0.02,
    min_confidence: float = 0.30,
    min_lift: float = 1.10,
) -> dict[str, int]:
    """Run the pipeline and write both deliverables; verify each is > 10 KB."""
    result = run_analysis(
        n_baskets=n_baskets,
        seed=seed,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "cross_sell_briefing.pdf")
    excel_path = os.path.join(output_dir, "market_basket_analysis.xlsx")
    export_pdf(result, pdf_path)
    export_excel(result, excel_path)

    sizes: dict[str, int] = {}
    for path in (pdf_path, excel_path):
        size = os.path.getsize(path)
        if size <= 10_000:
            raise RuntimeError(f"deliverable too small ({size} bytes): {path}")
        sizes[path] = size
    return sizes
