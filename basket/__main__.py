"""Command-line entry point.

``python -m basket``                prints an analysis summary to the console.
``python -m basket --deliverables`` writes the PDF + Excel deliverables.

Console output is deliberately plain ASCII and stdout is reconfigured to
UTF-8 with replacement so the tool is safe on any Windows code page.
"""

from __future__ import annotations

import argparse
import sys

from basket.exports import (
    DISCLAIMER,
    build_affinity_report,
    build_evaluation_report,
    build_stability_report,
    run_analysis,
    write_deliverables,
)
from basket.rules import format_rule


def _print_summary(args: argparse.Namespace) -> None:
    result = run_analysis(
        n_baskets=args.baskets,
        seed=args.seed,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
    )
    print(DISCLAIMER)
    print()
    print(f"Baskets analysed : {result.n_baskets}")
    print(f"Frequent itemsets: {len(result.itemsets)} (min support {args.min_support:.0%})")
    print(f"Rules kept       : {len(result.rules)} (confidence >= "
          f"{args.min_confidence:.0%}, lift >= {args.min_lift:.2f})")
    print()
    print(f"Top {args.top} rules by lift:")
    for rule in result.rules[: args.top]:
        thin = "  [thin support]" if rule.thin_support else ""
        print(
            f"  {format_rule(rule):<55s} lift={rule.lift:5.2f} "
            f"conf={rule.confidence:5.1%} support={rule.support:5.1%}{thin}"
        )
    print()
    print("Customer segments (k-means on spend shares):")
    for profile in result.segmentation.profiles:
        top_categories = ", ".join(name for name, _ in profile.top_categories)
        print(
            f"  Segment {profile.segment}: {profile.n_customers} customers "
            f"({profile.share_of_customers:.0%}), avg spend EUR "
            f"{profile.avg_total_spend_eur:,.0f}; top: {top_categories}"
        )
    print()
    print("Top cross-sell recommendations (observational; correlation != causation):")
    for recommendation in result.recommendations[:3]:
        print(f"  - {recommendation.headline}")
    print()
    affinity = build_affinity_report(result)
    print("Category affinity network (co-purchase communities, greedy modularity):")
    print(f"  {affinity.plain_language()}")
    for community in affinity.communities:
        if community.n_members < 2:
            continue
        print(
            f"  - community {community.community_id}: {', '.join(community.members)} "
            f"({community.n_internal_edges} edges, avg internal lift "
            f"{community.lift_mean:.2f})"
        )
    for bridge in affinity.bridges[:3]:
        print(
            f"  - bridge: {bridge.label} (lift {bridge.lift:.2f}) links communities "
            f"{affinity.membership[bridge.item_a]} and {affinity.membership[bridge.item_b]}"
        )
    print()
    stability = build_stability_report(result)
    print("Rule stability (robustness check across time splits):")
    print(f"  {stability.plain_language()}")
    for item in stability.rules:
        if not item.stable:
            print(
                f"  - window-specific: {item.label} "
                f"(in {item.n_present}/{item.n_splits} windows, lift {item.reference_lift:.2f})"
            )
    print()
    evaluation = build_evaluation_report(result)
    ratio = evaluation.hit_rate_ratio()
    print("Recommender back-test (leave-one-out predictive accuracy):")
    print(f"  {evaluation.plain_language()}")
    header_k = "  ".join(f"hit@{k}" for k in evaluation.k_values)
    print(f"                        {header_k}    MRR   coverage")
    rules_hits = "  ".join(f"{evaluation.rules.hit_rate[k]:5.1%}" for k in evaluation.k_values)
    pop_hits = "  ".join(f"{evaluation.popularity.hit_rate[k]:5.1%}" for k in evaluation.k_values)
    print(f"  association rules     {rules_hits}   {evaluation.rules.mrr:.3f}   "
          f"{evaluation.rules.coverage:.1%}")
    print(f"  popularity baseline   {pop_hits}   {evaluation.popularity.mrr:.3f}   "
          f"{evaluation.popularity.coverage:.1%}")
    ratio_str = "  ".join(f"{ratio[k]:4.2f}x" for k in evaluation.k_values)
    print(f"  rules / baseline      {ratio_str}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="python -m basket",
        description="Market-basket analysis on seeded synthetic B2B data.",
    )
    parser.add_argument(
        "--deliverables", action="store_true",
        help="write the executive PDF and Excel workbook to --outdir",
    )
    parser.add_argument("--outdir", default="deliverables")
    parser.add_argument("--baskets", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--min-lift", type=float, default=1.10)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    if args.deliverables:
        sizes = write_deliverables(
            output_dir=args.outdir,
            n_baskets=args.baskets,
            seed=args.seed,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_lift=args.min_lift,
        )
        print(DISCLAIMER)
        for path, size in sizes.items():
            print(f"wrote {path} ({size:,} bytes; verified non-empty)")
        return 0

    _print_summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
