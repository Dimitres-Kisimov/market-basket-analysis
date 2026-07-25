"""Command-line entry point.

``python -m basket``                prints an analysis summary to the console.
``python -m basket --deliverables`` writes the PDF + Excel deliverables.

Console output is deliberately plain ASCII and stdout is reconfigured to
UTF-8 with replacement so the tool is safe on any Windows code page.
"""

from __future__ import annotations

import argparse
import sys

from basket.exports import DISCLAIMER, run_analysis, write_deliverables
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
            print(f"wrote {path} ({size:,} bytes; verified > 10 KB)")
        return 0

    _print_summary(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
