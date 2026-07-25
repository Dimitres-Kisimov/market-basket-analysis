# Market-basket analysis and cross-sell for a B2B distributor

I built this project to answer a question every distributor's sales team argues about:
**which products should a rep offer next, given what is already in the customer's order?**
The setting is a fictional B2B maintenance & construction supplies distributor with 14
product categories, and the answer comes from association-rule mining and customer
segmentation implemented entirely from scratch on numpy/pandas -- no scikit-learn, no
mlxtend, no mining libraries.

**Everything here runs on synthetic data.** The generator is seeded and deterministic,
and it plants a handful of known co-purchase bundles (for example fasteners + power
tools + work gloves) so the mining code can be tested against a known ground truth.
I say this up front because the numbers below look like business results, and they are
not: they are measurements of my pipeline on a simulation I designed.

## What I measured (seed 42, 6,000 orders, min support 2%)

Running `python -m basket` end to end:

- **224 frequent itemsets** (14 single categories, 91 pairs, 119 triples) mined by
  Apriori with downward-closure pruning; an independent FP-growth implementation
  returns the **exact same itemsets and supports** (a test asserts equality).
- **254 association rules** kept at confidence >= 30% and lift >= 1.10, ranked by lift.
  None fall below the thin-support threshold (30 orders) at these settings.

Top 3 rules by lift:

| Rule | Support | Orders | Confidence | Lift |
|---|---|---|---|---|
| electrical + welding_supplies -> ppe_eyewear | 2.9% | 177 | 61.0% | **2.41** |
| ppe_eyewear + storage_handling -> welding_supplies | 2.5% | 152 | 66.1% | **2.39** |
| abrasives + ppe_eyewear -> welding_supplies | 4.6% | 276 | 65.9% | **2.39** |

The planted flagship bundle is recovered exactly where it should be: *fasteners +
ppe_gloves -> power_tools* comes out at lift 2.32 with 79.7% confidence across 1,038
orders -- the strongest high-volume rule in the set. All six planted category pairs are
recovered with lift >= 1.5 (tested).

- **3 customer segments** from my own k-means (k-means++ seeding, 8 restarts) on
  per-customer spend shares, matching the three archetypes the generator plants:
  - Segment 0 -- 173 customers (41%): power tools, storage & handling, hand tools
    (construction-like), avg spend EUR 5,867.
  - Segment 1 -- 137 customers (33%): electrical, power tools, pipe fittings
    (maintenance/MRO-like), avg spend EUR 5,025.
  - Segment 2 -- 110 customers (26%): welding supplies, cutting tools
    (fabrication-like), avg spend EUR 5,378.
- **Estimated cross-sell uplift**: the top-10 recommendation list carries estimated
  incremental basket values from EUR 9.82 to EUR 117.84 per targeted order (average
  EUR 44). **This is an ESTIMATE with a stated assumption**: it assumes a targeted
  buyer attaches the recommended category at the rule's confidence instead of the
  baseline rate, multiplied by the category's average line value. Real attach rates
  would have to come from an A/B test.

## How to run it

Python 3.10+ with numpy, pandas, matplotlib, openpyxl (see `requirements.txt`).

```bash
pip install -r requirements.txt
python -m basket                  # console summary: rules, segments, recommendations
python -m basket --deliverables   # writes deliverables/cross_sell_briefing.pdf
                                  #    and deliverables/market_basket_analysis.xlsx
python -m ruff check .            # lint (clean)
python -m pytest -q               # 18 tests (green)
```

Everything is reproducible: same seed, same numbers. `--seed`, `--baskets`,
`--min-support`, `--min-confidence` and `--min-lift` let you explore other settings.
There is also a `Dockerfile` (python:3.12-slim) that lints, tests and builds the
deliverables, and a GitHub Actions workflow that runs the same gates.

## Methods, from scratch

- `basket/data.py` -- seeded synthetic order generator: 14 categories, ~6,000 orders,
  three customer archetypes, four planted bundles exposed as ground truth for tests.
- `basket/apriori.py` -- Apriori (Agrawal & Srikant, 1994): level-wise candidate
  generation with prefix join and downward-closure pruning, up to 3-itemsets.
- `basket/fpgrowth.py` -- FP-growth (Han, Pei & Yin, 2000): FP-tree with header table,
  recursive mining over conditional pattern bases. Tested equal to Apriori.
- `basket/rules.py` -- support, confidence, lift, leverage, conviction; filtering,
  lift ranking, and a thin-support flag for rules backed by too few orders.
- `basket/segment.py` -- k-means with k-means++ seeding, restarts and empty-cluster
  repair, run on spend shares so large customers don't dominate.
- `basket/recommend.py` -- rules to a cross-sell action list plus a next-best-product
  lookup for a basket in progress.
- `basket/exports.py` -- a four-page executive PDF (cover with disclaimer, rules
  table, category-pair lift heatmap, segment profiles) and a four-sheet Excel
  workbook (Rules, Itemsets, Segments, Recommendations).

## Honesty notes

- **Synthetic data.** Every figure in this repository and in the deliverables comes
  from a seeded simulation I wrote. No real customer or sales data was used.
- **Correlation is not causation.** Lift describes co-purchase frequency relative to
  independence. It cannot tell you that recommending eyewear to welding-supply buyers
  *causes* extra sales -- only a controlled experiment can.
- **Thin support is flagged.** Rules backed by fewer than 30 orders are marked
  `thin_support` and excluded from recommendations by default, because their metrics
  are noise-prone.
- **Estimates are labelled.** Any EUR uplift figure carries its assumption inline.

More context in [docs/BUSINESS_CASE.md](docs/BUSINESS_CASE.md). © 2026 Dimitres Kisimov — all rights reserved; published for portfolio review. See LICENSE.
