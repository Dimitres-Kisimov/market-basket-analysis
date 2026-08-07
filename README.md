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

## Rule stability -- which rules are trustworthy, not just high-lift?

A high lift on the full dataset is not the same as a *repeatable* pattern. So the
pipeline re-mines itself: it re-runs the exact same Apriori + rule-scoring pipeline,
at the same thresholds, across **4 contiguous time windows** (baskets carry no
timestamp, so basket arrival order is used as a documented proxy for time) and,
optionally, across seeded bootstrap resamples. For each of the **top 20 rules by
lift** it reports a *stability score* -- the fraction of windows in which the rule
still clears support >= 2%, confidence >= 30% and lift >= 1.10 -- plus how far the
lift moves across windows.

Measured (seed 42, 6,000 orders, 4 time windows):

- **18 of the top 20 rules are stable** -- they reappear above all three thresholds
  in every window. The flagship *fasteners + ppe_gloves -> power_tools* is one of
  them (lift 2.32 in the full data; per-window lift 2.19-2.55).
- **2 are flagged window-specific**, appearing in only 3 of the 4 windows:
  *pipe_fittings + storage_handling -> adhesives_sealants* (lift 2.27) and
  *janitorial + ppe_eyewear -> welding_supplies* (lift 2.20). Both sit at ~2.15%
  support -- just above the 2% floor -- so in a single 1,500-order window they slip
  under the count threshold. They look as strong as the rest by lift alone; the
  stability check is what separates them.

Honest caveat: the synthetic generator is **stationary** (every basket is drawn from
the same distribution), so durable rules are *expected* to persist and the flicker
above is sampling noise near the threshold, not real drift. The value here is twofold
-- it validates the machinery, and on real order history (where demand drifts and
promotions come and go) the identical check would separate durable rules from
seasonal or one-off artefacts. Outputs: `deliverables/rule_stability.csv`,
`deliverables/rule_stability.svg`, a "Stability" sheet in the workbook, and a
stability page in the PDF.

## Recommender back-test -- does the cross-sell engine actually predict?

Stability asks whether a rule is *durable*; it does not ask whether the
recommender *predicts what a customer buys next*. So the pipeline back-tests the
recommender the way you would evaluate any top-N recommender: **leave-one-out on
held-out baskets**. Rules are mined on the first 70% of baskets by arrival order
(the same documented time proxy used for stability); for every basket in the
later 30% that has at least two categories, one category is hidden, the rest are
fed to the recommender, and I check whether the hidden category comes back in the
top-K. The identical trials are scored for a **popularity baseline** (recommend
the most-frequent training categories not already in the basket) so the rules'
predictive value is *measured against a trivial control*, not asserted.

Measured (seed 42, 6,000 orders, 70/30 arrival-order split, 4,200 train / 1,800
test, 1,594 eligible test baskets):

| Recommender | hit-rate@1 | hit-rate@3 | hit-rate@5 | MRR | coverage |
|---|---|---|---|---|---|
| **Association rules** | 30.4% | **60.2%** | 69.9% | **0.455** | 99.9% |
| Popularity baseline | 20.1% | 34.4% | 56.2% | 0.311 | 100% |
| Rules / baseline | 1.51x | **1.75x** | 1.24x | 1.46x | -- |

The rule recommender recovers the held-out category in **60.2% of cases within
its top 3**, versus **34.4%** for popularity -- **1.75x** the baseline hit-rate --
and ranks the true next item higher on average (MRR 0.455 vs 0.311). It fires on
99.9% of test baskets (a rule matched; the rest fall through to nothing).

Honest reading: this is **predictive fit on stationary synthetic data**, not the
causal revenue a campaign would earn -- only an A/B test measures that (the
recommendations page already says so). Because the generator draws every basket
from one distribution, the arrival-order split is not a real train-then-future
shift; its job here is to validate the evaluation harness and quantify how far the
rules beat popularity. On real order history the identical back-test would report
how well last quarter's rules predict this quarter's baskets. Outputs:
`deliverables/recommender_backtest.csv`, `deliverables/recommender_backtest.svg`,
an "Evaluation" sheet in the workbook, and a back-test page in the PDF.

## How to run it

Python 3.10+ with numpy, pandas, matplotlib, openpyxl (see `requirements.txt`).

```bash
pip install -r requirements.txt
python -m basket                  # console summary: rules, segments, recommendations,
                                  #    and the rule-stability read
python -m basket --deliverables   # writes deliverables/cross_sell_briefing.pdf,
                                  #    market_basket_analysis.xlsx, rule_stability.csv,
                                  #    rule_stability.svg, recommender_backtest.csv
                                  #    and recommender_backtest.svg
python -m ruff check .            # lint (clean)
python -m pytest -q               # 42 tests (green)
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
- `basket/stability.py` -- robustness trust layer: re-mines the top rules across
  time windows or seeded bootstrap folds (reusing Apriori + rule scoring) and scores
  how consistently each rule clears the thresholds; writes a CSV and a hand-drawn SVG.
- `basket/segment.py` -- k-means with k-means++ seeding, restarts and empty-cluster
  repair, run on spend shares so large customers don't dominate.
- `basket/recommend.py` -- rules to a cross-sell action list, a ranked top-K
  next-best-category function, and a single next-best-product lookup for a basket
  in progress.
- `basket/evaluate.py` -- recommender back-test: a leave-one-out, held-out-basket
  evaluation (reusing Apriori + rule scoring) that measures hit-rate@K, MRR and
  coverage for the rule recommender against a popularity baseline; writes a CSV and
  a hand-drawn SVG.
- `basket/exports.py` -- a six-page executive PDF (cover with disclaimer, rules
  table, category-pair lift heatmap, segment profiles, rule-stability page,
  recommender back-test page) and a six-sheet Excel workbook (Rules, Itemsets,
  Segments, Recommendations, Stability, Evaluation).

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
