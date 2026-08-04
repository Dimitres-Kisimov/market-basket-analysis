# Business case: rule-driven cross-sell for a B2B distributor

> **Scope note:** the "company" below is fictional and every number is either measured
> on the project's seeded synthetic dataset or an explicitly labelled ESTIMATE. This
> document exists to show how I would frame the analysis for stakeholders, not to
> report real results.

## Situation

A mid-size B2B distributor of maintenance and construction supplies (14 product
categories, ~420 active trade accounts in the simulated region) takes roughly 6,000
orders per period through inside sales and a webshop. Cross-selling today is ad hoc:
whether a rep suggests work gloves with a power-tool order depends on who picks up
the phone.

## Quantified problem

Measured on the synthetic order history (seed 42, 6,000 orders):

- The average order contains ~3.8 category lines; most orders leave obvious
  companion categories on the table.
- Example, measured: 79.7% of the 1,302 orders containing fasteners + work gloves
  also contain power tools -- but 264 of those orders (20.3%) contain no power tools
  at all. Each one forgoes a category with an average line value of EUR 260.
- Nothing systematic connects the webshop's "you may also like" slot, the reps'
  talk tracks, or the quarterly promo plan to actual co-purchase behaviour.

## Solution

Mine the order history for association rules and act on the strongest ones:

1. **Frequent-itemset mining** (Apriori, cross-checked by FP-growth) at 2% minimum
   support -- 224 itemsets.
2. **Rule scoring** with confidence, lift, leverage and conviction; keep 254 rules at
   confidence >= 30% and lift >= 1.10; flag thin-support rules (< 30 orders) and keep
   them out of recommendations.
3. **Customer segmentation** (k-means on spend shares) to route the right rule to the
   right account list -- three clear segments emerge (construction-, MRO- and
   fabrication-shaped).
4. **Action list**: a ranked top-10 cross-sell list plus a next-best-product function
   for the webshop basket and the rep's order screen.

## ROI (all figures labelled)

- Measured: the top-10 rules carry estimated incremental basket values of
  EUR 9.82-117.84 per targeted order (average EUR 44). *ESTIMATE -- assumes a
  targeted buyer attaches the recommended category at the rule's confidence rather
  than the baseline rate, valued at the category's average line value.*
- Illustrative arithmetic, not a forecast: the flagship rule (fasteners + gloves ->
  power tools) leaves 264 qualifying orders per period without power tools. If a
  suggestion at order entry converted 20% of them (assumed pilot conversion rate) at
  the EUR 260 average line value, that is ~EUR 13,700 per period from one rule.
  *ESTIMATE on synthetic data with an assumed conversion rate; the honest way to
  read this is "the prize is large enough to justify a pilot", nothing more.*
- Cost side (ESTIMATE): wiring the rule list into the webshop slot and the order
  screen is configuration work, not data science -- on the order of 2-3 weeks of
  one developer plus campaign ops time.
- **The go/no-go must come from an A/B test.** Lift is observational; only a
  controlled experiment measures true incremental revenue. The pilot design is:
  randomize qualifying orders 50/50, serve rule-driven suggestions to treatment,
  measure attach rate and basket value, decide on the measured difference.

## Stakeholders

- **Sales director** -- owns the rep talk tracks; consumes the top-10 rule list.
- **E-commerce lead** -- owns the webshop recommendation slot; consumes the
  next-best-product function.
- **Category managers** -- own bundle pricing and the promo calendar; consume the
  lift heatmap and segment profiles.
- **Data/BI** -- owns the pipeline, thresholds and the A/B test readout.

## Deliverable

`python -m basket --deliverables` produces, reproducibly:

- `deliverables/cross_sell_briefing.pdf` -- four-page executive briefing: headline
  numbers and disclaimer, top-rules table, category-pair lift heatmap, segment
  profiles.
- `deliverables/market_basket_analysis.xlsx` -- working file for analysts: all 254
  rules with full metrics, all 224 itemsets, segment assignments per customer, the
  ranked recommendation list with the estimate assumption spelled out per row, and a
  Stability sheet scoring how well the top rules persist across splits.
- `deliverables/rule_stability.csv` + `deliverables/rule_stability.svg` -- the
  robustness read-out: for the top 20 rules by lift, the fraction of time windows in
  which each rule still clears every threshold (18 of 20 stable on the default run),
  so a reviewer can tell durable patterns from window-specific ones before acting.
