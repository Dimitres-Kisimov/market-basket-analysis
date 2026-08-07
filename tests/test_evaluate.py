import pytest

from basket.evaluate import (
    HoldoutTrial,
    evaluate_recommenders,
    holdout_trials,
    popularity_predictions,
    popularity_ranking,
    render_evaluation_svg,
    score_recommender,
    train_test_split_by_arrival,
    write_evaluation_csv,
)


def test_train_test_split_by_arrival_is_contiguous():
    baskets = [frozenset({f"i{i}"}) for i in range(10)]
    train, test = train_test_split_by_arrival(baskets, train_fraction=0.7)
    assert len(train) == 7
    assert len(test) == 3
    # Contiguous cover in arrival order: train then test rebuilds the input.
    assert train + test == [frozenset(b) for b in baskets]


def test_train_test_split_validates_fraction():
    baskets = [frozenset({f"i{i}"}) for i in range(10)]
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            train_test_split_by_arrival(baskets, train_fraction=bad)


def test_holdout_trials_deterministic_and_eligible():
    test_baskets = [
        frozenset({"a", "b", "c"}),
        frozenset({"x"}),  # size 1: excluded
        frozenset({"m", "n"}),
    ]
    first = holdout_trials(test_baskets, seed=42)
    second = holdout_trials(test_baskets, seed=42)
    assert first == second  # same seed -> identical trials
    assert len(first) == 2  # the size-1 basket is dropped
    for trial, source in zip(first, [test_baskets[0], test_baskets[2]], strict=True):
        assert trial.held_out in source
        assert trial.query == source - {trial.held_out}
        assert trial.held_out not in trial.query
    # A different seed is allowed to pick different held-out items.
    other = holdout_trials(test_baskets, seed=7)
    assert len(other) == 2


def test_popularity_ranking_orders_by_frequency():
    train = [
        frozenset({"a", "b"}),
        frozenset({"a", "b"}),
        frozenset({"a"}),
        frozenset({"c"}),
    ]
    # a:3, b:2, c:1 -> descending count, name as tie-break.
    assert popularity_ranking(train) == ["a", "b", "c"]


def test_popularity_predictions_skips_query_and_respects_k():
    ranking = ["a", "b", "c", "d"]
    assert popularity_predictions(ranking, frozenset({"a"}), max_k=2) == ["b", "c"]
    assert popularity_predictions(ranking, frozenset({"a", "b"}), max_k=5) == ["c", "d"]


def test_score_recommender_hand_checked():
    # Three trials, k in {1,2,3}. preds[i] ranked, held[i] the hidden target.
    predictions = [["A", "B", "C"], ["A", "B", "C"], []]
    held = ["B", "A", "Z"]
    metrics = score_recommender("m", predictions, held, (1, 2, 3))
    # hit@1: only trial 1 (A at rank 1) -> 1/3.
    assert metrics.hit_rate[1] == pytest.approx(1 / 3)
    # hit@2: trial 0 (B at rank 2) and trial 1 -> 2/3; hit@3 unchanged (C not a target).
    assert metrics.hit_rate[2] == pytest.approx(2 / 3)
    assert metrics.hit_rate[3] == pytest.approx(2 / 3)
    # MRR: 1/2 (trial 0) + 1/1 (trial 1) + 0 (trial 2), averaged over 3.
    assert metrics.mrr == pytest.approx(0.5)
    # Coverage: two of three trials produced a non-empty list.
    assert metrics.coverage == pytest.approx(2 / 3)
    assert metrics.n_trials == 3


def test_score_recommender_validates_inputs():
    with pytest.raises(ValueError):
        score_recommender("m", [], [], (1,))
    with pytest.raises(ValueError):
        score_recommender("m", [["A"]], ["A"], (0,))  # non-positive K
    with pytest.raises(ValueError):
        score_recommender("m", [["A"]], ["A", "B"], (1,))  # misaligned lengths


def test_rules_beat_popularity_on_controlled_scenario():
    """Hand-built world where a rule captures a target that popularity misses.

    C->D is a perfect, non-thin rule (33/110 baskets); A,B are the popular
    categories. For the query {C} the rule recovers D at rank 1, while D is not
    among the two most popular categories, so the baseline misses it.
    """
    train = (
        [frozenset({"C", "D"})] * 33
        + [frozenset({"A", "B"})] * 66
        + [frozenset({"A"})] * 11
    )
    trials = [
        HoldoutTrial(frozenset({"C"}), "D"),  # rule hit, popularity miss
        HoldoutTrial(frozenset({"A"}), "B"),  # both hit
    ]
    report = evaluate_recommenders(
        train_baskets=train,
        trials=trials,
        k_values=(1, 2),
        min_support=0.2,
        min_confidence=0.3,
        min_lift=1.1,
    )
    assert report.n_train == 110
    assert report.n_test == 2
    assert report.n_trials == 2
    assert report.rules.hit_rate[1] == pytest.approx(1.0)
    assert report.rules.mrr == pytest.approx(1.0)
    assert report.popularity.hit_rate[1] == pytest.approx(0.5)
    assert report.popularity.mrr == pytest.approx(0.5)
    # The ratio helper reflects the 2x edge at K=1.
    assert report.hit_rate_ratio()[1] == pytest.approx(2.0)


def test_evaluate_requires_trials_or_baskets():
    with pytest.raises(ValueError):
        evaluate_recommenders()  # nothing to evaluate
    with pytest.raises(ValueError):
        evaluate_recommenders(train_baskets=[frozenset({"a", "b"})], trials=[])


def test_evaluate_on_synthetic_beats_popularity_and_is_deterministic(baskets):
    first = evaluate_recommenders(baskets, train_fraction=0.7, k_values=(1, 3, 5), seed=42)
    second = evaluate_recommenders(baskets, train_fraction=0.7, k_values=(1, 3, 5), seed=42)

    assert first.k_values == (1, 3, 5)
    assert first.n_train + first.n_test == len(baskets)
    assert 0 < first.n_trials <= first.n_test
    for metrics in (first.rules, first.popularity):
        for k in first.k_values:
            assert 0.0 <= metrics.hit_rate[k] <= 1.0
        assert 0.0 <= metrics.mrr <= 1.0
        assert 0.0 <= metrics.coverage <= 1.0
    # Hit-rate is monotonic non-decreasing in K for both recommenders.
    for metrics in (first.rules, first.popularity):
        hits = [metrics.hit_rate[k] for k in first.k_values]
        assert hits == sorted(hits)
    # The mined rules add predictive value over the popularity baseline.
    assert first.rules.hit_rate[3] > first.popularity.hit_rate[3]
    assert first.rules.mrr > first.popularity.mrr

    # Fully deterministic: identical metrics on a re-run with the same seed.
    for k in first.k_values:
        assert first.rules.hit_rate[k] == second.rules.hit_rate[k]
        assert first.popularity.hit_rate[k] == second.popularity.hit_rate[k]
    assert first.rules.mrr == second.rules.mrr


def test_plain_language_mentions_both_recommenders(baskets):
    report = evaluate_recommenders(baskets, train_fraction=0.7, seed=42)
    text = report.plain_language()
    assert str(report.n_trials) in text
    assert "popularity" in text.lower()
    assert "not the causal" in text.lower()


def test_csv_and_svg_writers_deterministic(baskets, tmp_path):
    report = evaluate_recommenders(baskets, train_fraction=0.7, seed=42)

    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    write_evaluation_csv(report, str(csv_a))
    write_evaluation_csv(report, str(csv_b))
    assert csv_a.read_bytes() == csv_b.read_bytes()
    lines = csv_a.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "recommender,n_trials,hit_rate_at_1,hit_rate_at_3,hit_rate_at_5,mrr,coverage"
    assert lines[1].startswith("association rules,")
    assert lines[2].startswith("popularity baseline,")

    svg = render_evaluation_svg(report)
    assert svg.startswith("<svg")
    assert "recommender back-test" in svg.lower()
    assert render_evaluation_svg(report) == svg  # pure function of the report
