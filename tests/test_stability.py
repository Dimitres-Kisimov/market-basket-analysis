import pytest

from basket.apriori import apriori
from basket.rules import generate_rules
from basket.stability import (
    bootstrap_splits,
    render_stability_svg,
    rule_stability,
    time_window_splits,
    write_stability_csv,
)


def _milk_bread_reference():
    """A hand-built split where milk -> bread clears conf 0.5 / lift 1.1.

    split1 = 3x{milk,bread}, {milk}, {bread}, {other}: support 0.5, conf 0.75,
    lift 1.125 -- the rule is present. Returns the reference rule list mined on
    this split alone so tests can control the reference independently of splits.
    """
    split1 = [
        frozenset({"milk", "bread"}),
        frozenset({"milk", "bread"}),
        frozenset({"milk", "bread"}),
        frozenset({"milk"}),
        frozenset({"bread"}),
        frozenset({"other"}),
    ]
    itemsets = apriori(split1, min_support=0.1, max_len=2)
    reference = generate_rules(itemsets, len(split1), min_confidence=0.5, min_lift=1.1)
    return split1, reference


def test_time_window_splits_partition(baskets):
    windows = time_window_splits(baskets, k=4)
    assert len(windows) == 4
    # Contiguous cover: concatenation reproduces the input exactly, in order.
    rebuilt = [b for window in windows for b in window]
    assert rebuilt == [frozenset(b) for b in baskets]
    # Near-equal sizes (differ by at most one).
    lengths = [len(w) for w in windows]
    assert max(lengths) - min(lengths) <= 1


def test_time_window_splits_validates_k(baskets):
    with pytest.raises(ValueError):
        time_window_splits(baskets, k=1)
    with pytest.raises(ValueError):
        time_window_splits(baskets[:3], k=5)


def test_bootstrap_splits_deterministic_and_sized(baskets):
    a = bootstrap_splits(baskets, n_folds=5, seed=42)
    b = bootstrap_splits(baskets, n_folds=5, seed=42)
    c = bootstrap_splits(baskets, n_folds=5, seed=7)
    assert len(a) == 5
    assert all(len(fold) == len(baskets) for fold in a)
    assert a == b  # same seed -> identical resamples
    assert a != c  # different seed -> different resamples
    with pytest.raises(ValueError):
        bootstrap_splits(baskets, n_folds=1)


def test_stability_score_counts_present_splits():
    split1, reference = _milk_bread_reference()
    # A second split where milk -> bread has confidence 0.33 (dropped).
    split2 = [
        frozenset({"milk"}),
        frozenset({"milk"}),
        frozenset({"bread"}),
        frozenset({"bread"}),
        frozenset({"milk", "bread"}),
        frozenset({"other"}),
    ]
    report = rule_stability(
        split1 + split2,
        reference,
        splits=[split1, split2],
        min_support=0.1,
        min_confidence=0.5,
        min_lift=1.1,
        top_n=5,
    )
    rule = next(r for r in report.rules if r.label == "milk -> bread")
    assert rule.n_splits == 2
    assert rule.n_present == 1  # present in split1 only
    assert rule.stability_score == pytest.approx(0.5)
    assert rule.stable is False

    # Present in every split -> stable, score 1.0.
    both = rule_stability(
        split1,
        reference,
        splits=[split1, split1],
        min_support=0.1,
        min_confidence=0.5,
        min_lift=1.1,
        top_n=5,
    )
    rule_both = next(r for r in both.rules if r.label == "milk -> bread")
    assert rule_both.n_present == 2
    assert rule_both.stability_score == pytest.approx(1.0)
    assert rule_both.stable is True
    assert rule_both.lift_std == pytest.approx(0.0)  # identical splits


def test_stability_reuses_miner_on_synthetic(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets), min_confidence=0.30, min_lift=1.10)
    report = rule_stability(baskets, rules, method="time_window", n_splits=4, top_n=10)

    assert report.method == "time_window"
    assert report.n_splits == 4
    assert len(report.rules) == 10
    for item in report.rules:
        assert 0.0 <= item.stability_score <= 1.0
        # Default threshold 1.0: stable iff present in every split.
        assert item.stable == (item.n_present == item.n_splits)
        if item.n_present:
            assert item.lift_min <= item.lift_mean <= item.lift_max
            assert item.lift_std >= 0.0
    assert 0 <= report.n_stable <= len(report.rules)


def test_stability_is_deterministic(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets), min_confidence=0.30, min_lift=1.10)
    first = rule_stability(baskets, rules, method="time_window", n_splits=4, top_n=10)
    second = rule_stability(baskets, rules, method="time_window", n_splits=4, top_n=10)
    assert [r.stability_score for r in first.rules] == [
        r.stability_score for r in second.rules
    ]
    assert [r.lift_mean for r in first.rules] == [r.lift_mean for r in second.rules]

    # Bootstrap method runs and is likewise reproducible.
    boot1 = rule_stability(baskets, rules, method="bootstrap", n_splits=5, top_n=10)
    boot2 = rule_stability(baskets, rules, method="bootstrap", n_splits=5, top_n=10)
    assert [r.stability_score for r in boot1.rules] == [
        r.stability_score for r in boot2.rules
    ]


def test_invalid_method_rejected(baskets):
    with pytest.raises(ValueError):
        rule_stability(baskets, [], method="nonsense", n_splits=4)


def test_plain_language_reports_counts(baskets):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets), min_confidence=0.30, min_lift=1.10)
    report = rule_stability(baskets, rules, method="time_window", n_splits=4, top_n=10)
    text = report.plain_language()
    assert str(report.n_stable) in text
    assert "top" in text.lower()


def test_csv_and_svg_writers_deterministic(baskets, tmp_path):
    itemsets = apriori(baskets, min_support=0.02, max_len=3)
    rules = generate_rules(itemsets, len(baskets), min_confidence=0.30, min_lift=1.10)
    report = rule_stability(baskets, rules, method="time_window", n_splits=4, top_n=10)

    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    write_stability_csv(report, str(csv_a))
    write_stability_csv(report, str(csv_b))
    assert csv_a.read_bytes() == csv_b.read_bytes()
    header = csv_a.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("rank,rule,antecedent,consequent")

    svg = render_stability_svg(report)
    assert svg.startswith("<svg")
    assert "Rule stability" in svg
    assert "CATEGORY ATLAS" in svg and "PLATE 06 / 07" in svg  # atlas plate header
    assert "atlas-hatch-warning" in svg  # verdict is never colour alone
    assert render_stability_svg(report) == svg  # pure function of the report
