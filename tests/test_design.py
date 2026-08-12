"""The category-atlas design system: palette maths, styles, plate registry.

The member ramps below are pinned to the exact hexes that were run through the
data-viz palette validator (ordinal mode: monotone lightness, adjacent
delta-L >= 0.06, light end >= 2:1 on the surface) so any drift in the colour
maths is caught here, not in a shipped deliverable.
"""

from basket import design

# Validator-passed member ramps (light -> dark) per community hue.
_EXPECTED_RAMPS = {
    ("#2a78d6", 5): ["#62a6ff", "#408cec", "#2573d1", "#005bb5", "#00458d"],
    ("#2a78d6", 4): ["#62a6ff", "#3784e3", "#0e63bf", "#00458d"],
    ("#eb6834", 5): ["#fd7845", "#e05e29", "#c44500", "#9f3600", "#7c2800"],
    ("#eb6834", 4): ["#fd7845", "#d7551e", "#ab3b00", "#7c2800"],
    ("#1baf7a", 5): ["#37bf89", "#00a672", "#008a5e", "#006f4b", "#005639"],
    ("#1baf7a", 4): ["#37bf89", "#009c6b", "#007851", "#005639"],
}


def test_member_ramps_pinned_to_validated_hexes():
    for (base, n), expected in _EXPECTED_RAMPS.items():
        assert design.member_ramp(base, n) == expected


def test_member_ramps_are_ordinal():
    """Monotone lightness, visible step gaps, light end readable on surface."""
    for base in design.COMMUNITY_HUES:
        for n in (2, 3, 4, 5):
            ramp = design.member_ramp(base, n)
            lightness = [design.hex_to_oklch(step)[0] for step in ramp]
            deltas = [
                lightness[i] - lightness[i + 1] for i in range(len(lightness) - 1)
            ]
            assert all(delta >= 0.06 for delta in deltas), (base, n, deltas)
            assert design.wcag_contrast(ramp[0], design.SURFACE) >= 2.0


def test_member_ramp_single_and_errors():
    assert design.member_ramp("#2a78d6", 1) == ["#2a78d6"]
    try:
        design.member_ramp("#2a78d6", 0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("n=0 must raise")


def test_category_styles_grouped_and_overflow():
    communities = [
        (0, ("abrasives", "cutting_tools", "lubricants")),
        (1, ("fasteners", "hand_tools")),
        (2, ("electrical", "janitorial")),
        (3, ("welding_supplies",)),  # singleton: folds to grey
    ]
    styles = design.category_styles(communities)
    assert len(styles) == 8
    assert styles["abrasives"].hue == design.COMMUNITY_HUES[0]
    assert styles["abrasives"].shape == "circle"
    assert styles["fasteners"].shape == "square"
    assert styles["electrical"].shape == "diamond"
    # Members step light -> dark in member order within one hue.
    l_first = design.hex_to_oklch(styles["abrasives"].fill)[0]
    l_last = design.hex_to_oklch(styles["lubricants"].fill)[0]
    assert l_first > l_last
    # The singleton wears the fold-to-other treatment, never a community hue.
    assert styles["welding_supplies"].fill == design.OVERFLOW_FILL
    assert not styles["welding_supplies"].grouped
    # Pure function: same input, same styles.
    assert design.category_styles(communities) == styles


def test_community_hue_bounds():
    assert design.community_hue(0, 5) == design.COMMUNITY_HUES[0]
    assert design.community_hue(0, 1) is None  # singleton
    assert design.community_hue(3, 5) is None  # beyond the validated hues


def test_plate_registry():
    numbers = sorted(design.PLATES.values())
    assert numbers == list(range(1, design.N_PLATES + 1))
    assert design.plate_eyebrow("network") == "CATEGORY ATLAS · PLATE 04 / 07"


def test_tint_and_mix():
    assert design.tint("#2a78d6", 0.0) == "#ffffff"
    assert design.tint("#2a78d6", 1.0) == "#2a78d6"
    assert design.mix("#000000", "#ffffff", 0.5) == "#808080"


def test_svg_mark_shapes():
    parts: list[str] = []
    design.svg_mark(parts, x=10, y=10, r=5, fill="#2a78d6", shape="circle")
    design.svg_mark(parts, x=10, y=10, r=5, fill="#eb6834", shape="square")
    design.svg_mark(parts, x=10, y=10, r=5, fill="#1baf7a", shape="diamond")
    assert parts[0].startswith("<circle")
    assert parts[1].startswith("<rect")
    assert parts[2].startswith("<polygon")
