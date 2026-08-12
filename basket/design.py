"""The "category atlas" design system shared by every deliverable.

One visual language across the PDF briefing, the Excel workbook and the
standalone SVG read-outs, so the four artifacts read as plates of a single
atlas rather than four separate charts:

* **Numbered plates.** Every subject has a fixed plate number (``PLATES``);
  a PDF page and the standalone SVG about the same subject carry the same
  number, under a common eyebrow line (``CATEGORY ATLAS · PLATE 04 / 07``).
* **One categorical palette for the 14 categories.** Fourteen distinct hues
  cannot pass colour-vision-deficiency separation, so identity is *grouped*:
  each co-purchase **community** takes one of three validated hues (the
  all-pairs-safe slots blue / orange / aqua), and **members** within a
  community take monotone lightness steps of that hue. Identity is never
  colour alone: every mark is directly labelled, and each community also has
  a fixed marker **shape** (circle / square / diamond). Communities beyond
  the third — and singleton communities — fold to neutral grey with labels.
* **Hairline chrome.** Recessive grids/rules (``GRID``), a warm off-white
  surface, ink hierarchy (primary / secondary / muted), tabular figures.
* **Honest footnotes as designed captions.** The synthetic-data and
  correlation-is-not-causation notes are part of the page anatomy (a
  hairline rule + muted caption), never removed or shrunk away.

Palette validation (scripts/validate_palette.js of the data-viz method,
light surface ``#fcfcfb`` — all runs exit 0):

* community hues ``#2a78d6, #eb6834, #1baf7a`` under ``--pairs all``:
  worst CVD pair ΔE 9.2 (target ≥ 8), worst normal-vision pair ΔE 24.0
  (floor ≥ 15); aqua sits at 2.74:1 contrast, which obligates the relief
  rule — satisfied because every mark in the atlas is directly labelled;
* member ramps (``member_ramp``) at n = 2..5 per hue under ``--ordinal``:
  monotone lightness, adjacent ΔL ≥ 0.06, light end ≥ 2:1 on the surface.

Everything here is pure, deterministic float/string math: no RNG, no wall
clock, so SVG/CSV bytes stay identical across runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Ink & chrome (light surface) — the single source of truth for every artifact.
# --------------------------------------------------------------------------- #

INK = "#0b0b0b"  # primary text
INK_SECONDARY = "#52514e"  # supporting text, table bodies
INK_MUTED = "#898781"  # captions, axes, footnotes
GRID = "#e1e0d9"  # hairline gridlines / table rules
BASELINE = "#c3c2b7"  # axis baselines, section rules
SURFACE = "#fcfcfb"  # chart surface
NEUTRAL_FILL = "#f0efec"  # table header bands, bar tracks, diverging midpoint

FONT_STACK = "Segoe UI, Helvetica, Arial, sans-serif"

# Series roles (not category identity).
SERIES_BLUE = "#2a78d6"  # slot-1 hue: single-series bars, "our" recommender
BASELINE_SERIES = "#898781"  # reference series (e.g. popularity baseline)
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")  # blue <-> neutral grey <-> red

# Status colours (reserved meaning, never reused as "series 4").
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_WARNING_DEEP = "#b47e00"  # tone-on-tone hatch ink for warning fills

# Table tints (12% of the hue over white; text on top stays ink).
TINT_GOOD = "#e2f4e2"
TINT_WARNING = "#fef6e3"
TINT_BLUE = "#e5effa"

# --------------------------------------------------------------------------- #
# The category palette: community hues + member lightness steps.
# --------------------------------------------------------------------------- #

#: All-pairs-validated community hues, in fixed order (never cycled).
COMMUNITY_HUES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")
COMMUNITY_HUE_NAMES: tuple[str, ...] = ("blue", "orange", "aqua")
#: Form redundancy: one marker shape per community (colour is never alone).
COMMUNITY_SHAPES: tuple[str, ...] = ("circle", "square", "diamond")
#: Fold-to-other treatment for singletons / communities beyond the third.
OVERFLOW_FILL = "#898781"
OVERFLOW_SHAPE = "circle"

#: Lightness span of a member ramp (OKLCH L). The light end clears 2:1 on the
#: surface for all three hues; five steps keep adjacent ΔL >= 0.06.
RAMP_L_TOP = 0.72
RAMP_L_BOTTOM = 0.40


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    return (
        int(hex_color[1:3], 16) / 255.0,
        int(hex_color[3:5], 16) / 255.0,
        int(hex_color[5:7], 16) / 255.0,
    )


def hex_to_oklch(hex_color: str) -> tuple[float, float, float]:
    """sRGB hex -> OKLCH (Björn Ottosson's OKLab, polar form)."""
    r, g, b = (_srgb_to_linear(c) for c in _hex_to_rgb(hex_color))
    long = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    medium = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    short = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = long ** (1 / 3), medium ** (1 / 3), short ** (1 / 3)
    lightness = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_axis = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    chroma = math.hypot(a, b_axis)
    hue = math.degrees(math.atan2(b_axis, a)) % 360.0
    return lightness, chroma, hue


def _oklch_to_rgb(lightness: float, chroma: float, hue: float) -> tuple[float, float, float]:
    a = chroma * math.cos(math.radians(hue))
    b_axis = chroma * math.sin(math.radians(hue))
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b_axis
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b_axis
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b_axis
    long, medium, short = l_**3, m_**3, s_**3
    r = 4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short
    g = -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short
    b = -0.0041960863 * long - 0.7034186147 * medium + 1.7076147010 * short
    return tuple(_linear_to_srgb(c) for c in (r, g, b))


def _in_gamut(rgb: tuple[float, float, float]) -> bool:
    return all(-1e-6 <= c <= 1 + 1e-6 for c in rgb)


def oklch_to_hex(lightness: float, chroma: float, hue: float) -> str:
    """OKLCH -> sRGB hex, bisecting chroma into gamut when needed."""
    rgb = _oklch_to_rgb(lightness, chroma, hue)
    if not _in_gamut(rgb):
        lo, hi = 0.0, chroma
        for _ in range(24):
            mid = (lo + hi) / 2.0
            if _in_gamut(_oklch_to_rgb(lightness, mid, hue)):
                lo = mid
            else:
                hi = mid
        rgb = _oklch_to_rgb(lightness, lo, hue)
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def member_ramp(base_hex: str, n: int) -> list[str]:
    """``n`` monotone lightness steps of one hue, light -> dark.

    Chroma and hue are held from ``base_hex`` (chroma clamped into gamut per
    step); lightness runs ``RAMP_L_TOP`` -> ``RAMP_L_BOTTOM``. Validated with
    the palette validator's ``--ordinal`` mode for n = 2..5 per community hue.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return [base_hex]
    _, chroma, hue = hex_to_oklch(base_hex)
    span = RAMP_L_TOP - RAMP_L_BOTTOM
    return [
        oklch_to_hex(RAMP_L_TOP - span * i / (n - 1), chroma, hue) for i in range(n)
    ]


def mix(hex_a: str, hex_b: str, t: float) -> str:
    """Linear sRGB-byte mix: ``t = 0`` -> a, ``t = 1`` -> b."""
    ra, ga, ba = (int(hex_a[i : i + 2], 16) for i in (1, 3, 5))
    rb, gb, bb = (int(hex_b[i : i + 2], 16) for i in (1, 3, 5))

    def blend(x: int, y: int) -> int:
        return round(x + (y - x) * t)

    return f"#{blend(ra, rb):02x}{blend(ga, gb):02x}{blend(ba, bb):02x}"


def tint(hex_color: str, strength: float) -> str:
    """``strength`` of the hue over white — for hull washes and cell fills."""
    return mix("#ffffff", hex_color, strength)


def wcag_contrast(hex_a: str, hex_b: str) -> float:
    """WCAG 2.x contrast ratio between two colours."""

    def _luminance(hex_color: str) -> float:
        r, g, b = (_srgb_to_linear(c) for c in _hex_to_rgb(hex_color))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    la, lb = _luminance(hex_a), _luminance(hex_b)
    la, lb = max(la, lb), min(la, lb)
    return (la + 0.05) / (lb + 0.05)


@dataclass(frozen=True)
class CategoryStyle:
    """How one category is drawn everywhere in the atlas."""

    fill: str  # the member step (community hue at this member's lightness)
    hue: str  # the community's base hue (hulls, tints, chips)
    shape: str  # circle | square | diamond (form redundancy)
    community_id: int
    grouped: bool  # False = fold-to-other grey (singleton / overflow)


def community_hue(community_id: int, n_members: int) -> str | None:
    """The community's base hue, or ``None`` for the grey fold-to-other case."""
    if n_members >= 2 and 0 <= community_id < len(COMMUNITY_HUES):
        return COMMUNITY_HUES[community_id]
    return None


def community_shape(community_id: int, n_members: int) -> str:
    if community_hue(community_id, n_members) is None:
        return OVERFLOW_SHAPE
    return COMMUNITY_SHAPES[community_id]


def category_styles(
    communities: list[tuple[int, tuple[str, ...]]],
) -> dict[str, CategoryStyle]:
    """Build the per-category atlas styles from ``(community_id, members)`` pairs.

    Members are assigned lightness steps in their (already sorted) member
    order, light -> dark, so a category keeps its colour for a given run of
    the pipeline across every artifact.
    """
    styles: dict[str, CategoryStyle] = {}
    for community_id, members in communities:
        hue = community_hue(community_id, len(members))
        if hue is None:
            for member in members:
                styles[member] = CategoryStyle(
                    fill=OVERFLOW_FILL,
                    hue=OVERFLOW_FILL,
                    shape=OVERFLOW_SHAPE,
                    community_id=community_id,
                    grouped=False,
                )
            continue
        ramp = member_ramp(hue, len(members))
        shape = community_shape(community_id, len(members))
        for step, member in zip(ramp, members, strict=True):
            styles[member] = CategoryStyle(
                fill=step,
                hue=hue,
                shape=shape,
                community_id=community_id,
                grouped=True,
            )
    return styles


# --------------------------------------------------------------------------- #
# The plate registry: one number per subject, shared by PDF page and SVG.
# --------------------------------------------------------------------------- #

PLATES: dict[str, int] = {
    "rules": 1,
    "redundancy": 2,
    "heatmap": 3,
    "network": 4,
    "segments": 5,
    "stability": 6,
    "backtest": 7,
}
N_PLATES = len(PLATES)


def plate_eyebrow(key: str) -> str:
    """The atlas eyebrow line, e.g. ``CATEGORY ATLAS · PLATE 04 / 07``."""
    return f"CATEGORY ATLAS · PLATE {PLATES[key]:02d} / {N_PLATES:02d}"


# --------------------------------------------------------------------------- #
# Shared SVG anatomy: plate header, marks, footer caption.
# --------------------------------------------------------------------------- #

SVG_MARGIN = 36


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_open(width: int, height: int) -> list[str]:
    """Document start: root element + full-bleed surface.

    Content-driven plates may open with a placeholder height and patch it via
    :func:`svg_close` once the real height is known.
    """
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT_STACK}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
    ]


def svg_close(parts: list[str], *, width: int, height: int) -> str:
    """Patch the document to its final height, close it, and join to a string."""
    parts[0] = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT_STACK}">'
    )
    parts[1] = f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>'
    parts.append("</svg>")
    return "\n".join(parts)


def svg_plate_header(
    parts: list[str],
    *,
    width: int,
    plate: str,
    title: str,
    subtitle: str,
    note: str | None = None,
) -> int:
    """Numbered plate header: eyebrow, title, subtitle, optional method note,
    closed by a hairline rule. Returns the y where content may begin."""
    x = SVG_MARGIN
    parts.append(
        f'<text x="{x}" y="30" font-size="10" letter-spacing="2.2" '
        f'fill="{INK_MUTED}">{svg_escape(plate_eyebrow(plate))}</text>'
    )
    parts.append(
        f'<text x="{x}" y="58" font-size="20" font-weight="bold" '
        f'fill="{INK}">{title}</text>'
    )
    parts.append(f'<text x="{x}" y="80" font-size="12.5" fill="{INK_SECONDARY}">{subtitle}</text>')
    rule_y = 92
    if note is not None:
        parts.append(f'<text x="{x}" y="98" font-size="11" fill="{INK_MUTED}">{note}</text>')
        rule_y = 110
    parts.append(
        f'<line x1="{x}" y1="{rule_y}" x2="{width - SVG_MARGIN}" y2="{rule_y}" '
        f'stroke="{GRID}" stroke-width="1"/>'
    )
    return rule_y + 26


def svg_footer_caption(parts: list[str], *, width: int, y: int, text: str) -> int:
    """The designed honesty caption: hairline rule + muted footnote.
    Returns the bottom y consumed."""
    parts.append(
        f'<line x1="{SVG_MARGIN}" y1="{y}" x2="{width - SVG_MARGIN}" y2="{y}" '
        f'stroke="{BASELINE}" stroke-width="1"/>'
    )
    parts.append(f'<text x="{SVG_MARGIN}" y="{y + 22}" font-size="10" fill="{INK_MUTED}">{text}</text>')
    return y + 34


def svg_mark(
    parts: list[str],
    *,
    x: float,
    y: float,
    r: float,
    fill: str,
    shape: str,
    stroke: str = SURFACE,
    stroke_width: float = 1.6,
) -> None:
    """One data mark in the community's shape, ringed in surface colour so
    overlapping marks stay separable (the 2px-surface-ring rule)."""
    common = f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"'
    if shape == "square":
        side = r * 1.8
        parts.append(
            f'<rect x="{x - side / 2:.1f}" y="{y - side / 2:.1f}" '
            f'width="{side:.1f}" height="{side:.1f}" rx="1.5" {common}/>'
        )
    elif shape == "diamond":
        d = r * 1.25
        parts.append(
            f'<polygon points="{x:.1f},{y - d:.1f} {x + d:.1f},{y:.1f} '
            f'{x:.1f},{y + d:.1f} {x - d:.1f},{y:.1f}" {common}/>'
        )
    else:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" {common}/>')
