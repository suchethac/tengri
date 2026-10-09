#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Shared drawing constants for the paper's demonstration figures.

The configuration census lived in three figure scripts at once, each with its
own copy. They agreed on the first three entries and none of them had the
other three, so every figure drew a six-configuration grid as though it were a
three-configuration one and no single edit could have caught it. One census,
imported.

The palette is Okabe-Ito, which stays distinguishable under the three common
forms of color blindness and in grayscale print.
"""

from __future__ import annotations

import numpy as np

#: The configurations Section 7 demonstrates, in the order it presents them.
#:
#: Configuration VI is deliberately NOT here (owner, 2026-09-28). Its row is
#: partial: nine cells of twenty, six of them adopted, and still running as of
#: 2026-09-28 -- re-read the count rather than trusting this line. The cost is
#: the reason for the deferral, not the count.
#: VI frees ``agn_lum_ratio``, a linear amplitude reaching zero, which is worse
#: conditioned than the log amplitude it replaced -- the adapted step size falls
#: about 4.5x and mean tree depth rises 6.21 -> 8.46, so a cell costs 4-5x more
#: gradient evaluations per draw. Attempt 1 is affordable at 2.4-6.1 h, but the
#: retune ladder is not: one cell's attempt-2 warmup alone ran 236,703 s.
#:
#: The AGN parametrization itself is sound and measured -- on the two galaxies
#: where the old prior floor dominated, divergences fell 107 -> 15 and 72 -> 8
#: with ``ess_min`` rising 75.3 -> 276.7 and 139.6 -> 284.3. What is missing is
#: wall-clock, not a working model. Keep the row's cells; do not draw it as one
#: of the demonstrated configurations.
CONFIG_ORDER: tuple[str, ...] = ("I", "II", "III", "IV", "V")

#: Run, kept, and not demonstrated. Held apart from CONFIG_ORDER rather than
#: deleted so a figure that wants to show the partial row can ask for it by
#: name, and so the palette below stays complete for those cells.
DEFERRED_CONFIGS: tuple[str, ...] = ("VI",)

CONFIG_COLORS: dict[str, str] = {
    "I": "#0072B2",  # blue
    "II": "#E69F00",  # orange
    "III": "#009E73",  # bluish green
    "IV": "#CC79A7",  # reddish purple
    "V": "#56B4E9",  # sky blue
    "VI": "#D55E00",  # vermillion
}

CONFIG_LABELS: dict[str, str] = {
    key: f"Configuration {key}" for key in (*CONFIG_ORDER, *DEFERRED_CONFIGS)
}

assert set(CONFIG_COLORS) == set(CONFIG_ORDER) | set(DEFERRED_CONFIGS), (
    "palette must cover every configuration that has cells, demonstrated or not"
)


#: SFR-panel limits come from this central range of all plotted values [percent],
#: padded by ``SFR_LIMIT_MARGIN`` [dex]; points outside are pinned at the edge.
SFR_LIMIT_PERCENTILES: tuple[float, float] = (2.0, 98.0)
SFR_LIMIT_MARGIN = 0.5

#: Marker pointing out of the panel for a point pinned at the left/right/bottom/top edge.
EDGE_MARKERS = {"left": "<", "right": ">", "bottom": "v", "top": "^"}


def robust_log_limits(values, default: tuple[float, float] = (-1.0, 2.0)) -> list[float]:
    """Axis limits [dex] from the bulk of ``values``, so one outlier cannot set the scale.

    Takes the ``SFR_LIMIT_PERCENTILES`` range of the finite values, pads it by
    ``SFR_LIMIT_MARGIN`` and rounds outward to 0.5 dex. ``default`` is returned when no
    value is finite.
    """
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return list(default)
    lo, hi = np.percentile(finite, SFR_LIMIT_PERCENTILES)
    return [
        float(np.floor((lo - SFR_LIMIT_MARGIN) * 2) / 2),
        float(np.ceil((hi + SFR_LIMIT_MARGIN) * 2) / 2),
    ]


def edge_side(x: float, y: float, lim: list[float]) -> str | None:
    """Edge a point outside ``lim`` is pinned to (the larger excursion), else ``None``."""
    lo, hi = lim
    excess = {
        "left": lo - x,
        "right": x - hi,
        "bottom": lo - y,
        "top": y - hi,
    }
    side, amount = max(excess.items(), key=lambda kv: kv[1])
    return side if amount > 0 else None
