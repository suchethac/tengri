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

CONFIG_ORDER: tuple[str, ...] = ("I", "II", "III", "IV", "V", "VI")

CONFIG_COLORS: dict[str, str] = {
    "I": "#0072B2",  # blue
    "II": "#E69F00",  # orange
    "III": "#009E73",  # bluish green
    "IV": "#CC79A7",  # reddish purple
    "V": "#56B4E9",  # sky blue
    "VI": "#D55E00",  # vermillion
}

CONFIG_LABELS: dict[str, str] = {key: f"Configuration {key}" for key in CONFIG_ORDER}

assert set(CONFIG_COLORS) == set(CONFIG_ORDER), (
    "palette and census must cover the same configurations"
)
