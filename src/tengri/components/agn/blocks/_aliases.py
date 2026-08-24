# SPDX-License-Identifier: BSD-3-Clause
"""Deprecated `lines` slot → (nlr, blr) alias expansion.

The composable AGN block grammar previously held both narrow-line region (NLR)
and broad-line region (BLR) under a single `lines` category. This module
provides the canonical mapping table and expansion helper for code that still
uses the deprecated `lines` selector, so it can emit a `DeprecationWarning` and
forward to the new independent `nlr` and `blr` categories.
"""

from __future__ import annotations

__all__ = ["expand_lines_alias"]

#: Canonical deprecated `lines` → (nlr, blr) mapping.
#: Each legacy `lines` block name maps to an (nlr_name, blr_name) pair.
_LINES_ALIAS: dict[str, tuple[str, str]] = {
    "none": ("none", "none"),
    "nlr": ("analytic", "none"),
    "blr": ("none", "analytic"),
    "nlr_synthesizer": ("synthesizer", "none"),
    "blr_synthesizer": ("none", "synthesizer"),
    "nlr_synthesizer_spectra": ("synthesizer_spectra", "none"),
    "blr_synthesizer_spectra": ("none", "synthesizer_spectra"),
    "nlr_blr": ("analytic", "analytic"),
    "nlr_blr_synthesizer": ("synthesizer", "synthesizer"),
    "nlr_blr_synthesizer_spectra": ("synthesizer_spectra", "synthesizer_spectra"),
    "grahsp": ("grahsp", "grahsp"),
    "qsogen": ("none", "qsogen"),
}


def expand_lines_alias(lines_type: str) -> tuple[str, str]:
    """Expand a deprecated `lines` block name to (nlr, blr) pair.

    Parameters
    ----------
    lines_type : str
        The legacy `lines` block name (e.g. ``"nlr"``, ``"blr"``,
        ``"nlr_blr"``).

    Returns
    -------
    tuple of str
        ``(nlr_block_name, blr_block_name)`` corresponding to the input.

    Raises
    ------
    ValueError
        If ``lines_type`` is not in the canonical mapping table.

    Notes
    -----
    This function is used internally by deprecated-alias entry points
    (e.g. ``Parameters(agn_lines_block=...)`` in the flat kwarg path) to
    emit a ``DeprecationWarning`` and convert the old selector to new
    independent `nlr` and `blr` names.

    Examples
    --------
    >>> expand_lines_alias("nlr_blr")
    ('analytic', 'analytic')
    >>> expand_lines_alias("qsogen")
    ('none', 'qsogen')
    """
    if lines_type not in _LINES_ALIAS:
        available = sorted(_LINES_ALIAS.keys())
        raise ValueError(
            f"Unknown deprecated 'lines' block type {lines_type!r}. Available: {available}."
        )
    return _LINES_ALIAS[lines_type]
