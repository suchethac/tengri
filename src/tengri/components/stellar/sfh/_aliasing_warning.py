# SPDX-License-Identifier: BSD-3-Clause
"""Build-time warning for SFH-burst-vs-SSP-grid Nyquist aliasing (#299).

Background
----------
The forward model interpolates ``SFR(t)`` linearly at the SSP grid age
nodes:

.. code-block:: python

    sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)

This is a *point-sample*, not a bin-integral. If the SFH carries a
burst that is narrower than the local SSP grid spacing, the sample
falls between adjacent grid points differently as the burst peak slides
across grid boundaries: producing a non-physical staircase in any
quantity that varies smoothly with stellar age (e.g. the Mg b absorption
strength in #299).

The proper fix is conservative rebinning (bin-integral averaging across
the SSP grid). That's a non-trivial refactor of the CSP integration
path. Until then, this module emits a build-time warning when a fixed
or median-prior burst width is narrower than the SSP grid spacing at
the burst peak, so users hit the failure mode at construction rather
than discovering it as a visual artefact in their predictions.

Reference: tengri issue #299.
"""

from __future__ import annotations

import numpy as np

from tengri.config.exceptions import warn_measured


class SFHBurstAliasingWarning(UserWarning):
    """A burst-width SFH parameter is narrower than the SSP grid spacing.

    Predictions will exhibit a non-physical staircase as the burst peak
    crosses adjacent SSP grid boundaries: the SFR(t) → SSP interpolation
    is a point-sample, not a bin-integral. Workaround: widen the burst
    until ``width_gyr ≳ grid_spacing_at_peak``. See #299.
    """


# Map: ``sfh_<variant>_width_gyr`` → corresponding ``sfh_<variant>_peak_lbt_gyr``.
# Only SFH variants whose ``width_gyr`` is genuinely in Gyr (linear time,
# not dex) are checked. lnorm's ``sfh_lnorm_width_gyr`` is actually a
# log-space width (dex) and is intentionally excluded.
_BURST_WIDTH_TO_PEAK: dict[str, str] = {
    "sfh_tsnorm_width_gyr": "sfh_tsnorm_peak_lbt_gyr",
    "sfh_snorm_width_gyr": "sfh_snorm_peak_lbt_gyr",
    "sfh_snorm_burst_width_gyr": "sfh_snorm_burst_peak_lbt_gyr",
    "sfh_tsnorm_burst_width_gyr": "sfh_tsnorm_burst_peak_lbt_gyr",
    "sfh_norm_width_gyr": "sfh_norm_peak_lbt_gyr",
}


def _ssp_grid_spacing_yr_at(ssp_ages_yr: np.ndarray, age_yr: float) -> float:
    """Local SSP grid spacing (years) in the bin bracketing ``age_yr``.

    Returns the larger of the two adjacent log-spaced widths so the
    warning errs toward the "spacing too small" side (i.e. fires when
    burst width is at most marginal, not only when it's catastrophically
    narrow).
    """
    ages = np.asarray(ssp_ages_yr)
    # Clamp to grid extent for the edge cases.
    if age_yr <= float(ages[0]):
        return float(ages[1] - ages[0])
    if age_yr >= float(ages[-1]):
        return float(ages[-1] - ages[-2])
    idx_hi = int(np.searchsorted(ages, age_yr))
    idx_lo = max(0, idx_hi - 1)
    idx_hi = min(len(ages) - 1, idx_hi)
    width_lo = float(ages[idx_hi] - ages[idx_lo]) if idx_hi > idx_lo else 0.0
    # Also probe the next bin to the right (the staircase appears as the
    # peak crosses *any* nearby boundary).
    idx_next = min(len(ages) - 1, idx_hi + 1)
    width_hi = float(ages[idx_next] - ages[idx_hi]) if idx_next > idx_hi else width_lo
    return max(width_lo, width_hi)


def _representative_value(distribution) -> float | None:
    """Extract a representative scalar from a ``Distribution``.

    For ``Fixed``: the fixed value.
    For other distributions: the midpoint of ``(lo, hi)``.
    Returns ``None`` if the distribution isn't representable as a real.
    """
    if distribution is None:
        return None
    val = getattr(distribution, "value", None)
    if val is not None and not isinstance(val, str):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    bounds = getattr(distribution, "bounds", None)
    if bounds is None:
        return None
    try:
        lo, hi = bounds
    except (TypeError, ValueError):
        return None
    if isinstance(lo, str) or isinstance(hi, str):
        return None
    try:
        return 0.5 * (float(lo) + float(hi))
    except (TypeError, ValueError):
        return None


def maybe_warn_burst_aliasing(spec, ssp_ages_yr) -> None:
    """Emit :class:`SFHBurstAliasingWarning` for any too-narrow burst.

    Walks the spec's resolved parameter distributions, looks for known
    ``sfh_<variant>_width_gyr`` parameters, compares the (fixed value or
    median-of-prior) width to the SSP grid spacing at the burst peak,
    and warns once per offending pair.
    """
    distributions = getattr(spec, "_distributions", None)
    if not distributions:
        return
    ssp_ages_arr = np.asarray(ssp_ages_yr)
    if ssp_ages_arr.size < 2:
        return

    for width_name, peak_name in _BURST_WIDTH_TO_PEAK.items():
        if width_name not in distributions or peak_name not in distributions:
            continue
        width_gyr = _representative_value(distributions[width_name])
        peak_gyr = _representative_value(distributions[peak_name])
        if width_gyr is None or peak_gyr is None:
            continue
        if width_gyr <= 0 or peak_gyr <= 0:
            continue
        peak_yr = peak_gyr * 1e9
        spacing_yr = _ssp_grid_spacing_yr_at(ssp_ages_arr, peak_yr)
        spacing_gyr = spacing_yr * 1e-9
        if width_gyr < spacing_gyr:
            warn_measured(
                f"SFH burst width {width_name}={width_gyr:.3g} Gyr is narrower "
                f"than the SSP grid spacing {spacing_gyr:.3g} Gyr at peak "
                f"{peak_name}={peak_gyr:.3g} Gyr. Predictions will show a "
                f"non-physical staircase as the burst peak crosses SSP grid "
                f"boundaries (#299). Widen the burst to at least "
                f"width_gyr ≳ {spacing_gyr:.3g} for smooth behavior.",
                SFHBurstAliasingWarning,
                stacklevel=3,
                burst_width_gyr=width_gyr,
                ssp_spacing_gyr=spacing_gyr,
                burst_peak_gyr=peak_gyr,
            )
