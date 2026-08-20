# SPDX-License-Identifier: BSD-3-Clause
r"""CB19 emission lines are exactly zero in pure float32 (#1568).

Not NaN, not ``inf`` — **zero**, which propagates silently and reads as a galaxy
with no emission lines. Two independent underflows, either of which alone is
sufficient, so fixing one changes nothing:

1. **The Q_H table.** ``compute_qh`` divides :math:`L_\nu` by :math:`h\nu`,
   which lands at ~1e41 before the trapezoid and ~1e46 after — both past
   float32's 3.4e38 ceiling. Every entry overflows to ``inf``, and
   ``sanitize_qh_table`` then rewrites non-finite to ``0.0``.

   That sanitizer has a guard for exactly this (#1491) and it cannot fire here.
   It separates dtype overflow from honest missing-UV data by asking whether the
   *surviving finite* entries sit against the dtype ceiling::

       largest_finite = max(where(finite, |qh|, 0.0))
       if largest_finite > 0.01 * ceiling: raise QHTableOverflowError

   When **every** entry overflows there are no survivors, ``largest_finite`` is
   ``0.0``, the condition is False, and the whole table is silently zeroed. A
   guard that infers overflow from the survivors is blind to total overflow —
   the worst case, not an edge case.

2. **The Hβ conversion constant.** ``_HB_PER_QH_LSUN = 4.78e-13 / L_sun =
   1.2487e-46`` is below float32's smallest *subnormal* (1.4013e-45), so it is
   ``0.0`` exactly. The code already documents this at the use site and casts
   with ``.astype(jnp.float64)`` — which is a no-op under
   ``jax.enable_x64(False)``, i.e. inert in precisely the mode it guards.

Both are fixed by keeping the chain in log space until the two out-of-range
constants have been combined into one in-range constant, which is what
``_integrate_nion_log10`` already does for the stellar path.
"""

import jax
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

_PARAMS = {"sfh_delayed_log_total_mass": 10.0, "sfh_delayed_tau_gyr": 1.0}


def _model(ssp):
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    scaled = SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * 1.0e-17,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )
    return SEDModel.build(
        ssp_data=scaled,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        redshift=Fixed(0.1),
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": Uniform(0.5, 3.0),
            "age_gyr": Fixed(5.0),
        },
        dust_attenuation={"law": "power_law", "type": "two_component", "all_params": FIXED},
        neb={"type": "cb19", "all_params": FIXED},
    )


def _lines(ssp, *, x64):
    """``(published erg/s, published log10)``.

    The two are not interchangeable, and only one can be finite in float32.
    ``line_lums`` is ~1e42 **erg/s** and does not fit float32 at all — that is
    the deliberate consequence of #1559 and the whole reason ``log_line_lums``
    exists. So the float32 contract is carried entirely by the log companion,
    which is derived on the [Lsun] side (~1e8) before the erg/s multiply.
    """
    with jax.enable_x64(x64):
        d = _model(ssp).predict_state(_PARAMS).derived
        return (
            np.asarray(d["line_lums"], dtype=np.float64),
            np.asarray(d["log_line_lums"], dtype=np.float64),
        )


@pytest.fixture(scope="module")
def both(synthetic_ssp_wide):
    return _lines(synthetic_ssp_wide, x64=True), _lines(synthetic_ssp_wide, x64=False)


def test_setup_float64_really_has_lines(both):
    """Guard the guard: if float64 were also dark, the float32 test proves nothing."""
    (lin64, log64), _ = both
    assert (lin64 > 0).all(), (
        f"float64 CB19 lines are not all positive: {lin64}. This fixture no longer "
        "exercises the defect"
    )
    assert np.isfinite(log64).all(), f"float64 log companion is not finite: {log64}"


def test_cb19_lines_are_not_silently_zero_in_float32(both):
    """The defect. A global zero is the failure mode ratio tests cannot see."""
    (_, log64), (_, log32) = both

    n_zero = int(np.isneginf(log32).sum())
    assert n_zero == 0, (
        f"{n_zero}/{log32.size} CB19 line luminosities are EXACTLY zero in pure float32 "
        f"(log = -inf) while float64 gives {log64.min():.2f}..{log64.max():.2f} dex. Zero "
        "propagates silently and reads as a galaxy with no lines — every per-backend test "
        "asserts ratios or runs in float64, and a global zero fails none of them (#1568)"
    )
    assert np.isfinite(log32).all(), (
        f"CB19 float32 log companion is non-finite: {log32}. It is derived on the [Lsun] "
        "side (~1e8), which fits float32 comfortably — a +inf here means the erg/s scale "
        "leaked back upstream of the log"
    )


def test_cb19_float32_lines_agree_with_float64(both):
    """Finite and non-zero is not enough — they must be the same luminosities."""
    (_, log64), (_, log32) = both
    usable = np.isfinite(log64) & np.isfinite(log32)
    assert usable.any(), "no comparable lines"

    # Absolute dex: these are logs, so a dex offset IS the error scale. 1e-3 dex
    # is 0.23% in linear — well above float32's ~7-digit noise on this chain and
    # far below any real mistake.
    delta = np.abs(log32[usable] - log64[usable]).max()
    assert delta < 1.0e-3, (
        f"CB19 float32 lines differ from float64 by {delta:.3e} dex. Finite, non-zero "
        "and wrong is the failure a zero-check cannot see"
    )


def test_the_published_erg_per_second_key_is_expected_to_overflow(both):
    """Pins the boundary, so the log companion's reason for existing stays visible.

    ``line_lums`` is [erg/s] (#1559) at ~1e42 — float32 cannot hold it, and that
    is correct rather than a defect. If this ever starts fitting, either the
    fixture stopped being physical or the unit silently changed back, and both
    are things the next person should be told rather than left to infer.
    """
    (lin64, _), (lin32, _) = both
    assert lin64.max() > 3.4e38, (
        f"float64 line_lums peak is {lin64.max():.3e}, inside the float32 window — this "
        "fixture no longer exercises the boundary the log companion exists for"
    )
    assert not np.isfinite(lin32).all(), (
        "line_lums now fits float32. If the erg/s unit changed, update #1559's contract "
        "and this file together"
    )


def test_the_log_companion_does_not_report_underflow_as_a_dark_line(both):
    """``-inf`` means 'exactly zero', which is a legitimate value — and a trap here.

    When the linear value underflows, ``log10_magnitude`` returns ``-inf``: the
    companion faithfully reports a destroyed answer as a genuinely dark line.
    Underflow-to-zero and true-zero share a sentinel, so the companion cannot be
    used to detect this class on its own — it has to agree with float64.
    """
    (lin64, log64), (_, log32) = both
    bright = lin64 > 0
    lost = bright & np.isneginf(log32) & np.isfinite(log64)
    assert not lost.any(), (
        f"{int(lost.sum())} lines report log = -inf ('exactly zero') in float32 while "
        f"float64 has them at {log64[lost][:3]} dex. The sentinel is honest and the "
        "value is not — nothing downstream can tell this from a real dark line (#1568)"
    )
