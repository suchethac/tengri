# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for BLR emission against the Vanden Berk+2001 composite.

Measurement convention
----------------------
A line's strength is its **energy** flux: ``L_nu`` integrated over *frequency*
across a window containing the line. This file previously integrated
``sed * gaussian_profile`` over frequency, which for a line whose shape is the
profile returns ``A * int(phi^2 dnu)`` -- a matched filter, not a flux. Since
``int(phi^2 dnu)`` scales as ``1 / sigma_nu``, every line came out scaled by
roughly its own wavelength, so Lyman alpha was reported four times too faint
relative to Hbeta and the test could not pass at any tolerance. It carried
``xfail(strict=False)`` and therefore asserted nothing. See
``test_the_matched_filter_estimator_is_biased_by_wavelength`` in the sibling
``test_vs_richardson_nlr.py``, which pins the same defect on the NLR side.

Grid
----
The lines span Lyman alpha to Halpha, and resolving a 500 km/s profile at
1215 A (sigma_lambda = 0.86 A) across that whole range uniformly would take
~200k points. Trapezoid does not require a uniform grid, so the spectrum is
sampled only where the lines are: one fine window per line, concatenated and
sorted. ``test_the_composite_grid_does_not_change_the_answer`` checks that
against a uniform control, because the shortcut would be invalid if
``compute_blr_sed`` normalized against the grid it was handed.
"""

import functools

import jax.numpy as jnp
import pytest

from tengri.components.agn.blr import (
    _BLR_FWHM_KMS,
    _BLR_LINES,
    compute_blr_sed,
)

#: Speed of light [Angstrom/s] and [km/s].
_C_AA = 2.99792458e18
_C_KMS = 2.99792458e5

#: Vacuum wavelengths [Angstrom].
_CENTERS_AA = {
    "lya": 1215.67,
    "civ": 1549.06,
    "mgii": 2799.12,
    "hbeta": 4862.69,
    "halpha": 6564.61,
}

#: VB01 Table 2 relative fluxes (col. 4, in 100 * F/F_Lya) divided by the Hbeta
#: value 8.649, i.e. normalized to Hbeta. Carried in ``_BLR_LINES``.
_VB01_RATIOS = {
    "lya": 11.5660,  # 100.0  / 8.649
    "civ": 2.9237,  # 25.291 / 8.649
    "mgii": 1.7033,  # 14.725 / 8.649
    "hbeta": 1.00,  # reference
    "halpha": 3.5666,  # 30.832 / 8.649
}

#: Measured at 500 km/s, not the 5000 km/s BLR default. Line *ratios* do not
#: depend on the width -- every amplitude scales together -- but the windows do:
#: at 5000 km/s sigma_lambda is 8.6 A at Lyman alpha, so a +-8 sigma window
#: spans +-69 A and swallows N V 1240, while at Halpha it runs off the end of
#: any grid that starts blueward of Lyman alpha. Same reasoning, and the same
#: 8 sigma window, as the NLR sibling.
_FWHM_KMS = 500.0
assert _BLR_FWHM_KMS == 5000.0, (
    "the BLR default width moved; re-check that 500 km/s still separates the "
    "measured lines from their neighbours"
)
_SIGMA_FRAC = (_FWHM_KMS / 2.354820045) / _C_KMS
_WINDOW_SIGMA = 8.0


def _sed_on(wave):
    return compute_blr_sed(
        wavelength=wave,
        l_disc_bol_erg=1e45,
        covering_fraction=0.1,
        fwhm_kms=_FWHM_KMS,
        agn_fe2_strength=0.0,  # Fe II off: it is a pseudo-continuum under the lines
        line_efficiency=0.08,
    )


@functools.lru_cache(maxsize=1)
def _composite_spectrum():
    """One fine window per measured line, concatenated. ~8k points."""
    pieces = [
        jnp.linspace(c * (1 - 14 * _SIGMA_FRAC), c * (1 + 14 * _SIGMA_FRAC), 1600)
        for c in _CENTERS_AA.values()
    ]
    wave = jnp.sort(jnp.concatenate(pieces))
    return wave, _sed_on(wave)


def _line_flux(wave, sed, center_aa, *, n_sigma=_WINDOW_SIGMA):
    """Energy flux of one line: ``L_nu`` integrated over frequency."""
    half_aa = n_sigma * _SIGMA_FRAC * center_aa
    mask = jnp.abs(wave - center_aa) < half_aa
    nu = _C_AA / wave
    order = jnp.argsort(nu)
    return float(jnp.abs(jnp.trapezoid(jnp.where(mask, sed, 0.0)[order], nu[order])))


@functools.lru_cache(maxsize=1)
def _ratios_to_hbeta():
    wave, sed = _composite_spectrum()
    flux = {k: _line_flux(wave, sed, c) for k, c in _CENTERS_AA.items()}
    return {k: v / flux["hbeta"] for k, v in flux.items()}


@pytest.mark.regression_paper
def test_blr_vanden_berk_line_ratios():
    """The emitted BLR spectrum reproduces the VB01 strengths it is built from.

    All five lines agree to four digits. The 5 % bound is the tolerance this
    test has always documented; the tighter assertion is what is actually
    measured, because 5 % against an exact result would not notice drift.

    References
    ----------
    Vanden Berk, D. E., et al. 2001, AJ, 122, 549. https://doi.org/10.1086/321167
    """
    measured = _ratios_to_hbeta()
    for name, expected in _VB01_RATIOS.items():
        rel = abs(measured[name] - expected) / expected
        assert rel < 0.05, (
            f"{name}: expected {expected:.4f}, got {measured[name]:.4f} ({100 * rel:.1f}% error)"
        )
        assert rel < 1e-3, (
            f"{name}: {measured[name]:.5f} vs {expected} -- inside the 5 % "
            f"contract but no longer reproducing the VB01 table exactly"
        )


@pytest.mark.regression_bug
def test_the_matched_filter_estimator_is_biased_by_wavelength():
    """Pin the defect that left this file asserting nothing.

    ``int(sed * phi) dnu`` reports each line scaled by roughly its own
    wavelength, so relative to Hbeta the bias runs from 0.25x at Lyman alpha to
    1.35x at Halpha. Lyman alpha is the extreme case and the one that made the
    5 % tolerance unreachable: the old estimator put it at ~2.9 against a table
    value of 11.57.
    """
    from tengri.components.agn._phys import gaussian_line_profile

    wave, sed = _composite_spectrum()
    nu = _C_AA / wave
    order = jnp.argsort(nu)

    def matched(center_aa):
        phi = gaussian_line_profile(wave, center_aa, _FWHM_KMS)
        return float(jnp.abs(jnp.trapezoid((sed * phi)[order], nu[order])))

    biased = matched(_CENTERS_AA["lya"]) / matched(_CENTERS_AA["hbeta"])
    honest = _ratios_to_hbeta()["lya"]
    lam_ratio = _CENTERS_AA["lya"] / _CENTERS_AA["hbeta"]

    deflation = biased / honest
    assert 0.20 < deflation < 0.30, (
        f"matched-filter/true = {deflation:.4f} for Lyman alpha; expected a "
        f"deflation of the order of the wavelength ratio {lam_ratio:.4f}"
    )
    assert abs(biased - _VB01_RATIOS["lya"]) / _VB01_RATIOS["lya"] > 0.5, (
        "the old estimator now agrees with the table to better than 50%; if "
        "that is real, this guard and the xfail it replaced are both obsolete"
    )


def test_the_composite_grid_does_not_change_the_answer():
    """The cheap grid is only valid if the normalization ignores it.

    If ``compute_blr_sed`` set its scale from the extent of the wavelength
    array, sampling only near the lines would silently renormalize every
    strength. A coarse uniform grid over the full span must give the same
    ratios.
    """
    uniform = jnp.linspace(1150.0, 6700.0, 25_000)
    sed = _sed_on(uniform)
    flux = {k: _line_flux(uniform, sed, c) for k, c in _CENTERS_AA.items()}
    ratios = {k: v / flux["hbeta"] for k, v in flux.items()}

    composite = _ratios_to_hbeta()
    for name in _CENTERS_AA:
        assert abs(ratios[name] - composite[name]) / composite[name] < 1e-3, (
            f"{name}: uniform grid gives {ratios[name]:.5f}, composite "
            f"{composite[name]:.5f} -- the normalization depends on the grid"
        )


def test_the_measurement_window_actually_contains_the_line():
    """A window narrower than the line reads as a physics discrepancy.

    Found while writing the NLR sibling: an off-by-0.6 A line centre against a
    window of +-0.9 A put Halpha 14 % low, which looks exactly like a real
    disagreement with the reference.
    """
    wave, sed = _composite_spectrum()
    for name, center in _CENTERS_AA.items():
        narrow = _line_flux(wave, sed, center)
        wide = _line_flux(wave, sed, center, n_sigma=12.0)
        assert narrow / wide > 0.999, (
            f"{name}: the +-{_WINDOW_SIGMA} sigma window holds only "
            f"{100 * narrow / wide:.2f}% of the line"
        )


@pytest.mark.regression_paper
def test_blr_line_count():
    """BLR line list covers all the major VB01 Table 2 broad permitted lines."""
    # 23 broad lines from VB01: Lyb, Lya, NV, SiII, CII, SiIV, OIV] (= 2-line
    # blend), CIV, HeII, OIII], AlIII, SiIII], CIII], CII], NeIV, MgII, He,
    # Hd, Hg, Hb, Ha, Pab, Pag. Paschen lines are approximate, the rest are
    # paper-traceable (see _BLR_LINES inline comments).
    n_lines = _BLR_LINES.shape[0]
    assert n_lines >= 23, (
        f"Expected >=23 BLR lines, got {n_lines}. "
        "If you removed a line from _BLR_LINES, update this assertion."
    )


@pytest.mark.regression_paper
def test_blr_covers_uv_optical():
    """Verify that BLR line list spans UV to optical wavelengths."""
    min_wave = jnp.min(_BLR_LINES[:, 0])
    max_wave = jnp.max(_BLR_LINES[:, 0])

    assert min_wave < 1250, f"Minimum wavelength {min_wave} should be <1250 A (Lya region)"
    assert max_wave > 6000, f"Maximum wavelength {max_wave} should be >6000 A (Ha region)"
