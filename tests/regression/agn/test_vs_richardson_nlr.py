# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for NLR emission against the Richardson+2014 a42 template.

Measurement convention
----------------------
A line's strength is its **energy** flux, so it is the integral of ``L_nu``
over *frequency* across a window containing the line. Two other estimators
appear in this repository's history and neither measures that:

* ``int(sed * gaussian_profile) dnu`` -- a matched filter. For a line whose
  shape *is* the profile it returns ``A * int(phi^2 dnu)``, and
  ``int(phi^2 dnu)`` scales as ``1 / sigma_nu``, hence roughly as ``lambda_0``
  (measured 1.32 across Halpha/Hbeta against a wavelength ratio of 1.35 -- the
  remainder is the profile's normalization convention). Every line is
  therefore biased by its own wavelength. This is what this file used
  until the estimator was fixed; it made the test unpassable, which is why it
  carried ``xfail(strict=False)`` and asserted nothing for as long as it did.
* ``int(sed) dlambda`` with ``sed`` in ``L_nu`` -- mixes the two axes and
  biases a ratio by ``(lambda_1 / lambda_2)^2``. Guarded below.

Lines are computed at ``_FWHM_KMS = 100`` rather than the 500 km/s NLR default
so that [NII] 6585 and Halpha 6564 (21 A apart) do not blend; at 500 km/s
sigma_lambda is ~4.6 A and no window is clean. Line *ratios* do not depend on
the width -- every amplitude scales together -- so narrowing is free. Same
trick, and same reason, as ``tests/crossval/_nlr_measure.py``.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.nlr import (
    compute_nlr_sed,
    compute_nlr_sed_richardson2014,
)

#: Speed of light [Angstrom/s] and [km/s].
_C_AA = 2.99792458e18
_C_KMS = 2.99792458e5

#: Vacuum wavelengths [Angstrom]. Air wavelengths are never used (NAMING_CONTRACT).
_CENTERS_AA = {
    "oiii_5007": 5008.24,
    "oiii_4959": 4960.30,
    "hbeta": 4862.69,
    "nii_6584": 6585.27,
    "halpha": 6564.61,
}

#: Richardson+2014 (MNRAS 437, 2376) Table 3 column 'a42', normalized to Hbeta.
_A42_RATIOS = {
    "oiii_5007": 8.53,
    "oiii_4959": 2.87,
    "hbeta": 1.00,
    "nii_6584": 2.13,
    "halpha": 2.86,
}

#: Narrow enough to resolve [NII] 6585 away from Halpha, wide enough that the
#: grid resolving it stays cheap. At 100 km/s sigma_lambda is 0.93 A at Halpha
#: and the nearest neighbor ([NII] 6549.86) is 15.8 sigma away; the +-8 sigma
#: measurement windows of the two do not touch. The 500 km/s NLR default puts
#: them 3.2 sigma apart, which is why the default cannot be used here.
_FWHM_KMS = 100.0
_SIGMA_FRAC = (_FWHM_KMS / 2.354820045) / _C_KMS

#: +-8 sigma. The ratios are converged to four digits from 5 sigma upward.
_WINDOW_SIGMA = 8.0


@functools.lru_cache(maxsize=1)
def _nlr_spectrum():
    """Richardson a42 NLR spectrum on a grid fine enough to resolve 100 km/s lines.

    sigma_lambda is 0.93 A at Halpha, so the 0.067 A sampling below puts ~14
    points per sigma.

    Cached: building the grid is the whole cost of this module, and six tests
    share one build.
    """
    wave = jnp.linspace(4700.0, 6700.0, 30_000)
    sed = compute_nlr_sed_richardson2014(
        wavelength=wave,
        l_disc_bol_erg=1e45,
        covering_fraction=0.1,
        fwhm_kms=_FWHM_KMS,
        line_efficiency=0.10,
    )
    return wave, sed


def _line_flux(wave, sed, center_aa, *, axis="nu"):
    """Integrate one line over +-``_WINDOW_SIGMA`` about its center.

    ``axis='nu'`` is the physical energy flux. ``axis='lambda'`` exists only so
    a test can demonstrate the bias that choice introduces.
    """
    half_aa = _WINDOW_SIGMA * _SIGMA_FRAC * center_aa
    mask = jnp.abs(wave - center_aa) < half_aa
    x = _C_AA / wave if axis == "nu" else wave
    order = jnp.argsort(x)
    return float(jnp.abs(jnp.trapezoid(jnp.where(mask, sed, 0.0)[order], x[order])))


@functools.lru_cache(maxsize=2)
def _ratios_to_hbeta(axis="nu"):
    wave, sed = _nlr_spectrum()
    flux = {k: _line_flux(wave, sed, c, axis=axis) for k, c in _CENTERS_AA.items()}
    return {k: v / flux["hbeta"] for k, v in flux.items()}


@pytest.mark.regression_paper
def test_richardson_nlr_line_ratios():
    """The emitted spectrum reproduces the a42 template it is built from.

    Every ratio lands within 0.05 % of the table (the assertion allows 5 %, the
    tolerance this test has always documented). Before the estimator was fixed
    the same five lines came out 36 %, 62 %, 7 % and 93 % wrong -- see
    ``test_the_matched_filter_estimator_is_biased_by_wavelength``.

    References
    ----------
    Richardson et al. 2014, MNRAS, 437, 2376. Table 3, column 'a42'.
    """
    measured = _ratios_to_hbeta()
    for name, expected in _A42_RATIOS.items():
        rel = abs(measured[name] - expected) / expected
        assert rel < 0.05, (
            f"{name}: expected {expected:.2f}, got {measured[name]:.4f} ({100 * rel:.1f}% error)"
        )
        # The 5 % above is the documented contract, but the agreement is exact
        # to four digits, so on its own it has 5000x headroom and would not
        # notice real drift. Pin what is actually measured as well.
        assert rel < 1e-3, (
            f"{name}: {measured[name]:.5f} vs {expected} -- within the 5 % "
            f"contract but no longer reproducing the table exactly, which is "
            f"what a spectrum built from that table should do"
        )


@pytest.mark.regression_paper
def test_oiii_doublet_matches_its_transition_probabilities():
    """[OIII] 5007/4959 = 2.98, fixed by atomic physics rather than by the template.

    The ratio of the two ``1D2 -> 3P2,1`` branches is set by their Einstein A
    values, so it is the one number in this file a wrong template cannot make
    right by construction: the test above compares the spectrum to the table it
    was built from, this one compares the table to nature.
    """
    r = _ratios_to_hbeta()
    doublet = r["oiii_5007"] / r["oiii_4959"]
    assert abs(doublet - 2.98) / 2.98 < 0.01, (
        f"[OIII] 5007/4959 = {doublet:.4f}, expected 2.98 from the A-values"
    )


@pytest.mark.regression_bug
def test_the_matched_filter_estimator_is_biased_by_wavelength():
    """Pin the defect that left this file asserting nothing.

    ``int(sed * phi) dnu`` returns ``A * int(phi^2 dnu)`` for a Gaussian line,
    and ``int(phi^2 dnu) = 1 / (2 sigma_nu sqrt(pi))`` grows with ``lambda_0``.
    So the estimator reports each line scaled by roughly its own wavelength,
    and no tolerance on the resulting ratios can be met.

    Measured on Halpha/Hbeta the inflation is **1.32**, against a wavelength
    ratio of 1.35 -- the same size and direction, not the same number, because
    the exact factor also carries the profile's normalization convention. The
    assertion brackets the measurement rather than asserting the idealized
    ``lambda`` law, which is accurate to about 2 % and no better.
    """
    from tengri.components.agn._phys import gaussian_line_profile

    wave, sed = _nlr_spectrum()
    nu = _C_AA / wave
    order = jnp.argsort(nu)

    def matched(center_aa):
        phi = gaussian_line_profile(wave, center_aa, _FWHM_KMS)
        return float(jnp.abs(jnp.trapezoid((sed * phi)[order], nu[order])))

    biased = matched(_CENTERS_AA["halpha"]) / matched(_CENTERS_AA["hbeta"])
    honest = _ratios_to_hbeta()["halpha"]
    lam_ratio = _CENTERS_AA["halpha"] / _CENTERS_AA["hbeta"]

    inflation = biased / honest
    assert 1.20 < inflation < 1.45, (
        f"matched-filter/true = {inflation:.4f}; expected an inflation of the "
        f"order of the wavelength ratio {lam_ratio:.4f} (measured 1.32). "
        f"Outside this bracket the bias model in the module docstring is wrong."
    )
    assert inflation > 1.0, "the bias must inflate the redder line, not deflate it"
    assert abs(biased - _A42_RATIOS["halpha"]) / _A42_RATIOS["halpha"] > 0.20, (
        "the old estimator now agrees with the table; if that is real, this "
        "guard and the xfail it replaced are both obsolete"
    )


@pytest.mark.regression_bug
def test_integrating_l_nu_over_wavelength_biases_a_ratio():
    """``int(L_nu) dlambda`` is not a flux, and the error is wavelength-dependent.

    Mixing the two axes multiplies a ratio by ``(lambda_1 / lambda_2)^2`` --
    2 % across the [OIII] doublet, but a factor 1.8 between Halpha and Hbeta.
    Stated here because the convention is easy to reintroduce in any helper
    that measures lines from a spectrum.
    """
    nu_r = _ratios_to_hbeta("nu")
    lam_r = _ratios_to_hbeta("lambda")

    d_nu = nu_r["oiii_5007"] / nu_r["oiii_4959"]
    d_lam = lam_r["oiii_5007"] / lam_r["oiii_4959"]
    predicted = (_CENTERS_AA["oiii_5007"] / _CENTERS_AA["oiii_4959"]) ** 2

    assert abs((d_lam / d_nu) / predicted - 1.0) < 0.002, (
        f"lambda/nu doublet ratio = {d_lam / d_nu:.4f}, expected {predicted:.4f}"
    )
    assert abs(lam_r["halpha"] - _A42_RATIOS["halpha"]) / _A42_RATIOS["halpha"] > 0.5, (
        "the wavelength-axis estimator no longer misses Halpha/Hbeta by >50%"
    )


@pytest.mark.regression_paper
def test_nlr_delegate_to_richardson():
    """``compute_nlr_sed`` delegates to the Richardson implementation."""
    wavelength = jnp.linspace(3000, 10000, 500)
    kwargs = dict(
        l_disc_bol_erg=1e45,
        covering_fraction=0.1,
        fwhm_kms=500.0,
        line_efficiency=0.10,
    )
    sed_delegated = compute_nlr_sed(wavelength=wavelength, **kwargs)
    sed_richardson = compute_nlr_sed_richardson2014(wavelength=wavelength, **kwargs)

    assert jnp.allclose(sed_delegated, sed_richardson, rtol=1e-6)
    assert float(jnp.max(sed_richardson)) > 0.0, "both SEDs are zero; allclose is vacuous"


@pytest.mark.contract
def test_the_measurement_window_actually_contains_the_line():
    """Guards the flaw found while writing this file.

    The first draft used the old centers (6564.0 for Halpha, 0.61 A off the
    vacuum value) with a +-5 sigma window. At 20 km/s that window is +-0.93 A,
    so it sat over the line's wing and Halpha measured 14 % low -- a wrong
    number that looked like a physics discrepancy. Asserting that the window
    captures essentially all of the line makes that failure mode loud.
    """
    wave, sed = _nlr_spectrum()
    for name, center in _CENTERS_AA.items():
        narrow = _line_flux(wave, sed, center)
        # 12 sigma, not more: a Gaussian is complete to ~1e-15 by 8 sigma, while
        # a 30 sigma reference at 100 km/s spans +-28 A and swallows Halpha,
        # 20.7 A from [NII] 6585 -- the reference would be the blend.
        half_aa = 12.0 * _SIGMA_FRAC * center
        mask = jnp.abs(wave - center) < half_aa
        nu = _C_AA / wave
        order = jnp.argsort(nu)
        wide = float(jnp.abs(jnp.trapezoid(jnp.where(mask, sed, 0.0)[order], nu[order])))
        assert narrow / wide > 0.999, (
            f"{name}: the +-{_WINDOW_SIGMA} sigma window holds only "
            f"{100 * narrow / wide:.2f}% of the line"
        )


if __name__ == "__main__":
    np.seterr(all="raise")
    test_richardson_nlr_line_ratios()
    test_oiii_doublet_matches_its_transition_probabilities()
    test_nlr_delegate_to_richardson()
    print("All NLR regression tests passed!")
