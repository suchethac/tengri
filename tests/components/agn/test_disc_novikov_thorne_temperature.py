# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the accretion-disc inner temperature follows Novikov-Thorne theory.

The big-blue-bump peak wavelength of a thin accretion disc is set by the inner
temperature

.. math::

    T_{\\rm in} = \\left(\\frac{3 G M \\dot M}{8 \\pi \\sigma R_{\\rm in}^3}\\right)^{1/4},
    \\qquad \\dot M = \\frac{L_{\\rm bol}}{\\eta c^2},

with the Novikov-Thorne radiative efficiency :math:`\\eta = 1 - \\sqrt{1 - 2/(3 r_{\\rm ISCO})}`
(η ≈ 0.057 for a Schwarzschild hole, :math:`r_{\\rm ISCO} = 6\\,r_g`). The disc
νLν peak sits near the Wien peak of the hottest annulus, :math:`T_{\\max} \\approx
0.488\\,T_{\\rm in}`.

This pins tengri's ``multicolor_disc`` (Shakura-Sunyaev) and ``kubota_done``
(Kubota & Done 2018) discs to that analytic prediction, so a regression in the
efficiency, inner radius, or temperature normalisation cannot slip through. It
also documents why the Synthesizer-reproduction §9a disc peak differs: that
reference is Synthesizer's bundled *test* AGN grid, whose incident disc is ~2.4×
cooler than Novikov-Thorne theory — tengri is the faithful side (#695).

References
----------
.. [1] I. D. Novikov, K. S. Thorne, "Astrophysics of Black Holes," in
   *Black Holes (Les Astres Occlus)*, 343 (1973).
.. [2] A. Kubota, C. Done, "A physical model of the broadband continuum of AGN
   and its implications for the UV/X relation and optical variability,"
   MNRAS, 480, 1247 (2018). https://doi.org/10.1093/mnras/sty1890
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.agn.disc import kubota_done_disc, multicolor_disc

pytestmark = pytest.mark.regression_paper

# CGS constants.
_G = 6.674e-8
_C = 2.998e10
_SIGMA = 5.670e-5
_MSUN = 1.989e33
_LSUN = 3.828e33


def _analytic_nu_lnu_peak_aa(log_mbh: float, log_lbol: float, a_spin: float = 0.0) -> float:
    """Novikov-Thorne νLν peak wavelength [Å] for the hottest disc annulus."""
    # ISCO radius (Bardeen+1972); a=0 -> 6 r_g.
    z1 = 1 + (1 - a_spin**2) ** (1 / 3) * ((1 + a_spin) ** (1 / 3) + (1 - a_spin) ** (1 / 3))
    z2 = np.sqrt(3 * a_spin**2 + z1**2)
    r_isco = 3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))  # prograde
    eta = 1 - np.sqrt(1 - 2 / (3 * r_isco))
    m_g = 10**log_mbh * _MSUN
    r_g = _G * m_g / _C**2
    r_in = r_isco * r_g
    mdot = 10**log_lbol * _LSUN / (eta * _C**2)
    t_in = (3 * _G * m_g * mdot / (8 * np.pi * _SIGMA * r_in**3)) ** 0.25
    t_max = 0.488 * t_in
    return 5.10e7 / t_max  # Wien peak of B_nu [Å·K] / T


def _nu_lnu_peak(disc_fn, log_mbh, log_lbol, a_spin=0.0):
    wave = np.logspace(np.log10(50.0), np.log10(1.0e5), 4000)
    lnu = np.asarray(
        disc_fn(
            wave,
            agn_log_lbol=log_lbol,
            agn_log_mbh=log_mbh,
            agn_log_ledd=np.log10(0.5),
            agn_a_spin=a_spin,
        )
    )
    sel = (wave > 100.0) & (wave < 1.0e4) & (lnu > 0)
    w = wave[sel]
    return float(w[np.argmax((lnu * _C * 1e8 / wave)[sel])])


def test_multicolor_disc_peak_matches_novikov_thorne():
    """Shakura-Sunyaev disc νLν peak tracks the analytic NT inner temperature."""
    pred = _analytic_nu_lnu_peak_aa(log_mbh=8.0, log_lbol=12.215)
    got = _nu_lnu_peak(multicolor_disc, 8.0, 12.215)
    assert 0.8 < got / pred < 1.4, (
        f"multicolor disc νLν peak {got:.0f} Å vs Novikov-Thorne {pred:.0f} Å "
        f"(ratio {got / pred:.2f}) — disc temperature normalisation has drifted"
    )


def test_kubota_done_disc_peak_matches_novikov_thorne():
    """K&D three-zone disc thermal peak is consistent with NT theory.

    Slightly redder than the bare multicolor disc because the inner annuli are
    re-assigned to the warm/hot Comptonising zones — a physical, not numerical,
    shift — so the upper bound is a touch wider.
    """
    pred = _analytic_nu_lnu_peak_aa(log_mbh=8.0, log_lbol=12.215)
    got = _nu_lnu_peak(kubota_done_disc, 8.0, 12.215)
    assert 0.8 < got / pred < 1.6, (
        f"kubota_done disc νLν peak {got:.0f} Å vs Novikov-Thorne {pred:.0f} Å "
        f"(ratio {got / pred:.2f}) — disc temperature normalisation has drifted"
    )


def test_disc_peak_scales_with_eddington_ratio():
    """T_in ∝ (Ṁ/M)^{1/4} ∝ λ_Edd^{1/4} ⟹ νLν peak λ ∝ λ_Edd^{-1/4} (NT scaling).

    The inner temperature is driven by the Eddington ratio (``agn_log_ledd``),
    not the overall luminosity scaling (``agn_log_lbol`` sets amplitude only).
    """
    wave = np.logspace(np.log10(50.0), np.log10(1.0e5), 4000)

    def peak(log_ledd):
        lnu = np.asarray(
            multicolor_disc(wave, agn_log_lbol=12.215, agn_log_mbh=8.0, agn_log_ledd=log_ledd)
        )
        sel = (wave > 100.0) & (wave < 1.0e4) & (lnu > 0)
        return wave[sel][np.argmax((lnu * _C * 1e8 / wave)[sel])]

    # -1 dex in Eddington ratio -> peak redder by 10^{1/4} = 1.78x.
    ratio = peak(np.log10(0.05)) / peak(np.log10(0.5))
    assert abs(ratio - 10**0.25) / 10**0.25 < 0.15, (
        f"disc peak should scale as λ_Edd^-0.25 (expect 1.78x for 1 dex); got {ratio:.2f}"
    )
