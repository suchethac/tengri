# SPDX-License-Identifier: BSD-3-Clause
"""AGN disc SEDs do not depend on the wavelength grid they are evaluated on (#2572).

A disc's normalization is a physical integral (the ring blackbody power, the disc bolometric,
the EUV-tail budget). It was a trapezoid on the CALLER's grid: the same disc evaluated on a grid
reaching 0.01 A and one starting at 10 A differed by a flat -4e-3 across 10-912 A (3.7e-3 at
log M_BH 6, -4.1e-3 at 8, -4.5e-3 at 10). The normalizations are now closed form (ring
``int B_nu dnu = (sigma/pi) T^4``; the power-law tail integral) or on fixed internal nodes (the
EUV-band excess, the ``powerlaw_disc`` band), so an SED value at a wavelength is a function of
that wavelength alone.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.agn.disc import (
    kubota_done_disc,
    multicolor_disc,
    powerlaw_disc,
)

pytestmark = pytest.mark.conservation

_LSUN = 3.828e33
_C_AA = 2.99792458e18
#: Wavelengths [A] at which the two grids are compared (all inside both grids).
_PROBES = np.geomspace(12.0, 5.0e7, 40)


def _grid(lo_aa, n, probes=_PROBES):
    """Log grid ``lo_aa``..1e8 A (``n`` nodes) plus the probe wavelengths, and the probe index."""
    wave = np.unique(np.concatenate([np.geomspace(lo_aa, 1.0e8, n), probes]))
    return wave, np.searchsorted(wave, probes)


def _disc_fns():
    def powerlaw(wave, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return powerlaw_disc(wave, kw["agn_log_lbol"])

    return {"kubota_done": kubota_done_disc, "multicolor": multicolor_disc, "powerlaw": powerlaw}


@pytest.mark.parametrize("disc", ["kubota_done", "multicolor", "powerlaw"])
@pytest.mark.parametrize("log_mbh", [6.0, 8.0, 10.0])
def test_sed_at_a_wavelength_is_independent_of_the_grid(disc, log_mbh):
    """L_nu at 40 probe wavelengths agrees to 1e-6 between two grids.

    A 10 A - 1e8 A grid (450 nodes) and a 0.01 A - 1e8 A grid (510 nodes): coarse enough
    (d ln lambda ~ 0.03) that a trapezoid of a hot ring's Planck peak on the grid is wrong at
    the 1e-4..1e-3 level, which is what the SSP-sampled production grids do.
    """
    fn = _disc_fns()[disc]
    wave_a, ia = _grid(10.0, 450)
    wave_b, ib = _grid(0.01, 510)
    kw = {"agn_log_lbol": 11.5, "agn_log_mbh": log_mbh}
    sed_a = np.asarray(fn(jnp.asarray(wave_a), **kw))[ia]
    sed_b = np.asarray(fn(jnp.asarray(wave_b), **kw))[ib]
    assert np.all(sed_a > 0) and np.all(np.isfinite(sed_a))
    rel = np.abs(sed_a / sed_b - 1.0)
    assert rel.max() < 1e-6, f"{disc} logM={log_mbh}: worst grid dependence {rel.max():.3e}"


def _integrated_power(disc, cos_inc, log_lbol=11.5):
    """``int L_nu dnu`` [erg/s] on a dense 0.01 A - 1e8 A log grid."""
    wave = np.geomspace(0.01, 1.0e8, 40001)
    nu = _C_AA / wave
    lnu = np.asarray(
        _disc_fns()[disc](
            jnp.asarray(wave),
            agn_log_lbol=log_lbol,
            agn_log_mbh=8.0,
            agn_lum_ratio=1.0,
            agn_cos_inc=cos_inc,
        )
    )
    return float(np.trapezoid(lnu[::-1], nu[::-1]))


@pytest.mark.parametrize("disc", ["kubota_done", "multicolor"])
def test_integrated_lbol_equals_the_declared_target(disc):
    """At cos i = 0.5 the SED integrates to the accretion power ``L_acc``.

    ``L_nu(i) = 2 cos i D_nu + H_nu`` with ``int (D_nu + H_nu) dnu = L_acc``, so the power
    equals ``10**agn_log_lbol L_sun`` at ``cos i = 0.5``. A dense 0.01 A - 1e8 A log grid; the
    residual is that grid's own trapezoid error, which the closed-form normalization removes
    from the SED itself.
    """
    total = _integrated_power(disc, 0.5)
    np.testing.assert_allclose(total / (10.0**11.5 * _LSUN), 1.0, atol=2e-3)


def test_multicolor_power_at_the_default_inclination_is_two_cos_i_lacc():
    """A pure disc integrates to ``2 cos i L_acc`` at any inclination (no corona)."""
    cos_i = 0.866
    total = _integrated_power("multicolor", cos_i)
    np.testing.assert_allclose(total / (10.0**11.5 * _LSUN), 2.0 * cos_i, atol=2e-3 * 2.0 * cos_i)


def test_kubota_done_power_is_two_cos_i_disc_plus_corona():
    """``P(c) = 2 c D + H``: ``P(1) - P(0) = 2 D``, ``P(0) = H``, ``D + H = L_acc``."""
    p0, p1, p_half = (_integrated_power("kubota_done", c) for c in (0.0, 1.0, 0.5))
    d_disc, h_corona = 0.5 * (p1 - p0), p0
    np.testing.assert_allclose((d_disc + h_corona) / (10.0**11.5 * _LSUN), 1.0, atol=2e-3)
    np.testing.assert_allclose(p_half, d_disc + h_corona, rtol=1e-9)
    cos_i = 0.866
    p_i = _integrated_power("kubota_done", cos_i)
    np.testing.assert_allclose(p_i, 2.0 * cos_i * d_disc + h_corona, rtol=1e-9)


def test_ring_power_is_two_cos_i_times_the_two_face_power():
    """Energy budget: a ring's L_nu integrates to ``2 cos i`` times its two-face power.

    ``L_nu = 4 pi B_nu(T) 2 pi r dr cos i`` (``ring_area``) integrates to
    ``4 sigma T^4 2 pi r dr cos i = 2 cos i x (2 sigma T^4 2 pi r dr)``: ``2 cos i`` times the
    ring's two-face power ``dD = 2 sigma T^4 2 pi r dr``, which is what the bolometric
    normalizers sum (no cos i). The closed form the normalizers use
    (``_BNU_BOL_PER_T4 T^4``) equals the Stefan-Boltzmann one, and the numerical integral of
    the ring's ``L_nu``. The hot corona enters with its full isotropic ``l_hot_erg``.
    """
    from tengri.components.agn import disc as D
    from tengri.components.agn._phys import ring_area

    t, r, dr, cos_i = 4.0e4, 3.0e15, 1.0e14, 0.6
    closed = D._BNU_BOL_PER_T4 * t**4 * float(ring_area(r, dr, cos_i))
    two_face = 2.0 * D._SIGMA_SB * t**4 * 2.0 * np.pi * r * dr
    np.testing.assert_allclose(
        closed, 2.0 * cos_i * two_face, rtol=2e-5
    )  # sigma_Planck vs sigma_SB
    wave = np.geomspace(1.0, 1.0e8, 200001)
    nu = _C_AA / wave
    lnu = np.asarray(D._planck_lnu(jnp.asarray(nu), t)) * float(ring_area(r, dr, cos_i))
    np.testing.assert_allclose(np.trapezoid(lnu[::-1], nu[::-1]), closed, rtol=1e-5)
