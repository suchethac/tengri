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


@pytest.mark.parametrize("disc", ["kubota_done", "multicolor"])
def test_integrated_lbol_equals_the_declared_target(disc):
    """On a grid that covers the emission the SED integrates to ``L_bol``.

    A dense 0.01 A - 1e8 A log grid; the residual is that grid's own trapezoid error, which
    the closed-form normalization removes from the SED itself.
    """
    wave = np.geomspace(0.01, 1.0e8, 40001)
    nu = _C_AA / wave
    lnu = np.asarray(
        _disc_fns()[disc](jnp.asarray(wave), agn_log_lbol=11.5, agn_log_mbh=8.0, agn_lum_ratio=1.0)
    )
    total = float(np.trapezoid(lnu[::-1], nu[::-1]))
    np.testing.assert_allclose(total / (10.0**11.5 * _LSUN), 1.0, atol=2e-3)


def test_ring_power_is_one_face_cos_i_weighted():
    """Energy budget as implemented: a ring's bolometric content is ONE face, projected by cos i.

    ``L_nu = pi B_nu(T) 2 pi r dr cos i`` integrates to ``sigma T^4 2 pi r dr cos i`` (not
    ``2 sigma T^4 ...``, not cos-free): the closed form the normalizers use equals the Stefan-
    Boltzmann one-face power, and equals the numerical integral of the ring's ``L_nu``. The
    bolometric sums of ``_compute_zone_luminosities`` and ``multicolor_disc`` use the same
    ``sigma T^4 2 pi r dr cos i`` per ring; the hot corona enters with its full ``l_hot_erg``.
    """
    from tengri.components.agn import disc as D
    from tengri.components.agn._phys import ring_area

    t, r, dr, cos_i = 4.0e4, 3.0e15, 1.0e14, 0.6
    closed = D._BNU_BOL_PER_T4 * t**4 * float(ring_area(r, dr, cos_i))
    one_face = D._SIGMA_SB * t**4 * 2.0 * np.pi * r * dr * cos_i
    np.testing.assert_allclose(closed, one_face, rtol=2e-5)  # sigma_Planck vs tabulated sigma_SB
    wave = np.geomspace(1.0, 1.0e8, 200001)
    nu = _C_AA / wave
    lnu = np.asarray(D._planck_lnu(jnp.asarray(nu), t)) * float(ring_area(r, dr, cos_i))
    np.testing.assert_allclose(np.trapezoid(lnu[::-1], nu[::-1]), closed, rtol=1e-5)
