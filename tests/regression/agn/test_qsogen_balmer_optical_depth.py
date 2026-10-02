# SPDX-License-Identifier: BSD-3-Clause
"""QSOGen Balmer continuum: optical depth runs with the photoionisation cross-section.

The bound-free cross-section of hydrogen n=2 scales as sigma_bf ~ nu^-3
(Grandi 1982; Osterbrock & Ferland 2006), so the optical depth of the slab is

    tau(nu) = tau_BE * (nu_BE / nu)^3 = tau_BE * (lambda / lambda_BE)^3,

largest **at the Balmer edge** and falling toward the blue.  The original
QSOGen (Temple, Hewett & Banerji 2021, ``qsosed.py::add_balmer_continuum``)
writes this in frequency, ``tau = taube * (nuzero / nu)**3``.  The tengri
transcription wrote the ratio in wavelength as ``(lambda_BE / lambda)**3``,
which is the inverse: tau grew toward the blue.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.qsogen import _balmer_continuum

HC_OVER_K = 1.43877735e8  # h c / k_B [K Angstrom]
WAV_BE = 3646.0


def _b_lambda(wave, tbc):
    """QSOGen ``bb()`` convention: wav^-3 / (exp(hc/(k T wav)) - 1)."""
    return wave ** (-3.0) / np.expm1(HC_OVER_K / (tbc * wave))


def _reference(wave, cont, bcnorm, tbc, taube, wavbe):
    """Direct evaluation of the Grandi (1982) / QSOGen Balmer continuum.

    tau is computed in *frequency* exactly as upstream does:
    ``taube * (nu_zero / nu)**3``; normalised to ``bcnorm * continuum(3000 A)``.
    """
    c = 2.99792458e18  # Angstrom / s
    nu0 = c / wavbe

    def shape(w):
        tau = taube * (nu0 / (c / w)) ** 3
        return _b_lambda(w, tbc) * (1.0 - np.exp(-np.minimum(tau, 50.0)))

    cont3000 = np.interp(np.log10(3000.0), np.log10(wave), cont)
    scale = bcnorm * cont3000 / shape(3000.0)
    sigmoid = 1.0 / (1.0 + np.exp(50.0 * (np.log10(wave) - np.log10(wavbe))))
    return scale * shape(wave) * sigmoid


def test_optical_depth_increases_toward_the_balmer_edge():
    """With tau << 1 the BC is B * tau: (BC / B) must rise as lambda^3 toward the edge."""
    wave = np.linspace(1500.0, 2400.0, 400)
    full = np.geomspace(1000.0, 8000.0, 4000)
    cont = np.ones_like(full)
    bc = np.interp(
        wave,
        full,
        np.asarray(_balmer_continuum(jnp.asarray(full), jnp.asarray(cont), 1.0, 15000.0, 1e-6)),
    )
    ratio = bc / _b_lambda(wave, 15000.0)
    assert np.all(np.diff(ratio) > 0.0)
    slope = np.polyfit(np.log(wave), np.log(ratio), 1)[0]
    assert slope == pytest.approx(3.0, abs=0.02)


@pytest.mark.parametrize(
    ("bcnorm", "tbc", "taube"),
    [(1.0, 15000.0, 1.0), (0.7, 12000.0, 1.3), (2.0, 20000.0, 0.3)],
)
def test_balmer_continuum_matches_direct_grandi_evaluation(bcnorm, tbc, taube):
    wave = np.geomspace(1200.0, 7000.0, 3000)
    cont = 1e-3 * (wave / 3000.0) ** -0.5
    got = np.asarray(
        _balmer_continuum(jnp.asarray(wave), jnp.asarray(cont), bcnorm, tbc, taube, WAV_BE)
    )
    ref = _reference(wave, cont, bcnorm, tbc, taube, WAV_BE)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=0.0)


def test_no_balmer_continuum_at_default_zero_bcnorm():
    """agn_bcnorm=0 (the default) switches the component off exactly."""
    wave = np.geomspace(1200.0, 7000.0, 500)
    out = np.asarray(_balmer_continuum(jnp.asarray(wave), jnp.ones_like(jnp.asarray(wave)), 0.0))
    assert np.all(out == 0.0)
