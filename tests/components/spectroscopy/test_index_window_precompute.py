# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the FeaturePrecomp index-window LUT reproduces measure_index_jax.

A break / EW spectral index is a scale-invariant functional of the SED, and the
SED is a weight-sum of SSP spectra, so an index measured on
``SED = Σ_ij w_ij · SSP_ij`` equals the same index evaluated from per-window SSP
integrals precomputed once at build time (the WavePrecomp-analog for spectral
features). This pins that the LUT path is *bit-exact* with the exact
``measure_index_jax`` on a dust-free SED — the window mean commutes with the
SFH weight sum — so the fast feature path cannot silently drift from the exact
one. Slope indices are not expressible from a single window integral and must
be flagged for the exact fallback.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    measure_index_jax,
    measure_indices_from_windows,
    precompute_index_windows,
)

pytestmark = pytest.mark.contract

_BREAK_EW = ["Dn4000", "D4000", "HdA", "HdF", "Hbeta", "Mgb", "HgA", "Fe5270"]


def _mock_ssp(n_met=4, n_age=12, n_wave=4000):
    """A smooth synthetic SSP cube covering the optical index windows.

    No real SSP grid needed — parity is a property of the linear window
    functional, independent of the SED's physical content.
    """
    wave = jnp.linspace(3600.0, 5600.0, n_wave)
    key = jax.random.PRNGKey(0)
    # positive, structured spectra (continuum + a few absorption dips)
    base = 1.0 + 0.3 * jnp.sin(wave / 40.0)
    met_age = jax.random.uniform(key, (n_met, n_age), minval=0.5, maxval=2.0)
    flux = met_age[:, :, None] * base[None, None, :]
    # add per-(met,age) tilt so windows differ across the grid
    tilt = jax.random.normal(jax.random.PRNGKey(1), (n_met, n_age))
    flux = flux * jnp.exp(tilt[:, :, None] * (wave[None, None, :] - 4600.0) / 4000.0)
    return wave, flux


def _weights(n_met, n_age, seed):
    w = jax.random.normal(jax.random.PRNGKey(seed), (n_met * n_age,))
    return jax.nn.softmax(w).reshape(n_met, n_age)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_index_lut_bit_exact_vs_measure_index_jax(seed):
    wave, flux = _mock_ssp()
    n_met, n_age, _ = flux.shape
    w = _weights(n_met, n_age, seed)
    sed = jnp.tensordot(w, flux, axes=([0, 1], [0, 1]))  # (n_wave,) dust-free SED

    defs = [STANDARD_INDICES[n] for n in _BREAK_EW]
    exact = np.asarray(jnp.stack([measure_index_jax(wave, sed, d) for d in defs]))

    pc = precompute_index_windows(wave, flux, defs)
    window_means = jnp.tensordot(w, pc.window_integrals, axes=([0, 1], [0, 1])) / pc.window_norms
    lut = np.asarray(measure_indices_from_windows(window_means, pc))

    # bit-exact up to float64 round-off (window mean commutes with the weight sum)
    np.testing.assert_allclose(lut, exact, rtol=1e-9, atol=0)
    assert not pc.has_slope


def test_slope_index_flagged_and_returns_nan_sentinel():
    wave, flux = _mock_ssp()
    slope = [STANDARD_INDICES["uv_slope_beta"]]
    pc = precompute_index_windows(wave, flux, slope)
    assert pc.has_slope
    assert pc.window_integrals.shape[-1] == 0  # no break/EW windows
    out = measure_indices_from_windows(jnp.zeros(0), pc)
    assert bool(np.isnan(np.asarray(out)[0]))  # caller must use the exact path


def test_shared_windows_deduplicated():
    """Two indices sharing a continuum band integrate that window once."""
    wave, flux = _mock_ssp()
    # HdA and HgA share no window, but Dn4000/D4000 both have red/blue bands;
    # build a case with a deliberate shared band via two identical breaks.
    defs = [STANDARD_INDICES["Dn4000"], STANDARD_INDICES["Dn4000"]]
    pc = precompute_index_windows(wave, flux, defs)
    # only 2 unique windows (blue, red) despite 2 indices × 2 windows = 4 refs
    assert pc.window_integrals.shape[-1] == 2
