# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the WavePrecomp dust energy-balance LUT matches the exact path.

Enabling dust IR re-emission under ``approx=WavePrecomp()`` used to drag the
full-wavelength stellar cube back into every evaluation (it feeds ``L_ir``),
costing ~40x. The model now precomputes the bolometric absorbed luminosity on a
``(tau_bc, tau_diff)`` grid (``energy_balance_precompute``) and contracts it with
the runtime DSPS weights — no per-call spectral cube. The spectral integral is
exact at the grid nodes, so the only approximation is the smooth interpolation in
optical depth; the projected photometry must therefore track the exact full-wave
energy balance to well within the WavePrecomp approximation budget.

``WavePrecomp(fast_dust_emission=True)`` additionally samples the IR template at
the filter effective wavelength instead of integrating it through each band — a
deliberately coarser, faster projection. It must still run and stay finite.

CI-runnable on the synthetic wide SSP (no ``data/`` grids needed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
)
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _obs() -> Observation:
    # Optical bands probe the absorbed (energy-balance) light; the far-IR band
    # (100 um) is where the re-emitted dust luminosity lands, so it is the band
    # that actually exercises L_ir.
    centers = (3500.0, 4800.0, 6200.0, 9000.0, 1.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(ssp, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=_obs(),
        approx=approx,
        sfh=builders.sfh.tsnorm(defaults=FIXED),
        dust=builders.dust.two_component(
            defaults=FIXED,
            law_bc="calzetti",
            tau_bc=Uniform(0.0, 1.0),
            emission=builders.dust.emission.modified_blackbody(defaults=FIXED),
        ),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
    )


def test_eb_lut_engages_and_matches_exact(synthetic_ssp_wide):
    """WavePrecomp+IR-emission engages the LUT and tracks the exact energy balance."""
    ssp = synthetic_ssp_wide
    m_lut = _build(ssp, WavePrecomp())
    m_exact = _build(ssp, None)

    # The LUT is precomputed at construction for this (fixed-shape) config.
    assert getattr(m_lut, "_energy_balance_lut_cache", None) is not None

    f_lut = jax.jit(m_lut.predict_photometry)
    f_exact = jax.jit(m_exact.predict_photometry)
    base = {**m_lut.spec.get_fixed_values(), **m_lut.spec.sample(jax.random.PRNGKey(0))}

    worst = 0.0
    for tau in np.linspace(0.0, 1.0, 9):
        p = dict(base)
        p["dust_tau_bc"] = jnp.asarray(float(tau))
        a = np.asarray(f_lut(p))
        b = np.asarray(f_exact(p))
        worst = max(worst, float(np.abs(a - b).max() / np.abs(b).max()))
    # WavePrecomp's own dust-attenuation approximation already sits near ~1%; the
    # energy-balance LUT must not blow past that budget across the tau range.
    assert worst < 0.02, f"EB-LUT vs exact drifted {worst:.3%} (> 2%)"


def test_fast_dust_emission_runs_and_is_finite(synthetic_ssp_wide):
    """The coarser ``fast_dust_emission`` projection runs and stays finite."""
    ssp = synthetic_ssp_wide
    m_fast = _build(ssp, WavePrecomp(fast_dust_emission=True))
    f = jax.jit(m_fast.predict_photometry)
    p = {**m_fast.spec.get_fixed_values(), **m_fast.spec.sample(jax.random.PRNGKey(1))}
    out = np.asarray(f(p))
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)


def test_eb_lut_gradient_is_finite(synthetic_ssp_wide):
    """L_ir flows through the LUT differentiably (the fit path needs gradients)."""
    ssp = synthetic_ssp_wide
    m = _build(ssp, WavePrecomp())
    p = {**m.spec.get_fixed_values(), **m.spec.sample(jax.random.PRNGKey(2))}
    g = jax.jit(jax.grad(lambda q: jnp.sum(m.predict_photometry(q))))(p)
    assert np.all(np.isfinite(np.asarray(g["dust_tau_bc"])))
