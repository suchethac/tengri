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
from tengri.components.dust.energy_balance_precompute import (
    EnergyBalanceLUT,
    lut_l_absorbed_stellar,
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
            law="calzetti",
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


# ── the interpolation only ever needs four nodes ──────────────────────────
#
# Bilinear interpolation on (tau_bc, tau_diff) touches four nodes of ``G``.
# The LUT used to reach them with a *dense* weight vector — zero everywhere
# except at two nodes per axis — and contract the whole optical-depth grid
# against it. On the shipped (15, 93, 24, 24) LUT that is 803,520 multiply-adds
# to use 5,580 of them, and it dominated the entire WavePrecomp forward pass
# (70% of it; more than the stellar LUT, the dust screen and the IGM combined).
#
# The dense form was chosen to keep XLA from constant-folding a gather. A
# ``dynamic_slice`` on traced indices cannot be constant-folded, so the four
# nodes can be sliced out directly — same arithmetic, 144x less of it.


def _dense_weight_vector_reference(lut, joint_weights, mass_scale, tau_bc, tau_diff):
    """The dense contraction the sparse bracket replaces, written out independently.

    Kept here — rather than imported — so the parity check compares two
    implementations of bilinear interpolation, not one implementation against
    itself.
    """

    def weights(grid, x):
        n = grid.shape[0]
        if n == 1:
            return jnp.ones((1,), dtype=grid.dtype)
        dx = grid[1] - grid[0]
        return jnp.clip(1.0 - jnp.abs(x - grid) / dx, 0.0, 1.0)

    g_interp = jnp.einsum(
        "maij,i,j->ma",
        lut.G,
        weights(lut.tau_bc_grid, tau_bc),
        weights(lut.tau_diff_grid, tau_diff),
    )
    return mass_scale * jnp.sum(joint_weights * (lut.B - g_interp))


def _synthetic_lut(n_met=4, n_age=7, n_bc=24, n_diff=24):
    rng = np.random.default_rng(11)
    return EnergyBalanceLUT(
        B=jnp.asarray(rng.random((n_met, n_age))),
        G=jnp.asarray(rng.random((n_met, n_age, n_bc, n_diff))),
        tau_bc_grid=jnp.linspace(0.0, 4.0, n_bc),
        tau_diff_grid=jnp.linspace(0.0, 3.0, n_diff),
    )


def _lut_args(lut):
    rng = np.random.default_rng(3)
    joint_weights = jnp.asarray(rng.random(lut.B.shape))
    return joint_weights, jnp.asarray(1.234e40)


# node, midpoint, both edges, and outside the grid on either side — the dense
# form returns all-zero weights beyond one spacing out, and the sparse bracket
# must reproduce that quirk rather than clamping to the edge value.
@pytest.mark.parametrize(
    "tau_bc,tau_diff",
    [
        (4.0 * 7 / 23, 3.0 * 3 / 23),  # exactly on a node
        (0.5 * (4.0 * 7 / 23 + 4.0 * 8 / 23), 1.0),  # between nodes
        (0.0, 0.0),  # lower edge
        (4.0, 3.0),  # upper edge
        (-0.05, 0.5),  # just below the grid
        (-9.0, 0.5),  # far below: all-zero weights
        (99.0, 0.5),  # far above: all-zero weights
        (1.8, 1.0),  # the DESI corner
    ],
)
def test_sparse_bracket_matches_the_dense_weight_vector(tau_bc, tau_diff):
    """The four-node slice is an identity on the dense contraction, not an approximation."""
    lut = _synthetic_lut()
    joint_weights, mass_scale = _lut_args(lut)
    args = (joint_weights, mass_scale, jnp.asarray(tau_bc), jnp.asarray(tau_diff))

    got = lut_l_absorbed_stellar(lut, *args)
    want = _dense_weight_vector_reference(lut, *args)

    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-13, atol=0.0)


def test_sparse_bracket_gradient_matches_the_dense_weight_vector():
    lut = _synthetic_lut()
    joint_weights, mass_scale = _lut_args(lut)

    def sparse(tb, td):
        return lut_l_absorbed_stellar(lut, joint_weights, mass_scale, tb, td)

    def dense(tb, td):
        return _dense_weight_vector_reference(lut, joint_weights, mass_scale, tb, td)

    for tau_bc, tau_diff in [(1.8, 1.0), (2.5, 1.2), (0.05, 0.05)]:
        args = (jnp.asarray(tau_bc), jnp.asarray(tau_diff))
        got = jax.grad(sparse, argnums=(0, 1))(*args)
        want = jax.grad(dense, argnums=(0, 1))(*args)
        assert np.all(np.isfinite(np.asarray(got)))
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-9, atol=0.0)


def test_interpolation_does_not_contract_the_whole_optical_depth_grid():
    """The LUT must cost four nodes, not every node.

    Compares the shipped kernel's compiled FLOPs against the dense contraction's
    on the *same* LUT, so the bound cannot go stale as grids change — and so a
    revert to the dense form fails this outright rather than merely getting
    slower.
    """
    lut = _synthetic_lut(n_bc=32, n_diff=32)  # 1024 nodes; bilinear needs 4
    joint_weights, mass_scale = _lut_args(lut)
    args = (jnp.asarray(1.8), jnp.asarray(1.0))

    def flops(fn):
        jitted = jax.jit(lambda tb, td: fn(lut, joint_weights, mass_scale, tb, td))
        return jitted.lower(*args).compile().cost_analysis()["flops"]

    sparse = flops(lut_l_absorbed_stellar)
    dense = flops(_dense_weight_vector_reference)

    # 4 nodes of 1024 -> the sparse form should be orders below the dense one.
    # A 5x margin is far looser than the ~250x actually achieved, so this fails
    # only on a genuine regression, not on XLA accounting drift.
    assert sparse < dense / 5, (
        f"the optical-depth grid is being contracted whole: {sparse:,.0f} FLOPs "
        f"vs {dense:,.0f} for the dense form (expected roughly dense * 4/1024)"
    )
