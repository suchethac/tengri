# SPDX-License-Identifier: BSD-3-Clause
"""Witt & Gordon (2000) attenuation (FSPS dust_type=3) — grid, runtime, live wiring.

Covers the vendored ``data/wg00_attenuation_grid.h5`` (committed, so these run on
CI without ``$SPS_HOME``), the pure-JAX ``A(λ; τ_V)`` interpolation, and the
end-to-end builder path ``dust={'type': 'wg00', ...}`` — proving the WG00 screen
actually attenuates the forward SED (not a silent no-op), responds to ``τ_V``,
greys with geometry, is gradient-safe, and surfaces its citation.

Data source: Witt & Gordon 2000 (ApJ 528, 799), FSPS ``alldirty_{h,c}.dat``.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.contract

# Skip cleanly if the vendored grid is absent (e.g. partial checkout).
wg00 = pytest.importorskip("tengri.components.dust.wg00")
_GRID = None
try:
    _GRID = wg00._find_wg00_grid()
except FileNotFoundError:
    pytest.skip("WG00 grid not vendored; run scripts/build_wg00_grid.py", allow_module_level=True)


# ── Grid + runtime ──────────────────────────────────────────────────────────


@pytest.mark.regression_paper
def test_grid_matches_fsps_table_nodes():
    """Vendored grid reproduces published FSPS WG00 values at tabulated nodes.

    First two rows of ``alldirty_h.dat`` (homogeneous): τ_V=0.25, λ=1000 Å gives
    MW+dusty/shell/cloudy = 0.589 / 1.002 / 0.276 and SMC+shell = 1.677.
    """
    import h5py

    with h5py.File(_GRID, "r") as f:
        a = f["wg00"]["a_lambda"][:]  # (struct, dust, geom, tau, wave)
    # structure homogeneous=0; dust mw=0/smc=1; geom dusty=0/shell=1/cloudy=2; tau idx0, λ idx0
    assert a[0, 0, 0, 0, 0] == pytest.approx(0.589, abs=1e-3)
    assert a[0, 0, 1, 0, 0] == pytest.approx(1.002, abs=1e-3)
    assert a[0, 0, 2, 0, 0] == pytest.approx(0.276, abs=1e-3)
    assert a[0, 1, 1, 0, 0] == pytest.approx(1.677, abs=1e-3)


@pytest.mark.regression_paper
def test_interp_matches_table_at_interior_nodes():
    """Triweight interp reproduces the table at interior τ_V nodes (sub-percent)."""
    fn = wg00.create_wg00_from_grid(
        _GRID, dust_curve="mw", geometry="shell", structure="homogeneous"
    )
    # MW+shell @1571 Å: table τ_V=2 → 3.0430, τ_V=3 → 4.7150.
    assert float(fn(jnp.array([1571.0]), 2.0)[0]) == pytest.approx(3.043, rel=0.01)
    assert float(fn(jnp.array([1571.0]), 3.0)[0]) == pytest.approx(4.715, rel=0.01)


@pytest.mark.bounds
def test_geometry_graying_ordering():
    """WG00 graying: at fixed τ_V the shell screen attenuates most, cloudy least."""
    w, tau = jnp.array([1500.0]), 3.0
    a = {
        g: float(
            wg00.create_wg00_from_grid(
                _GRID, dust_curve="mw", geometry=g, structure="homogeneous"
            )(w, tau)[0]
        )
        for g in ("shell", "dusty", "cloudy")
    }
    assert a["shell"] > a["dusty"] > a["cloudy"], a


@pytest.mark.bounds
def test_attenuation_monotonic_in_tau():
    """A(λ; τ_V) increases monotonically with τ_V."""
    fn = wg00.create_wg00_from_grid(_GRID, dust_curve="smc", geometry="shell")
    w = jnp.array([2000.0])
    vals = [float(fn(w, t)[0]) for t in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert all(b > a for a, b in itertools.pairwise(vals)), vals


@pytest.mark.gradient
def test_curve_gradient_safe_in_tau():
    """A(λ; τ_V) is differentiable and JIT-able in τ_V (fitted-param requirement)."""
    fn = wg00.create_wg00_from_grid(_GRID)
    g = assert_grad_matches_fd(lambda t: fn(jnp.array([2000.0]), t)[0], 3.0)
    assert np.isfinite(g) and g > 0.0
    out = jax.jit(lambda t: fn(jnp.array([2000.0, 6000.0]), t))(1.5)
    assert np.all(np.isfinite(np.asarray(out)))


def test_invalid_selector_raises():
    with pytest.raises(ValueError, match="Invalid WG00 selector"):
        wg00.create_wg00_from_grid(_GRID, geometry="banana")


# ── Live forward wiring via the builder ──────────────────────────────────────


def _sed(model):
    pred = model.predict(model.spec.sample(jax.random.PRNGKey(0))).sed
    return np.asarray(pred._wave()), np.asarray(pred._sed())


@pytest.mark.limit
def test_wg00_attenuates_forward_sed(synthetic_ssp_wide):
    """dust={'type':'wg00'} actually attenuates predict() output (not a no-op)."""
    from tengri import Fixed, SEDModel

    m_lo = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        dust={"type": "wg00", "geometry": "shell", "tau_v": Fixed(0.25)},
        redshift=Fixed(0.1),
    )
    m_hi = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        dust={"type": "wg00", "geometry": "shell", "tau_v": Fixed(5.0)},
        redshift=Fixed(0.1),
    )
    w, s_lo = _sed(m_lo)
    _, s_hi = _sed(m_hi)
    i = int(np.argmin(np.abs(w - 1500.0)))
    assert s_hi[i] < 0.5 * s_lo[i], (s_lo[i], s_hi[i])


@pytest.mark.bounds
def test_wg00_geometry_changes_forward_curve(synthetic_ssp_wide):
    """Switching geometry (shell→cloudy) changes the attenuated SED (graying)."""
    from tengri import Fixed, SEDModel

    def build(geom):
        return SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            dust={"type": "wg00", "geometry": geom, "tau_v": Fixed(3.0)},
            redshift=Fixed(0.1),
        )

    w, s_shell = _sed(build("shell"))
    _, s_cloudy = _sed(build("cloudy"))
    i = int(np.argmin(np.abs(w - 1500.0)))
    assert s_shell[i] < s_cloudy[i]  # shell screen attenuates more than cloudy


@pytest.mark.contract
def test_wg00_tau_v_is_free_param_and_cited(synthetic_ssp_wide):
    """τ_V registers as a free param and the model cites Witt & Gordon (2000)."""
    import tengri
    from tengri import Fixed, SEDModel, Uniform

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        dust={
            "type": "wg00",
            "dust_curve": "smc",
            "geometry": "dusty",
            "structure": "clumpy",
            "tau_v": Uniform(0.25, 10.0),
        },
        redshift=Fixed(0.1),
    )
    assert "dust_tau_v" in model.spec.free_params
    assert model.spec.dust_model == "wg00"
    assert model.spec.dust_wg00_geometry == "dusty"
    assert "witt_gordon2000" in [c.key for c in tengri.collect_citations(model)]


@pytest.mark.gradient
def test_wg00_forward_gradient_in_tau_v(synthetic_ssp_wide):
    """predict() is differentiable w.r.t. the fitted dust_tau_v."""
    from tengri import Fixed, SEDModel, Uniform

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        dust={"type": "wg00", "tau_v": Uniform(0.25, 10.0)},
        redshift=Fixed(0.1),
    )
    base = dict(model.spec.sample(jax.random.PRNGKey(1)))

    def loss(tau):
        p = {**base, "dust_tau_v": tau}
        return jnp.sum(model.predict(p).sed._sed())

    g = assert_grad_matches_fd(loss, 3.0)
    assert np.isfinite(g) and abs(float(g)) > 0.0
