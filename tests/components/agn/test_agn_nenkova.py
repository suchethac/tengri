# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Nenkova+2008 CLUMPY torus component.

This torus is Prospector's only AGN SED component (FSPS CLUMPY templates,
Johnson et al. 2021). The grid is vendored into ``data/nenkova08_torus_grid.h5``
by ``scripts/build_nenkova_grid.py`` and interpolated with a pure-JAX triweight
kernel so that ``agn_tau`` is a fully differentiable, JIT/vmap-safe *fitted*
parameter.

Regression guard (``regression_bug``): the historical implementation called
``scipy.interpolate.interp1d`` and ``float(agn_tau)`` on every call, raising
``ConcretizationTypeError`` whenever ``agn_tau`` was traced — i.e. it could
never be sampled/optimized by MAP, NUTS, or VI. These tests freeze the fix.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._jit_parity import assert_jit_matches_eager

jax.config.update("jax_enable_x64", True)

_GRID_PATH = Path(__file__).resolve().parents[3] / "data" / "nenkova08_torus_grid.h5"
_has_grid = _GRID_PATH.is_file()

# The grid is committed (~26 KB, whitelisted in .gitignore), so these run on CI.
pytestmark = pytest.mark.skipif(
    not _has_grid,
    reason=(
        "Nenkova+2008 grid not built. Run: python scripts/build_nenkova_grid.py "
        '--input "$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat"'
    ),
)


@pytest.fixture(scope="module")
def wavelength() -> jnp.ndarray:
    return jnp.geomspace(1e3, 1e6, 512)


@pytest.fixture(scope="module")
def torus_fn():
    from tengri.components.agn.torus import create_nenkova_from_grid

    return create_nenkova_from_grid(str(_GRID_PATH))


@pytest.mark.contract
def test_grid_metadata_sensible() -> None:
    import h5py

    with h5py.File(_GRID_PATH, "r") as f:
        g = f["nenkova"]
        tau = g["tau_axis"][:]
        wave = g["wavelength"][:]
        tpl = g["template"][:]

    assert tau.ndim == 1 and tau.size == 9
    assert jnp.all(tau[1:] > tau[:-1]), "tau_axis must be ascending"
    # FSPS CLUMPY equatorial optical depths.
    np.testing.assert_allclose(tau, [5, 10, 20, 30, 40, 60, 80, 100, 150])
    assert wave.ndim == 1 and wave.size >= 100
    assert jnp.all(wave[1:] > wave[:-1]), "wavelength must be ascending"
    chex.assert_shape(tpl, (tau.size, wave.size))


@pytest.mark.bounds
def test_output_shape_and_finiteness(torus_fn, wavelength) -> None:
    sed = torus_fn(wavelength, agn_log_lbol=44.0, agn_tau=30.0, agn_torus_frac=0.5)
    chex.assert_equal_shape([sed, wavelength])
    chex.assert_tree_all_finite(sed)
    assert jnp.all(sed >= 0.0)
    assert float(sed.max()) > 0.0


@pytest.mark.bounds
def test_mir_peak_in_torus_range(torus_fn, wavelength) -> None:
    """A dusty torus peaks in the mid-IR (~10-40 um), not the optical or FIR.

    This is the headline sanity check vs Prospector's torus SED shape.
    """
    sed = torus_fn(wavelength, agn_log_lbol=44.0, agn_tau=30.0)
    peak_um = float(wavelength[int(jnp.argmax(sed))]) / 1e4
    assert 8.0 < peak_um < 50.0, f"torus peak at {peak_um:.1f} um is unphysical"


@pytest.mark.bounds
def test_luminosity_scales_linearly_with_lbol(torus_fn, wavelength) -> None:
    sed_lo = torus_fn(wavelength, agn_log_lbol=44.0, agn_tau=30.0)
    sed_hi = torus_fn(wavelength, agn_log_lbol=45.0, agn_tau=30.0)
    mask = sed_lo > 0.0
    ratio = jnp.where(mask, sed_hi / jnp.where(mask, sed_lo, 1.0), 10.0)
    assert jnp.allclose(ratio[mask], 10.0, rtol=1e-5)


@pytest.mark.bounds
def test_torus_frac_scales_linearly(torus_fn, wavelength) -> None:
    sed_half = torus_fn(wavelength, agn_log_lbol=44.0, agn_torus_frac=0.5)
    sed_full = torus_fn(wavelength, agn_log_lbol=44.0, agn_torus_frac=1.0)
    mask = sed_half > 0.0
    ratio = jnp.where(mask, sed_full / jnp.where(mask, sed_half, 1.0), 2.0)
    assert jnp.allclose(ratio[mask], 2.0, rtol=1e-5)


@pytest.mark.regression_paper
def test_on_grid_reproduces_clumpy_template(torus_fn) -> None:
    """At a grid-node tau, the interpolated SED matches the tabulated CLUMPY
    template shape (faithfulness to the FSPS/Prospector templates)."""
    from tengri.components.agn.torus import _load_nenkova_arrays

    raw = _load_nenkova_arrays(str(_GRID_PATH))
    i30 = int(np.argmin(np.abs(raw["tau_axis"] - 30.0)))
    ref = np.asarray(raw["template"][i30])
    wave = jnp.asarray(raw["wavelength"])
    sed = np.asarray(torus_fn(wave, agn_log_lbol=44.0, agn_tau=30.0))
    # Triweight is a smoothing kernel (not exact at nodes), so compare the
    # normalized spectral shape rather than asserting bit-exact node recovery.
    corr = float(np.corrcoef(sed, ref)[0, 1])
    assert corr > 0.999, f"on-grid shape correlation {corr:.6f} too low"


@pytest.mark.bounds
def test_tau_actually_interpolated(torus_fn, wavelength) -> None:
    """Distinct optical depths must yield distinct SEDs (tau is a real axis)."""
    sed_low = torus_fn(wavelength, agn_tau=10.0)
    sed_high = torus_fn(wavelength, agn_tau=120.0)
    diff = jnp.linalg.norm(sed_high - sed_low)
    norm = jnp.linalg.norm(sed_low) + 1e-300
    assert float(diff / norm) > 1e-3


@pytest.mark.regression_bug
def test_jit_with_traced_tau(torus_fn, wavelength) -> None:
    """``agn_tau`` must survive ``jax.jit`` as a *traced* argument.

    Regression for the scipy/``float(agn_tau)`` defect that made the torus
    unusable in inference (every backend traces fitted parameters).
    """
    sed = assert_jit_matches_eager(
        lambda lbol, tau: torus_fn(wavelength, agn_log_lbol=lbol, agn_tau=tau),
        jnp.array(44.0),
        jnp.array(30.0),
    )
    chex.assert_equal_shape([sed, wavelength])
    chex.assert_tree_all_finite(sed)


@pytest.mark.gradient
def test_grad_flows_through_tau(torus_fn, wavelength) -> None:
    """Gradient must flow through ``agn_tau`` (triweight kernel is C²)."""

    def scalar_loss(tau: float) -> float:
        sed = torus_fn(wavelength, agn_log_lbol=44.0, agn_tau=tau)
        return jnp.log1p(jnp.sum(sed))

    g = jax.grad(scalar_loss)(40.0)
    assert jnp.isfinite(g)
    assert abs(float(g)) > 0.0, "gradient w.r.t. tau is identically zero"


@pytest.mark.contract
def test_vmap_over_tau(torus_fn, wavelength) -> None:
    """``jax.vmap`` over ``agn_tau`` (e.g. posterior-sample batches) works."""
    taus = jnp.array([10.0, 30.0, 80.0, 140.0])
    batch = jax.vmap(lambda t: torus_fn(wavelength, agn_log_lbol=44.0, agn_tau=t))(taus)
    chex.assert_shape(batch, (taus.size, wavelength.size))
    chex.assert_tree_all_finite(batch)


@pytest.mark.contract
def test_public_nenkova_torus_is_jit_safe(wavelength) -> None:
    """The public ``nenkova_torus`` entry point is itself JIT/grad-safe."""
    from tengri.components.agn.torus import nenkova_torus

    sed = assert_jit_matches_eager(
        lambda tau: nenkova_torus(wavelength, agn_log_lbol=44.0, agn_tau=tau), jnp.array(30.0)
    )
    chex.assert_tree_all_finite(sed)
    g = jax.grad(lambda tau: jnp.sum(nenkova_torus(wavelength, agn_log_lbol=44.0, agn_tau=tau)))(
        50.0
    )
    assert jnp.isfinite(g)


# ──────────────────────────────────────────────────────────────────────
# Builder + forward integration: agn_tau must be a fittable parameter that
# threads end-to-end (param registry -> builder grammar -> forward allowlist
# -> torus block). Prospector treats agn_tau as a standard free parameter.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_builder_exposes_agn_tau_as_free(synthetic_ssp) -> None:
    """``SEDModel.build`` must accept ``torus={'type':'nenkova','tau':...}``
    and expose ``agn_tau`` as a free parameter (registry + grammar wiring)."""
    import tengri

    model = tengri.SEDModel.build(
        synthetic_ssp,
        sfh={"type": "const", "*": tengri.FIXED, "log_total_mass": -10.0},
        dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
        agn={
            "disc": {"type": "multicolor", "*": tengri.FIXED},
            "torus": {"type": "nenkova", "*": tengri.FIXED, "tau": tengri.Uniform(5, 150)},
            "*": tengri.FIXED,
            "log_lbol": 12.0,
            "frac": 1.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    assert "agn_tau" in model.spec.free_params


# A real (MIR-reaching) SSP is needed to see the torus bump; the synthetic
# optical-only SSP cannot. Gate on the default SSP being present.
def _has_real_ssp() -> bool:
    try:
        import tengri

        tengri.load_ssp()
        return True
    except Exception:
        return False


@pytest.mark.regression_bug
@pytest.mark.skipif(not _has_real_ssp(), reason="default SSP grid not available")
def test_agn_tau_threads_through_model_layer() -> None:
    """End-to-end: varying ``agn_tau`` through ``predict_rest_sed`` changes the
    mid-IR torus SED and yields a finite, non-zero gradient.

    Regression for the forward-layer param allowlist (sed_model / nonstell):
    ``agn_tau`` must be forwarded to the composable AGN model, not silently
    dropped. Guards all four wiring layers of the Nenkova torus fix.
    """
    import tengri

    ssp = tengri.load_ssp()
    model = tengri.SEDModel.build(
        ssp,
        sfh={"type": "const", "*": tengri.FIXED, "log_total_mass": -10.0},
        dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
        agn={
            "disc": {"type": "multicolor", "*": tengri.FIXED},
            "torus": {"type": "nenkova", "*": tengri.FIXED, "tau": tengri.Uniform(5, 150)},
            "*": tengri.FIXED,
            "log_lbol": 12.5,
            "frac": 1.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    wave = np.asarray(model.predict_rest_sed(p).wavelength)
    mir = (wave > 5e4) & (wave < 4e5)  # 5-40 um torus bump
    sed_lo = np.asarray(model.predict_rest_sed({**p, "agn_tau": jnp.float64(10.0)}).sed)
    sed_hi = np.asarray(model.predict_rest_sed({**p, "agn_tau": jnp.float64(140.0)}).sed)
    rel = np.linalg.norm(sed_hi[mir] - sed_lo[mir]) / (np.linalg.norm(sed_lo[mir]) + 1e-300)
    assert rel > 1e-2, f"agn_tau does not affect the MIR SED (rel change {rel:.2e})"

    grad = jax.grad(lambda t: jnp.sum(model.predict_rest_sed({**p, "agn_tau": t}).sed))(
        jnp.float64(40.0)
    )
    assert jnp.isfinite(grad) and float(grad) != 0.0
