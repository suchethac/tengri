# SPDX-License-Identifier: BSD-3-Clause
"""#2683 follow-up: every surface that reads the age weights sees the same, correct ones.

The age weights feed the SED, the derived quantities (formed mass, SFR, ages),
the young/old split behind the birth-cloud screen, the SED-free fast path behind
``WavePrecomp`` and the line/Q_H routes, per-age metallicity tables, vmapped
catalog evaluation and float32 fits. ``age_kernel='dsps'`` used to differ from
the first-order integration on each of them in a different way (a table that
could not see an edge, a refused fast path with a per-age metallicity table, a
zero time-parameter gradient). All of them now go through one integration.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.stellar.component import StellarSEDComponent
from tengri.parameters.resolve import merge_fixed_params

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "sdss_u", "sdss_r", "2mass_h"]
#: Round-off bound for two evaluations of one integration in different graphs.
TIGHT = 1e-9

PSB = {
    "type": "psb_flex",
    "ratio_young": Fixed(0.6),
    "ratio_flex_0": Fixed(0.5),
    "ratio_flex_1": Fixed(-0.4),
    "ratio_flex_2": Fixed(0.3),
    "ratio_flex_3": Fixed(-0.5),
    "ratio_old_0": Fixed(0.3),
    "ratio_old_1": Fixed(-0.2),
    "ratio_old_2": Fixed(0.2),
}


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def obs():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _model(ssp, obs, kernel, sfh, *, z=0.3, **groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            neb=groups.pop("neb", {"type": "none"}),
            redshift=Fixed(z),
            sfh={**sfh, "all_params": Fixed(DEFAULT), "age_kernel": kernel},
            **groups,
        )


def _stellar(model):
    return next(c for c in model._build_component_chain() if isinstance(c, StellarSEDComponent))


def _full_params(model):
    return merge_fixed_params(model.spec, dict(model.spec.sample(jax.random.PRNGKey(0))))


@pytest.mark.parametrize("sfh", [PSB, {"type": "continuity"}, {"type": "periodic"}], ids=str)
def test_a_state_and_derived_quantities_are_kernel_independent(ssp, obs, sfh):
    """Every published float array (SED, joint weights, derived SFR, masses, ages) agrees."""
    a = _model(ssp, obs, "cic", sfh).predict_state({})
    b = _model(ssp, obs, "dsps", sfh).predict_state({})
    worst, worst_key = 0.0, None
    n_checked = 0
    for key in a.derived:
        x, y = np.asarray(a.derived[key]), np.asarray(b.derived[key])
        if x.dtype.kind != "f" or x.shape != y.shape or x.size == 0:
            continue
        fin = np.isfinite(x) & np.isfinite(y)
        scale = max(float(np.max(np.abs(x[fin]))) if fin.any() else 0.0, 1e-300)
        err = float(np.max(np.abs(x[fin] - y[fin]))) / scale if fin.any() else 0.0
        n_checked += 1
        if err > worst:
            worst, worst_key = err, key
    assert n_checked > 20
    assert worst <= TIGHT, f"derived[{worst_key!r}] differs by {worst:.2e} (relative to its max)"
    sed = float(np.max(np.abs(np.asarray(a.sed_intrinsic) / np.asarray(b.sed_intrinsic) - 1.0)))
    assert sed <= TIGHT, f"sed_intrinsic differs by {sed:.2e}"


def test_b_young_old_split_is_kernel_independent(ssp, obs):
    """The birth-cloud screen reads the younger-than-boundary mass fraction per node."""
    groups = {
        "dust_attenuation": {
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.8,
            "tau_diff": 0.3,
        }
    }
    models = {k: _model(ssp, obs, k, PSB, **groups) for k in ("cic", "dsps")}
    frac = {
        k: np.asarray(m.predict_state({}).derived["age_boundary_younger_fraction"])
        for k, m in models.items()
    }
    assert frac["cic"].size > 0 and float(frac["cic"].max()) > 0.0
    np.testing.assert_allclose(frac["dsps"], frac["cic"], rtol=0.0, atol=TIGHT)
    phot = {k: np.asarray(m.predict_photometry({})) for k, m in models.items()}
    np.testing.assert_allclose(phot["dsps"], phot["cic"], rtol=TIGHT, atol=0.0)


_T_GYR = np.concatenate([np.array([0.0]), np.linspace(1.0, 13.0, 39)])


def _table_model(ssp, obs, kernel):
    """A tabulated SFH with a tabulated Z(t): the per-age metallicity joint-weight route."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "table", "age_kernel": kernel},
            met={"type": "table"},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
                "tau_bc": 0.5,
                "tau_diff": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )


def _table_params(model):
    params = _full_params(model)
    shape = np.ones(_T_GYR.shape[0])
    shape[0] = 0.0
    params["dust_tau_diff"] = jnp.asarray(0.3)
    params["sfh_t_gyr"] = jnp.asarray(_T_GYR)
    params["sfh_sfr"] = jnp.asarray(3.0 * shape * (1.0 + 0.5 * np.sin(_T_GYR)))
    params["met_history"] = jnp.asarray(np.linspace(-1.2, -0.2, _T_GYR.shape[0]))
    return params


def test_c_fast_path_serves_dsps_with_a_per_age_metallicity_table(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """compute_joint_weights raised NotImplementedError for dsps + a Z(t) table.

    Now it serves it: finite, normalized, equal to the first-order weights and to
    the exact forward's published weights.
    """
    dsps = _table_model(synthetic_ssp_wide, synthetic_tophat_obs, "dsps")
    cic = _table_model(synthetic_ssp_wide, synthetic_tophat_obs, "cic")
    jw, total_mass, _ = _stellar(dsps).compute_joint_weights(_table_params(dsps))
    jw_cic, _, _ = _stellar(cic).compute_joint_weights(_table_params(cic))
    assert bool(jnp.all(jnp.isfinite(jw))) and float(total_mass) > 0.0
    assert abs(float(jw.sum()) - 1.0) <= 1e-12
    np.testing.assert_allclose(np.asarray(jw), np.asarray(jw_cic), rtol=TIGHT, atol=1e-15)
    free = {
        k: v
        for k, v in _table_params(dsps).items()
        if k in set(dsps.spec.free_params) | {"sfh_t_gyr", "sfh_sfr", "met_history"}
    }
    exact = np.asarray(dsps.predict_state(free).derived["joint_weights"])
    np.testing.assert_allclose(np.asarray(jw), exact, rtol=1e-11, atol=1e-15)


def test_d_fast_path_matches_exact_forward_for_dsps(ssp, obs):
    """Delta metallicity: the SED-free weights are the weights apply publishes."""
    model = _model(ssp, obs, "dsps", {"type": "dpl"})
    jw, total_mass, _ = _stellar(model).compute_joint_weights(_full_params(model))
    state = model.predict_state({})
    np.testing.assert_allclose(
        np.asarray(jw), np.asarray(state.derived["joint_weights"]), rtol=1e-12, atol=1e-15
    )
    assert abs(float(total_mass) / 1e10 - 1.0) <= 1e-9


#: Photometry is ~1e-12 in cgs; a float32 reverse pass underflows on that scale for
#: either kernel (it is the fit surface's job to scale the loss), so the gradient
#: is taken of the flux in units of 1e-12.
FLUX_SCALE = 1e12


def _tflex_flux(ssp, obs, kernel):
    model = _model(ssp, obs, kernel, {**PSB, "tflex_gyr": Uniform(0.5, 5.0)}, z=0.0)
    return lambda x: FLUX_SCALE * model.predict_photometry({"sfh_psb_flex_tflex_gyr": x})[2]


def test_e_float32_gradient_is_finite_and_nonzero(ssp, obs):
    """float32 fit surface: d(r flux)/d(tflex) for dsps is finite, non-zero, and the cic one."""
    flux = {k: _tflex_flux(ssp, obs, k) for k in ("cic", "dsps")}
    with jax.enable_x64(False):
        x = jnp.asarray(2.3, dtype=jnp.float32)
        g = {k: np.asarray(jax.jit(jax.grad(f))(x)) for k, f in flux.items()}
        v = {k: np.asarray(jax.jit(f)(x)) for k, f in flux.items()}
    assert g["dsps"].dtype == np.float32
    assert np.isfinite(g["dsps"]) and float(abs(g["dsps"])) > 0.0
    assert np.isfinite(v["dsps"]) and float(v["dsps"]) > 0.0
    np.testing.assert_allclose(g["dsps"], g["cic"], rtol=1e-4)
    np.testing.assert_allclose(v["dsps"], v["cic"], rtol=1e-5)


def test_f_vmapped_population_matches_the_scalar_evaluations(ssp, obs):
    """Catalog-style vmap over tflex: finite, smooth, equal to the loop (and non-constant)."""
    f = _tflex_flux(ssp, obs, "dsps")
    xs = jnp.linspace(2.0, 3.0, 9)
    batched = np.asarray(jax.jit(jax.vmap(f))(xs))
    looped = np.array([float(f(x)) for x in xs])
    np.testing.assert_allclose(batched, looped, rtol=1e-10)
    grads = np.asarray(jax.jit(jax.vmap(jax.grad(f)))(xs))
    assert np.all(np.isfinite(grads)) and np.all(np.abs(grads) > 0.0)
    assert float(np.ptp(batched)) > 0.0
