# SPDX-License-Identifier: BSD-3-Clause
"""Regression: a custom bin ladder gets its own edges as integrand knots (#3038).

#765 injects a binned SFH's exact bin edges as knots into the dense lookback
integrand, so each SFR step is represented sharply. For ``continuity`` and
``dirichlet`` (and the registry entries that name them), ``sfh_bin_edges_yr``
returned ``DEFAULT_BIN_EDGES_GYR`` whatever ladder the model was built with.
A model with its own ``bin_edges_gyr``, including the redshift-built ladder
``SEDModel.build`` gives every fixed-redshift model (#2521), got knots at
0.03 ... 13.7 Gyr while its own steps fell between grid nodes and were
smeared. Measured on ``make_agebins_from_zred(0.05)`` (FSPS grid, ratios drawn
from N(0, 0.8)): the intrinsic SED was 1.6e-3 (median, 1000-3500 A) off a
64x-denser reference with the same knots; with the model's own knots it is
4e-6 off.

Issue: https://github.com/suchethac/tengri/issues/3038
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri.components.stellar.component as stellar_component
import tengri.components.stellar.sfh.nonparametric as nonparametric
from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.component import _cic_integrand
from tengri.components.stellar.sfh.nonparametric import (
    DEFAULT_BIN_EDGES_GYR,
    continuity,
    dirichlet,
    make_agebins_from_zred,
    sfh_bin_edges_yr,
)

pytestmark = pytest.mark.regression_bug

Z_OBS = 0.05
#: The #765 accuracy target: the operational integrand against a converged one.
REF_RTOL = 1e-4
RATIOS = np.random.default_rng(0).normal(0.0, 0.8, 6)


def _default_edges_yr(fn, sfh_kwargs):
    """The pre-#3038 ``sfh_bin_edges_yr``: the default ladder for continuity/dirichlet."""
    if fn is continuity or fn is dirichlet:
        return jnp.asarray(np.asarray(DEFAULT_BIN_EDGES_GYR)) * 1e9
    return _ORIGINAL(fn, sfh_kwargs)


_ORIGINAL = sfh_bin_edges_yr


@pytest.mark.parametrize("fn", [continuity, dirichlet])
def test_edges_follow_the_model_ladder(fn):
    """The knots are the ladder the shape function is called with, else the default."""
    edges = np.asarray(make_agebins_from_zred(Z_OBS))
    got = np.asarray(sfh_bin_edges_yr(fn, {"bin_edges_gyr": jnp.asarray(edges)}))
    np.testing.assert_array_equal(got, edges * 1e9)
    np.testing.assert_array_equal(
        np.asarray(sfh_bin_edges_yr(fn, {})), np.asarray(DEFAULT_BIN_EDGES_GYR) * 1e9
    )


def test_traced_ladder_under_vmap_matches_each_concrete_ladder(ssp_data_fsps):
    """Per-galaxy traced ladders (the batched stack) give each row its own knots.

    Only the ladder's length is static; the values may be traced. Each row of a
    vmapped integrand equals the integrand built from that ladder as a concrete
    array.
    """
    ssp_ages_yr = 10.0 ** jnp.asarray(ssp_data_fsps.ssp_lg_age_gyr) * 1e9
    ladders = jnp.asarray([make_agebins_from_zred(z) for z in (0.02, 0.05, 0.1)])
    kw = {"log_total_mass": 9.0, **{f"ratio_{i}": float(r) for i, r in enumerate(RATIOS)}}

    def integrand(edges, t_obs_gyr):
        sfh_kwargs = {**kw, "bin_edges_gyr": edges}
        return _cic_integrand(ssp_ages_yr, continuity, sfh_kwargs, continuity, None, t_obs_gyr)

    t_obs = ladders[:, -1]
    age_b, sfr_b = jax.jit(jax.vmap(integrand))(ladders, t_obs)
    for i in range(ladders.shape[0]):
        age_i, sfr_i = integrand(ladders[i], t_obs[i])
        np.testing.assert_allclose(np.asarray(age_b[i]), np.asarray(age_i), rtol=1e-14)
        np.testing.assert_allclose(np.asarray(sfr_b[i]), np.asarray(sfr_i), rtol=1e-12)
        for e in np.asarray(ladders[i, 1:-1]) * 1e9:
            assert np.min(np.abs(np.asarray(age_b[i]) - e)) < e * 1e-5, (i, e)


def _sed(ssp, edges):
    sfh = {
        "type": "continuity",
        "bin_edges_gyr": edges,
        "log_total_mass": Fixed(9.0),
        **{f"ratio_{i}": Fixed(float(r)) for i, r in enumerate(RATIOS)},
        "all_params": Fixed(DEFAULT),
    }
    model = SEDModel.build(
        ssp_data=ssp,
        sfh=sfh,
        met={"logzsol": Fixed(-0.5), "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(Z_OBS),
    )
    jax.clear_caches()  # the integrand and its knots are baked into compiled kernels
    return np.asarray(model.predict_state({}).sed_intrinsic)


def test_custom_ladder_matches_a_dense_reference(ssp_data_fsps, monkeypatch):
    """Own-edge knots reach the converged SED; default-ladder knots do not.

    The reference is the same model on a 4x denser integrand (64 nodes per SSP
    interval instead of 16) with the model's own knots. With the default-ladder
    knots the moved steps are smeared and the SED is 1.6e-3 off.
    """
    edges = np.asarray(make_agebins_from_zred(Z_OBS))
    try:
        new = _sed(ssp_data_fsps, edges)
        with monkeypatch.context() as m:
            m.setattr(nonparametric, "sfh_bin_edges_yr", _default_edges_yr)
            old = _sed(ssp_data_fsps, edges)
        with monkeypatch.context() as m:
            m.setattr(stellar_component, "INTEGRAND_FACTOR_PARAMETRIC", 64)
            ref = _sed(ssp_data_fsps, edges)
    finally:
        jax.clear_caches()
    wave = np.asarray(ssp_data_fsps.ssp_wave)
    sel = (wave > 1000.0) & (wave < 10000.0) & (ref > 0.0)
    err_new = np.max(np.abs(new[sel] / ref[sel] - 1.0))
    err_old = np.median(np.abs(old[sel] / ref[sel] - 1.0))
    assert err_new < REF_RTOL, f"own-ladder knots are {err_new:.2e} off the dense reference"
    assert err_old > 5.0 * REF_RTOL, f"the default-ladder arm no longer fails ({err_old:.2e})"


def test_default_ladder_is_bit_identical(ssp_data_fsps, monkeypatch):
    """A model on ``DEFAULT_BIN_EDGES_GYR`` gets the same knots as before #3038."""
    edges = np.asarray(DEFAULT_BIN_EDGES_GYR)
    try:
        new = _sed(ssp_data_fsps, edges)
        with monkeypatch.context() as m:
            m.setattr(nonparametric, "sfh_bin_edges_yr", _default_edges_yr)
            old = _sed(ssp_data_fsps, edges)
    finally:
        jax.clear_caches()
    assert np.all(np.isfinite(new)) and np.any(new > 0.0)
    np.testing.assert_array_equal(new, old)
