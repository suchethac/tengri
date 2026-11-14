#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""A per-age [alpha/Fe] ramp (``alpha_fe_evolving=True``) on a 4D ([alpha/Fe]-axis) SSP grid.

The ramp is linear in lookback time, ``[alpha/Fe](t) = young + (old - young) t / t_universe``
(:func:`~tengri.components.stellar.sps.dsps_wrapper.compute_alpha_fe_evolving`). Each SSP age
collapses the alpha axis at its own node pair, so the exact forward, the lookup tables and the
fast spectral-index path all see one [alpha/Fe] per age. Contract:

a. flat limit: ``old == young == v`` equals the scalar model at ``met_alpha_fe = v``;
b. composition: each per-age row of the evolving model equals row ``a`` of the scalar model at
   ``alpha(a)``, the independent check;
c. direction: with old > young an old population's Mg b-like trough is deeper than at young
   and shallower than at old;
d. gradients in the free ``met_alpha_fe_old`` and ``met_alpha_fe_young`` match finite differences;
e. lookup tables (WavePrecomp, SpectrumPrecomp, the fast spectral-index path) reproduce the exact
   evolving model;
f. grammar: ``met={'alpha_fe_evolving': True, ...}`` builds, round-trips and refuses
   misuse by name;
g. the per-age kernel :func:`collapse_alpha_axis` equals a loop over ages.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import (
    SSPData,
    alpha_bracket,
    collapse_alpha_axis,
    compute_alpha_fe_evolving,
    has_alpha_grid,
)
from tengri.config.exceptions import ParameterError
from tengri.cosmology import age_at_z
from tengri.forward.sed_model import SpectrumPrecomp, WavePrecomp
from tengri.observation.spectral_indices import STANDARD_INDICES
from tengri.parameters import Fixed
from tengri.parameters.groups import parse_groups
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
MGB = STANDARD_INDICES["Mgb"]
REDSHIFT = 0.05


@pytest.fixture(scope="module")
def ssp4d():
    """3 Z x 5 [alpha/Fe] x 20 ages x 1200 wavelengths, 500 to 10000 A.

    The alpha axis deepens a Mg b-like trough at 5175 A, so every node differs."""
    n_met, n_age, n_wave = 3, 20, 1200
    wave = jnp.linspace(500.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(7)
    base = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-3
    feature = jnp.exp(-0.5 * ((wave - 5175.0) / 40.0) ** 2)
    cubes = [base * (1.0 - 0.25 * (1.0 + a) * feature)[None, None, :] for a in ALPHA_NODES]
    flux = jnp.stack(cubes, axis=1)  # (n_met, n_alpha, n_age, n_wave)
    lgmet = jnp.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES),
    )


def _slab(ssp4d, k):
    return SSPData(
        ssp_wave=ssp4d.ssp_wave,
        ssp_flux=ssp4d.ssp_flux[:, k],
        ssp_lg_age_gyr=ssp4d.ssp_lg_age_gyr,
        ssp_lgmet=ssp4d.ssp_lgmet,
    )


@pytest.fixture(scope="module")
def observation():
    wave_obs = jnp.linspace(4000.0, 8000.0, 400)
    return tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]),
        spectroscopy=tengri.Spectroscopy(wave_obs=wave_obs, resolution=jnp.full(400, 2000.0)),
    )


@pytest.fixture(scope="module")
def point_observation():
    """Spectroscopy sampled at the model pixel centers, which SpectrumPrecomp requires."""
    wave_obs = jnp.linspace(4000.0, 8000.0, 400)
    return tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]),
        spectroscopy=tengri.Spectroscopy(
            wave_obs=wave_obs, resolution=jnp.full(400, 2000.0), resample="point"
        ),
    )


@pytest.fixture(scope="module")
def photometry_only():
    return tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]))


def _model(ssp, observation, met, *, age_gyr=3.0, redshift=None):
    redshift = Fixed(REDSHIFT) if redshift is None else redshift
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(age_gyr),
            "log_total_mass": Fixed(10.0),
        },
        met={"logzsol": Fixed(-0.3), **met},
        redshift=redshift,
        neb={"type": "ssp"},
    )


def _scalar(ssp, observation, value, **kw):
    return _model(ssp, observation, {"alpha_fe": Fixed(value)}, **kw)


def _evolving(ssp, observation, old, young, **kw):
    return _model(
        ssp,
        observation,
        {
            "alpha_fe_evolving": True,
            "alpha_fe_old": Fixed(old),
            "alpha_fe_young": Fixed(young),
        },
        **kw,
    )


def _alpha_of_age(ssp, old, young):
    t_universe = float(age_at_z(REDSHIFT))
    return np.asarray(
        compute_alpha_fe_evolving(jnp.asarray(ssp.ssp_lg_age_gyr), old, young, t_universe)
    )


# ── a. flat limit ──────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [0.0, 0.3])
def test_flat_ramp_equals_the_scalar_model(ssp4d, observation, value):
    evo = _evolving(ssp4d, observation, value, value).predict({})
    sca = _scalar(ssp4d, observation, value).predict({})
    np.testing.assert_allclose(
        np.asarray(evo.photometry()), np.asarray(sca.photometry()), rtol=1e-12
    )
    np.testing.assert_allclose(np.asarray(evo.rest_sed()), np.asarray(sca.rest_sed()), rtol=1e-12)
    for name in ("stellar_mass_surviving", "log_q_h"):
        np.testing.assert_allclose(
            np.asarray(evo.properties[name]), np.asarray(sca.properties[name]), rtol=1e-12
        )


# ── b. composition: each age is the scalar model at alpha(a) ───────────


def test_each_age_row_equals_the_scalar_model_at_that_alpha(ssp4d, observation):
    old, young = 0.5, 0.0
    row = np.asarray(
        _evolving(ssp4d, observation, old, young).predict_state({}).derived["lnu_age"]
    )
    alpha = _alpha_of_age(ssp4d, old, young)
    scalar = _model(ssp4d, observation, {"alpha_fe": Uniform(-0.2, 0.6)})
    assert row.shape == (len(alpha), 1200)
    for a, value in enumerate(alpha):
        ref = np.asarray(scalar.predict_state({"met_alpha_fe": float(value)}).derived["lnu_age"])
        np.testing.assert_allclose(row[a], ref[a], rtol=1e-10, atol=0)


# ── c. direction: an old population's trough sits between the two scalars ──


def test_old_population_trough_lies_between_young_and_old_scalars(ssp4d, observation):
    def window_flux(model):
        pred = model.predict({})
        wave = np.asarray(pred.wave_rest)
        pix = (wave > 5150.0) & (wave < 5200.0)
        return float(np.sum(np.asarray(pred.rest_sed())[pix]))

    evo = window_flux(_evolving(ssp4d, observation, 0.5, 0.0, age_gyr=10.0))
    at_young = window_flux(_scalar(ssp4d, observation, 0.0, age_gyr=10.0))
    at_old = window_flux(_scalar(ssp4d, observation, 0.5, age_gyr=10.0))
    assert at_old < evo < at_young
    # Not merely inside the bracket: the ramp must reach most of the way to the old node
    # for an old population, so the trough sits near the middle of the two scalars.
    fraction = (at_young - evo) / (at_young - at_old)
    assert 0.2 < fraction < 0.8, fraction


# ── d. gradients in the free old and young [alpha/Fe] ──────────────────


def test_gradients_in_old_and_young_match_finite_differences(ssp4d, observation):
    model = _model(
        ssp4d,
        observation,
        {
            "alpha_fe_evolving": True,
            "alpha_fe_old": Uniform(0.0, 0.6),
            "alpha_fe_young": Uniform(-0.2, 0.4),
        },
    )
    assert set(model.spec.free_params) == {"met_alpha_fe_old", "met_alpha_fe_young"}

    def loss(x, y):
        pred = model.predict_photometry({"met_alpha_fe_old": x, "met_alpha_fe_young": y})
        return jnp.sum(pred) * 1e15  # photometry is ~1e-15 in cgs; scale to O(1)

    g_old, g_young = (float(v) for v in jax.grad(loss, argnums=(0, 1))(0.5, 0.0))
    h = 1e-5
    fd_old = float((loss(0.5 + h, 0.0) - loss(0.5 - h, 0.0)) / (2 * h))
    fd_young = float((loss(0.5, h) - loss(0.5, -h)) / (2 * h))
    assert np.isfinite(g_old) and np.isfinite(g_young)
    assert g_old != 0.0 and g_young != 0.0
    np.testing.assert_allclose(g_old, fd_old, rtol=1e-4)
    np.testing.assert_allclose(g_young, fd_young, rtol=1e-4)


# ── e. lookup tables and the fast spectral-index path ──────────────────


def _lut_tolerance(control_rel: float) -> float:
    """Twice the scalar-alpha control's deviation, floored at round-off and below 1e-2."""
    return min(max(2.0 * control_rel, 1e-12), 1e-2)


def _max_rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a / b - 1.0)))


def _approx_model(ssp, observation, met, approx, *, redshift=None):
    redshift = Fixed(REDSHIFT) if redshift is None else redshift
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met={"logzsol": Fixed(-0.3), **met},
        redshift=redshift,
        neb={"type": "ssp"},
        approx=approx,
    )


def _evo_met(old, young):
    return {"alpha_fe_evolving": True, "alpha_fe_old": Fixed(old), "alpha_fe_young": Fixed(young)}


@pytest.mark.parametrize("old,young", [(0.5, 0.0), (0.3, 0.3)])
def test_wave_precomp_photometry_fixed_z(ssp4d, observation, old, young):
    exact = _approx_model(ssp4d, observation, _evo_met(old, young), None).predict_photometry({})
    lut = _approx_model(ssp4d, observation, _evo_met(old, young), WavePrecomp())
    control_exact = _approx_model(ssp4d, observation, {"alpha_fe": Fixed(0.25)}, None)
    control_lut = _approx_model(ssp4d, observation, {"alpha_fe": Fixed(0.25)}, WavePrecomp())
    ctrl = _max_rel(control_lut.predict_photometry({}), control_exact.predict_photometry({}))
    rel = _max_rel(lut.predict_photometry({}), exact)
    assert rel <= _lut_tolerance(ctrl)


@pytest.mark.parametrize("old,young", [(0.5, 0.0), (0.3, 0.3)])
def test_wave_precomp_photometry_free_z(ssp4d, observation, old, young):
    free_z = Uniform(0.05, 0.3)
    met = _evo_met(old, young)
    exact = _approx_model(ssp4d, observation, met, None, redshift=free_z)
    lut = _approx_model(ssp4d, observation, met, WavePrecomp(), redshift=free_z)
    c_exact = _approx_model(ssp4d, observation, {"alpha_fe": Fixed(0.25)}, None, redshift=free_z)
    c_lut = _approx_model(
        ssp4d, observation, {"alpha_fe": Fixed(0.25)}, WavePrecomp(), redshift=free_z
    )
    for z in (0.07, 0.2):
        ctrl = _max_rel(
            c_lut.predict_photometry({"redshift": z}), c_exact.predict_photometry({"redshift": z})
        )
        rel = _max_rel(
            lut.predict_photometry({"redshift": z}), exact.predict_photometry({"redshift": z})
        )
        assert rel <= _lut_tolerance(ctrl)


@pytest.mark.parametrize("old,young", [(0.5, 0.0), (0.3, 0.3)])
def test_spectrum_precomp_fixed_z(ssp4d, point_observation, old, young):
    met = _evo_met(old, young)
    exact = _approx_model(ssp4d, point_observation, met, None).predict_spectrum({})
    lut = _approx_model(ssp4d, point_observation, met, SpectrumPrecomp()).predict_spectrum({})
    c_exact = _approx_model(
        ssp4d, point_observation, {"alpha_fe": Fixed(0.25)}, None
    ).predict_spectrum({})
    c_lut = _approx_model(
        ssp4d, point_observation, {"alpha_fe": Fixed(0.25)}, SpectrumPrecomp()
    ).predict_spectrum({})
    assert _max_rel(lut, exact) <= _lut_tolerance(_max_rel(c_lut, c_exact))


@pytest.mark.parametrize("old,young", [(0.5, 0.0), (0.3, 0.3)])
def test_spectrum_precomp_free_z(ssp4d, point_observation, old, young):
    free_z = Uniform(0.05, 0.3)
    met = _evo_met(old, young)
    exact = _approx_model(ssp4d, point_observation, met, None, redshift=free_z)
    lut = _approx_model(ssp4d, point_observation, met, SpectrumPrecomp(), redshift=free_z)
    c_exact = _approx_model(
        ssp4d, point_observation, {"alpha_fe": Fixed(0.25)}, None, redshift=free_z
    )
    c_lut = _approx_model(
        ssp4d, point_observation, {"alpha_fe": Fixed(0.25)}, SpectrumPrecomp(), redshift=free_z
    )
    for z in (0.07, 0.2):
        ctrl = _max_rel(
            c_lut.predict_spectrum({"redshift": z}), c_exact.predict_spectrum({"redshift": z})
        )
        rel = _max_rel(
            lut.predict_spectrum({"redshift": z}), exact.predict_spectrum({"redshift": z})
        )
        assert rel <= _lut_tolerance(ctrl)


def test_fast_spectral_index_path_matches_exact(ssp4d, observation):
    model = _evolving(ssp4d, observation, 0.5, 0.0)
    fast = np.asarray(model.predict_spectral_indices({}, (MGB,), approx=True))
    exact = np.asarray(model.predict_spectral_indices({}, (MGB,), approx=False))
    np.testing.assert_allclose(fast, exact, rtol=1e-10)


def test_fast_line_fluxes_match_exact(ssp4d, observation):
    from tengri.observation.line_measurement import default_line_defs

    model = _evolving(ssp4d, observation, 0.5, 0.0)
    line_defs = default_line_defs([5175.0 * 1.05])
    fast = np.asarray(model.measure_line_fluxes({}, line_defs, approx=True))
    exact = np.asarray(model.measure_line_fluxes({}, line_defs, approx=False))
    assert np.all(np.isfinite(fast))
    np.testing.assert_allclose(fast, exact, rtol=1e-10)


def test_log_nion_matches_the_published_value(ssp4d, observation):
    model = _evolving(ssp4d, observation, 0.5, 0.0)
    chain = model._feature_chain()
    from tengri.components.stellar.component import StellarSEDComponent

    stellar = next(c for c in chain if isinstance(c, StellarSEDComponent))
    fast = float(stellar.compute_log_nion(model._evaluation_params({}, None)))
    published = float(np.asarray(model.predict_state({}).derived["log_nion"]))
    assert np.isfinite(published)
    np.testing.assert_allclose(fast, published, rtol=1e-12)


# ── f. grammar and refusals ────────────────────────────────────────────


def test_grammar_builds_round_trips_and_names_free_params(ssp4d, observation):
    model = _model(
        ssp4d,
        observation,
        {
            "alpha_fe_evolving": True,
            "alpha_fe_old": Uniform(0.0, 0.6),
            "alpha_fe_young": Fixed(0.0),
        },
    )
    assert model.spec.free_params == ["met_alpha_fe_old"]
    spec2 = parse_groups(**model.spec.to_groups())
    assert spec2.free_params == model.spec.free_params
    assert spec2.alpha_fe_evolving is True


def test_scalar_and_evolving_spellings_are_refused_together(ssp4d, observation):
    with pytest.raises(ValueError, match="exclusive"):
        _model(
            ssp4d,
            observation,
            {
                "alpha_fe_evolving": True,
                "alpha_fe": Fixed(0.1),
                "alpha_fe_old": Uniform(0.0, 0.6),
            },
        )


def test_pair_without_the_flag_is_refused(ssp4d, observation):
    with pytest.raises(ValueError, match="alpha_fe_evolving"):
        _model(ssp4d, observation, {"alpha_fe_old": Uniform(0.0, 0.6)})


def test_evolving_on_the_3d_slab_is_refused_by_name(ssp4d, observation):
    with pytest.raises(ParameterError, match="alpha_fe_evolving=True needs"):
        SEDModel.build(
            ssp_data=_slab(ssp4d, 1),
            observation=observation,
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "tau_gyr": Fixed(2.0),
                "age_gyr": Fixed(3.0),
                "log_total_mass": Fixed(10.0),
            },
            met={
                "logzsol": Fixed(-0.3),
                "alpha_fe_evolving": True,
                "alpha_fe_old": Fixed(0.5),
                "alpha_fe_young": Fixed(0.0),
            },
            redshift=Fixed(REDSHIFT),
            neb={"type": "ssp"},
        )


def test_free_old_or_young_with_photometry_only_is_refused(ssp4d, photometry_only):
    for name in ("alpha_fe_old", "alpha_fe_young"):
        with pytest.raises(ParameterError, match=f"met_{name}"):
            _model(
                ssp4d,
                photometry_only,
                {"alpha_fe_evolving": True, name: tengri.FREE},
            )


def test_free_old_with_spectroscopy_takes_the_grid_node_range(ssp4d, observation):
    model = _model(
        ssp4d,
        observation,
        {"alpha_fe_evolving": True, "alpha_fe_old": tengri.FREE, "alpha_fe_young": Fixed(0.0)},
    )
    dist = model.spec.get_distribution("met_alpha_fe_old")
    assert tuple(float(b) for b in dist.bounds) == (-0.2, 0.6)


# ── g. the per-age kernel ──────────────────────────────────────────────


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_vector_bracket_equals_a_loop_over_ages(axis):
    rng = np.random.default_rng(3)
    n_alpha, n_age = 5, 7
    shape = [2, 3, 4]
    shape.insert(axis, n_alpha)
    shape.insert(axis + 1, n_age)
    x = rng.normal(size=tuple(shape))
    nodes = jnp.asarray(ALPHA_NODES)
    alpha = jnp.asarray(rng.uniform(-0.2, 0.6, size=n_age))
    ia, fa = alpha_bracket(nodes, alpha)
    got = np.asarray(collapse_alpha_axis(jnp.asarray(x), ia, fa, axis=axis))
    ia, fa = np.asarray(ia), np.asarray(fa)
    ref = np.stack(
        [
            (1.0 - fa[a]) * np.take(np.take(x, ia[a], axis=axis), a, axis=axis)
            + fa[a] * np.take(np.take(x, ia[a] + 1, axis=axis), a, axis=axis)
            for a in range(n_age)
        ],
        axis=axis,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_constant_vector_bracket_equals_the_scalar_bracket(axis):
    rng = np.random.default_rng(5)
    shape = [2, 3, 4]
    shape.insert(axis, len(ALPHA_NODES))
    shape.insert(axis + 1, 6)
    x = jnp.asarray(rng.normal(size=tuple(shape)))
    nodes = jnp.asarray(ALPHA_NODES)
    ia_s, fa_s = alpha_bracket(nodes, jnp.asarray(0.33))
    ia_v, fa_v = alpha_bracket(nodes, jnp.full(6, 0.33))
    scalar = collapse_alpha_axis(x, ia_s, fa_s, axis=axis)
    per_age = collapse_alpha_axis(x, ia_v, fa_v, axis=axis)
    np.testing.assert_allclose(np.asarray(per_age), np.asarray(scalar), rtol=1e-14, atol=0)


def test_vector_bracket_length_mismatch_raises():
    x = jnp.zeros((5, 4, 3))
    with pytest.raises(ValueError, match="age axis"):
        collapse_alpha_axis(x, jnp.zeros(6, dtype=int), jnp.zeros(6), axis=0)


def test_grid_check_helper_is_consistent(ssp4d):
    assert has_alpha_grid(ssp4d)
