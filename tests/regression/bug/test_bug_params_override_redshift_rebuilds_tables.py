# SPDX-License-Identifier: BSD-3-Clause
"""params_override redshift on a non-catalog model must equal a build at that redshift.

``Fitter(model, ..., params_override={"redshift": z2})`` on a model built with
``redshift=Fixed(z1)`` and no ``catalog_z_range`` used to merge ``z2`` into the
loss's fixed values only. Every table the model captured at construction (the
fixed-z stellar photometry LUT, IGM band factors, nebular grid reference, the
dust-IR band response, the energy-balance LUT, the radio/X-ray term responses,
the luminosity distance) stayed at ``z1``, so the compiled program read z1 tables
at z2. Measured: a 45.8% loss error on a ``WavePrecomp`` model moved from z=0.05
to 1.0. The exact path (``approx=None``) had no tables and was already right.

The fix evaluates a model *built* at the override redshift, so the override is
exactly a direct build. Every comparison here is against that direct build,
paired with a vacuity check that the un-rebuilt z1 model genuinely disagrees.
"""

from __future__ import annotations

import warnings
from unittest import mock

import jax
import numpy as np
import pytest

from tengri import DEFAULT, FREE, FeaturePrecomp, Fitter, Fixed, Observation, SEDModel, Uniform
from tengri.forward import convenience
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population import Population
from tengri.forward.sed_model import WavePrecomp
from tengri.inference.context import InferenceContext
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug

OPT_BANDS = ["des_g", "des_r", "des_z", "wise_w1", "wise_w2"]
IR_BANDS = ["des_r", "des_z", "wise_w1", "wise_w2", "wise_w3", "wise_w4"]

RTOL = 1e-6
#: Two genuinely different redshifts must disagree by far more than RTOL.
VACUITY_FLOOR = 1e-3
Z_BASE = 0.05
Z_OVERRIDE = 1.0

#: Physical evaluation point, valid at every redshift used here (the universe is
#: still older than sfh_dpl_age_gyr at z=2.5).
PARAMS = {
    "sfh_dpl_log_total_mass": 10.0,
    "sfh_dpl_alpha": 2.0,
    "sfh_dpl_tau_gyr": 3.0,
    "sfh_dpl_age_gyr": 0.5,
    "sfh_dpl_beta": 1.0,
    "neb_logU": -2.5,
    "neb_logZ_gas": 0.0,
    "dust_tau_bc": 0.5,
    "dust_tau_diff": 0.2,
}

_SFH = {"type": "dpl", "all_params": FREE}
_TWO_COMPONENT = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_bc": Uniform(0.0, 4.0),
    "tau_diff": Uniform(0.0, 3.0),
}
_CUE = {
    "type": "cue",
    "all_params": Fixed(DEFAULT),
    "logU": Uniform(-4.0, -1.0),
    "logZ_gas": Uniform(-1.5, 0.3),
}


def _build(ssp, obs, z, approx, **blocks):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(z),
            sfh=_SFH,
            met={"logzsol": Fixed(0.0)},
            approx=approx,
            **blocks,
        )


def _phot_only(ssp, obs, z, approx):
    return _build(
        ssp,
        obs,
        z,
        approx,
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


def _cue_dust_free(ssp, obs, z, approx):
    return _build(
        ssp,
        obs,
        z,
        approx,
        neb=_CUE,
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


def _dust_emission(ssp, obs, z, approx):
    return _build(
        ssp,
        obs,
        z,
        approx,
        dust_attenuation=_TWO_COMPONENT,
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
    )


def _inner(model):
    """The SEDModel behind a fit model (a bare SEDModel or a ForwardModel)."""
    return model.populations[0].sed if hasattr(model, "populations") else model


def _mock(model, params):
    phot = np.asarray(model.predict_photometry(params))
    return phot, 0.05 * np.abs(phot) + 1e-31


def _fitter(model, data, noise, approx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # profile_mass=False: analytic mass marginalization fits out the overall
        # amplitude, and the synthetic SSP's near power-law shape makes what is
        # left almost redshift-blind, which would make every loss below vacuous.
        return Fitter(
            model,
            data=data,
            noise=noise,
            data_type="photometry",
            approx=approx,
            profile_mass=False,
            **kw,
        )


def _loss(fitter, key=11):
    ctx = InferenceContext.from_target(fitter)
    p_u = ctx.initial_params(jax.random.PRNGKey(key))
    return float(ctx.neg_log_posterior_fn(p_u, ctx.data_args))


def _params_for(model):
    free = set(model.spec.free_params)
    return {k: v for k, v in PARAMS.items() if k in free}


def _override_vs_direct(builder, ssp, bands, approx, z2=Z_OVERRIDE):
    obs = Observation(photometry=Photometry.from_names(bands))
    base = builder(ssp, obs, Z_BASE, approx)
    direct = builder(ssp, obs, z2, approx)
    data, noise = _mock(direct, _params_for(direct))
    f_over = _fitter(base, data, noise, approx, params_override={"redshift": z2})
    f_direct = _fitter(direct, data, noise, approx)
    f_base = _fitter(base, data, noise, approx)
    return f_over, f_direct, f_base, base, direct


def _assert_matches_direct(builder, ssp, bands, approx, z2=Z_OVERRIDE):
    f_over, f_direct, f_base, _, _ = _override_vs_direct(builder, ssp, bands, approx, z2)
    loss_o, loss_d, loss_b = _loss(f_over), _loss(f_direct), _loss(f_base)
    np.testing.assert_allclose(loss_o, loss_d, rtol=RTOL)
    assert abs(loss_b - loss_d) > VACUITY_FLOOR * abs(loss_d), (
        "vacuity: the un-overridden z1 model has the same loss as the z2 build, "
        "so the equality above cannot observe a stale table"
    )
    params = _params_for(f_direct.model)
    phot_o = np.asarray(f_over.model.predict_photometry(params))
    phot_d = np.asarray(f_direct.model.predict_photometry(params))
    np.testing.assert_allclose(phot_o, phot_d, rtol=RTOL)
    phot_b = np.asarray(f_base.model.predict_photometry(params))
    assert np.max(np.abs(phot_b - phot_d) / np.abs(phot_d)) > VACUITY_FLOOR
    return f_over, f_direct


def test_photometry_override_matches_direct_build(synthetic_ssp_wide):
    """T1: the 45.8% defect. Fixed(0.05)+WavePrecomp overridden to 1.0 == Fixed(1.0)."""
    _assert_matches_direct(_phot_only, synthetic_ssp_wide, OPT_BANDS, WavePrecomp())


def test_cue_nebular_override_matches_direct_build(synthetic_ssp_wide):
    """T2: a dust-free Cue model, so the nebular grid's reference redshift matters."""
    approx = (WavePrecomp(), FeaturePrecomp(n_grid=4))
    f_over, f_direct = _assert_matches_direct(
        _cue_dust_free, synthetic_ssp_wide, OPT_BANDS, approx
    )
    grid_o = _inner(f_over.model)._nebular_grid_table
    grid_d = _inner(f_direct.model)._nebular_grid_table
    assert grid_o is not None, "the model has no nebular grid, so T2 tests nothing about it"
    np.testing.assert_allclose(np.asarray(grid_o.wavelengths), np.asarray(grid_d.wavelengths))


def test_dust_emission_band_response_override_matches_direct_build(synthetic_ssp_wide):
    """T3: the dust-IR band response R and energy-balance LUT are built at the override z."""
    f_over, _ = _assert_matches_direct(_dust_emission, synthetic_ssp_wide, IR_BANDS, WavePrecomp())
    sed = _inner(f_over.model)
    assert sed._dust_band_response_cache is not None, (
        "no band response was built, so T3 does not exercise R"
    )


def test_exact_path_override_matches_direct_build(synthetic_ssp_wide):
    """T4: the exact path has no tables and matched before; it must still match."""
    f_over, f_direct, f_base, _, _ = _override_vs_direct(
        _phot_only, synthetic_ssp_wide, OPT_BANDS, None
    )
    loss_d = _loss(f_direct)
    np.testing.assert_allclose(_loss(f_over), loss_d, rtol=RTOL)
    assert abs(_loss(f_base) - loss_d) > VACUITY_FLOOR * abs(loss_d)


@pytest.mark.parametrize("z2", [0.3, 1.0, 2.5])
def test_override_matches_direct_build_across_redshifts(synthetic_ssp_wide, z2):
    """T5a: the equivalence holds across the redshift range."""
    _assert_matches_direct(_phot_only, synthetic_ssp_wide, OPT_BANDS, WavePrecomp(), z2)


def test_two_overrides_on_one_base_model_do_not_share_a_program(synthetic_ssp_wide):
    """T5b: overrides on one base model each get their own program (distinct z, distinct loss)."""
    obs = Observation(photometry=Photometry.from_names(OPT_BANDS))
    approx = WavePrecomp()
    base = _phot_only(synthetic_ssp_wide, obs, Z_BASE, approx)
    direct = {z: _phot_only(synthetic_ssp_wide, obs, z, approx) for z in (0.3, 2.5)}
    data, noise = _mock(direct[2.5], _params_for(direct[2.5]))

    fitters = {
        z: _fitter(base, data, noise, approx, params_override={"redshift": z}) for z in (0.3, 2.5)
    }
    # Interleave: construct both, then evaluate, so a shared cache entry would bite.
    losses = {z: _loss(fitters[z]) for z in (0.3, 2.5)}
    expected = {z: _loss(_fitter(direct[z], data, noise, approx)) for z in (0.3, 2.5)}
    for z in (0.3, 2.5):
        np.testing.assert_allclose(losses[z], expected[z], rtol=RTOL)
    assert abs(losses[0.3] - losses[2.5]) > VACUITY_FLOOR * abs(losses[2.5])
    assert fitters[0.3]._engine_cache_key() != fitters[2.5]._engine_cache_key()
    assert fitters[0.3].model is not fitters[2.5].model


def test_the_callers_model_is_not_mutated(synthetic_ssp_wide):
    """The override rebuilds; it never rewrites the model the caller passed in."""
    obs = Observation(photometry=Photometry.from_names(OPT_BANDS))
    base = _phot_only(synthetic_ssp_wide, obs, Z_BASE, WavePrecomp())
    data, noise = _mock(base, _params_for(base))
    f = _fitter(base, data, noise, WavePrecomp(), params_override={"redshift": Z_OVERRIDE})
    assert base.spec.get_fixed_values()["redshift"] == Z_BASE
    assert f.spec.get_fixed_values()["redshift"] == Z_OVERRIDE
    assert f._fixed_values["redshift"] == Z_OVERRIDE


# ── Catalog rows: the per-galaxy redshift rides params_override ────────────
#
# A catalog fit whose model has no ``catalog_z_range`` fits each row through
# ``Fitter(..., params_override={"redshift": z_i})`` (catalog_fitter.py). The
# fix above IS the catalog fix for that route. Only the mass is free, so a row
# that evaluates the right tables recovers the truth it was mocked from, and a
# row evaluating another redshift's tables is off by the flux ratio (~2 dex).

CATALOG_Z = (0.3, 0.9, 1.6)
TRUTH_LOGM = 9.5


def _mass_only(ssp, obs, z, approx, *, cue=False, dust_emission=False):
    blocks = {
        "dust_attenuation": {"type": "none"},
        "dust_emission": {"type": "none"},
    }
    if cue:
        blocks["neb"] = {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Fixed(-2.5),
            "logZ_gas": Fixed(0.0),
        }
    if dust_emission:
        blocks["dust_attenuation"] = {
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.2),
        }
        blocks["dust_emission"] = {"type": "dale2014", "all_params": Fixed(DEFAULT)}
    return _build_mass_only(ssp, obs, z, approx, blocks)


def _mass_only_builder(ssp, obs, z, approx):
    return _mass_only(ssp, obs, z, approx)


def _build_mass_only(ssp, obs, z, approx, blocks):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(z),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(8.0, 12.0),
            },
            met={"logzsol": Fixed(0.0)},
            approx=approx,
            **blocks,
        )


def _catalog_rows(ssp, obs, approx, zs, **variant):
    """Rows mocked from single-galaxy Fixed(z_i) builds at the same truth."""
    nb = len(obs.photometry.filters)
    rows = []
    for z in zs:
        direct = _mass_only(ssp, obs, z, approx, **variant)
        phot = np.asarray(direct.predict_photometry({"sfh_dpl_log_total_mass": TRUTH_LOGM}))
        row = {"z": float(z)}
        row.update({f"f{j}": float(phot[j]) for j in range(nb)})
        row.update({f"e{j}": float(0.05 * phot[j]) for j in range(nb)})
        rows.append(row)
    return rows, nb


def _fit_catalog(base, rows, nb):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return convenience.fit_batch(
            base,
            rows,
            flux_cols=[f"f{j}" for j in range(nb)],
            err_cols=[f"e{j}" for j in range(nb)],
            redshift_col="z",
            method="map",
            n_steps=200,
            verbose=False,
        )


_CATALOG_VARIANTS = {
    "wave": (OPT_BANDS, WavePrecomp(), {}),
    "cue_feature": (OPT_BANDS, (WavePrecomp(), FeaturePrecomp(n_grid=4)), {"cue": True}),
    "dust_emission": (IR_BANDS, WavePrecomp(), {"dust_emission": True}),
}


@pytest.mark.parametrize("variant", sorted(_CATALOG_VARIANTS))
def test_catalog_rows_without_catalog_z_range_recover_each_galaxys_truth(
    synthetic_ssp_wide, variant
):
    """Each catalog row is scored on tables built at ITS redshift, not the model's."""
    bands, approx, kw = _CATALOG_VARIANTS[variant]
    obs = Observation(photometry=Photometry.from_names(bands))
    rows, nb = _catalog_rows(synthetic_ssp_wide, obs, approx, CATALOG_Z, **kw)
    base = _mass_only(synthetic_ssp_wide, obs, Z_BASE, approx, **kw)

    posteriors = _fit_catalog(base, rows, nb)

    for z, post in zip(CATALOG_Z, posteriors, strict=True):
        logm = float(post.params["sfh_dpl_log_total_mass"])
        assert abs(logm - TRUTH_LOGM) < 0.05, (
            f"z={z}: recovered log M = {logm:.3f}, truth {TRUTH_LOGM}; the row was "
            f"evaluated on tables built at another redshift"
        )
        assert float(post.fixed_values["redshift"]) == z


def test_catalog_rebuilds_once_per_distinct_redshift(synthetic_ssp_wide):
    """Repeated catalog redshifts share one rebuilt model, and the advisory still fires."""
    obs = Observation(photometry=Photometry.from_names(OPT_BANDS))
    approx = WavePrecomp()
    zs = (0.3, 0.9, 0.3, 0.9, 0.3)
    rows, nb = _catalog_rows(synthetic_ssp_wide, obs, approx, zs)
    base = _mass_only(synthetic_ssp_wide, obs, Z_BASE, approx)

    from tengri.forward.forward_model import ForwardModel

    real = ForwardModel.with_fixed_redshift
    with (
        mock.patch.object(
            ForwardModel, "with_fixed_redshift", autospec=True, side_effect=real
        ) as spy,
        pytest.warns(
            UserWarning, match=r"catalog_z_range.*recompiles|recompiles.*catalog_z_range"
        ),
    ):
        convenience.fit_batch(
            base,
            rows,
            flux_cols=[f"f{j}" for j in range(nb)],
            err_cols=[f"e{j}" for j in range(nb)],
            redshift_col="z",
            method="map",
            n_steps=2,
            verbose=False,
        )
    assert sorted(round(c.args[1], 6) for c in spy.call_args_list) == [0.3, 0.9], (
        "expected exactly one rebuild per distinct catalog redshift"
    )


# ── Multi-population forwards ─────────────────────────────────────────────


def _two_pop(base_builder, ssp, obs, z):
    """Two populations (a galaxy decomposition) sharing one redshift."""
    return ForwardModel.build(
        populations=[
            Population(name="a", sed=base_builder(ssp, obs, z, WavePrecomp())),
            Population(name="b", sed=base_builder(ssp, obs, z, WavePrecomp())),
        ],
        observation=obs,
    )


def test_multi_population_override_matches_direct_build(synthetic_ssp_wide):
    """Every population is rebuilt at the override redshift, names and mode kept."""
    obs = Observation(photometry=Photometry.from_names(OPT_BANDS))
    base = _two_pop(_mass_only_builder, synthetic_ssp_wide, obs, Z_BASE)
    direct = _two_pop(_mass_only_builder, synthetic_ssp_wide, obs, Z_OVERRIDE)

    rebuilt = base.with_fixed_redshift(Z_OVERRIDE)
    assert [p.name for p in rebuilt.populations] == ["a", "b"]
    assert rebuilt.mode == base.mode
    for pop in rebuilt.populations:
        assert pop.sed.spec.get_fixed_values()["redshift"] == Z_OVERRIDE
    assert base.populations[0].sed.spec.get_fixed_values()["redshift"] == Z_BASE

    params = {"sfh_dpl_log_total_mass": 10.0}
    phot = lambda m: np.asarray(m.predict_observables(params)["phot_fnu"])  # noqa: E731
    np.testing.assert_allclose(phot(rebuilt), phot(direct), rtol=RTOL)
    assert np.max(np.abs(phot(base) - phot(direct)) / np.abs(phot(direct))) > VACUITY_FLOOR


def test_hierarchical_population_override_rebuilds_the_template(synthetic_ssp_wide):
    """A hierarchical forward rebuilds its template SED and keeps galaxies and priors."""
    from tengri.forward.population_sed_model import PopulationSEDModel

    obs = Observation(photometry=Photometry.from_names(OPT_BANDS))
    template = _mass_only(synthetic_ssp_wide, obs, Z_BASE, WavePrecomp())
    flux = np.asarray(template.predict_photometry({"sfh_dpl_log_total_mass": TRUTH_LOGM}))
    galaxies = [{"flux_obs": flux, "noise": 0.05 * flux}, {"flux_obs": flux, "noise": 0.1 * flux}]
    pop = PopulationSEDModel(template, galaxies)
    fm = ForwardModel.build(population=pop, observation=obs)

    rebuilt = fm.with_fixed_redshift(Z_OVERRIDE)
    sub = rebuilt.populations[0].sed
    assert rebuilt.mode == "hierarchical"
    assert sub.sed.spec.get_fixed_values()["redshift"] == Z_OVERRIDE
    assert len(sub.galaxies) == 2
    assert sub.shared == pop.shared
    assert dict(sub.priors) == dict(pop.priors)
    assert pop.sed.spec.get_fixed_values()["redshift"] == Z_BASE


def test_a_free_redshift_population_refuses_the_rebuild(synthetic_ssp_wide):
    """A free redshift is fit, not pinned: the error names the parameter, not NotImplemented."""
    from tengri.config.exceptions import ParameterError

    obs = Observation(photometry=Photometry.from_names(OPT_BANDS))
    fixed = _phot_only(synthetic_ssp_wide, obs, Z_BASE, WavePrecomp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        free = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=obs,
            redshift=Uniform(0.01, 0.2),
            sfh=_SFH,
            met={"logzsol": Fixed(0.0)},
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
        )
    fm = ForwardModel.build(
        populations=[Population(name="a", sed=fixed), Population(name="b", sed=free)],
        observation=obs,
    )
    with pytest.raises(ParameterError, match="redshift"):
        fm.with_fixed_redshift(Z_OVERRIDE)


# ── The default path: profile_mass at its default, real SSP ───────────────


def test_default_profile_mass_override_matches_direct_build_on_real_ssp(ssp_data_fsps):
    """Fits default to profile_mass="auto"; the override must hold on that objective too."""
    obs = Observation(photometry=Photometry.from_names(OPT_BANDS))
    approx = WavePrecomp()
    base = _mass_only(ssp_data_fsps, obs, Z_BASE, approx)
    direct = _mass_only(ssp_data_fsps, obs, Z_OVERRIDE, approx)
    params = {"sfh_dpl_log_total_mass": TRUTH_LOGM}
    data = np.asarray(direct.predict_photometry(params))
    noise = 0.05 * data

    def make(model, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Fitter(  # profile_mass left at its default
                model, data=data, noise=noise, data_type="photometry", approx=approx, **kw
            )

    f_over = make(base, params_override={"redshift": Z_OVERRIDE})
    f_direct = make(direct)
    f_base = make(base)
    assert f_over._profile_mass_reason == f_direct._profile_mass_reason
    loss_o, loss_d, loss_b = _loss(f_over), _loss(f_direct), _loss(f_base)
    np.testing.assert_allclose(loss_o, loss_d, rtol=RTOL)
    assert abs(loss_b - loss_d) > VACUITY_FLOOR * abs(loss_d), (
        "vacuity: the un-overridden model has the same default-path loss"
    )
    phot_o = np.asarray(f_over.model.predict_photometry(params))
    np.testing.assert_allclose(phot_o, data, rtol=RTOL)
