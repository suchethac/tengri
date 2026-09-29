# SPDX-License-Identifier: BSD-3-Clause
r"""Regression: Fixed redshift propagation to line-flux calculations with catalog_z_range.

When using ``WavePrecomp(catalog_z_range=...)`` with a ``Fixed(z)`` redshift, the model's
own ``_z_fixed`` / ``_dl_cm_fixed`` are deliberately left ``None`` (so many galaxies can
share one compiled kernel with a runtime redshift). Before the fix, every line-flux
evaluation path (build-time nebular precompute, and the ``predict_line_fluxes`` /
``predict_line_ratios`` / ``measure_line_fluxes`` evaluation methods) resolved redshift
through ``_get_redshift``'s old ``_z_fixed``-only fallback, so it raised ``KeyError``
the moment a ``Fixed`` redshift was legitimately absent from the free-only ``params``
dict.

The fix makes ``_get_redshift`` / ``_get_dl_cm`` resolve through
``SEDModel._evaluation_params`` (the spec's own declared ``Fixed`` values, merged with
an optional evaluation-time ``fixed_values`` override -- e.g. a ``Fitter``'s runtime
redshift under ``catalog_z_range``, see ``fitter.py``'s ``_runtime_redshift`` / spec
#1320 Section 9.4), and threads that ``fixed_values`` channel through every line/feature
evaluation method and their internal ``merge_fixed_params`` call sites.

These are VALUE tests (not just "does it build"): every comparison below matches a
catalog-built model against a plain model built directly at the same redshift, and every
positive comparison is paired with a vacuity check that a genuinely different redshift
produces a genuinely different answer.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    FeaturePrecomp,
    Fitter,
    Fixed,
    Observation,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.inference.context import InferenceContext
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineList
from tengri.observation.line_ratio_data import LineRatioData
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug

BANDS = ["des_g", "des_r", "des_z", "wise_w1", "wise_w2"]
LINES = ["OII_3726", "Hbeta", "OIII_5007", "Halpha", "SII_6717"]

#: Shared evaluation point. Hand-picked (not sampled) so every test runs on the
#: identical, physically valid point regardless of how the model's free-parameter
#: spec was declared, and stays valid across every build redshift used below
#: (0.1-1.5): age_at_z at the highest of those is still >> sfh_dpl_age_gyr.
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
SFH_PARAMS = {k: v for k, v in PARAMS.items() if k.startswith("sfh_")}

#: Loose enough to comfortably cover float64 rounding between two independently
#: built models (a cached ``_dl_cm_fixed`` vs a freshly computed one, etc.), tight
#: enough that a redshift resolved from the wrong value cannot pass by accident.
RTOL = 1e-6
#: A vacuity threshold: two genuinely different redshifts must disagree by far
#: more than RTOL, or the "positive" comparison above it would be uninformative.
VACUITY_FLOOR = 1000 * RTOL


def synthetic_ssp_wide():
    """Minimal synthetic SSP for testing."""
    n_met, n_age = 3, 25
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


def line_data():
    """Minimal line-flux data for testing."""
    cat = LineList.default_optical()
    waves = jnp.asarray([float(w) for n, w in zip(cat.names, cat.wavelengths) if n in LINES])
    f = jnp.ones(len(LINES))
    return LineFluxData(
        names=tuple(LINES),
        fluxes=f,
        errors=jnp.abs(f) * 0.1 + 1e-30,
        wavelengths=waves,
    )


def _build_model(ssp, obs, z, approx):
    """One builder for every test below: only ``z`` / ``approx`` vary.

    Cue nebular backend (Q_H-linear, required for ``FeaturePrecomp``'s per-Q_H
    grid). ``met_logzsol`` is Fixed (not a grid axis) so ``FeaturePrecomp``'s
    grid stays 2-D (``neb_logU`` x ``neb_logZ_gas``) and small ``n_grid`` builds
    run in well under a second.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(z),
            sfh={"type": "dpl", "all_params": FREE},
            met={"logzsol": Fixed(0.0)},
            neb={
                "type": "cue",
                "all_params": Fixed(DEFAULT),
                "logU": Uniform(-4.0, -1.0),
                "logZ_gas": Uniform(-1.5, 0.3),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_bc": Uniform(0.0, 4.0),
                "tau_diff": Uniform(0.0, 3.0),
            },
            dust_emission={"type": "none"},
            approx=approx,
        )


def _build_model_fixed_dust(ssp, obs, z, approx, *, tau_bc):
    """A variant with EVERY dust/nebular parameter Fixed (only SFH is free).

    Used by the ``params_override`` dust test: ``params_override`` (and
    ``predict_line_fluxes(..., fixed_values=...)``) may only re-pin a Fixed
    parameter (#1329), so the parameter under test must not be free here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(z),
            sfh={"type": "dpl", "all_params": FREE},
            met={"logzsol": Fixed(0.0)},
            neb={
                "type": "cue",
                "all_params": Fixed(DEFAULT),
                "logU": Fixed(-2.5),
                "logZ_gas": Fixed(0.0),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_bc": Fixed(tau_bc),
                "tau_diff": Fixed(0.2),
            },
            dust_emission={"type": "none"},
            approx=approx,
        )


def _relative_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def _full_fixed_values(model, **overrides):
    """A COMPLETE fixed-values dict (every spec Fixed value) with overrides.

    ``predict_state``'s ``fixed_values`` kwarg is an "already resolved, trust
    me" escape hatch (see its docstring): when not ``None`` it skips
    ``merge_fixed_params`` entirely and merges ``{**fixed_values, **params}``
    verbatim, so a caller-supplied ``fixed_values`` must be the model's FULL
    Fixed-value set, not a sparse override -- exactly what
    ``Fitter._build_data_args`` passes (``dict(model.spec.get_fixed_values())``
    updated with ``params_override``). A test calling ``predict_line_fluxes(...,
    fixed_values=...)`` with only the one key under test would hit this
    contract the moment the method's ``state is None`` branch threads it into
    ``predict_state`` internally, and raise ``KeyError`` for every OTHER Fixed
    parameter instead.
    """
    return {**dict(model.spec.get_fixed_values()), **overrides}


# ── A: build-time (enable_fast_nebular / FeaturePrecomp) ──────────────────


def test_catalog_grid_matches_plain_grid_and_fast_line_fluxes():
    """catalog_z_range + FeaturePrecomp builds the SAME grid/line fluxes as plain.

    Before the fix this raised ``KeyError`` inside ``_snap_to_nebular_catalog``
    (build-time) the moment ``_z_fixed`` was left ``None`` under
    ``catalog_z_range``. This upgrades the old "builds without raising" smoke
    check to a VALUE comparison against a non-catalog build at the same z.
    """
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())

    catalog_model = _build_model(
        ssp,
        obs,
        0.1,
        (WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50), FeaturePrecomp(n_grid=4)),
    )
    plain_model = _build_model(ssp, obs, 0.1, (WavePrecomp(), FeaturePrecomp(n_grid=4)))

    np.testing.assert_allclose(
        np.asarray(catalog_model._nebular_grid_table.wavelengths),
        np.asarray(plain_model._nebular_grid_table.wavelengths),
        rtol=RTOL,
    )

    lf_catalog = np.asarray(catalog_model.predict_line_fluxes(PARAMS))
    lf_plain = np.asarray(plain_model.predict_line_fluxes(PARAMS))
    np.testing.assert_allclose(lf_catalog, lf_plain, rtol=RTOL)


# ── B: evaluation methods (exact path, no build-time FeaturePrecomp) ──────


def test_predict_line_fluxes_matches_plain_model_at_same_z():
    """predict_line_fluxes: catalog model == plain model at the same Fixed(z)."""
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())

    catalog_01 = _build_model(ssp, obs, 0.1, WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50))
    plain_01 = _build_model(ssp, obs, 0.1, WavePrecomp())
    plain_08 = _build_model(ssp, obs, 0.8, WavePrecomp())

    lf_catalog = np.asarray(catalog_01.predict_line_fluxes(PARAMS))
    lf_plain01 = np.asarray(plain_01.predict_line_fluxes(PARAMS))
    lf_plain08 = np.asarray(plain_08.predict_line_fluxes(PARAMS))

    np.testing.assert_allclose(lf_catalog, lf_plain01, rtol=RTOL)
    assert _relative_diff(lf_plain08, lf_plain01) > VACUITY_FLOOR, (
        "vacuity: z=0.8 and z=0.1 line fluxes must differ by far more than RTOL"
    )


def test_measure_line_fluxes_matches_plain_model_at_same_z():
    """measure_line_fluxes: catalog model == plain model at the same Fixed(z)."""
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())

    catalog_01 = _build_model(ssp, obs, 0.1, WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50))
    plain_01 = _build_model(ssp, obs, 0.1, WavePrecomp())
    plain_08 = _build_model(ssp, obs, 0.8, WavePrecomp())

    mlf_catalog = np.asarray(catalog_01.measure_line_fluxes(PARAMS))
    mlf_plain01 = np.asarray(plain_01.measure_line_fluxes(PARAMS))
    mlf_plain08 = np.asarray(plain_08.measure_line_fluxes(PARAMS))

    np.testing.assert_allclose(mlf_catalog, mlf_plain01, rtol=RTOL)
    assert _relative_diff(mlf_plain08, mlf_plain01) > VACUITY_FLOOR, (
        "vacuity: z=0.8 and z=0.1 measured line fluxes must differ by far more than RTOL"
    )


def test_predict_line_ratios_matches_plain_model_at_same_z():
    """predict_line_ratios: catalog model == plain model at the same Fixed(z).

    ``LineRatioData`` is constructible here (``LineRatioData.from_dict``); the
    observed ratio/error values are irrelevant to ``predict_line_ratios`` (it
    returns the MODEL ratio only), only the numerator/denominator line names
    matter.
    """
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())

    catalog_01 = _build_model(ssp, obs, 0.1, WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50))
    plain_01 = _build_model(ssp, obs, 0.1, WavePrecomp())
    plain_08 = _build_model(ssp, obs, 0.8, WavePrecomp())

    lrd = LineRatioData.from_dict({("Halpha", "Hbeta"): (2.86, 0.1)})

    lr_catalog = np.asarray(catalog_01.predict_line_ratios(PARAMS, lrd))
    lr_plain01 = np.asarray(plain_01.predict_line_ratios(PARAMS, lrd))
    lr_plain08 = np.asarray(plain_08.predict_line_ratios(PARAMS, lrd))

    np.testing.assert_allclose(lr_catalog, lr_plain01, rtol=RTOL)
    # No vacuity-by-redshift check here: the observed-flux scale
    # `1 / (4 pi d_L(z)^2)` is COMMON to the numerator and denominator and
    # cancels exactly in the ratio (see predict_line_ratios's own comment on
    # this), so a line ratio is provably z-independent by design -- z=0.8 and
    # z=0.1 line ratios are expected to agree, not differ. The equality check
    # above still exercises (and would catch a KeyError from) the redshift
    # resolution inside `_get_dl_cm`.
    np.testing.assert_allclose(lr_plain08, lr_plain01, rtol=RTOL)


# ── C: Fitter runtime redshift under catalog_z_range ───────────────────────


def _mock_photometry(model, truth):
    phot = np.asarray(model.predict_photometry(truth))
    return phot, 0.05 * np.abs(phot) + 1e-31


def test_fitter_runtime_redshift_matches_plain_model_loss():
    """A Fitter's runtime z override (catalog_z_range) matches a direct build.

    Must match a model built directly at that redshift -- same loss, at the
    same standardized point, for
    the SAME (photometry + line-flux) data. Explicitly hands ``approx=`` to
    every Fitter (matching the model's own build-time config) so the comparison
    stays on the EXACT line-flux path on both sides: FeaturePrecomp's per-Q_H
    grid is a genuine (if small, ~0.1-1%) reconstruction approximation, and its
    reference SFH depends on the model's OWN build redshift, so two
    differently-built grids would not agree to RTOL even with a correct fix.
    """
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())

    wp_catalog = WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50)
    catalog_model = _build_model(ssp, obs, 0.1, wp_catalog)
    plain_08 = _build_model(ssp, obs, 0.8, WavePrecomp())
    plain_03 = _build_model(ssp, obs, 0.3, WavePrecomp())

    phot, phot_err = _mock_photometry(plain_08, PARAMS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_catalog = Fitter(
            catalog_model,
            data=phot,
            noise=phot_err,
            data_type="photometry",
            approx=wp_catalog,
            params_override={"redshift": 0.8},
        )
        f_plain08 = Fitter(
            plain_08, data=phot, noise=phot_err, data_type="photometry", approx=WavePrecomp()
        )
        f_plain03 = Fitter(
            plain_03, data=phot, noise=phot_err, data_type="photometry", approx=WavePrecomp()
        )

    assert "redshift" in f_catalog._data_args, (
        "catalog_z_range + params_override redshift must route through data_args (#1316)"
    )

    ctx_c = InferenceContext.from_target(f_catalog)
    ctx_08 = InferenceContext.from_target(f_plain08)
    ctx_03 = InferenceContext.from_target(f_plain03)
    p_u = ctx_c.initial_params(jax.random.PRNGKey(7))

    loss_c_at_08 = float(ctx_c.neg_log_posterior_fn(p_u, ctx_c.data_args))
    loss_plain_08 = float(ctx_08.neg_log_posterior_fn(p_u, ctx_08.data_args))
    np.testing.assert_allclose(loss_c_at_08, loss_plain_08, rtol=RTOL)

    # ONE compiled catalog loss, re-evaluated at a different data_args redshift,
    # must match a SEPARATELY built plain model at that redshift too (#1316
    # spec Sec. 9.4: distinct runtime z must not fork the compiled program).
    data_args_03 = {**ctx_c.data_args, "redshift": jnp.asarray(0.3)}
    loss_c_at_03 = float(ctx_c.neg_log_posterior_fn(p_u, data_args_03))
    loss_plain_03 = float(ctx_03.neg_log_posterior_fn(p_u, ctx_03.data_args))
    np.testing.assert_allclose(loss_c_at_03, loss_plain_03, rtol=RTOL)

    assert _relative_diff(loss_c_at_03, loss_c_at_08) > VACUITY_FLOOR, (
        "vacuity: the same compiled loss at two different runtime redshifts "
        "must give genuinely different values"
    )


# ── D: params_override redshift on a NON-catalog model (compile-time bake) ─


def test_non_catalog_params_override_redshift_matches_direct_build():
    """params_override redshift on a non-catalog model bakes, not routes (#1331).

    Without catalog_z_range, params_override redshift is a compile constant
    rather than a data_args runtime input, but the evaluation methods
    (fixed_values-aware since this fix) must still see the OVERRIDDEN z, not
    the model's own build-time Fixed value -- this is exactly mutant M3's
    target (``_get_dl_cm`` preferring ``_dl_cm_fixed``, which IS set here,
    over ``fixed_values``).
    """
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())

    plain_01 = _build_model(ssp, obs, 0.1, WavePrecomp())
    plain_08 = _build_model(ssp, obs, 0.8, WavePrecomp())

    phot, phot_err = _mock_photometry(plain_08, PARAMS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_override = Fitter(
            plain_01,
            data=phot,
            noise=phot_err,
            data_type="photometry",
            approx=WavePrecomp(),
            params_override={"redshift": 0.8},
        )
        f_direct = Fitter(
            plain_08, data=phot, noise=phot_err, data_type="photometry", approx=WavePrecomp()
        )

    assert "redshift" not in f_override._data_args, (
        "no catalog_z_range: the override must stay a compile constant (#1331), "
        "not a data_args runtime input -- if this now fails, params_override "
        "started routing through data_args even without a ztable and this test "
        "should be updated to match"
    )

    ctx_o = InferenceContext.from_target(f_override)
    ctx_d = InferenceContext.from_target(f_direct)
    p_u = ctx_o.initial_params(jax.random.PRNGKey(11))

    loss_o = float(ctx_o.neg_log_posterior_fn(p_u, ctx_o.data_args))
    loss_d = float(ctx_d.neg_log_posterior_fn(p_u, ctx_d.data_args))
    np.testing.assert_allclose(loss_o, loss_d, rtol=RTOL)


# ── Mutation-targeted guards ─────────────────────────────────────────────


def test_predict_line_fluxes_dust_merge_respects_fixed_values_override():
    """M6 guard: predict_line_fluxes' dust-attenuation merge must respect fixed_values.

    Must consult fixed_values, not just the spec's own Fixed value. Exercises the
    non-``FeaturePrecomp`` (exact) merge site: ``redden=True`` with no grid
    attached, so ``full_params`` is built solely by that site.
    """
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())

    model = _build_model_fixed_dust(ssp, obs, 0.1, WavePrecomp(), tau_bc=1.0)

    baseline = np.asarray(model.predict_line_fluxes(SFH_PARAMS))
    overridden = np.asarray(
        model.predict_line_fluxes(
            SFH_PARAMS, fixed_values=_full_fixed_values(model, dust_tau_bc=2.5)
        )
    )

    direct_model = _build_model_fixed_dust(ssp, obs, 0.1, WavePrecomp(), tau_bc=2.5)
    direct = np.asarray(direct_model.predict_line_fluxes(SFH_PARAMS))

    np.testing.assert_allclose(overridden, direct, rtol=RTOL)
    assert _relative_diff(overridden, baseline) > VACUITY_FLOOR, (
        "vacuity: overriding dust_tau_bc via fixed_values did not change the predicted line fluxes"
    )


def test_predict_line_fluxes_grid_branch_dust_merge_respects_fixed_values_override():
    """M6 sibling: the SAME check on the ``FeaturePrecomp`` grid-branch site.

    ``predict_line_fluxes`` has TWO dust/nebular ``merge_fixed_params`` ->
    ``_evaluation_params`` sites: this one (``grid is not None``, feeding both
    the DIG mix and its reused redden screen) and the exact-path fallback the
    sibling test above exercises. A ``fixed_values``-blind mutation at THIS
    site is invisible to every test that calls ``predict_line_fluxes`` with
    ``fixed_values=None`` (identical execution, not merely unmeasured, since
    ``_evaluation_params(p, None) == merge_fixed_params(self.spec, p)``
    exactly) -- which is every other test in this file, including test A.
    """
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())

    approx = (WavePrecomp(), FeaturePrecomp(n_grid=4))
    model = _build_model_fixed_dust(ssp, obs, 0.1, approx, tau_bc=1.0)

    baseline = np.asarray(model.predict_line_fluxes(SFH_PARAMS))
    overridden = np.asarray(
        model.predict_line_fluxes(
            SFH_PARAMS, fixed_values=_full_fixed_values(model, dust_tau_bc=2.5)
        )
    )

    direct_model = _build_model_fixed_dust(ssp, obs, 0.1, approx, tau_bc=2.5)
    direct = np.asarray(direct_model.predict_line_fluxes(SFH_PARAMS))

    np.testing.assert_allclose(overridden, direct, rtol=RTOL)
    assert _relative_diff(overridden, baseline) > VACUITY_FLOOR, (
        "vacuity: overriding dust_tau_bc via fixed_values did not change the "
        "predicted line fluxes on the grid branch"
    )


def test_compute_log_nion_respects_fixed_values_redshift_override():
    """Guard for the grid branch's no-state ``_compute_log_nion`` fallback.

    Not one of the four lettered scenarios above (A-D deliberately avoid
    FeaturePrecomp for an exact rtol=1e-6 cross-build comparison, see test C's
    docstring), but ``_compute_log_nion`` shares the exact same
    ``merge_fixed_params`` -> ``_evaluation_params`` bug/fix as
    ``predict_line_fluxes``'s own merges, and reaching it needs a
    FeaturePrecomp-attached model called with ``state=None`` -- the ``fast_lines
    and not needs_state`` case in ``loss_functions._build_prediction``. Compares
    it against the SAME model's ``predict_state``-published ``log_nion`` at the
    SAME overridden redshift (both single-source the identical stellar Q_H
    computation, so this is grid-approximation-free unlike a cross-build
    comparison).
    """
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())
    model = _build_model(
        ssp,
        obs,
        0.1,
        (WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50), FeaturePrecomp(n_grid=4)),
    )

    # age_gyr=10.0 (unlike the shared PARAMS' 0.5) is close enough to
    # age_at_z(0.8) (~6.7 Gyr) that `compute_joint_weights`'s age-of-universe
    # SSP-bin mask (components/stellar/component.py's `_cic_parcels`) actually
    # clips differently at z=0.1 (age_at_z ~ 12.4 Gyr, no clip) vs z=0.8 --
    # measured 0.26 dex (~86% linear) apart, comfortably above VACUITY_FLOOR.
    # The shared PARAMS' age_gyr=0.5 measures bit-identical Q_H at both
    # redshifts (all its star formation sits at lookback << either age_at_z),
    # which is why it is not used for this check.
    params = {**PARAMS, "sfh_dpl_age_gyr": 10.0}

    fixed_at_08 = _full_fixed_values(model, redshift=0.8)
    log_nion_helper = np.asarray(model._compute_log_nion(params, fixed_values=fixed_at_08))
    with warnings.catch_warnings():
        # SFHBeforeBigBangWarning: age_gyr=10.0 deliberately exceeds age_at_z(0.8)
        # (see comment above) so the orchestrator's own age-of-universe guard
        # (distinct from, and downstream of, the `_compute_log_nion` fix under
        # test) fires here; not the subject of this test.
        warnings.simplefilter("ignore")
        state_08 = model.predict_state(params, fixed_values=fixed_at_08)
    log_nion_from_state = np.asarray(state_08.derived["log_nion"])
    if log_nion_from_state.ndim:
        log_nion_from_state = np.asarray(np.log10(np.sum(10.0**log_nion_from_state)))
    np.testing.assert_allclose(log_nion_helper, log_nion_from_state, rtol=RTOL)

    log_nion_baked = np.asarray(model._compute_log_nion(params))
    assert _relative_diff(log_nion_helper, log_nion_baked) > VACUITY_FLOOR, (
        "vacuity: overriding redshift via fixed_values did not change log_nion "
        "(the age-of-universe SFH cutoff must depend on the evaluation z)"
    )
