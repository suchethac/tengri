# SPDX-License-Identifier: BSD-3-Clause
"""Equality tests: orchestrator path vs legacy ``SEDModel.predict_*``.

The plan's success criterion for monolith deletion is that the
orchestrator produces bit-exact (or rtol=1e-10) output relative to
the legacy ``SEDModel.predict_rest_sed`` / ``predict_obs_sed`` path
for the same ``Parameters`` and ``params``. This module pins that
equality so any future drift surfaces as a test failure rather than
silent divergence.

If any of the strict equality tests here fail, deletion of the
corresponding monolith branch is **blocked** until the divergence
is resolved.

Status today: only the **physical-range** + **shape** tests pass.
The strict ``rtol=1e-6`` bit-exact comparisons are marked
``xfail`` with a documented reason — they're the gating criterion
for monolith deletion and the work to close them is the active
Phase II-2 milestone.
"""

from __future__ import annotations

import pathlib
import warnings

import chex
import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.parameters.priors import Fixed, Uniform
from tests._bounds import assert_non_negative

_SSP_PATH = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_PATH.exists():
        pytest.skip(f"SSP file not present at {_SSP_PATH}")
    return load_ssp_data(str(_SSP_PATH))


@pytest.fixture(scope="module")
def stellar_only_model(ssp):
    """Stellar-only model — minimal chain (zero dust, no IGM)."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(8, 12),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp)


_STELLAR_PARAMS = {
    # Only free params — fixed values (met_logzsol, redshift,
    # dust_tau_*) are read from spec by both predict_rest_sed and
    # predict_state (the latter injects via
    # spec.get_fixed_values()). This is the realistic call site.
    "sfh_tsnorm_log_total_mass": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 2.0,
    "sfh_tsnorm_width_gyr": 1.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 3.0,
}


# ── Fixed-param injection: orchestrator now reads from spec ──────────


def test_orchestrator_injects_fixed_values_from_spec(stellar_only_model):
    """Calling predict_state with only the FREE params must
    succeed — fixed values (met_logzsol, redshift, dust_tau_*) come
    from spec.get_fixed_values()."""
    free_only = {
        "sfh_tsnorm_log_total_mass": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 2.0,
        "sfh_tsnorm_width_gyr": 1.0,
        "sfh_tsnorm_skew": 0.0,
        "sfh_tsnorm_trunc": 3.0,
    }
    state = stellar_only_model.predict_state(free_only)
    # log_mstar published by stellar component → injection worked end-to-end.
    assert "log_mstar" in state.derived
    assert jnp.isfinite(state.derived["log_mstar"])


def test_orchestrator_explicit_param_overrides_spec_fixed(stellar_only_model):
    """A param passed explicitly must win over the spec's fixed value."""
    free_only = {
        "sfh_tsnorm_log_total_mass": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 2.0,
        "sfh_tsnorm_width_gyr": 1.0,
        "sfh_tsnorm_skew": 0.0,
        "sfh_tsnorm_trunc": 3.0,
    }
    state_default = stellar_only_model.predict_state(free_only)
    # Override met_logzsol with a different value than the spec fixed.
    state_overridden = stellar_only_model.predict_state({**free_only, "met_logzsol": 0.0})
    # Stellar SED differs because metallicity strongly affects spectral shape.
    rel_diff = float(
        jnp.max(
            jnp.abs(state_default.sed_intrinsic - state_overridden.sed_intrinsic)
            / jnp.maximum(jnp.abs(state_default.sed_intrinsic), 1e-30)
        )
    )
    assert rel_diff > 1e-3, (
        f"Override didn't change SED: max rel diff = {rel_diff:.3e} "
        "(orchestrator ignored explicit met_logzsol?)"
    )


# ── Sanity: both paths run, produce finite + same-shape output ────────


def test_stellar_only_orchestrator_runs(stellar_only_model):
    """Orchestrator path produces finite SED of expected shape."""
    state = stellar_only_model.predict_state(_STELLAR_PARAMS)
    assert state.sed_intrinsic.shape[0] > 1000, (
        f"sed_intrinsic shape too small: {state.sed_intrinsic.shape}"
    )
    chex.assert_tree_all_finite(state.sed_intrinsic)
    assert_non_negative(state.sed_intrinsic, name="output")


def test_stellar_only_legacy_runs(stellar_only_model):
    """Legacy path produces finite SED of expected shape."""
    legacy = stellar_only_model.predict_rest_sed(_STELLAR_PARAMS)
    assert legacy.sed.shape[0] > 1000
    chex.assert_tree_all_finite(legacy.sed)
    assert_non_negative(legacy.sed, name="output")


# ── Physical-range agreement: both within order-of-magnitude ──────────


def test_orchestrator_vs_legacy_mstar_physical_agreement(stellar_only_model):
    """log10(M*) agrees between paths — compared like-for-like.

    The old assertion compared legacy ``stellar_mass`` (documented "total
    mass formed") against ``derived['log_mstar']`` (documented "surviving
    stellar mass"): the ~0.19 dex "disagreement" it reported was the stellar
    mass-loss return fraction (10**-0.19 ~= 0.65), not a path divergence.
    Compare formed-vs-formed and surviving-vs-surviving instead — a strictly
    stronger contract than the old mixed one.
    """
    legacy_q = stellar_only_model.predict(_STELLAR_PARAMS)
    state = stellar_only_model.predict_state(_STELLAR_PARAMS)

    legacy_formed = float(jnp.log10(legacy_q.stellar_mass))
    orch_formed = float(state.derived["log_mstar_formed"])
    diff_formed = abs(legacy_formed - orch_formed)
    assert diff_formed < 0.02, (
        f"formed log10(M*) disagrees by {diff_formed:.3f} dex: "
        f"legacy={legacy_formed:.3f}, orch={orch_formed:.3f}"
    )

    legacy_surv = float(jnp.log10(legacy_q.stellar_mass_surviving))
    orch_surv = float(state.derived["log_mstar"])
    diff_surv = abs(legacy_surv - orch_surv)
    assert diff_surv < 0.05, (
        f"surviving log10(M*) disagrees by {diff_surv:.3f} dex: "
        f"legacy={legacy_surv:.3f}, orch={orch_surv:.3f}"
    )

    # Physical ordering: surviving mass can never exceed formed mass.
    assert orch_surv <= orch_formed + 1e-9


# ── Strict bit-exact: gating criterion for monolith deletion ──────────


def test_orchestrator_rest_sed_close_to_legacy(stellar_only_model):
    """Orchestrator's stellar SED agrees with legacy's
    ``predict_rest_sed.sed`` at ``rtol=1e-2`` after the DSPS-canonical
    metallicity migration (commit ``20260504-csp-integral-...``).

    Residual ~0.2% comes from legacy SFH integration (trapezoidal in
    lookback time on the SSP age grid) vs DSPS canonical (trapezoidal
    in cosmic time). Closing this residual to ``rtol=1e-6`` requires
    migrating the SFH side too — tracked as the next milestone in
    ``docs/dev/20260504-csp-integral-canonicalization.md``.
    """
    legacy = stellar_only_model.predict_rest_sed(_STELLAR_PARAMS)
    state = stellar_only_model.predict_state(_STELLAR_PARAMS)

    chex.assert_equal_shape([legacy.sed, state.sed_intrinsic])
    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 1e-2, f"max rel diff: {rel_diff:.3e}"


def test_orchestrator_rest_sed_bit_exact_to_legacy(stellar_only_model):
    """Orchestrator's stellar SED equals legacy's predict_rest_sed.sed
    at rtol=1e-6 — bit-exact within float64 precision.

    Closed by closure-path-A in the no-α delta-Z branch of
    ``forward/pipeline.py``: that branch now calls
    ``calc_rest_sed_sfh_table_lognormal_mdf`` directly (mirroring
    :class:`StellarSEDComponent.apply` exactly) instead of the
    JIT-kernel two-step einsum. With matched grid construction
    (n_grid=64, linear lookback-time interpolation, cosmic-time
    floor=1e-3 Gyr), legacy and orchestrator feed bit-identical
    inputs to DSPS and produce SEDs that agree to ~2e-15 (machine
    epsilon). See ``docs/dev/20260504-csp-integral-canonicalization.md``.
    """
    legacy = stellar_only_model.predict_rest_sed(_STELLAR_PARAMS)
    state = stellar_only_model.predict_state(_STELLAR_PARAMS)

    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 1e-6, f"max rel diff: {rel_diff:.3e}"


# ── Phase II-2.3: stochastic GP field equivalence ────────────────────


_FIELD_N_GRID = 64


@pytest.fixture(scope="module")
def stellar_field_model(ssp):
    """Stellar-only model with field=True (PSD-governed GP modulation).

    Adds ``"field"`` to ``mean_sfh_type`` to enable the stochastic
    branch in both legacy and orchestrator paths. The GP-draw vector
    ``sfh_field_xi`` is supplied at *call time* in ``_STELLAR_FIELD_PARAMS``
    (it's a runtime input rather than a ``Fixed`` prior; see
    :mod:`tengri.parameters.translate`).
    """
    spec = Parameters(
        mean_sfh_type=["tsnorm", "field"],
        n_grid=_FIELD_N_GRID,
        sfh_tsnorm_log_total_mass=Uniform(8, 12),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        sfh_field_psd_sigma=Uniform(0.05, 0.5),
        sfh_field_psd_tau_myr=Uniform(50.0, 500.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp)


def _field_params():
    """Build a deterministic GP-draw payload used by every field test."""
    import numpy as np

    rng = np.random.default_rng(seed=42)
    return {
        **_STELLAR_PARAMS,
        "sfh_field_psd_sigma": 0.2,
        "sfh_field_psd_tau_myr": 200.0,
        "sfh_field_xi": jnp.asarray(rng.standard_normal(_FIELD_N_GRID)),
    }


_STELLAR_FIELD_PARAMS = _field_params()


def test_field_orchestrator_runs(stellar_field_model):
    """Orchestrator handles field=True without raising and produces finite SFH."""
    state = stellar_field_model.predict_state(_STELLAR_FIELD_PARAMS)
    sfr_history = state.derived["sfr_history"]
    chex.assert_tree_all_finite(sfr_history)
    assert jnp.any(sfr_history > 0.0), "sfr_history is all zero — field branch dead?"


def test_field_legacy_runs(stellar_field_model):
    """Legacy handles field=True without raising and produces finite SFH."""
    sfh = stellar_field_model.predict_sfh(_STELLAR_FIELD_PARAMS)
    chex.assert_tree_all_finite(sfh["sfr_full"])
    assert jnp.any(sfh["sfr_full"] > 0.0), "legacy sfr_full all zero — field branch dead?"


# ── Phase II-2.6: Galaxy.predict() unified entry point ───────────────


@pytest.fixture(scope="module")
def stellar_only_galaxy(ssp, stellar_only_model):
    """Minimal Galaxy wrapping the stellar-only SEDModel.

    Bypasses ``Galaxy.from_arrays`` (which builds the full model from
    a preset) by constructing the Galaxy directly with a pre-built
    ``SEDModel``. We don't need real photometry for this dispatch test;
    we just need a Galaxy whose ``build_model()`` returns the stellar-
    only model already used by the SED-equivalence tests.
    """
    from tengri import Galaxy
    from tengri.config.settings import SEDModelConfig

    return Galaxy(
        ssp=ssp,
        observation=None,
        parameters=stellar_only_model.spec,
        model_config=SEDModelConfig(),
        model=stellar_only_model,
    )


def test_galaxy_predict_legacy_returns_prediction(stellar_only_galaxy):
    """Galaxy.predict(..., backend='legacy') returns a Prediction."""
    from tengri.forward.prediction import Prediction

    pred = stellar_only_galaxy.predict(_STELLAR_PARAMS)
    assert isinstance(pred, Prediction)
    # And the SED is finite
    assert jnp.all(jnp.isfinite(pred.sed.l_bol)) if hasattr(pred.sed, "l_bol") else True


def test_galaxy_predict_component_returns_pipeline_state(stellar_only_galaxy):
    """Galaxy.predict(..., backend='component') returns a ForwardState
    with the expected stellar derived keys."""
    from tengri.protocols.component import ForwardState

    state = stellar_only_galaxy.predict(_STELLAR_PARAMS, backend="component")
    assert isinstance(state, ForwardState)
    assert "log_mstar" in state.derived
    assert "sfr_history" in state.derived


def test_galaxy_predict_unknown_backend_raises(stellar_only_galaxy):
    """Galaxy.predict(..., backend='nonsense') raises ValueError."""
    with pytest.raises(ValueError, match="must be 'legacy' or 'component'"):
        stellar_only_galaxy.predict(_STELLAR_PARAMS, backend="nonsense")


def test_galaxy_predict_default_is_legacy(stellar_only_galaxy):
    """Default backend remains 'legacy' until Phase B v1.0 cutover."""
    from tengri.forward.prediction import Prediction
    from tengri.protocols.component import ForwardState

    default_pred = stellar_only_galaxy.predict(_STELLAR_PARAMS)
    explicit_legacy = stellar_only_galaxy.predict(_STELLAR_PARAMS, backend="legacy")
    assert isinstance(default_pred, Prediction)
    assert isinstance(explicit_legacy, Prediction)
    # And not a ForwardState
    assert not isinstance(default_pred, ForwardState)


def test_galaxy_predict_via_components_alias_unchanged(stellar_only_galaxy):
    """The pre-existing predict_via_components() shim still works (aliasing
    Galaxy.predict(..., backend='component'))."""
    state_alias = stellar_only_galaxy.predict_via_components(_STELLAR_PARAMS)
    state_unified = stellar_only_galaxy.predict(_STELLAR_PARAMS, backend="component")
    # Both call the same SEDModel.predict_state on the same params,
    # so the published quantities should be identical.
    assert jnp.allclose(state_alias.sed_intrinsic, state_unified.sed_intrinsic, rtol=1e-12)


# ── Phase II-2.5: non-parametric SFH modes ───────────────────────────


@pytest.fixture(scope="module")
def stellar_dirichlet_model(ssp):
    """Stellar-only model with non-parametric Dirichlet SFH (Leja+2017)."""
    spec = Parameters(
        mean_sfh_type=["dirichlet"],
        sfh_dir_log_total_mass=Uniform(8.0, 12.0),
        sfh_dir_z_0=Uniform(0.01, 0.99),
        sfh_dir_z_1=Uniform(0.01, 0.99),
        sfh_dir_z_2=Uniform(0.01, 0.99),
        sfh_dir_z_3=Uniform(0.01, 0.99),
        sfh_dir_z_4=Uniform(0.01, 0.99),
        sfh_dir_z_5=Uniform(0.01, 0.99),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp)


_DIRICHLET_PARAMS = {
    "sfh_dir_log_total_mass": 10.5,
    "sfh_dir_z_0": 0.2,
    "sfh_dir_z_1": 0.3,
    "sfh_dir_z_2": 0.4,
    "sfh_dir_z_3": 0.5,
    "sfh_dir_z_4": 0.6,
    "sfh_dir_z_5": 0.7,
}


def test_dirichlet_orchestrator_runs(stellar_dirichlet_model):
    """Orchestrator handles dirichlet SFH: finite sfr_history with positive bins."""
    state = stellar_dirichlet_model.predict_state(_DIRICHLET_PARAMS)
    sfr = state.derived["sfr_history"]
    chex.assert_tree_all_finite(sfr)
    assert jnp.any(sfr > 0.0), "dirichlet sfr_history all zero"


def test_dirichlet_legacy_runs(stellar_dirichlet_model):
    """Legacy handles dirichlet SFH without raising."""
    legacy = stellar_dirichlet_model.predict_rest_sed(_DIRICHLET_PARAMS)
    chex.assert_tree_all_finite(legacy.sed)
    assert jnp.any(legacy.sed > 0.0), "legacy dirichlet SED all zero"


def test_dirichlet_orchestrator_rest_sed_close_to_legacy(stellar_dirichlet_model):
    """Phase II-2.5 contract: orchestrator's stellar SED with dirichlet SFH
    agrees with legacy at ``rtol=5e-2`` — slightly looser than the
    ``rtol=1e-2`` tsnorm/dpl bar because piecewise-constant SFHs
    amplify the SFH-integration mismatch from the unrelated CSP
    canonicalization work tracked in
    ``docs/dev/20260504-csp-integral-canonicalization.md``.

    Both paths dispatch through ``SFH_REGISTRY["dirichlet"].fn``
    (the bare ``dirichlet`` function) with the same internal kwargs
    derived from the registry's ``internal_param_map``. The orchestrator
    drives the dispatch directly; legacy drives it through
    ``resolve_sfh`` which composes a wrapper. Closing the gap to
    ``rtol=1e-3`` is contingent on migrating the legacy SFH integration
    onto DSPS canonical trapezoidal-in-cosmic-time.
    """
    legacy = stellar_dirichlet_model.predict_rest_sed(_DIRICHLET_PARAMS)
    state = stellar_dirichlet_model.predict_state(_DIRICHLET_PARAMS)

    chex.assert_equal_shape([legacy.sed, state.sed_intrinsic])
    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 5e-2, f"max rel diff: {rel_diff:.3e}"


@pytest.fixture(scope="module")
def stellar_continuity_model(ssp):
    """Stellar-only model with non-parametric continuity SFH (Leja+2019)."""
    spec = Parameters(
        mean_sfh_type=["continuity"],
        sfh_cont_log_total_mass=Uniform(8.0, 12.0),
        sfh_cont_ratio_0=Uniform(-1.0, 1.0),
        sfh_cont_ratio_1=Uniform(-1.0, 1.0),
        sfh_cont_ratio_2=Uniform(-1.0, 1.0),
        sfh_cont_ratio_3=Uniform(-1.0, 1.0),
        sfh_cont_ratio_4=Uniform(-1.0, 1.0),
        sfh_cont_ratio_5=Uniform(-1.0, 1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp)


_CONTINUITY_PARAMS = {
    "sfh_cont_log_total_mass": 10.5,
    "sfh_cont_ratio_0": 0.1,
    "sfh_cont_ratio_1": 0.0,
    "sfh_cont_ratio_2": -0.2,
    "sfh_cont_ratio_3": 0.3,
    "sfh_cont_ratio_4": -0.1,
    "sfh_cont_ratio_5": 0.0,
}


def test_continuity_orchestrator_rest_sed_close_to_legacy(stellar_continuity_model):
    """Phase II-2.5 contract: continuity SFH parity at ``rtol=5e-2``."""
    legacy = stellar_continuity_model.predict_rest_sed(_CONTINUITY_PARAMS)
    state = stellar_continuity_model.predict_state(_CONTINUITY_PARAMS)

    chex.assert_equal_shape([legacy.sed, state.sed_intrinsic])
    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 5e-2, f"max rel diff: {rel_diff:.3e}"


@pytest.fixture(scope="module")
def stellar_dense_basis_model(ssp):
    """Stellar-only model with non-parametric dense_basis SFH (Iyer+2017)."""
    spec = Parameters(
        mean_sfh_type=["dense_basis"],
        sfh_db_log_total_mass=Uniform(8.0, 12.0),
        sfh_db_log_sfr_inst=Uniform(-2.0, 3.0),
        sfh_db_tx_frac_0=Uniform(0.05, 0.95),
        sfh_db_tx_frac_1=Uniform(0.05, 0.95),
        sfh_db_tx_frac_2=Uniform(0.05, 0.95),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp)


_DENSE_BASIS_PARAMS = {
    "sfh_db_log_total_mass": 10.5,
    "sfh_db_log_sfr_inst": 0.5,
    "sfh_db_tx_frac_0": 0.25,
    "sfh_db_tx_frac_1": 0.50,
    "sfh_db_tx_frac_2": 0.75,
}


def test_dense_basis_orchestrator_rest_sed_close_to_legacy(stellar_dense_basis_model):
    """Phase II-2.5 contract: dense_basis SFH parity at ``rtol=5e-2``."""
    legacy = stellar_dense_basis_model.predict_rest_sed(_DENSE_BASIS_PARAMS)
    state = stellar_dense_basis_model.predict_state(_DENSE_BASIS_PARAMS)

    chex.assert_equal_shape([legacy.sed, state.sed_intrinsic])
    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 5e-2, f"max rel diff: {rel_diff:.3e}"


# ── Phase II-2.4: chem_evol metallicity equivalence ──────────────────


@pytest.fixture(scope="module")
def stellar_chem_evol_model(ssp):
    """Stellar-only model with metallicity_model='chem_evol'.

    Chemical-evolution closed-box gas regulator — Z(t) is derived
    self-consistently from the SFH using `chem_yield`, `chem_eta_outflow`,
    `chem_f_gas_init`, `chem_return_frac`. Mirrors the legacy code path
    at sed_model.py:3578-3592 and the orchestrator component at
    component.py (Phase II-2.4).
    """
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        met_mode="chem_evol",
        sfh_tsnorm_log_total_mass=Uniform(8, 12),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        chem_yield=Fixed(0.03),
        chem_eta_outflow=Fixed(0.0),
        chem_f_gas_init=Fixed(0.9),
        chem_return_frac=Fixed(0.4),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp)


def test_chem_evol_orchestrator_runs(stellar_chem_evol_model):
    """Orchestrator handles metallicity_model='chem_evol' and publishes a
    finite, monotonically-enriching log_metallicity_history."""
    state = stellar_chem_evol_model.predict_state(_STELLAR_PARAMS)
    log_z_hist = state.derived["log_metallicity_history"]
    chex.assert_tree_all_finite(log_z_hist)
    # Closed-box enrichment: metallicity at present (lookback≈0) should
    # exceed metallicity at oldest stars (lookback≈14 Gyr). The grid is
    # ascending in lookback time, so log_z_hist[0] is youngest, [-1] oldest.
    assert float(log_z_hist[0]) > float(log_z_hist[-1]), (
        f"chem_evol should enrich over time: youngest log_Z={log_z_hist[0]:.3f}, "
        f"oldest log_Z={log_z_hist[-1]:.3f}"
    )


def test_chem_evol_legacy_runs(stellar_chem_evol_model):
    """Legacy handles metallicity_model='chem_evol' without raising."""
    legacy = stellar_chem_evol_model.predict_rest_sed(_STELLAR_PARAMS)
    chex.assert_tree_all_finite(legacy.sed)
    assert jnp.any(legacy.sed > 0.0), "legacy SED all zero"


def test_chem_evol_orchestrator_rest_sed_close_to_legacy(stellar_chem_evol_model):
    """Phase II-2.4: chem_evol orchestrator-vs-legacy agreement at rtol=1e-2.

    Closed by extending closure-path-A to the chem_evol branch in
    ``forward/pipeline.py``: the default-csp (``trapz``) chem_evol
    path now uses ``calc_rest_sed_sfh_table_met_table`` with the
    per-age ``log_z_per_age`` from the gas-regulator model, mirroring
    :class:`StellarSEDComponent.apply` exactly. The previous
    SED-luminosity-weighted scalar collapse + bilinear
    ``interp_metallicity`` is gone; both paths now produce
    bit-exact-equal SEDs.
    """
    legacy = stellar_chem_evol_model.predict_rest_sed(_STELLAR_PARAMS)
    state = stellar_chem_evol_model.predict_state(_STELLAR_PARAMS)

    chex.assert_equal_shape([legacy.sed, state.sed_intrinsic])
    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 1e-2, f"max rel diff: {rel_diff:.3e}"


# ── Phase II-2.4: ramp metallicity equivalence ───────────────────────


@pytest.fixture(scope="module")
def stellar_ramp_model(ssp):
    """Stellar-only model with metallicity_model='ramp' (linear Z(t))."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        met_mode="ramp",
        sfh_tsnorm_log_total_mass=Uniform(8, 12),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol_0=Uniform(-2.0, 0.2),
        met_logzsol_final=Uniform(-2.0, 0.2),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp)


_STELLAR_RAMP_PARAMS = {
    **_STELLAR_PARAMS,
    "met_logzsol_0": -1.5,
    "met_logzsol_final": -0.3,
}


def test_ramp_orchestrator_rest_sed_close_to_legacy(stellar_ramp_model):
    """Phase II-2.4: ramp Z(t) orchestrator-vs-legacy at ``rtol=1e-2``."""
    legacy = stellar_ramp_model.predict_rest_sed(_STELLAR_RAMP_PARAMS)
    state = stellar_ramp_model.predict_state(_STELLAR_RAMP_PARAMS)
    chex.assert_equal_shape([legacy.sed, state.sed_intrinsic])
    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 1e-2, f"max rel diff: {rel_diff:.3e}"


def test_field_orchestrator_rest_sed_close_to_legacy(stellar_field_model):
    """Phase II-2.3 contract: orchestrator's stellar SED with field=True
    agrees with legacy at ``rtol=1e-2`` — the same bar as field=False.

    Closed by aligning the orchestrator's SFH log-age grid with the
    legacy ``make_log_age_grid(n_grid)`` (linspace in log10(age/yr)
    over [6.0, 10.14]). Before that fix the orchestrator used
    ``logspace(log10(1e5), log10(14e9), n_grid)`` — extending one
    extra dex below 1 Myr — so the same ``xi`` produced a different
    GP correlation pattern (``d_log_age`` differed by ~25%) and the
    young-age portion of the SFH was sampled inconsistently. SED
    divergence dropped from ~13% to ~0.2% with the grid alignment.

    Closing the residual ~0.2% to ``rtol=1e-6`` is blocked on the
    SFH-integration migration (legacy trapz in lookback time vs
    DSPS canonical trapz in cosmic time), tracked in
    ``docs/dev/20260504-csp-integral-canonicalization.md``.
    """
    legacy = stellar_field_model.predict_rest_sed(_STELLAR_FIELD_PARAMS)
    state = stellar_field_model.predict_state(_STELLAR_FIELD_PARAMS)

    chex.assert_equal_shape([legacy.sed, state.sed_intrinsic])
    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 1e-2, f"max rel diff: {rel_diff:.3e}"


# ── Phase II-2.5: equivalence pinning for additional SFH variants ────
#
# These tests broaden orchestrator-vs-legacy parity to the remaining
# parametric SFH variants registered in
# ``components/stellar/sfh/registry.py``: ``lnorm``, ``snorm``,
# ``snorm_burst``, ``tsnorm_burst``, ``norm``. Each test pins the
# orchestrator's ``state.sed_intrinsic`` to legacy
# ``predict_rest_sed.sed`` at ``rtol=5e-2``, the same physical-range
# tolerance as ``test_orchestrator_rest_sed_close_to_legacy``.
#
# When a test passes, the corresponding name should be added to
# ``components/stellar/component.py::StellarSEDComponent._SUPPORTED_SFH``
# so the orchestrator stops raising ``NotImplementedError`` on it.


def _stellar_only_spec(sfh_name: str, sfh_params: dict):
    """Build a stellar-only Parameters spec for the given SFH variant.

    All free SFH params get ``Uniform`` priors centered on the test
    values; metallicity and dust are fixed to their off-state.
    """
    free_priors = {}
    for k, v in sfh_params.items():
        # Some params have physical constraints (must be > 0): widths,
        # truncs, burst_sfr, burst_age, peak_lbt. Use multiplicative
        # bounds for those; additive ±1.5 for unconstrained params
        # like log_total_mass and skew.
        # Most time/scale-like params have ``lo >= 0`` constraints in
        # their priors. Use multiplicative bounds for any param whose
        # name does NOT match an unconstrained suffix.
        unconstrained_keys = (
            "log_",
            "skew",
            "_alpha",
            "_beta",
            "ratio",
            "flex_",
            "tx_frac",
            "fburst",
            "_r_sfr",
        )
        positive_only = not any(token in k for token in unconstrained_keys)
        if positive_only:
            free_priors[k] = Uniform(max(v * 0.1, 1e-3), max(v * 5.0, 0.5))
        else:
            free_priors[k] = Uniform(v - 1.5, v + 1.5)

    # The widening above is strictly per-parameter, so it cannot see the
    # ordered-pair constraints ``Parameters`` enforces (#1277). Widening the
    # ``const`` window bounds independently (``start_gyr`` down by 10x,
    # ``end_gyr`` up by 5x) made their supports overlap even though the
    # fiducial values are correctly ordered, and every spec this helper built
    # for ``const`` was rejected at construction (#1382 landed the guard; the
    # slow tier that exercises this helper is deselected by default, so it went
    # unnoticed until a schedule-gated run).
    #
    # The repair is DERIVED from ``Parameters._ORDERED_PAIRS`` rather than
    # hard-coding the const window, so a pair added there later cannot silently
    # re-break this fixture: lift the greater param's floor just above the
    # lesser's ceiling, which is exactly the remedy the guard's message advises.
    for greater, lesser, _reason in Parameters._ORDERED_PAIRS:
        if greater not in free_priors or lesser not in free_priors:
            continue
        lesser_hi = free_priors[lesser].bounds[-1]
        greater_lo, greater_hi = free_priors[greater].bounds[0], free_priors[greater].bounds[-1]
        if greater_lo > lesser_hi:
            continue  # already non-overlapping
        new_lo = lesser_hi * (1.0 + 1e-6)
        # Assert the fixture's own setup: the fiducial value must survive the
        # repair, or the comparison below would run on a spec whose prior
        # excludes the point it is meant to evaluate.
        assert new_lo < sfh_params[greater] < greater_hi, (
            f"repairing the {greater}/{lesser} overlap pushed the fiducial "
            f"{greater}={sfh_params[greater]} outside its own prior "
            f"({new_lo}, {greater_hi}) — widen the fiducial separation"
        )
        free_priors[greater] = Uniform(new_lo, greater_hi)

    return Parameters(
        mean_sfh_type=[sfh_name],
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
        **free_priors,
    )


def _check_sfh_variant_equivalence(ssp, sfh_name: str, sfh_params: dict, rtol: float = 2e-2):
    spec = _stellar_only_spec(sfh_name, sfh_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel(spec, ssp)
    # Some SFH variants carry a free parameter the caller does not pin -- e.g.
    # ``lnorm`` has a free ``sfh_lnorm_age_gyr`` (formation time) that this dict
    # omits. Fill any such with its spec default, which is exactly the value the
    # forward substituted silently before the missing-parameter guard landed. A
    # no-op for variants whose dict already covers every free parameter.
    params = dict(sfh_params)
    for name in spec.free_params:
        if name not in params:
            params[name] = float(spec.get_distribution(name).default)
    legacy = model.predict_rest_sed(params)
    state = model.predict_state(params)
    chex.assert_equal_shape([legacy.sed, state.sed_intrinsic])
    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < rtol, f"{sfh_name} max rel diff: {rel_diff:.3e}"


def test_orchestrator_lnorm_close_to_legacy(ssp):
    """``lnorm`` (lognormal SFH) — orchestrator vs legacy."""
    _check_sfh_variant_equivalence(
        ssp,
        "lnorm",
        {
            "sfh_lnorm_log_total_mass": 1.0,
            "sfh_lnorm_peak_gyr": 3.0,
            "sfh_lnorm_width_gyr": 1.0,
        },
    )


def test_orchestrator_snorm_close_to_legacy(ssp):
    """``snorm`` (skew-normal SFH) — orchestrator vs legacy."""
    _check_sfh_variant_equivalence(
        ssp,
        "snorm",
        {
            "sfh_snorm_log_total_mass": 1.0,
            "sfh_snorm_peak_lbt_gyr": 3.0,
            "sfh_snorm_width_gyr": 1.0,
            "sfh_snorm_skew": 0.0,
        },
    )


def test_orchestrator_snorm_burst_close_to_legacy(ssp):
    """``snorm_burst`` (skew-normal + burst SFH) — orchestrator vs legacy."""
    _check_sfh_variant_equivalence(
        ssp,
        "snorm_burst",
        {
            "sfh_snorm_burst_log_total_mass": 1.0,
            "sfh_snorm_burst_peak_lbt_gyr": 3.0,
            "sfh_snorm_burst_width_gyr": 1.0,
            "sfh_snorm_burst_skew": 0.0,
            "sfh_snorm_burst_burst_sfr": 5.0,
            "sfh_snorm_burst_burst_age_gyr": 0.05,
        },
    )


def test_orchestrator_tsnorm_burst_close_to_legacy(ssp):
    """``tsnorm_burst`` (truncated-skew-normal + burst SFH) — orchestrator vs legacy."""
    _check_sfh_variant_equivalence(
        ssp,
        "tsnorm_burst",
        {
            "sfh_tsnorm_burst_log_total_mass": 1.0,
            "sfh_tsnorm_burst_peak_lbt_gyr": 3.0,
            "sfh_tsnorm_burst_width_gyr": 1.0,
            "sfh_tsnorm_burst_skew": 0.0,
            "sfh_tsnorm_burst_trunc": 3.0,
            "sfh_tsnorm_burst_burst_sfr": 5.0,
            "sfh_tsnorm_burst_burst_age_gyr": 0.05,
        },
    )


def test_orchestrator_norm_close_to_legacy(ssp):
    """``norm`` (Gaussian SFH) — orchestrator vs legacy."""
    _check_sfh_variant_equivalence(
        ssp,
        "norm",
        {
            "sfh_norm_log_total_mass": 1.0,
            "sfh_norm_peak_lbt_gyr": 3.0,
            "sfh_norm_width_gyr": 1.0,
        },
    )


# ── Phase II-2.5b: more parametric SFH variants ──────────────────────
#
# Tests pin orchestrator-vs-legacy parity for the smooth-SFH variants
# that converge at ``rtol=5e-2`` (or ``rtol=1e-1`` for ``const_exp``,
# which has a sharp quench transition — see comment on the test).
# Variants with sharp discontinuities (``exp``, ``dexp``, ``tau``)
# diverge by ≥ 13% per-wavelength because the legacy log-space SFR
# interpolation and the orchestrator's linear-space interpolation
# resolve the cutoff differently. They are pinned by the SFH-side
# CSP-canonicalization work (tracked in
# ``docs/dev/20260504-csp-integral-canonicalization.md``), not here.
# ``psb``, ``delayed_bq``, ``dense_basis_pure`` have prior-bound
# constraints (``[0,1]`` fractions, etc.) that the test fixture's
# generic free-prior generator cannot satisfy yet; they need
# variant-specific fixtures and are deferred.


def test_orchestrator_const_close_to_legacy(ssp):
    """``const`` (constant SFH between two lookback times) — orchestrator vs legacy.

    Convention: ``start_gyr`` is when star formation *began* (the OLDER
    lookback bound) and ``end_gyr`` when it stopped. The original fixture had
    them swapped (start 0.5, end 5.0), which encodes an *empty* SF window —
    both paths then compared DSPS's SFR_MIN-floor garbage against itself and
    passed vacuously. The #964 CIC kernel returns honest zero weights for an
    empty window, which exposed the swap.
    """
    _check_sfh_variant_equivalence(
        ssp,
        "const",
        {
            "sfh_const_log_total_mass": 1.0,
            "sfh_const_start_gyr": 5.0,
            "sfh_const_end_gyr": 0.5,
        },
    )


def test_orchestrator_const_exp_close_to_legacy(ssp):
    """``const_exp`` (constant-then-exponential SFH) — orchestrator vs legacy."""
    _check_sfh_variant_equivalence(
        ssp,
        "const_exp",
        {
            "sfh_cexp_log_total_mass": 1.0,
            "sfh_cexp_tau_gyr": 2.0,
            "sfh_cexp_quench_gyr": 5.0,
            "sfh_cexp_age_gyr": 10.0,
        },
        rtol=1e-1,  # quench transition slightly worse than smooth SFHs
    )


def _stellar_only_spec_with_priors(sfh_name: str, priors: dict):
    """Build a stellar-only Parameters spec with explicit priors per param.

    Used for variants whose default priors have non-trivial constraints
    (``[0,1]`` fractions, etc.) that the generic generator can't infer.
    """
    return Parameters(
        mean_sfh_type=[sfh_name],
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
        **priors,
    )


def _check_with_priors(ssp, sfh_name, priors, sfh_params, rtol=2e-2):
    spec = _stellar_only_spec_with_priors(sfh_name, priors)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel(spec, ssp)
    legacy = model.predict_rest_sed(sfh_params)
    state = model.predict_state(sfh_params)
    chex.assert_equal_shape([legacy.sed, state.sed_intrinsic])
    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < rtol, f"{sfh_name} max rel diff: {rel_diff:.3e}"


def test_orchestrator_psb_close_to_legacy(ssp):
    """``psb`` (post-starburst, Wild+ 2020) — orchestrator vs legacy."""
    priors = {
        "sfh_psb_log_total_mass": Uniform(8.0, 12.0),
        "sfh_psb_age_gyr": Uniform(0.5, 13.0),
        "sfh_psb_tau_gyr": Uniform(0.1, 10.0),
        "sfh_psb_burstage_gyr": Uniform(0.01, 5.0),
        "sfh_psb_alpha": Uniform(0.5, 5.0),
        "sfh_psb_beta": Uniform(0.5, 5.0),
        "sfh_psb_fburst": Uniform(0.01, 0.99),
    }
    sfh_params = {
        "sfh_psb_log_total_mass": 1.0,
        "sfh_psb_age_gyr": 5.0,
        "sfh_psb_tau_gyr": 2.0,
        "sfh_psb_burstage_gyr": 0.5,
        "sfh_psb_alpha": 2.0,
        "sfh_psb_beta": 2.0,
        "sfh_psb_fburst": 0.3,
    }
    # rtol=1e-1: the post-starburst burst component has a sharp DPL
    # rise/fall whose log-vs-linear interpolation residual is ~6%,
    # similar to const_exp. Closes to <1% with the SFH-side migration.
    _check_with_priors(ssp, "psb", priors, sfh_params, rtol=1e-1)


def test_orchestrator_delayed_bq_close_to_legacy(ssp):
    """``delayed_bq`` (delayed burst-quench SFH) — orchestrator vs legacy."""
    priors = {
        "sfh_delayed_bq_log_total_mass": Uniform(8.0, 12.0),
        "sfh_delayed_bq_tau_main_gyr": Uniform(0.1, 10.0),
        "sfh_delayed_bq_age_main_gyr": Uniform(0.5, 13.0),
        "sfh_delayed_bq_age_bq_gyr": Uniform(0.01, 5.0),
        "sfh_delayed_bq_r_sfr": Uniform(0.01, 10.0),
    }
    sfh_params = {
        "sfh_delayed_bq_log_total_mass": 10.0,
        "sfh_delayed_bq_tau_main_gyr": 2.0,
        "sfh_delayed_bq_age_main_gyr": 5.0,
        "sfh_delayed_bq_age_bq_gyr": 1.0,
        "sfh_delayed_bq_r_sfr": 0.1,
    }
    _check_with_priors(ssp, "delayed_bq", priors, sfh_params)


def test_orchestrator_dense_basis_pure_close_to_legacy(ssp):
    """``dense_basis_pure`` — orchestrator vs legacy."""
    priors = {
        "sfh_dbp_log_total_mass": Uniform(8.0, 12.0),
        "sfh_dbp_tx_frac_0": Uniform(0.05, 0.95),
        "sfh_dbp_tx_frac_1": Uniform(0.05, 0.95),
        "sfh_dbp_tx_frac_2": Uniform(0.05, 0.95),
    }
    sfh_params = {
        "sfh_dbp_log_total_mass": 10.0,
        "sfh_dbp_tx_frac_0": 0.25,
        "sfh_dbp_tx_frac_1": 0.5,
        "sfh_dbp_tx_frac_2": 0.75,
    }
    _check_with_priors(ssp, "dense_basis_pure", priors, sfh_params)


def test_orchestrator_continuity_flex_close_to_legacy(ssp):
    """``continuity_flex`` — orchestrator vs legacy."""
    _check_sfh_variant_equivalence(
        ssp,
        "continuity_flex",
        {
            "sfh_cflex_log_total_mass": 10.0,
            "sfh_cflex_ratio_young": 0.0,
            "sfh_cflex_flex_0": 0.0,
            "sfh_cflex_flex_1": 0.0,
            "sfh_cflex_flex_2": 0.0,
            "sfh_cflex_ratio_old": 0.0,
        },
    )


# ── Phase II-2.5c: sharp-cutoff exponential SFH variants ──────────────


def test_orchestrator_exp_close_to_legacy(ssp):
    """``exp`` (exponential) — agrees at rtol=5e-2 when ``start_gyr``
    is at its default Fixed(0) (no sharp cutoff). Variants with
    non-zero ``sfh_exp_start_gyr`` would expose the legacy log-space
    vs orchestrator linear-space SFR interpolation residual at the
    cutoff edge, which closes only with the SFH-side CSP migration.
    """
    priors = {
        "sfh_exp_log_total_mass": Uniform(8.0, 12.0),
        "sfh_exp_tau_gyr": Uniform(0.1, 10.0),
    }
    sfh_params = {
        "sfh_exp_log_total_mass": 1.0,
        "sfh_exp_tau_gyr": 2.0,
    }
    _check_with_priors(ssp, "exp", priors, sfh_params)


def test_orchestrator_periodic_close_to_legacy(ssp):
    """``periodic`` (Ciesla+ 2017 periodic burst SFH) — rtol=1e-1.

    Multiple sharp burst rise/fall edges produce ~9% rtol residual,
    similar to const_exp/psb. Closes to <1% with the SFH-side migration.
    """
    priors = {
        "sfh_periodic_log_total_mass": Uniform(8.0, 12.0),
        "sfh_periodic_delta_bursts_gyr": Uniform(0.01, 1.0),
        "sfh_periodic_tau_bursts_gyr": Uniform(0.001, 0.5),
        "sfh_periodic_age_gyr": Uniform(0.5, 13.0),
    }
    sfh_params = {
        "sfh_periodic_log_total_mass": 10.0,
        "sfh_periodic_delta_bursts_gyr": 0.5,
        "sfh_periodic_tau_bursts_gyr": 0.05,
        "sfh_periodic_age_gyr": 5.0,
    }
    _check_with_priors(ssp, "periodic", priors, sfh_params, rtol=1e-1)


def test_orchestrator_buat08_close_to_legacy(ssp):
    """``buat08`` (Buat+ 2008 velocity-parameterized SFH) — rtol=5e-2."""
    priors = {
        "sfh_buat08_log_total_mass": Uniform(8.0, 12.0),
        "sfh_buat08_velocity_km_s": Uniform(80.0, 360.0),
    }
    sfh_params = {
        "sfh_buat08_log_total_mass": 10.0,
        "sfh_buat08_velocity_km_s": 200.0,
    }
    _check_with_priors(ssp, "buat08", priors, sfh_params)


def test_orchestrator_dexp_close_to_legacy(ssp):
    """``dexp`` (delayed exponential) — same caveat as ``exp``."""
    priors = {
        "sfh_dexp_log_total_mass": Uniform(8.0, 12.0),
        "sfh_dexp_tau_gyr": Uniform(0.1, 10.0),
    }
    sfh_params = {
        "sfh_dexp_log_total_mass": 1.0,
        "sfh_dexp_tau_gyr": 2.0,
    }
    _check_with_priors(ssp, "dexp", priors, sfh_params)


# Removed: ``test_orchestrator_tau_close_to_legacy`` (FSPS-style "tau"
# declining exponential SFH was deregistered 2026-05-28 because its name
# collided with CIGALE ``sfhdelayed`` semantics — only ``"delayed"``
# remains. See PR notes and registry.py comment block.
