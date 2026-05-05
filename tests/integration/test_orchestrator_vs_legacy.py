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

import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.parameters.priors import Fixed, Uniform

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
        sfh_tsnorm_log_peak_sfr=Uniform(-1, 3),
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
    # predict_via_orchestrator (the latter injects via
    # spec.get_fixed_values()). This is the realistic call site.
    "sfh_tsnorm_log_peak_sfr": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 2.0,
    "sfh_tsnorm_width_gyr": 1.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 3.0,
}


# ── Fixed-param injection: orchestrator now reads from spec ──────────


def test_orchestrator_injects_fixed_values_from_spec(stellar_only_model):
    """Calling predict_via_orchestrator with only the FREE params must
    succeed — fixed values (met_logzsol, redshift, dust_tau_*) come
    from spec.get_fixed_values()."""
    free_only = {
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 2.0,
        "sfh_tsnorm_width_gyr": 1.0,
        "sfh_tsnorm_skew": 0.0,
        "sfh_tsnorm_trunc": 3.0,
    }
    state = stellar_only_model.predict_via_orchestrator(free_only)
    # log_mstar published by stellar component → injection worked end-to-end.
    assert "log_mstar" in state.derived
    assert jnp.isfinite(state.derived["log_mstar"])


def test_orchestrator_explicit_param_overrides_spec_fixed(stellar_only_model):
    """A param passed explicitly must win over the spec's fixed value."""
    free_only = {
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 2.0,
        "sfh_tsnorm_width_gyr": 1.0,
        "sfh_tsnorm_skew": 0.0,
        "sfh_tsnorm_trunc": 3.0,
    }
    state_default = stellar_only_model.predict_via_orchestrator(free_only)
    # Override met_logzsol with a different value than the spec fixed.
    state_overridden = stellar_only_model.predict_via_orchestrator(
        {**free_only, "met_logzsol": 0.0}
    )
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
    state = stellar_only_model.predict_via_orchestrator(_STELLAR_PARAMS)
    assert state.sed_intrinsic.shape[0] > 1000, (
        f"sed_intrinsic shape too small: {state.sed_intrinsic.shape}"
    )
    assert jnp.all(jnp.isfinite(state.sed_intrinsic))
    assert jnp.all(state.sed_intrinsic >= 0)


def test_stellar_only_legacy_runs(stellar_only_model):
    """Legacy path produces finite SED of expected shape."""
    legacy = stellar_only_model.predict_rest_sed(_STELLAR_PARAMS)
    assert legacy.sed.shape[0] > 1000
    assert jnp.all(jnp.isfinite(legacy.sed))
    assert jnp.all(legacy.sed >= 0)


# ── Physical-range agreement: both within order-of-magnitude ──────────


def test_orchestrator_vs_legacy_mstar_physical_agreement(stellar_only_model):
    """log10(M*) from both paths agrees within 0.1 dex."""
    legacy_q = stellar_only_model.predict(_STELLAR_PARAMS)
    state = stellar_only_model.predict_via_orchestrator(_STELLAR_PARAMS)

    legacy_log_mstar = float(jnp.log10(legacy_q.stellar_mass))
    orch_log_mstar = float(state.derived["log_mstar"])
    diff_dex = abs(legacy_log_mstar - orch_log_mstar)

    assert diff_dex < 0.1, (
        f"log10(M*) disagrees by {diff_dex:.3f} dex: "
        f"legacy={legacy_log_mstar:.3f}, orch={orch_log_mstar:.3f}"
    )


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
    state = stellar_only_model.predict_via_orchestrator(_STELLAR_PARAMS)

    assert legacy.sed.shape == state.sed_intrinsic.shape

    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 1e-2, f"max rel diff: {rel_diff:.3e}"


@pytest.mark.xfail(
    reason=(
        "Strict bit-exact (rtol=1e-6) parity requires also migrating "
        "the SFH integration in the legacy trapz path to DSPS canonical "
        "trapezoidal-in-cosmic-time. Tracked as the next milestone in "
        "docs/dev/20260504-csp-integral-canonicalization.md."
    ),
    strict=True,
)
def test_orchestrator_rest_sed_bit_exact_to_legacy(stellar_only_model):
    """Orchestrator's stellar SED must equal legacy's predict_rest_sed.sed
    at rtol=1e-6 for the gating-criterion sign-off."""
    legacy = stellar_only_model.predict_rest_sed(_STELLAR_PARAMS)
    state = stellar_only_model.predict_via_orchestrator(_STELLAR_PARAMS)

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
        sfh_tsnorm_log_peak_sfr=Uniform(-1, 3),
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
    state = stellar_field_model.predict_via_orchestrator(_STELLAR_FIELD_PARAMS)
    sfr_history = state.derived["sfr_history"]
    assert jnp.all(jnp.isfinite(sfr_history)), "sfr_history contains NaN/Inf"
    assert jnp.any(sfr_history > 0.0), "sfr_history is all zero — field branch dead?"


def test_field_legacy_runs(stellar_field_model):
    """Legacy handles field=True without raising and produces finite SFH."""
    sfh = stellar_field_model.predict_sfh(_STELLAR_FIELD_PARAMS)
    assert jnp.all(jnp.isfinite(sfh["sfr_full"])), "legacy sfr_full contains NaN/Inf"
    assert jnp.any(sfh["sfr_full"] > 0.0), "legacy sfr_full all zero — field branch dead?"


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
    state = stellar_field_model.predict_via_orchestrator(_STELLAR_FIELD_PARAMS)

    assert legacy.sed.shape == state.sed_intrinsic.shape

    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 1e-2, f"max rel diff: {rel_diff:.3e}"
