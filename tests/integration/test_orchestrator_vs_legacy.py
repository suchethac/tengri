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
    "sfh_tsnorm_log_peak_sfr": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 2.0,
    "sfh_tsnorm_width_gyr": 1.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 3.0,
    # Orchestrator path expects ALL parameters (free + fixed) in
    # the params dict. Legacy path reads fixed values from spec.
    "met_logzsol": -0.5,
    "redshift": 0.05,
    "dust_tau_bc": 0.0,
    "dust_tau_diff": 0.0,
    "dust_slope": 0.0,
    "dust_T": 35.0,
    "dust_beta_ir": 1.6,
}


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


@pytest.mark.xfail(
    reason=(
        "Strict orchestrator-vs-legacy bit-exact parity is the gating "
        "criterion for monolith deletion (Phase II-2 success bar). "
        "Current divergence sources: (1) DSPS joint vs separable mass "
        "factorization (legacy still on separable path), (2) interface "
        "mismatch in fixed-param injection. Closing this xfail is the "
        "blocking work item for the deletion phase."
    ),
    strict=True,
)
def test_orchestrator_rest_sed_bit_exact_to_legacy(stellar_only_model):
    """Orchestrator's stellar SED must equal legacy's predict_rest_sed.sed
    at rtol=1e-6 for the gating-criterion sign-off."""
    legacy = stellar_only_model.predict_rest_sed(_STELLAR_PARAMS)
    state = stellar_only_model.predict_via_orchestrator(_STELLAR_PARAMS)

    if legacy.sed.shape != state.sed_intrinsic.shape:
        pytest.fail(
            f"Shape mismatch — legacy {legacy.sed.shape} vs orch {state.sed_intrinsic.shape}"
        )

    rel_diff = float(
        jnp.max(
            jnp.abs(legacy.sed - state.sed_intrinsic) / jnp.maximum(jnp.abs(legacy.sed), 1e-30)
        )
    )
    assert rel_diff < 1e-6, f"max rel diff: {rel_diff:.3e}"
