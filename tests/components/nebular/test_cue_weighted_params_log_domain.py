# SPDX-License-Identifier: BSD-3-Clause
"""Contract: CueBackend._compute_weighted_cue_params uses log-domain arithmetic to prevent
float32 overflow (#1206).

The old implementation overflows in pure float32 when logqion_all ~ 44-52, causing
10**logqion_all -> inf, which then gets zeroed out by the isfinite guard. This is a
fail-open silent error that must be prevented by using log-domain logsumexp + per-segment
max-offset scaling.

This test suite:
1. Validates f64 exactness vs frozen old arithmetic (rtol 1e-12)
2. Validates pure-f32 correctness (must rebuild backend inside jax.enable_x64(False))
3. Tests degenerate population behavior
4. Validates gradient non-zero and smooth via existing test imports
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import load_ssp_data
from tengri.components.nebular.cue import _MAX_NEB_LOG_AGE, CueBackend
from tengri.components.nebular.ionizing_spectrum import (
    interpolate_ionizing_params,
    interpolate_ionizing_seglum,
)

pytestmark = pytest.mark.regression_bug

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_CUE_WEIGHTS = "data/cue_weights.npz"


def _load_ssp():
    if not Path(_BARE).is_file():
        pytest.skip(f"missing bare SSP {_BARE}")
    return load_ssp_data(_BARE)


def _frozen_weighted_combine(logqion_all, log_seglum_all, ionspec_all, ssp_weights, young_mask):
    """FROZEN pre-log_nion Cue combine (#1206).

    This is the exact old linear-domain arithmetic from cue.py lines 1401-1463
    (commit before the log-domain reformulation). Used as the ground truth for
    f64 exactness tests.
    """
    # Q_H per bin, masked to young bins with positive weights.
    # #1001 defense
    qh_per_bin = 10.0**logqion_all  # (n_age,)
    qh_per_bin = jnp.where(jnp.isfinite(qh_per_bin), qh_per_bin, 0.0)
    weighted_qh = ssp_weights * qh_per_bin  # (n_age,)
    weighted_qh = jnp.where(young_mask & (ssp_weights > 0), weighted_qh, 0.0)

    total_qh = jnp.sum(weighted_qh)
    total_logqion = jnp.where(total_qh > 0, jnp.log10(total_qh), -99.0)

    # Effective ionizing-spectrum shape (#1018)
    seg_per_bin = 10.0**log_seglum_all
    seg_per_bin = jnp.where(jnp.isfinite(seg_per_bin), seg_per_bin, 0.0)

    w_mass = jnp.where(young_mask & (ssp_weights > 0), ssp_weights, 0.0)  # (n_age,)
    seg_w = w_mass[:, None] * seg_per_bin  # (n_age, 4)
    seg_tot = jnp.sum(seg_w, axis=0)  # (4,)
    seg_safe = jnp.maximum(seg_tot, 1e-300)

    alpha_eff = jnp.sum(seg_w * ionspec_all[:, :4], axis=0) / seg_safe  # (4,)
    logLratio_eff = jnp.diff(jnp.log10(seg_safe))  # (3,)
    i7_weighted = jnp.concatenate([alpha_eff, logLratio_eff])

    # Fully degenerate case fallback
    i7 = jnp.where(total_qh > 0, i7_weighted, ionspec_all[jnp.argmax(weighted_qh)])

    return total_logqion, i7


def test_f64_exactness_vs_frozen_arithmetic():
    """New log-domain method must match frozen old arithmetic at rtol=1e-12 in f64."""
    ssp = _load_ssp()
    be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)

    # Test across 3 metallicities and 2 SFH weightings
    metallicities = [-3.5, -2.0, -1.0]  # log10(Z) absolute
    sfh_names = ["young_burst", "older_population"]

    for met_abs in metallicities:
        for sfh_label in sfh_names:
            # Create a test SFH weighting
            log_ages = (
                jnp.log10(be._ssp_ages_yr)
                if hasattr(be, "_ssp_ages_yr")
                else np.log10(np.linspace(1e6, 1.3e10, 30))
            )
            n_age = len(log_ages) if hasattr(log_ages, "__len__") else 30

            if sfh_label == "young_burst":
                # Concentrated at young ages (log_age ~ 7-8)
                ssp_weights = jnp.where(
                    jnp.log10(np.arange(1, n_age + 1)) <= 8.0, 1.0 / n_age, 0.0
                )
            else:
                # More distributed, still some young
                ssp_weights = jnp.exp(-np.abs(np.arange(n_age) - n_age / 3) / 5.0)

            ssp_weights = ssp_weights / jnp.sum(ssp_weights)  # normalize
            ssp_log_ages = np.log10(np.linspace(1e6, 1.3e10, n_age))

            # Get interpolated parameters for this population
            young_mask = ssp_log_ages <= _MAX_NEB_LOG_AGE
            ionspec_all, logqion_all = jax.vmap(
                lambda log_age_yr, met_abs=met_abs: interpolate_ionizing_params(
                    be._ionspec_table,
                    be._logqion_table,
                    be._ssp_lgmet,
                    be._ssp_log_age_yr,
                    met_abs,
                    log_age_yr,
                )
            )(jnp.asarray(ssp_log_ages))

            log_seglum_all = jax.vmap(
                lambda log_age_yr, met_abs=met_abs: interpolate_ionizing_seglum(
                    be._seglum_table,
                    be._ssp_lgmet,
                    be._ssp_log_age_yr,
                    met_abs,
                    log_age_yr,
                )
            )(jnp.asarray(ssp_log_ages))

            # Check precondition: young population has all 4 segments populated
            seg_per_bin_frozen = 10.0**log_seglum_all
            seg_per_bin_frozen = jnp.where(
                jnp.isfinite(seg_per_bin_frozen), seg_per_bin_frozen, 0.0
            )
            w_mass_frozen = jnp.where(young_mask & (ssp_weights > 0), ssp_weights, 0.0)
            seg_w_frozen = w_mass_frozen[:, None] * seg_per_bin_frozen
            seg_tot_frozen = jnp.sum(seg_w_frozen, axis=0)
            assert jnp.all(seg_tot_frozen > 0), (
                f"precondition failed: not all 4 segments populated at met={met_abs}, "
                f"sfh={sfh_label}. segments: {seg_tot_frozen}"
            )

            # Get the frozen (old) result
            total_logqion_frozen, i7_frozen = _frozen_weighted_combine(
                logqion_all, log_seglum_all, ionspec_all, ssp_weights, young_mask
            )

            # Get the new method's result
            result = be._compute_weighted_cue_params(ssp_weights, ssp_log_ages, met_abs)
            gas_logqion_new = result["gas_logqion"]
            i7_new = jnp.array(
                [
                    result["ionspec_index1"],
                    result["ionspec_index2"],
                    result["ionspec_index3"],
                    result["ionspec_index4"],
                    result["ionspec_logLratio1"],
                    result["ionspec_logLratio2"],
                    result["ionspec_logLratio3"],
                ]
            )

            # Assert match at rtol=1e-12
            assert jnp.allclose(gas_logqion_new, total_logqion_frozen, rtol=1e-12, atol=1e-12), (
                f"gas_logqion mismatch at met={met_abs}, sfh={sfh_label}: "
                f"new={gas_logqion_new}, frozen={total_logqion_frozen}"
            )
            assert jnp.allclose(i7_new, i7_frozen, rtol=1e-12, atol=1e-12), (
                f"ionspec params mismatch at met={met_abs}, sfh={sfh_label}: "
                f"new={i7_new}, frozen={i7_frozen}"
            )


def test_pure_float32_correctness():
    """Pure f32 must be finite and match f64 reference (anti-silent-zeroing gate).

    This test MUST rebuild the CueBackend inside jax.enable_x64(False) context,
    or the f64 tables will mask the overflow. It MUST assert dtype == float32.
    """
    ssp = _load_ssp()

    # Compute f64 reference OUTSIDE any context
    be64 = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
    n_age = 20
    ssp_weights_f64 = jnp.ones(n_age) / n_age
    ssp_log_ages = np.log10(np.linspace(1e6, 1.3e10, n_age))
    log_z = -2.0  # absolute

    result_f64 = be64._compute_weighted_cue_params(ssp_weights_f64, ssp_log_ages, log_z)
    gas_logqion_f64 = result_f64["gas_logqion"]
    i7_f64 = jnp.array(
        [
            result_f64["ionspec_index1"],
            result_f64["ionspec_index2"],
            result_f64["ionspec_index3"],
            result_f64["ionspec_index4"],
            result_f64["ionspec_logLratio1"],
            result_f64["ionspec_logLratio2"],
            result_f64["ionspec_logLratio3"],
        ]
    )

    # NOW rebuild inside f32 context
    with jax.enable_x64(False):
        be32 = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
        # CRITICAL: assert the backend was rebuilt in f32
        assert be32._logqion_table.dtype == jnp.float32, (
            f"CueBackend was NOT rebuilt in f32 context: dtype={be32._logqion_table.dtype}"
        )

        # Convert inputs to f32
        ssp_weights_f32 = jnp.asarray(ssp_weights_f64, dtype=jnp.float32)
        ssp_log_ages_f32 = jnp.asarray(ssp_log_ages, dtype=jnp.float32)

        result_f32 = be32._compute_weighted_cue_params(ssp_weights_f32, ssp_log_ages_f32, log_z)
        gas_logqion_f32 = result_f32["gas_logqion"]
        i7_f32 = jnp.array(
            [
                result_f32["ionspec_index1"],
                result_f32["ionspec_index2"],
                result_f32["ionspec_index3"],
                result_f32["ionspec_index4"],
                result_f32["ionspec_logLratio1"],
                result_f32["ionspec_logLratio2"],
                result_f32["ionspec_logLratio3"],
            ]
        )

    # Assert f32 is finite (not the old bug's -99.0 silencing)
    assert jnp.all(jnp.isfinite(gas_logqion_f32)), (
        f"f32 gas_logqion is non-finite: {gas_logqion_f32}"
    )
    assert jnp.all(jnp.isfinite(i7_f32)), f"f32 ionspec params contain non-finite values: {i7_f32}"

    # Assert f32 matches f64 reference (NOT merely finite)
    # The old bug returns finite -99.0 which is WRONG
    assert jnp.allclose(gas_logqion_f32, gas_logqion_f64, atol=5e-3), (
        f"f32/f64 gas_logqion mismatch: f32={gas_logqion_f32}, f64={gas_logqion_f64}, "
        f"difference={float(gas_logqion_f32 - gas_logqion_f64)}"
    )
    assert jnp.allclose(i7_f32, i7_f64, atol=0.02), (
        f"f32/f64 ionspec mismatch: f32={i7_f32}, f64={i7_f64}"
    )


def test_degenerate_population():
    """All-old or all-zero population must return gas_logqion=-99.0 with finite ionspec shape."""
    ssp = _load_ssp()
    be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)

    n_age = 20
    ssp_log_ages = np.log10(np.linspace(1e6, 1.3e10, n_age))
    young_mask = ssp_log_ages <= _MAX_NEB_LOG_AGE

    # Test 1: All-old population (weights only on old ages)
    ssp_weights_old = jnp.where(~young_mask, 1.0 / jnp.sum(~young_mask), 0.0)
    result_old = be._compute_weighted_cue_params(ssp_weights_old, ssp_log_ages, log_z=-2.0)
    assert result_old["gas_logqion"] == -99.0, (
        f"all-old population should have gas_logqion=-99.0, got {result_old['gas_logqion']}"
    )
    i7_old = jnp.array(
        [
            result_old["ionspec_index1"],
            result_old["ionspec_index2"],
            result_old["ionspec_index3"],
            result_old["ionspec_index4"],
            result_old["ionspec_logLratio1"],
            result_old["ionspec_logLratio2"],
            result_old["ionspec_logLratio3"],
        ]
    )
    assert jnp.all(jnp.isfinite(i7_old)), (
        f"all-old population ionspec should be finite, got {i7_old}"
    )

    # Test 2: All-zero weights
    ssp_weights_zero = jnp.zeros(n_age)
    result_zero = be._compute_weighted_cue_params(ssp_weights_zero, ssp_log_ages, log_z=-2.0)
    assert result_zero["gas_logqion"] == -99.0, (
        f"zero weights should have gas_logqion=-99.0, got {result_zero['gas_logqion']}"
    )
    i7_zero = jnp.array(
        [
            result_zero["ionspec_index1"],
            result_zero["ionspec_index2"],
            result_zero["ionspec_index3"],
            result_zero["ionspec_index4"],
            result_zero["ionspec_logLratio1"],
            result_zero["ionspec_logLratio2"],
            result_zero["ionspec_logLratio3"],
        ]
    )
    assert jnp.all(jnp.isfinite(i7_zero)), f"zero weights ionspec should be finite, got {i7_zero}"


def test_gradient_nonzero_and_smooth():
    """Gradient of gas_logqion w.r.t. ssp_weights must be non-zero and finite."""
    ssp = _load_ssp()
    be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)

    n_age = 20
    ssp_weights = jnp.ones(n_age) / n_age
    ssp_log_ages = np.log10(np.linspace(1e6, 1.3e10, n_age))
    log_z = -2.0

    def gas_logqion_fn(w):
        result = be._compute_weighted_cue_params(w, ssp_log_ages, log_z)
        return result["gas_logqion"]

    # Compute gradient for young population
    g_young = jax.grad(gas_logqion_fn)(ssp_weights)
    assert jnp.all(jnp.isfinite(g_young)), (
        f"gradient w.r.t. ssp_weights has NaN/inf for young population: {g_young}"
    )

    # Gradient should be non-zero for a young population (i.e., SFH changes affect Q_H)
    assert jnp.any(jnp.abs(g_young) > 1e-10), (
        f"gradient w.r.t. ssp_weights is zero for young population: {g_young}. "
        "This indicates SFH has no effect on Q_H, which is wrong."
    )

    # Test degenerate case: gradient must still be finite even if emission is zero
    ssp_log_ages_old = np.log10(np.linspace(1e9, 1.3e10, n_age))
    ssp_weights_old = jnp.ones(n_age) / n_age

    def gas_logqion_fn_old(w):
        result = be._compute_weighted_cue_params(w, ssp_log_ages_old, log_z)
        return result["gas_logqion"]

    g_old = jax.grad(gas_logqion_fn_old)(ssp_weights_old)
    assert jnp.all(jnp.isfinite(g_old)), (
        f"gradient w.r.t. ssp_weights has NaN/inf for degenerate population: {g_old}"
    )
    assert jnp.any(g_old != 0.0), (
        "`g_old` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
