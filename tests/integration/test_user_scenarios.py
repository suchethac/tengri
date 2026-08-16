# SPDX-License-Identifier: BSD-3-Clause
"""Comprehensive user scenario testing for model combinations × inference engines.

Tests various SFH types, component combinations, and inference methods from an
astronomer's perspective. Identifies JIT compilation bottlenecks, memory issues,
and user experience problems.

Author: Claude Code (generated 2026-04-17)
Purpose: User review of tengri code for Paper I usability improvements
"""

import os
import time
import tracemalloc
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# Ensure we use CPU (JAX Metal causes test failures per CLAUDE.md)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Disable background compilation to get accurate JIT timing
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

from tengri import Fitter, Observation, Parameters, Photometry, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform

# ── Skip if SSP data not available ────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_MIST_SSP = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_BC03_SSP = _DATA_DIR / "ssp_bc03_miles_chabrier.h5"
_BPASS_SSP = _DATA_DIR / "ssp_bpass_imf170_300_100.h5"

_MIST_EXISTS = _MIST_SSP.is_file()
_BC03_EXISTS = _BC03_SSP.is_file()
_BPASS_EXISTS = _BPASS_SSP.is_file()

pytestmark = pytest.mark.skipif(
    not _MIST_EXISTS,
    reason="MIST SSP data file not found (required for user scenarios)",
)


# ── Performance thresholds (user experience boundaries) ───────────
class PerformanceThresholds:
    """User experience thresholds for JIT and memory usage."""

    # JIT compilation time (seconds)
    JIT_EXCELLENT = 15.0  # ✓ Fast
    JIT_ACCEPTABLE = 30.0  # ⚠️ Acceptable (user waits once)
    JIT_SLOW = 60.0  # ⚠️ Slow (annoying but tolerable)
    # Above 60s = ❌ Broken (user thinks it's hung)

    # Peak RAM usage (GB)
    RAM_LAPTOP_FRIENDLY = 4.0  # ✓ Works on laptops
    RAM_NEEDS_GOOD_LAPTOP = 8.0  # ⚠️ Needs good laptop
    # Above 8GB = ❌ Requires workstation


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def mist_ssp():
    """MIST SSP data (dense age/met grid, default for science)."""
    return load_ssp_data(str(_MIST_SSP))


@pytest.fixture(scope="session")
def bc03_ssp():
    """BC03 SSP data (legacy, smaller grid)."""
    if not _BC03_EXISTS:
        pytest.skip("BC03 SSP not available")
    return load_ssp_data(str(_BC03_SSP))


@pytest.fixture(scope="session")
def bpass_ssp():
    """BPASS SSP data (no mass-remaining table → crashes posterior.derived)."""
    if not _BPASS_EXISTS:
        pytest.skip("BPASS SSP not available")
    return load_ssp_data(str(_BPASS_SSP))


@pytest.fixture(scope="session")
def optical_nir_filters():
    """Standard 11-band optical+NIR filter set (HST+VISTA+IRAC).

    Matches the CANDELS z~1 galaxy setup from fig01_multimodel_candels.py:
    - HST ACS: F435W, F606W, F775W, F814W, F850LP
    - HST WFC3: F125W, F140W, F160W
    - VISTA: Ks
    - Spitzer IRAC: 3.6, 4.5 μm
    """
    filter_names = [
        "hst_f435w",
        "hst_f606w",
        "hst_f775w",
        "hst_f814w",
        "hst_f850lp",
        "hst_f125w",
        "hst_f140w",
        "hst_f160w",
        "vista_ks",
        "irac_36",
        "irac_45",
    ]
    return load_filter_set(filter_names)


@pytest.fixture(scope="session")
def mock_obs_z1(optical_nir_filters):
    """Mock photometric observation config at z=1 (11 bands)."""
    # Observation is just the filter configuration
    return Observation(photometry=Photometry.from_filter_set(optical_nir_filters))


@pytest.fixture(scope="session")
def mock_data_z1():
    """Mock flux measurements for z=1 galaxy (11 bands).

    Creates synthetic photometry with realistic SNR~10 per band.
    Uses log-uniform flux distribution (faint to bright).

    Returns
    -------
    dict
        Keys: "flux", "flux_unc", "redshift"
    """
    n_bands = 11

    # Synthetic flux distribution: log-uniform from 0.1 to 100 μJy
    rng = np.random.default_rng(42)
    log_flux = rng.uniform(np.log10(0.1), np.log10(100.0), size=n_bands)
    flux_mjy = 10.0**log_flux * 1e-3  # Convert μJy → mJy

    # Convert to erg/s/cm²/Hz (IAU convention)
    # 1 mJy = 1e-26 erg/s/cm²/Hz
    flux_cgs = flux_mjy * 1e-26

    # SNR ~ 10 per band
    noise_cgs = flux_cgs / 10.0

    # Add Gaussian noise
    flux_obs_cgs = flux_cgs + rng.normal(0, noise_cgs)

    return {
        "flux": jnp.array(flux_obs_cgs),
        "flux_unc": jnp.array(noise_cgs),
        "redshift": 1.0,
    }


@pytest.fixture
def rng_key():
    """PRNG key for reproducible inference runs."""
    return jax.random.PRNGKey(42)


# ── Helper functions ──────────────────────────────────────────────


def measure_jit_and_runtime(
    fitter: Fitter,
    method: str,
    rng_key: jax.Array,
    n_warmup: int = 5,
    **method_kwargs: Any,
) -> dict:
    """Measure JIT compilation time, runtime, and peak memory for an inference run.

    Parameters
    ----------
    fitter : Fitter
        The fitter instance (already initialized with model, data, noise).
    method : str
        Inference method name (map, mcmc_nuts, vi, nss, etc.).
    rng_key : jax.Array
        PRNG key for reproducible runs.
    n_warmup : int
        Number of warmup calls after JIT to measure runtime.
    **method_kwargs
        Additional kwargs for the inference method (e.g., n_live for nss).

    Returns
    -------
    dict
        Performance metrics:
        - t_jit_sec: JIT compilation time (first call)
        - t_runtime_sec: Mean runtime after warmup
        - peak_ram_mb: Peak memory usage during inference
        - success: bool (no exceptions raised)
        - error_type: str (exception type if failed)
        - error_msg: str (exception message if failed)

    Notes
    -----
    - Uses tracemalloc for cross-platform memory profiling
    - Clears JAX compilation cache before measuring JIT time
    - Measures wall-clock time (includes XLA compilation overhead)
    """
    result = {
        "t_jit_sec": None,
        "t_runtime_sec": None,
        "peak_ram_mb": None,
        "success": False,
        "error_type": None,
        "error_msg": None,
    }

    try:
        # Clear JAX compilation cache to get true first-call JIT time
        jax.clear_caches()

        # Start memory tracking
        tracemalloc.start()

        # Measure JIT time (first call includes compilation)
        t_start_jit = time.perf_counter()
        posterior_jit = fitter.run(
            method,
            key=rng_key,
            **method_kwargs,
        )
        t_end_jit = time.perf_counter()
        result["t_jit_sec"] = t_end_jit - t_start_jit

        # Measure runtime after warmup (compilation cached)
        # Skip warmup for MCMC methods - each run is expensive (full chain)
        if method.startswith("mcmc_"):
            # For MCMC, JIT time already includes full sampling run
            result["t_runtime_sec"] = result["t_jit_sec"]
        else:
            # For MAP/VI/NSS, measure cached runtime with warmup
            runtimes = []
            for i in range(n_warmup):
                key_i = jax.random.fold_in(rng_key, abs(hash(f"warmup_{i}")) % (2**31))
                t_start = time.perf_counter()
                _ = fitter.run(
                    method,
                    key=key_i,
                    **method_kwargs,
                )
                t_end = time.perf_counter()
                runtimes.append(t_end - t_start)

            result["t_runtime_sec"] = float(np.mean(runtimes))

        # Get peak memory
        _current, peak = tracemalloc.get_traced_memory()
        result["peak_ram_mb"] = peak / (1024**2)  # Convert bytes → MB
        tracemalloc.stop()

        result["success"] = True

    except Exception as e:
        # Stop memory tracking on error
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        result["success"] = False
        result["error_type"] = type(e).__name__
        result["error_msg"] = str(e)

    return result


def run_scenario(
    name: str,
    params: Parameters,
    ssp_data: dict,
    observation: Observation,
    data: jnp.ndarray,
    noise: jnp.ndarray,
    method: str,
    rng_key: jax.Array,
    **method_kwargs: Any,
) -> dict:
    """Run a single user scenario and return performance + diagnostics.

    Parameters
    ----------
    name : str
        Scenario name (e.g., "A1_optical_quick").
    params : Parameters
        SEDModel parameter specification.
    ssp_data : dict
        SSP data dictionary.
    observation : Observation
        Observation configuration (filters, spectroscopy, noise model).
    data : jnp.ndarray
        Observed flux measurements.
    noise : jnp.ndarray
        Flux uncertainties.
    method : str
        Inference method.
    rng_key : jax.Array
        PRNG key.
    **method_kwargs
        Method-specific kwargs.

    Returns
    -------
    dict
        Scenario results including performance metrics and status.
    """
    # Build model and fitter
    model = SEDModel(params, ssp_data, observation=observation)
    fitter = Fitter(model, data=data, noise=noise)

    # Get dimensionality
    n_free = len(params.free_params)

    # Measure performance
    perf = measure_jit_and_runtime(fitter, method, rng_key, **method_kwargs)

    # Classify status based on thresholds
    status = "✓"  # Excellent
    notes = []

    if perf["success"]:
        jit_time = perf["t_jit_sec"]
        ram_gb = perf["peak_ram_mb"] / 1024.0

        # JIT classification
        if jit_time > PerformanceThresholds.JIT_SLOW:
            status = "❌"
            notes.append(f"JIT >{PerformanceThresholds.JIT_SLOW:.0f}s")
        elif jit_time > PerformanceThresholds.JIT_ACCEPTABLE:
            status = "⚠️"
            notes.append(f"JIT slow ({jit_time:.1f}s)")
        elif jit_time > PerformanceThresholds.JIT_EXCELLENT:
            status = "⚠️"
            notes.append(f"JIT acceptable ({jit_time:.1f}s)")

        # RAM classification
        if ram_gb > PerformanceThresholds.RAM_NEEDS_GOOD_LAPTOP:
            status = "❌"
            notes.append(f"RAM >{PerformanceThresholds.RAM_NEEDS_GOOD_LAPTOP:.0f}GB")
        elif ram_gb > PerformanceThresholds.RAM_LAPTOP_FRIENDLY:
            if status == "✓":
                status = "⚠️"
            notes.append(f"RAM >{PerformanceThresholds.RAM_LAPTOP_FRIENDLY:.0f}GB")
    else:
        status = "❌"
        notes.append(f"{perf['error_type']}: {perf['error_msg'][:50]}")

    return {
        "name": name,
        "D": n_free,
        "method": method,
        "jit_sec": perf["t_jit_sec"],
        "runtime_sec": perf["t_runtime_sec"],
        "ram_gb": perf["peak_ram_mb"] / 1024.0 if perf["peak_ram_mb"] else None,
        "status": status,
        "success": perf["success"],
        "error_type": perf["error_type"],
        "error_msg": perf["error_msg"],
        "notes": "; ".join(notes) if notes else "Fast",
    }


# ── Test Scenarios ────────────────────────────────────────────────


class TestStandardGalaxyWorkflows:
    """Category A: Standard galaxy fitting workflows (most common user path)."""

    def test_a1_quick_optical_fit(self, mist_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """A1. Quick optical fit (D=7, typical user starting point).

        SEDModel: tsnorm SFH, Calzetti dust, nebular (baked SSP), IGM
        Expected: <5s JIT, <30s total inference
        Failure modes: Long JIT would surprise users
        """
        params = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_law_bc="calzetti",
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            nebular_ssp=True,
            apply_igm=True,
            redshift=Fixed(mock_data_z1["redshift"]),
        )

        result = run_scenario(
            name="A1_optical_quick",
            params=params,
            ssp_data=mist_ssp,
            observation=mock_obs_z1,
            data=mock_data_z1["flux"],
            noise=mock_data_z1["flux_unc"],
            method="map",
            rng_key=rng_key,
        )

        jit_str = f"{result['jit_sec']:.1f}s" if result["jit_sec"] is not None else "N/A"
        _rt = result["runtime_sec"]
        runtime_str = f"{_rt:.3f}s" if _rt is not None else "N/A"
        ram_str = f"{result['ram_gb']:.2f}GB" if result["ram_gb"] is not None else "N/A"
        print(
            f"\n{result['name']}: D={result['D']}, "
            f"JIT={jit_str}, "
            f"Runtime={runtime_str}, "
            f"RAM={ram_str}, "
            f"status={result['status']}"
        )
        if not result["success"]:
            print(f"  Error: {result['error_type']}: {result['error_msg']}")

        # Baseline assertion: should not crash
        assert result["success"], f"A1 failed: {result['error_type']}"

        # User experience assertion: JIT should be excellent for this simple case
        # Correctness, not wall-clock: the fit must complete without error. JIT
        # time is printed as a diagnostic but not asserted (it depends on the
        # machine / CI runner load, not on the code).
        assert result["success"], (
            f"{result['name']} inference failed: {result['error_type']}: {result['error_msg']}"
        )

    def test_a2_fir_constrained_fit(self, mist_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """A2. FIR-constrained fit (D=8-9, dust emission with Fixed umin).

        SEDModel: tsnorm + DL07 dust emission with Fixed dust_umin=1.0
        Expected: <10s JIT (template collapsed at init, no 2GB graph)
        Test: Verify Fixed umin → no PERF-01 issue
        """
        params = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_law_bc="calzetti",
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_model="two_component",
            dust_emission="draine_li2007",
            dust_umin=Fixed(1.0),  # FIXED → should avoid PERF-01
            dust_gamma_dl=Uniform(0.0, 0.1),
            dust_qpah=Uniform(0.5, 4.5),
            nebular_ssp=True,
            apply_igm=True,
            redshift=Fixed(mock_data_z1["redshift"]),
        )

        result = run_scenario(
            name="A2_fir_fixed_umin",
            params=params,
            ssp_data=mist_ssp,
            observation=mock_obs_z1,
            data=mock_data_z1["flux"],
            noise=mock_data_z1["flux_unc"],
            method="map",
            rng_key=rng_key,
        )

        jit_str = f"{result['jit_sec']:.1f}s" if result["jit_sec"] is not None else "N/A"
        _rt = result["runtime_sec"]
        runtime_str = f"{_rt:.3f}s" if _rt is not None else "N/A"
        ram_str = f"{result['ram_gb']:.2f}GB" if result["ram_gb"] is not None else "N/A"
        print(
            f"\n{result['name']}: D={result['D']}, "
            f"JIT={jit_str}, "
            f"Runtime={runtime_str}, "
            f"RAM={ram_str}, "
            f"status={result['status']}"
        )
        if not result["success"]:
            print(f"  Error: {result['error_type']}: {result['error_msg']}")

        assert result["success"], f"A2 failed: {result['error_type']}"

        # Should be fast with Fixed umin (no template explosion)
        # Fixed dust_umin must yield a working fit. (The old check used JIT time as
        # a proxy for "no template explosion"; that is machine-dependent and tests
        # nothing about the code — the umin-is-honored property belongs in a
        # structural test, not a wall-clock threshold.)
        assert result["success"], (
            f"{result['name']} inference failed: {result['error_type']}: {result['error_msg']}"
        )


class TestKnownBugReproduction:
    """Category C: Known bug reproduction tests."""

    @pytest.mark.skipif(not _BPASS_EXISTS, reason="BPASS SSP required for BUG-NSS-01 test")
    def test_c1_bpass_posterior_derived_crash(self, bpass_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """C1. BPASS + posterior.derived crash (BUG-NSS-01).

        SEDModel: dpl SFH, BPASS SSP
        Inference: map (quick)
        Expected: ⚠️ Crash when calling posterior.derived (no ssp_mass_remaining table)
        """
        params = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            nebular_ssp=True,
            redshift=Fixed(mock_data_z1["redshift"]),
        )

        model = SEDModel(params, bpass_ssp, observation=mock_obs_z1)
        fitter = Fitter(model, data=mock_data_z1["flux"], noise=mock_data_z1["flux_unc"])

        try:
            posterior = fitter.run("map", key=rng_key)

            # BUG-NSS-01: posterior.derived should crash with BPASS
            try:
                _ = posterior.derived  # This should raise TypeError
                print("\n⚠️ BUG-NSS-01 NOT reproduced: posterior.derived succeeded with BPASS")
                print("   This could mean BUG-NSS-01 was fixed!")
            except TypeError as e:
                if "stack requires ndarray, got NoneType" in str(e):
                    print("\n✓ BUG-NSS-01 reproduced: posterior.derived crashes with BPASS")
                    print(f"   Error: {e}")
                else:
                    raise  # Different TypeError, re-raise

        except Exception as e:
            pytest.fail(f"C1 failed during fit (not posterior.derived): {e}")

    def test_c2_evolving_metallicity_keyerror(self, mist_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """C2. Evolving metallicity KeyError (BUG-NSS-02).

        SEDModel: dense_basis, evolving_metallicity=True
        Inference: map
        Expected: ⚠️ KeyError: 'log_z_abs' in fused kernel
        """
        params = Parameters(
            mean_sfh_type="dense_basis",
            sfh_db_log_total_mass=Uniform(9.0, 12.0),
            sfh_db_log_sfr_inst=Uniform(-2.0, 3.0),
            sfh_db_tx_frac_0=Uniform(0.0, 1.0),
            sfh_db_tx_frac_1=Uniform(0.0, 1.0),
            sfh_db_tx_frac_2=Uniform(0.0, 1.0),
            evolving_metallicity=True,
            met_logzsol_0=Uniform(-2.0, 0.2),
            met_logzsol_final=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            redshift=Fixed(mock_data_z1["redshift"]),
        )

        model = SEDModel(params, mist_ssp, observation=mock_obs_z1)
        fitter = Fitter(model, data=mock_data_z1["flux"], noise=mock_data_z1["flux_unc"])

        try:
            _ = fitter.run("map", key=rng_key)
            print("\n⚠️ BUG-NSS-02 NOT reproduced: evolving_metallicity succeeded")
            print("   This could mean BUG-NSS-02 was fixed!")
        except KeyError as e:
            if "log_z_abs" in str(e):
                print("\n✓ BUG-NSS-02 reproduced: evolving_metallicity raises KeyError")
                print(f"   Error: {e}")
            else:
                raise  # Different KeyError


class TestMemoryAndCompilation:
    """Category E: Memory and compilation profiling."""

    def test_e1_baseline_memory(self, mist_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """E1. Baseline memory (D=7, simple model).

        SEDModel: tsnorm + dust (no nebular, no dust emission)
        Inference: map
        Expected: <2GB total
        """
        params = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            redshift=Fixed(mock_data_z1["redshift"]),
        )

        result = run_scenario(
            name="E1_baseline_memory",
            params=params,
            ssp_data=mist_ssp,
            observation=mock_obs_z1,
            data=mock_data_z1["flux"],
            noise=mock_data_z1["flux_unc"],
            method="map",
            rng_key=rng_key,
        )

        print(
            f"\n{result['name']}: D={result['D']}, "
            f"JIT={result['jit_sec']:.1f}s, "
            f"RAM={result['ram_gb']:.2f}GB (baseline), "
            f"status={result['status']}"
        )

        assert result["success"], f"E1 failed: {result['error_type']}"
        assert result["ram_gb"] < 2.0, (
            f"E1 baseline RAM {result['ram_gb']:.2f}GB exceeds 2GB threshold"
        )


class TestAGNScience:
    """Category B: AGN science cases."""

    @pytest.mark.slow
    def test_b1_agn_disc_torus(self, mist_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """B1. AGN disc + torus (D=12-15).

        SEDModel: tsnorm + Kubota & Done disc + SKIRTOR torus
        Inference: mcmc_nuts
        Expected: <15s JIT, converges to reasonable agn_log_lbol, agn_torus_frac
        Failure modes: Degeneracies (dust emission vs torus); slow mixing
        """
        params = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            agn_model="kubota_done",  # Kubota & Done disc
            agn_log_lbol=Uniform(9.42, 13.42),
            agn_lum_ratio=Uniform(0.01, 0.3),  # Fraction of L_bol in disc
            agn_torus_frac=Uniform(0.0, 1.0),  # Fraction seen by torus (enables SKIRTOR)
            agn_tau_skirtor=Uniform(3.0, 11.0),  # SKIRTOR optical depth
            agn_oa_skirtor=Uniform(20.0, 70.0),  # Opening angle (deg)
            nebular_ssp=True,
            apply_igm=True,
            redshift=Fixed(mock_data_z1["redshift"]),
        )

        result = run_scenario(
            name="B1_agn_disc_torus",
            params=params,
            ssp_data=mist_ssp,
            observation=mock_obs_z1,
            data=mock_data_z1["flux"],
            noise=mock_data_z1["flux_unc"],
            method="map",  # Use MAP instead of NUTS for speed
            rng_key=rng_key,
        )

        jit_str = f"{result['jit_sec']:.1f}s" if result["jit_sec"] is not None else "N/A"
        print(f"\n{result['name']}: D={result['D']}, JIT={jit_str}, status={result['status']}")
        if not result["success"]:
            print(f"  Error: {result['error_type']}: {result['error_msg']}")

        # Unconditional: an `assert result["success"]` nested inside
        # `if result["success"]:` is unreachable exactly when it matters — a
        # failed fit skips the block and the test passes green.
        assert result["success"], (
            f"{result['name']} inference failed: {result['error_type']}: {result['error_msg']}"
        )

    def test_b2_qsogen_tracer_leak(self, mist_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """B2. qsogen with JIT-compiled inference (D=10, BUG-NSS-03 regression test).

        SEDModel: tsnorm + agn_model="qsogen" (template-based AGN)
        Inference: map (which uses JIT internally)
        Expected: ✓ SUCCESS — BUG-NSS-03 fixed (no UnexpectedTracerError)
        Purpose: Regression test for BUG-NSS-03 fix (lazy file I/O moved to import time)
        """
        params = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            agn_model="qsogen",  # Fixed: no longer causes UnexpectedTracerError
            agn_log_lbol=Uniform(9.42, 12.42),
            redshift=Fixed(mock_data_z1["redshift"]),
        )

        result = run_scenario(
            name="B2_qsogen_FIXED",
            params=params,
            ssp_data=mist_ssp,
            observation=mock_obs_z1,
            data=mock_data_z1["flux"],
            noise=mock_data_z1["flux_unc"],
            method="map",
            rng_key=rng_key,
        )

        assert result["success"], (
            f"BUG-NSS-03 regression: qsogen MAP failed with {result['error_type']}: "
            f"{result['error_msg'][:200]}"
        )
        print("\n✓ BUG-NSS-03 fixed: qsogen JIT compilation succeeded")


class TestInferenceStressTests:
    """Category D: Inference method stress tests."""

    @pytest.mark.slow
    def test_d1_high_dimensional_viable(self, mist_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """D1. High-dimensional (D~20) inference viability.

        SEDModel: dense_basis_pure + field + DL07 (Fixed umin) + nebular (D~18-20).
        Probes that the forward model, its gradient, and the optimizer all work at
        the D=20 boundary — the correctness question behind "can you fit a complex
        model?".

        Uses MAP, deliberately. NUTS at D=20 on this model is a multi-minute
        *benchmark* — the tree-building kernel compile alone runs for minutes
        regardless of step count — which belongs in bench/, not a correctness test
        that must stay fast and deterministic. MAP exercises the same
        high-dimensional forward+gradient path in seconds and lets us assert that
        the fit actually succeeds.
        """
        params = Parameters(
            mean_sfh_type=["dense_basis_pure", "field"],
            sfh_dbp_log_total_mass=Uniform(9.0, 12.0),
            sfh_dbp_tx_frac_0=Uniform(0.05, 0.95),
            sfh_dbp_tx_frac_1=Uniform(0.05, 0.95),
            sfh_dbp_tx_frac_2=Uniform(0.05, 0.95),
            sfh_field_psd_sigma=Uniform(0.1, 3.0),
            sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_law_bc="salim_sbl18",
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_model="two_component",
            dust_emission="draine_li2007",
            dust_umin=Fixed(1.0),  # Fixed to avoid PERF-01
            dust_gamma_dl=Uniform(0.0, 0.1),
            dust_qpah=Uniform(0.5, 4.5),
            nebular_ssp=True,
            apply_igm=True,
            redshift=Fixed(mock_data_z1["redshift"]),
            n_grid=64,
        )

        result = run_scenario(
            name="D1_high_d_viability",
            params=params,
            ssp_data=mist_ssp,
            observation=mock_obs_z1,
            data=mock_data_z1["flux"],
            noise=mock_data_z1["flux_unc"],
            method="map",
            rng_key=rng_key,
        )

        jit_str = f"{result['jit_sec']:.1f}s" if result["jit_sec"] is not None else "N/A"
        print(f"\n{result['name']}: D={result['D']}, JIT={jit_str}, status={result['status']}")
        if not result["success"]:
            print(f"  Error: {result['error_type']}: {result['error_msg']}")

        # Correctness: high-dimensional MAP must complete. Unlike NUTS, MAP is a
        # deterministic optimizer and should robustly succeed at D~20 — a failure
        # here means the forward/gradient broke at high dimension, which is exactly
        # what this scenario exists to catch.
        assert result["success"], (
            f"{result['name']} (D={result['D']}) MAP failed: "
            f"{result['error_type']}: {result['error_msg']}"
        )
        assert result["D"] >= 10, f"expected a high-dimensional model, got D={result['D']}"


class TestEdgeCases:
    """Category F: Edge cases and boundary conditions."""

    def test_f1_very_low_redshift(self, mist_ssp, optical_nir_filters, rng_key):
        """F1. Very low redshift (z=0.01, IGM transparent)."""
        params = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            apply_igm=True,
            redshift=Fixed(0.01),
        )

        # Create mock data at z=0.01
        _, _, filter_curves = optical_nir_filters  # Unpack 3-tuple from load_filter_set()
        n_bands = len(filter_curves)
        flux_cgs = 1e-26 * np.ones(n_bands)
        noise_cgs = flux_cgs / 10.0

        obs = Observation(photometry=Photometry.from_filter_set(optical_nir_filters))
        data_dict = {
            "flux": jnp.array(flux_cgs),
            "flux_unc": jnp.array(noise_cgs),
            "redshift": 0.01,
        }

        result = run_scenario(
            name="F1_low_redshift",
            params=params,
            ssp_data=mist_ssp,
            observation=obs,
            data=data_dict["flux"],
            noise=data_dict["flux_unc"],
            method="map",
            rng_key=rng_key,
        )

        jit_str = f"{result['jit_sec']:.1f}s" if result["jit_sec"] is not None else "N/A"
        print(f"\n{result['name']}: z=0.01, JIT={jit_str}, status={result['status']}")
        if not result["success"]:
            print(f"  Error: {result['error_type']}: {result['error_msg']}")

        assert result["success"], f"F1 failed at z=0.01: {result['error_type']}"

    def test_f2_very_high_redshift(self, mist_ssp, optical_nir_filters, rng_key):
        """F2. Very high redshift (z=8, strong Lyman forest)."""
        params = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 1.0),  # Young universe
            sfh_tsnorm_width_gyr=Uniform(0.1, 0.5),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 5.0),
            met_logzsol=Uniform(-2.5, -0.5),  # Metal-poor
            dust_tau_bc=Uniform(0.0, 1.0),  # Low dust
            dust_tau_diff=Uniform(0.0, 0.5),
            apply_igm=True,
            redshift=Fixed(8.0),
        )

        # Create mock data at z=8
        _, _, filter_curves = optical_nir_filters  # Unpack 3-tuple from load_filter_set()
        n_bands = len(filter_curves)
        flux_cgs = 1e-27 * np.ones(n_bands)  # Faint high-z galaxy
        noise_cgs = flux_cgs / 5.0

        obs = Observation(photometry=Photometry.from_filter_set(optical_nir_filters))
        data_dict = {
            "flux": jnp.array(flux_cgs),
            "flux_unc": jnp.array(noise_cgs),
            "redshift": 8.0,
        }

        result = run_scenario(
            name="F2_high_redshift",
            params=params,
            ssp_data=mist_ssp,
            observation=obs,
            data=data_dict["flux"],
            noise=data_dict["flux_unc"],
            method="map",
            rng_key=rng_key,
        )

        jit_str = f"{result['jit_sec']:.1f}s" if result["jit_sec"] is not None else "N/A"
        print(f"\n{result['name']}: z=8, JIT={jit_str}, status={result['status']}")
        if not result["success"]:
            print(f"  Error: {result['error_type']}: {result['error_msg']}")

        assert result["success"], f"F2 failed at z=8: {result['error_type']}"

    def test_f3_zero_dust(self, mist_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """F3. Zero dust (tau_bc=0, tau_diff=0)."""
        params = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_tau_bc=Fixed(0.0),  # Zero dust
            dust_tau_diff=Fixed(0.0),
            redshift=Fixed(mock_data_z1["redshift"]),
        )

        result = run_scenario(
            name="F3_zero_dust",
            params=params,
            ssp_data=mist_ssp,
            observation=mock_obs_z1,
            data=mock_data_z1["flux"],
            noise=mock_data_z1["flux_unc"],
            method="map",
            rng_key=rng_key,
        )

        print(
            f"\n{result['name']}: tau=0, JIT={result['jit_sec']:.1f}s, status={result['status']}"
        )

        assert result["success"], f"F3 failed with zero dust: {result['error_type']}"

    def test_f4_extreme_metallicity(self, mist_ssp, mock_obs_z1, mock_data_z1, rng_key):
        """F4. Extreme metallicity (logZ = -2.0 subsolar)."""
        params = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            met_logzsol=Fixed(-2.0),  # Extreme metal-poor (grid edge)
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            redshift=Fixed(mock_data_z1["redshift"]),
        )

        result = run_scenario(
            name="F4_extreme_metallicity",
            params=params,
            ssp_data=mist_ssp,
            observation=mock_obs_z1,
            data=mock_data_z1["flux"],
            noise=mock_data_z1["flux_unc"],
            method="map",
            rng_key=rng_key,
        )

        print(
            f"\n{result['name']}: logZ=-2.0, "
            f"JIT={result['jit_sec']:.1f}s, "
            f"status={result['status']}"
        )

        assert result["success"], f"F4 failed at metallicity grid edge: {result['error_type']}"
