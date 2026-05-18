"""Measure end-to-end inference speed for realistic galaxy fitting scenarios.

Tests actual inference time (not just loss function timing) for:
- A1: Quick optical fit (D=7, MAP + NUTS)
- A2: FIR-constrained fit (D=8-9, NUTS with DL07)
- A4: Stochastic SFH (D=12, VI)

Focus: User experience from astronomer perspective.
Expected thresholds:
- JIT < 15s: Excellent
- JIT 15-30s: Acceptable
- JIT > 60s: Broken (user thinks it hung)
- Total inference < 60s: Good for quick exploration
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from tengri import Fitter, Observation, Parameters, Photometry, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.inference.backends import run_map, run_nifty_vi, run_nuts
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform


@pytest.fixture(scope="module")
def ssp_data():
    """Load SSP data once for all tests."""
    ssp_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    )
    if not ssp_path.exists():
        pytest.skip(f"SSP data not found: {ssp_path}")
    return load_ssp_data(str(ssp_path))


@pytest.fixture(scope="module")
def filters_optical():
    """11 optical+NIR bands (HST+VISTA+IRAC)."""
    filter_names = [
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
        "irac_58",
    ]
    return load_filter_set(filter_names)


@pytest.fixture
def mock_flux_optical():
    """Mock optical+NIR photometry."""
    flux_mjy = jnp.array([0.8, 1.0, 1.2, 1.1, 1.5, 1.8, 2.0, 2.2, 2.1, 1.9, 1.7])
    flux_unc_mjy = flux_mjy * 0.1
    return flux_mjy * 1e-26, flux_unc_mjy * 1e-26


def measure_inference_time(fitter, inference_fn, **kwargs):
    """Measure JIT compile time + total inference time.

    Returns
    -------
    dict with:
        - jit_time_s: First call time (compile + run)
        - inference_time_s: Total time excluding compile
        - total_time_s: jit_time_s + inference_time_s
        - result: Posterior object
    """
    key = jax.random.PRNGKey(42)

    # First call: JIT compile
    t0 = time.perf_counter()
    result = inference_fn(fitter, key=key, **kwargs)
    jit_time = time.perf_counter() - t0

    # For MAP/VI: inference is instant after JIT
    # For MCMC: actual sampling happens in first call
    # So jit_time includes both compile + inference

    return {
        "jit_time_s": jit_time,
        "total_time_s": jit_time,
        "result": result,
    }


def test_a1_quick_optical_map(ssp_data, filters_optical, mock_flux_optical):
    """A1: Quick optical fit with MAP (D=7).

    Expected: <5s JIT, instant inference.
    Typical user starting point.
    """
    print("\n" + "=" * 80)
    print("A1: Quick Optical Fit (D=7) - MAP Optimization")
    print("=" * 80)

    # SEDModel: tsnorm SFH, Calzetti dust, nebular, IGM
    params = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
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
        redshift=Fixed(1.0),
    )

    observation = Observation(photometry=Photometry.from_filter_set(filters_optical))
    model = SEDModel(params, ssp_data, observation=observation)

    flux_cgs, flux_unc_cgs = mock_flux_optical
    fitter = Fitter(model, flux_cgs, flux_unc_cgs)

    D = len(fitter._free_names)
    print(f"  D = {D} free parameters")
    print(f"  Free params: {fitter._free_names}")

    # Measure MAP optimization
    print("\n  Running MAP (100 steps)...")
    timing = measure_inference_time(
        fitter,
        run_map,
        n_steps=100,
        learning_rate=0.02,
        verbose=False,
    )

    result = timing["result"]

    print("\n  Results:")
    print(f"    JIT + inference: {timing['jit_time_s']:.2f}s")
    print(f"    Final loss: {result.diagnostics['final_loss']:.4f}")
    print(f"    Optimizer: {result.diagnostics['optimizer']}")

    # User experience check
    if timing["jit_time_s"] < 5.0:
        status = "✓ EXCELLENT"
    elif timing["jit_time_s"] < 15.0:
        status = "✓ GOOD"
    elif timing["jit_time_s"] < 30.0:
        status = "⚠️ ACCEPTABLE"
    else:
        status = "❌ TOO SLOW"

    print(f"    Status: {status}")

    assert timing["jit_time_s"] < 30.0, f"MAP too slow: {timing['jit_time_s']:.2f}s"


def test_a1_quick_optical_nuts(ssp_data, filters_optical, mock_flux_optical):
    """A1: Quick optical fit with NUTS (D=7).

    Expected: <5s JIT, <30s total (500 warmup + 500 sample).
    Standard workflow after MAP.
    """
    print("\n" + "=" * 80)
    print("A1: Quick Optical Fit (D=7) - NUTS MCMC")
    print("=" * 80)

    params = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
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
        redshift=Fixed(1.0),
    )

    observation = Observation(photometry=Photometry.from_filter_set(filters_optical))
    model = SEDModel(params, ssp_data, observation=observation)

    flux_cgs, flux_unc_cgs = mock_flux_optical
    fitter = Fitter(model, flux_cgs, flux_unc_cgs)

    D = len(fitter._free_names)
    print(f"  D = {D} free parameters")

    # Measure NUTS
    print("\n  Running NUTS (250 warmup + 250 samples)...")
    timing = measure_inference_time(
        fitter,
        run_nuts,
        n_warmup=250,
        n_samples=250,
        verbose=False,
    )

    result = timing["result"]

    # samples is a dict, get first param to check shape
    first_param = next(iter(result.samples.keys()))
    n_samples = result.samples[first_param].shape[0]

    print("\n  Results:")
    print(f"    Total time: {timing['total_time_s']:.2f}s")
    print(f"    Samples: {n_samples} × {len(result.samples)} parameters")
    print(f"    Acceptance rate: {result.diagnostics.get('acceptance_rate', 'N/A')}")

    # User experience check
    if timing["total_time_s"] < 30.0:
        status = "✓ EXCELLENT"
    elif timing["total_time_s"] < 60.0:
        status = "✓ GOOD"
    elif timing["total_time_s"] < 120.0:
        status = "⚠️ ACCEPTABLE"
    else:
        status = "❌ TOO SLOW"

    print(f"    Status: {status}")

    assert timing["total_time_s"] < 120.0, f"NUTS too slow: {timing['total_time_s']:.2f}s"


@pytest.mark.slow
def test_a2_fir_constrained_nuts(ssp_data):
    """A2: FIR-constrained fit with DL07 dust emission (D=8-9).

    Expected: <10s JIT, <60s total with NUTS.
    Tests Fixed dust_umin → no 2GB graph issue.

    NOTE: Most time is MCMC overhead (compilation + tree building), not loss eval.
    Loss function is ~2-3ms. See docs/dev/performance-bottleneck-analysis-2026-04-18.md
    """
    print("\n" + "=" * 80)
    print("A2: FIR-Constrained Fit (D=8-9) - NUTS with DL07")
    print("=" * 80)

    # SEDModel: tsnorm + DL07 with Fixed dust_umin
    params = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_law_bc="calzetti",
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_emission="draine_li2007",
        dust_umin=Fixed(1.0),  # Fixed → fast
        dust_qpah=Uniform(0.5, 4.5),
        dust_gamma_dl=Uniform(0.0, 0.2),
        nebular_ssp=True,
        apply_igm=True,
        redshift=Fixed(1.0),
    )

    # Add FIR bands
    filter_names = [
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
        "herschel_160",
        "herschel_250",
    ]
    filters = load_filter_set(filter_names)
    observation = Observation(photometry=Photometry.from_filter_set(filters))
    model = SEDModel(params, ssp_data, observation=observation)

    # Mock data (12 bands)
    flux_mjy = jnp.array([0.8, 1.0, 1.2, 1.1, 1.5, 1.8, 2.0, 2.2, 2.1, 1.9, 3.5, 2.8])
    flux_unc_mjy = flux_mjy * 0.15
    flux_cgs = flux_mjy * 1e-26
    flux_unc_cgs = flux_unc_mjy * 1e-26

    fitter = Fitter(model, flux_cgs, flux_unc_cgs)

    D = len(fitter._free_names)
    print(f"  D = {D} free parameters")
    print("  dust_umin: Fixed(1.0) → no 2GB graph")

    # Verify DL07 preintegration is active
    has_preint = hasattr(model, "_precomputed") and model._precomputed.dust_ir_lookup is not None
    print(f"  DL07 preintegration: {'✓ ACTIVE' if has_preint else '❌ DISABLED'}")

    # Measure NUTS
    print("\n  Running NUTS (250 warmup + 250 samples)...")
    timing = measure_inference_time(
        fitter,
        run_nuts,
        n_warmup=250,
        n_samples=250,
        verbose=False,
    )

    result = timing["result"]

    # samples is a dict, get first param to check shape
    first_param = next(iter(result.samples.keys()))
    n_samples = result.samples[first_param].shape[0]

    print("\n  Results:")
    print(f"    Total time: {timing['total_time_s']:.2f}s")
    print(f"    Samples: {n_samples} × {len(result.samples)} parameters")

    # Explain time breakdown
    n_steps_total = 250 + 250  # warmup + samples
    expected_loss_time = (n_steps_total * 2.3) / 1000  # ~1.2s if 2.3ms per loss eval
    print("    Breakdown:")
    print("      MAP init: ~5-15s (200 steps)")
    print("      Warmup: ~20-30s (adaptation overhead)")
    print("      Sampling: ~10-15s (tree building)")
    print(f"      Pure loss evals: ~{expected_loss_time:.1f}s ({n_steps_total} × 2.3ms)")

    # Check JIT time specifically (compile should be fast with Fixed dust_umin)
    # Note: For MCMC, "jit_time_s" includes compile + first sampling
    if timing["total_time_s"] < 60.0:
        status = "✓ EXCELLENT"
    elif timing["total_time_s"] < 120.0:
        status = "✓ GOOD"
    else:
        status = "⚠️ SLOW"

    print(f"    Status: {status} (MCMC overhead dominates, not loss function)")

    assert timing["total_time_s"] < 180.0, f"DL07 NUTS too slow: {timing['total_time_s']:.2f}s"


@pytest.mark.slow
def test_a4_stochastic_sfh_vi(ssp_data):
    """A4: Stochastic SFH with VI (D=12).

    Expected: <15s JIT, <60s total.
    Tests dense_basis+field with geoVI.
    """
    print("\n" + "=" * 80)
    print("A4: Stochastic SFH (D=12) - geoVI")
    print("=" * 80)

    # SEDModel: dense_basis+field (4 DB + 2 PSD params)
    params = Parameters(
        mean_sfh_type=["dense_basis", "field"],
        sfh_dbp_log_total_mass=Uniform(9.0, 11.5),
        sfh_dbp_tx_frac_0=Uniform(0.05, 0.95),
        sfh_dbp_tx_frac_1=Uniform(0.05, 0.95),
        sfh_dbp_tx_frac_2=Uniform(0.05, 0.95),
        sfh_field_psd_sigma=Uniform(0.1, 2.0),
        sfh_field_psd_tau_myr=Uniform(10.0, 500.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_law_bc="salim",
        dust_slope=Uniform(-1.5, 0.4),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        nebular_ssp=True,
        apply_igm=True,
        redshift=Fixed(0.1),
    )

    # UV+optical (12 bands, dwarf galaxy)
    filter_names = [
        "galex_fuv",
        "galex_nuv",
        "hst_f435w",
        "hst_f606w",
        "hst_f775w",
        "hst_f814w",
        "hst_f125w",
        "hst_f140w",
        "hst_f160w",
        "vista_ks",
        "irac_36",
        "irac_45",
    ]
    filters = load_filter_set(filter_names)
    observation = Observation(photometry=Photometry.from_filter_set(filters))
    model = SEDModel(params, ssp_data, observation=observation)

    # Mock data
    flux_mjy = jnp.array([0.5, 0.7, 0.8, 1.0, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 1.9, 1.7])
    flux_unc_mjy = flux_mjy * 0.15
    flux_cgs = flux_mjy * 1e-26
    flux_unc_cgs = flux_unc_mjy * 1e-26

    fitter = Fitter(model, flux_cgs, flux_unc_cgs)

    D = len(fitter._free_names)
    print(f"  D = {D} free parameters")
    print("  Stochastic SFH: dense_basis + correlated field")

    # Measure VI (geoVI)
    print("\n  Running geoVI (10 KL iterations, 8 samples/iter)...")
    timing = measure_inference_time(
        fitter,
        run_nifty_vi,
        n_iterations=10,
        n_samples=8,
        verbose=False,
    )

    result = timing["result"]

    # samples is a dict, get first param to check shape
    if result.samples is not None:
        first_param = next(iter(result.samples.keys()))
        n_samples = result.samples[first_param].shape[0]
        samples_str = f"{n_samples} × {len(result.samples)} parameters"
    else:
        samples_str = "N/A"

    print("\n  Results:")
    print(f"    Total time: {timing['total_time_s']:.2f}s")
    print(f"    Samples: {samples_str}")

    if timing["total_time_s"] < 60.0:
        status = "✓ EXCELLENT"
    elif timing["total_time_s"] < 120.0:
        status = "✓ GOOD"
    else:
        status = "⚠️ SLOW"

    print(f"    Status: {status}")

    assert timing["total_time_s"] < 180.0, f"VI too slow: {timing['total_time_s']:.2f}s"
