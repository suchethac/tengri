#!/usr/bin/env python3
"""Profile JIT opportunities: identify real performance improvements.

Answers: "SEDModel evaluation takes milliseconds, why does loss eval take longer?"

Strategy:
1. Time pure model prediction (forward model only)
2. Time full loss (prediction + chi-square + prior)
3. Break down overhead
4. Test mode='auto' vs mode='_traceable'
5. Identify JIT opportunities
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from tengri import SEDModel, Fitter, Parameters, Observation, Photometry
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform


def time_function(fn, *args, n_warmup=10, n_evals=50, **kwargs):
    """Time a function with proper warmup."""
    # Compile + first call
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    compile_time = time.perf_counter() - t0

    # Warmup
    for _ in range(n_warmup):
        _ = fn(*args, **kwargs)

    # Measure
    times = []
    for _ in range(n_evals):
        t0 = time.perf_counter()
        _ = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return {
        "compile_time_s": compile_time,
        "mean_ms": np.mean(times_arr) * 1000,
        "std_ms": np.std(times_arr) * 1000,
        "min_ms": np.min(times_arr) * 1000,
        "max_ms": np.max(times_arr) * 1000,
        "median_ms": np.median(times_arr) * 1000,
        "result": result,
    }


def profile_model_photometry(model, fitter):
    """Profile predict_photometry with different modes."""
    print("\n" + "=" * 80)
    print("MODEL.PREDICT_PHOTOMETRY PROFILING")
    print("=" * 80)

    from tengri.utils.transforms import to_bounded

    key = jax.random.PRNGKey(42)
    params_unbounded = fitter._initialize_unbounded(key)

    # Convert unbounded → bounded
    param_dict = {}
    for name in fitter._free_names:
        lo, hi = fitter._bounds[name]
        param_dict[name] = to_bounded(params_unbounded[name], lo, hi)
    for name, val in fitter._fixed_values.items():
        param_dict[name] = val

    print("\n  Testing mode='auto'...")
    timing_auto = time_function(model.predict_photometry, param_dict, mode="auto")

    print("\n  Testing mode='_traceable'...")
    timing_traceable = time_function(model.predict_photometry, param_dict, mode="_traceable")

    print("\n  Photometry timing (mode='auto'):")
    print(f"    Compile: {timing_auto['compile_time_s']:.3f}s")
    print(f"    Mean: {timing_auto['mean_ms']:.3f} ± {timing_auto['std_ms']:.3f} ms")
    print(f"    Median: {timing_auto['median_ms']:.3f} ms")

    print("\n  Photometry timing (mode='_traceable'):")
    print(f"    Compile: {timing_traceable['compile_time_s']:.3f}s")
    print(f"    Mean: {timing_traceable['mean_ms']:.3f} ± {timing_traceable['std_ms']:.3f} ms")
    print(f"    Median: {timing_traceable['median_ms']:.3f} ms")

    if timing_auto["mean_ms"] < timing_traceable["mean_ms"]:
        speedup = timing_traceable["mean_ms"] / timing_auto["mean_ms"]
        print(f"\n  ✓ mode='auto' is {speedup:.2f}x FASTER")
    else:
        speedup = timing_auto["mean_ms"] / timing_traceable["mean_ms"]
        print(f"\n  ✓ mode='_traceable' is {speedup:.2f}x FASTER")

    return timing_auto, timing_traceable


def profile_loss_components(fitter):
    """Profile loss function components separately."""
    print("\n" + "=" * 80)
    print("LOSS FUNCTION COMPONENT PROFILING")
    print("=" * 80)

    from tengri.utils.transforms import to_bounded

    loss_fn = fitter._get_or_build_loss_fn(mode="auto")
    data_args = fitter._data_args

    key = jax.random.PRNGKey(42)
    params_unbounded = fitter._initialize_unbounded(key)

    # Time full loss
    print("\n  Full loss function...")
    timing_loss = time_function(lambda p: loss_fn(p, data_args), params_unbounded)

    print(f"\n  Full loss timing:")
    print(f"    Compile: {timing_loss['compile_time_s']:.3f}s")
    print(f"    Mean: {timing_loss['mean_ms']:.3f} ± {timing_loss['std_ms']:.3f} ms")
    print(f"    Median: {timing_loss['median_ms']:.3f} ms")

    # Now break down what's in the loss
    # Loss = predict + chi-square + prior
    # We need to create manual versions to time each piece

    # 1. Just prediction (via fitter's internal model)
    # Convert unbounded → bounded
    param_dict = {}
    for name in fitter._free_names:
        lo, hi = fitter._bounds[name]
        param_dict[name] = to_bounded(params_unbounded[name], lo, hi)
    for name, val in fitter._fixed_values.items():
        param_dict[name] = val

    @jax.jit
    def just_predict(p_dict):
        return fitter.model.predict_photometry(p_dict, mode="_traceable")

    print("\n  Just prediction (no chi-square, no prior)...")
    timing_predict = time_function(just_predict, param_dict)

    print(f"    Mean: {timing_predict['mean_ms']:.3f} ± {timing_predict['std_ms']:.3f} ms")

    # 2. Prediction + chi-square (no prior)
    obs_flux = data_args["data"]
    obs_unc = data_args["noise"]

    @jax.jit
    def predict_and_chisq(p_dict):
        model_flux = fitter.model.predict_photometry(p_dict, mode="_traceable")
        chi = (model_flux - obs_flux) / obs_unc
        return jnp.sum(chi**2)

    print("\n  Prediction + chi-square (no prior)...")
    timing_chisq = time_function(predict_and_chisq, param_dict)

    print(f"    Mean: {timing_chisq['mean_ms']:.3f} ± {timing_chisq['std_ms']:.3f} ms")

    # Overhead breakdown
    chisq_overhead = timing_chisq["mean_ms"] - timing_predict["mean_ms"]
    prior_overhead = timing_loss["mean_ms"] - timing_chisq["mean_ms"]

    print("\n  Overhead breakdown:")
    print(
        f"    Prediction: {timing_predict['mean_ms']:.3f} ms ({timing_predict['mean_ms'] / timing_loss['mean_ms'] * 100:.1f}%)"
    )
    print(
        f"    Chi-square: {chisq_overhead:.3f} ms ({chisq_overhead / timing_loss['mean_ms'] * 100:.1f}%)"
    )
    print(
        f"    Prior eval: {prior_overhead:.3f} ms ({prior_overhead / timing_loss['mean_ms'] * 100:.1f}%)"
    )
    print(f"    Total: {timing_loss['mean_ms']:.3f} ms")

    return timing_loss, timing_predict, timing_chisq


def profile_model_components(model, param_dict):
    """Profile individual model components."""
    print("\n" + "=" * 80)
    print("MODEL COMPONENT PROFILING")
    print("=" * 80)

    # We'll call internal components directly to see where time is spent
    # This requires accessing model internals

    from tengri.forward.pipeline import assemble_sed

    # Get the component outputs
    result = model.predict(param_dict, mode="_traceable")

    print("\n  Component outputs:")
    print(f"    SFH: {hasattr(result.components, 'sfh')}")
    print(f"    Stellar: {hasattr(result.components, 'stellar')}")
    print(f"    Dust attenuation: {hasattr(result.components, 'dust_attenuation')}")
    print(f"    Dust emission: {hasattr(result.components, 'dust_emission')}")
    print(f"    Nebular: {hasattr(result.components, 'nebular')}")
    print(f"    IGM: {hasattr(result.components, 'igm')}")

    # Individual component timing is hard without refactoring the pipeline
    # But we can test model variants to isolate components

    print("\n  Testing model variants to isolate components...")

    # Variant 1: No dust emission (set dust_emission=None)
    params_no_dust_em = Parameters(
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
        dust_emission=None,  # Disable dust emission
        nebular_ssp=True,
        apply_igm=True,
        redshift=Fixed(1.0),
    )

    from tengri.observation.filters import load_filter_set

    filters = load_filter_set(
        [
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
    )
    observation = Observation(photometry=Photometry.from_filter_set(filters))
    model_no_dust_em = SEDModel(params_no_dust_em, model._ssp_data, observation=observation)

    print("\n  SEDModel without dust emission...")
    timing_no_dust_em = time_function(
        model_no_dust_em.predict, param_dict, mode="_traceable", n_evals=30
    )
    print(f"    Mean: {timing_no_dust_em['mean_ms']:.3f} ± {timing_no_dust_em['std_ms']:.3f} ms")

    # Variant 2: No nebular
    params_no_neb = Parameters(
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
        dust_emission=None,
        nebular_ssp=False,  # Disable nebular
        apply_igm=True,
        redshift=Fixed(1.0),
    )
    model_no_neb = SEDModel(params_no_neb, model._ssp_data, observation=observation)

    print("\n  SEDModel without nebular...")
    timing_no_neb = time_function(model_no_neb.predict, param_dict, mode="_traceable", n_evals=30)
    print(f"    Mean: {timing_no_neb['mean_ms']:.3f} ± {timing_no_neb['std_ms']:.3f} ms")

    # Variant 3: No IGM
    params_no_igm = Parameters(
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
        dust_emission=None,
        nebular_ssp=False,
        apply_igm=False,  # Disable IGM
        redshift=Fixed(1.0),
    )
    model_no_igm = SEDModel(params_no_igm, model._ssp_data, observation=observation)

    print("\n  SEDModel without IGM...")
    timing_no_igm = time_function(model_no_igm.predict, param_dict, mode="_traceable", n_evals=30)
    print(f"    Mean: {timing_no_igm['mean_ms']:.3f} ± {timing_no_igm['std_ms']:.3f} ms")

    # Original model timing for comparison
    print("\n  Full model (with DL07, nebular, IGM)...")
    timing_full = time_function(model.predict, param_dict, mode="_traceable", n_evals=30)
    print(f"    Mean: {timing_full['mean_ms']:.3f} ± {timing_full['std_ms']:.3f} ms")

    # Component cost estimates
    print("\n  Component cost estimates:")
    dust_em_cost = timing_full["mean_ms"] - timing_no_dust_em["mean_ms"]
    neb_cost = timing_no_dust_em["mean_ms"] - timing_no_neb["mean_ms"]
    igm_cost = timing_no_neb["mean_ms"] - timing_no_igm["mean_ms"]
    base_cost = timing_no_igm["mean_ms"]

    print(
        f"    Base (SFH + stellar + dust atten): {base_cost:.3f} ms ({base_cost / timing_full['mean_ms'] * 100:.1f}%)"
    )
    print(f"    IGM: {igm_cost:.3f} ms ({igm_cost / timing_full['mean_ms'] * 100:.1f}%)")
    print(f"    Nebular: {neb_cost:.3f} ms ({neb_cost / timing_full['mean_ms'] * 100:.1f}%)")
    print(
        f"    Dust emission (DL07): {dust_em_cost:.3f} ms ({dust_em_cost / timing_full['mean_ms'] * 100:.1f}%)"
    )
    print(f"    Total: {timing_full['mean_ms']:.3f} ms")


def analyze_jit_fusion_opportunities(fitter):
    """Analyze JIT compilation and fusion opportunities."""
    print("\n" + "=" * 80)
    print("JIT FUSION ANALYSIS")
    print("=" * 80)

    loss_fn = fitter._get_or_build_loss_fn(mode="auto")
    data_args = fitter._data_args

    key = jax.random.PRNGKey(42)
    params = fitter._initialize_unbounded(key)

    # Get the HLO (high-level optimizer) graph
    print("\n  Analyzing JIT compilation graph...")

    # Create a jitted version and get lowered representation
    jitted_loss = jax.jit(lambda p: loss_fn(p, data_args))

    # Lower to see the compiled graph
    lowered = jitted_loss.lower(params)

    print(f"\n  Lowered computation:")
    print(f"    Input shapes: {jax.tree_util.tree_map(lambda x: x.shape, params)}")

    # Count operations in HLO
    hlo_text = lowered.as_text()

    # Extract some stats
    n_ops = hlo_text.count("\n")
    n_calls = hlo_text.count("call")
    n_dots = hlo_text.count("dot")
    n_broadcasts = hlo_text.count("broadcast")

    print(f"\n  HLO statistics:")
    print(f"    Lines: {n_ops}")
    print(f"    Function calls: {n_calls}")
    print(f"    Dot products: {n_dots}")
    print(f"    Broadcasts: {n_broadcasts}")

    # Look for potential fusion opportunities
    print("\n  Potential optimization opportunities:")

    # Check if loss is being recompiled on different inputs
    print("\n  Testing recompilation on different parameter values...")

    times = []
    for i in range(10):
        key_i = jax.random.PRNGKey(i + 100)
        params_i = fitter._initialize_unbounded(key_i)

        t0 = time.perf_counter()
        _ = jitted_loss(params_i)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    if times_arr.std() / times_arr.mean() > 0.5:
        print(f"    ⚠️  HIGH VARIANCE detected ({times_arr.std() * 1000:.2f}ms std)")
        print(f"    Possible recompilation on different inputs")
    else:
        print(f"    ✓ Stable timing ({times_arr.std() * 1000:.2f}ms std)")
        print(f"    No recompilation detected")


def main():
    print("=" * 80)
    print("JIT OPTIMIZATION OPPORTUNITY PROFILER")
    print("=" * 80)

    # Load SSP data
    print("\nLoading SSP data...")
    ssp_path = (
        Path(__file__).parent.parent
        / "data"
        / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    )
    if not ssp_path.exists():
        print(f"  ❌ SSP data not found at {ssp_path}")
        return

    ssp_data = load_ssp_data(str(ssp_path))
    print(f"  ✓ Loaded")

    # Create test model (same as test_a2)
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
        dust_emission="draine_li2007",
        dust_umin=Fixed(1.0),
        dust_qpah=Uniform(0.5, 4.5),
        dust_gamma_dl=Uniform(0.0, 0.2),
        nebular_ssp=True,
        apply_igm=True,
        redshift=Fixed(1.0),
    )

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

    # Mock data
    flux_mjy = jnp.array([0.8, 1.0, 1.2, 1.1, 1.5, 1.8, 2.0, 2.2, 2.1, 1.9, 3.5, 2.8])
    flux_unc_mjy = flux_mjy * 0.15
    flux_cgs = flux_mjy * 1e-26
    flux_unc_cgs = flux_unc_mjy * 1e-26

    fitter = Fitter(model, flux_cgs, flux_unc_cgs)
    D = len(fitter._free_names)
    print(f"\n  D = {D} free parameters")

    # 1. Profile photometry
    timing_auto, timing_traceable = profile_model_photometry(model, fitter)

    # 2. Profile loss components
    timing_loss, timing_predict, timing_chisq = profile_loss_components(fitter)

    # 3. Analyze JIT fusion
    analyze_jit_fusion_opportunities(fitter)

    # Final summary
    print("\n" + "=" * 80)
    print("ANSWERING: 'SEDModel eval takes milliseconds, why does loss take longer?'")
    print("=" * 80)

    print(f"\n  1. Pure model.predict_photometry (mode='auto'): {timing_auto['mean_ms']:.3f} ms")
    print(
        f"  2. Pure model.predict_photometry (mode='_traceable'): {timing_traceable['mean_ms']:.3f} ms"
    )
    print(f"  3. Just prediction (in loss context): {timing_predict['mean_ms']:.3f} ms")
    print(f"  4. Prediction + chi-square: {timing_chisq['mean_ms']:.3f} ms")
    print(f"  5. Full loss (predict + chi² + prior): {timing_loss['mean_ms']:.3f} ms")

    pred_overhead = timing_predict["mean_ms"] - timing_traceable["mean_ms"]
    chisq_overhead = timing_chisq["mean_ms"] - timing_predict["mean_ms"]
    prior_overhead = timing_loss["mean_ms"] - timing_chisq["mean_ms"]

    print(f"\n  Overhead breakdown:")
    print(f"    Base model eval: {timing_traceable['mean_ms']:.3f} ms")
    print(f"    + Loss context overhead: {pred_overhead:.3f} ms")
    print(f"    + Chi-square computation: {chisq_overhead:.3f} ms")
    print(f"    + Prior evaluation: {prior_overhead:.3f} ms")
    print(f"    = Total loss: {timing_loss['mean_ms']:.3f} ms")

    print(f"\n  User's observation: 'SEDModel eval takes milliseconds'")
    print(f"  ✓ CONFIRMED: model.predict_photometry = {timing_auto['mean_ms']:.3f} ms")

    print(f"\n  Question: 'Why does loss eval take longer?'")
    print(f"  Answer: Loss = model eval + chi² + prior")
    print(
        f"          The {timing_loss['mean_ms'] - timing_traceable['mean_ms']:.3f}ms overhead comes from:"
    )
    print(
        f"          - Chi-square: {chisq_overhead:.3f}ms ({chisq_overhead / timing_loss['mean_ms'] * 100:.1f}%)"
    )
    print(
        f"          - Prior: {prior_overhead:.3f}ms ({prior_overhead / timing_loss['mean_ms'] * 100:.1f}%)"
    )
    print(
        f"          - Loss context: {pred_overhead:.3f}ms ({pred_overhead / timing_loss['mean_ms'] * 100:.1f}%)"
    )

    print(f"\n  JIT optimization opportunities:")
    if timing_auto["mean_ms"] < timing_traceable["mean_ms"]:
        speedup = timing_traceable["mean_ms"] / timing_auto["mean_ms"]
        print(f"    1. ✓ mode='auto' already used, {speedup:.2f}x faster than traceable")
    else:
        speedup = timing_auto["mean_ms"] / timing_traceable["mean_ms"]
        print(f"    1. ⚠️  Switch to mode='_traceable' for {speedup:.2f}x speedup")

    print(f"    2. Check variance in timing (see fusion analysis above)")
    print(f"    3. Component-level fusion (would require kernel refactor)")


if __name__ == "__main__":
    main()
