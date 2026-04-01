"""Run the full tengri profiling suite.

One command to profile everything: pipeline breakdown, scaling tests,
memory footprint, and generate analysis plots.

Usage::

    cd ~/Projects/tengri
    python profiling/run_all.py --quick          # CI mode (~3 min)
    python profiling/run_all.py --full            # comprehensive (~20 min)
    python profiling/run_all.py --pipeline-only   # just pipeline breakdown
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description="Tengri profiling suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode for CI")
    parser.add_argument("--full", action="store_true", help="Comprehensive mode")
    parser.add_argument("--pipeline-only", action="store_true",
                        help="Only run pipeline breakdown")
    parser.add_argument("--scaling-only", action="store_true",
                        help="Only run scaling tests")
    parser.add_argument("--memory-only", action="store_true",
                        help="Only run memory profiling")
    parser.add_argument("--output-dir", default="profiling/outputs",
                        help="Output directory")
    args = parser.parse_args()

    quick = args.quick or not args.full
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TENGRI PROFILING SUITE")
    print("=" * 70)
    print(f"Platform: {sys.platform}, JAX backend: {jax.default_backend()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Mode: {'quick' if quick else 'full'}")
    print(f"Output: {output_dir}")

    # Load shared data
    from tengri import (
        Fixed,
        Model,
        ParamSpec,
        Uniform,
        load_filter_set,
        load_ssp_data,
    )

    print("\nLoading SSP data...")
    ssp = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    print(f"SSP shape: {ssp.ssp_flux.shape}")

    run_all = not (args.pipeline_only or args.scaling_only or args.memory_only)

    # -----------------------------------------------------------------
    # 1. Pipeline breakdown
    # -----------------------------------------------------------------
    if run_all or args.pipeline_only:
        print("\n" + "=" * 70)
        print("PIPELINE PROFILING")
        print("=" * 70)

        from tengri.profiling.pipeline import profile_pipeline

        n_iters = 50 if quick else 200

        # Smooth model (D=7)
        spec_smooth = ParamSpec(
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.5, 3.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            met_logzsol=Uniform(-2.0, 0.5),
            dust_tau_bc=Uniform(0.0, 2.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.1),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_fused = Model(spec_smooth, ssp, filters=filters, precompute=True)

        params = spec_smooth.sample(jax.random.PRNGKey(42))
        report = profile_pipeline(
            model_fused, params, n=n_iters, config_name="smooth_D7_fused"
        )
        print(report.summary())
        report.to_csv(str(output_dir / "pipeline_smooth_fused.csv"))

        # Exact path comparison
        model_exact = Model(spec_smooth, ssp, filters=filters, precompute=False)
        report_exact = profile_pipeline(
            model_exact, params, n=n_iters, config_name="smooth_D7_exact"
        )
        print("\n" + report_exact.summary())
        report_exact.to_csv(str(output_dir / "pipeline_smooth_exact.csv"))

        speedup = report_exact.total_us / report.total_us if report.total_us > 0 else 0
        print(f"\nFused speedup: {speedup:.1f}x")

        # Stochastic model (D~137)
        spec_stoch = ParamSpec(
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.5, 3.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            sfh_field_psd_sigma=Uniform(0.01, 1.0),
            sfh_field_psd_tau_myr=Uniform(10, 500),
            met_logzsol=Uniform(-2.0, 0.5),
            dust_tau_bc=Uniform(0.0, 2.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.1),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_stoch = Model(spec_stoch, ssp, filters=filters, precompute=True)

        params_stoch = spec_stoch.sample(jax.random.PRNGKey(42))
        report_stoch = profile_pipeline(
            model_stoch, params_stoch, n=n_iters, config_name="stochastic_D137_fused"
        )
        print("\n" + report_stoch.summary())
        report_stoch.to_csv(str(output_dir / "pipeline_stochastic_fused.csv"))

    # -----------------------------------------------------------------
    # 2. Scaling tests
    # -----------------------------------------------------------------
    if run_all or args.scaling_only:
        print("\n" + "=" * 70)
        print("SCALING PROFILING")
        print("=" * 70)

        from profiling.profile_scaling import (
            plot_scaling,
            scale_bands,
            scale_dimension,
            scale_spectral,
        )
        import json

        results = {}
        results["dimension"] = scale_dimension(ssp, filters, quick=quick)
        if not quick:
            results["bands"] = scale_bands(ssp, quick=quick)
            results["spectral"] = scale_spectral(ssp, filters, quick=quick)

        with open(output_dir / "scaling_results.json", "w") as f:
            json.dump(results, f, indent=2)

        try:
            plot_scaling(results, output_dir)
        except Exception as e:
            print(f"  Plot generation failed: {e}")

    # -----------------------------------------------------------------
    # 3. Memory profiling
    # -----------------------------------------------------------------
    if run_all or args.memory_only:
        print("\n" + "=" * 70)
        print("MEMORY PROFILING")
        print("=" * 70)

        from tengri.profiling.memory import profile_memory

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_mem = Model(spec_smooth, ssp, filters=filters, precompute=True)

        mem_report = profile_memory(model_mem)
        print(mem_report.summary())
        mem_report.to_csv(str(output_dir / "memory_footprint.csv"))

    # -----------------------------------------------------------------
    # 4. Generate analysis plots
    # -----------------------------------------------------------------
    if run_all:
        print("\n" + "=" * 70)
        print("GENERATING ANALYSIS PLOTS")
        print("=" * 70)

        try:
            from profiling.analyse_results import (
                generate_text_report,
                plot_pipeline_breakdown,
            )

            for csv_file in output_dir.glob("pipeline*.csv"):
                plot_pipeline_breakdown(str(csv_file), str(output_dir))

            generate_text_report(str(output_dir))
        except Exception as e:
            print(f"  Analysis failed: {e}")

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print(f"Results in: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
