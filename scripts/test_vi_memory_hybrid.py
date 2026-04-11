"""Memory test: VI with hybrid (precomputed) vs auto (compositional) modes.

Runs MAP + VI and measures peak RSS.
Must run each mode in a separate process for clean measurement.

Usage:
    source .venv/bin/activate
    JAX_PLATFORMS=cpu python scripts/test_vi_memory_hybrid.py [mode]
    # mode: "auto", "compositional", "hybrid", "exact" (default: "auto")
"""

import os
import resource
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    print(f"=== Prediction mode: {mode} ===")

    rss0 = rss_gb()
    print(f"[0] Baseline RSS: {rss0:.2f} GB")

    # --- Import + SSP ---
    from tengri import Fitter, Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
    from tengri.models.sps.dsps_wrapper import load_ssp_data

    ssp_path = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    if not os.path.exists(ssp_path):
        ssp_path = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    if not os.path.exists(ssp_path):
        print("ERROR: No SSP file found")
        sys.exit(1)

    ssp_data = load_ssp_data(ssp_path)

    rss1 = rss_gb()
    print(f"[1] After imports + SSP load: {rss1:.2f} GB")

    # --- Build model (photometry, dense_basis, 5 SDSS filters) ---
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
    )
    spec = Parameters(
        mean_sfh_type="dense_basis",
        sfh_db_log_total_mass=Uniform(8, 12),
        sfh_db_log_sfr_inst=Uniform(-3, 3),
        sfh_db_tx_frac_0=Uniform(0.05, 0.95),
        sfh_db_tx_frac_1=Uniform(0.05, 0.95),
        sfh_db_tx_frac_2=Uniform(0.05, 0.95),
        met_logzsol=Uniform(-2, 0.2),
        dust_tau_bc=Uniform(0, 2),
        dust_tau_diff=Uniform(0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )
    model = SEDModel(spec, ssp_data, observation=obs)

    rss2 = rss_gb()
    print(f"[2] After model build: {rss2:.2f} GB")

    # Check which kernels are available
    has_comp = model._compositional.photometry is not None
    has_hybrid = model._hybrid.photometry is not None
    print(f"    Compositional kernel: {has_comp}")
    print(f"    Hybrid kernel: {has_hybrid}")

    # --- Mock data ---
    key = jax.random.PRNGKey(42)
    true_params = spec.sample(key)
    true_flux = model.predict_photometry(true_params, mode="exact")
    noise = true_flux * 0.05
    mock_flux = true_flux + noise * jax.random.normal(key, shape=true_flux.shape)

    rss3 = rss_gb()
    print(f"[3] After mock data: {rss3:.2f} GB")

    # --- Fitter ---
    fitter = Fitter(model, mock_flux, noise)

    rss4 = rss_gb()
    print(f"[4] After fitter build: {rss4:.2f} GB")

    # --- MAP ---
    t0 = time.perf_counter()
    fitter.run("map", n_steps=200, verbose=False)
    t_map = time.perf_counter() - t0

    rss5 = rss_gb()
    print(f"[5] After MAP (200 steps): {rss5:.2f} GB  ({t_map:.1f}s)")

    # --- VI ---
    t0 = time.perf_counter()
    fitter.run(
        "vi",
        n_iterations=6,
        n_samples=3,
        verbose=False,
    )
    t_vi = time.perf_counter() - t0

    rss6 = rss_gb()
    print(f"[6] After VI (6 iter, 3 samples): {rss6:.2f} GB  ({t_vi:.1f}s)")

    # --- NUTS ---
    t0 = time.perf_counter()
    fitter.run(
        "mcmc_nuts",
        n_warmup=50,
        n_samples=50,
        verbose=False,
    )
    t_nuts = time.perf_counter() - t0

    rss7 = rss_gb()
    print(f"[7] After NUTS (50+50): {rss7:.2f} GB  ({t_nuts:.1f}s)")

    # --- Raytrace ---
    t0 = time.perf_counter()
    fitter.run(
        "mcmc_raytrace",
        n_steps=100,
        verbose=False,
    )
    t_rt = time.perf_counter() - t0

    rss8 = rss_gb()
    print(f"[8] After Raytrace (100 steps): {rss8:.2f} GB  ({t_rt:.1f}s)")

    # --- Summary ---
    print()
    print(f"{'Stage':<35} {'RSS (GB)':>10} {'Time':>10}")
    print("-" * 57)
    print(f"{'Baseline':<35} {rss0:>10.2f}")
    print(f"{'After imports + SSP load':<35} {rss1:>10.2f}")
    print(f"{'After model build':<35} {rss2:>10.2f}")
    print(f"{'After mock data':<35} {rss3:>10.2f}")
    print(f"{'After fitter build':<35} {rss4:>10.2f}")
    print(f"{'After MAP (200 steps)':<35} {rss5:>10.2f} {t_map:>9.1f}s")
    print(f"{'After VI (6 iter, 3 samples)':<35} {rss6:>10.2f} {t_vi:>9.1f}s")
    print(f"{'After NUTS (50+50)':<35} {rss7:>10.2f} {t_nuts:>9.1f}s")
    print(f"{'After Raytrace (100 steps)':<35} {rss8:>10.2f} {t_rt:>9.1f}s")
    print()
    print(f"Peak RSS: {rss8:.2f} GB")
    print(f"Total memory delta: {rss8 - rss0:.2f} GB")


if __name__ == "__main__":
    main()
