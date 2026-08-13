#!/usr/bin/env python
"""
End-to-end hierarchical PSD recovery test.

Truthvalues: sigma_true = 1.3 dex, tau_true_myr = 150.0 Myr
Configuration: z = 0.1, DPL + stochastic field SFH with n_grid=16 (D=25),
               10-band photometry SNR 20 + 8 optical lines SNR 10
Sampler: mcmc_hmc, n_leapfrog_steps=100, dense_mass_matrix=True

Run with:
  export PYTHONPATH=src
  JAX_PLATFORMS=cpu python recovery_run.py
"""

import os
import sys
import time
from pathlib import Path

import jax
import numpy as np
import psutil

# Enforce CPU-only mode to avoid MPS flakiness
os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    load_ssp_data,
)
from tengri.analysis.population_mocks import (
    assert_truth_is_discriminating,
    make_population,
)
from tengri.inference.population import (
    SharedGrid,
    fit_interim,
    shared_log_posterior,
)

# =============================================================================
# Configuration
# =============================================================================

TRUTH_SIGMA = 0.75  # dex
TRUTH_TAU_MYR = 150.0  # Myr
REDSHIFT = 0.1
N_GRID = 16  # D = 25 per galaxy with DPL
AGE_GYR = 11.0  # < 12.47 Gyr cosmic age
SNR_PHOT = 20.0
SNR_LINE = 10.0
N_LEAPFROG_STEPS = 100
DENSE_MASS_MATRIX = True

# Interim priors (bounds for the independent fits)
INTERIM_SIGMA_BOUNDS = (0.01, 1.0)
INTERIM_TAU_BOUNDS_MYR = (10.0, 500.0)

# Shared grid for hierarchical inference
GRID_SIGMA_BOUNDS = (0.01, 1.0)
GRID_TAU_BOUNDS_YR = (10.0 * 1e6, 500.0 * 1e6)  # Convert to years
N_SIGMA_NODES = 60
N_TAU_NODES = 60

# Output directory
output_dir = Path(".superpowers/sdd/2026-07-29-hierarchical-psd-recovery")
output_dir.mkdir(parents=True, exist_ok=True)
report_file = output_dir / "recovery-run-report.md"


def load_ssp():
    """Load SSP data, checking that it exists."""
    ssp_path = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    if not Path(ssp_path).exists():
        raise FileNotFoundError(f"SSP data not found at {ssp_path}")
    return load_ssp_data(ssp_path)


BANDS = [
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "wise_w1",
    "wise_w2",
    "wise_w3",
    "wise_w4",
    "hst_f160w",
]


def _build(ssp_data, *, line_flux_data=None, approx=None):
    """One build path, optionally declaring lines and/or precompute."""
    from tengri import FREE

    obs_kw = {"photometry": Photometry.from_names(BANDS)}
    if line_flux_data is not None:
        obs_kw["line_fluxes"] = line_flux_data
    build_kw = {} if approx is None else {"approx": approx}
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=Observation(**obs_kw),
        sfh={"type": ["dpl", "field"], "*": FREE, "age_gyr": 11.0},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FREE},
        neb={"type": "none"},
        redshift=Fixed(REDSHIFT),
        n_grid=N_GRID,
        **build_kw,
    )


def build_model(ssp_data, *, line_flux_data=None, approx=None):
    """Build the SEDModel for hierarchical inference.

    Declaring ``line_flux_data`` is what puts the emission lines INTO the
    likelihood. Without it the Observation carries no lines, no line adapter is
    built, and mock line fluxes are generated and then silently discarded — the
    fit is photometry-only while appearing to use lines. It is also what lets
    ``FeaturePrecomp`` build at all; without lines it raises "no emission lines
    to tabulate".
    """
    return _build(ssp_data, line_flux_data=line_flux_data, approx=approx)


def get_process_memory_mb():
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def sanity_check_chi_squared(model, mock, truth_params):
    """Verify chi-squared at truth parameters is reasonable (order of data size).

    Units mismatch or scale errors produce chi-squared > 1e30.
    A probe must assert its own setup.
    """
    # Predict at truth
    pred = model.predict_photometry(truth_params)
    pred_arr = np.asarray(pred)
    phot_obs = np.asarray(mock.table["phot_flux_obs"][0])
    phot_err = np.asarray(mock.table["phot_flux_err"][0])

    # Chi-squared for photometry
    phot_resid = (phot_obs - pred_arr) / phot_err
    phot_chi2 = np.sum(phot_resid**2)

    # Check sanity: chi-squared should be order of number of data points, not 1e60
    n_bands = len(phot_obs)
    if phot_chi2 > 1e6:
        raise ValueError(
            f"Photometry chi-squared at truth is {phot_chi2:.2e}, "
            f"expected ~{n_bands}. Units mismatch or scale error detected. STOP."
        )

    print(f"  Photometry chi-squared at truth: {phot_chi2:.2f} (n_bands={n_bands})")

    return True


def run_recovery(n_galaxies):
    """Run the full hierarchical PSD recovery pipeline."""
    print(f"\n{'=' * 80}")
    print(f"HIERARCHICAL PSD RECOVERY: N = {n_galaxies}")
    print(f"{'=' * 80}")
    print(f"Truth: σ = {TRUTH_SIGMA} dex, τ = {TRUTH_TAU_MYR} Myr")
    print(f"Interim priors: σ ∈ {INTERIM_SIGMA_BOUNDS}, τ ∈ {INTERIM_TAU_BOUNDS_MYR} Myr")
    print(f"Sampler: mcmc_hmc, L={N_LEAPFROG_STEPS}, dense_mass_matrix={DENSE_MASS_MATRIX}")
    print()

    wall_t0 = time.time()
    mem_peak = get_process_memory_mb()

    # --- Step 0: Validate truths ---
    print("[0/5] Validating injected truths...")
    try:
        assert_truth_is_discriminating(
            TRUTH_SIGMA,
            INTERIM_SIGMA_BOUNDS,
            name="sfh_field_psd_sigma",
        )
        assert_truth_is_discriminating(
            TRUTH_TAU_MYR,
            INTERIM_TAU_BOUNDS_MYR,
            name="sfh_field_psd_tau_myr",
        )
        print("  ✓ Truths discriminate from prior.")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")
        return None

    # --- Step 1: Load and build model ---
    #
    # Two passes, because the Observation's line declaration and the mock's
    # lines must agree and each needs the other. Pass 1 builds a photometry-only
    # model purely to MEASURE a line template off the forward model; pass 2
    # rebuilds declaring those lines, so make_population then reads its line set
    # from the model rather than inventing one. Skipping pass 2 is what made
    # every earlier run photometry-only.
    print("[1/5] Loading SSP data and building model...")
    ssp = load_ssp()

    bootstrap = build_model(ssp)
    boot_pop = make_population(
        bootstrap,
        n_galaxies=1,
        sigma_true=TRUTH_SIGMA,
        tau_true_myr=TRUTH_TAU_MYR,
        key=jax.random.PRNGKey(42),
        snr_phot=SNR_PHOT,
        snr_line=SNR_LINE,
    )
    line_template = boot_pop.line_flux_data()

    # approx= is deliberately OFF. Declaring lines is what lets FeaturePrecomp
    # build at all, so it was tried here — and measured a net LOSS for fitting:
    #
    #                    forward     gradient
    #   no approx        0.265 ms    0.904 ms
    #   Wave+Feature     0.193 ms    0.936 ms
    #
    # The forward speedup is real (1.37x) but the backward is worse, because the
    # LUT swaps a dense contraction for a gather whose VJP is a scatter-add. A
    # fit is ~100% backward passes — 200k gradients and zero bare forwards — so
    # the forward win never lands. See issue #1503. Pass approx= explicitly if
    # you want it for forward-only work such as mock generation or plotting.
    model = build_model(ssp, line_flux_data=line_template)
    print(
        f"  ✓ Model: D = {len(model.spec.free_params)}, n_grid = {model.spec.n_grid}, "
        f"lines = {model.observation.n_data_lines}, approx = None (see #1503)"
    )

    # --- Step 2: Generate mock population ---
    print(f"[2/5] Generating {n_galaxies} mock galaxies...")
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    mock = make_population(
        model,
        n_galaxies=n_galaxies,
        sigma_true=TRUTH_SIGMA,
        tau_true_myr=TRUTH_TAU_MYR,
        key=subkey,
        snr_phot=SNR_PHOT,
        snr_line=SNR_LINE,
    )
    print(
        f"  ✓ Generated {n_galaxies} galaxies; "
        f"Halpha absorption: {mock.n_halpha_absorption} events"
    )
    mem_peak = max(mem_peak, get_process_memory_mb())

    # --- Sanity check: chi-squared at truth ---
    print("  Sanity check: chi-squared at truth parameters...")
    sanity_check_chi_squared(model, mock, mock.truth_params[0])

    # --- Step 3: Interim fits (per-galaxy) ---
    print("[3/5] Running per-galaxy interim fits...")
    key, subkey = jax.random.split(key)
    interim = fit_interim(
        model,
        mock,
        key=subkey,
        interim_bounds={
            "sigma_bounds": INTERIM_SIGMA_BOUNDS,
            "tau_bounds_myr": INTERIM_TAU_BOUNDS_MYR,
        },
        n_leapfrog_steps=N_LEAPFROG_STEPS,
        dense_mass_matrix=DENSE_MASS_MATRIX,
        n_warmup=1000,
        n_samples=1000,
        n_chains=4,
    )
    print(f"  ✓ Interim fits completed in {interim.wall_time_s:.1f}s")
    print(f"    R-hat (max, incl. psd_xi): {max(interim.rhat.values()):.4f}")
    print(f"    Divergences (total): {np.sum(interim.n_divergent)}")
    print(
        f"    Divergences per galaxy: min={np.min(interim.n_divergent)}, "
        f"median={np.median(interim.n_divergent):.0f}, "
        f"max={np.max(interim.n_divergent)}"
    )
    mem_peak = max(mem_peak, get_process_memory_mb())

    # --- Step 4: Shared posterior (B2 method) ---
    print("[4/5] Computing shared posterior (B2 method)...")
    grid = SharedGrid.uniform(
        sigma_bounds=GRID_SIGMA_BOUNDS,
        tau_bounds_yr=GRID_TAU_BOUNDS_YR,
        n_sigma=N_SIGMA_NODES,
        n_tau=N_TAU_NODES,
    )
    log_posterior_b2, ess_b2 = shared_log_posterior(
        interim.fields, interim.times_yr, grid, method="b2"
    )

    # Normalize posterior for inference
    lnorm_b2 = jax.scipy.special.logsumexp(log_posterior_b2)
    posterior_b2 = np.exp(np.asarray(log_posterior_b2) - lnorm_b2)

    # Find posterior mode
    best_idx_b2 = np.argmax(posterior_b2)
    best_node_b2 = grid.nodes[best_idx_b2]
    sigma_mode_b2 = best_node_b2[0]
    tau_mode_b2 = best_node_b2[1] / 1e6  # Convert back to Myr

    # ESS diagnostics
    ess_at_mode_b2 = np.asarray(ess_b2.at_mode)
    ess_min_high_mass_b2 = np.asarray(ess_b2.min_high_mass)

    print("  ✓ B2 posterior computed")
    print(f"    ESS at mode: {ess_at_mode_b2}")
    print(
        f"    ESS min (top 99% mass): min={np.min(ess_min_high_mass_b2):.1f}, "
        f"median={np.median(ess_min_high_mass_b2):.1f}"
    )
    print(f"    Mode: σ = {sigma_mode_b2:.3f} dex, τ = {tau_mode_b2:.1f} Myr")

    # --- Step 5: Shared posterior (B1 method for cross-check) ---
    print("[5/5] Computing shared posterior (B1 method, cross-check)...")
    log_posterior_b1, ess_b1 = shared_log_posterior(
        interim.fields, interim.times_yr, grid, method="b1"
    )
    lnorm_b1 = jax.scipy.special.logsumexp(log_posterior_b1)
    posterior_b1 = np.exp(np.asarray(log_posterior_b1) - lnorm_b1)

    best_idx_b1 = np.argmax(posterior_b1)
    best_node_b1 = grid.nodes[best_idx_b1]
    sigma_mode_b1 = best_node_b1[0]
    tau_mode_b1 = best_node_b1[1] / 1e6

    print("  ✓ B1 posterior computed")
    print(f"    Mode: σ = {sigma_mode_b1:.3f} dex, τ = {tau_mode_b1:.1f} Myr")

    # --- Extract credible intervals ---
    sigma_b2 = grid.sigma
    tau_b2_myr = grid.tau_yr / 1e6

    # Marginalize to 1D
    posterior_b2_sigma = np.array(
        [
            np.sum(posterior_b2[i * len(grid.tau_yr) : (i + 1) * len(grid.tau_yr)])
            for i in range(len(grid.sigma))
        ]
    )
    posterior_b2_tau = np.array(
        [np.sum(posterior_b2[i :: len(grid.tau_yr)]) for i in range(len(grid.tau_yr))]
    )

    # Normalize marginals
    posterior_b2_sigma /= np.sum(posterior_b2_sigma)
    posterior_b2_tau /= np.sum(posterior_b2_tau)

    # 68% credible intervals (16th and 84th percentiles)
    sigma_cdf = np.cumsum(posterior_b2_sigma)
    tau_cdf = np.cumsum(posterior_b2_tau)

    idx_sigma_16 = np.searchsorted(sigma_cdf, 0.16)
    idx_sigma_84 = np.searchsorted(sigma_cdf, 0.84)
    sigma_16 = sigma_b2[idx_sigma_16]
    sigma_84 = sigma_b2[idx_sigma_84]
    sigma_med = sigma_b2[np.searchsorted(sigma_cdf, 0.5)]

    idx_tau_16 = np.searchsorted(tau_cdf, 0.16)
    idx_tau_84 = np.searchsorted(tau_cdf, 0.84)
    tau_16 = tau_b2_myr[idx_tau_16]
    tau_84 = tau_b2_myr[idx_tau_84]
    tau_med = tau_b2_myr[np.searchsorted(tau_cdf, 0.5)]

    # Check if truths fall inside intervals
    sigma_truth_in = sigma_16 <= TRUTH_SIGMA <= sigma_84
    tau_truth_in = tau_16 <= TRUTH_TAU_MYR <= tau_84

    wall_t1 = time.time()
    wall_total = wall_t1 - wall_t0
    wall_per_galaxy = wall_total / n_galaxies

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Parameter':<15} {'Truth':>10} {'Median':>10} {'68% Interval':>25} {'In CI':>8}")
    print("-" * 70)
    print(
        f"{'σ [dex]':<15} {TRUTH_SIGMA:>10.3f} {sigma_med:>10.3f} "
        f"[{sigma_16:.3f}, {sigma_84:.3f}]{'':<12} {'YES' if sigma_truth_in else 'NO':>8}"
    )
    print(
        f"{'τ [Myr]':<15} {TRUTH_TAU_MYR:>10.1f} {tau_med:>10.1f} "
        f"[{tau_16:.1f}, {tau_84:.1f}]{'':<15} {'YES' if tau_truth_in else 'NO':>8}"
    )

    print("\nEffective Sample Size (B2):")
    print(f"  At mode (median over galaxies): {float(np.median(ess_at_mode_b2)):.1f}")
    print(f"  Min (top 99% mass): {float(np.min(ess_min_high_mass_b2)):.1f}")
    print(f"  Median (top 99% mass): {float(np.median(ess_min_high_mass_b2)):.1f}")

    print("\nConvergence:")
    _rh = interim.rhat
    _shared_rh = {k: v for k, v in _rh.items() if "psd_sigma" in k or "psd_tau" in k}
    _xi_rh = {k: v for k, v in _rh.items() if "psd_xi" in k or "field_xi" in k}
    _other_rh = {k: v for k, v in _rh.items() if k not in _shared_rh and k not in _xi_rh}
    print(f"  Max R-hat (all params, incl. psd_xi): {max(_rh.values()):.4f}")
    for _k, _v in sorted(_shared_rh.items()):
        print(f"    SHARED  {_k:26s} R-hat {_v:.4f}")
    if _xi_rh:
        print(f"    latents psd_xi (max over modes)    R-hat {max(_xi_rh.values()):.4f}")
    if _other_rh:
        print(f"    other   (max over per-galaxy)      R-hat {max(_other_rh.values()):.4f}")
    print(f"  Total divergences: {np.sum(interim.n_divergent)}")

    print("\nWall clock:")
    print(f"  Total: {wall_total:.1f}s ({wall_total / 60:.1f}m)")
    print(f"  Per galaxy: {wall_per_galaxy:.1f}s")
    print(f"  Peak memory: {mem_peak:.0f} MB")

    print("\nB1 vs B2 comparison:")
    print(f"  B1 mode: σ = {sigma_mode_b1:.3f} dex, τ = {tau_mode_b1:.1f} Myr")
    print(f"  B2 mode: σ = {sigma_mode_b2:.3f} dex, τ = {tau_mode_b2:.1f} Myr")
    sigma_b1_b2_diff = abs(sigma_mode_b1 - sigma_mode_b2)
    tau_b1_b2_diff = abs(tau_mode_b1 - tau_mode_b2)
    print(f"  Difference: Δσ = {sigma_b1_b2_diff:.3f} dex, Δτ = {tau_b1_b2_diff:.1f} Myr")

    return {
        "n_galaxies": n_galaxies,
        "sigma_med": float(sigma_med),
        "sigma_16": float(sigma_16),
        "sigma_84": float(sigma_84),
        "sigma_truth_in": bool(sigma_truth_in),
        "tau_med": float(tau_med),
        "tau_16": float(tau_16),
        "tau_84": float(tau_84),
        "tau_truth_in": bool(tau_truth_in),
        "ess_at_mode": float(np.median(ess_at_mode_b2)),
        "ess_min_high_mass": float(np.min(ess_min_high_mass_b2)),
        "ess_median_high_mass": float(np.median(ess_min_high_mass_b2)),
        "rhat_max": float(max(interim.rhat.values())),
        "rhat_shared_max": float(max(_shared_rh.values())) if _shared_rh else float("nan"),
        "rhat_xi_max": float(max(_xi_rh.values())) if _xi_rh else float("nan"),
        "n_divergent_total": int(np.sum(interim.n_divergent)),
        "wall_total_s": float(wall_total),
        "wall_per_galaxy_s": float(wall_per_galaxy),
        "peak_memory_mb": float(mem_peak),
        "sigma_b1": float(sigma_mode_b1),
        "tau_b1_myr": float(tau_mode_b1),
        "sigma_b2": float(sigma_mode_b2),
        "tau_b2_myr": float(tau_mode_b2),
    }


if __name__ == "__main__":
    try:
        # Start with N=4 shakedown
        result = run_recovery(n_galaxies=4)

        if result is None:
            print("\n[ERROR] Recovery failed.")
            sys.exit(1)

        # Write brief report
        report_content = f"""# Hierarchical PSD Recovery Run
## 2026-07-29

### Configuration
- Truth: σ = {TRUTH_SIGMA} dex, τ = {TRUTH_TAU_MYR} Myr
- N galaxies: {result["n_galaxies"]}
- Redshift: {REDSHIFT}
- SFH: DPL + field, n_grid={N_GRID}, D=25
- Photometry: 10 bands, SNR={SNR_PHOT}
- Lines: 8 optical, SNR={SNR_LINE}
- Sampler: mcmc_hmc, L={N_LEAPFROG_STEPS}, dense_mass_matrix={DENSE_MASS_MATRIX}

### Results

| Parameter | Truth | Median | 68% Interval | Truth In CI |
|-----------|-------|--------|--------------|-------------|
| σ [dex] | {TRUTH_SIGMA:.3f} | {result["sigma_med"]:.3f} | [{result["sigma_16"]:.3f}, {result["sigma_84"]:.3f}] | {"✓" if result["sigma_truth_in"] else "✗"} |
| τ [Myr] | {TRUTH_TAU_MYR:.1f} | {result["tau_med"]:.1f} | [{result["tau_16"]:.1f}, {result["tau_84"]:.1f}] | {"✓" if result["tau_truth_in"] else "✗"} |

### Diagnostics

**Effective Sample Size (B2):**
- At mode: {result["ess_at_mode"]:.1f}
- Min (top 99% mass): {result["ess_min_high_mass"]:.1f}
- Median (top 99% mass): {result["ess_median_high_mass"]:.1f}

**Convergence:**
- Max R-hat (incl. psd_xi): {result["rhat_max"]:.4f}
- Total divergences: {result["n_divergent_total"]}

**Performance:**
- Total wall clock: {result["wall_total_s"]:.1f}s ({result["wall_total_s"] / 60:.1f}m)
- Per galaxy: {result["wall_per_galaxy_s"]:.1f}s
- Peak RSS: {result["peak_memory_mb"]:.0f} MB

### B1 vs B2 Comparison

- B1 mode: σ = {result["sigma_b1"]:.3f} dex, τ = {result["tau_b1_myr"]:.1f} Myr
- B2 mode: σ = {result["sigma_b2"]:.3f} dex, τ = {result["tau_b2_myr"]:.1f} Myr
- Δσ = {abs(result["sigma_b1"] - result["sigma_b2"]):.3f} dex
- Δτ = {abs(result["tau_b1_myr"] - result["tau_b2_myr"]):.1f} Myr
"""

        report_file.write_text(report_content)
        print(f"\n✓ Report written to {report_file}")

    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
