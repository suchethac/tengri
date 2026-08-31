#!/usr/bin/env python3
"""Figure 8 — Gradient sensitivity (Jacobian and Fisher matrix)."""

import json
import os
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from configs import config_II, load_ssp_for

import tengri
from tengri import (
    Photometry,
)

jax.config.update("jax_enable_x64", True)

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def resolve_filter_names():
    """Map requested filter names to tengri's filter list."""
    filter_map = {
        "HST ACS F435W": "hst_f435w",
        "HST ACS F606W": "hst_f606w",
        "HST ACS F775W": "hst_f775w",
        "HST ACS F814W": "hst_f814w",
        "HST ACS F850LP": "hst_f850lp",
        "HST WFC3 F105W": "hst_f105w",
        "HST WFC3 F125W": "hst_f125w",
        "HST WFC3 F160W": "hst_f160w",
        "VISTA Ks": "vista_ks",
        "Spitzer IRAC 3.6 µm": "irac_36",
        "Spitzer IRAC 4.5 µm": "irac_45",
        "Spitzer IRAC 5.8 µm": "irac_58",
        "Spitzer IRAC 8.0 µm": "irac_80",
    }

    all_filters = tengri.list_filters()
    filter_lookup = {}
    for item in all_filters:
        name = item.get("name", "").lower()
        alias = item.get("alias", "")
        if alias:
            for a in alias.split(","):
                a = a.strip().lower()
                filter_lookup[a] = item
        filter_lookup[name] = item

    used_filters = []
    for req_name, search_key in filter_map.items():
        search_lower = search_key.lower()
        if search_lower in filter_lookup:
            item = filter_lookup[search_lower]
            alias = item.get("alias", "")
            if alias:
                main_alias = alias.split(",")[0].strip()
            else:
                main_alias = search_lower
            used_filters.append(main_alias)
            print(f"  ✓ {req_name:30s} -> {main_alias}")
        else:
            print(f"  ✗ {req_name:30s} -> {search_key} NOT FOUND (skipped)")

    if not used_filters:
        raise ValueError("No filters could be resolved!")

    return used_filters


def compute_jacobian_linear_flux(model, params):
    """
    Compute ∂f_b/∂θ_k (Jacobian of linear flux, bands x params).
    """
    free_params = model.spec.free_params

    def flux_fn(theta_dict):
        full_params = dict(params)
        full_params.update(theta_dict)
        flux = model.predict_photometry(full_params)
        return flux

    theta_subset = {k: params[k] for k in free_params}
    jac_fn = jax.jacfwd(flux_fn)
    jac_dict = jac_fn(theta_subset)

    flux_at_point = model.predict_photometry(params)
    n_bands = len(flux_at_point)
    n_params = len(free_params)

    J = np.zeros((n_bands, n_params))
    for i, param_name in enumerate(free_params):
        J[:, i] = np.asarray(jac_dict[param_name])

    return J, flux_at_point, free_params


def compute_log_jacobian(J, flux):
    """
    Convert linear flux Jacobian to log10 flux Jacobian.
    ∂log10(f)/∂θ = (1/ln(10)) * (1/f) * ∂f/∂θ
    """
    log_J = np.zeros_like(J)
    for i in range(J.shape[0]):
        if flux[i] > 0:
            log_J[i, :] = (1.0 / np.log(10.0)) * J[i, :] / flux[i]
    return log_J


def check_jacobian(log_J, free_params):
    """
    Run checks on log-Jacobian matrix.
    """
    messages = []
    passed = True

    print("\n--- CHECK 1: Column magnitudes (log-Jacobian) ---")
    for i, param_name in enumerate(free_params):
        col = log_J[:, i]
        col_max = np.max(np.abs(col))
        if col_max < 1e-12:
            msg = f"FAIL: Column {i} ({param_name}) has max|entry| = {col_max:.3e} < 1e-12"
            messages.append(msg)
            print(msg)
            passed = False
        else:
            msg = f"PASS: Column {i} ({param_name:30s}) max|entry| = {col_max:.3e}"
            messages.append(msg)
            print(msg)

    print("\n--- CHECK 2: Matrix rank ---")
    rank_J = np.linalg.matrix_rank(log_J)
    n_params = log_J.shape[1]
    if rank_J == n_params:
        msg = f"PASS: rank(J) = {rank_J} = n_params ({n_params})"
        messages.append(msg)
        print(msg)
    else:
        msg = f"FAIL: rank(J) = {rank_J} != n_params ({n_params})"
        messages.append(msg)
        print(msg)
        passed = False

    return passed, messages


def compute_fisher_matrix(J, flux, free_params, frac_uncertainty=0.05):
    """
    Compute Fisher matrix F = J^T N^-1 J using linear flux Jacobian.
    """
    n_params = len(free_params)
    messages = []

    sigma = frac_uncertainty * flux
    N_inv = np.diag(1.0 / (sigma**2))
    fisher = J.T @ N_inv @ J

    print("\n--- CHECK 3: Fisher matrix condition number ---")
    try:
        cond_F = np.linalg.cond(fisher)
        msg = f"Fisher condition number: {cond_F:.3e}"
        messages.append(msg)
        print(msg)

        if cond_F > 1e12:
            print(
                "WARNING: Condition number > 1e12; adding unit prior regularization (standardized coordinates)"
            )
            msg_reg = "Regularized with unit prior (I) in standardized coordinates"
            messages.append(msg_reg)
            fisher_reg = fisher + np.eye(n_params)
            fisher_inv = np.linalg.inv(fisher_reg)
        else:
            fisher_inv = np.linalg.inv(fisher)

    except np.linalg.LinAlgError as e:
        print(f"ERROR: Could not invert Fisher matrix: {e}")
        messages.append(f"ERROR: Fisher matrix inversion failed: {e}")
        return fisher, np.eye(n_params), np.zeros(n_params), messages

    forecast_sigmas = np.sqrt(np.diag(fisher_inv))

    cov_matrix = fisher_inv
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
    np.fill_diagonal(corr_matrix, 1.0)

    return fisher, corr_matrix, forecast_sigmas, messages


def plot_figures(
    log_J,
    J,
    flux,
    fisher,
    corr_matrix,
    forecast_sigmas,
    used_filters,
    free_params,
    check_messages,
    fisher_messages,
):
    """Plot figures from precomputed data (no recomputation)."""
    n_bands = J.shape[0]
    n_params = J.shape[1]

    print("\n11. Creating final publication figure...")
    fig_final = plt.figure(figsize=(7.0, 3.4), dpi=150)

    # Use manual margins instead of constrained_layout
    # Increase wspace significantly to prevent colorbar overlap
    gs = fig_final.add_gridspec(
        1, 2, width_ratios=[1.2, 1.0], wspace=0.75, left=0.08, right=0.98, top=0.95, bottom=0.12
    )
    ax_a = fig_final.add_subplot(gs[0])
    ax_b = fig_final.add_subplot(gs[1])

    # Panel (a): Log-Jacobian
    mat = log_J.T
    vmax = np.max(np.abs(mat))
    im_a = ax_a.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_a.set_xticks(range(n_bands))
    ax_a.set_xticklabels(used_filters, rotation=45, ha="right", fontsize=9)
    ax_a.set_yticks(range(n_params))
    label_map = {
        "sfh_dpl_alpha": r"$\alpha$",
        "sfh_dpl_beta": r"$\beta$",
        "sfh_dpl_tau_gyr": r"$\tau$ (Gyr)",
        "sfh_dpl_age_gyr": "age (Gyr)",
        "sfh_dpl_log_total_mass": r"$\log M_*$",
        "met_logzsol": r"$\log Z$",
        "dust_tau_bc": r"$\tau_{\rm bc}$",
        "dust_tau_diff": r"$\tau_{\rm diff}$",
    }
    param_labels = [label_map.get(p, p) for p in free_params]
    ax_a.set_yticklabels(param_labels, fontsize=10)
    ax_a.set_title("(a) Jacobian", fontsize=10, fontweight="bold")
    ax_a.set_xlabel("Filter", fontsize=10)
    ax_a.set_ylabel("Parameter", fontsize=10)
    ax_a.tick_params(axis="both", which="major", labelsize=9)
    cbar_a = plt.colorbar(im_a, ax=ax_a, shrink=0.9)
    cbar_a.set_label(r"∂$\log_{10}$ f / ∂θ", fontsize=10)
    cbar_a.ax.tick_params(labelsize=9)

    # Panel (b): Fisher correlation matrix
    im_b = ax_b.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_b.set_xticks(range(n_params))
    ax_b.set_yticks(range(n_params))
    ax_b.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=10)
    ax_b.set_yticklabels(param_labels, fontsize=10)
    ax_b.set_title("(b) Fisher correlation", fontsize=10, fontweight="bold")
    ax_b.tick_params(axis="both", which="major", labelsize=9)
    cbar_b = plt.colorbar(im_b, ax=ax_b, shrink=0.9)
    cbar_b.set_label("Correlation", fontsize=10)
    cbar_b.ax.tick_params(labelsize=9)

    fig_final.savefig(OUTPUT_DIR / "fig08_gradient_sensitivity.pdf", dpi=150, bbox_inches="tight")
    fig_final.savefig(OUTPUT_DIR / "fig08_gradient_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"   ✓ Saved to {OUTPUT_DIR / 'fig08_gradient_sensitivity.pdf'}")
    print(f"   ✓ Saved to {OUTPUT_DIR / 'fig08_gradient_sensitivity.png'}")


def main():
    print("=" * 80)
    print("Figure 8: Gradient Sensitivity (Jacobian and Fisher Matrix)")
    print("=" * 80)

    z = 1.1
    frac_uncertainty = 0.05

    eval_point = {
        "sfh_dpl_log_total_mass": 10.5,
        "sfh_dpl_age_gyr": 4.0,
        "sfh_dpl_tau_gyr": 2.0,
        "sfh_dpl_alpha": 2.0,
        "sfh_dpl_beta": 1.0,
        "met_logzsol": -0.3,
        "dust_tau_bc": 1.0,
        "dust_tau_diff": 0.5,
    }

    print("\n1. Resolving filters...")
    used_filters = resolve_filter_names()
    print(f"\n   Resolved {len(used_filters)} filters:")
    for name in used_filters:
        print(f"     - {name}")

    print("\n2. Loading SSP grid...")
    ssp = load_ssp_for("II")
    print("   SSP grid loaded")

    print("\n3. Building observation...")
    obs = tengri.Observation(photometry=Photometry.from_names(used_filters))
    print(f"   Observation: {len(used_filters)} bands")

    print(f"\n4. Building model (Config II, z={z})...")
    model = config_II(ssp, obs, z)
    print(f"   Model built with {len(model.spec.free_params)} free parameters")
    print(f"   Free params: {model.spec.free_params}")

    print("\n5. Setting evaluation point (fixed, physically realistic)...")
    params_at_eval = dict(eval_point)
    print("\n   Evaluation point:")
    for param_name in sorted(params_at_eval.keys()):
        print(f"     {param_name:30s} = {params_at_eval[param_name]:.6f}")

    print("\n6. Computing Jacobian (linear flux units)...")
    J, flux, free_params = compute_jacobian_linear_flux(model, params_at_eval)
    print(f"   Jacobian shape: {J.shape} (bands x params)")
    print(f"   Flux range: {flux.min():.3e} to {flux.max():.3e} erg/s/cm^2/Hz")

    print("\n7. Computing log-Jacobian (∂log10(f)/∂θ)...")
    log_J = compute_log_jacobian(J, flux)
    print("   Log-Jacobian computed")

    print("\n8. Running Jacobian checks (on log-Jacobian)...")
    checks_passed, check_messages = check_jacobian(log_J, free_params)
    if not checks_passed:
        print("\n*** JACOBIAN CHECKS FAILED ***")
        for msg in check_messages:
            if "FAIL" in msg:
                print(msg)
        print("\nTerminating run due to failed checks.")
        return None

    print("\n9. Computing Fisher matrix (using linear flux Jacobian)...")
    fisher, corr_matrix, forecast_sigmas, fisher_messages = compute_fisher_matrix(
        J, flux, free_params, frac_uncertainty
    )

    print("\n   Forecast 1-sigma marginal uncertainties:")
    for i, param_name in enumerate(free_params):
        print(f"     {param_name:30s} : {forecast_sigmas[i]:.8f}")

    print("\n10. Measuring timing...")
    for _ in range(3):
        _ = model.predict_photometry(params_at_eval)

    t_forward = []
    for _ in range(20):
        t0 = time.time()
        _ = model.predict_photometry(params_at_eval)
        t_forward.append(time.time() - t0)
    t_forward_median = np.median(t_forward)

    def compute_jac():
        _, _, _ = compute_jacobian_linear_flux(model, params_at_eval)

    for _ in range(3):
        compute_jac()
    t_jac = []
    for _ in range(20):
        t0 = time.time()
        compute_jac()
        t_jac.append(time.time() - t0)
    t_jac_median = np.median(t_jac)

    print(f"\n   Forward pass median: {t_forward_median:.6f} s")
    print(f"   Jacobian median:     {t_jac_median:.6f} s")
    print(f"   Ratio (Jac/Forward): {t_jac_median / t_forward_median:.2f}x")

    # Plot the figures
    plot_figures(
        log_J,
        J,
        flux,
        fisher,
        corr_matrix,
        forecast_sigmas,
        used_filters,
        free_params,
        check_messages,
        fisher_messages,
    )

    print("\n12. Saving results JSON...")
    results = {
        "configuration": "II",
        "redshift": z,
        "n_bands": J.shape[0],
        "n_free_params": J.shape[1],
        "filter_names": used_filters,
        "parameter_names": free_params,
        "evaluation_point": {
            "description": "representative point chosen inside the support",
            "values": eval_point,
        },
        "jacobian_log_flux": log_J.tolist(),
        "jacobian_linear_flux": J.tolist(),
        "flux": flux.tolist(),
        "fisher_matrix": fisher.tolist(),
        "correlation_matrix": corr_matrix.tolist(),
        "forecast_sigmas": forecast_sigmas.tolist(),
        "forecast_sigma_by_param": {
            param_name: float(forecast_sigmas[i]) for i, param_name in enumerate(free_params)
        },
        "noise_model": {
            "type": "fractional",
            "fractional_uncertainty": frac_uncertainty,
        },
        "checks": {
            "jacobian_checks_passed": checks_passed,
            "messages": check_messages + fisher_messages,
        },
        "timings": {
            "forward_pass_median_s": float(t_forward_median),
            "jacobian_median_s": float(t_jac_median),
            "jacobian_to_forward_ratio": float(t_jac_median / t_forward_median),
        },
        "metadata": {
            "date": "2026-08-30",
            "jax_version": jax.__version__,
            "platform": "CPU",
            "ssp_grid": "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0",
        },
    }

    json_path = RESULTS_DIR / "fig08_gradient_sensitivity_data.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ Saved to {json_path}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(
        f"\nFigures:\n  {OUTPUT_DIR / 'fig08_gradient_sensitivity.pdf'}\n  {OUTPUT_DIR / 'fig08_gradient_sensitivity.png'}"
    )
    print(f"\nResults:\n  {json_path}")

    return results


if __name__ == "__main__":
    results = main()
