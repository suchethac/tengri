#!/usr/bin/env python3
"""Figure 8: End-to-end gradient sensitivity.

Shows the Jacobian ∂flux/∂θ for all physical parameters and derived
quantities, demonstrating that the forward model is fully differentiable
and that photometric bands carry parameter-specific information.

Usage:
    python analysis/fig08_gradient_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    FIG_DIR,
    PAPER_FIG_DIR,
    setup_matplotlib,
)

from tengri import Gaussian, Model, ParamSpec, Uniform, load_ssp_data


def compute_jacobian(model, params, filter_names):
    """Compute ∂flux/∂θ for all free physical parameters."""
    free_params = [k for k in params if k != "psd_xi"]

    def flux_fn(p):
        full = dict(params)
        full.update(p)
        return model.predict_photometry(full)

    subset = {k: params[k] for k in free_params}
    jac = jax.jacfwd(flux_fn)(subset)

    # jac[param_name] has shape (n_bands,) or (n_bands, ...)
    return {k: np.array(v) for k, v in jac.items()}


def plot_gradient_heatmap(jac, param_names, filter_names):
    """Plot gradient sensitivity heatmap."""
    plt = setup_matplotlib()

    n_bands = len(filter_names)
    n_params = len(param_names)

    # Build matrix: rows=params, cols=bands
    # Normalize each row by its max for visibility
    mat = np.zeros((n_params, n_bands))
    for i, p in enumerate(param_names):
        row = jac[p]
        if row.ndim > 1:
            row = row.flatten()[:n_bands]
        mat[i, :] = row

    # Normalize rows
    mat_norm = np.zeros_like(mat)
    for i in range(n_params):
        max_abs = np.max(np.abs(mat[i]))
        if max_abs > 0:
            mat_norm[i] = mat[i] / max_abs

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 0.5 * n_params + 2), gridspec_kw={"width_ratios": [1, 1]}
    )

    # Left: raw Jacobian (log scale)
    mat_log = np.sign(mat) * np.log10(np.abs(mat) + 1e-35)
    im1 = ax1.imshow(
        mat_log,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.max(np.abs(mat_log)),
        vmax=np.max(np.abs(mat_log)),
    )
    ax1.set_xticks(range(n_bands))
    ax1.set_xticklabels(filter_names, rotation=45, ha="right")
    ax1.set_yticks(range(n_params))

    label_map = {
        "sfh_dpl_alpha": r"$\alpha$ (falling slope)",
        "sfh_dpl_beta": r"$\beta$ (rising slope)",
        "sfh_dpl_tau_gyr": r"$\tau_{\rm peak}$ (Gyr)",
        "sfh_dpl_log_total_mass": r"$\log M_{\star}$ (Msun)",
        "sfh_field_psd_sigma": r"$\sigma_{\rm PSD}$",
        "sfh_field_psd_tau_myr": r"$\tau_{\rm PSD}$ (Myr)",
        "met_logzsol": r"$\log Z/Z_\odot$",
        "dust_tau_bc": r"$\hat{\tau}_{V,1}$ (birth cloud)",
        "dust_tau_diff": r"$\hat{\tau}_{V,2}$ (diffuse)",
    }
    ax1.set_yticklabels([label_map.get(p, p) for p in param_names])
    ax1.set_title("Jacobian  (signed log scale)")
    plt.colorbar(im1, ax=ax1, shrink=0.8, label=r"sgn$\times\log_{10}|\partial f/\partial\theta|$")

    # Right: normalized sensitivity
    im2 = ax2.imshow(np.abs(mat_norm), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_xticks(range(n_bands))
    ax2.set_xticklabels(filter_names, rotation=45, ha="right")
    ax2.set_yticks(range(n_params))
    ax2.set_yticklabels([label_map.get(p, p) for p in param_names])
    ax2.set_title("Normalized sensitivity  (|∂f/∂θ| / max)")
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="Relative sensitivity")

    # Annotate values
    for i in range(n_params):
        for j in range(n_bands):
            val = np.abs(mat_norm[i, j])
            if val > 0.3:
                ax2.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val > 0.7 else "black",
                )

    plt.tight_layout()
    return fig


def main():
    print("Computing end-to-end gradient sensitivity...")

    # Use a star-forming galaxy at z=0.1 with all params free
    spec_kwargs = dict(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(1.0, 8.0),
        sfh_dpl_log_total_mass=Uniform(10.0, 11.5),
        sfh_field_psd_sigma=Uniform(0.1, 4.0),
        sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
        met_logzsol=Gaussian(-0.5, 0.3, lo=-2.0, hi=0.0),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.0),
        dust_slope=-0.7,
        redshift=0.1,
        stochastic=True,
        n_grid=128,
    )
    spec = ParamSpec(**spec_kwargs)
    ssp = load_ssp_data(
        str(
            Path(__file__).resolve().parent.parent
            / "data"
            / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
        )
    )

    filter_names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    from tengri import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(filter_names))
    model = Model(spec, ssp, observation=obs)

    # Evaluate at a specific point
    key = jax.random.PRNGKey(7)
    params = spec.sample(key)

    # Physical params only (exclude psd_xi)
    param_names = [
        k for k in sorted(params.keys()) if k != "psd_xi" and k not in ("dust_slope", "redshift")
    ]

    print(f"  Parameters: {param_names}")
    print(f"  Filters: {filter_names}")

    jac = compute_jacobian(model, params, filter_names)

    print("  Jacobian computed. Plotting...")
    fig = plot_gradient_heatmap(jac, param_names, filter_names)

    out_path = FIG_DIR / "fig08_gradient_sensitivity.pdf"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")

    if PAPER_FIG_DIR.exists():
        paper_path = PAPER_FIG_DIR / "fig08_gradient_sensitivity.pdf"
        fig.savefig(paper_path)
        print(f"Saved: {paper_path}")


if __name__ == "__main__":
    main()
