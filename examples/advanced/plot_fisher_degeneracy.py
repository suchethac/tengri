"""
Age-Dust-Metallicity Degeneracy: Fisher Analysis
=================================================

The Cramér-Rao bound from the Fisher Information Matrix shows that SDSS
5-band photometry alone cannot separately constrain age, dust, and
metallicity. Adding NIR or MIR bands breaks the degeneracy by factors of
2–5×, quantifying the information gain from multiwavelength coverage.

Reference: Fisher Information Matrix in parameter estimation; see
Conroy 2013 (ARA&A, 51, 393) for SED fitting context.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.diagnostics.fisher import compute_fisher_matrix, fisher_parameter_errors

tengri.analysis.plotting.setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()

_FILTER_DIR = next(
    (
        str(d)
        for d in [
            Path("data/filters"),
            Path("../data/filters"),
            Path("../../data/filters"),
            Path("../../../data/filters"),
        ]
        if d.exists()
    ),
    "data/filters",
)

FILTER_SETS = {
    "SDSS (5)": ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    "+ NIR (8)": [
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "2mass_j",
        "2mass_h",
        "2mass_ks",
    ],
    "+ MIR (10)": [
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "2mass_j",
        "2mass_h",
        "2mass_ks",
        "wise_w1",
        "wise_w2",
    ],
}

fisher_params = ["met_logzsol", "dust_tau_bc", "dust_tau_diff"]
PARAM_LABELS = [r"$\log(Z/Z_\odot)$", r"$\tau_{\rm bc}$", r"$\tau_{\rm diff}$"]
COLORS_BAR = ["#4477AA", "#EE6677", "#228833"]

key = jax.random.PRNGKey(42)
true_params = {
    "sfh_tsnorm_log_total_mass": 10.5,
    "sfh_tsnorm_peak_lbt_gyr": 4.0,
    "sfh_tsnorm_width_gyr": 2.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.8,
    "dust_tau_diff": 0.4,
    "dust_slope": -0.7,
    "redshift": 0.1,
}

sigmas = {}
first_failure: Exception | None = None
for fname, filters in FILTER_SETS.items():
    try:
        obs = tengri.Observation(
            photometry=tengri.Photometry.from_names(filters, cache_dir=_FILTER_DIR)
        )
        mdl = tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": tengri.FIXED},
            dust={"type": "two_component", "all_params": tengri.FIXED},
            redshift=tengri.Fixed(0.1),
        )
        phot = jnp.abs(mdl.predict_photometry(true_params))
        noise = phot / 20.0
        fim, _ = compute_fisher_matrix(
            mdl, true_params, noise, data_type="photometry", param_names=fisher_params
        )
        errs = np.array(fisher_parameter_errors(fim))
        errs = np.where(np.isfinite(errs) & (errs > 0), errs, 5.0)
        sigmas[fname] = np.minimum(errs, 5.0)
    except Exception as e:
        if first_failure is None:
            first_failure = e
        print(f"[{fname}] skipped: {e}")

# This guard was already here and is the pattern the other examples now follow.
# It only gains the cause: "check filter availability" was a guess at why, and
# when the reason was something else (a dust-law KeyError, say) it sent the
# reader looking in the wrong place.
if not sigmas:
    raise RuntimeError(
        f"Fisher computation failed for every filter set. First failure: "
        f"{type(first_failure).__name__}: {first_failure}"
    ) from first_failure

x = np.arange(len(fisher_params))
width = 0.22
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (fname, sigma_arr) in enumerate(sigmas.items()):
    ax.bar(x + (i - 1) * width, sigma_arr, width, label=fname, color=COLORS_BAR[i], alpha=0.85)

ax.set_yscale("log")
ax.set_ylim(1e-3, 1e1)
ax.set_xticks(x)
ax.set_xticklabels(PARAM_LABELS, fontsize=10)
ax.set_ylabel(r"Cramér-Rao $1\sigma$ bound (log scale)")
ax.legend(fontsize=10, frameon=False)
fig.tight_layout()
plt.savefig("plot_fisher_degeneracy.png", dpi=150, bbox_inches="tight")
