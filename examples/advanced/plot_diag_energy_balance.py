"""
Energy balance: dust absorption vs. emission
==============================================

Tests dust attenuation–emission consistency via energy conservation. Sweeps
diffuse optical depth τ_diff while measuring agreement between independent
attenuation and emission modules. Ratio = L_emitted / L_absorbed should equal 1.

Reference: Draine & Li 2007, ApJ, 657, 810.
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Physical constants
C_AA_PER_S = 2.998e18  # Speed of light in Å/s

# Star-forming galaxy baseline (fixed, only tau_diff swept)
SFH = {
    "type": "dpl",
    "*": tengri.FIXED,
    "tau_gyr": 1.0,
    "log_total_mass": 10.0,
    "alpha": 2.5,
    "beta": 1.5,
}

DUST_BASE = {
    "type": "two_component",
    "law_bc": "calzetti",
    "law_diff": "calzetti",
    "*": tengri.FIXED,
    "tau_bc": 0.2,
    "emission": {"type": "draine_li2007", "*": tengri.FIXED},
}

# Load SSP and build model with dust emission enabled
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
model = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust=DUST_BASE,
    redshift=tengri.Fixed(0.0),
)

# Wavelength grid: rest-frame, covers UV–optical–IR
wave_jax = jnp.logspace(2.0, 5.5, 2500)  # ~100 Å to ~316 μm
wave = np.asarray(wave_jax)

# Dust optical depth sweep (5 values)
tau_diffs = np.array([0.1, 0.3, 0.5, 1.0, 2.0])
ratios = []
L_abs_list = []
L_emit_list = []

for tau_diff in tau_diffs:
    # Baseline parameters (redshift=0, SFH fixed, dust bc fixed)
    p_base = dict(model.spec.sample(jax.random.PRNGKey(0)))

    p_int = {**p_base, "dust_tau_bc": 0.0, "dust_tau_diff": 0.0}
    sed_int = np.asarray(model.predict_rest_sed(p_int, wave=wave_jax)).sum(axis=0)
    p_full = {**p_base, "dust_tau_bc": 0.2, "dust_tau_diff": tau_diff}
    sed_full = np.asarray(model.predict_rest_sed(p_full, wave=wave_jax)).sum(axis=0)
    nu = C_AA_PER_S / wave

    # L = ∫ L_ν dν.  np.trapz needs frequency in INCREASING order.
    mask_uv_opt = (wave >= 912.0) & (wave <= 30000.0)
    nu_uv = nu[mask_uv_opt]
    order = np.argsort(nu_uv)
    L_absorbed = float(
        np.trapz(
            (sed_int[mask_uv_opt] - sed_full[mask_uv_opt])[order],
            nu_uv[order],
        )
    )
    mask_fir = (wave >= 80000.0) & (wave <= 1.0e7)
    nu_fir = nu[mask_fir]
    order = np.argsort(nu_fir)
    L_emitted = float(np.trapz(sed_full[mask_fir][order], nu_fir[order]))
    ratio = L_emitted / L_absorbed if L_absorbed > 0 else np.nan
    ratios.append(ratio)
    L_abs_list.append(L_absorbed)
    L_emit_list.append(L_emitted)

# Convert to arrays
ratios = np.array(ratios)
L_abs_list = np.array(L_abs_list)
L_emit_list = np.array(L_emit_list)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))
ax_l.errorbar(
    tau_diffs, ratios, fmt="o", markersize=8, lw=2, color="C0", label="L_emit / L_absorb"
)
ax_l.axhline(1.0, ls="--", color="gray", lw=1, alpha=0.7, label="Expected")
ax_l.set_xlabel(r"$\tau_{\rm diff}$"), ax_l.set_ylabel("Ratio")
ax_l.legend(frameon=False, fontsize=9), ax_l.grid(True, alpha=0.3)
x_pos = np.arange(len(tau_diffs))
width = 0.35
ax_r.bar(x_pos - width / 2, L_abs_list / 1.0e43, width, label="L_absorbed", color="C0")
ax_r.bar(x_pos + width / 2, L_emit_list / 1.0e43, width, label="L_emitted", color="C1")
ax_r.set_xlabel(r"$\tau_{\rm diff}$"), ax_r.set_ylabel(r"Luminosity [$10^{43}$ erg s$^{-1}$]")
ax_r.set_xticks(x_pos), ax_r.set_xticklabels([f"{t:.1f}" for t in tau_diffs])
ax_r.legend(frameon=False, fontsize=9), ax_r.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
plt.savefig("plot_diag_energy_balance.png", dpi=150, bbox_inches="tight")

# Report findings
print("Energy balance diagnostic results:")
print(f"{'tau_diff':<10} {'Ratio':<10} {'L_abs (erg/s)':<20} {'L_emit (erg/s)':<20}")
for tau, r, la, le in zip(tau_diffs, ratios, L_abs_list, L_emit_list):
    print(f"{tau:<10.1f} {r:<10.3f} {la:<20.3e} {le:<20.3e}")
