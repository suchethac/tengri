"""
Validating IGM transmission against Madau 1995 published table
===============================================================

Diagnostic: Lyman-series line optical depth τ_LS vs observed wavelength in the
Lyman-alpha forest, comparing tengri's Madau+1995 model to manual calculation
from published coefficients (Madau 1995 Table 1, Eq. 15).
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

import tengri
from tengri.igm import igm_transmission_madau

# Setup house style (required for all gallery scripts)
tengri.analysis.plotting.setup_style()

# ── Published Madau 1995 Table 1: all 17 Lyman-series rest wavelengths ──────
# Coefficients from Madau (1995) Table 1, lines j=2 (Ly-alpha) to j=18
madau_table1 = {
    "Ly_alpha": (1215.67, 3.6e-3),
    "Ly_beta": (1025.72, 1.7e-3),
    "Ly_gamma": (972.537, 1.1846e-3),
    "Ly_delta": (949.743, 9.41e-4),
    "Ly_epsilon": (937.803, 7.96e-4),
    "Ly_zeta": (930.748, 6.967e-4),
    "j=8": (926.226, 6.236e-4),
    "j=9": (923.150, 5.665e-4),
    "j=10": (920.963, 5.200e-4),
    "j=11": (919.352, 4.817e-4),
    "j=12": (918.129, 4.487e-4),
    "j=13": (917.181, 4.200e-4),
    "j=14": (916.429, 3.947e-4),
    "j=15": (915.824, 3.720e-4),
    "j=16": (915.329, 3.520e-4),
    "j=17": (914.919, 3.334e-4),
    "j=18": (914.576, 3.1644e-4),
}

# ── Manual Lyman-series line opacity (Madau Eq. 15) ────────────────────────
def tau_lyman_series_manual(wave_obs, z_source):
    """
    Sum Lyman-series line opacities: τ_j = A_j × (λ_obs/λ_j)^3.46
    for λ_j ≤ λ_obs ≤ λ_j(1+z), per Madau+1995 Eq. 15 (Table 1).
    """
    tau = np.zeros_like(wave_obs, dtype=float)
    for name, (lam_j, a_j) in madau_table1.items():
        lam_max = lam_j * (1.0 + z_source)
        in_range = (wave_obs >= lam_j) & (wave_obs <= lam_max)
        tau = np.where(in_range, tau + a_j * (wave_obs / lam_j) ** 3.46, tau)
    return tau


# ── Setup: source at z=4.0, sample λ_obs in Lyman-alpha forest ─────────────
z_source = 4.0
# Lyman limit at z=4: 911.75 × 5 = 4559 Å (continuum active below this)
# Lyman-alpha forest: above Lyman limit, between Ly-alpha and Ly-beta resonance
# At z=4: Ly-beta = 1026 × 5 = 5130 Å is redward boundary
# Sample in the true forest region (Lyman limit < λ_obs < Ly-alpha max)
wave_forest = np.array([4700.0, 4900.0, 5100.0, 5300.0, 5800.0])

# ── Compute Lyman-series line opacity: tengri igm_transmission_madau vs manual ─
# Extract the line component only (ignore continuum for ground-truth comparison)
T_tengri = igm_transmission_madau(jnp.array(wave_forest), z=z_source)
tau_total_tengri = -np.log(np.clip(T_tengri.tolist(), a_min=1e-8, a_max=None))
tau_lyman_manual = tau_lyman_series_manual(wave_forest, z_source)

# Residual (line component only)
residual_line = tau_total_tengri - tau_lyman_manual
rel_error_line = residual_line / np.maximum(np.abs(tau_lyman_manual), 1e-6)

# ── Left panel: τ_LS vs λ_obs ──────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.0, 4.0))

ax_left.plot(wave_forest, tau_lyman_manual, "s--", lw=1.4,
             label="Manual (Madau Eq. 15, Table 1)", markersize=6, color="C1")
ax_left.plot(wave_forest, tau_total_tengri, "o-", lw=1.4,
             label="tengri igm_transmission_madau", markersize=6, color="C0")
ax_left.set_xlabel(r"$\lambda_{\mathrm{obs}}$ [Å]", fontsize=11)
ax_left.set_ylabel(r"$\tau_{\mathrm{LS}}$ [dimensionless]", fontsize=11)
ax_left.legend(frameon=False, fontsize=9)
ax_left.grid(True, alpha=0.3)

# ── Right panel: relative residual ─────────────────────────────────────────
ax_right.plot(wave_forest, rel_error_line * 100, "s-", lw=1.4, markersize=6,
              color="C2", label="Relative error")
ax_right.axhline(+10, color="gray", linestyle=":", alpha=0.6, lw=1)
ax_right.axhline(-10, color="gray", linestyle=":", alpha=0.6, lw=1)
ax_right.fill_between(wave_forest, -10, +10, alpha=0.1, color="gray",
                      label="±10% band")
ax_right.set_xlabel(r"$\lambda_{\mathrm{obs}}$ [Å]", fontsize=11)
ax_right.set_ylabel(r"Relative error [%]", fontsize=11)
ax_right.legend(frameon=False, fontsize=9)
ax_right.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_diag_madau_published_table.png", dpi=150, bbox_inches="tight")

# ── Verification report ────────────────────────────────────────────────────
max_abs_error = np.max(np.abs(residual_line))
max_rel_error = np.max(np.abs(rel_error_line)) * 100

print(f"Madau 1995 diagnostic: z_source={z_source}")
print(f"λ_obs range: {wave_forest.min():.0f}–{wave_forest.max():.0f} Å")
print(f"Max absolute τ_LS error: {max_abs_error:.2e}")
print(f"Max relative error: {max_rel_error:.2f}%")
if max_rel_error > 10.0:
    print("WARNING: Residuals exceed ±10% threshold.")
else:
    print("✓ PASS: Residuals within ±10% band.")
