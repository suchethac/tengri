"""
Diagnostic: Calzetti 2000 attenuation law vs. published formula
===============================================================

Verify tengri's Calzetti implementation against Eq. 1 in Calzetti et al. 2000
(ApJ 533, 682). The canonical k(V=5500 Å) = 4.05 must be reproduced exactly.
"""

import matplotlib.pyplot as plt
import numpy as np

import tengri.analysis.plotting
from tengri.dust import calzetti


def published_calzetti_kv(wave_um):
    """Calzetti 2000 piecewise formula from ApJ 533, 682 Eq. 1.

    wave_um : wavelength in microns
    Returns k(lambda) before normalization (k'(lambda) in the paper).
    """
    x = 1.0 / wave_um

    # UV: 0.12 - 0.63 μm
    k_uv = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + 4.05

    # IR: 0.63 - 2.20 μm
    k_ir = 2.659 * (-1.857 + 1.040 * x) + 4.05

    return np.where(wave_um >= 0.63, k_ir, k_uv)


# Setup
tengri.analysis.plotting.setup_style()
fig, (ax_curve, ax_res) = plt.subplots(
    2, 1, figsize=(6.5, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

# Wavelength grid: 1200 Å to 22000 Å
wave_aa = np.logspace(3.08, 4.34, 300)  # ~1200 to 22000 Å
wave_um = wave_aa / 1e4

# Tengri's k(lambda) — normalized so k(V) = 1
k_tengri = calzetti(wave_aa)

# Published Calzetti formula — returns k'(lambda) before normalization
k_pub_unnorm = published_calzetti_kv(wave_um)

# Published k(lambda) normalized: k'(lambda) / R_V
k_published = k_pub_unnorm / 4.05

# Find V-band value (5500 Å)
i_v = np.argmin(np.abs(wave_aa - 5500.0))
k_v_tengri = k_tengri[i_v]
k_v_published = k_published[i_v]

# Plot curves
ax_curve.loglog(wave_um, k_tengri, "C0-", lw=1.4, label="tengri", alpha=0.8)
ax_curve.loglog(wave_um, k_published, "C1--", lw=1.2, label="Calzetti 2000", alpha=0.7)

# Mark V-band point
ax_curve.plot(5500.0 / 1e4, k_v_tengri, "C0o", markersize=7, zorder=5)
ax_curve.text(5500.0 / 1e4 * 1.1, k_v_tengri, f"k(V) = {k_v_tengri:.4f}", fontsize=9, va="center")

ax_curve.set(ylabel=r"$k(\lambda)$", ylim=[0.1, 10])
ax_curve.legend(frameon=False, fontsize=9, loc="upper left")
ax_curve.grid(True, which="both", alpha=0.2)

# Residual panel: percent difference
residual_pct = 100.0 * (k_tengri - k_published) / k_published
ax_res.semilogx(wave_um, residual_pct, "k-", lw=1.2)
ax_res.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax_res.axhline(1, color="gray", linestyle=":", alpha=0.3)
ax_res.axhline(-1, color="gray", linestyle=":", alpha=0.3)
ax_res.set(xlabel=r"$\lambda$ [$\mu$m]", ylabel=r"Residual [\%]")
ax_res.grid(True, which="both", alpha=0.2)

fig.tight_layout()
plt.savefig("plot_diag_calzetti_kv_norm.png", dpi=150, bbox_inches="tight")
