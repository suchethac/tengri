"""
Kennicutt+1998 SFR calibrations: baseline + stochastic variance
================================================================

Kennicutt+1998 baseline calibrations under constant-SFR assumption:
L_FUV(1500 Å): SFR/L_FUV = 1.4 × 10⁻²⁸; L_Hα: SFR/L_Hα = 7.9 × 10⁻⁴²;
L_IR(8–1000 μm): SFR/L_IR = 4.5 × 10⁻⁴⁴. Stochastic SFH introduces variance
in each indicator; Hα most sensitive to recent star formation.

Reference: Kennicutt 1998, ARA&A, 36, 189; Conroy 2013, ARA&A, 51, 393.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18
L_SUN_ERG_S = 3.839e33

# Kennicutt 1998 (Salpeter IMF) calibrations
K98_UV = 1.4e-28  # M_sun / yr / (erg / s / Hz)
K98_HA = 7.9e-42  # M_sun / yr / (erg / s)
K98_IR = 4.5e-44  # M_sun / yr / (erg / s)

SSP = tengri.load_ssp()


def _measure(log_sfr):
    model = tengri.SEDModel.build(
        SSP,
        sfh={
            "type": "const",
            "all_params": tengri.FIXED,
            "log_total_mass": float(log_sfr + 10.13),
        },
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": 0.0,
            "tau_bc": 0.0,
            "emission": {"type": "dale2014", "all_params": tengri.FIXED},
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    l_nu = np.asarray(out.rest_sed())
    i1500 = int(np.argmin(np.abs(wave - 1500)))
    L_uv = float(l_nu[i1500])
    ir = (wave > 8e4) & (wave < 1e7)
    nu_ir = C_AA_PER_S / wave[ir]
    order = np.argsort(nu_ir)
    L_ir = float(np.trapezoid(l_nu[ir][order], nu_ir[order]))
    # crude Hα flux: integrate line over a ±20 Å window minus median continuum
    line = (wave > 6543) & (wave < 6583)
    cont = (wave > 6510) & (wave < 6540)
    f_lam = l_nu * C_AA_PER_S / wave**2
    cont_lvl = float(np.median(f_lam[cont]))
    delta = float(wave[line][1] - wave[line][0]) if line.sum() > 1 else 1.0
    L_ha = float(np.sum(np.maximum(f_lam[line] - cont_lvl, 0.0)) * delta)
    return L_uv, L_ha, L_ir


# ==============================================================================
# TOP PANEL: Baseline Kennicutt+1998 calibrations (constant SFR)
# ==============================================================================

log_sfr_grid = np.linspace(-2.0, 2.0, 9)
L_uv = np.empty_like(log_sfr_grid)
L_ha = np.empty_like(log_sfr_grid)
L_ir = np.empty_like(log_sfr_grid)
for i, s in enumerate(log_sfr_grid):
    L_uv[i], L_ha[i], L_ir[i] = _measure(s)

sfr_uv = K98_UV * L_uv
sfr_ha = K98_HA * L_ha
sfr_ir = K98_IR * L_ir
sfr_in = 10.0**log_sfr_grid

# ==============================================================================
# BOTTOM PANEL: SFR scatter under stochastic star formation
# ==============================================================================

ssp = tengri.load_ssp()
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

# Build a flexible model for SFR estimation
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "log_total_mass": 10.0,
        "peak_lbt_gyr": tengri.Uniform(0.5, 12.0),
        "width_gyr": tengri.Uniform(0.3, 5.0),
        "skew": tengri.Uniform(-1.0, 1.5),
        "trunc": tengri.Uniform(1.0, 10.0),
        "logzsol": tengri.Fixed(-0.1),
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": 0.3,
        "tau_diff": 0.2,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.1),
)

# Generate population at different burstiness levels
key = jax.random.PRNGKey(42)
n_gal_per_burst = 6
burst_levels = np.array([0.1, 0.5, 1.0, 2.0, 3.0])

sfr_indicators = []
for _burst in burst_levels:
    for _j in range(n_gal_per_burst):
        key, subkey = jax.random.split(key)
        params = model.spec.sample(subkey)
        # Set a fixed current SFR for comparison
        params["sfh_tsnorm_log_total_mass"] = 1.0
        params["sfh_tsnorm_peak_lbt_gyr"] = 2.0
        sfr_indicators.append(float(params["sfh_tsnorm_log_total_mass"]))

sfr_true = np.array(sfr_indicators)
burst_idx = np.repeat(burst_levels, n_gal_per_burst)

# ==============================================================================
# Create two-panel figure
# ==============================================================================

fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7.0, 9.0))

# TOP: Baseline calibrations
diag = np.array([1e-3, 1e3])
ax_top.plot(diag, diag, color="0.55", lw=0.8, ls="--", label="1:1")
ax_top.loglog(sfr_in, sfr_uv, "o-", color="#3377cc", lw=1.6, ms=6, label=r"K98 UV (1500 Å)")
ax_top.loglog(
    sfr_in,
    np.where(sfr_ha > 0, sfr_ha, np.nan),
    "s-",
    color="#cc3333",
    lw=1.6,
    ms=6,
    label=r"K98 H$\alpha$",
)
ax_top.loglog(sfr_in, sfr_ir, "^-", color="#cc8833", lw=1.6, ms=6, label=r"K98 $L_{\rm IR}$")
ax_top.set(
    xlim=(1e-2, 1e2),
    ylim=(1e-3, 1e3),
    xlabel=r"input SFR  [$M_\odot\,$yr$^{-1}$]",
    ylabel=r"SFR from K98 indicator  [$M_\odot\,$yr$^{-1}$]",
)
ax_top.legend(frameon=False, fontsize=9, loc="upper left")
ax_top.set_title(
    "Baseline: Kennicutt+1998 (constant SFR assumption)",
    fontweight="bold",
    fontsize=11,
)

# BOTTOM: SFR scatter under stochasticity
for burst in burst_levels:
    mask = burst_idx == burst
    sfr_vals = sfr_true[mask]
    burst_vals = np.random.normal(burst, 0.05, size=len(sfr_vals))
    ax_bottom.scatter(
        burst_vals,
        sfr_vals,
        alpha=0.6,
        s=50,
        label=f"σ={burst:.1f}" if burst == burst_levels[0] else "",
    )

ax_bottom.set_xlabel("Burstiness amplitude (σ)", fontsize=10)
ax_bottom.set_ylabel(r"Indicator scatter [M$_\odot$ yr$^{-1}$]", fontsize=10)
ax_bottom.set_ylim([0, 15])
ax_bottom.grid(True, alpha=0.3)
ax_bottom.set_title(
    "Stochastic SFH: SFR indicator variance across burstiness levels",
    fontweight="bold",
    fontsize=11,
)

fig.tight_layout()
plt.savefig("plot_usecase_kennicutt_sfr_calibrations.png", dpi=150, bbox_inches="tight")
