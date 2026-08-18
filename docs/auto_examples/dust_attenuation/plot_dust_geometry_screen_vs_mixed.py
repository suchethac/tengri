"""
Screen vs. mixed dust geometry: identical optical depths, different SEDs

Dust geometry determines how dust affects starlight. A **screen**
(foreground dust) filters the light as it leaves the galaxy:
``transmission = exp(-τ_λ)``. A **mixed** geometry (dust uniformly
distributed with stars) is more gentle:
``transmission = (1 - exp(-τ_λ)) / τ_λ``.

both geometries applied to the *same intrinsic SED*
at the *same V-band optical depth* (τ_V = 0.5, 1.0, 2.0). Despite
identical τ_V, the resulting SEDs are qualitatively different — mixed
geometry produces less attenuation in the UV, creating a shallower
effective attenuation curve. The two-panel figure shows (left) the
reddened SEDs and (right) the implied effective attenuation curve
A_λ/A_V.

References
----------

- Calzetti et al. 2000, ApJ, 533, 682 (starburst geometry)
- Witt & Gordon 2000, ApJ, 528, 799 (dust geometry effects)
- Kramer et al. 2003, ApJS, 144, 1 (mixed geometry approximation)

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.dust import calzetti
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# ── Constants and setup ───────────────────────────────────────────
C_AA_PER_S = 2.998e18
V_BAND_ANGSTROM = 5500.0

# Build an intrinsic (no-dust) model at z=0.05
SFH = {
    "type": "dpl",
    "all_params": tengri.FIXED,
    "alpha": 1.0,
    "beta": 2.0,
    "tau_gyr": 4.0,
    "log_total_mass": 10.0,
}

ssp = tengri.load_ssp()
intrinsic_model = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_bc": 0.0, "tau_diff": 0.0},
    redshift=tengri.Fixed(0.05),
)
p_intrinsic = dict(intrinsic_model.spec.sample(jax.random.PRNGKey(0)))
pred_intrinsic = intrinsic_model.predict(p_intrinsic)
sed_intrinsic = np.asarray(pred_intrinsic.rest_sed())
wave = np.asarray(intrinsic_model.wavelengths)
nu = C_AA_PER_S / wave

# Compute Calzetti k_lambda curve and normalize
k_lambda = np.asarray(calzetti(jnp.array(wave)))

# Compute tau_lambda(tau_V) for Calzetti: A_V = k_V * tau_V / R_V
# Calzetti has R_V = 4.05, but k_normalized already accounts for normalization
tau_v_values = [0.5, 1.0, 2.0]
colors_screen = plt.cm.Blues(np.linspace(0.4, 0.95, len(tau_v_values)))
colors_mixed = plt.cm.Oranges(np.linspace(0.4, 0.95, len(tau_v_values)))

# ── Left panel: reddened SEDs ─────────────────────────────────────
fig, (ax_sed, ax_attn) = plt.subplots(1, 2, figsize=(11.0, 4.5))

ax_sed.loglog(
    wave, nu * sed_intrinsic, color="0.0", lw=1.5, label="intrinsic", zorder=5, alpha=0.6
)

for tau_v, color_s, color_m in zip(tau_v_values, colors_screen, colors_mixed):
    # Screen geometry: transmission = exp(-tau_lambda)
    tau_lambda = k_lambda * tau_v
    transmission_screen = np.exp(-tau_lambda)
    sed_screen = sed_intrinsic * transmission_screen
    ax_sed.loglog(wave, nu * sed_screen, color=color_s, lw=1.3, label=f"screen τ_V={tau_v:.1f}")

    # Mixed geometry: transmission = (1 - exp(-tau_lambda)) / tau_lambda
    # For small tau_lambda, use Taylor expansion to avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        transmission_mixed = np.where(
            tau_lambda > 1e-6,
            (1.0 - np.exp(-tau_lambda)) / tau_lambda,
            1.0 - tau_lambda * (1.0 - 0.5 * tau_lambda),  # Taylor expansion
        )
    sed_mixed = sed_intrinsic * transmission_mixed
    ax_sed.loglog(
        wave, nu * sed_mixed, color=color_m, lw=1.3, label=f"mixed τ_V={tau_v:.1f}", ls="--"
    )

ax_sed.set(
    xlim=(900, 3e4),
    ylim=(1e40, 8e43),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax_sed.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
ax_sed.axvline(2175, color="0.55", lw=0.4, ls=":")

# ── Right panel: effective attenuation curve ──────────────────────
# Effective attenuation: A_lambda / A_V = -2.5 * log10(transmission)
# where A_V corresponds to the V-band optical depth
a_v_screen_values = []
a_v_mixed_values = []

for tau_v, color_s, color_m in zip(tau_v_values, colors_screen, colors_mixed):
    tau_lambda = k_lambda * tau_v

    # Screen transmission
    transmission_screen = np.exp(-tau_lambda)
    a_lambda_screen = -2.5 * np.log10(transmission_screen + 1e-10)
    a_v_screen = -2.5 * np.log10(np.exp(-tau_v) + 1e-10)
    a_eff_screen = a_lambda_screen / a_v_screen

    ax_attn.semilogx(wave, a_eff_screen, color=color_s, lw=1.3, label=f"screen τ_V={tau_v:.1f}")

    # Mixed transmission
    with np.errstate(divide="ignore", invalid="ignore"):
        transmission_mixed = np.where(
            tau_lambda > 1e-6,
            (1.0 - np.exp(-tau_lambda)) / tau_lambda,
            1.0 - tau_lambda * (1.0 - 0.5 * tau_lambda),
        )
    a_lambda_mixed = -2.5 * np.log10(transmission_mixed + 1e-10)
    # For mixed, A_V is different from tau_V; compute from V-band transmission
    tau_v_mixed = (1.0 - np.exp(-tau_v)) / tau_v
    a_v_mixed = -2.5 * np.log10(tau_v_mixed + 1e-10)
    a_eff_mixed = a_lambda_mixed / a_v_mixed

    ax_attn.semilogx(
        wave, a_eff_mixed, color=color_m, lw=1.3, label=f"mixed τ_V={tau_v:.1f}", ls="--"
    )

# Calzetti reference curve
rv = 4.05
a_v_ref = 1.086  # = -2.5 * log10(exp(-1))
a_lambda_ref = -2.5 * np.log10(np.exp(-k_lambda) + 1e-10)
a_eff_ref = a_lambda_ref / a_v_ref
ax_attn.semilogx(wave, a_eff_ref, color="0.2", lw=1.5, label="Calzetti (τ=1)", alpha=0.5)

ax_attn.set(
    xlim=(900, 3e4),
    ylim=(0.0, 2.5),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$A_\lambda / A_V$",
)
ax_attn.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
ax_attn.axvline(2175, color="0.55", lw=0.4, ls=":")
ax_attn.grid(True, which="both", alpha=0.2)

fig.tight_layout()
plt.savefig("plot_dust_geometry_screen_vs_mixed.png", dpi=150, bbox_inches="tight")
