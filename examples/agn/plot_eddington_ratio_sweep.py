"""
Eddington ratio sweep: multicolor disc thermal scaling
=======================================================

At fixed black hole mass M_BH = 10^8 M_sun, the accretion disc luminosity
and spectral shape scale with Eddington ratio λ_Edd = L_bol / L_Edd.
Here we sweep λ_Edd from 0.001 to 1.0 at five logarithmic steps and overlay
the disc continuum (100–3000 Å) to show how lower accretion rates produce
fainter discs with unchanged spectral shape (Shakura & Sunyaev 1973).

This example demonstrates:

- Bolometric luminosity constraint: L_bol = λ_Edd × L_Edd(M_BH)
- Black hole physics encoding in disc models via M_BH and λ_Edd
- Fixed spectral shape under Eddington scaling (thin-disc invariance)

**References:**

.. [1] Eddington, A. S. (1926). The internal constitution of the stars.
       Cambridge University Press.

.. [2] Shakura, N. I., & Sunyaev, R. A. (1973). Black holes in binary systems.
       Observational appearance. Astronomy and Astrophysics, 24, 337–355.
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Physical constants
LSUN = 3.828e33  # [erg s^-1]
LEDD_SUN_COEFF = 3.2e4  # L_Edd / (M_BH / M_sun) in L_sun units

# Load SSP data and construct model
ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "tau_gyr": 3.0,
        "log_peak_sfr": 0.5,
        "alpha": 2.0,
        "beta": 2.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    agn={
        "type": "composable",
        "disc": {"type": "multicolor", "*": tengri.FIXED},
        "torus": {"type": "skirtor", "*": tengri.FIXED},
        "lines": {"type": "nlr", "*": tengri.FIXED},
        "*": tengri.FIXED,
    },
    redshift=tengri.Fixed(0.05),
)

# Sample baseline parameters
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Eddington ratio sweep: log10(L_bol / L_Edd) from -3 to 0
log_edd_values = np.linspace(-3.0, 0.0, 5)  # λ_Edd = 0.001, 0.01, 0.1, 1.0 (and intermediate)

# Colormap setup
norm = mpl.colors.Normalize(vmin=log_edd_values.min(), vmax=log_edd_values.max())
cmap = plt.get_cmap("viridis")

# Plot setup with style control
fig, ax = plt.subplots(figsize=(8, 5.5))

# Loop over Eddington ratio values
for log_edd in log_edd_values:
    # Compute bolometric luminosity from M_BH and Eddington ratio
    # log_mbh is fixed at 8.0 (10^8 M_sun)
    # L_Edd(M_BH) = LEDD_SUN_COEFF * M_BH [M_sun] [L_sun]
    # log_lbol = log(lambda_Edd * L_Edd / L_sun)
    #          = log(lambda_Edd) + log(L_Edd / L_sun)
    #          = log_lambda_Edd + log(LEDD_SUN_COEFF * 10^log_mbh)
    log_mbh = 8.0
    log_ledd = LEDD_SUN_COEFF * (10.0**log_mbh)  # L_Edd in L_sun
    log_lbol = log_edd + np.log10(log_ledd)

    # Update parameters: set log_lbol and log_ledd, keep M_BH fixed
    params = {
        **baseline,
        "agn_log_lbol": jnp.float64(log_lbol),
        "agn_log_mbh": jnp.float64(log_mbh),
        "agn_log_ledd": jnp.float64(log_edd),
    }

    # Predict rest-frame SED
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    sed = np.asarray(out.sed)

    # Restrict to disc-dominated regime (100-3000 Angstrom)
    mask = (wave >= 100.0) & (wave <= 3000.0)
    wave_cut = wave[mask]
    sed_cut = sed[mask]

    # Convert to nu * L_nu for canonical SED plotting
    nu = 2.998e18 / wave_cut  # [Hz]
    nu_l_nu = nu * sed_cut

    # Plot with colormap scaled by Eddington ratio
    label = rf"$\log(L/L_{{\mathrm{{Edd}}}})={log_edd:.1f}$"
    ax.loglog(
        wave_cut,
        nu_l_nu,
        color=cmap(norm(log_edd)),
        lw=1.8,
        label=label,
    )

# Axis labels and limits
ax.set_xlim(100, 3000)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]", fontsize=11)
ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]", fontsize=11)
ax.set_title(
    r"Multicolor Disc: Eddington Ratio Sweep ($M_{\mathrm{BH}}=10^8 M_\odot$)",
    fontsize=12,
    pad=10,
)

# Legend
ax.legend(
    loc="upper right",
    fontsize=9,
    framealpha=0.95,
    title=r"Eddington Ratio",
    title_fontsize=9,
)

# Grid
ax.grid(alpha=0.25, which="both", linestyle="-", linewidth=0.4)

fig.tight_layout()
plt.savefig("plot_eddington_ratio_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
