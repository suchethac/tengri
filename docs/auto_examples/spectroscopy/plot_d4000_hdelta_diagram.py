"""
Kauffmann+2003 D_n(4000) vs Hδ_A Diagram
=========================================

Population diagnostics: single-burst SSP populations (3 SFH shapes × 5 ages
× 3 metallicities = 45 points) colored by SFH shape and marked by
metallicity. The Hδ_A vs D_n(4000) diagram discriminates starburst
(high Hδ_A, low D_n(4000)) from quiescent (low Hδ_A, high D_n(4000))
populations and is sensitive to recent star formation and metal enrichment.

References
----------
.. [1] Kauffmann, G., Heckman, T. M., White, S. D. M., et al. 2003,
       MNRAS, 341, 33 (D_n(4000) and Hδ_A diagnostics)
.. [2] Worthey, G., & Ottaviani, D. L. 1997, ApJS, 111, 377
       (Hδ_A window definitions)
.. [3] Balogh, M. L., Morris, S. L., Yee, H. K. C., et al. 1999,
       ApJ, 527, 54 (D_n(4000) break strength)
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Physical constants
C_AA_PER_S = 2.998e18  # Speed of light in Angstrom/s


def _compute_d4000(wave, l_nu):
    """Compute D_n(4000) break strength per Balogh+1999.

    D_n(4000) = mean flux at 4000-4100 Å / mean flux at 3850-3950 Å.

    Parameters
    ----------
    wave : ndarray
        Rest-frame wavelength [Angstrom].
    l_nu : ndarray
        Monochromatic luminosity [erg/s/Hz].

    Returns
    -------
    float
        D_n(4000) break strength.
    """
    blue = (wave >= 3850) & (wave <= 3950)
    red = (wave >= 4000) & (wave <= 4100)
    return float(np.mean(l_nu[red]) / np.mean(l_nu[blue]))


def _compute_hdelta_a(wave, l_nu):
    """Compute Hδ_A absorption equivalent width per Worthey+1997.

    Hδ_A defined as rest equivalent width in the 4080-4120 Å window,
    subtending the H-delta Balmer line at 4101.7 Å.

    Parameters
    ----------
    wave : ndarray
        Rest-frame wavelength [Angstrom].
    l_nu : ndarray
        Monochromatic luminosity [erg/s/Hz].

    Returns
    -------
    float
        Hδ_A equivalent width [Angstrom]. Negative for absorption
        (post-starburst), positive for emission (HII).
    """
    # Define window (Worthey & Ottaviani 1997)
    line = (wave >= 4080) & (wave <= 4120)
    cont_blue = (wave >= 4050) & (wave <= 4080)
    cont_red = (wave >= 4120) & (wave <= 4170)

    # Convert L_nu to F_lambda for equivalent width calculation
    f_lam = l_nu * C_AA_PER_S / wave**2

    # Continuum estimate via linear interpolation
    if cont_blue.sum() > 0 and cont_red.sum() > 0:
        lam_blue = np.mean(wave[cont_blue])
        lam_red = np.mean(wave[cont_red])
        f_blue = np.mean(f_lam[cont_blue])
        f_red = np.mean(f_lam[cont_red])
        slope = (f_red - f_blue) / (lam_red - lam_blue)
        cont = f_blue + slope * (wave[line] - lam_blue)
    else:
        cont = np.mean(f_lam[line])

    # Equivalent width: sum of (1 - F_line / F_cont) × delta_lambda
    if line.sum() > 0:
        delta = wave[line][1] - wave[line][0] if line.sum() > 1 else 1.0
        ew = float(np.sum((1.0 - f_lam[line] / np.maximum(cont, 1e-30)) * delta))
    else:
        ew = 0.0

    return ew


ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")  # Cue needs bare-stellar SSP

# Three SFH widths under tsnorm, scanned over age and Z. tsnorm accepts
# peak_lbt_gyr so the inner loop can vary the burst lookback time.
sfh_shapes = [
    (
        "instantaneous",
        {"type": "tsnorm", "width_gyr": 0.05, "log_total_mass": 10.0, "skew": 0.0, "trunc": 13.5},
    ),
    (
        "extended",
        {"type": "tsnorm", "width_gyr": 0.50, "log_total_mass": 10.0, "skew": 0.0, "trunc": 13.5},
    ),
    (
        "rising",
        {"type": "tsnorm", "width_gyr": 0.30, "log_total_mass": 10.0, "skew": 1.5, "trunc": 13.5},
    ),
]
ages_gyr = np.array([0.5, 1.0, 2.0, 4.0, 5.5])  # avoid SSP age boundary step
metallicities = np.array([-0.5, 0.0, 0.3])  # log10(Z/Zsun)

n_pop = len(sfh_shapes) * len(ages_gyr) * len(metallicities)
d4000 = np.empty(n_pop)
hdelta_a = np.empty(n_pop)
sfh_shape_idx = np.empty(n_pop, dtype=int)
metallicity_idx = np.empty(n_pop, dtype=int)

idx = 0
for shape_i, (_shape_name, sfh_dict) in enumerate(sfh_shapes):
    for age in ages_gyr:
        for met_i, z_sun in enumerate(metallicities):
            # Build model with this configuration
            # Use a single-burst-like SFH with varying peak age
            sfh = {
                **sfh_dict,
                "peak_lbt_gyr": age,  # Lookback time to burst
            }
            dust = {
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.FIXED,
                "tau_bc": 0.0,
                "tau_diff": 0.0,
            }
            neb = {"type": "cue", "all_params": tengri.FIXED}

            model = tengri.SEDModel.build(
                ssp,
                sfh=sfh,
                dust_attenuation=dust,
                neb=neb,
                redshift=tengri.Fixed(0.05),  # avoid numerical issues at z=0
            )

            # Sample baseline parameters
            baseline = dict(model.spec.sample(jax.random.PRNGKey(idx)))

            # Override metallicity (convert from absolute to relative log10(Z/Zsun))
            LOG10_ZSUN = -1.8477  # Asplund 2009
            baseline["neb_logZ_gas"] = z_sun - LOG10_ZSUN
            baseline["met_logzsol"] = z_sun - LOG10_ZSUN

            # Predict spectrum at R~2000 resolution (~150 km/s at 4000 Å)
            # Use a narrow rest-frame window covering D4000 and Hδ_A
            wave_rest = np.linspace(3800, 4200, 320)  # ~1.25 Å resolution ≈ 95 km/s

            # Get rest-frame SED
            pred = model.predict(baseline)
            wave = np.asarray(model.wavelengths)
            l_nu = np.asarray(pred.rest_sed())

            # Extract D4000 and Hδ_A
            d4000[idx] = _compute_d4000(wave, l_nu)
            hdelta_a[idx] = _compute_hdelta_a(wave, l_nu)

            sfh_shape_idx[idx] = shape_i
            metallicity_idx[idx] = met_i

            idx += 1

# Plot Kauffmann+2003 diagram
fig, ax = plt.subplots(figsize=(7.0, 5.5))

# Define colors for SFH shapes and markers for metallicity
colors = ["C0", "C1", "C2"]
labels_shape = ["Instantaneous", "Extended", "Rising"]
markers = ["o", "s", "^"]
labels_met = [r"$\log_{10}(Z/Z_\odot) = -0.85$", r"$0.85$", r"$1.15$"]

# Plot each combination
for shape_i in range(len(sfh_shapes)):
    for met_i in range(len(metallicities)):
        mask = (sfh_shape_idx == shape_i) & (metallicity_idx == met_i)
        ax.scatter(
            d4000[mask],
            hdelta_a[mask],
            color=colors[shape_i],
            marker=markers[met_i],
            s=80,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.4,
            label=f"{labels_shape[shape_i]}, {labels_met[met_i]}" if shape_i == 0 else "",
        )

# Annotations for quiescent/starburst regions
ax.text(1.65, -1.5, "Quiescent", fontsize=9, color="0.5", ha="center")
ax.text(1.35, 3.0, "Starburst", fontsize=9, color="0.5", ha="center")

# Mark the Kauffmann+2003 dividing line approximately
d4000_div = np.linspace(1.3, 1.8, 50)
hdelta_div = 8.0 - 4.5 * d4000_div  # Approximate linear divider
ax.plot(d4000_div, hdelta_div, "k--", alpha=0.3, lw=1.2, label="Kauffmann+2003 divider")

ax.set_xlabel(r"$D_n(4000)$ break strength", fontsize=11)
ax.set_ylabel(r"H$\delta_A$ equivalent width  [$\mathrm{\AA}$]", fontsize=11)
ax.axhline(0, color="0.8", lw=0.8, zorder=0)
ax.set_xlim(1.2, 1.9)
ax.set_ylim(-2.5, 3.5)

# Legend for shapes (compact)
from matplotlib.patches import Patch

legend_patches = [
    Patch(facecolor=colors[i], label=labels_shape[i]) for i in range(len(sfh_shapes))
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9, title="SFH shape")

# Add marker legend for metallicity as text annotations
fig.text(
    0.72,
    0.25,
    "Metallicity:\n" + "\n".join([f"{m} {l}" for m, l in zip(markers, labels_met)]),
    fontsize=8,
    family="monospace",
)

plt.savefig("plot_d4000_hdelta_diagram.png", dpi=150, bbox_inches="tight")
