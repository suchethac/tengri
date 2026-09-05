r"""
Greybody dust opacity: optically thin vs. general form across temperature
=========================================================================

The optically thin dust emission (modified blackbody) is ν^β B_ν(T),
where dust temperature T and emissivity index β determine the SED shape.
Adding a frequency-dependent opacity (general greybody) introduces
(1 - exp(-(λ_0/λ)^β)) B_ν(T), which peaks in the FIR before flattening
in the sub-mm — this shape is also used in CIGALE. A pure blackbody is
obtained by setting β = 0 in the modified blackbody form. Both models
assume a constant dust mass and fixed attenuation, varying only temperature.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18


def _build_model(dust_emission_dict):
    """Build a model with the specified dust_emission configuration."""
    dust = {
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "tau_diff": 0.5,
        "tau_bc": 1.0,
    }
    model = tengri.SEDModel.build(
        tengri.load_ssp(),
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT), "log_total_mass": 11.13},
        dust_attenuation=dust,
        dust_emission=dust_emission_dict,
        redshift=tengri.Fixed(0.05),
    )
    return model


def _get_dust_ir_normalized(model):
    """
    Extract isolated dust emission normalized by L_IR.
    Returns (wave_um, normalized_lnu) where normalized_lnu = (c/lambda) * sed_dust_ir / L_ir.
    """
    state = model.predict_state({})
    wave_aa = np.asarray(state.wave)
    sed_dust_ir = np.asarray(state.derived["sed_dust_ir"])
    l_ir = float(state.derived["L_ir"])

    # Compute nu = c / lambda (speed of light / wavelength in Angstrom)
    c_aa_s = 2.99792458e18  # Angstrom/s
    nu = c_aa_s / wave_aa

    # Compute nu*L_nu / L_IR (dimensionless shape)
    shape = nu * sed_dust_ir / l_ir

    # Convert wavelength to micrometers
    wave_um = wave_aa * 1e-4

    return wave_um, shape


# Temperature grid for both panels
T_grid = np.array([25.0, 50.0, 100.0, 200.0])

fig, (ax_thin, ax_thick) = plt.subplots(
    2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"hspace": 0.06}
)

norm_T = mpl.colors.Normalize(vmin=T_grid.min(), vmax=T_grid.max())
cmap = plt.get_cmap("viridis")

# Top panel: optically thin (modified blackbody)
for T in T_grid:
    dust_emission_thin = {
        "type": "modified_blackbody",
        "beta_ir": tengri.Fixed(2.0),
        "T": tengri.Fixed(T),
        "all_params": tengri.Fixed(tengri.DEFAULT),
    }
    model = _build_model(dust_emission_thin)
    wave_um, shape = _get_dust_ir_normalized(model)
    ax_thin.loglog(wave_um, shape, color=cmap(norm_T(T)), lw=1.4)

# Bottom panel: general opacity (greybody)
for T in T_grid:
    dust_emission_thick = {
        "type": "greybody",
        "beta_ir": tengri.Fixed(2.0),
        "lambda_0_um": tengri.Fixed(100.0),
        "T": tengri.Fixed(T),
        "all_params": tengri.Fixed(tengri.DEFAULT),
    }
    model = _build_model(dust_emission_thick)
    wave_um, shape = _get_dust_ir_normalized(model)
    ax_thick.loglog(wave_um, shape, color=cmap(norm_T(T)), lw=1.4)

# Set axis labels and limits
for ax in (ax_thin, ax_thick):
    # Frame from 1 um to 1 cm with appropriate y-range for dimensionless shape curves
    ax.set(xlim=(1, 1e4), ylim=(1e-3, 2), ylabel=r"$\nu L_\nu / L_{\rm IR}$")

ax_thick.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")

# Single colorbar spanning both axes
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm_T, cmap=cmap), ax=[ax_thin, ax_thick], pad=0.01
)
cbar.set_label(r"$T_{\rm dust}$  [K]")

plt.savefig("plot_greybody_opacity_temperature.png", dpi=150, bbox_inches="tight")
