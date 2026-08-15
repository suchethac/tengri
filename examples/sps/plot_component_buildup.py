"""
Building the panchromatic SED component by component
=====================================================

Energy is conserved: dust attenuation removes UV/optical flux, which is
re-radiated in the far-infrared (Dale 2014 templates restore the balance).
IGM absorption (Inoue 2014) sculpts the rest-frame continuum below the
Lyman break (912 Å).
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

Z = 0.2
ssp = tengri.load_ssp()

# Common base: tsnorm SFH, no AGN, no IGM, no dust, no nebular.
BASE_SFH = {
    "type": "tsnorm",
    "all_params": tengri.FIXED,
    "log_total_mass": 10.63,
    "peak_lbt_gyr": 3.0,
    "width_gyr": 2.0,
    "skew": 0.3,
    "trunc": 3.0,
}


def build(label, **kw):
    return label, tengri.SEDModel.build(ssp, sfh=BASE_SFH, redshift=tengri.Fixed(Z), **kw)


STAGES = [
    build("1. stellar only"),
    build(
        "2. + nebular",
        # BakedIn nebular is on by default when the SSP is wNE; the
        # explicit None turns the dust + AGN off so we isolate the
        # nebular addition.
    ),
    build(
        "3. + dust attenuation",
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": 0.35,
            "tau_bc": 0.4,
        },
    ),
    build(
        "4. + dust IR emission",
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": 0.35,
            "tau_bc": 0.4,
            "emission": {"type": "dale2014", "all_params": tengri.FIXED},
        },
    ),
    build(
        "5. + AGN",
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": 0.35,
            "tau_bc": 0.4,
            "emission": {"type": "dale2014", "all_params": tengri.FIXED},
        },
        agn={
            "type": "composable",
            "all_params": tengri.FIXED,
            "log_lbol": 11.5,
            "disc": {"type": "multicolor", "all_params": tengri.FIXED},
            "torus": {"type": "none"},
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "feii": {"type": "none"},
            "atten": {"type": "none"},
        },
    ),
    build(
        "6. + IGM",
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": 0.35,
            "tau_bc": 0.4,
            "emission": {"type": "dale2014", "all_params": tengri.FIXED},
        },
        agn={
            "type": "composable",
            "all_params": tengri.FIXED,
            "log_lbol": 11.5,
            "disc": {"type": "multicolor", "all_params": tengri.FIXED},
            "torus": {"type": "none"},
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "feii": {"type": "none"},
            "atten": {"type": "none"},
        },
        apply_igm=True,
    ),
]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
cmap = plt.get_cmap("viridis")

for i, (label, model) in enumerate(STAGES):
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    sed = model.predict(params)
    wave = np.asarray(model.wavelengths)
    L_nu = np.asarray(sed.rest_sed())
    # rest-frame nu L_nu in erg/s
    nu_L_nu = (2.998e18 / wave) * L_nu
    color = cmap(i / (len(STAGES) - 1))
    alpha = 0.45 if i < len(STAGES) - 1 else 1.0
    lw = 0.9 if i < len(STAGES) - 1 else 1.6
    ax.plot(wave, nu_L_nu, color=color, lw=lw, alpha=alpha, label=label)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Rest-frame wavelength  [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax.set_xlim(8.0e2, 5.0e6)
ymax = ax.get_ylim()[1]
ax.set_ylim(ymax * 1e-6, ymax)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)

fig.savefig("plot_component_buildup.png", dpi=150, bbox_inches="tight")
