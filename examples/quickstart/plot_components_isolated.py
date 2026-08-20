"""
Each tengri SED component shown in isolation
==============================================

Six physics blocks added cumulatively to the same star-forming host
so the contribution of each is visible at every wavelength.

Reading order: stellar continuum (gray), then nebular (HII region
lines added at the source), then dust (UV attenuated, reprocessed
into the FIR), then AGN (disc + torus + NLR), then radio, then X-ray.
The color at each wavelength tells you which block matters most.
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


SSP = tengri.load_ssp("fsps_prsc_miles_chabrier")
HOST = dict(
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "tau_gyr": 1.5,
        "log_total_mass": 10.0,
        "alpha": 2.5,
        "beta": 2.0,
    },
    redshift=tengri.Fixed(0.05),
)
DUST_ON = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 0.4,
    "tau_bc": 0.6,
}
# dale2014_cigale: the "+ radio" run composes this dust block with the
# radio component, and plain dale2014 embeds its own SF radio continuum —
# the pair is refused at build as a double-count (#1970). The stripped
# template also keeps the "+ dust" curve honest in the radio band: dust
# alone contributes nothing there.
DUST_EMISSION = {"type": "dale2014_cigale", "all_params": tengri.FIXED}
DUST_OFF = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}


def _nuLnu(**blocks):
    model = tengri.SEDModel.build(SSP, **HOST, **blocks)
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    return wave, 2.998e18 / wave * np.asarray(out.rest_sed())


RUNS = [
    ("stellar", "#666666", dict(dust_attenuation=DUST_OFF)),
    (
        "+ nebular",
        "#33aa55",
        dict(dust_attenuation=DUST_OFF, neb={"type": "cue", "all_params": tengri.FIXED}),
    ),
    ("+ dust", "#cc6633", dict(dust_attenuation=DUST_ON, dust_emission=DUST_EMISSION)),
    (
        "+ AGN",
        "#cc3399",
        dict(
            dust_attenuation=DUST_ON,
            dust_emission=DUST_EMISSION,
            agn={
                "disc": {"type": "multicolor", "all_params": tengri.FIXED},
                "torus": {"type": "skirtor", "all_params": tengri.FIXED},
                "nlr": {"type": "analytic", "all_params": tengri.FIXED},
                "blr": {"type": "none", "all_params": tengri.FIXED},
                "all_params": tengri.FIXED,
                "log_lbol": 11.5,
                "lum_ratio": 0.5,
            },
        ),
    ),
    (
        "+ radio",
        "#3377cc",
        dict(
            dust_attenuation=DUST_ON,
            dust_emission=DUST_EMISSION,
            radio={
                "sf": {"type": "bell2003"},
                "agn": {"type": "powerlaw"},
                "all_params": tengri.FIXED,
            },
        ),
    ),
    (
        "+ X-ray (XRBs)",
        "#9933cc",
        dict(
            dust_attenuation=DUST_ON,
            dust_emission=DUST_EMISSION,
            xray={"type": "simple", "all_params": tengri.FIXED},
        ),
    ),
]

fig, ax = plt.subplots(figsize=(8.0, 5.0))
for label, color, blocks in RUNS:
    wave, nuL = _nuLnu(**blocks)
    ax.loglog(wave, nuL, color=color, lw=1.4, label=label)

for x, name in [(1000, "UV"), (5500, "optical"), (1e5, "MIR"), (1e8, "radio")]:
    ax.text(x, 4e44, name, fontsize=7, color="0.5", ha="center", alpha=0.7)

ax.set(
    xlim=(10, 1e10),
    ylim=(1e35, 1e45),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2)

fig.tight_layout()
plt.savefig("plot_components_isolated.png", dpi=150, bbox_inches="tight")
