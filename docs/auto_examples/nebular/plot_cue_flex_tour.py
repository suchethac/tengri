"""
Cue knob flexibility: six dimensions of HII region control
===========================================================

Cue has six tuning knobs that control HII-region ionization and the
diffuse ionized gas. This six-panel tour sweeps each knob individually
and reports the ``L_Hα`` response *relative to the baseline*, in dex.
A flat line means the parameter has no effect on Hα at fixed other
knobs.

Per-panel summary:

- ``logU``    — modest, ±0.01 dex (Hα is set by recombination rate)
- ``logZ_gas`` — modest, ±0.02 dex (metals shift HII cooling)
- ``fesc``    — strong, −1 dex by ``fesc=0.95`` (LyC photons leak)
- ``fesc_lya`` — flat (only the Lyα line is affected)
- ``dig_frac`` — flat (suspected wiring bug, issue #259)
- ``dig_delta_logU`` — flat (confirmed wiring bug, issue #259)

References:
- Li, Leja & Speagle 2023, ApJ, 956, 23 (Cue)
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

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
SFH = {
    "type": "dpl",
    "*": tengri.FIXED,
    "tau_gyr": 0.3,
    "log_peak_sfr": 1.5,
    "alpha": 3.0,
    "beta": 2.0,
}
DUST = {"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.05, "tau_bc": 0.1}

model = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust=DUST,
    neb={
        "type": "cue",
        "*": tengri.FIXED,
        "logU": tengri.Uniform(-4.0, -1.0),
        "logZ_gas": tengri.Uniform(-2.0, 0.5),
        "fesc": tengri.Uniform(0.0, 1.0),
        "fesc_lya": tengri.Uniform(0.0, 1.0),
        "dig_frac": tengri.Uniform(0.0, 1.0),
        "dig_delta_logU": tengri.Uniform(-4.0, 0.0),
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))


def _halpha_lum(params):
    return float(model.predict_emission_lines(params).halpha)


SWEEPS = [
    ("neb_logU", r"$\log\,U$", 7, -3.8, -1.5),
    ("neb_logZ_gas", r"$\log\,Z_{\rm gas}/Z_\odot$", 7, -1.5, 0.4),
    ("neb_fesc", r"$f_{\rm esc}$", 7, 0.0, 0.95),
    ("neb_fesc_lya", r"$f_{\rm esc,Ly\alpha}$", 7, 0.0, 0.95),
    ("neb_dig_frac", r"DIG frac", 7, 0.0, 0.6),
    ("neb_dig_delta_logU", r"DIG $\Delta\log\,U$", 7, -2.0, 0.0),
]

fig, axes = plt.subplots(
    2, 3, figsize=(11, 6.6), sharey=True, gridspec_kw={"hspace": 0.35, "wspace": 0.15}
)
axes_flat = axes.ravel()
colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

ha_baseline = _halpha_lum(baseline)

for ax_idx, (param_name, label, n_vals, lo, hi) in enumerate(SWEEPS):
    ax = axes_flat[ax_idx]
    values = np.linspace(lo, hi, n_vals)
    halpha_vals = np.array([_halpha_lum({**baseline, param_name: jnp.float64(v)}) for v in values])
    delta_dex = np.log10(halpha_vals / ha_baseline)
    ax.plot(values, delta_dex, "o-", color=colors[ax_idx], lw=1.6, markersize=5)
    ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
    ax.set_xlabel(label, fontsize=9)
    ax.text(
        0.05,
        0.93,
        param_name.replace("neb_", ""),
        transform=ax.transAxes,
        fontsize=8,
        color="0.4",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.4),
    )

for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Delta \log_{10}\,L_{\rm H\alpha}$  [dex]", fontsize=9)

fig.savefig("plot_cue_flex_tour.png", dpi=150, bbox_inches="tight")
