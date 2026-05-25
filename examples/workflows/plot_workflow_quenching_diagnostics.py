"""
Three diagnostics of quenching epoch in one figure
====================================================

When did star formation in a galaxy stop? Optical-only color, the
4000 Å break, and Hα equivalent width respond on different timescales:
NUV − r reddens within ~100 Myr of quenching (loss of O/B stars),
``D_n(4000)`` continues to rise over 1–3 Gyr as A stars evolve, and
Hα EW drops fastest of all (within ~10 Myr) since it tracks only the
youngest ionizing photons.

We sweep the truncation epoch ``t_quench`` of a constant-SFR
population from 10 Myr to 5 Gyr ago and show all three diagnostics on
the same x-axis so the reader can see which observation is the right
clock for a given quenching timescale.

References:
- Kauffmann+2003, MNRAS, 341, 33 (D_n(4000) vs Hα EW)
- Martin+2007, ApJS, 173, 342 (NUV − r and the green valley)
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18


def _d4000(wave: np.ndarray, l_nu: np.ndarray) -> float:
    blue = (wave >= 3850) & (wave <= 3950)
    red = (wave >= 4000) & (wave <= 4100)
    return float(np.mean(l_nu[red]) / np.mean(l_nu[blue]))


def _halpha_ew(wave: np.ndarray, l_nu: np.ndarray) -> float:
    """Approximate Hα equivalent width in Å via narrow / continuum windows."""
    cont = ((wave >= 6450) & (wave <= 6540)) | ((wave >= 6580) & (wave <= 6650))
    line = (wave >= 6545) & (wave <= 6580)
    f_lam = l_nu * C_AA_PER_S / wave**2
    f_cont = np.median(f_lam[cont])
    delta = wave[line][1] - wave[line][0] if line.sum() > 1 else 1.0
    excess = np.sum(f_lam[line] - f_cont) * delta
    return float(excess / f_cont) if f_cont > 0 else 0.0


obs = tengri.Observation(photometry=tengri.Photometry.from_names(["galex_nuv", "sdss_r"]))
model = tengri.SEDModel.build(
    tengri.load_ssp(),
    observation=obs,
    sfh={
        "type": "tsnorm",
        "*": tengri.FIXED,
        "peak_lbt_gyr": tengri.Uniform(0.01, 8.0),
        "width_gyr": 0.05,  # narrow truncation: an explicit quench
        "log_peak_sfr": 1.0,
        "skew": 0.0,
        "trunc": 13.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

t_q = np.geomspace(0.02, 5.0, 22)
nuv_r = np.empty_like(t_q)
d4000 = np.empty_like(t_q)
ha_ew = np.empty_like(t_q)

for i, t in enumerate(t_q):
    params = {**baseline, "sfh_tsnorm_peak_lbt_gyr": jnp.float64(t)}
    flux = np.asarray(model.predict_photometry(params))
    nuv_r[i] = -2.5 * np.log10(flux[0] / flux[1])
    sed = model.predict_rest_sed(params)
    wave = np.asarray(sed.wavelength)
    l_nu = np.asarray(sed.sed)
    d4000[i] = _d4000(wave, l_nu)
    ha_ew[i] = _halpha_ew(wave, l_nu)

fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.4), sharex=True, gridspec_kw={"hspace": 0.08})

ax = axes[0]
ax.plot(t_q, np.abs(ha_ew), color="C3", lw=1.6)
ax.set_yscale("log")
ax.set_ylabel(r"H$\alpha$ EW  [$\mathrm{\AA}$]")
ax.text(
    0.97, 0.92, "fast clock — ~10 Myr", transform=ax.transAxes, ha="right", fontsize=8, color="0.3"
)

ax = axes[1]
ax.plot(t_q, nuv_r, color="C0", lw=1.6)
ax.axhline(4.0, color="0.55", lw=0.6, ls=":")
ax.text(t_q[-1] * 0.6, 4.07, "green-valley NUV–r ≈ 4", fontsize=7, color="0.4")
ax.set_ylabel(r"NUV $-$ r  [AB mag]")
ax.text(
    0.97,
    0.92,
    "medium clock — ~100 Myr",
    transform=ax.transAxes,
    ha="right",
    fontsize=8,
    color="0.3",
)

ax = axes[2]
ax.plot(t_q, d4000, color="C2", lw=1.6)
ax.axhspan(1.5, 1.6, color="0.92", alpha=0.8, lw=0)
ax.text(t_q[-1] * 0.6, 1.55, "K+2003 green-valley", fontsize=7, color="0.35", va="center")
ax.set_ylabel(r"$D_n(4000)$")
ax.text(
    0.97,
    0.92,
    "slow clock — ~1–3 Gyr",
    transform=ax.transAxes,
    ha="right",
    fontsize=8,
    color="0.3",
)
ax.set_xscale("log")
ax.set_xlabel(r"Lookback time since quenching $t_{\rm q}$  [Gyr]")

plt.savefig("plot_workflow_quenching_diagnostics.png", dpi=150, bbox_inches="tight")
