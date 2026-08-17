"""
The photo-z degeneracy: dusty z ≈ 0.3 vs unobscured z ≈ 3.5
===========================================================

Two galaxies with different physical properties can produce nearly
identical broadband fluxes when the 4000 Å break of a dusty low-z galaxy
and the Lyman break of a high-z galaxy land at the same observed wavelength.

References: Steidel+1996; Massarotti+2001.
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

BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])


def _make(z_value: float, tau_diff: float, peak_lbt: float, log_total_mass: float):
    """Build a model at a chosen redshift and a couple of SFH/dust knobs."""
    model = tengri.SEDModel.build(
        tengri.load_ssp(),
        observation=obs,
        sfh={
            "type": "tsnorm",
            "all_params": tengri.FIXED,
            "peak_lbt_gyr": peak_lbt,
            "width_gyr": 1.5,
            "log_total_mass": 10.0,
            "skew": 0.0,
            "trunc": 10.0,
        },
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": tau_diff,
            "tau_bc": 0.4,
            "slope": -0.7,
        },
        redshift=tengri.Fixed(z_value),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    return model, p


m_low, p_low = _make(z_value=0.3, tau_diff=1.4, peak_lbt=3.0, log_total_mass=10.0)
m_hi, p_hi = _make(z_value=3.5, tau_diff=0.05, peak_lbt=0.2, log_total_mass=10.0)

flux_low = np.array(m_low.predict_photometry(p_low))
flux_hi = np.array(m_hi.predict_photometry(p_hi))
flux_hi = flux_hi * (flux_low[2] / flux_hi[2])


def _rest_to_obs(model, params, z, scale):
    out = model.predict(params)
    wave_obs = np.asarray(model.wavelengths) * (1.0 + z)
    fnu_obs = scale * np.asarray(out.rest_sed())
    return wave_obs, fnu_obs


# Anchor SEDs to land near the photometric flux on the figure.
def _band_scale(model, params, z, target_flux_r):
    wave_obs, sed = _rest_to_obs(model, params, z, 1.0)
    i = int(np.argmin(np.abs(wave_obs - wave_eff[2])))
    return target_flux_r / sed[i]


scale_low = _band_scale(m_low, p_low, 0.3, flux_low[2])
scale_hi = _band_scale(m_hi, p_hi, 3.5, flux_hi[2])
wave_low, fnu_low = _rest_to_obs(m_low, p_low, 0.3, scale_low)
wave_hi, fnu_hi = _rest_to_obs(m_hi, p_hi, 3.5, scale_hi)

fig, (ax_sed, ax_phot) = plt.subplots(
    2,
    1,
    figsize=(7.0, 5.4),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.05},
)

vis = (wave_low > 2.0e3) & (wave_low < 1.3e4)
ax_sed.plot(
    wave_low[vis],
    fnu_low[vis],
    color="C3",
    lw=1.0,
    label=r"$z=0.3$ dusty SF ($\tau_{\rm diff}=1.4$)",
)
vis_h = (wave_hi > 2.0e3) & (wave_hi < 1.3e4)
ax_sed.plot(wave_hi[vis_h], fnu_hi[vis_h], color="C0", lw=1.0, label=r"$z=3.5$ unobscured LBG")
ax_sed.set_yscale("log")
ax_sed.set_ylabel(r"$F_\nu$ [arbitrary scale]")
ax_sed.legend(frameon=False, fontsize=9, loc="lower right")
# Mark the two breaks
break_4000_obs = 4000 * 1.3
break_lyman_obs = 1216 * 4.5
ax_sed.axvline(break_4000_obs, color="C3", lw=0.6, ls=":", alpha=0.7)
ax_sed.axvline(break_lyman_obs, color="C0", lw=0.6, ls=":", alpha=0.7)
ymin, ymax = ax_sed.get_ylim()
ax_sed.text(
    break_4000_obs * 1.02, ymax * 0.4, r"4000 Å$\,(z{=}0.3)$", color="C3", fontsize=8, va="top"
)
ax_sed.text(
    break_lyman_obs * 0.98,
    ymax * 0.04,
    r"Ly$\alpha\,(z{=}3.5)$",
    color="C0",
    fontsize=8,
    va="top",
    ha="right",
)
ax_sed.text(
    0.04,
    0.10,
    "drop the u-band\n→ photo-z is bimodal",
    transform=ax_sed.transAxes,
    fontsize=8,
    color="0.3",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.5),
)

ax_phot.plot(wave_eff, flux_low, "o", color="C3", ms=7, mfc="none", mew=1.4, label=r"low-z fluxes")
ax_phot.plot(
    wave_eff,
    flux_hi,
    "s",
    color="C0",
    ms=7,
    mfc="none",
    mew=1.4,
    label=r"high-z fluxes (rescaled)",
)
ax_phot.set_yscale("log")
ax_phot.set_xscale("log")
ax_phot.set_ylabel(r"$F_\nu$  band")
ax_phot.set_xlabel(r"Observed wavelength $\lambda$ [$\mathrm{\AA}$]")
ax_phot.set_xlim(2.5e3, 1.2e4)
ax_phot.legend(frameon=False, fontsize=8, loc="lower right")

plt.savefig("plot_workflow_photoz_degeneracy.png", dpi=150, bbox_inches="tight")
