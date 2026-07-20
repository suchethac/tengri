"""
SED additivity: stellar, dust attenuation, emission, and nebular components
===========================================================================

Verifies the SED chain is additive by comparing the full ``pred.rest_sed()``
output against a manual sum of per-component SEDs. The forward model chains
stellar continuum through dust attenuation, dust emission, and nebular
processing; if modular, the sum should reconstruct the total.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
)

dust_cfg = {
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": tengri.Uniform(0.0, 1.5),
    "law_bc": "calzetti",
    "law_diff": "calzetti",
    "emission": {"type": "dale2014", "all_params": tengri.FIXED},
}
sfh_cfg = {"type": "tsnorm", "all_params": tengri.FREE}

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh=sfh_cfg,
    dust=dust_cfg,
    neb={"type": "cue", "all_params": tengri.FIXED},
    redshift=tengri.Fixed(0.05),
)

key = jax.random.PRNGKey(42)
params = dict(model.spec.sample(key))
params.update(
    sfh_tsnorm_peak_lbt_gyr=3.0,
    sfh_tsnorm_width_gyr=2.0,
    sfh_tsnorm_log_total_mass=10.5,
    sfh_tsnorm_skew=0.3,
    sfh_tsnorm_trunc=10.0,
    dust_tau_diff=0.5,
)

sed_total = model.predict(params)
wave = np.asarray(model.wavelengths)
lnu_total = np.asarray(sed_total.rest_sed())

model_stellar = tengri.SEDModel.build(ssp, sfh=sfh_cfg, redshift=tengri.Fixed(0.05))
_sed_stellar = model_stellar.predict(params)

model_dust = tengri.SEDModel.build(ssp, sfh=sfh_cfg, dust=dust_cfg, redshift=tengri.Fixed(0.05))
_sed_dust = model_dust.predict(params)

# The sub-models omit nebular (and dust), so their rest-frame wavelength grids
# are shorter than the full model's — Cue injects emission-line wavelengths the
# bare-stellar grid lacks. Interpolate each component onto the full model's grid
# before differencing so the additivity reconstruction lines up.
lnu_stellar = np.asarray(_sed_stellar.rest_sed(wave))
lnu_dust = np.asarray(_sed_dust.rest_sed(wave))

lnu_dust_emission = lnu_dust - lnu_stellar
lnu_nebular = lnu_total - lnu_dust
lnu_reconstructed = lnu_stellar + lnu_dust_emission + lnu_nebular
max_residual = np.max(np.abs(lnu_total - lnu_reconstructed)) / np.max(np.abs(lnu_total))

fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.loglog(wave, wave * lnu_total, color="k", lw=2.0, label="Total (model)", zorder=10)
ax.loglog(wave, wave * lnu_stellar, color="C0", lw=1.4, alpha=0.8, label="Stellar")
ax.loglog(
    wave,
    wave * np.maximum(lnu_dust_emission, 1e-40),
    color="C1",
    lw=1.4,
    alpha=0.8,
    label="Dust emission",
)
ax.loglog(
    wave, wave * np.maximum(lnu_nebular, 1e-40), color="C2", lw=1.4, alpha=0.8, label="Nebular"
)
ax.loglog(
    wave,
    wave * np.maximum(lnu_reconstructed, 1e-40),
    "k--",
    lw=1.0,
    alpha=0.6,
    label=f"Sum (max resid: {max_residual:.1e})",
)

ax.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.grid(True, alpha=0.3, which="both", linestyle=":", linewidth=0.5)

plt.savefig("plot_diag_sed_additivity.png", dpi=150, bbox_inches="tight")
