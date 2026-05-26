"""
Velocity-dispersion broadening of stellar absorption features
================================================================

The Mg b 5170 Å region of an old stellar population observed at
spectral resolution R = 3000, convolved with increasing stellar
velocity dispersion ``σ_v`` from 50 to 400 km/s. The classic kinematic
diagnostic — line core depth tracks σ_v, asymmetric wings appear with
rotational broadening (not modelled here, sigma only).
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


SIGMA_GRID = np.linspace(50.0, 400.0, 8)
Z = 0.05  # avoid predict_spectrum NaN at z=0
WAVE_OBS = jnp.linspace(5000 * (1 + Z), 5350 * (1 + Z), 1400)
WAVE_REST = np.asarray(WAVE_OBS) / (1 + Z)

ssp = tengri.load_ssp()
spec = tengri.Spectroscopy(wave_obs=WAVE_OBS, resolution=3000.0)
obs = tengri.Observation(spectroscopy=spec)
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "*": tengri.FIXED, "peak_lbt_gyr": 7.0,
         "width_gyr": 1.5, "log_peak_sfr": 1.0,
         "skew": 0.0, "trunc": 13.5},
    dust={"type": "two_component", "*": tengri.FIXED,
          "tau_diff": 0.0, "tau_bc": 0.0},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

norm = mpl.colors.Normalize(vmin=SIGMA_GRID.min(), vmax=SIGMA_GRID.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(7.0, 4.6))
for sigma in SIGMA_GRID:
    params = {**baseline, "sigma_v_kms": jnp.float64(sigma)}
    spec_out = model.predict_spectrum(params, wave_obs=WAVE_OBS)
    flux = np.asarray(spec_out)
    cont_mask = (WAVE_REST >= 5200) & (WAVE_REST <= 5230)
    f_cont = np.median(flux[cont_mask])
    ax.plot(WAVE_REST, flux / f_cont, color=cmap(norm(sigma)), lw=1.2)

ax.axvline(5167, color="0.55", lw=0.4, ls=":")
ax.axvline(5173, color="0.55", lw=0.4, ls=":")
ax.axvline(5184, color="0.55", lw=0.4, ls=":")
ax.text(5175, 1.06, "Mg b triplet", fontsize=8, color="0.4", ha="center")

ax.set(xlim=(5050, 5300), ylim=(0.78, 1.10),
       xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
       ylabel=r"$F_\lambda\,/\,F_{\rm cont}$  (normalised at 5200-5230 Å)")
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cb.set_label(r"$\sigma_v$  [km s$^{-1}$]")

fig.tight_layout()
plt.savefig("plot_sigma_v_absorption_broadening.png", dpi=150,
            bbox_inches="tight")
