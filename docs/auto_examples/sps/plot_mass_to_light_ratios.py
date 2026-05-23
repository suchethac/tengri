"""
Stellar mass-to-light ratios vs SSP age, per band
====================================================

The conversion from observed flux to stellar mass — the mass-to-light
ratio ``M/L`` — depends on stellar age and the photometric band used.
For a single-burst stellar population, ``M/L`` rises monotonically
with age in every band; the rise is steepest in the blue, where
massive bright stars dominate young populations, and shallowest in
the K band, where light is dominated by red giants at every age past
~1 Gyr.

We compute ``M_star / L_band`` (in ``M_sun / L_sun``) for a narrow
single-burst SSP swept from 30 Myr to 11 Gyr in four bands:
SDSS ``g``, SDSS ``r``, SDSS ``i``, 2MASS ``Ks``.
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

# AB-mag → nu*L_nu zeropoint per band needs the filter effective wavelength.
# We'll compute L_band = integral of L_nu T_nu dnu / integral T_nu dnu directly
# from predict_photometry, then convert to L_band-in-Lsun via the band's
# effective frequency.

C_AA_PER_S = 2.998e18
L_SUN_ERG_S = 3.839e33

BANDS = [
    ("sdss_g",   "g (SDSS)"),
    ("sdss_r",   "r (SDSS)"),
    ("sdss_i",   "i (SDSS)"),
    ("2mass_ks", "K_s (2MASS)"),
]
COLORS = plt.cm.viridis(np.linspace(0.05, 0.92, len(BANDS)))

obs = tengri.Observation(
    photometry=tengri.Photometry.from_names([b for b, _ in BANDS])
)
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "*": tengri.FIXED,
         "peak_lbt_gyr": tengri.Uniform(0.03, 13.0),
         "width_gyr": 0.05,
         "log_peak_sfr": 1.0, "skew": 0.0, "trunc": 13.0},
    dust={"type": "two_component", "*": tengri.FIXED,
          "tau_diff": 0.0, "tau_bc": 0.0},
    redshift=tengri.Fixed(0.01),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

ages = np.geomspace(0.03, 1.0, 16)  # SFH normalisation gets noisy past ~1 Gyr
                                      # (burst clipped at universe age)
ml_grid = np.empty((len(BANDS), ages.size))

for j, age in enumerate(ages):
    p = {**baseline, "sfh_tsnorm_peak_lbt_gyr": jnp.float64(age)}
    # Total stellar mass from the SFH normalisation:
    sfh = model.predict_sfh(p)
    t = np.asarray(sfh["t_gyr"])
    sfr = np.asarray(sfh["sfr_mean"])
    m_star = float(np.trapezoid(sfr, t * 1e9))
    # Band luminosity: F_nu × 4π d_L^2 to get L_nu, then × nu_eff to L_band.
    # At z=0.01 the flux is essentially L_nu / (4π d_L^2). We avoid the
    # cosmology by computing L_band ratio M/L using the model's photometry
    # output scaled to L_sun via the effective-frequency multiplication.
    flux = np.asarray(model.predict_photometry(p))     # erg/s/cm^2/Hz
    # Approximate distance modulus at z=0.01: d_L ≈ 43.6 Mpc.
    d_l_cm = 43.6 * 3.086e24
    L_nu = flux * 4 * np.pi * d_l_cm**2            # erg/s/Hz
    nu_eff = C_AA_PER_S / wave_eff                 # Hz
    L_band = L_nu * nu_eff / L_SUN_ERG_S           # L_sun
    ml_grid[:, j] = m_star / np.maximum(L_band, 1e-12)

fig, ax = plt.subplots(figsize=(7.0, 4.6))
for (band, label), color, m_l in zip(BANDS, COLORS, ml_grid):
    ax.loglog(ages, m_l, color=color, lw=1.6, label=label)

ax.set_xlabel("Stellar burst age  [Gyr]")
ax.set_ylabel(r"$M_\star\,/\,L$  [$M_\odot\,/\,L_\odot$]")
ax.set_ylim(0.03, 3.0)
ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.tight_layout()
fig.savefig("plot_mass_to_light_ratios.png", dpi=150, bbox_inches="tight")
