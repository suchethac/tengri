"""
D_n(4000) – specific SFR: the Kauffmann+2003 sequence
=======================================================

The Kauffmann+2003 separation of star-forming and quiescent SDSS
galaxies plotted as a sample track: stellar-burst age varied from
30 Myr to 11 Gyr (single-burst SSP), with each model giving a
(``D_n(4000)``, ``sSFR``) pair.

Young bursts have ``D_n(4000) ≲ 1.2`` and ``log sSFR ≳ −9`` (active
star formation, weak break). Old populations climb to
``D_n(4000) ≈ 1.9`` and ``log sSFR ≲ −12`` (quenched, deep break).
The Kauffmann+2003 green-valley cut at ``D_n(4000) ≈ 1.5`` is the
horizontal divider.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18


def _d4000(wave, l_nu):
    blue = (wave >= 3850) & (wave <= 3950)
    red = (wave >= 4000) & (wave <= 4100)
    return float(np.mean(l_nu[red]) / np.mean(l_nu[blue]))


ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "peak_lbt_gyr": tengri.Uniform(0.03, 13.0),
        "width_gyr": 0.05,
        "log_total_mass": 10.0,
        "skew": 0.0,
        "trunc": 13.0,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

ages = np.geomspace(0.03, 11.0, 28)
d4000 = np.empty_like(ages)
ssfr = np.empty_like(ages)

for i, age in enumerate(ages):
    p = {**baseline, "sfh_tsnorm_peak_lbt_gyr": jnp.float64(age)}
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    l_nu = np.asarray(out.rest_sed())
    d4000[i] = _d4000(wave, l_nu)
    sfh = model.predict_sfh(p)
    t = np.asarray(sfh["t_gyr"])
    sfr = np.asarray(sfh["sfr_mean"])
    m_star = float(np.trapezoid(sfr, t * 1e9))
    # current sSFR: SFR at t=0 / M_star
    sfr_now = float(sfr[0]) if sfr[0] > 0 else 1e-15
    ssfr[i] = sfr_now / max(m_star, 1e-30)

fig, ax = plt.subplots(figsize=(6.4, 5.0))
sc = ax.scatter(
    d4000,
    np.log10(np.maximum(ssfr, 1e-15)),
    c=ages,
    cmap="viridis",
    s=44,
    lw=0.4,
    edgecolor="0.2",
    norm=plt.matplotlib.colors.LogNorm(),
)

ax.axvspan(1.5, 1.6, color="0.92", alpha=0.6, lw=0)
ax.text(1.62, -8.5, "Kauffmann+2003\ngreen-valley cut", fontsize=8, color="0.4")

ax.set(
    xlabel=r"$D_n(4000)$",
    ylabel=r"$\log\,\mathrm{sSFR}$  [yr$^{-1}$]",
    xlim=(0.95, 2.05),
    ylim=(-13.5, -8.0),
)
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("Stellar burst age [Gyr]")

fig.tight_layout()
plt.savefig("plot_usecase_d4000_vs_ssfr.png", dpi=150, bbox_inches="tight")
