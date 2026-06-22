"""
Raw SKIRTOR (Stalevski 2016): the published radiative-transfer template
=======================================================================

``agn={"type": "skirtor_stalevski"}`` returns the SKIRTOR 2016 SED **exactly as
Stalevski's radiative transfer computed it** — accretion disc + clumpy torus +
scattered light, read straight from the full-coverage grid with *no* analytic-disc
substitution and *no* re-normalisation. This is the faithful template that codes
reading SKIRTOR directly (e.g. ProSpect's ``SKIRTOR_interp``) reproduce.

It is deliberately distinct from the other two SKIRTOR flavours tengri ships:

- the composable ``disc.skirtor`` + ``torus.skirtor`` blocks pair CIGALE's
  *analytic* disc with the torus under ``norm="cigale_joint"`` energy balance —
  the right choice for reproducing CIGALE's ``skirtor2016``;
- the monolithic ``agn={"type": "skirtor"}`` pairs the torus with a *power-law*
  disc.

All three are swappable — see ``tengri.list_agn_models()``. This example shows the
**raw** template's two defining behaviours: the inclination-dependent anisotropy
of the unified model, and the silicate 9.7 / 18 μm features that flip from
emission to absorption as the optical depth and viewing angle increase.

**References:**

.. [1] Stalevski, M., Fritz, J., Baes, M., et al. (2012).
   3D radiative-transfer modelling of the dusty torus around AGN.
   MNRAS, 420, 2756. arXiv:1109.1286.

.. [2] Stalevski, M., Ricci, C., Ueda, Y., et al. (2016).
   The dust covering factor in AGN — combining the IR torus emission with the
   polar dust component. MNRAS, 458, 2288. arXiv:1602.01954.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

C_AA_PER_S = 2.998e18

# Negligible host (total mass ~1e-10 Msun) so the raw AGN template dominates.
SFH = {"type": "const", "*": tengri.FIXED, "log_total_mass": -10.0}
DUST = {"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

ssp = tengri.load_ssp()

model = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust=DUST,
    agn={
        "type": "skirtor_stalevski",
        "*": tengri.FIXED,
        "agn_log_lbol": 12.0,
        "agn_frac_agn": 1.0,
        "agn_cos_inc": 0.5,
        "agn_tau_skirtor": 7.0,  # optical depth at 9.7 um (grid [3, 11])
        "agn_oa_skirtor": 40.0,  # torus half-opening angle [deg]
        "agn_p_skirtor": 1.0,
        "agn_q_skirtor": 1.0,
    },
    redshift=tengri.Fixed(0.0),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))


def raw_sed(**overrides):
    out = model.predict_rest_sed({**baseline, **{k: np.float64(v) for k, v in overrides.items()}})
    return np.asarray(out.wavelength), np.asarray(out.sed)


fig, (ax_inc, ax_sil) = plt.subplots(1, 2, figsize=(13.0, 5.0))

# ── Panel (a): inclination sweep (full SED) ───────────────────────────
inclination_deg = np.array([0.0, 30.0, 50.0, 65.0, 75.0, 85.0, 90.0])
cos_inc_values = np.cos(np.radians(inclination_deg))
colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(inclination_deg)))

for cos_inc, inc_deg, color in zip(cos_inc_values, inclination_deg, colors):
    wave, sed = raw_sed(agn_cos_inc=cos_inc)
    ax_inc.loglog(
        wave, C_AA_PER_S / wave * sed, color=color, lw=1.6, alpha=0.85, label=f"{inc_deg:4.0f}°"
    )

ax_inc.set(
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
    xlim=(1e3, 1e6),
    ylim=(1e43, 1e47),
    title="(a) Raw template anisotropy vs inclination",
)
ax_inc.legend(title="Inclination", frameon=False, fontsize=8, loc="lower center", ncol=2)
for um in (3.0, 10.0, 30.0, 100.0):
    ax_inc.axvline(um * 1e4, color="0.85", lw=0.5, alpha=0.5)

# ── Panel (b): silicate 9.7/18 um vs optical depth ────────────────────
tau_values = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
colors_tau = plt.cm.inferno(np.linspace(0.15, 0.85, len(tau_values)))

for tau, color in zip(tau_values, colors_tau):
    # edge-on (cos_inc small) so the silicate band is seen through the torus
    wave, sed = raw_sed(agn_tau_skirtor=tau, agn_cos_inc=0.2)
    mask = (wave >= 3e4) & (wave <= 3e5)
    ax_sil.loglog(
        wave[mask] / 1e4, sed[mask], color=color, lw=2.0, label=rf"$\tau_{{9.7}}={tau:.0f}$"
    )

for um, name in ((9.7, "9.7 μm"), (18.0, "18 μm")):
    ax_sil.axvline(um, color="gray", ls="--", alpha=0.5, lw=1.0)
    ax_sil.text(
        um,
        ax_sil.get_ylim()[1] * 0.6,
        name,
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

ax_sil.set(
    xlim=(3, 30),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mu$m]",
    ylabel=r"$L_\nu$  [erg s$^{-1}$ Hz$^{-1}$]",
    title="(b) Silicate features vs optical depth (edge-on)",
)
ax_sil.legend(frameon=True, fontsize=9, loc="upper right", framealpha=0.95)
ax_sil.grid(True, alpha=0.3, which="both")

fig.tight_layout()
plt.savefig("plot_skirtor_stalevski_raw_template.png", dpi=150, bbox_inches="tight")
print("Saved plot_skirtor_stalevski_raw_template.png")
