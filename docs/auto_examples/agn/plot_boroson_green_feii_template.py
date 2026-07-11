"""
Boroson & Green (1992) Fe II template: the optical/UV iron blanket
==================================================================

``agn={"feii": {"type": "boroson_green"}}`` adds the empirical **Boroson & Green
(1992)** Fe II pseudo-continuum to a composable AGN — the broad iron-line blanket
that fills the 2200–3000 Å and 4400–4700 Å windows of type-1 quasars. Its
amplitude is set by ``agn_fe2_strength`` (the standard :math:`R_{\\rm Fe}` ratio of
Fe II to broad H\\ :math:`\\beta`).

The template is the PyQSOFit empirical Fe II compilation (Vestergaard & Wilkes
2001 in the UV, Boroson & Green 1992 in the optical), so the line ratios within
each iron complex follow the *measured* I Zw 1 spectrum rather than a smooth
parametrization. ``boroson_green`` exposes it as a standalone ``feii`` block —
the same template the analytic ``blr`` lines carry, usable on its own. Other Fe II
options (``grahsp``, ``qsogen_balmer``) are swappable — see
``tengri.list_agn_models()``.

Panel (a) plots the Fe II **excess** over the ``agn_fe2_strength = 0`` baseline,
which isolates the iron humps; panel (b) shows the same Fe II riding on the bright
accretion-disc continuum, where it fills in the windows between the broad lines.

**References:**

.. [1] Boroson, T. A. & Green, R. F. (1992). The emission-line properties of
   low-redshift quasi-stellar objects. ApJS, 80, 109.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

C_AA_PER_S = 2.998e18

SFH = {"type": "const", "*": tengri.FIXED, "log_total_mass": -10.0}
DUST = {"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

ssp = tengri.load_ssp()


def _feii_model(feii_type):
    return tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust=DUST,
        agn={
            "type": "composable",
            "disc": {"type": "multicolor", "*": tengri.FIXED},
            "feii": {"type": feii_type, "*": tengri.FIXED},
            "*": tengri.FIXED,
            "log_lbol": 12.0,
            "frac": 1.0,
        },
        redshift=tengri.Fixed(0.0),
    )


def _feii_excess(model, strength):
    """Fe II-only Lnu: SED at the requested strength minus the strength=0 SED."""
    base = dict(model.spec.sample(jax.random.PRNGKey(0)))
    wave = np.asarray(model.predict_rest_sed(base).wavelength)
    sed0 = np.asarray(model.predict_rest_sed({**base, "agn_fe2_strength": np.float64(0.0)}).sed)
    sed = np.asarray(
        model.predict_rest_sed({**base, "agn_fe2_strength": np.float64(strength)}).sed
    )
    return wave, sed - sed0


fig, (ax_sweep, ax_cmp) = plt.subplots(1, 2, figsize=(13.0, 5.0))

# ── Panel (a): Fe II strength sweep (boroson_green) ───────────────────
bg = _feii_model("boroson_green")
strengths = np.linspace(0.25, 2.0, 6)
norm = mpl.colors.Normalize(vmin=strengths.min(), vmax=strengths.max())
cmap = plt.get_cmap("plasma")

for strength in strengths:
    wave, excess = _feii_excess(bg, strength)
    mask = (wave >= 2000) & (wave <= 5200)
    ax_sweep.plot(wave[mask], excess[mask], color=cmap(norm(strength)), lw=1.5)

for ang, name in ((2480, "UV Fe II\n2200–3000 Å"), (4570, "optical Fe II\n4400–4700 Å")):
    ax_sweep.axvline(ang, color="0.6", ls="--", lw=0.8, alpha=0.6)
    ax_sweep.text(ang, ax_sweep.get_ylim()[1] * 0.02, name, fontsize=8, ha="center", va="bottom")

ax_sweep.set(
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"Fe II excess  $L_\nu$  [erg s$^{-1}$ Hz$^{-1}$]",
    xlim=(2000, 5200),
    title="(a) Boroson & Green Fe II vs strength",
)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax_sweep, pad=0.01).set_label(r"$R_{\rm Fe}$ (agn_fe2_strength)", fontsize=9)

# ── Panel (b): Fe II riding on the disc continuum ─────────────────────
base = dict(bg.spec.sample(jax.random.PRNGKey(0)))
wave = np.asarray(bg.predict_rest_sed(base).wavelength)
mask = (wave >= 2000) & (wave <= 5200)
sed_off = np.asarray(bg.predict_rest_sed({**base, "agn_fe2_strength": np.float64(0.0)}).sed)
sed_on = np.asarray(bg.predict_rest_sed({**base, "agn_fe2_strength": np.float64(1.5)}).sed)

ax_cmp.plot(
    wave[mask], sed_off[mask], color="0.55", lw=1.5, ls="--", label=r"disc only ($R_{\rm Fe}=0$)"
)
ax_cmp.plot(
    wave[mask], sed_on[mask], color="#cc4477", lw=1.7, label=r"disc + Fe II ($R_{\rm Fe}=1.5$)"
)
ax_cmp.fill_between(wave[mask], sed_off[mask], sed_on[mask], color="#cc4477", alpha=0.2)

ax_cmp.set(
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$L_\nu$  [erg s$^{-1}$ Hz$^{-1}$]",
    xlim=(2000, 5200),
    title="(b) Fe II filling the continuum",
)
ax_cmp.legend(frameon=False, fontsize=9, loc="upper right")
ax_cmp.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("plot_boroson_green_feii_template.png", dpi=150, bbox_inches="tight")
print("Saved plot_boroson_green_feii_template.png")
