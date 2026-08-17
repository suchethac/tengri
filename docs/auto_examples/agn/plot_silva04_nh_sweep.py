"""
Silva+04 torus: Obscuration and the 9.7 μm silicate feature
===============================================================

The Silva, Maiolino & Granato (2004) AGN torus templates are empirical
reprocessed-dust SEDs binned by line-of-sight hydrogen column density
``agn_log_nh_silva``. As the column rises from unobscured (Type-1-like,
:math:`N_\\mathrm{H} \\sim 10^{22}\\,\\mathrm{cm^{-2}}`) to Compton-thick
(:math:`N_\\mathrm{H} \\sim 10^{25}\\,\\mathrm{cm^{-2}}`):

- the near-IR continuum is increasingly suppressed, and
- the **9.7 μm silicate feature** turns from weak emission into deep
  absorption — the classic signature of an edge-on, dusty torus.

The left panel shows the full reprocessed SED; the right panel zooms on the
mid-IR so the silicate feature's evolution with column density is visible.
Column density is encoded by color.

References
----------
.. [1] L. Silva, A. Maiolino & G. L. Granato, "Connecting the active galactic
   nucleus and the host galaxy in the infrared," MNRAS 355, 973 (2004).
   arXiv:astro-ph/0403468. https://doi.org/10.1111/j.1365–2966.2004.08380.x
.. [2] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the radio-to-X-ray
   spectral energy distributions of AGNs," A&A 688, A46 (2024).
   arXiv:2405.12111. https://doi.org/10.1051/0004–6361/202449329
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
from tengri.utils.physics_constants import C_AA  # speed of light [Angstrom/s]

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp()

# Minimal host: stellar light suppressed so the torus stands alone.
SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": -10.0}
DUST = {"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

# Silva+04 grid: log N_H in [22, 25]. Sweep the full obscuration range.
LOG_NH_VALUES = np.linspace(22.0, 25.0, 7)

# A torus block normalizes to the disc luminosity, so a disc must be present;
# the torus contribution is then isolated by subtracting the disc-only SED.
BASE_AGN = {
    "disc": {"type": "multicolor", "all_params": tengri.FIXED},
    "all_params": tengri.FIXED,
    "log_lbol": 12.0,
    "lum_ratio": 1.0,
}


def _build_sed(torus: dict | None) -> tuple[np.ndarray, np.ndarray]:
    agn = dict(BASE_AGN)
    if torus is not None:
        agn["torus"] = torus
    model = tengri.SEDModel.build(ssp, sfh=SFH, dust=DUST, agn=agn, redshift=tengri.Fixed(0.05))
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    return np.asarray(model.wavelengths), np.asarray(out.rest_sed())


WAVE, DISC_ONLY = _build_sed(None)  # disc-only baseline (subtracted off)


def torus_sed(log_nh: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelength [AA], nu*L_nu [erg/s]) for the Silva+04 torus alone."""
    wave, total = _build_sed(
        {"type": "silva04", "all_params": tengri.FIXED, "log_nh_silva": log_nh}
    )
    disc = np.interp(wave, WAVE, DISC_ONLY)  # disc build uses a different grid
    return wave, C_AA / wave * np.clip(total - disc, 0.0, None)


fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

norm = mpl.colors.Normalize(vmin=LOG_NH_VALUES.min(), vmax=LOG_NH_VALUES.max())
cmap = plt.get_cmap("cividis")

for log_nh in LOG_NH_VALUES:
    wave, nu_l_nu = torus_sed(log_nh)
    color = cmap(norm(log_nh))
    axes[0].loglog(wave, nu_l_nu, color=color, lw=1.6)
    axes[1].loglog(wave, nu_l_nu, color=color, lw=1.6)

# ── Panel 1: full reprocessed SED ───────────────────────────────────────
axes[0].set_xlim(3e3, 3e6)
axes[0].set_ylim(2e43, 3e45)
axes[0].set_title("Full reprocessed SED", fontsize=10)
axes[0].set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

# ── Panel 2: mid-IR zoom on the 9.7 micron silicate feature ─────────────
axes[1].set_xlim(3e4, 3e5)  # 3-30 micron
axes[1].set_ylim(3e44, 1.3e45)
axes[1].set_title(r"Mid-IR zoom: 9.7 $\mu$m silicate feature", fontsize=10)
axes[1].axvline(9.7e4, color="0.4", ls=":", lw=1.0, zorder=0)
axes[1].text(9.7e4, 3.4e44, r"9.7 $\mu$m", fontsize=8, ha="center", color="0.4", rotation=90)

for ax in axes:
    ax.set_xlabel(r"Rest-frame wavelength $\lambda$  [$\mathrm{\AA}$]")
    ax.grid(True, which="major", alpha=0.2)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=axes, pad=0.01, fraction=0.046)
cb.set_label(r"$\log_{10}(N_\mathrm{H} / \mathrm{cm^{-2}})$", fontsize=9)

fig.suptitle(
    "Silva+04 torus: obscuration and the silicate feature",
    fontsize=11.5,
    weight="bold",
)
plt.savefig("plot_silva04_nh_sweep.png", dpi=150, bbox_inches="tight")
