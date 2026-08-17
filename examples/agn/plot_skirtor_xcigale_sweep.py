"""
SKIRTOR torus (full X-CIGALE grid): optical depth and inclination
=================================================================

This is tengri's full-grid SKIRTOR torus (Stalevski+2012, 2016), following the
X-CIGALE ``skirtor2016`` conventions: a 5-D clumpy two-phase library indexed by
equatorial optical depth ``tau``, radial and polar density gradients ``p`` / ``q``,
half-opening angle ``oa``, and inclination ``cos i`` (plus an optional Casey-2012
polar-dust graybody). It is the science-grade counterpart to the
parameter-*averaged* ``skirtor_agnfitter`` library — and, having the full grid,
it responds strongly to its parameters.

The panels sweep the two most consequential axes:

- **Equatorial optical depth** ``tau`` (left): a thicker torus deepens the
  9.7 μm silicate absorption and redistributes the reprocessed power.

- **Inclination** (right): edge-on lines of sight (low ``cos i``) graze the
  optically-thick equatorial dust (Type-2-like, deep silicate absorption);
  face-on sight lines look down the polar funnel (Type-1-like).

The full-grid SKIRTOR peaks near 40 μm, redward of the ~25 μm
AGNfitter-averaged ``skirtor_agnfitter`` library (see
``plot_skirtor_agnfitter_sweep.py``). The torus contribution is isolated by
subtracting the disc-only SED.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty torus
   around AGN," MNRAS 420, 2756 (2012). arXiv:1109.1286.
.. [2] M. Stalevski et al., "The dust covering factor in active galactic
   nuclei," MNRAS 458, 2288 (2016). arXiv:1602.01954.
.. [3] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission,"
   A&A 622, A103 (2019). arXiv:1811.03094.
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

SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": -10.0}
DUST = {"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

# SKIRTOR grid (CIGALE subset): tau 3-11, oa 10-80, cos i in (0, 1].
TAU_VALUES = np.linspace(3.0, 11.0, 7)
COS_INC_VALUES = np.linspace(0.1, 0.95, 7)
OA_REF = 40.0
COS_INC_REF = 0.85
TAU_REF = 7.0

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


WAVE, DISC_ONLY = _build_sed(None)


def torus_sed(oa: float, cos_inc: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelength [AA], nu*L_nu [erg/s]) for the SKIRTOR torus alone."""
    wave, total = _build_sed(
        {
            "type": "skirtor",
            "all_params": tengri.FIXED,
            "oa_skirtor": oa,
            "cos_inc": cos_inc,
            "tau_skirtor": tau,
        }
    )
    disc = np.interp(wave, WAVE, DISC_ONLY)
    return wave, C_AA / wave * np.clip(total - disc, 0.0, None)


fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

# ── Panel 1: equatorial optical depth sweep ─────────────────────────────
ax = axes[0]
norm_t = mpl.colors.Normalize(vmin=TAU_VALUES.min(), vmax=TAU_VALUES.max())
cmap_t = plt.get_cmap("viridis")
for tau in TAU_VALUES:
    wave, nu_l_nu = torus_sed(OA_REF, COS_INC_REF, tau)
    ax.loglog(wave, nu_l_nu, color=cmap_t(norm_t(tau)), lw=1.6)
ax.set_title(rf"Optical-depth sweep  ($\cos i = {COS_INC_REF}$)", fontsize=10)
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
sm_t = mpl.cm.ScalarMappable(norm=norm_t, cmap=cmap_t)
fig.colorbar(sm_t, ax=ax, pad=0.01).set_label(r"equatorial optical depth  $\tau$", fontsize=9)

# ── Panel 2: inclination sweep ──────────────────────────────────────────
ax = axes[1]
norm_i = mpl.colors.Normalize(vmin=COS_INC_VALUES.min(), vmax=COS_INC_VALUES.max())
cmap_i = plt.get_cmap("cividis")
for cos_inc in COS_INC_VALUES:
    wave, nu_l_nu = torus_sed(OA_REF, cos_inc, TAU_REF)
    ax.loglog(wave, nu_l_nu, color=cmap_i(norm_i(cos_inc)), lw=1.6)
ax.set_title(rf"Inclination sweep  ($\tau = {TAU_REF:.0f}$)", fontsize=10)
sm_i = mpl.cm.ScalarMappable(norm=norm_i, cmap=cmap_i)
fig.colorbar(sm_i, ax=ax, pad=0.01).set_label(r"$\cos i$  (1 = face-on)", fontsize=9)

for ax in axes:
    ax.set_xlim(3e3, 3e6)
    ax.set_ylim(3e43, 3e45)
    ax.set_xlabel(r"Rest-frame wavelength $\lambda$  [$\mathrm{\AA}$]")
    ax.grid(True, which="major", alpha=0.2)

fig.suptitle(
    "SKIRTOR torus (full X-CIGALE grid): optical depth and inclination",
    fontsize=11.5,
    weight="bold",
)
fig.tight_layout()
plt.savefig("plot_skirtor_xcigale_sweep.png", dpi=150, bbox_inches="tight")
