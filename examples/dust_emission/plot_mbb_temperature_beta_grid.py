"""
Modified blackbody: T_dust × β grid
=====================================

Casey 2012 modified blackbody dust SEDs across the canonical fitter's
two knobs — dust temperature ``T_dust`` and emissivity index ``β``.
Each curve in the top panel is a fixed ``β = 1.8`` MBB swept in T;
the bottom panel fixes T = 30 K and sweeps β. The peak shifts by
~40 μm per 10 K of warming; the sub-mm slope steepens by one
power-law index per Δβ = 1.

Useful when interpreting FIR fits as the ``(T, β)`` degeneracy
projected onto a single sub-mm photometric point.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

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

C_AA_PER_S = 2.998e18

ssp = tengri.load_ssp()


def _build(t_dust=None, beta=None):
    dust = {
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_diff": 0.5,
        "tau_bc": 1.0,
        "emission": {"type": "modified_blackbody", "*": tengri.FIXED},
    }
    model = tengri.SEDModel.build(
        ssp,
        sfh={"type": "const", "*": tengri.FIXED, "log_sfr": 1.0},
        dust=dust,
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    if t_dust is not None:
        p["dust_T"] = jnp.float64(t_dust)
    if beta is not None:
        p["dust_beta_ir"] = jnp.float64(beta)
    return model, p


# Sanity-check what knobs the MBB exposes
_, p_check = _build()
mbb_keys = sorted(
    k
    for k in p_check
    if k.startswith("dust_")
    and ("temp" in k.lower() or "beta" in k.lower() or "t_dust" in k.lower())
)
# Will print to log so the reader can re-discover the actual MBB knob names
# if the API ever drifts; not part of the figure.

T_grid = np.linspace(20.0, 60.0, 7)
beta_grid = np.linspace(1.0, 2.5, 6)

fig, (ax_T, ax_b) = plt.subplots(
    2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"hspace": 0.06}
)

norm_T = mpl.colors.Normalize(vmin=T_grid.min(), vmax=T_grid.max())
for T in T_grid:
    model, p = _build(t_dust=float(T), beta=1.8)
    out = model.predict_rest_sed(p)
    w = np.asarray(out.wavelength)
    ax_T.loglog(w, C_AA_PER_S / w * np.asarray(out.sed), color=plt.cm.plasma(norm_T(T)), lw=1.4)

norm_b = mpl.colors.Normalize(vmin=beta_grid.min(), vmax=beta_grid.max())
for beta in beta_grid:
    model, p = _build(t_dust=30.0, beta=float(beta))
    out = model.predict_rest_sed(p)
    w = np.asarray(out.wavelength)
    ax_b.loglog(
        w, C_AA_PER_S / w * np.asarray(out.sed), color=plt.cm.viridis(norm_b(beta)), lw=1.4
    )

for ax in (ax_T, ax_b):
    ax.set(xlim=(1e5, 1e7), ylim=(1e40, 1e44), ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]")

ax_b.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")

cb_T = fig.colorbar(plt.cm.ScalarMappable(norm=norm_T, cmap="plasma"), ax=ax_T, pad=0.01)
cb_T.set_label(r"$T_{\rm dust}$  [K]   (β = 1.8 fixed)")
cb_b = fig.colorbar(plt.cm.ScalarMappable(norm=norm_b, cmap="viridis"), ax=ax_b, pad=0.01)
cb_b.set_label(r"emissivity index $\beta$   ($T_{\rm dust}$ = 30 K fixed)")

plt.savefig("plot_mbb_temperature_beta_grid.png", dpi=150, bbox_inches="tight")
