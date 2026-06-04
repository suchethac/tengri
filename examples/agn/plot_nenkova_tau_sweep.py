"""
CLUMPY torus (Nenkova+2008): optical depth as a fitted parameter
================================================================

The Nenkova et al. (2008) CLUMPY library is the AGN dusty-torus model used
by FSPS and Prospector. tengri ships the same templates (vendored from FSPS
as ``data/nenkova08_torus_grid.h5``) and interpolates them with a pure-JAX
triweight kernel, so the **equatorial optical depth ``agn_tau`` is a fully
differentiable, fitted parameter** — it can be sampled by NUTS, optimised by
MAP, or marginalised by VI, just like in Prospector.

Here we sweep ``agn_tau`` over the grid extent (5–150) at fixed bolometric
luminosity. Higher optical depths reprocess more of the accretion-disc
continuum into the infrared and deepen the 9.7 μm silicate feature, shifting
the mid-IR torus bump.

Because the interpolation is pure JAX, ``jax.grad`` of the SED with respect to
``agn_tau`` is finite — the property that makes the optical depth usable in
gradient-based inference (and which the previous SciPy-interpolated
implementation lacked).

References
----------
Nenkova, M., Sirocky, M. M., Nikutta, R., Ivezić, Ž., & Elitzur, M. 2008,
ApJ, 685, 147 ("AGN Dusty Tori. I. Handling of Clumpy Media").
Conroy, C. & Gunn, J. E. 2010, ApJ, 712, 833 (FSPS).
Johnson, B. D. et al. 2021, ApJS, 254, 22 (Prospector).
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

C_AA_PER_S = 2.998e18

# A luminous AGN on a quiescent host so the torus dominates the mid-IR.
# ``frac=1.0`` scales the AGN to the host bolometric luminosity; ``log_lbol``
# sets the absolute AGN power. ``tau`` (= agn_tau) is the free torus axis.
ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={"type": "const", "*": tengri.FIXED, "log_total_mass": -10.0},
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    agn={
        "disc": {"type": "multicolor", "*": tengri.FIXED},
        "torus": {"type": "nenkova", "*": tengri.FIXED, "tau": tengri.Uniform(5, 150)},
        "*": tengri.FIXED,
        "log_lbol": 12.5,
        "frac": 1.0,
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

tau_values = np.array([10.0, 30.0, 60.0, 100.0, 150.0])
norm = mpl.colors.Normalize(vmin=tau_values.min(), vmax=tau_values.max())
cmap = plt.get_cmap("inferno")

fig, ax = plt.subplots(figsize=(6.6, 4.3))
for tau in tau_values:
    out = model.predict_rest_sed({**baseline, "agn_tau": jnp.float64(tau)})
    wave = np.asarray(out.wavelength)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(tau)), lw=1.5)

# Silicate feature marker.
ax.axvline(9.7e4, color="0.6", ls=":", lw=1.0)
ax.text(9.7e4, ax.get_ylim()[1] * 0.5, r"  9.7 μm silicate", color="0.4", fontsize=8)

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r"equatorial optical depth $\tau$")

ax.set(
    xlim=(1e3, 3e6),
    ylim=(1e43, 5e46),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
    title="CLUMPY torus (Nenkova+2008): optical-depth sweep",
)

plt.tight_layout()
plt.savefig("plot_nenkova_tau_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)
