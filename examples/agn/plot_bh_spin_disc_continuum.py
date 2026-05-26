"""
Black hole spin effect on accretion disc UV peak temperature
=============================================================

The dimensionless spin parameter a* determines the innermost stable circular
orbit (ISCO). Higher spin pushes ISCO inward, raising peak disc temperature
and shifting the UV bump bluer. This demonstrates the classic Kerr black hole
effect on thin disc accretion: Schwarzschild (a*=0) → near-extremal Kerr
(a*=0.998).
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "tau_gyr": 3.0,
        "log_total_mass": 10.0,
        "alpha": 2.0,
        "beta": 2.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    agn={
        "type": "composable",
        "disc": {"type": "multicolor", "*": tengri.FIXED},
        "torus": {"type": "skirtor", "*": tengri.FIXED, "tau_skirtor": 7.0},
        "lines": {"type": "nlr", "*": tengri.FIXED},
        "*": tengri.FIXED,
        "log_lbol": 11.5,
        "log_mbh": 8.0,
        "log_ledd": -0.5,
        "frac": 1.0,  # without this the composable AGN is multiplied by zero
        # Promote a_spin to FREE so the sweep at predict time actually flows;
        # a bare tengri.FREE at per-param level is swallowed by '*: FIXED'.
        "a_spin": tengri.Uniform(0.0, 0.998),
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Spin values: Schwarzschild, intermediate, near-extremal Kerr
a_spin_values = np.array([0.0, 0.5, 0.9, 0.998])
norm = mpl.colors.Normalize(vmin=a_spin_values.min(), vmax=a_spin_values.max())
cmap = plt.get_cmap("hot_r")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for a_spin in a_spin_values:
    params = {**baseline, "agn_a_spin": jnp.float64(a_spin)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(a_spin)), lw=1.5, label=f"a* = {a_spin:.3f}")

ax.set_xlim(100, 3000)
ax.set_ylim(5e43, 2e45)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax.legend(loc="upper right", framealpha=0.95)

fig.tight_layout()
plt.savefig("plot_bh_spin_disc_continuum.png", dpi=150, bbox_inches="tight")
