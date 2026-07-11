"""
Two-component dust: birth cloud obscures only young stars
==========================================================

The Charlot & Fall two-component dust model separates birth-cloud dust (young
stars only, age < ~10 Myr) from diffuse ISM dust (all stars). Two panels show:
(left) V-band transmission versus age for three (τ_bc, τ_diff) combinations,
revealing the sharp ~10 Myr transition; (right) full transmission spectra for
1 Myr and 1 Gyr stars under the same dust column.

Reference: Charlot & Fall 2000, ApJ, 539, 718 (age-dependent dust attenuation).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

wave = jnp.linspace(1000.0, 10000.0, 500)
log_ages = jnp.linspace(5.0, 10.2, 200)
ages = 10.0**log_ages
LAW_KW = dict(law_bc="power_law", law_diff="power_law", n_slope=-0.7)

fig, (ax_age, ax_spec) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A — V-band transmission vs stellar age, three dust columns
i_v = int(jnp.argmin(jnp.abs(wave - 5500.0)))
configs = [
    (1.5, 0.3, r"$\tau_{\rm bc}=1.5$, $\tau_{\rm diff}=0.3$"),
    (0.5, 0.5, r"$\tau_{\rm bc}=0.5$, $\tau_{\rm diff}=0.5$"),
    (0.0, 0.8, r"$\tau_{\rm bc}=0.0$, $\tau_{\rm diff}=0.8$"),
]
for tau_bc, tau_diff, label in configs:
    trans = tengri.two_component_dust(wave, ages, tau_bc, tau_diff, **LAW_KW)
    ax_age.plot(np.array(log_ages), np.array(trans[:, i_v]), lw=1.4, label=label)
ax_age.axvline(7.0, ls=":", color="gray", lw=0.8, alpha=0.6)
ax_age.set_xlabel(r"log$_{10}$(stellar age / yr)")
ax_age.set_ylabel("Transmission at V-band")
ax_age.legend(fontsize=9, frameon=False)

# Panel B — spectrum for young vs old star under canonical (τ_bc=1.5, τ_diff=0.3)
trans = tengri.two_component_dust(wave, ages, tau_v1=1.5, tau_v2=0.3, **LAW_KW)
i_young = int(jnp.argmin(jnp.abs(log_ages - 6.0)))  # 1 Myr
i_old = int(jnp.argmin(jnp.abs(log_ages - 9.0)))  # 1 Gyr
ax_spec.plot(wave / 1e4, np.array(trans[i_young]), lw=1.4, label="Young star (1 Myr)")
ax_spec.plot(wave / 1e4, np.array(trans[i_old]), lw=1.4, label="Old star (1 Gyr)")
ax_spec.set_xlabel(r"Wavelength [$\mu$m]")
ax_spec.set_ylabel("Transmission")
ax_spec.legend(fontsize=9, frameon=False)

fig.tight_layout()
plt.savefig("plot_two_component.png", dpi=150, bbox_inches="tight")
