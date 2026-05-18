"""
Stochastic SFH: Burstiness Timescale τ
========================================

τ (in Myr) controls how long bursts last. Short τ = fast flickering;
long τ = sustained episodes.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_psd_tau_sweep_001.png
   :alt: plot_psd_tau_sweep
   :class: sphx-glr-single-img

"""

import jax
import matplotlib.pyplot as plt

from tengri import Fixed, Parameters, SEDModel, Uniform, load_ssp, setup_style
from tengri.analysis.plotting import sfh_sed_comparison

setup_style()


ssp = load_ssp()

# Build Parameters with tsnorm + GP field for stochastic SFH
spec = Parameters(
    mean_sfh_type=["tsnorm", "field"],
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
    sfh_tsnorm_width_gyr=Fixed(2.0),
    sfh_tsnorm_skew=Fixed(0.3),
    sfh_tsnorm_trunc=Fixed(2.0),
    sfh_field_psd_sigma=Fixed(1.0),
    sfh_field_psd_tau_myr=Uniform(30, 3000),  # will be overridden
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

model = SEDModel(spec, ssp)

# Sweep parameter with stochastic samples
key = jax.random.PRNGKey(42)
values = [30, 100, 300, 1000, 3000]

fig = sfh_sed_comparison(
    model, "sfh_field_psd_tau_myr", values, cmap="viridis", n_stochastic=5, key=key
)
fig.suptitle("Stochastic SFH: Burstiness Timescale τ", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig("plot_psd_tau_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
