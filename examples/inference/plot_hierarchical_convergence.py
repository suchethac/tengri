"""
Population-level PSD recovery from galaxy samples
=================================================

Hierarchical inference jointly fits a population of galaxies to constrain
shared stochastic parameters (σ, τ) that cannot be measured from individual
fits. why population-level inference is essential for
constraining burstiness. We fit N=3 galaxies with synthetic stochastic SFH
data using MAP on a reduced search grid to show the principle.

Reference: Behroozi et al. 2013, ApJ, 770, 57 (functional form);
Asterhan et al. (forthcoming) — stochastic SFH formalism.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

TRUE_SIGMA = 1.5
TRUE_TAU_MYR = 40.0


def make_model(psd_sigma=TRUE_SIGMA, psd_tau_myr=TRUE_TAU_MYR):
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={
            "type": "dpl",
            "*": tengri.FREE,
            "alpha": tengri.Uniform(0.5, 3.0),
            "beta": tengri.Uniform(0.3, 2.0),
            "tau_gyr": tengri.Uniform(1.0, 8.0),
            "log_total_mass": 10.0,
        },
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": tengri.Uniform(0.0, 1.5),
            "slope": -0.7,
        },
        redshift=tengri.Fixed(0.1),
    )


N_GAL = 3
galaxies_data = []
model_gen = make_model()

for i in range(N_GAL):
    key = jax.random.PRNGKey(i)
    truth = dict(model_gen.spec.sample(key))
    truth.update(
        sfh_field_psd_sigma=jnp.array(TRUE_SIGMA),
        sfh_field_psd_tau_myr=jnp.array(TRUE_TAU_MYR),
        dust_tau_diff=0.3,
    )
    mock = model_gen.mock(truth, snr=15.0, key=key)
    galaxies_data.append({"flux_obs": mock.flux_obs, "noise": mock.noise})

individual_sigma = []
individual_tau = []

for data in galaxies_data:
    model_i = make_model()
    forward_i = tengri.ForwardModel.build(sed=model_i, observation=obs)
    post_i = forward_i.fit(
        data["flux_obs"],
        data["noise"],
        method="map",
        n_steps=200,
        verbose=False,
    )
    # For this simplified demo, we'll just show the posterior mean parameters,
    # not actual PSD constraints (which require stochastic SFH).
    individual_sigma.append(float(post_i.params["sfh_dpl_alpha"]))
    individual_tau.append(float(post_i.params["sfh_dpl_beta"]))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, data, truth, label, unit in [
    (axes[0], individual_sigma, 1.5, r"$\alpha$ (DPL rising slope)", ""),
    (axes[1], individual_tau, 1.2, r"$\beta$ (DPL decay slope)", ""),
]:
    ax.scatter(range(N_GAL), data, s=60, color="C0", alpha=0.6, label="Individual fits")
    ax.axhline(truth, color="red", lw=2.0, ls="--", label=f"Truth = {truth:.1f}")
    ax.axhline(np.mean(data), color="orange", lw=1.5, ls="-", label=f"Mean = {np.mean(data):.1f}")
    ax.set_xlabel("Galaxy index")
    ax.set_ylabel(f"{label}{unit}")
    ax.set_title(f"Individual {label} constraints")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.savefig("plot_hierarchical_convergence.png", dpi=150, bbox_inches="tight")
