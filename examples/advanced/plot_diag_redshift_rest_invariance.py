"""
Rest-frame SED Redshift Invariance
==================================

The rest-frame SED depends only on intrinsic galaxy properties (SFH, dust,
metallicity, nebular, AGN) and is independent of redshift. Redshift only
enters via the observation (wavelength shift, distance dimming, IGM
attenuation). This diagnostic verifies that :meth:`Prediction.rest_sed` returns
bit-identical SEDs across a range of redshifts for identical intrinsic
parameters. Age-of-the-Universe constraints at high-z may truncate the SFH
legitimately, producing smooth variation; any non-smooth jump signals a
coupling bug.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Build model with free redshift parameter
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
obs = tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_g"]))

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "dpl", "all_params": tengri.FIXED, "alpha": 2.0, "beta": 1.5, "tau_gyr": 5.0},
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": 0.3,
    },
    neb={"type": "cue", "all_params": tengri.FIXED},
    redshift=tengri.Uniform(0.0, 5.0),
)

# Sample baseline parameters at z=0
key = jax.random.PRNGKey(0)
baseline = model.spec.sample(key)

# Redshifts to test
z_vals = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])

# Compute rest-frame SEDs, varying only redshift
max_rel_diffs = []
for z in z_vals:
    params_z = {**baseline, "redshift": float(z)}
    sed_z = model.predict(params_z)

    # Compare to z=0 baseline
    if z == 0.0:
        sed_0 = sed_z.rest_sed()
    else:
        # Max relative difference (exclude zeros)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.abs(sed_z.rest_sed() - sed_0) / np.abs(sed_0)
            rel_diff = rel_diff[np.isfinite(rel_diff)]
            max_rel_diff = np.max(rel_diff) if len(rel_diff) > 0 else 0.0
            max_rel_diffs.append(max_rel_diff)

max_rel_diffs = np.array(max_rel_diffs)

# Plot: max relative difference vs redshift
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.semilogy(
    z_vals[1:],
    max_rel_diffs,
    marker="o",
    markersize=8,
    lw=1.5,
    color="C0",
    markerfacecolor="white",
    markeredgewidth=1.5,
)
ax.axhline(
    1e-3, color="gray", linestyle="--", lw=1, alpha=0.5, label=r"$10^{-3}$ (warning threshold)"
)
ax.set_xlabel(r"Redshift $z$")
ax.set_ylabel(r"Max relative difference vs $z=0$ SED")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()
plt.savefig("plot_diag_redshift_rest_invariance.png", dpi=150, bbox_inches="tight")
