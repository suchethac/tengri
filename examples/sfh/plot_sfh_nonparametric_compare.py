"""
Non-parametric SFH families compared
======================================

The parametric SFH atlas (``plot_sfh_form_compare.py``) shows seven
classical analytic SFH shapes. Beyond those, tengri ships three
non-parametric families that bin the mass formed in successive lookback
intervals — useful when the data resolve more than ~5 SFR bins and you
want a flexible prior that doesn't impose a strong shape.

Three forms overlaid at their default priors:

- ``continuity``: Leja+2019 fixed-bin continuity prior
- ``dirichlet``: Leja+2017 Dirichlet over normalized bin weights
- ``dense_basis``: Iyer+2019 GP-regularized cumulative SFH

All three are drawn 24 times from their priors and shown as light
trajectories; the parametric ``dpl`` is overlaid as a thick black
reference. The dispersion of each family is the visual prior — wide
clouds mean the form is permissive, narrow clouds mean the form is
restrictive.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

FORMS = [
    ("continuity", "continuity (Leja+2019)", "#cc4477"),
    ("dirichlet", "Dirichlet (Leja+2017)", "#ee8833"),
    ("dense_basis", "dense basis (Iyer+2019)", "#3388aa"),
]
N_DRAWS = 24

ssp = tengri.load_ssp()
fig, axes = plt.subplots(
    1, 3, figsize=(11.5, 4.0), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08}
)

ref = tengri.SEDModel.build(
    ssp, sfh={"type": "dpl", "all_params": tengri.FIXED}, redshift=tengri.Fixed(0.0)
)
p_ref = dict(ref.spec.sample(jax.random.PRNGKey(99)))
sfh_ref = ref.predict_sfh(p_ref)
t_ref = np.asarray(sfh_ref["t_gyr"])
sfr_ref = np.asarray(sfh_ref["sfr_mean"])
mass_ref = np.trapezoid(sfr_ref, t_ref * 1.0e9)
sfr_ref = sfr_ref / mass_ref if mass_ref > 0 else sfr_ref

key0 = jax.random.PRNGKey(7)
for ax, (form, label, color) in zip(axes, FORMS):
    model = tengri.SEDModel.build(
        ssp,
        sfh={"type": form, "all_params": tengri.FREE},
        redshift=tengri.Fixed(0.0),
    )
    for _i, sub_key in enumerate(jax.random.split(key0, N_DRAWS)):
        p = dict(model.spec.sample(sub_key))
        sfh = model.predict_sfh(p)
        t = np.asarray(sfh["t_gyr"])
        sfr = np.asarray(sfh["sfr_mean"])
        mass = np.trapezoid(sfr, t * 1.0e9)
        if mass > 0:
            sfr = sfr / mass
        ax.plot(t, sfr, color=color, lw=0.6, alpha=0.4)
    ax.plot(t_ref, sfr_ref, color="0.15", lw=1.6, ls="--", label="dpl reference")
    ax.text(
        0.95,
        0.93,
        label,
        transform=ax.transAxes,
        fontsize=8,
        color="0.15",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5),
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"lookback time  [Gyr]")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(1e-12, 5e-9)

axes[0].set_ylabel(r"SFR$(t)$ / $M_\star^{\rm tot}$  [yr$^{-1}$]")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")

plt.savefig("plot_sfh_nonparametric_compare.png", dpi=150, bbox_inches="tight")
