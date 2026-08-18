"""
Prospector prior families: continuity vs bursty vs Dirichlet vs PSB
====================================================================

Tengri ships the non-parametric SFH priors that appear most often in
Prospector papers, all with the *published* prior on the SFR ratios:

- ``continuity``: Leja+2019 — StudentT(μ=0, σ=0.3, df=2) on log-SFR
  ratios between adjacent bins.

- ``bursty_continuity``: Tacchella+2022 — same shape, but the per-ratio σ
  switches between 1.0 dex (younger edge < 1 Gyr) and 0.3 dex (older). The
  prior is intentionally wider on recent ratios so the data can drive
  burstiness.

- ``dirichlet``: Leja+2017 — symmetric Dirichlet(1,…,1) on mass
  fractions via Beta(1, 1) stick-breaking auxiliaries.

- ``psb_suess2022``: Suess+2022 — post-starburst SFH that splits the
  recent past into a youngest [0, t_last] bin and a [t_last, t_flex] flex
  zone. ``t_last`` and ``t_flex`` are free, with uniform priors.

This script draws ``N_DRAWS`` SFR trajectories from each family's prior
and overlays them. The visual width of each band *is* the prior — bursty
should be the widest at recent lookback times, continuity the narrowest.

A fifth family, ``prospector_beta`` (Wang+2024), is also registered with
the same Student-t ratio prior; constructing it with redshift-dependent
edges via :func:`~tengri.make_agebins_from_zred` requires passing
``bin_edges_gyr`` through the composer, which is a separate plumbing
follow-up.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import Parameters, SEDModel
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

N_DRAWS = 40

# ``bursty_continuity`` is registered but not yet validated against the DSPS
# *flux* forward path, so it is gated out of ``SEDModel.build``. Since we only
# draw the star-formation *history* (via ``predict_sfh``, which the gate does
# not touch), we construct every family through the expert flat-kwarg
# ``Parameters(...)`` escape hatch — uniform across validated and gated types.


def _sfh_model(sfh_type):
    spec = Parameters(mean_sfh_type=sfh_type, redshift=0.0)
    return SEDModel(spec=spec, ssp_data=ssp, precompute=False)


FORMS = [
    ("continuity", "continuity (Leja+19)", "#3388aa"),
    ("bursty_continuity", "bursty (Tacchella+22)", "#cc4477"),
    ("dirichlet", "Dirichlet (Leja+17)", "#ee8833"),
    ("psb_suess2022", "PSB (Suess+22)", "#55aa44"),
]

ssp = tengri.load_ssp()

fig, axes = plt.subplots(
    1, 4, figsize=(15.0, 4.0), sharex=True, sharey=True, gridspec_kw={"wspace": 0.06}
)

key0 = jax.random.PRNGKey(7)
for ax, (form, label, color) in zip(axes, FORMS):
    model = _sfh_model(form)

    sfr_stack = []
    for sub_key in jax.random.split(key0, N_DRAWS):
        p = dict(model.spec.sample(sub_key))
        sfh = model.predict_sfh(p)
        t = np.asarray(sfh["t_gyr"])
        sfr = np.asarray(sfh["sfr_mean"])
        mass = float(np.trapezoid(sfr, t * 1.0e9))
        if mass > 0:
            sfr = sfr / mass
        sfr_stack.append(sfr)
        ax.plot(t, sfr, color=color, lw=0.4, alpha=0.25)

    sfr_arr = np.asarray(sfr_stack)
    median = np.median(sfr_arr, axis=0)
    ax.plot(t, median, color="0.1", lw=1.4, label="median")
    ax.text(
        0.97,
        0.94,
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

plt.savefig("plot_prospector_priors_compare.png", dpi=150, bbox_inches="tight")
