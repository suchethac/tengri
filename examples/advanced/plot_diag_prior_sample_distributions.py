"""
Diagnosing prior sampling distributions with empirical histograms
==================================================================

When constructing a model with priors like ``Uniform(0, 2)``, the sampling
method ``model.spec.sample(key)`` should actually draw from that declared
distribution. This example verifies the sampling implementation empirically:
we draw 10000 samples from a model with mixed prior types (Uniform, LogUniform)
and compare each empirical histogram against its theoretical PDF.

For each parameter, we compute the Kolmogorov-Smirnov (KS) statistic to flag
any significant deviation (KS > 0.05 suggests a bug in the sampler).
"""

import warnings
from functools import partial

import jax
import jax.random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

model = tengri.SEDModel.build(
    tengri.load_ssp("fsps_prsc_miles_chabrier"),
    sfh={"type": "dpl", "alpha": tengri.FREE, "beta": tengri.FREE, "all_params": tengri.FIXED},
    dust={"type": "two_component", "all_params": tengri.FIXED},
)
spec = model.spec
free_params = spec.free_params
free_dists = {name: spec._distributions[name] for name in free_params}

n_samples = 10000
key = jax.random.PRNGKey(42)
samples = [dict(spec.sample(jax.random.fold_in(key, i))) for i in range(n_samples)]
param_samples = {name: np.array([s[name] for s in samples]) for name in free_params}

n_params = len(free_params)
ncols, nrows = 2, (n_params + 1) // 2
fig, axes = plt.subplots(nrows, ncols, figsize=(8.0, 2.8 * nrows))
axes = np.atleast_1d(axes).flatten()

ks_stats = {}


def _log_uniform_cdf(x, lo, hi):
    """CDF for LogUniform distribution."""
    return np.log(x / lo) / np.log(hi / lo)


def _uniform_cdf(x, lo, hi):
    """CDF for Uniform distribution."""
    return (x - lo) / (hi - lo)


for idx, (param_name, samples_arr) in enumerate(param_samples.items()):
    ax = axes[idx]
    dist = free_dists[param_name]
    ax.hist(samples_arr, bins=50, density=True, alpha=0.6, color="C0", edgecolor="k", lw=0.5)

    lo, hi = dist.bounds
    is_loguniform = "LogUniform" in dist.__class__.__name__
    if is_loguniform:
        x_eval = np.logspace(np.log10(lo), np.log10(hi), 200)
    else:
        x_eval = np.linspace(lo, hi, 200)
    log_prob_vals = np.array([dist.log_prob(np.array(x)) for x in x_eval])
    ax.plot(x_eval, np.exp(log_prob_vals), "r-", lw=1.5, label="Declared PDF")

    if "LogUniform" in dist.__class__.__name__:
        cdf_fn = partial(_log_uniform_cdf, lo=lo, hi=hi)
    elif "Uniform" in dist.__class__.__name__:
        cdf_fn = partial(_uniform_cdf, lo=lo, hi=hi)
    else:
        cdf_fn = None

    if cdf_fn is not None:
        ks_stat = stats.kstest(samples_arr, cdf_fn)[0]
        ks_stats[param_name] = ks_stat
        ax.text(
            0.98,
            0.97,
            f"KS = {ks_stat:.4f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel(param_name, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.tick_params(labelsize=8)

for idx in range(len(free_params), len(axes)):
    axes[idx].set_visible(False)

fig.tight_layout()
plt.savefig("plot_diag_prior_sample_distributions.png", dpi=150, bbox_inches="tight")
