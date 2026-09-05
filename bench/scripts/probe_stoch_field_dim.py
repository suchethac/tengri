#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Report the sampled dimension and metric conditioning of the field fixture.

``n_free`` counts NAMED free parameters and deliberately omits the field latent
vector; ``n_latent`` is the flattened size the sampler actually sees (#1408).
Quoting the first for a field model says D = 10 for a posterior a sampler
experiences as D = 74, and that distinction is the entire reason the fixture
exists -- ``blackjax.window_adaptation_low_rank``'s ``max_rank`` defaults to 10,
so a "low-rank" mass matrix is a full-rank one below D = 10.

Also reports the analytic metric's condition number at the MAP, because the
low-rank question is a question about correlation structure and this is the
cheapest description of it available.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/probe_stoch_field_dim.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tengri  # noqa: E402
from benchmark_notebook_sampler import NOTEBOOKS  # noqa: E402
from tengri import Fitter  # noqa: E402
from tengri.analysis.mock import generate_mock  # noqa: E402
from tengri.inference._sample_utils import _maybe_map_init  # noqa: E402
from tengri.inference.backends.mcmc._shared import _get_flat_logdensity  # noqa: E402
from tengri.inference.preconditioning import negative_hessian_metric  # noqa: E402


def main(notebook: str = "stoch-field") -> None:
    """Build the fixture, report D, then the metric spectrum at the MAP."""
    cfg = NOTEBOOKS[notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    print(f"fixture {notebook}")
    print(f"  n_free (named)   = {sed.spec.n_free}")
    print(f"  n_latent         = {sed.spec.n_latent}")
    print(f"  named free       = {', '.join(sed.spec.free_params)}")

    k_truth, k_mock, k_fit = jax.random.split(jax.random.PRNGKey(cfg["seed"]), 3)
    mock = generate_mock(sed, sed.spec.sample(k_truth), key=k_mock, snr=cfg["snr"])
    fitter = Fitter(
        sed,
        np.asarray(mock["flux_obs"]),
        np.asarray(mock["noise"]),
        data_type="photometry",
    )
    init_params, _ = _maybe_map_init(fitter, k_fit, None, False)
    log_p2, _unravel, init_flat, data_args = _get_flat_logdensity(fitter, init_params)
    print(f"  sampled D        = {int(init_flat.shape[0])}")

    metric = negative_hessian_metric(log_p2, init_flat, data_args)
    eigenvalues = np.sort(np.asarray(jnp.linalg.eigvalsh(metric)))[::-1]
    cond = float(eigenvalues[0] / eigenvalues[-1])
    print(f"  metric condition = {cond:.4g}")
    # The question low-rank asks: how many directions carry the curvature? The
    # method fits a rank-k correction to a diagonal, so it pays exactly when
    # the spectrum has a short heavy tail above the bulk.
    for k in (3, 5, 10, 20):
        if k < len(eigenvalues):
            print(
                f"  lambda_{k:<3} / lambda_max = {eigenvalues[k] / eigenvalues[0]:.3e}"
                f"   lambda_{k} / lambda_min = {eigenvalues[k] / eigenvalues[-1]:.3e}"
            )
    above_bulk = int(np.sum(eigenvalues > 2.0 * np.median(eigenvalues)))
    print(f"  eigenvalues above 2x the median = {above_bulk} of {len(eigenvalues)}")
    print(
        "  (blackjax's low-rank adaptation masks eigenvalues inside "
        "[1/cutoff, cutoff] with cutoff=2, so the count above is roughly how "
        "many directions its correction has to represent)"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stoch-field")
