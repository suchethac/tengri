#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
r"""Does ``blackjax.mgrad_gaussian`` apply to tengri's stochastic SFH field?

The case for it is strong on its face. ``mgrad_gaussian`` is the marginal
sampler for latent Gaussian models -- targets `q(x) ∝ exp(f(x)) N(x; m, C)` --
and tengri's stochastic SFH is a Gaussian-process field: eleven physical
parameters plus a latent vector over the lookback grid. If any configuration in
the library was designed for a configuration tengri actually has, that is it.

This script checks the two facts that decide it, and both are cheap enough that
there is no excuse for arguing them instead:

1. **What is `C` for tengri?** The field is parameterized **non-centered**:
   ``compute_field_gp`` maps ``xi ~ N(0, I)`` through the OU Cholesky, and
   ``drw_latent_log_prior`` at the shipped ``centering = 1.0`` is exactly
   ``-1/2 zeta^T zeta`` plus a constant. Checked numerically below against a
   standard normal, over a range of ``psd_sigma``, because the claim that
   matters is that `C` does **not** depend on the PSD hyperparameters either.
2. **What does `mgrad_gaussian` do with `C = I`?** ``svd_from_covariance``
   is a dense SVD of `C`; at `C = I` its `U` is orthonormal-arbitrary and its
   `Gamma` is all ones, so the kernel's ``Gamma_1``, ``Gamma_2``, ``Gamma_3``
   collapse to scalars and the two `O(D^2)` matvecs per step are a rotation
   applied and undone. Checked numerically below.

It also reports the D = 75 census, because no report in this project has: how
many free parameters ``recipes.stochastic_sfh_jwst`` actually has and how many
of them are field latents.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/probe_latent_gaussian_fit.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tengri  # noqa: E402
from tengri.components.stellar.sfh.gp_sfh import drw_latent_log_prior  # noqa: E402


def check_latent_prior(n: int = 64) -> None:
    """Is the field latent prior exactly ``N(0, I)``, whatever the PSD says?"""
    key = jax.random.PRNGKey(0)
    zeta = jax.random.normal(key, (n,))
    reference = -0.5 * float(zeta @ zeta) - 0.5 * n * np.log(2.0 * np.pi)
    print("1. field latent prior at the shipped centering = 1.0")
    print(f"   -1/2 zeta^T zeta - n/2 log(2 pi)      = {reference:.10f}")
    for sigma in (0.1, 0.3, 1.0, 3.0):
        got = float(drw_latent_log_prior(zeta, sigma, centering=1.0))
        print(
            f"   drw_latent_log_prior(zeta, sigma={sigma:<4}) = {got:.10f}"
            f"   delta = {got - reference:+.3e}"
        )
    print(
        "   => C = I exactly, and independent of psd_sigma. The prior carries "
        "no correlation for a latent-Gaussian method to exploit."
    )


def check_mgrad_degeneracy(n: int = 64) -> None:
    """What does ``mgrad_gaussian``'s machinery reduce to when ``C = I``?"""
    from blackjax.mcmc.marginal_latent_gaussian import svd_from_covariance

    cov = jnp.eye(n)
    U, Gamma, U_t = svd_from_covariance(cov)
    delta = 0.7
    Gamma_1 = Gamma * delta / (delta + 2 * Gamma)
    Gamma_3 = (delta + 2 * Gamma) / (delta + 4 * Gamma)
    print("\n2. mgrad_gaussian's SVD machinery at C = I")
    print(f"   Gamma unique values          = {np.unique(np.asarray(Gamma))}")
    print(f"   Gamma_1 unique values        = {np.unique(np.asarray(Gamma_1))}")
    print(f"   Gamma_3 unique values        = {np.unique(np.asarray(Gamma_3))}")
    print(f"   max |U U^T - I|              = {float(jnp.max(jnp.abs(U @ U_t - cov))):.3e}")
    print(
        "   => every per-coordinate weight is the SAME scalar, and the two "
        "O(D^2) matvecs per step apply and undo one rotation. The kernel is an "
        "isotropic first-order step -- MALA with extra bookkeeping."
    )


def census_stochastic_model() -> None:
    """How many free parameters does the D = 75 stochastic recipe actually have?"""
    print("\n3. the D = 75 stochastic-SFH census (no fit, model build only)")
    try:
        # The recipe carries neb={'type': 'cue'}, so it needs the BARE-stellar
        # grid; pairing it with a wNE grid double-counts nebular emission and
        # the loader says so.
        ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)
    except Exception as exc:  # noqa: BLE001
        print(f"   SKIPPED -- SSP grid unavailable: {type(exc).__name__}: {exc}")
        return
    try:
        from tengri import SEDModel, recipes
        from tengri.observation import Photometry

        bands = [
            "jwst_f090w",
            "jwst_f115w",
            "jwst_f150w",
            "jwst_f200w",
            "jwst_f277w",
            "jwst_f356w",
            "jwst_f444w",
        ]
        obs = Photometry.from_names(bands)
        model = SEDModel.build(ssp_data=ssp, observation=obs, **recipes.stochastic_sfh_jwst())
        free = list(model.spec.free_params)
    except Exception as exc:  # noqa: BLE001
        print(f"   SKIPPED -- model build failed: {type(exc).__name__}: {str(exc)[:200]}")
        return
    # ``free_params`` counts NAMED parameters and deliberately omits the field
    # latent vector; ``n_latent`` is the flattened size the sampler actually
    # sees (#1408). Reporting only the first would say D = 11 for a posterior
    # a sampler experiences as D = 75, which is the whole distinction.
    print(f"   named free parameters        = {len(free)}")
    print(f"   n_free                       = {model.spec.n_free}")
    print(f"   n_latent (sampled dimension) = {model.spec.n_latent}")
    print(f"   field latents                = {model.spec.n_latent - model.spec.n_free}")
    print("   named free: " + ", ".join(free))


def main() -> None:
    """Run the three checks in order."""
    check_latent_prior()
    check_mgrad_degeneracy()
    census_stochastic_model()


if __name__ == "__main__":
    main()
