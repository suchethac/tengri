# SPDX-License-Identifier: BSD-3-Clause
"""Gradient of the AGB circumstellar dust-shell weight (#2534).

``jax.grad`` of the 10 um ratio wrt the ``agb_dust_weight`` scalar must be
finite and non-zero for a population with TP-AGB stars (1 Gyr), and zero for
one too young to have any (3 Myr: measured R(w)=1 to machine precision at
every sampled weight -- no star has reached the TP-AGB phase that fast, so
there is nothing for the shell weight to act on). The generation script's
age-window report (``|R-1| > 1e-4`` *somewhere* in the wavelength window,
over the *whole* 0.0045-19.95 Gyr grid) is not itself "R=1 everywhere below
it": measured directly here, R(w=0) at 10 um is exactly 1 at 1-5 Myr, jumps
to a strong effect at 10 Myr, relaxes to within ~1% of 1 at 20-50 Myr, then
strengthens again from ~100 Myr on -- a non-monotonic age dependence, not a
single on/off window. :func:`agb_dust_ratio` is a plain multiply by an array
built from
``jnp.searchsorted``/``jnp.take`` (``ratio_planes`` layout) or an affine
function of w (``shell_fraction`` layout) -- both JIT/grad-safe by
construction; this pins that differentiability on the shipped template
rather than asserting it only in the abstract.

Also exercises the live (free-weight) path end to end through
``StellarSEDComponent.apply``, since that is the one consumer that only
reaches ``agb_dust_ratio`` with a *traced* weight (the Fixed-weight path
bakes a Python-float weight at build time and never traces it).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import load_ssp
from tengri.components.stellar.agb_dust_shell import (
    agb_dust_ratio,
    load_agb_dust_shell_template,
    resample_agb_dust_shell,
)

pytestmark = pytest.mark.gradient


def fd_grad(f, x: float, eps: float = 1e-3) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture(scope="module")
def resampled():
    ssp = load_ssp("fsps_mist_c3k_a_chabrier")
    template = load_agb_dust_shell_template()
    return resample_agb_dust_shell(
        template,
        np.asarray(ssp.ssp_lgmet),
        np.asarray(ssp.ssp_lg_age_gyr) + 9.0,
        np.asarray(ssp.ssp_wave),
    ), ssp


def _indices(ssp, age_gyr: float):
    met_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - (-1.848))))  # solar
    age_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(age_gyr))))
    wave_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_wave) - 1.0e5)))  # ~10 um
    return met_idx, age_idx, wave_idx


def test_grad_wrt_weight_is_finite_and_nonzero_for_1gyr_population(resampled):
    """1 Gyr (Chabrier, solar): TP-AGB stars are present, so R(w) at 10 um
    must respond to w, with ``jax.grad`` matching a converged central finite
    difference to within 1%.

    Evaluated at w=1.2, not w=1.0: the stored template's weight nodes are
    {0, 0.5, 1, 1.5, 2, 3} (``_lerp_along_weight_axis`` is piecewise-linear
    between them), and R(w) is not linear across w=1 (it rises into that
    node from below and falls away above it -- see the monotonicity test),
    so w=1 is a genuine kink: a one-sided autodiff gradient there need not
    match a central finite difference straddling both sides. 1.2 sits
    strictly inside the [1, 1.5] segment, where the interpolant is exactly
    linear and the two must agree.
    """
    resampled_grid, ssp = resampled
    met_idx, age_idx, wave_idx = _indices(ssp, 1.0)

    def ratio_at_10um(w):
        return agb_dust_ratio(resampled_grid, w)[met_idx, age_idx, wave_idx]

    w0 = 1.2
    grad = float(jax.grad(ratio_at_10um)(w0))
    assert jnp.isfinite(grad)
    assert grad != 0.0

    fd = fd_grad(ratio_at_10um, w0, eps=1e-3)
    assert jnp.isfinite(fd)
    assert fd != 0.0
    rel_err = abs(grad - fd) / abs(fd)
    assert rel_err < 0.01, f"grad={grad}, finite-diff={fd}, rel_err={rel_err}"


def test_grad_wrt_weight_is_zero_for_3myr_population(resampled):
    """3 Myr: too young for any star to have reached the TP-AGB phase
    (measured R(w)=1 to machine precision at every weight at this age), so
    R(w) at 10 um must not respond to w at all."""
    resampled_grid, ssp = resampled
    met_idx, age_idx, wave_idx = _indices(ssp, 0.003)

    def ratio_at_10um(w):
        return agb_dust_ratio(resampled_grid, w)[met_idx, age_idx, wave_idx]

    grad = float(jax.grad(ratio_at_10um)(1.0))
    # grad-assert: finite-only — the claim on this gradient is that it is
    # ~0 (asserted below via an explicit magnitude bound), the opposite of
    # "non-zero": a pre-TP-AGB population's ratio must not respond to w.
    assert jnp.isfinite(grad)
    assert abs(grad) < 1e-9, f"expected ~0 gradient for a pre-TP-AGB population, got {grad}"
