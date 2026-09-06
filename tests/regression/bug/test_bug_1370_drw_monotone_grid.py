# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the DRW innovations recursion does not fail open on a descending grid.

``drw_innovations_gp_from_xi`` computes per-step gaps from the physical-time grid
and turns them into a correlation ``rho_i = exp(-dt_i / tau)`` and an innovation
scale ``sqrt(var (1 - rho_i**2))``. Computing ``dt`` as a *signed*
``t_i - t_{i-1}`` makes that arithmetic order-dependent, and the failure is silent:

* a descending grid gives ``dt < 0``, hence ``rho > 1`` and ``1 - rho**2 < 0``;
* the ``clip(1 - rho**2, 0, None)`` floor -- present so float round-off cannot push
  the square root's argument slightly negative -- converts the would-be-loud
  ``sqrt(negative) = NaN`` into ``innov = 0``;
* the recursion degenerates to ``s_i = rho_i s_{i-1}`` with ``rho > 1``, which grows
  geometrically to a **finite, unflagged** ~1e17 sigma (measured 2.1e17 at n=256).

A guard whose failure mode is silent garbage is the bug. The fix takes the gap
magnitude, ``dt = abs(diff(t))``, which is not merely defensive: a DRW kernel
depends only on ``|t_i - t_j|``, and along any *monotone* sequence the consecutive
``|dt|`` telescope to exactly that. So a descending grid becomes a genuine square
root of the same ``K`` rather than garbage, and the clip goes back to guarding only
what it was documented to guard.

Exposure when filed was latent -- the function is not public and every call site
(``sed_model.py``, ``sfh/component.py``, ``registry.py``) passes the canonical
ascending ``make_log_age_grid``. The dense path this replaced
(``drw_linear_gp_from_xi``, built from ``|t_i - t_j|`` directly) was order-agnostic,
so the precondition was introduced by the O(n) swap. These tests pin it closed.

References
----------
.. [1] K. G. Iyer et al., "The star formation history and variability of galaxies,"
   MNRAS, 498, 430 (2020). [physical decorrelation timescale]
.. [2] N. Caplar & S. Tacchella, MNRAS, 487, 3845 (2019). [PSD amplitude, dex]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import make_log_age_grid
from tengri.components.stellar.sfh.gp_sfh import drw_innovations_gp_from_xi

pytestmark = pytest.mark.regression_bug

_LN10 = float(np.log(10.0))

#: Grid size. Small enough that the dense (n, n) oracle is free, large enough that
#: a ``rho > 1`` recursion compounds far past any tolerance. Measured with the fix
#: reverted, on this grid at tau = 3e8 yr: the descending realization reaches
#: **1.06e20** sigma and the non-monotone one 3.6e8 sigma -- both entirely finite.
_N = 48


def _exact_drw_K(log_age_grid, sigma_dex, tau_yr):
    """The DRW covariance on the grid *as ordered*, built independently of the code."""
    t = 10.0 ** np.asarray(log_age_grid)
    var = (sigma_dex * _LN10) ** 2
    return var * np.exp(-np.abs(t[:, None] - t[None, :]) / tau_yr)


def _induced_matrix(sigma_dex, tau_yr, grid):
    """Dense ``M`` with ``gp_x = M xi``, recovered column-wise (the map is linear)."""
    n = grid.shape[0]

    def col(e):
        return drw_innovations_gp_from_xi(e, sigma_dex, tau_yr, grid)[0]

    return np.asarray(jax.vmap(col)(jnp.eye(n)).T)


def _gram(M):
    """``M @ M.T`` with the FPU flags quieted -- see the note below.

    ``exp(-dt/tau)`` underflows to exactly zero wherever a grid step far exceeds
    ``tau``. That is the correct answer, but it sets a shared FPU flag that the
    *next* numpy call reports, so the matmul warns about a condition it did not
    cause. Suppressing it does not weaken anything here: the assertions, not the
    warnings, are what fail if the map is wrong -- pre-fix this product overflowed
    to ``inf`` and every caller below caught it on the value, not the warning.
    """
    with np.errstate(all="ignore"):
        return M @ M.T


@pytest.fixture(scope="module")
def grids():
    """Ascending canonical grid and its reversal, with the ordering asserted.

    Guards the setup: if ``make_log_age_grid`` ever stopped being ascending, every
    assertion below would still pass while testing nothing.
    """
    asc = np.asarray(make_log_age_grid(_N))
    assert np.all(np.diff(asc) > 0), "probe setup failed: canonical grid is not ascending"
    desc = asc[::-1].copy()
    assert np.all(np.diff(desc) < 0), "probe setup failed: reversed grid is not descending"
    return asc, desc


@pytest.mark.parametrize("tau_yr", [1e7, 3e8])
def test_descending_grid_is_a_valid_square_root_not_garbage(grids, tau_yr):
    """LOAD-BEARING. Neuter: restore ``dt = jnp.diff(t)`` and this explodes.

    Verified by reverting the fix: this test and the six others below fail, while
    the two ``..._ascending_path_is_bit_identical`` cases still pass -- they are the
    controls, since the canonical path is genuinely unchanged either way.

    The pre-fix result was not merely inaccurate, it was finite garbage: no NaN, no
    warning, nothing for a caller to catch. The tolerance is machine-precision
    because there is no regime in which a partially-wrong square root is useful.
    """
    _, desc = grids
    sigma = 0.7
    M = _induced_matrix(sigma, tau_yr, desc)
    K = _exact_drw_K(desc, sigma, tau_yr)
    rel = np.max(np.abs(_gram(M) - K)) / np.max(np.abs(K))
    assert rel < 1e-10, (
        f"descending grid does not realize the DRW kernel: ||M M^T - K||/||K|| = "
        f"{rel:.3e}. A signed dt makes rho > 1 and the clip floor swallows the NaN."
    )


def test_descending_realization_stays_within_a_sane_number_of_sigma(grids):
    """The blunt symptom, asserted directly: no 1e17-sigma output.

    ``test_..._valid_square_root`` above would also catch this, but only through a
    covariance identity. This pins the observable a user would actually hit -- a
    log-SFH modulation that a downstream ``exp()`` turns into inf or 0.
    """
    _, desc = grids
    sigma = 0.7
    xi = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (_N,)))
    gp_x = np.asarray(drw_innovations_gp_from_xi(xi, sigma, 3e8, desc)[0])
    assert np.all(np.isfinite(gp_x))
    assert np.max(np.abs(gp_x)) / (sigma * _LN10) < 10.0, (
        f"descending realization reached {np.max(np.abs(gp_x)) / (sigma * _LN10):.3e} "
        "sigma -- the recursion is compounding rho > 1"
    )


def test_descending_grid_gives_the_cholesky_in_that_node_order(grids):
    """Stronger than 'same covariance': it is *the* Cholesky factor of that K.

    Same-covariance alone would admit any right-multiplication by an orthogonal
    matrix. The recursion is structurally lower-triangular with a positive diagonal,
    and such a factor is unique -- so descending is not an alternative square root,
    it is the canonical one transposed onto the reversed node order.
    """
    _, desc = grids
    sigma = 0.7
    M = _induced_matrix(sigma, 3e8, desc)
    with np.errstate(all="ignore"):  # exp(-dt/tau) legitimately underflows to 0
        L = np.linalg.cholesky(_exact_drw_K(desc, sigma, 3e8))
    assert np.max(np.abs(np.triu(M, 1))) == 0.0
    assert bool(np.all(np.diag(M) > 0.0))
    assert np.max(np.abs(M - L)) / (sigma * _LN10) < 1e-10


def test_marginal_variance_is_preserved_in_both_directions(grids):
    """Every node keeps variance ``var`` whichever way the grid runs."""
    asc, desc = grids
    sigma = 0.7
    var = (sigma * _LN10) ** 2
    for name, grid in (("ascending", asc), ("descending", desc)):
        M = _induced_matrix(sigma, 3e8, grid)
        diag = np.diag(_gram(M))
        assert np.allclose(diag, var, rtol=1e-10), f"{name}: marginal variance drifted"


@pytest.mark.parametrize("tau_yr", [1e7, 3e8])
def test_the_canonical_ascending_path_is_bit_identical(grids, tau_yr):
    """``abs`` must be a no-op where ``dt > 0`` already -- this fix changes nothing.

    Asserted as exact equality against an independent reimplementation of the
    recursion, not against a stored snapshot: a snapshot would also pass if both
    sides drifted together.
    """
    asc, _ = grids
    sigma = 0.7
    xi = np.asarray(jax.random.normal(jax.random.PRNGKey(11), (_N,)))
    got = np.asarray(drw_innovations_gp_from_xi(xi, sigma, tau_yr, asc)[0])

    t = 10.0 ** np.asarray(asc)
    var = (sigma * _LN10) ** 2
    sigma_s = np.sqrt(var)
    rho = np.exp(-(t[1:] - t[:-1]) / tau_yr)  # signed on purpose: dt > 0 here
    innov = sigma_s * np.sqrt(1.0 - rho**2)
    want = np.empty(_N)
    want[0] = sigma_s * xi[0]
    for i in range(1, _N):
        want[i] = rho[i - 1] * want[i - 1] + innov[i - 1] * xi[i]
    assert np.all(np.isfinite(got)), "canonical ascending path returned non-finite"
    assert np.max(np.abs(got - want)) / sigma_s < 1e-12


def test_non_monotone_grid_is_bounded_even_though_meaningless(grids):
    """A non-monotone grid has no DRW square root; it must still not blow up.

    The contract this pins is boundedness, not correctness -- consecutive ``|dt|``
    only telescope to ``|t_i - t_j|`` along a monotone sequence, so the induced
    covariance is legitimately not ``K`` here (measured O(1) relative deviation).
    What must never return is the pre-fix behavior: a finite 1e17-sigma answer.
    """
    asc, _ = grids
    mixed = np.asarray(asc).copy()
    mixed[3], mixed[_N - 5] = mixed[_N - 5], mixed[3]
    assert not np.all(np.diff(mixed) > 0) and not np.all(np.diff(mixed) < 0)

    sigma = 0.7
    xi = np.asarray(jax.random.normal(jax.random.PRNGKey(5), (_N,)))
    gp_x = np.asarray(drw_innovations_gp_from_xi(xi, sigma, 3e8, mixed)[0])
    assert np.all(np.isfinite(gp_x))
    assert np.max(np.abs(gp_x)) / (sigma * _LN10) < 10.0


def test_descending_grid_is_still_jit_and_grad_safe(grids):
    """``abs`` sits inside the traced path -- confirm it did not break the transforms.

    ``abs`` is non-differentiable at 0, which a degenerate (zero-gap) grid could hit;
    the canonical grids here have strictly non-zero gaps, so gradients stay finite.
    """
    _, desc = grids
    xi = jax.random.normal(jax.random.PRNGKey(2), (_N,))

    def summ(sigma, tau):
        gp_x, _ = drw_innovations_gp_from_xi(xi, sigma, tau, desc)
        return jnp.sum(gp_x**2)

    val = jax.jit(summ)(0.7, 3e8)
    assert bool(jnp.isfinite(val))
    g_sigma, g_tau = jax.grad(summ, argnums=(0, 1))(0.7, 3e8)
    assert np.isfinite(float(g_sigma)) and float(g_sigma) != 0.0
    assert np.isfinite(float(g_tau))
    assert np.any(float(g_tau) != 0.0), (
        "`float(g_tau)` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
