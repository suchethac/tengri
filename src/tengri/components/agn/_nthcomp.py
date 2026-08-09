# SPDX-License-Identifier: BSD-3-Clause
"""nthcomp warm Comptonization precomputed template support.

Provides JAX-compatible log-space trilinear interpolation over a precomputed
table of Kompaneets equation solutions.  The table is built by
``scripts/build_nthcomp_templates.py``, which calls RELAGN's ``pyNTHCOMP``
(scotthgn/RELAGN, credit A.D. Thomas, ported from XSpec donthcomp.f) as an
external dependency — tengri does **not** ship the Kompaneets solver itself.

Usage
-----
The template table is loaded from ``data/nthcomp_templates.npz`` at import
time if the file exists, and exposed as JAX arrays for JIT-compatible
trilinear interpolation.

Build the template file once with::

    # First clone RELAGN:
    git clone --depth=1 https://github.com/scotthgn/RELAGN.git /tmp/relagn_ref
    # Then build:
    python scripts/build_nthcomp_templates.py

When the file is absent, ``_TABLE_AVAILABLE`` is ``False`` and
``nthcomp_lnu_interp`` raises ``RuntimeError``.  Callers (disc.py) fall back
to the simplified QSOSED-style power-law proxy and emit a one-time warning.

References
----------
Kubota & Done (2018) MNRAS 480 1247 Section 2.2 — warm Comptonization zone.
Zdziarski, Johnson & Magdziarz (1996) MNRAS 283 193 — Kompaneets solver.
"""

from __future__ import annotations

import functools
import warnings
from typing import NamedTuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from tengri._data_setup import package_or_env_data_path

# ── Template loading (lazy — no computation at import time) ───────

_DEFAULT_TEMPLATE_PATH = package_or_env_data_path("nthcomp_templates.h5")


@functools.cache
def _load_nthcomp_templates_impl():
    """Load precomputed nthcomp templates from file.

    Returns a tuple (gamma, kte, ktbb, nu, table_log, available).
    """
    fpath = _DEFAULT_TEMPLATE_PATH
    if not fpath.exists():
        return None, None, None, None, None, False

    try:
        with h5py.File(fpath, "r") as f:
            gamma_jax = jnp.array(f["gamma_grid"][:], dtype=jnp.float32)
            kte_jax = jnp.array(f["kte_grid"][:], dtype=jnp.float32)
            ktbb_jax = jnp.array(f["ktbb_grid"][:], dtype=jnp.float32)
            nu_jax = jnp.array(f["nu_grid"][:], dtype=jnp.float32)
            table = f["table"][:]
        table_log_jax = jnp.array(np.log(np.maximum(table, 1e-37)), dtype=jnp.float32)
        return gamma_jax, kte_jax, ktbb_jax, nu_jax, table_log_jax, True
    except Exception as exc:
        warnings.warn(
            f"Failed to load nthcomp templates from {fpath}: {exc}. "
            "Run scripts/build_nthcomp_templates.py to build them.",
            stacklevel=2,
        )
        return None, None, None, None, None, False


def _get_nthcomp_templates():
    """Get cached nthcomp templates, loading on first call."""
    gamma, kte, ktbb, nu, table_log, available = _load_nthcomp_templates_impl()
    return gamma, kte, ktbb, nu, table_log, available


#: Backward-compat global accessors (for interpolation functions below)
def _get_gamma_jax():
    """Return photon index grid from cached nthcomp templates."""
    gamma, _, _, _, _, _ = _get_nthcomp_templates()
    return gamma


def _get_kte_jax():
    """Return electron temperature grid from cached nthcomp templates."""
    _, kte, _, _, _, _ = _get_nthcomp_templates()
    return kte


def _get_ktbb_jax():
    """Return seed blackbody temperature grid from cached nthcomp templates."""
    _, _, ktbb, _, _, _ = _get_nthcomp_templates()
    return ktbb


def _get_nu_jax():
    """Return frequency grid from cached nthcomp templates."""
    _, _, _, nu, _, _ = _get_nthcomp_templates()
    return nu


def _get_table_jax():
    """Return log-space nthcomp template table from cached templates."""
    _, _, _, _, table_log, _ = _get_nthcomp_templates()
    return table_log


def _is_table_available():
    """Check if nthcomp templates are loaded and available."""
    _, _, _, _, _, available = _get_nthcomp_templates()
    return available


_TABLE_AVAILABLE = _is_table_available()


# ── JAX-compatible interpolation (only valid when _TABLE_AVAILABLE is True)


def _clamp_interp_index(val: jnp.ndarray, grid: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (i_lo, frac) for clamped linear interpolation of val in grid."""
    n = grid.shape[0]
    i_hi = jnp.searchsorted(grid, val, side="right")
    i_lo = jnp.clip(i_hi - 1, 0, n - 2)
    i_hi_c = jnp.clip(i_hi, 1, n - 1)
    span = grid[i_hi_c] - grid[i_lo]
    # Safe gradient pattern: avoid division by near-zero in unselected branch
    # by pre-masking the divisor (double-where idiom).
    span_safe = jnp.where(span > 0, span, 1.0)
    frac = jnp.where(span > 0, (val - grid[i_lo]) / span_safe, 0.0)
    return i_lo, jnp.clip(frac, 0.0, 1.0)


class NthcompTable(NamedTuple):
    """nthcomp Comptonization template arrays, as a JAX pytree.

    Attributes
    ----------
    gamma : ndarray, shape (n_gamma,)
        Photon-index axis.
    kte : ndarray, shape (n_kte,)
        Electron-temperature axis [keV].
    ktbb : ndarray, shape (n_ktbb,)
        Seed-blackbody-temperature axis [keV].
    nu : ndarray, shape (n_nu,)
        Template frequency grid [Hz].
    table_log : ndarray, shape (n_gamma, n_kte, n_ktbb, n_nu)
        ``log`` of the spectral shape.

    Notes
    -----
    A pytree, so it can be handed to ``jax.jit`` as an argument. Closing over
    these arrays instead freezes ~15 MB into every graph that touches a
    Comptonized disc.
    """

    gamma: jnp.ndarray
    kte: jnp.ndarray
    ktbb: jnp.ndarray
    nu: jnp.ndarray
    table_log: jnp.ndarray


def load_nthcomp_table() -> NthcompTable | None:
    """Load the packaged nthcomp templates as a :class:`NthcompTable` pytree.

    This is the ``template_loader`` the Comptonized disc blocks register.

    Returns
    -------
    NthcompTable or None
        ``None`` when the templates are absent — callers then fall back to
        their analytic path, as they did before threading existed.

    Notes
    -----
    **JIT-compatible**: no, deliberately — call it before tracing.
    """
    gamma, kte, ktbb, nu, table_log, available = _get_nthcomp_templates()
    if not available:
        return None
    return NthcompTable(gamma=gamma, kte=kte, ktbb=ktbb, nu=nu, table_log=table_log)


def _nthcomp_lnu_interp_impl(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
    table: NthcompTable | None = None,
) -> jnp.ndarray:
    """Implementation of nthcomp interpolation (used by both forward and VJP).

    ``table`` carries the template arrays. Passing them in — rather than
    reading the module-level cache here — is what lets the forward model
    thread the ~15 MB library through ``jax.jit`` as a ``Parameter`` instead
    of freezing it into the graph as ``Constant`` ops (#1383).
    """
    if table is None:
        if not _is_table_available():
            raise RuntimeError(
                "nthcomp templates not loaded. Run scripts/build_nthcomp_templates.py first."
            )
        table = load_nthcomp_table()

    g = jnp.asarray(gamma, dtype=jnp.float32)
    t = jnp.asarray(kTe_keV, dtype=jnp.float32)
    b = jnp.asarray(kTbb_keV, dtype=jnp.float32)

    gamma_jax = jnp.asarray(table.gamma)
    kte_jax = jnp.asarray(table.kte)
    ktbb_jax = jnp.asarray(table.ktbb)
    ig, fg = _clamp_interp_index(g, gamma_jax)
    it, ft = _clamp_interp_index(t, kte_jax)
    ib, fb = _clamp_interp_index(b, ktbb_jax)

    table_jax = jnp.asarray(table.table_log)
    nu_jax = jnp.asarray(table.nu)

    def _c(dg: int, dt: int, db: int) -> jnp.ndarray:
        """Return table value at the interpolation-cell corner offset (dg, dt, db)."""
        return table_jax[ig + dg, it + dt, ib + db]

    # Trilinear interpolation over 8 corners (gamma × kTe × kTbb) in log space.
    # table_jax stores log(spectral_shape); exponentiating after interpolation
    # gives exact results for exponentially varying features (e.g. Wien seed-BB
    # tail), avoiding the large errors that linear interpolation produces there.
    s00 = _c(0, 0, 0) * (1 - fg) + _c(1, 0, 0) * fg
    s10 = _c(0, 1, 0) * (1 - fg) + _c(1, 1, 0) * fg
    s01 = _c(0, 0, 1) * (1 - fg) + _c(1, 0, 1) * fg
    s11 = _c(0, 1, 1) * (1 - fg) + _c(1, 1, 1) * fg
    s0 = s00 * (1 - ft) + s10 * ft
    s1 = s01 * (1 - ft) + s11 * ft
    log_shape_on_table_grid = s0 * (1 - fb) + s1 * fb
    shape_on_table_grid = jnp.exp(log_shape_on_table_grid)

    # Resample onto the requested nu grid
    nu_f = jnp.asarray(nu, dtype=jnp.float32)
    lnu = jnp.interp(nu_f, nu_jax, shape_on_table_grid, left=0.0, right=0.0)
    return jnp.maximum(lnu, 0.0)


@jax.custom_vjp
def _nthcomp_interp(
    table: NthcompTable,
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
) -> jnp.ndarray:
    """Return the normalized nthcomp L_nu shape via trilinear interpolation.

    The custom-VJP kernel. ``table`` is primal argument 0 so the library can
    be threaded through ``jax.jit`` as an argument; :func:`nthcomp_lnu_interp`
    is the public entry point that resolves it. Extrapolation beyond grid
    bounds is clamped to boundary values.

    Parameters
    ----------
    table : NthcompTable
        Template arrays. Also saved in the residuals, because the backward
        pass re-evaluates the interpolation at ``gamma + eps``.
    nu : jnp.ndarray
        Frequency grid [Hz].
    gamma : scalar jnp array
        Photon index.  Clamped to grid range.
    kTe_keV : scalar jnp array
        Electron temperature [keV].  Clamped to grid range.
    kTbb_keV : scalar jnp array
        Seed temperature [keV].  Clamped to grid range.

    Returns
    -------
    lnu_shape : jnp.ndarray, shape (len(nu),)
        Non-negative spectral shape (integrates to ~1 over nu).

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and JAX-registered VJP.

    The underlying templates are precomputed Kompaneets equation solutions
    in log-space (Kubota & Done 2018, Section 2.2). Trilinear interpolation
    is performed in log(spectral shape) to improve accuracy for exponentially
    varying features (Wien seed-BB tail), then exponentiated. Extrapolation
    beyond grid bounds is clamped to preserve monotonicity at boundaries.

    **Custom VJP**: A finite-difference approximation is used for the gamma
    gradient to work around JAX autodiff limitations with composed operations
    involving ``jnp.interp`` and large output scalars. See _nthcomp_lnu_interp_bwd
    (lines 234–286) for implementation details.

    References
    ----------
    .. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
       of AGN and its implications for the UV/X relation and optical variability,"
       MNRAS, 480, 1247 (2018). arXiv:1804.00171.
       https://doi.org/10.1093/mnras/sty1890
    .. [2] A. A. Zdziarski, G. M. Johnson, and M. Magdziarz, "Inverse Compton
       dominance in the torus emission," MNRAS, 283, 193 (1996).
       https://doi.org/10.1093/mnras/283.1.193
    """
    return _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV, table)


def _nthcomp_lnu_interp_fwd(
    table: NthcompTable,
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
) -> tuple[jnp.ndarray, tuple]:
    """Forward pass for custom VJP of nthcomp_lnu_interp.

    Computes the function value and saves residuals for backward differentiation.
    """
    # Compute the actual output via the implementation function
    result = _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV, table)

    # Save residuals for the backward pass. The table rides along because the
    # backward pass re-evaluates the interpolation at gamma + eps.
    residuals = (table, nu, gamma, kTe_keV, kTbb_keV)
    return result, residuals


def _nthcomp_lnu_interp_bwd(residuals: tuple, g_out: jnp.ndarray) -> tuple:
    """Backward pass for custom VJP of nthcomp_lnu_interp.

    Uses finite-difference approximation for gamma gradient to avoid the
    jnp.interp gradient NaN issue (JAX autodiff limitation with interpolation
    and extrapolation in composed operations).

    To handle overflow when g_out contains very large values, we:
    1. Compute the finite-difference gradient of the raw output
    2. Apply the cotangent vector with care to avoid overflow

    Other parameters (nu, kTe_keV, kTbb_keV) are not differentiated since they
    are held fixed during typical inference workflows (gamma is the primary
    Comptonization parameter tuned during fitting).
    """
    table, nu, gamma, kTe_keV, kTbb_keV = residuals

    # Finite-difference approximation for gamma (the problematic parameter)
    # Use adaptive epsilon based on gamma value
    eps = jnp.maximum(1e-6 * jnp.abs(gamma), 1e-6)
    gamma_plus = gamma + eps
    result_plus = _nthcomp_lnu_interp_impl(nu, gamma_plus, kTe_keV, kTbb_keV, table)
    result = _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV, table)

    # Finite-difference gradient w.r.t. gamma (element-wise)
    # This gives the derivative of each output element w.r.t. gamma
    fd_grad_per_element = (result_plus - result) / eps

    # Chain rule: compute sum of g_out * fd_grad_per_element
    # To avoid overflow: divide by max absolute gradient before accumulating,
    # then rescale using the safe gradient pattern.
    max_grad = jnp.max(jnp.abs(fd_grad_per_element))
    # Safe gradient pattern: pre-mask divisor to avoid NaN from unselected branch
    max_grad_safe = jnp.where(max_grad > 0, max_grad, 1.0)

    # Normalize to unit scale to avoid overflow in multiplication
    # Use pre-masked max_grad_safe to ensure gradient flows safely
    fd_grad_normalized = fd_grad_per_element / max_grad_safe
    g_out_normalized = g_out / max_grad_safe

    # Compute normalized product and sum
    g_gamma_normalized = jnp.sum(g_out_normalized * fd_grad_normalized)

    # Restore the scale using the safe value
    g_gamma = g_gamma_normalized * max_grad_safe * max_grad_safe

    # Ensure result is finite
    g_gamma = jnp.where(jnp.isfinite(g_gamma), g_gamma, 0.0)

    # Return zero gradients for other parameters (held fixed in fitting)
    g_nu = jnp.zeros_like(nu)
    g_kTe = jnp.zeros_like(kTe_keV)
    g_kTbb = jnp.zeros_like(kTbb_keV)

    # The template library is data, never a fit parameter, so its cotangent is
    # structurally zero. It must still be returned: custom_vjp requires one
    # cotangent per primal. Nothing consumes it, so XLA drops the zeros.
    g_table = jax.tree.map(jnp.zeros_like, table)

    return (g_table, g_nu, g_gamma, g_kTe, g_kTbb)


# Register the VJP rule
_nthcomp_interp.defvjp(_nthcomp_lnu_interp_fwd, _nthcomp_lnu_interp_bwd)


def nthcomp_lnu_interp(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
    _template: NthcompTable | None = None,
) -> jnp.ndarray:
    """Normalized nthcomp :math:`L_\\nu` shape via trilinear interpolation.

    Thin dispatcher over the custom-VJP kernel. See :func:`_nthcomp_interp`
    for the physics and the gradient treatment.

    Parameters
    ----------
    nu : ndarray, shape (n_nu,)
        Frequency grid [Hz].
    gamma, kTe_keV, kTbb_keV : Array
        Photon index, electron temperature [keV], seed temperature [keV].
        Each is clamped to the grid range.
    _template : NthcompTable, optional
        Pre-loaded templates, threaded in as a JIT argument by the forward
        model. ``None`` (default) reads the module-level cache, which — under
        trace — bakes ~15 MB into the graph as constants.

    Returns
    -------
    ndarray, shape (n_nu,)
        Non-negative spectral shape.

    Notes
    -----
    **JIT-compatible**: yes. The gradient w.r.t. ``gamma`` is the
    finite-difference rule registered on :func:`_nthcomp_interp`; the
    template cotangent is structurally zero.
    """
    table = _template if _template is not None else load_nthcomp_table()
    if table is None:
        raise RuntimeError(
            "nthcomp templates not loaded. Run scripts/build_nthcomp_templates.py first."
        )
    return _nthcomp_interp(table, nu, gamma, kTe_keV, kTbb_keV)
