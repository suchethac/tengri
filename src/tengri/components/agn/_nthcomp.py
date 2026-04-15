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

import warnings
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Template loading (lazy — no computation at import time)
# ---------------------------------------------------------------------------

#: Grid axes stored in the npz file (set by load_nthcomp_templates).
_GAMMA_JAX: jnp.ndarray | None = None
_KTE_JAX: jnp.ndarray | None = None
_KTBB_JAX: jnp.ndarray | None = None
_NU_JAX: jnp.ndarray | None = None
_TABLE_JAX: jnp.ndarray | None = None

#: True once templates are successfully loaded.
_TABLE_AVAILABLE: bool = False

_DEFAULT_TEMPLATE_PATH = Path(__file__).parents[4] / "data" / "nthcomp_templates.h5"


def load_nthcomp_templates(path: Path | None = None) -> bool:
    """Load precomputed nthcomp templates from an npz file.

    Called automatically at import time if ``data/nthcomp_templates.h5``
    exists.  Also callable manually to load from a custom path.

    Parameters
    ----------
    path : Path, optional
        Path to the npz file.  Defaults to ``data/nthcomp_templates.npz``
        relative to the package root.

    Returns
    -------
    bool
        True if templates were loaded successfully.
    """
    global _GAMMA_JAX, _KTE_JAX, _KTBB_JAX, _NU_JAX, _TABLE_JAX, _TABLE_AVAILABLE

    fpath = Path(path) if path is not None else _DEFAULT_TEMPLATE_PATH
    if not fpath.exists():
        return False

    try:
        with h5py.File(fpath, "r") as f:
            _GAMMA_JAX = jnp.array(f["gamma_grid"][:], dtype=jnp.float32)
            _KTE_JAX = jnp.array(f["kte_grid"][:], dtype=jnp.float32)
            _KTBB_JAX = jnp.array(f["ktbb_grid"][:], dtype=jnp.float32)
            _NU_JAX = jnp.array(f["nu_grid"][:], dtype=jnp.float32)
            table = f["table"][:]
        # Store log(table) for log-space trilinear interpolation.
        # Clamped to a small positive floor to avoid log(0); zeros in the
        # table correspond to spectral regions where nthcomp returns no flux.
        _TABLE_JAX = jnp.array(np.log(np.maximum(table, 1e-37)), dtype=jnp.float32)
        _TABLE_AVAILABLE = True
        return True
    except Exception as exc:
        warnings.warn(
            f"Failed to load nthcomp templates from {fpath}: {exc}. "
            "Run scripts/build_nthcomp_templates.py to build them.",
            stacklevel=2,
        )
        return False


# Try to auto-load at import time.
load_nthcomp_templates()


# ---------------------------------------------------------------------------
# JAX-compatible interpolation (only valid when _TABLE_AVAILABLE is True)
# ---------------------------------------------------------------------------


def _clamp_interp_index(val: jnp.ndarray, grid: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (i_lo, frac) for clamped linear interpolation of val in grid."""
    n = grid.shape[0]
    i_hi = jnp.searchsorted(grid, val, side="right")
    i_lo = jnp.clip(i_hi - 1, 0, n - 2)
    i_hi_c = jnp.clip(i_hi, 1, n - 1)
    span = grid[i_hi_c] - grid[i_lo]
    frac = jnp.where(span > 0, (val - grid[i_lo]) / span, 0.0)
    return i_lo, jnp.clip(frac, 0.0, 1.0)


def _nthcomp_lnu_interp_impl(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
) -> jnp.ndarray:
    """Implementation of nthcomp interpolation (used by both forward and VJP)."""
    if not _TABLE_AVAILABLE:
        raise RuntimeError(
            "nthcomp templates not loaded. Run scripts/build_nthcomp_templates.py first."
        )

    g = jnp.asarray(gamma, dtype=jnp.float32)
    t = jnp.asarray(kTe_keV, dtype=jnp.float32)
    b = jnp.asarray(kTbb_keV, dtype=jnp.float32)

    ig, fg = _clamp_interp_index(g, _GAMMA_JAX)
    it, ft = _clamp_interp_index(t, _KTE_JAX)
    ib, fb = _clamp_interp_index(b, _KTBB_JAX)

    def _c(dg: int, dt: int, db: int) -> jnp.ndarray:
        return _TABLE_JAX[ig + dg, it + dt, ib + db]

    # Trilinear interpolation over 8 corners (gamma × kTe × kTbb) in log space.
    # _TABLE_JAX stores log(spectral_shape); exponentiating after interpolation
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
    lnu = jnp.interp(nu_f, _NU_JAX, shape_on_table_grid, left=0.0, right=0.0)
    return jnp.maximum(lnu, 0.0)


@jax.custom_vjp
def nthcomp_lnu_interp(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
) -> jnp.ndarray:
    """Return the normalised nthcomp L_nu shape via trilinear interpolation.

    Requires templates to have been loaded (``_TABLE_AVAILABLE`` is True).
    Extrapolation beyond grid bounds is clamped to boundary values.

    Parameters
    ----------
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
    """
    return _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV)


def _nthcomp_lnu_interp_fwd(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
) -> tuple[jnp.ndarray, tuple]:
    """Forward pass for custom VJP of nthcomp_lnu_interp.

    Computes the function value and saves residuals for backward differentiation.
    """
    # Compute the actual output via the implementation function
    result = _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV)

    # Save residuals for the backward pass
    residuals = (nu, gamma, kTe_keV, kTbb_keV)
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
    nu, gamma, kTe_keV, kTbb_keV = residuals

    # Finite-difference approximation for gamma (the problematic parameter)
    # Use adaptive epsilon based on gamma value
    eps = jnp.maximum(1e-6 * jnp.abs(gamma), 1e-6)
    gamma_plus = gamma + eps
    result_plus = _nthcomp_lnu_interp_impl(nu, gamma_plus, kTe_keV, kTbb_keV)
    result = _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV)

    # Finite-difference gradient w.r.t. gamma (element-wise)
    # This gives the derivative of each output element w.r.t. gamma
    fd_grad_per_element = (result_plus - result) / eps

    # Chain rule: compute sum of g_out * fd_grad_per_element
    # To avoid overflow: divide by max absolute gradient before accumulating,
    # then rescale
    max_grad = jnp.max(jnp.abs(fd_grad_per_element))
    max_grad_safe = jnp.where(max_grad > 0, max_grad, 1.0)

    # Normalize to unit scale to avoid overflow in multiplication
    fd_grad_normalized = fd_grad_per_element / max_grad_safe
    g_out_normalized = g_out / max_grad_safe

    # Compute normalized product and sum
    g_gamma_normalized = jnp.sum(g_out_normalized * fd_grad_normalized)

    # Restore the scale
    g_gamma = g_gamma_normalized * max_grad_safe * max_grad_safe

    # Ensure result is finite
    g_gamma = jnp.where(jnp.isfinite(g_gamma), g_gamma, 0.0)

    # Return zero gradients for other parameters (held fixed in fitting)
    g_nu = jnp.zeros_like(nu)
    g_kTe = jnp.zeros_like(kTe_keV)
    g_kTbb = jnp.zeros_like(kTbb_keV)

    return (g_nu, g_gamma, g_kTe, g_kTbb)


# Register the VJP rule
nthcomp_lnu_interp.defvjp(_nthcomp_lnu_interp_fwd, _nthcomp_lnu_interp_bwd)
