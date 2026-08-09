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


def _nthcomp_lnu_interp_impl(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
) -> jnp.ndarray:
    """Implementation of nthcomp interpolation (used by both forward and VJP)."""
    if not _is_table_available():
        raise RuntimeError(
            "nthcomp templates not loaded. Run scripts/build_nthcomp_templates.py first."
        )

    g = jnp.asarray(gamma, dtype=jnp.float32)
    t = jnp.asarray(kTe_keV, dtype=jnp.float32)
    b = jnp.asarray(kTbb_keV, dtype=jnp.float32)

    gamma_jax = _get_gamma_jax()
    kte_jax = _get_kte_jax()
    ktbb_jax = _get_ktbb_jax()
    ig, fg = _clamp_interp_index(g, gamma_jax)
    it, ft = _clamp_interp_index(t, kte_jax)
    ib, fb = _clamp_interp_index(b, ktbb_jax)

    table_jax = _get_table_jax()
    nu_jax = _get_nu_jax()

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


@jax.custom_jvp
def nthcomp_lnu_interp(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
) -> jnp.ndarray:
    """Return the normalized nthcomp L_nu shape via trilinear interpolation.

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

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and JAX-registered VJP.

    The underlying templates are precomputed Kompaneets equation solutions
    in log-space (Kubota & Done 2018, Section 2.2). Trilinear interpolation
    is performed in log(spectral shape) to improve accuracy for exponentially
    varying features (Wien seed-BB tail), then exponentiated. Extrapolation
    beyond grid bounds is clamped to preserve monotonicity at boundaries.

    **Custom JVP**: a finite-difference approximation supplies the ``gamma``
    derivative, because differentiating the composed ``jnp.interp`` chain
    directly returns NaN. ``nu``, ``kTe_keV`` and ``kTbb_keV`` are held fixed
    during fitting and carry exactly zero derivative, as before.

    The rule is a ``custom_jvp``, not a ``custom_vjp`` (#1206). A ``custom_vjp``
    is **opaque to forward mode** — ``jax.jvp`` raises ``TypeError: can't apply
    forward-mode autodiff (jvp) to a custom_vjp function`` — which takes out
    geoVI, whose metric is built with forward mode, for every AGN model
    reaching this kernel. A ``custom_jvp`` serves forward mode directly and
    reverse mode by transposition; the transpose of ``fd_grad * d_gamma`` is
    ``sum(g * fd_grad)``, exactly the reverse pass this replaces.

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
    return _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV)


@nthcomp_lnu_interp.defjvp
def _nthcomp_lnu_interp_jvp(primals: tuple, tangents: tuple) -> tuple:
    """Forward-mode rule: a finite-difference derivative in ``gamma`` only.

    Parameters
    ----------
    primals : tuple
        ``(nu, gamma, kTe_keV, kTbb_keV)`` -- see :func:`nthcomp_lnu_interp`.
    tangents : tuple
        Tangents of those same four operands. Only the ``gamma`` tangent
        contributes; the other three are discarded, matching the reverse rule
        this replaces.

    Returns
    -------
    tuple
        ``(primal_out, tangent_out)``, both ``ndarray, shape (n_nu,)``
        [dimensionless -- a normalized spectral shape].

    Notes
    -----
    **JIT/grad/vmap-safe**: yes -- and, unlike the ``custom_vjp`` spelling this
    replaces, ``jvp``-safe, which is what geoVI's forward-mode metric needs
    (#1206).

    The ``gamma`` derivative is a one-sided finite difference with an adaptive
    step, because differentiating the composed ``jnp.interp`` chain
    analytically returns NaN.

    **``kTe_keV`` carries a real derivative that this rule deliberately drops.**
    The wording here used to be that the other three operands "are held fixed
    during fitting and carry exactly zero derivative". The first half is a
    configuration choice, not a fact — ``kd18_disc_model.py`` declares
    ``kt_warm = Uniform(0.1, 0.5, ...)``, so a user can and does free it — and
    the second half is false: a central difference gives
    ``d ln f / d ln kTe`` ~ -0.24, an order-unity sensitivity, where this rule
    returns exactly ``0.0``.

    Supplying it would cost a second :func:`_nthcomp_lnu_interp_impl` call on
    *every* AGN JVP (~+50% on this kernel), paid by the majority of fits that
    leave ``kt_warm`` pinned. So the trade-off stands, but it is a trade-off and
    is now documented as one. The silent half is handled at build time:
    ``_warn_dead_gradient_params`` in ``forward/sed_model.py`` emits a
    :class:`~tengri.config.exceptions.DeadGradientParameterWarning` when
    ``agn_kt_warm`` is freed, so a fit cannot quietly return the prior.

    **No cotangent rescaling.** The previous reverse rule divided by
    ``max|fd_grad|`` "to avoid overflow" and restored the scale with
    ``* max * max``. That inverted the intended effect: with ``max|fd_grad|``
    ~1e-17, an incoming cotangent of 1e30 became ``g / max`` ~1e47 -- past
    float32's 3.4e38 -- and the trailing ``where(isfinite(g), g, 0.0)`` turned
    the resulting ``inf`` into a **silent zero gradient**. Measured before this
    change: ``d/d(gamma)`` came back exactly ``0.0`` at cotangent 1e30 for every
    ``(gamma, kTe)`` tried, while being correct at cotangent 1. The unscaled
    product ``sum(g * fd_grad)`` is ~1e13 at that cotangent -- nowhere near the
    limit -- so dropping the rescaling removes both the overflow and the
    fail-open that hid it.
    """
    nu, gamma, kTe_keV, kTbb_keV = primals
    _, d_gamma, _, _ = tangents

    primal_out = _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV)

    # Adaptive one-sided step: relative for large gamma, absolute near zero.
    #
    # 1e-3, not the 1e-6 carried over from the custom_vjp spelling. The impl is a
    # composed ``jnp.interp`` chain, so the finite difference is a subtraction of two
    # nearly equal ~1e-16 values: at 1e-6 the surviving digits are cancellation
    # remainder, not slope. Measured against a converged central difference at three
    # off-node gammas (2.37/2.53/2.64 — 2.5 is a grid node where the derivative is
    # genuinely undefined and any FD comparison is meaningless)::
    #
    #     h        2.37      2.53      2.64
    #     1e-7     -100%     -100%     -100%     <- differences to exactly 0.0
    #     ~2.5e-6   -21%      +47%      +5.9%    <- the old step
    #     1e-4      +0.6%     +0.0%     -1.3%
    #     1e-3      -0.1%     +0.0%     -0.2%    <- plateau
    #     1e-2      +0.6%     +0.3%     +0.3%
    #
    # The old step was not uniformly biased — it was wrong by -10% to +54% depending
    # on where in the grid gamma sat, which is why a single-step check never caught
    # it. The plateau is two decades wide; 1e-3 sits in its middle.
    eps = jnp.maximum(1e-3 * jnp.abs(gamma), 1e-3)
    shifted = _nthcomp_lnu_interp_impl(nu, gamma + eps, kTe_keV, kTbb_keV)
    fd_grad = (shifted - primal_out) / eps

    # The tangent dtype must MATCH the primal's, exactly — a ``custom_jvp``
    # contract that ``custom_vjp`` did not impose, so it is the one way this
    # conversion can regress. ``nu`` sets the primal dtype while ``gamma`` sets
    # the tangent's: a float32 SED grid with a float64 ``gamma`` promotes the
    # product to float64 and JAX rejects the rule outright::
    #
    #     TypeError: Custom JVP rule must produce primal and tangent outputs
    #     with corresponding shapes and dtypes. Expected float32[5994]
    #     (tangent type of float32[5994]) but got float64[5994].
    #
    # That is a hard error at trace time, not a wrong number, and it took out
    # the B1_agn_disc_torus scenario — a mixed-dtype path that no unit test
    # reaches, only the slow integration tier.
    return primal_out, jnp.asarray(fd_grad * d_gamma, dtype=primal_out.dtype)
