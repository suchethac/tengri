# SPDX-License-Identifier: BSD-3-Clause
"""nthcomp warm Comptonization precomputed template support.

Provides JAX-compatible log-space trilinear interpolation over a precomputed
table of Kompaneets equation solutions.  The table is built by
``scripts/build_nthcomp_templates.py``, which calls RELAGN's ``pyNTHCOMP``
(scotthgn/RELAGN, credit A.D. Thomas, ported from XSpec donthcomp.f) as an
external dependency: tengri does **not** ship the Kompaneets solver itself.

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
Kubota & Done (2018) MNRAS 480 1247 Section 2.2: warm Comptonization zone.
Zdziarski, Johnson & Magdziarz (1996) MNRAS 283 193: Kompaneets solver.
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

# ── Template loading (lazy: no computation at import time) ───────

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
        ``None`` when the templates are absent: callers then fall back to
        their analytic path, as they did before threading existed.

    Notes
    -----
    **JIT-compatible**: no, deliberately; call it before tracing.
    """
    gamma, kte, ktbb, nu, table_log, available = _get_nthcomp_templates()
    if not available:
        return None
    return NthcompTable(gamma=gamma, kte=kte, ktbb=ktbb, nu=nu, table_log=table_log)


def _axis_slope(val: jnp.ndarray, grid: jnp.ndarray, i_lo: jnp.ndarray) -> jnp.ndarray:
    """d(frac)/d(val) of :func:`_clamp_interp_index`: ``1/span`` in-cell, 0 where clamped."""
    span = grid[jnp.clip(i_lo + 1, 1, grid.shape[0] - 1)] - grid[i_lo]
    span_safe = jnp.where(span > 0, span, 1.0)
    raw = (val - grid[i_lo]) / span_safe
    return jnp.where((span > 0) & (raw >= 0.0) & (raw <= 1.0), 1.0 / span_safe, 0.0)


def _interp_with_slopes(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
    table: NthcompTable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Trilinear-in-log template shape and its exact slopes in all three operands.

    Returns ``(shape, slopes)`` on the requested ``nu`` grid, preserving input
    precision; ``slopes`` has shape ``(3, n_nu)`` and holds
    ``d shape / d(gamma, kTe_keV, kTbb_keV)``. ``shape`` is the forward value,
    unchanged from the previous inline implementation.

    The interpolant is piecewise linear in each operand, so its derivative is
    exact and local: inside a template cell it is the cell slope, and it is 0
    where the operand is clamped to the grid edge (the clamp makes the shape
    independent of it). Nothing here needs a finite-difference step.
    """
    g = jnp.asarray(gamma)
    t = jnp.asarray(kTe_keV)
    b = jnp.asarray(kTbb_keV)

    gamma_jax = jnp.asarray(table.gamma).astype(g.dtype)
    kte_jax = jnp.asarray(table.kte).astype(t.dtype)
    ktbb_jax = jnp.asarray(table.ktbb).astype(b.dtype)
    ig, fg = _clamp_interp_index(g, gamma_jax)
    it, ft = _clamp_interp_index(t, kte_jax)
    ib, fb = _clamp_interp_index(b, ktbb_jax)

    table_jax = jnp.asarray(table.table_log)

    def _c(dg: int, dt: int, db: int) -> jnp.ndarray:
        """Return table value at the interpolation-cell corner offset (dg, dt, db)."""
        return table_jax[ig + dg, it + dt, ib + db]

    # Trilinear interpolation over 8 corners (gamma x kTe x kTbb) in log space.
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

    # d(log shape)/d(frac) along each axis of the multilinear form, then the
    # chain rule through frac(val) = (val - node) / span.
    dl_dfg = (1 - fb) * ((1 - ft) * (_c(1, 0, 0) - _c(0, 0, 0)) + ft * (_c(1, 1, 0) - _c(0, 1, 0)))
    dl_dfg = dl_dfg + fb * (
        (1 - ft) * (_c(1, 0, 1) - _c(0, 0, 1)) + ft * (_c(1, 1, 1) - _c(0, 1, 1))
    )
    dl_dft = (1 - fb) * (s10 - s00) + fb * (s11 - s01)
    dl_dfb = s1 - s0
    dlog = jnp.stack(
        [
            dl_dfg * _axis_slope(g, gamma_jax, ig),
            dl_dft * _axis_slope(t, kte_jax, it),
            dl_dfb * _axis_slope(b, ktbb_jax, ib),
        ]
    )
    dshape_on_table_grid = shape_on_table_grid * dlog

    # Resample onto the requested nu grid (linear, so the slopes resample too).
    nu_f = jnp.asarray(nu)
    nu_jax_interp = jnp.asarray(table.nu).astype(nu.dtype)
    lnu = jnp.interp(nu_f, nu_jax_interp, shape_on_table_grid, left=0.0, right=0.0)
    dlnu = jax.vmap(lambda d: jnp.interp(nu_f, nu_jax_interp, d, left=0.0, right=0.0))(
        dshape_on_table_grid
    )
    return lnu, dlnu


def _nthcomp_lnu_interp_impl(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
    table: NthcompTable | None = None,
) -> jnp.ndarray:
    """Implementation of nthcomp interpolation (used by both forward and VJP).

    ``table`` carries the template arrays. Passing them in: rather than
    reading the module-level cache here: is what lets the forward model
    thread the ~15 MB library through ``jax.jit`` as a ``Parameter`` instead
    of freezing it into the graph as ``Constant`` ops (#1383).
    """
    if table is None:
        if not _is_table_available():
            raise RuntimeError(
                "nthcomp templates not loaded. Run scripts/build_nthcomp_templates.py first."
            )
        table = load_nthcomp_table()

    lnu, _ = _interp_with_slopes(nu, gamma, kTe_keV, kTbb_keV, table)

    # Return in the CALLER's precision, not the table's (#1822).
    #
    # The table is float32 and the interpolation is done there, which is right;
    # promoting a float32 library to float64 buys no accuracy. But *returning*
    # float32 forced the caller's precision too, and that is what broke reverse
    # mode: ``custom_jvp`` takes the cotangent in the primal's dtype, and this
    # kernel's output gets multiplied by a ring luminosity in ``disc.py``, so the
    # cotangent handed back is ~1e66: fine in float64, **inf in float32**, whose
    # ceiling is 3.4e38. ``inf * fd_grad`` is NaN, so ``jax.grad`` w.r.t.
    # ``agn_gamma_warm`` returned NaN on every realistic disc while ``jax.jvp``
    # returned 5.2e30. Every gradient backend (MAP, NUTS, VI) is reverse-mode.
    #
    # The rule's docstring argued the old ``custom_vjp``'s overflow rescaling was
    # unnecessary because "forward mode never forms the cotangent product". True,
    # and beside the point: ``jax.grad`` transposes the jvp and forms exactly
    # that product. Widening the output is the fix that needs no rescaling;
    # float32 -> float64 is exact, so no forward value moves.
    #
    # A caller working entirely in float32 (the #1206 path) still gets float32
    # out, and is still exposed to the same ceiling; that is inherent to float32
    # and is why the disc's float32 branch folds the tiny shape in first.
    out_dtype = jnp.result_type(nu, gamma, kTe_keV, kTbb_keV)
    return jnp.maximum(lnu, 0.0).astype(out_dtype)


@jax.custom_jvp
def _nthcomp_interp(
    table: NthcompTable,
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
) -> jnp.ndarray:
    """Return the normalized nthcomp L_nu shape via trilinear interpolation.

    The custom-JVP kernel. ``table`` is primal argument 0 so the library can
    be threaded through ``jax.jit`` as an argument; :func:`nthcomp_lnu_interp`
    is the public entry point that resolves it. Extrapolation beyond grid
    bounds is clamped to boundary values.

    Parameters
    ----------
    table : NthcompTable
        Template arrays. Also read by the JVP rule, which re-evaluates the
        interpolation at shifted operands.
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
    **JIT-compatible**: yes, uses ``jnp`` primitives and a JAX-registered JVP.

    The underlying templates are precomputed Kompaneets equation solutions
    in log-space (Kubota & Done 2018, Section 2.2). Trilinear interpolation
    is performed in log(spectral shape) to improve accuracy for exponentially
    varying features (Wien seed-BB tail), then exponentiated. Extrapolation
    beyond grid bounds is clamped to preserve monotonicity at boundaries.

    **Custom JVP**: differentiating the composed ``jnp.interp`` chain directly
    returns NaN, so :func:`_nthcomp_interp_jvp` supplies the exact
    piecewise-linear slopes instead (#2572). It is a ``custom_jvp`` rather than a
    ``custom_vjp`` because a ``custom_vjp`` is opaque to forward mode, which
    takes out geoVI.

    **Which operands carry a tangent is documented on that rule, and is
    deliberately not repeated here** -- the contract lives in one place so it
    cannot drift.

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


@_nthcomp_interp.defjvp
def _nthcomp_interp_jvp(primals: tuple, tangents: tuple) -> tuple:
    """Forward-mode rule: exact interpolant slopes in ``gamma``, ``kTe`` and ``kTbb``.

    Parameters
    ----------
    primals : tuple
        ``(table, nu, gamma, kTe_keV, kTbb_keV)`` -- see :func:`_nthcomp_interp`.
    tangents : tuple
        Tangents of those same five operands. ``gamma``, ``kTe_keV`` and
        ``kTbb_keV`` contribute; ``nu`` is a fixed grid and ``table`` is a
        library, never a fit parameter, so their tangents are structurally zero.

    Returns
    -------
    tuple
        ``(primal_out, tangent_out)``, both ``ndarray, shape (n_nu,)``
        [dimensionless -- a normalized spectral shape].

    Notes
    -----
    **JIT-compatible**: yes.

    **Exact slopes, not finite differences (#2572).** The forward value is a
    trilinear interpolation in log space followed by a linear resample onto
    ``nu``, so it is piecewise linear in each operand and its derivative is
    known in closed form (:func:`_interp_with_slopes`). All three operand tangents
    are carried:

    * ``kTbb_keV = k_B * t_ring`` from ``disc.py``'s warm zone, so it moves with
      ``agn_log_mbh`` and ``agn_log_lbol`` through every ring; in the UV and soft
      X-ray, where the Comptonized shape carries the flux, omitting its tangent
      would misstate the log-derivative of the disc SED by up to 0.4.
    * ``gamma`` and ``kTe_keV`` slopes are taken on the template cell the operand
      sits in; a finite-difference step spans template nodes, where the slope is
      discontinuous, and returns a cell-averaged slope instead.

    Exact slopes cost one kernel evaluation instead of the three a finite-difference
    rule needs.

    **A ``custom_jvp``, not a ``custom_vjp`` (#1206).** A ``custom_vjp`` is
    *opaque to forward mode* -- ``jax.jvp`` raises ``TypeError`` -- which takes
    out geoVI, whose metric is built with forward mode. A ``custom_jvp`` serves
    forward mode directly and reverse mode by transposition (#1822: the kernel
    returns the caller's precision, so the transposed cotangent product stays
    finite in float64).
    """
    table, nu, gamma, kTe_keV, kTbb_keV = primals
    _, _, d_gamma, d_kTe, d_kTbb = tangents

    _, slopes = _interp_with_slopes(nu, gamma, kTe_keV, kTbb_keV, table)
    primal_out = _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV, table)

    # The tangent dtype must MATCH the primal's, exactly -- a ``custom_jvp``
    # contract. ``nu`` sets the primal dtype while the operand tangents set the
    # product's: a float32 SED grid with a float64 ``gamma`` would otherwise
    # promote to float64 and JAX rejects the rule at trace time.
    #
    # The slopes and tangents are widened to that dtype BEFORE they multiply, so
    # the transposed (reverse-mode) cotangent product happens in the caller's
    # precision: ``disc.py`` hands back a ~1e66 cotangent that is fine in float64
    # and ``inf`` in float32 (#1822). Casting only the finished sum would put the
    # float32 ceiling back into the transpose.
    dt = primal_out.dtype
    slopes = slopes.astype(dt)
    tangent_out = (
        slopes[0] * jnp.asarray(d_gamma, dtype=dt)
        + slopes[1] * jnp.asarray(d_kTe, dtype=dt)
        + slopes[2] * jnp.asarray(d_kTbb, dtype=dt)
    )
    return primal_out, tangent_out


def nthcomp_lnu_interp(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
    _template: NthcompTable | None = None,
) -> jnp.ndarray:
    """Normalized nthcomp :math:`L_\\nu` shape via trilinear interpolation.

    Thin dispatcher over the custom-JVP kernel. See :func:`_nthcomp_interp`
    for the physics and :func:`_nthcomp_interp_jvp` for the gradient treatment.

    Parameters
    ----------
    nu : ndarray, shape (n_nu,)
        Frequency grid [Hz].
    gamma, kTe_keV, kTbb_keV : Array
        Photon index, electron temperature [keV], seed temperature [keV].
        Each is clamped to the grid range.
    _template : NthcompTable, optional
        Pre-loaded templates, threaded in as a JIT argument by the forward
        model. ``None`` (default) reads the module-level cache, which: under
        trace: bakes ~15 MB into the graph as constants.

    Returns
    -------
    ndarray, shape (n_nu,)
        Non-negative spectral shape.

    Notes
    -----
    **JIT-compatible**: yes. Derivatives come from the finite-difference rule
    registered on :func:`_nthcomp_interp`; which operands carry a tangent is
    documented there and not restated here (#1822).
    """
    table = _template if _template is not None else load_nthcomp_table()
    if table is None:
        raise RuntimeError(
            "nthcomp templates not loaded. Run scripts/build_nthcomp_templates.py first."
        )
    return _nthcomp_interp(table, nu, gamma, kTe_keV, kTbb_keV)
