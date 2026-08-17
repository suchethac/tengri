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

    # Return in the CALLER's precision, not the table's (#1822).
    #
    # The table is float32 and the interpolation is done there, which is right —
    # promoting a float32 library to float64 buys no accuracy. But *returning*
    # float32 forced the caller's precision too, and that is what broke reverse
    # mode: ``custom_jvp`` takes the cotangent in the primal's dtype, and this
    # kernel's output gets multiplied by a ring luminosity in ``disc.py``, so the
    # cotangent handed back is ~1e66 — fine in float64, **inf in float32**, whose
    # ceiling is 3.4e38. ``inf * fd_grad`` is NaN, so ``jax.grad`` w.r.t.
    # ``agn_gamma_warm`` returned NaN on every realistic disc while ``jax.jvp``
    # returned 5.2e30. Every gradient backend (MAP, NUTS, VI) is reverse-mode.
    #
    # The rule's docstring argued the old ``custom_vjp``'s overflow rescaling was
    # unnecessary because "forward mode never forms the cotangent product". True,
    # and beside the point: ``jax.grad`` transposes the jvp and forms exactly
    # that product. Widening the output is the fix that needs no rescaling —
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
    **JIT-compatible**: yes — uses ``jnp`` primitives and a JAX-registered JVP.

    The underlying templates are precomputed Kompaneets equation solutions
    in log-space (Kubota & Done 2018, Section 2.2). Trilinear interpolation
    is performed in log(spectral shape) to improve accuracy for exponentially
    varying features (Wien seed-BB tail), then exponentiated. Extrapolation
    beyond grid bounds is clamped to preserve monotonicity at boundaries.

    **Custom JVP**: differentiating the composed ``jnp.interp`` chain directly
    returns NaN, so :func:`_nthcomp_interp_jvp` supplies finite-difference
    tangents instead. It is a ``custom_jvp`` rather than the ``custom_vjp`` this
    used to be (#1206) because a ``custom_vjp`` is opaque to forward mode, which
    takes out geoVI.

    **Which operands carry a tangent is documented on that rule, and is
    deliberately not repeated here.** This paragraph used to keep its own copy,
    and the copy went stale the moment the rule changed: after #1822 gave
    ``kTe`` a tangent, this text still read "``nu``, ``kTe_keV`` and ``kTbb_keV``
    are held fixed during fitting and carry exactly zero derivative" — the
    precise false belief #1822 existed to correct, restated one screen above the
    correction. Two copies of a contract do not stay in sync; one does.

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
    """Forward-mode rule: finite-difference derivatives in ``gamma`` and ``kTe``.

    Parameters
    ----------
    primals : tuple
        ``(table, nu, gamma, kTe_keV, kTbb_keV)`` -- see :func:`_nthcomp_interp`.
    tangents : tuple
        Tangents of those same five operands. ``gamma`` and ``kTe_keV``
        contribute; ``nu`` and ``kTbb_keV`` do not, and ``table`` is a library,
        never a fit parameter, so its tangent is structurally zero -- the
        forward-mode counterpart of the zero cotangent the reverse rule used to
        return for it.

        **Why ``kTbb`` is still dropped, and why that is not the same omission
        as ``kTe`` was (#1822).** It reaches this kernel only as
        ``kTbb_keV = k_B * t_ring`` from ``disc.py``'s warm zone, and ``t_ring``
        also drives ``_planck_lnu`` on the same ring -- a path that *is*
        differentiated and that dominates. Measured through
        ``kubota_done_disc``: ``d/d(agn_log_mbh)`` agrees with a central
        difference to **0.00%** at log M_BH = 7.5, 8.0 and 8.5 with the tangent
        dropped. So the missing term is not detectable in the observable it
        feeds, and supplying it would cost a third kernel evaluation per JVP for
        no measured accuracy. ``kTe`` was the opposite case: -100%, because
        ``agn_kt_warm`` reaches the SED through this kernel and nothing else.
        Re-measure before assuming either still holds.

    Returns
    -------
    tuple
        ``(primal_out, tangent_out)``, both ``ndarray, shape (n_nu,)``
        [dimensionless -- a normalized spectral shape].

    Notes
    -----
    **JIT-compatible**: yes.

    **A ``custom_jvp``, not a ``custom_vjp`` (#1206).** A ``custom_vjp`` is
    *opaque to forward mode* -- ``jax.jvp`` raises ``TypeError: can't apply
    forward-mode autodiff (jvp) to a custom_vjp function`` -- which takes out
    geoVI, whose metric is built with forward mode, for every AGN model reaching
    this kernel. A ``custom_jvp`` serves forward mode directly and reverse mode
    by transposition; the transpose of ``fd_grad * d_gamma`` is
    ``sum(g * fd_grad)``, exactly the reverse pass it replaces.

    The overflow-safe rescaling the reverse rule performed on ``g_out`` is not
    needed here: forward mode never forms the cotangent product, so there is no
    ``sum(g_out * fd_grad)`` to overflow.
    """
    table, nu, gamma, kTe_keV, kTbb_keV = primals
    _, _, d_gamma, d_kTe, _ = tangents

    primal_out = _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV, kTbb_keV, table)

    # Adaptive one-sided step: relative for large gamma, absolute near zero.
    #
    # 1e-3, not the 1e-6 carried by the ``custom_vjp`` spelling. The impl is a
    # composed ``jnp.interp`` chain, so the finite difference is a subtraction of two
    # nearly equal ~1e-16 values: at 1e-6 the surviving digits are cancellation
    # remainder, not slope. Measured against a converged central difference at three
    # off-node gammas (2.37/2.53/2.64 -- 2.5 is a grid node where the derivative is
    # genuinely undefined and any FD comparison is meaningless)::
    #
    #     h        2.37      2.53      2.64
    #     1e-7     -100%     -100%     -100%     <- differences to exactly 0.0
    #     ~2.5e-6   -21%      +47%      +5.9%    <- the old step
    #     1e-4      +0.6%     +0.0%     -1.3%
    #     1e-3      -0.1%     +0.0%     -0.2%    <- plateau
    #     1e-2      +0.6%     +0.3%     +0.3%
    #
    # The old step was not uniformly biased -- it was wrong by -10% to +54% depending
    # on where in the grid gamma sat, which is why a single-step check never caught
    # it. The plateau is two decades wide; 1e-3 sits in its middle.
    eps = jnp.maximum(1e-3 * jnp.abs(gamma), 1e-3)
    shifted = _nthcomp_lnu_interp_impl(nu, gamma + eps, kTe_keV, kTbb_keV, table)
    fd_grad = (shifted - primal_out) / eps

    # The kTe tangent, by the same one-sided rule (#1822). Discarding it made
    # ``agn_kt_warm`` — declared ``Uniform(0.1, 0.5)`` and freeable — a parameter
    # no gradient backend could move: measured exactly 0.0 against a central
    # difference of 7.0e41 through ``kubota_done_disc``, i.e. -100%. The forward
    # sensitivity is large (18.1x in sum(L_nu) across that prior), so the
    # posterior came back as the prior and nothing downstream could tell that
    # apart from an honestly-unconstrained fit.
    #
    # Step chosen the same way as gamma's, against a converged central difference
    # at three off-node kTe (grid nodes are kinks where the derivative is
    # genuinely undefined, so an FD comparison there is meaningless)::
    #
    #     h        0.1304    0.1625    0.1946
    #     1e-7     +169%      -95%     +249%    <- cancellation, not slope
    #     1e-6      +8.2%     -8.9%     +3.4%
    #     1e-5      +1.2%     +0.0%     +0.1%
    #     1e-4      -0.0%     -1.5%     -0.9%   <- plateau, chosen
    #     1e-3      +1.0%     -0.9%     -0.4%
    #     1e-2     +13.1%     +5.9%     +4.3%   <- 1/3 of a cell; crosses nodes
    #
    # The kTe axis is spaced 0.0321 apart, an order of magnitude finer than
    # gamma's 0.105, which is why the usable window sits a decade lower and
    # gamma's 1e-3 floor would be a poor default here.
    eps_t = jnp.maximum(1e-3 * jnp.abs(kTe_keV), 1e-4)
    shifted_t = _nthcomp_lnu_interp_impl(nu, gamma, kTe_keV + eps_t, kTbb_keV, table)
    fd_grad_t = (shifted_t - primal_out) / eps_t

    # The tangent dtype must MATCH the primal's, exactly -- a ``custom_jvp``
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
    # the B1_agn_disc_torus scenario -- a mixed-dtype path that no unit test
    # reaches, only the slow integration tier.
    return primal_out, jnp.asarray(fd_grad * d_gamma + fd_grad_t * d_kTe, dtype=primal_out.dtype)


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
        model. ``None`` (default) reads the module-level cache, which — under
        trace — bakes ~15 MB into the graph as constants.

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
