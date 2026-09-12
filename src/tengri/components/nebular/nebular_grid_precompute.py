# SPDX-License-Identifier: BSD-3-Clause
r"""Adaptive-axis nebular line grid: variable ionization via precompute (#950).

Generalizes the fixed-ionization line-per-Q_H table (:mod:`line_precompute`,
#955) to a grid over the ionization parameters that are **free** in a given
model. Cue makes each nebular line luminosity exactly linear in the hydrogen-
ionizing photon rate Q_H, and the per-Q_H factor depends only on the stellar
metallicity (which sets the ionizing-spectrum shape) and the gas conditions
(logU, gas-phase metallicity): **not** on the star-formation-history shape
(#950, CV = 0 % across SFH draws). So

.. math::

    F_{\rm line}(\theta) = \frac{n_{\rm ion}(\theta)\,
        \ell\bigl(Z_\star, \log U, \log Z_{\rm gas}\bigr)}{4\pi\,d_L(z)^2}

where :math:`\ell` is a stored **luminosity per Q_H** (distance-independent, so
the cosmology is applied at the evaluation redshift) interpolated over whichever
of ``met_logzsol`` / ``neb_logU`` / ``neb_logZ_gas`` are free. Fixed parameters
are baked into the grid at their spec value, so a tighter setup gets a smaller,
faster grid automatically:

* all three fixed  → 0 axes (one template; pure Q_H scaling: the #955 case);
* ``neb_logU`` free → 1 axis;
* ``neb_logU`` + ``neb_logZ_gas`` free → 2 axes; …

Interpolation is **node-exact** monotone-cubic PCHIP
(:func:`tengri.utils.grid_interp.interp_nd_pchip`, the SKIRTOR pattern) in
log10-luminosity space, so the reconstruction is exact at grid nodes,
JIT/gradient-safe, and free of the smoothing bias a kernel smoother introduces on
the steeply logU-varying lines. The ionizing-spectrum shape is
**not** a grid axis: it is carried by the ``met_logzsol`` axis (SFH-independent
to ~0.2 %; #1018). See issue #950.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._params import PARAMS as _NEB_PARAM_DECLARATIONS
from tengri.components.nebular.line_precompute import _log10_four_pi_dl2
from tengri.components.stellar.reference_history import reference_history_params
from tengri.parameters.translate import LOG10_ZSUN
from tengri.protocols.component import declared_default
from tengri.utils.grid_interp import interp_nd_pchip
from tengri.utils.scale import apply_log10_scale, pow10

#: Parameters that may become grid axes when free. ``met_logzsol`` sets the
#: ionizing-spectrum shape; ``neb_logU`` / ``neb_logZ_gas`` are the gas
#: conditions. ``neb_fesc`` stays fixed (it rescales the escaping continuum, not
#: a smooth interpolation axis) and the ionizing-spectrum params are SSP-derived.
#: ``neb_logU`` also joins the axes whenever DIG mixing could be active, even
#: when it is itself Fixed (#2222): see ``_dig_may_be_active``.
_CANDIDATE_AXES = ("met_logzsol", "neb_logU", "neb_logZ_gas")

#: Fallback grid bounds used ONLY when a free axis's prior exposes no finite
#: support (e.g. an unbounded Gaussian). Kept at least as wide as the standard
#: priors (met_logzsol ~ Uniform(-2, 0.2/0.5)) so the fallback still spans the
#: sampled region: the primary path reads the prior's actual bounds.
_DEFAULT_RANGE = {
    "met_logzsol": (-2.0, 0.5),
    "neb_logU": (-4.0, -1.0),
    "neb_logZ_gas": (-1.0, 0.5),
    # Only reached if neb_dig_delta_logU is free with a prior exposing no
    # finite bounds; mirrors its registered Uniform(-4, 0) support (#2222).
    "neb_dig_delta_logU": (-4.0, 0.0),
}

#: ``neb_logU``'s declared registry default (#2222): the query-point value
#: used for the HII lookup when ``neb_logU`` joins ``axis_names`` purely
#: because DIG mixing could be active and the caller's params dict does not
#: carry an explicit value for it (a Fixed parameter is not guaranteed
#: present in every params dict a caller builds by hand). Read from the one
#: declaration (``NebularSEDComponent``'s ``PARAMS``) rather than repeated as
#: a literal, so this and ``component.py``'s own ``common_kwargs`` default
#: cannot drift apart.
NEB_LOGU_DEFAULT: float = declared_default(_NEB_PARAM_DECLARATIONS, "neb_logU")

#: ``neb_dig_delta_logU``'s declared registry default, for the same reason
#: (used only internally, by :func:`_dig_delta_logU_support`).
_NEB_DIG_DELTA_LOGU_DEFAULT: float = declared_default(
    _NEB_PARAM_DECLARATIONS, "neb_dig_delta_logU"
)

#: Extra resolution on the ``met_logzsol`` axis relative to the smooth gas axes.
#:
#: Applied **only** when the met axis cannot be snapped to the SSP metallicity
#: nodes (``snap_met_to_ssp_nodes=False``, or a model exposing no SSP metallicity
#: grid). A snapped axis puts knots on the kinks and interpolates linearly across
#: them, which converges: blind densification of an unsnapped axis does not, so
#: it needs the extra points more (#1020).
_MET_AXIS_DENSITY_FACTOR = 2

#: A uniform grid point within this fraction of a cell width of an SSP metallicity
#: node is dropped in favor of the node (see :func:`_snap_axis_to_nodes`), so
#: snapping never creates a near-degenerate interpolation cell.
_SNAP_MERGE_FRAC = 0.25

#: Points used for an axis the caller did not resolve explicitly: both the
#: scalar default and the per-axis fallback for a dict that omits an axis.
_DEFAULT_N_GRID = 16

#: Fewest points an interpolation axis can carry. One knot cannot interpolate;
#: an axis that should not vary belongs fixed in the spec, not shrunk to a point.
_MIN_N_GRID = 2


def validate_n_grid(n_grid):
    """Validate a scalar or per-axis ``n_grid`` before any grid is built.

    Parameters
    ----------
    n_grid : int or dict
        Points per free ionization axis. A scalar applies to every axis; a dict
        ``{axis_name: n}`` sets axes individually and falls back to
        :data:`_DEFAULT_N_GRID` for any it omits.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``n_grid``, or any dict value, is not an integer.
    ValueError
        If a dict key names something that is not a griddable axis, or any
        resolution is below :data:`_MIN_N_GRID`.

    Notes
    -----
    Runs both at :class:`~tengri.forward.sed_model.FeaturePrecomp` construction
    and again inside :func:`precompute_nebular_grid`, so a misspelled axis raises
    where it was written. Before #1311 an unrecognized key was silently dropped
    by the ``dict.get(name, default)`` lookup and the axis quietly took the
    default resolution: the user got a grid they did not ask for, with no
    warning.
    """

    def _check_one(value, where):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"n_grid{where} must be an integer; got {type(value).__name__} ({value!r})."
            )
        if int(value) < _MIN_N_GRID:
            raise ValueError(
                f"n_grid{where} must be >= {_MIN_N_GRID} (an interpolation axis needs at "
                f"least two knots); got {value}. To drop an axis entirely, fix its "
                f"parameter in the spec rather than shrinking its grid."
            )

    if isinstance(n_grid, dict):
        unknown = sorted(k for k in n_grid if k not in _CANDIDATE_AXES)
        if unknown:
            raise ValueError(
                f"n_grid names {unknown!r}, which are not griddable ionization axes. "
                f"Valid axes are {', '.join(_CANDIDATE_AXES)}. Axes omitted from the "
                f"dict default to {_DEFAULT_N_GRID}."
            )
        for name, value in n_grid.items():
            _check_one(value, f"[{name!r}]")
    else:
        _check_one(n_grid, "")


#: Grid nodes evaluated per vmapped batch at build time.
#:
#: The build vmaps one Cue forward per node. vmap is *batched*, not streamed, so a
#: single call over every node holds every node's intermediates live at once, and
#: peak memory scales with the node count: a three-axis grid at ``n_grid=8``
#: (16 x 8 x 8 = 1024 nodes, ~11 MB of intermediates each) peaked at **11.7 GB**,
#: enough to OOM a 16 GB CI runner or an ordinary laptop (#1361).
#:
#: Chunking bounds that peak at ``chunk x per-node`` while evaluating exactly the
#: same nodes: vmap applies no cross-node reduction, so the per-node result does
#: not depend on who else is in the batch. Grids at or below this size take the
#: single-call path unchanged, so the common one-axis grid is untouched.
_BUILD_CHUNK_NODES = 64


@dataclasses.dataclass(frozen=True)
class NebularGridTable:
    """Adaptive-axis grid of per-Q_H line luminosities for variable ionization.

    Attributes
    ----------
    axis_names : tuple of str
        Free parameters gridded, in interpolation order (subset of
        :data:`_CANDIDATE_AXES`). Empty when all ionization params are fixed.
        ``neb_logU`` is included even when Fixed whenever DIG mixing could be
        active (:func:`_dig_may_be_active`), since the DIG lookup always
        needs a second query point distinct from the HII one (#2222).
    axes : tuple of ndarray
        One grid-value array per axis, ascending.
    log_line_per_qh : ndarray, shape ``(*grid_dims, n_lines)``
        ``log10`` of the line **luminosity** per unit ``nion`` [erg/s per
        (photons/s)], distance-independent. Stored in log space because line
        luminosities span decades across (met, logU, logZ_gas): geometric
        (log-space) interpolation is far more accurate than arithmetic there.
        ``grid_dims`` matches ``axes``; a 0-axis table is shape ``(n_lines,)``.
    wavelengths : ndarray, shape (n_lines,)
        Rest-frame vacuum line wavelengths [Angstrom].
    log_phot_per_qh : ndarray, shape ``(*grid_dims, n_filter)`` or None
        ``log10`` of the **intrinsic** (un-reddened) nebular filter-integrated
        rest-frame ``L_nu`` per unit ``nion`` [erg/s/Hz per (photons/s)]: the
        broadband analog of ``log_line_per_qh``, one column per photometric
        filter. Reconstructs ``nebular_phot_lnu_precomp`` (the key
        :meth:`Observation.predict_via_precomp` consumes) without the per-eval
        Cue forward + filter integration. ``None`` when the reference model had
        no ``WavePrecomp`` filters to integrate against (line-only grid).
    axis_kinds : tuple of str
        Interpolation kind per axis (``'pchip'`` / ``'linear'``), matching
        ``axes``. The ``met_logzsol`` axis is ``'linear'`` when its knots are
        snapped to the SSP metallicity nodes: the exact emissivity has C0 kinks
        there, and a cubic's two-sided tangent straddles them (#1020). Empty
        tuple means PCHIP everywhere (the pre-#1020 default).
    log_restband_per_qh : ndarray, shape ``(*grid_dims, n_filter)`` or None
        The **rest-frame** twin of ``log_phot_per_qh``: the same filters
        integrated at ``redshift=0`` instead of at the model redshift, so the
        band sits in the rest frame and samples the rest SED at its own pivot
        (#1148). Reconstructs ``nebular_restband_lnu_precomp``.

        This channel exists because the two publishes are **twins**: the exact
        path emits them together, so a grid carrying only the observed band
        makes every rest-frame consumer silently lose the nebular contribution.
        That was #1665: all 13 spectral indices moved, worst ``HgA`` by +1733%,
        with no exception raised. ``None`` only on tables built before #1665;
        :func:`precompute_nebular_grid` now always populates it alongside
        ``log_phot_per_qh``, and a table missing it disables the fast path
        rather than serving a half-answer.
    """

    axis_names: tuple
    axes: tuple
    log_line_per_qh: jnp.ndarray
    wavelengths: jnp.ndarray
    log_phot_per_qh: jnp.ndarray | None = None
    axis_kinds: tuple = ()
    log_restband_per_qh: jnp.ndarray | None = None


def _ssp_met_nodes(model):
    """The SSP metallicity grid nodes, in user-facing ``met_logzsol`` units.

    Returns ``None`` when the model exposes no SSP metallicity axis.
    """
    ssp = getattr(model, "ssp_data", None)
    lgmet = getattr(ssp, "ssp_lgmet", None)
    if lgmet is None:
        return None
    return np.asarray(lgmet, dtype=float) - LOG10_ZSUN  # absolute log10(Z) -> log10(Z/Zsun)


def _snap_axis_to_nodes(lo, hi, n, nodes):
    """Ascending grid on ``[lo, hi]``: the uniform axis **plus** every interior node.

    ``met_logzsol`` reaches the forward through a linear interpolation over the SSP
    metallicity axis, whose derivative jumps at each SSP node. The exact per-Q_H
    emissivity is therefore piecewise-smooth with C0 kinks at *known* locations. A
    PCHIP interpolant is node-exact, so a knot placed on a kink reproduces it exactly
    instead of smearing it across a cell -- an O(h) error becomes O(h^2).

    The nodes are **added to** a uniform axis of ``n`` points rather than replacing
    it. Two measured facts force that choice (dense 401-point sweep, FSPS/MILES):

    * Distributing a fixed budget across the node intervals -- by width, or by
      greedily bisecting the widest cell -- leaves the near-solar cells unrefined and
      the worst-case [OIII] error pinned at ~1.9 % regardless of ``n``. The residual
      is **curvature-driven, not width-driven**: it sits in a cell *narrower* than the
      widest one, because the SSP spectra vary fastest near solar.
    * Adding the nodes on top of the uniform axis keeps the density everywhere at
      least that of the uniform grid, so the snapped axis cannot be worse -- it is
      strictly the uniform grid with the kinks resolved.

    Uniform points landing within ``_SNAP_MERGE_FRAC`` of a cell width of a node are
    dropped in favor of the node, so the axis never develops a degenerate cell.

    Parameters
    ----------
    lo, hi : float
        Axis bounds [log10(Z/Zsun)].
    n : int
        Uniform-axis point count. The returned axis has ``n`` to ``n + n_interior``
        points depending on how many uniform points merge into nodes.
    nodes : array_like, shape (n_ssp_met,)
        SSP metallicity nodes [log10(Z/Zsun)]; those outside ``(lo, hi)`` are ignored.

    Returns
    -------
    ndarray, shape (>= n,)
        Strictly ascending grid including ``lo``, ``hi``, and every interior node.
    """
    n = int(n)
    uniform = np.linspace(lo, hi, n)
    nodes = np.unique(np.asarray(nodes, dtype=float))
    tol = 1e-9 * max(hi - lo, 1.0)
    interior = nodes[(nodes > lo + tol) & (nodes < hi - tol)]
    if interior.size == 0:
        return uniform

    # drop uniform points that a node effectively replaces (no degenerate cells)
    merge_within = _SNAP_MERGE_FRAC * (hi - lo) / max(n - 1, 1)
    keep = np.min(np.abs(uniform[:, None] - interior[None, :]), axis=1) > merge_within
    keep[0] = keep[-1] = True  # lo/hi are not nodes' to consume
    return np.sort(np.concatenate([uniform[keep], interior]))


def _finite_prior_bounds(dist):
    """(lo, hi) finite bounds of ``dist``, or ``None`` if it exposes none.

    Reads the bounded prior's support (``bounds`` tuple, else ``lo``/``hi``: the
    attributes tengri's :class:`Uniform` / :class:`LogUniform` expose). Shared
    by :func:`_axis_range` (candidate grid axes) and
    :func:`_dig_delta_logU_support` (``neb_dig_delta_logU``, not itself an
    axis, but read the same way to size the DIG-extended ``neb_logU`` range,
    #2222), so the bound-extraction rule cannot drift between the two.
    """
    candidates = []
    b = getattr(dist, "bounds", None)
    if b is not None:
        with contextlib.suppress(TypeError, ValueError, IndexError):
            candidates.append((float(b[0]), float(b[1])))
    lo, hi = getattr(dist, "lo", None), getattr(dist, "hi", None)
    if lo is not None and hi is not None:
        with contextlib.suppress(TypeError, ValueError):
            candidates.append((float(lo), float(hi)))
    for clo, chi in candidates:
        if math.isfinite(clo) and math.isfinite(chi) and clo < chi:
            return clo, chi
    return None


def _axis_range(spec, name):
    """(lo, hi) grid bounds for a free axis: from its prior's finite support.

    Only when the prior has no finite support (e.g. an unbounded Gaussian used
    as an axis) does it fall back to :data:`_DEFAULT_RANGE`, and it warns
    rather than silently ignoring the prior: a too-narrow grid would
    extrapolate and bias the fit.
    """
    dist = spec.get_distribution(name)
    bounds = _finite_prior_bounds(dist)
    if bounds is not None:
        return bounds
    warnings.warn(
        f"nebular grid: prior for {name!r} ({type(dist).__name__}) exposes no "
        f"finite [lo, hi] support; falling back to default range "
        f"{_DEFAULT_RANGE[name]}. If the prior is wider, pass ranges={{'{name}': "
        f"(lo, hi)}} or the grid will extrapolate. ",
        stacklevel=3,
    )
    return _DEFAULT_RANGE[name]


def _nion_of_state(state) -> jnp.ndarray:
    nion = state.derived["nion"]
    return jnp.sum(nion) if jnp.ndim(nion) else nion


def _log_nion_of_state(state) -> jnp.ndarray:
    """log10 Q_H [dex re photons/s], never materializing the ~1e53 linear value.

    Q_H overflows float32 (max 3.4e38), so the stellar component publishes
    ``log_nion`` alongside ``nion`` for exactly this reason. Falls back to the
    log of the linear publish for a state that carries only the latter.
    """
    log_nion = state.derived.get("log_nion")
    if log_nion is None:
        return jnp.log10(_nion_of_state(state))
    log_nion = jnp.asarray(log_nion)
    if not jnp.ndim(log_nion):
        return log_nion
    # ``_nion_of_state`` sums a multi-component Q_H; the log-domain sum is
    # logsumexp, not ``log10(sum(10**x))``, whose intermediate is the overflow
    # this helper exists to avoid.
    from jax.scipy.special import logsumexp

    ln10 = jnp.log(jnp.asarray(10.0, dtype=log_nion.dtype))
    return logsumexp(log_nion * ln10) / ln10


def _refuse_tabulated_metallicity(model):
    """Refuse a tabulated metallicity, whose LUT axis cannot exist (#1718).

    The axes are ``tuple(p for p in _CANDIDATE_AXES if p in free)``: free
    *parameters*. ``met_mode='table'`` declares none, so ``met_logzsol`` is not
    merely fixed, it is absent, and the metallicity axis disappears from the grid
    with nothing raised. The table is then built at a single reference
    metallicity and reconstructed at that one value for every galaxy.

    Measured against the exact path on a tabulated SFH whose Z(t) runs -2.1 to
    +0.4: **OIII_5007 off by 17.5%**, NII_6584 by 5.3%, against 0.3% for the same
    model with a parametric metallicity. Metal-line ratios are what a nebular
    fit is *for*, so this is refused rather than warned about.

    This is the same reasoning as ``_REQUIRED_FIXED`` in ``line_precompute``,
    which refuses a *free* ionization parameter because the single-metallicity
    axis makes reconstruction wrong away from its reference. Here the axis is
    missing outright.

    A tabulated **SFH** is fine and deliberately still allowed: the table is
    per-Q_H and so SFH-independent to 0.3% end-to-end (see
    ``components/stellar/reference_history.py`` for the measurement).

    Raises
    ------
    ValueError
        If the model's ``metallicity_model`` is ``'table'``.
    """
    from tengri.components.stellar.reference_history import stellar_config_of

    cfg = stellar_config_of(model)
    if cfg is None or getattr(cfg, "metallicity_model", None) != "table":
        return
    raise ValueError(
        "FeaturePrecomp cannot serve a tabulated metallicity. Its grid axes are "
        "the model's free parameters, and met={'type': 'table'} declares "
        "none: so met_logzsol is absent, the metallicity axis silently drops, "
        "and the whole table would be built at one reference metallicity. "
        "Measured that way against the exact path, OIII_5007 came out 17.5% "
        "wrong and NII_6584 5.3%, which is precisely the line-ratio information "
        "a nebular fit exists to use. Either drop FeaturePrecomp and keep the "
        "exact line path (WavePrecomp alone is unaffected and still applies), or "
        "use a parametric metallicity: a tabulated SFH with a free met_logzsol "
        "is supported and agrees with exact to 0.3% (#1718)."
    )


def _dig_may_be_active(spec) -> bool:
    """True if DIG mixing may be active: ``neb_dig_frac`` free, or fixed non-zero.

    Governs two grid-building decisions (#2222): whether ``neb_logU`` joins
    ``axis_names`` even when it is itself Fixed (the DIG lookup always needs a
    second query point distinct from the HII one), and how far the
    ``neb_logU`` axis range must extend to cover it
    (:func:`_dig_extended_logU_range`).

    Before #2222 this same predicate (then named ``_refuse_active_dig_mixing``)
    raised ``DIGNotOnNebularGridError``; the grid now serves DIG mixing via two
    lookups (:func:`~tengri.components.nebular.dig.mix_dig_grid_reconstruction`)
    rather than refusing it.

    Parameters
    ----------
    spec : Parameters
        The model's parameter specification.

    Returns
    -------
    bool
        Whether ``neb_dig_frac`` can take a nonzero value for this spec.
    """
    if "neb_dig_frac" not in spec.all_params:
        return False
    if "neb_dig_frac" in spec.free_params:
        return True
    value = spec.fixed_value("neb_dig_frac")
    return value is not None and float(value) != 0.0


def _own_logU_support(spec):
    """(lo, hi) support of ``neb_logU`` itself: prior bounds if free, else its
    pinned scalar as a degenerate ``(value, value)`` point.

    The "own" support this model's ``neb_logU`` explores, before any
    DIG-driven extension (:func:`_dig_extended_logU_range`) widens it.
    """
    if "neb_logU" in spec.free_params:
        return _axis_range(spec, "neb_logU")
    value = spec.fixed_value("neb_logU")
    value = NEB_LOGU_DEFAULT if value is None else float(value)
    return value, value


def _dig_delta_logU_support(spec):
    """(lo, hi) support of ``neb_dig_delta_logU``: prior bounds if free, else
    its pinned scalar as a degenerate ``(value, value)`` point.
    """
    if "neb_dig_delta_logU" not in spec.all_params:
        return _NEB_DIG_DELTA_LOGU_DEFAULT, _NEB_DIG_DELTA_LOGU_DEFAULT
    if "neb_dig_delta_logU" not in spec.free_params:
        value = spec.fixed_value("neb_dig_delta_logU")
        value = _NEB_DIG_DELTA_LOGU_DEFAULT if value is None else float(value)
        return value, value
    dist = spec.get_distribution("neb_dig_delta_logU")
    bounds = _finite_prior_bounds(dist)
    if bounds is not None:
        return bounds
    warnings.warn(
        f"nebular grid: prior for 'neb_dig_delta_logU' ({type(dist).__name__}) "
        f"exposes no finite [lo, hi] support; falling back to its registered "
        f"range {_DEFAULT_RANGE['neb_dig_delta_logU']}. The neb_logU axis "
        f"extension for active DIG mixing (#2222) may be too narrow.",
        stacklevel=3,
    )
    return _DEFAULT_RANGE["neb_dig_delta_logU"]


def _dig_extended_logU_range(own_lo, own_hi, delta_bounds):
    r"""Extend a ``neb_logU`` range to cover the DIG-shifted query point (#2222).

    DIG mixing evaluates the grid a second time at
    :math:`\log U + \Delta\log U` (:func:`~tengri.components.nebular.dig.
    mix_dig_grid_reconstruction`). The returned range is the union of
    ``(own_lo, own_hi)`` with that range shifted by ``delta_bounds`` (the
    Minkowski sum of the two intervals), so a subsequent
    ``jnp.linspace(lo, hi, n)`` axis covers both the HII query point and every
    DIG query point the model's priors can produce -- never clipping, which
    :func:`~tengri.utils.grid_interp.interp_nd_pchip` does silently outside the
    axis and which an earlier probe (kept in the #2222 research notes) measured
    at 3/10 random seeds giving 5.1e-2 / 4.2e-1 worst-case relative error,
    versus 1.17e-3 / 1.41e-3 with the range extended (measured on this
    implementation; see :func:`precompute_nebular_grid`'s Notes).

    Parameters
    ----------
    own_lo, own_hi : float
        ``neb_logU``'s own support (:func:`_own_logU_support`). [log10(U)]
    delta_bounds : tuple of float
        ``(lo, hi)`` support of ``neb_dig_delta_logU``
        (:func:`_dig_delta_logU_support`). [dex]

    Returns
    -------
    tuple of float
        ``(lo, hi)`` extended range. [log10(U)]
    """
    delta_lo, delta_hi = delta_bounds
    lo = min(own_lo, own_lo + delta_lo)
    hi = max(own_hi, own_hi + delta_hi)
    return lo, hi


def _dig_logU_axis_extension(spec, user_range):
    r"""Resolve whether ``neb_logU`` needs a DIG-extended axis, and its range.

    Single entry point for both the axis-selection decision (does
    ``neb_logU`` join ``axis_names``?) and the range that axis is built over,
    so the two cannot disagree (#2222 review I2, I3).

    The "own" (HII) range is ``user_range`` when the caller supplied one for
    ``neb_logU`` via ``ranges=`` -- the ruling is that a user-supplied range
    is the HII support, and the DIG image is derived from it exactly like the
    prior-derived case, rather than bypassing the extension (review I2,
    measured 5.4e-2 worst-case relative error, above this module's own 3e-2
    ceiling, when an explicit ``ranges={'neb_logU': (lo, hi)}`` was taken
    verbatim) -- else :func:`_own_logU_support`.

    A **degenerate** extension (``ext_lo == ext_hi``, e.g. ``neb_logU`` Fixed
    with ``neb_dig_delta_logU`` Fixed at exactly 0.0: the HII and DIG query
    points then coincide) means DIG mixing is arithmetically the HII term at
    every possible query, so no axis is needed -- an all-identical axis is
    otherwise silently a divide-by-zero for
    :func:`~tengri.utils.grid_interp.interp_nd_pchip`'s PCHIP slopes, which
    returned NaN with no warning before this fix (review I3). Before #2222
    this combination was refused outright (``DIGNotOnNebularGridError``); it
    needs no refusal now because it needs no axis.

    Parameters
    ----------
    spec : Parameters
        The model's parameter specification.
    user_range : tuple of float or None
        Caller-supplied ``(lo, hi)`` for ``neb_logU`` (``ranges.get("neb_logU")``),
        or ``None`` to use the spec's own support.

    Returns
    -------
    needs_axis : bool
        Whether ``neb_logU`` should join ``axis_names``.
    own_lo, own_hi : float
        The "own" (HII) range used as the extension's base -- ``user_range``
        if given, else :func:`_own_logU_support`. [log10(U)]
    ext_lo, ext_hi : float
        The DIG-extended range. Equal to ``(own_lo, own_hi)`` when
        ``needs_axis`` is ``False``. [log10(U)]
    """
    if not _dig_may_be_active(spec):
        own_lo, own_hi = user_range if user_range is not None else _own_logU_support(spec)
        return False, own_lo, own_hi, own_lo, own_hi
    own_lo, own_hi = user_range if user_range is not None else _own_logU_support(spec)
    ext_lo, ext_hi = _dig_extended_logU_range(own_lo, own_hi, _dig_delta_logU_support(spec))
    return ext_lo != ext_hi, own_lo, own_hi, ext_lo, ext_hi


def _preserve_spacing_n(base_n, own_lo, own_hi, ext_lo, ext_hi):
    """Node count for an extended ``neb_logU`` axis that keeps the original spacing.

    When ``neb_logU`` was already going to be a free axis before DIG-awareness
    widened its range, resolving the wider range at the SAME node count
    (``base_n``) would thin out the nodes covering the original HII-only
    region, degrading ``neb_dig_frac = 0`` parity purely because DIG was
    armed elsewhere in the spec. Scaling the node count by the range ratio
    keeps the spacing, and hence that parity, unchanged (#2222).

    Parameters
    ----------
    base_n : int
        Node count :func:`precompute_nebular_grid` would have used for
        ``neb_logU`` absent the DIG extension.
    own_lo, own_hi : float
        ``neb_logU``'s own (un-extended) support. [log10(U)]
    ext_lo, ext_hi : float
        The DIG-extended range (:func:`_dig_extended_logU_range`). [log10(U)]

    Returns
    -------
    int
        Node count for the extended axis. Equal to ``base_n`` when
        ``own_lo == own_hi`` (``neb_logU`` is Fixed and joins the axis purely
        because DIG could be active: there is no pre-existing spacing to
        preserve), else scaled up to hold the un-extended spacing fixed.
    """
    if own_hi <= own_lo:
        return base_n
    dx = (own_hi - own_lo) / max(base_n - 1, 1)
    width = ext_hi - ext_lo
    return max(_MIN_N_GRID, math.ceil(width / dx) + 1)


def precompute_nebular_grid(
    model,
    wavelengths,
    *,
    n_grid: int = 16,
    ranges: dict | None = None,
    ref_params: dict | None = None,
    snap_met_to_ssp_nodes: bool = True,
) -> NebularGridTable:
    """Build the adaptive-axis per-Q_H line grid for ``model``.

    Auto-detects which of ``met_logzsol`` / ``neb_logU`` / ``neb_logZ_gas`` are
    free in ``model.spec`` and grids exactly those; fixed ones are baked at their
    spec value. The per-Q_H factor is SFH-shape-independent only to ~0.2 % (#1018:
    the ionizing-spectrum shape is the Q_H-weighted age mix, so re-weighting the SFH
    shifts the forbidden-line emissivity slightly), so the reference SFH is nearly
    but not exactly arbitrary: well inside the interpolation error.

    ``neb_logU`` is also gridded, and its range extended (never fixed at a
    single point), whenever DIG mixing could be active (``neb_dig_frac`` free,
    or fixed non-zero; see :func:`_dig_may_be_active`), because
    :func:`~tengri.components.nebular.dig.mix_dig_grid_reconstruction` needs a
    second query point at ``neb_logU + neb_dig_delta_logU`` (#2222).

    Parameters
    ----------
    model : SEDModel
        A model with a Q_H-linear nebular backend (Cue / CloudyGrid).
    wavelengths : array_like, shape (n_lines,)
        Rest-frame vacuum target line wavelengths [Angstrom].
    n_grid : int or dict, default 16
        Grid points per free axis. As a scalar it resolves the smooth gas axes
        (``neb_logU`` / ``neb_logZ_gas``) at ``n_grid``; the ``met_logzsol`` axis
        also starts at ``n_grid`` and then gains the interior SSP metallicity nodes
        (see ``snap_met_to_ssp_nodes``), so it ends up slightly larger. Pass a dict
        ``{axis_name: n}`` to set each axis explicitly; omitted axes take
        :data:`_DEFAULT_N_GRID`, and an explicit per-axis number is used verbatim
        (the unsnapped-met densification applies only to the default). Keys are
        validated against :data:`_CANDIDATE_AXES`, so a misspelled axis raises
        instead of silently selecting the default (#1311). Since build cost is the
        *product* over axes, per-axis resolution is the lever for a model whose
        axes differ in sensitivity. Validate any accuracy claim with a dense sweep
        strictly inside the grid range, never with random draws: a narrow feature
        hides from random draws, and an error that ignores ``n_grid`` is an
        unresolved kink, not interpolation error.
    ranges : dict, optional
        Override ``{param: (lo, hi)}`` grid bounds. Defaults to each free param's
        prior support (else :data:`_DEFAULT_RANGE`).
    ref_params : dict, optional
        Reference parameter dict (grid axes are overwritten per point). Defaults
        to a mid-range sample.
    snap_met_to_ssp_nodes : bool, optional
        Place knots on the SSP metallicity nodes and interpolate that axis
        linearly (default True). ``met_logzsol`` reaches the forward through a
        bilinear interpolation of the ionizing-spectrum tables, so the exact
        per-Q_H emissivity has C0 kinks exactly at ``ssp_data.ssp_lgmet``. Knots
        on the kinks + a C0 interpolant converge normally; a uniform axis, or a
        cubic whose tangent straddles a kink, does not (#1020). Set False to
        recover the pre-#1020 uniform + PCHIP axis.

    Returns
    -------
    NebularGridTable

    Notes
    -----
    **Build cost**: ``n_grid ** n_free_axes`` forward evaluations, once at
    construction. They are JIT'd and vmapped over the grid: one compile, not one
    eager forward per node: and evaluated in batches of
    :data:`_BUILD_CHUNK_NODES` so peak memory is bounded by the chunk rather than
    by the node count (#1361). (This note previously described a build-time loop
    over concrete grid points; that was the pre-vmap implementation.)

    **Accuracy** (dense 401-point sweep inside the bounds; FSPS/MILES, dpl SFH,
    z = 0.15, met the only free axis; worst-case relative error, requires the
    #1018 ionizing-shape fix):

    ==========================  ========  ==========  ==========
    met axis                    n points  [OIII]5007  Balmer
    ==========================  ========  ==========  ==========
    uniform + PCHIP (pre-#1020)       32      1.31 %      1.23 %
    snapped + linear                  23      0.46 %      0.24 %
    snapped + linear                  30      0.28 %      0.15 %
    ==========================  ========  ==========  ==========

    **DIG mixing (#2222)**: two lookups against this same table (HII at
    ``neb_logU``, DIG at ``neb_logU + neb_dig_delta_logU``), mixed by
    ``neb_dig_frac``. Measured worst-case relative error over 10 seeds
    (FSPS/MILES, dpl SFH, z = 0.15, ``neb_logU`` the only free axis,
    ``neb_dig_delta_logU = -1.0``, against the exact path): 1.17e-3
    (photometry) / 1.41e-3 (lines) at ``neb_dig_frac = 0.3`` with the axis
    extended from ``(-4, -1)`` to ``(-5, -1)`` (14 nodes to 19) to cover both
    query points, versus 1.72e-3 / 2.04e-3 at ``neb_dig_frac = 0`` (DIG
    absent, un-extended, 14 nodes) on the same fixture -- DIG mixing costs no
    accuracy relative to the table's own baseline; both comfortably inside
    this repository's 3e-2 parity ceiling. Freeing ``neb_dig_frac`` too (same
    fixture) gives 1.34e-3 / 1.91e-3; freeing ``neb_dig_delta_logU`` as well
    (the widest extension, ``(-8, -1)``, 32 nodes) and querying at a forced
    ``neb_dig_frac = 0`` still gives 4.6e-4 / 1.35e-3 -- at least as tight as
    the baseline, confirming the node-spacing preservation
    (:func:`_preserve_spacing_n`) does its job. Without the extension,
    :func:`~tengri.utils.grid_interp.interp_nd_pchip` clips the DIG query
    silently whenever it falls outside the un-extended axis: an earlier probe
    (#2222 research notes) measured 3 of 10 seeds at 5.1e-2 / 4.2e-1, an order
    of magnitude worse. Before #2222 an active ``neb_dig_frac`` raised
    ``ValueError`` at this call (``DIGNotOnNebularGridError``, now removed);
    this is the only
    unsupported-combination refusal this module carried, so removing it
    leaves ``_refuse_tabulated_metallicity`` as the sole remaining one.
    """
    validate_n_grid(n_grid)

    spec = model.spec
    free = set(spec.free_params)
    ranges = ranges or {}
    # neb_logU joins the axes whenever DIG mixing could be active (#2222) AND
    # the resulting extension is non-degenerate (#2222 review I3): the DIG
    # lookup needs a second query point (neb_logU + neb_dig_delta_logU)
    # distinct from the HII one, but if that second point coincides with the
    # first (neb_logU Fixed, neb_dig_delta_logU Fixed at exactly 0.0), DIG
    # mixing is arithmetically the HII term and no axis is needed. The
    # extension itself honors a user-supplied ranges['neb_logU'] as the HII
    # support to extend from, not only the spec's own (review I2), computed
    # once here and reused unchanged in the per-axis loop below.
    (
        dig_logU_needs_axis,
        dig_logU_own_lo,
        dig_logU_own_hi,
        dig_logU_ext_lo,
        dig_logU_ext_hi,
    ) = _dig_logU_axis_extension(spec, ranges.get("neb_logU"))
    axis_names = tuple(
        p for p in _CANDIDATE_AXES if p in free or (dig_logU_needs_axis and p == "neb_logU")
    )

    _refuse_tabulated_metallicity(model)

    met_nodes = _ssp_met_nodes(model) if snap_met_to_ssp_nodes else None

    # Per-axis resolution matched to the physics. A scalar ``n_grid`` resolves every
    # axis at ``n_grid``; a dict ``{axis: n}`` sets each explicitly and the rest fall
    # back to ``_DEFAULT_N_GRID``. An UNSNAPPED met axis is densified by
    # ``_MET_AXIS_DENSITY_FACTOR`` because it must resolve the SSP-node kinks by brute
    # force; a snapped one gets the nodes for free. Densification applies to the
    # *default* only: a per-axis number is an explicit request and is honored verbatim,
    # so ``{'met_logzsol': 30}`` builds 30 knots rather than silently doubling to 60.
    def _axis_n(name):
        if isinstance(n_grid, dict):
            if name in n_grid:
                return int(n_grid[name])
            requested = _DEFAULT_N_GRID
        else:
            requested = n_grid
        if name == "met_logzsol" and met_nodes is None:
            return int(requested * _MET_AXIS_DENSITY_FACTOR)
        return int(requested)

    # Guard: an UNSNAPPED free met axis cannot resolve the C0 kinks the ionizing-
    # spectrum tables put at every SSP metallicity node, so the forbidden lines
    # converge only as O(h) there. Snapping is the default; this fires when the
    # caller disabled it or the model exposes no SSP metallicity grid (#1020).
    if "met_logzsol" in axis_names and met_nodes is None:
        warnings.warn(
            f"nebular fast grid: met_logzsol is a FREE axis resolved on a UNIFORM "
            f"grid of {_axis_n('met_logzsol')} points. The exact per-Q_H emissivity "
            f"has C0 kinks at the SSP metallicity nodes (the ionizing-spectrum "
            f"tables interpolate bilinearly in met), which a uniform axis straddles: "
            f"the collisionally-excited lines then converge only as O(h): a dense "
            f"sweep shows [OIII] worst-case ~1.3 % at n=32 versus ~0.5 % for a "
            f"node-snapped axis of 23 points. Balmer lines are shape-insensitive and "
            f"are less affected. Prefer snap_met_to_ssp_nodes=True (the default), "
            f"which needs a model carrying ssp_data.ssp_lgmet.",
            stacklevel=2,
        )

    wavelengths = jnp.asarray(wavelengths)
    if ref_params is None:
        ref_params = dict(spec.sample(jax.random.PRNGKey(0)))
    else:
        ref_params = dict(ref_params)
    ref_z = ref_params.get("redshift", 0.0)
    # A tabulated SFH declares no parameters, so `spec.sample` cannot produce
    # its runtime arrays and the stellar component raises before the first row.
    # This table is per-Q_H and so independent of the SFH that built it (#1718),
    # which is why a stand-in serves: and why one already has to, since the
    # whole grid is built at a single sampled SFH for parametric models too.
    ref_params = {**reference_history_params(model, redshift=ref_z), **ref_params}
    # The table stores ONE photoionization regime per node (pure HII, no DIG):
    # DIG mixing happens at RECONSTRUCTION time via two lookups into this same
    # table (dig.py's mix_dig_grid_reconstruction), not by baking a mix into
    # the node itself. Each per-node build below calls model.predict_state /
    # predict_line_fluxes on the ORIGINAL (not-yet-grid-attached) chain, which
    # still applies the exact path's own DIG mix
    # (mix_dig_emission/mix_dig_line_luminosities) whenever neb_dig_frac is
    # nonzero -- and ref_params may carry the model's actual Fixed value
    # (#2222 allows building a grid from a DIG-active model). A Python-literal
    # 0.0 here makes that mix's own short-circuit fire during every node
    # build, regardless of the reference model's disposition, so the stored
    # per-Q_H value is always the undiluted HII term.
    ref_params["neb_dig_frac"] = 0.0
    log10_ref_divisor = _log10_four_pi_dl2(ref_z)  # observed flux -> luminosity

    axes, axis_kinds = [], []
    for name in axis_names:
        if name == "neb_logU" and dig_logU_needs_axis:
            # Extend to cover the DIG-shifted query point (never clip, never
            # refuse: #2222) -- on top of a user-supplied ranges['neb_logU']
            # too (review I2), and preserve the node spacing neb_logU would
            # have had absent DIG so neb_dig_frac=0 parity does not degrade.
            # dig_logU_ext_lo/hi/own_lo/own_hi are computed once, above, by
            # _dig_logU_axis_extension, which already folded in ranges.
            lo, hi = dig_logU_ext_lo, dig_logU_ext_hi
            n_points = _preserve_spacing_n(_axis_n(name), dig_logU_own_lo, dig_logU_own_hi, lo, hi)
        elif name in ranges:
            lo, hi = ranges[name]
            n_points = _axis_n(name)
        else:
            lo, hi = _axis_range(spec, name)
            n_points = _axis_n(name)
        if name == "met_logzsol" and met_nodes is not None:
            # knots on the kinks -> the cubic's cross-kink tangent is the error
            # floor, so this axis interpolates linearly (#1020)
            axes.append(jnp.asarray(_snap_axis_to_nodes(lo, hi, n_points, met_nodes)))
            axis_kinds.append("linear")
        else:
            axes.append(jnp.linspace(lo, hi, n_points))
            axis_kinds.append("pchip")
    axes = tuple(axes)
    axis_kinds = tuple(axis_kinds)

    def _row(point_values):
        """(line, phot|None, restband|None) at one grid point: one eager Cue forward.

        Kept for the single reference evaluation below (photometry-channel probe +
        vmap sanity check); the full grid is built vmapped, not by looping this.
        """
        row = jnp.asarray([float(v) for v in point_values])
        line, phot, rest = _row_traced(row, want_phot=True)
        return line, (None if phot is None else phot), (None if rest is None else rest)

    def _row_traced(row, *, want_phot):
        """Per-Q_H line (and optionally phot) vector at one grid point, tracer-safe.

        ``row`` is a ``(n_axes,)`` array so this vmaps: ``predict_state`` compiles
        once and runs batched over every node, instead of one eager forward per node
        (the #950 build looped ``predict_state`` ~n_grid**n_axes times: 256 eager
        forwards, ~11 min; vmapped it is one compile, ~seconds).
        """
        p = dict(ref_params)
        for i, name in enumerate(axis_names):
            p[name] = row[i]
        state = model.predict_state(p)
        # Q_H is ~1e53 photons/s, so the LINEAR ``nion`` is ``inf`` in float32 and
        # ``inv_qh`` is then exactly 0. The reciprocal is only ever used as a
        # divisor, so take it as a log offset instead and it never materializes
        # (#1859). ``log10(1e-30) == -30`` reproduces the old clamp.
        neg_log_qh = -jnp.maximum(_log_nion_of_state(state), -30.0)
        # intrinsic (redden=False) observed flux -> luminosity per Q_H
        flux = model.predict_line_fluxes(
            p, target_wavelengths=wavelengths, redden=False, state=state
        )
        # The un-divided luminosity is ~1e40 against a float32 max of 3.4e38, so
        # recovering it from the flux and *then* dividing by Q_H was ``inf * 0``.
        # Both offsets are ~+55 and ~-53 dex and cancel to an O(1e-13) answer;
        # applying them together is what keeps every intermediate in range.
        line_per_qh = apply_log10_scale(jnp.asarray(flux), log10_ref_divisor + neg_log_qh)
        if not want_phot:
            return line_per_qh, None, None
        # intrinsic nebular filter-integrated rest-frame L_nu per Q_H (the exact
        # per-eval publish, captured once at build time). Absent when the model
        # has no WavePrecomp filters (line-only grid).
        # Same reciprocal, same reason: ``inv_qh`` is ~1e-53, below float32's
        # smallest subnormal (1.4e-45), so the linear multiply flushes the whole
        # band to zero even though the ~1e-25 answer is representable.
        neb_phot = state.derived.get("nebular_phot_lnu_precomp")
        phot_per_qh = (
            None if neb_phot is None else apply_log10_scale(jnp.asarray(neb_phot), neg_log_qh)
        )
        # ...and its rest-frame twin, captured in the SAME forward (#1665).
        # Capturing only the observed band is what silently stripped the nebular
        # contribution out of every rest-frame band on the fast path.
        neb_rest = state.derived.get("nebular_restband_lnu_precomp")
        rest_per_qh = (
            None if neb_rest is None else apply_log10_scale(jnp.asarray(neb_rest), neg_log_qh)
        )
        return line_per_qh, phot_per_qh, rest_per_qh

    if not axis_names:
        grid_shape: tuple = ()
        points: list = [()]
    else:
        grid_shape = tuple(len(a) for a in axes)
        points = list(itertools.product(*[list(a) for a in axes]))

    # One eager reference forward: detects the photometry channel (line-only grids
    # have none) and anchors the vmap sanity check below.
    ref_line, ref_phot, ref_rest = _row(points[0])
    has_phot = ref_phot is not None
    if has_phot and ref_rest is None:
        # The exact path publishes the observed band and its rest-frame twin
        # together, so a reference model that emits one without the other is a
        # contract break upstream, not a grid variant. Refuse at BUILD time:
        # shipping the half-grid is what made #1665 silent for a whole release.
        raise RuntimeError(
            "nebular fast grid: the reference model published "
            "'nebular_phot_lnu_precomp' but not 'nebular_restband_lnu_precomp'. "
            "The two are twins (#1148/#1665); a grid carrying only the observed "
            "band silently drops the nebular contribution from every rest-frame "
            "band. Refusing to build a half-grid."
        )

    pts_arr = jnp.asarray([[float(v) for v in pt] for pt in points])  # (n_points, n_axes)

    def _in_chunks(fn, pts, n_out):
        """Run the vmapped ``fn`` over ``pts`` in batches of ``_BUILD_CHUNK_NODES``.

        Each chunk is forced to completion before the next is dispatched. Without
        that, JAX's async dispatch queues every chunk and holds all their
        intermediates live anyway: which is the very thing chunking is for.
        """
        n = pts.shape[0]
        if n <= _BUILD_CHUNK_NODES:
            return fn(pts)
        parts = []
        for i in range(0, n, _BUILD_CHUNK_NODES):
            part = fn(pts[i : i + _BUILD_CHUNK_NODES])
            parts.append(jax.block_until_ready(part))
        if n_out == 1:
            return jnp.concatenate(parts, axis=0)
        return tuple(jnp.concatenate([p[k] for p in parts], axis=0) for k in range(n_out))

    if has_phot:

        @jax.jit
        @jax.vmap
        def _eval_both(row):
            line, phot, rest = _row_traced(row, want_phot=True)
            return line, phot, rest

        # (n_points, n_line), (n_points, n_phot), (n_points, n_phot)
        line_all, phot_all, rest_all = _in_chunks(_eval_both, pts_arr, 3)
    else:

        @jax.jit
        @jax.vmap
        def _eval_line(row):
            line, _, _ = _row_traced(row, want_phot=False)
            return line

        line_all = _in_chunks(_eval_line, pts_arr, 1)  # (n_points, n_line)
        phot_all = None
        rest_all = None

    # Sanity: the vmapped first node must reproduce the eager reference forward.
    # The tolerance follows the working dtype: 1e-5 is the historical float64
    # bar and sits above float32's accumulated rounding on CPU, but a CUDA
    # float32 build differs from the eager forward by up to 1.2e-5 (measured,
    # reduction order), so float32 gets 256 ulp (3.1e-5). A vmap/tracer
    # regression is orders of magnitude away from either.
    parity_rtol = max(1e-5, 256.0 * float(jnp.finfo(line_all.dtype).eps))
    if not bool(jnp.allclose(line_all[0], ref_line, rtol=parity_rtol, atol=0.0)):
        raise RuntimeError(
            "nebular fast grid: vmapped build disagrees with the eager reference "
            "forward at the first node: a tracer/vmap regression, not a rounding gap."
        )

    def _stack_log(arr) -> jnp.ndarray:
        # log space: nebular luminosities span decades across the ionization grid
        return jnp.log10(jnp.maximum(arr, 1e-300)).reshape(*grid_shape, arr.shape[-1])

    log_line = _stack_log(line_all)
    log_phot = None if phot_all is None else _stack_log(phot_all)
    log_rest = None if rest_all is None else _stack_log(rest_all)

    return NebularGridTable(
        axis_names=axis_names,
        axes=axes,
        log_line_per_qh=log_line,
        wavelengths=wavelengths,
        log_phot_per_qh=log_phot,
        axis_kinds=axis_kinds,
        log_restband_per_qh=log_rest,
    )


def _kinds(table):
    """Per-axis interpolation kinds, tolerating tables pickled before #1020."""
    return tuple(table.axis_kinds) or None


def reconstruct_nebular_lines(nion, params, redshift, table) -> jnp.ndarray:
    r"""Reconstruct observed line fluxes from the grid: no Cue forward.

    .. math::

        F_{\rm line} = \frac{n_{\rm ion}\,
            \mathrm{interp}(\ell;\,Z_\star,\log U,\log Z_{\rm gas})}{4\pi\,d_L(z)^2}

    Parameters
    ----------
    nion : float
        Ionizing photon rate for this evaluation (stellar-published; == q_h).
    params : Mapping
        Parameter dict: the free-axis values (``params[name]`` for ``name`` in
        ``table.axis_names``) locate the query point.
    redshift : float
        Evaluation redshift: the cosmology is applied here, not baked in.
    table : NebularGridTable
        The grid from :func:`precompute_nebular_grid`.

    Returns
    -------
    ndarray, shape (n_lines,)
        Observed line fluxes [erg/s/cm^2] at the evaluation redshift.

    Notes
    -----
    **JIT-compatible / gradient-safe**: yes, node-exact PCHIP interpolation + a
    scalar multiply + the cosmology divisor.
    """
    # The intrinsic luminosity (~1e40 erg/s) and the divisor (~1e57) are both out
    # of float32 range and in opposite directions while the flux (~1e-16) is not,
    # so neither end is materialized: one exponent, one ``pow10`` (#1859).
    return pow10(
        reconstruct_nebular_line_log_lums(jnp.log10(jnp.asarray(nion)), params, table)
        - _log10_four_pi_dl2(redshift)
    )


def reconstruct_nebular_line_lums(nion, params, table) -> jnp.ndarray:
    r"""Intrinsic line **luminosities** [erg/s] from the grid: no Cue, no cosmology.

    The distance-independent core of :func:`reconstruct_nebular_lines`. Returns
    the intrinsic (un-reddened) line luminosities so a caller can apply dust
    attenuation at the line wavelengths and the cosmology dimming itself: the
    order :meth:`SEDModel.predict_line_fluxes` uses (redden the intrinsic
    catalog, then convert ``L / 4 pi d_L^2``).

    Parameters
    ----------
    nion : float
        Ionizing photon rate for this evaluation (stellar-published; == q_h).
    params : Mapping
        Parameter dict: the free-axis values (``params[name]`` for ``name`` in
        ``table.axis_names``) locate the query point. Use full public names
        (``met_logzsol`` / ``neb_logU`` / ``neb_logZ_gas``).
    table : NebularGridTable
        The grid from :func:`precompute_nebular_grid`.

    Returns
    -------
    ndarray, shape (n_lines,)
        Intrinsic line luminosities [erg/s].

    Notes
    -----
    **Not float32-safe, by construction**: a line luminosity is ~1e40 erg/s and
    ``nion`` ~1e53, both past float32's 3.4e38 ceiling, so this returns ``inf``
    there. That is a property of the erg/s contract, not a defect. Callers that
    must work at either precision take
    :func:`reconstruct_nebular_line_log_lums` and stay in the exponent, which is
    what :meth:`~tengri.forward.sed_model.SEDModel.predict_line_fluxes` does.
    """
    return pow10(reconstruct_nebular_line_log_lums(jnp.log10(jnp.asarray(nion)), params, table))


def reconstruct_nebular_line_log_lums(log_nion, params, table) -> jnp.ndarray:
    r"""log10 intrinsic line luminosities from the grid — the float32-safe form.

    .. math::

        \log_{10} L_{\rm line} = \log_{10} n_{\rm ion} + \log_{10}\ell

    with :math:`\ell` the interpolated luminosity-per-Q_H and :math:`n_{\rm ion}`
    the ionizing photon rate [photons/s].

    The linear sibling :func:`reconstruct_nebular_line_lums` multiplies a ~1e53
    :math:`n_{\rm ion}` by a ~1e-13 table value. Against a float32 max of 3.4e38
    the factor is ``inf`` before the multiply and the ~1e40 product would be out of
    range anyway, so the linear spelling is ``inf`` and the flux it feeds is
    ``nan`` (#1859) — while the ~1e-16 flux at the end of the chain is
    representable throughout. Adding the exponents keeps every intermediate in
    range, and it is the same closing step :func:`reconstruct_nebular_phot`
    already uses for the broadband twin — the line channel simply never got it.

    Parameters
    ----------
    log_nion : ndarray, shape ()
        log10 ionizing photon rate [dex re photons/s] (stellar-published
        ``log_nion``; == log10(q_h)).
    params : Mapping
        Parameter dict: the free-axis values (``params[name]`` for ``name`` in
        ``table.axis_names``) locate the query point. Use full public names
        (``met_logzsol`` / ``neb_logU`` / ``neb_logZ_gas``).
    table : NebularGridTable
        The grid from :func:`precompute_nebular_grid`.

    Returns
    -------
    ndarray, shape (n_lines,)
        log10 intrinsic line luminosities [dex re erg/s].

    Notes
    -----
    **JIT-compatible / gradient-safe**: yes, node-exact PCHIP interpolation plus
    a scalar add.
    """
    if not table.axis_names:
        log_lpq = table.log_line_per_qh
    else:
        point = tuple(jnp.asarray(params[name]).reshape(()) for name in table.axis_names)
        log_lpq = interp_nd_pchip(table.log_line_per_qh, table.axes, point, _kinds(table))
    return jnp.asarray(log_nion) + log_lpq  # node-exact geometric interp


def reconstruct_nebular_phot(log_nion, params, table) -> jnp.ndarray:
    r"""Reconstruct the intrinsic nebular photometry precompute: no Cue forward.

    The broadband analog of :func:`reconstruct_nebular_lines`. Returns the
    **rest-frame** filter-integrated ``L_nu`` (one column per filter) that the
    nebular component would publish as ``nebular_phot_lnu_precomp``:

    .. math::

        L_\nu^{\rm neb}(b) = 10^{\log_{10} n_{\rm ion} + \log_{10}\ell_b}

    **No cosmology or dust here**: unlike the line channel, this matches the
    intrinsic precompute contract: :meth:`Observation.predict_via_precomp`
    applies the young-limit dust screen (at the filter level) and the
    ``(1+z)/(4 pi d_L^2)`` dimming downstream, exactly as it does for the exact
    per-eval publish.

    Parameters
    ----------
    log_nion : float
        log10 ionizing photon rate for this evaluation [dex re photons/s]
        (stellar-published; == log10(q_h)).
    params : Mapping
        Parameter dict: the free-axis values locate the query point.
    table : NebularGridTable
        The grid from :func:`precompute_nebular_grid`, built from a
        ``WavePrecomp`` model so ``log_phot_per_qh`` is populated.

    Returns
    -------
    ndarray, shape (n_filter,)
        Intrinsic nebular filter-integrated rest-frame ``L_nu`` [erg/s/Hz].

    Raises
    ------
    ValueError
        If the table carries no photometry channel (``log_phot_per_qh is None``)
        rebuild from a ``WavePrecomp`` model with photometric filters.

    Notes
    -----
    **JIT-compatible / gradient-safe**: yes, node-exact PCHIP + log-domain add.
    The sibling :func:`reconstruct_nebular_line_lums` and
    :func:`reconstruct_nebular_lines` still take linear ``nion`` (their erg/s
    output is deferred to #1206 items 2/3).
    """
    return _reconstruct_band_channel(log_nion, params, table, "log_phot_per_qh", "photometry")


def _reconstruct_band_channel(log_nion, params, table, field, label) -> jnp.ndarray:
    """Shared core of the two per-filter reconstructions (#1665).

    ``reconstruct_nebular_phot`` and :func:`reconstruct_nebular_restband` differ
    only in which grid channel they read: same axes, same interpolation, same
    ``L_nu = 10^(log_nion + log_channel)`` closing step. One body so the twins
    cannot drift apart, which is the failure this issue was.

    Parameters
    ----------
    log_nion : float
        log10 ionizing photon rate [dex re photons/s].
    params : Mapping
        Parameter dict; free-axis values locate the query point.
    table : NebularGridTable
        The grid from :func:`precompute_nebular_grid`.
    field : str
        Grid attribute to read (``'log_phot_per_qh'`` / ``'log_restband_per_qh'``).
    label : str
        Human name of the channel for the error message (``'photometry'`` /
        ``'rest-band'``).

    Returns
    -------
    ndarray, shape (n_filter,)
        Intrinsic nebular filter-integrated rest-frame ``L_nu`` [erg/s/Hz].

    Notes
    -----
    **JIT-compatible / gradient-safe**: yes, node-exact PCHIP + log-domain add.
    """
    log_channel = getattr(table, field, None)
    if log_channel is None:
        raise ValueError(
            f"NebularGridTable has no {label} channel ({field} is None). "
            "Rebuild precompute_nebular_grid from a model built with "
            "approx=WavePrecomp() and photometric filters."
        )
    if not table.axis_names:
        log_cpq = log_channel
    else:
        point = tuple(jnp.asarray(params[name]).reshape(()) for name in table.axis_names)
        log_cpq = interp_nd_pchip(log_channel, table.axes, point, _kinds(table))
    return pow10(jnp.asarray(log_nion) + log_cpq)  # rest-frame L_nu; consumer applies dust + z


def reconstruct_nebular_restband(log_nion, params, table) -> jnp.ndarray:
    r"""Intrinsic nebular **rest-band** ``L_nu`` from the grid: the twin of phot.

    Same filters as :func:`reconstruct_nebular_phot`, integrated at
    ``redshift=0`` so the band sits in the rest frame (#1148). Reconstructs
    ``nebular_restband_lnu_precomp``, which every rest-frame consumer
    (spectral indices, rest-frame colors) reads.

    Parameters
    ----------
    log_nion : float
        log10 ionizing photon rate for this evaluation [dex re photons/s]
        (stellar-published; == log10(q_h)).
    params : Mapping
        Parameter dict: the free-axis values locate the query point.
    table : NebularGridTable
        The grid from :func:`precompute_nebular_grid`, built from a
        ``WavePrecomp`` model so ``log_restband_per_qh`` is populated.

    Returns
    -------
    ndarray, shape (n_filter,)
        Intrinsic nebular rest-frame band-integrated ``L_nu`` [erg/s/Hz].

    Raises
    ------
    ValueError
        If the table predates #1665 and carries no rest-band channel: rebuild
        the grid rather than serving the observed band in its place.

    Notes
    -----
    **JIT-compatible / gradient-safe**: yes, node-exact PCHIP + log-domain add.

    Omitting this publish on the fast path was #1665: all 13 spectral indices
    moved off the exact path, worst ``HgA`` by +1733%, silently.
    """
    return _reconstruct_band_channel(log_nion, params, table, "log_restband_per_qh", "rest-band")
