# SPDX-License-Identifier: BSD-3-Clause
r"""Adaptive-axis nebular line grid — variable ionization via precompute (#950).

Generalizes the fixed-ionization line-per-Q_H table (:mod:`line_precompute`,
#955) to a grid over the ionization parameters that are **free** in a given
model. Cue makes each nebular line luminosity exactly linear in the hydrogen-
ionizing photon rate Q_H, and the per-Q_H factor depends only on the stellar
metallicity (which sets the ionizing-spectrum shape) and the gas conditions
(logU, gas-phase metallicity) — **not** on the star-formation-history shape
(#950, CV = 0 % across SFH draws). So

.. math::

    F_{\rm line}(\theta) = \frac{n_{\rm ion}(\theta)\,
        \ell\bigl(Z_\star, \log U, \log Z_{\rm gas}\bigr)}{4\pi\,d_L(z)^2}

where :math:`\ell` is a stored **luminosity per Q_H** (distance-independent, so
the cosmology is applied at the evaluation redshift) interpolated over whichever
of ``met_logzsol`` / ``neb_logU`` / ``neb_logZ_gas`` are free. Fixed parameters
are baked into the grid at their spec value, so a tighter setup gets a smaller,
faster grid automatically:

* all three fixed  → 0 axes (one template; pure Q_H scaling — the #955 case);
* ``neb_logU`` free → 1 axis;
* ``neb_logU`` + ``neb_logZ_gas`` free → 2 axes; …

Interpolation is **node-exact** monotone-cubic PCHIP
(:func:`tengri.utils.grid_interp.interp_nd_pchip`, the SKIRTOR pattern) in
log10-luminosity space, so the reconstruction is exact at grid nodes,
JIT/gradient-safe, and free of the smoothing bias a kernel smoother introduces on
the steeply logU-varying lines. The ionizing-spectrum shape is
**not** a grid axis — it is carried by the ``met_logzsol`` axis (SFH-independent
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

from tengri.components.nebular.line_precompute import _four_pi_dl2
from tengri.components.stellar.reference_history import reference_history_params
from tengri.parameters.translate import LOG10_ZSUN
from tengri.utils.grid_interp import interp_nd_pchip
from tengri.utils.scale import pow10

#: Parameters that may become grid axes when free. ``met_logzsol`` sets the
#: ionizing-spectrum shape; ``neb_logU`` / ``neb_logZ_gas`` are the gas
#: conditions. ``neb_fesc`` stays fixed (it rescales the escaping continuum, not
#: a smooth interpolation axis) and the ionizing-spectrum params are SSP-derived.
_CANDIDATE_AXES = ("met_logzsol", "neb_logU", "neb_logZ_gas")

#: Fallback grid bounds used ONLY when a free axis's prior exposes no finite
#: support (e.g. an unbounded Gaussian). Kept at least as wide as the standard
#: priors (met_logzsol ~ Uniform(-2, 0.2/0.5)) so the fallback still spans the
#: sampled region — the primary path reads the prior's actual bounds.
_DEFAULT_RANGE = {
    "met_logzsol": (-2.0, 0.5),
    "neb_logU": (-4.0, -1.0),
    "neb_logZ_gas": (-1.0, 0.5),
}

#: Extra resolution on the ``met_logzsol`` axis relative to the smooth gas axes.
#:
#: Applied **only** when the met axis cannot be snapped to the SSP metallicity
#: nodes (``snap_met_to_ssp_nodes=False``, or a model exposing no SSP metallicity
#: grid). A snapped axis puts knots on the kinks and interpolates linearly across
#: them, which converges — blind densification of an unsnapped axis does not, so
#: it needs the extra points more (#1020).
_MET_AXIS_DENSITY_FACTOR = 2

#: A uniform grid point within this fraction of a cell width of an SSP metallicity
#: node is dropped in favor of the node (see :func:`_snap_axis_to_nodes`), so
#: snapping never creates a near-degenerate interpolation cell.
_SNAP_MERGE_FRAC = 0.25

#: Points used for an axis the caller did not resolve explicitly — both the
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
    default resolution — the user got a grid they did not ask for, with no
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
#: same nodes — vmap applies no cross-node reduction, so the per-node result does
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
    axes : tuple of ndarray
        One grid-value array per axis, ascending.
    log_line_per_qh : ndarray, shape ``(*grid_dims, n_lines)``
        ``log10`` of the line **luminosity** per unit ``nion`` [erg/s per
        (photons/s)], distance-independent. Stored in log space because line
        luminosities span decades across (met, logU, logZ_gas) — geometric
        (log-space) interpolation is far more accurate than arithmetic there.
        ``grid_dims`` matches ``axes``; a 0-axis table is shape ``(n_lines,)``.
    wavelengths : ndarray, shape (n_lines,)
        Rest-frame vacuum line wavelengths [Angstrom].
    log_phot_per_qh : ndarray, shape ``(*grid_dims, n_filter)`` or None
        ``log10`` of the **intrinsic** (un-reddened) nebular filter-integrated
        rest-frame ``L_nu`` per unit ``nion`` [erg/s/Hz per (photons/s)] — the
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
    """

    axis_names: tuple
    axes: tuple
    log_line_per_qh: jnp.ndarray
    wavelengths: jnp.ndarray
    log_phot_per_qh: jnp.ndarray | None = None
    axis_kinds: tuple = ()


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


def _axis_range(spec, name):
    """(lo, hi) grid bounds for a free axis — from its prior's finite support.

    Reads the bounded prior's support (``bounds`` tuple, else ``lo``/``hi`` — the
    attributes tengri's :class:`Uniform` / :class:`LogUniform` expose). Only when
    the prior has no finite support (e.g. an unbounded Gaussian used as an axis)
    does it fall back to :data:`_DEFAULT_RANGE`, and it warns rather than silently
    ignoring the prior — a too-narrow grid would extrapolate and bias the fit.
    """
    dist = spec.get_distribution(name)
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


def _refuse_tabulated_metallicity(model):
    """Refuse a tabulated metallicity, whose LUT axis cannot exist (#1718).

    The axes are ``tuple(p for p in _CANDIDATE_AXES if p in free)`` — free
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
        "the model's free parameters, and stellar={'met_mode': 'table'} declares "
        "none — so met_logzsol is absent, the metallicity axis silently drops, "
        "and the whole table would be built at one reference metallicity. "
        "Measured that way against the exact path, OIII_5007 came out 17.5% "
        "wrong and NII_6584 5.3%, which is precisely the line-ratio information "
        "a nebular fit exists to use. Either drop FeaturePrecomp and keep the "
        "exact line path (WavePrecomp alone is unaffected and still applies), or "
        "use a parametric metallicity — a tabulated SFH with a free met_logzsol "
        "is supported and agrees with exact to 0.3% (#1718)."
    )


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
    but not exactly arbitrary — well inside the interpolation error.

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
        strictly inside the grid range, never with random draws — a narrow feature
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
    construction. They are JIT'd and vmapped over the grid — one compile, not one
    eager forward per node — and evaluated in batches of
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
    """
    validate_n_grid(n_grid)

    spec = model.spec
    free = set(spec.free_params)
    axis_names = tuple(p for p in _CANDIDATE_AXES if p in free)
    ranges = ranges or {}

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
            f"the collisionally-excited lines then converge only as O(h) — a dense "
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
    # which is why a stand-in serves — and why one already has to, since the
    # whole grid is built at a single sampled SFH for parametric models too.
    ref_params = {**reference_history_params(model, redshift=ref_z), **ref_params}
    ref_divisor = _four_pi_dl2(ref_z)  # observed flux -> luminosity

    axes, axis_kinds = [], []
    for name in axis_names:
        lo, hi = ranges.get(name, _axis_range(spec, name))
        if name == "met_logzsol" and met_nodes is not None:
            # knots on the kinks -> the cubic's cross-kink tangent is the error
            # floor, so this axis interpolates linearly (#1020)
            axes.append(jnp.asarray(_snap_axis_to_nodes(lo, hi, _axis_n(name), met_nodes)))
            axis_kinds.append("linear")
        else:
            axes.append(jnp.linspace(lo, hi, _axis_n(name)))
            axis_kinds.append("pchip")
    axes = tuple(axes)
    axis_kinds = tuple(axis_kinds)

    def _row(point_values):
        """(line_per_qh, phot_per_qh|None) at one grid point — one eager Cue forward.

        Kept for the single reference evaluation below (photometry-channel probe +
        vmap sanity check); the full grid is built vmapped, not by looping this.
        """
        row = jnp.asarray([float(v) for v in point_values])
        line, phot = _row_traced(row, want_phot=True)
        return line, (None if phot is None else phot)

    def _row_traced(row, *, want_phot):
        """Per-Q_H line (and optionally phot) vector at one grid point, tracer-safe.

        ``row`` is a ``(n_axes,)`` array so this vmaps: ``predict_state`` compiles
        once and runs batched over every node, instead of one eager forward per node
        (the #950 build looped ``predict_state`` ~n_grid**n_axes times — 256 eager
        forwards, ~11 min; vmapped it is one compile, ~seconds).
        """
        p = dict(ref_params)
        for i, name in enumerate(axis_names):
            p[name] = row[i]
        state = model.predict_state(p)
        inv_qh = 1.0 / jnp.maximum(_nion_of_state(state), 1e-30)
        # intrinsic (redden=False) observed flux -> luminosity per Q_H
        flux = model.predict_line_fluxes(
            p, target_wavelengths=wavelengths, redden=False, state=state
        )
        line_per_qh = jnp.asarray(flux) * ref_divisor * inv_qh
        if not want_phot:
            return line_per_qh, None
        # intrinsic nebular filter-integrated rest-frame L_nu per Q_H (the exact
        # per-eval publish, captured once at build time). Absent when the model
        # has no WavePrecomp filters (line-only grid).
        neb_phot = state.derived.get("nebular_phot_lnu_precomp")
        phot_per_qh = None if neb_phot is None else jnp.asarray(neb_phot) * inv_qh
        return line_per_qh, phot_per_qh

    if not axis_names:
        grid_shape: tuple = ()
        points: list = [()]
    else:
        grid_shape = tuple(len(a) for a in axes)
        points = list(itertools.product(*[list(a) for a in axes]))

    # One eager reference forward: detects the photometry channel (line-only grids
    # have none) and anchors the vmap sanity check below.
    ref_line, ref_phot = _row(points[0])
    has_phot = ref_phot is not None

    pts_arr = jnp.asarray([[float(v) for v in pt] for pt in points])  # (n_points, n_axes)

    def _in_chunks(fn, pts, n_out):
        """Run the vmapped ``fn`` over ``pts`` in batches of ``_BUILD_CHUNK_NODES``.

        Each chunk is forced to completion before the next is dispatched. Without
        that, JAX's async dispatch queues every chunk and holds all their
        intermediates live anyway — which is the very thing chunking is for.
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
            line, phot = _row_traced(row, want_phot=True)
            return line, phot

        # (n_points, n_line), (n_points, n_phot)
        line_all, phot_all = _in_chunks(_eval_both, pts_arr, 2)
    else:

        @jax.jit
        @jax.vmap
        def _eval_line(row):
            line, _ = _row_traced(row, want_phot=False)
            return line

        line_all = _in_chunks(_eval_line, pts_arr, 1)  # (n_points, n_line)
        phot_all = None

    # Sanity: the vmapped first node must reproduce the eager reference forward.
    if not bool(jnp.allclose(line_all[0], ref_line, rtol=1e-5, atol=0.0)):
        raise RuntimeError(
            "nebular fast grid: vmapped build disagrees with the eager reference "
            "forward at the first node — a tracer/vmap regression, not a rounding gap."
        )

    def _stack_log(arr) -> jnp.ndarray:
        # log space: nebular luminosities span decades across the ionization grid
        return jnp.log10(jnp.maximum(arr, 1e-300)).reshape(*grid_shape, arr.shape[-1])

    log_line = _stack_log(line_all)
    log_phot = None if phot_all is None else _stack_log(phot_all)

    return NebularGridTable(
        axis_names=axis_names,
        axes=axes,
        log_line_per_qh=log_line,
        wavelengths=wavelengths,
        log_phot_per_qh=log_phot,
        axis_kinds=axis_kinds,
    )


def _kinds(table):
    """Per-axis interpolation kinds, tolerating tables pickled before #1020."""
    return tuple(table.axis_kinds) or None


def reconstruct_nebular_lines(nion, params, redshift, table) -> jnp.ndarray:
    r"""Reconstruct observed line fluxes from the grid — no Cue forward.

    .. math::

        F_{\rm line} = \frac{n_{\rm ion}\,
            \mathrm{interp}(\ell;\,Z_\star,\log U,\log Z_{\rm gas})}{4\pi\,d_L(z)^2}

    Parameters
    ----------
    nion : float
        Ionizing photon rate for this evaluation (stellar-published; == q_h).
    params : Mapping
        Parameter dict — the free-axis values (``params[name]`` for ``name`` in
        ``table.axis_names``) locate the query point.
    redshift : float
        Evaluation redshift — the cosmology is applied here, not baked in.
    table : NebularGridTable
        The grid from :func:`precompute_nebular_grid`.

    Returns
    -------
    ndarray, shape (n_lines,)
        Observed line fluxes [erg/s/cm^2] at the evaluation redshift.

    Notes
    -----
    **JIT-compatible / gradient-safe**: yes — node-exact PCHIP interpolation + a
    scalar multiply + the cosmology divisor.
    """
    return reconstruct_nebular_line_lums(nion, params, table) / _four_pi_dl2(redshift)


def reconstruct_nebular_line_lums(nion, params, table) -> jnp.ndarray:
    r"""Intrinsic line **luminosities** [erg/s] from the grid — no Cue, no cosmology.

    The distance-independent core of :func:`reconstruct_nebular_lines`. Returns
    the intrinsic (un-reddened) line luminosities so a caller can apply dust
    attenuation at the line wavelengths and the cosmology dimming itself — the
    order :meth:`SEDModel.predict_line_fluxes` uses (redden the intrinsic
    catalog, then convert ``L / 4 pi d_L^2``).

    Parameters
    ----------
    nion : float
        Ionizing photon rate for this evaluation (stellar-published; == q_h).
    params : Mapping
        Parameter dict — the free-axis values (``params[name]`` for ``name`` in
        ``table.axis_names``) locate the query point. Use full public names
        (``met_logzsol`` / ``neb_logU`` / ``neb_logZ_gas``).
    table : NebularGridTable
        The grid from :func:`precompute_nebular_grid`.

    Returns
    -------
    ndarray, shape (n_lines,)
        Intrinsic line luminosities [erg/s].
    """
    if not table.axis_names:
        log_lpq = table.log_line_per_qh
    else:
        point = tuple(jnp.asarray(params[name]).reshape(()) for name in table.axis_names)
        log_lpq = interp_nd_pchip(table.log_line_per_qh, table.axes, point, _kinds(table))
    return jnp.asarray(nion) * (10.0**log_lpq)  # node-exact geometric interp


def reconstruct_nebular_phot(log_nion, params, table) -> jnp.ndarray:
    r"""Reconstruct the intrinsic nebular photometry precompute — no Cue forward.

    The broadband analog of :func:`reconstruct_nebular_lines`. Returns the
    **rest-frame** filter-integrated ``L_nu`` (one column per filter) that the
    nebular component would publish as ``nebular_phot_lnu_precomp``:

    .. math::

        L_\nu^{\rm neb}(b) = 10^{\log_{10} n_{\rm ion} + \log_{10}\ell_b}

    **No cosmology or dust here** — unlike the line channel, this matches the
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
        Parameter dict — the free-axis values locate the query point.
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
        — rebuild from a ``WavePrecomp`` model with photometric filters.

    Notes
    -----
    **JIT-compatible / gradient-safe**: yes — node-exact PCHIP + log-domain add.
    The sibling :func:`reconstruct_nebular_line_lums` and
    :func:`reconstruct_nebular_lines` still take linear ``nion`` (their erg/s
    output is deferred to #1206 items 2/3).
    """
    if table.log_phot_per_qh is None:
        raise ValueError(
            "NebularGridTable has no photometry channel (log_phot_per_qh is None). "
            "Rebuild precompute_nebular_grid from a model built with "
            "approx=WavePrecomp() and photometric filters."
        )
    if not table.axis_names:
        log_ppq = table.log_phot_per_qh
    else:
        point = tuple(jnp.asarray(params[name]).reshape(()) for name in table.axis_names)
        log_ppq = interp_nd_pchip(table.log_phot_per_qh, table.axes, point, _kinds(table))
    return pow10(jnp.asarray(log_nion) + log_ppq)  # rest-frame L_nu; consumer applies dust + z
