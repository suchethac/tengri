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

from tengri.components.nebular.line_precompute import _four_pi_dl2
from tengri.utils.grid_interp import interp_nd_pchip

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
#: .. warning::
#:    **The met axis does NOT converge, and denser grids cannot fix it.** The exact
#:    Cue forward is *discontinuous* in ``met_logzsol``: ``cue.py`` selects the
#:    ionizing-spectrum shape from a single ``argmax``-chosen age bin, so as the
#:    metallicity varies the dominant bin flips and the collisionally-excited lines
#:    step (~33 % in [OIII], measured at met ~ -1.0955 on the FSPS/MILES grid).
#:    No interpolant can cross a jump, so a systematic dense sweep shows [OIII]
#:    worst-case error plateauing at ~10-23 % regardless of ``n_grid``.
#:    Recombination lines (Balmer, ∝ Q_H) are shape-insensitive and DO converge.
#:    The real fix is a Q_H-weighted ionizing-spectrum shape in the Cue backend
#:    (continuous + differentiable); until then this factor only buys a modest
#:    reduction away from the flips. **Validate with a dense sweep inside the grid
#:    range — random draws under-sample the jump and give ~100x optimistic bounds.**
_MET_AXIS_DENSITY_FACTOR = 2


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
    """

    axis_names: tuple
    axes: tuple
    log_line_per_qh: jnp.ndarray
    wavelengths: jnp.ndarray
    log_phot_per_qh: jnp.ndarray | None = None


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


def precompute_nebular_grid(
    model,
    wavelengths,
    *,
    n_grid: int = 16,
    ranges: dict | None = None,
    ref_params: dict | None = None,
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
        (``neb_logU`` / ``neb_logZ_gas``) at ``n_grid`` and gives the
        ``met_logzsol`` axis :data:`_MET_AXIS_DENSITY_FACTOR` x that. Pass a dict
        ``{axis_name: n}`` to set each axis explicitly (unspecified axes default to
        16). The gas axes converge with resolution; the met axis does **not** —
        see the :data:`_MET_AXIS_DENSITY_FACTOR` warning (the exact forward is
        discontinuous in met). Validate any accuracy claim with a dense sweep
        strictly inside the grid range, never with random draws.
    ranges : dict, optional
        Override ``{param: (lo, hi)}`` grid bounds. Defaults to each free param's
        prior support (else :data:`_DEFAULT_RANGE`).
    ref_params : dict, optional
        Reference parameter dict (grid axes are overwritten per point). Defaults
        to a mid-range sample.

    Returns
    -------
    NebularGridTable

    Notes
    -----
    **Build cost**: ``n_grid ** n_free_axes`` forward evaluations, once at
    construction (not JIT'd — a build-time loop over concrete grid points).
    """
    spec = model.spec
    free = set(spec.free_params)
    axis_names = tuple(p for p in _CANDIDATE_AXES if p in free)
    ranges = ranges or {}

    # Per-axis resolution matched to the physics. A scalar ``n_grid`` resolves the
    # smooth gas axes at ``n_grid`` and auto-densifies the sharp met axis by
    # ``_MET_AXIS_DENSITY_FACTOR``; a dict ``{axis: n}`` sets each explicitly.
    def _axis_n(name):
        if isinstance(n_grid, dict):
            return int(n_grid.get(name, 16))
        return int(n_grid * _MET_AXIS_DENSITY_FACTOR) if name == "met_logzsol" else int(n_grid)

    # Loud guard: a free met axis cannot be reconstructed reliably for the
    # collisionally-excited lines, at ANY resolution, because the exact Cue forward
    # is discontinuous in met (argmax age-bin selection of the ionizing-spectrum
    # shape — see _MET_AXIS_DENSITY_FACTOR). Warn unconditionally; denser grids only
    # shrink the error away from the jumps, never across them.
    if "met_logzsol" in axis_names:
        warnings.warn(
            f"nebular fast grid: met_logzsol is a FREE axis (resolved at "
            f"{_axis_n('met_logzsol')} points). The exact Cue forward is "
            f"DISCONTINUOUS in met — it picks the ionizing-spectrum shape from a "
            f"single argmax-selected age bin, so [OIII] steps ~33 % when the "
            f"dominant bin flips. No grid can interpolate across that: a dense "
            f"systematic sweep shows [OIII] worst-case ~10-23 % at any n_grid "
            f"(random-draw checks under-sample the jump and look ~100x better). "
            f"Balmer lines (recombination, proportional to Q_H) are shape-insensitive "
            f"and DO converge. Fix met_logzsol, restrict to Balmer, or use the exact "
            f"path until the Cue ionizing-shape weighting is corrected.",
            stacklevel=2,
        )

    wavelengths = jnp.asarray(wavelengths)
    if ref_params is None:
        ref_params = dict(spec.sample(jax.random.PRNGKey(0)))
    else:
        ref_params = dict(ref_params)
    ref_z = ref_params.get("redshift", 0.0)
    ref_divisor = _four_pi_dl2(ref_z)  # observed flux -> luminosity

    axes = []
    for name in axis_names:
        lo, hi = ranges.get(name, _axis_range(spec, name))
        axes.append(jnp.linspace(lo, hi, _axis_n(name)))
    axes = tuple(axes)

    def _row(point_values):
        """(line_per_qh, phot_per_qh|None) at one grid point — the Cue forward."""
        p = dict(ref_params)
        for name, val in zip(axis_names, point_values, strict=True):
            p[name] = jnp.asarray(float(val))
        state = model.predict_state(p)
        inv_qh = 1.0 / jnp.maximum(_nion_of_state(state), 1e-30)
        # intrinsic (redden=False) observed flux -> luminosity per Q_H
        flux = model.predict_line_fluxes(
            p, target_wavelengths=wavelengths, redden=False, state=state
        )
        line_per_qh = jnp.asarray(flux) * ref_divisor * inv_qh
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
    rows = [_row(pt) for pt in points]

    def _stack_log(vectors) -> jnp.ndarray:
        # log space: nebular luminosities span decades across the ionization grid
        arr = jnp.stack(vectors)  # (n_points, n_channel)
        return jnp.log10(jnp.maximum(arr, 1e-300)).reshape(*grid_shape, arr.shape[-1])

    log_line = _stack_log([r[0] for r in rows])
    log_phot = None if rows[0][1] is None else _stack_log([r[1] for r in rows])

    return NebularGridTable(
        axis_names=axis_names,
        axes=axes,
        log_line_per_qh=log_line,
        wavelengths=wavelengths,
        log_phot_per_qh=log_phot,
    )


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
        log_lpq = interp_nd_pchip(table.log_line_per_qh, table.axes, point)
    return jnp.asarray(nion) * (10.0**log_lpq)  # node-exact geometric interp


def reconstruct_nebular_phot(nion, params, table) -> jnp.ndarray:
    r"""Reconstruct the intrinsic nebular photometry precompute — no Cue forward.

    The broadband analog of :func:`reconstruct_nebular_lines`. Returns the
    **rest-frame** filter-integrated ``L_nu`` (one column per filter) that the
    nebular component would publish as ``nebular_phot_lnu_precomp``:

    .. math::

        L_\nu^{\rm neb}(b) = n_{\rm ion}\,
            \mathrm{interp}\bigl(\ell_b;\,Z_\star,\log U,\log Z_{\rm gas}\bigr)

    **No cosmology or dust here** — unlike the line channel, this matches the
    intrinsic precompute contract: :meth:`Observation.predict_via_precomp`
    applies the young-limit dust screen (at the filter level) and the
    ``(1+z)/(4 pi d_L^2)`` dimming downstream, exactly as it does for the exact
    per-eval publish.

    Parameters
    ----------
    nion : float
        Ionizing photon rate for this evaluation (stellar-published; == q_h).
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
    **JIT-compatible / gradient-safe**: yes — node-exact PCHIP + a scalar multiply.
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
        log_ppq = interp_nd_pchip(table.log_phot_per_qh, table.axes, point)
    return jnp.asarray(nion) * (10.0**log_ppq)  # rest-frame L_nu; consumer applies dust + z
