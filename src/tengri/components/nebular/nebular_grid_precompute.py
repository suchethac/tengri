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
**not** a grid axis — it is carried by the ``met_logzsol`` axis (SFH-independent,
#950). See issue #950.
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

#: Fallback grid bounds when a free axis's prior exposes no ``low``/``high``.
_DEFAULT_RANGE = {
    "met_logzsol": (-1.8, 0.4),
    "neb_logU": (-4.0, -1.0),
    "neb_logZ_gas": (-1.0, 0.5),
}


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
    """

    axis_names: tuple
    axes: tuple
    log_line_per_qh: jnp.ndarray
    wavelengths: jnp.ndarray


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
    spec value. Because the per-Q_H factor is SFH-shape-independent, the reference
    SFH is arbitrary.

    Parameters
    ----------
    model : SEDModel
        A model with a Q_H-linear nebular backend (Cue / CloudyGrid).
    wavelengths : array_like, shape (n_lines,)
        Rest-frame vacuum target line wavelengths [Angstrom].
    n_grid : int, default 16
        Grid points per free axis. Denser → tighter interpolation (the met
        dependence is nonlinear near solar; 16-24 gives < few e-3 on strong DESI
        lines with the triweight kernel).
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
        axes.append(jnp.linspace(lo, hi, n_grid))
    axes = tuple(axes)

    def _row(point_values) -> jnp.ndarray:
        p = dict(ref_params)
        for name, val in zip(axis_names, point_values, strict=True):
            p[name] = jnp.asarray(float(val))
        state = model.predict_state(p)
        # intrinsic (redden=False) observed flux -> luminosity per Q_H
        flux = model.predict_line_fluxes(
            p, target_wavelengths=wavelengths, redden=False, state=state
        )
        nion = _nion_of_state(state)
        return jnp.asarray(flux) * ref_divisor / jnp.maximum(nion, 1e-30)

    if not axis_names:
        line_per_qh = _row(())  # (n_lines,)
    else:
        grid_shape = tuple(len(a) for a in axes)
        rows = [_row(pt) for pt in itertools.product(*[list(a) for a in axes])]
        line_per_qh = jnp.stack(rows).reshape(*grid_shape, wavelengths.shape[0])

    return NebularGridTable(
        axis_names=axis_names,
        axes=axes,
        # log space: line luminosities span decades across the ionization grid
        log_line_per_qh=jnp.log10(jnp.maximum(line_per_qh, 1e-300)),
        wavelengths=wavelengths,
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
    **JIT-compatible / gradient-safe**: yes — triweight interpolation + a scalar
    multiply + the cosmology divisor.
    """
    if not table.axis_names:
        log_lpq = table.log_line_per_qh
    else:
        point = tuple(jnp.asarray(params[name]).reshape(()) for name in table.axis_names)
        log_lpq = interp_nd_pchip(table.log_line_per_qh, table.axes, point)
    lum = jnp.asarray(nion) * (10.0**log_lpq)  # node-exact geometric interp
    return lum / _four_pi_dl2(redshift)
