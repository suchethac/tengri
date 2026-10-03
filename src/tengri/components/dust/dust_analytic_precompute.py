# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapters for analytic dust-emission models.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
four analytic dust-emission models:

1. **modified_blackbody**: Optically-thin modified blackbody SED.
   Two free axes: ``dust_T`` (temperature), ``dust_beta_ir`` (emissivity index).

2. **casey2012**: Casey (2012) modified blackbody + mid-IR power law.
   Four free axes: ``dust_T`` (temperature), ``dust_beta_ir`` (emissivity index),
   ``dust_alpha_mir`` (mid-IR power-law slope), ``dust_lambda_0_um`` (opacity
   pivot wavelength).

3. **graybody**: General-opacity graybody with free pivot wavelength.
   Three free axes: ``dust_T`` (temperature), ``dust_beta_ir`` (emissivity index),
   ``dust_lambda_0_um`` (opacity pivot wavelength).

4. **pah_drude**: Sum of 18 PAH Drude profiles (Smith et al. 2007).
   No free axes (pure template; runtime amplitudes scale the combined PAH profile).

Each model is preintegrated through filter curves at model-initialization time.
The band integral is a Gauss-Legendre quadrature of the closed-form spectrum read at
the filter's own wavelengths, ``lambda_obs / (1 + z)`` in the rest frame, sub-divided
until the Wien-side slope is resolved (``tengri.components.dust._analytic_band_quadrature``),
so a filter outside the 0.01 um - 10 mm normalization grid, or on the Wien tail of a
cold spectrum, reads the true band flux. The table is carried as ``ln(flux)`` on
nodes spanning the declared prior range of each parameter and read with a tensor-product
cubic spline (``tengri.components.dust._analytic_table_interp``): node-exact, C2 in the
parameters, and accurate to :data:`TABLE_RTOL` off-node wherever the flux exceeds
:data:`TABLE_FLUX_FLOOR` per unit absorbed luminosity.

Auto-collapses axes whose corresponding parameters are ``Fixed`` in the user's
``Parameters``: e.g., a user who pins ``dust_T`` gets a 1D grid. The collapsed table is
the spline evaluated at the fixed value, so it reproduces the full lookup there.

References
----------
.. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
       MNRAS, 425, 3094 (2012). arXiv:1206.1595.
       https://doi.org/10.1111/j.1365-2966.2012.21455.x
.. [2] Smith, J. D., et al., "The mid-infrared emission of ultraluminous
       infrared galaxies," ApJ, 656, 770 (2007). arXiv:astro-ph/0701042.
       https://doi.org/10.1086/510378
.. [3] Hildebrand, R. H., "The determination of cloud masses from dust continuum
       emission," QJRAS, 24, 267 (1983).

"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.dust._analytic_band_quadrature import band_quadrature
from tengri.components.dust._analytic_table_interp import (
    AXIS_TRANSFORMS,
    LogSplineTable,
    build_log_spline_table,
    evaluate_log_spline,
    slice_log_spline,
)
from tengri.components.dust._params import PARAMS
from tengri.components.dust.drude_profiles import SMITH2007_PAH_FEATURES
from tengri.components.dust.emission import (
    casey2012 as _casey2012,
    graybody as _graybody,
    modified_blackbody as _modified_blackbody,
    pah_drude as _pah_drude,
)
from tengri.forward.precompute.templates import collapse_fixed_axes
from tengri.utils.grid_interp import PreintegratedGrid
from tengri.utils.interpolation import edges_for_grid
from tengri.utils.physics_constants import (
    AA_TO_CM as _AA_TO_CM,
    C_CGS as _C_CGS,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZ,
)
from tengri.utils.scale import log10_flux_scale as _log10_flux_scale

# ── Axis definitions per model ──────────────────────────────────

# modified_blackbody: parametrized by temperature and emissivity index
AXIS_PARAMS_MBB = ("dust_T", "dust_beta_ir")

# casey2012: parametrized by temperature, emissivity index, mid-IR slope, opacity pivot
AXIS_PARAMS_CASEY = ("dust_T", "dust_beta_ir", "dust_alpha_mir", "dust_lambda_0_um")

# graybody: parametrized by temperature, emissivity index, opacity pivot
AXIS_PARAMS_GRAYBODY = ("dust_T", "dust_beta_ir", "dust_lambda_0_um")

# pah_drude: pure template; no grid axes (runtime amplitude scaling)
AXIS_PARAMS_PAH = ()

# Protocol-required dict form for multi-model module
AXIS_PARAMS: dict[str, tuple[str, ...]] = {
    "modified_blackbody": AXIS_PARAMS_MBB,
    "casey2012": AXIS_PARAMS_CASEY,
    "graybody": AXIS_PARAMS_GRAYBODY,
    "pah_drude": AXIS_PARAMS_PAH,
}

#: The thermal-continuum closures: their spectra are normalized to ``L_absorbed`` by
#: the frequency integral over the rest grid they are handed, and carry the
#: da Cunha et al. (2013) CMB terms at the redshift they are built for.
_CONTINUUM_CLOSURES = {
    "modified_blackbody": _modified_blackbody,
    "casey2012": _casey2012,
    "graybody": _graybody,
}

#: Spline coordinate of each axis parameter (key of ``AXIS_TRANSFORMS``) and its node
#: count over the declared prior range. Node counts are set by the off-node accuracy
#: of the log flux in each coordinate, :data:`TABLE_RTOL`.
AXIS_TRANSFORM_KEYS: dict[str, str] = {
    "dust_T": "neg_inverse",
    "dust_beta_ir": "identity",
    "dust_alpha_mir": "log_shift_half",
    "dust_lambda_0_um": "log",
}
DEFAULT_AXIS_NODES: dict[str, int] = {
    "dust_T": 25,
    "dust_beta_ir": 6,
    "dust_alpha_mir": 11,
    "dust_lambda_0_um": 15,
}

#: Relative accuracy of an off-node lookup against the exact band integral over the
#: declared parameter ranges and ``0 <= z <= 6``, for fluxes above
#: :data:`TABLE_FLUX_FLOOR`.
TABLE_RTOL: float = 1.0e-3

#: Band flux per unit absorbed luminosity [1/Hz] below which the closures themselves
#: saturate (they clip ``h nu / k T`` at 500, a flux of ~1e-217): the lookup is
#: accurate to :data:`TABLE_RTOL` above this value and positive below it.
TABLE_FLUX_FLOOR: float = 1.0e-150


# Rest-frame normalization grid of the thermal-continuum models (modified_blackbody,
# casey2012, graybody): 0.01 um to 10 mm, log-spaced. The closures normalize their SED
# to L_absorbed by integrating over the grid they are given, so the grid has to contain
# the whole thermal bump; the band integrals read the spectrum at the filter's own
# wavelengths and do not depend on this grid's resolution. 3000 points hold the
# normalization trapezoid to ~1e-6.
_CONTINUUM_LOG10_WAVE_AA_MIN = 2.0
_CONTINUUM_LOG10_WAVE_AA_MAX = 8.0
_CONTINUUM_N_WAVE = 3000

# h c / k_B [K Angstrom]: the Wien exponent is x = _WIEN_K_AA / (lambda_AA * T).
_WIEN_K_AA = _H_PLANCK * _C_CGS / (_K_BOLTZ * _AA_TO_CM)
# The closures clip h nu / k T at 500 (the spectrum is flat beyond it), so the log slope
# of the spectrum never exceeds that exponent plus the margin below.
_WIEN_X_CAP = 500.0
# Added to the Wien exponent to bound the log slope of the spectrum prefactors
# (nu**(3 + beta), opacity, CMB contrast, mid-IR power law).
_SLOPE_MARGIN = 8.0
# |d ln D / d ln lambda| of a Drude profile of fractional width gamma peaks at 2 / gamma.
_PAH_LOG_SLOPE = 2.0 / min(f.gamma for f in SMITH2007_PAH_FEATURES) + _SLOPE_MARGIN

# Largest block of (table nodes x wavelength points) evaluated per jitted call.
_CHUNK_POINTS = 4_000_000


def _continuum_wave_rest() -> np.ndarray:
    """Rest-frame normalization grid [Angstrom] of the thermal-continuum builders."""
    return np.logspace(
        _CONTINUUM_LOG10_WAVE_AA_MIN,
        _CONTINUUM_LOG10_WAVE_AA_MAX,
        _CONTINUUM_N_WAVE,
        dtype=np.float64,
    )


def declared_range(name: str) -> tuple[float, float]:
    """Declared free-prior range ``(lo, hi)`` of an axis parameter.

    Parameters
    ----------
    name : str
        Parameter name, one of the continuum models' axis parameters.

    Returns
    -------
    tuple of float
        Lower and upper bound of the parameter's declared free prior.
    """
    decl = next(p for p in PARAMS if p.name == name)
    lo, hi = decl.free_prior.bounds
    return float(lo), float(hi)


def default_axis(name: str) -> np.ndarray:
    """Default node values of an axis: uniform in its spline coordinate over its declared range.

    Parameters
    ----------
    name : str
        Axis parameter name.

    Returns
    -------
    ndarray
        ``DEFAULT_AXIS_NODES[name]`` ascending nodes spanning :func:`declared_range`.
    """
    fwd, inv = AXIS_TRANSFORMS[AXIS_TRANSFORM_KEYS[name]]
    lo, hi = declared_range(name)
    return inv(np.linspace(fwd(lo), fwd(hi), DEFAULT_AXIS_NODES[name]))


def _resolve_axes(model: str, user_axes: dict[str, np.ndarray | None]) -> tuple[np.ndarray, ...]:
    """Axis nodes of ``model``: the caller's grid where given, else the declared-range default."""
    axes = []
    for name in AXIS_PARAMS[model]:
        given = user_axes.get(name)
        axes.append(default_axis(name) if given is None else np.asarray(given, dtype=np.float64))
    return tuple(axes)


def _evaluate_table_nodes(spectrum, axes, wave, pos, weights, owner, n_filters, norm=None):
    """Band flux at every node of the Cartesian product of ``axes`` (float64, host).

    ``norm`` is ``(norm_wave, norm_pos)`` for the normalized closures: the spectrum is also
    evaluated on ``norm_wave`` alone and the band flux rescaled by the ratio of the two
    normalizations, so the normalization integral is over the rest grid whatever the filters
    are (a filter node beyond the grid would otherwise extend the integral).
    """
    mesh = np.meshgrid(*axes, indexing="ij") if axes else []
    params = np.stack([m.ravel() for m in mesh], axis=-1) if axes else np.zeros((1, 0))
    chunk = max(1, min(_CHUNK_POINTS // wave.size, params.shape[0]))
    with jax.enable_x64(True):
        wave_j = jnp.asarray(wave)
        pos_j, w_j, own_j = jnp.asarray(pos), jnp.asarray(weights), jnp.asarray(owner)
        if norm is not None:
            norm_wave_j, norm_pos_j = jnp.asarray(norm[0]), jnp.asarray(norm[1])

        @jax.jit
        def block(p_block):
            def one(p):
                s = spectrum(wave_j, p)
                band = jax.ops.segment_sum(s[pos_j] * w_j, own_j, num_segments=n_filters)
                if norm is None:
                    return band
                s_norm = spectrum(norm_wave_j, p)
                ref = jnp.argmax(s_norm)
                return band * (s_norm[ref] / s[norm_pos_j[ref]])

            return jax.vmap(one)(p_block)

        out = []
        for start in range(0, params.shape[0], chunk):
            blk = params[start : start + chunk]
            pad = chunk - blk.shape[0]
            if pad:
                blk = np.concatenate([blk, np.repeat(blk[-1:], pad, axis=0)])
            out.append(np.asarray(block(jnp.asarray(blk)))[: chunk - pad])
    phot = np.concatenate(out)
    return phot.reshape((*(a.size for a in axes), n_filters))


def _build_continuum_phot(
    model: str,
    axes: tuple[np.ndarray, ...],
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    L_absorbed_ref: float,
) -> np.ndarray:
    """Band flux table of a thermal-continuum model, shape ``(*axis_lengths, n_filters)``.

    The closure is evaluated once per table node on the union of the normalization grid
    and the filters' quadrature nodes, so its trapezoid normalization sees the whole
    thermal bump and the bands read the spectrum itself, not an interpolation of it. The
    union grid's normalization is rescaled to that of the rest grid alone.
    """
    closure = _CONTINUUM_CLOSURES[model]
    names = AXIS_PARAMS[model]
    t_min = float(np.min(axes[0]))
    quad = band_quadrature(
        filter_waves,
        filter_trans,
        redshift,
        lambda w_aa: np.minimum(_WIEN_K_AA / (w_aa * t_min), _WIEN_X_CAP) + _SLOPE_MARGIN,
    )
    norm_grid = _continuum_wave_rest()
    wave, inverse = np.unique(np.concatenate([norm_grid, quad.wave_rest]), return_inverse=True)
    pos = inverse[norm_grid.size :]
    norm_pos = inverse[: norm_grid.size]

    def spectrum(wave_j, p):
        kw = {name: p[i] for i, name in enumerate(names)}
        return closure(wave_j, L_absorbed_ref, redshift=float(redshift), **kw)

    return _evaluate_table_nodes(
        spectrum,
        axes,
        wave,
        pos,
        quad.weights,
        quad.owner,
        quad.n_filters,
        norm=(norm_grid, norm_pos),
    )


def _build_pah_phot(
    filter_waves: list, filter_trans: list, redshift: float, L_absorbed_ref: float
) -> np.ndarray:
    """Band flux of the PAH Drude template, shape ``(1, n_filters)``."""
    quad = band_quadrature(
        filter_waves, filter_trans, redshift, lambda w_aa: np.full_like(w_aa, _PAH_LOG_SLOPE)
    )
    pos = np.arange(quad.wave_rest.size)

    def spectrum(wave_j, p):
        return _pah_drude(wave_j, L_absorbed_ref, redshift=float(redshift))

    phot = _evaluate_table_nodes(
        spectrum, (), quad.wave_rest, pos, quad.weights, quad.owner, quad.n_filters
    )
    return phot.reshape(1, quad.n_filters)


def _effective_wavelengths(filter_waves: list, filter_trans: list) -> np.ndarray:
    """Photon-weighted effective wavelength of each filter, observed frame [Angstrom]."""
    out = []
    for fw, ft in zip(filter_waves, filter_trans, strict=True):
        fw_np, ft_np = np.asarray(fw, dtype=np.float64), np.asarray(ft, dtype=np.float64)
        tw = ft_np / fw_np
        out.append(np.trapezoid(tw * fw_np, fw_np) / np.trapezoid(tw, fw_np))
    return np.asarray(out)


def _assemble_preint(
    log_phot: np.ndarray,
    axes: tuple[np.ndarray, ...],
    eff_obs: np.ndarray,
    redshift: float,
) -> PreintegratedGrid:
    """Wrap a log table as the typed :class:`PreintegratedGrid` (``phot`` = ``exp(log_phot)``)."""
    axes_j = tuple(jnp.asarray(a) for a in axes)
    return PreintegratedGrid(
        phot=jnp.asarray(np.exp(log_phot)),
        moment=None,
        axes=axes_j,
        edges=tuple(edges_for_grid(a) for a in axes_j),
        effective_wavelengths=jnp.asarray(eff_obs),
        effective_wavelengths_rest=jnp.asarray(eff_obs / (1.0 + redshift)),
        log10_flux_scale=_log10_flux_scale(redshift, 1.0),
        n_filters=int(eff_obs.size),
    )


def _fixed_axis_values(
    axis_params: tuple[str, ...], parameters: Any, model: str
) -> dict[int, float]:
    """Axis index -> value for every axis whose parameter is ``Fixed``.

    Decided by :func:`collapse_fixed_axes` on a two-node stand-in grid, so the
    Fixed-name matching, the dead-axis warning and the alignment checks are the shared
    ones; only the contraction differs (the spline, not the triweight kernel).
    """
    if parameters is None or not axis_params:
        return {}
    stub_axes = tuple(jnp.asarray([0.0, 1.0]) for _ in axis_params)
    stub = PreintegratedGrid(
        phot=jnp.zeros((2,) * len(axis_params) + (1,)),
        moment=None,
        axes=stub_axes,
        edges=tuple(edges_for_grid(a) for a in stub_axes),
        effective_wavelengths=jnp.ones(1),
        effective_wavelengths_rest=jnp.ones(1),
        log10_flux_scale=0.0,
        n_filters=1,
    )
    _, _, fixed = collapse_fixed_axes(
        stub, axis_params, parameters, origin=f"dust_analytic_precompute[{model}]"
    )
    return fixed


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    model: str = "modified_blackbody",
    T_grid: np.ndarray | None = None,
    beta_grid: np.ndarray | None = None,
    alpha_mir_grid: np.ndarray | None = None,
    lambda_0_um_grid: np.ndarray | None = None,
) -> dict:
    """Build preintegrated analytic dust grid, auto-collapsing Fixed-parameter axes.

    Multi-model entry point. Dispatches to the appropriate builder based
    on ``model`` parameter.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0-1).
    redshift : float
        Source redshift. The band integral reads the rest-frame spectrum at
        ``lambda_obs / (1 + z)``; the closures carry the CMB terms at this redshift. No
        ``(1 + z)`` or distance factor is applied. [dimensionless]
    parameters : Parameters | None
        Parameters spec, used to detect Fixed-axis parameters.
    model : str, keyword-only
        One of "modified_blackbody", "casey2012", "graybody", "pah_drude".
        Default: "modified_blackbody".
    T_grid : ndarray, optional
        Temperature nodes for modified_blackbody/casey2012/graybody [K]. If None,
        :data:`DEFAULT_AXIS_NODES` nodes spanning the declared range of ``dust_T``,
        uniform in ``-1/T``.
    beta_grid : ndarray, optional
        Emissivity-index nodes [dimensionless]. If None, uniform over the declared
        range of ``dust_beta_ir``.
    alpha_mir_grid : ndarray, optional
        Mid-IR power-law slope nodes for casey2012 [dimensionless]. If None, uniform in
        ``ln(alpha - 1/2)`` over the declared range of ``dust_alpha_mir``.
    lambda_0_um_grid : ndarray, optional
        Opacity pivot wavelength nodes for graybody and casey2012 [micron]. If None,
        uniform in ``ln(lambda_0)`` over the declared range of ``dust_lambda_0_um``.

    Returns
    -------
    dict
        Keys: "grid_phot" (photometry array, linear flux per unit absorbed luminosity),
        "axes" (free axes), "_preint" (PreintegratedGrid), "_table" (the
        ``LogSplineTable`` that
        :func:`build_lookup` reads), optionally "_collapsed_axes" (if any axes fixed).

    Raises
    ------
    ValueError
        For an unknown ``model``, a filter without positive transmission, a node grid
        that is not strictly ascending, or a band flux that is not positive and finite.

    References
    ----------
    .. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
           MNRAS, 425, 3094 (2012).
    .. [2] Smith, J. D., et al., "The mid-infrared emission of ultraluminous
           infrared galaxies," ApJ, 656, 770 (2007).

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.

    The table is built in float64 whatever the session's JAX precision.
    """
    if model not in AXIS_PARAMS:
        raise ValueError(f"Unknown analytic dust model: {model}")
    user_axes = {
        "dust_T": T_grid,
        "dust_beta_ir": beta_grid,
        "dust_alpha_mir": alpha_mir_grid,
        "dust_lambda_0_um": lambda_0_um_grid,
    }
    axes = _resolve_axes(model, user_axes)
    axis_params = AXIS_PARAMS[model]
    fw = [np.asarray(f, dtype=np.float64) for f in filter_waves]
    ft = [np.asarray(t, dtype=np.float64) for t in filter_trans]

    if model == "pah_drude":
        phot = _build_pah_phot(fw, ft, redshift, 1.0)
    else:
        phot = _build_continuum_phot(model, axes, fw, ft, redshift, 1.0)
    if not (np.all(np.isfinite(phot)) and np.all(phot > 0.0)):
        raise ValueError(
            f"dust_analytic_precompute[{model}]: band flux is not positive and finite at "
            f"every node (min {np.min(phot):.3e}); the log-carried table cannot represent it."
        )

    table = build_log_spline_table(
        np.log(phot), axes, tuple(AXIS_TRANSFORM_KEYS[n] for n in axis_params)
    )
    eff_obs = _effective_wavelengths(fw, ft)
    fixed = _fixed_axis_values(axis_params, parameters, model)
    table = slice_log_spline(table, fixed)
    remaining = tuple(table.axes)
    preint = _assemble_preint(np.asarray(table.log_phot), remaining, eff_obs, redshift)

    result = {
        "grid_phot": preint.phot,
        "axes": tuple(jnp.asarray(a) for a in remaining),
        "_preint": preint,
        "_table": table,
    }
    if fixed:
        result["_collapsed_axes"] = fixed
    return result


def build_lookup(
    preint: dict,
    *,
    model: str = "modified_blackbody",
    free_param_names: tuple[str, ...] | None = None,
):
    """Build the runtime analytic dust photometry lookup from a preintegrated dict.

    Reads the log-carried band table with a tensor-product cubic spline.

    Parameters
    ----------
    preint : dict
        Output of :func:`precompute`.
    model : str, keyword-only
        Analytic dust model name (for documentation; not used in lookup logic).
    free_param_names : tuple of str, optional
        Names of remaining free axes in the collapsed case. Not used.

    Returns
    -------
    callable
        JIT-compiled photometry lookup function with signature::

            fn(L_absorbed, *free_axis_values) -> ndarray, shape (n_filters,)

        Returns dust emission L_nu [erg/s/Hz]. Caller applies flux scaling. The
        ``pah_drude`` lookup, which has no axes, returns shape ``(1, n_filters)``.
        Values are the exponential of the spline of ``ln(flux)``; query values outside
        the node range are clipped to it.

    References
    ----------
    .. [1] Casey, C. M., "Dusty star-forming galaxies at high redshift,"
           MNRAS, 425, 3094 (2012).

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes; the interpolant is C2 in each query value (zero gradient
    where the query is clipped), and a query made only of float32 values is evaluated
    in float32, where the log carrier keeps cold-dust fluxes representable down to the
    float32 range of ``exp``.
    """
    table: LogSplineTable = preint["_table"]

    @jax.jit
    def dust_phot(L_absorbed, *free_axis_values):
        """Dust photometry [erg/s/Hz]: ``L_absorbed * exp(spline of ln flux)``."""
        return L_absorbed * jnp.exp(evaluate_log_spline(table, free_axis_values))

    return dust_phot
