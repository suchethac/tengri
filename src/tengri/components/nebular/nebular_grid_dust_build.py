# SPDX-License-Identifier: BSD-3-Clause
"""Build-time helpers for the dust channels of the per-Q_H nebular grid.

Two channel groups let a multiplicative dust screen be applied to the tabulated
nebular emission at runtime without the full-wavelength Cue continuum:

* **Sub-band photometry.** Each filter's band integral is split into ``K``
  equal-filter-mass chunks with the SAME quadrature the stellar LUT uses
  (:func:`~tengri.utils.grid_interp.subband_quadrature`); a screen ``T`` is then
  applied as ``sum_k Phi_k T(lambda_k)``.
* **Energy balance.** The LyC-masked absorbed nebular luminosity per unit Q_H,
  through the nebular screen, at every node of the stellar energy-balance LUT's
  ``(tau_a, tau_b)`` grid.

Everything here is numpy (or one small ``lax.map``) work executed once, outside
the vmapped per-node forward.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tengri.forward.energy_balance import bolometric_absorbed_log10
from tengri.parameters.resolve import require_redshift
from tengri.utils.filter_convention import FilterConvention, filter_weight_np
from tengri.utils.grid_interp import (
    _np_trapezoid,
    _vectorized_interp,
    subband_quadrature,
)
from tengri.utils.physics_constants import C_AA, LYMAN_LIMIT_AA
from tengri.utils.scale import representable_denominator, representable_floor

__all__ = [
    "_lyc_cutoff_for",
    "_nebular_eb_channel",
    "_nebular_screen_for",
    "_nebular_subband_channels",
]


def _lyc_cutoff_for(dust) -> float | None:
    """The short-wavelength edge of the dust component's energy-balance integral [Angstrom].

    ``None`` when the component counts the Lyman continuum in the absorbed
    luminosity (``dust.config.lyc_in_energy_balance``), the Lyman edge
    (:data:`tengri.components.lyc.LYMAN_LIMIT_AA`) otherwise. The grid's
    absorbed-energy channel and the table's record of the choice both read it
    here, so the configuration key is spelled in one place.
    """
    return None if dust.config.lyc_in_energy_balance else LYMAN_LIMIT_AA


def _nebular_subband_channels(
    sed_nodes_rest: np.ndarray,
    wave_rest: np.ndarray,
    filters: Sequence[tuple[np.ndarray, np.ndarray]],
    redshift: float,
    n_subbands: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sub-band integrals and quadrature nodes of per-node nebular SEDs.

    Mirrors the union-grid quadrature of
    :func:`~tengri.utils.grid_interp.preintegrate_grid`, so the chunks factor the
    exact filter integral the stellar LUT factors.

    Parameters
    ----------
    sed_nodes_rest : ndarray, shape (n_points, n_wave)
        Rest-frame nebular L_nu of every grid node on ``wave_rest``.
    wave_rest : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom], ascending.
    filters : sequence of (wave, trans)
        Observed-frame filter tables [Angstrom, dimensionless].
    redshift : float
        Redshift at which the filters are projected (0 for the rest-band twin).
    n_subbands : int
        Number of chunks K per filter.

    Returns
    -------
    phi : ndarray, shape (n_points, n_filter, K)
        Filter integral restricted to each chunk; sums over K to the whole
        band's ``lnu_filter_integral`` [erg/s/Hz].
    lam_rest : ndarray, shape (n_points, n_filter, K)
        Flux-weighted rest-frame wavelength of the SED inside each chunk
        [Angstrom]; the chunk filter-mass centroid where the chunk is empty.

    Notes
    -----
    **JIT-compatible**: no; build-time numpy precompute.
    """
    sed = np.asarray(sed_nodes_rest, dtype=np.float64)
    wave_obs = np.asarray(wave_rest, dtype=np.float64) * (1.0 + redshift)
    n_points = sed.shape[0]
    phi = np.zeros((n_points, len(filters), n_subbands))
    lam_rest = np.zeros_like(phi)
    for f_idx, (fw, ft) in enumerate(filters):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        grid = np.sort(np.concatenate([wave_obs, fw_np]))
        # Same clip as preintegrate_grid: keep one node past each filter edge.
        lo = max(np.searchsorted(grid, fw_np[0]) - 1, 0)
        hi = min(np.searchsorted(grid, fw_np[-1], side="right") + 1, grid.size)
        grid = grid[lo:hi]
        trans = np.interp(grid, fw_np, ft_np, left=0.0, right=0.0)
        tw_grid = trans * filter_weight_np(grid, FilterConvention.BESSELL)
        denom = _np_trapezoid(tw_grid, grid)
        eff_obs = _np_trapezoid(tw_grid * grid, grid) / np.maximum(
            denom, representable_denominator(1e-30)
        )
        integrand = _vectorized_interp(grid, wave_obs, sed) * tw_grid[None, :]
        phi[:, f_idx, :], nodes = subband_quadrature(
            grid, tw_grid, integrand, denom, n_subbands, float(eff_obs)
        )
        lam_rest[:, f_idx, :] = nodes / (1.0 + redshift)
    return phi, lam_rest


def _nebular_screen_for(
    dust,
    params: Mapping[str, jnp.ndarray],
    wave: jnp.ndarray,
    neb_weights: Sequence[float] | None = None,
) -> jnp.ndarray:
    """Nebular-screen transmission of ``dust`` at ``wave``, as ``apply`` resolves it.

    Uses the component's ``nebular_screen_transmission`` when it declares one;
    otherwise reproduces the resolution ``apply`` performs for the nebular
    continuum (two-component: ``config.nebular_screen`` through the resolved
    birth-cloud and diffuse laws; single screen: ``exp(-tau_v k)``).

    Parameters
    ----------
    dust : DustSEDComponent or DustAttenuationSEDComponent
        The chain's attenuator.
    params : Mapping
        Fully resolved parameters, including the dust optical depths.
    wave : ndarray, shape (n_wave,)
        Rest-frame wavelengths [Angstrom].
    neb_weights : sequence of float, optional
        Age-interval weights of an age-split attenuator's nebular screen (the
        table bakes the pure screens, ``(1, 0)`` and ``(0, 1)``, and the runtime
        mixes them); ignored by a single screen.

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission in ``[0, 1]``.

    Notes
    -----
    **JIT-compatible**: yes, ``jnp`` primitives plus static registry lookups.
    """
    method = getattr(dust, "nebular_screen_transmission", None)
    if method is not None:
        if neb_weights is None:
            return method(params, wave)  # a single screen has no age intervals to weigh
        return method(params, wave, jnp.asarray(neb_weights))

    from tengri.components.dust.component import DustAttenuationSEDComponent

    if isinstance(dust, DustAttenuationSEDComponent):
        k = dust._curve(params)(wave)
        return jnp.exp(-jnp.asarray(params["dust_tau_v"]) * k)

    from tengri.components.dust._apply import (
        merge_neb_screen_live_overrides,
        resolve_bc_diff_law_params,
    )
    from tengri.components.dust.laws._registry import select_law_kwargs

    cfg = dust.config
    bc_law_params, diff_law_params = resolve_bc_diff_law_params(
        params,
        dict(cfg.bc_law_overrides),
        dict(cfg.diff_law_overrides),
        cfg.live_shape_params,
        bc_law=cfg.law_bc,
        diff_law=cfg.law_diff,
        redshift=params.get("redshift"),
    )
    neb_law = cfg.law_neb or cfg.law_bc
    neb_overrides = merge_neb_screen_live_overrides(
        params, cfg.neb_law_overrides, cfg.live_shape_params
    )
    neb_bc_params = {
        k: jnp.asarray(v)
        for k, v in select_law_kwargs(neb_law, {**bc_law_params, **neb_overrides}).items()
    }
    diff_kw = {k: jnp.asarray(v) for k, v in diff_law_params.items()}
    return dust._line_transmission(
        params, jnp.asarray(wave), neb_law, neb_bc_params, diff_kw, jnp.asarray(neb_weights)
    )


def _tau_params(dust, ref_params: Mapping, tau_a: float, tau_b: float) -> dict:
    """``ref_params`` with the dust optical depths at grid node ``(tau_a, tau_b)``."""
    from tengri.components.dust.component import DustAttenuationSEDComponent

    q = dict(ref_params)
    if isinstance(dust, DustAttenuationSEDComponent):
        q["dust_tau_v"] = float(tau_b)
    else:
        q["dust_tau_bc"], q["dust_tau_diff"] = float(tau_a), float(tau_b)
    return q


def _nebular_eb_channel(
    sed_nodes_rest: np.ndarray,
    neg_log_qh: np.ndarray,
    wave_rest: np.ndarray,
    dust,
    ref_params: Mapping,
    eb_tau_grids: tuple[Sequence[float], Sequence[float]],
) -> np.ndarray:
    r"""LyC-masked absorbed nebular luminosity per unit Q_H on the tau grid.

    Parameters
    ----------
    sed_nodes_rest : ndarray, shape (n_points, n_wave)
        Rest-frame nebular L_nu of every grid node [erg/s/Hz].
    neg_log_qh : ndarray, shape (n_points,)
        ``-log10 Q_H`` of every node, as the grid build applies it.
    wave_rest : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    dust : DustSEDComponent or DustAttenuationSEDComponent
        The attenuator whose nebular screen is applied.
    ref_params : Mapping
        Resolved reference parameters (law shapes, redshift).
    eb_tau_grids : tuple
        ``(tau_a_grid, tau_b_grid)`` of the stellar energy-balance LUT.

    Returns
    -------
    ndarray, shape (n_points, K, n_tau_a, n_tau_b)
        SIGNED absorbed luminosity per unit nion [erg/s per (photon/s)], one
        channel ``K`` per pure nebular screen (1 for a single screen, 2 for an
        age-split attenuator: the young and the old screen),
        positively oriented (+1 for a net absorber); exactly 0 where the screen
        is unity.

    Notes
    -----
    **JIT-compatible**: no; build-time precompute (one ``lax.map`` per node
    batch to bound memory).
    """
    tau_a = np.asarray(eb_tau_grids[0], dtype=float)
    tau_b = np.asarray(eb_tau_grids[1], dtype=float)
    wave = jnp.asarray(wave_rest)
    nu = C_AA / wave
    cutoff = _lyc_cutoff_for(dust)
    # One channel per pure screen of an age-split attenuator (the absorbed energy
    # is linear in the nebular screen, so the runtime mixes the channels by the
    # ionizing-luminosity weights); a single screen has the one.
    channels = getattr(dust, "nebular_weight_channels", None) or (None,)
    t_stacks = [
        jnp.stack(
            [
                _nebular_screen_for(dust, _tau_params(dust, ref_params, ta, tb), wave, weights)
                for ta, tb in itertools.product(tau_a, tau_b)
            ]
        )
        for weights in channels
    ]  # each (n_a * n_b, n_wave)

    @jax.jit
    def _node(sed, t_stack):
        def one(t):
            log_abs, sign = bolometric_absorbed_log10(
                sed, sed * t, nu, wave=wave, lyman_cutoff_aa=cutoff
            )
            return log_abs, sign

        return jax.lax.map(one, t_stack)

    channel_tables = []
    for t_stack in t_stacks:
        results = np.stack(
            [
                np.asarray(_node(jnp.asarray(row), t_stack), dtype=np.float64)
                for row in np.asarray(sed_nodes_rest)
            ]
        )  # (n_points, 2, n_a * n_b)
        log_abs = results[:, 0, :]
        sign = results[:, 1, :]
        finite = np.isfinite(log_abs)
        exponent = np.where(
            finite, log_abs + np.asarray(neg_log_qh, dtype=np.float64)[:, None], 0.0
        )
        eb = np.where(finite, sign * 10.0**exponent, 0.0)
        channel_tables.append(eb.reshape(eb.shape[0], tau_a.size, tau_b.size))
    return np.stack(channel_tables, axis=1)  # (n_points, K, n_a, n_b)


_SUBBAND_NODE_CHUNK = 128
_CONSERVATION_FLOOR = 1e-12


def _nebular_filters(model) -> tuple | None:
    """Observed-frame ``(wave, trans)`` tables the nebular component cached, or None."""
    from tengri.components.nebular.component import NebularSEDComponent

    chain = model._cached_component_chain or model._build_component_chain()
    for comp in chain:
        if isinstance(comp, NebularSEDComponent):
            state = comp._state
            if state is None or state.filter_waves is None or state.filter_trans is None:
                return None
            return tuple(
                (np.asarray(fw, dtype=np.float64), np.asarray(ft, dtype=np.float64))
                for fw, ft in zip(state.filter_waves, state.filter_trans, strict=True)
            )
    return None


def _chunked_subbands(sed_all, neg_log_qh, wave_rest, filters, redshift, n_subbands):
    """:func:`_nebular_subband_channels` in node batches, scaled to per-Q_H (float64)."""
    phis, lams = [], []
    for i in range(0, sed_all.shape[0], _SUBBAND_NODE_CHUNK):
        sl = slice(i, i + _SUBBAND_NODE_CHUNK)
        phi, lam = _nebular_subband_channels(sed_all[sl], wave_rest, filters, redshift, n_subbands)
        phis.append(phi * 10.0 ** neg_log_qh[sl, None, None])
        lams.append(lam)
    return np.concatenate(phis, axis=0), np.concatenate(lams, axis=0)


def _check_conservation(parts: np.ndarray, whole, label: str, n_nodes: int) -> float:
    """Raise unless the sub-band sum matches the whole band; return the worst residual.

    A trapezoid over ``n_nodes`` samples in the precision of ``whole`` accumulates
    about ``eps * sqrt(n_nodes)`` relative error; the bound allows eight times that,
    which is 1e-8 in float64 (the floor) and about 1e-4 in float32 on the SSP
    wavelength grid, while a genuine divergence of the two quadratures is 1e-3 or
    worse.
    """
    whole_np = np.asarray(whole, dtype=np.float64)
    eps = float(np.finfo(np.asarray(whole).dtype).eps)
    tol = max(1e-8, 8.0 * eps * math.sqrt(n_nodes))
    floor = _CONSERVATION_FLOOR * max(float(np.max(np.abs(whole_np))), 1e-300)
    rel = np.abs(parts.sum(axis=-1) - whole_np) / np.maximum(np.abs(whole_np), floor)
    worst = float(np.max(rel))
    if not worst <= tol:
        raise RuntimeError(
            f"nebular fast grid: the {label} sub-band channels do not sum to the whole-band "
            f"channel (worst relative residual {worst:.3e} > {tol:.1e}); the sub-band "
            "quadrature and the exact filter integral have diverged."
        )
    return worst


def _log_channel(per_qh: np.ndarray, grid_shape: tuple) -> jnp.ndarray:
    return jnp.asarray(
        np.log10(np.maximum(per_qh, representable_floor(1e-300))).reshape(
            *grid_shape, *per_qh.shape[1:]
        )
    )


def _build_dust_channels(
    model,
    *,
    grid_shape: tuple,
    phot_all,
    rest_all,
    sed_all,
    neg_log_qh_all,
    wave_rest,
    ref_params: Mapping,
    dust_component,
    eb_tau_grids,
    n_subbands,
) -> dict:
    """The ``NebularGridTable`` dust-channel fields, or ``{}`` where none is requested.

    Parameters
    ----------
    model : SEDModel
        The reference model (source of the cached filter tables).
    grid_shape : tuple
        ``tuple(len(a) for a in axes)``.
    phot_all, rest_all : ndarray, shape (n_points, n_filter)
        Whole-band per-Q_H photometry (observed and rest-band twins) the
        sub-band channels must sum to.
    sed_all : ndarray, shape (n_points, n_wave) or None
        Per-node materialized nebular SED [erg/s/Hz].
    neg_log_qh_all : ndarray, shape (n_points,)
        ``-log10 Q_H`` per node.
    wave_rest : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    ref_params : Mapping
        Resolved reference parameters.
    dust_component, eb_tau_grids, n_subbands
        As in :func:`precompute_nebular_grid`.

    Returns
    -------
    dict
        Field name -> array for :class:`NebularGridTable`; empty when ``sed_all``
        is None.

    Raises
    ------
    RuntimeError
        When the sub-band sum disagrees with the whole-band channel.

    Notes
    -----
    **JIT-compatible**: no; build-time precompute.
    """
    if sed_all is None:
        return {}
    sed = np.asarray(sed_all)
    nlq = np.asarray(neg_log_qh_all, dtype=np.float64)
    out: dict = {}
    filters = _nebular_filters(model) if n_subbands is not None and phot_all is not None else None
    if filters is not None:
        k = int(n_subbands)
        ref_z = float(
            require_redshift(
                ref_params, "components.nebular.nebular_grid_dust_build._build_dust_channels"
            )
        )
        n_wave_nodes = int(np.asarray(wave_rest).size)
        phi, lam = _chunked_subbands(sed, nlq, wave_rest, filters, ref_z, k)
        _check_conservation(phi, phot_all, "observed-band", n_nodes=n_wave_nodes)
        phi_r, lam_r = _chunked_subbands(sed, nlq, wave_rest, filters, 0.0, k)
        _check_conservation(phi_r, rest_all, "rest-band", n_nodes=n_wave_nodes)
        out.update(
            log_phot_subband_per_qh=_log_channel(phi, grid_shape),
            phot_subband_waves_rest=jnp.asarray(lam.reshape(*grid_shape, *lam.shape[1:])),
            log_restband_subband_per_qh=_log_channel(phi_r, grid_shape),
            restband_subband_waves_rest=jnp.asarray(lam_r.reshape(*grid_shape, *lam_r.shape[1:])),
        )
    if dust_component is not None and eb_tau_grids is not None:
        eb = _nebular_eb_channel(sed, nlq, wave_rest, dust_component, ref_params, eb_tau_grids)
        out.update(
            eb_absorbed_per_qh=jnp.asarray(eb.reshape(*grid_shape, *eb.shape[1:])),
            eb_tau_a_grid=jnp.asarray(eb_tau_grids[0]),
            eb_tau_b_grid=jnp.asarray(eb_tau_grids[1]),
            lyc_in_energy_balance=_lyc_cutoff_for(dust_component) is None,
        )
    return out
