# SPDX-License-Identifier: BSD-3-Clause
"""Age-resolved band table for the CLOUDY nebular backend (#2324).

The per-call photometric path projects the full nebular SED through every
filter on every evaluation. Band projection is linear in the SED, so the
same answer is a contraction over SSP ages of tabulated per-age quantities.

Continuum: the band of each (age, Z_gas, logU) node is tabulated (log10),
interpolated in (Z_gas, logU) and contracted over ages.

Lines: the per-line luminosity of each (age, Z_gas, logU) node is tabulated
(log10), interpolated per line, summed over ages, and projected onto the
filters with a fixed band coefficient per line. Band-level tabulation of the
line sum is not used: lines with different (Z, U) slopes make the log of the
sum differ from the sum of the logs by more than 1e-3.

Band coefficients depend on the velocity width the lines are rendered at, so
a table holds one fixed ``line_sigma_kms``. A caller whose width is free must
not use it.

Units: band values are in erg/s/Hz (``_LSUN_ERG`` applied), matching
``lnu_filter_integral`` on ``predict_nebular_sed``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.lyc import lyc_shares
from tengri.components.nebular._constants import _LSUN_ERG
from tengri.components.nebular._recombination_coeffs import lyc_dust_escape_factor
from tengri.components.nebular._shared import (
    _interp_index_weight,
    apply_lya_escape,
    interp_continuum_with_freefree_tail,
    render_nebular_lines,
)
from tengri.observation.photometry import lnu_filter_integral
from tengri.utils.grid_interp import interp_nd_pchip
from tengri.utils.scale import pow10

#: log10 floor for a band that is exactly zero (no filter overlap).
LOG_FLOOR: float = -300.0

#: Rest-frame wavelength of Ly-alpha [Angstrom]; the line nearest it gets
#: the independent ``neb_fesc_lya`` factor.
_LYA_WAVE: float = 1215.67


class CloudyBandTable(NamedTuple):
    """Per-age tables on the CLOUDY grid's own (Z_gas, logU) nodes.

    Attributes
    ----------
    log_age_yr : ndarray, shape (n_young,)
        log10(age/yr) of the young SSP bins the table covers.
    cont_z, cont_u : ndarray
        Continuum node axes (absolute log10 Z_gas, log10 U).
    line_z, line_u : ndarray
        Line node axes (absolute log10 Z_gas, log10 U).
    log_cont : ndarray, shape (2, n_young, n_cont_z, n_cont_u, n_filt)
        log10 continuum band per unit Q_H and unit k-factor. Axis 0 is the
        frame: 0 = observed (redshift ``redshift``), 1 = rest (redshift 0).
    log_lines : ndarray, shape (n_young, n_line_z, n_line_u, n_lines)
        log10 per-line luminosity per unit Q_H and unit k-factor [Lsun].
    line_coeff : ndarray, shape (n_lines, 2, n_filt)
        Band of a unit-luminosity line through each filter, both frames
        [erg/s/Hz per Lsun].
    line_wavelengths : ndarray, shape (n_lines,)
        Rest-frame line wavelengths [Angstrom]; the Ly-alpha escape is applied
        through ``apply_lya_escape`` (in ``tengri.components.nebular._shared``) on
        these, the same helper the per-call backend uses.
    line_sigma_kms : float
        Velocity width the line coefficients were rendered at.
    redshift : float
        Redshift of the observed frame.
    """

    log_age_yr: np.ndarray
    cont_z: np.ndarray
    cont_u: np.ndarray
    line_z: np.ndarray
    line_u: np.ndarray
    log_cont: np.ndarray
    log_lines: np.ndarray
    line_coeff: np.ndarray
    line_wavelengths: np.ndarray
    line_sigma_kms: float
    redshift: float


def _band_row(sed, wave, filter_waves, filter_trans, redshift):
    """Filter-integrated L_nu of one SED through every filter, shape (n_filt,)."""
    return jnp.stack(
        [
            lnu_filter_integral(sed, wave, fw, ft, redshift=redshift)
            for fw, ft in zip(filter_waves, filter_trans, strict=True)
        ]
    )


def _band_both_frames(sed, wave, filter_waves, filter_trans, redshift):
    """Observed (index 0) and rest (index 1) band rows, shape (2, n_filt)."""
    obs = _band_row(sed, wave, filter_waves, filter_trans, redshift)
    rest = _band_row(sed, wave, filter_waves, filter_trans, 0.0)
    return jnp.stack([obs, rest])


def _line_band_coefficients(
    wave, line_wavelengths, sigma_kms, filter_waves, filter_trans, redshift
) -> np.ndarray:
    """Band of one unit-luminosity line per line, shape (n_lines, 2, n_filt).

    The rendered profile is linear in the luminosity, so the band of any line
    mix is the luminosity-weighted sum of these rows.
    """
    n_lines = line_wavelengths.shape[0]
    eye = jnp.eye(n_lines)

    def _one(lum_vec):
        sed = render_nebular_lines(line_wavelengths, lum_vec, wave, 0.0, sigma_kms) * _LSUN_ERG
        return _band_both_frames(sed, wave, filter_waves, filter_trans, redshift)

    return np.asarray(jax.vmap(_one)(eye))


def _grid2d(z_axis: np.ndarray, u_axis: np.ndarray) -> jnp.ndarray:
    """All (Z, U) node pairs, shape (n_z * n_u, 2)."""
    zz, uu = np.meshgrid(z_axis, u_axis, indexing="ij")
    return jnp.asarray(np.stack([zz.ravel(), uu.ravel()], axis=-1))


def _to_log(arr: np.ndarray) -> np.ndarray:
    """log10 with a finite floor, so an exactly zero band stays representable."""
    with np.errstate(divide="ignore"):
        logged = np.log10(np.maximum(arr, 10.0**LOG_FLOOR))
    return np.maximum(logged, LOG_FLOOR)


def build_cloudy_band_table(
    backend,
    wave,
    filter_waves,
    filter_trans,
    redshift: float,
    line_sigma_kms: float,
) -> CloudyBandTable:
    """Tabulate the CLOUDY nebular band projection per age and node.

    Parameters
    ----------
    backend : CloudyGridBackend
        Backend whose grid and young-age bins are used.
    wave : array_like, shape (n_wave,)
        SED wavelength grid the per-call path projects (``state.wave``).
    filter_waves, filter_trans : list of array_like
        Filter curves, one per band.
    redshift : float
        Source redshift for the observed frame.
    line_sigma_kms : float
        Fixed line velocity width the lines are rendered at.

    Returns
    -------
    CloudyBandTable
    """
    g = backend.grid
    young = np.asarray(backend._young_idx)
    scale = float(backend._log_qh_scale)
    wave = jnp.asarray(wave)
    cont_wave = jnp.asarray(g.cont_wavelength)
    line_wave = np.asarray(g.line_wavelengths)

    interp_cont = backend._make_interp_fn(
        g.cont_luminosity,
        g.cont_log_met,
        g.cont_log_age,
        g.cont_log_U,
        edges_z=getattr(backend, "_edges_z_cont", None),
        edges_age=getattr(backend, "_edges_age_cont", None),
        edges_u=getattr(backend, "_edges_u_cont", None),
    )
    interp_line = backend._make_interp_fn(
        g.line_luminosity,
        g.line_log_met,
        g.line_log_age,
        g.line_log_U,
        edges_z=getattr(backend, "_edges_z_line", None),
        edges_age=getattr(backend, "_edges_age_line", None),
        edges_u=getattr(backend, "_edges_u_line", None),
    )
    log_age = np.asarray(backend._qh_log_age)[young]

    def _cont_row(p, age):
        """Continuum band (2, n_filt) at one (Z, U) node for one age."""
        lum = pow10(interp_cont(p[0], age, p[1]) + scale)
        sed = interp_continuum_with_freefree_tail(wave, cont_wave, lum) * _LSUN_ERG
        return _band_both_frames(sed, wave, filter_waves, filter_trans, redshift)

    def _line_lum(p, age):
        """Unit-Q_H per-line luminosity (n_lines,) at one (Z, U) node for one age."""
        return pow10(interp_line(p[0], age, p[1]) + scale)

    cz, cu = np.asarray(g.cont_log_met), np.asarray(g.cont_log_U)
    lz, lu = np.asarray(g.line_log_met), np.asarray(g.line_log_U)
    cont_nodes = _grid2d(cz, cu)
    line_nodes = _grid2d(lz, lu)
    cont_block = jax.jit(jax.vmap(_cont_row, in_axes=(0, None)))
    line_block = jax.jit(jax.vmap(_line_lum, in_axes=(0, None)))

    cont_tab, line_tab = [], []
    for age in log_age:
        cont_tab.append(np.asarray(cont_block(cont_nodes, age)).reshape(len(cz), len(cu), 2, -1))
        line_tab.append(np.asarray(line_block(line_nodes, age)).reshape(len(lz), len(lu), -1))

    coeff = _line_band_coefficients(
        wave, jnp.asarray(line_wave), line_sigma_kms, filter_waves, filter_trans, redshift
    )
    # (n_young, n_cont_z, n_cont_u, 2, n_filt) -> (2, n_young, n_cont_z, n_cont_u, n_filt)
    log_cont = np.moveaxis(_to_log(np.stack(cont_tab)), 3, 0)
    return CloudyBandTable(
        log_age_yr=log_age,
        cont_z=cz,
        cont_u=cu,
        line_z=lz,
        line_u=lu,
        # Device arrays: the table rides in the jitted template data every call,
        # and a host array there is re-transferred each time (~5 MB here).
        log_cont=jnp.asarray(log_cont),
        log_lines=jnp.asarray(_to_log(np.stack(line_tab))),
        line_coeff=jnp.asarray(coeff),
        line_wavelengths=np.asarray(line_wave),
        line_sigma_kms=float(line_sigma_kms),
        redshift=float(redshift),
    )


def _bilinear_log(log_tab, z_axis, u_axis, z, u, axis: int):
    """Bilinear interpolation of log10 values in (Z, U), then pow10.

    The Z axis of ``log_tab`` is ``axis`` and the U axis is ``axis + 1``.
    Query points are clipped to the node range, as ``_interp_index_weight``
    does. Trailing and leading dimensions are carried through.

    Scheme: linear weights in (log Z, log U), the (Z, U) restriction of the
    ``grid_interp="linear"`` path of
    ``CloudyGridBackend._make_interp_fn`` (``_trilinear_interp`` in
    ``cloudy_grid.py``), which interpolates log10 values. The table holds the
    band of each node (line luminosities for lines), not the luminosity behind
    the continuum band, so the continuum interpolates log10(band) where the
    backend interpolates log10(luminosity); the difference is inside the
    band-table accuracy bar. With ``grid_interp="triweight"`` the age axis
    follows the backend but this (Z, U) contraction stays linear, so the table
    does not reproduce that mode.
    """
    iz, wz = _interp_index_weight(z, z_axis)
    iu, wu = _interp_index_weight(u, u_axis)

    def _at(iz_, iu_):
        return jnp.take(jnp.take(log_tab, iz_, axis=axis), iu_, axis=axis)

    logv = (
        (1 - wz) * (1 - wu) * _at(iz, iu)
        + wz * (1 - wu) * _at(iz + 1, iu)
        + (1 - wz) * wu * _at(iz, iu + 1)
        + wz * wu * _at(iz + 1, iu + 1)
    )
    return pow10(logv)


def _pchip_log(log_tab, z_axis, u_axis, z, u, axis: int):
    """Node-exact PCHIP of log10 values in (Z, U), then pow10.

    The same ``interp_nd_pchip`` the CLOUDY backend uses for its ``"pchip"``
    default (``CloudyGridBackend._make_interp_fn``), with the Z axis first and
    the U axis second, clipped at the node range (no extrapolation).
    """
    moved = jnp.moveaxis(log_tab, (axis, axis + 1), (0, 1))
    return pow10(interp_nd_pchip(moved, (z_axis, u_axis), (z, u)))


def _read_log_zu(log_tab, z_axis, u_axis, z, u, axis: int, grid_interp: str):
    """Interpolate a log table in (Z, U) with the backend's scheme."""
    if grid_interp == "pchip":
        return _pchip_log(log_tab, z_axis, u_axis, z, u, axis)
    if grid_interp == "linear":
        return _bilinear_log(log_tab, z_axis, u_axis, z, u, axis)
    raise ValueError(
        f"the CLOUDY band table reproduces grid_interp 'pchip' or 'linear', not {grid_interp!r}"
    )


def contract_cloudy_band_table(
    table: CloudyBandTable,
    age_weights,
    neb_logz_gas_abs,
    neb_logu,
    fesc_lya=0.0,
    grid_interp: str = "pchip",
):
    """Contract the band table over ages at one (Z_gas, logU) point.

    Parameters
    ----------
    table : CloudyBandTable
        Built by :func:`build_cloudy_band_table`.
    age_weights : array, shape (n_young,)
        ``w_i * Q_H(log_z, age_i) * k_factor`` for the young bins.
    neb_logz_gas_abs : float
        Absolute gas-phase log10(Z_gas).
    neb_logu : float
        log10 U.
    fesc_lya : float
        Ly-alpha escape fraction; scales the Ly-alpha line alone.
    grid_interp : {"pchip", "linear"}
        The interpolation of the CLOUDY backend the table was built from.

    Returns
    -------
    array, shape (2, n_filt)
        Observed (0) and rest (1) band L_nu [erg/s/Hz].
    """
    aw = jnp.asarray(age_weights)
    cont = _read_log_zu(
        jnp.asarray(table.log_cont),
        jnp.asarray(table.cont_z),
        jnp.asarray(table.cont_u),
        neb_logz_gas_abs,
        neb_logu,
        axis=2,
        grid_interp=grid_interp,
    )  # (2, n_young, n_filt)
    cont_band = jnp.einsum("i,cif->cf", aw, cont)

    lines = _read_log_zu(
        jnp.asarray(table.log_lines),
        jnp.asarray(table.line_z),
        jnp.asarray(table.line_u),
        neb_logz_gas_abs,
        neb_logu,
        axis=1,
        grid_interp=grid_interp,
    )  # (n_young, n_lines)
    lum = jnp.einsum("i,ij->j", aw, lines)
    lum = apply_lya_escape(lum, jnp.asarray(table.line_wavelengths), fesc_lya)
    line_band = jnp.einsum("j,jcf->cf", lum, jnp.asarray(table.line_coeff))
    return cont_band + line_band


def table_covers(table: CloudyBandTable, z_range, u_range) -> bool:
    """Whether the table's (Z_gas, logU) axes contain both closed ranges.

    Parameters
    ----------
    table : CloudyBandTable
    z_range, u_range : (float, float)
        Absolute log10 Z_gas and log10 U ranges the model can reach.

    Returns
    -------
    bool
        True when every node of both the continuum and the line axes spans
        the requested range. Outside the axes the per-call path clips while
        the table would too, so a model that reaches past them must decline.
    """
    z_axis = np.concatenate([table.cont_z, table.line_z])
    u_axis = np.concatenate([table.cont_u, table.line_u])
    return bool(
        z_axis.min() <= z_range[0]
        and z_axis.max() >= z_range[1]
        and u_axis.min() <= u_range[0]
        and u_axis.max() >= u_range[1]
    )


def runtime_age_weights(backend, ssp_log_ages_yr, ssp_weights, log_z, neb_fesc, neb_fdust_frac):
    """Per-young-age weights ``w_i * Q_H(log_z, age_i) * k_factor``, shape (n_young,).

    Mirrors the age weighting of ``CloudyGridBackend.predict_nebular_split``
    (same young-bin selection, same Q_H lookup, same ``lyc_shares`` k-factor),
    so the table contraction is the per-call sum regrouped.
    """
    young = np.asarray(backend._young_idx)
    ages = jnp.asarray(ssp_log_ages_yr)[young]
    weights = jnp.asarray(ssp_weights)[young]
    qh = jax.vmap(lambda a: backend._get_qh_at(log_z, a))(ages)
    _, f_dust, _ = lyc_shares(neb_fesc, neb_fdust_frac)
    k_factor = lyc_dust_escape_factor(neb_fesc, f_dust)
    return weights * qh * k_factor


def table_band_photometry(
    table: CloudyBandTable,
    backend,
    ssp_log_ages_yr,
    ssp_weights,
    log_z,
    log_z_gas,
    neb_logu,
    neb_fesc,
    neb_fdust_frac,
    neb_fesc_lya,
):
    """Observed and rest band L_nu of the CLOUDY nebular emission, shape (2, n_filt).

    The runtime entry point: age weights from the SSP layout and Q_H, then the
    contraction of :func:`contract_cloudy_band_table`.
    """
    aw = runtime_age_weights(
        backend, ssp_log_ages_yr, ssp_weights, log_z, neb_fesc, neb_fdust_frac
    )
    return contract_cloudy_band_table(
        table, aw, log_z_gas, neb_logu, neb_fesc_lya, grid_interp=backend._grid_interp
    )
