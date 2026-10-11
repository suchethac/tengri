# SPDX-License-Identifier: BSD-3-Clause
"""Bolometric normalization of the GRAHSP AGN model, one component at a time.

GRAHSP is linear in ``l5100``. The published SED is ``l5100 * shape``, and
``l5100`` is chosen so the intrinsic bolometric equals the requested power. That
bolometric is a sum of per-component integrals. Each is taken on the component's
own grid, never on the caller's wavelength grid:

* BBB side, above the Lyman limit: the disc (SBPL power law on its own log grid,
  or the Netzer template on its native nodes), the Balmer continuum (its own grid
  up to the Balmer edge), the Gaussian lines (closed form, error function) and
  the FeII forest (template nodes, exact for its linear interpolation).
* Torus side, over all wavelengths: the torus (log-Gaussian pair on its own log
  grid, or the MN12 template on its native nodes) and the silicate feature
  (Gaussians on their own grid, or the template's native nodes).

Template integrals use ``native_bolometric_nu`` in ``tengri.components.agn._template_grid``,
the power-law interpolant that :func:`~tengri.utils.grid_interp.resample_template`
builds, so the normalization equals the integral of the published template.
Every function here is evaluated at unit ``l5100``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array

from tengri.components.agn._template_grid import native_bolometric_nu
from tengri.components.agn.grahsp.balmer import _BALMER_EDGE_NM, balmer_continuum
from tengri.components.agn.grahsp.bbb import floor_disc_xray, sbpl_bbb
from tengri.components.agn.grahsp.bolometric import LYMAN_LIMIT_NM
from tengri.components.agn.grahsp.disc import select_disc_model
from tengri.components.agn.grahsp.lines import feii_forest_power, gaussian_lines_power_above
from tengri.components.agn.grahsp.torus import (
    mn12_native_shape,
    si_feature,
    torus_dust_continuum,
)
from tengri.utils.grid_interp import resample_template
from tengri.utils.physics_constants import C_AA

if TYPE_CHECKING:  # pragma: no cover
    from tengri.components.agn.grahsp.model import GRAHSPParams
    from tengri.components.agn.grahsp.templates import GRAHSPTemplates

__all__ = [
    "BALMER_NORM_GRID_NM",
    "BBB_NORM_GRID_NM",
    "SI_NORM_GRID_NM",
    "TORUS_NORM_GRID_NM",
    "bbb_continuum_power_nm",
    "bbb_side_unit_power",
    "native_power_nm",
    "torus_side_unit_power",
    "unit_bolometric_power",
]

BBB_NORM_GRID_NM: np.ndarray = np.geomspace(LYMAN_LIMIT_NM, 1.0e10, 1024)
"""SBPL disc grid [nm], log-spaced from the Lyman limit to 1e10 nm.

Trapezoid in ln(lambda). Relative error against a 4e5-node reference is 3e-6 at
the default bend width and 6e-6 at width 0.3; the tail beyond 1e10 nm is below 1e-5.
"""

BALMER_NORM_GRID_NM: np.ndarray = np.geomspace(LYMAN_LIMIT_NM, _BALMER_EDGE_NM, 512)
"""Balmer continuum grid [nm], ending exactly on the Balmer edge (the shape is cut
there). Relative error 4e-5 at 512 nodes."""

TORUS_NORM_GRID_NM: np.ndarray = np.geomspace(1.0, 1.0e7, 128)
"""Gaussian torus grid [nm], 1e-3 to 1e4 um. The log-Gaussian pair is below 1e-8 at
both ends. Exact to machine precision at 128 nodes."""

SI_NORM_GRID_NM: np.ndarray = np.linspace(2.0e3, 3.0e4, 256)
"""Silicate-feature grid [nm], 2 to 30 um. Spacing 110 nm against a 1 um feature."""


def _log_trapezoid_weights(grid_nm: np.ndarray) -> np.ndarray:
    """Weights ``w`` with ``sum(w * f)`` = integral of ``f`` d(lambda), trapezoid in ln."""
    half = 0.5 * np.diff(np.log(grid_nm))
    weights = np.zeros_like(grid_nm)
    weights[:-1] += half
    weights[1:] += half
    return weights * grid_nm


def _linear_trapezoid_weights(grid_nm: np.ndarray) -> np.ndarray:
    """Weights ``w`` with ``sum(w * f)`` = trapezoid of ``f`` over a linear grid."""
    half = 0.5 * np.diff(grid_nm)
    weights = np.zeros_like(grid_nm)
    weights[:-1] += half
    weights[1:] += half
    return weights


_W_BBB = _log_trapezoid_weights(BBB_NORM_GRID_NM)
_W_BALMER = _log_trapezoid_weights(BALMER_NORM_GRID_NM)
_W_TORUS = _log_trapezoid_weights(TORUS_NORM_GRID_NM)
_W_SI = _linear_trapezoid_weights(SI_NORM_GRID_NM)


def native_power_nm(wave_nm: Array, l_lambda_nm: Array) -> Array:
    r"""Integral of a tabulated :math:`L_\lambda` over its own nodes [erg/s].

    Parameters
    ----------
    wave_nm : array_like, shape (n,)
        Template wavelength nodes [nm], ascending.
    l_lambda_nm : array_like, shape (n,)
        Template :math:`L_\lambda` on those nodes [erg/s/nm; any scale, the
        result carries the same scale].

    Returns
    -------
    ndarray, shape ()
        :math:`\int L_\lambda\,\mathrm{d}\lambda`, taken with the power law the
        template is resampled with (see :func:`native_bolometric_nu`).

    Notes
    -----
    :func:`native_bolometric_nu` works in Angstrom and in :math:`L_\nu`; this
    converts once, so its nm inputs are unchanged.
    """
    wave_aa = jnp.asarray(wave_nm) * 10.0
    l_lambda_aa = jnp.asarray(l_lambda_nm) * 0.1
    return native_bolometric_nu(l_lambda_aa * wave_aa**2 / C_AA, wave_aa)


def bbb_continuum_power_nm(
    uvslope: float,
    plslope: float,
    plbendloc_nm: float,
    plbendwidth: float,
    cutoff_nm: float,
) -> Array:
    """Unit-``l5100`` SBPL disc power above the Lyman limit [erg/s].

    Used by the GRAHSP disc block (BBB-only normalization) and by the full model.
    """
    nodes = jnp.asarray(BBB_NORM_GRID_NM)
    l_lambda = floor_disc_xray(
        nodes,
        sbpl_bbb(
            wave_nm=nodes,
            l5100=1.0,
            uvslope=uvslope,
            plslope=plslope,
            plbendloc_nm=plbendloc_nm,
            plbendwidth=plbendwidth,
            cutoff_nm=cutoff_nm,
        ),
    )
    return jnp.sum(jnp.asarray(_W_BBB) * l_lambda)


def _netzer_disc_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """Netzer disc template power above the Lyman limit [erg/s], native nodes.

    The template starts below 91.2 nm, so its first segment is cut at the
    Lyman limit: the template value there is prepended as a node, which keeps
    the power-law interpolant exact on the retained part.
    """
    idx = select_disc_model(
        templates.disc_m,
        templates.disc_a,
        templates.disc_mdot,
        m=params.disc_m,
        a=params.disc_a,
        mdot=params.disc_mdot,
    )
    wave = np.asarray(templates.disc_wave_nm, dtype=float)
    lumin = jnp.asarray(templates.disc_lumin[idx])
    start = int(np.searchsorted(wave, LYMAN_LIMIT_NM))
    edge_value = resample_template(jnp.asarray([LYMAN_LIMIT_NM]), wave, lumin, left=0.0, right=0.0)
    nodes = np.concatenate([[LYMAN_LIMIT_NM], wave[start:]])
    values = jnp.concatenate([edge_value, lumin[start:]])
    return native_power_nm(nodes, values)


def _balmer_power(params: GRAHSPParams) -> Array:
    """Balmer continuum power above the Lyman limit [erg/s] (type-1 only)."""
    nodes = jnp.asarray(BALMER_NORM_GRID_NM)
    sed = balmer_continuum(
        wave_nm=nodes,
        l5100=1.0,
        a_bc=params.a_bc,
        linewidth_kms=params.linewidth_kms,
    )
    return jnp.sum(jnp.asarray(_W_BALMER) * sed)


def bbb_side_unit_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """Unit-``l5100`` BBB-side bolometric power above the Lyman limit [erg/s].

    Disc + lines + FeII + (type 1 only) Balmer continuum: the same terms that
    :func:`~tengri.components.agn.grahsp.model.evaluate_grahsp_agn` puts in
    ``l_bol_bbb``.
    """
    if params.disc_model == "netzer":
        disc = _netzer_disc_power(params, templates)
    else:
        disc = bbb_continuum_power_nm(
            params.uvslope,
            params.plslope,
            params.plbendloc_nm,
            params.plbendwidth,
            params.cutoff_nm,
        )
    total = disc + gaussian_lines_power_above(
        line_wave_nm=templates.line_wave_nm,
        line_broad=templates.line_broad,
        line_narrow_sy2=templates.line_narrow_sy2,
        line_narrow_liner=templates.line_narrow_liner,
        l5100=1.0,
        a_lines=params.a_lines,
        linewidth_kms=params.linewidth_kms,
        agn_type=params.agn_type,
        lower_nm=LYMAN_LIMIT_NM,
    )
    total = total + feii_forest_power(
        template_wave_nm=(
            templates.feii_vc04_wave_nm
            if params.feii_template == "veroncetty2004"
            else templates.feii_wave_nm
        ),
        template_lumin=(
            templates.feii_vc04_lumin
            if params.feii_template == "veroncetty2004"
            else templates.feii_lumin
        ),
        l5100=1.0,
        a_lines=params.a_lines,
        a_feii=params.a_feii,
    )
    if params.agn_type == 1:
        total = total + _balmer_power(params)
    return total


def _torus_gaussian_power(params: GRAHSPParams) -> Array:
    nodes = jnp.asarray(TORUS_NORM_GRID_NM)
    sed = torus_dust_continuum(
        wave_nm=nodes,
        l5100=1.0,
        fcov=params.fcov,
        cool_lam_um=params.cool_lam_um,
        cool_width=params.cool_width,
        hot_lam_um=params.hot_lam_um,
        hot_width=params.hot_width,
        hot_fcov=params.hot_fcov,
    )
    return jnp.sum(jnp.asarray(_W_TORUS) * sed)


def _si_gaussian_power(params: GRAHSPParams) -> Array:
    nodes = jnp.asarray(SI_NORM_GRID_NM)
    feature = si_feature(wave_nm=nodes, l5100=1.0, fcov=params.fcov, si=params.si)
    return jnp.sum(jnp.asarray(_W_SI) * feature)


def _mn12_torus_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """MN12 torus power, integrated on the template's own 239 native nodes."""
    shape = mn12_native_shape(
        templates.torus_mn12_wave_nm,
        params.tor_temp,
        params.tor_cutoff_um,
        templates.torus_mn12_avg,
        templates.torus_mn12_lo,
        templates.torus_mn12_hi,
    )
    l_torus = 2.5 * params.fcov / 12.0 * 0.510  # l5100 = 1
    return l_torus * native_power_nm(templates.torus_mn12_wave_nm, shape)


def _mn12_si_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """MN12 silicate feature power, on the template's own native nodes."""
    l_torus = 2.5 * params.fcov / 12.0 * 0.510  # l5100 = 1
    return (
        l_torus
        * params.si
        * native_power_nm(templates.torus_mn12_si_wave_nm, templates.torus_mn12_si_lumin)
    )


def torus_side_unit_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """Unit-``l5100`` torus-side bolometric power, all wavelengths [erg/s].

    Torus + silicate feature, the terms that ``evaluate_grahsp_agn`` puts in
    ``l_bol_torus``. The pointwise clip ``si >= -torus`` of the published SED is
    not applied here: it is inactive unless the silicate absorption exceeds the
    continuum.
    """
    if params.torus_model == "mn12":
        return _mn12_torus_power(params, templates) + _mn12_si_power(params, templates)
    return _torus_gaussian_power(params) + _si_gaussian_power(params)


def unit_bolometric_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """Unit-``l5100`` intrinsic bolometric power [erg/s]: BBB side + torus side.

    The published SED is divided by this and multiplied by the requested
    bolometric. It is a property of the model parameters only, so it does not
    change with the caller's wavelength grid.
    """
    return bbb_side_unit_power(params, templates) + torus_side_unit_power(params, templates)
