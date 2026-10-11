# SPDX-License-Identifier: BSD-3-Clause
"""Bolometric normalization of the GRAHSP AGN model, one component at a time.

GRAHSP is linear in ``l5100``. The published SED is ``l5100 * shape``, and
``l5100`` is chosen so the BBB-side bolometric luminosity (paper lumBolBBB,
:func:`~tengri.components.agn.grahsp.bolometric.bolometric_luminosity_bbb`)
equals the requested power. That bolometric is a sum of per-component integrals
above the Lyman limit. Each is taken on the component's own grid, never on the
caller's wavelength grid, so the unit power does not depend on how the caller
samples the spectrum:

* disc: SBPL power law on its own log grid (floored at the X-ray corona edge),
  or the Netzer template integrated on its native nodes with the power-law
  interpolant the template is resampled with;
* Gaussian lines: closed form (complementary error function), exact;
* FeII forest: exact trapezoid of the piecewise-linear template interpolant on
  its own nodes;
* Balmer continuum (type 1 only): log-trapezoid on its own grid up to the
  resampler's 1e6 nm clip.

Every function here is evaluated at unit ``l5100``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy import special as jsp_special

from tengri.components.agn._template_grid import native_bolometric_nu
from tengri.components.agn.grahsp.balmer import balmer_continuum
from tengri.components.agn.grahsp.bbb import LAMBDA_5100_NM, floor_disc_xray, sbpl_bbb
from tengri.components.agn.grahsp.bolometric import LYMAN_LIMIT_NM
from tengri.components.agn.grahsp.disc import select_disc_model
from tengri.components.agn.grahsp.lines import (
    _FWHM_TO_SIGMA,
    _HBETA_BROAD_RATIO,
    _HBETA_NARROW_RATIO,
    AGN_TYPE_BL,
    AGN_TYPE_LINER,
)
from tengri.utils.grid_interp import resample_template
from tengri.utils.physics_constants import C_AA

if TYPE_CHECKING:  # pragma: no cover
    from tengri.components.agn.grahsp.model import GRAHSPParams
    from tengri.components.agn.grahsp.templates import GRAHSPTemplates

__all__ = [
    "BALMER_NORM_GRID_NM",
    "BBB_NORM_GRID_NM",
    "bbb_side_unit_power",
    "disc_unit_power",
    "native_power_nm",
]

BBB_NORM_GRID_NM: np.ndarray = np.geomspace(LYMAN_LIMIT_NM, 1.0e10, 1024)
"""SBPL disc grid [nm], log-spaced from the Lyman limit to 1e10 nm."""

BALMER_NORM_GRID_NM: np.ndarray = np.geomspace(LYMAN_LIMIT_NM, 1.0e6, 2048)
"""Balmer continuum grid [nm] from the Lyman limit to the 1e6 nm clip in
:func:`~tengri.components.agn.grahsp.balmer.balmer_continuum`."""


def _log_trapezoid_weights(grid_nm: np.ndarray) -> np.ndarray:
    """Weights ``w`` with ``sum(w * f)`` = integral of ``f`` d(lambda), trapezoid in ln."""
    half = 0.5 * np.diff(np.log(grid_nm))
    weights = np.zeros_like(grid_nm)
    weights[:-1] += half
    weights[1:] += half
    return weights * grid_nm


_W_BBB = _log_trapezoid_weights(BBB_NORM_GRID_NM)
_W_BALMER = _log_trapezoid_weights(BALMER_NORM_GRID_NM)


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
        template is resampled with (see ``native_bolometric_nu``).
    """
    wave_aa = jnp.asarray(wave_nm) * 10.0
    l_lambda_aa = jnp.asarray(l_lambda_nm) * 0.1
    return native_bolometric_nu(l_lambda_aa * wave_aa**2 / C_AA, wave_aa)


def sbpl_disc_power_nm(
    uvslope: float,
    plslope: float,
    plbendloc_nm: float,
    plbendwidth: float,
    cutoff_nm: float,
) -> Array:
    """Unit-``l5100`` SBPL disc power above the Lyman limit [erg/s], floored."""
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


def netzer_disc_power_nm(
    disc_m: float,
    disc_a: float,
    disc_mdot: float,
    templates: GRAHSPTemplates,
) -> Array:
    """Unit-``l5100`` Netzer disc template power above the Lyman limit [erg/s].

    The template is resampled with :func:`~tengri.utils.grid_interp.resample_template`
    and anchored at 510 nm (:func:`~tengri.components.agn.grahsp.disc.netzer_disc`).
    Its first segment crosses the Lyman limit, so the interpolated value at the
    limit is prepended as a node, which keeps the power-law interpolant exact on
    the retained part.
    """
    idx = select_disc_model(
        templates.disc_m,
        templates.disc_a,
        templates.disc_mdot,
        m=disc_m,
        a=disc_a,
        mdot=disc_mdot,
    )
    wave = np.asarray(templates.disc_wave_nm, dtype=float)
    lumin = jnp.asarray(templates.disc_lumin[idx])
    anchor = resample_template(jnp.asarray([LAMBDA_5100_NM]), wave, lumin, left=0.0, right=0.0)[0]
    shape = lumin / (LAMBDA_5100_NM * anchor)
    nodes = _retained_nodes(wave)
    values = resample_template(jnp.asarray(nodes), wave, shape, left=0.0, right=0.0)
    return native_power_nm(nodes, values)


def disc_unit_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """Unit-``l5100`` disc power above the Lyman limit [erg/s] (no lines, no FeII)."""
    if params.disc_model == "netzer":
        return netzer_disc_power_nm(params.disc_m, params.disc_a, params.disc_mdot, templates)
    return sbpl_disc_power_nm(
        params.uvslope,
        params.plslope,
        params.plbendloc_nm,
        params.plbendwidth,
        params.cutoff_nm,
    )


def _retained_nodes(nodes_nm: np.ndarray) -> np.ndarray:
    """Template nodes above the Lyman limit, with the limit itself added when the
    template crosses it.

    A template that starts above the limit is zero below its first node, so no
    node is added there; adding one would create a spurious ramp from zero.
    """
    above = nodes_nm[nodes_nm > LYMAN_LIMIT_NM]
    if nodes_nm[0] < LYMAN_LIMIT_NM:
        return np.concatenate([[LYMAN_LIMIT_NM], above])
    return above


def _line_power(line_wave_nm: Array, line_lumin_unit: Array, linewidth_kms: float) -> Array:
    """Power above the Lyman limit of unit-area Gaussian lines [erg/s].

    Each line is ``L_i * 510 / sqrt(2 pi sigma^2) * exp(-(lam - lam_i)^2 / 2 sigma^2)``
    (the convention of :func:`~tengri.components.agn.grahsp.lines.gaussian_lines`).
    Its integral from the Lyman limit to infinity is ``L_i * 510 * erfc(x_i) / 2``
    with ``x_i = (91.2 - lam_i) / (sqrt(2) sigma_i)``, exact for any grid.
    """
    width_nm = line_wave_nm * (linewidth_kms / 299792.458)
    sigma = width_nm * _FWHM_TO_SIGMA
    x = (LYMAN_LIMIT_NM - line_wave_nm) / (jnp.sqrt(2.0) * sigma)
    return jnp.sum(line_lumin_unit * LAMBDA_5100_NM * 0.5 * jsp_special.erfc(x))


def line_unit_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """Unit-``l5100`` broad + narrow line power above the Lyman limit [erg/s]."""
    l_agn = 1.0 / LAMBDA_5100_NM
    l_broad = _HBETA_BROAD_RATIO * l_agn * params.a_lines
    l_narrow = _HBETA_NARROW_RATIO * l_agn * params.a_lines
    line_wave = jnp.asarray(templates.line_wave_nm)
    narrow = (
        templates.line_narrow_liner
        if params.agn_type == AGN_TYPE_LINER
        else templates.line_narrow_sy2
    )
    total = _line_power(line_wave, l_narrow * jnp.asarray(narrow), params.linewidth_kms)
    if params.agn_type == AGN_TYPE_BL:
        total = total + _line_power(
            line_wave,
            l_broad * jnp.asarray(templates.line_broad),
            params.linewidth_kms,
        )
    return total


def feii_unit_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """Unit-``l5100`` FeII forest power above the Lyman limit [erg/s] (type 1 only).

    The forest is the piecewise-linear (in lambda) template interpolant, zero
    outside its support. Its integral is the linear trapezoid on the template's
    own nodes above the Lyman limit (plus the limit itself), exact for that
    interpolant.
    """
    if params.agn_type != AGN_TYPE_BL:
        return jnp.asarray(0.0)
    if params.feii_template == "veroncetty2004":
        twave, tlumin = templates.feii_vc04_wave_nm, templates.feii_vc04_lumin
    else:
        twave, tlumin = templates.feii_wave_nm, templates.feii_lumin
    twave = np.asarray(twave, dtype=float)
    nodes = _retained_nodes(twave)
    values = jnp.interp(
        jnp.asarray(nodes),
        jnp.asarray(twave),
        jnp.asarray(tlumin),
        left=0.0,
        right=0.0,
    )
    l_broadlines = _HBETA_BROAD_RATIO * (1.0 / LAMBDA_5100_NM) * params.a_lines
    return params.a_feii * l_broadlines * jnp.trapezoid(values, jnp.asarray(nodes))


def balmer_unit_power(params: GRAHSPParams) -> Array:
    """Unit-``l5100`` Balmer continuum power above the Lyman limit [erg/s] (type 1)."""
    if params.agn_type != AGN_TYPE_BL:
        return jnp.asarray(0.0)
    sed = balmer_continuum(
        wave_nm=jnp.asarray(BALMER_NORM_GRID_NM),
        l5100=1.0,
        a_bc=params.a_bc,
        linewidth_kms=params.linewidth_kms,
    )
    return jnp.sum(jnp.asarray(_W_BALMER) * sed)


def bbb_side_unit_power(params: GRAHSPParams, templates: GRAHSPTemplates) -> Array:
    """Unit-``l5100`` BBB-side bolometric power above the Lyman limit [erg/s].

    Disc + broad and narrow lines + FeII + (type 1 only) Balmer continuum: the
    terms of ``l_bol_bbb`` in :func:`~tengri.components.agn.grahsp.model.evaluate_grahsp_agn`.
    The torus is not included, as in the paper's lumBolBBB.
    """
    return (
        disc_unit_power(params, templates)
        + line_unit_power(params, templates)
        + feii_unit_power(params, templates)
        + balmer_unit_power(params)
    )
