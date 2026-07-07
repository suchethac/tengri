# SPDX-License-Identifier: BSD-3-Clause
r"""Measure emission-line fluxes from a model spectrum, the way a catalog does.

A spectroscopic pipeline (DESI FastSpecFit, SDSS MPA-JHU, …) measures an
emission-line flux by estimating a local continuum, subtracting it, and
integrating the emission that remains:

.. math::

    F_{\rm line} = \frac{1}{4\pi d_L^2}
        \int_{\rm line}\bigl(L_\nu - L_\nu^{\rm cont}\bigr)\,d\nu
      \approx \frac{(\langle L_\nu\rangle_{\rm feat} - L_\nu^{\rm cont})
        \,(c/\lambda_c^2)\,\Delta\lambda_{\rm feat}}{4\pi d_L^2}

with the continuum a linear interpolation between blue and red side-band means.
This module applies that **same operator** to a *model* spectrum — "measure the
observable the way it was measured" — so a forward-modelled line flux is directly
comparable to a catalog line flux (including, self-consistently, the stellar
Balmer absorption under the emission for baked-in nebular SSPs).

The reduction :func:`_line_flux_from_means` takes window mean fluxes and is
single-sourced across two window-mean sources:

* **exact** (:func:`measure_line_flux_jax`) — means from a reconstructed
  rest-frame :math:`L_\nu` SED (any nebular backend: Cue additive, baked-in, …);
* **fast** (:func:`measure_line_fluxes_from_window_lut`) — means from the
  precomputed SSP window-integral LUT contracted with SFH+metallicity weights
  and the per-age dust screen (baked-in / LUT-eligible models only).

Both are bit-exact where the LUT reconstructs the SED. See issue #950.

Notes
-----
**Continuum caveat (Balmer).** The side-band continuum carries the stellar
absorption around the line. For a baked-in SSP the stellar and nebular light are
inseparable, so the measured Balmer flux depends on the continuum windows exactly
as a real pipeline's does — a crude window pair biases the Balmer decrement (see
the #950 probe). Choose continuum windows that match the target catalog, and
prefer the direct nebular luminosity (Cue ``predict_line_fluxes``) when an
absorption-clean intrinsic line flux is wanted.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from tengri.observation.spectral_indices import _window_mean_flux, soft_window_ssp_integral
from tengri.utils.physics_constants import C_AA


@dataclasses.dataclass(frozen=True)
class LineDef:
    """Definition of one emission line measured catalog-style from a spectrum.

    Parameters
    ----------
    name : str
        Line identifier (e.g. ``"Halpha"``).
    wavelength : float
        Rest-frame **vacuum** line centre [Å] (for labelling / data alignment).
    continuum : tuple of (float, float)
        ``((blue_lo, blue_hi), (red_lo, red_hi))`` — the two pseudo-continuum
        side-bands [Å], used for a linear continuum under the line.
    feature : tuple of float
        ``(lo, hi)`` — the feature window [Å] over which the continuum-subtracted
        emission is integrated. Its centre is the effective :math:`\\lambda_c`.
    """

    name: str
    wavelength: float
    continuum: tuple
    feature: tuple


def _line_flux_from_means(feat_mean, cont_mean, lam_c, feat_width, four_pi_dl2):
    r"""Observed line flux from feature + continuum mean fluxes — the one operator.

    ``feat_mean`` / ``cont_mean`` are physical mean :math:`L_\nu` [erg/s/Hz]
    (SFH-weighted, dust-attenuated, mass-scaled). Converts the continuum-
    subtracted mean to a per-wavelength emission, multiplies by the feature width
    (rectangular narrow-line approximation of :math:`\int (L_\nu-L_\nu^{\rm cont})
    \,d\nu`), and divides by :math:`4\pi d_L^2`.

    Parameters
    ----------
    feat_mean, cont_mean : ndarray, shape ()
        Feature-window and continuum mean :math:`L_\nu` [erg/s/Hz].
    lam_c : float
        Feature-window centre :math:`\lambda_c` [Å].
    feat_width : float
        Feature-window width :math:`\Delta\lambda` [Å].
    four_pi_dl2 : ndarray, shape ()
        :math:`4\pi d_L^2` [cm^2] at the evaluation redshift.

    Returns
    -------
    ndarray, shape ()
        Observed line flux [erg/s/cm^2]; positive for emission.
    """
    l_line = (feat_mean - cont_mean) * (C_AA / lam_c**2) * feat_width  # erg/s
    return l_line / four_pi_dl2


def _continuum_at(lam_c, x_blue, x_red, f_blue, f_red):
    """Linear side-band continuum evaluated at ``lam_c`` (in L_nu)."""
    return f_blue + (f_red - f_blue) * (lam_c - x_blue) / (x_red - x_blue)


def measure_line_flux_jax(wave, sed_lnu, line_def, four_pi_dl2):
    r"""Exact catalog-style line flux from a rest-frame :math:`L_\nu` SED.

    Backend-agnostic: measures whatever line is present in ``sed_lnu`` (Cue
    additive emission, baked-in SSP emission, …). The continuum is a linear fit
    between the blue and red side-band means; the emission is integrated over the
    (soft) feature window.

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    sed_lnu : ndarray, shape (n_wave,)
        Rest-frame spectral luminosity :math:`L_\nu` [erg/s/Hz] (dust-attenuated
        total SED, e.g. ``predict_rest_sed(...).sed``).
    line_def : LineDef
        The line + continuum window definition.
    four_pi_dl2 : ndarray, shape ()
        :math:`4\pi d_L^2` [cm^2] at the evaluation redshift.

    Returns
    -------
    ndarray, shape ()
        Observed line flux [erg/s/cm^2].

    Notes
    -----
    **JIT-compatible / differentiable**: yes (soft sigmoid windows).
    """
    (blo, bhi), (rlo, rhi) = line_def.continuum
    flo, fhi = line_def.feature
    f_blue = _window_mean_flux(wave, sed_lnu, blo, bhi)
    f_red = _window_mean_flux(wave, sed_lnu, rlo, rhi)
    f_feat = _window_mean_flux(wave, sed_lnu, flo, fhi)
    lam_c = 0.5 * (flo + fhi)
    cont = _continuum_at(lam_c, 0.5 * (blo + bhi), 0.5 * (rlo + rhi), f_blue, f_red)
    return _line_flux_from_means(f_feat, cont, lam_c, fhi - flo, four_pi_dl2)


@dataclasses.dataclass(frozen=True)
class LineWindowPrecomputation:
    """Precomputed SSP window integrals for catalog-style line-flux measurement.

    Built once per model + line set; consumed by
    :func:`measure_line_fluxes_from_window_lut`. Mirrors
    :class:`tengri.observation.spectral_indices.IndexWindowPrecomputation` but
    carries the per-line ``(blue, red, feature, lambda_c, width)`` recipe needed
    for the physical (scale + distance) line flux.

    Attributes
    ----------
    window_integrals : ndarray, shape (n_met, n_age, n_window)
        Soft-window SSP integrals [erg/s/Hz/Msun · Å], deduplicated windows.
    window_norms : ndarray, shape (n_window,)
        Window normalizations (``mean = integral / norm``).
    window_centers : ndarray, shape (n_window,)
        Window mid-wavelengths [Å] (for per-window dust + the continuum slope).
    line_slots : tuple
        Per line, ``(name, blue_slot, red_slot, feat_slot, lambda_c, width)``.
    names : tuple of str
        Line names in order.
    """

    window_integrals: jnp.ndarray
    window_norms: jnp.ndarray
    window_centers: jnp.ndarray
    line_slots: tuple
    names: tuple


def precompute_line_windows(ssp_wave, ssp_flux, line_defs, edge_width: float = 1.0):
    """Precompute SSP window integrals for a set of :class:`LineDef`.

    Parameters
    ----------
    ssp_wave : ndarray, shape (n_wave,)
        SSP wavelength grid [Å].
    ssp_flux : ndarray, shape (n_met, n_age, n_wave)
        SSP spectra [erg/s/Hz/Msun].
    line_defs : sequence of LineDef
        Lines to precompute.
    edge_width : float, default 1.0
        Sigmoid edge width [Å] — MUST match :func:`_window_mean_flux`.

    Returns
    -------
    LineWindowPrecomputation

    Notes
    -----
    **JIT-compatible**: yes (built once at construction). Windows shared across
    lines (e.g. a common continuum band) are integrated once.
    """
    ssp_wave = jnp.asarray(ssp_wave)
    ssp_flux = jnp.asarray(ssp_flux)

    unique: dict[tuple[float, float], int] = {}
    integrals: list[jnp.ndarray] = []
    norms: list[jnp.ndarray] = []
    centers: list[float] = []

    def _slot(lo, hi) -> int:
        key = (round(float(lo), 4), round(float(hi), 4))
        if key in unique:
            return unique[key]
        integral, norm = soft_window_ssp_integral(ssp_wave, ssp_flux, lo, hi, edge_width)
        integrals.append(integral)
        norms.append(norm)
        centers.append(0.5 * (float(lo) + float(hi)))
        unique[key] = len(integrals) - 1
        return unique[key]

    line_slots = []
    names = []
    for ld in line_defs:
        (blo, bhi), (rlo, rhi) = ld.continuum
        flo, fhi = ld.feature
        b = _slot(blo, bhi)
        r = _slot(rlo, rhi)
        f = _slot(flo, fhi)
        lam_c = 0.5 * (flo + fhi)
        line_slots.append((ld.name, b, r, f, lam_c, fhi - flo))
        names.append(ld.name)

    return LineWindowPrecomputation(
        window_integrals=jnp.stack(integrals, axis=-1),
        window_norms=jnp.stack(norms),
        window_centers=jnp.asarray(centers),
        line_slots=tuple(line_slots),
        names=tuple(names),
    )


def measure_line_fluxes_from_window_lut(joint_weights, scale, transmission, precomp, four_pi_dl2):
    r"""Fast catalog-style line fluxes from the SSP window-integral LUT.

    The FeaturePrecomp line-flux path: contract precomputed SSP window integrals
    with the SFH+metallicity weights, apply the age-dependent two-component screen
    at each window centre, then run the same :func:`_line_flux_from_means`
    reduction as the exact path — no full-grid SED reconstruction.

    Parameters
    ----------
    joint_weights : ndarray, shape (n_met, n_age)
        Published SFH × metallicity CSP weights (sum to 1).
    scale : ndarray, shape ()
        ``stellar_mass_scale`` = ``total_mass · L_sun`` [erg/s per Msun weight].
    transmission : ndarray, shape (n_age, n_window)
        Two-component transmission at each window centre per SSP age.
    precomp : LineWindowPrecomputation
        Per-(met, age) window integrals + per-line window recipe.
    four_pi_dl2 : ndarray, shape ()
        :math:`4\pi d_L^2` [cm^2] at the evaluation redshift.

    Returns
    -------
    ndarray, shape (n_line,)
        Observed line fluxes [erg/s/cm^2] in ``precomp.names`` order.

    Notes
    -----
    **JIT-compatible**: yes. Bit-exact with :func:`measure_line_flux_jax` where
    the LUT reconstructs the SED (baked-in / LUT-eligible models).
    """
    wint_age = jnp.einsum("ma,maw->aw", joint_weights, precomp.window_integrals)
    window_means = scale * jnp.sum(transmission * wint_age, axis=0) / precomp.window_norms
    centers = precomp.window_centers
    out = []
    for _name, b, r, f, lam_c, width in precomp.line_slots:
        cont = _continuum_at(lam_c, centers[b], centers[r], window_means[b], window_means[r])
        out.append(_line_flux_from_means(window_means[f], cont, lam_c, width, four_pi_dl2))
    return jnp.stack(out)


#: Illustrative DESI-like emission-line set (rest-frame **vacuum** centres [Å]).
#: The continuum side-bands are reasonable defaults for clean regions; the
#: crowded Halpha+[NII]+[SII] complex is approximate and should be tuned to the
#: target survey's continuum definition (see the module Balmer caveat).
DESI_LINES = (
    LineDef("Hbeta", 4862.71, ((4820.0, 4845.0), (4880.0, 4905.0)), (4855.0, 4871.0)),
    LineDef("OIII_5007", 5008.24, ((4975.0, 4995.0), (5020.0, 5045.0)), (5000.0, 5017.0)),
    LineDef("Halpha", 6564.61, ((6505.0, 6535.0), (6600.0, 6620.0)), (6556.0, 6573.0)),
    LineDef("NII_6584", 6585.27, ((6505.0, 6535.0), (6600.0, 6620.0)), (6577.0, 6593.0)),
    LineDef("SII_6717", 6718.29, ((6690.0, 6708.0), (6745.0, 6770.0)), (6711.0, 6725.0)),
)
