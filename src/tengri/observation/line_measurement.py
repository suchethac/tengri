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
This module applies that **same operator** to a *model* spectrum, "measure the
observable the way it was measured", so a forward-modeled line flux is directly
comparable to a catalog line flux (including, self-consistently, the stellar
Balmer absorption under the emission for baked-in nebular SSPs).

The reduction :func:`_line_flux_from_means` takes window mean fluxes and is
single-sourced across two window-mean sources:

* **exact** (:func:`measure_line_flux_jax`), means from a reconstructed
  rest-frame :math:`L_\nu` SED (any nebular backend: Cue additive, baked-in, …);
* **fast** (:func:`measure_line_fluxes_from_window_lut`), means from the
  precomputed SSP window-integral LUT contracted with SFH+metallicity weights
  and the per-age dust screen (baked-in / LUT-eligible models only).

Both are bit-exact where the LUT reconstructs the SED. See issue #950.

Notes
-----
**Continuum caveat (Balmer).** The side-band continuum carries the stellar
absorption around the line. For a baked-in SSP the stellar and nebular light are
inseparable, so the measured Balmer flux depends on the continuum windows exactly
as a real pipeline's does, a crude window pair biases the Balmer decrement (see
the #950 probe). Choose continuum windows that match the target catalog, and
prefer the direct nebular luminosity (Cue ``predict_line_fluxes``) when an
absorption-clean intrinsic line flux is wanted.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp

from tengri.observation.spectral_indices import _window_mean_flux, soft_window_ssp_integral
from tengri.utils.physics_constants import C_AA, L_SUN
from tengri.utils.scale import apply_log10_scale


@dataclasses.dataclass(frozen=True)
class LineDef:
    """Definition of one emission line measured catalog-style from a spectrum.

    Parameters
    ----------
    name : str
        Line identifier (e.g. ``"Halpha"``).
    wavelength : float
        Rest-frame **vacuum** line center [Å] (for labeling / data alignment).
    continuum : tuple of (float, float)
        ``((blue_lo, blue_hi), (red_lo, red_hi))``, the two pseudo-continuum
        side-bands [Å], used for a linear continuum under the line.
    feature : tuple of float
        ``(lo, hi)``, the feature window [Å] over which the continuum-subtracted
        emission is integrated. Its center is the effective :math:`\\lambda_c`.
    """

    name: str
    wavelength: float
    continuum: tuple
    feature: tuple


def _line_flux_from_means(feat_mean, cont_mean, lam_c, feat_width, log10_four_pi_dl2):
    r"""Observed line flux from feature + continuum mean fluxes, the one operator.

    ``feat_mean`` / ``cont_mean`` are physical mean :math:`L_\nu` [erg/s/Hz]
    (SFH-weighted, dust-attenuated, mass-scaled). Converts the continuum-
    subtracted mean to a per-wavelength emission, multiplies by the feature width
    (rectangular narrow-line approximation of :math:`\int (L_\nu-L_\nu^{\rm cont})
    \,d\nu`), and divides by :math:`4\pi d_L^2`.

    .. math::

        F_{\rm line} = (\bar L_\nu^{\rm feat} - \bar L_\nu^{\rm cont})
                       \frac{c}{\lambda_c^2}\,\Delta\lambda \big/ 4\pi d_L^2

    with :math:`\bar L_\nu` [erg/s/Hz], :math:`c` [Å/s], :math:`\lambda_c` and
    :math:`\Delta\lambda` [Å], :math:`d_L` [cm], giving [erg/s/cm^2].

    Parameters
    ----------
    feat_mean, cont_mean : ndarray, shape ()
        Feature-window and continuum mean :math:`L_\nu` [erg/s/Hz].
    lam_c : float
        Feature-window center :math:`\lambda_c` [Å].
    feat_width : float
        Feature-window width :math:`\Delta\lambda` [Å].
    log10_four_pi_dl2 : ndarray, shape ()
        :math:`\log_{10}(4\pi d_L^2)` [dex], from
        :func:`tengri.utils.scale.log10_four_pi_dl2`.

    Returns
    -------
    ndarray, shape ()
        Observed line flux [erg/s/cm^2]; positive for emission.

    Notes
    -----
    JIT/grad/vmap-safe.

    **Neither the numerator nor the denominator is materialized** (#1859). Both
    are out of float32 range and in opposite directions, while the answer sits
    comfortably inside it: for an ordinary galaxy the ``erg/s`` line luminosity is
    ~1.4e40 (float32 max 3.4e38) and :math:`4\pi d_L^2` is ~1.0e57, so the linear
    spelling was ``inf/inf``, ``nan`` at *every* redshift, including the 10-pc
    :math:`z=0` convention. Grouping the two conversion constants with the
    distance into one log offset and applying it to the O(1e28) mean keeps every
    intermediate in range.

    The line-luminosity overflow is distance-independent, so repairing only the
    divisor would have left the ``nan`` in place.
    """
    # log10 of (c / lam_c^2) * feat_width / (4 pi d_L^2), a ~-45 dex offset that
    # exists only as an exponent. feat_width == 0 gives -inf, which powers back to
    # an exact 0.0, matching the linear form's multiply-by-zero.
    log10_conv = jnp.log10(C_AA) - 2.0 * jnp.log10(lam_c) + jnp.log10(feat_width)
    return apply_log10_scale(feat_mean - cont_mean, log10_conv - log10_four_pi_dl2)


def _continuum_at(lam_c, x_blue, x_red, f_blue, f_red):
    """Linear side-band continuum evaluated at ``lam_c`` (in L_nu)."""
    return f_blue + (f_red - f_blue) * (lam_c - x_blue) / (x_red - x_blue)


def measure_line_flux_jax(wave, sed_lnu, line_def, log10_four_pi_dl2):
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
    log10_four_pi_dl2 : ndarray, shape ()
        :math:`\log_{10}(4\pi d_L^2)` [dex] at the evaluation redshift, from
        :func:`tengri.utils.scale.log10_four_pi_dl2`. The linear divisor is
        ``inf`` in float32 at every distance (#1859).

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
    return _line_flux_from_means(f_feat, cont, lam_c, fhi - flo, log10_four_pi_dl2)


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
        Sigmoid edge width [Å], MUST match :func:`_window_mean_flux`.

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


#: ``L_sun`` [erg/s] split as ``_LSUN_MANTISSA * 2**_LSUN_EXP2``, exactly.
#:
#: ``total_mass * L_sun`` is ~4e43 for an ordinary galaxy against a float32 max of
#: 3.4e38, so the **scale constant** overflows to ``inf`` on its own — while the
#: window mean it multiplies (~6e28) and the flux that comes out (~1e-16) are both
#: comfortably in range. Every one of the offending decades sits in ``L_sun``'s
#: binary exponent (112), so stripping it leaves the arithmetic at the mass's own
#: scale and :func:`jax.numpy.ldexp` puts it back at the end.
#:
#: The split is **bit-exact in float64**: scaling by a power of two is exact and
#: commutes with rounding, so ``ldexp(fl(m*x), k) == fl(2**k*m*x)`` for every
#: intermediate below. That is the same property that makes
#: :data:`~tengri.utils.scale.DEFAULT_COTANGENT_BOOST` safe to divide back out, and
#: it is why this repair moves no float64 result.
_LSUN_MANTISSA, _LSUN_EXP2 = math.frexp(L_SUN)


def measure_line_fluxes_from_window_lut(
    joint_weights, total_mass, transmission, precomp, log10_four_pi_dl2
):
    r"""Fast catalog-style line fluxes from the SSP window-integral LUT.

    The FeaturePrecomp line-flux path: contract precomputed SSP window integrals
    with the SFH+metallicity weights, apply the age-dependent two-component screen
    at each window center, then run the same :func:`_line_flux_from_means`
    reduction as the exact path, no full-grid SED reconstruction.

    Parameters
    ----------
    joint_weights : ndarray, shape (n_met, n_age)
        Published SFH × metallicity CSP weights (sum to 1).
    total_mass : ndarray, shape ()
        Total formed stellar mass [Msun]. The ``erg/s`` scale is
        ``total_mass · L_sun``, applied here rather than by the caller so the
        power-of-two split that keeps it inside float32 stays in one place
        (see :data:`_LSUN_MANTISSA`).
    transmission : ndarray, shape (n_age, n_window)
        Two-component transmission at each window center per SSP age.
    precomp : LineWindowPrecomputation
        Per-(met, age) window integrals + per-line window recipe.
    log10_four_pi_dl2 : ndarray, shape ()
        :math:`\log_{10}(4\pi d_L^2)` [dex] at the evaluation redshift, from
        :func:`tengri.utils.scale.log10_four_pi_dl2`.

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
    # ``L_sun`` is carried as a binary exponent, not as a factor: the
    # product runs at the mass's own scale (~1e-5) and ``ldexp`` restores the
    # ~1e28 erg/s/Hz window mean in one exact step. Spelling this as
    # ``(total_mass * L_sun) * ...`` was ``inf * finite`` in float32, and the
    # ``feat - cont`` below then read ``inf - inf`` -> ``nan`` on every line (#1859).
    scale = total_mass * _LSUN_MANTISSA
    window_means = jnp.ldexp(
        scale * jnp.sum(transmission * wint_age, axis=0) / precomp.window_norms, _LSUN_EXP2
    )
    centers = precomp.window_centers
    out = []
    for _name, b, r, f, lam_c, width in precomp.line_slots:
        cont = _continuum_at(lam_c, centers[b], centers[r], window_means[b], window_means[r])
        out.append(_line_flux_from_means(window_means[f], cont, lam_c, width, log10_four_pi_dl2))
    return jnp.stack(out)


def default_line_defs(
    wavelengths,
    names=None,
    *,
    feature_halfwidth: float = 8.0,
    cont_gap: float = 17.0,
    cont_width: float = 20.0,
):
    """Build generic :class:`LineDef` windows around a set of line centers.

    Used to fit line fluxes through the measure-as-catalog path when only line
    *centers* are known (e.g. a :class:`LineFluxData` set), the likelihood needs
    continuum windows and these are concrete, built once at fitter setup (never
    from traced ``data_args``).

    Parameters
    ----------
    wavelengths : array_like, shape (n_line,)
        Rest-frame vacuum line centers [Å].
    names : sequence of str, optional
        Per-line names; defaults to ``line_<λ>``.
    feature_halfwidth : float, default 8.0
        Half-width of the feature window [Å].
    cont_gap, cont_width : float, default 17.0, 20.0
        The continuum side-bands sit at ``[λ ± (gap+width), λ ± gap]`` [Å].

    Returns
    -------
    tuple of LineDef

    Notes
    -----
    These are **generic** windows: they clear the line itself but not necessarily
    neighboring lines in crowded regions (Hα+[NII]+[SII]). For science against a
    real catalog, pass survey-matched :class:`LineDef` windows explicitly.
    """
    import numpy as _np

    waves = _np.atleast_1d(_np.asarray(wavelengths, dtype=float))
    out = []
    for i, lam in enumerate(waves):
        lam = float(lam)
        name = names[i] if names is not None else f"line_{lam:.0f}"
        out.append(
            LineDef(
                name=name,
                wavelength=lam,
                continuum=(
                    (lam - cont_gap - cont_width, lam - cont_gap),
                    (lam + cont_gap, lam + cont_gap + cont_width),
                ),
                feature=(lam - feature_halfwidth, lam + feature_halfwidth),
            )
        )
    return tuple(out)


#: Illustrative DESI-like emission-line set (rest-frame **vacuum** centers [Å]).
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


def resolve_line_defs(line_defs, observation=None):
    """Line windows to measure: explicit argument, else the observation's own set.

    Parameters
    ----------
    line_defs : sequence of LineDef or None
        Explicit windows. Returned as a tuple when given.
    observation : Observation or None, optional
        The model's observation. When it declares ``line_fluxes``, its line
        identities and wavelengths are used.

    Returns
    -------
    tuple of LineDef
        Windows in the order the caller should interpret the returned fluxes.

    Notes
    -----
    :meth:`SEDModel.measure_line_fluxes` used to fall back to :data:`DESI_LINES`
    whenever ``line_defs`` was omitted, ignoring the observation entirely. A model
    built with an eight-line :class:`LineFluxData` therefore returned **five**
    fluxes, for different lines, in a different order -- a plausible float array of
    the wrong length that raises nothing on its own. Zipped against the caller's
    own list of names it yields silently mislabeled fluxes (#1500).

    ``DESI_LINES`` remains the fallback only when nothing declares a line set.
    """
    if line_defs is not None:
        return tuple(line_defs)
    lfd = getattr(observation, "line_fluxes", None) if observation is not None else None
    if lfd is None:
        return tuple(DESI_LINES)

    import numpy as _np

    # Prefer the CURATED window for a line DESI already defines. default_line_defs
    # builds generic +/-8 A feature and 17-20 A continuum side-bands from a center;
    # DESI_LINES carries hand-chosen side-bands that dodge neighboring features.
    # Rebuilding a generic window for, say, [OIII] 5007 measures a measurably
    # different flux -- enough to move it ~19% against the direct nebular
    # luminosity. Curated where available, generic only for the rest.
    curated = {d.name: d for d in DESI_LINES}
    waves = _np.asarray(lfd.wavelengths)
    names = tuple(lfd.names)
    out = []
    for name, wave in zip(names, waves):
        known = curated.get(name)
        if known is not None and abs(float(known.wavelength) - float(wave)) < 1.0:
            out.append(known)
        else:
            out.append(default_line_defs(_np.asarray([wave]), (name,))[0])
    return tuple(out)
