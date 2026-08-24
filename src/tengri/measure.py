# SPDX-License-Identifier: BSD-3-Clause
r"""Model-free measurement operators: indices, line fluxes, synthetic photometry.

The measurement counterpart to the forward model. Where ``model.predict_*``
asks *"what does this galaxy emit?"*, ``tengri.measure.*`` asks *"what would a
pipeline measure off this spectrum?"*: and it will measure **any** spectrum:
one tengri just predicted, one exported to disk, or one that a user reduced
themselves.

"Model-free" means **no** :class:`~tengri.forward.sed_model.SEDModel` is
required: every operator here is a pure array-in / number-out function. It does
**not** mean these are reduction tools: tengri does not do continuum placement,
sky subtraction, or bad-pixel repair. Those are a reduction pipeline's job, and
the choices they involve are not ones an SED model should be making quietly on
your behalf.

Why the façade exists
---------------------
The underlying engines are correct but each speaks its own dialect, and the
three dialects are precisely where astronomers get burned:

============  =====================================  ==============================
Convention    The engines' native form               What ``measure`` asks you for
============  =====================================  ==============================
**frame**     rest-frame wavelengths [Å]             rest-frame wavelengths [Å]
**units**     rest-frame :math:`L_\nu` [erg/s/Hz]    rest-frame :math:`L_\nu`
**distance**  ``compute_photometry`` -> ``dl_cm``;   a single ``redshift=``
              ``measure_line_flux_jax`` ->
              ``log10_four_pi_dl2``
============  =====================================  ==============================

This module adds **no new algorithms**. It dispatches to
:func:`~tengri.observation.spectral_indices.measure_index_jax`,
:func:`~tengri.observation.line_measurement.measure_line_flux_jax`, and
:func:`~tengri.observation.photometry.compute_photometry`, and its entire value
is that the conventions above are handled once, here, instead of at every call
site.

The same ruler on both sides
----------------------------
tengri's summary-statistic likelihood
(:class:`~tengri.observation.spectral_indices.SpectralIndexData`,
:class:`~tengri.observation.line_flux_data.LineFluxData`) compares a *model*
index against an *observed* index. If your observed values came from a pipeline
whose windows or continuum estimator differ from tengri's, model and data are
being measured with different rulers; a systematic that is invisible in the
residuals and hard to unpick later.

Because these operators need no model, you can close that gap yourself: point
them at your own reduced arrays and measure the observed side with **exactly**
the operator the model side uses.

>>> from tengri import measure
>>> obs_value = measure.spectral_index(wave_rest, flux_obs, "Dn4000")  # doctest: +SKIP
>>> # ...now `obs_value` and the model's Dn4000 are on the same ruler.

Examples
--------
Measure off a prediction (the convenient path; inherits the model's
filter convention and rest-frame grid):

>>> from tengri import measure
>>> pred = model.predict(params)  # doctest: +SKIP
>>> out = measure.from_prediction(pred, indices=("Dn4000",), lines=("Halpha",))  # doctest: +SKIP
>>> out["Dn4000"], out["Halpha"]  # doctest: +SKIP

Measure off bare arrays (no model anywhere):

>>> value = measure.spectral_index(wave_rest, lnu, "Dn4000")  # doctest: +SKIP

See Also
--------
tengri.forward.sed_model.SEDModel.predict_spectral_indices : model-side twin, same operator.
tengri.forward.sed_model.SEDModel.measure_line_fluxes : model-side twin, same operator.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.observation.line_measurement import (
    DESI_LINES,
    LineDef,
    measure_line_flux_jax,
)
from tengri.observation.photometry import compute_photometry
from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    CompositeIndexDef,
    SpectralIndexDef,
    measure_index_jax,
)
from tengri.parameters.resolve import require_redshift
from tengri.utils.filter_convention import FilterConvention

__all__ = [
    "DESI_LINES",
    "STANDARD_INDICES",
    "from_prediction",
    "line_flux",
    "photometry",
    "spectral_index",
]


class _Unset:
    """Sentinel for "argument not supplied".

    Needed because ``None`` is already meaningful for ``from_prediction``'s
    ``filters``: it is what :meth:`Prediction.photometry` takes to mean *"the
    filters the model was built with"*. Reusing ``None`` for *"skip photometry"*
    would make the two surfaces disagree on the same word.
    """

    def __repr__(self):  # pragma: no cover; debugging aid
        return "<unset>"


_UNSET = _Unset()


# ── name resolution ───────────────────────────────────────────────


def _resolve_index(index_def):
    """Accept a name or a definition; never return ``None`` for a typo."""
    if isinstance(index_def, (SpectralIndexDef, CompositeIndexDef)):
        return index_def
    try:
        return STANDARD_INDICES[index_def]
    except KeyError:
        raise KeyError(
            f"unknown spectral index {index_def!r}. Available: "
            f"{', '.join(sorted(STANDARD_INDICES))}"
        ) from None


def _resolve_line(line_def):
    """Accept a name or a :class:`LineDef`; never return ``None`` for a typo."""
    if isinstance(line_def, LineDef):
        return line_def
    by_name = {ln.name: ln for ln in DESI_LINES}
    try:
        return by_name[line_def]
    except KeyError:
        raise KeyError(
            f"unknown emission line {line_def!r}. Available: {', '.join(sorted(by_name))}. "
            "For a line outside this set, pass a LineDef."
        ) from None


def _log10_four_pi_dl2(redshift, dl_cm=None):
    r"""``log10(4 pi d_L^2)`` [dex]: the distance convention ``measure_line_flux_jax`` wants.

    Log, not linear: :math:`4\pi d_L^2` is ``inf`` in float32 at every distance
    (#1859).
    """
    from tengri.cosmology import luminosity_distance
    from tengri.utils.scale import log10_four_pi_dl2

    if dl_cm is None:
        dl_cm = jnp.asarray(luminosity_distance(jnp.asarray(redshift))).reshape(())
    return log10_four_pi_dl2(dl_cm)


# ── the operators ─────────────────────────────────────────────────


def spectral_index(wave_rest, flux, index_def):
    r"""Measure a spectral index (break, EW, or slope) on a **rest-frame** spectrum.

    Parameters
    ----------
    wave_rest : array_like, shape (n_wave,)
        **Rest-frame** wavelengths [Angstrom]. Must span every window the index
        defines: de-redshift an observed grid before passing it (``wave_obs /
        (1 + z)``).
    flux : array_like, shape (n_wave,)
        Flux density on ``wave_rest``. **Any consistent units**; indices are
        ratios of window means, so :math:`L_\nu` [erg/s/Hz] and :math:`F_\nu`
        [erg/s/cm^2/Hz] give the same answer, and no distance is needed.
    index_def : str or SpectralIndexDef or CompositeIndexDef
        An index name from :data:`STANDARD_INDICES` (e.g. ``"Dn4000"``,
        ``"HdA"``) or a definition object.

    Returns
    -------
    ndarray, shape ()
        The index value. Units follow ``index_def.units``: [Angstrom] for
        equivalent widths, [dimensionless] for breaks and slopes.

    Raises
    ------
    KeyError
        If ``index_def`` is a name that is not in :data:`STANDARD_INDICES`. The
        message lists the available names; a typo never silently returns NaN.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes, the window edges are soft
    sigmoids, so the measurement is differentiable w.r.t. ``flux``.

    Delegates to :func:`~tengri.observation.spectral_indices.measure_index_jax`;
    this is the same operator :meth:`SEDModel.predict_spectral_indices` applies
    to the model, so an observed value measured here is on the model's ruler.

    Examples
    --------
    >>> from tengri import measure
    >>> measure.spectral_index(wave_rest, lnu, "Dn4000")  # doctest: +SKIP
    Array(1.43, dtype=float64)
    """
    return measure_index_jax(jnp.asarray(wave_rest), jnp.asarray(flux), _resolve_index(index_def))


def line_flux(wave_rest, lnu, line_def, *, redshift, dl_cm=None):
    r"""Measure an emission-line flux catalog-style from a **rest-frame** :math:`L_\nu` SED.

    Applies the operator a spectroscopic pipeline applies to data; estimate a
    linear continuum from the two side-bands, subtract it, integrate the
    residual emission over the feature window: so the result is directly
    comparable to a catalog's continuum-subtracted line flux, and it carries the
    stellar Balmer absorption under the line self-consistently.

    Parameters
    ----------
    wave_rest : array_like, shape (n_wave,)
        **Rest-frame** wavelengths [Angstrom].
    lnu : array_like, shape (n_wave,)
        **Rest-frame** spectral luminosity :math:`L_\nu` [erg/s/Hz]. Unlike
        :func:`spectral_index`, the units matter here: the output is an absolute
        flux, so a dimensionless or :math:`F_\nu` input gives a wrong answer.
    line_def : str or LineDef
        A line name from :data:`DESI_LINES` (e.g. ``"Halpha"``) or a
        :class:`~tengri.observation.line_measurement.LineDef` for anything else.
    redshift : float
        Source redshift, used **only** to set the luminosity distance.
    dl_cm : float, optional
        Luminosity distance [cm]. Pass this to override the package cosmology;
        otherwise it is derived from ``redshift`` via
        :func:`tengri.cosmology.luminosity_distance`.

    Returns
    -------
    ndarray, shape ()
        Observed line flux [erg/s/cm^2]; positive for emission.

    Raises
    ------
    KeyError
        If ``line_def`` names a line outside :data:`DESI_LINES`.

    Notes
    -----
    **JIT-compatible / differentiable**: yes (soft sigmoid windows).

    .. math::

        F_{\rm line} = \frac{(\bar{L}_\nu^{\rm feat} - \bar{L}_\nu^{\rm cont})
                             \, c / \lambda_c^2 \; \Delta\lambda}{4\pi d_L^2}

    where :math:`\bar{L}_\nu^{\rm feat}` and :math:`\bar{L}_\nu^{\rm cont}` are
    the feature-window and interpolated side-band continuum mean luminosities
    [erg/s/Hz], :math:`\lambda_c` the feature-window center [Å],
    :math:`\Delta\lambda` its width [Å], and :math:`d_L` the luminosity distance
    [cm]. The rectangular :math:`\Delta\lambda` is the narrow-line approximation
    of :math:`\int (L_\nu - L_\nu^{\rm cont})\,d\nu`.

    **Distance convention.** The engine
    (:func:`~tengri.observation.line_measurement.measure_line_flux_jax`) takes
    :math:`\log_{10}(4\pi d_L^2)` directly; this wrapper derives it, which is the
    point of the façade.

    Examples
    --------
    >>> from tengri import measure
    >>> measure.line_flux(wave_rest, lnu, "Halpha", redshift=0.7)  # doctest: +SKIP
    Array(3.1e-16, dtype=float64)
    """
    return measure_line_flux_jax(
        jnp.asarray(wave_rest),
        jnp.asarray(lnu),
        _resolve_line(line_def),
        _log10_four_pi_dl2(redshift, dl_cm),
    )


def photometry(wave_rest, lnu, filters, *, redshift, convention=None, dl_cm=None):
    r"""Synthesize observed-frame photometry from a **rest-frame** :math:`L_\nu` SED.

    Parameters
    ----------
    wave_rest : array_like, shape (n_wave,)
        **Rest-frame** wavelengths [Angstrom].
    lnu : array_like, shape (n_wave,)
        **Rest-frame** spectral luminosity :math:`L_\nu` [erg/s/Hz]. Units matter
        the output is an absolute flux density.
    filters : Photometry or sequence of FilterCurve
        The bands to integrate through. **Passing a**
        :class:`~tengri.observation.photometry_config.Photometry` **is the safe
        form**: it carries its own ``convention``, so the measurement cannot
        silently disagree with the model that owns it.
    redshift : float
        Source redshift. Sets both the wavelength shift and (via the cosmology)
        the luminosity distance.
    convention : FilterConvention, optional
        The bandpass weight :math:`w(\lambda)`. Resolved in this order: an
        explicit argument wins; else the ``convention`` carried by a
        ``Photometry`` object; else :attr:`FilterConvention.BESSELL`
        (photon-counting; the tengri, FSPS, and prospector default).

        .. warning::

           A bare list of ``FilterCurve`` carries **no** convention. If the model
           you are comparing against was built on
           :attr:`~FilterConvention.ENERGY` (the CIGALE / bagpipes convention),
           passing bare curves here silently answers in BESSELL and the two
           disagree by ~0.5-0.8 %. Pass the model's ``Photometry``; or
           :func:`from_prediction`, which inherits it: and the question cannot
           arise.

    dl_cm : float, optional
        Luminosity distance [cm]; overrides the package cosmology.

    Returns
    -------
    ndarray, shape (n_filters,)
        Observed-frame flux density per band, :math:`F_\nu` [erg/s/cm^2/Hz], in
        the order the filters were given.

    Notes
    -----
    **JIT-compatible**: no; :func:`~tengri.observation.photometry.compute_photometry`
    loops over filters in Python. **Gradient-safe**: yes. For a jittable path
    over many bands use
    :func:`~tengri.observation.photometry.compute_flux_density_batch`, or the
    model's own :meth:`SEDModel.predict_photometry`.

    **No IGM.** The SED is integrated exactly as handed over. A model's
    ``predict_photometry`` applies IGM attenuation inside the kernel from
    ``state.derived``; if you want that here, either multiply the transmission
    into ``lnu`` yourself or use :func:`from_prediction`, which measures the
    model's own attenuated SED.

    **Distance convention.** The engine takes ``dl_cm``; the line-flux engine
    takes :math:`4\pi d_L^2`. This façade derives whichever the engine wants
    from one ``redshift=``.

    Examples
    --------
    >>> from tengri import measure
    >>> measure.photometry(wave, lnu, model.observation.photometry, redshift=0.7)  # doctest: +SKIP
    Array([1.2e-29, 3.4e-29, ...], dtype=float64)
    """
    curves = getattr(filters, "filters", filters)
    if convention is None:
        convention = getattr(filters, "convention", FilterConvention.BESSELL)

    if dl_cm is None:
        from tengri.cosmology import luminosity_distance

        dl_cm = jnp.asarray(luminosity_distance(jnp.asarray(redshift))).reshape(())

    return compute_photometry(
        jnp.asarray(lnu),
        jnp.asarray(wave_rest),
        list(curves),
        jnp.asarray(redshift),
        jnp.asarray(dl_cm),
        convention=convention,
    )


def from_prediction(pred, *, indices=None, lines=None, filters=_UNSET):
    r"""Measure indices, line fluxes, and photometry off a :class:`Prediction`.

    The convenient path. It pairs the model's rest-frame SED with the grid it
    lives on (``Prediction.rest_sed()`` alone does not carry its axis), reads the
    redshift from the prediction's parameters, and; for photometry: routes
    through the model's own exact projector so the **filter convention and IGM
    attenuation are inherited rather than re-derived**.

    Parameters
    ----------
    pred : Prediction
        A cached prediction, from ``model.predict(params)``.
    indices : sequence of str or SpectralIndexDef, optional
        Indices to measure. Names resolve against :data:`STANDARD_INDICES`.
    lines : sequence of str or LineDef, optional
        Emission lines to measure. Names resolve against :data:`DESI_LINES`.
    filters : None or sequence of str, optional
        Bands to synthesize, delegated to :meth:`Prediction.photometry`; so the
        model's filter convention and IGM attenuation apply. ``filters=None``
        means *the filters the model was built with* (matching
        ``Prediction.photometry``); a sequence of registered filter **names**
        means those bands. Omit the argument entirely to skip photometry.

    Returns
    -------
    dict
        One flat key per requested quantity; index names and line names as
        given: plus ``"photometry"`` (shape ``(n_filters,)``) whenever
        ``filters`` was supplied at all (including as ``None``). Empty dict if
        nothing was requested.

    Notes
    -----
    **JIT-compatible**: no; it returns a Python dict and touches the
    ``Prediction`` cache. Inside a JIT/vmap, call the array operators
    (:func:`spectral_index`, :func:`line_flux`) directly, or use
    :meth:`SEDModel.predict_properties`.

    Examples
    --------
    >>> from tengri import measure
    >>> pred = model.predict(params)  # doctest: +SKIP
    >>> measure.from_prediction(pred, indices=("Dn4000",), lines=("Halpha",))  # doctest: +SKIP
    {'Dn4000': Array(1.43, ...), 'Halpha': Array(3.1e-16, ...)}
    """
    model, params = pred._model, pred._params
    out = {}

    if indices or lines:
        rest = model._predict_rest_sed(params)
        wave, lnu = rest.wavelength, rest.sed

    for name in indices or ():
        key = name if isinstance(name, str) else name.name
        out[key] = spectral_index(wave, lnu, name)

    if lines:
        z = require_redshift(params, "measure.from_prediction")
        for name in lines:
            key = name if isinstance(name, str) else name.name
            out[key] = line_flux(wave, lnu, name, redshift=z)

    if filters is not _UNSET:
        # Delegate: Prediction.photometry owns the exact projector, the model's
        # FilterConvention, and the IGM factor. Re-deriving any of the three
        # here is how Phase 2 shipped a silent 0.5-0.8 % offset (#1097).
        out["photometry"] = pred.photometry(filters=filters)

    return out
