# SPDX-License-Identifier: BSD-3-Clause
"""One decision cascade for projecting an additive emitter onto photometric filters.

Radio, X-ray, dust IR emission and shock nebular each publish a
``*_phot_lnu_precomp`` LUT family under ``WavePrecomp``, and each wrote the same
three-branch choice by hand. Four copies meant four fallbacks that disagreed --
radio/X-ray and dust degraded to a point sample, shock published nothing at all.
This module holds the choice once; each caller still supplies its own band
response, redshift and fallback, because those legitimately differ (#1738).
"""

from collections.abc import Callable

import jax.numpy as jnp

from tengri.utils.band_projection import BandSpec, project_bands


def project_additive_onto_photometry(
    precomputed_phot: jnp.ndarray | None,
    sed: jnp.ndarray,
    wave: jnp.ndarray,
    filter_eff_waves: jnp.ndarray,
    bands: BandSpec | None,
    redshift: jnp.ndarray,
    fallback_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    r"""Project an additive emitter's rest-frame SED onto photometric bands.

    Selects, in order, the most accurate branch whose inputs are available:

    1. **Band response** -- exact *and* fast. The caller already reduced the
       emitter to scalar amplitudes times a response integrated through the true
       filter transmission at build time.
    2. **Dense filter integral** -- exact, but reads the full-resolution SED.
    3. **Effective wavelength** -- approximate; a single sample per band.

    Parameters
    ----------
    precomputed_phot : ndarray, shape (n_filters,), optional
        Band-response result [erg/s/Hz], or ``None`` when the caller has no
        response available. When given it is returned unchanged.
    sed : ndarray, shape (n_wave,)
        Rest-frame specific luminosity of this emitter [erg/s/Hz]. Read only by
        branches 2 and 3.
    wave : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom], ascending.
    filter_eff_waves : ndarray, shape (n_filters,)
        Filter effective wavelengths [Angstrom]. Read by branch 3 only.
    bands : BandSpec, optional
        The band spec the observation published in state: filter curves and the
        convention the exact integral weights them with. ``None`` selects branch 3.
    redshift : ndarray, scalar
        Source redshift [dimensionless]. Read by branch 2 only.
    fallback_fn : callable, optional
        Maps wavelengths [Angstrom] to specific luminosity [erg/s/Hz]. Lets a
        caller re-evaluate its own emission at the effective wavelengths rather
        than interpolate a precomputed SED. Defaults to interpolating ``sed``.

    Returns
    -------
    ndarray, shape (n_filters,)
        Filter-weighted rest-frame L_nu per band [erg/s/Hz] -- **not** a flux.
        Cosmological dimming is applied later, after the L_nu families are
        summed (see ``observation/redshift_kernel.py``).

    Notes
    -----
    **JIT-compatible**: yes -- and gradient- and vmap-safe. Every branch is
    chosen on the *presence* of a build-time array, never on a traced value, so
    the choice is static at trace time.

    Branch 1 exists for speed, not only accuracy: it reaches the answer without
    referencing the full-resolution SED, which is what lets XLA eliminate the
    dense chain. That elimination is where the ``WavePrecomp`` speedup lives, so
    a caller that can supply a response should (#1109).

    Branch 3 is the one that has already shipped a bug. An emission model
    normalizes its frequency integral over whatever wavelength array it is
    handed, so sampling a structured emitter at a few filter pivots renormalizes
    far-IR flux onto those points -- measured at 293% on the reddest band before
    branch 2 was introduced (#629). It remains here only for configurations that
    publish no padded curves.

    References
    ----------
    .. [1] tengri #1109 -- the WavePrecomp speedup depends on the dense SED
       staying unreferenced on the LUT path.
    .. [2] tengri #629 -- exact filter integration for additive emitters, after
       effective-wavelength sampling inflated a band by ~4x.
    """
    if precomputed_phot is not None:
        return precomputed_phot

    if bands is not None:
        return project_bands(sed, wave, bands, redshift)

    if fallback_fn is not None:
        return fallback_fn(filter_eff_waves)

    return jnp.interp(filter_eff_waves, wave, sed)
