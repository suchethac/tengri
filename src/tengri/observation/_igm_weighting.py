# SPDX-License-Identifier: BSD-3-Clause
r"""Spectrum-weighted IGM transmission for the non-stellar LUT photometry.

Under ``WavePrecomp`` the stellar continuum carries the IGM inside its sub-band
quadrature. Every other contributor arrives at :meth:`Observation.predict_via_precomp`
as a band flux and used to be multiplied by the filter-averaged transmission

.. math::

   \langle T\rangle_b = \frac{\int T\,R_b\,w\,d\lambda}{\int R_b\,w\,d\lambda},

which forms :math:`\langle S\rangle\langle T\rangle` where the flux needs
:math:`\langle S T\rangle`. For nebular emission near Ly-alpha that is the
dominant error of the LUT at high redshift: a Cue model at z = 7.3 read F115W
+10.3 % against the exact path, its Ly-alpha line sitting on the break.

Where the component's own dense spectrum :math:`S_c` exists at runtime (it is
what its band flux was integrated from), its band transmission is instead

.. math::

   T_{c,b} = \frac{\int S_c\,T\,R_b\,w\,d\lambda}{\int S_c\,R_b\,w\,d\lambda},

and the band flux :math:`F_{c,b}` enters as :math:`F_{c,b} T_{c,b}`. This is
the exact path's integral whenever :math:`S_c` is the spectrum :math:`F_{c,b}`
came from, including its dust screen. Where it is not (the nebular grid of
``FeaturePrecomp`` publishes no dense spectrum; a single-screen dust model
screens the nebular bucket at the effective wavelength), the weight is the
closest spectrum available and a zero spectrum falls back to
:math:`\langle T\rangle_b`, i.e. to the behavior before this module.

JIT-compatible: pure ``jnp``; the reachable-filter indices are build-time
constants.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "igm_weighted_parts",
    "spectral_igm_correction",
    "subband_igm_correction",
]


def _sum_seds(*seds):
    live = [s for s in seds if s is not None]
    if not live:
        return None
    out = live[0]
    for s in live[1:]:
        out = out + s
    return out


def igm_weighted_parts(derived, dust_mode: str) -> list[tuple]:
    """Each non-stellar band contribution to the LUT total, with its dense spectrum.

    Mirrors how :meth:`Observation.predict_via_precomp` assembles the total in
    each dust branch, so every band flux listed here is exactly the term that
    reached ``total_lnu``.

    Parameters
    ----------
    derived : DerivedState
        The forward state's derived keys.
    dust_mode : {"two_component", "single", "none"}
        Which dust branch the projector took.

    Returns
    -------
    list of (ndarray, ndarray or None)
        ``(band flux, shape (n_filters,) [erg/s/Hz], dense rest spectrum,
        shape (n_wave,) [erg/s/Hz])`` pairs. Bands carried by a component with
        no spectrum here keep the filter-averaged transmission.
    """
    get = derived.get
    parts = []
    neb, shock = get("nebular_phot_lnu_precomp"), get("shock_phot_lnu_precomp")
    sed_neb, sed_shock = get("sed_nebular"), get("sed_shock")

    if dust_mode == "two_component":
        neb_att = get("nebular_phot_lnu_attenuated_precomp")
        if neb_att is not None:
            parts.append((neb_att, get("sed_nebular_attenuated_precomp")))
        elif neb is not None:
            screen = get("dust_diff_attenuation_precomp") * get("dust_bc_attenuation_precomp")
            parts.append((screen * neb, sed_neb))
        shock_att = get("shock_phot_lnu_attenuated_precomp")
        if shock_att is not None:
            parts.append((shock_att, get("sed_shock_attenuated_precomp")))
    elif dust_mode == "single":
        # One λ_eff screen over the nebular bucket, which also holds shock here.
        bucket = _sum_seds(neb, shock)
        if bucket is not None:
            parts.append((get("dust_attenuation_precomp") * bucket, _sum_seds(sed_neb, sed_shock)))
    else:
        # The nebular grid serves its band flux as sub-band chunks instead, which
        # :func:`subband_igm_correction` weights.
        if neb is not None and get("nebular_phot_lnu_subband_precomp") is None:
            parts.append((neb, sed_neb))
        if shock is not None:
            parts.append((shock, sed_shock))

    agn_att = get("agn_phot_lnu_attenuated_precomp")
    if agn_att is not None:
        parts.append((agn_att, get("sed_agn_attenuated_precomp")))
    elif get("agn_phot_lnu_precomp") is not None:
        parts.append((get("agn_phot_lnu_precomp"), get("sed_agn")))
    return parts


def spectral_igm_correction(
    parts, transmission, reach, wave, fw_pad, ft_pad, z, band_factor, *, convention
):
    r"""Correction from the band-averaged transmission to each spectrum's own :math:`T_{c,b}`.

    Parameters
    ----------
    parts : list of (ndarray, ndarray or None)
        From :func:`igm_weighted_parts`.
    transmission : ndarray, shape (n_wave,)
        IGM transmission on the rest grid at ``z`` [dimensionless].
    reach : ndarray of int, shape (n_reach,)
        Filters the IGM can reach; every other band is left untouched.
    wave : ndarray, shape (n_wave,)
        Rest wavelength grid [Angstrom].
    fw_pad, ft_pad : ndarray, shape (n_filters, n_pad)
        Padded filter curves, as the band fluxes were integrated with.
    z : float
        Redshift.
    band_factor : ndarray, shape (n_filters,)
        :math:`\langle T\rangle_b`, which the caller has already applied.
    convention : FilterConvention
        Bandpass weight.

    Returns
    -------
    ndarray, shape (n_filters,)
        :math:`\sum_c F_{c,b}\,(T_{c,b} - \langle T\rangle_b)` [erg/s/Hz], zero
        outside ``reach`` and wherever a spectrum is absent or zero in the band.

    Notes
    -----
    **JIT-compatible**: yes.
    """
    from tengri.observation.photometry import lnu_filter_integral_batch

    n_filters = band_factor.shape[0]
    fw_r, ft_r = fw_pad[reach], ft_pad[reach]
    t_avg = band_factor[reach]
    correction = jnp.zeros(n_filters, dtype=band_factor.dtype)
    for band_flux, sed in parts:
        if sed is None:
            continue
        num = lnu_filter_integral_batch(
            sed * transmission, wave, fw_r, ft_r, z, convention=convention
        )
        den = lnu_filter_integral_batch(sed, wave, fw_r, ft_r, z, convention=convention)
        live = den != 0.0
        t_own = jnp.where(live, num / jnp.where(live, den, 1.0), t_avg)
        correction = correction.at[reach].add(band_flux[reach] * (t_own - t_avg))
    return correction


def subband_igm_correction(chunks, nodes_rest, transmission, wave, reach, band_factor):
    r"""Correction for a band flux served as sub-band chunks, with no dense spectrum.

    The nebular grid serves a dusty model's nebular band flux as screened chunks
    :math:`\Phi_k` at flux-weighted rest nodes :math:`\lambda_k` (#2570) and never
    materializes the spectrum :func:`spectral_igm_correction` would weight by.
    Each chunk instead takes :math:`T` at its own node, the K-point form the
    stellar continuum used before the exact fold (#1135), so the band moves from
    :math:`\langle T\rangle_b\sum_k\Phi_k` to :math:`\sum_k\Phi_k T(\lambda_k)` (#2679).

    Parameters
    ----------
    chunks : ndarray, shape (n_filters, n_subbands)
        Screened sub-band band fluxes [erg/s/Hz].
    nodes_rest : ndarray, shape (n_filters, n_subbands)
        Their rest-frame nodes [Angstrom].
    transmission : ndarray, shape (n_wave,)
        IGM transmission on the rest grid at the runtime redshift.
    wave : ndarray, shape (n_wave,)
        Rest wavelength grid [Angstrom].
    reach : ndarray of int, shape (n_reach,)
        Filters the IGM can reach.
    band_factor : ndarray, shape (n_filters,)
        The band-averaged transmission the caller already applied.

    Returns
    -------
    ndarray, shape (n_filters,)
        :math:`\sum_k \Phi_k (T(\lambda_k) - \langle T\rangle_b)` [erg/s/Hz], zero
        outside ``reach``.

    Notes
    -----
    **JIT-compatible**: yes.
    """
    t_nodes = jnp.interp(nodes_rest[reach], wave, transmission)
    delta = jnp.sum(chunks[reach] * (t_nodes - band_factor[reach][:, None]), axis=-1)
    return jnp.zeros_like(band_factor).at[reach].add(delta)
