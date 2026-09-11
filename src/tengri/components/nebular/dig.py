# SPDX-License-Identifier: BSD-3-Clause
"""Diffuse Ionized Gas (DIG) nebular emission model.

Models the mixing of ionizing photon-powered emission from two gas components:
dense HII regions and diffuse ionized gas (DIG). DIG is low-density ionized gas
between HII regions, characterized by lower ionization parameter (log U ~ −4 vs
−2.5 in HII regions). This produces a distinct emission-line signature: enhanced
low-ionization diagnostics ([NII]/Hα, [SII]/Hα, [OI]/Hα).

Observationally, DIG contributes ~30–60% of Hα flux in local star-forming
galaxies (Reynolds 1984; Haffner et al. 2009; Tacchella et al. 2022).

**Mixing formula**: Returns a weighted average of nebular emission at two
ionization parameters (HII + DIG):

    L_total = (1 − f_DIG) × L(log U_HII) + f_DIG × L(log U_DIG)

where log U_DIG = log U_HII + Δ log U, with Δ log U = −1 dex (default).

When f_DIG = 0 (default), collapses to pure HII region emission.

Three public channel entry points share one mixing core
(:func:`_mix_dig_backend_evaluations`): :func:`mix_dig_emission` for the
continuum SED (``predict_nebular_sed``) and :func:`mix_dig_line_luminosities`
for the discrete line catalog (``predict_nebular_line_luminosities``), both
against a live backend forward. Both build one frozen keyword dict and reuse
it for the HII and DIG backend calls, so any extra backend-specific kwarg
(e.g. an ionizing population resolved by the caller) reaches both evaluations
identically (#2221).

:func:`mix_dig_grid_reconstruction` is the third: the same two-query-point mix
against the tabulated per-Q_H nebular grid
(:mod:`~tengri.components.nebular.nebular_grid_precompute`) instead of a live
backend forward, for the ``FeaturePrecomp`` fast path (#2222).
"""

import jax.numpy as jnp


def _mix_dig_backend_evaluations(evaluate, combine, neb_logU, neb_dig_frac, neb_dig_delta_logU):
    """Evaluate a backend at the HII and (if needed) DIG ionization parameters.

    Parameters
    ----------
    evaluate : callable
        ``evaluate(logU) -> result``. Called once at ``neb_logU`` (HII) and,
        unless the short-circuit fires, again at ``neb_logU +
        neb_dig_delta_logU`` (DIG). ``result`` may be a single array (the SED
        channel) or a tuple (the line-luminosity channel); this function does
        not interpret its shape.
    combine : callable
        ``combine(hii_result, dig_result, neb_dig_frac) -> mixed_result``.
        Only called when the DIG evaluation runs.
    neb_logU : float
        HII region ionization parameter. [log10(U)]
    neb_dig_frac : float
        DIG mass fraction. [dimensionless, in [0, 1]]
    neb_dig_delta_logU : float
        Offset in ionization parameter for DIG (negative). [dex]

    Returns
    -------
    Whatever ``evaluate`` or ``combine`` returns: the HII-only result when
    ``neb_dig_frac`` short-circuits, else ``combine``'s result.
    """
    hii_result = evaluate(neb_logU)

    # Short-circuit: when neb_dig_frac is a Python literal 0.0, skip the extra
    # forward pass entirely. Under JIT with a traced value both passes execute.
    if isinstance(neb_dig_frac, (int, float)) and neb_dig_frac == 0.0:
        return hii_result

    dig_result = evaluate(neb_logU + neb_dig_delta_logU)
    return combine(hii_result, dig_result, neb_dig_frac)


def _linear_mix(hii, dig, frac):
    """Linear mass-fraction mix: ``(1.0 - frac) * hii + frac * dig``.

    Parameters
    ----------
    hii : array_like
        HII-region value(s).
    dig : array_like
        DIG value(s), same shape as ``hii``.
    frac : float
        DIG mass fraction. [dimensionless, in [0, 1]]
    """
    return (1.0 - frac) * hii + frac * dig


def mix_dig_emission(
    nebular_backend,
    ssp_wave: jnp.ndarray,
    ssp_weights: jnp.ndarray,
    ssp_log_ages_yr: jnp.ndarray,
    log_z: float,
    neb_logU: float = -3.0,
    neb_logZ_gas: float | None = None,
    neb_fesc: float = 0.0,
    neb_fesc_lya: float = 0.0,
    neb_dig_frac: float = 0.0,
    neb_dig_delta_logU: float = -1.0,
    line_sigma_aa: float = 0.0,
    **kwargs,
) -> jnp.ndarray:
    r"""Predict nebular SED with HII region and diffuse ionized gas components.

    Computes emission from two ionization regimes (HII + DIG) using any backend,
    then combines via mass-weighted mixing. This captures the enhanced low-ionization
    emission ([NII], [SII], [OI]) characteristic of warm, ionized ISM.

    Parameters
    ----------
    nebular_backend : NebularBackend (CloudyGridBackend or CueBackend)
        Backend with predict_nebular_sed() method. Called twice (HII, DIG).
    ssp_wave : array, shape (n_wave,)
        SSP wavelength grid in Å. [Å]
    ssp_weights : array, shape (n_age,)
        Stellar mass weights of composite stellar population per age bin.
        [Msun]
    ssp_log_ages_yr : array, shape (n_age,)
        Age grid of SSP basis (in cosmic time). [log10(yr)]
    log_z : float
        Stellar metallicity (SSP grid metallicity). [log10(Z)]
    neb_logU : float, optional
        HII region ionization parameter. Default: −3.0. [log10(U)]
    neb_logZ_gas : float, optional
        Gas-phase metallicity (relative to solar). If None, defaults to
        stellar metallicity. Default: None. [log10(Z/Zsun)]
    neb_fesc : float, optional
        Ionizing photon escape fraction (applies to both HII and DIG).
        Default: 0.0. [dimensionless, ∈ [0, 1]]
    neb_fesc_lya : float, optional
        Lyman-α-specific escape fraction. Default: 0.0. [dimensionless, ∈ [0, 1]]
    neb_dig_frac : float, optional
        DIG mass fraction. Default: 0.0. [dimensionless, ∈ [0, 1]]
    neb_dig_delta_logU : float, optional
        Offset in ionization parameter for DIG (negative). Default: −1.0 dex.
        [dex]
    line_sigma_aa : float, optional
        Gaussian line width for emission-line placement. Default: 0.0 (delta).
        [Å]
    **kwargs
        Additional backend-specific keyword arguments (passed to both calls).

    Returns
    -------
    array, shape (n_wave,)
        Combined nebular SED: (1 − f_DIG) × L_HII + f_DIG × L_DIG.
        [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives, but note
    that when neb_dig_frac is a traced JAX value, both HII and DIG forward
    passes execute (no short-circuit optimization).

    **Mixing model** (Haffner et al. 2009, Tacchella et al. 2022):
        The diffuse ionized gas (DIG) has a lower ionization parameter than
        HII regions, producing a distinct line-ratio signature. We approximate
        the total emission as a linear combination:

        .. math::

            L_{\mathrm{total}}(\lambda) = (1 - f_{\mathrm{DIG}}) \,
                L_{\mathrm{HII}}(\lambda, \log U_{\mathrm{HII}}) +
                f_{\mathrm{DIG}} \, L_{\mathrm{DIG}}(\lambda, \log U_{\mathrm{DIG}})

        where log U_DIG = log U_HII + Δ log U, and both components use the
        same metallicity, escape fraction, and stellar population weights.

    **Physical picture**:

        - **HII regions**: dense, photoionized by young (< 1 Myr) OB stars.
          Log U ~ −2.5 to −3. Dominated by recombination lines ([OIII], [SIII],
          etc.).
        - **DIG**: diffuse, warm (~8000 K), ionized by stellar radiation and
          shocks. Log U ~ −3 to −4. Dominated by forbidden lines ([NII], [SII],
          [OI]).

    **Escape fraction**:
        Both HII and DIG share the same f_esc (stellar-population-level escape).
        This is a simplification; physically, dust might preferentially shield
        DIG, but we do not model this spatial variation.

    **Pitfall**: When neb_dig_frac is a JAX-traced (differentiable) value,
    both forward passes (HII + DIG) execute and contribute to gradients, even
    if neb_dig_frac = 0 at runtime. Pre-compute DIG mixing only when needed.

    References
    ----------
    .. [1] L. M. Haffner et al., "The warm ionized medium in spiral galaxies,"
       Rev. Mod. Phys., 81, 969 (2009).
       https://doi.org/10.1103/RevModPhys.81.969
    .. [2] S. Tacchella et al., "H-alpha emission in local galaxies: star
       formation, time variability, and the diffuse ionized gas," MNRAS, 513,
       2904 (2022). arXiv:2112.00027. https://doi.org/10.1093/mnras/stac818
    .. [3] R. J. Reynolds, "A Measurement of the Hydrogen Recombination Rate
       in the Diffuse Interstellar Medium," ApJ, 282, 191 (1984).
       https://doi.org/10.1086/162190

    """
    common_kw = dict(
        ssp_wave=ssp_wave,
        ssp_weights=ssp_weights,
        ssp_log_ages_yr=ssp_log_ages_yr,
        log_z=log_z,
        neb_logZ_gas=neb_logZ_gas,
        neb_fesc=neb_fesc,
        neb_fesc_lya=neb_fesc_lya,
        line_sigma_aa=line_sigma_aa,
        **kwargs,
    )

    def _evaluate(logU):
        return nebular_backend.predict_nebular_sed(neb_logU=logU, **common_kw)

    def _combine(neb_hii, neb_dig, frac):
        return _linear_mix(neb_hii, neb_dig, frac)

    return _mix_dig_backend_evaluations(
        _evaluate, _combine, neb_logU, neb_dig_frac, neb_dig_delta_logU
    )


def mix_dig_line_luminosities(
    nebular_backend,
    ssp_wave: jnp.ndarray,
    ssp_weights: jnp.ndarray,
    ssp_log_ages_yr: jnp.ndarray,
    log_z: float,
    neb_logU: float = -3.0,
    neb_logZ_gas: float | None = None,
    neb_fesc: float = 0.0,
    neb_fesc_lya: float = 0.0,
    neb_dig_frac: float = 0.0,
    neb_dig_delta_logU: float = -1.0,
    line_sigma_aa: float = 0.0,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Predict discrete line luminosities with HII and DIG components.

    Same contract as :func:`mix_dig_emission`, but calls
    ``nebular_backend.predict_nebular_line_luminosities`` (a
    ``(line_waves, line_lums)`` pair) instead of ``predict_nebular_sed``.
    Only the luminosities are mixed; the wavelengths are the HII call's
    (the DIG call's own ``line_waves`` are discarded), since both calls
    catalog the same species at the same rest wavelengths and only the HII
    evaluation's copy is kept.

    Parameters
    ----------
    nebular_backend : NebularBackend (CloudyGridBackend, CueBackend, or CB19Backend)
        Backend with a ``predict_nebular_line_luminosities()`` method. Called
        twice (HII, DIG).
    ssp_wave : array, shape (n_wave,)
        SSP wavelength grid in Å. [Å]
    ssp_weights : array, shape (n_age,)
        Stellar mass weights of composite stellar population per age bin.
        [Msun]
    ssp_log_ages_yr : array, shape (n_age,)
        Age grid of SSP basis (in cosmic time). [log10(yr)]
    log_z : float
        Stellar metallicity (SSP grid metallicity). [log10(Z)]
    neb_logU : float, optional
        HII region ionization parameter. Default: −3.0. [log10(U)]
    neb_logZ_gas : float, optional
        Gas-phase metallicity (relative to solar). If None, defaults to
        stellar metallicity. Default: None. [log10(Z/Zsun)]
    neb_fesc : float, optional
        Ionizing photon escape fraction (applies to both HII and DIG).
        Default: 0.0. [dimensionless, ∈ [0, 1]]
    neb_fesc_lya : float, optional
        Lyman-α-specific escape fraction. Default: 0.0. [dimensionless, ∈ [0, 1]]
    neb_dig_frac : float, optional
        DIG mass fraction. Default: 0.0. [dimensionless, ∈ [0, 1]]
    neb_dig_delta_logU : float, optional
        Offset in ionization parameter for DIG (negative). Default: −1.0 dex.
        [dex]
    line_sigma_aa : float, optional
        Unused by the line-luminosity channel (kept for a signature identical
        to :func:`mix_dig_emission`; forwarded via ``**kwargs`` to the backend,
        which ignores it). Default: 0.0. [Å]
    **kwargs
        Additional backend-specific keyword arguments (passed to both calls).

    Returns
    -------
    line_waves : array, shape (n_lines,)
        Rest-frame line wavelengths, from the HII evaluation. [Å]
    line_lums : array, shape (n_lines,)
        Combined line luminosities: (1 − f_DIG) × L_HII + f_DIG × L_DIG.
        [Lsun]

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives, but note
    that when neb_dig_frac is a traced JAX value, both HII and DIG forward
    passes execute (no short-circuit optimization).

    **Mixing model** (Haffner et al. 2009, Tacchella et al. 2022): identical
    formula to :func:`mix_dig_emission`, applied to each cataloged line's
    luminosity independently:

    .. math::

        L_{\mathrm{total},i} = (1 - f_{\mathrm{DIG}}) \,
            L_{\mathrm{HII},i}(\log U_{\mathrm{HII}}) +
            f_{\mathrm{DIG}} \, L_{\mathrm{DIG},i}(\log U_{\mathrm{DIG}})

    where log U_DIG = log U_HII + Δ log U and :math:`i` indexes the cataloged
    lines.

    **Pitfall**: When neb_dig_frac is a JAX-traced (differentiable) value,
    both forward passes (HII + DIG) execute and contribute to gradients, even
    if neb_dig_frac = 0 at runtime. Pre-compute DIG mixing only when needed.

    References
    ----------
    .. [1] L. M. Haffner et al., "The warm ionized medium in spiral galaxies,"
       Rev. Mod. Phys., 81, 969 (2009).
       https://doi.org/10.1103/RevModPhys.81.969
    .. [2] S. Tacchella et al., "H-alpha emission in local galaxies: star
       formation, time variability, and the diffuse ionized gas," MNRAS, 513,
       2904 (2022). arXiv:2112.00027. https://doi.org/10.1093/mnras/stac818
    .. [3] R. J. Reynolds, "A Measurement of the Hydrogen Recombination Rate
       in the Diffuse Interstellar Medium," ApJ, 282, 191 (1984).
       https://doi.org/10.1086/162190

    """
    common_kw = dict(
        ssp_wave=ssp_wave,
        ssp_weights=ssp_weights,
        ssp_log_ages_yr=ssp_log_ages_yr,
        log_z=log_z,
        neb_logZ_gas=neb_logZ_gas,
        neb_fesc=neb_fesc,
        neb_fesc_lya=neb_fesc_lya,
        line_sigma_aa=line_sigma_aa,
        **kwargs,
    )

    def _evaluate(logU):
        return nebular_backend.predict_nebular_line_luminosities(neb_logU=logU, **common_kw)

    def _combine(hii_result, dig_result, frac):
        line_waves, line_lums_hii = hii_result
        _, line_lums_dig = dig_result
        return line_waves, _linear_mix(line_lums_hii, line_lums_dig, frac)

    return _mix_dig_backend_evaluations(
        _evaluate, _combine, neb_logU, neb_dig_frac, neb_dig_delta_logU
    )


def mix_dig_grid_reconstruction(
    reconstruct, amplitude, point, table, neb_dig_frac, neb_dig_delta_logU
):
    r"""Mix two per-Q_H grid reconstructions (HII + DIG) at a shifted ``neb_logU`` (#2222).

    The tabulated-grid analog of :func:`mix_dig_emission` /
    :func:`mix_dig_line_luminosities`: instead of two live backend forwards,
    this calls the same
    :mod:`~tengri.components.nebular.nebular_grid_precompute` reconstruction
    function twice -- once at ``point["neb_logU"]`` (HII) and, unless the
    zero-fraction short-circuit fires, again at ``point["neb_logU"] +
    neb_dig_delta_logU`` (DIG) -- and mixes with the same core
    (:func:`_mix_dig_backend_evaluations`) the other two wrappers share.
    ``neb_logU`` is one of ``table.axis_names`` whenever DIG mixing could be
    active (``nebular_grid_precompute._dig_may_be_active``), with its range
    extended to cover both query points, so both are always resolvable
    against ``table`` without clipping.

    Parameters
    ----------
    reconstruct : callable
        ``reconstruct(amplitude, point, table) -> array``. One of
        :func:`~tengri.components.nebular.nebular_grid_precompute.reconstruct_nebular_phot`,
        :func:`~tengri.components.nebular.nebular_grid_precompute.reconstruct_nebular_restband`,
        or
        :func:`~tengri.components.nebular.nebular_grid_precompute.reconstruct_nebular_line_lums`.
    amplitude : float or array
        The Q_H-derived amplitude ``reconstruct`` expects: ``log_nion`` [dex
        re photons/s] for the photometry / rest-band channels, linear
        ``nion`` [photons/s] for the line-luminosity channel.
    point : Mapping
        Interpolation point: ``point[name]`` for every ``name`` in
        ``table.axis_names``, including ``neb_logU`` whenever that is one of
        them. A ``point`` with no ``"neb_logU"`` key is fine when
        ``"neb_logU"`` is not a table axis (DIG necessarily inactive then).
    table : NebularGridTable
        The grid to interpolate
        (:func:`~tengri.components.nebular.nebular_grid_precompute.precompute_nebular_grid`).
    neb_dig_frac : float
        DIG mass fraction. [dimensionless, in [0, 1]]
    neb_dig_delta_logU : float
        Offset in ionization parameter for DIG (negative). [dex]

    Returns
    -------
    array
        Whatever ``reconstruct`` returns: the HII-only reconstruction when
        ``neb_dig_frac`` short-circuits, else the linear mix
        ``(1 - f) * hii + f * dig``.

    Notes
    -----
    **JIT-compatible**: yes -- node-exact PCHIP interpolation (inside
    ``reconstruct``) plus a linear mix, both pure ``jnp`` operations. Same
    short-circuit contract as :func:`mix_dig_emission`: a Python-literal
    ``neb_dig_frac == 0.0`` skips the second interpolation entirely (one
    ``reconstruct`` call, the pre-DIG cost); a traced ``neb_dig_frac`` runs
    both.

    References
    ----------
    .. [1] L. M. Haffner et al., "The warm ionized medium in spiral galaxies,"
       Rev. Mod. Phys., 81, 969 (2009).
       https://doi.org/10.1103/RevModPhys.81.969
    .. [2] S. Tacchella et al., "H-alpha emission in local galaxies: star
       formation, time variability, and the diffuse ionized gas," MNRAS, 513,
       2904 (2022). arXiv:2112.00027. https://doi.org/10.1093/mnras/stac818
    """
    # ``point.get(..., 0.0)``, not ``point["neb_logU"]``: DIG mixing is
    # unconditionally wired into every use_grid call site, including tables
    # built with DIG inactive (neb_dig_frac Fixed at its declared 0.0), whose
    # axes may not include "neb_logU" at all (e.g. only met_logzsol free).
    # ``reconstruct`` reads only ``point[name] for name in table.axis_names``,
    # so a "neb_logU" key that names no axis is inert regardless of its
    # value; the 0.0 placeholder is never read in that case. When
    # "neb_logU" IS an axis, the caller (component.py / sed_model.py) always
    # supplies it, so the placeholder never masks a real value.
    neb_logU = jnp.asarray(point.get("neb_logU", 0.0))

    def _evaluate(logU):
        query = dict(point)
        query["neb_logU"] = logU
        return reconstruct(amplitude, query, table)

    def _combine(hii_result, dig_result, frac):
        return _linear_mix(hii_result, dig_result, frac)

    return _mix_dig_backend_evaluations(
        _evaluate, _combine, neb_logU, neb_dig_frac, neb_dig_delta_logU
    )
