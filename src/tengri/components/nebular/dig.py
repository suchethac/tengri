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

from tengri.utils.scale import pow10


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


def _log10_weighted_mix(log_hii, log_dig, frac):
    r"""Float32-safe log-domain analog of :func:`_linear_mix` (#2222, #2269).

    ``log10((1 - frac) * 10**log_hii + frac * 10**log_dig)``, computed by
    factoring out the larger of the two magnitudes so neither term is
    exponentiated at its own magnitude:

    .. math::

        \ell_{\max} = \max(\ell_{\mathrm{hii}}, \ell_{\mathrm{dig}})

        R = \ell_{\max} + \log_{10}\!\left[(1 - f)\,
            10^{\ell_{\mathrm{hii}} - \ell_{\max}} +
            f\, 10^{\ell_{\mathrm{dig}} - \ell_{\max}}\right]

    This is the mixing step :func:`reconstruct_nebular_line_log_lums
    <tengri.components.nebular.nebular_grid_precompute.reconstruct_nebular_line_log_lums>`
    needs: that function's single-lookup form already carries a line
    luminosity (~1e40 erg/s) as an exponent rather than a value so it never
    overflows float32 (#1859); a linear ``(1 - f) * hii + f * dig`` mix of two
    such lookups would reintroduce exactly the overflow the log carrier
    exists to avoid, so the mix itself has to stay in log space too.

    The mass fraction ``frac`` enters the factored sum linearly, NOT as a
    ``log10(frac)`` addend: an earlier version of this helper built
    ``log10(1 - frac) + log_hii`` / ``log10(frac) + log_dig`` and combined
    them with a log-sum-exp, which is finite in the forward direction but
    gives a NaN gradient w.r.t. ``frac`` at exactly ``frac == 0`` or
    ``frac == 1`` (``d/dfrac log10(frac)`` is ``+-inf`` there, multiplied by
    the log-sum-exp's own zero weight for the dropped term -- ``0 * inf``).
    Those edges are reachable, not measure-zero: the declared
    ``neb_dig_frac`` prior is ``Uniform(0, 1)``, and its unconstrained
    sampler coordinate saturates to exactly ``1.0`` for any ``|z| >= 9`` in
    float64 (``|z| >= 8`` in float32) -- ordinary during HMC/NUTS warmup.
    Keeping ``frac`` linear inside the sum avoids ever differentiating
    ``log10(frac)``, so the gradient is analytic-exact at every edge
    (measured 1.970e-16 max relative error vs. a float64 linear reference
    over 2000 trials, and exact agreement with the analytic derivative at
    ``frac`` in ``{0, 1e-8, 0.3, 1 - 1e-8, 1}``).

    Parameters
    ----------
    log_hii, log_dig : array_like
        log10 magnitudes of the HII and DIG evaluations [dex], same shape.
    frac : float
        DIG mass fraction. [dimensionless, in [0, 1]]

    Returns
    -------
    ndarray
        log10 of the mixed magnitude [dex]. ``-inf`` when both ``log_hii``
        and ``log_dig`` are ``-inf`` (no NaN: the offset subtraction is
        skipped when the offset itself is non-finite).

    Notes
    -----
    **JIT-compatible / gradient-safe**: yes -- the offset-and-exponentiate
    step is the same factoring :func:`~tengri.utils.scale.log10_add` uses,
    and ``frac`` is never logarithmed, so the gradient w.r.t. ``frac`` is
    finite everywhere on ``[0, 1]``, including both endpoints.
    """
    log_hii = jnp.asarray(log_hii)
    log_dig = jnp.asarray(log_dig)
    frac = jnp.asarray(frac)
    offset = jnp.maximum(log_hii, log_dig)
    # When both inputs are -inf, offset is -inf too, and log_hii - offset
    # would be `-inf - (-inf)` = NaN. Route the subtraction through a finite
    # stand-in offset in that case only (never used to compute the returned
    # value): pow10(-inf - 0.0) = pow10(-inf) = 0.0 for both terms, the
    # weighted sum is exactly 0.0, and log10(0.0) = -inf, not NaN.
    safe_offset = jnp.where(jnp.isfinite(offset), offset, 0.0)
    weighted_sum = (1.0 - frac) * pow10(log_hii - safe_offset) + frac * pow10(
        log_dig - safe_offset
    )
    return safe_offset + jnp.log10(weighted_sum)


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
    reconstruct, amplitude, point, table, neb_dig_frac, neb_dig_delta_logU, *, log_domain=False
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

    .. math::

        R_{\mathrm{total}} = (1 - f_{\mathrm{DIG}}) \,
            \mathrm{reconstruct}(A, \log U) +
            f_{\mathrm{DIG}} \, \mathrm{reconstruct}(A, \log U + \Delta \log U)

    where :math:`\mathrm{reconstruct}` is ``reconstruct``, :math:`A` is
    ``amplitude``, :math:`f_{\mathrm{DIG}}` is ``neb_dig_frac``
    [dimensionless, in 0 to 1], and :math:`\Delta \log U` is
    ``neb_dig_delta_logU`` [dex].

    Parameters
    ----------
    reconstruct : callable
        ``reconstruct(amplitude, point, table) -> ndarray``. One of
        :func:`~tengri.components.nebular.nebular_grid_precompute.reconstruct_nebular_phot`,
        :func:`~tengri.components.nebular.nebular_grid_precompute.reconstruct_nebular_restband`
        (both ``ndarray, shape (n_filter,)``, linear L_nu -- pass with
        ``log_domain=False``), or
        :func:`~tengri.components.nebular.nebular_grid_precompute.reconstruct_nebular_line_log_lums`
        (``ndarray, shape (n_lines,)``, log10 erg/s -- pass with
        ``log_domain=True``). The plain linear
        :func:`~tengri.components.nebular.nebular_grid_precompute.reconstruct_nebular_line_lums`
        is float32-unsafe by construction (its own docstring) and should not
        be mixed here; use the log form instead.
    amplitude : float or array_like, shape ()
        The Q_H-derived amplitude every ``reconstruct`` variant above expects:
        ``log_nion`` [dex re photons/s].
    point : Mapping
        Interpolation point: ``point[name]`` for every ``name`` in
        ``table.axis_names``, including ``neb_logU`` whenever that is one of
        them. A ``point`` with no ``"neb_logU"`` key is fine only when
        ``"neb_logU"`` is not a table axis (DIG necessarily inactive then);
        when it is an axis, a missing ``"neb_logU"`` raises.
    table : NebularGridTable
        The grid to interpolate
        (:func:`~tengri.components.nebular.nebular_grid_precompute.precompute_nebular_grid`).
    neb_dig_frac : float
        DIG mass fraction. [dimensionless, in [0, 1]]
    neb_dig_delta_logU : float
        Offset in ionization parameter for DIG (negative). [dex]
    log_domain : bool, optional
        ``False`` (default) mixes ``reconstruct``'s return value linearly
        (:func:`_linear_mix`) -- correct for the photometry / rest-band
        channels, whose linear L_nu never leaves float32 range. ``True``
        mixes it as a log10 magnitude (:func:`_log10_weighted_mix`) --
        required for
        :func:`~tengri.components.nebular.nebular_grid_precompute.reconstruct_nebular_line_log_lums`,
        whose linear form (~1e40 erg/s) overflows float32 (#2269): mixing the
        log10 values directly, rather than exponentiating each to mix and
        re-logging, keeps every intermediate in range.

    Returns
    -------
    ndarray, shape (n_filter,) or (n_lines,)
        Whatever ``reconstruct`` returns: the HII-only reconstruction when
        ``neb_dig_frac`` short-circuits, else the mix (linear or log10
        magnitude, per ``log_domain``) of ``(1 - f) * hii + f * dig``.

    Raises
    ------
    KeyError
        If ``"neb_logU"`` is one of ``table.axis_names`` but ``point`` does
        not carry it: every axis the table was built over must have a
        matching entry in the query point (#2222 review I1). A caller must
        merge the model's Fixed values in first (mirrors ``predict_state``'s
        ``full_params = {**fixed_values, **params}``) rather than rely on a
        registry-default placeholder, which would substitute the wrong value
        for a model whose Fixed pin differs from the default.

    Notes
    -----
    **JIT-compatible**: yes -- node-exact PCHIP interpolation (inside
    ``reconstruct``) plus a linear or log10-domain mix (per ``log_domain``),
    both pure ``jnp`` operations. Same short-circuit contract as
    :func:`mix_dig_emission`: a Python-literal ``neb_dig_frac == 0.0`` skips
    the second interpolation entirely (one ``reconstruct`` call, the pre-DIG
    cost); a traced ``neb_dig_frac`` runs both.

    References
    ----------
    .. [1] L. M. Haffner et al., "The warm ionized medium in spiral galaxies,"
       Rev. Mod. Phys., 81, 969 (2009).
       https://doi.org/10.1103/RevModPhys.81.969
    .. [2] S. Tacchella et al., "H-alpha emission in local galaxies: star
       formation, time variability, and the diffuse ionized gas," MNRAS, 513,
       2904 (2022). arXiv:2112.00027. https://doi.org/10.1093/mnras/stac818
    """
    # A presence check keyed on table.axis_names, not a placeholder sentinel
    # (#2222 review I1): when "neb_logU" IS an axis, every caller must supply
    # it (a Fixed value that is absent silently substituted a wrong default
    # before this fix -- 9.3e-1 measured relative error). When it is NOT an
    # axis, no value is read downstream (reconstruct() only reads
    # point[name] for name in table.axis_names), so the shift is skipped
    # rather than requiring a value nothing will use.
    if "neb_logU" in table.axis_names:
        if "neb_logU" not in point:
            raise KeyError(
                "mix_dig_grid_reconstruction: 'neb_logU' is one of "
                "table.axis_names but is missing from `point` "
                f"(point keys: {sorted(point)!r}). Every axis name in "
                "table.axis_names must have a matching entry in `point` -- "
                "merge the model's Fixed values in first (e.g. "
                "{**self.spec.get_fixed_values(), **params}), do not rely on "
                "a registry-default placeholder."
            )
        neb_logU = jnp.asarray(point["neb_logU"])
    else:
        neb_logU = jnp.asarray(0.0)

    def _evaluate(logU):
        query = dict(point)
        query["neb_logU"] = logU
        return reconstruct(amplitude, query, table)

    def _combine(hii_result, dig_result, frac):
        if log_domain:
            return _log10_weighted_mix(hii_result, dig_result, frac)
        return _linear_mix(hii_result, dig_result, frac)

    return _mix_dig_backend_evaluations(
        _evaluate, _combine, neb_logU, neb_dig_frac, neb_dig_delta_logU
    )
