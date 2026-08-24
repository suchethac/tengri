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
"""

import jax.numpy as jnp


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
    .. [3] S. P. Reynolds, "Supernova Remnants as Cosmic Ray Sources," ApJ,
       282, 191 (1984). https://doi.org/10.1086/162189

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

    # HII component (standard ionization parameter)
    neb_hii = nebular_backend.predict_nebular_sed(neb_logU=neb_logU, **common_kw)

    # Short-circuit: when neb_dig_frac is a Python literal 0.0, skip the extra
    # forward pass entirely.  Under JIT with a traced value both passes execute.
    if isinstance(neb_dig_frac, (int, float)) and neb_dig_frac == 0.0:
        return neb_hii

    # DIG component (lower ionization parameter)
    logU_dig = neb_logU + neb_dig_delta_logU
    neb_dig = nebular_backend.predict_nebular_sed(neb_logU=logU_dig, **common_kw)

    # Linear mix
    return (1.0 - neb_dig_frac) * neb_hii + neb_dig_frac * neb_dig
