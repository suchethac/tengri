# SPDX-License-Identifier: BSD-3-Clause
"""Informative AGN prior penalty terms, transcribed from AGNfitter-rX.

These are the eight optional, composite log-prior penalty terms AGNfitter-rX
adds on top of the per-parameter priors (Uniform/Gaussian/etc.) when the
corresponding ``modelsettings`` flag is enabled (``PRIORS_AGNfitter.py:1-75``,
the ``PRIORS()`` dispatcher). Unlike per-parameter priors, each of these links
**multiple** predicted quantities (e.g. the galaxy-attenuated luminosity
against the starburst-emitted luminosity), so they cannot be expressed as a
bound on a single free parameter.

Each function here is a pure, JIT/grad-safe JAX transcription of one upstream
branch, taking the physical scalar(s) (luminosities/fluxes in erg/s or
erg/s/Hz, magnitudes, redshift) that upstream's own local variables hold
immediately before the prior arithmetic -- the array slicing, band selection,
and de-reddening bookkeeping that upstream performs *to compute* those scalars
from its internal model-dictionary objects is a separate, model-specific
concern (see "Adapters" below), not part of the prior formula itself.

.. list-table:: The eight priors
   :header-rows: 1

   * - Function
     - Upstream
     - Lines
   * - :func:`prior_energy_balance`
     - ``prior_energy_balance``
     - 78-106
   * - :func:`prior_agn_fraction`
     - ``prior_AGNfraction``
     - 109-199
   * - :func:`prior_stellar_mass`
     - ``prior_stellar_mass``
     - 201-210
   * - :func:`prior_ir_syn_fraction`
     - ``prior_IR_SYNfraction``
     - 212-254
   * - :func:`prior_uv_xrays`
     - ``prior_UV_xrays``
     - 256-299
   * - :func:`prior_ir_xrays`
     - ``prior_IR_XRays``
     - 302-324
   * - :func:`prior_midir_uv`
     - ``prior_midIR_UV``
     - 327-357
   * - :func:`prior_low_agn_fraction`
     - ``prior_low_AGNfraction``
     - 360-425

``AGNFITTER_HARD_REJECT`` (below) is upstream's literal sentinel for "outside
the physically-allowed region": a large finite negative number (``-9999``),
**not** ``-inf``. Upstream's own comment marks the ambivalence
(``return -9999 #-np.inf``, ``PRIORS_AGNfitter.py:98``). tengri reproduces the
finite sentinel exactly, since a finite value stays gradient-safe (``-inf``
has an ill-defined gradient at that point) and is what upstream's own
posteriors were actually computed with.

Gaussian normalization
-----------------------
Every Gaussian branch below is evaluated through :func:`gaussian_log_prior`,
which includes the ``log(1/(sqrt(2*pi)*sigma))`` normalization constant that
upstream's own shared ``Gaussian_prior`` helper carries
(``PRIORS_AGNfitter.py:428-430``). The constant is irrelevant to posterior
*shape* at fixed sigma (cancels in MCMC acceptance ratios) but is not
irrelevant to a nested-sampling evidence (Z) integral -- upstream explicitly
supports ``ultranest`` as a sampler, and any comparison of evidences between
models with different sigma needs it.

Adapters
--------
Turning a tengri :class:`~tengri.forward.prediction.Prediction` /
``ForwardState`` into the physical scalars each function above expects
(rest-1500 A disc/galaxy fluxes, nu*L_nu(6 micron) of the torus, L_IR of the
cold dust, ...) is model-specific glue outside the scope of this module. The
fitter hook (``Fitter(..., extra_log_prior=...)``, see
:func:`tengri.inference.loss_functions.build_logprior_fn`) is the place a user
wires a closure that does that extraction and calls the functions below.

Breaking change (this rewrite)
-------------------------------
The previous three functions in this module (``agn_prior_energy_balance``,
``agn_prior_agn_fraction_floor``, ``agn_prior_midir_uv_tie``) did not
reproduce their cited upstream branches (missing the Gaussian normalization;
the energy-balance "flexible" mode applied a Gaussian penalty instead of
upstream's unconditional 0; the AGN-fraction floor compared a bolometric
ratio, not upstream's redshift-dependent rest-1500 A flux-ratio branch) and
had no callers anywhere in tengri (``grep -rn`` outside this file and its own
test returned nothing). They are removed rather than aliased.

Implements the same AGN prior penalties as AGNfitter-rX (Martinez-Ramirez
et al. 2024, A&A, 688, A46, arXiv:2405.12111,
doi:10.1051/0004-6361/202449329); validated against faithful transcriptions
of ``functions/PRIORS_AGNfitter.py`` in
``tests/crossval/test_agn_priors_vs_agnfitter.py``.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

#: Upstream's finite reject sentinel for a physically-disallowed region
#: (``PRIORS_AGNfitter.py:98,192``: ``return -9999 #-np.inf``). A finite value
#: keeps ``jax.grad`` well-defined at the boundary (``-inf`` does not); this is
#: also the literal value upstream's own posteriors were computed with, so it
#: is reproduced exactly rather than "corrected" to ``-jnp.inf``.
AGNFITTER_HARD_REJECT = -9999.0


def gaussian_log_prior(mu, sigma, par):
    r"""Gaussian log-prior density, including the normalization constant.

    .. math::

        \ln P(par) = \ln\!\left(\frac{1}{\sqrt{2\pi}\,\sigma}\right)
            - \frac{1}{2}\left(\frac{par - \mu}{\sigma}\right)^2

    where :math:`\mu` is the mean, :math:`\sigma` the standard deviation, and
    :math:`par` the value at which the density is evaluated (same units for
    all three).

    Parameters
    ----------
    mu : float
        Mean [same units as ``par``].
    sigma : float
        Standard deviation [same units as ``par``]. Must be > 0.
    par : float or jnp.ndarray
        Value(s) at which to evaluate the log-density.

    Returns
    -------
    float or jnp.ndarray
        Log-probability density. Can exceed 0 when ``sigma < 1/sqrt(2*pi)``
        (a density, not a bounded penalty).

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes (smooth everywhere for
    ``sigma > 0``).

    Faithful transcription of upstream's shared ``Gaussian_prior`` helper
    (``PRIORS_AGNfitter.py:428-430``), including the normalization term that
    every prior below needs for evidence (Z) comparisons across different
    sigma (see module docstring).

    References
    ----------
    .. [1] G. Calistro Rivera et al., "AGNfitter: A Bayesian MCMC Approach
       to Fitting Spectral Energy Distributions of AGNs," ApJ, 833, 98
       (2016). arXiv:1606.05648. doi:10.3847/1538-4357/833/1/98.
    """
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    par = jnp.asarray(par)
    return jnp.log(1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)) - 0.5 * ((par - mu) / sigma) ** 2


def _characteristic_uv_magnitude(redshift):
    r"""Characteristic rest-frame UV absolute magnitude of the UV luminosity function.

    .. math::

        M^*(z) = \frac{-35.4\,(1+z)^{0.524}}{1 + (1+z)^{0.678}}

    Parameters
    ----------
    redshift : float
        Source redshift [dimensionless].

    Returns
    -------
    float
        Characteristic absolute UV magnitude [mag].

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes.

    Transcription of the ``characteristic_mag`` expression that appears
    identically at ``PRIORS_AGNfitter.py:172`` (``prior_AGNfraction``) and
    ``:406`` (``prior_low_AGNfraction``). Upstream attributes this fit
    inconsistently -- the inline comment at line 172 cites "Parsa, Dunlop et
    al. 2014" and the (textually identical) comment at line 403-405 cites
    "Parsa, Dunlop et al. 2016" for the same formula. Neither citation
    (journal, volume, DOI/arXiv) appears anywhere in tengri's existing
    bibliography (``src/tengri/citations/references.bib``), so per project
    convention (never write a citation from memory) only the author names and
    upstream's own two conflicting years are recorded here; verify
    independently before citing this formula outside tengri.
    """
    z = jnp.asarray(redshift, dtype=float)
    return -35.4 * (1.0 + z) ** 0.524 / (1.0 + (1.0 + z) ** 0.678)


def prior_energy_balance(
    l_gal_att,
    l_sb_emit,
    mode: Literal["flexible", "restrictive"] = "flexible",
):
    r"""Soft penalty enforcing galaxy-attenuated <= starburst-emitted luminosity.

    Encodes the physical requirement that energy absorbed by dust in the
    galaxy screen must be re-emitted in the infrared: the model cannot emit
    less IR luminosity than it absorbed.

    .. math::

        P = \begin{cases}
        \text{AGNFITTER\_HARD\_REJECT} & L_{sb,emit} < L_{gal,att}
            \quad \text{(both modes)} \\
        0 & L_{sb,emit} \geq L_{gal,att},\ \text{mode = "flexible"} \\
        \mathcal{N}\!\left(0,\ 0.1^2;\ \log_{10}\frac{L_{sb,emit}}{L_{gal,att}}
            \right) & L_{sb,emit} \geq L_{gal,att},\ \text{mode = "restrictive"}
        \end{cases}

    Parameters
    ----------
    l_gal_att : float
        Galaxy-attenuated (dust-absorbed) luminosity [erg/s].
    l_sb_emit : float
        Starburst/cold-dust re-emitted IR luminosity [erg/s].
    mode : {"flexible", "restrictive"}, optional
        ``"flexible"``: only a hard floor -- zero penalty for any excess IR
        emission. ``"restrictive"``: additionally penalizes deviation from
        exact equality with a narrow (sigma=0.1) Gaussian. Default
        ``"flexible"``.

    Returns
    -------
    float or jnp.ndarray
        Log-prior contribution. ``AGNFITTER_HARD_REJECT`` (-9999) if
        ``l_sb_emit < l_gal_att`` in EITHER mode (upstream's ``if`` at
        ``PRIORS_AGNfitter.py:97-98`` is outside the mode branches); else 0.0
        (flexible) or a Gaussian log-density (restrictive).

    Raises
    ------
    ValueError
        If ``mode`` is not ``"flexible"`` or ``"restrictive"``.

    Notes
    -----
    **JIT-compatible**: yes (``mode`` is a Python-level static string, not
    traced). **Grad-compatible**: yes, via ``jnp.where`` (both branches are
    always evaluated so no branch is ever pruned from the trace).

    Faithful transcription of ``prior_energy_balance``
    (``PRIORS_AGNfitter.py:78-106``). A prior implementation of this function
    in tengri (removed by this rewrite) applied the restrictive-mode Gaussian
    in flexible mode too, and never hard-rejected -- see the module
    docstring's "Breaking change" note.

    References
    ----------
    .. [1] L. N. Martinez-Ramirez et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
       (2024). arXiv:2405.12111. doi:10.1051/0004-6361/202449329.
    """
    if mode not in ("flexible", "restrictive"):
        raise ValueError(f"mode must be 'flexible' or 'restrictive', got {mode!r}")
    l_gal_att = jnp.asarray(l_gal_att, dtype=float)
    l_sb_emit = jnp.asarray(l_sb_emit, dtype=float)
    is_rejected = l_sb_emit < l_gal_att
    frac_sb_att_gal = jnp.log10(l_sb_emit / l_gal_att)
    if mode == "flexible":
        physical_value = jnp.zeros_like(frac_sb_att_gal)
    else:
        physical_value = gaussian_log_prior(0.0, 0.1, frac_sb_att_gal)
    return jnp.where(is_rejected, AGNFITTER_HARD_REJECT, physical_value)


def prior_stellar_mass(ga):
    r"""Soft Gaussian prior on the raw galaxy flux-normalization scalar.

    .. math::

        P = \mathcal{N}(4.5,\ 1.5^2;\ GA)

    Parameters
    ----------
    ga : float
        Galaxy (host stellar) flux-normalization scalar, upstream's ``GA``
        -- the log10 amplitude multiplying the galaxy template flux
        (``Lgal = GALAXYFdict[...] * 10**GA``). Dimensionless (a log10
        normalization exponent, not a luminosity).

    Returns
    -------
    float or jnp.ndarray
        Log-prior contribution (Gaussian log-density; can exceed 0).

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes.

    Faithful transcription of ``prior_stellar_mass``
    (``PRIORS_AGNfitter.py:201-210``): ``GA < 3`` upstream's comment reads
    corresponds to :math:`M_* < 10^9\,M_\odot`. Upstream sums this with
    :func:`prior_agn_fraction` under a single ``PRIOR_AGNfraction`` settings
    flag (``prior1 + prior2`` at ``PRIORS_AGNfitter.py:46-48``); tengri keeps
    them as two functions so a caller can enable either independently.

    References
    ----------
    .. [1] L. N. Martinez-Ramirez et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
       (2024). arXiv:2405.12111. doi:10.1051/0004-6361/202449329.
    """
    return gaussian_log_prior(4.5, 1.5, ga)


def prior_agn_fraction(bbb_flux_1500, gal_flux_1500, data_flux_1500, dlum, redshift):
    r"""Soft prior tying the rest-1500 A AGN/galaxy flux ratio to a UV luminosity function.

    .. math::

        \mathrm{AGNfrac}_{1500} = \log_{10}\!\left(\frac{F_{bbb,1500}}
            {F_{gal,1500}}\right)

        m_{\rm abs} = 51.6 - 2.5\log_{10}\!\left(4\pi d_L^2\,F_{data,1500}
            \right)

    with the characteristic magnitude :math:`M^*(z)` of
    :func:`_characteristic_uv_magnitude`. Branch structure:

    .. math::

        P = \begin{cases}
        \mathcal{N}(-2,\ 2^2;\ \mathrm{AGNfrac}_{1500})
            & m_{\rm abs} > M^*(z) - 1 \quad \text{(galaxy regime)} \\
        \text{AGNFITTER\_HARD\_REJECT} & m_{\rm abs} \leq M^*(z) - 1,\
            \mathrm{AGNfrac}_{1500} < 0 \\
        \mathcal{N}(+2,\ 2^2;\ \mathrm{AGNfrac}_{1500})
            & m_{\rm abs} \leq M^*(z) - 1,\ \mathrm{AGNfrac}_{1500} \geq 0
        \end{cases}

    Parameters
    ----------
    bbb_flux_1500 : float
        Model accretion-disc (BBB) flux density at rest-frame 1500 A
        [erg/s/cm^2/Hz].
    gal_flux_1500 : float
        Model galaxy flux density at rest-frame 1500 A [erg/s/cm^2/Hz].
    data_flux_1500 : float
        Observed flux density at rest-frame 1500 A [erg/s/cm^2/Hz].
    dlum : float
        Luminosity distance [cm].
    redshift : float
        Source redshift [dimensionless].

    Returns
    -------
    float or jnp.ndarray
        Log-prior contribution. ``AGNFITTER_HARD_REJECT`` in the bright/QSO
        regime when the AGN is fainter than the galaxy at 1500 A
        (``AGNfrac1500 < 0``); a Gaussian log-density otherwise.

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes, via ``jnp.where``.

    Faithful transcription of ``prior_AGNfraction``
    (``PRIORS_AGNfitter.py:109-199``), specifically its branch tail
    (:170-198`). This operates on a **rest-1500 A flux ratio** gated by a
    **redshift-dependent UV luminosity-function threshold**, not a bolometric
    AGN-fraction floor -- see the module docstring's "Breaking change" note
    for what the previous (removed) tengri implementation did instead.
    Upstream sums this with :func:`prior_stellar_mass` under one settings
    flag; see that function's Notes.

    References
    ----------
    .. [1] L. N. Martinez-Ramirez et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
       (2024). arXiv:2405.12111. doi:10.1051/0004-6361/202449329.
    .. [2] Parsa, S. & Dunlop, J. S. et al. -- characteristic UV
       luminosity-function magnitude fit used by
       :func:`_characteristic_uv_magnitude`; see that function's Notes for
       the citation-year caveat.
    """
    bbb_flux_1500 = jnp.asarray(bbb_flux_1500, dtype=float)
    gal_flux_1500 = jnp.asarray(gal_flux_1500, dtype=float)
    agn_frac_1500 = jnp.log10(bbb_flux_1500 / gal_flux_1500)

    lumfactor = 4.0 * jnp.pi * jnp.asarray(dlum, dtype=float) ** 2
    data_lum_1500 = lumfactor * jnp.asarray(data_flux_1500, dtype=float)
    abs_mag_data = 51.6 - 2.5 * jnp.log10(data_lum_1500)
    characteristic_mag = _characteristic_uv_magnitude(redshift)

    is_galaxy_regime = abs_mag_data > (characteristic_mag - 1.0)
    galaxy_value = gaussian_log_prior(-2.0, 2.0, agn_frac_1500)
    qso_value = gaussian_log_prior(2.0, 2.0, agn_frac_1500)
    is_rejected = agn_frac_1500 < 0.0
    qso_branch = jnp.where(is_rejected, AGNFITTER_HARD_REJECT, qso_value)
    return jnp.where(is_galaxy_regime, galaxy_value, qso_branch)


def prior_low_agn_fraction(bbb_flux_1500, gal_flux_1500, data_flux_1500, dlum, redshift):
    r"""Soft prior favoring a galaxy-dominated rest-1500 A flux ratio.

    .. math::

        P = \begin{cases}
        \mathcal{N}(-2,\ 0.5^2;\ \mathrm{AGNfrac}_{1500})
            & m_{\rm abs} > M^*(z) - 3 \\
        \mathcal{N}(-2,\ 2^2;\ \mathrm{AGNfrac}_{1500}) & \text{otherwise}
        \end{cases}

    with :math:`\mathrm{AGNfrac}_{1500}` and :math:`m_{\rm abs}` as in
    :func:`prior_agn_fraction`.

    Parameters
    ----------
    bbb_flux_1500 : float
        Model accretion-disc (BBB) flux density at rest-frame 1500 A
        [erg/s/cm^2/Hz].
    gal_flux_1500 : float
        Model galaxy flux density at rest-frame 1500 A [erg/s/cm^2/Hz].
    data_flux_1500 : float
        Observed flux density at rest-frame 1500 A [erg/s/cm^2/Hz].
    dlum : float
        Luminosity distance [cm].
    redshift : float
        Source redshift [dimensionless].

    Returns
    -------
    float or jnp.ndarray
        Log-prior contribution: a Gaussian log-density, mean -2 in both
        regimes, narrower (sigma=0.5) when the source is UV-faint relative to
        the (redshift-dependent, -3 mag offset) luminosity-function
        threshold, wider (sigma=2) otherwise. No hard-reject branch (unlike
        :func:`prior_agn_fraction`).

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes, via ``jnp.where``.

    Faithful transcription of ``prior_low_AGNfraction``
    (``PRIORS_AGNfitter.py:360-425``), specifically its branch tail
    (:412-423). Distinct from :func:`prior_agn_fraction`: same mean (-2) in
    both regimes here (vs. -2/+2 there), a -3 mag threshold offset (vs. -1),
    different sigma values (0.5/2 vs. 2/2), and no hard-reject branch at all.
    A prior tengri implementation of the AGN-fraction floor (removed by this
    rewrite) had accidentally borrowed this function's mu=-2, sigma=0.5
    constants while citing ``prior_AGNfraction`` -- see the module
    docstring's "Breaking change" note.

    References
    ----------
    .. [1] L. N. Martinez-Ramirez et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
       (2024). arXiv:2405.12111. doi:10.1051/0004-6361/202449329.
    .. [2] Parsa, S. & Dunlop, J. S. et al. -- see
       :func:`_characteristic_uv_magnitude` Notes for the citation-year
       caveat.
    """
    bbb_flux_1500 = jnp.asarray(bbb_flux_1500, dtype=float)
    gal_flux_1500 = jnp.asarray(gal_flux_1500, dtype=float)
    agn_frac_1500 = jnp.log10(bbb_flux_1500 / gal_flux_1500)

    lumfactor = 4.0 * jnp.pi * jnp.asarray(dlum, dtype=float) ** 2
    data_lum_1500 = lumfactor * jnp.asarray(data_flux_1500, dtype=float)
    abs_mag_data = 51.6 - 2.5 * jnp.log10(data_lum_1500)
    characteristic_mag = _characteristic_uv_magnitude(redshift)

    is_narrow_regime = abs_mag_data > (characteristic_mag - 3.0)
    narrow_value = gaussian_log_prior(-2.0, 0.5, agn_frac_1500)
    wide_value = gaussian_log_prior(-2.0, 2.0, agn_frac_1500)
    return jnp.where(is_narrow_regime, narrow_value, wide_value)


def prior_ir_syn_fraction(
    data_flux_rad, data_nu_rad, data_flux_ir, data_nu_ir, sb_flux_ir, syn_flux_ir
):
    r"""Soft prior on the synchrotron/starburst flux fraction at the cold-dust peak.

    .. math::

        F_{\rm syn,exp} = F_{\rm rad}\left(\frac{\nu_{IR}}{\nu_{\rm rad}}
            \right)^{-0.75}

        \mathrm{SYNfrac}_{IR} = \log_{10}\!\left(\frac{F_{\rm syn}(\nu_{IR})}
            {F_{\rm sb}(\nu_{IR})}\right)

        P = \begin{cases}
        \mathcal{N}(+2,\ 2^2;\ \mathrm{SYNfrac}_{IR}) & F_{IR}/F_{\rm syn,exp}
            < 2 \\
        \mathcal{N}(-2,\ 2^2;\ \mathrm{SYNfrac}_{IR}) & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    data_flux_rad : float
        Observed radio flux density (highest-frequency detection in the
        0.1-10 GHz rest-frame window) [erg/s/cm^2/Hz].
    data_nu_rad : float
        log10(frequency) [log10(Hz)] of ``data_flux_rad``.
    data_flux_ir : float
        Observed flux density at the cold-dust spectral peak
        [erg/s/cm^2/Hz].
    data_nu_ir : float
        log10(frequency) [log10(Hz)] of ``data_flux_ir``.
    sb_flux_ir : float
        Model starburst/cold-dust flux density at ``data_nu_ir``
        [erg/s/cm^2/Hz].
    syn_flux_ir : float
        Model synchrotron flux density at ``data_nu_ir`` [erg/s/cm^2/Hz].

    Returns
    -------
    float or jnp.ndarray
        Log-prior contribution: a Gaussian log-density with mean +2 (radio
        power law can already explain the IR peak -- favor synchrotron) or
        -2 (it cannot -- favor starburst), sigma=2 either way.

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes, via ``jnp.where``.

    Faithful transcription of ``prior_IR_SYNfraction``
    (``PRIORS_AGNfitter.py:212-254``), specifically its arithmetic core
    (:237-254`); the caller-level early-return-0 when no radio/IR data is
    present (``:225-226,:232-233``) is a data-availability check that belongs
    to the adapter computing these inputs, not to this formula. Upstream's
    branch is undefined exactly AT the ``== 2`` boundary (neither ``if`` nor
    ``elif`` fires, an upstream gap); this transcription resolves the tie to
    the ``< 2`` branch's complement (``>= 2``) rather than reproducing an
    undefined return.

    References
    ----------
    .. [1] L. N. Martinez-Ramirez et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
       (2024). arXiv:2405.12111. doi:10.1051/0004-6361/202449329.
    """
    data_flux_rad = jnp.asarray(data_flux_rad, dtype=float)
    data_nu_rad = jnp.asarray(data_nu_rad, dtype=float)
    data_flux_ir = jnp.asarray(data_flux_ir, dtype=float)
    data_nu_ir = jnp.asarray(data_nu_ir, dtype=float)
    sb_flux_ir = jnp.asarray(sb_flux_ir, dtype=float)
    syn_flux_ir = jnp.asarray(syn_flux_ir, dtype=float)

    syn_exp_ir = data_flux_rad * (10.0**data_nu_ir / 10.0**data_nu_rad) ** (-0.75)
    syn_frac_ir = jnp.log10(syn_flux_ir / sb_flux_ir)

    is_radio_explained = (data_flux_ir / syn_exp_ir) < 2.0
    mu = jnp.where(is_radio_explained, 2.0, -2.0)
    return gaussian_log_prior(mu, 2.0, syn_frac_ir)


def prior_uv_xrays(log_l2500a_data, log_l2kev_data):
    r"""Soft prior tying the accretion-disc UV luminosity to the observed X-ray flux.

    .. math::

        \log_{10}L_{2500\,\rm A}^{\rm pred} = \frac{\log_{10}L_{2\,\rm keV}
            - \gamma}{\beta},\quad \beta = 0.643,\ \gamma = 6.8734

        P = \mathcal{N}\!\left(0,\ 0.4^2;\ \log_{10}L_{2500\,\rm A}^{\rm data}
            - \log_{10}L_{2500\,\rm A}^{\rm pred}\right)

    Parameters
    ----------
    log_l2500a_data : float
        log10 of the model's dereddened accretion-disc luminosity density at
        rest-frame 2500 A [log10(erg/s/Hz)].
    log_l2kev_data : float
        log10 of the observed luminosity density at rest-frame 2 keV
        [log10(erg/s/Hz)].

    Returns
    -------
    float or jnp.ndarray
        Log-prior contribution (Gaussian log-density).

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes.

    Faithful transcription of ``prior_UV_xrays``
    (``PRIORS_AGNfitter.py:256-299``), specifically the ``alpha_OX`` inverse
    relation and Gaussian penalty (:258-265,:291-297`). Upstream's own
    in-line docstring for ``alpha_OX`` attributes the (beta, gamma)
    coefficients to "Lusso&Risaliti +16 gives beta=[0.6-0.65], gamma=[7-8]"
    (:260`), while the caller-site comment two functions up names "the
    alpha_ox correlation by Just et al. 2007" (:64`) for the same relation --
    upstream cites both for one formula; both are reproduced faithfully
    below rather than resolved.

    References
    ----------
    .. [1] E. Lusso & G. Risaliti, 2016, ApJ, 819, 154. (beta, gamma range
       cited by upstream's ``alpha_OX`` docstring; exact DOI/arXiv not
       independently re-verified beyond tengri's existing xray-component
       citation of this same journal reference,
       ``src/tengri/components/xray/xray.py``.)
    .. [2] D. W. Just et al., 2007, ApJ, 665, 1004. (named by upstream's
       caller-site comment for this same correlation,
       ``PRIORS_AGNfitter.py:64``; title and DOI disagree between two
       existing citations of this same journal reference inside tengri's own
       ``xray.py``, so neither is repeated here -- see
       :func:`prior_ir_xrays` Notes.)
    """
    beta = 0.643
    gamma = 6.8734
    log_l2500a_model = (jnp.asarray(log_l2kev_data, dtype=float) - gamma) / beta
    ratio = jnp.asarray(log_l2500a_data, dtype=float) - log_l2500a_model
    return gaussian_log_prior(0.0, 0.4, ratio)


def prior_ir_xrays(log_f2_10kev_data, nulnu_6um):
    r"""Soft prior tying the torus mid-IR flux to the observed 2-10 keV flux.

    .. math::

        x = \log_{10}\!\left(\frac{\nu L_\nu(6\,\mu{\rm m})}{10^{41}\,
            {\rm erg/s}}\right)

        \log_{10}f_{2-10\,\rm keV}^{\rm pred} = 22.9494264 + 1.024\,x -
            0.047\,x^2

        P = \mathcal{N}\!\left(0,\ 0.5^2;\ \log_{10}f_{2-10\,\rm keV}^{\rm
            data} - \log_{10}f_{2-10\,\rm keV}^{\rm pred}\right)

    Parameters
    ----------
    log_f2_10kev_data : float
        log10 of the observed monochromatic flux (luminosity) at the
        2-10 keV band center [log10(erg/s/Hz)].
    nulnu_6um : float
        nu*L_nu of the torus at rest-frame 6 microns [erg/s].

    Returns
    -------
    float or jnp.ndarray
        Log-prior contribution (Gaussian log-density).

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes.

    Faithful transcription of ``prior_IR_XRays``
    (``PRIORS_AGNfitter.py:302-324``), specifically the Stern (2015) mid-IR--
    X-ray correlation and Gaussian penalty (:309-322`).

    References
    ----------
    .. [1] D. Stern, mid-infrared--X-ray luminosity correlation for AGN
       (Stern 2015), as cited by upstream's in-line comment at
       ``PRIORS_AGNfitter.py:310``. Volume/page/DOI not found in tengri's
       existing bibliography (``src/tengri/citations/references.bib``) for
       this task and so are left absent rather than guessed (per project
       citation policy); verify independently before citing this formula
       elsewhere.
    """
    x = jnp.log10(jnp.asarray(nulnu_6um, dtype=float) / 1e41)
    log_f_2_10kev_model = 22.9494264 + 1.024 * x - 0.047 * x**2
    ratio = jnp.asarray(log_f2_10kev_data, dtype=float) - log_f_2_10kev_model
    return gaussian_log_prior(0.0, 0.5, ratio)


def prior_midir_uv(log_l2500a_bbmodel, nulnu_6um):
    r"""Soft prior tying the accretion-disc UV luminosity to the torus mid-IR flux.

    Predicts an accretion-disc UV luminosity FROM the torus's 6-micron
    emission by composing the Stern (2015) mid-IR--X-ray correlation with the
    Just+2007 alpha_ox X-ray--UV correlation -- **not** a direct
    :math:`L_{\rm mir} = L_{\rm uv}` comparison.

    .. math::

        x = \log_{10}\!\left(\nu L_\nu(6\,\mu{\rm m})\ [{\rm erg/s}]\right)
            - 27.30103

        \log_{10}L_{2500\,\rm A}^{\rm pred} = \frac{16.2530786 + 1.024\,x -
            0.047\,x^2}{0.643}

        P = \mathcal{N}\!\left(0,\ 0.6^2;\ \log_{10}L_{2500\,\rm A}^{\rm bb}
            - \log_{10}L_{2500\,\rm A}^{\rm pred}\right)

    Parameters
    ----------
    log_l2500a_bbmodel : float
        log10 of the model's (unreddened) accretion-disc luminosity density
        at rest-frame 2500 A [log10(erg/s/Hz)].
    nulnu_6um : float
        nu*L_nu of the torus at rest-frame 6 microns [erg/s].

    Returns
    -------
    float or jnp.ndarray
        Log-prior contribution (Gaussian log-density).

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes.

    Faithful transcription of ``prior_midIR_UV``
    (``PRIORS_AGNfitter.py:327-357``), specifically the composite
    correlation and Gaussian penalty (:333-355`). The sigma=0.6 combines the
    mid-IR--X-ray scatter (~0.5, :func:`prior_ir_xrays`) with the alpha_ox
    scatter (~0.1, per upstream's in-line comment at
    ``PRIORS_AGNfitter.py:354``). A prior tengri implementation (removed by
    this rewrite) compared :math:`L_{\rm mir}` and :math:`L_{\rm uv}`
    directly -- see the module docstring's "Breaking change" note; at equal
    nominal log-luminosities that comparison and this one disagree by
    hundreds of dex, because equal luminosities are nowhere near this
    correlation's mean relation.

    References
    ----------
    .. [1] D. Stern (Stern 2015) -- mid-IR--X-ray correlation; see
       :func:`prior_ir_xrays` Notes for the citation-verification caveat.
    .. [2] D. W. Just et al., 2007, ApJ, 665, 1004 -- alpha_ox correlation;
       see :func:`prior_uv_xrays` Notes for the citation-verification
       caveat.
    """
    x = jnp.log10(jnp.asarray(nulnu_6um, dtype=float)) - 27.30103
    log_l2500a_tomodel = (16.2530786 + 1.024 * x - 0.047 * x**2) / 0.643
    ratio = jnp.asarray(log_l2500a_bbmodel, dtype=float) - log_l2500a_tomodel
    return gaussian_log_prior(0.0, 0.6, ratio)
