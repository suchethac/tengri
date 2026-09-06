# SPDX-License-Identifier: BSD-3-Clause
"""Informative AGN prior penalty terms, implementing the same physics as AGNfitter-rX.

These are the eight optional, composite log-prior penalty terms AGNfitter-rX
adds on top of the per-parameter priors (Uniform/Gaussian/etc.) when the
corresponding ``modelsettings`` flag is enabled (``PRIORS_AGNfitter.py:1-75``,
the ``PRIORS()`` dispatcher). Unlike per-parameter priors, each of these links
**multiple** predicted quantities (e.g. the galaxy-attenuated luminosity
against the starburst-emitted luminosity), so they cannot be expressed as a
bound on a single free parameter.

Each function here is a pure, JIT/grad-safe JAX implementation of one upstream
branch's physics, taking the physical scalar(s) (luminosities/fluxes in erg/s or
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
cold dust, ...) is model-specific glue. Two routes, for two different jobs:

- :func:`agnfitter_priors` -- the convenience adapter, built on the public,
  rich :meth:`~tengri.forward.sed_model.SEDModel.predict` /
  :attr:`Prediction.sed.components
  <tengri.forward.prediction.SEDProperties.components>` surface. Computes
  every prior's physical inputs from that dict plus (for the priors that
  compare against actual data, not just model self-consistency) explicit
  data keyword arguments, and returns ``(total_log_prior, breakdown_dict)``.
  **Not JIT-compatible** (``Prediction`` is a Python-cached exploration
  object per its own docstring) -- for post-fit inspection / reporting, one
  prediction at a time.
- ``Fitter(..., extra_log_prior=...)`` (see
  :func:`tengri.inference.loss_functions.build_logprior_fn` and
  :func:`~tengri.inference.loss_functions.build_loss_fn`) -- the JIT-safe
  fitting hook. A user's closure receives ``(params, state)`` where ``state``
  is the internal ``ForwardState`` (``model.predict_state(params)``, JIT/grad
  safe) and calls the prior functions directly on ``state.derived`` keys;
  this is the route that actually reaches MAP/VI/MCMC/nested-sampling
  optimization.

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
doi:10.1051/0004-6361/202449329); validated against independent
reimplementations of ``functions/PRIORS_AGNfitter.py`` (the validation
oracle) in ``tests/crossval/test_agn_priors_vs_agnfitter.py``.
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

    Implements the same density as upstream's shared ``Gaussian_prior`` helper
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

    Implements the same ``characteristic_mag`` expression that appears
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

    Implements the same prior as upstream's ``prior_energy_balance``
    (``PRIORS_AGNfitter.py:78-106``), validated against it. A prior implementation of this function
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

    Implements the same prior as upstream's ``prior_stellar_mass``
    (``PRIORS_AGNfitter.py:201-210``), validated against it: ``GA < 3`` upstream's comment reads
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
        Model accretion-disc (BBB) brightness at rest-frame 1500 A. Only the
        ratio ``bbb_flux_1500 / gal_flux_1500`` enters this prior (see
        ``AGNfrac_1500`` above), so this need not be a flux density in the
        strict [erg/s/cm^2/Hz] sense -- any common normalization is accepted
        (flux density, specific luminosity L_nu [erg/s/Hz], or another
        quantity proportional to those), **provided ``gal_flux_1500`` shares
        the same one**. The public :func:`agnfitter_priors` adapter passes
        specific luminosity (L_nu), not flux density.
    gal_flux_1500 : float
        Model galaxy brightness at rest-frame 1500 A, in the same units and
        normalization as ``bbb_flux_1500`` -- see that parameter's Notes.
    data_flux_1500 : float
        Observed flux density at rest-frame 1500 A [erg/s/cm^2/Hz]. Unlike
        ``bbb_flux_1500``/``gal_flux_1500`` this enters only through
        ``abs_mag_data`` (an absolute magnitude, not a ratio), so it MUST be
        a genuine physical flux density -- see the equation above.
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

    Implements the same prior as upstream's ``prior_AGNfraction``
    (``PRIORS_AGNfitter.py:109-199``), validated against it, specifically its branch tail
    (:170-198`). This operates on a **rest-1500 A flux ratio** gated by a
    **redshift-dependent UV luminosity-function threshold**, not a bolometric
    AGN-fraction floor -- see the module docstring's "Breaking change" note
    for what the previous (removed) tengri implementation did instead.
    Upstream sums this with :func:`prior_stellar_mass` under one settings
    flag; see that function's Notes.

    Upstream's caller (``PRIORS_AGNfitter.py:177-178``) applies
    ``if BB==0: bbb_flux_1500Angs /= 4*pi*dlum**2`` before computing
    ``AGNfrac1500`` -- a model-bookkeeping detail of the ``R06``/``THB21``
    accretion-disc normalization convention (``BB`` there is the disc's log10
    flux-normalization scalar, and ``BB==0`` flags "already physical, not a
    fittable amplitude"), not a physics term of this prior. Because only the
    ``bbb_flux_1500 / gal_flux_1500`` ratio enters ``AGNfrac1500`` (see
    Parameters above), that branch is intentionally not reproduced here
    regardless of which disc model or normalization convention the caller
    uses; the caller supplying ``bbb_flux_1500``/``gal_flux_1500`` is
    responsible for using one common normalization for the pair.

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
        Model accretion-disc (BBB) brightness at rest-frame 1500 A. Only the
        ratio ``bbb_flux_1500 / gal_flux_1500`` enters this prior (same
        ``AGNfrac_1500`` as :func:`prior_agn_fraction`), so this need not be
        a flux density in the strict [erg/s/cm^2/Hz] sense -- any common
        normalization is accepted (flux density, specific luminosity L_nu
        [erg/s/Hz], or another quantity proportional to those), **provided
        ``gal_flux_1500`` shares the same one**. See
        :func:`prior_agn_fraction`'s Parameters for the full discussion; the
        public :func:`agnfitter_priors` adapter passes specific luminosity
        (L_nu), not flux density.
    gal_flux_1500 : float
        Model galaxy brightness at rest-frame 1500 A, in the same units and
        normalization as ``bbb_flux_1500`` -- see that parameter's Notes.
    data_flux_1500 : float
        Observed flux density at rest-frame 1500 A [erg/s/cm^2/Hz]. Unlike
        ``bbb_flux_1500``/``gal_flux_1500`` this enters only through
        ``abs_mag_data`` (an absolute magnitude, not a ratio), so it MUST be
        a genuine physical flux density.
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

    Implements the same prior as upstream's ``prior_low_AGNfraction``
    (``PRIORS_AGNfitter.py:360-425``), validated against it, specifically its branch tail
    (:412-423). Distinct from :func:`prior_agn_fraction`: same mean (-2) in
    both regimes here (vs. -2/+2 there), a -3 mag threshold offset (vs. -1),
    different sigma values (0.5/2 vs. 2/2), and no hard-reject branch at all.
    A prior tengri implementation of the AGN-fraction floor (removed by this
    rewrite) had accidentally borrowed this function's mu=-2, sigma=0.5
    constants while citing ``prior_AGNfraction`` -- see the module
    docstring's "Breaking change" note.

    Unlike :func:`prior_agn_fraction`, upstream's ``prior_low_AGNfraction``
    (``PRIORS_AGNfitter.py:360-425``) has no ``if BB==0`` caller-side
    normalization branch on ``bbb_flux_1500Angs`` at all (verified by reading
    the full function body: no such conditional appears there) -- that branch
    is specific to ``prior_AGNfraction``; see that function's Notes.

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

    Implements the same prior as upstream's ``prior_IR_SYNfraction``
    (``PRIORS_AGNfitter.py:212-254``), validated against it, specifically its arithmetic core
    (:237-254`); the caller-level early-return-0 when no radio/IR data is
    present (``:225-226,:232-233``) is a data-availability check that belongs
    to the adapter computing these inputs, not to this formula. Upstream's
    branch is undefined exactly AT the ``== 2`` boundary (neither ``if`` nor
    ``elif`` fires, an upstream gap); this implementation resolves the tie to
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

    Implements the same prior as upstream's ``prior_UV_xrays``
    (``PRIORS_AGNfitter.py:256-299``), validated against it, specifically the ``alpha_OX`` inverse
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


def _x_from_nulnu_6um(nulnu_6um):
    r"""Stern (2015) mid-IR correlation variable, from nu*L_nu at 6 microns.

    .. math::

        x = \log_{10}\!\left(\frac{\nu L_\nu(6\,\mu{\rm m})}{10^{41}\,
            {\rm erg/s}}\right)

    Parameters
    ----------
    nulnu_6um : float
        nu*L_nu of the torus at rest-frame 6 microns [erg/s].

    Returns
    -------
    float or jnp.ndarray
        Dimensionless log-luminosity variable ``x``.

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes.

    Shared by :func:`prior_ir_xrays` and :func:`prior_midir_uv`, both built on
    the Stern (2015) mid-IR--X-ray correlation (``PRIORS_AGNfitter.py:309``,
    ``:333``) evaluated at the SAME physical quantity -- upstream's own two
    call sites compute it in what look like different unit conventions but
    are algebraically the same relation:

    - ``prior_IR_XRays`` (:302-324`) explicitly forms
      ``nuLnu_6microns = 10**13.69897 * tor_flux_6microns * lumfactor`` (i.e.
      nu*L_nu, erg/s) THEN ``x = log10(nuLnu_6microns/1e41)``
      (:308-309`, ``= log10(nuLnu_6microns) - 41``).
    - ``prior_midIR_UV`` (:327-357`) instead uses
      ``L_nu_6um = tor_flux_6microns * lumfactor`` directly (a SPECIFIC
      luminosity, erg/s/Hz, NOT multiplied by the 6-micron frequency) and
      computes ``x = log10(L_nu_6um) - 27.30103`` (:333`).

    These agree because ``13.69897 + 27.30103 = 41`` exactly: writing
    ``nuLnu_6um = nu_6um * L_nu_6um`` with
    :math:`\nu_{6\mu m} = 10^{13.69897}\,{\rm Hz}` (``PRIORS_AGNfitter.py:306,
    331``),

    .. math::

        \log_{10}(L_{\nu,6\mu m}) - 27.30103
            = \log_{10}(\nu L_{\nu,6\mu m}) - 13.69897 - 27.30103
            = \log_{10}(\nu L_{\nu,6\mu m}) - 41

    is the SAME ``x`` as ``prior_IR_XRays``'s. A previous version of
    :func:`prior_midir_uv` (fixed by this revision) took ``nulnu_6um`` (the
    SAME nu*L_nu input as :func:`prior_ir_xrays`, per its own docstring) but
    then subtracted the ``L_nu``-calibrated constant 27.30103 directly from
    ``log10(nulnu_6um)`` without the ``-13.69897`` frequency correction --
    off by ``log10(nu_6um) = 13.69897`` in ``x`` (before squaring), verified
    against upstream's own two-different-looking formulas in
    ``tests/crossval/test_agn_priors_vs_agnfitter.py``. tengri standardizes on
    nu*L_nu [erg/s] as the one public input for both functions; this helper is
    the single place that conversion happens, so the two priors cannot drift
    apart again.
    """
    return jnp.log10(jnp.asarray(nulnu_6um, dtype=float) / 1e41)


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

    Implements the same prior as upstream's ``prior_IR_XRays``
    (``PRIORS_AGNfitter.py:302-324``), validated against it, specifically the Stern (2015) mid-IR--
    X-ray correlation and Gaussian penalty (:309-322`). ``x`` is computed by
    :func:`_x_from_nulnu_6um`, shared with :func:`prior_midir_uv` -- see that
    helper's Notes for why the two priors must agree on ``x`` for the same
    physical input.

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
    x = _x_from_nulnu_6um(nulnu_6um)
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

        x = \log_{10}\!\left(\frac{\nu L_\nu(6\,\mu{\rm m})\ [{\rm erg/s}]}
            {10^{41}\,{\rm erg/s}}\right)

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
        nu*L_nu of the torus at rest-frame 6 microns [erg/s]. Same physical
        quantity, and same units, as :func:`prior_ir_xrays`'s ``nulnu_6um``.

    Returns
    -------
    float or jnp.ndarray
        Log-prior contribution (Gaussian log-density).

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes.

    Implements the same prior as upstream's ``prior_midIR_UV``
    (``PRIORS_AGNfitter.py:327-357``), validated against it, specifically the composite
    correlation and Gaussian penalty (:333-355`). ``x`` is computed by
    :func:`_x_from_nulnu_6um`; upstream's own formula for this ``x``
    (``log10(tor_flux_6microns*lumfactor) - 27.30103``, a SPECIFIC luminosity
    L_nu, not nu*L_nu) is algebraically identical to that helper's nu*L_nu
    form once the ``-13.69897`` frequency offset is accounted for -- see the
    helper's Notes for the derivation and the bug it fixes (a previous
    version of this function took the nu*L_nu input but applied the
    L_nu-calibrated ``-27.30103`` offset directly, off by
    ``log10(nu_6um) = 13.69897`` in ``x``).

    The sigma=0.6 combines the mid-IR--X-ray scatter (~0.5,
    :func:`prior_ir_xrays`) with the alpha_ox scatter (~0.1, per upstream's
    in-line comment at ``PRIORS_AGNfitter.py:354``). A prior tengri
    implementation (removed well before this revision) compared
    :math:`L_{\rm mir}` and :math:`L_{\rm uv}` directly -- see the module
    docstring's "Breaking change" note; at equal nominal log-luminosities
    that comparison and this one disagree by hundreds of dex, because equal
    luminosities are nowhere near this correlation's mean relation.

    References
    ----------
    .. [1] D. Stern (Stern 2015) -- mid-IR--X-ray correlation; see
       :func:`prior_ir_xrays` Notes for the citation-verification caveat.
    .. [2] D. W. Just et al., 2007, ApJ, 665, 1004 -- alpha_ox correlation;
       see :func:`prior_uv_xrays` Notes for the citation-verification
       caveat.
    """
    x = _x_from_nulnu_6um(nulnu_6um)
    log_l2500a_tomodel = (16.2530786 + 1.024 * x - 0.047 * x**2) / 0.643
    ratio = jnp.asarray(log_l2500a_bbmodel, dtype=float) - log_l2500a_tomodel
    return gaussian_log_prior(0.0, 0.6, ratio)


# ===========================================================================
# Public adapter: agnfitter_priors -- from a Prediction to a total log-prior.
# ===========================================================================

#: Default enable/mode settings, matching ``SETTINGS_AGNfitter.py`` (upstream
#: example config, ``example/SETTINGS_AGNfitter.py:225-247``):
#: ``PRIOR_energy_balance='Flexible'`` (on, flexible), ``PRIOR_AGNfraction=True``
#: (on), ``PRIOR_midIR_UV=False`` (off), ``PRIOR_galaxy_only=False`` (off,
#: this is the flag that selects ``prior_low_AGNfraction``). ``XRAYS`` in that
#: same file is the boolean ``True``, which matches NEITHER of the two literal
#: strings (``'Prior_UV'``, ``'Prior_midIR'``) the ``PRIORS()`` dispatcher
#: actually branches on (``PRIORS_AGNfitter.py:63,68``) -- i.e. upstream's own
#: shipped example, read literally, leaves both X-ray-tied priors OFF; tengri
#: mirrors that literal reading rather than guessing an intended default.
#: ``prior_ir_syn_fraction`` has no dedicated ``PRIOR_*`` flag upstream at all
#: (gated directly on ``RADIO``, ``PRIORS_AGNfitter.py:57``); off by default
#: here since it always needs explicit radio+IR data kwargs regardless.
AGNFITTER_PRIOR_DEFAULTS: dict = {
    "energy_balance": True,
    "energy_balance_mode": "flexible",
    "stellar_mass": False,
    "agn_fraction": True,
    "low_agn_fraction": False,
    "midir_uv": False,
    "uv_xrays": False,
    "ir_xrays": False,
    "ir_syn_fraction": False,
}


def _integrate_l_nu(wave_aa, l_nu):
    r"""Bolometric-like luminosity from an L_nu(wave) array: :math:`\int L_\nu\,d\nu`.

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom], ascending.
    l_nu : array_like, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Returns
    -------
    float
        Integrated luminosity [erg/s].

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes.

    Thin unit wrapper over
    :func:`tengri.utils.sed_quantities.compute_bolometric_luminosity`
    (Lsun, range-safe in float32 via its peak-factored integrand), converted
    to erg/s -- rather than a second, naive ``jnp.trapezoid`` reimplementation
    of the same integral that would not share that range-safety.
    """
    from tengri.utils.physics_constants import L_SUN
    from tengri.utils.sed_quantities import compute_bolometric_luminosity

    l_nu = jnp.asarray(l_nu, dtype=float)
    wave_aa = jnp.asarray(wave_aa, dtype=float)
    return compute_bolometric_luminosity(l_nu, wave_aa) * L_SUN


def _l_nu_at_wave(wave_aa, l_nu, target_wave_aa):
    r"""Interpolate a rest-frame :math:`L_\nu` array onto one target wavelength.

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom], ascending.
    l_nu : array_like, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].
    target_wave_aa : float
        Rest-frame wavelength at which to evaluate [Angstrom].

    Returns
    -------
    float
        :math:`L_\nu` at ``target_wave_aa`` [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes. **Grad-compatible**: yes (piecewise-linear).
    Approximation: linear interpolation on the model's native rest-frame
    grid, not a native node value -- adequate for the smoothly-varying AGN
    disc/torus continuum this adapter reads, but not a spectral-feature
    measurement.
    """
    return jnp.interp(
        jnp.asarray(target_wave_aa, dtype=float),
        jnp.asarray(wave_aa, dtype=float),
        jnp.asarray(l_nu, dtype=float),
    )


def _nulnu_at_wave(wave_aa, l_nu, target_wave_aa):
    r"""nu*L_nu at one target rest-frame wavelength [erg/s]."""
    from tengri.utils.physics_constants import C_AA

    l_nu_target = _l_nu_at_wave(wave_aa, l_nu, target_wave_aa)
    nu_target = C_AA / jnp.asarray(target_wave_aa, dtype=float)
    return nu_target * l_nu_target


def agnfitter_priors(
    pred,
    *,
    redshift,
    dlum,
    torus_key: str = "sed_agn",
    disc_key: str = "sed_agn",
    dust_ir_key: str = "sed_dust_ir",
    galaxy_key: str = "sed_attenuated",
    energy_balance_mode: str = AGNFITTER_PRIOR_DEFAULTS["energy_balance_mode"],
    enable_energy_balance: bool = AGNFITTER_PRIOR_DEFAULTS["energy_balance"],
    enable_stellar_mass: bool = AGNFITTER_PRIOR_DEFAULTS["stellar_mass"],
    enable_agn_fraction: bool = AGNFITTER_PRIOR_DEFAULTS["agn_fraction"],
    enable_low_agn_fraction: bool = AGNFITTER_PRIOR_DEFAULTS["low_agn_fraction"],
    enable_midir_uv: bool = AGNFITTER_PRIOR_DEFAULTS["midir_uv"],
    enable_uv_xrays: bool = AGNFITTER_PRIOR_DEFAULTS["uv_xrays"],
    enable_ir_xrays: bool = AGNFITTER_PRIOR_DEFAULTS["ir_xrays"],
    enable_ir_syn_fraction: bool = AGNFITTER_PRIOR_DEFAULTS["ir_syn_fraction"],
    ga=None,
    data_flux_1500=None,
    log_l2kev_data=None,
    log_f2_10kev_data=None,
    data_flux_rad=None,
    data_nu_rad=None,
    data_flux_ir=None,
    data_nu_ir=None,
):
    r"""Evaluate the AGNfitter-rX informative priors on a tengri prediction.

    Computes each enabled prior's physical inputs from the PUBLIC prediction
    surface -- ``pred.sed.components`` (rest-frame ``wavelength`` [Angstrom]
    plus per-component :math:`L_\nu` [erg/s/Hz]: ``sed_intrinsic``,
    ``sed_attenuated``, ``sed_dust_ir``, ``sed_agn``, ``sed_xray``,
    ``sed_radio``) -- and returns the total log-prior plus a per-prior
    breakdown.

    Parameters
    ----------
    pred : Prediction
        A single-galaxy prediction, ``model.predict(params)``.
    redshift : float
        Source redshift [dimensionless]. Used by
        :func:`prior_agn_fraction` / :func:`prior_low_agn_fraction`.
    dlum : float
        Luminosity distance [cm]. Used by the same two functions.
    torus_key, disc_key : str, optional
        Which ``pred.sed.components`` key to read the torus 6-micron nu*L_nu
        and the disc 1500/2500 A flux from. Both default to ``"sed_agn"``
        (disc+torus combined) -- **known approximation**: this reads the
        disc and torus continua at the SAME wavelengths from the SAME
        combined array, so the 6-micron value carries whatever residual
        disc contributes there (small; discs fall steeply into the IR) and
        the 1500/2500 A values carry whatever residual torus contributes
        there (small; tori peak in the IR, torus UV is typically a scattered
        fraction). A later task publishes per-sub-block AGN SEDs
        (``sed_agn_disc``, ``sed_agn_torus``); switching to those once
        available is a one-line change: ``torus_key="sed_agn_torus",
        disc_key="sed_agn_disc"``.
    dust_ir_key : str, optional
        ``pred.sed.components`` key for the cold-dust/starburst IR emission
        (:func:`prior_energy_balance`'s ``l_sb_emit``). Default
        ``"sed_dust_ir"``.
    galaxy_key : str, optional
        ``pred.sed.components`` key for the dust-attenuated stellar SED
        (used both as :func:`prior_energy_balance`'s attenuated-luminosity
        reference and as :func:`prior_agn_fraction`'s galaxy flux). Default
        ``"sed_attenuated"``.
    energy_balance_mode : {"flexible", "restrictive"}, optional
        Passed to :func:`prior_energy_balance`. Default ``"flexible"``
        (``SETTINGS_AGNfitter.py``'s ``PRIOR_energy_balance`` default).
    enable_energy_balance, enable_stellar_mass, enable_agn_fraction,
    enable_low_agn_fraction, enable_midir_uv, enable_uv_xrays,
    enable_ir_xrays, enable_ir_syn_fraction : bool, optional
        Which of the eight priors to evaluate. Defaults are
        :data:`AGNFITTER_PRIOR_DEFAULTS`, matching
        ``example/SETTINGS_AGNfitter.py`` where a corresponding ``PRIOR_*``
        flag exists (see that dict's own docstring for the two flags that
        do not map cleanly, and why).
    ga : float, optional
        Upstream's raw galaxy flux-normalization scalar (``GA``), required
        only if ``enable_stellar_mass=True``. tengri has no automatically
        -derived equivalent (it is a template-normalization exponent
        specific to upstream's model-dictionary bookkeeping, not a
        published prediction quantity) -- callers wanting this term supply
        it explicitly, e.g. from their own fit parameters if they have
        constructed an analogous quantity.
    data_flux_1500 : float, optional
        Observed flux density at rest-frame 1500 A [erg/s/cm^2/Hz], required
        if ``enable_agn_fraction`` or ``enable_low_agn_fraction`` is
        ``True`` (both need :func:`prior_agn_fraction` /
        :func:`prior_low_agn_fraction`'s ``data_flux_1500``, a genuinely
        observed quantity, not a model prediction).
    log_l2kev_data : float, optional
        log10 of the observed luminosity density at rest-frame 2 keV
        [log10(erg/s/Hz)], required if ``enable_uv_xrays=True``.
    log_f2_10kev_data : float, optional
        log10 of the observed monochromatic flux at the 2-10 keV band
        center [log10(erg/s/Hz)], required if ``enable_ir_xrays=True``.
    data_flux_rad, data_nu_rad, data_flux_ir, data_nu_ir : float, optional
        Observed radio flux/frequency and cold-dust-peak flux/frequency
        (``data_nu_*`` in log10(Hz)), required if
        ``enable_ir_syn_fraction=True``. ``sb_flux_ir``/``syn_flux_ir`` (the
        model-side inputs to :func:`prior_ir_syn_fraction`) are read from
        ``dust_ir_key`` / ``sed_radio`` at the wavelength corresponding to
        ``data_nu_ir``.

    Returns
    -------
    total : float or jnp.ndarray
        Sum of every enabled prior's log-prior contribution.
    breakdown : dict of str to (float or jnp.ndarray)
        Per-prior log-prior contribution, keyed by the same names as the
        ``enable_*`` parameters (without the ``enable_`` prefix), for every
        ENABLED prior only.

    Raises
    ------
    ValueError
        If a prior is enabled but a required data keyword argument is
        ``None``.

    Notes
    -----
    **JIT-compatible**: no. ``pred`` (:class:`~tengri.forward.prediction.
    Prediction`) is a Python-cached exploration object by its own docstring
    ("Not JIT-compatible (uses Python caching)"); this adapter is for
    post-fit inspection and reporting, one prediction at a time -- not the
    fitting hot path. For fitting, use ``Fitter(..., extra_log_prior=...)``
    directly on ``state.derived`` (JIT/grad-safe); see the module
    docstring's "Adapters" section.

    **Grad-compatible**: the arithmetic inside is JAX-differentiable, but
    since the whole function is not JIT-traced there is limited practical
    reason to differentiate through it; differentiate the individual prior
    functions directly if needed.

    References
    ----------
    .. [1] L. N. Martinez-Ramirez et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
       (2024). arXiv:2405.12111. doi:10.1051/0004-6361/202449329.

    Examples
    --------
    >>> pred = model.predict(params)  # doctest: +SKIP
    >>> total, breakdown = agnfitter_priors(  # doctest: +SKIP
    ...     pred,
    ...     redshift=1.0,
    ...     dlum=6.6e27,
    ...     data_flux_1500=3e-28,
    ... )
    >>> breakdown.keys()  # doctest: +SKIP
    dict_keys(['energy_balance', 'agn_fraction'])
    """
    components = pred.sed.components
    wave = components["wavelength"]

    breakdown: dict = {}

    if enable_energy_balance:
        from tengri.utils.physics_constants import L_SUN
        from tengri.utils.sed_quantities import compute_l_dust_absorbed

        l_absorbed = (
            compute_l_dust_absorbed(components["sed_intrinsic"], components[galaxy_key], wave)
            * L_SUN
        )
        l_sb_emit = _integrate_l_nu(wave, components[dust_ir_key])
        breakdown["energy_balance"] = prior_energy_balance(
            l_absorbed, l_sb_emit, mode=energy_balance_mode
        )

    if enable_stellar_mass:
        if ga is None:
            raise ValueError(
                "enable_stellar_mass=True requires ga=... (upstream's raw GA "
                "flux-normalization scalar; tengri has no automatically-derived "
                "equivalent, see this function's docstring)."
            )
        breakdown["stellar_mass"] = prior_stellar_mass(ga)

    if enable_agn_fraction or enable_low_agn_fraction:
        if data_flux_1500 is None:
            raise ValueError(
                "enable_agn_fraction / enable_low_agn_fraction require "
                "data_flux_1500=... (the OBSERVED rest-1500 A flux; these "
                "priors tie the model to data, not just to itself)."
            )
        bbb_flux_1500 = _l_nu_at_wave(wave, components[disc_key], 1500.0)
        gal_flux_1500 = _l_nu_at_wave(wave, components[galaxy_key], 1500.0)
        if enable_agn_fraction:
            breakdown["agn_fraction"] = prior_agn_fraction(
                bbb_flux_1500, gal_flux_1500, data_flux_1500, dlum, redshift
            )
        if enable_low_agn_fraction:
            breakdown["low_agn_fraction"] = prior_low_agn_fraction(
                bbb_flux_1500, gal_flux_1500, data_flux_1500, dlum, redshift
            )

    if enable_midir_uv:
        nulnu_6um = _nulnu_at_wave(wave, components[torus_key], 60000.0)
        log_l2500a_bbmodel = jnp.log10(_l_nu_at_wave(wave, components[disc_key], 2500.0))
        breakdown["midir_uv"] = prior_midir_uv(log_l2500a_bbmodel, nulnu_6um)

    if enable_uv_xrays:
        if log_l2kev_data is None:
            raise ValueError("enable_uv_xrays=True requires log_l2kev_data=... (observed).")
        log_l2500a_data = jnp.log10(_l_nu_at_wave(wave, components[disc_key], 2500.0))
        breakdown["uv_xrays"] = prior_uv_xrays(log_l2500a_data, log_l2kev_data)

    if enable_ir_xrays:
        if log_f2_10kev_data is None:
            raise ValueError("enable_ir_xrays=True requires log_f2_10kev_data=... (observed).")
        nulnu_6um = _nulnu_at_wave(wave, components[torus_key], 60000.0)
        breakdown["ir_xrays"] = prior_ir_xrays(log_f2_10kev_data, nulnu_6um)

    if enable_ir_syn_fraction:
        from tengri.utils.physics_constants import C_AA

        if any(v is None for v in (data_flux_rad, data_nu_rad, data_flux_ir, data_nu_ir)):
            raise ValueError(
                "enable_ir_syn_fraction=True requires data_flux_rad, data_nu_rad, "
                "data_flux_ir, data_nu_ir=... (all observed)."
            )
        wave_ir_aa = C_AA / (10.0 ** jnp.asarray(data_nu_ir, dtype=float))
        sb_flux_ir = _l_nu_at_wave(wave, components[dust_ir_key], wave_ir_aa)
        syn_flux_ir = _l_nu_at_wave(wave, components["sed_radio"], wave_ir_aa)
        breakdown["ir_syn_fraction"] = prior_ir_syn_fraction(
            data_flux_rad, data_nu_rad, data_flux_ir, data_nu_ir, sb_flux_ir, syn_flux_ir
        )

    total = jnp.asarray(0.0)
    for term in breakdown.values():
        total = total + term
    return total, breakdown
