# SPDX-License-Identifier: BSD-3-Clause
"""Lazy prediction object for derived physical quantities.

The :class:`Prediction` class provides on-demand computation of derived
quantities from a tengri forward model. Properties are only computed
when first accessed, and intermediate results (SFR, SED, emission lines)
are cached so related quantities share the expensive computation.

Two usage modes
---------------

**Mode 1 — Single-galaxy exploration (lazy):**

.. code-block:: python

    pred = model.predict(params)

    # SFH quantities — triggers only SFH computation (~100 μs)
    pred.sfh.stellar_mass
    pred.sfh.mass_weighted_age_gyr

    # SED quantities — triggers full SED computation (~300 μs)
    pred.sed.l_bol
    pred.sed.uv_slope_beta
    pred.sed.dn4000

    # Emission lines — triggers nebular computation (~200 μs)
    pred.lines.halpha
    pred.lines.bpt_nii

    # Radio and X-ray (empirical scaling relations)
    pred.radio.l_1p4ghz
    pred.xray.l_x_xrb

**Mode 2 — Population / batch computation (JIT + vmap):**

For computing derived quantities over many parameter sets (posterior
chains, mock catalogs), use the JIT-compatible group methods instead:

.. code-block:: python

    import jax

    # Batch of 10,000 parameter sets
    params_batch = spec.sample_batch(jax.random.PRNGKey(0), n=10_000)

    # vmap over SFH quantities — returns SFHQuantities with shape (10000,)
    sfh_fn = jax.vmap(model.predict_sfh_quantities)
    sfh_batch = sfh_fn(params_batch)
    sfh_batch.stellar_mass  # shape (10000,)

    # vmap over SED quantities — returns SEDQuantities
    sed_fn = jax.vmap(model.predict_sed_quantities)
    sed_batch = sed_fn(params_batch)
    sed_batch.m_uv  # shape (10000,)

The lazy :class:`Prediction` object is NOT JIT-compatible (it uses
Python-level caching). For inference loops and batches, always use
the JIT-compatible methods (``predict_sfh_quantities``, etc.).

Caching hierarchy
-----------------

Three computation levels, triggered on demand:

============  ============================  ==========
Level         Triggered by                  Approx cost
============  ============================  ==========
**SFH**       Any ``pred.sfh.*``            ~100 μs
**SED**       Any ``pred.sed.*``            ~300 μs
**Lines**     Any ``pred.lines.*``          ~200 μs
============  ============================  ==========

Each level auto-triggers its dependencies: SED triggers SFH first;
Lines triggers SFH first. Luminosity-weighted quantities in
``pred.sfh`` trigger SED (they need per-bin luminosities).
"""

from typing import NamedTuple

import jax.numpy as jnp

from tengri.components.stellar.sps.dsps_wrapper import (
    compute_surviving_mass,
    interpolate_mass_remaining,
)
from tengri.utils.sed_quantities import (
    KEY_LINES,
    compute_balmer_break,
    compute_bolometric_luminosity,
    compute_dn4000,
    compute_fuv_flux,
    compute_ionizing_efficiency,
    compute_irx,
    compute_l_dust_absorbed,
    compute_l_radio_1p4ghz_from_sfr,
    compute_l_radio_thermal,
    compute_l_tir,
    compute_l_x_agn,
    compute_l_x_xrb,
    compute_luminosity_weighted_age,
    compute_luminosity_weighted_metallicity,
    compute_m_uv,
    compute_mass_weighted_age,
    compute_mass_weighted_metallicity,
    compute_nuv_flux,
    compute_q_ir,
    compute_rest_uv_color,
    compute_uv_luminosity_1600,
    compute_uv_slope_beta,
    extract_line_luminosity,
)

# ── NamedTuples for JIT-compatible batch computation ──────────────


class SFHQuantities(NamedTuple):
    """Derived quantities from the star formation history.

    All fields are JAX arrays (scalars). This is a proper JAX pytree,
    so it works with ``jax.jit``, ``jax.vmap``, and ``jax.grad``.

    Attributes
    ----------
    stellar_mass : jnp.ndarray
        Total formed stellar mass [Msun].
    stellar_mass_surviving : jnp.ndarray
        Surviving mass in living stars + remnants [Msun].
        Returns NaN if the mass-remaining table was not loaded.
    sfr_100myr : jnp.ndarray
        Star formation rate averaged over the last 100 Myr [Msun/yr].
    sfr_10myr : jnp.ndarray
        Star formation rate averaged over the last 10 Myr [Msun/yr].
    ssfr : jnp.ndarray
        Specific star formation rate SFR/M* [1/yr].
    mass_weighted_age_gyr : jnp.ndarray
        Mass-weighted stellar age [Gyr].
    mass_weighted_metallicity : jnp.ndarray
        Mass-weighted metallicity log10(Z), evolving-Z aware.

    Returns
    -------
    This is a NamedTuple (JAX pytree) returned by
    :meth:`SEDModel.predict_sfh_quantities`.

    Notes
    -----
    JAX-compatible array container. All fields are JAX arrays compatible with
    ``jax.jit`` and ``jax.vmap``. Returned by :meth:`SEDModel.predict_sfh_quantities`
    and :attr:`Prediction.sfh` when accessed.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import SFHQuantities
    >>> q = SFHQuantities(
    ...     stellar_mass=jnp.array(1e10),
    ...     stellar_mass_surviving=jnp.array(6e9),
    ...     sfr_100myr=jnp.array(5.0),
    ...     sfr_10myr=jnp.array(8.0),
    ...     ssfr=jnp.array(5e-10),
    ...     mass_weighted_age_gyr=jnp.array(3.5),
    ...     mass_weighted_metallicity=jnp.array(-0.5),
    ... )
    >>> float(q.stellar_mass)
    10000000000.0
    >>> "stellar_mass" in q._fields and "sfr_100myr" in q._fields
    True
    """

    stellar_mass: jnp.ndarray
    stellar_mass_surviving: jnp.ndarray
    sfr_100myr: jnp.ndarray
    sfr_10myr: jnp.ndarray
    ssfr: jnp.ndarray
    mass_weighted_age_gyr: jnp.ndarray
    mass_weighted_metallicity: jnp.ndarray


class SEDQuantities(NamedTuple):
    """Derived quantities from the spectral energy distribution.

    All fields are JAX arrays. Proper JAX pytree for ``jit``/``vmap``.

    Attributes
    ----------
    l_bol : jnp.ndarray
        Bolometric luminosity [Lsun].
    l_tir : jnp.ndarray
        Total infrared luminosity 8–1000 μm [Lsun].
    l_dust_absorbed : jnp.ndarray
        Dust-absorbed luminosity [Lsun]. Returns NaN if no intrinsic SED.
    irx : jnp.ndarray
        Infrared excess log10(L_TIR / νLν_1600) [dimensionless].
    uv_slope_beta : jnp.ndarray
        UV spectral slope β in range 1250–2600 Å [dimensionless].
    dn4000 : jnp.ndarray
        Narrow 4000 Å break, Balogh et al. 1999 [dimensionless].
    balmer_break : jnp.ndarray
        Modified Balmer break, Wang et al. 2024 [dimensionless].
    m_uv : jnp.ndarray
        Absolute UV magnitude at rest-frame 1500 Å [AB].
    fuv_flux : jnp.ndarray
        Mean flux density in FUV 1000–1700 Å [erg/s/Hz].
    nuv_flux : jnp.ndarray
        Mean flux density in NUV 1700–3200 Å [erg/s/Hz].
    fuv_flux_intrinsic : jnp.ndarray
        Dust-free FUV flux [erg/s/Hz]. Returns NaN if no intrinsic SED.
    nuv_flux_intrinsic : jnp.ndarray
        Dust-free NUV flux [erg/s/Hz]. Returns NaN if no intrinsic SED.
    rest_uv_color : jnp.ndarray
        Rest-frame U-V color [AB magnitudes].
    luminosity_weighted_age_gyr : jnp.ndarray
        Luminosity-weighted age [Gyr].
    luminosity_weighted_metallicity : jnp.ndarray
        Luminosity-weighted metallicity log10(Z).

    Returns
    -------
    This is a NamedTuple (JAX pytree) returned by
    :meth:`SEDModel.predict_sed_quantities`.

    Notes
    -----
    JAX-compatible array container. All fields are JAX arrays compatible with
    ``jax.jit`` and ``jax.vmap``. Returned by :meth:`SEDModel.predict_sed_quantities`
    and :attr:`Prediction.sed` when accessed.

    Examples
    --------
    Access via :attr:`Prediction.sed` after calling :meth:`SEDModel.predict`:

    .. code-block:: python

        pred = model.predict(params)
        sed = pred.sed  # SEDQuantities NamedTuple
        print(float(sed.l_bol))  # bolometric luminosity [Lsun]
        print(float(sed.dn4000))  # 4000 Å break strength
        print(float(sed.uv_slope_beta))  # UV slope beta
    """

    l_bol: jnp.ndarray
    l_tir: jnp.ndarray
    l_dust_absorbed: jnp.ndarray
    irx: jnp.ndarray
    uv_slope_beta: jnp.ndarray
    dn4000: jnp.ndarray
    balmer_break: jnp.ndarray
    m_uv: jnp.ndarray
    fuv_flux: jnp.ndarray
    nuv_flux: jnp.ndarray
    fuv_flux_intrinsic: jnp.ndarray
    nuv_flux_intrinsic: jnp.ndarray
    rest_uv_color: jnp.ndarray
    luminosity_weighted_age_gyr: jnp.ndarray
    luminosity_weighted_metallicity: jnp.ndarray


class EmissionLines(NamedTuple):
    """Emission line luminosities — headline survey lines plus the full backend catalog.

    NaN for the headline fields and empty arrays for ``all_*`` when no
    nebular model is active. For doublets ([O II], C IV) the headline
    fields sum both components.

    The full ~271-line Cue catalog (and equivalent grids for CloudyGrid)
    is exposed via :attr:`all_waves` / :attr:`all_lums` so users can read
    species the headline NamedTuple does not name explicitly (HeII 1640,
    HeI 10830, NIII] 1750, [O III] 4363, etc.). See :meth:`get` for the
    nearest-wavelength accessor.

    Attributes
    ----------
    lya : jnp.ndarray
        Lyman-alpha at 1216 Å [Lsun].
    civ_1549 : jnp.ndarray
        C IV doublet 1548+1551 Å, summed [Lsun].
    oii : jnp.ndarray
        [OII] doublet 3726+3729 Å, summed [Lsun].
    hbeta : jnp.ndarray
        H-beta at 4861 Å [Lsun].
    oiii_4959 : jnp.ndarray
        [OIII] at 4959 Å [Lsun].
    oiii_5007 : jnp.ndarray
        [OIII] at 5007 Å [Lsun].
    nii_6548 : jnp.ndarray
        [NII] at 6548 Å [Lsun].
    halpha : jnp.ndarray
        H-alpha at 6563 Å [Lsun].
    nii_6584 : jnp.ndarray
        [NII] at 6584 Å [Lsun].
    sii_6717 : jnp.ndarray
        [SII] at 6717 Å [Lsun].
    sii_6731 : jnp.ndarray
        [SII] at 6731 Å [Lsun].
    all_waves : jnp.ndarray, shape ``(n_lines,)``
        Vacuum rest-frame wavelengths of every species published by the
        active nebular backend [Angstrom]. Empty when the backend does
        not expose a discrete catalog (BakedIn, shock).
    all_lums : jnp.ndarray, shape ``(n_lines,)``
        Luminosities at ``all_waves``, in the same dust regime as the
        headline fields (i.e. attenuated by the active dust model when
        present) [Lsun].

    Returns
    -------
    This is a NamedTuple (JAX pytree) returned by
    :meth:`SEDModel.predict_emission_lines`.

    Notes
    -----
    JAX-compatible array container. All fields are JAX arrays compatible with
    ``jax.jit`` and ``jax.vmap``. Returned by :attr:`Prediction.lines` when
    accessed. All fields return NaN if no nebular model is active in the SEDModel.

    Examples
    --------
    Access via :attr:`Prediction.lines` after calling :meth:`SEDModel.predict`:

    .. code-block:: python

        pred = model.predict(params)
        lines = pred.lines  # EmissionLines NamedTuple
        print(float(lines.halpha))  # H-alpha luminosity [Lsun]
        print(float(lines.oiii_5007))  # [OIII] 5007 Å luminosity [Lsun]
        # BPT diagram
        bpt_x = float(lines.nii_6584 / lines.halpha)
        bpt_y = float(lines.oiii_5007 / lines.hbeta)
        # Access lines outside the headline catalog via nearest-wavelength
        heii_1640 = lines.get(1640.42)  # closest match in ``all_waves``
    """

    lya: jnp.ndarray
    civ_1549: jnp.ndarray
    oii: jnp.ndarray
    hbeta: jnp.ndarray
    oiii_4959: jnp.ndarray
    oiii_5007: jnp.ndarray
    nii_6548: jnp.ndarray
    halpha: jnp.ndarray
    nii_6584: jnp.ndarray
    sii_6717: jnp.ndarray
    sii_6731: jnp.ndarray
    all_waves: jnp.ndarray
    all_lums: jnp.ndarray

    def get(self, wavelength: float, tol_aa: float = 2.0) -> jnp.ndarray:
        """Return the luminosity at the species nearest ``wavelength`` Å.

        Parameters
        ----------
        wavelength : float
            Rest-frame vacuum wavelength to look up [Angstrom].
        tol_aa : float, optional
            Acceptable distance to the nearest cataloged line [Angstrom].
            Returns ``nan`` if the nearest line is further than this. Default 2.0.

        Returns
        -------
        jnp.ndarray
            Luminosity at the matched line [Lsun], or ``nan`` if no line
            is within ``tol_aa``. Returns ``nan`` if the active backend
            did not publish a discrete catalog.
        """
        if self.all_waves.size == 0:
            return jnp.asarray(jnp.nan)
        diff = jnp.abs(self.all_waves - jnp.asarray(wavelength))
        idx = jnp.argmin(diff)
        return jnp.where(diff[idx] <= tol_aa, self.all_lums[idx], jnp.asarray(jnp.nan))


class DerivedQuantities(NamedTuple):
    """All derived physical quantities (convenience container).

    Returned by ``model.predict_derived()``.

    Attributes
    ----------
    sfh : SFHQuantities
        Star formation history derived quantities.
    sed : SEDQuantities
        Spectral energy distribution derived quantities.

    Returns
    -------
    This is a NamedTuple (JAX pytree) returned by
    :meth:`SEDModel.predict_derived`.

    Notes
    -----
    JAX-compatible array container combining :class:`SFHQuantities` and
    :class:`SEDQuantities`. Compatible with ``jax.jit`` and ``jax.vmap``.
    Returned by :meth:`SEDModel.predict_derived`.

    Examples
    --------
    .. code-block:: python

        from tengri import DerivedQuantities

        derived = model.predict_derived(params)
        print(float(derived.sfh.stellar_mass))  # [Msun]
        print(float(derived.sed.dn4000))  # 4000 Å break
        print(float(derived.sed.uv_slope_beta))  # UV slope β
    """

    sfh: SFHQuantities
    sed: SEDQuantities


# ── Lazy property group base ──────────────────────────────────────


class _CachedBase:
    """Base class for lazy-cached prediction property groups.

    Each subclass accesses the parent :class:`Prediction` object's
    shared cache via ``self._pred._cache`` and triggers the appropriate
    computation level via ``self._pred._ensure_*()`` methods.
    """

    __slots__ = ("_pred",)

    def __init__(self, prediction):
        self._pred = prediction


# ── SFH properties (lazy) ─────────────────────────────────────────


class SFHProperties(_CachedBase):
    """Lazy property accessor for SFH-derived quantities.

    Accessing any property triggers SFH computation (SFR, CSP weights)
    if not already cached. Luminosity-weighted quantities additionally
    trigger SED computation.

    Attributes
    ----------
    stellar_mass : property
        Total formed stellar mass [Msun].
    stellar_mass_surviving : property
        Surviving stellar mass [Msun].
    sfr_100myr : property
        SFR averaged over 100 Myr [Msun/yr].
    sfr_10myr : property
        SFR averaged over 10 Myr [Msun/yr].
    ssfr : property
        Specific SFR [1/yr].
    mass_weighted_age_gyr : property
        Mass-weighted age [Gyr].
    mass_weighted_metallicity : property
        Mass-weighted metallicity log10(Z).
    luminosity_weighted_age_gyr : property
        Luminosity-weighted age [Gyr].
    luminosity_weighted_metallicity : property
        Luminosity-weighted metallicity log10(Z).

    Notes
    -----
    JAX-compatible array container. Properties are lazy-cached within a
    :class:`Prediction` object. Returned by :attr:`Prediction.sfh`.
    Not JIT-compatible (uses Python caching). For batch computation, use
    JIT-compatible methods :meth:`SEDModel.predict_sfh_quantities` instead.

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.sfh.stellar_mass  # triggers SFH computation
    Array(1.23e10, dtype=float64)
    >>> pred.sfh.mass_weighted_age_gyr  # reuses cached weights
    Array(3.45, dtype=float64)
    """

    @property
    def stellar_mass(self):
        """Total formed stellar mass.

        Returns
        -------
        float
            Total stellar mass ever formed [Msun].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sfh()
        return jnp.sum(self._pred._cache["weights"])

    @property
    def stellar_mass_surviving(self):
        """Surviving stellar mass in living stars and remnants.

        Returns NaN if the mass-remaining table was not loaded in the
        SSP data.

        Returns
        -------
        float
            Surviving stellar mass [Msun], or NaN if mass table unavailable.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sfh()
        model = self._pred._model
        if model.ssp_data.ssp_mass_remaining is None:
            return jnp.array(jnp.nan)
        log_z = self._pred._cache["p"].get("log_z_abs", 0.0)
        mr_at_met = interpolate_mass_remaining(
            model.ssp_data.ssp_mass_remaining, model.ssp_data.ssp_lgmet, log_z
        )
        return compute_surviving_mass(self._pred._cache["weights"], mr_at_met)

    @property
    def sfr_100myr(self):
        """Star formation rate averaged over the last 100 Myr.

        Returns
        -------
        float
            Time-averaged SFR over the last 100 Myr [Msun/yr].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sfh()
        sfr = self._pred._cache["sfr"]
        model = self._pred._model
        mask = model.age_yr <= 1e8
        return jnp.where(
            jnp.sum(mask) > 0,
            jnp.sum(sfr * mask) / jnp.maximum(jnp.sum(mask), 1.0),
            sfr[0],
        )

    @property
    def sfr_10myr(self):
        """Star formation rate averaged over the last 10 Myr.

        Returns
        -------
        float
            Time-averaged SFR over the last 10 Myr [Msun/yr].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sfh()
        sfr = self._pred._cache["sfr"]
        model = self._pred._model
        mask = model.age_yr <= 1e7
        return jnp.where(
            jnp.sum(mask) > 0,
            jnp.sum(sfr * mask) / jnp.maximum(jnp.sum(mask), 1.0),
            sfr[0],
        )

    @property
    def ssfr(self):
        """Specific star formation rate normalized by stellar mass.

        Uses surviving mass if available, otherwise formed mass.

        Returns
        -------
        float
            Specific star formation rate SFR/M* [1/yr].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        mass_surv = self.stellar_mass_surviving
        mass = jnp.where(jnp.isnan(mass_surv), self.stellar_mass, mass_surv)
        return self.sfr_100myr / jnp.maximum(mass, 1.0)

    @property
    def mass_weighted_age_gyr(self):
        """Mass-weighted stellar age.

        Returns
        -------
        float
            Age weighted by stellar mass [Gyr].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sfh()
        return compute_mass_weighted_age(
            self._pred._cache["weights"], self._pred._model.ssp_ages_yr
        )

    @property
    def mass_weighted_metallicity(self):
        """Mass-weighted metallicity.

        For single metallicity models, returns the metallicity parameter.
        For evolving metallicity, computes Σ(w·Z)/Σ(w).

        Returns
        -------
        float
            Metallicity weighted by stellar mass, log10(Z).

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sfh()
        p = self._pred._cache["p"]
        return compute_mass_weighted_metallicity(
            self._pred._cache["weights"],
            self._pred._model.ssp_ages_yr,
            p.get("log_z_abs", 0.0),
            log_z_initial=p.get("log_z_abs_initial"),
            log_z_final=p.get("log_z_abs_final"),
        )

    @property
    def luminosity_weighted_age_gyr(self):
        """Luminosity-weighted stellar age.

        Accessing this property triggers SED computation.

        Returns
        -------
        float
            Age weighted by stellar luminosity [Gyr].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sed()
        return compute_luminosity_weighted_age(
            self._pred._cache["weights"],
            self._pred._cache["ssp_flux_at_z"],
            self._pred._model.ssp_ages_yr,
            self._pred._model.ssp_data.ssp_wave,
        )

    @property
    def luminosity_weighted_metallicity(self):
        """Luminosity-weighted metallicity.

        Accessing this property triggers SED computation.

        Returns
        -------
        float
            Metallicity weighted by stellar luminosity, log10(Z).

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sed()
        p = self._pred._cache["p"]
        return compute_luminosity_weighted_metallicity(
            self._pred._cache["weights"],
            self._pred._cache["ssp_flux_at_z"],
            self._pred._model.ssp_ages_yr,
            self._pred._model.ssp_data.ssp_wave,
            p.get("log_z_abs", 0.0),
            log_z_initial=p.get("log_z_abs_initial"),
            log_z_final=p.get("log_z_abs_final"),
        )


# ── SED properties (lazy) ─────────────────────────────────────────


class SEDProperties(_CachedBase):
    """Lazy property accessor for SED-derived quantities.

    Accessing any property triggers the full SED computation (dust
    attenuation, emission, AGN, etc.) if not already cached.

    Attributes
    ----------
    l_bol : property
        Bolometric luminosity [Lsun].
    l_tir : property
        Total infrared luminosity [Lsun].
    l_dust_absorbed : property
        Dust-absorbed luminosity [Lsun].
    irx : property
        Infrared excess [dimensionless].
    uv_slope_beta : property
        UV spectral slope [dimensionless].
    dn4000 : property
        4000 Å break [dimensionless].
    balmer_break : property
        Balmer break [dimensionless].
    m_uv : property
        Absolute UV magnitude [AB].
    fuv_flux : property
        FUV flux density [erg/s/Hz].
    nuv_flux : property
        NUV flux density [erg/s/Hz].
    fuv_flux_intrinsic : property
        Dust-free FUV flux [erg/s/Hz].
    nuv_flux_intrinsic : property
        Dust-free NUV flux [erg/s/Hz].
    rest_uv_color : property
        Rest-frame U-V color [AB magnitudes].
    luminosity_weighted_age_gyr : property
        Luminosity-weighted age [Gyr].
    luminosity_weighted_metallicity : property
        Luminosity-weighted metallicity log10(Z).

    Notes
    -----
    JAX-compatible array container. Properties are lazy-cached within a
    :class:`Prediction` object. Returned by :attr:`Prediction.sed`.
    Not JIT-compatible (uses Python caching). For batch computation,
    use JIT-compatible methods :meth:`SEDModel.predict_sed_quantities`.

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.sed.l_bol  # triggers SED computation
    Array(2.5e43, dtype=float64)
    >>> pred.sed.uv_slope_beta  # reuses cached SED
    Array(-1.8, dtype=float64)
    """

    def _wave(self):
        """Get rest-frame wavelength array from model."""
        return self._pred._model.ssp_data.ssp_wave

    @property
    def components(self):
        r"""Per-component SED decomposition on the rest-frame grid.

        The single-prediction counterpart of
        :meth:`Posterior.sed_components
        <tengri.inference.posterior.Posterior.sed_components>` — both
        read the per-component arrays every adapter publishes into
        ``state.derived`` (ADR-0009) via
        :func:`tengri.forward.state_to_sed_components`. Reuses the
        prediction's cached forward state: no extra forward pass.

        Returns
        -------
        dict
            ``wavelength`` [Angstrom] plus rest-frame :math:`L_\nu`
            [erg/s/Hz] per component, each shape ``(n_wave,)``:
            ``sed_total``, ``sed_intrinsic`` (stellar pre-dust),
            ``sed_attenuated`` (stellar post-dust), ``sed_nebular``,
            ``sed_shock``, ``sed_dust_ir``, ``sed_agn``, ``sed_radio``,
            ``sed_xray`` — zeros for components not in the chain.

        Examples
        --------
        >>> pred = model.predict(params)
        >>> comp = pred.sed.components
        >>> comp["sed_dust_ir"]  # one component
        >>> (comp["sed_agn"] / comp["sed_total"]).max()  # AGN fraction
        """
        from tengri.forward.component_factory import state_to_sed_components

        self._pred._ensure_sfh()
        return state_to_sed_components(self._pred._cache["_state"])

    def _sed(self):
        """Retrieve cached total SED, computing if necessary."""
        self._pred._ensure_sed()
        return self._pred._cache["sed_total"]

    def _sed_intrinsic(self):
        """Retrieve cached intrinsic (unattenuated) SED, computing if necessary."""
        self._pred._ensure_sed()
        return self._pred._cache.get("sed_intrinsic")

    @property
    def l_bol(self):
        """Bolometric luminosity.

        Returns
        -------
        float
            Total bolometric luminosity [Lsun].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return compute_bolometric_luminosity(self._sed(), self._wave())

    @property
    def l_tir(self):
        """Total infrared luminosity.

        Returns
        -------
        float
            Integrated luminosity in the 8–1000 μm range [Lsun].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return compute_l_tir(self._sed(), self._wave())

    @property
    def l_dust_absorbed(self):
        """Dust-absorbed luminosity.

        Returns NaN if no intrinsic SED is available.

        Returns
        -------
        float
            Luminosity absorbed by dust [Lsun], or NaN if unavailable.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        sed_intr = self._sed_intrinsic()
        if sed_intr is None:
            return jnp.array(jnp.nan)
        self._pred._ensure_sed()
        return compute_l_dust_absorbed(sed_intr, self._pred._cache["sed_attenuated"], self._wave())

    @property
    def irx(self):
        """Infrared excess.

        Returns
        -------
        float
            Infrared excess IRX = log10(L_TIR / νLν_1600) [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        l_tir = self.l_tir
        l_uv = compute_uv_luminosity_1600(self._sed(), self._wave())
        return compute_irx(l_tir, l_uv)

    @property
    def uv_slope_beta(self):
        """UV spectral slope.

        Returns
        -------
        float
            Spectral slope β in the rest-frame range 1250–2600 Å [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return compute_uv_slope_beta(self._sed(), self._wave())

    @property
    def dn4000(self):
        """Narrow 4000 Å break.

        Returns
        -------
        float
            Narrow D_n(4000) break from Balogh et al. 1999 [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return compute_dn4000(self._sed(), self._wave())

    @property
    def balmer_break(self):
        """Modified Balmer break.

        Returns
        -------
        float
            Modified Balmer break from Wang et al. 2024 [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return compute_balmer_break(self._sed(), self._wave())

    @property
    def m_uv(self):
        """Absolute UV magnitude.

        Returns
        -------
        float
            Absolute magnitude at rest-frame 1500 Å [AB].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return compute_m_uv(self._sed(), self._wave())

    @property
    def fuv_flux(self):
        """Mean flux density in the FUV.

        Returns
        -------
        float
            Mean flux density in the FUV 1000–1700 Å range [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return compute_fuv_flux(self._sed(), self._wave())

    @property
    def nuv_flux(self):
        """Mean flux density in the NUV.

        Returns
        -------
        float
            Mean flux density in the NUV 1700–3200 Å range [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return compute_nuv_flux(self._sed(), self._wave())

    @property
    def fuv_flux_intrinsic(self):
        """Dust-free FUV flux.

        Returns NaN if no intrinsic SED is available.

        Returns
        -------
        float
            Dust-free flux density in the FUV 1000–1700 Å range [erg/s/Hz],
            or NaN if unavailable.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        sed_intr = self._sed_intrinsic()
        if sed_intr is None:
            return jnp.array(jnp.nan)
        return compute_fuv_flux(sed_intr, self._wave())

    @property
    def nuv_flux_intrinsic(self):
        """Dust-free NUV flux.

        Returns NaN if no intrinsic SED is available.

        Returns
        -------
        float
            Dust-free flux density in the NUV 1700–3200 Å range [erg/s/Hz],
            or NaN if unavailable.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        sed_intr = self._sed_intrinsic()
        if sed_intr is None:
            return jnp.array(jnp.nan)
        return compute_nuv_flux(sed_intr, self._wave())

    @property
    def rest_uv_color(self):
        """Rest-frame U-V color.

        Returns
        -------
        float
            U-V color in rest frame [AB magnitudes].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return compute_rest_uv_color(self._sed(), self._wave())

    @property
    def luminosity_weighted_age_gyr(self):
        """Luminosity-weighted stellar age.

        Returns
        -------
        float
            Age weighted by stellar luminosity [Gyr].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sed()
        return compute_luminosity_weighted_age(
            self._pred._cache["weights"],
            self._pred._cache["ssp_flux_at_z"],
            self._pred._model.ssp_ages_yr,
            self._pred._model.ssp_data.ssp_wave,
        )

    @property
    def luminosity_weighted_metallicity(self):
        """Luminosity-weighted metallicity.

        Returns
        -------
        float
            Metallicity weighted by stellar luminosity, log10(Z).

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sed()
        p = self._pred._cache["p"]
        return compute_luminosity_weighted_metallicity(
            self._pred._cache["weights"],
            self._pred._cache["ssp_flux_at_z"],
            self._pred._model.ssp_ages_yr,
            self._pred._model.ssp_data.ssp_wave,
            p.get("log_z_abs", 0.0),
            log_z_initial=p.get("log_z_abs_initial"),
            log_z_final=p.get("log_z_abs_final"),
        )


# ── Emission line properties (lazy) ───────────────────────────────

# Floor used in BPT and other line-ratio diagnostics to avoid log10(0)
# when a line is undetected at this S/N. Set well below any realistic
# flux (typical detection limits ≈ 1e-18 erg/s/cm²) so the floor never
# affects detected lines but yields a finite, sortable −∞-equivalent
# value for missing lines.
_LINE_RATIO_FLOOR = 1e-50


class LineProperties(_CachedBase):
    """Lazy property accessor for emission line luminosities and diagnostic ratios.

    Accessing any line property triggers the nebular computation
    if not already cached. All line luminosities are in Lsun. Diagnostic
    ratios are dimensionless log10 values.

    If no nebular model is active, all line luminosities return NaN
    and all ratios return NaN.

    Attributes
    ----------
    lya : property
        Lyman-alpha [Lsun].
    civ_1549 : property
        C IV doublet [Lsun].
    oii : property
        [OII] doublet [Lsun].
    hbeta : property
        H-beta [Lsun].
    oiii_4959 : property
        [OIII] 4959 [Lsun].
    oiii_5007 : property
        [OIII] 5007 [Lsun].
    nii_6548 : property
        [NII] 6548 [Lsun].
    halpha : property
        H-alpha [Lsun].
    nii_6584 : property
        [NII] 6584 [Lsun].
    sii_6717 : property
        [SII] 6717 [Lsun].
    sii_6731 : property
        [SII] 6731 [Lsun].
    bpt_nii : property
        BPT [NII] diagnostic [dimensionless].
    bpt_sii : property
        BPT [SII] diagnostic [dimensionless].
    o3hb : property
        [OIII]/Hβ diagnostic [dimensionless].
    r23 : property
        R23 metallicity diagnostic [dimensionless].
    o32 : property
        O32 ionization parameter [dimensionless].
    balmer_decrement : property
        Hα/Hβ ratio [dimensionless].

    Notes
    -----
    JAX-compatible array container. Properties are lazy-cached within a
    :class:`Prediction` object. Returned by :attr:`Prediction.lines`.
    Not JIT-compatible (uses Python caching). For batch computation,
    use JIT-compatible methods :meth:`SEDModel.predict_emission_lines`.

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.lines.halpha  # Hα luminosity in Lsun
    Array(1.5e8, dtype=float64)
    >>> pred.lines.bpt_nii  # log10([NII]6584 / Hα)
    Array(-0.45, dtype=float64)
    """

    def _get_line(self, name):
        """Extract emission line luminosity from cached grid, computing if necessary."""
        self._pred._ensure_lines()
        lw = self._pred._cache["line_waves"]
        ll = self._pred._cache["line_lums"]
        return extract_line_luminosity(lw, ll, KEY_LINES[name])

    # --- Individual lines ---

    @property
    def lya(self):
        """Lyman-alpha at 1216 Å.

        Returns
        -------
        float
            Line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("lya")

    @property
    def civ_1549(self):
        """C IV doublet at 1548+1551 Å.

        Returns
        -------
        float
            Summed line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("civ_1549")

    @property
    def oii(self):
        """[OII] doublet at 3726+3729 Å.

        Returns
        -------
        float
            Summed line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("oii")

    @property
    def hbeta(self):
        """H-beta at 4861 Å.

        Returns
        -------
        float
            Line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("hbeta")

    @property
    def oiii_4959(self):
        """[OIII] at 4959 Å.

        Returns
        -------
        float
            Line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("oiii_4959")

    @property
    def oiii_5007(self):
        """[OIII] at 5007 Å.

        Returns
        -------
        float
            Line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("oiii_5007")

    @property
    def nii_6548(self):
        """[NII] at 6548 Å.

        Returns
        -------
        float
            Line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("nii_6548")

    @property
    def halpha(self):
        """H-alpha at 6563 Å.

        Returns
        -------
        float
            Line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("halpha")

    @property
    def nii_6584(self):
        """[NII] at 6584 Å.

        Returns
        -------
        float
            Line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("nii_6584")

    @property
    def sii_6717(self):
        """[SII] at 6717 Å.

        Returns
        -------
        float
            Line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("sii_6717")

    @property
    def sii_6731(self):
        """[SII] at 6731 Å.

        Returns
        -------
        float
            Line luminosity [Lsun], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self._get_line("sii_6731")

    # --- Diagnostic ratios ---

    @property
    def bpt_nii(self):
        """BPT-NII diagnostic ratio.

        Returns
        -------
        float
            log10([NII]6584 / Hα) for BPT diagram [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return jnp.log10(
            jnp.maximum(self.nii_6584, _LINE_RATIO_FLOOR)
            / jnp.maximum(self.halpha, _LINE_RATIO_FLOOR)
        )

    @property
    def bpt_sii(self):
        """BPT-SII diagnostic ratio.

        Returns
        -------
        float
            log10(([SII]6717+6731) / Hα) for BPT diagram [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        sii_total = self.sii_6717 + self.sii_6731
        return jnp.log10(
            jnp.maximum(sii_total, _LINE_RATIO_FLOOR) / jnp.maximum(self.halpha, _LINE_RATIO_FLOOR)
        )

    @property
    def o3hb(self):
        """[OIII]5007/Hβ diagnostic ratio.

        Returns
        -------
        float
            log10([OIII]5007 / Hβ), the BPT diagram y-axis [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return jnp.log10(
            jnp.maximum(self.oiii_5007, _LINE_RATIO_FLOOR)
            / jnp.maximum(self.hbeta, _LINE_RATIO_FLOOR)
        )

    @property
    def r23(self):
        """R23 metallicity diagnostic indicator.

        Returns
        -------
        float
            log10(([OII]+[OIII]4959+5007)/Hβ), metallicity indicator [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        numerator = self.oii + self.oiii_4959 + self.oiii_5007
        return jnp.log10(
            jnp.maximum(numerator, _LINE_RATIO_FLOOR) / jnp.maximum(self.hbeta, _LINE_RATIO_FLOOR)
        )

    @property
    def o32(self):
        """O32 ionization parameter.

        Returns
        -------
        float
            log10([OIII]5007 / [OII]), ionization indicator [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return jnp.log10(
            jnp.maximum(self.oiii_5007, _LINE_RATIO_FLOOR)
            / jnp.maximum(self.oii, _LINE_RATIO_FLOOR)
        )

    @property
    def balmer_decrement(self):
        """Balmer decrement ratio.

        Returns
        -------
        float
            Hα/Hβ intensity ratio [dimensionless]. Case B intrinsic value: 2.86.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self.halpha / jnp.maximum(self.hbeta, _LINE_RATIO_FLOOR)


# ── Radio properties (lazy) ───────────────────────────────────────


class RadioProperties(_CachedBase):
    """Lazy property accessor for radio-derived quantities.

    These use empirical scaling relations: the FIR-radio correlation
    (Bell 2003; Murphy+2011) and free-free emission from the ionizing
    photon budget.

    Attributes
    ----------
    l_1p4ghz : property
        Radio luminosity at 1.4 GHz [erg/s/Hz].
    l_thermal : property
        Thermal free-free luminosity [erg/s/Hz].
    l_nonthermal : property
        Non-thermal synchrotron luminosity [erg/s/Hz].
    q_ir : property
        FIR-radio correlation parameter [dimensionless].

    Notes
    -----
    JAX-compatible array container. Properties are lazy-cached within a
    :class:`Prediction` object. Returned by :attr:`Prediction.radio`.
    Not JIT-compatible (uses Python caching).

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.radio.l_1p4ghz  # 1.4 GHz luminosity
    Array(5.2e28, dtype=float64)
    """

    @property
    def l_1p4ghz(self):
        """Radio luminosity at 1.4 GHz.

        Returns
        -------
        float
            Radio luminosity density at 1.4 GHz [erg/s/Hz], from SFR scaling.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        sfr = self._pred.sfh.sfr_100myr
        return compute_l_radio_1p4ghz_from_sfr(sfr)

    @property
    def l_thermal(self):
        """Thermal radio luminosity at 1.4 GHz.

        Returns
        -------
        float
            Free-free radio luminosity density at 1.4 GHz [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        q_h = self._pred.ionizing.q_h
        return compute_l_radio_thermal(q_h)

    @property
    def l_nonthermal(self):
        """Non-thermal radio luminosity at 1.4 GHz.

        Returns
        -------
        float
            Synchrotron radio luminosity density [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self.l_1p4ghz - self.l_thermal

    @property
    def q_ir(self):
        """FIR-radio correlation parameter.

        Returns
        -------
        float
            FIR-radio correlation parameter q_TIR [dimensionless].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        l_tir = self._pred.sed.l_tir
        return compute_q_ir(l_tir, self.l_1p4ghz)


# ── X-ray properties (lazy) ───────────────────────────────────────


class XRayProperties(_CachedBase):
    """Lazy property accessor for X-ray derived quantities.

    Uses empirical scaling relations from Lehmer et al. (2010, 2016) for
    X-ray binaries and Duras et al. (2020) for AGN bolometric corrections.

    Attributes
    ----------
    l_x_xrb : property
        X-ray binary luminosity [erg/s].
    l_x_agn : property
        AGN X-ray luminosity [erg/s].
    l_x_total : property
        Total X-ray luminosity [erg/s].

    Notes
    -----
    JAX-compatible array container. Properties are lazy-cached within a
    :class:`Prediction` object. Returned by :attr:`Prediction.xray`.
    Not JIT-compatible (uses Python caching).

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.xray.l_x_xrb  # XRB luminosity (0.5-8 keV)
    Array(3.1e40, dtype=float64)
    """

    @property
    def l_x_xrb(self):
        """X-ray luminosity from X-ray binaries.

        Returns
        -------
        float
            XRB X-ray luminosity in 0.5–8 keV band [erg/s].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        sfr = self._pred.sfh.sfr_100myr
        mstar = self._pred.sfh.stellar_mass
        return compute_l_x_xrb(sfr, mstar)

    @property
    def l_x_agn(self):
        """AGN X-ray luminosity.

        Returns
        -------
        float
            AGN X-ray luminosity in 2–10 keV band [erg/s].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_sed()
        agn_bol = self._pred._cache.get("agn_bol_erg", 0.0)
        return compute_l_x_agn(agn_bol)

    @property
    def l_x_total(self):
        """Total X-ray luminosity.

        Returns
        -------
        float
            Combined XRB and AGN X-ray luminosity [erg/s].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        return self.l_x_xrb + self.l_x_agn


# ── Ionizing properties (lazy) ────────────────────────────────────


class IonizingProperties(_CachedBase):
    """Lazy property accessor for ionizing photon budget quantities.

    The ionizing photon rate Q_H is extracted from the nebular model
    backend (Cloudy grid or Cue emulator). If no nebular model is
    active, returns NaN.

    Attributes
    ----------
    q_h : property
        Ionizing photon production rate [photons/s].
    xi_ion : property
        Ionizing photon production efficiency [Hz/erg].

    Notes
    -----
    JAX-compatible array container. Properties are lazy-cached within a
    :class:`Prediction` object. Returned by :attr:`Prediction.ionizing`.
    Not JIT-compatible (uses Python caching).

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.ionizing.xi_ion  # ionizing efficiency
    Array(25.3, dtype=float64)
    """

    @property
    def q_h(self):
        """Total ionizing photon production rate.

        Returns NaN if no nebular model is active.

        Returns
        -------
        float
            Ionizing photon production rate [photons/s], or NaN if unavailable.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred._cache.get("q_h_total", jnp.array(jnp.nan))

    @property
    def xi_ion(self):
        """Ionizing photon production efficiency.

        Defined as Q_H / L_UV(1600 Å). Key parameter for cosmic
        reionization studies. Typical values: 25.0–25.6.

        Returns
        -------
        float
            Ionizing photon efficiency log10(ξ_ion) [Hz/erg].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        q_h = self.q_h
        l_uv = compute_uv_luminosity_1600(self._pred.sed._sed(), self._pred.sed._wave())
        return compute_ionizing_efficiency(q_h, l_uv)


# ── Main Prediction class ─────────────────────────────────────────


class Prediction:
    """Lazy prediction object with on-demand computation of derived quantities.

    Created via ``model.predict(params)``. Properties are computed on
    first access and cached. The cache is shared across all property
    groups (``sfh``, ``sed``, ``lines``, ``radio``, ``xray``,
    ``ionizing``), so related quantities share the expensive
    intermediates.

    Parameters
    ----------
    model : SEDModel
        The tengri SEDModel instance.
    params : dict
        Parameter values (public names).

    Attributes
    ----------
    sfh : SFHProperties
        Star formation history derived quantities. Lazy accessor.
    sed : SEDProperties
        Spectral energy distribution derived quantities. Lazy accessor.
    lines : LineProperties
        Emission line luminosities and diagnostic ratios. Lazy accessor.
    radio : RadioProperties
        Radio-derived quantities from empirical relations. Lazy accessor.
    xray : XRayProperties
        X-ray derived quantities from empirical relations. Lazy accessor.
    ionizing : IonizingProperties
        Ionizing photon budget quantities. Lazy accessor.

    Returns
    -------
    Prediction
        Lazy prediction object with cached computed quantities.

    Notes
    -----
    This class is NOT JIT-compatible due to Python-level caching. For
    batch computations over many parameter sets (MCMC chains, mock
    catalogs), use the JIT-compatible methods :meth:`SEDModel.predict_sfh_quantities`,
    :meth:`SEDModel.predict_sed_quantities`, etc. instead. Those return
    JAX pytrees (:class:`SFHQuantities`, :class:`SEDQuantities`,
    :class:`DerivedQuantities`) suitable for :func:`jax.vmap`,
    :func:`jax.jit`, and :func:`jax.grad`.

    Examples
    --------
    **Two equivalent ways to access derived quantities:**

    >>> pred = model.predict(params)
    >>> pred.stellar_mass  # flat shortcut
    >>> pred.sfh.stellar_mass  # grouped form (same value)
    >>> pred.dn4000  # flat
    >>> pred.sed.dn4000  # grouped
    >>> pred.halpha  # flat
    >>> pred.lines.halpha  # grouped

    The grouped form (``pred.sfh``, ``pred.sed``, ``pred.lines``,
    ``pred.radio``, ``pred.xray``, ``pred.ionizing``) exposes every
    derived quantity. The top-level shortcuts cover the most-used
    quantities for quick access; for less-common ones use the grouped
    form. Both share the same lazy cache, so accessing a quantity by
    either route triggers computation only once.

    **Accessing the full SED or photometry:**

    >>> pred.sed_array  # shape (n_wave,)
    >>> pred.photometry  # shape (n_filters,)

    **For batch computation, use JIT-compatible methods instead:**

    >>> sfh_batch = jax.vmap(model.predict_sfh_quantities)(params_batch)
    """

    __slots__ = ("_cache", "_model", "_params", "ionizing", "lines", "radio", "sed", "sfh", "xray")

    def __init__(self, model, params):
        self._model = model
        self._params = params
        self._cache = {}
        self.sfh = SFHProperties(self)
        self.sed = SEDProperties(self)
        self.lines = LineProperties(self)
        self.radio = RadioProperties(self)
        self.xray = XRayProperties(self)
        self.ionizing = IonizingProperties(self)

    def _ensure_sfh(self):
        """Populate SFH cache (SFR history, age weights, internal params).

        Reads the orchestrator's :class:`ForwardState` to keep SFH-only
        consumers (``stellar_mass``, ``sfr_*``, ``mass_weighted_age``)
        and SED-consuming consumers
        (``luminosity_weighted_age``, ``dn4000``) on the same numerics.
        ``weights`` is the orchestrator's DSPS-canonical
        ``age_weights`` (Msun per SSP age bin); ``sfr`` is
        ``sfr_history`` on the SFH lookback grid (same shape as
        ``model.age_yr``). The orchestrator state itself is cached on
        ``_state`` so :meth:`_ensure_sed` can reuse it without a
        second forward-pass.
        """
        if "weights" in self._cache:
            return
        p = self._model._get_internal_params(self._params)
        state = self._model.predict_state(self._params)
        derived = state.derived
        # The stellar block integrates the SFH on
        # ``spec.n_grid`` (default 64) regardless of whether the model
        # is stochastic. Legacy ``SEDModel`` uses ``n_grid=256`` for
        # non-stochastic configs, so cache consumers index ``sfr``
        # against ``model.age_yr`` of length 256. Resample the
        # orchestrator's SFR history to the legacy grid so masks like
        # ``model.age_yr <= 1e8`` still align.
        sfh_grid = jnp.asarray(derived["sfh_grid_lbt_yr"])
        sfr_history = jnp.asarray(derived["sfr_history"])
        sfr_on_legacy_grid = jnp.interp(self._model.age_yr, sfh_grid, sfr_history)
        self._cache.update(
            {
                "p": p,
                "sfr": sfr_on_legacy_grid,
                "weights": jnp.asarray(derived["age_weights"]),
                "_state": state,
            }
        )

    def _ensure_sed(self):
        """Populate SED cache from the orchestrator's ForwardState.

        Re-uses the state computed in :meth:`_ensure_sfh` (cached on
        ``self._cache["_state"]``). Reconstructs the legacy cache
        contract:

        - ``sed_total`` ← ``state.sed_intrinsic`` (post-dust total)
        - ``sed_intrinsic`` ← ``sum(lnu_age, axis=0)`` (pre-dust stellar)
        - ``sed_attenuated`` ← ``state.derived["sed_dust_attenuated"]``
        - ``ssp_flux_at_z`` ← ``lnu_age / (age_weights * LSUN_ERG)``
          (safe-divided where ``age_weights`` is zero)
        - ``agn_bol_erg`` ← ``state.derived["L_agn_bol"]`` if present.
        """
        if "sed_total" in self._cache:
            return
        self._ensure_sfh()
        state = self._cache["_state"]
        derived = state.derived

        self._cache["sed_total"] = state.sed_intrinsic

        lnu_age = derived.get("lnu_age")
        if lnu_age is not None:
            from tengri.utils.physics_constants import L_SUN

            lnu_age_arr = jnp.asarray(lnu_age)
            self._cache["sed_intrinsic"] = jnp.sum(lnu_age_arr, axis=0)
            aw = jnp.asarray(self._cache["weights"])
            aw_safe = jnp.maximum(aw, 1e-30)
            self._cache["ssp_flux_at_z"] = lnu_age_arr / (aw_safe[:, None] * L_SUN)

        sed_attenuated = derived.get("sed_dust_attenuated")
        if sed_attenuated is not None:
            self._cache["sed_attenuated"] = jnp.asarray(sed_attenuated)

        if "L_agn_bol" in derived:
            self._cache["agn_bol_erg"] = jnp.asarray(derived["L_agn_bol"])

    def _ensure_lines(self):
        """Compute and cache nebular emission line luminosities.

        Reads the discrete catalog published by
        :class:`~tengri.components.nebular.component.NebularSEDComponent`
        (``state.derived["line_waves"]`` / ``["line_lums"]``). Matches
        legacy-path luminosities within numerical tolerance for both
        Cue and CloudyGrid backends.

        Issues a one-time :class:`UserWarning` when the active backend
        doesn't expose a per-line luminosity catalog (BakedIn / Shock).
        Without the warning, ``pred.lines.halpha`` etc. silently return
        NaN — see #361.
        """
        if "line_waves" in self._cache:
            return
        self._ensure_sfh()
        model = self._model
        backend = model._nebular_backend

        if backend is None or not hasattr(backend, "predict_nebular_line_luminosities"):
            import warnings

            backend_name = type(backend).__name__ if backend is not None else "None"
            warnings.warn(
                f"Nebular backend {backend_name!r} does not publish a "
                "per-line luminosity catalog, so pred.lines.halpha, "
                ".hbeta, .bpt_nii, etc. will return NaN. To get discrete "
                "line luminosities, rebuild the model with neb={'type': "
                "'cue'}, 'cloudy', or 'cb19' (each requires a "
                "compatible SSP and any backing grid; see "
                "tengri.list_nebular_backends() for details). See #361.",
                UserWarning,
                stacklevel=3,
            )
            self._cache["line_waves"] = jnp.array([])
            self._cache["line_lums"] = jnp.array([])
            self._cache["q_h_total"] = jnp.array(jnp.nan)
            return

        # Pull the catalog from the orchestrator's NebularSEDComponent
        # publication. BakedIn / Shock backends won't publish it; fall
        # back to all-NaN for those (matches the legacy "no catalog"
        # behavior without raising).
        state = model.predict_state(self._params)
        derived = state.derived
        if "line_waves" in derived and "line_lums" in derived:
            self._cache["line_waves"] = jnp.asarray(derived["line_waves"])
            self._cache["line_lums"] = jnp.asarray(derived["line_lums"])
        else:
            self._cache["line_waves"] = jnp.array([])
            self._cache["line_lums"] = jnp.array([])

        # Q_H: compute from backend's precomputed table if available.
        # Uses the orchestrator-published ``age_weights`` (Msun/bin) +
        # ``log_metallicity_history`` (present-day value) so the value
        # matches the legacy path even when other consumers of the
        # cache still go through the legacy ``_ensure_sed``.
        weights_orch = derived.get("age_weights")
        log_z_history = derived.get("log_metallicity_history")
        if (
            weights_orch is not None
            and log_z_history is not None
            and hasattr(backend, "_qh_table")
            and backend._qh_table is not None
        ):
            log_z = jnp.asarray(log_z_history)[0]
            young_idx = backend._young_idx
            young_ages = model.ssp_log_ages_yr[young_idx]
            young_weights = jnp.asarray(weights_orch)[young_idx]

            def _qh_one_bin(log_age_i, w_i):
                """Ionizing photon production rate for one age bin."""
                return w_i * backend._get_qh_at(log_z, log_age_i)

            import jax

            q_h_per_bin = jax.vmap(_qh_one_bin)(young_ages, young_weights)
            neb_fesc = jnp.asarray(self._params.get("neb_fesc", 0.0))
            self._cache["q_h_total"] = jnp.sum(q_h_per_bin) * (1.0 - neb_fesc)
        else:
            self._cache["q_h_total"] = jnp.array(jnp.nan)

    @property
    def sed_array(self):
        """Full rest-frame SED array.

        Returns
        -------
        ndarray, shape (n_wave,)
            Total spectral energy distribution [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._ensure_sed()
        return self._cache["sed_total"]

    @property
    def photometry(self):
        """Observed photometric flux densities.

        Returns
        -------
        ndarray, shape (n_filters,)
            Photometry at the filters defined in the SEDModel [erg/s/cm²/Hz].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.

        Examples
        --------
        .. code-block:: python

            pred = model.predict(params)
            phot = pred.photometry  # ndarray, shape (n_filters,)
            print(phot.shape)  # e.g. (8,) for 8 photometric bands
        """
        return self._model.predict_photometry(self._params)

    # ── Top-level shortcuts to grouped derived quantities ───────────────
    # ``pred.stellar_mass`` and ``pred.sfh.stellar_mass`` return the same value;
    # the flat form is for tab-completion convenience and aligns with how
    # astronomers typically refer to derived quantities (no domain prefix).
    # Where two groups expose the same name (e.g. luminosity_weighted_*),
    # the flat shortcut points to the SED version — that's the canonical
    # "luminosity-weighted" meaning (uses attenuated SED, not stellar-only).

    # --- SFH-derived (forward to pred.sfh) ---

    @property
    def stellar_mass(self):
        """Total stellar mass formed [M☉]. Same as ``pred.sfh.stellar_mass``."""
        return self.sfh.stellar_mass

    @property
    def stellar_mass_surviving(self):
        """Surviving stellar + remnant mass [M☉]. Same as ``pred.sfh.stellar_mass_surviving``."""
        return self.sfh.stellar_mass_surviving

    @property
    def sfr_100myr(self):
        """SFR averaged over last 100 Myr [M☉/yr]. Same as ``pred.sfh.sfr_100myr``."""
        return self.sfh.sfr_100myr

    @property
    def sfr_10myr(self):
        """SFR averaged over last 10 Myr [M☉/yr]. Same as ``pred.sfh.sfr_10myr``."""
        return self.sfh.sfr_10myr

    @property
    def ssfr(self):
        """Specific SFR [yr⁻¹]. Same as ``pred.sfh.ssfr``."""
        return self.sfh.ssfr

    @property
    def mass_weighted_age_gyr(self):
        """Mass-weighted stellar age [Gyr]. Same as ``pred.sfh.mass_weighted_age_gyr``."""
        return self.sfh.mass_weighted_age_gyr

    @property
    def mass_weighted_metallicity(self):
        """Mass-weighted log₁₀(Z/Z☉). Same as ``pred.sfh.mass_weighted_metallicity``."""
        return self.sfh.mass_weighted_metallicity

    # --- SED-derived (forward to pred.sed) ---

    @property
    def l_bol(self):
        """Bolometric luminosity [L☉]. Same as ``pred.sed.l_bol``."""
        return self.sed.l_bol

    @property
    def l_tir(self):
        """Total infrared (8–1000 μm) luminosity [L☉]. Same as ``pred.sed.l_tir``."""
        return self.sed.l_tir

    @property
    def l_dust_absorbed(self):
        """Dust-absorbed luminosity [L☉]. Same as ``pred.sed.l_dust_absorbed``."""
        return self.sed.l_dust_absorbed

    @property
    def irx(self):
        """Infrared excess L_TIR / L_UV(1600 Å). Same as ``pred.sed.irx``."""
        return self.sed.irx

    @property
    def uv_slope_beta(self):
        """UV slope β in f_λ ∝ λ^β. Same as ``pred.sed.uv_slope_beta``."""
        return self.sed.uv_slope_beta

    @property
    def dn4000(self):
        """D_n(4000) break ratio. Same as ``pred.sed.dn4000``."""
        return self.sed.dn4000

    @property
    def balmer_break(self):
        """Balmer break flux ratio. Same as ``pred.sed.balmer_break``."""
        return self.sed.balmer_break

    @property
    def m_uv(self):
        """Absolute magnitude at 1500 Å. Same as ``pred.sed.m_uv``."""
        return self.sed.m_uv

    @property
    def fuv_flux(self):
        """FUV flux at 1500 Å [erg/s/cm²]. Same as ``pred.sed.fuv_flux``."""
        return self.sed.fuv_flux

    @property
    def nuv_flux(self):
        """NUV flux at 2300 Å [erg/s/cm²]. Same as ``pred.sed.nuv_flux``."""
        return self.sed.nuv_flux

    @property
    def fuv_flux_intrinsic(self):
        """Dust-free FUV flux. Same as ``pred.sed.fuv_flux_intrinsic``."""
        return self.sed.fuv_flux_intrinsic

    @property
    def nuv_flux_intrinsic(self):
        """Dust-free NUV flux. Same as ``pred.sed.nuv_flux_intrinsic``."""
        return self.sed.nuv_flux_intrinsic

    @property
    def rest_uv_color(self):
        """Rest-frame UV color (f_1500 − f_2300). Same as ``pred.sed.rest_uv_color``."""
        return self.sed.rest_uv_color

    @property
    def luminosity_weighted_age_gyr(self):
        """Luminosity-weighted age [Gyr]. Same as ``pred.sed.luminosity_weighted_age_gyr``.

        Both ``pred.sfh`` and ``pred.sed`` define this; the top-level shortcut
        forwards to the SED version (canonical "luminosity-weighted" using the
        attenuated stellar SED).
        """
        return self.sed.luminosity_weighted_age_gyr

    @property
    def luminosity_weighted_metallicity(self):
        """Luminosity-weighted log₁₀(Z/Z☉).

        Same as ``pred.sed.luminosity_weighted_metallicity``.
        """
        return self.sed.luminosity_weighted_metallicity

    # --- Emission lines (forward to pred.lines) ---

    @property
    def halpha(self):
        """Hα 6564 Å luminosity [erg/s]. Same as ``pred.lines.halpha``."""
        return self.lines.halpha

    @property
    def hbeta(self):
        """Hβ 4862 Å luminosity [erg/s]. Same as ``pred.lines.hbeta``."""
        return self.lines.hbeta

    @property
    def oiii_5007(self):
        """[O III] 5007 Å luminosity [erg/s]. Same as ``pred.lines.oiii_5007``."""
        return self.lines.oiii_5007

    @property
    def balmer_decrement(self):
        """Hα/Hβ flux ratio. Same as ``pred.lines.balmer_decrement``."""
        return self.lines.balmer_decrement

    # --- Ionizing budget (forward to pred.ionizing) ---

    @property
    def q_h(self):
        """Total ionizing photon production rate [s⁻¹]. Same as ``pred.ionizing.q_h``."""
        return self.ionizing.q_h

    @property
    def xi_ion(self):
        """Ionizing photon production efficiency [Hz·erg⁻¹]. Same as ``pred.ionizing.xi_ion``."""
        return self.ionizing.xi_ion
