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

from tengri._mapping import ReadOnlyPropertyMapping


class _SEDCallable:
    """A callable SED accessor that refuses to be mistaken for an array.

    ``pred.rest_sed`` / ``pred.obs_sed`` are *methods* — ``pred.rest_sed()``
    gives the model's own grid, ``pred.rest_sed(wave)`` resamples onto yours
    (contract §4b.3: uniform callables with defaults, like ``photometry()``
    and ``spectrum()``).

    They were briefly plain properties. Had they simply become methods, the
    forgotten-parenthesis mistake would fail *silently*: ``np.asarray`` of a
    bound method yields a ``dtype=object`` array, which plots and arithmetics
    happily turn into garbage rather than an exception. Every numeric dunder
    below therefore raises with the fix spelled out. Failing loudly on a
    misuse is the whole point — this package has shipped enough silent
    NaN-and-carry-on bugs already.
    """

    __slots__ = ("_fn", "_name")

    def __init__(self, fn, name):
        self._fn = fn
        self._name = name

    def __call__(self, wave=None):
        return self._fn(wave)

    def _not_an_array(self, *_args, **_kwargs):
        raise TypeError(
            f"Prediction.{self._name} is a method, not an array — you left off the "
            f"parentheses. Use pred.{self._name}() for the model's own grid, or "
            f"pred.{self._name}(wave) to resample onto your own grid [Angstrom]. "
            f"The matching wavelength axis is pred."
            f"{'wave_rest' if self._name == 'rest_sed' else 'wave_obs'}."
        )

    # Everything that would otherwise silently coerce to a dtype=object array.
    __array__ = _not_an_array
    __len__ = _not_an_array
    __iter__ = _not_an_array
    __getitem__ = _not_an_array
    __add__ = __radd__ = _not_an_array
    __sub__ = __rsub__ = _not_an_array
    __mul__ = __rmul__ = _not_an_array
    __truediv__ = __rtruediv__ = _not_an_array

    def __repr__(self):
        return f"<Prediction.{self._name}(wave=None) — a method; call it to get the array>"


# ── Module-level warn-once guard ──────────────────────────────────

_WARNED_RUNTIME_PHOTOMETRY = False


def _warn_runtime_photometry_once():
    """Issue one-time warning for runtime photometry filter construction."""
    global _WARNED_RUNTIME_PHOTOMETRY
    if not _WARNED_RUNTIME_PHOTOMETRY:
        import warnings

        _WARNED_RUNTIME_PHOTOMETRY = True
        warnings.warn(
            "Runtime filter resolution (Photometry.from_names) uses the exact path "
            "(full SED integration) and is slower than build-time filters. "
            "For repeated calls with the same filters, cache the result. "
            "For significantly faster photometry, rebuild the model with "
            "approx=WavePrecomp() and use pred.photometry(fast=True).",
            UserWarning,
            stacklevel=4,
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
        Lyman-alpha at 1216 Å [erg/s].
    civ_1549 : jnp.ndarray
        C IV doublet 1548+1551 Å, summed [erg/s].
    oii : jnp.ndarray
        [OII] doublet 3726+3729 Å, summed [erg/s].
    hbeta : jnp.ndarray
        H-beta at 4861 Å [erg/s].
    oiii_4959 : jnp.ndarray
        [OIII] at 4959 Å [erg/s].
    oiii_5007 : jnp.ndarray
        [OIII] at 5007 Å [erg/s].
    nii_6548 : jnp.ndarray
        [NII] at 6548 Å [erg/s].
    halpha : jnp.ndarray
        H-alpha at 6563 Å [erg/s].
    nii_6584 : jnp.ndarray
        [NII] at 6584 Å [erg/s].
    sii_6717 : jnp.ndarray
        [SII] at 6717 Å [erg/s].
    sii_6731 : jnp.ndarray
        [SII] at 6731 Å [erg/s].
    all_waves : jnp.ndarray, shape ``(n_lines,)``
        Vacuum rest-frame wavelengths of every species published by the
        active nebular backend [Angstrom]. Empty when the backend does
        not expose a discrete catalog (BakedIn, shock).
    all_lums : jnp.ndarray, shape ``(n_lines,)``
        Luminosities at ``all_waves``, in the same dust regime as the
        headline fields (i.e. attenuated by the active dust model when
        present) [erg/s].

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
        print(float(lines.halpha))  # H-alpha luminosity [erg/s]
        print(float(lines.oiii_5007))  # [OIII] 5007 Å luminosity [erg/s]
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
            Luminosity at the matched line [erg/s], or ``nan`` if no line
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

    Returned by ``model._predict_derived()``.

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

        derived = model._predict_derived(params)
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
        return self._pred.properties["stellar_mass"]

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
        return self._pred.properties["stellar_mass_surviving"]

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
        return self._pred.properties["sfr_100myr"]

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
        return self._pred.properties["sfr_10myr"]

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
        return self._pred.properties["ssfr"]

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
        return self._pred.properties["mass_weighted_age_gyr"]

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
        return self._pred.properties["mass_weighted_metallicity"]

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
        return self._pred.properties["luminosity_weighted_age_gyr"]

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
        return self._pred.properties["luminosity_weighted_metallicity"]


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
        """Rest-frame wavelength grid of the cached ForwardState.

        The pipeline evaluates on its own grid (auto-extended for dust
        emission, trimmed to the modeling range) — NOT the raw
        ``ssp_data.ssp_wave``. Every cached SED array shares the state's
        axis, so quantities integrating sed × wave must use it too.
        """
        self._pred._ensure_sfh()
        return self._pred._cache["_state"].wave

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
        return self._pred.properties["l_bol"]

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
        return self._pred.properties["l_tir"]

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
        return self._pred.properties["l_dust_absorbed"]

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
        return self._pred.properties["irx"]

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
        return self._pred.properties["uv_slope_beta"]

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
        return self._pred.properties["dn4000"]

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
        return self._pred.properties["balmer_break"]

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
        return self._pred.properties["m_uv"]

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
        return self._pred.properties["fuv_flux"]

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
        return self._pred.properties["nuv_flux"]

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
        return self._pred.properties["fuv_flux_intrinsic"]

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
        return self._pred.properties["nuv_flux_intrinsic"]

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
        return self._pred.properties["rest_uv_color"]

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
        return self._pred.properties["luminosity_weighted_age_gyr"]

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
        return self._pred.properties["luminosity_weighted_metallicity"]


# ── Emission line properties (lazy) ───────────────────────────────

# ``_LINE_RATIO_FLOOR = 1e-50`` stood here until #1568, described as the floor
# "used in BPT and other line-ratio diagnostics to avoid log10(0)". It had no
# readers — the BPT ratios live in ``NebularSEDComponent`` and use that module's
# own floor. Removed rather than made representable: a dead constant that is
# also 0.0 in float32 is the worst of both, since it reads as a live guard.


class LineProperties(_CachedBase):
    """Lazy property accessor for emission line luminosities and diagnostic ratios.

    Accessing any line property triggers the nebular computation
    if not already cached. All line luminosities are in erg/s. Diagnostic
    ratios are dimensionless log10 values.

    If no nebular model is active, all line luminosities return NaN
    and all ratios return NaN.

    Attributes
    ----------
    lya : property
        Lyman-alpha [erg/s].
    civ_1549 : property
        C IV doublet [erg/s].
    oii : property
        [OII] doublet [erg/s].
    hbeta : property
        H-beta [erg/s].
    oiii_4959 : property
        [OIII] 4959 [erg/s].
    oiii_5007 : property
        [OIII] 5007 [erg/s].
    nii_6548 : property
        [NII] 6548 [erg/s].
    halpha : property
        H-alpha [erg/s].
    nii_6584 : property
        [NII] 6584 [erg/s].
    sii_6717 : property
        [SII] 6717 [erg/s].
    sii_6731 : property
        [SII] 6731 [erg/s].
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
    >>> pred.lines.halpha  # Hα luminosity in erg/s
    Array(2.4e41, dtype=float64)
    >>> pred.lines.bpt_nii  # log10([NII]6584 / Hα)
    Array(-0.45, dtype=float64)
    """

    # --- Individual lines ---

    @property
    def lya(self):
        """Lyman-alpha at 1216 Å.

        Returns
        -------
        float
            Line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["lya"]

    @property
    def civ_1549(self):
        """C IV doublet at 1548+1551 Å.

        Returns
        -------
        float
            Summed line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["civ_1549"]

    @property
    def oii(self):
        """[OII] doublet at 3726+3729 Å.

        Returns
        -------
        float
            Summed line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["oii"]

    @property
    def hbeta(self):
        """H-beta at 4861 Å.

        Returns
        -------
        float
            Line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["hbeta"]

    @property
    def oiii_4959(self):
        """[OIII] at 4959 Å.

        Returns
        -------
        float
            Line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["oiii_4959"]

    @property
    def oiii_5007(self):
        """[OIII] at 5007 Å.

        Returns
        -------
        float
            Line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["oiii_5007"]

    @property
    def nii_6548(self):
        """[NII] at 6548 Å.

        Returns
        -------
        float
            Line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["nii_6548"]

    @property
    def halpha(self):
        """H-alpha at 6563 Å.

        Returns
        -------
        float
            Line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["halpha"]

    @property
    def nii_6584(self):
        """[NII] at 6584 Å.

        Returns
        -------
        float
            Line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["nii_6584"]

    @property
    def sii_6717(self):
        """[SII] at 6717 Å.

        Returns
        -------
        float
            Line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["sii_6717"]

    @property
    def sii_6731(self):
        """[SII] at 6731 Å.

        Returns
        -------
        float
            Line luminosity [erg/s], or NaN if no nebular model.

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.
        """
        self._pred._ensure_lines()
        return self._pred.properties["sii_6731"]

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
        self._pred._ensure_lines()
        return self._pred.properties["bpt_nii"]

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
        self._pred._ensure_lines()
        return self._pred.properties["bpt_sii"]

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
        self._pred._ensure_lines()
        return self._pred.properties["o3hb"]

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
        self._pred._ensure_lines()
        return self._pred.properties["r23"]

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
        self._pred._ensure_lines()
        return self._pred.properties["o32"]

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
        self._pred._ensure_lines()
        return self._pred.properties["balmer_decrement"]


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
        return self._pred.properties["l_1p4ghz"]

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
        return self._pred.properties["l_thermal"]

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
        return self._pred.properties["l_nonthermal"]

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
        return self._pred.properties["q_ir"]


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
        return self._pred.properties["l_x_xrb"]

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
        return self._pred.properties["l_x_agn"]

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
        return self._pred.properties["l_x_total"]


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
        return self._pred.properties["q_h"]

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
        return self._pred.properties["xi_ion"]


# ── Property catalog accessor ────────────────────────────────────


class PropertyCatalog(ReadOnlyPropertyMapping):
    """View of computed properties with dict-like and attribute access.

    A PropertyCatalog is a dict-like accessor bound to a Prediction that
    lazily evaluates each property from the cached ForwardState. It
    implements ``__getitem__``, ``__contains__``, iteration, and
    ``to_dict(names=...)`` for batch export.

    Parameters
    ----------
    prediction : Prediction
        Parent Prediction object (which caches the ForwardState and params).

    Attributes
    ----------
    _prediction : Prediction
        Reference to parent Prediction instance.
    """

    def __init__(self, prediction):
        """Initialize the catalog bound to a Prediction."""
        object.__setattr__(self, "_prediction", prediction)

    def __getitem__(self, name: str):
        """Access a property by name (dict-like syntax).

        Parameters
        ----------
        name : str
            Property name (e.g., ``"stellar_mass"``).

        Returns
        -------
        scalar
            Computed property value.

        Raises
        ------
        KeyError
            If the property is unknown or not available in this model.
        """
        pred = object.__getattribute__(self, "_prediction")
        catalog = pred._model._ensure_property_catalog()
        if name not in catalog:
            from tengri.forward.properties import missing_property_message

            raise KeyError(missing_property_message(name, available=catalog))
        entry = catalog[name]
        state = pred._ensure_state()
        return entry.fn(state, pred._params)

    def __contains__(self, name: str) -> bool:
        """Check if a property is available."""
        pred = object.__getattribute__(self, "_prediction")
        return name in pred._model._ensure_property_catalog()

    def __iter__(self):
        """Iterate over property names."""
        pred = object.__getattribute__(self, "_prediction")
        return iter(sorted(pred._model._ensure_property_catalog().keys()))

    def get(self, name: str, default=None):
        """Return the property value for ``name``, or ``default`` if absent.

        Mirrors :meth:`dict.get` so the catalog is safe to probe without a
        ``try``/``except KeyError`` when a property may not be available for
        this model (e.g. asking for ``"agn_luminosity"`` on a galaxy-only fit).

        Parameters
        ----------
        name : str
            Property name (e.g., ``"stellar_mass"``).
        default : object, optional
            Value to return when ``name`` is not available. Default ``None``.

        Returns
        -------
        scalar or object
            Computed property value, or ``default`` if ``name`` is unknown.
        """
        if name in self:
            return self[name]
        return default

    def values(self):
        """Return computed values for every available property.

        Returns
        -------
        list
            One value per name in :meth:`keys`, in the same order.
        """
        return [self[name] for name in self]

    def items(self):
        """Return ``(name, value)`` pairs for every available property.

        Returns
        -------
        list of tuple
            ``(name, value)`` for each name in :meth:`keys`.
        """
        return [(name, self[name]) for name in self]

    # ``keys`` / ``to_dict`` / read-only ``__setattr__`` come from
    # ReadOnlyPropertyMapping (#1431). ``get`` / ``values`` / ``items`` stay
    # here: they return plain lists, which a Mapping base would turn into views.


from tengri.parameters.resolve import resolve_fixed_params

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
    catalogs), use :meth:`SEDModel.predict_properties` — the one
    JIT/vmap-safe surface for derived quantities — or
    :meth:`SEDModel.predict_photometry` on an inference hot path. Both
    return plain JAX values suitable for :func:`jax.vmap`,
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
    >>> pred.photometry()  # shape (n_filters,) -- call it, it is a method

    **For batch computation, use JIT-compatible methods instead:**

    >>> batch = jax.vmap(lambda p: model.predict_properties(p, names=("stellar_mass",)))(
    ...     params_batch
    ... )
    """

    __slots__ = (
        "_cache",
        "_model",
        "_params",
        "_photometry_cache",
        "ionizing",
        "lines",
        "properties",
        "radio",
        "sed",
        "sfh",
        "xray",
    )

    def __init__(self, model, params):
        self._model = model
        self._params = resolve_fixed_params(model, params)
        self._cache = {}
        self._photometry_cache = {}
        self.sfh = SFHProperties(self)
        self.sed = SEDProperties(self)
        self.lines = LineProperties(self)
        self.radio = RadioProperties(self)
        self.xray = XRayProperties(self)
        self.ionizing = IonizingProperties(self)
        self.properties = PropertyCatalog(self)

    def __getattr__(self, name: str):
        """Attribute-access sugar for derived properties.

        Supports ``pred.stellar_mass`` as a shorthand for
        ``pred.properties["stellar_mass"]``. Falls back to AttributeError
        if the name is not in the property catalog.

        Parameters
        ----------
        name : str
            Property name.

        Returns
        -------
        scalar
            Computed property value from the property catalog.

        Raises
        ------
        AttributeError
            If the name is not a known property, with a message listing
            available properties.

        Notes
        -----
        This method is only called when standard attribute lookup fails,
        so it cannot shadow normal attributes like ``_model``, ``_params``,
        or the ``sfh``/``sed``/``lines`` groups (which are set in __init__).
        """
        # Avoid infinite recursion by checking if we're still initializing
        if name.startswith("_"):
            raise AttributeError(f"No attribute '{name}'")
        try:
            properties = object.__getattribute__(self, "properties")
            if name in properties:
                return properties[name]
        except AttributeError:
            pass
        # Not a property — raise AttributeError
        props = object.__getattribute__(self, "properties")
        available = sorted(props.keys()) if props else []
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Available properties: {available}"
        )

    def __dir__(self) -> list[str]:
        """Include property names in tab completion and dir().

        Returns
        -------
        list[str]
            All normal attributes plus property catalog names for autocomplete.
        """
        base = [
            "_cache",
            "_model",
            "_params",
            "ionizing",
            "lines",
            "properties",
            "radio",
            "sed",
            "sfh",
            "xray",
        ]
        properties = object.__getattribute__(self, "properties")
        if properties:
            base.extend(sorted(properties.keys()))
        return base

    def __jax_array__(self):
        """Guard against accidental JIT/vmap tracing of Prediction objects.

        Raises
        ------
        TypeError
            Always — with a message directing users to use
            ``model.predict_properties(...)``, which is JIT-compatible.
        """
        raise TypeError(
            "Prediction objects are not JIT/vmap-compatible due to Python-level "
            "caching. For batch computations use model.predict_properties("
            "params, names=(...)) — the one JIT/vmap-safe surface for derived "
            "quantities — or model.predict_photometry(params) on an inference "
            "hot path. Both return plain JAX values suitable for jax.vmap "
            "and jax.jit."
        )

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

    def _ensure_state(self):
        """Ensure the ForwardState is cached and return it.

        Returns
        -------
        ForwardState
            The cached orchestrator state, computed on first call to
            _ensure_sfh and reused by all downstream consumers.
        """
        if "_state" not in self._cache:
            self._ensure_sfh()
        return self._cache["_state"]

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
        # behavior without raising). Reuse the ForwardState cached by
        # _ensure_sfh() — re-running predict_state here doubled the
        # forward pass (and its transient memory) for pred.lines.
        state = self._cache["_state"]
        derived = state.derived
        if "line_waves" in derived and "line_lums" in derived:
            waves = jnp.asarray(derived["line_waves"])
            lums = jnp.asarray(derived["line_lums"])
            self._cache["line_waves"] = waves
            # Backends publish INTRINSIC line_lums; apply dust reddening so the
            # interactive catalog is observed-frame, matching predict_line_fluxes
            # (single-sourced via SEDModel._attenuate_line_catalog). Without this
            # the .lines catalog is silently intrinsic despite its docstring.
            self._cache["line_lums"] = model._attenuate_line_catalog(self._params, waves, lums)
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

    def photometry(self, filters=None, fast=False):
        r"""Observed-frame photometric flux densities.

        Integrates the model SED through each filter and applies the cosmological
        dimming, returning :math:`F_\nu` in the observer frame. For the same
        quantity as AB magnitudes, see :meth:`magnitudes`.

        **Exact by default, fast by choice.** The default integrates the full SED
        — including on a model built with ``approx=WavePrecomp(...)``, where the
        lean :meth:`~tengri.SEDModel.predict_photometry` would instead read the
        lookup table. That is deliberate: ``pred.photometry()`` must mean the same
        thing on every model. The LUT is an approximation carrying real error (of
        order a few percent at high redshift), so you reach it only by asking for
        it, with ``fast=True``.

        Parameters
        ----------
        filters : sequence of str or FilterCurve, optional
            Filter names (e.g., ``["jwst_f356w", "jwst_f444w"]``) or
            :class:`~tengri.observation.photometry.FilterCurve` objects.
            If None, uses the filters configured at model build time.
        fast : bool, optional
            If True, use the build-time WavePrecomp LUT (20–50× faster).
            Only valid when the model was built with ``approx=WavePrecomp(...)``.
            If `filters` is also provided, raises ValueError.
            Default: False (exact path).

        Returns
        -------
        ndarray, shape (n_filters,)
            Observed-frame flux densities [erg/s/cm²/Hz].

        Raises
        ------
        ValueError
            If the model was not built with ``approx=WavePrecomp(...)`` but
            ``fast=True`` is requested, or if both `filters` and `fast=True`.
        ValueError
            If no photometry is configured on the model.

        Notes
        -----
        **JIT-compatible**: no — Python method with caching. Use in postprocessing,
        not inside :func:`jax.jit`.

        **Default (exact) path**: integrates ``state.sed_intrinsic`` through each
        filter via :func:`~tengri.observation.photometry.project_photometry` — the
        same kernel the likelihood uses — on the ``ForwardState`` this Prediction
        already cached, so it costs the filter integration and not a second
        forward pass.

        **Fast path** (``fast=True``): Uses the precomputed photometric LUT
        (filter effective wavelengths × SSP grid). Requires
        ``approx=WavePrecomp(...)`` at model build time. Runtime cost is
        ~150 μs per forward model (20–50× speedup for typical models).

        **Runtime filters** (``filters=...``): Constructs a runtime Photometry
        object from filter names or objects. Cached on ``self._photometry_cache``
        keyed by filter tuple for zero-cost repeat calls. Issues a one-time
        UserWarning (exact-path cost). The exact path is always used; a LUT
        built for one filter set cannot cover filters it was never built for.

        Examples
        --------
        **Default exact path:**

        >>> pred = model.predict(params)
        >>> phot = pred.photometry()  # shape (n_filters,)
        >>> phot.shape
        (8,)

        **Fast path (requires WavePrecomp):**

        >>> phot_fast = pred.photometry(fast=True)  # ≈ same result, 20–50× faster
        >>> jnp.allclose(phot, phot_fast, rtol=1e-12)
        True

        **Runtime filter set:**

        >>> phot_custom = pred.photometry(filters=["jwst_f356w", "jwst_f444w"])
        >>> phot_custom.shape
        (2,)
        """
        # A model built without an observation is a valid rest-frame-only model
        # (pred.rest_sed(), pred.stellar_mass, ... all work), but has nothing to
        # project photometry onto. Name the fix instead of crashing with a bare
        # ``'NoneType' object has no attribute 'photometry'`` — the lean
        # model.predict_photometry() already raises a helpful ValueError here.
        if self._model.observation is None:
            raise ValueError(
                "No observation is configured on the model, so photometry cannot "
                "be projected. Build the model with observation=... (or filters=...) "
                "to get photometry; without one, only rest-frame quantities "
                "(pred.rest_sed(), pred.stellar_mass, ...) are available."
            )
        # Validate arguments
        if fast and filters is not None:
            raise ValueError(
                "fast=True and filters=[...] are mutually exclusive. "
                "A LUT built for one filter set cannot cover filters it was never built for."
            )

        # Mode 1: Fast path (WavePrecomp LUT).
        #
        # Route through the lean jitted shortcut, NOT through
        # ``predict_via_precomp`` on ``self._ensure_state()``. The whole saving of
        # WavePrecomp is that XLA dead-code-eliminates the full-resolution SED
        # einsum when only the LUT is consumed — and ``_ensure_state`` materializes
        # that state eagerly, so the einsum runs and the saving is already spent.
        # Going through the state made ``fast=True`` *slower* than the exact
        # default (measured 0.7-0.8x) while also returning an approximation: the
        # worst of both. ``predict_photometry`` is the same LUT to ~1 ULP and keeps
        # the elision, so it is both faster and the number the fit itself sees.
        if fast:
            if not self._model._approx.get("wave_precomp"):
                raise ValueError(
                    "fast=True requires the model to be built with approx=WavePrecomp(...). "
                    "Rebuild the model with approx=WavePrecomp() and try again."
                )
            return self._model.predict_photometry(self._params)

        # Mode 2: Runtime filters
        if filters is not None:
            # The convention (Bessell photon-counting vs energy, ADR-0017) is part
            # of the filter integral, not of the filter. ``Photometry.from_names``
            # defaults to Bessell, so resolving runtime filters without passing the
            # model's own convention silently answers a different question than
            # ``photometry()`` does — same filters, two numbers (~0.5% apart on an
            # energy-convention model). It is part of the cache key for the same
            # reason.
            from tengri.observation.photometry_config import resolve_runtime_photometry
            from tengri.utils.filter_convention import FilterConvention

            build_time = self._model.observation.photometry
            convention = (
                build_time.convention if build_time is not None else FilterConvention.BESSELL
            )

            filter_names = tuple(f if isinstance(f, str) else f.name for f in filters)
            cache_key = (filter_names, convention)
            if cache_key in self._photometry_cache:
                phot_obj = self._photometry_cache[cache_key]
            else:
                # The shared normalizer, so ``filters=`` resolves identically here
                # and on Posterior.observables (#1129).
                phot_obj = resolve_runtime_photometry(filters, build_time=build_time)
                self._photometry_cache[cache_key] = phot_obj
                _warn_runtime_photometry_once()

            from tengri.observation.photometry import project_photometry

            return project_photometry(self._ensure_state(), self._params, phot_obj)

        # Mode 3: Default — the EXACT path, on the build-time filters.
        #
        # This deliberately does NOT call ``model.predict_photometry``: that is
        # the lean inference shortcut, and on a model built with
        # ``approx=WavePrecomp(...)`` it returns the LUT. Routing the default
        # here through it would make ``pred.photometry()`` silently mean "exact"
        # on one model and "approximate" on another, which is precisely the
        # ambiguity the exact-by-default rule exists to kill. The fast path is
        # reachable — but only by asking for it, with ``fast=True``.
        if self._model.observation.photometry is None:
            raise ValueError(
                "No photometry is configured on the model. "
                "Build the model with observation=Observation(photometry=...) and try again."
            )
        from tengri.observation.photometry import project_photometry

        return project_photometry(
            self._ensure_state(), self._params, self._model.observation.photometry
        )

    def magnitudes(self, filters=None, fast=False):
        r"""AB magnitudes through filters.

        Computes AB magnitudes by converting observed flux densities from
        :meth:`photometry` using the AB magnitude system.

        Parameters
        ----------
        filters : sequence of str or FilterCurve, optional
            Filter names or :class:`~tengri.observation.photometry.FilterCurve`
            objects. If None, uses the filters configured at model build time.
        fast : bool, optional
            If True, use the build-time WavePrecomp LUT (20–50× faster).
            Only valid when the model was built with ``approx=WavePrecomp(...)``.
            If `filters` is also provided, raises ValueError.
            Default: False (exact path).

        Returns
        -------
        ndarray, shape (n_filters,)
            AB magnitudes [dimensionless].

        Raises
        ------
        ValueError
            If the model was not built with ``approx=WavePrecomp(...)`` but
            ``fast=True`` is requested, or if both `filters` and `fast=True`.
        ValueError
            If no photometry is configured on the model.

        Notes
        -----
        **JIT-compatible**: no — Python method delegating to :meth:`photometry`.
        Use in postprocessing, not inside :func:`jax.jit`.

        **Semantics**: Computes :meth:`photometry` in the requested mode
        (exact, fast, or runtime filters), then converts flux densities to
        AB magnitudes via:

        .. math::

            m_{\mathrm{AB}} = -2.5 \log_{10}(f_\nu / f_0)

        where f_0 = 3.631×10⁻²⁰ erg s⁻¹ cm⁻² Hz⁻¹ is the AB zeropoint.

        References
        ----------
        Oke & Gunn 1983, ApJ, 266, 713 — AB magnitude definition.

        Examples
        --------
        >>> pred = model.predict(params)
        >>> mags = pred.magnitudes()  # AB mags, shape (n_filters,)
        >>> mags[0]  # first filter AB magnitude
        20.5
        """
        from tengri.utils.magnitudes import fnu_to_ab_mag

        fnu = self.photometry(filters=filters, fast=fast)
        return fnu_to_ab_mag(fnu)

    def spectrum(self, wave_obs=None, fast=False):
        r"""Instrument-grid, LSF-convolved spectrum.

        Computes the observed-frame spectrum at the instrument wavelength grid,
        convolved with the line-spread function and calibrated.

        Parameters
        ----------
        wave_obs : ndarray, optional
            Custom observed-frame wavelength grid [Angstrom]. If None, uses
            the grid configured in the spectroscopy setup at model build time.
        fast : bool, optional
            If True, use the build-time ``SpectrumPrecomp`` LUT. Only valid when
            the model was built with ``approx=SpectrumPrecomp(...)``.
            Default False — the exact projector.

        Returns
        -------
        ndarray, shape (n_wave_obs,)
            Observed-frame flux density [erg/s/cm^2/Hz].

        Raises
        ------
        ValueError
            If no spectroscopy is configured on the model, or if ``fast=True``
            on a model not built with ``approx=SpectrumPrecomp(...)``.

        Notes
        -----
        **JIT-compatible**: no — Python accessor delegating to
        :meth:`SEDModel.predict_spectrum`. Use in postprocessing,
        not inside :func:`jax.jit`.

        **Naming contract**: SED = panchromatic model-grid array
        (:meth:`rest_sed` / :meth:`obs_sed`); spectrum = instrument-grid,
        LSF-convolved, calibrated observable (:meth:`spectrum`).

        **Preprocessing**: The instrument LSF (Gaussian with resolution R),
        velocity broadening, and calibration polynomial all apply on top
        of the SED redshift.

        Examples
        --------
        >>> pred = model.predict(params)
        >>> spec = pred.spectrum()  # shape (n_wave,)
        >>> spec.shape
        (2048,)
        """
        if self._model.observation is None or self._model.observation.spectroscopy is None:
            raise ValueError(
                "No spectroscopy is configured on the model. "
                "Build the model with observation=Observation(spectroscopy=...) and try again."
            )

        # Same rule as :meth:`photometry`, for the same reason. ``predict_spectrum``
        # honors the SpectrumPrecomp LUT, so defaulting to it would make
        # ``pred.spectrum()`` mean "exact" on one model and "approximate" on another
        # — measured 5-7% apart on a SpectrumPrecomp model, the same order as the
        # photometry LUT error, and not a rounding difference.
        if fast:
            if not self._model._approx.get("spectrum_precomp"):
                raise ValueError(
                    "fast=True requires the model to be built with "
                    "approx=SpectrumPrecomp(...). Rebuild the model with "
                    "approx=SpectrumPrecomp() and try again."
                )
            return self._model.predict_spectrum(self._params, wave_obs=wave_obs)

        # Exact: project the cached ForwardState through the shared spectrum
        # projector. ``Observation.predict`` is the canonical exact path — it calls
        # ``project_spectrum`` (#1052) and applies the flux calibration (#1086) —
        # and it never falls through to the LUT.
        out = self._model.observation.predict(
            self._ensure_state(), self._params, wave_obs=wave_obs
        )
        return out["spec_fnu"]

    @property
    def rest_sed(self):
        r"""Rest-frame panchromatic SED — **call it**: ``pred.rest_sed()``.

        A uniform callable with a default, like :meth:`photometry`,
        :meth:`magnitudes` and :meth:`spectrum` (contract §4b.3).

        Parameters
        ----------
        wave : array_like, shape (n_out,), optional
            Rest-frame wavelength grid to resample onto [Angstrom]. Default
            ``None`` returns the SED on the model's own grid
            (:attr:`wave_rest`), with no interpolation.

        Returns
        -------
        ndarray, shape (n_wave,) or (n_out,)
            Rest-frame luminosity density [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: no — a postprocessing accessor. Use the lean
        ``model.predict_photometry`` / ``predict_properties`` shortcuts inside
        :func:`jax.jit`.

        **Naming contract**: SED = panchromatic model-grid array
        (:meth:`rest_sed` / :meth:`obs_sed`); spectrum = instrument-grid,
        LSF-convolved, calibrated observable (:meth:`spectrum`).

        **Grid**: the model's SSP grid, auto-extended when dust emission,
        radio or X-ray components are configured. The axis is
        :attr:`wave_rest` — the SED array does not carry it.

        **Resampling** is ``jnp.interp`` onto ``wave``, bit-exact with the
        wavelength argument of the deprecated ``model.predict_rest_sed``.

        Examples
        --------
        >>> pred = model.predict(params)
        >>> lnu = pred.rest_sed()  # the model's own grid
        >>> lnu = pred.rest_sed(np.logspace(3, 5, 500))  # your grid
        >>> plt.loglog(pred.wave_rest, pred.rest_sed())

        See Also
        --------
        obs_sed : Observed-frame SED (redshifted + IGM + DLA).
        wave_rest : The wavelength axis this SED lives on.
        """
        return _SEDCallable(self._rest_sed_on, "rest_sed")

    def _rest_sed_on(self, wave):
        """Rest-frame L_nu, optionally resampled onto ``wave`` [Angstrom]."""
        # The fast-nebular grid zeroes the Cue continuum, so ``sed_intrinsic``
        # would come back without the nebular continuum or the lines (#1665).
        # Same census as predict_spectrum / predict_spectral_indices.
        state = self._ensure_state()
        sed = state.sed_intrinsic
        if wave is None:
            return sed
        # Same interpolation the deprecated predict_rest_sed(wave=...) used, so
        # migrating a call site changes no number.
        return jnp.interp(jnp.asarray(wave), state.wave, sed)

    @property
    def obs_sed(self):
        r"""Observed-frame panchromatic SED — **call it**: ``pred.obs_sed()``.

        The rest-frame SED moved onto the observed-frame wavelength axis, with
        IGM (and DLA) absorption applied. "Observed" refers to the **frame**,
        not to a flux conversion.

        Parameters
        ----------
        wave_obs : array_like, shape (n_out,), optional
            **Observed**-frame wavelength grid to resample onto [Angstrom] —
            the same frame as :attr:`wave_obs`, and as :meth:`spectrum`. Default
            ``None`` returns the SED on the model's own grid.

        Returns
        -------
        ndarray, shape (n_wave,)
            Luminosity density L_nu [erg/s/Hz] — **not** a flux.

        Notes
        -----
        **Units — read this.** This returns **L_nu [erg/s/Hz]**, exactly like
        :meth:`rest_sed`. It does **NOT** apply the cosmological dimming factor
        ``(1+z) / (4 pi d_L^2)``. The only differences from :meth:`rest_sed` are
        the wavelength axis and IGM absorption; at z = 3 the two arrays are
        identical everywhere above rest-frame Lyman-alpha.

        The distance is applied at the **projection** step, not here — see
        ``observation/redshift_kernel.py``. The surfaces that return a genuine
        flux F_nu [erg/s/cm^2/Hz] are :meth:`photometry`, :meth:`magnitudes`
        and :meth:`spectrum`. Integrating ``obs_sed()`` as if it were a flux is
        wrong by ~57 orders of magnitude.

        (This docstring previously claimed the opposite — "Returns F_nu, not
        L_nu ... accounts for the (1+z)/(4 pi d_L^2) dimming factor". It was
        false, and the claim had propagated into the naming contract. Measured,
        not assumed.)

        **JIT-compatible**: no — a postprocessing accessor.

        **Naming contract**: SED = panchromatic model-grid array
        (:meth:`rest_sed` / :meth:`obs_sed`); spectrum = instrument-grid,
        LSF-convolved, calibrated observable (:meth:`spectrum`).

        **Wavelength frame**: observed-frame (rest x (1+z)) [Angstrom]; the
        matching axis is :attr:`wave_obs`.

        **Absorption**: Includes:

        - IGM Lyman absorption (Inoue et al. 2014 [1]_) when ``igm=True``
        - DLA (damping wing absorption) when ``dla=True``
        - Reionization epoch (CGM) when configured
        - Patchy reionization when configured

        References
        ----------
        .. [1] Inoue A. K., et al. 2014, MNRAS, 442, 1805 — IGM absorption
           tables and mean transmission.

        Examples
        --------
        >>> pred = model.predict(params)
        >>> fnu = pred.obs_sed()  # the model's own grid
        >>> fnu = pred.obs_sed(np.logspace(3, 5, 500))  # your grid
        >>> plt.loglog(pred.wave_obs, pred.obs_sed())

        See Also
        --------
        rest_sed : Rest-frame SED (no absorption, no redshift).
        wave_obs : The observed-frame wavelength axis this SED lives on.
        spectrum : Instrument-grid observable (LSF-convolved).
        """
        return _SEDCallable(self._obs_sed_on, "obs_sed")

    def _obs_sed_on(self, wave_obs):
        """Observed-frame F_nu, optionally resampled onto ``wave_obs`` [Angstrom].

        Refuses on a fast-nebular model for the same reason as
        :meth:`_rest_sed_on` — this is the same SED, on a different axis (#1665).

        ``wave_obs`` is OBSERVED-frame, matching this SED's own axis
        (:attr:`wave_obs`) and :meth:`spectrum`. The deprecated
        ``model.predict_obs_sed(params, wave=...)`` took a *rest*-frame grid
        and redshifted it — an observed-frame result with a rest-frame
        argument. That asymmetry was a footgun; it is not reproduced here.
        """
        result = self._model._predict_obs_sed(self._params)
        if wave_obs is None:
            return result.sed
        return jnp.interp(jnp.asarray(wave_obs), result.wavelength, result.sed)

    @property
    def wave_rest(self):
        r"""Rest-frame wavelength grid.

        Returns the wavelength grid on which ``rest_sed`` and ``obs_sed`` are
        evaluated. The grid is determined by the model's SSP wavelength array,
        optionally extended for dust emission (IR), radio, and X-ray components.

        Returns
        -------
        ndarray, shape (n_wave,)
            Rest-frame wavelengths [Angstrom].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.

        Pairs exactly with ``rest_sed`` and forms the x-axis for plotting
        the rest-frame SED. The same wavelength grid is shared with derived
        properties (e.g., ``l_bol``, ``l_tir``, ``fuv_flux``) that integrate
        over it.

        Examples
        --------
        >>> pred = model.predict(params)
        >>> wave_rest = pred.wave_rest  # shape (n_wave,)
        >>> sed_rest = pred.rest_sed()  # the SED is a call, not an attribute
        >>> wave_rest.shape == sed_rest.shape
        True

        See Also
        --------
        wave_obs : Observed-frame wavelength grid.
        rest_sed : Rest-frame SED on this grid.
        """
        self._ensure_sfh()
        return self._cache["_state"].wave

    @property
    def wave_obs(self):
        r"""Observed-frame wavelength grid.

        Returns the observed-frame wavelength grid, computed as the rest-frame
        wavelengths shifted by redshift: ``wave_obs = wave_rest * (1 + z)``.
        Pairs exactly with ``obs_sed``.

        Returns
        -------
        ndarray, shape (n_wave,)
            Observed-frame wavelengths [Angstrom].

        Notes
        -----
        **JIT-compatible**: no — Python property accessor. Use in postprocessing,
        not inside :func:`jax.jit`.

        The redshift value is resolved through the model's spec:

        - If ``redshift`` is a free parameter, it is taken from ``params``.
        - If ``redshift`` is fixed in the spec, the fixed value is used.
        - If ``redshift`` is not found in either place, raises ``KeyError``.

        This design avoids the silent bugs (#1097, #1124, #1127) that arose
        from using ``params.get("redshift", 0.0)``, which can silently return
        a default value instead of raising an error when redshift is absent.

        Examples
        --------
        >>> pred = model.predict(params)
        >>> wave_obs = pred.wave_obs  # shape (n_wave,)
        >>> sed_obs = pred.obs_sed()  # the SED is a call, not an attribute
        >>> wave_obs.shape == sed_obs.shape
        True
        >>> # For a model with Fixed redshift:
        >>> (wave_obs / wave_rest - 1).mean()  # should be z
        0.1

        See Also
        --------
        wave_rest : Rest-frame wavelength grid.
        obs_sed : Observed-frame SED on this grid.
        """
        z = self._model._get_redshift(self._params)
        return self.wave_rest * (1.0 + z)

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
