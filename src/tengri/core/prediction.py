"""Lazy prediction object for derived physical quantities.

The :class:`Prediction` class provides on-demand computation of derived
quantities from a diffsed forward model. Properties are only computed
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

from diffsed.models.sps.dsps_wrapper import (
    compute_csp_weights,
    compute_surviving_mass,
    interpolate_mass_remaining,
)
from diffsed.utils.sed_quantities import (
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

# ---------------------------------------------------------------------------
# NamedTuples for JIT-compatible batch computation
# ---------------------------------------------------------------------------


class SFHQuantities(NamedTuple):
    """Derived quantities from the star formation history.

    All fields are JAX arrays (scalars). This is a proper JAX pytree,
    so it works with ``jax.jit``, ``jax.vmap``, and ``jax.grad``.

    Attributes
    ----------
    stellar_mass : jnp.ndarray
        Total formed stellar mass (Msun).
    stellar_mass_surviving : jnp.ndarray
        Surviving mass in living stars + remnants (Msun).
        NaN if the mass-remaining table was not loaded.
    sfr_100myr : jnp.ndarray
        Star formation rate averaged over the last 100 Myr (Msun/yr).
    sfr_10myr : jnp.ndarray
        Star formation rate averaged over the last 10 Myr (Msun/yr).
    ssfr : jnp.ndarray
        Specific star formation rate SFR/M* (yr⁻¹).
    mass_weighted_age_gyr : jnp.ndarray
        Mass-weighted stellar age (Gyr).
    mass_weighted_metallicity : jnp.ndarray
        Mass-weighted metallicity log10(Z). Evolving-Z aware.
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
        Bolometric luminosity (Lsun).
    l_tir : jnp.ndarray
        Total infrared luminosity 8–1000 μm (Lsun).
    l_dust_absorbed : jnp.ndarray
        Dust-absorbed luminosity (Lsun). NaN if no intrinsic SED.
    irx : jnp.ndarray
        Infrared excess log10(L_TIR / νLν_1600).
    uv_slope_beta : jnp.ndarray
        UV spectral slope β (1250–2600 Å).
    dn4000 : jnp.ndarray
        Narrow 4000 Å break (Balogh et al. 1999).
    balmer_break : jnp.ndarray
        Modified Balmer break (Wang et al. 2024).
    m_uv : jnp.ndarray
        Absolute UV magnitude at rest-frame 1500 Å (AB).
    fuv_flux : jnp.ndarray
        Mean flux density in FUV 1000–1700 Å (erg/s/Hz).
    nuv_flux : jnp.ndarray
        Mean flux density in NUV 1700–3200 Å (erg/s/Hz).
    fuv_flux_intrinsic : jnp.ndarray
        Dust-free FUV flux (erg/s/Hz). NaN if no intrinsic SED.
    nuv_flux_intrinsic : jnp.ndarray
        Dust-free NUV flux (erg/s/Hz). NaN if no intrinsic SED.
    rest_uv_color : jnp.ndarray
        Rest-frame U-V color (AB magnitudes).
    luminosity_weighted_age_gyr : jnp.ndarray
        Luminosity-weighted age (Gyr).
    luminosity_weighted_metallicity : jnp.ndarray
        Luminosity-weighted metallicity log10(Z).
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
    """Key emission line luminosities in Lsun.

    NaN for all fields when no nebular model is active. For doublets
    ([OII], C IV), the luminosities of both components are summed.

    Attributes
    ----------
    lya : jnp.ndarray
        Lyman-alpha 1216 Å.
    civ_1549 : jnp.ndarray
        C IV doublet 1548+1551 Å (sum).
    oii : jnp.ndarray
        [OII] doublet 3726+3729 Å (sum).
    hbeta : jnp.ndarray
        H-beta 4861 Å.
    oiii_4959 : jnp.ndarray
        [OIII] 4959 Å.
    oiii_5007 : jnp.ndarray
        [OIII] 5007 Å.
    nii_6548 : jnp.ndarray
        [NII] 6548 Å.
    halpha : jnp.ndarray
        H-alpha 6563 Å.
    nii_6584 : jnp.ndarray
        [NII] 6584 Å.
    sii_6717 : jnp.ndarray
        [SII] 6717 Å.
    sii_6731 : jnp.ndarray
        [SII] 6731 Å.
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


class DerivedQuantities(NamedTuple):
    """All derived physical quantities (convenience container).

    Returned by ``model.predict_derived()`` for backward compatibility
    with the grouped API.

    Attributes
    ----------
    sfh : SFHQuantities
        Star formation history derived quantities.
    sed : SEDQuantities
        Spectral energy distribution derived quantities.
    """

    sfh: SFHQuantities
    sed: SEDQuantities


# ---------------------------------------------------------------------------
# Lazy property group base
# ---------------------------------------------------------------------------


class _CachedBase:
    """Base class for lazy-cached prediction property groups.

    Each subclass accesses the parent :class:`Prediction` object's
    shared cache via ``self._pred._cache`` and triggers the appropriate
    computation level via ``self._pred._ensure_*()`` methods.
    """

    __slots__ = ("_pred",)

    def __init__(self, prediction):
        self._pred = prediction


# ---------------------------------------------------------------------------
# SFH properties (lazy)
# ---------------------------------------------------------------------------


class SFHProperties(_CachedBase):
    """Lazy SFH-derived quantities.

    Accessing any property triggers SFH computation (SFR, CSP weights)
    if not already cached. Luminosity-weighted quantities additionally
    trigger SED computation.

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
        """Total formed stellar mass (Msun)."""
        self._pred._ensure_sfh()
        return jnp.sum(self._pred._cache["weights"])

    @property
    def stellar_mass_surviving(self):
        """Surviving stellar mass in living stars + remnants (Msun).

        Returns NaN if the mass-remaining table was not loaded in the
        SSP data.
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
        """SFR averaged over the last 100 Myr (Msun/yr)."""
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
        """SFR averaged over the last 10 Myr (Msun/yr)."""
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
        """Specific star formation rate SFR/M* (yr⁻¹).

        Uses surviving mass if available, otherwise formed mass.
        """
        mass_surv = self.stellar_mass_surviving
        mass = jnp.where(jnp.isnan(mass_surv), self.stellar_mass, mass_surv)
        return self.sfr_100myr / jnp.maximum(mass, 1.0)

    @property
    def mass_weighted_age_gyr(self):
        """Mass-weighted stellar age (Gyr)."""
        self._pred._ensure_sfh()
        return compute_mass_weighted_age(
            self._pred._cache["weights"], self._pred._model.ssp_ages_yr
        )

    @property
    def mass_weighted_metallicity(self):
        """Mass-weighted metallicity log10(Z).

        For single metallicity models, returns the metallicity parameter.
        For evolving metallicity, computes Σ(w·Z)/Σ(w).
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
        """Luminosity-weighted age (Gyr). Triggers SED computation."""
        self._pred._ensure_sed()
        return compute_luminosity_weighted_age(
            self._pred._cache["weights"],
            self._pred._cache["ssp_flux_at_z"],
            self._pred._model.ssp_ages_yr,
            self._pred._model.ssp_data.ssp_wave,
        )

    @property
    def luminosity_weighted_metallicity(self):
        """Luminosity-weighted metallicity log10(Z). Triggers SED computation."""
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


# ---------------------------------------------------------------------------
# SED properties (lazy)
# ---------------------------------------------------------------------------


class SEDProperties(_CachedBase):
    """Lazy SED-derived quantities.

    Accessing any property triggers the full SED computation (dust
    attenuation, emission, AGN, etc.) if not already cached.

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.sed.l_bol  # triggers SED computation
    Array(2.5e43, dtype=float64)
    >>> pred.sed.uv_slope_beta  # reuses cached SED
    Array(-1.8, dtype=float64)
    """

    def _wave(self):
        return self._pred._model.ssp_data.ssp_wave

    def _sed(self):
        self._pred._ensure_sed()
        return self._pred._cache["sed_total"]

    def _sed_intrinsic(self):
        self._pred._ensure_sed()
        return self._pred._cache.get("sed_intrinsic")

    @property
    def l_bol(self):
        """Bolometric luminosity (Lsun)."""
        return compute_bolometric_luminosity(self._sed(), self._wave())

    @property
    def l_tir(self):
        """Total infrared luminosity 8–1000 μm (Lsun)."""
        return compute_l_tir(self._sed(), self._wave())

    @property
    def l_dust_absorbed(self):
        """Dust-absorbed luminosity (Lsun). NaN if no intrinsic SED."""
        sed_intr = self._sed_intrinsic()
        if sed_intr is None:
            return jnp.array(jnp.nan)
        self._pred._ensure_sed()
        return compute_l_dust_absorbed(sed_intr, self._pred._cache["sed_attenuated"], self._wave())

    @property
    def irx(self):
        """Infrared excess IRX = log10(L_TIR / νLν_1600)."""
        l_tir = self.l_tir
        l_uv = compute_uv_luminosity_1600(self._sed(), self._wave())
        return compute_irx(l_tir, l_uv)

    @property
    def uv_slope_beta(self):
        """UV spectral slope β (1250–2600 Å)."""
        return compute_uv_slope_beta(self._sed(), self._wave())

    @property
    def dn4000(self):
        """Narrow 4000 Å break (Balogh et al. 1999)."""
        return compute_dn4000(self._sed(), self._wave())

    @property
    def balmer_break(self):
        """Modified Balmer break (Wang et al. 2024)."""
        return compute_balmer_break(self._sed(), self._wave())

    @property
    def m_uv(self):
        """Absolute UV magnitude at rest-frame 1500 Å (AB)."""
        return compute_m_uv(self._sed(), self._wave())

    @property
    def fuv_flux(self):
        """Mean flux density in FUV 1000–1700 Å (erg/s/Hz)."""
        return compute_fuv_flux(self._sed(), self._wave())

    @property
    def nuv_flux(self):
        """Mean flux density in NUV 1700–3200 Å (erg/s/Hz)."""
        return compute_nuv_flux(self._sed(), self._wave())

    @property
    def fuv_flux_intrinsic(self):
        """Dust-free FUV flux (erg/s/Hz). NaN if no intrinsic SED."""
        sed_intr = self._sed_intrinsic()
        if sed_intr is None:
            return jnp.array(jnp.nan)
        return compute_fuv_flux(sed_intr, self._wave())

    @property
    def nuv_flux_intrinsic(self):
        """Dust-free NUV flux (erg/s/Hz). NaN if no intrinsic SED."""
        sed_intr = self._sed_intrinsic()
        if sed_intr is None:
            return jnp.array(jnp.nan)
        return compute_nuv_flux(sed_intr, self._wave())

    @property
    def rest_uv_color(self):
        """Rest-frame U-V color (AB mag)."""
        return compute_rest_uv_color(self._sed(), self._wave())

    @property
    def luminosity_weighted_age_gyr(self):
        """Luminosity-weighted age (Gyr)."""
        self._pred._ensure_sed()
        return compute_luminosity_weighted_age(
            self._pred._cache["weights"],
            self._pred._cache["ssp_flux_at_z"],
            self._pred._model.ssp_ages_yr,
            self._pred._model.ssp_data.ssp_wave,
        )

    @property
    def luminosity_weighted_metallicity(self):
        """Luminosity-weighted metallicity log10(Z)."""
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


# ---------------------------------------------------------------------------
# Emission line properties (lazy)
# ---------------------------------------------------------------------------


class LineProperties(_CachedBase):
    """Lazy emission line luminosities and diagnostic ratios.

    Accessing any line property triggers the nebular computation
    if not already cached. All luminosities are in Lsun. Diagnostic
    ratios are dimensionless log10 values.

    If no nebular model is active, all line luminosities return NaN
    and all ratios return NaN.

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.lines.halpha  # Hα luminosity in Lsun
    Array(1.5e8, dtype=float64)
    >>> pred.lines.bpt_nii  # log10([NII]6584 / Hα)
    Array(-0.45, dtype=float64)
    """

    def _get_line(self, name):
        self._pred._ensure_lines()
        lw = self._pred._cache["line_waves"]
        ll = self._pred._cache["line_lums"]
        return extract_line_luminosity(lw, ll, KEY_LINES[name])

    # --- Individual lines ---

    @property
    def lya(self):
        """Lyman-alpha 1216 Å (Lsun)."""
        return self._get_line("lya")

    @property
    def civ_1549(self):
        """C IV doublet 1548+1551 Å (Lsun, sum)."""
        return self._get_line("civ_1549")

    @property
    def oii(self):
        """[OII] doublet 3726+3729 Å (Lsun, sum)."""
        return self._get_line("oii")

    @property
    def hbeta(self):
        """H-beta 4861 Å (Lsun)."""
        return self._get_line("hbeta")

    @property
    def oiii_4959(self):
        """[OIII] 4959 Å (Lsun)."""
        return self._get_line("oiii_4959")

    @property
    def oiii_5007(self):
        """[OIII] 5007 Å (Lsun)."""
        return self._get_line("oiii_5007")

    @property
    def nii_6548(self):
        """[NII] 6548 Å (Lsun)."""
        return self._get_line("nii_6548")

    @property
    def halpha(self):
        """H-alpha 6563 Å (Lsun)."""
        return self._get_line("halpha")

    @property
    def nii_6584(self):
        """[NII] 6584 Å (Lsun)."""
        return self._get_line("nii_6584")

    @property
    def sii_6717(self):
        """[SII] 6717 Å (Lsun)."""
        return self._get_line("sii_6717")

    @property
    def sii_6731(self):
        """[SII] 6731 Å (Lsun)."""
        return self._get_line("sii_6731")

    # --- Diagnostic ratios ---

    @property
    def bpt_nii(self):
        """BPT-NII ratio: log10([NII]6584 / Hα)."""
        return jnp.log10(jnp.maximum(self.nii_6584, 1e-50) / jnp.maximum(self.halpha, 1e-50))

    @property
    def bpt_sii(self):
        """BPT-SII ratio: log10(([SII]6717+6731) / Hα)."""
        sii_total = self.sii_6717 + self.sii_6731
        return jnp.log10(jnp.maximum(sii_total, 1e-50) / jnp.maximum(self.halpha, 1e-50))

    @property
    def o3hb(self):
        """[OIII]5007/Hβ ratio: log10([OIII]5007 / Hβ). BPT y-axis."""
        return jnp.log10(jnp.maximum(self.oiii_5007, 1e-50) / jnp.maximum(self.hbeta, 1e-50))

    @property
    def r23(self):
        """R23 metallicity indicator: log10(([OII]+[OIII]4959+5007)/Hβ)."""
        numerator = self.oii + self.oiii_4959 + self.oiii_5007
        return jnp.log10(jnp.maximum(numerator, 1e-50) / jnp.maximum(self.hbeta, 1e-50))

    @property
    def o32(self):
        """O32 ionization parameter: log10([OIII]5007 / [OII])."""
        return jnp.log10(jnp.maximum(self.oiii_5007, 1e-50) / jnp.maximum(self.oii, 1e-50))

    @property
    def balmer_decrement(self):
        """Balmer decrement Hα/Hβ. Case B value: 2.86."""
        return self.halpha / jnp.maximum(self.hbeta, 1e-50)


# ---------------------------------------------------------------------------
# Radio properties (lazy)
# ---------------------------------------------------------------------------


class RadioProperties(_CachedBase):
    """Lazy radio-derived quantities from empirical scaling relations.

    These use the FIR-radio correlation (Bell 2003; Murphy+2011)
    and free-free emission from the ionizing photon budget.

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.radio.l_1p4ghz  # 1.4 GHz luminosity
    Array(5.2e28, dtype=float64)
    """

    @property
    def l_1p4ghz(self):
        """Radio luminosity at 1.4 GHz (erg/s/Hz) from SFR."""
        sfr = self._pred.sfh.sfr_100myr
        return compute_l_radio_1p4ghz_from_sfr(sfr)

    @property
    def l_thermal(self):
        """Thermal (free-free) radio luminosity at 1.4 GHz (erg/s/Hz)."""
        q_h = self._pred.ionizing.q_h
        return compute_l_radio_thermal(q_h)

    @property
    def l_nonthermal(self):
        """Non-thermal (synchrotron) radio luminosity (erg/s/Hz)."""
        return self.l_1p4ghz - self.l_thermal

    @property
    def q_ir(self):
        """FIR-radio correlation parameter q_TIR."""
        l_tir = self._pred.sed.l_tir
        return compute_q_ir(l_tir, self.l_1p4ghz)


# ---------------------------------------------------------------------------
# X-ray properties (lazy)
# ---------------------------------------------------------------------------


class XRayProperties(_CachedBase):
    """Lazy X-ray derived quantities from empirical scaling relations.

    Uses Lehmer et al. (2010, 2016) for XRBs and Duras et al. (2020)
    for AGN bolometric corrections.

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.xray.l_x_xrb  # XRB luminosity (0.5-8 keV)
    Array(3.1e40, dtype=float64)
    """

    @property
    def l_x_xrb(self):
        """X-ray luminosity from XRBs, 0.5–8 keV (erg/s)."""
        sfr = self._pred.sfh.sfr_100myr
        mstar = self._pred.sfh.stellar_mass
        return compute_l_x_xrb(sfr, mstar)

    @property
    def l_x_agn(self):
        """AGN X-ray luminosity, 2–10 keV (erg/s)."""
        self._pred._ensure_sed()
        agn_bol = self._pred._cache.get("agn_bol_erg", 0.0)
        return compute_l_x_agn(agn_bol)

    @property
    def l_x_total(self):
        """Total X-ray luminosity (erg/s)."""
        return self.l_x_xrb + self.l_x_agn


# ---------------------------------------------------------------------------
# Ionizing properties (lazy)
# ---------------------------------------------------------------------------


class IonizingProperties(_CachedBase):
    """Lazy ionizing photon budget quantities.

    The ionizing photon rate Q_H is extracted from the nebular model
    backend (Cloudy grid or Cue emulator). If no nebular model is
    active, returns NaN.

    Examples
    --------
    >>> pred = model.predict(params)
    >>> pred.ionizing.xi_ion  # ionizing efficiency
    Array(25.3, dtype=float64)
    """

    @property
    def q_h(self):
        """Total ionizing photon production rate (photons/s).

        NaN if no nebular model is active.
        """
        self._pred._ensure_lines()
        return self._pred._cache.get("q_h_total", jnp.array(jnp.nan))

    @property
    def xi_ion(self):
        """Ionizing photon production efficiency log10(ξ_ion) (Hz/erg).

        Defined as Q_H / L_UV(1600 Å). Key parameter for cosmic
        reionization studies. Typical values: 25.0–25.6.
        """
        q_h = self.q_h
        l_uv = compute_uv_luminosity_1600(self._pred.sed._sed(), self._pred.sed._wave())
        return compute_ionizing_efficiency(q_h, l_uv)


# ---------------------------------------------------------------------------
# Main Prediction class
# ---------------------------------------------------------------------------


class Prediction:
    """Lazy prediction object with on-demand computation of derived quantities.

    Created via ``model.predict(params)``. Properties are computed on
    first access and cached. The cache is shared across all property
    groups (``sfh``, ``sed``, ``lines``, ``radio``, ``xray``,
    ``ionizing``), so related quantities share the expensive
    intermediates.

    Parameters
    ----------
    model : Model
        The diffsed Model instance.
    params : dict
        Parameter values (public names).

    Examples
    --------
    **Single-galaxy exploration:**

    >>> pred = model.predict(params)
    >>> pred.sfh.stellar_mass  # only SFH computed
    >>> pred.sed.l_bol  # full SED now computed
    >>> pred.lines.bpt_nii  # nebular lines now computed

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
        """Compute and cache SFH intermediates (SFR, weights, internal params)."""
        if "weights" in self._cache:
            return
        p = self._model._get_internal_params(self._params)
        sfr = self._model._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self._model.ssp_log_ages_yr, self._model.log_age_grid, sfr)
        weights = compute_csp_weights(sfr_on_ssp, self._model.ssp_ages_yr)
        self._cache.update({"p": p, "sfr": sfr, "weights": weights})

    def _ensure_sed(self):
        """Compute and cache full SED intermediates."""
        if "sed_total" in self._cache:
            return
        self._ensure_sfh()
        comp = self._model._compute_sed_components(
            self._params,
            _sfr=self._cache["sfr"],
            _weights=self._cache["weights"],
            need_intrinsic=True,
        )
        self._cache.update(comp)

    def _ensure_lines(self):
        """Compute and cache nebular emission line luminosities."""
        if "line_waves" in self._cache:
            return
        self._ensure_sfh()
        model = self._model
        backend = model._nebular_backend

        if backend is None or not hasattr(backend, "predict_nebular_line_luminosities"):
            self._cache["line_waves"] = jnp.array([])
            self._cache["line_lums"] = jnp.array([])
            self._cache["q_h_total"] = jnp.array(jnp.nan)
            return

        p = self._cache["p"]
        weights = self._cache["weights"]

        line_waves, line_lums = backend.predict_nebular_line_luminosities(
            ssp_weights=weights,
            ssp_log_ages_yr=model.ssp_log_ages_yr,
            log_z=p.get("log_z_abs", 0.0),
            neb_logU=p.get("neb_logU", -3.0),
            neb_logZ_gas=p.get("neb_logZ_gas", None),
            neb_fesc=p.get("neb_fesc", 0.0),
        )

        self._cache["line_waves"] = line_waves
        self._cache["line_lums"] = line_lums

        # Q_H: compute from backend's precomputed table if available
        if hasattr(backend, "_qh_table") and backend._qh_table is not None:
            log_z = p.get("log_z_abs", 0.0)
            young_idx = backend._young_idx
            young_ages = model.ssp_log_ages_yr[young_idx]
            young_weights = weights[young_idx]

            def _qh_one_bin(log_age_i, w_i):
                return w_i * backend._get_qh_at(log_z, log_age_i)

            import jax

            q_h_per_bin = jax.vmap(_qh_one_bin)(young_ages, young_weights)
            neb_fesc = p.get("neb_fesc", 0.0)
            self._cache["q_h_total"] = jnp.sum(q_h_per_bin) * (1.0 - neb_fesc)
        else:
            self._cache["q_h_total"] = jnp.array(jnp.nan)

    @property
    def sed_array(self):
        """Full rest-frame SED array in erg/s/Hz, shape (n_wave,)."""
        self._ensure_sed()
        return self._cache["sed_total"]

    @property
    def photometry(self):
        """Observed photometric flux densities, shape (n_filters,).

        Uses the Model's predict_photometry method.
        """
        return self._model.predict_photometry(self._params)
