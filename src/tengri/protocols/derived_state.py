# SPDX-License-Identifier: BSD-3-Clause
"""Typed container for cross-component derived quantities on :attr:`ForwardState.derived`.

``DerivedState`` is a frozen dataclass with one field per documented
inter-component quantity (``L_ir``, ``lnu_age``, …). Writers go through
``state.derived.with_(L_ir=...)``; a typo (``L_ie``) becomes a
``TypeError`` at trace time instead of a silent missing key.

Read sites keep using mapping syntax (``state.derived.get("L_ir", 0.0)``,
``state.derived["lnu_age"]``, ``"L_ir" in state.derived``) — DerivedState
implements ``__getitem__``, ``.get()``, ``__contains__``, ``keys()``,
``items()``, ``values()``, ``__iter__``, and ``__len__``. A field whose
value is ``None`` reads as "not present", so missing quantities behave
like a missing dict key.

Unknown-key errors include a Levenshtein-2 ``Did you mean: ...`` hint.

Adding a new derived key: add a field here, then add the same key to
``_CANONICAL_UNITS`` in :mod:`tengri.forward.orchestrator`. The field
set is hand-maintained on purpose — same friction model as the
inputs/outputs declaration on each component (ADR-0009), and a
deliberate guard against ``derived`` becoming a junk drawer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import Any

import jax.numpy as jnp

__all__ = ["DerivedState"]


_MISSING = object()  # sentinel for .get(name) without default — match dict.get


@dataclass(frozen=True)
class DerivedState:
    r"""Typed container for cross-component derived data.

    Fields mirror :data:`tengri.forward.orchestrator._CANONICAL_UNITS`
    one-for-one — every canonical derived key has a field with the same
    name. ``None`` means "not populated by any upstream component".

    Mutation via :meth:`with_` (the same pattern
    :class:`tengri.protocols.ForwardState` uses); never assign in place
    because the dataclass is frozen.

    Dict compatibility
    ------------------

    All of these work, returning ``True`` / the value / iterating
    non-None fields exactly as a plain ``dict`` would::

        "L_ir" in bundle
        bundle["L_ir"]
        bundle.get("L_ir", 0.0)
        bundle.keys()
        bundle.items()
        bundle.values()
        list(bundle)
        len(bundle)

    JAX pytree
    ----------

    Registered with :func:`jax.tree_util.register_dataclass` so the
    bundle can ride through ``jax.lax.scan`` / ``jax.tree_map`` along
    with the rest of :class:`ForwardState`. Every field is a *data*
    field — JAX traces each one. ``None`` defaults are static at trace
    time; switching which fields are populated invalidates the JIT
    cache, which is acceptable because a given :class:`tengri.SEDModel`
    pins its component list at construction and never reshuffles.
    """

    # Stellar — formed/surviving mass + SFR variants
    #
    # ``log_mstar`` falls back to the formed mass when the SSP grid carries no
    # mass-remaining table, because downstream normalization needs *a* mass.
    # ``log_mstar_surviving`` does not: it is NaN when the grid cannot answer,
    # so the user-facing ``stellar_mass_surviving`` and ``ssfr`` properties never
    # silently report the formed mass as the surviving one (#1131).
    log_mstar: jnp.ndarray | None = None
    log_mstar_formed: jnp.ndarray | None = None
    log_mstar_surviving: jnp.ndarray | None = None
    sfr: jnp.ndarray | None = None
    sfr_10myr: jnp.ndarray | None = None
    sfr_100myr: jnp.ndarray | None = None

    # Stellar — age-resolved tensors and ionizing rate
    L_age: jnp.ndarray | None = None
    #: float32-safe companion to ``L_age`` (#1534). ``L_age`` is ~1e42-1e46
    #: erg/s and reads as ``inf`` in pure float32; ``-inf`` here means a bin
    #: genuinely emits nothing, ``+inf`` that the input was corrupt (#1527).
    log_L_age: jnp.ndarray | None = None
    lnu_age: jnp.ndarray | None = None
    # DSPS joint (metallicity, age) weights and the total_mass x L_sun scaling,
    # published so DustSEDComponent can evaluate L_ir from a precomputed
    # bolometric (tau_bc, tau_diff) LUT instead of the full stellar cube.
    joint_weights: jnp.ndarray | None = None
    stellar_mass_scale: jnp.ndarray | None = None
    #: log10(total_mass x L_sun) [dex] — the float32-safe form of
    #: ``stellar_mass_scale``, which is ~1e43 and so overflows float32 for
    #: any galaxy above ~9e4 Msun (#1206).
    log_stellar_mass_scale: jnp.ndarray | None = None
    ssp_ages_yr: jnp.ndarray | None = None
    age_weights: jnp.ndarray | None = None
    nion: jnp.ndarray | None = None
    log_nion: jnp.ndarray | None = None

    # Stellar — SFH grid + chemistry history (diagnostic)
    sfh_grid_lbt_yr: jnp.ndarray | None = None
    sfr_history: jnp.ndarray | None = None
    log_metallicity_history: jnp.ndarray | None = None

    # Stellar — photometry LUT (published only when
    # ``approx=WavePrecomp()`` is set on SEDModel).
    # Rest-frame F_nu through the configured filters, in erg/s/Hz at
    # the source (no redshift / luminosity distance applied).
    stellar_phot_lnu_precomp: jnp.ndarray | None = None
    # Stellar — Taylor moment for filter-level dust attenuation
    # (published alongside stellar_phot_lnu_precomp). First
    # spectral moment of the CSP within each filter:
    # Ψ_b = ∫ L_ν(λ) (λ - λ_eff_b) T_b(λ) λ dλ / ∫ T_b(λ) λ dλ.
    # Used by the filter-level dust integration via the expansion
    # f_b ≈ A(λ_eff)·Φ_b + A'(λ_eff)·Ψ_b (Zacharegkas+2025).
    # Units: erg/s/Hz × Å.
    stellar_phot_moment_precomp: jnp.ndarray | None = None
    # Stellar — age-resolved per-filter LUT. Shape
    # ``(n_age, n_filter)``, units erg/s/Hz. The age axis is NOT
    # marginalized. Sum over the age axis equals
    # ``stellar_phot_lnu_precomp``. Published only when
    # ``approx=WavePrecomp()`` is set. Consumed by the two-
    # component dust LUT path to apply per-age
    # attenuation ``T(a, λ) = T_diff(λ) × T_bc(λ)^y(a)``.
    stellar_phot_lnu_per_age_precomp: jnp.ndarray | None = None
    stellar_phot_moment_per_age_precomp: jnp.ndarray | None = None

    # Sub-band quadrature for the multiplicative dust screen (#1122), shape
    # ``(n_age, n_filter, n_subbands)``. ``..._subband_precomp`` is the filter
    # integral restricted to each sub-band (sums over k to the per-age LUT);
    # ``..._waves_rest_precomp`` is that sub-band's quadrature node — the
    # template's own flux-weighted centroid, rest frame [Angstrom]. Both are
    # build-time constants, so the dust screen is EVALUATED at K points per band
    # rather than Taylor-extrapolated from one (which diverges in the rest-UV).
    stellar_phot_lnu_per_age_subband_precomp: jnp.ndarray | None = None
    stellar_subband_waves_rest_precomp: jnp.ndarray | None = None

    #: The same sub-band tensor with the IGM transmission folded in at the
    #: quadrature nodes, shape ``(n_age, n_filter, n_subbands)`` [erg/s/Hz]
    #: (#1135). Published only when a mean-IGM model is precomputable — patchy
    #: reionization and DLAs read free parameters, so ``T`` is not a function of
    #: ``(λ, z)`` alone and those configs keep the live per-call path.
    #:
    #: ``igm_phot_factor`` band-averages ``T`` *alone*, unweighted by the
    #: spectrum, forming ``⟨S⟩·⟨T⟩`` where the flux needs ``⟨S·T⟩``. Across GALEX
    #: FUV at z≈0.8 the transmission runs from ~1 to ~0 *inside* the bandpass and
    #: that covariance term reaches −9.5 %. Folding ``T`` into the sub-band
    #: weights captures it in the same contraction as the dust screen.
    #:
    #: Folded at BUILD time, into the ``(n_met, n_age, n_filter, n_subbands)``
    #: SSP node tensor — where the nodes actually live. The node is a
    #: metallicity-weighted average whose weights move with the free
    #: ``met_logzsol``, so a table keyed on ``(z, age, filter, k)`` alone would be
    #: valid at one metallicity only (the node shifts by up to 68 % of a sub-band
    #: width across the SSP grid; ``T`` there by up to 1.3 % in GALEX FUV). Folding
    #: before the met contraction handles that exactly, and for free — the runtime
    #: einsum is the same shape and cost as the IGM-free one.
    #:
    #: The IGM-free twin above is retained, not replaced: ``phot_rest_fnu`` is
    #: projected at z=0 and carries no IGM by contract.
    stellar_phot_lnu_per_age_subband_igm_precomp: jnp.ndarray | None = None

    # ── The REST-frame band (#1148) ────────────────────────────────────────────
    # ``phot_rest_fnu`` (and ``Observables.mag_absolute``) is the SED reprojected at
    # z=0, d_L=10 pc — *the galaxy as it is*. The filter therefore sits in the REST
    # frame and samples the rest SED at its OWN pivot wavelength. That is a
    # different integral from the ``*_phot_lnu_precomp`` family above, which places
    # the filter in the observed frame and so samples rest λ_eff/(1+z).
    #
    # The LUT used to reuse the observed-band tensors for the rest-frame flux. It
    # therefore reported a *different physical quantity* from the exact path — 769 %
    # apart in des_g at z=0.5, orders of magnitude in the blue — so an object's
    # ABSOLUTE magnitude depended on its redshift, and on which `approx` you passed.
    #
    # Every emitter publishes ``<name>_restband_lnu_precomp``; the projector
    # auto-discovers them exactly as it does the observed family, so a new component
    # gets rest-frame photometry for free and cannot silently omit it.
    #
    # None of these carry a redshift axis: the rest band does not move with z, so
    # they are build-time constants and cost nothing at runtime.
    stellar_restband_lnu_precomp: jnp.ndarray | None = None
    nebular_restband_lnu_precomp: jnp.ndarray | None = None
    # Its own integral at redshift=0, not the observed-band value reused: the
    # rest band samples the SED at its own pivot (#1148, #1375).
    shock_restband_lnu_precomp: jnp.ndarray | None = None
    agn_restband_lnu_precomp: jnp.ndarray | None = None
    radio_restband_lnu_precomp: jnp.ndarray | None = None
    xray_restband_lnu_precomp: jnp.ndarray | None = None
    #: Per-age sub-band quadrature over the rest band, shape (n_age, n_filter, K),
    #: and its nodes [Angstrom] — the #1122 machinery applied to the rest-frame
    #: projection, so the dust screen is EVALUATED across the band rather than
    #: extrapolated from its pivot.
    stellar_restband_lnu_per_age_subband_precomp: jnp.ndarray | None = None
    stellar_restband_subband_waves_precomp: jnp.ndarray | None = None
    #: The wavelength the rest band samples — the filter's own pivot, since the
    #: projection is at z=0 [Angstrom]. Contrast ``filter_eff_waves`` = pivot/(1+z),
    #: the rest wavelength the OBSERVED band samples.
    filter_restband_eff_waves: jnp.ndarray | None = None
    #: The dust screen evaluated on the rest band — at its pivot, and at its
    #: quadrature nodes. Separate from the observed-band screens because the two
    #: bands sample different rest wavelengths.
    dust_restband_attenuation_precomp: jnp.ndarray | None = None
    dust_bc_restband_attenuation_precomp: jnp.ndarray | None = None
    dust_diff_restband_attenuation_precomp: jnp.ndarray | None = None
    dust_restband_attenuation_subband_precomp: jnp.ndarray | None = None
    dust_bc_restband_attenuation_subband_precomp: jnp.ndarray | None = None
    dust_diff_restband_attenuation_subband_precomp: jnp.ndarray | None = None

    # Dust attenuation / emission
    L_ir: jnp.ndarray | None = None
    L_absorbed: jnp.ndarray | None = None
    #: log10(L_ir / (erg/s)) [dex] — the float32-safe form of ``L_ir``, which
    #: is ~1e43 and therefore outside the float32 range entirely (#1206).
    log_L_ir: jnp.ndarray | None = None
    dust_attenuation_factor: jnp.ndarray | None = None
    sed_dust_attenuated: jnp.ndarray | None = None
    sed_dust_ir: jnp.ndarray | None = None
    # Dust attenuation per filter. A(λ_eff) and its
    # wavelength derivative A'(λ_eff) at each filter pivot, used
    # to apply Taylor-expansion attenuation in
    # ``Observation.predict_via_precomp``:
    # f_b ≈ A(λ_eff)·Φ_b + A'(λ_eff)·Ψ_b
    # Shape ``(n_filters,)``. Published only when
    # ``approx=WavePrecomp()`` is set.
    dust_attenuation_precomp: jnp.ndarray | None = None
    dust_attenuation_slope_precomp: jnp.ndarray | None = None
    # Two-component dust. Birth-cloud and diffuse
    # attenuation factors per filter pivot, plus their wavelength
    # slopes. Two-component dust factorizes as
    # ``T(a, λ) = T_diff(λ) × T_bc(λ)^y(a)`` — the per-age dependence
    # comes from the young indicator ``y(a)`` below. Used to apply
    # the per-age expansion in predict_via_precomp.
    dust_bc_attenuation_precomp: jnp.ndarray | None = None
    dust_bc_attenuation_slope_precomp: jnp.ndarray | None = None
    dust_diff_attenuation_precomp: jnp.ndarray | None = None
    dust_diff_attenuation_slope_precomp: jnp.ndarray | None = None
    # The same two transmissions, evaluated at the sub-band quadrature nodes
    # (#1122), shape ``(n_age, n_filter, n_subbands)``. The law is evaluated live
    # on the node grid rather than tabulated, so its shape parameters (``n_slope``,
    # bump) stay FREE — no gate, unlike a tau-axis LUT.
    dust_bc_attenuation_subband_precomp: jnp.ndarray | None = None
    dust_diff_attenuation_subband_precomp: jnp.ndarray | None = None
    #: Single-screen counterpart, shape ``(n_age, n_filter, n_subbands)``.
    dust_attenuation_subband_precomp: jnp.ndarray | None = None
    # Log-attenuation slopes d(ln A)/dλ = −τ·k'(λ_eff), per filter. Published so
    # the two-component first-order Taylor projection (#617) can be written
    # T_a' = T_a·(logslope_diff + y·logslope_bc) — NaN-safe where A → 0 (avoids
    # the A_bc^(y−1) pole at X-ray/UV bands far off the dust curve).
    dust_bc_log_attenuation_slope_precomp: jnp.ndarray | None = None
    dust_diff_log_attenuation_slope_precomp: jnp.ndarray | None = None
    # Young-star indicator on the SSP age grid,
    # shape ``(n_age,)``. Smooth sigmoid transition around
    # ``DustSEDComponent.config.t_birth_yr``. ``y(a)`` = 1 for fully
    # young, 0 for fully old, with a logistic transition controlled by
    # ``transition_width_dex``.
    dust_young_indicator: jnp.ndarray | None = None
    # Filter pivot wavelengths in the rest frame (published by stellar
    # when wave_precomp is on; shared by downstream filter-level
    # consumers like the dust attenuation LUT). Shape ``(n_filters,)``,
    # units Å.
    filter_eff_waves: jnp.ndarray | None = None
    # Zero-padded observed-frame filter curves, published alongside
    # ``filter_eff_waves`` so additive, unattenuated emitters (dust IR, radio,
    # X-ray, AGN) can project their dense rest-frame SED through the *true*
    # filter transmission via ``lnu_filter_integral_batch`` — bit-exact vs the
    # exact path, instead of sampling at the effective wavelength. Shapes
    # ``(n_filters, max_len)``; ``phot_filter_waves_padded`` units Å.
    phot_filter_waves_padded: jnp.ndarray | None = None
    phot_filter_trans_padded: jnp.ndarray | None = None
    # Spectrum pixel effective wavelengths in the rest frame (published
    # when approx=SpectrumPrecomp() is set). Per-pixel effective
    # wavelengths in the galaxy rest frame, computed as observed wavelengths
    # divided by (1 + z). Shape ``(n_spec_pixel,)``, units Å.
    spec_eff_waves: jnp.ndarray | None = None

    # AGN (incl. GRAHSP alternates)
    L_agn_bol: jnp.ndarray | None = None
    # log10(L_agn_bol / (erg/s)) — float32-safe companion to ``L_agn_bol``
    # (~1e46 erg/s overflows float32 max 3.4e38). Consumed by the radio jet
    # term (#1206) via ``apply_log10_scale`` so the AGN-driven radio emission
    # never materializes the out-of-range linear luminosity.
    log_L_agn_bol: jnp.ndarray | None = None
    L_agn_torus: jnp.ndarray | None = None
    L_agn_absorbed: jnp.ndarray | None = None
    # Intrinsic (un-reddened) disc monochromatic L_nu at 2500 A [erg/s/Hz];
    # drives X-ray alpha_ox (Just+2007). Published by the composable AGN (#722).
    L_2500_intrinsic: jnp.ndarray | None = None
    # Intrinsic (un-reddened) disc monochromatic L_nu at 4400 A [erg/s/Hz];
    # drives radio loudness normalization (B-band). Published by composable AGN.
    L_4400_intrinsic: jnp.ndarray | None = None
    # AGN cos(i) [dimensionless]. The X-ray corona tilts its Yang+2022
    # anisotropy to the same sightline as the disc/torus, exactly as
    # X-CIGALE forwards cos i from the AGN module into yang20 (#980).
    agn_cos_inc: jnp.ndarray | None = None
    sed_agn: jnp.ndarray | None = None
    sed_grahsp: jnp.ndarray | None = None
    # AGN — filter-integrated LUT. Rest-frame Lν of
    # the AGN contribution per filter, shape ``(n_filters,)``. Published
    # only when ``approx=WavePrecomp()`` is set and AGN is
    # configured. Consumed by ``predict_via_precomp`` via the multi-
    # component sum of ``*_phot_lnu_precomp`` keys.
    agn_phot_lnu_precomp: jnp.ndarray | None = None

    # Nebular
    sed_nebular: jnp.ndarray | None = None
    sed_shock: jnp.ndarray | None = None
    line_waves: jnp.ndarray | None = None
    line_lums: jnp.ndarray | None = None
    #: float32-safe companion to ``line_lums`` (#1534); ~1e41 erg/s is ``inf``
    #: in pure float32. Same sentinel convention as ``log_L_age``.
    log_line_lums: jnp.ndarray | None = None
    #: ``line_lums`` after the dust screen, published by whichever dust
    #: component ran, with the same law/params/clip it applies to the nebular
    #: continuum (#1867). ``None`` when no dust component ran, and
    #: ``_line_luminosity_helper`` then falls back to the intrinsic
    #: ``line_lums`` — the correct answer in that case.
    #:
    #: A typed field rather than an ``_extras`` entry deliberately: both public
    #: line surfaces (``pred.lines.*`` and ``predict_properties``) resolve
    #: through it, and #1867 was exactly a reddened catalog going somewhere
    #: nothing read. A typed field is greppable and checkable; an untyped extra
    #: would reproduce the failure mode.
    line_lums_attenuated: jnp.ndarray | None = None
    # Stellar Lyman-continuum survival fraction where(λ<912, neb_fesc, 1),
    # published by photoionized backends so two-component dust can honor the
    # fesc absorption on the per-age lnu_age path (#824).
    lyc_transmission: jnp.ndarray | None = None
    # Nebular — photometry LUT (published only when
    # ``approx=WavePrecomp()`` is set on SEDModel and the nebular
    # backend supports filter-level precomputation (Cue / CloudyGrid).
    # For BakedIn nebular this is None — the nebular emission is already
    # baked into the SSP grid and therefore included in stellar_phot_lnu_precomp.
    nebular_phot_lnu_precomp: jnp.ndarray | None = None
    # Shock (MAPPINGS V) per filter — rest-frame Lν, erg/s/Hz, intrinsic (no
    # dust, no cosmology), exactly like ``nebular_phot_lnu_precomp``. A separate
    # additive component from the photoionized backend (#851), so it carries its
    # own key. ``ShockNebular.apply`` publishes it by filter-integrating the
    # full-grid shock SED; the inherited effective-wavelength LUT cannot, because
    # shock emission is line-dominated and its norm="frac" anchor needs the
    # accumulated SED's bolometric (#1375). ``predict_via_precomp`` reddens this
    # with the young-limit screen alongside the nebular bucket.
    shock_phot_lnu_precomp: jnp.ndarray | None = None
    # Dust IR re-emission per filter (rest-frame Lν, erg/s/Hz). Additive and
    # unattenuated — summed by predict_via_precomp like the other families.
    # Published by the two-component dust component under WavePrecomp (#622).
    dust_emission_phot_lnu_precomp: jnp.ndarray | None = None
    # Radio (synchrotron + free-free) and X-ray per filter (rest-frame Lν,
    # erg/s/Hz). Additive, unattenuated power-law emitters summed by
    # predict_via_precomp under WavePrecomp (#624).
    radio_phot_lnu_precomp: jnp.ndarray | None = None
    xray_phot_lnu_precomp: jnp.ndarray | None = None
    # Shock is an additive emitter like radio/xray, and SEDModelComponent
    # publishes ``{name}_phot_lnu_precomp`` generically — but this field was
    # never added, so any model with a shock component raised ComponentIOError
    # ("_extras is non-empty") the moment the photometry LUT ran, i.e. on every
    # inference call. Typed here so the shock LUT rides the same contract.
    shock_phot_lnu_precomp: jnp.ndarray | None = None

    # Spectrum LUT (published only when approx=SpectrumPrecomp()
    # is set). Per-pixel rest-frame Lν contributions from each component
    # at spectrum pixel centers (effective wavelengths).
    # Shape ``(n_spec_pixel,)``, units erg/s/Hz.
    stellar_spec_lnu_precomp: jnp.ndarray | None = None
    nebular_spec_lnu_precomp: jnp.ndarray | None = None
    dust_spec_lnu_precomp: jnp.ndarray | None = None
    # Dust IR re-emission per spectrum pixel (rest-frame Lν, erg/s/Hz);
    # additive/unattenuated, summed by predict_spectrum_via_precomp (#622).
    dust_emission_spec_lnu_precomp: jnp.ndarray | None = None
    # Radio / X-ray per spectrum pixel (rest-frame Lν, erg/s/Hz); additive,
    # unattenuated, summed by predict_spectrum_via_precomp (#624).
    radio_spec_lnu_precomp: jnp.ndarray | None = None
    xray_spec_lnu_precomp: jnp.ndarray | None = None
    agn_spec_lnu_precomp: jnp.ndarray | None = None
    igm_spec_transmission_precomp: jnp.ndarray | None = None
    # Age-resolved per-pixel stellar LUT, shape ``(n_age, n_spec_pixel)``,
    # erg/s/Hz — needed for two-component (Charlot & Fall) dust on the
    # spectrum LUT path (sum over age == stellar_spec_lnu_precomp).
    stellar_spec_lnu_per_age_precomp: jnp.ndarray | None = None
    # Dust transmission at spectrum pixel centers (dimensionless, in [0, 1]).
    # Unlike photometry, no Taylor moment is needed — a pixel is a single
    # wavelength, so A(λ_pix) is exact. ``dust_spec_transmission_precomp`` is
    # the single-component / diffuse transmission; the two-component split
    # publishes ``dust_spec_bc_transmission_precomp`` (birth cloud) and
    # ``dust_spec_diff_transmission_precomp`` (diffuse) applied with the
    # shared ``dust_young_indicator`` age weighting.
    dust_spec_transmission_precomp: jnp.ndarray | None = None
    dust_spec_bc_transmission_precomp: jnp.ndarray | None = None
    dust_spec_diff_transmission_precomp: jnp.ndarray | None = None

    # Radio / X-ray / IGM / shock
    sed_radio: jnp.ndarray | None = None
    sed_xray: jnp.ndarray | None = None
    igm_transmission: jnp.ndarray | None = None
    # Filter-averaged IGM transmission <T>_f, shape (n_filters,), dimensionless.
    # The WavePrecomp twin of ``igm_transmission``: the LUT projector needs one
    # number per band, and <T>_f depends only on (z, filter, convention) — the
    # transmission is averaged alone, never weighted by the SED — so the IGM
    # component tabulates it against z at build time. Consuming the full-grid
    # curve here instead keeps the whole model grid live and defeats the
    # dead-code elimination that IS the WavePrecomp speedup (#932).
    igm_phot_factor: jnp.ndarray | None = None
    # Per-pixel IGM transmission, shape (n_pix,), dimensionless — the
    # SpectrumPrecomp twin of ``igm_phot_factor``. A pixel's rest effective
    # wavelength is wave_obs/(1+z) and the curve is T(wave_rest*(1+z), z), so
    # sampling one at the other collapses to T at the FIXED observed instrument
    # grid: a function of (z, pixel) alone, tabulated at build time.
    igm_spec_factor: jnp.ndarray | None = None
    shock_log_lhalpha: jnp.ndarray | None = None

    # Spatial — 2D surface-brightness profile and the (x, y) kpc grid that
    # underlies it. Published by spatial components (Sersic, Exponential,
    # FlatSlab, …). Reserved B-path keys (``spatial_profile_per_age``,
    # ``spatial_profile_per_wave``) will be added when those components land.
    # See architecture spec §3.3.
    spatial_profile_2d: jnp.ndarray | None = None
    # ``spatial_grid_xy_kpc`` is a tuple of two 2D arrays ``(x_grid, y_grid)``
    # so it is intentionally not a jnp.ndarray field. The bundle still
    # accepts it via ``with_(spatial_grid_xy_kpc=...)``; the type annotation
    # is permissive.
    spatial_grid_xy_kpc: Any = None

    # Free-form spillover dict for keys not yet promoted to fields.
    # Empty in steady state; provides a graceful path when an
    # in-flight migration declares a new key in _CANONICAL_UNITS
    # before adding the matching field here. Always-non-None
    # (default empty dict) so JAX pytree registration sees a stable
    # structure across instances.
    _extras: dict[str, Any] = field(default_factory=dict)

    # ── helpers ────────────────────────────────────────────────

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Tuple of every typed-field name (excludes ``_extras``)."""
        return tuple(f.name for f in fields(cls) if f.name != "_extras")

    def with_(self, **overrides: Any) -> DerivedState:
        """Return a copy with selected fields replaced.

        Unknown keys raise :class:`TypeError` (the dataclass-frozen
        default) with a Levenshtein-2 ``Did you mean: ...`` hint if
        the typo is close to a known field.
        """
        from dataclasses import replace

        # Pre-check unknown keys to produce a friendly hint message.
        # ``_extras`` is the documented escape hatch for keys not yet
        # promoted to typed fields — explicitly allow it.
        known = set(self.field_names()) | {"_extras"}
        unknown = [k for k in overrides if k not in known]
        if unknown:
            offender = unknown[0]
            hint = _did_you_mean(offender, known)
            suffix = f" (Did you mean: {hint!r}?)" if hint else ""
            raise TypeError(
                f"DerivedState has no field {offender!r}.{suffix} "
                f"Use ``with_(_extras={{...}})`` to attach values that "
                f"haven't been promoted to typed fields yet."
            )
        return replace(self, **overrides)

    # ── dict compatibility ────────────────────────────────────

    def __getitem__(self, name: str) -> Any:
        if name in self.field_names():
            value = getattr(self, name)
            if value is None:
                raise KeyError(name)
            return value
        if name in self._extras:
            return self._extras[name]
        raise KeyError(name)

    def get(self, name: str, default: Any = _MISSING) -> Any:
        try:
            return self[name]
        except KeyError:
            if default is _MISSING:
                return None
            return default

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        if name in self.field_names():
            return getattr(self, name) is not None
        return name in self._extras

    def keys(self) -> list[str]:
        """Names of fields with non-None values + any keys in _extras."""
        live = [n for n in self.field_names() if getattr(self, n) is not None]
        live.extend(self._extras.keys())
        return live

    def values(self) -> list[Any]:
        return [
            getattr(self, n) for n in self.field_names() if getattr(self, n) is not None
        ] + list(self._extras.values())

    def items(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        for n in self.field_names():
            v = getattr(self, n)
            if v is not None:
                out.append((n, v))
        out.extend(self._extras.items())
        return out

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    # ── migration helpers ─────────────────────────────────────

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        *,
        allow_extras: bool = False,
    ) -> DerivedState:
        """Build a bundle from a plain dict.

        Strict by default: when ``allow_extras=False``, keys outside
        the typed set raise :class:`TypeError` with a Levenshtein-2
        "Did you mean: ..." hint. Production write paths
        (:meth:`ForwardState.__post_init__`, :meth:`ForwardState.with_`)
        rely on this strictness to catch typos at trace time.

        Pass ``allow_extras=True`` to spill unknown keys into
        ``_extras`` (still readable via mapping syntax). Used by the
        legacy-compat tests; not used by production code.
        """
        known = set(cls.field_names())
        typed = {k: v for k, v in d.items() if k in known}
        extras = {k: v for k, v in d.items() if k not in known}
        if extras and not allow_extras:
            # Pick the first offender so the error message is short
            # and stable. The hint pinpoints likely typos at distance
            # ≤ 2 — same UX as ``with_(unknown=…)``.
            offender = next(iter(extras))
            hint = _did_you_mean(offender, known)
            suffix = f" (Did you mean: {hint!r}?)" if hint else ""
            raise TypeError(
                f"DerivedState.from_dict received {len(extras)} unknown "
                f"key(s): first is {offender!r}.{suffix} The dict-style "
                f"write path is closed as of ADR-0007 Phase 4 — use "
                f"``state.derived.with_(...)`` to set typed fields, or "
                f"pass ``allow_extras=True`` to opt into the legacy "
                f"spillover-to-_extras shim for migration / debugging."
            )
        return cls(**typed, _extras=extras)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict — the inverse of :meth:`from_dict`.

        Non-None typed fields are included; ``_extras`` is merged in.
        Useful for serialization, debugging, and gradual migration.
        """
        out: dict[str, Any] = {
            n: getattr(self, n) for n in self.field_names() if getattr(self, n) is not None
        }
        out.update(self._extras)
        return out


def _did_you_mean(target: str, options) -> str | None:
    """Levenshtein-2 nearest match, or None if no close match."""
    from tengri.utils.strings import closest

    return closest(target, options, max_distance=2)


# ── JAX pytree registration ──────────────────────────────────
#
# Each typed field is a data field. ``_extras`` is also a data field
# (a dict; JAX's default-dict-pytree handler will recurse into it).
# The frozen-dataclass + register_dataclass combo is identical to how
# ForwardState is registered in tengri.protocols.component.

from jax import tree_util as _tree_util

_DATA_FIELDS = tuple(f.name for f in fields(DerivedState))

_tree_util.register_dataclass(
    DerivedState,
    data_fields=_DATA_FIELDS,
    meta_fields=(),
)

del _tree_util
