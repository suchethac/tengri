# SPDX-License-Identifier: BSD-3-Clause
"""The policy ledger behind :meth:`~tengri.forward.sed_model.SEDModel.compile_signature`.

``compile_signature()`` used to be a hand-written 68-field tuple: every
JIT-affecting attribute had to be remembered and added by name. Four bugs
shipped from exactly that omission — #1122 (``n_subbands`` missing from the
z-table key), #1462 (seven AGN and X-ray fields missing from this
signature), #2145 (the cosmology latent in the z-table key), #2237 (the live
dust shape-parameter set missing from this signature) — each a field nobody
added to a list, each silently serving one model's cached computation to a
structurally different one (#1973, the SSP metallicity grid VALUES, was the
same class one key earlier).

This module replaces the hand list with a policy ledger over
``vars(model)``, consumed by ``tengri._cache_keys.derive_key``. Every
instance attribute :class:`~tengri.forward.sed_model.SEDModel` can carry is
classified here under one of three modes:

- ``content(reason)``: hash the attribute's full value. The default for
  anything that can change what the compiled kernel computes or returns.
- ``shape(reason)``: hash only shape + dtype (unused on this ledger today;
  every SEDModel array-valued attribute is small enough, and structurally
  significant enough, to key by content instead).
- ``exclude(reason)``: omit the attribute. Reserved for memo caches
  recomputed from already-keyed structure, the runtime state container, and
  the memo (and version) of the signature itself.

**An attribute this ledger does not classify is a bug, not a shrug.**
``tengri._cache_keys.classify`` defaults an unclassified attribute to
``"content"`` (fail-safe: an unnecessary cache miss over a silent wrong
answer), but ``tengri._cache_keys.assert_policy_complete`` — run in
``tests/contract/test_compile_signature_invariants.py`` against every
representative build — fails the moment a NEW attribute appears that this
module has not named. That is the whole point: the #1122/#1462/#2145/#2237
class of bug becomes a failing test instead of a silent miscompilation, on
the day the attribute is added rather than however many releases later
someone notices two models sharing a kernel they should not.

To add a new attribute:

1. Decide its mode (see the three bullets above).
2. Add a row to ``SIGNATURE_POLICY`` with a one-line reason.
3. Nothing else. ``compile_signature()`` itself never changes.

Nested objects that already carry their own ``cache_key()`` ledger
(``Observation``, ``Parameters`` — read as ``model.spec`` — and
``SSPData``) are listed here as ordinary ``content`` rows: ``baked()``
detects the ``cache_key()`` method and delegates to it, so this ledger
does not re-list their internals.

Bump ``SIGNATURE_VERSION`` whenever the derivation itself changes, OR
whenever an existing row's MODE changes (e.g. a row moves from
``exclude`` to ``content``) — either can change what two otherwise-equal
models hash to, and a version bump forces every cached engine to
recompile rather than silently keep serving a stale one.
"""

from tengri._cache_keys import KeyPolicy, content, exclude

#: Bump whenever the derivation changes, or an existing row's MODE changes.
#: See the module docstring for the full rule.
SIGNATURE_VERSION = 1

SIGNATURE_POLICY: KeyPolicy = {
    # ── Nested objects with their own cache_key() ledger (#2163 E.2) ──
    "observation": content("Observation.cache_key() covers filters, spectroscopy, noise"),
    "spec": content("Parameters.cache_key() covers every structural spec setting"),
    "ssp_data": content("SSPData.cache_key() covers the SSP grid shape, values, and metadata"),
    # ── Component chain (#2163 E.3 step 1) ──────────────────────────
    "_component_configs": content(
        "the (name, config) tuple in chain order; each config is a frozen "
        "dataclass keyed field-by-field via SEDComponentConfig.cache_key(). "
        "Built eagerly and unconditionally in __init__ (see "
        "SEDModel._build_chain_configs), so a fresh model's signature never "
        "depends on whether approx=WavePrecomp()/SpectrumPrecomp() triggered "
        "an eager _build_component_chain() call. This is what makes a config "
        "field nobody added to a hand-written list (#1462's AGN blocks, "
        "#2237's dust shape-parameter set) impossible to miss: it is read "
        "off the SAME configs the chain runs, not re-derived in parallel."
    ),
    # ── SSP / age grid (small arrays; JAX arrays memoize by identity) ──
    "_rest_wavelength": content("rest wavelength grid the compiled kernels close over"),
    "age_yr": content("SSP age grid in years"),
    "log_age_grid": content("SSP log-age grid"),
    "d_log_age": content("log-age grid spacing"),
    "ssp_ages_yr": content("SSP age nodes in years"),
    "ssp_log_ages_yr": content("SSP log10(age/yr) nodes"),
    "_csp_age_dt": exclude(
        "deterministic function of ssp_ages_yr (content-keyed above) AND the "
        "deprecated, no-effect csp_integration knob (#1500, excluded below): "
        "csp_age_dt(ssp_ages_yr, csp_integration) reads the knob directly, so "
        "keying this row transitively re-admits csp_integration's dead variation "
        "and reopens exactly the bug _csp_integration's own exclusion closed --"
        "measured: four of the five csp_integration values produced four distinct "
        "signatures via this row alone before it was excluded here"
    ),
    "_csp_matrix": exclude(
        "same hazard as _csp_age_dt immediately above: None for four of the five "
        "deprecated csp_integration values and, on the fifth ('log_interp'), a "
        "deterministic function of the already-keyed ssp_ages_yr"
    ),
    "_dl_cm_fixed": content("fixed luminosity distance, derived from _z_fixed"),
    # ── SSP-derived scalars ──────────────────────────────────────────
    "_n_grid": content("stochastic-SFH grid resolution changes the graph shape"),
    "_lgmet_scatter": content(
        "metallicity-scatter width changes the age-weight kernel output (#2163: "
        "one of the four attributes the hand-written list never keyed)"
    ),
    "_met_interp": content("metallicity interpolation mode changes the SSP lookup"),
    "_met_mode": content("metallicity model selection determines parameters and graph shape"),
    "_z_interp": content("redshift interpolation mode changes the ztable lookup"),
    # ── SFH ────────────────────────────────────────────────────────────
    "_sfh_fn": content("SFH function, kept by qualified name"),
    "_sfh_internal_names": content("internal SFH parameter names determine the param map"),
    "_sfh_public_names": content("public SFH parameter names determine the param map"),
    "_sfh_settings": content("resolved SFH settings dict (mean type, field composition, ...)"),
    "_uses_stochastic_sfh": content("stochastic vs parametric SFH changes the graph shape"),
    "_field_centering": content("GP-field centering changes the xi -> SFH map (#1355)"),
    "_gp_kernel": content(
        "GP-field kernel choice (drw vs ...) changes the field draw (#2163: one of "
        "the four attributes the hand-written list never keyed)"
    ),
    # ── Dust ───────────────────────────────────────────────────────────
    "_dust_model": content("dust model selection determines the graph shape"),
    "_dust_scheme": content("dust scheme (single/two-component) determines the graph shape"),
    "_dust_emission_model": content("dust emission model selection determines the graph shape"),
    "_dust_law_bc": content("birth-cloud dust law name"),
    "_dust_law_diff": content("diffuse dust law name"),
    "_dust_law_neb": content("nebular dust law name (None -> inherits bc)"),
    "_dust_law_bc_fn": content("birth-cloud dust law function, kept by qualified name"),
    "_dust_law_diff_fn": content("diffuse dust law function, kept by qualified name"),
    "_dust_nebular_screen": content(
        "which screen the nebular continuum and line catalog pass through (#2234); also "
        "carried by the retained DustSEDComponentConfig"
    ),
    "_dust_shock_screen": content("which screen the shock SED passes through (#2234)"),
    "_dust_agn_screen": content("galaxy screen on AGN light; none until the AGN change lands"),
    "_dust_law_overrides": content("per-component dust law parameter overrides"),
    "_dust_lyman_cutoff_aa": content("Lyman-limit clip wavelength changes the FUV curve"),
    "_dust_lyc_absorb_all": content("young-only vs absorb-all stellar LyC changes the chain"),
    "_dust_eb_include_lyc": content("LyC-in-energy-balance flag rescales L_IR"),
    "_astrodust_spinning_dust": content("astrodust AME enable flag changes the emitted SED"),
    "_astrodust_f_cnm": content("astrodust cold-neutral-medium filling fraction"),
    "_wg00_dust_curve": content("WG00 dust curve selector (dust_type=3 only)"),
    "_wg00_geometry": content("WG00 geometry selector (dust_type=3 only)"),
    "_wg00_structure": content("WG00 structure selector (dust_type=3 only)"),
    # ── Nebular ────────────────────────────────────────────────────────
    "_nebular_backend": content("backend instance; delegates to its own cache_key() ledger"),
    "_nebular_model": content("nebular mode string, used for wavelength-extension routing"),
    "_nebular_grid_table": content(
        "fast-nebular per-Q_H grid attached by enable_fast_nebular(); a different "
        "compiled graph AND the kernel closes over the grid arrays"
    ),
    # ── IGM ────────────────────────────────────────────────────────────
    "_uses_igm": content("whether IGM absorption is applied"),
    "_igm_model": content("IGM model selection determines the transmission curve"),
    "_igm_fn": content("IGM transmission function, kept by qualified name"),
    "_igm_patchy": content(
        "patchy vs uniform IGM changes the transmission curve (#2163: one of the "
        "four attributes the hand-written list never keyed)"
    ),
    "_uses_dla": content("whether DLA absorption is applied"),
    # ── AGN ────────────────────────────────────────────────────────────
    "_agn_model": content(
        "AGN model selection (carries no discriminating power alone, "
        "but is cheap and keeps the row list honest)"
    ),
    "_agn_luminosity_mode": content("AGN luminosity mode changes the parameterization"),
    "_agn_disc_block": content("AGN disc block selector (#1462)"),
    "_agn_torus_block": content("AGN torus block selector (#1462)"),
    "_agn_nlr_block": content("AGN NLR block selector (#1462)"),
    "_agn_blr_block": content("AGN BLR block selector (#1462)"),
    "_agn_feii_block": content("AGN FeII block selector (#1462)"),
    "_agn_attenuation_block": content("AGN attenuation block selector (#1462)"),
    "_agn_norm": content("AGN cross-block normalization policy changes the emitted SED"),
    "_agn_config": content("frozen AGNConfig, keyed field by field"),
    "_agn_ir_frac_dist": content("AGN IR-fraction Distribution, keyed via jit_cache_key"),
    "_agn_lbol_dist": content("AGN bolometric-luminosity Distribution, keyed via jit_cache_key"),
    "_needs_agn_lbol_flat_check": content(
        "owner-flagged: a build-time flag that selects a validation branch"
    ),
    "_agn_lbol_is_user_fixed": content(
        "owner-flagged: the sibling of the row above, same shape. It records "
        "whether agn_log_lbol was spelled out by the user rather than left "
        "free (R55), and selects the wording of the flat-direction refusal "
        "_check_agn_lbol_flat_direction raises. Build-time only, reaching no "
        "kernel -- content is the fail-safe classification, as for the flag "
        "that gates the same check"
    ),
    # ── Radio / X-ray / shock ──────────────────────────────────────────
    "_uses_radio": content("whether radio emission is attached"),
    # No ``_radio_include_freefree`` row, deliberately: a 2026-09 citation audit
    # deleted that attribute. It was read from a ``spec.radio_include_freefree``
    # the public grammar never set, fed only the old hand-written signature
    # tuple, and never reached ``RadioSEDComponentConfig`` -- the object
    # ``radio_freefree`` actually gates on, which ``component_factory`` never
    # accepted it for. 100% dead, so its removal changes no model's compiled
    # HLO and is intentionally not replaced with anything here.
    "_radio_sfr_mode": content("radio SFR-tracer model selection"),
    "_radio_agn_model": content("radio AGN model selection"),
    "_uses_xray": content("whether X-ray emission is attached"),
    "_xray_model": content("WHICH X-ray model, not merely whether one is attached (#1462)"),
    "_uses_shock": content("whether shock nebular emission is attached"),
    "_shock_norm": content("shock normalization mode changes the emitted SED"),
    "_shock_abundance": content("shock abundance set changes the emitted SED"),
    "_shock_component": content("shock component selector changes the emitted SED"),
    # ── Instrument / spectroscopy-adjacent scalars ─────────────────────
    "_sigma_lib_kms": content("SSP library velocity dispersion affects LSF deconvolution"),
    "_lsf_resolution": content("spectral resolution affects LSF convolution"),
    "_lsf_n_bins": content(
        "LSF approximation bin count affects accuracy (#2163: one of the four "
        "attributes the hand-written list never keyed)"
    ),
    "_has_sigma_v": content("whether a velocity-dispersion free parameter is registered"),
    # ── Redshift / cosmology ────────────────────────────────────────────
    "_z_fixed": content("fixed-redshift VALUE; compiled kernels close over derived tables"),
    "_catalog_z_range": content("catalog-fit redshift range; the ztable shape can differ"),
    # ── Param map (build-time, frozen) ─────────────────────────────────
    "_param_map": content("owner-flagged: the frozen name map decides which runtime inputs exist"),
    # ── Compile-time knobs ──────────────────────────────────────────────
    "_compile_mode": content("compile mode changes the JIT strategy"),
    "_wave_chunk_size": content("wavelength chunking changes the graph shape"),
    "_alpha_fe_evolving": content("alpha-Fe evolution model selection changes parameters"),
    "_fast_line_measurement": content("FeaturePrecomp routing flag for the line channel (#1748)"),
    # ── Approximation policy ────────────────────────────────────────────
    "_approx": content("ApproxPolicy is a Mapping; bakes generically via the Mapping branch"),
    "_approx_config": content("resolved WavePrecomp config or None"),
    "_approx_config_wave": content("resolved WavePrecomp config or None (wave channel)"),
    "_approx_config_spec": content("resolved SpectrumPrecomp config or None"),
    "_approx_config_feature": content("resolved FeaturePrecomp config or None"),
    # ── Precision ────────────────────────────────────────────────────────
    # forward_dtype is excluded below (#1433); precision itself enters via the
    # x64 tail passed to derive_key(), not through any instance attribute.
    # ── The memo, and true no-ops ──────────────────────────────────────
    "_signature_memo": exclude("the memo itself: compile_signature() reads/writes this"),
    "_Observables": exclude("class object assembling accessors; no structure of its own"),
    "_cached_component_chain": exclude(
        "memo of the built chain; the chain's configs are keyed through _component_configs instead"
    ),
    "_cached_full_state_chain": exclude(
        "memo of the built chain; the chain's configs are keyed through _component_configs instead"
    ),
    "_dust_band_response_cache": exclude("memo cache computed from keyed structure"),
    "_energy_balance_lut_cache": exclude("memo cache computed from keyed structure"),
    "_radio_term_response_cache": exclude("memo cache computed from keyed structure"),
    "_xray_term_response_cache": exclude("memo cache computed from keyed structure"),
    "_index_window_lut_cache": exclude("memo cache computed from keyed structure"),
    "_line_window_lut_cache": exclude("memo cache computed from keyed structure"),
    "_property_catalog": exclude("memo cache computed from keyed structure"),
    "_state": exclude("runtime state container rebuilt from keyed structure"),
    "_csp_integration": exclude(
        "deliberately NOT part of the signature. Every value produces an "
        "identical program (#1500); including it split the compile cache five "
        "ways and recompiled the whole model to compute the same numbers. "
        "Measured: 5 distinct signatures, 0 differing outputs."
    ),
    "_forward_dtype": exclude(
        "deliberately NOT part of this key (#1433). It is retired and casts "
        "nothing, so two models differing only in it compute bit-identical "
        "results; keying on it bought a second compile of an identical kernel "
        "and nothing else. Anyone who wires it must put it back here in the "
        "same change, or the two precisions will share a kernel."
    ),
}
