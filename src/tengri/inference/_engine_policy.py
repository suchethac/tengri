# SPDX-License-Identifier: BSD-3-Clause
"""The two policy ledgers behind ``Fitter._engine_cache_key()`` and ``_data_fingerprint()``.

``Fitter._engine_cache_key()`` used to be a hand-written ~20-entry tuple built by
reading through ``__init__`` and remembering which fields the compiled loss
closure bakes in. ``_data_fingerprint()`` was a second, independent hand list
(``_FINGERPRINTED_ATTRS`` in ``_sample_utils.py``) of the same Fitter, keying the
complementary MAP/adaptation caches by data *content* rather than program
*structure*. Nobody had ever checked the two lists against each other, or
against the live attribute set: this is the same #2163 bug class the model's
own ``compile_signature()`` ledger (``forward._signature_policy``) fixed for
``SEDModel`` -- a field nobody added to a hand list, silently serving one
Fitter's compiled loss (or cached MAP/adaptation) to a structurally, or
observationally, different one.

This module replaces both hand lists with two policy ledgers over
``vars(fitter)``, consumed by ``tengri._cache_keys.derive_key``:

- ``ENGINE_POLICY`` -- everything that changes what the compiled loss
  closure computes: which components are active, which parameters are free,
  what values are baked as constants, which observation channels are wired
  up on a *per-Fitter* basis. Two Fitters agreeing on every ``ENGINE_POLICY``
  row may safely share one compiled JIT engine.
- ``FINGERPRINT_POLICY`` -- the data this Fitter was constructed against:
  the arrays whose *values* (not merely shape) decide where a MAP search or
  MCMC warmup lands. Two Fitters agreeing on every ``FINGERPRINT_POLICY`` row
  are fitting the same target and may safely share a cached MAP or a tuned
  step size / mass matrix.

The two ledgers are deliberately complementary, not independent: every
Fitter attribute is classified ``content``/``shape`` in *at most one* of
them, the engine's ``shape`` rows (``data``, ``noise``, ``data_mask``,
``presence``) are exactly the fingerprint's ``content`` rows, and
``_runtime_redshift`` -- excluded from the engine key because it rides
``data_args`` as a traced input (#1316) -- is the fingerprint's fifth
content row for the same reason. ``tests/contract/test_inference_cache_keys.py``
asserts this partition holds over the representative Fitters below, so a
new Fitter attribute that is unclassified in either ledger, or wrongly
double-classified, fails a test instead of shipping a silent bug.

**An attribute either ledger does not classify is a bug, not a shrug.**
``tengri._cache_keys.classify`` defaults an unclassified attribute to
``"content"`` (fail-safe), but the policy-complete test fails the moment a
NEW Fitter attribute appears that this module has not named.

Two things ``ENGINE_POLICY`` deliberately does NOT re-derive, and why:

1. **The model's own structure.** ``model`` (and its pre-approx twin
   ``_pre_approx_model``) are excluded here: ``SEDModel`` has no
   ``cache_key()`` for ``tengri._cache_keys.baked`` to delegate to, and
   every cache keyed by ``_engine_cache_key()`` is ALREADY namespaced per
   ``model`` object (``_model_cache_owner.get_or_compile_model(model)`` in
   ``jit_engine.py`` and ``backends/mcmc/_shared.py``), so the model's
   structure -- including ``Observation.cache_key()``'s own coverage of
   filters, spectroscopy, noise, line fluxes/ratios/spectral indices -- is
   already what distinguishes one namespace from another. ``Fitter.compile_signature()``
   additionally pairs ``model.compile_signature()`` alongside this key for
   any caller that mixes engines across models. Re-including ``model``
   content here would need a bespoke, non-``cache_key()`` hook and would add
   no coverage a ``baked()`` fallback (a custom ``__repr__``) could not get
   wrong.
2. **Redundant derived booleans.** ``has_noise_model(spec)`` (a pure
   function of which ``noise_*`` names are free, covered by ``spec``'s own
   ``cache_key()`` tail, and whether ``noise_frac_cal`` is fixed nonzero,
   covered by the ``fixed_values`` tail row below) and the presence checks
   on ``model.observation.line_ratios`` / ``.spectral_indices`` (covered by
   ``Observation.cache_key()``, reachable through case 1 above) are provably
   determined by rows already in this ledger or the model's own signature.
   Only ``_line_flux_override`` -- the one place a Fitter can hand a
   *different* per-galaxy schema than the model's own ``observation.line_fluxes``
   (#1599) -- needs its own row, and it gets one below (delegating to
   ``LineFluxData.cache_key()``, which is a strictly more complete check
   than the wavelengths-plus-limit-mask-presence tuple the old hand key
   computed by itself).

Two Fitter attributes -- ``_fixed_values`` and ``_params_override`` -- are
ALSO excluded from generic per-attribute classification here, despite
carrying engine-relevant content: both dicts can carry a per-galaxy,
runtime-routed redshift override value (merged in unconditionally by
``__init__``, see ``Fitter._fixed_value_key`` and ``_params_override_key``),
and a raw ``content`` row would bake that per-galaxy value into the key,
defeating exactly the #1316 sharing the routed-redshift exclusion exists to
protect. Both are instead carried by dedicated, redshift-filtered TAIL rows
(``"fixed_values"``, ``"params_override"``) built from
``Fitter._fixed_value_key()`` / ``Fitter._params_override_key()`` -- the same
methods the old hand key called inline, now surfaced through the tail
instead of raw positional tuple slots. The ``"priors"`` tail row
(``Fitter._free_prior_key()``, today's ``#1972`` free-parameter-prior-identity
row) and ``"mirrors"`` (``Fitter._mirror_key()``, ``#1972`` instance 3, the
target -> source mirror map) round out the tail; ``"data_arg_names"`` carries
``_data_args``'s KEY SET (structure -- which channels are wired up) while
``_data_args`` itself is excluded (its VALUES are the fingerprint's job).

To add a new Fitter attribute:

1. Decide its mode in ``ENGINE_POLICY`` (does the compiled program change if
   this value changes? -- ``content``; is it a per-galaxy array? -- ``shape``;
   is it a memo, or derived from something already keyed? -- ``exclude``).
2. Decide its mode in ``FINGERPRINT_POLICY``: is it one of the five data
   rows (``data``, ``noise``, ``data_mask``, ``presence``,
   ``_runtime_redshift``)? -- ``content``. Otherwise -- ``exclude``, with the
   reason "structure, not data: keyed by ENGINE_POLICY" (or "memo/threading"
   for the background-compile machinery).
3. Run ``tests/contract/test_inference_cache_keys.py``; it names anything
   left unclassified in either ledger, or double-classified.

Bump ``ENGINE_VERSION`` / ``FINGERPRINT_VERSION`` whenever the derivation
itself changes, OR an existing row's MODE changes -- either can change what
two otherwise-equal Fitters hash to.
"""

from tengri._cache_keys import KeyPolicy, content, exclude, shape

#: Bump whenever the engine-key derivation changes, or an existing row's
#: MODE changes. See the module docstring for the full rule.
ENGINE_VERSION = 2  # 2026-09-12: _profile_mass joined the engine key

#: Bump whenever the fingerprint derivation changes, or an existing row's
#: MODE changes.
FINGERPRINT_VERSION = 1

# ---------------------------------------------------------------------------
# ENGINE_POLICY -- what the compiled loss closure bakes in.
# ---------------------------------------------------------------------------

ENGINE_POLICY: KeyPolicy = {
    # ── Background-compile / memo machinery (never in the key) ────────
    "cache": exclude("CompileCache memo, not part of the loss's structure"),
    "_compilation_event": exclude("background-compile machinery"),
    "_compilation_lock": exclude("background-compile machinery"),
    "_compilation_thread": exclude("background-compile machinery"),
    "_compilation_error": exclude("background-compile machinery"),
    "_jit_sampler": exclude("memo: the already-built engine, not an input to building one"),
    "_lut_bias_checked": exclude(
        "a one-shot advisory flag (#1671); reports on the LUT bias, does not change it"
    ),
    # ── Lazy post-construction memo caches (absent until first exercised;
    # a representative Fitter that never ran a MAP/MCMC/native-VI batch path
    # never materializes these -- see _memo_batch_kernel and vi/native.py) ──
    "_batch_map_kernel_cache": exclude(
        "memo: compiled jax.jit(jax.vmap(...)) batch-MAP kernel, keyed internally "
        "by its own tuple (see _memo_batch_map_kernel); not an engine-key input"
    ),
    "_batch_mcmc_kernel_cache": exclude(
        "memo: compiled jax.jit(jax.vmap(...)) batch-MCMC kernel, keyed internally; "
        "not an engine-key input"
    ),
    "_batch_adapt_kernel_cache": exclude(
        "memo: compiled batched-adaptation kernel, keyed internally; not an engine-key input"
    ),
    "_blackjax_draw_kernel_cache": exclude(
        "memo: compiled posterior-draw kernel, keyed internally; not an engine-key input"
    ),
    "_map_multistart_kernel_cache": exclude(
        "memo: compiled vmapped MAP-multistart kernel (map_dispatch.py), keyed "
        "internally by (compile_signature, optimizer, n_steps, learning_rate, "
        "n_restarts) -- already covered by compile_signature() and the engine key; "
        "not itself an engine-key input"
    ),
    "_map_multistart_qn_kernel_cache": exclude(
        "memo: compiled vmapped JAX-BFGS MAP-multistart kernel (map_dispatch.py), "
        "keyed internally by (compile_signature, optimizer, n_steps, tol, "
        "n_restarts); not itself an engine-key input -- and it must not be, "
        "or memoizing the kernel changes the signature it is keyed on"
    ),
    "_native_vi_nonlinear_engine": exclude(
        "memo: the already-built native-VI-nonlinear engine (vi/native.py), not an "
        "input to building one -- same category as _jit_sampler"
    ),
    # ── Analysis-layer / scheduling knobs: never reach the compiled program ──
    "_memory_mode": exclude(
        "'fast' vs 'low' only changes Python-side posterior-chunking strategy over an "
        "ALREADY-COMPILED engine's draw_samples call, not the compiled HLO graph itself "
        "-- test_memory_mode_does_not_affect_signature pins this; a content row here "
        "would recompile on every run() that flips memory_mode"
    ),
    "_posterior_chunk_size": exclude(
        "a chunk COUNT over an already-compiled engine's draw_samples call (see "
        "_convert_samples_chunked): changes how many times the SAME compiled function "
        "is invoked, never what gets compiled"
    ),
    "_target_modes": exclude(
        "which inference methods a BACKGROUND thread proactively pre-compiles; the "
        "engine each method builds is the SAME whether pre-warmed or built lazily on "
        "first run(), so this is a scheduling knob, not a structural one"
    ),
    # ── The model itself: structure lives elsewhere (see module docstring) ──
    "model": exclude(
        "SEDModel has no cache_key() for baked() to delegate to; every cache keyed "
        "by _engine_cache_key() is already namespaced per model object "
        "(_model_cache_owner.get_or_compile_model(model)), and Fitter.compile_signature() "
        "additionally pairs model.compile_signature() alongside this key -- re-including it "
        "here would need a bespoke hook and add no coverage"
    ),
    "_pre_approx_model": exclude(
        "the pre-approx twin, kept only for the #1671 bias advisory in run(); its "
        "structure is the SAME model's own signature, excluded above for the same reason"
    ),
    # ── Data arguments: values are the fingerprint's job, key SET is structure ──
    "_data_args": exclude(
        "the runtime data dict handed to the compiled loss as a traced argument: its "
        "VALUES are exactly what _data_fingerprint keys, and re-baking them here would "
        "recompile per galaxy. Its KEY SET (which channels are wired up) IS structure and "
        "rides the tail as ('data_arg_names', tuple(sorted(self._data_args)))"
    ),
    # ── Runtime redshift: keyed by the fingerprint, not the engine (#1316) ──
    "_runtime_redshift": exclude(
        "a per-galaxy redshift override threaded through data_args as a traced input "
        "under catalog_z_range (#1316), so distinct redshifts legitimately share one "
        "compiled program; keyed by _data_fingerprint instead. Used here only to decide "
        "the redshift exclusion inside the 'fixed_values' / 'params_override' tail rows"
    ),
    # ── Baked scalars that need the routed-redshift exclusion: tail-only ──
    "_fixed_values": exclude(
        "the override-merged dict (see __init__): _params_override is merged in "
        "UNCONDITIONALLY, including a routed redshift, so a raw content row here would "
        "leak the per-galaxy override value into the key and defeat the #1316 sharing "
        "the routed-redshift exclusion exists to protect (see Fitter._fixed_value_key's "
        "own docstring). Carried instead by the redshift-filtered 'fixed_values' tail row, "
        "sourced from spec.get_fixed_values(), NOT this dict"
    ),
    "_params_override": exclude(
        "same hazard as _fixed_values, one layer over: a raw content row would bake the "
        "per-fit override's redshift value even when it is routed through data_args. "
        "Carried instead by the redshift-filtered 'params_override' tail row (today's rule, "
        "preserved: Fitter._params_override_key())"
    ),
    # ── Per-galaxy arrays: shape fixes the program, values do not ──────
    "data": shape("per-galaxy data; program shape depends on length, not values"),
    "noise": shape("per-galaxy data; program shape depends on length, not values"),
    "data_mask": shape(
        "per-galaxy censoring flags; PRESENCE of a mask selects the censored "
        "likelihood adapter (structure), the flag VALUES ride data_args"
    ),
    "presence": shape("per-galaxy detection flags; batched-catalog data, not structure"),
    # ── Structural settings: content ────────────────────────────────────
    "data_type": content("photometry/spectroscopy/joint selects the loss dispatch"),
    "spec": content(
        "Parameters.cache_key() covers every structural spec setting AND which "
        "parameter names are free/fixed; can differ from model.spec because the "
        "eline-amplitude-fitted merge (_init_eline_arrays) reassigns fitter.spec"
    ),
    "use_components": content("orchestrator-vs-legacy predict path is a different graph"),
    "_free_names": content("which parameters are latent determines the unravel structure"),
    "_bounds": content(
        "per-free-parameter (lo, hi) bounds; redundant with the 'priors' tail row's "
        "jit_cache_key() but cheap and directly observable -- kept for defense in depth"
    ),
    "_has_spectroscopy": content("spectroscopy-vs-photometry data_type derivative gate"),
    "_auto_protocol_likelihood": content(
        "gates whether a Likelihood is auto-built at construction; the resulting "
        "_user_likelihood is already keyed below, this is cheap defense in depth"
    ),
    "_user_likelihood": content(
        "a custom Likelihood replaces the entire chi-squared dispatch (#2163: the old "
        "hand key never included this at all, so two Fitters with different custom "
        "likelihoods silently shared one loss). Every shipped adapter "
        "(GaussianLikelihood/PhotometryLikelihood/SpectroscopyLikelihood/CompositeLikelihood/"
        "StudentTLikelihood/CensoredLikelihood/MultivariateGaussianLikelihood/the "
        "marginalized adapters) is a frozen dataclass, baked field-by-field automatically. "
        "A user-supplied Likelihood that is neither a dataclass nor defines cache_key() nor "
        "a custom __repr__ raises TypeError here -- the documented, intentional baked() "
        "contract ('no address-bearing keys, ever'), not a new failure mode"
    ),
    # ── Calibration ──────────────────────────────────────────────────────
    # ── Mass profiling: the resolved flag bakes the marginal likelihood ────
    "_profile_mass": content(
        "profile_mass engaged: the loss is the mass-marginalized likelihood over "
        "D-1 parameters (mass_profile.py); the free-name tuple also moves, but the "
        "flag is the honest input"
    ),
    "_profile_mass_resolved": exclude("mirror of _profile_mass kept for diagnostics"),
    "_profile_mass_reason": exclude("diagnostic text explaining the resolved flag"),
    "_profile_mass_requested": exclude(
        "the constructor argument (True/False/'auto'); only the resolved flag reaches the loss"
    ),
    "_profile_mass_name": exclude("derived from the spec; _free_names and the model key cover it"),
    "_profile_mass_prior": exclude("the mass prior object, already part of the model's own key"),
    "_profile_mass_bounds": exclude("derived from _profile_mass_prior"),
    "_profile_mass_original_spec": exclude(
        "the pre-profiling spec kept to undo the rewrite; structure the model key covers"
    ),
    "_calibration_marginalize": content("enables the calibration-marginalized likelihood"),
    "_cal_n_poly": content("calibration polynomial order changes the design matrix shape"),
    "_cal_prior_sigma": content("calibration prior width is baked into the marginal likelihood"),
    # ── Emission lines: every _eline_* attribute bakes the loss ─────────
    "_eline_marginalize": content("enables the eline-marginalized likelihood"),
    "_eline_fitted": content("enables explicit eline-amplitude latent parameters"),
    "_eline_prior_type": content("eline amplitude prior family changes the likelihood"),
    "_eline_prior_sigma": content("eline marginalization prior width is baked in"),
    "_eline_prior_width_dex": content("eline amplitude prior width is baked in"),
    "_eline_wavelengths": content("eline wavelength set determines the likelihood function"),
    "_eline_independent_wavelengths": content(
        "independent (non-doublet-secondary) eline wavelengths determine the design"
    ),
    "_eline_names": content("eline name set determines which lines are fit"),
    "_eline_constraint_matrix": content(
        "doublet-fixing constraint matrix changes the design matrix shape"
    ),
    "_eline_amplitude_names": content("fitted eline-amplitude latent parameter names"),
    "_eline_amp_priors": content(
        "fitted eline-amplitude priors merged into fitter.spec; present only when "
        "eline_fitted or eline_marginalize triggered _init_eline_arrays. Redundant with "
        "the free names already appearing in spec/priors once merged, kept for defense "
        "in depth since it is what __init__ actually re-applies after the approx-policy "
        "spec reassignment"
    ),
    # ── Per-galaxy line-flux schema override (#1599) ────────────────────
    "_line_flux_override": content(
        "when set, a per-galaxy LineFluxData whose SCHEMA (names/wavelengths/"
        "is_upper_limit/is_lower_limit) can differ from model.observation.line_fluxes; "
        "delegates to LineFluxData.cache_key(), which is strictly more complete than the "
        "old hand key's (rounded wavelengths, any-limit-mask-present) check -- it reaches "
        "per-line upper/lower-limit flags individually. None when unset, matching every "
        "Fitter sharing the model's own (already-covered) line-flux schema"
    ),
}


# ---------------------------------------------------------------------------
# FINGERPRINT_POLICY -- the data this Fitter was constructed against.
# ---------------------------------------------------------------------------
# The five content rows below are exactly ``ENGINE_POLICY``'s four ``shape``
# rows plus ``_runtime_redshift``; every other Fitter attribute is excluded
# here because it is structure, already covered by ENGINE_POLICY.

_STRUCTURE_REASON = "structure, not data: keyed by ENGINE_POLICY"
_MEMO_REASON = "memo/threading, not data"

FINGERPRINT_POLICY: KeyPolicy = {
    # ── The five data rows ──────────────────────────────────────────────
    "data": content("observed values; decide where a MAP/warmup lands"),
    "noise": content("observed uncertainties; decide the posterior geometry"),
    "data_mask": content("censoring flag VALUES; decide the posterior geometry"),
    "presence": content("per-galaxy detection flag VALUES; decide the posterior geometry"),
    "_runtime_redshift": content(
        "a per-galaxy redshift override rides data_args as a traced input (#1316), so it "
        "is target-identifying data here even though it is excluded from ENGINE_POLICY"
    ),
    # ── Everything else: structure, kept by ENGINE_POLICY ───────────────
    "cache": exclude(_MEMO_REASON),
    "_compilation_event": exclude(_MEMO_REASON),
    "_compilation_lock": exclude(_MEMO_REASON),
    "_compilation_thread": exclude(_MEMO_REASON),
    "_compilation_error": exclude(_MEMO_REASON),
    "_jit_sampler": exclude(_MEMO_REASON),
    "_batch_map_kernel_cache": exclude(_MEMO_REASON),
    "_batch_mcmc_kernel_cache": exclude(_MEMO_REASON),
    "_batch_adapt_kernel_cache": exclude(_MEMO_REASON),
    "_blackjax_draw_kernel_cache": exclude(_MEMO_REASON),
    "_map_multistart_kernel_cache": exclude(_MEMO_REASON),
    "_map_multistart_qn_kernel_cache": exclude(_MEMO_REASON),
    "_native_vi_nonlinear_engine": exclude(_MEMO_REASON),
    "_lut_bias_checked": exclude(_MEMO_REASON),
    "model": exclude("structure (kept by the model's own compile_signature(), not either ledger)"),
    "_pre_approx_model": exclude(_STRUCTURE_REASON),
    "_data_args": exclude(_STRUCTURE_REASON),
    "_fixed_values": exclude(_STRUCTURE_REASON),
    "_params_override": exclude(_STRUCTURE_REASON),
    "data_type": exclude(_STRUCTURE_REASON),
    "spec": exclude(_STRUCTURE_REASON),
    "use_components": exclude(_STRUCTURE_REASON),
    "_free_names": exclude(_STRUCTURE_REASON),
    "_bounds": exclude(_STRUCTURE_REASON),
    "_has_spectroscopy": exclude(_STRUCTURE_REASON),
    "_memory_mode": exclude(
        "an analysis-layer scheduling knob (chunking strategy over an already-compiled "
        "engine), not data: excluded from ENGINE_POLICY too, see that ledger's reason"
    ),
    "_posterior_chunk_size": exclude(
        "an analysis-layer scheduling knob (chunk count over an already-compiled "
        "engine), not data: excluded from ENGINE_POLICY too, see that ledger's reason"
    ),
    "_target_modes": exclude(
        "a background-compile scheduling knob, not data: excluded from ENGINE_POLICY "
        "too, see that ledger's reason"
    ),
    "_auto_protocol_likelihood": exclude(_STRUCTURE_REASON),
    "_user_likelihood": exclude(_STRUCTURE_REASON),
    "_calibration_marginalize": exclude(_STRUCTURE_REASON),
    "_profile_mass": exclude(_STRUCTURE_REASON),
    "_profile_mass_resolved": exclude(_STRUCTURE_REASON),
    "_profile_mass_reason": exclude(_STRUCTURE_REASON),
    "_profile_mass_requested": exclude(_STRUCTURE_REASON),
    "_profile_mass_name": exclude(_STRUCTURE_REASON),
    "_profile_mass_prior": exclude(_STRUCTURE_REASON),
    "_profile_mass_bounds": exclude(_STRUCTURE_REASON),
    "_profile_mass_original_spec": exclude(_STRUCTURE_REASON),
    "_cal_n_poly": exclude(_STRUCTURE_REASON),
    "_cal_prior_sigma": exclude(_STRUCTURE_REASON),
    "_eline_marginalize": exclude(_STRUCTURE_REASON),
    "_eline_fitted": exclude(_STRUCTURE_REASON),
    "_eline_prior_type": exclude(_STRUCTURE_REASON),
    "_eline_prior_sigma": exclude(_STRUCTURE_REASON),
    "_eline_prior_width_dex": exclude(_STRUCTURE_REASON),
    "_eline_wavelengths": exclude(_STRUCTURE_REASON),
    "_eline_independent_wavelengths": exclude(_STRUCTURE_REASON),
    "_eline_names": exclude(_STRUCTURE_REASON),
    "_eline_constraint_matrix": exclude(_STRUCTURE_REASON),
    "_eline_amplitude_names": exclude(_STRUCTURE_REASON),
    "_eline_amp_priors": exclude(_STRUCTURE_REASON),
    "_line_flux_override": exclude(_STRUCTURE_REASON),
}
