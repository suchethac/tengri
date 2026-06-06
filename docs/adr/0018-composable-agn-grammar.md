# ADR-0018: Composable AGN Grammar — Canonical Block Design

**Status**: Accepted  
**Date**: 2026-06-05  
**Authors**: AGN Composable Task Force

## Context

Tengri's AGN physics previously lived in two separate registries:
1. Monolithic models (qsogen, SKIRTOR, GRAHSP, etc.) — each bundled disc + torus + lines + attenuation.
2. Minimal stand-alone disc/torus/attenuation functions scattered across submodules.

This split prevented users from mixing, e.g., GRAHSP's UV/optical disc with SKIRTOR's torus. PR A (https://github.com/data-unit/tengri/pull/699) introduced the composable AGN blocks system — a fine-grained pipeline that decomposes the AGN SED into six pluggable stages:

```
disc → nlr → blr → feii → torus → attenuation
```

Each stage has a registry of swappable implementations (e.g., `agn_disc_block="multicolor"`, `agn_torus_block="skirtor"`). This ADR locks the design of that grammar and establishes composable as the canonical user-facing AGN surface.

## Decision

1. **Fixed Slot Set**: The six categories (`disc`, `nlr`, `blr`, `feii`, `torus`, `attenuation`) are the canonical decomposition. No new categories without a full ADR review.

2. **NLR/BLR Split**: Narrow and broad lines are separate stages because:
   - NLR is spatially extended, isotropic (illuminated by intrinsic bolometric).
   - BLR is compact, anisotropic (illuminated by disc 5100 Å).
   - They normalize to different references (L_bol vs l5100_disc).

3. **Normalization Contract**:
   - **Disc block** returns L_λ [erg/s/Å] at rest-frame wavelengths.
   - **NLR block** receives `agn_log_lbol` (bolometric), ignores `l5100_disc`, returns isotropic tuple (maskable, isotropic).
   - **BLR, FeII, torus** receive `l5100_disc` (disc λL_λ at 5100 Å) for line/component normalization.
   - **All additive blocks** are normalized to `l5100_disc` unless self-computing (e.g., qsogen anchors to continuum shape at line wavelengths).
   - **Attenuation block** returns multiplicative factor [0, 1], dimensionless.

4. **Stage-4.5 Masking** (Type-1/2 screening):
   - **Maskable channels**: disc, blr, feii — obscured at Type-2 (edge-on) sightlines.
   - **Isotropic channel**: nlr — visible at all inclinations.
   - **Torus/attenuation**: Applied post-masking; polar dust carries geometry in its per-wavelength extinction curve.

5. **Block Metadata** (Citation, Status, Short Doc):
   - Every registered block includes `citation`, `status` ("production" / "experimental" / "deprecated"), and `short_doc`.
   - Matches monolithic registry entry structure for consistency.
   - Enables `list_agn_blocks()` and `describe_agn_block(name)` introspection.

6. **Monolithic Registry Deprecation Policy**:
   - Monolithic models (qsogen, skirtor, etc.) remain resolvable via `agn_model="qsogen"` for backward compatibility.
   - New code should use `agn_model="composable"` + per-category selectors.
   - Composable is the single canonical user-facing AGN surface; monolithic entries are thin wrappers.

7. **Distinct AGN Surfaces**:
   - **Composable `agn_frac`** (this system): covers all AGN-specific free parameters within the block framework.
   - **CIGALE-embedded `dust_frac_agn`** (dust SEDModelComponent): a separate DALE2014 fixture allowing a fixed AGN luminosity proxy inside the dust continuum. These are orthogonal and should not coexist (duplication warning at model construction time).
   - Why composable blocks are canonical for AGN but dust-emission is a SEDModelComponent (#718): AGN requires per-wavelength masking (Type-1/2 screens) that the component protocol cannot express, while dust-emission is purely multiplicative.

8. **Consumed Keys Contract** (from #722):
   - Runner publishes `L_agn_bol` = intrinsic bolometric luminosity (4π-averaged).
   - Composable AGN publishes `L_2500_intrinsic` = un-reddened disc monochromatic luminosity at 2500 Å (captures disc shape). Monolithic models publish 0.0 (disabled).
   - X-ray component consumes `L_2500_intrinsic` (fallback: `L_2500_30deg` from SKIRTOR, then L_bol BC) to compute AGN corona via the α_ox–L_2500 relation (disc-shape-dependent).
   - Radio component anchors to the B-band (4400 Å) via an L_bol BC (NOT UV-based) — radio remains independent of `L_2500_intrinsic` to preserve loudness physics; UV-anchored radio is a noted follow-up.
   - X-ray/radio are NOT slots in the AGN grammar; they are separate downstream components that depend on AGN luminosity.

8a. **Radio's own composable sub-block grammar** (canonical builder surface):
   - Radio is itself decomposed into two **additive** axes via the nested-dict
     grammar: `radio={'sf': {'type': ...}, 'agn': {'type': ...}}`. SF
     synchrotron (`bell2003`/`delvecchio2021`/`mccheyne2022`/`none`) and AGN
     radio (`powerlaw`/`dpl`/`none`) are selected independently. This mirrors
     the AGN block grammar's sub-block nesting (cf. `dust.emission`,
     `agn.disc`) — the builder grammar (`SEDModel.build` /
     `builders.radio.sf.<variant>()` / `builders.radio.agn.<variant>()`) is the
     **canonical surface**, and the contract is verified through it.
   - **FIRRC evolution coefficients are free parameters.** The mass/redshift
     `q_IR(M*, z)` coefficients `(q0, mass_slope, z_slope)` are surfaced as
     fittable, **model-specific** `radio_delv_*` / `radio_mcch_*` params,
     addressable in the `radio.sf` sub-block. Names are model-specific (not a
     shared `radio_q0`) because the two calibrations carry genuinely different
     literature defaults and mass-slope **sign conventions**; a shared name
     would silently apply the wrong default to the inactive model. Only the
     active `sfr_mode`'s triplet is consumed — a `radio.sf` `'*'` wildcard is
     scoped to the active model (block-scoped, as in the AGN grammar), and the
     inactive model's coefficients stay Fixed no-ops. DPL turnover knobs
     (`radio_alpha_thin`/`_thick`, `radio_log_nu_t`/`_nu_cut`) are the symmetric
     `radio.agn` sub-block params. These coefficients are designed to promote
     to `PopulationFitter` hyperparameters without code changes.

9. **Impact on Posterior**:
   - Users prefer one canonical surface. Composable blocks are simpler to discover, mix, and extend than hand-written monolithic adapters.
   - Fine-grained control (e.g., swapping lines while keeping disc fixed) reduces model complexity and speeds iteration.
   - Grid-backed blocks (skirtor, fritz, GRAHSP) now have the same discovery and metadata exposure as analytic blocks.

## Consequences

- **Backward Compatibility**: Monolithic names resolve as deprecated aliases; no breaking changes to existing model specs using `agn_model="qsogen"`, etc.
- **Block Extension**: Adding a new disc model is now a single ~50-line file (one block implementation) instead of a full monolithic adapter.
- **Discovery**: `list_agn_blocks()` and `describe_agn_block()` are the recommended API for choosing components; REGISTRY.md is a frozen reference.
- **Testing**: New blocks inherit a conformance test suite; no need to write per-block tests (though physics assertions are required).
- **Documentation**: Docstrings on blocks use standard numpydoc format with References sections extractable to `citation=` metadata.

## References

1. **PR A Composable Blocks Protocol** (https://github.com/data-unit/tengri/pull/699): Introduced AGN_BLOCKS registry and @register_agn_block decorator.
2. **ADR-0009: Typed Pipeline Contract** (#xxx): Component inputs/outputs/optional_inputs protocol; composable blocks follow this convention.
3. **ADR-0011: SEDModelComponent Base** (#xxx): Base class used for dust-emission and nebular components; composable blocks use a separate block protocol (lighter-weight for pure spectra functions).
