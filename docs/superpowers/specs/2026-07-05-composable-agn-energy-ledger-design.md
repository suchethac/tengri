# Composable AGN energy ledger — one self-consistent framework, per-code policies

**Date:** 2026-07-05 · **Branch:** cs/agn-energy-ledger-design · **Status:** draft (design review)

## Goal

Make the composable AGN grammar (`agn={'type':'composable', ...}`) the single,
energy-**self-consistent** framework for AGN SEDs: the disc's intrinsic
bolometric luminosity is the one energy source, every reprocessing component
(torus, polar dust, lines) *debits* it, and the total emitted energy conserves
`L_bol` by construction. A small, explicit **normalization-policy layer**
(`agn_norm`) selects how the energy is allocated, so the *same* grammar
reproduces each reference code's bookkeeping (CIGALE, AGNfitter, GRAHSP,
Synthesizer). The headline user-facing deliverable is a set of **turnkey
reproduction recipes** — `recipes.agn_cigale_skirtor()`,
`recipes.agn_synthesizer_unified()`, `recipes.agn_grahsp()`, … — each a curated
`(disc + torus + lines + atten + policy)` bundle that reproduces one reference
code's AGN model out of the box, backed by a parity test. Once the composable
path reproduces each of the 14 monolithic `AGN_MODELS`, those are retired onto
faithful presets — completing #909 / #846 the right way and superseding the
broken PR #916.

## Why now

PR #916 tried to retire the monolithic `AGN_MODELS` by aliasing each old name to
a composable "preset." Direct verification (monolithic function vs composable
preset, matched `agn_frac`) showed it is **not physics-equivalent**:

| model | preset / monolithic amplitude | cause |
|---|---|---|
| multicolor_agn / kubota_done | 1.00 (equivalent) | — |
| silva04, cat3d_wind | 1.96× | torus **added on top**, not debited |
| adaf | 1.64× | same |
| skirtor | 0.51× | `cigale_joint` coupling inactive at `fracAGN=0` |
| skirtor_stalevski | 3.93× | raw-SKIRTOR SED not reproduced by disc+torus |
| qsogen, grahsp, richards2006, unified_nlr_blr | raises `ValueError` | preset references non-existent block names |
| adaf, kubota_done_full | `agn_ebv_disc` inert | preset drops disc reddening |

Root cause (confirmed in `blocks/runner.py:466-587`): energy-conserving
normalization (`cigale_joint`) is wired **only** for `torus="skirtor"` with
`fracAGN>0`. Every other torus falls back to `agn_norm="independent"`, which
adds `L_disc + L_torus` with no conservation. The band-integrated energy proves
it: increasing `agn_torus_frac` from 0.0 → 0.6 keeps the **monolithic** total
flat at 3.831e78 (conserving) but grows the **composable independent** total
3.831e78 → 6.129e78 (adds energy).

The 2026-06-04 "fully composable AGN" spec deliberately deferred exactly this:
its Out-of-Scope lists *"bit-exact numerical parity with unified_nlr_blr
(different disc/torus coupling)"* and *"retiring the monolithic models."* This
design delivers that deferred layer. The #556 work already articulated the
target — *"block-level faithfulness isn't enough; the cross-block
normalization/energy-balance must be explicit and switchable so the same grammar
reproduces each code's bookkeeping"* — and shipped the `agn_norm` seed
(`cigale_joint` | `independent`). This generalizes that seed to all tori.

## How the reference codes allocate AGN energy (prior art)

Grounded in `docs/dev/archive/agn-model-comparison.md`,
`docs/dev/agn-reference-crosscheck-2026-05-24.md`, and the X-CIGALE / GRAHSP /
AGNfitter sources:

- **CIGALE / X-CIGALE (Yang+2020)** — north-star. Single `agn_power` reference;
  disc/torus/polar tied by fixed template ratio `R = lumin_disk/lumin_dust`;
  `fracAGN = L_AGN/L_total` in a configurable band. Energy-conserving.
- **AGNfitter-rX** — each component has an independent log-normalization free
  param on [−10,+10]; energy balance only via optional informative priors.
- **GRAHSP (Buchner+2024)** — components normalized to disc λL_λ(5100 Å).
- **Synthesizer UnifiedAGN** — disc `L_bol` from M_BH·ṁ·η; NLR/BLR/torus via
  covering fractions; conserving by construction.
- **Prospector/FSPS** — one CLUMPY template; `fagn = L_AGN/L_stellar_bol`.
- **tengri monolithic** — `agn_torus_frac` splits `L_bol` (disc gets 1−frac,
  torus reprocesses frac). Conserving. This is the convention #916 dropped.

## Decisions (maintainer)

1. **Self-consistency** = an energy-conserving composition is the canonical
   default, with named policies reproducing each code on top.
2. **Energy scope** = AGN-internal (disc↔torus↔polar↔lines share one AGN L_bol
   ledger; X-ray corona and radio consume `L_2500`/`L_bol` via the existing
   ADR-0009 cross-component contract; stellar↔dust DL07 balance unchanged).
3. **Monolithic models** = retire onto faithful presets, gated by an
   equivalence test (bit-exact, or documented tolerance where coupling
   genuinely differs).
4. **Architecture** = explicit energy ledger (Approach A); CIGALE's
   single-reference is the `cigale_joint` *policy* within the ledger.
5. **No backward compatibility** = the ledger is *the* behavior; no legacy
   additive dual-path, no "existing configs unchanged" guarantee. Existing
   composable configs move to the conserved (correct) numbers.
6. **Disc-reddening param** = canonical `agn_ebv_disc`; `agn_polar_ebv` removed.

## Section 1 — The energy-ledger runner (architecture & data flow)

`compose_l_nu` (`blocks/runner.py`) becomes the single ledger keeper:

```
 INPUT: agn_log_lbol → L_bol = 10^lbol · L_sun            (the ledger total)

 Stage 0  disc block → S_disc(λ)         intrinsic shape, ∫ normalized to L_bol
 Stage 1  ALLOCATE (policy-driven):
            f_tor  = torus intercepted fraction     ┐
            f_pol  = polar intercepted fraction     ├─ Σf ≤ 1 (policy sets these)
            f_line = line covering fractions        ┘
 Stage 2  reprocessors emit at allocated budget (shape × budget):
            L_torus = normalize(S_torus) · f_tor ·L_bol
            L_polar = normalize(S_polar) · f_pol ·L_bol   (from reddening-removed disc UV)
            L_lines = S_lines · f_line·L_bol              (NLR isotropic, BLR maskable)
 Stage 3  disc observed = (1 − f_tor − f_pol) · L_bol · Ŝ_disc      ← DEBITED
 Stage 4  Type-1/2 mask on {disc, BLR, FeII}     (existing Stage 4.5 — unchanged)
 Stage 5  host/foreground attenuation screen     (existing Stage 5 — unchanged)
 ────────────────────────────────────────────────────────────────
 INVARIANT:  ∫(L_disc + L_torus + L_polar + L_lines) dν  ≈  L_bol
```

The masking (Stage 4.5) and attenuation (Stage 5) stages shipped in the
2026-06-04 spec are untouched; the energy accounting slots underneath them. The
SKIRTOR `#556` `agn_power×R` machinery (`runner.py:466-587`) becomes the
`cigale_joint` implementation of "compute `f_tor` from R," no longer a hardcoded
`if torus=="skirtor"` branch.

## Section 2 — Block contract (shape-providers; no back-compat)

Blocks stop owning absolute normalization; they provide **shapes** and the
runner scales each to its allocated budget.

| Category | Today (`_protocol.py`) | Under the ledger |
|---|---|---|
| **disc** | returns `L_λ` at full `L_bol` | returns `L_λ` shape; runner reads `∫` as `L_bol` and **debits** by Σf |
| **torus** | `L_λ` scaled to `l5100_disc`, added | shape; runner normalizes `∫=f_tor·L_bol` (`f_tor` from `agn_torus_frac` default, or template-R under cigale) |
| **polar** | ad-hoc in runner (`agn_polar_ebv`) | reprocesses reddening-removed disc UV → conserved, uniform across tori |
| **nlr/blr/feii** | `l5100_disc`-normalized, added | shapes; drawn from covering-fraction budget under conserving/synthesizer policy |
| **attenuation** | multiplicative factor | unchanged |

The registry (`register_agn_block`) gains one optional field
`reprocessed_fraction_param` (e.g. torus → `"agn_torus_frac"`); the runner uses
it to allocate. Because there is no back-compat, all torus/polar/lines blocks
migrate to shape-providers as part of Phase 1 — there is no dual "additive
legacy" path to maintain.

## Section 3 — The normalization-policy layer (`agn_norm`)

The policy governs only how `L_bol` is allocated and what reference the disc
scales to. The L_bol *source* (fixed `agn_log_lbol` vs M_BH·ṁ) lives in the disc
block; line/torus *shapes* live in their blocks. So "which code" =
`(disc block + line/torus block + policy)`, keeping the policy set small:

| `agn_norm` | Reproduces | Allocation rule | Conserving |
|---|---|---|---|
| **`conserving`** *(default)* | tengri monolithic; Synthesizer (qsosed disc + grids) | disc `L_bol` from `agn_log_lbol`; `f_tor=agn_torus_frac`, `f_pol` from polar covering, lines from covering; `disc_obs=(1−Σf)·L_bol` | yes (structural) |
| **`cigale_joint`** | X-CIGALE (Yang+2020) | single `agn_power` reference; disc/torus/polar tied by R (template-R for SKIRTOR, covering-R for analytic tori); `fracAGN` band | yes |
| **`l5100`** | GRAHSP (Buchner+2024) | components normalized to disc λL_λ(5100 Å) — the current anchor as explicit policy | anchor-convention |
| **`fagn`** | Prospector/FSPS | AGN allocated as a fraction of the *stellar* bolometric (`L_AGN = fagn·L_★`); disc/torus conserve internally | yes (AGN-internal) |
| **`independent`** | AGNfitter-rX | each component carries its own log-norm; no debiting | no (comparison-only) |

Two allocation modes sit outside this table by construction:
- **Self-contained tori** (`qsogen`, `grahsp` — the `_SELF_CONTAINED_TORI` set)
  bundle disc+lines+torus in one template; they bypass the ledger and are
  self-normalized. Their recipes select the self-contained block and ignore
  `agn_norm`.
- **External-reference policies** (`cigale_joint`'s `fracAGN`, `fagn`) set the
  allocation *fractions* from a luminosity outside the AGN (galaxy total /
  stellar bolometric), consumed via the existing ADR-0009 cross-component
  contract. The AGN-internal conservation (`Σ=L_bol`) still holds; only the
  fraction that defines `L_bol` relative to the host comes from outside.

## Section 4 — Reddening unification

Today disc reddening exists twice: `agn_ebv_disc` (monolithic `_redden_disc`,
inert in the composable path — the #916 bug) and `agn_polar_ebv` (composable
runner, type-1 gated, energy routed to polar; `runner.py:459-464`). Unify:

- Canonical **`agn_ebv_disc`** (the disc is what's reddened); type-1 sightline
  gate retained (`cos_inc ≥ sin(oa)`).
- Removed UV energy is **conserved into the polar-dust graybody** (the
  CIGALE-correct behavior; the ledger makes it uniform, not SKIRTOR-special).
- `agn_polar_ebv` removed (no back-compat) — one name, one mechanism.

## Section 5 — Retirement + the equivalence gate

The 14 monolithic models retire onto `conserving`-policy presets with **correct
block names** (fixing the #916 typos — `richards2006`, `grahsp_sbpl`, `qsogen`,
`analytic`) and restored disc reddening.

```
tests/regression/agn/test_monolithic_equivalence.py  (CI, marker: regression_bug)
  for each of 14 models:
     sed_mono   = <monolithic function>(wave, lbol, **matched_params)
     sed_preset = resolve_agn_model(name)(wave, lbol, **matched_params)   # composable conserving preset
     assert allclose(sed_mono, sed_preset, rtol=GATE[name])
        GATE = 1e-10          where coupling is identical (multicolor, kubota_done, silva04, cat3d, adaf, …)
             = documented tol  where coupling genuinely differs (skirtor η(i) ~12%, per #556) — tol + comment
```

Only a model whose gate is green may have its monolithic registration + function
deleted. The ratchets (`check_single_dispatch`, `check_registry_completeness`)
then confirm `AGN_MODELS` shrinks to `{composable}` — closing #909 / #846. Old
names still *resolve* to the faithful preset (the retirement mechanism), not as
behavioral back-compat.

## Section 5b — Reference-code reproduction recipes (the headline product)

Named turnkey configs under `tengri.recipes.*`, each returning a composable
grammar dict that reproduces one reference code's AGN model. They compose the
existing blocks + the policy layer; the value is curation + a shipped parity
test. This is what a user reaches for to say "give me CIGALE's AGN" or "give me
Synthesizer's UnifiedAGN."

| recipe | disc | torus | lines | atten | `agn_norm` | reproduces | parity test |
|---|---|---|---|---|---|---|---|
| `agn_cigale_skirtor()` | schartmann2005 (=skirtor disk_type=1) | skirtor | none | polar_dust | `cigale_joint` | X-CIGALE SKIRTOR (Yang+2020) | `reproduction/cigale` §9 (exists) |
| `agn_cigale_fritz()` | schartmann2005 | fritz | none | polar_dust | `cigale_joint` | CIGALE Fritz (2006) | new — Fritz grid parity |
| `agn_synthesizer_unified()` | kubota_done (KD18 = qsosed) | two_temperature | nlr + blr = `synthesizer_spectra` | none | `conserving` | Synthesizer UnifiedAGN | `reproduction/synthesizer` (exists) |
| `agn_grahsp()` | grahsp_sbpl | grahsp | grahsp (nlr+blr+feii) | grahsp_biatten | `l5100` | GRAHSP (Buchner+2024) | grahsp parity (exists) |
| `agn_agnfitter(disc=, torus=)()` | R06 / SN12 / KD18 / THB21 | S04 / nenkova / skirtor / cat3d_wind | (per disc) | smc_prevot | `independent` | AGNfitter-rX 4×4 | `reproduction/agnfitter` (exists) |
| `agn_qsogen()` | qsogen | qsogen (self-contained) | qsogen | qsogen_smc | self-normalized | qsogen (Temple+2021) | qsogen parity (exists) |
| `agn_prospector()` | (CLUMPY-shaped) | nenkova | none | none | `fagn` (L_AGN/L_stellar) | Prospector/FSPS | new — 2-param fagn |

Notes:
- Self-contained tori (`qsogen`, `grahsp`) already bundle their own components
  (`_SELF_CONTAINED_TORI` in the runner) → they bypass the ledger and are
  self-normalized; the recipe simply selects the self-contained block.
- `agn_agnfitter(...)` is parameterized (4 discs × 4 tori) rather than 16
  separate functions — one factory, autocomplete-friendly.
- Each recipe docstring states its SSP requirement (bare-stellar vs any),
  cites the reference code + paper, and names its parity test. Recipes live
  under the existing `tengri.recipes.*` surface (alongside `describe_agn_model`).
  Line regions are separate `nlr`/`blr` block categories, so a recipe sets both
  (there is no combined `nlr_blr` block).
- Where a monolithic model maps 1:1 to a code recipe (e.g. `skirtor` →
  `agn_cigale_skirtor`), the Section-5 retirement preset and the Section-5b
  recipe share one definition — no duplication.

## Section 6 — Testing & governance

| Test | Marker | Guarantees |
|---|---|---|
| Conservation invariant | `conservation` | for every (disc × torus) under `conserving`: `∫Σ_emitted dν ≈ L_bol`, swept over `agn_torus_frac ∈ [0,0.9]` (the axis #916 broke) |
| Monolithic equivalence gate | `regression_bug` | each of 14 models: composable preset == monolithic (bit-exact or documented-tol) |
| Per-code parity | `regression_paper` | `cigale_joint` holds the §9 CIGALE reproduction; `l5100` holds GRAHSP parity; `independent` holds AGNfitter comparison |
| No-op param guards | `gradient` | `agn_ebv_disc` changes `predict()` w/ nonzero gradient; `agn_torus_frac` redistributes not adds |

Existing guards (`check_single_dispatch`, `check_registry_completeness`,
param-prefix, taxonomy markers) keep running. The conservation-invariant test is
the one that would have caught #916 immediately.

## Section 7 — Blast radius & phasing

Five independently-green PRs (supersedes #916; retirement lands Phase 4):

| Phase | Scope | Files |
|---|---|---|
| 1 | Energy-ledger spine + `conserving` default + conservation invariant | `blocks/runner.py`, torus/polar/lines blocks → shape-providers, param declarations |
| 2 | Fold SKIRTOR #556 into `cigale_joint`; add `l5100`/`independent`/`fagn` policies + per-code parity | `blocks/runner.py`, `blocks/registry.py`, reproduction/cigale regression |
| 3 | Reddening unification (`agn_ebv_disc` canonical; remove `agn_polar_ebv`) | `blocks/runner.py`, param declarations, dust SMC path |
| 4 | Fix 14 presets + equivalence gate; retire monolithic model-by-model | `unified.py`, `test_monolithic_equivalence.py` |
| 5 | Reference-code reproduction recipes (`agn_cigale_skirtor`, `agn_synthesizer_unified`, `agn_grahsp`, `agn_agnfitter`, `agn_qsogen`, `agn_prospector`) + their parity tests | `recipes/`, `reproduction/*` wiring, `list_recipes`/`describe_recipe` |

**Risks:** Phase 2 touches the #556-validated `cigale_joint` path → CIGALE §9
reproduction is a regression lock. Heavy builds under
`LIMIT_GB=20 scripts/run_with_oom_monitor.sh`. SSP-gated tests skip in CI; run
the clean-main differential to catch precompute regressions. JIT rule: policy
dispatch is on static Python strings at trace time — never trace `agn_norm` or
the block registry.

## Out of scope

- Panchromatic energy ledger (stellar→dust→AGN→X-ray→radio one budget) — this
  design is AGN-internal; X-ray/radio consume via the existing contract.
- New disc/torus/line *physics* (no new blocks) — this is composition/energy
  bookkeeping only. New-physics ports remain their own issues (#898 ADAF,
  #901 tbabs, etc.).
- Changing the Type-1/2 masking or attenuation stages (shipped 2026-06-04).

## References

- `blocks/runner.py:380-644` — current `compose_l_nu` (verified 2026-07-05).
- `blocks/_protocol.py` — block registry contract.
- `components/agn/unified.py` — monolithic models + `_AGN_PRESETS` (#916).
- `docs/dev/archive/agn-model-comparison.md` — cross-code AGN comparison.
- `docs/dev/agn-reference-crosscheck-2026-05-24.md` — CIGALE/PyQSOFit cross-check.
- `docs/superpowers/specs/2026-06-04-fully-composable-agn-design.md` — the
  masking/lines-split layer this design builds on.
- #556 / #710 — the `agn_norm` policy seed (cigale_joint | independent).
- Yang et al. 2020, MNRAS 491, 740 (X-CIGALE); Stalevski et al. 2016
  (SKIRTOR); Buchner et al. 2024 (GRAHSP); Zhuang et al. 2024 (AGNfitter-rX).
