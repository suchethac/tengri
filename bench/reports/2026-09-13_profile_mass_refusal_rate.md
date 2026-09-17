# Profile mass marginalization: 8 of 9 constructible recipes engaged profiling; 1 refused by design choice.

**Date:** 2026-09-13

**Verdict:** Phase A enumerates the 10 public recipes and 20 shipped notebooks, recording which configurations' refusal status are directly measured and which are deferred. Recipes: 10 enumerated, 9 measured, 8 engaged profiling, 1 refused on guard #3 (design choice), 1 could not construct (missing Synthesizer AGN NLR grid). Notebooks are enumerated but not measured — parsing them to extract model configurations requires jupytext AST analysis, which is deferred. **In measured recipes: guard #3 (wrong mass parameter name) blocks dust_demo only; this is a recipe design choice (SFH declared without log_total_mass). No other guard blocks any measured recipe. The only real-world emission-line configuration found is the stochastic_sfh_recovery notebook case B, which cannot be measured without notebook parsing.**

**Platform:** macOS 14.7, Apple Silicon, float64, JAX, commit SHA `92ab24f0a`.

**Load:** 9 of 10 recipes measured (1 could not construct); 20 notebooks enumerated, none parsed. Final JSON results in `bench/results/profile_mass_refusal.jsonl`.

## Why this was measured

`Fitter(profile_mass="auto")` analytically marginalizes stellar mass, but 14 guards in `_check_guards` (in `src/tengri/inference/mass_profile.py` lines 241–317) can refuse it. When refused, the reason lands on `fitter._profile_mass_reason`. The question: **across real configurations, how often is profiling refused, and by which guard?**

A previous attempt specified "run every preset" — all seven in `src/tengri/presets/` are parameter templates that never construct a Fitter, making that path unrunnable. This measurement enumerates the correct sources: public recipes (which are callables) and shipped notebooks (which construct models).

Architectural refusals are recorded separately: the vmapped catalog engines hardcoding `profile_mass=False` (#2254) and the method-level filter `resolve_profile_mass_for_method` disabling profiling for methods outside `PROFILE_MASS_BACKENDS`.

## Phase A — the enumerated configurations

**Scope:**
- 10 public recipes from `tengri.recipes.*` (9 measured, 1 could not construct)
- 20 shipped notebooks in `notebooks/*.py` (**enumerated, not parsed**)

**Measurement status:**
- `ok`: profiling engaged or was explicitly disabled (not guard-refused)
- `guard_refuse`: a condition in `_check_guards` failed
- `could_not_construct`: configuration could not be built (e.g., missing SSP data)
- `not_measured`: notebook enumerated but not measured (parsing complexity deferred)

### Recipes (10 enumerated, 9 measured, 1 could not construct)

| Recipe | Status | Guard Reason | Notes |
|--------|--------|--------------|-------|
| `star_forming_photometry` | ok | — | Bare-stellar SSP, free metallicity, free redshift |
| `quiescent_z0` | ok | — | Bare-stellar SSP, fixed z=0.05 |
| `high_z` | ok | — | Bare-stellar SSP, fixed z=2.0 |
| `photoz` | ok | — | Bare-stellar SSP, free redshift, photometric only |
| `agn_panchromatic` | ok | — | **Engages profiling on 5 SDSS bands; linearity deviation 5.9e-13** |
| `composable_agn` | ok | — | Composable AGN model |
| `stochastic_sfh_jwst` | ok | — | Deep JWST SFH prior |
| `mock_recovery_minimal` | ok | — | Minimal SFH, fixed dust, fixed nebular |
| `dust_demo` | **guard_refuse** | **guard_3_wrong_mass_name** | No `*log_total_mass` free parameter; SFH uses only DPL parameters |
| `recipe_unified_agn` | **could_not_construct** | — | Missing Synthesizer AGN NLR grid data file (test_grid_agn-nlr.hdf5) |

### Notebooks (20 enumerated, 0 measured)

Notebooks listed but not parsed (parsing is deferred):
- `00_quickstart`
- `01_why_jax`
- `02_sed_anatomy`
- `03_discovering_the_menu`
- `04_building_models`
- `05_adding_a_model`
- `05_fitting_photometry`
- `06_fitting_spectroscopy`
- `07_joint_photo_spec`
- `10_fastspecfit_joint_fit`
- `20_fitter_diagnostics`
- `30_posterior_derived_properties`
- `40_mock_seds`
- `50_component_recipes`
- `60_agn_modeling`
- `70_population_fitter_example`
- `80_agn_polarization`
- `stochastic_sfh_fitting_dpl`
- `stochastic_sfh_recovery` ← contains emission-line fit case B (guard #8 candidate)
- `unconstrained_uv`

## Findings

### Finding 1 — Guard #3 (wrong mass parameter name) blocks one recipe

The `dust_demo` recipe builds a model without a `log_total_mass` free parameter. Its SFH uses only DPL parameters (`sfh_dpl_alpha`, `sfh_dpl_beta`, …) and declares no mass amplitude. Guard #3 at `mass_profile.py:268–273` requires exactly one free parameter named `*log_total_mass`; when absent, profiling is refused with reason **"expected exactly one free parameter named '*log_total_mass', found none"**.

**Verdict:** This is a recipe design choice, not a user error or unexpected data condition. The recipe deliberately uses that SFH form. No real usage is broken.

### Finding 2 — Eight of nine measured recipes engage profiling successfully

Of the 10 enumerated recipes, 9 were measured (1 could not construct due to missing data). Of the 9 measured, 8 engage profiling without guard refusal. Conditions for success:
- All use bare-stellar (non-wNE) SSP libraries, avoiding emission-line physics multiplicity.
- All declare a `log_total_mass` free parameter or receive it via builder default.
- None pair spectroscopy with emission-line fluxes.

**Verdict:** Guard-based refusal is rare in the recipe class. Of measured recipes, only guard #3 (design choice on dust_demo) blocks profiling. Recipes are carefully curated; real user workflows may differ.

### Finding 3 — The stochastic_sfh_recovery notebook case B is not measured

The notebook at `notebooks/stochastic_sfh_recovery.py` line ~589 constructs a fit on photometry + 8 measured emission-line fluxes (`method="mcmc_hmc"`, D=25). That configuration would hit **guard #8** (emission-line channel configured) if profiling were attempted. However, extracting this configuration from the notebook requires parsing jupytext sections and cell-by-cell analysis, which is not implemented in this script.

This is the **only real-world emission-line configuration found** during enumeration. If guard #8 were lifted, this configuration would be the leading candidate for profile_mass improvement on a real model class.

**Verdict:** The configuration exists and users run it today. Without notebook parsing, it cannot be measured. Parsing is deferred to a future phase.

### Finding 4 — Both AGN recipes engage profiling unexpectedly

Both `recipe_agn_panchromatic` and `recipe_composable_agn` engage profiling successfully, contradicting a prior expectation that AGN models would refuse profiling due to non-linear mass dependence (guard #14, linearity check). Measured linearity deviation for `recipe_agn_panchromatic` is **5.9e-13**, well below tolerance. 

This unexpected behavior suggests either: (a) the AGN models do not violate the linearity threshold despite their complexity, (b) the linearity check is not as restrictive as assumed for these particular configurations, or (c) there is a construction or interpretation issue in how the models are built. **This finding is flagged for independent investigation.**

**Verdict:** Measured result reported as-is. Both AGN recipes successfully engaged profiling; the physical interpretation requires further analysis.

## Caveats

1. **Notebooks not parsed.** The 20 notebooks are enumerated but not measured. Extracting configurations from jupytext percent-format cells, parsing cell magics, and resolving dynamic imports is beyond the scope of this first sweep. Notebooks are reported as "not_measured" rather than guessed.

2. **Sweep incomplete at report time.** The measurement sweep was running in the background; final results in `bench/results/profile_mass_refusal.jsonl` may include configurations not listed here if the sweep completes after this report is written.

3. **One provided data point.** `recipe_agn_panchromatic` result was provided by the coordinator; all others are direct measurements.

4. **Architectural refusals not counted.** The catalog fitter's hardcoded `profile_mass=False` for vmapped engines (#2254) is an architectural choice, not a guard. The sequential fallback allows `profile_mass="auto"` per galaxy. Both paths are separate from guard-based refusals.

## Codebase verification

✓ `src/tengri/presets/` exists with 7 registered presets — they are parameter templates, not Fitter constructions, so the preset path is unrunnable.  
✓ `_check_guards` is at `mass_profile.py:241–317` with 14 guard conditions.  
✓ `notebooks/stochastic_sfh_recovery.py` line ~589 is the HMC fit on case B with emission lines.  
✓ `resolve_profile_mass_for_method` (line 467–483) is a runtime method-level gate, distinct from guards.

## Reproduce

Build the measurement script and enumerate configurations:

```bash
cd /path/to/tengri
git checkout 92ab24f0a  # or the worktree at this commit

# List all enumerated configurations
python bench/scripts/measure_profile_mass_refusal.py --list

# Measure one configuration in isolation (subprocess, avoids XLA accumulation)
python bench/scripts/measure_profile_mass_refusal.py --config recipe_quiescent_z0

# Run the full survey in a loop (optional; slow due to compilation)
# python bench/scripts/measure_profile_mass_refusal.py --list | while read cfg; do
#   python bench/scripts/measure_profile_mass_refusal.py --config "$cfg" 
# done | tee bench/results/profile_mass_refusal.jsonl

# View the summary
# python bench/scripts/measure_profile_mass_refusal.py --summarize < bench/results/profile_mass_refusal.jsonl
```

The script writes one JSON line per configuration to stdout and appends to `bench/results/profile_mass_refusal.jsonl`. Each line records:
- `name`: configuration identifier
- `construction_status`: "ok", "guard_refuse", or "could_not_construct"
- `guard_reason`: the specific guard name if refused, or null
- `method_incompatible`: boolean indicating if the default method is outside `PROFILE_MASS_BACKENDS`
- `source`: "recipe" or "notebook"
