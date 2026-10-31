# Batched spectroscopic stack for the DESI DR1 dwarfs: #3115–#3118, plus the tengri-side limitations on Risa's pages

## 0. Execution model (binding)

- **The orchestrator (this session) never implements.** Every source, test, doc and measurement edit is done by a
  **Haiku** implementer launched with `isolation: "worktree"` from the main checkout
  (the repository root; cwd must be the repo before each launch). The orchestrator writes
  briefs (`$CLAUDE_JOB_DIR/tmp/briefs/`), posts claim comments, verifies every lane's diff on disk (grep the claimed
  symbols, re-run its RED/GREEN commands, one mutant per new test file), runs read-only gates through **Haiku gate
  lanes** with exact commands, pushes verified heads by refspec, opens and labels PRs, and messages the merge
  orchestrator session (`orchestrator session tengri [8e7a9a]`). It never merges.
- **Escalation:** Sonnet only after two verified Haiku failures on one item, said so in the report. The test portion
  of each item is briefed as its own item.
- **Worktree rules in every brief:** open with `pwd` + `git branch --show-current` + `git rev-parse --show-toplevel`
  and STOP unless under `.claude/worktrees/agent-`; never `git checkout/restore/stash/reset`; never call
  `EnterWorktree`; mutation restores from file copies in job tmp; `git diff --cached --stat` before every commit;
  report RED output, GREEN output, `git diff --stat`, exact commands and exit codes.
- **Stacking:** a lane starting from a pushed branch runs `git fetch origin <br> && git merge --no-ff
  --no-rerere-autoupdate origin/<br>` first. The orchestrator pushes `git push origin <sha>:refs/heads/<branch>`
  and verifies with `gh api pulls/N --jq .head.sha`. Never resume an isolation agent that stopped with no changes.
  Never switch the orchestrator into a worktree while any lane runs. One worktree per Bash call.
- **No population-fit tests; `JAX_PLATFORMS=cpu`; shared venv `.venv` (jax 0.11.1).** Pytest restates the markers:
  `-o addopts="--tb=short --strict-markers -m 'not crossval and not slow and not benchmark and not population_fit' -n 0 --dist=loadfile"`
  (slow trees: drop `not slow`). One pytest process per file for `tests/contract/`.

## 1. Context

The batched spectroscopic stack (`observation/batched.py`, `forward/batched_spectrum.py`,
`inference/batched_spectro.py`; #2845/#2880/#3075) compiles once per `SpectroBatchSpec` and traces per-galaxy data as
`SpectroBatch` leaves, but refuses emission lines (`check_template_model`, `eline_mode != "off"`) and has no
analytic marginalization. Risa Wechsler's DESI DR1 dwarf project (pages read 2026-10-31; her project repo is private,
her tengri fork `rhw/tengri` is public) is the motivating user: a production fit holds an A100 for 12.6 min of which
2.7 min is sampling, because the model is rebuilt and recompiled per galaxy. Issues #3115–#3118 are the owner's spec;
the owner then asked to also fix the tengri-side limitations on her pages in the same run, with **design decisions
upheld across modules** (no batched-only math), and asked what of her work can go to main.

### 1a. Her pages → tengri actions

| Her limitation / design need | Action |
|---|---|
| Compile per galaxy, data closed over as constants | #3115–#3117, #3118 states the measured gain |
| Lines fitted with free amplitudes integrated out **in the same linear solve** as the calibration (production §2) | #3117 as an **exact joint** marginal, both paths (closes #2354) |
| Per-camera scales + shared Chebyshev tilt; flat prior in data-flux units (+k ln a); Gaussian prior on the mean multiplier q from the catalog fiber fraction (pipeline §3) | **PR-B**: per-segment scale columns, the scale-free prior with k counting only model-scaled columns, the q prior as a **log-normal on ln q** (her sampled Gaussian-in-q puts 7.4% at q < 0); names need the owner's OK |
| Banded pixel covariance from constant-velocity rebinning; her 161-diagonal matvec is a truncated inverse | **PR-A**: banded *covariance* with an exact banded Cholesky solve on both paths (never a truncated precision) |
| MEADS K=128, nested R-hat stopping, scattered starts (pipeline §5); upstream `mcmc_ghmc` is `tier="broken"` while her 40/40 mock calibration says otherwise | **PR-C**: `"ghmc"` in the catalog engine; `nested_rhat` (Margossian+2024) rank-normalized + tail ESS; distinct starts per superchain; wall hits rejected without NaN propagation; tier relabel is the owner's call with her evidence |
| Cue's lines stop at H10 (3797.9 Å, verified in `data/cue_reference_outputs.npz`); `LineList.default_optical()` stops at Hε; two-thirds of her young-bin pull sits below 3780 Å rest | **PR-D** (physics, owner go/no-go): Balmer series above Cue's last line from Case-B ratios (Storey & Hummer 1995 via pyNeb 1.1.30, cited), scaled to Cue's Hβ; H8–H14 + He I 3889 + [Ne III] 3968 in `default_optical()` with verified vacuum wavelengths |
| Lines untied from the model (no Balmer-decrement dust constraint; both referees) | **PR-G**: `eline_prior_type="model"`: Gaussian prior on amplitudes centered on the model's own line prediction with a separate line aperture factor, through the one primitive |
| MW reddening of the model spectrum (F99, R_V 3.1, SFD E(B−V)); upstream has Cardelli only, applied in `obs_sed` only, E(B−V) a build-time float | **PR-I**: Fitzpatrick (1999) law; the foreground screen on the projection path (spectrum and photometry consistently); per-galaxy `ebmv_mw` leaf on the batched path; note the SF11 0.86 recalibration in the docstring |
| Her two fork commits not on main: Cue tables from the near-solar [α/Fe] slab (999c29e1f, +14/−1 + a 99-line contract test) and contract-over-metallicity-per-α-node before interpolating (87c8bdc64, +71/−20; removes a 26.75 GiB allocation under vmap×128) | **PR-F**: carry both onto current main, Risa as co-author (the #3024/#3063/#3075 precedent); re-run her contract test |
| `Spectroscopy(nebular_lines=False)` (#3112, MERGEABLE/CLEAN, all checks green; touches one line of `predict_batched_observables`) | ask the merge orchestrator to land #3112 first; #3115 re-merges main after it |
| σ_gas fixed at 30 km/s, no velocity offsets | no code: `eline_sigma_kms` / `eline_delta_v_kms` are per-galaxy latents on the batched path when FREE; #3118 documents it |
| Filter-curve sort constant-folded per compile (`observation/photometry.py:179,194`) | **PR-E** (perf, optional): hoist to build time if the diff stays small |
| youngfrac coordinates for the continuity SFH (upstream has Dirichlet stick-breaking only) | owner decision: file an issue; not in this run unless asked |
| Degrade-and-rebin helper with covariance propagation (her preprocessing) | optional **PR-J**, after PR-A, if the owner wants it upstream |
| Three processes per galaxy, un-jitted starts, re-traces | her code; no action |
| Science caveats (young-bin systematic, prior-dependent SFH, mass outliers, aperture, solar-scaled isochrones) | referee comments only (§7) |

**Risa's measured spread (Slack, 2026-10-31 14:47, relayed by the owner):** one observed wavelength grid for the
whole 10k; fitted pixels 4,159–4,751 (within 1%); 26–36 line columns; four band sets (DECaLS / BASS+MzLS, each
with and without W1). One padding bucket covers everything; the model grid is shared and only the LSF, z, SFH
edges, line columns and pixel masks vary. Her likelihood needs beyond #3115–#3117: (a) a banded covariance
(correlated pixels from resampling), (b) the fiber-photometry prior on the flux scale, (c) per-camera scales +
Chebyshev tilt with the Ca K core masked; she samples with MEADS (ChEES is the closest catalog sampler today).
She will time the batched path on one bucket with lines off this week. Design consequence: the four band sets are
one template with the union of both filter systems plus W1 and the per-galaxy `phot_presence` mask (one compile);
`quantum=256` or `n_max=4751` directly for the pixels; the Ca K core is a `pix_mask` entry.

### 1b. Inhomogeneities in the references (removed, not duplicated)

1. `CalibrationELineMarginalizedLikelihood` is a sequential plug-in (#2354); under `C = 1 + Σ c_j T_j` the joint
   model `d ≈ m + [m·T | G]·[c; a]` is linear, so one exact Gaussian marginal exists.
2. `marginalize_calibration` anchors on `wavelength[0]`/`[-1]`, the fitted path on `min`/`max`; both count every
   pixel in `N` and `Σ ln σ` (no mask).
3. `marginalize_emission_lines` counts the prior penalty twice (`r̃ᵀr̃ − âᵀ(GᵀN⁻¹G)â − âᵀΛ⁻¹â`; the marginal has
   `r̃ᵀr̃ − âᵀAâ`) and whitens `G` by σ, which overflows float32 on F_nu-scale data.
4. Pixel covariance: dense on the non-batched path, refused on the batched path, banded in her code.

## 2. Design: one linear-Gaussian marginal, both paths

**New `src/tengri/observation/linear_marginal.py`** (Tier 2 docstrings; `.. math::`; JIT/grad/vmap notes):

```
marginalize_linear_nuisance(data, model0, design, *, sigma=None, cov=None, pix_mask,
                            prior_mean=None, prior_precision=None, col_mask=None, scale=None)
   -> MarginalResult(log_like, coef_hat, coef_cov)
```
- `d = m0 + X β + ε`, `ε ~ N(0, N)`; `N` diagonal (`sigma`) or a banded `BandedMatrix` covariance (`cov`, PR-A,
  whitened by a banded Cholesky factor computed in a `lax.scan`).
- `pix_mask` multiplies the residual and every row of `X`; `N_eff = Σ mask`, `Σ_mask ln σ`: masked pixels
  contribute exactly nothing, so padding is exact.
- `col_mask`: masked columns zeroed; the normal matrix reduced exactly by the identity-block substitution
  `M' = D M D + (I − D)`, `b' = D b`, `D = diag(col_mask)`; log-det and prior normalization count unmasked columns
  only. Equals deleting the column; no floor, clamp or jitter (`check_zero_hiding_clamps`, `check_numeric_guards`
  do not fire).
- `β ~ N(μ, Λ)` with `prior_precision` a full `(k,k)` matrix (diagonal usual; rank-1 `w wᵀ/σ_q²` for PR-B; zero for
  flat; the +k ln a Jacobian of a scale-free prior counts only columns proportional to the model).
- `scale` (traced per galaxy): the solve runs in units of `scale`, log terms restored analytically, so float32 never
  forms `(G/σ)²` at F_nu scale. Identical value in exact arithmetic.
- `log_like = −½ N_eff ln 2π − Σ_mask ln σ − ½ r̃ᵀr̃ + ½ b'ᵀ M'⁻¹ b' − ½ ln det M' + ½ ln det Λ⁻¹_eff − ½ μᵀΛ⁻¹μ`
  (`jnp.linalg.solve` + `slogdet`, no explicit inverse).

**Adapters, all thin, all on both paths:** `calibration_basis(wave, lo, hi, order, segments=None)` is the one basis
builder (fitted path and marginal; `(lo, hi)` = real grid min/max = `cal_lo`/`cal_hi`); `marginalize_calibration`
and `marginalize_emission_lines` become wrappers (exact formulas); the three `marginalized.py` classes delegate, the
combined one as the exact joint; `build_eline_design_matrix` gains `line_mask=None`; the line-profile scalar R is one
rule on both paths (`_eline_scalar_resolution`, moved to `observation/eline_marginalization.py`: scalar R, else the
median of the per-pixel array, else 2000), carried on the batched path as a per-galaxy leaf `eline_resolution`
computed at build time by the same function.

## 3. The PR ladder (one branch and PR per issue, each off `origin/main`; `full-tests` + area labels on every PR)

### PR-3115 `feat/3115-batched-elines` — `eline_mode="fitted"` on the batched path [parallel with PR-3116]
`observation/batched.py`: leaves `line_mask (B, n_lines)`, `eline_resolution (B,)`; `SpectroBatchSpec.n_lines` from
the template's `effective_catalog` when `eline_mode != "off"`, else 0; `line_mask[i] = 1 iff λ_i(1+z) ∈
[wave_real.min(), wave_real.max()]`. `forward/batched_spectrum.py`: `batched_eline_design(model_consts, obs,
params)` → `build_eline_design_matrix(obs.wave, λ, obs.eline_resolution, obs.z, σ, Δv, line_mask=obs.line_mask)` ×
`pix_mask[:, None]`, then `apply_doublet_constraints`; fitted mode adds `G @ amps` after calibration (mirrors
`ELineFittedLikelihood`); refusal lifted for `"fitted"` only; `"marginalized"`, `eline_broad=True`,
`eline_prior_type="cloudy"` keep raising with messages naming the alternative. `inference/batched_spectro.py`:
`_latent_spec(model)` = `model.spec.merge_observation_params(**amp_priors)` (`eline_amp_<name>` ~
`Uniform(±10·eline_prior_sigma)`, as `Fitter._init_eline_arrays`), used by every latent-space helper; amplitude keys
stripped before `predict_state`.
Tests (`tests/contract/test_batched_elines.py`, `contract`, SPDX): single galaxy == `ELineFittedLikelihood` value
and gradient (rtol 1e-10); all-lines-outside == `eline_mode="off"` bit for bit; **one compile per spec with different
masks** (`jax._src.test_util.count_jit_compilation_cache_miss`, 0 misses on the second batch); padding round-off only
(extend `test_batched_padding_exactness.py`); float32 gradient finite AND non-zero; refusals kept
(`test_batched_spectro_guards.py`). Slow tier: 3 matched mocks, batched NUTS vs `Fitter` NUTS medians within 0.05
widths. Also verify and **file (not fix)** the log-uniform-grid bug (`_resolution_for` passes `resolution_pp[0]`).

### PR-3116 `feat/3116-batched-cal-marginal` — the primitive + calibration marginalization, both paths [parallel]
New `observation/linear_marginal.py`; `calibration.py` (basis builder, wrapper, `pix_mask`, `lo/hi`);
`marginalized.py` (calibration class delegates); `batched_spectro.py` options **`calibration_marginalize=False`,
`cal_n_poly=3`, `cal_prior_sigma=1.0`** on `batched_log_likelihood`, `batched_neg_log_posterior`,
`fit_spectra_map_vmap`, `fit_spectra_catalog_mcmc` (the Fitter's names); `ConfigError` with `calibration_order > 0`
(wording of `inference/likelihood.py:223`); CHANGELOG.
Tests: primitive vs the old closed form (rtol 1e-12, monotonic grids); masked-pixel exactness; single galaxy ==
`CalibrationMarginalizedLikelihood` to round-off; padding; one compile; float32 finite non-zero gradient; matched
posteriors (slow); existing calibration tests unchanged and green.

### PR-3117 `feat/3117-batched-eline-marginal` — line marginalization, exact joint, both paths [after 3115 + 3116]
`eline_marginalization.py` (wrapper, exact formula, `line_mask`); `marginalized.py` (`ELineMarginalizedLikelihood`,
exact joint `CalibrationELineMarginalizedLikelihood`, flat prior `Λ = eline_prior_sigma²`); `batched_spectro.py`
(`eline_mode="marginalized"`: `X = G` or `[m·basis | G]`); refusal lifted for `"marginalized"` + `"flat"`; #2354
closed; the two plug-in tests (`test_likelihood_full_cohort.py:313-377`, `test_calibration_eline_plugin_disclosed.py`)
rewritten to the exact statement.
Tests: single galaxy == non-batched class, lines-only and joint; masking a line == removing its column (flat prior and
Λ⁻¹ = 0); the joint equals a nested integration done the long way on a 3-line toy; every line masked stays finite;
one compile; padding; float32; matched posteriors (slow).

### PR-F `feat/alpha-4d-cue-slab-and-contraction` — carry Risa's two commits [independent; first]
Cherry-pick 999c29e1f and 87c8bdc64 from `rhw/tengri` onto main (co-author trailer `Risa Wechsler`), resolve against
the current `_spectrum_lut_families` block, run her `tests/contract/test_alpha_fe_4d_luts.py` plus
`tests/components/sps/test_alpha_fe.py` and the whole contract tree. Labels `area:stellar`, `area:nebular`.

### PR-A `feat/banded-covariance-both-paths` (new issue) [after 3116]
`Spectroscopy(covariance=BandedMatrix | dense)`, `SpectroBatch.cov_data (B, K_c, n_max)` + `SpectroBatchSpec.
cov_offsets`; banded Cholesky (`lax.scan`) and whitening inside the primitive; `MultivariateGaussianLikelihood` and
the batched Gaussian term through one `banded_whiten`; masked pixels zero rows and columns exactly (identity-block).
Tests: banded == dense on a toy; padding exact; one compile; single-galaxy parity.

### PR-B `feat/calibration-segments-scale-prior` (new issue; **names need the owner's OK**) [after 3116]
Proposed: `Spectroscopy(calibration_segments=...)` → per-segment scale columns; `cal_prior="gaussian" |
"scale_free"` (columns normalized by the model's masked mean flux; k counts those columns only);
`cal_scale_prior=(log10_center, sigma_dex)` per galaxy (a leaf on the batched path): a log-normal on the mean
multiplier over a chosen pixel window, carried exactly as one extra latent `ln q` with the rest analytic. Both paths.

### PR-G `feat/eline-prior-from-model` (new issue) [after 3117]
`eline_prior_type="model"`: Gaussian prior on each line amplitude centered on the model's own line flux
(`state.derived["line_lums"]` → observed F_nu·Å through the same projection as the spectrum, dust included), width
`eline_prior_width_dex`, a separate line aperture factor (free or fixed); masked lines exact via `col_mask`. Both paths.

### PR-C `feat/catalog-engine-ghmc-meads` (new issue) [independent]
`"ghmc"` in `_SAMPLERS` driving `blackjax.meads_adaptation` + `ghmc` over `n_chains` per galaxy; distinct start per
superchain; adaptive warmup stop (ensemble step size and damping stationary; min 500, max 2000); wall hits rejected
with finite gradients; `analysis/diagnostics.nested_rhat` rank-normalized and folded + tail ESS. Conformance test in
`tests/inference/` (slow). The `broken` tier relabel is the owner's call; the PR body cites her 40/40.

### PR-D `feat/balmer-series-completion` (new issue, physics; **owner go/no-go**) [independent]
Nebular: Balmer lines above Cue's last line at Case-B ratios to Cue's Hβ (pyNeb `RecAtom('H',1)`, Storey & Hummer
1995, at the backend's T_e/n_e; cited), through the existing line-profile path so dust and the kernel split see them.
`LineList.default_optical()`: H8–H14, He I 3889, [Ne III] 3968 with vacuum wavelengths verified against NIST /
FastSpecFit's emlines table, with the fixed-ratio group option. Tests: ratios vs pyNeb; flux conservation; a Cue
spectrum below 3800 Å changes only at the new lines.

### PR-I `feat/mw-foreground-f99-projection-path` (new issue) [independent]
Fitzpatrick (1999) R_V = 3.1 law in `components/dust/attenuation.py`; `foreground=` applied on the observed-frame
projection path (spectrum and photometry) rather than `obs_sed` only; per-galaxy `ebmv_mw` leaf on the batched path
(`GalaxySpectrum.ebmv_mw`); docstring notes the SF11 recalibration. Tests: F99 curve vs the paper's tabulated
values; photometry and spectrum reddened consistently; batched == per-galaxy.

### PR-E `perf/photometry-grid-sort-hoist` (optional) [independent]
Hoist the merged-grid sort to build time; bit-identical photometry test; compile-time measurement in the PR body.

### PR-3118 `docs/3118-batched-spectroscopy-page` [last]
`docs/batched_spectroscopy.md` + toctree entry under **Fitting workflows** (next to `spine/11_catalog_fits`); one
executable `SEDModel.build(...)` listing (the `check_build_listings` guard runs it on the tracked wNE grid),
`<!-- no-exec -->` on fit fences. When to use; the workflow (`GalaxySpectrum` → `build_spectro_batches(quantum=)` →
`fit_spectra_map_vmap` / `fit_spectra_catalog_mcmc`); traced vs template-fixed; choosing `quantum`; **measured**
numbers from a Haiku measurement lane (per-galaxy `Fitter` build + compile + sample vs batched one-compile on 3 and 12
mocks, Mac CPU; script and raw output in job tmp, quoted in the PR); what shipped; what it does not support stated as
what to do, with links only to open issues; no developmental language.

## 4. Gates before each handoff (Haiku gate lanes; the orchestrator reads exit codes, not prose)

1. Every `run:` step of the `lint` job (PR-template one-liner) plus the `smoke` job's `tools/check_*.py` steps.
2. `tests/contract/` one file per process (591 files, background script, `FILES_WITH_FAILURES`), `tests/physics/`,
   every `tests/**/test_batched_*.py`, the marginalization tests (`inference/test_eline_marginalization.py`,
   `inference/test_likelihood_full_cohort.py`, `inference/test_eline_fitting.py`,
   `physics/bounds/test_marginalize_calibration.py`, `regression/bug/test_calibration_*.py`,
   `contract/test_calibration_contract.py`, `contract/test_eline_resolution.py`), and for every lifted refusal each
   test constructing through it (`grep -rln "eline_mode" tests notebooks examples bench docs`). Any red is re-run on
   a clean `origin/main` export and reported as member / pre-existing.
3. Docstring standard checked on every new public function; one mutant per new test file.
4. Handoff to the orchestrator session: PR number, head SHA, labels, gate results by name with counts, the lint run,
   squash subject + body, the #3112 ordering note. Then stand down.

## 5. Owner rulings (2026-10-31, one prompt in this session; relayed to the merge orchestrator)

1. **Exact joint marginalization replaces the plug-in on the non-batched path too** (closes #2354; one primitive). YES.
2. **Batched option names follow the Fitter** (`calibration_marginalize`, `cal_n_poly`, `cal_prior_sigma`). YES (plan approved).
3. **PR-D Balmer-series completion:** GO, both halves (model lines above Cue's H10 and the catalog lines).
4. **PR-I MW foreground (F99) on the projection path:** GO.
5. **PR-B with the proposed names** (`calibration_segments`, `cal_prior="gaussian"|"scale_free"`, `cal_scale_prior=(log10_center, sigma_dex)` as a log-normal on ln q): GO.
6. **`mcmc_ghmc` tier:** relabel from `broken` only after PR-C's conformance test passes on the repo's own mocks.
7. **Young-fraction (stick-breaking) coordinates for the continuity SFH:** GO as an implementation (new PR-H), not an issue only.
8. **PR-E filter-sort hoist:** GO if the diff stays small.
Order after the four issues: PR-F (running), PR-A, PR-C, PR-G, PR-D, PR-I, PR-B, PR-H, PR-E; PR-J only if wanted.

**Tracking issues filed 2026-10-31:** PR-F #3124 · PR-A #3133 · PR-B #3134 · PR-C #3135 · PR-D #3136 · PR-E #3137 ·
PR-G #3138 · PR-H #3139 · PR-I #3140 · log-uniform-R bug #3145 (filed from the #3115 probe, not fixed). Claims
posted on #3115–#3118. Branch `feat/3115-batched-elines` carries lane A's WIP (366a1a212); a Sonnet lane finishes it. Merge orchestrator: #3112 lands via verdict
batch #3122 first; the #3115 parity lane waits for its merge SHA.

## 6. Verification of the whole

Each PR green on its gates (§4), slow-tier matched-galaxy tests run locally once. One end-to-end script (job tmp)
fits 12 DESI-like mocks (three pixel counts, `quantum=128`, banded resolution, lines marginalized jointly with the
calibration, banded covariance once PR-A lands) with `fit_spectra_catalog_mcmc` and with the per-galaxy `Fitter`:
compile count (one per bucket), wall time per galaxy both ways, posterior-median differences ≤ 0.05 widths. Its
numbers are what #3118 quotes. Final report: PR numbers/heads, gate counts, new issues filed (A–J + the log-uniform-R
bug if confirmed), the comments for Risa (§7).

## 8. Where the record of what was fixed lives (binding; each PR carries its own part)

Future agents find things in this order: CLAUDE.md → `docs/dev/` → CHANGELOG → issues/PRs → docstrings. Each PR
updates the layers it touches; the orchestrator refuses a handoff that leaves one out.

1. **`CLAUDE.md`** (read by every session): one new entry under "Critical gotchas", written once in PR-3116 and
   extended by 3117/A/B: *"One linear-Gaussian marginal on both paths"* — `observation/linear_marginal.py` is the
   only marginalization math; the three `marginalized.py` classes and `inference/batched_spectro.py` are adapters;
   never add batched-only math; the refusal table is `check_template_model` and every entry names the alternative;
   the exact joint replaced the plug-in (#2354); option names follow the Fitter. Paths in code markup must exist
   (`tools/check_claude_md_paths.py`). The "Package structure" line for `observation/` gains `linear_marginal`.
2. **`docs/dev/batched-spectroscopy.md`** (new contributor narrative, excluded from the published site): the leaf
   table of `SpectroBatch` with shapes and how each is built, the `SpectroBatchSpec` key fields, the refusal table
   with the reason and the issue or PR that lifted each entry, the one-primitive design (§2 of this plan, with the
   identity-block reduction and the scale trick), the homogeneity rule, and the per-PR "what changed and why" in
   dated bullets. `docs/dev/where-things-live.md` gains the new modules. PR-3116 creates it; every later PR appends.
3. **`docs/internal/plans/2026-10-31-batched-desi-dwarfs.md`**: this plan, committed verbatim in PR-3116 (the
   project's home for agent-authored plans per CLAUDE.md), updated at the end with the owner's rulings from §5.
4. **`CHANGELOG.md`**: one bullet per PR in the repo's `<type>(area): ... (#PR)` form; the grouped-dust bullet
   that cites #2879 is corrected to #3024 in PR-F since that PR touches the same physics.
5. **Issues and PRs**: each of #3115–#3118 is closed by its PR with `Closes #N`; the new issues A–J are filed before
   their lanes launch, each linking this plan file's path, and closed by their PRs; #2354 closed by PR-3117; the
   log-uniform-R bug filed from PR-3115 with the reproducer. Every PR body has a "Record" line naming the
   CLAUDE.md entry, the docs/dev section and the CHANGELOG bullet it added.
6. **Docstrings** are the API reference (`tools/check_doc_examples.py`): the module docstrings of
   `observation/batched.py`, `forward/batched_spectrum.py` and `inference/batched_spectro.py` state what is
   supported and what still raises, dated; `check_template_model` lists every refusal with its reason.
7. **The published page** (#3118) states only what shipped and what to do instead; no issue numbers except links
   to open work.
8. **Memory** (this account): one `project` memory `tengri-batched-stack-2026-10-31.md` with the owner's rulings
   (§5), the PR numbers, and the location of the Risa comments, linked from MEMORY.md; written after the rulings.

## 7. Comments for Risa (consolidated from the three review lanes; arithmetic re-checked by the orchestrator)

**Method (must-fix):** (1) with line columns added, the `+k ln a` Jacobian must count only the five model-scaled
calibration columns, not 5 + n_L — please confirm in the source and say so on the page; (2) the sampled q prior is
Gaussian in q (7.4% of its mass at q < 0, +7 nats at +0.6 dex against the intended 0.3-dex log-normal used at the MAP):
sample ln q or integrate it by 1-D quadrature; (3) the age calibration failure is a +0.8 sd *bias* (Table 5), which
is exactly the 52% coverage: split same-family vs misspecified mocks, run ≥100 same-family mocks, and do not publish
ages or 0–30 Myr fractions with current intervals; (4) a banded P is not the inverse of a banded C: state whether the
161-diagonal matrix is a truncated inverse and compare with an exact banded Cholesky on a few galaxies.
**Method (should):** σ_gas is degenerate with the LSF (a 5% LSF error is 15–20% of σ_gas) and the mocks tested 100
km/s, not 30; free either-sign columns at FWHM 2.5–4 Å can act as partial masks on the stellar Balmer cores (refit
with non-negative amplitudes; report the negative fraction); four of eight superchains share the MAP start; nested
R-hat on means with ~3 effective draws per chain has a ~0.1 sd floor (add rank-normalized/folded R-hat and tail ESS);
the GPU-vs-CPU 0.10 sd median agreement is what Monte Carlo noise predicts at ESS 400; "1% of prior mass" is per
ratio (5.8% over six); wall hits may feed NaN gradients into the step-size adaptation (the frozen ensembles); apply
the SF11 0.86 recalibration of SFD consistently to the model reddening and the photometry.
**Science (must-fix):** the PPC shows the −0.36 dex shift is expected *if* the joint solution is true, not that it
is; a joint [Fe/H] bias of order 0.1 dex is not excluded; "shrinkage toward the prior" is the joint (Z, τ_diff)
prior-volume effect (a pull toward the Z prior median predicts −0.04 dex and the photometry-only medians sit below
it); remeasure the shift and the calibration in the production configuration and tabulate which configuration each
number comes from; "masses agree to better than 0.1 dex" holds for the median only (NMAD 0.12, 7.8% outliers with
COG/MSTAR_MASKBIT galaxies over-represented); the FastSpecFit "−0.06 dex if h = 1" is +0.28 with +0.34 applied, not
evidence, and would put FastSpecFit above the fiducial mass; the Kirby comparison should stay in [Fe/H], state the
weighting, and report slope/intercept/scatter forward-modeled through the same pipeline.
**Science (should):** the two-fiber test bounds an aperture effect at ≲0.1 dex rather than ruling it out (f_fib never
approaches 1; the tilt absorbs color gradients); Probe C does not "identify" the broad term; resolve
"two-thirds of the pull below 3780 Å" vs "cutting it removes a third of the shift"; one sign convention for the shift
across pages; one S/N definition per cut.
**What she can take from upstream today:** redshift-set continuity edges and per-galaxy edges on the batched stack,
truncated `StudentT(lo=-3, hi=3)` with a smooth standardized bijection (may remove the wall), grouped dust and
`transition_width_dex`, `lyc_max_age_yr=1e8`, the DESI reader / R-matrix / library deconvolution, sky and line masks,
`apply_zp_floor`, `vmap_chunked`, the frozen-chain health check, and the batched template.
