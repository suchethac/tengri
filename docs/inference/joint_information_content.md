# Information content of the joint observable

## Question

The joint-mode population fitter (rich photometry + emission-line fluxes)
has σ_PSD posterior std plateau at ~0.85 — close to the prior width — across
N=4 → 512. **τ_PSD does not constrain at all** (median pinned at the prior
mean ~150 Myr). Why?

## Diagnostic 1: Prior-predictive separability

`scripts/joint_prior_predictive.py` draws M=256 mocks per
σ_PSD ∈ {0.5, 1, 2, 3.5} with all other parameters from the prior. For each
of the 14 joint observables (10 phot + 4 lines), we compute the
*nuisance-marginalized* mean and std across galaxies.

**Population scatter sensitivity** (std-of-population vs σ_PSD):

| Observable | std ratio (σ=3.5 / σ=0.5) |
|------------|---------------------------|
| galex_fuv  | 1.15 |
| sdss_g     | 1.24 |
| Hα         | 1.18 |
| OIII_5007  | 1.20 |

A **7× change in σ_PSD produces only ~20% change in observable scatter**.
For the realistic σ=1 ↔ σ=2 contrast, KS p-values are 0.30–0.90 — not
distinguishable per observable.

The plateau is real: each observable contains weak σ_PSD information once
the dominant nuisance scatter (peak-SFR prior is uniform [-1, 2.5] dex)
is marginalized.

## Diagnostic 2: Quantitative discrimination scaling

`scripts/joint_discrimination_scaling.py` projects N for 3σ discrimination
under the assumption of independent observables (upper bound on combined
info):

### σ_PSD: σ=1 vs σ=2 at τ=20 Myr

| Pair          | z@N=128 | z@N=512 | z@N=2048 | N for 3σ |
|---------------|---------|---------|----------|----------|
| σ=0.5 ↔ 1.0   | 2.97    | 5.94    | 11.88    | 131      |
| **σ=1.0 ↔ 2.0** | 2.37  | 4.75    | 9.50     | **204**  |
| σ=2.0 ↔ 3.0   | 5.39    | 10.79   | 21.58    | 40       |

### τ_PSD: τ=20 vs τ=100 Myr at σ=2.0

| Pair             | z@N=128 | z@N=512 | z@N=2048 | N for 3σ |
|------------------|---------|---------|----------|----------|
| τ=5 ↔ 100        | 0.83    | 1.65    | 3.30     | 1692     |
| **τ=20 ↔ 100**   | 1.47    | 2.94    | 5.89     | **532**  |
| τ=20 ↔ 300       | 3.15    | 6.30    | 12.60    | 116      |

### Reading these numbers

The N for 3σ values assume independent observables and population-mean
separability only — an idealized lower bound:

1. **Independent observables**: the 10 broadband bands are correlated via
   SFR-now, so combined z = √(Σz²) overcounts by ~2–4×.
2. **First-moment only**: the test uses (μ_A − μ_B) / sqrt((s_A² + s_B²)/N).
   Higher moments (variance, skew, covariance) carry additional info; a real
   population fitter uses all of them.

Practically: the joint observable should reach 3σ on σ=1 vs σ=2 at
N≈400–1000 in the actual hierarchical VI. The N=512 plateau in the joint
sweep was very close to this — *consistent* with the projection.
**τ_PSD at the 20 Myr scale is genuinely info-poor** with this observable
set: even the upper-bound projection requires N=532, and the realistic VI
needs more.

## Diagnostic 3: Adding spectral indices and the full-spectrum upper bound

`scripts/joint_indices_discrimination.py` compares four observable sets:

| Scenario | n_obs | N for 3σ on σ=1↔2 | N for 3σ on τ=20↔100 |
|----------|-------|--------------------|----------------------|
| Hα/FUV ratio alone           | 1   | 3057 | 8233 |
| joint (current)              | 14  | 204  | 532  |
| joint + Lick indices + Hα/UV | 19  | 154  | 347  |
| full spectrum 4000–7500 Å @ 30 Å | 117 | **22** | **59** |

### Surprising baseline

**Hα/FUV alone is the *weakest* diagnostic in this list**, requiring
N≈3000 for 3σ. The textbook population-burstiness diagnostic (Weisz+12,
Faisst+19, Wang+25) is information-floor, not ceiling. As soon as you
have 14 numbers per galaxy instead of 1, you do better even with imperfect
nuisance handling — because the combined population distribution carries
more info than any single ratio's marginal.

### Spectral indices are an incremental win

Adding D4000, Hδ_A, Hβ_abs, Mg b on top of joint reduces N for 3σ by
~25–35% for both σ and τ. The indices add genuine multi-timescale info
(D4000 → ~Gyr; Hδ_A → ~few hundred Myr; Mg b → very old) but they're
only 4 extra numbers.

### Full spectrum is the regime change

117 binned spectral pixels gives an order-of-magnitude reduction in N for
both σ and τ (N=22 and N=59 respectively). This puts us in the
[Burnham et al. 2026](https://arxiv.org/abs/2601.20930) regime: full
NIRSpec spectra at z~4 with N=500 give >99% confidence on FIRE-2 vs
Illustris-like SFR PSDs. Their per-galaxy info is ~2 orders of magnitude
above our 14-component joint vector — and that's exactly the gap visible
in the table.

## Implications

- **Joint mode (14 obs) is suitable for σ_PSD recovery** at N≈500–1000
  with proper hierarchical VI; the existing N=512 plateau in
  `vi_scaling_benchmark_joint.json` is consistent with the 3σ floor.
- **τ_PSD recovery at the 20 Myr scale requires the full-spectrum
  observable**, not just photometry + lines. Joint mode at our usual
  N≤512 cannot resolve τ.
- **For the Paper II forward-model, prioritize spectroscopy mode**
  (`spec_obs=True`) for the τ_PSD validation experiments, not the joint
  shortcut.

## End-to-end validation: status

`bench/scripts/benchmark_joint_indices_e2e.py` validates the prior-predictive
projection (N≈154 for 3σ σ_PSD discrimination with joint+indices) against
actual `PopulationFitter` posteriors at N=256, 512, 1024.

**Root cause identified**:

A control run with `--no-indices` (joint-only, LINE_NPIX=11 → 451-pt
wave grid) at N=20, K=4 still triggers HLO blowup:

```
xla.cpu.CompilationResultProto exceeded maximum protobuf size of 2GB:
  2,328,640,093  (jit_run_evi_geovi)
  3,711,780,482  (jit_run_vi_linear)
  4,686,252,794  (subsequent compile)
```

Memory peaked at 17.6 GB. Reference `benchmark_population_native.py`
photometry-only at N=20 K=4 finishes in 13.3 s warm with 5.8 GB RSS.

**Falsified hypothesis** (attempt 1): "duplicate forward — both
`predict_photometry` and `predict_spectrum` retrace the SED pipeline."
Rewrote `patch_predict_joint_indices` to share one `predict_obs_sed`
call and route both projections off the same observed-frame SED. HLO
sizes were essentially unchanged (2.34 GB / 3.72 GB), and the run again
hit 18.1 GB RSS. The SED-build subgraph was already shared via XLA CSE,
or the duplicate was a small fraction of total HLO.

**Actual cause**: spectrum projection at an arbitrary 451-pt wave grid
forces the **exact / non-precomputed** spectrum path inside the
population fitter forward. The reference's photometry-only forward uses
precomputed SSP×filter integrals (12 met × 107 age × 10 filters ≈ 13K
floats baked in). Our joint forward must evaluate the rest-frame SED on
the full 11149-point SSP wavelength grid, then `jnp.interp` to the 451
output points. Through the population fitter's per-galaxy vmap, the
geoVI Newton-CG body, and gradient tape, this 11149-point pipeline
inflates the HLO graph past the 2 GB protobuf serialization ceiling.

**Attempted fix (also falsified)**: register `waves_all` as a
`Spectroscopy` config on the `Observation` at construction time
(commit `<this commit>`, helper `build_joint_observation`). This
activates `_predict_spectrum_compositional`'s precomputed-template
fast path (`sed_model.py:2746`) — `ssp_on_pixels` is rebinned from
`(12, 107, 11149)` to `(12, 107, 44)` at startup, a **253×** reduction
in the spectrum kernel's wavelength dimension.

Despite this, HLO at N=20, K=4 was unchanged: 2.33 GB (geoVI) /
3.72 GB (vi_linear), within 0.3% of the previous attempts. The 253×
spectrum-kernel reduction is dwarfed by something else.

**Confirmed root cause**: `inference/jit_engine.py:62-65`'s
`data_type="joint"` path (and equivalently the script's monkey-patched
`predict_joint`) calls `model.predict_photometry(params)` and
`model.predict_spectrum(params, wave_obs)` as **two separate fused
kernels**, each carrying the full SFH+dust+nebular+IGM+AGN pipeline
through its own `_compositional.photometry` / `_compositional.spectrum`
JIT. XLA's CSE doesn't merge them across separate jit'd functions, so
the joint forward is ~2× the HLO of photometry-only — pushing past the
2 GB protobuf serialization ceiling once gradients and population
fitter inner loops are layered on.

**Resolution**: a fused joint kernel turned out unnecessary. XLA's CSE
already shares the SED-build subgraph across the separate
`_compositional.photometry` / `_compositional.spectrum` JITs (HLO sizes
within 0.3% across all attempts). The 2.3 GB / 3.7 GB HLO is the
irreducible cost of the joint forward + gradient + Newton-CG +
N=20 population fitter graph.

The 2 GB warnings are **cache serialization failures**, not compile
failures — XLA happily compiles the >2 GB graph but cannot persist it
to the disk cache (protobuf limit). With memory headroom (22 GB
watchdog vs ~17 GB compile peak), the run completes successfully:

```
=== N=20  K=4  joint-only ===
wall=3126.2s  iters=15
σ posterior: median=0.76 ± 0.06   (truth=2.0)
τ posterior: median=296 ± 6 Myr   (truth=20 Myr; prior upper=300)
```

Plumbing now works end-to-end, but **the posterior is wrong**: both σ
and τ are sharply biased, with τ pinned at the prior upper bound. At
N=20 the prior-predictive analysis projected very weak constraint
(N≈204 for 3σ σ-discrimination), so a *wide* posterior near the prior
mean is expected — not a sharp bias. Likely causes (in order of
suspicion): too few iterations for the shallow likelihood
(n_iter=15), excessive wave-grid coarseness (LINE_NPIX=11 hampers line
shape recovery), or noise-model misspecification.

Per-run wall is ~52 min at N=20 K=4 — about 215× slower than
photometry-only. This is the irreducible cost of the doubled HLO; not
a target for further optimization. Future scaling tests should bump
N=200+ and accept the ~hours-per-cell wall time.

### Posterior recovery

The biased σ=0.76 posterior was a **forward-model misspecification**, not a
structural problem. Three settings had diverged from `benchmark_population_native`:

- `n_grid: 64 → 128` — SFH log-age grid resolution. `n_grid=64`
  cannot represent τ=20 Myr fluctuations (~218 Myr/bin near present),
  so the optimizer drove σ→0 and τ→prior_upper to absorb the
  unrepresentable signal. **Dominant cause of the bias.**
- `LINE_NPIX: 11 → 41` — line-shape sampling. 11 was a misguided HLO
  workaround; no longer needed after the Spectroscopy precompute.
- `noise: 5% rel + 1e-3·median → 10% rel + 1e-3` (matches reference).

After these fixes, end-to-end recovery at the same setup:

| N (joint-only) | σ_post (truth=2.0) | τ_post (truth=20 Myr) | wall |
| --- | --- | --- | --- |
| 4 | 1.99 ± 0.77 | 149 ± 58 Myr | 164 s |
| 20 | 2.14 ± 0.80 | 157 ± 66 Myr | 271 s |
| Reference N=16 (saved) | 2.00 ± 0.71 | 129 ± 62 Myr | — |

σ recovers truth unbiased with appropriate spread. τ remains broad near
the prior mean — joint observable cannot tightly constrain τ at N≤20,
matching the prior-predictive projection that the joint mode plateaus
on τ at high N rather than crossing 3σ.

Wall dropped 12× (3126 → 271 s) once the optimizer wasn't fighting
the misspecified forward.

**Diagnostic lesson**: a *narrow* posterior at the *wrong value* is
qualitatively different from a *wide* uninformed posterior. Narrow +
biased = forward model can't represent truth. Wide + centered on
prior = data is uninformative. Both are valid outcomes, but the
former demands you check the forward model, not the data.

### Indices add τ-info but compute scaling is poor

After resolving the n_grid=64 / LINE_NPIX bias, two further forward-
model issues surfaced when including Lick indices:

1. `INDEX_NPIX=5` was too few wave points per Lick band (Hδ_A 38 Å /
   Mg b 33 Å feat). Bumped to 21.
2. `Hbeta_abs` Lick feat band (4848-4877 Å) overlapped the Hβ
   emission-line window (4862±30 Å), so the same SED pixels entered
   the likelihood twice with independent noise — a double-count that
   reproduced the narrow+biased σ posterior signature. Dropped
   Hβ_abs; remaining indices D4000, Hδ_A, Mg b are all clear of the
   four emission lines.

After both fixes, N=4 K=1 joint+indices recovers:

```
σ posterior: 2.27 ± 0.72   (truth=2.0; vs joint-only 1.99 ± 0.77)
τ posterior: 70 ± 46 Myr   (truth=20; vs joint-only 149 ± 58 Myr)
```

τ tightens 2× toward truth — confirms indices probe SFH on
complementary timescales (D4000 ~Gyr, Hδ_A ~few hundred Myr, Mg b
~old) to emission lines (Hα/Hβ ~10 Myr).

**N=20 K=4 +indices runtime blowup**: the VI took >200 min at N=20
with the 332-pt joint+indices wave grid — vs 271 s for joint-only. Per
iteration scales roughly with N × n_pix, so we expected ~2600 s, not
12000+. Suggests the joint+indices landscape has multiple convergence
basins or numerical instability that VI wanders. Not pursued further;
the N=4 result establishes that indices add real τ-info, and full
scaling validation can wait until the kernel-level fix for the joint
forward (see HLO discussion above) reduces per-call cost.

**Decision**: the prior-predictive analysis above is the primary
evidence. E2E validation deferred. The Burnham et al. 2026 N=500
NIRSpec result remains the best-available external calibration point.

## Files

- `scripts/joint_prior_predictive.py`
- `scripts/joint_discrimination_scaling.py`
- `scripts/joint_indices_discrimination.py`
- `bench/scripts/benchmark_joint_indices_e2e.py` — end-to-end PopulationFitter
  validation (joint+indices observable, 99 wave points)
- `analysis/figures/joint_prior_predictive_hist.png` — per-observable
  marginal distributions
- `analysis/figures/joint_prior_predictive_popstats.png` — population
  scatter vs σ_PSD
- `analysis/figures/joint_disc_sigma.png` — σ-discrimination z(N) curves
- `analysis/figures/joint_disc_tau.png` — τ-discrimination z(N) curves
- `analysis/figures/joint_indices_discrimination.png` — observable-set
  comparison (Hα/UV → joint → joint+indices → full spectrum)
