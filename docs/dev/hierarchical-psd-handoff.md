# Hierarchical SFH-PSD recovery — handoff

**Branch:** `worktree-hierarchical-psd-spec` · **PR:** #1479
**Status:** the estimator is correct — **now demonstrated against an exact
analytic posterior** (§4b): given true per-galaxy draws it recovers σ and τ and
shows no railing out to N=512. End-to-end it still has a **ceiling on N around
32–64**, above which the shared posterior jumps to a grid corner. The cause is a
small per-galaxy bias in `Z_i(σ,τ)` amplified linearly by N, traced to the
mismatch between the true interim posterior and the Laplace draws (§4a).
Everything below is measured, not assumed.

---

## 1. What this is

Recover the shared burstiness hyperparameters `sfh_field_psd_sigma` (σ, dex) and
`sfh_field_psd_tau_myr` (τ, Myr) from a population of galaxies, for the
companion PSD paper. Truth injected at **σ = 0.75, τ = 150 Myr**, z = 0.1,
10 broadband filters (GALEX + SDSS + 2MASS), `n_grid = 16`, D = 26 per galaxy.

**Two-step estimator (approach B).** Galaxies are conditionally independent
given (σ, τ), so the shared block is 2-D:

    p(σ, τ | {d}) ∝ p(σ, τ) · Π_i Z_i(σ, τ)

Step 1 fits each galaxy independently under a deliberately wide *interim* prior.
Step 2 reweights those draws onto a 60×60 (σ, τ) quadrature grid. `method="b2"`
is production (importance reweighting with the interim pushforward correction);
`method="b1"` is the marginal-posterior product, kept only as a
differently-wrong cross-check.

---

## 2. Where it stands

### Works

| Claim | Evidence |
|---|---|
| Estimator, kernel, grid, quadrature, bounds are correct | Fed the **true** fields on the production grid: σ = 0.734–0.747, τ = 143–152 Myr at N=256 |
| It works where the weights survive | σ covers truth at N = 4, 8; **τ covers truth at N = 32** (61.9–153.4), where ESS is still 81.5 |
| B2 beats B1 | Measured 4.6× closer to closed form on the toy |
| Estimator is memory-flat in N | Streaming: 1.07 GB at N=256 vs 7.4 GB materialized |
| High N is affordable to *fit* | Laplace 3.2 s/galaxy; a 2048-galaxy bank in ~1.8 h |

### Does not work — the N ceiling

**Definitive run** (`psd_bank_fixed`, N=2048, K=4000 unthinned,
`min_eigenvalue=1.0`, zero galaxies rejected):

| N | σ (truth 0.75) | τ Myr (truth 150) | ESS **min** ⚠ |
|---|---|---|---|
| 4 | 0.692–0.930 OK | 11.9–29.6 | 54.6 |
| 8 | 0.699–0.894 OK | 12.7–30.7 | 89.9 |
| 16 | 0.757–0.944 | 39.2–112.6 | 100.7 |
| 32 | 0.753–0.918 | **61.9–153.4 OK** | 81.5 |
| 64 | 0.958–0.995 | 434.6–491.2 | 5.1 |
| 128 | 0.970–0.996 | 472.1–494.7 | 5.1 |
| 256 | 0.985–0.997 | 473.0–494.9 | 2.9 |
| 512 | 0.986–0.997 | 473.1–494.9 | 1.0 |
| 1024 | 0.986–0.997 | 473.1–494.9 | 1.0 |

σ slope −0.672 ± 0.080 (PASS), τ slope −0.058 ± 0.127 (FAIL).

**The ESS column above is `min` over galaxies and is MISLEADING — do not read
it as a collapse.** A minimum over 1024 galaxies is necessarily below a minimum
over 4. The distribution is healthy throughout:

| N | ESS min | p10 | **median** | p90 | % < 10 | **mode σ** | **mode τ** |
|---|---|---|---|---|---|---|---|
| 4 | 54.6 | 78.5 | 251.4 | 2398 | 0% | 0.866 | 18.2 |
| 8 | 89.9 | 102.1 | 179.2 | 1111 | 0% | 0.799 | 18.2 |
| 16 | 100.7 | 266.4 | 521.0 | 1035 | 0% | 0.866 | 68.4 |
| 32 | 81.5 | 218.5 | 631.1 | 1361 | 0% | 0.832 | 95.3 |
| 64 | 5.1 | 304.6 | **518.3** | 1240 | 2% | **1.000** | **500.0** |
| 128 | 5.1 | 322.2 | **530.3** | 1098 | 1% | **1.000** | **500.0** |
| 256 | 2.9 | 345.3 | **550.4** | 1121 | 1% | **1.000** | **500.0** |
| 512 | 1.0 | 327.5 | **550.2** | 1143 | 1% | **1.000** | **500.0** |

**Median ESS is flat at ~550 from N=16 to N=512**, p10 never below 218, and only
1–2% of galaxies fall under 10 — while `min` falls 100.7 → 1.0 over the same
range. **The weights are fine.** The `min` column is pure order statistic.

**The actual mechanism is in the mode column.** Between N=32 and N=64 the shared
posterior jumps discontinuously from (σ=0.832, τ=95.3) — near the truth
(0.75, 150) — to (σ=1.000, τ=500.0), the extreme corner with both parameters at
their grid maximum. Min-ESS is low at N≥64 *because* ESS is evaluated at the
mode and the mode is now in a corner where a few galaxies have degenerate
weights. **Low ESS is a consequence of the railing, not its cause.**

Since `log p = Σ_i log Z_i`, a small systematic per-galaxy tilt in `Z_i` grows
linearly in N while noise grows as √N. At N=32 noise still dominates and the
answer is roughly right; by N=64 the bias has won. This is "pooling reduces
variance, not bias" playing out directly, and it is **the same root cause as the
τ bias** rather than a separate failure.

Consequence for the estimator design: eliminating the importance weights
(e.g. by closed-form Gaussian marginalization) would **not** fix this. The
weights were never the problem. The per-galaxy `Z_i` is biased and that must be
found first.

Bears on the user's earlier 8192-galaxy MGVI experiment: at that N a per-galaxy
tilt this size is overwhelming, so flat/railed intervals there are this
signature rather than evidence about the physics.

### Superseded claim — do not cite

An earlier partial bank (309 galaxies, K=500 thinned, `min_eigenvalue=1e-6`,
45% of galaxies discarded, N capped at 128) gave σ slope −0.448 ± 0.053 and τ
slope −0.516 ± 0.076, both PASS, and σ covering truth to N=64. **That does not
survive on the clean bank at higher N.** It was measured on a contaminated
sample over too short a lever arm.

---

## 3. Bugs found and fixed (each was a real defect)

1. **`min_eigenvalue=1e-6` is a variance *ceiling*, not a regularizer** (#1515).
   `run_laplace` floors Hessian eigenvalues then takes `cov = H⁻¹`, so the floor
   *assigns* clipped directions variance 1e6 (std 1000). Measured: galaxy 4 had
   11/26 eigenvalues clipped and ξ std 682, against an expected 1. Ten filters
   constrain ~4 modes of a 16-node field, so ~12 field directions are genuinely
   flat — but an unconstrained direction is not infinitely uncertain: ξ has a
   N(0,1) prior, so variance ≤ 1. Flooring at **1.0** takes ξ std for galaxies
   0/3/4/5 from 1.20/108.1/682.2/85.8 → 0.98/0.94/0.96/0.95 and removes the
   **45–56% pathological rate**. Not universal — only valid because these
   latents are unit-normal in unbounded space.

2. **Interim/grid prior mismatch on τ.** Interim fits use `Uniform(10, 500)` on
   `tau_myr` (linear-uniform); `SharedGrid` geomspaced τ with flat weights
   (log-uniform). B2 divides by the interim pushforward, which is a *fact* about
   how draws were made, not a modelling choice — mismatched by a factor ∝ τ,
   50× end-to-end. Fixed via `SharedGrid.uniform(tau_prior=...)`. **Did not cure
   the railing on its own.**

3. **Thinning i.i.d. Laplace draws.** `thin=8` was carried over from the HMC
   path where it removes autocorrelation. Laplace draws are independent, so it
   discarded 87% of the sample for nothing. ESS 9.4/10.3/3.8 → 25.0/92.3/46.3.
   Default is now method-aware.

4. **Wrong convergence diagnostic.** The estimator consumes the reconstructed
   field `m = L(σ,τ)·ξ`, not ξ. Chains agreeing on ξ (R̂ 0.994–0.998) while
   disagreeing on σ (R̂ 4.4) **disagree on m**. `Posterior.rhat` excludes
   `psd_xi` by default, so this read as "latents converged". The bank now gates
   on field R̂.

5. `make_population` called without required `snr_phot`/`snr_line` (would have
   died on first run); `credible_interval` had `np.exp` without max-subtraction
   (NaN at large N) and `searchsorted` quantizing widths to whole grid cells.

---

## 4. The open problem: τ

**Ruled out** (each measured, do not re-try):

| Hypothesis | Test | Result |
|---|---|---|
| Kernel/grid/quadrature wrong | Truth fields, same grid | τ = 143–152 ✓ — they're fine |
| Grid bounds too tight | Truth sits comfortably inside | Not it |
| B2 degenerates with field dimension | Toy at n_grid 4→24 | ESS/K stays 0.97 → 0.88, recovers both params. **Refuted** |
| Weight degeneracy | K=500→4000, ESS 3.8→46 | τ unchanged (11.6–29.0 vs 11.9–31.0) |
| Pathological galaxies | Dropped them | τ unchanged |
| τ prior mismatch | Fixed | τ unchanged |
| Field too rough (excess HF power) | lag-1 correlation | true +0.2566 vs healthy Laplace +0.2506 — **same** |
| Not enough HMC compute | 1000/1000 → 4000/4000 | **Worse**: σ 0.597 → 1.000 (ceiling), R̂ 4.42 → 1.61 |

**What this leaves.** The bias is in *what the interim posteriors are*, not in
the estimator. Two live leads:

- **σ is an amplitude** (recoverable from the field's marginal variance, which a
  Gaussian approximation gets right). **τ is a correlation length**, recoverable
  only from joint structure across age nodes — the thing a per-galaxy Gaussian
  is least likely to preserve. That lag-1 matches while τ doesn't suggests the
  discrepancy is at longer lags or higher order. **Next test: compare the full
  empirical covariance across age nodes, truth vs Laplace, not just lag-1.**
- B2 assumes draws come from the *actual* interim posterior. A Laplace Gaussian
  is not one. That bias does not shrink with N — consistent with τ tightening
  around a wrong value. Testing this needs honest HMC posteriors, which means
  addressing the funnel (below).

---

## 4a-bis. What σ and τ mean, and what is identified

`SFR(t) = SFR_smooth(t) · exp(gp(t) − K(0)/2)` with
`K_ij = (σ·ln10)² exp(−|tᵢ−tⱼ|/τ)`.

- **σ [dex]** — RMS scatter of log₁₀(SFR) about the secular track. σ = 0.75
  means factor-5.6 swings at 1σ. It is `sqrt(K(0))/ln10`, the field's marginal
  standard deviation. The `−K(0)/2` term keeps the modulation mean-preserving.
- **τ [Myr]** — e-folding time of the autocorrelation; how long a burst
  remembers itself.
- **PSD**: `S(f) = 2σ²τ / (1 + (2πfτ)²)`. Flat at `2σ²τ` below the break
  `f ≈ 1/2πτ`, falling as f⁻² above it. Only if the data lie entirely below the
  break does the product σ²τ become the sole identified combination.

**In this setup σ is identified and τ is not, and the product is worse than
either** — see §5 step (a) for the measurement. Do not report σ²τ here.

---

## 4b. The estimator is sound — the τ bias is NOT an information limit

> **This section previously concluded the opposite** ("the τ bias is not a
> defect, it is the correct answer to an ill-posed question"). That conclusion
> rested on one experiment. The experiment was measured and found broken twice
> over. The corrected experiment reverses the finding. The old text is kept
> below, marked, because the reasoning is a trap worth recognizing again.

**REFUTED — the SIR exact-posterior experiment.** It obtained per-galaxy
ensembles by sampling-importance-resampling 200k particles **from the prior**,
weighted by a Gaussian likelihood of width 0.15. Measured after the fact
(`sir_ess.py`), the importance weights are degenerate:

| constrained modes | Kish ESS (of 200,000) | unique particles per 800-draw ensemble | max weight |
|---|---|---|---|
| 4 | **4.9** [1.6, 48.7] | **15** [2, 127] | 0.365 |
| 8 | **1.0** [1.0, 1.7] | **1** [1, 4] | 1.000 |
| 16 | **1.0** [1.0, 1.6] | **1** [1, 2] | 1.000 |

At 8 and 16 modes every "800 exact posterior draws" is **800 copies of a single
prior draw**. The collapse is structural, not tuning: the prior std along the
leading OU modes is 4.59, 2.04, 1.74, … against a likelihood width of 0.15, so
the proposal is 11–31× too broad *per constrained direction*, and the
inefficiency compounds multiplicatively with each one.

It was also wrong a second, independent way: it proposed particles uniformly
over grid **indices** — log-uniform in τ, since `SharedGrid.uniform` spaces τ
geometrically — while scoring them against a `tau_prior="uniform"` pushforward
in `p_0`. That is exactly the ∝τ, 50×-end-to-end prior mismatch that
`SharedGrid.uniform`'s own docstring warns about, sitting inside the experiment
built to isolate the estimator.

The 16-mode "recovery" was therefore circular. With ESS = 1, the surviving
particle is the single prior draw closest to a field generated at truth, so it
is preferentially a particle whose own generating hyperparameters are near
truth — measured: survivors' mean (σ, τ) = (0.643, 175 Myr) against a grid
midpoint of (0.505, 127) and a truth of (0.75, 150). B2 was reading the label
off one particle, not recovering τ from data.

**Corrected experiment — no importance sampling at all.** This toy's posterior
is available in closed form: a Gaussian likelihood on `P @ m` against a prior
that is a finite Gaussian mixture (one component per grid node) gives a Gaussian
mixture posterior. `exact_post_analytic.py` draws from it directly — i.i.d.,
ESS = `n_samples` by construction, with component weights taken from
`grid.log_prior` itself so the proposal *cannot* disagree with `p_0`. Self-test:
the posterior-predictive residual is 0.150 at every setting, exactly the
injected noise.

| constrained modes | τ Myr — SIR (refuted) | τ Myr — analytic (truth 150) |
|---|---|---|
| 4 | 26–34 | **80.0–132.3** |
| 6 | 35–45 | **96.1–183.1 ✓** |
| 8 | 36–43 | **128.5–189.9 ✓** |
| 10 | 40–48 | **139.6–182.8 ✓** |
| 12 | 65–76 | **130.3–160.3 ✓** |
| 16 | 128–150 | **143.5–171.9 ✓** |

τ is recoverable from **6** constrained modes, not 16. At 4 modes it is biased
low by 1.1–1.9×, not by 5×.

**And the clean estimator does not rail.** The real bank's signature failure is
τ pinned at 473–495 Myr against a U(10, 500) prior for every N ≥ 64. Running the
analytic estimator over the same N (`exact_scaling.py`, two seeds per cell):

| modes | N | τ 68% Myr, seed 0 | τ 68% Myr, seed 1 |
|---|---|---|---|
| 4 | 16 | 141.0–384.9 | 121.2–292.2 |
| 4 | 64 | 80.0–132.3 | 182.5–286.5 |
| 4 | 256 | 108.2–142.0 | 157.6–197.7 |
| 4 | 512 | **119.8–145.4** | **144.8–173.0** |
| 8 | 512 | **129.5–151.0** | **112.1–137.3** |

Intervals *tighten* with N (≈240 Myr wide at N=16 → ≈26 Myr at N=512) and stay
centered near truth. No railing at any N. Residual: τ coverage at 4 modes misses
5 of 8 against 2.7 expected for a 68% interval — a mild under-coverage worth
noting, and categorically different from pinning at the prior edge.

**Consequence — the suspicion returns to the per-galaxy posteriors.** Given
genuinely exact per-galaxy posteriors, B2 recovers σ and τ at this information
content and this N. So the real bank's railing is not explained by the estimator
and not by the sparse-information regime. What remains is whatever makes the
real per-galaxy posteriors differ from exact ones: the **Laplace/Gaussian
approximation**, the **D=26 nuisance-parameter coupling**, or the **nonlinear
SED likelihood**. The toy exonerates the estimator; it does not by itself
convict any one of those three. Note also that the biases point in *opposite*
directions — the toy at 4 modes is biased low, the real bank rails high — which
a pure information limit would not do.

---

### Superseded reasoning (kept as a cautionary record)

The following was the previous conclusion. It is **wrong**, but the failure mode
is instructive: a single unverified experiment, whose intended property
("exact posterior draws") was never measured, was allowed to carry a
architectural conclusion.

> **The τ bias is not a defect. It is the correct answer to an ill-posed
> question.** τ needs essentially **all** modes. Set that against measured
> `n_eff` for real observables: GALEX+SDSS **3.21**, +2MASS 3.43, COSMOS 20
> bands 3.95, GALEX+SDSS **+8 emission lines 4.17**, COSMOS-20+lines 4.40.
> **Adding the emission lines will NOT rescue τ at z=0.1** — they buy ~1 mode,
> which gives τ ≈ 30–45 against a truth of 150.

The `n_eff` numbers themselves were measured independently and **stand**. The
conclusion drawn from them does not: on the corrected table 4 → 6 modes is the
difference between a 1.6× low bias and covering truth, so ~1 extra mode is no
longer obviously worthless. Whether emission lines rescue τ is **reopened**.

**What Burnham+2026 (arXiv:2601.20930) does differently** (unaffected by the
above — these are facts about their paper, not about this experiment).

1. **No per-galaxy SFH fitting at all.** Population-level SBI (SNPE/MAF) on the
   *distribution* of ~5 compressed summaries (Hα, dust-corrected FUV, NUV,
   Balmer break, U−V). They cite Wang+2025 that single-galaxy SEDs lack the
   information to recover rapid SFHs — exactly the degeneracy that breaks the
   approach here.
2. **Flex-PSD, not a single DRW.** Five independent logPSD bins at fixed
   timescales plus two recent-slope nuisances. For a DRW the PSD below the break
   is flat at σ²τ, so unless the data resolve the break only the **product** is
   identified. Prior work here concluded quoting independent (σ, τ) from one DRW
   is **not well-posed**, and that this is precisely why Burnham uses flex-PSD.
3. **A far richer observable** — z≈4 NIRSpec Hα+FUV+NUV+Balmer break. This
   project's own z≈4 replication found that observable lifts the amplitude
   degeneracy "that z=0.1 narrow-optical could not."
4. **They are biased too** — burstiness ~1 dex low at ≥100 Myr from outshining;
   best-constrained band 100 Myr–1 Gyr; the 1 Myr and ≥1 Gyr bins poorly
   calibrated.

**Refuted here, do not re-check:** the field is generated by
`drw_innovations_gp_from_xi`, the exact linear-time OU recursion (#865), which
matches the estimator's `ou_logpdf` — there is **no** kernel mismatch. (This
rests on #865 and on direct comparison of the two code paths, *not* on the
retracted SIR experiment.) The historical `log_age_ref=8.0` single-reference
Jacobian issue does not apply to the current field model.

**Status of the two caveats that prompted this rework.**

1. **Overconfidence — resolved in the toy, still open in the real bank.** The
   8 Myr-wide interval centered 5× from truth was itself an SIR artifact: with
   ~15 unique particles the ensemble is far under-dispersed, and B2 reads a
   near-degenerate "posterior" as sharp information. With exact draws the same
   4-mode case returns 80.0–132.3 Myr — 52 Myr wide, an interval commensurate
   with the information. But the **real bank** still returns a narrow interval
   pinned at 473–495, and prior work records τ's posterior at 0.98× the prior
   width (`project_psd_sigma_is_identified_tau_is_the_prior`). That remains
   unexplained and is now the live defect.

2. **SIR degeneracy — confirmed, and it was fatal.** Measured above. This is why
   §4b's conclusion inverted. Recorded as a standing lesson in §7.

**Consequences for the plan.** Chasing the τ bias *is* now worth effort, and it
has a specific target: the per-galaxy posteriors, not the estimator. The
options below remain on the table but are no longer forced:
(a) **change the measurand — report σ alone. NOT σ²τ.** Measured pushforward of
    the 2-D grid posterior onto v = σ²τ, fractional 68% widths at N=32:
    **σ 0.202, τ 0.962, σ²τ 1.237** — the product is the *worst* constrained of
    the three. A genuine σ²τ degeneracy would leave both marginals broad and the
    product tight, along a ridge of constant σ²τ; here σ is 5× tighter than the
    product. Reason: σ² = K(0) is the field's marginal *variance*, which the ~4
    smooth low-frequency modes photometry constrains do carry, while τ needs
    correlation structure across time separations that those modes miss. The two
    are constrained by disjoint features, so multiplying them only imports τ's
    noise. (The σ²τ framing is correct in the z≈4 regime recorded in
    `project_psd_z4_burnham_replication`, where the data probed only *below* the
    break frequency and the plateau 2σ²τ was all that was identified. It does
    **not** transfer here — verified, not assumed.);
(b) **change the observable** — z≈4 rest-UV + Hα, where the prior replication
    got σ to separate cleanly and recover at N=256;
(c) **change the architecture** — flex-PSD bins and/or population-level
    inference on summaries instead of per-galaxy field marginalization.
σ is sound in the current setup and is worth reporting on its own.

---

## 4a. The tilt, localized

Per-galaxy `log Z_i(σ,τ)` surfaces computed two ways on the identical grid, by
running `shared_log_posterior` on one galaxy at a time (so
`log_posterior = log_prior + log Z_i` comes out of the production path):

| | median per-galaxy peak | summed over 24 | `logZ(corner) − logZ(truth)` | % favoring corner |
|---|---|---|---|---|
| **Truth field** (K=1) | σ 0.690, τ **132.8** | σ 0.715, τ **141.9** ✓ | **−2.433** nats/gal | 4% |
| **Laplace** (K=1000) | σ 0.757, τ **12.2** | σ **1.000**, τ 162.0 ✗ | **+0.098** nats/gal | 38% |

**The tilt is +0.098 nats per galaxy.** Multiply by N: ~+3 nats at N=32, still
losable to noise; +6.3 at N=64, enough to overwhelm the prior. **The observed
N≈32–64 threshold falls straight out of this number.**

**The damage is entirely in τ, per galaxy, before any pooling.** Laplace gets σ
right (median peak 0.757 vs truth 0.75) but pins τ to the grid floor — 10.0–13.0
for six of eight galaxies, median 12.2, against truth-field peaks that scatter
73–438 Myr but center correctly.

**Mechanism.** `log Z_i = logsumexp_k[log p(m_k|σ,τ) − log p_0(m_k)]` averages
over the draw ensemble. A small-τ kernel is nearly white and gives moderate
density to *every* draw; a large-τ kernel gives high density to smooth draws and
near-zero to rough ones. An ensemble containing prior-like roughness — exactly
what the ~12 unconstrained field directions contribute — therefore favors small
τ. With K=1 there is no ensemble, the single true field is smooth, and large τ
wins. That is why the truth-field test passes and the pipeline does not.

Dividing by `p_0` is precisely what should cancel this. It fails because `p_0`
is the **grid-averaged pushforward** while the draws actually came from a
**Laplace Gaussian** — the correction is for the wrong sampling density, and a
residual tilt survives. This is a subtlety in the estimator's assumptions, not a
coding error, which is why every code-level remedy left it untouched.

**§4b's corrected experiment confirms this mechanism.** In
`exact_post_analytic.py` the draws come from the exact posterior under the
grid-mixture prior, so `p_0` — the grid-averaged pushforward — *is* the true
interim density, by construction rather than by approximation. In that setting
the estimator recovers σ and τ and shows no railing out to N=512. The
sampling-density mismatch identified here is therefore the leading explanation
for the real bank's failure, and §4a and §4b now agree. (For a while they did
not: §4b's superseded "ill-posed measurand" reading contradicted this section
and was the weaker of the two.)

---

## 5. The funnel (not yet addressed)

`s = L(σ,τ)·ξ` is **bilinear**, so the (σ, ξ) geometry is a funnel. Symptoms:
static HMC R̂(σ) = 4.42 at 1000/1000 and still 1.61 at 4000/4000 while ξ's own
R̂ is 0.994–0.998; and seed-to-seed swings of σ from 0.597 to 1.000.

Machinery exists but is **not usable yet**: `compute_field_gp(..., centering=a)`
is wired (commit `dfec36027`), with `drw_partial_gp_from_zeta` and
`drw_latent_log_prior` from #1355. To sample `a < 1` the loss needs its prior
term corrected at `src/tengri/inference/loss_functions.py:478` — the prior on ζ
becomes `N(0, σ_s^(2−2a) I)`, and the `−n(1−a)·log σ_s` normalizer is **not
optional**.

**NUTS has never been run here.** Every fit this session is `mcmc_hmc` with
fixed `n_leapfrog_steps=100` — the sampler least equipped for varying curvature.
This also makes divergence counts uninformative (blackjax's static-HMC
`divergence_threshold` is 1000, so "zero divergences" means nothing). Note
`dense_mass_matrix=True` at D=26 risks the 20+ GB NUTS warmup documented in
CLAUDE.md; use `dense_mass_matrix=False`.

---

## 6. How to run it

Always `PYTHONPATH=<worktree>/src:.` — the editable install points at the **main**
checkout, which has no `tengri.inference.population`. Scripts hard-fail on this.

```bash
# Fit a bank once (resumable, per-galaxy checkpoints). Laplace ~3.2 s/galaxy;
# mcmc_hmc ~155 s/galaxy.
bash scripts/run_psd_bank.sh 2048 psd_bank_fixed --method laplace \
    --n-samples 1000 --n-chains 4

# Every N by subsetting — the bank is nested, so the first N galaxies ARE the
# N-galaxy population (jax.random.split is prefix-stable; verified to 8192).
PYTHONPATH=src:. JAX_PLATFORMS=cpu python scripts/hierarchical_psd_subset_scaling.py \
    --bank psd_bank_fixed --node-chunk 4
```

`--node-chunk` is the memory knob: peak ≈ `chunk × N × K × 8 × ~15` bytes. At
N=2048, K=4000 use 4 or less.

```bash
# Validate the estimator itself against an EXACT analytic posterior (§4b).
# No SED model, no Laplace, no importance sampling -- closed-form Gaussian
# mixture, so the draws are i.i.d. exact and p_0 is correct by construction.
PYTHONPATH=src:. JAX_PLATFORMS=cpu python \
    scripts/hierarchical_psd_estimator_validation.py --bank psd_bank_fixed \
    --modes 4 8 16 --n-galaxies 64 512 --seeds 0 1
```

The `resid` column is a self-test: it must equal `--noise` (0.150). Anything
else means the draws are not posterior draws.

### Files

| Path | Role |
|---|---|
| `src/tengri/inference/population/estimator.py` | `SharedGrid`, `shared_log_posterior` (streaming, b1/b2) |
| `src/tengri/inference/population/kernel.py` | `ou_logpdf`, exact O(n) via Markov factorization |
| `src/tengri/inference/population/reconstruct.py` | `centered_fields`, ξ → m |
| `src/tengri/inference/population/interim.py` | per-galaxy HMC driver |
| `src/tengri/inference/population/diagnostics.py` | `credible_interval`, `interval_width_scaling` |
| `src/tengri/analysis/population_mocks.py` | `make_population`, truth guards |
| `scripts/hierarchical_psd_fit_bank.py` | fit the bank (laplace or mcmc_hmc) |
| `scripts/hierarchical_psd_subset_scaling.py` | the scaling curve + gate |
| `scripts/hierarchical_psd_estimator_validation.py` | estimator vs exact analytic posterior (§4b) |
| `scripts/run_psd_bank.sh` | restart-to-completion wrapper |

40+ contract tests under `tests/contract/test_population_*.py`.

---

## 7. Traps

- **The width-scaling gate is necessary, not sufficient — demonstrated three
  ways.** τ passed it at −0.516 ± 0.076 while being 10× wrong; σ passes it at
  −0.672 ± 0.080 on the clean bank while railing into a grid corner (the
  shrinkage IS the failure); and the two disagree within a single run. Always
  check coverage AND ESS.
- **Pooling reduces variance, not bias.** Unconverged interim fits at large N
  give a *tighter* wrong answer. Report interim R̂ before the intervals.
- **`ess.at_mode` is per-galaxy; the scaling script reports `min`.** A minimum
  over a growing sample can only fall, so it looks like a collapse when the
  distribution is flat. Report the median. This cost a wrong mechanism once.
- **A single bad galaxy destroys B2.** Field std 1155 → its OU density is ~0
  everywhere but one node, its weight buries the other 499 draws, `logsumexp`
  collapses to `max`, and the posterior pins to a grid corner — *identically at
  every N*. Byte-identical intervals across N means it is not averaging.
- **Laplace R̂ is vacuous** (i.i.d. draws → ~1.0 by construction). Stored as NaN
  deliberately. The Laplace failure mode is the wrong covariance *shape*, which
  no between-chain statistic can see. The field-scale guard is what catches it.
- **Truth-field and toy tests cannot see interim/pushforward mismatches.** With
  K=1 `p_0` is a per-galaxy constant that cancels; the toy generates and scores
  under the same prior.
- **An *intended* property is not a *measured* one — this cost the biggest wrong
  conclusion in this work.** §4b rested on draws described as "exact posterior
  draws". They were obtained by importance sampling, whose ESS was never
  checked; measured, it was 1–5 out of 200,000, and at 8+ modes every ensemble
  was 800 copies of one particle. The conclusion inverted once fixed. **Any
  time an ensemble is produced by reweighting or resampling, report its ESS
  next to the result** — and prefer a closed form when one exists (this toy's
  posterior was an analytic Gaussian mixture all along).
- **Importance sampling from a prior dies exponentially in constrained
  dimensions.** Prior std 4.59, 2.04, 1.74, … against likelihood width 0.15 is
  11–31× too broad *per direction*, and the cost multiplies. Four such
  directions is already fatal at 200k particles. This is structural, not a
  tuning failure — do not try to fix it by adding particles.
- **Shared machine.** A total-RSS guard SIGKILLs python largest-first at 15 GB
  across *all* sessions (log: `/tmp/oom_guard_15gb.log`). Silent death with no
  traceback is this, not your bug — read the guard log first. Hence the
  resumable bank.

---

## 8. Next steps, ranked

1. **Fix the per-galaxy draws — NOT `p_0`.** §4b now shows the estimator is
   sound given exact per-galaxy posteriors, and §4a localizes the damage to the
   mismatch between the true interim posterior `q_i` and the Laplace Gaussian
   `q̂` the draws actually come from. Do NOT reach for weight removal
   (closed-form Gaussian marginalization): median ESS is ~550, the weights are
   not the problem.

   **Correction to an earlier version of this list — do not do this.** It
   previously ranked "make `p_0` match the actual sampling density" as the most
   likely single fix. That is degenerate. B2 requires dividing by the interim
   **prior** with draws from the interim **posterior**:
   `Z_i = C_i E_{q_i}[p(m|σ,τ)/p_0(m)]`. Substituting the Laplace density gives
   `E_{q̂}[p(m|σ,τ)/q̂(m)] = ∫ p(m|σ,τ) dm = 1` for every (σ,τ) — a perfectly
   flat surface carrying no information at all. `p_0` as coded (the
   grid-averaged pushforward) is **correct**; `q_i` is what is wrong.

   Routes, cheapest first:
   (a) **Importance-correct the Laplace draws back to `q_i`.** Weight draw `m_k`
   by `w_k ∝ p(d_i|m_k) p_0(m_k) / q̂(m_k)` and use the self-normalized estimate.
   Legitimate and cheap. **Measure the per-galaxy ESS of these weights before
   trusting any result** — this is the same correction, in the same geometry,
   that collapsed to ESS ≈ 1 in §4b.
   (b) Use honest posteriors (NUTS with the funnel addressed) so `q_i` is right
   and no correction is needed.
   (c) Restrict the field to its data-constrained subspace (~4 of 16 modes)
   before scoring, removing the prior-like roughness that drives the tilt.
2. **Full covariance comparison, truth vs Laplace fields** across age nodes (not
   just lag-1). Cheapest test of the leading τ hypothesis. ~30 min.
3. **Run `mcmc_nuts`** (`dense_mass_matrix=False`) on ~16 galaxies and compare
   field R̂ and τ against static HMC. Never tried; static HMC is the wrong tool
   for a funnel. ~1 h.
4. **Finish partial centering**: the loss-prior correction at
   `loss_functions.py:478`. The principled funnel fix.
5. ~~File the `min_eigenvalue=1e-6` footgun~~ — filed as **#1515**. Worked
   around here with `--min-eigenvalue 1.0`; the library default is unchanged.
6. Acceptance criteria 3 (two-population separation) and 6 (coverage across ≥3
   realizations) are still unrun. Criterion 2 needs multiple realizations —
   single-realization coverage was over-read earlier in this work.
