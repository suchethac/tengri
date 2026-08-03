# Hierarchical SFH-PSD recovery — handoff

**Branch:** `worktree-hierarchical-psd-spec` · **PR:** #1479

**Status.** The estimator is correct — demonstrated against an exact analytic
posterior (§4b): given true per-galaxy draws it recovers σ and τ with no railing
out to N=512. End-to-end it still has a **ceiling on N around 32–64**, above
which the shared posterior jumps to a grid corner. **The cause is now
identified and proven sufficient (§4e): the anisotropy of the per-galaxy ξ
covariance.** Matching its eigenvalue spectrum alone — random eigenvectors —
reproduces the railing to three significant figures; isotropic ξ does not rail.
That anisotropy is **real, not a Laplace artifact** (§4f): a converged NUTS
posterior has the same shape and the estimator does worse on it. So the fix is
not a better sampler, and not more draws — the tilt is converged in K. The
remaining suspect is `p_0` (§8).

Everything below is measured, not assumed. Several sections record conclusions
that were later **refuted by measurement**; they are kept, marked, because the
reasoning errors recur.

**Reading order** (the §4 lettering accreted; this is the sequence):

| § | Says |
|---|---|
| **4** | the open problem, and what is ruled out |
| **4a-bis** | what σ and τ mean; what is identified |
| **4b** | the estimator is sound (exact-posterior test) — supersedes an "ill-posed measurand" claim |
| **4a** | the per-galaxy tilt, localized — its *mechanism* is superseded by §4e |
| **4c** | four suspects eliminated: funnel, nonlinearity, nuisance coupling, reconstruction |
| **4d** | `corr(tilt, interim τ) = +0.856` — a real diagnostic, but **refuted as the cause** |
| **4e** | **the cause: the ξ covariance spectrum, proven necessary and sufficient** |
| **4f** | the anisotropy is **real** — NUTS confirms it; the sampler is not the fix |

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

**Does NOT bear on the earlier 8192-galaxy MGVI experiment** (corrected
2026-08-02; an earlier revision of this section claimed it did). The +0.098
nats/galaxy tilt is a property of B2 specifically — it arises from dividing by
`p_0`, the interim pushforward, in §4a. A joint MGVI fit has no interim prior,
no importance weights and no `p_0`, so the mechanism cannot occur there. The
railing seen in that experiment has a separate, already-documented cause: the
`sigma^2 tau` degeneracy of a single DRW below its break frequency, plus
Gaussian-VI miscalibration (SBC gave 0/20 coverage on sigma). Attributing both
to one cause would retire the wrong hypothesis.

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
   how draws were made, not a modeling choice — mismatched by a factor ∝ τ,
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
>
> **Corroborated on the real observable by a joint-NUTS control** (2026-08-02,
> `docs/dev/hierarchical-psd-preliminary-results.md` §5). The refutation below
> is a *toy* whose per-mode noise (0.15) is not calibrated to SNR = 20
> photometry, so it establishes "the estimator is not what breaks τ" but cannot
> speak to whether the real observable identifies τ. The joint fit can, and
> does: on the same four galaxies as the N = 4 row, joint NUTS returns
> τ = 104–430 Myr — a posterior **0.98× the prior width**, i.e. it correctly
> reports learning nothing — while B2 returns 12–31 Myr, excluding truth by
> 5–12×. **The correct answer to an unidentified parameter is its prior, and B2
> does not give it.** Same verdict as this section, reached without a toy:
> the wall and the tilt are both real.

**REFUTED — the SIR exact-posterior experiment.** It obtained per-galaxy
ensembles by sampling-importance-resampling 200k particles **from the prior**,
weighted by a Gaussian likelihood of width 0.15. Measured after the fact
(throwaway probe, reproducing the SIR loop verbatim), the importance weights
are degenerate:

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
mixture posterior. `scripts/hierarchical_psd_estimator_validation.py` draws from it directly — i.i.d.,
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

**This does NOT say τ is recoverable from the real observable.** The toy's
per-mode noise (0.15) is a stand-in, not calibrated to the SNR=20 photometry,
so its "4 modes" and the measured `n_eff ≈ 3.2` are not the same quantity. A
**joint NUTS** fit on the real observable (PR #1520, N=4, D=98) returns τ at
**0.98× the prior width** — 104.5–429.7 against a prior of 88.4–421.6 — i.e.
τ genuinely is *not identified* by z=0.1 broadband photometry. What this section
establishes is narrower and sharper: **the estimator is not what breaks it.**

The two facts compose, and both are needed:

| | τ at N=4 (truth 150) | reading |
|---|---|---|
| Joint NUTS | 104.5–429.7 | returns the prior — correct behavior for an unidentified parameter |
| Two-step B2 | 12.2–30.9 | tight, excludes truth by 5–12× — **a defect** |

An unidentified parameter *should* come back as its prior. B2 instead returns a
confident wrong answer, and at N ≥ 64 rails to the grid edge. That gap is the
+0.098 nats/galaxy tilt of §4a, sitting **on top of** an identifiability wall.
Do not let the identifiability story retire the tilt bug.

**And the clean estimator does not rail.** The real bank's signature failure is
τ pinned at 473–495 Myr against a U(10, 500) prior for every N ≥ 64. Running the
analytic estimator over the same N (same script, `--n-galaxies`/`--seeds`):

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
genuinely exact per-galaxy posteriors, B2 recovers σ and τ and does not rail out
to N=512. So the *railing* is not a property of the estimator, and it is not
what non-identifiability looks like either — non-identifiability returns the
prior, as joint NUTS does. What remains is whatever makes the real per-galaxy
posteriors differ from exact ones: the **Laplace/Gaussian approximation**, the
**D=26 nuisance-parameter coupling**, or the **nonlinear SED likelihood**. The
toy exonerates the estimator; it does not by itself convict any one of those
three. Note also that the biases point in *opposite* directions — the toy at
4 modes is biased low, the real bank rails high — which a pure information
limit would not do.

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
    **not** transfer here — verified, not assumed.)

    *Independently corroborated by a second method* — eigendecomposing the
    mass-weighted covariance of the same posterior in (log σ, log τ) gives a
    tight direction of **σ¹ τ^−0.09**, stable across N = 4–32, against the
    σ¹ τ^+0.5 a power degeneracy requires. Two different statistics of the same
    grid, same conclusion. See
    `scripts/hierarchical_psd_identified_combination.py`, which also reports
    **edge mass** (posterior mass on the grid boundary) — 0.02–0.07 through
    N=32, **0.79 at N=64**, 0.99 at N=128, localizing the N ceiling to one sharp
    truncation event;
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

**Mechanism — SUPERSEDED, see §4d.** The reading below (prior-like roughness
from unconstrained directions drives a small-τ preference) is **refuted**:
sweeping the toy's information content from `n_eff` 4.0 down to 1.0 leaves the
tilt flat at −0.88 to −0.98 and moves the peak *up*, not toward the floor. The
actual driver is measured in §4d and it is not roughness.

> `log Z_i = logsumexp_k[log p(m_k|σ,τ) − log p_0(m_k)]` averages over the draw
> ensemble. A small-τ kernel is nearly white and gives moderate density to
> *every* draw; a large-τ kernel gives high density to smooth draws and
> near-zero to rough ones. An ensemble containing prior-like roughness — exactly
> what the ~12 unconstrained field directions contribute — therefore favors
> small τ. With K=1 there is no ensemble, the single true field is smooth, and
> large τ wins.

Dividing by `p_0` is precisely what should cancel this. It fails because `p_0`
is the **grid-averaged pushforward** while the draws actually came from a
**Laplace Gaussian** — the correction is for the wrong sampling density, and a
residual tilt survives. This is a subtlety in the estimator's assumptions, not a
coding error, which is why every code-level remedy left it untouched.

**§4b's corrected experiment confirms this mechanism.** In
`scripts/hierarchical_psd_estimator_validation.py` the draws come from the exact posterior under the
grid-mixture prior, so `p_0` — the grid-averaged pushforward — *is* the true
interim density, by construction rather than by approximation. In that setting
the estimator recovers σ and τ and shows no railing out to N=512. The
sampling-density mismatch identified here is therefore the leading explanation
for the real bank's failure, and §4a and §4b now agree. (For a while they did
not: §4b's superseded "ill-posed measurand" reading contradicted this section
and was the weaker of the two.)

---

## 4c. Two suspects eliminated — the railing is still unexplained

§4b left three candidates for what makes real per-galaxy posteriors differ from
exact ones: the Laplace approximation, the D=26 nuisance coupling, and the
likelihood nonlinearity. Two are now measured and **neither reproduces the
railing**. Each test is an A/B in the §4b toy with one ingredient added and
everything else — grid, estimator, `p_0` — held fixed.

**1. Laplace + the funnel is NOT sufficient.** Interim fit over (ξ, σ, τ) done
exactly as the pipeline does it (MAP in unconstrained space, Hessian,
eigenvalues floored at 1.0, `cov = H⁻¹`, draw, reconstruct
`m = mean(σ) + L(σ,τ)ξ`), against exact draws on the same galaxies:

| N | exact | Laplace |
|---|---|---|
| 16 | 176.0–370.9 | 162.3–280.3 |
| 64 | 148.6–236.3 ✓ | 150.1–197.3 |
| 256 | **132.9–171.0 ✓** | **144.3–167.4 ✓** |

Laplace tracks exact and recovers τ at N=256. Its *per-galaxy* τ is biased high
(median 247.6 vs truth 150 — the same direction as the real railing), but that
bias does **not** propagate: B2 reweights fields, not per-galaxy τ.

**2. Likelihood nonlinearity is NOT sufficient either.** The toy above is unfair
to reality in one specific way — conditional on (σ, τ) its posterior over ξ is
*exactly* Gaussian, so Laplace is near-perfect and only the funnel stresses it.
Real photometry is Gaussian in flux, and flux is linear in SFR while `m` is
log-SFR: `f = W exp(m)`. Swapping in that observable (5 nonneg age kernels,
SNR 20) and changing nothing else:

| N | σ (truth 0.75) | τ Myr (truth 150) |
|---|---|---|
| 16 | 0.610–0.692 | 85.9–161.4 ✓ |
| 64 | 0.617–0.659 | 116.7–156.3 ✓ |
| 256 | 0.612–0.630 | 110.8–129.0 |

Still no railing. It biases σ **low** (field std 1.25 against an expected 1.73 —
under-dispersed) and τ mildly low. **Both toys bias low; the real bank rails
high.** Whatever is missing does not merely amplify these effects, it reverses
their sign.

**3. The real field ensembles are only mildly anomalous.** Reconstructed
`centered_fields` for 64 bank galaxies (256k draws), against the interim prior
mixture — the distribution `p_0` itself represents:

| | bank | prior mix | truth OU |
|---|---|---|---|
| field std | **1.810** | 1.562 | 1.726 |
| implied τ from correlation-vs-lag | 6364 Myr | 6914 Myr | 153.6 Myr |

The bank is **over-dispersed** relative to the prior mixture (+16%) and
modestly over-correlated at 0.5–4 Gyr lags (+0.05 to +0.17 in correlation).
Both push toward high σ and high τ, which is the railing direction, and both are
small — the right magnitude for §4a's +0.098 nats/galaxy rather than a gross
reconstruction error. **No gross defect in the field reconstruction exists to
find.**

> ⚠ **A trap that nearly produced a false smoking gun here.** Comparing the bank
> to a *fixed*-σ truth-OU control shows implied τ = 6364 Myr against 153.6 — a
> 42× discrepancy that looks damning. It is entirely an artifact: the bank pools
> draws whose σ varies across the posterior, and a scale mixture inflates
> long-lag correlation. The interim prior mixture shows the **same** inflation
> (6914 Myr), and `p_0` is that mixture, so the estimator divides it out.
> **A posterior mixture must be compared against the prior mixture, never
> against a fixed-parameter draw.**

**4. Nuisance coupling is NOT sufficient either.** The third increment adds the
degeneracies the real D=26 fit has and the toys lacked, each aimed at a
different part of the field: a free log-normalization (degenerate with the
**mean** of `m`), a smooth SFH slope (degenerate with the lowest-frequency
**ramp** mode), and dust attenuation (mimics age through **color**, competing
with the low-frequency structure the ~4 constrained modes carry). 21 parameters
against the real fit's 26; everything else unchanged.

| N | σ (truth 0.75) | τ Myr (truth 150) |
|---|---|---|
| 16 | 0.603–0.712 | 116.3–224.2 ✓ |
| 64 | 0.620–0.675 | 110.0–150.6 ✓ |
| 256 | 0.637–0.664 | **127.8–150.3 ✓** |

τ *covers truth at every N*. Still no railing.

**5. The field reconstruction is sound — the draws are self-consistent.** For
each stored draw, a synthetic OU field was generated at that draw's own
`(σ_k, τ_k)` and the per-draw statistics compared. Matched by construction in
σ, τ, node count and correlation, so any offset is a property of the draws:

| per-draw statistic | bank | synthetic at same (σ_k, τ_k) | ratio |
|---|---|---|---|
| std(m) | 1.146 | 1.153 | **0.994** |
| lag-1 correlation | 0.251 | 0.270 | 0.927 |
| corr of std(m) with own σ·ln10 | 0.767 | 0.763 | — |

`centered_fields` is not the problem, and the draws genuinely carry the
amplitude and smoothness their own hyperparameters imply.

> ⚠ **The same mixture confound, a second time.** This test was motivated by an
> apparent contradiction — per-galaxy posterior mean σ = 0.639, but pooled field
> std 1.810 implying σ ≈ 0.786. There is no contradiction: pooling draws over a
> *varying* σ raises the pooled std above the median σ. Any statistic computed
> by pooling across a mixture needs a mixture-matched control, not a
> point-estimate comparison.

**Where this leaves the search.** Six things are now measured and cleared: the
estimator (§4b), the importance weights (median ESS ~550), the field
reconstruction, the funnel, the likelihood nonlinearity, and the nuisance
coupling. Every caricature recovers τ; only the real pipeline rails. The
difference is therefore **specific to the real forward model or the real
`run_laplace` fit**, not to the generic structure — so more toy-building has hit
diminishing returns.

The one route left that uses the real forward model: **fit the same handful of
real galaxies with `mcmc_nuts` (`dense_mass_matrix=False`) and compare their
`log Z_i(σ,τ)` surfaces against the Laplace ones directly.** §4a already has the
Laplace surfaces and the +0.098 nats/galaxy tilt; the missing half is a
trustworthy reference for the same galaxies. That is a small, targeted run — not
a re-fit of the bank.

---

## 4f. The anisotropy is REAL — NUTS confirms it, so the sampler is not the fix

§4e proved the ξ covariance spectrum sufficient for the railing and left one
question: is that anisotropy a Laplace artifact, or a real feature of the
posterior? **It is real.** NUTS — converged, R̂ ≤ 1.05, zero divergences —
produces a spectrum as anisotropic as Laplace's, slightly more so.

Same 8 galaxies, uniform K=1000 in both arms (median across galaxies; the ξ
prior is N(0, I), so isotropic would be all 1.00):

```
laplace  1.19 1.11 1.09 1.04 1.01 0.97 0.94 0.91 0.87 0.86 0.81 0.77 0.68 0.63 0.50 0.25   total 13.6/16
nuts     1.34 1.30 1.23 1.16 1.07 1.03 1.00 0.92 0.88 0.84 0.79 0.74 0.67 0.62 0.56 0.28   total 14.4/16
```

And the estimator does **worse** on the NUTS posteriors, not better:

| arm | mean tilt | pooled τ 68% (truth 150) | mode | ESS |
|---|---|---|---|---|
| laplace | −1.540 | 13.4–30.9 | (0.849, 19) | 45 |
| **nuts** | −1.018 | **10.0–15.2** | (0.883, **10.0**) | **14** |

NUTS pins τ to the grid floor. (At N=8 both arms are biased *low* — this is the
small-N end of the §2 sweep, where the per-galaxy `Z_i` peaks dominate; the
railing *high* emerges at N ≥ 64 as the accumulated tilt takes over. The mean
tilt over only 8 galaxies is noisy given the −9 to +5 spread, so the **spectrum**
comparison is the robust result here, not the tilt.)

**The tilt is not a finite-K artifact either.** A natural hypothesis: `Z_i` is
estimated by Monte Carlo and then logged, and `log Ẑ` is biased low by Jensen,
by roughly the relative weight variance (1/ESS) — which is (σ,τ)-shaped and so
would not cancel. That predicts a tilt shrinking like 1/K. Measured on the real
bank, it does the opposite — it **grows and saturates**:

| K | 125 | 250 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|
| mean tilt | +0.146 | +0.167 | +0.246 | +0.314 | +0.318 | +0.318 |
| tilt × K | +18 | +42 | +123 | +314 | +637 | +1273 |

`tilt × K` grows monotonically — definitively not 1/K. The tilt is **converged**
by K ≈ 1000. More draws do not help.

**What this rules out.** The entire "fix the per-galaxy fits" direction, which
§4e ranked first. Laplace is exonerated: a converged NUTS posterior has the same
shape and the estimator fails on it identically. Combined with the K-sweep, the
position is uncomfortable and worth stating plainly:

* `Z_i = C_i E_{q_i}[p(m|σ,τ)/p_0(m)]` is exact for **any** `q_i`.
* `q_i` is now known to be right (NUTS).
* The Monte Carlo estimate has **converged** (flat in K past ~1000).
* And the answer is still wrong.

Those four cannot all hold, so one of the identity's assumptions is violated in
the implementation. The most likely candidate is `p_0`: it must be the *exact*
marginal interim prior on `m` that the per-galaxy fits actually ran under, and
it is currently a 60×60 grid quadrature of that integral. That is the next thing
to test — not another sampler.

---

## 4e. The cause — the ξ covariance spectrum, proven sufficient

**Matching the ξ covariance eigenvalue spectrum alone reproduces the railing to
three significant figures.** Holding `(σ_k, τ_k)` fixed at the bank's stored
values and varying *only* the ξ ensemble:

| N | real ξ | synthetic, **matched spectrum, random eigenvectors** | synthetic isotropic |
|---|---|---|---|
| 16 | 38.5–91.1 | 132.0–429.4 ✓ | 52.1–139.1 |
| 64 | 433.3–490.9 **RAILED** | **437.4–491.5 RAILED** | 108.0–174.8 ✓ |
| 256 | 473.0–494.9 **RAILED** | **472.9–494.8 RAILED** | 154.2–196.8 |

The eigenvectors are random — only the spectrum is matched — and it rails
identically. Isotropic ξ does not rail. **The ξ covariance spectrum is necessary
and sufficient.**

**The spectrum** (median across galaxies; the ξ prior is N(0, I), so isotropic
would be all 1.00):

```
1.27  1.19  1.13  1.06  1.01  0.96  0.92  0.87  0.83  0.78  0.73  0.67  0.61  0.55  0.42  0.20
```

Total variance 13.2 against 16. Two features matter: the data shrinks the
constrained directions (down to 0.20), and **four to five directions sit ABOVE
the unit prior**. The latter is the signature of marginalizing over correlated
nuisances — `cov = H⁻¹` is formed on the full D=26 Hessian, and the ξ-block of
the inverse is not the inverse of the ξ-block. Degeneracy with mass, dust and
(σ, τ) inflates those directions past the prior.

The reconstructed field ensemble then has a covariance that **no single
OU(σ, τ) can represent**, and the estimator settles on whichever grid point fits
least badly — the corner. Per-galaxy that is +0.27 to +0.32 nats; times N it
beats the prior between N=32 and N=64, exactly where the jump is.

**How the elimination converged.** Holding everything else fixed and swapping
only the ξ ensemble:

| ξ | (σ, τ) | τ at N=256 | |
|---|---|---|---|
| real, paired with own (σ,τ) | real | 473.0–494.9 | RAILED |
| real, **shuffled** within galaxy | real | 473.0–494.9 | RAILED |
| fresh N(0, I) | real | 154.1–197.6 | not railed |
| synthetic, matched spectrum | real | 472.9–494.8 | RAILED |

Shuffling changes nothing, and the within-galaxy `corr(mean|ξ|, τ)` is +0.018 —
so the ξ–(σ,τ) *coupling* is irrelevant. Only the ξ *distribution* matters.

**Why the earlier self-consistency check missed it.** §4c compared per-draw
`std(m)` against synthetic draws and got a ratio of 0.994 — reassuring, and
useless here. Total ξ variance is 13.2 vs 16, a std ratio of 0.96, which that
test cannot distinguish from noise. The defect is in the **shape** of the
covariance, not its scale, and a scalar summary is blind to shape.

**What is NOT yet established.** Whether this anisotropy is a faithful feature
of the true posterior or an artifact of Laplace. Marginalizing over nuisances
*can* legitimately inflate a marginal variance above its prior, so the spectrum
is not self-evidently wrong. This matters for the fix:

* **If NUTS gives a different spectrum** → Laplace artifact → fix the per-galaxy
  fits and the estimator is fine as written.
* **If NUTS gives the same spectrum** → the anisotropy is real, `q_i` is
  genuinely this shape, and since `Z_i = C_i E_{q_i}[p(m|σ,τ)/p_0(m)]` is exact
  for *any* `q_i`, the fault would lie in how the estimator handles it.

That is now a single, precisely scoped measurement on ~8 galaxies: **compare the
ξ covariance spectrum, Laplace vs NUTS.** Not a bank re-fit.

---

## 4d. The correlation — a symptom, not the cause

> ⚠ **This section's claim to be "THE MECHANISM" is REFUTED — see §4e.** The
> correlation below is real and reproducible, but it is a **symptom**. The
> decisive test: rebuild the ensembles from the bank's own `(σ_k, τ_k)` with
> fresh isotropic ξ, preserving the entire hyperparameter distribution this
> section blames. Result: **τ 154.1–197.6 at N=256, no railing** — against the
> real 473.0–494.9. The (σ, τ) spread is *not sufficient*. §4e finds what is:
> the ξ covariance spectrum, which reproduces the railing to three significant
> figures on its own.
>
> Kept because the correlation is a genuine, cheap, pre-pooling diagnostic, and
> because the reasoning error is worth seeing: a strong correlation plus a
> plausible mechanical story is not causation, and the positive control that
> would have caught it took twenty minutes.

**`corr(per-galaxy tilt, per-galaxy interim posterior τ) = +0.856`**, measured
on 64 galaxies of `psd_bank_fixed`. A real and useful diagnostic — but §4e shows
it is downstream of the ξ spectrum, not the cause.

**The tilt is not a uniform bias — it is a heavy tail.** §4a's "+0.098
nats/galaxy" is reproduced on the current bank (+0.102, 38% favoring the corner
— so §4a is *not* stale), but the mean hides the structure. Per-galaxy tilts run
**−9.14 to +5.05**, and the `Z_i` peak spans the entire grid, 10 to 500 Myr:

| gal | tilt | `Z_i` peak τ | interim τ | σ |
|---|---|---|---|---|
| 0035 | **+5.88** | 500.0 | 309.0 | 0.605 |
| 0050 | +5.46 | 500.0 | 349.8 | 0.627 |
| 0019 | +5.05 | 500.0 | 350.4 | 0.650 |
| 0042 | +4.97 | 500.0 | 363.5 | 0.663 |
| 0036 | +4.32 | 500.0 | 402.3 | 0.633 |
| … | | | | |
| 0028 | −3.39 | 10.0 | 56.5 | 0.750 |
| 0037 | −4.07 | 15.9 | 41.4 | 0.844 |
| 0001 | **−9.14** | 13.0 | 21.3 | 0.875 |

**Dropping the worst 5 of 64 flips the sign of the total** (+20.4 → −5.3 nats).
A handful of galaxies carries the entire effect.

**Why.** The stored draws are built as `m_k = mean(σ_k) + L(σ_k, τ_k) ξ_k`. A
galaxy whose interim posterior put τ high produces a *smooth* ensemble by
construction, and `ou_logpdf` then scores that ensemble highest at large τ — so
`Z_i` peaks at the grid maximum. `Z_i` is reading the interim posterior's own τ
back out. Dividing by `p_0` cannot undo this: `p_0` is the shared interim
**prior**, identical for every galaxy, while what leaks in is each galaxy's
**posterior** position. The identity `Z_i = C_i E_{q_i}[p(m|σ,τ)/p_0(m)]` is
exact only when `q_i` is the true posterior; with `q̂` Laplace, the residual is
proportional to how far that galaxy's τ wandered.

**Why it rails, and why at N≈64.** τ is not identified per galaxy (joint NUTS
returns the prior, §4b), so per-galaxy interim τ scatters across the whole prior
range — measured 21 to 407 Myr. The resulting tilt distribution is asymmetric
(τ is compressed against its floor at 10 and stretched toward 500), leaving a
positive mean of +0.318 nats/galaxy on this sample. Multiplied by N that beats
the prior somewhere between N=32 and N=64, exactly where the jump is observed.

**Why every toy missed it — measured, not assumed.** Per-galaxy interim τ,
toy vs bank:

| | median | min | max | IQR |
|---|---|---|---|---|
| toy (21-param nuisance) | 246.3 | 156.0 | 265.5 | 41.0 |
| bank | 246.4 | **21.3** | **406.8** | **73.6** |

The medians are identical to 0.1 Myr — this is purely a **spread** difference,
1.80× by IQR, and far larger in range. The toy's τ never leaves 156–266, so the
galaxies that carry the entire tilt (interim τ 309–407 at the top, 21–79 at the
bottom) *do not exist in it*. Its tilts are correspondingly tight and negative,
0% favoring the corner, at every information content tested.

The toys reproduce the *estimator*; they do not reproduce the *spread of
per-galaxy τ posteriors* that real, τ-unidentified fits produce. **That spread,
not the estimator, is the disease.** It also explains the sign flip that puzzled
§4c: the toys bias low because they lack the high-τ tail entirely.

**What this implies for the fix.**
1. The heavy tail is diagnosable *before* pooling — the per-galaxy `Z_i` peak
   sitting at a grid bound is a per-galaxy red flag, and it correlates 0.856
   with the stored interim τ, which is already in every checkpoint.
2. Reducing the interim prior's τ range would shrink the leak directly, at the
   cost of assuming what you are trying to measure. Worth quantifying, not
   worth shipping silently.
3. The principled fix remains importance-correcting `q̂ → q_i` (§8b), and this
   section says exactly where the correction has to bite: on the galaxies whose
   τ wandered furthest.

**Status.** Every number in this section is measured on `psd_bank_fixed`. The
one supporting claim that was initially assumed — the toy/bank τ-spread
difference — has since been measured too (table above). What is *not* yet
established is the fix: no remedy has been tried against this mechanism.

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
- **A scalar summary is blind to the shape of a covariance.** The per-draw
  `std(m)` check passed at ratio 0.994 while the ξ covariance was anisotropic
  enough to rail the whole estimator (§4e) — total variance 13.2 vs 16 is a
  0.96 std ratio, indistinguishable from noise. When the suspect is a
  covariance, compare **spectra**, not scalars.
- **A strong correlation plus a plausible mechanism is not causation.** §4d had
  `corr = +0.856` and a mechanical story, and was wrong: the positive control
  (rebuild from the blamed quantity alone) did not rail. Twenty minutes. Run the
  positive control *before* writing the mechanism up.
- **Any statistic computed by POOLING across a mixture needs a mixture-matched
  control.** This confound produced two separate false leads in one session: a
  42× "over-smoothing" signal (below), and an apparent σ contradiction
  (per-galaxy mean 0.639 vs pooled-implied 0.786) that is just pooling over a
  varying σ. Neither survived a matched control. §4c.
- **A posterior mixture must be compared against the PRIOR MIXTURE, never
  against a fixed-parameter draw.** The bank's draws carry a varying σ; a scale
  mixture inflates long-lag correlation on its own. Against a fixed-σ control
  the bank's implied τ is 6364 Myr vs 153.6 — a 42× "smoking gun" that
  evaporates once the interim prior mixture (6914 Myr) is used instead, which is
  what `p_0` actually divides out. §4c.
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

   **§4c has since eliminated ALL THREE candidates**, plus the field
   reconstruction. Laplace + the funnel does not rail; adding the likelihood
   nonlinearity does not; adding the nuisance coupling does not. Every
   caricature recovers τ, and the toys bias *low* while the real bank rails
   *high*. Do not build a fourth toy — start here instead:

   **§4d has since found the mechanism** — `corr(tilt, per-galaxy interim τ) =
   +0.856`; `Z_i` reads the interim posterior's τ back out, a handful of
   galaxies carries the whole effect, and dropping the worst 5 of 64 flips the
   sign. Start there:

   **§4e has since found the cause and proven it sufficient**: the ξ covariance
   spectrum. Matching it alone — random eigenvectors — reproduces the railing to
   three significant figures; isotropic ξ does not rail. One question decides
   the fix:

   **§4f has since run this**: the anisotropy is **real**. NUTS gives the same
   spectrum (slightly more anisotropic) and the estimator does worse on it. The
   sampler is not the fix, and the tilt is converged in K, so more draws are not
   either. What is left:

   (a) **Verify `p_0` against the interim prior it is supposed to represent.**
   Four things cannot all be true at once (§4f): the B2 identity is exact for
   any `q_i`; `q_i` is right; the Monte Carlo has converged; the answer is
   wrong. The remaining suspect is `p_0` — it must be the *exact* marginal
   interim prior on `m` under which the fits ran, and it is currently a 60×60
   grid quadrature of that integral. Test it directly: draw `m` from the interim
   prior by ancestral sampling (draw σ, τ from their `Uniform` priors, then the
   OU field) and check that the estimator's `p_0` matches a kernel-density or
   analytic evaluation on those draws. A quadrature error, a normalization
   dropped, or a τ-Jacobian mismatch would all show up here and all produce a
   (σ,τ)-shaped distortion of exactly the kind observed.
   (b) **Importance-correct the Laplace draws back to `q_i`.** Weight draw `m_k`
   by `w_k ∝ p(d_i|m_k) p_0(m_k) / q̂(m_k)` and use the self-normalized estimate.
   **Measure the per-galaxy ESS of these weights before trusting any result** —
   this is the same correction, in the same geometry, that collapsed to ESS ≈ 1
   in §4b.
   (c) Restrict the field to its data-constrained subspace (~4 of 16 modes)
   before scoring, removing the prior-like roughness that drives the tilt.
2. **Full covariance comparison, truth vs Laplace fields** across age nodes (not
   just lag-1). Cheapest test of the leading τ hypothesis. ~30 min.
3. **Run `mcmc_nuts`** (`dense_mass_matrix=False`) on ~16 galaxies and compare
   field R̂ and τ against static HMC. Never tried; static HMC is the wrong tool
   for a funnel. *Corrected 2026-08-02: this is not the ~1 h job it was billed
   as.* `PopulationFitter.run` has **no NUTS** — its `_method_map` holds only
   the geoVI variants, the two `tier=broken` `native_vi_*`, and
   `mcmc_raytrace`; its own docstring says so. The flat seam that would open
   the other backends (#1394) is unmerged. NUTS on the shared block is
   reachable only through the canonical `Fitter(ForwardModel.build(
   population=...))` path — which this branch had broken (§9).
4. **Finish partial centering**: the loss-prior correction at
   `loss_functions.py:478`. The principled funnel fix.
5. ~~File the `min_eigenvalue=1e-6` footgun~~ — filed as **#1515**. Worked
   around here with `--min-eigenvalue 1.0`; the library default is unchanged.
6. Acceptance criteria 3 (two-population separation) and 6 (coverage across ≥3
   realizations) are still unrun. Criterion 2 needs multiple realizations —
   single-realization coverage was over-read earlier in this work.

---

## 9. Regression this branch introduced, and the fix (2026-08-02)

**This branch broke the canonical joint hierarchical fit.** Measured on the
same file, same command:

    tests/contract/test_single_hamiltonian_path_probe.py
      main            3 passed
      this branch     2 failed   TypeError: mul got incompatible shapes
                                 for broadcasting: (256,), (3,)

PR #1479 has **zero CI checks**, which is why it went unseen.

**Cause — two fail-open omissions masking each other, and fixing one exposed a
third.** On main, `ForwardModel` did not delegate `ssp_data`, so
`Fitter._build_data_args` raised `AttributeError`, a blanket
`contextlib.suppress` swallowed it, and `_jit_inputs` was never built. Every
fit therefore took the *eager* `predict_photometry` fallback in
`_build_prediction`. That fallback is the only path the population model has
ever supported.

This branch correctly fixes the delegation (#1496 — without it the SSP grid
inlines as 267.6 MB of a 274.6 MB program and XLA is OOM-killed). But
`_jit_inputs` is now built for **every** model, so hierarchical fits route into
`threaded_impl`, the **single-galaxy** orchestrator. Per-galaxy parameters
carry a leading `(N,)` axis, reach scalar component code, and die broadcasting
`(N,)` against `(n_grid,)`.

**Fix**: `_build_data_args` skips `_jit_inputs` when the model wraps a
`PopulationSEDModel`, via a new `_population_sed(model)` predicate that
`_maybe_extract_batched_data` now shares. Single-galaxy fits keep the #1496
threading win; hierarchical fits return to the eager path. Verified: 31 passed
across `test_single_hamiltonian_path_probe.py` and
`test_loss_ssp_threading.py`.

**What the fix does NOT do.** Hierarchical fits still closure-capture the SSP
grid, so they keep paying the baking cost #1496 removed for single-galaxy fits.
Removing it needs the batched-vmap forward (#211), not a wider gate. Anyone
running a large joint fit should expect the compile-memory cliff until then.

**Standing lesson.** A guard that fails open converts a one-line omission into
an invisible cliff, and the population path passed its own contract tests for
as long as the fallback happened to carry it. The tests were not wrong — they
were green on a path nobody intended to be load-bearing.
