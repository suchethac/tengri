# Hierarchical SFH-PSD recovery — handoff

**Branch:** `worktree-hierarchical-psd-spec` · **PR:** #1479

**Status.** The estimator is correct — demonstrated against an exact analytic
posterior (§4b): given true per-galaxy draws it recovers σ and τ with no railing
out to N=512. End-to-end it still has a **ceiling on N around 32–64**, above
which the shared posterior jumps to a grid corner. **The cause is now
identified and proven sufficient (§4e): the anisotropy of the per-galaxy ξ
covariance.** Matching its eigenvalue spectrum alone — random eigenvectors —
reproduces the railing to three significant figures; isotropic ξ does not rail.
That anisotropy is **real, not a Laplace artifact** (§4f) — but §4g then shows
anisotropy is **not the mechanism**: B2 recovers on exact posteriors that are
*more* anisotropic still. `p_0` is verified correct to 0.005 nats. Every
component now checks out individually while the composite fails, which leaves
one premise: **the real per-galaxy ensembles are not posteriors under the prior
`p_0` assumes** (§4g, §8).

**ROOT CAUSE FOUND (§4i, #1537): `run_laplace`'s finite-difference Hessian collapses ~13% of per-galaxy posteriors to near-singular; those galaxies carry the entire railing.** Everything below is measured, not assumed. Several sections record conclusions
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
| **4e** | ~~the ξ covariance spectrum~~ — **RETRACTED, method invalid** (synthetic ensembles are not posteriors) |
| **4f** | the anisotropy is **real** — NUTS confirms it; the sampler is not the fix |
| **4g** | **`p_0` is correct; §4e's cause is refuted** — B2 recovers on *more* anisotropic true posteriors |
| **4h** | **the driver: 13% of galaxies have a COLLAPSED ξ posterior and carry the whole tilt** |
| **4i** | **ROOT CAUSE: `run_laplace`'s finite-difference Hessian (#1537)** |

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

> **RESOLVED 2026-08-05 — the N ceiling below was a fitting bug, not a
> statistical one.** The bank was fit at `--n-map-steps 4000`, which leaves
> galaxies at a **non-stationary** point; `run_laplace` then inverted the
> Hessian there and returned a far-too-narrow posterior with nothing raised.
> Refitting with the MAP converged removes the railing entirely and **σ covers
> truth at every N from 4 to 128** (§4i-ter). The tables below are the pre-fix
> record.
>
> **τ, however, is now a sharper problem than this document has been calling
> it.** "Unidentified — it reads the prior" is no longer accurate: on the
> converged bank at N=1024 τ is **11.5–13.7 Myr** against a truth of 150 —
> excluding truth and tightening with N. That is a **biased** estimator, not an
> uninformative one. σ carries a smaller version of the same problem (~3.7%
> high, missing at N ≥ 256). Both width-scaling slopes PASS, so **pooling works
> mechanically and converges to the wrong point.**
>
> §4i-ter has the full curve and, more usefully, the elimination: the MAP, the
> Gaussian approximation, weight degeneracy, collapsed ξ posteriors and the
> fields' own correlation structure are all now excluded **by measurement**.
> What is left is the estimator's reading of the fields.

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

## 4i. ROOT CAUSE — `run_laplace` expands about non-modes (#1537)

> **The mechanism named in this section's first draft was wrong.** It blamed
> `_finite_diff_hessian`. Measured against `jax.hessian` on all nine affected
> galaxies, FD agrees to five or more digits and **both** report the collapse —
> see §4i-bis, which supersedes the "Mechanism" paragraph below. The *symptom*,
> the *downstream consequence* and the *exclusion test* in this section all
> stand; only the attribution changed. The corrected cause is that `run_map`
> returns an under-converged point and `run_laplace` inverts the Hessian there
> without checking that the gradient is zero.

**The collapse of §4h is a Laplace artifact, and it is the root cause of the
railing.** NUTS on the two worst-collapsed galaxies (R̂ ≤ 1.01, 0 divergences):

| gal | method | ξ total | min eigenvalue | tilt | `Z_i` peak τ |
|---|---|---|---|---|---|
| **19** | laplace | 3.49 | **0.000** | **+5.05** | 500.0 |
| **19** | **nuts** | **12.87** | 0.220 | **−0.21** | 10.7 |
| **35** | laplace | 2.81 | **0.001** | **+5.88** | 500.0 |
| **35** | **nuts** | **13.93** | 0.459 | **+0.32** | 73.1 |
| 7 (healthy) | laplace | 12.70 | 0.237 | −0.91 | 13.9 |
| 7 (healthy) | **nuts** | **12.68** | 0.242 | −0.40 | 10.0 |

For collapsed galaxies NUTS returns a normal spectrum and the tilt falls from
+5 to ≈0. For a healthy galaxy the two methods agree to three digits — which is
precisely why §4f, which sampled only galaxies 0–7, concluded there was no
difference.

**Mechanism.** `run_laplace` builds its Hessian by central finite differences on
the compiled gradient (`backends/laplace.py:27`) with step
`h = 1e-5 · max(|θ|, 1)`. The forward model contains piecewise interpolations
(PCHIP, `jnp.interp`) whose **gradients are discontinuous at knots**; a step
straddling a kink produces an enormous spurious second derivative, and since
`cov = H⁻¹` the variance in that direction collapses to ~0. A minimum posterior
variance of 0.000 implies a Hessian eigenvalue of order 10⁶ — impossible when
the data constrain 3–4 modes. An exact `jax.hessian` path exists in the same
function but is only taken when `grad_fn is None`, and `map_dispatch.py:783`
always supplies one. Filed as **#1537**.

**This closes the investigation.** The chain, end to end:

1. the FD Hessian blows up for ~13% of galaxies → near-singular posterior;
2. those ensembles are not posteriors, so B2's identity does not apply to them;
3. their `Z_i` pins to the grid corner (peak τ = 500) with tilt ≈ +5;
4. 17 such galaxies contribute +73.1 nats against 111 healthy ones at −17.0;
5. multiplied by N, that beats the prior between N=32 and N=64 — exactly the
   observed threshold.

Everything else was correctly exonerated along the way: the B2 identity,
`ou_logpdf` (exact to 5e-13), `p_0` and its quadrature (0.005 nats), the implied
interim prior (σ, τ flat; ξ isotropic), Monte Carlo convergence in K, nuisance
degeneracy, and the field reconstruction.

**The fix, in order of cost.**
1. **Detect and refit** — the cheapest, and it needs no library change.
   `eigvalsh(cov(psd_xi)).sum() < 0.4·n_params` or `min < 1e-2` flags the
   collapsed fits from stored draws alone, before any pooling. Refit those with
   `mcmc_nuts` (~200–500 s each; at 13% of a 2048 bank that is ~270 galaxies).
2. **Fix the Hessian** (#1537) — validate FD against `jax.hessian`, or expose
   the exact path through the public API.
3. Re-run the scaling curve on the repaired bank and re-check the N ceiling.

**Confirmed end-to-end.** Excluding the flagged fits (71 of 512, 14%) removes
the railing, and **σ recovers to cover truth**:

| N kept | arm | σ 68% (truth 0.75) | τ 68% Myr (truth 150) |
|---|---|---|---|
| 64 | all | 0.958–0.995 | 434.6–491.2 **RAILED** |
| **55** | **healthy** | **0.708–0.811 ✓** | 36.6–64.7 |
| 128 | all | 0.970–0.996 | 472.1–494.7 **RAILED** |
| **109** | **healthy** | **0.702–0.776 ✓** | 42.1–70.7 |

τ stops railing high and instead sits low — which is the *pre-existing*
non-identifiability (§4b: joint NUTS returns τ at 0.98× the prior width) plus
the small-τ preference §4a measured for healthy galaxies, not a new defect.

> ⚠ **Exclusion is not a valid estimator.** Selecting galaxies on an inferred
> quantity biases the population. This is a *mechanism* test — it shows the
> collapsed fits cause the railing. The shipping fix is **refit, not drop**:
> flag with `eigvalsh(cov(psd_xi)).sum() < 0.4·n_params or min < 1e-2`, then
> re-run those galaxies under `mcmc_nuts`. At 14% of a 2048 bank that is ~290
> galaxies at 200–500 s each, roughly 24 h — or much less once #1537 is fixed
> and Laplace is trustworthy everywhere.

## 4i-bis. The corrected mechanism — and it is much cheaper to fix

§4i's attribution to the finite-difference Hessian did not survive being
measured. The verdict rule was written into the probe before it ran: *the fix
is warranted iff the exact Hessian recovers the collapsed galaxies and FD
asymmetry separates the two groups.* Both failed.

| gal | group | FD asym. | exact asym. | **FD covtot** | **exact covtot** | max rel. diff |
|---|---|---|---|---|---|---|
| 1 | healthy | 7.6e-09 | 4.5e-15 | 16.597 | 16.597 | 4.4e-06 |
| 3 | healthy | 7.3e-07 | 5.7e-13 | 20.126 | 20.126 | 9.4e-05 |
| 19 | collapsed | 2.3e-08 | 1.1e-13 | **5.151** | **5.151** | 5.8e-05 |
| 35 | collapsed | 8.1e-07 | 9.0e-13 | **5.591** | **5.591** | 2.9e-04 |
| 61 | collapsed | 2.0e-06 | 4.4e-12 | **5.087** | **5.087** | 1.6e-03 |

FD and `jax.hessian` agree to five digits, and both report the collapse. The
curvature genuinely is that high. Two incidental corrections: `jax.hessian`
costs **0.05 s** on this model, not the 55 s its docstring claims, and the FD
error is small enough that the symmetrization on `laplace.py:135` discards
nothing.

**The real cause: the expansion point is not a mode.** `cov = H⁻¹` is a
covariance only at a stationary point. `run_map` takes a fixed number of Adam
steps with no convergence test, and `run_laplace` accepted whatever came back.

| gal | group | \|grad\| at MAP | loss along tightest eigendirection, −2σ/−1σ/+1σ/+2σ |
|---|---|---|---|
| 1 | healthy | 8.6e-02 | 2.057 / 0.507 / 0.493 / 1.945 |
| 2 | healthy | 1.1e-01 | 1.946 / 0.493 / 0.507 / 2.055 |
| 10 | collapsed | 5.4e+03 | −17.69 / −9.33 / +10.33 / +21.69 |
| 13 | collapsed | 6.0e+04 | −105.4 / −53.2 / +54.2 / +109.4 |
| 19 | collapsed | 3.0e+05 | −245.6 / −123.3 / +124.3 / +249.6 |

A parabola gives 2.0 / 0.5 / 0.5 / 2.0 — the healthy rows match it. The
collapsed rows are pure slope: the loss still *falls* by 245 nats going one way,
so there is no minimum there at all. The high curvature is real, and it is the
curvature of a hillside.

The MAP location is otherwise sound. MAP σ is 0.58–0.72 for collapsed galaxies
against 0.52–0.88 for healthy ones, agrees with the NUTS σ posterior within 2σ
in every case, and has seed-to-seed spread 0.005–0.10. Only convergence differs.

**Raising `n_map_steps` fixes it — no NUTS required.**

| gal | steps | \|grad\| | Newton decrement | ξ covtot |
|---|---|---|---|---|
| 10 | 4 000 | 5.42e+03 | 843.3 | 8.04 |
| 10 | **40 000** | 3.15e-01 | **0.0097** | **16.70** |
| 13 | 4 000 | 6.02e+04 | 179 760 | 7.51 |
| 13 | **40 000** | 4.31e-01 | **0.0219** | **16.57** |
| 19 | 4 000 | 3.02e+05 | 21 219 111 | 5.15 |
| 19 | **40 000** | 2.25e-01 | **0.0125** | **17.15** |
| 35 | 4 000 | 7.75e+04 | 3 073 568 | 5.59 |
| 35 | **40 000** | 5.44e-01 | **0.0008** | **17.70** |

Seconds per galaxy against ~600 s for a NUTS refit. This **replaces** the repair
plan in §4i step 1 and in the box above: the ~24 h estimate for a 2048 bank
becomes minutes. Galaxy 3, nominally healthy, also sat at decrement 1.30 at
4 000 steps — the failure is a continuum, not a category, so the ξ-spectrum flag
was always a proxy for it rather than the thing itself.

**Shipped fix.** `run_laplace` reports the **Newton decrement**
`d = ½ gᵀH⁻¹g` in `Posterior.diagnostics` and warns above 0.1 nat via
`LaplaceNotAtModeWarning`. The decrement rather than `|grad|`: it is invariant
under affine reparameterization, so one threshold means the same thing in every
parameterization, and it is in nats — an offset of δ standard deviations gives
`d = δ²/2`, so 0.1 nat is ≈0.45σ. Measured separation: converged fits score
0.0005–0.075, non-converged ones 1.3 upward. Detection only, never
auto-correction — a Newton step from these points overshoots catastrophically
(galaxy 13's raised the loss by ~1e79).

Pinned by `tests/regression/bug/test_bug_1537_laplace_expansion_point_not_a_mode.py`
on an analytic loss whose curvature grows away from its mode, so the test runs
in the PR gate rather than in the SSP-gated tier.

### The population-level confirmation

The whole N=64 bank refit at `--n-map-steps 40000` (≈7 s per galaxy, ~7 min
total), pooled against the two repair arms and the exclusion reference. Same 64
galaxies in every arm except `healthy-only`, so nothing here is a population
difference.

**Per-galaxy, before pooling: 9 collapsed fits → 0.** Converging the MAP repairs
every one, including galaxy 61, which resisted NUTS at 12× warmup.

| arm | n | σ 68% (truth 0.75) | τ 68% Myr (truth 150) | mode |
|---|---|---|---|---|
| laplace-4k — the bank as measured | 64 | 0.959–0.995 **MISS** | 274.4–424.3 **RAILED** | (1.000, 384) |
| **laplace-40k** | 64 | **0.743–0.812 ✓** | 21.8–34.6 | (0.782, 29) |
| nuts-repaired (8 of 9) | 64 | 0.806–0.905 MISS | 41.0–66.2 | (0.883, 52) |
| healthy-only (biased ref) | 55 | 0.712–0.853 ✓ | 40.0–68.1 | (0.799, 56) |

The railing is gone, and `laplace-40k` gives the **tightest σ interval of any
arm** while covering truth. It beats `nuts-repaired`, which still misses — that
arm carries galaxy 61's unrepaired collapsed fit, the one NUTS could not rescue.
So the cheap fix is not merely as good as the expensive one here; it is better,
because it converges on galaxies NUTS does not.

**Where the arms disagree, stated plainly.** τ is 21.8–34.6 for `laplace-40k`
against 40.0–68.1 for `healthy-only` — all arms miss τ = 150, but they do not
agree with each other either. That is the pre-existing non-identifiability
(§4b: joint NUTS returns τ at 0.98× the prior width), so the τ interval is
largely reading the prior plus the small-τ preference of §4a, and small
differences in the fits move it freely. **Do not read the τ column as a
measurement in any arm.** σ is the identified parameter; see §4a-bis.

**Method note.** The FD accusation was written, argued and filed as an issue
before it was measured. Writing the verdict rule into the probe *first* is what
caught it: had the fix been written before the measurement, swapping in the
exact Hessian would have run clean, changed nothing about the railing, and
shipped a slower default plus a false root-cause note in the paper.

**Repair progress and what the trend actually shows.** Every successful refit
turns a collapsed spectrum into a normal one — 6 for 6 so far:

| gal | Laplace ξ total | refit | note |
|---|---|---|---|
| 10, 12 | 5.75, 4.65 | 14.55, 15.43 | |
| 19, 35 | 3.49, 2.81 | 12.87, 13.93 | |
| 13 | 5.06 | 13.55 | R̂ 1.09 |
| **39** | 4.32 | **14.70** | needed **3× warmup** |
| **42** | 3.11 | R̂ 1.01 | needed **6× warmup** |
| 50, 61 | 2.46, 2.63 | — | resist at 6×; retrying at 12× |

The warmup ladder is itself a finding: default settings repaired five of nine,
3× warmup rescued one more, 6× another. Two still resist. Budget for it — a
production repair pass cannot assume one configuration fits every galaxy.

Pooling at N=64 as the repairs land (all arms K=400):

| repaired | σ 68% (0.75) | τ 68% Myr (150) |
|---|---|---|
| 0 of 9 (all-laplace) | 0.959–0.995 | 274.4–424.3 (high) |
| 4 of 9 | 0.908–0.991 | 134.6–219.6 |
| 6 of 9 | 0.812–0.953 | 72.3–139.9 |
| **7 of 9** | **0.817–0.929** | **53.6–105.1** |
| exclusion, for reference | 0.712–0.853 ✓ | 40.0–68.1 (low) |

**Repair converges to exclusion.** That is the consistency check that matters:
the unbiased fix (refit) and the biased shortcut (drop) agree, which confirms
the collapsed galaxies were the entire railing story and that nothing else
distinguishes them from the healthy population.

> ⚠ **τ is passing THROUGH truth, not converging to it.** An earlier revision of
> this section read the 4-of-9 row as "repair recovers τ". It does not: as more
> fits are repaired the answer moves monotonically toward the healthy-only
> value, which is biased *low*. Crossing 150 on the way from high to low is
> a coincidence of partial repair, and quoting it would be cherry-picking a
> midpoint. **σ** is genuinely converging toward truth (0.959 → 0.908 → 0.812,
> against 0.75). What repair removes is the **railing**; what remains
> underneath is the τ-low preference of §4a on healthy fits, sitting on the
> identifiability wall of §4b.

**These galaxies are hard for NUTS too.** The resistant fits complete in ~6.5 s
with 4 unique draws out of 4000 — one per chain, every chain frozen at its start
— consistent with warmup adapting the step size down against curvature of order
10⁶ until trees terminate at depth 0. Longer warmup rescued gal 39 (3×, R̂ 1.00)
but not 42, 50 or 61. **"Refit with NUTS" is therefore not a complete fix**; a
subset of galaxies needs more than a sampler swap, and the FD Hessian of #1537
is reporting absurd curvature on posteriors whose geometry is genuinely bad.

**Superseded partial reading (4 of 9):** Replacing only galaxies 10, 12,
19 and 35 (the four that already have clean NUTS refits) and leaving the other
five collapsed:

| arm | σ 68% (0.75) | τ 68% Myr (150) | mode |
|---|---|---|---|
| all-laplace | 0.959–0.995 | 274.4–424.3 | (1.000, 384) |
| **repaired (4 of 9)** | 0.908–0.991 | **134.6–219.6 ✓** | (1.000, 198) |
| healthy-only (excluded) | 0.712–0.853 ✓ | 40.0–68.1 | (0.799, 56) |

Repairing fewer than half the collapsed fits already moves τ from missing high
to **covering truth**. σ remains railed, as expected while five broken fits are
still in the pool. Run `scripts/hierarchical_psd_repaired_pool.py` — it reports
which collapsed galaxies are still unrepaired rather than pooling them silently.

> These three arms share `K=400` and so are comparable to each other, but **not**
> to the `K=4000` numbers in §2 (all-laplace reads 274–424 here against 435–491
> there). That is the K-dependence of §4f: the tilt grows with K and saturates
> near K≈1000. Compare within a run, never across.

**Remaining — and it is one command.** The repaired bank (collapsed galaxies
*refit* rather than dropped) has not been built. Blocked on machine contention,
not on anything unknown: a 4-chain `mcmc_nuts` fit peaks at **6.7 GB**, and the
shared-machine guard SIGKILLs at a 15 GB total that other sessions were already
holding at 14–14.9 GB.

The cheapest decisive version needs **five fits**. The railing first appears at
N=64, nine galaxies below index 64 are collapsed
(`10 12 13 19 35 39 42 50 61`), and four already have good refits — 19 and 35 in
`psd_bank_nuts`, 10 and 12 in `psd_bank_repair`. So:

```bash
# when the machine is quiet (needs ~7 GB free; check /tmp/oom_guard_15gb.log)
PYTHONPATH=src:. JAX_PLATFORMS=cpu python scripts/hierarchical_psd_fit_bank.py \
    --n 2048 --out psd_bank_repair64 --only 13 39 42 50 61 \
    --method mcmc_nuts --n-samples 1000 --n-chains 4 --thin 1
```

Then re-pool at N=64 replacing only the nine collapsed fits, keeping every
healthy galaxy on its original Laplace fit, so the comparison isolates the
repair rather than confounding it with a change of method. The all-Laplace arm
gives τ = 434.6–491.2 (railed); the prediction is that the repaired arm does not
rail and σ covers truth, matching the exclusion result.

> ⚠ **Use 4 chains × 1000, not 2 × 2000.** The 2-chain variant was a memory
> workaround and produced **seven dead fits out of nine** — every draw
> bit-identical, zero divergences, full sample count, sane-looking marginals
> (σ 0.62–0.70, |ξ|max 2.3–4.8), and only the ξ covariance total exposed it at
> **0.00** against a healthy ~14. Galaxies 19 and 35 fit cleanly at 4 chains
> (R̂ 1.01, 1.00) and died at 2. The bank script now refuses to checkpoint a
> degenerate fit, so this fails loudly rather than silently — but the
> configuration still matters.
>
> That is the **third** silent dead-chain failure in this work (#1529's MAP
> cache, the first NUTS batch, and this). In all three `n_divergent` was **0**.
> **A zero divergence count is equally consistent with a sampler that took no
> steps at all.**

---

## 4i-ter. The converged bank — σ is fixed, and τ is BIASED, not unidentified

The whole bank refit with `run_laplace` escalating `n_map_steps` while the fit
reports it is not at a mode (`--max-map-escalations`, tripling up to 27×). This
supersedes the flat `40000` of §4i-bis: 10× was one bank's answer, not a rule.

**Per-galaxy, before any pooling** (138 galaxies):

| `n_map_steps` used | galaxies | share |
|---|---|---|
| 4 000 (converged first try) | 63 | 45.7% |
| 12 000 | 21 | 15.2% |
| 36 000 | 51 | 37.0% |
| 108 000 | 3 | 2.2% |

Newton decrement: median 0.0060, **max 0.091 — none above the 0.1 tolerance**.
ξ covariance total: median 13.69, **min 11.96 — none below 6.0**, against 9 of
the first 64 before.

**54.3% of galaxies needed escalation.** §4h's ξ-spectrum flag found ~14%, so
that flag was a proxy for the worst tail, not a census of the defect — which
is what "the failure is a continuum, not a category" means quantitatively. Any
future claim of the form "X% of fits are affected" should be read as "X% were
bad enough for my chosen threshold to notice."

### σ: the railing is gone; a small high bias may remain

| N | σ 68% — pre-fix | σ 68% — converged | τ 68% Myr (truth 150) | ESS |
|---|---|---|---|---|
| 4 | 0.692–0.930 ok | 0.680–0.910 ok | 11.5–28.1 | 63.9 |
| 8 | 0.699–0.894 ok | 0.693–0.874 ok | 11.6–27.4 | 78.8 |
| 16 | 0.757–0.944 | 0.740–0.875 ok | 11.9–27.0 | 60.1 |
| 32 | 0.753–0.918 | 0.715–0.816 ok | 12.0–25.4 | 53.9 |
| 64 | **0.958–0.995 MISS** | **0.741–0.813 ok** | 13.2–25.6 | 65.8 |
| 128 | **0.970–0.996 MISS** | **0.741–0.795 ok** | 12.5–20.2 | 53.9 |
| 256 | **0.985–0.997 MISS** | **0.765–0.803 MISS** | 12.5–17.7 | 45.0 |
| 512 | **0.986–0.997 MISS** | **0.771–0.798 MISS** | 11.2–14.5 | 30.7 |
| 1024 | **0.986–0.997 MISS** | **0.768–0.789 MISS** | 11.5–13.7 | 30.5 |

σ slope **−0.439 ± 0.007**, τ slope **−0.373 ± 0.050** — both PASS the
width-scaling gate. **Pooling works mechanically; it converges to the wrong
point.** That is the one-line summary of this whole section: the widths shrink
as ~1/√N exactly as they should, around a center that is not truth.

The railing that defined §2's "N ceiling" is gone — σ moves from 0.96–0.99 to
0.74–0.79 — and σ covers truth at N = 4…128.

**It misses at N=256, 512 and 1024**, converging on **~0.778, about 3.7%
high**. At N=1024 truth sits 0.018 outside a 0.021-wide interval. That is the
signature of a *biased but consistent* estimator: coverage survives while the
interval is wide enough to swallow the bias, then fails as pooling tightens it.

τ misses at **all nine** N, converging on ~12.6 Myr against a truth of 150.

> ⚠ **Do not score this as "two misses in eight trials".** The N values are
> nested subsets of the *same* galaxies, so they are not independent draws and
> no binomial argument applies. What makes it convincing is the *convergence on
> a fixed wrong value*: a sampling fluctuation does not tighten around 0.784,
> a bias does. The **magnitude** still rests on one realization — coverage
> across ≥3 realizations is acceptance criterion 6 and remains unrun — so quote
> the direction and the mechanism, not "4.5%" as a calibrated number.

An earlier revision of this section called the single N=256 miss "suggestive,
not conclusive" and said not to quote a bias at all. N=512 supersedes that.

### τ: the correction this document owes

**"τ is unidentified — it reads the prior" is no longer accurate**, and it was
the reading this handoff carried from §4b through §4i-bis. On the converged
bank τ *tightens* with N — upper bound 28.1 → 20.2, slope **−0.191 ± 0.050**
excluding zero — converging on **~15 Myr against a truth of 150**.

An unidentified parameter returns the prior width and covers truth by accident.
This one **excludes truth and grows more confident with N**. That is a *biased*
estimator, which is strictly worse than an uninformative one: pooling shrinks
the interval around the wrong value instead of widening it.

Note the sign flip that makes this legible. The pre-fix bank railed **high**
(434–491 Myr at N=64); the converged bank converges **low** (12–20). Same
estimator, opposite direction — so the old railing was never τ's behavior at
all, it was the collapsed fits. What is left underneath is τ's real pathology,
and it was hidden by a louder one the whole time.

ESS is 54–79 throughout, so this is not weight degeneracy.

### τ: confirmed biased, and it is an ESTIMATOR effect

N=256 settled the bias-vs-plateau question: τ keeps tightening (upper bound
28.1 → 20.2 → 17.7), slope **−0.262 ± 0.054**, excluding zero. **7 of 7 N
values miss**, all in the same direction, by an order of magnitude. That is not
the ~1/3 miss rate a 68% interval is entitled to.

**The pooled answer lies far outside what any single galaxy supports.** With
the interim τ prior at 10–500 Myr:

| quantity | value |
|---|---|
| per-galaxy τ posterior medians | **222.9** (p16 151.1, p84 252.6) |
| prior median, if uninformative | 255.0 |
| truth | 150 |
| **pooled τ, N=256** | **12.5–17.7** |

The per-galaxy posteriors *are* mildly informative — pulled from 255 down
toward truth. The pooled result then lands an order of magnitude below every
one of them, in the bottom ~4% of the prior range.

**Note the sign flip.** Pre-fix the pool railed to the **upper** corner
(434–491 against a 500 bound); converged, it concentrates at the **lower**
corner. Two opposite corner-seeking behaviors from one estimator — which
retroactively explains a puzzle in the old data. §2's table shows τ apparently
"covering truth at N=32" (61.9–153.4) *before* the fix. That was never
recovery: it was **two errors of opposite sign partially canceling** at
intermediate N. Fixing the fitting bug unmasked the estimator one.

Two explanations are already excluded: ESS is 45–79 throughout, so it is not
weight degeneracy, and every per-galaxy fit is now provably at its mode, so it
is not §4i's non-stationary expansion.

> A hierarchical posterior *can* legitimately concentrate outside the range of
> the individual point estimates — it is a product of broad likelihoods whose
> shapes may agree somewhere none of them peaks. So this is a strong, localized
> anomaly, not yet a proven estimator bug. It is, however, now the **only**
> open problem, and it is much better localized than "τ is unidentified".

### Where τ's pull comes from — Laplace amplifies it ~2.8×, but does not create it

Per-galaxy tilt toward the short-τ corner, at truth σ, for the 17 galaxies
that have **both** a converged Laplace fit and an `mcmc_nuts` fit:

    tilt_i = log Z_i(sigma_truth, 15 Myr) - log Z_i(sigma_truth, 150 Myr)

| draws | mean tilt [nats/galaxy] | × N=256 |
|---|---|---|
| Laplace | **+1.044** | +267 |
| `mcmc_nuts` | **+0.377** | +97 |

Positive = prefers 15 Myr. **Both do.** The Gaussian approximation roughly
triples the pull but is not its origin — so **refitting the bank with NUTS
would cut τ's bias by about two-thirds and still rail low.** That is worth
knowing before spending the compute: NUTS is ~600 s/galaxy against ~10 s.

> ⚠ **The raw numbers from this probe were −1.258 and −1.925 — the opposite
> sign.** `SharedGrid.log_prior` carries the **quadrature weight**, and the τ
> grid is log-spaced, so that weight scales as τ: `log_prior(15) −
> log_prior(150) = log(15/150) = −2.3026` exactly. Calling
> `shared_log_posterior` once per galaxy adds it once per galaxy instead of
> once, which for 17 galaxies is 16 spurious copies — enough to flip the sign.
> A consistency check (sum of individual tilts vs the pooled surface at the
> same nodes) caught it: −21.39 against +15.45, reconciling exactly as
> 16 × (−2.3026). **Never read a per-galaxy `shared_log_posterior` value
> without subtracting `grid.log_prior`.**

### σ and τ are probably ONE displacement, not two biases

| | σ | τ Myr |
|---|---|---|
| pooled mode (17 galaxies, full grid) | **0.832** | **11.4** |
| truth | 0.750 | 150 |

σ high *and* τ short is exactly the degeneracy direction of the OU kernel
`(σ ln10)² exp(−|Δt|/τ)`: more variance with a shorter correlation time
reproduces the same short-lag power. The σ misses at N=256/512 and τ's bias are
plausibly the same slide along one ridge, which would explain why σ's error
appears only once the interval is tight enough to resolve 4%.

The two displacements are **σ +4.5%, τ −92%** on the converged bank at N=512.

**Tested, and the simple ridge is refuted.** No combination is conserved
between truth and the pooled mode:

| combination | truth (0.750, 150) | mode (0.832, 11.4) | ratio |
|---|---|---|---|
| σ²/τ | 0.00375 | 0.06072 | 16× |
| σ²τ | 84.38 | 7.89 | 10.7× |
| σ² | 0.5625 | 0.6922 | 1.23× |

So it is not a slide along a degeneracy. σ² is closest to invariant but still
23% off.

### Hypothesis (NOT yet measured): prior roughness in the unconstrained modes

The grid geometry is suggestive. Node spacing on the 16-node age grid runs
0.888 Myr to 6.49 Gyr, **median 75.9 Myr** — and the pooled τ of 11.4 Myr sits
*below* that median, with 10 of 15 gaps exceeding it.

A mechanism consistent with every number above, and with the fact that the pull
**survives NUTS** (+0.377 nats/galaxy):

1. Ten broadbands constrain `n_eff ≈ 3–4` of 16 field modes (§4b).
2. The constrained modes are the **smooth**, low-frequency ones — that is what
   broadband colors see.
3. So the posterior shrinks the smooth modes toward their fitted values while
   the ~12 rough modes stay at full prior amplitude.
4. The reconstructed field is then **relatively richer in high-frequency power
   than a genuine OU draw** at the same (σ, τ) — smooth part shrunk, rough part
   not.
5. The estimator infers τ from that roughness and reads the excess as a short
   correlation time; σ rises to absorb the extra variance (+23% in σ²).

Because this is a property of the *posterior* rather than of the Gaussian
approximation, it would survive an exact sampler — which is what the Laplace
vs NUTS tilts show (+1.044 vs +0.377: NUTS reduces it by ~64% but does not
remove it).

> **This is a hypothesis, not a measurement.** It is consistent with the tilt
> decomposition, the non-conserved combinations, the grid spacing, and the
> n_eff count — but none of those tests it directly. The falsifiable prediction
> is that the bias scales with how *few* modes are constrained: adding bands or
> spectroscopy should shrink it. The direct test is to pool fields whose
> unconstrained modes have been re-drawn from the *fitted* OU rather than left
> at prior amplitude, and see whether τ recovers.

### …and the roughness hypothesis is REFUTED

Tested directly with a normalized adjacent-node roughness
`R = <Σ_j (m_{j+1} − m_j)²> / <var(m)>`, on 120 galaxies × 200 draws, against
two matched synthetic controls on the same grid:

| arm | median R | p16 | p84 |
|---|---|---|---|
| bank — reconstructed fields | **22.26** | 18.37 | 25.25 |
| OU at each galaxy's own fitted (σ, τ) | 22.09 | 21.26 | 23.36 |
| OU at truth (0.75, 150 Myr) | 22.37 | 21.68 | 22.96 |

Ratios: bank/ou-fitted **1.007**, bank/ou-truth **0.995**. The reconstructed
fields are *not* rougher than genuine OU draws. The hypothesis above is wrong.

**The statistic is not blind — that was checked.** R against τ at σ=0.75,
20 000 draws:

| τ [Myr] | 5 | 15 | 50 | 150 | 500 | 2000 |
|---|---|---|---|---|---|---|
| R | 28.33 | 26.27 | 24.27 | 22.51 | 20.74 | 18.21 |

Monotone over a 55% range, so a null result here means something. (Worth the
check: R gave nearly the same value at τ=150 and τ=218, which looked like
insensitivity until the full sweep showed otherwise.)

### The contradiction this leaves, which is now the whole problem

The bank's fields score R = 22.26, i.e. they look like OU draws at
**τ ≈ 150–220 Myr** — consistent with their own fitted median of 218 Myr. A
field at τ = 12 Myr would score ≈ 26.5, far outside the bank's p16–p84.

**So the fields carry approximately the right correlation structure, and the
estimator returns 11–14 Myr from them.**

Every other explanation is now excluded by measurement:

| suspect | status |
|---|---|
| non-stationary MAP (§4i) | excluded — all fits provably at a mode |
| Laplace Gaussian approximation | amplifies ×2.8, but NUTS still tilts +0.377 |
| importance-weight degeneracy | excluded — ESS 31–79 |
| collapsed ξ posteriors (§4h) | excluded — 0 of 1024 below covtot 6.0 |
| field correlation structure | **excluded — R matches OU at the fitted τ** |

What remains is the estimator's own reading: the B2 ratio, `p_0`, or the
density evaluation. That collides head-on with §4b ("B2 is sound given exact
posteriors") and §4g ("`p_0` is correct to 0.005 nats"). **One of those
premises must fail, and identifying which is the next step.** The most likely
candidate is that §4b's analytic toy differs from the real case in a way that
matters — it has no nuisance parameters, a linear projection, and an exactly
Gaussian posterior, and the real case has none of those.

Not purely a ridge effect, though: at **fixed** truth σ the pooled surface
still prefers τ=15 over τ=150 by 19.4 nats. Both a genuine short-τ preference
at fixed σ *and* a ridge that carries σ upward.

---

## 4h. THE DRIVER — 13% of galaxies have a COLLAPSED ξ posterior

**The railing is carried by a small subset of broken per-galaxy fits, and the
medians hid it.** Across 128 bank galaxies, split by the total of the ξ
covariance spectrum (the ξ prior is N(0, I), so the prior total is 16):

| group | n | mean tilt | % favoring corner | **sum tilt** |
|---|---|---|---|---|
| **collapsed** (ξ total < 6) | 17 | **+4.301** | **100%** | **+73.1** |
| normal (ξ total ≥ 6) | 111 | −0.153 | 33% | **−17.0** |
| all | 128 | +0.439 | 42% | +56.1 |

`corr(tilt, ξ total) = −0.775`. **Drop the collapsed 17 and the total flips
sign** — the remaining 111 galaxies push *away* from the corner. The
distribution is bimodal: a healthy bulk (median ξ total 13.2, p75 14.0) and a
broken tail reaching **1.9 of 16**.

**The collapse cannot be physical.** A ξ total of 1.9 means the data would have
to constrain ~14 of 16 field directions to well below the unit prior. Ten
broadband filters constrain `n_eff ≈ 3–4` (§4b). So these are not posteriors —
they are a Hessian or MAP artifact of the Laplace fit.

**The bank itself is clean.** Refitting with a **fresh model instance per
galaxy**, which defeats any model-level cache (the failure mode of #1529),
reproduces the stored draws **bit-identically** — σ, τ, ξ std, field std and the
ξ spectrum total all agree to every printed digit for galaxies 3, 7, 19 and 35.
So the collapse is a genuine property of the Laplace fit for those galaxies, not
contamination across the loop.

| gal | σ | τ | ξ total | tilt (§4d) |
|---|---|---|---|---|
| 3 | 0.628 | 238 | 13.72 | — |
| 7 | 0.749 | 180 | 12.70 | — |
| **19** | 0.650 | 350 | **3.49** | **+5.05** |
| **35** | 0.605 | 309 | **2.81** | **+5.88** |

> ⚠ **This invalidates §4f's NUTS comparison, and the reason is worth keeping.**
> §4f concluded "the anisotropy is real, NUTS confirms it" from galaxies 0–7.
> Those galaxies are almost all *normal* — gal 3 is 13.72, gal 7 is 12.70. With
> 13% affected, eight consecutive galaxies had a good chance of containing none
> of the pathology, and they did. **The decisive comparison was run on precisely
> the galaxies where nothing is wrong.**
>
> The medians compounded it. §4e and §4f quoted median spectra throughout;
> a median of 13.2 looks healthy and averages the broken tail away. §4d had
> already reported that a handful of galaxies carries the effect — that finding
> was recorded and then not acted on. **When a diagnostic says a minority drives
> the outcome, sample the minority, not the population.**

**Also eliminated this round: nuisance degeneracy.** The one untested
combination — nuisances *with exact posteriors* — was run by letting the
nuisances enter linearly with a Gaussian prior, so marginalizing them
analytically leaves the likelihood Gaussian in `m` and the posterior a
closed-form mixture. Nuisance directions were a constant offset (degenerate with
total mass) and a log-age ramp (degenerate with the smooth SFH slope):

| `s_nu` | residual | τ 68% at N=256 (truth 150) | ξ spectrum total |
|---|---|---|---|
| 0.0 | 0.150 | 155.7–193.6 | 8.8 |
| 0.5 | 0.288 | 158.7–197.3 | 8.9 |
| 2.0 | 0.940 | 150.5–194.2 | 9.3 |
| **8.0** | **3.255** | **139.2–196.0 ✓** | **10.2** |

At `s_nu = 8` the nuisances dominate the error budget (residual 3.255 against a
noise of 0.15) and the posterior is measurably more anisotropic — and τ *covers
truth*. Nuisance degeneracy does not break B2; it mildly helps. (`s_nu = 0`
reproduces §4b, the known-answer validation this construct had to pass first.)

**Next.** Whether the collapse is a Laplace artifact is a two-galaxy question:
fit 19 and 35 with `mcmc_nuts` and compare their ξ spectra. A normal spectrum
means Laplace is at fault, and the fix is concrete — **the ξ spectrum total is a
per-galaxy red flag computable before any pooling, from data already in every
checkpoint.** Detect, refit, and the pooled estimate may come back.

---

## 4g. `p_0` is correct — and §4e's "cause" does not survive contact with posteriors

Two results, and together they undo §4e while narrowing the problem to one
statement.

**`p_0` is exonerated, four ways.**

| test | result |
|---|---|
| `ou_logpdf` vs explicit multivariate normal, 16 (σ,τ) points | **4.9e-13** — exact |
| p_0 quadrature, 60×60 vs a 240×240 reference | mean **+0.0024** nats |
| p_0 vs 40 000 Monte Carlo draws from the **continuous** interim prior | mean **+0.0047** nats |
| railing vs grid resolution 30 / 60 / 120 | 411–487 / 435–491 / 441–493 — **unmoved** |

Against a tilt of +0.318 nats/galaxy, a 0.005-nat quadrature error is three
orders of magnitude too small. The grid quadrature, the density, and the
normalization are all right, and refining the grid does not move the answer.

**§4e's sufficiency demonstration does not apply to posteriors.** §4e built its
anisotropic ensembles synthetically — a Gaussian with a matched eigenvalue
spectrum and *random* eigenvectors. Such an ensemble is the posterior of no
likelihood, and `Z_i = C_i E_{q_i}[p/p_0]` is exact only when `q_i` *is* a
posterior under `p_0`. B2 owes an arbitrary distribution nothing. Measured on
the §4b toy's **exact** posteriors, in the same ξ convention the bank stores:

| ensemble | spectrum total | above 1.05 | below 0.5 | B2 |
|---|---|---|---|---|
| toy exact posterior, 4 modes | 12.8/16 | 5 | 3 | **recovers** |
| toy exact posterior, 8 modes | **8.8/16** | 3 | **7** | **recovers** |
| real bank (laplace) | 13.6/16 | 2 | 1 | rails |
| real bank (nuts) | 14.4/16 | 4 | 1 | rails |

The toy's genuine posteriors are **more** anisotropic than the real bank's — at
8 modes, seven of sixteen directions below 0.5 against the bank's one — and B2
recovers there. **Anisotropy of a true posterior does not break B2.** §4e
therefore identified a property of non-posterior ensembles, not the cause.

**What that leaves.** Every component is now verified individually:

* the B2 identity is exact for any true `q_i` (algebra);
* `ou_logpdf` is exact to 5e-13;
* `p_0` matches the continuous interim prior to 0.005 nats;
* the Monte Carlo has converged (flat in K past ~1000, §4f);
* B2 recovers on exact posteriors that are *more* anisotropic than the real ones.

And the composite still fails. The only remaining possibility is the one
premise that is measured rather than derived: **the real per-galaxy ensembles
are not posteriors under the interim prior `p_0` assumes** — despite NUTS
reporting R̂ ≤ 1.05. R̂ is necessary, not sufficient; it certifies that chains
agree, not that they sample the intended target.

**The implied prior is correct — candidate 1 eliminated.** Measured by running
a fit with the likelihood switched off (flux errors inflated 1e6x, so the
posterior *is* the prior) and sampling it with the same NUTS backend, 2000
draws:

| | measured | KS vs `Uniform` | |
|---|---|---|---|
| σ | mean 0.50 (expect 0.51) | D=0.029, p=0.072 | FLAT |
| τ | mean 257.1 Myr (expect 255.0) | D=0.017, p=0.59 | FLAT |
| ξ covariance | total 15.8/16, mean 0.989, std 0.995 | — | ISOTROPIC |

So the fits do run under exactly the prior `p_0` assumes: flat in σ, flat in τ
*itself* (no unbounded-coordinate Jacobian leak), and a unit-isotropic field
prior.

> ⚠ **This run also calibrates the ξ spectrum, and corrects an over-reading in
> §4e.** The *prior* ξ spectrum — which is the identity by construction — comes
> back as `1.17 1.17 1.13 … 0.85 0.77 0.74` from 2000 draws. That spread is pure
> sampling noise on a 16×16 covariance. §4e read the bank's four-to-five
> eigenvalues above 1.0 as "the signature of marginalizing over correlated
> nuisances"; at K=1000 those values sit **within the noise floor** and support
> no such reading. The bank's *low* tail (0.25–0.28) is far below the floor and
> is a genuine data constraint. **Always calibrate a spectrum against the
> spectrum of the same number of draws from the known-isotropic prior.**

**Remaining candidates**, cheapest first. Note that the synthetic-substitution
route is now closed (§4e retracted): any future test must compare against
*genuine posteriors*, and must be validated by handing it the truth first.

1. **Nuisance parameters with EXACT posteriors** — the one untested combination.
   §4b used exact draws with no nuisances and recovers; §4c added nuisances but
   with Laplace draws. The real fit has ~8 nuisances alongside 16 field latents,
   and the §4b analytic machinery can be extended to carry them.
2. The SFH truncation. The forward model warns
   `SFHBeforeBigBangWarning: forms 3% of its stellar mass before the Big Bang …
   that mass is truncated` — so the likelihood does not see all of `m`.
2. Nuisance parameters. The §4b toy that recovers has **none**; the real fit has
   ~8 alongside the 16 field latents. Note the combination "exact posteriors
   **and** nuisances" has never been tested: §4c added nuisances but with
   Laplace draws, and §4b used exact draws with no nuisances.
3. The interaction between the ξ ensemble and the per-galaxy (σ, τ) *spread*.
   §4e's synthetic anisotropic ξ was paired with the bank's wide τ spread
   (21–407 Myr) and railed; the toy's exact posteriors carry a narrow spread
   (156–266) and recover. Pair §4e's synthetic ξ with a narrow spread to
   separate the two.

---

## 4f. The anisotropy is real — NUTS confirms it, so the sampler is not the fix

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

## 4e. The ξ covariance spectrum — SUPERSEDED by §4g

> ⚠ **RETRACTED — this section's method is invalid, not merely its conclusion.**
> Every experiment here (and in the follow-ups that used the same trick) built
> ensembles **synthetically**: OU draws at per-draw `(σ_k, τ_k)`, sometimes with
> a shaped ξ covariance. Such an ensemble is the posterior of no likelihood, and
> `Z_i = C_i E_{q_i}[p/p_0]` is exact only when `q_i` IS a posterior under
> `p_0`.
>
> **The control that settles it.** Build every galaxy's ensemble as OU draws at
> *exactly the truth* (σ=0.75, τ=150). B2 returns **τ = 105.3–119.0 at N=256** —
> it does not recover the very value it was handed. The reason is elementary:
> `E_q[p_θ/p_0]` for `q = p_{θ₀}` is `∫ p_{θ₀} p_θ / p_0`, and the `1/p_0`
> reweighting moves the maximum off `θ₀`. **B2 is biased on prior-like ensembles
> by construction.**
>
> So a synthetic ensemble railing proves nothing about the real one, and a
> 2×2 factorial over {isotropic, anisotropic} × {wide, narrow τ spread} confirms
> it: **all four cells are biased** at N=256 (154–198, 194–224, 388–483,
> 255–302). There is no clean arm to compare against.
>
> Positively: B2 **recovers** on genuine posteriors — §4b's analytic exact
> posteriors, and §4g's measurement that those are *more* anisotropic than the
> real bank's (total 8.8/16 at 8 modes, seven directions below 0.5, against the
> bank's one).
>
> Kept because the reasoning error is the sharpest of this investigation. A
> synthetic construct reproduced the symptom **to three significant figures**,
> which reads as proof. Reproducing a symptom with an object outside the
> estimator's domain of validity proves nothing — and the check that would have
> caught it (feed the construct the truth and see whether it comes back) was one
> line away the whole time.

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
- **When a diagnostic says a MINORITY drives the outcome, sample the minority.**
  13% of galaxies carry the entire railing (§4h); §4f's decisive NUTS
  comparison used galaxies 0-7, which contain none of them, and concluded the
  opposite. Medians made it worse — a median ξ total of 13.2 looks healthy while
  the tail reaches 1.9. §4d reported the minority effect and it was recorded,
  then not acted on.
- **Feed a diagnostic construct the truth and check it comes back.** Every
  synthetic-ensemble experiment in §4e was void because B2 is only defined for
  posteriors: handed OU draws at *exactly* (0.75, 150) it returns 105–119, not
  150. One line, and it would have caught a construct that reproduced the
  symptom to three significant figures. **Reproducing a symptom with an object
  outside the method's domain of validity proves nothing.**
- **Calibrate a spectrum against the same number of draws from the KNOWN
  prior.** The prior ξ covariance is the identity by construction, yet 2000
  draws return `1.17 … 0.74`. §4e read the bank's eigenvalues above 1.0 as
  evidence of nuisance marginalization; they are inside that noise floor. Only
  the low tail (0.25) clears it. §4g.
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

> **0. DONE (2026-08-05) — converge the MAP.** §4i-bis: the per-galaxy draws
> were bad because `run_map` returned a non-stationary point and `run_laplace`
> inverted the Hessian there. `--n-map-steps 40000` fixes every affected fit at
> seconds per galaxy. `run_laplace` now reports the Newton decrement and warns
> above 0.1 nat (#1537), and the MAP-init cache is keyed on the data (#1529).
> **Everything below item 0 was written before that was known.** The list is
> kept as a record of the search — several entries chase mechanisms that the
> measurement has since eliminated, and are marked where they do.

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

   **§4g has since verified `p_0` and refuted §4e's mechanism.** `p_0` matches
   the continuous interim prior to 0.005 nats, `ou_logpdf` is exact to 5e-13,
   and refining the grid does not move the answer. B2 recovers on exact
   posteriors *more* anisotropic than the real ones. Every component checks out
   while the composite fails, leaving one premise to test:

   (a) **Find where the fitted model's implied prior on `m` departs from
   `p_0`.** Cheapest first: (i) sample the interim prior directly, with the
   likelihood switched off, and compare the recovered (σ, τ) density against
   `Uniform` — a prior flat in an unbounded coordinate is *not* flat in τ, and a
   τ-shaped prior mismatch is already bug #2 in §3; (ii) the SFH truncation, the
   forward model warns it discards ~3% of the stellar mass before the Big Bang,
   so the likelihood does not see all of `m`; (iii) nuisance parameters — the
   §4b toy that recovers has none, the real fit has ~8.

   (b) ~~Verify `p_0` against the interim prior it is supposed to represent.~~
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
