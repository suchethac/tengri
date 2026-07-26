# Field-SFH posterior reparameterization: OU state-space (innovations) — design

**Issue:** #1301 · **Follow-up enhancement:** #1333 · **Date:** 2026-07-24
**Labels:** `area:sfh`, `area:inference`, `bug`

---

## ⚠️ CORRECTION (2026-07-24, after implementation) — the proposed remedy does not work

**The OU innovations recursion *is* the Cholesky factor.** Unrolling it gives
`s = M ξ` with `M` lower-triangular and positive on the diagonal, and `M Mᵀ = K`.
The Cholesky factor is the **unique** matrix with those properties, so `M = L`
exactly — measured agreement `≤ 5e-14` against an un-jittered
`np.linalg.cholesky` across n ∈ {16,32,64} × σ ∈ {0.3,0.8} × τ ∈ {1e7,1e8,5e8},
and the induced matrix's strict upper triangle is *exactly* zero.

Consequences:

1. **The ξ→SFH map is numerically identical, so the posterior geometry is
   unchanged.** This change cannot reduce the #1301 divergences, and #1301 stays
   open.
2. The "Mechanism" section below is still correct about *the problem*: `L(σ,τ)`
   does re-orient with τ. The error was in the *remedy* — §"THE THEOREM" already
   proves every **exact** square root of `K(τ)` must carry that τ-dependence
   (the kernels don't commute, so no τ-independent eigenbasis exists). The
   innovations recursion is exact, therefore the theorem applies to it too. The
   theorem was stated and then not applied to the proposed fix.
3. **The only escape remains changing the representation** — the uniform
   linear-time Fourier basis (#1333), which buys zero rotation at the cost of a
   changed prior (circulant boundary ≠ finite-domain OU). That trade is now the
   live option for #1301.
4. **What the implementation does deliver** is real but different from the goal:
   `O(n)` instead of `O(n³)`, and removal of the `1e-6` relative
   positive-definiteness jitter, so the realized prior is the exact `K` rather
   than `K + 1e-6·var·I` (the dense path perturbed every realization by ~1e-5
   relative). It ships on those grounds, not as a geometry fix.

The A/B that appeared to show innovations *worse* (21 vs 3 divergences,
single chain) was measuring chaotic amplification of that ~1e-5 jitter difference
in two non-converged chains (split-R̂ 1.69 and 1.14; σ̂ 0.543 vs 0.403). It is not
evidence in either direction and should not be cited.

**Success criterion 1 below is NOT met and cannot be met by this change.**
Criterion 2 (prior unchanged) is met in the strongest possible sense: not merely
the same distribution, the same map.

---

## Problem

Post-#1271 the GP field latents actually reach the likelihood, so the field
posterior is genuinely 25-D (16 latents + 9 physical). HMC does not converge
cleanly at a reasonable budget: on the 18-cell recovery study (`n_grid=16`, dense
mass, 4 chains, 4000 warmup / 2000 samples, L=80) max R̂ reaches **1.16** and up to
**864 divergences per 8000 draws**, concentrated in the emission-line arm (the
tighter likelihood). χ²/N at the MAP is healthy (0.32–1.06) in all 18 cells, so the
fits find good modes — **this is geometry, not a bad model.** Recovery *values* are
usable; the *uncertainties* are not yet quotable.

### Mechanism (confirmed in current code)

`compute_field_gp` (registry.py:1946) routes the default `drw` field to
`drw_linear_gp_from_xi` (gp_sfh.py:167), which realizes

```
K(σ,τ)_ij = (σ ln10)² exp(−|tᵢ−tⱼ|/τ),   tᵢ = 10^uᵢ      # physical time
s = cholesky(K) · ξ,    ξ ~ N(0, I)
```

The Cholesky factor `L(σ,τ)` is a **dense rotation-and-scaling that re-orients as τ
moves.** The set of ξ the data permits therefore rotates with the sampled
hyperparameter τ. HMC carries one global mass matrix — a single fixed metric — and
cannot track a target whose principal axes rotate. Divergences concentrate in the
line arm because that is where the likelihood is tight enough for the mismatch to
bite. σ enters as a pure scalar (`L = σ·L₀(τ)`), so the σ-funnel is already handled
by non-centering; **the τ-coupling is the rotation**, and that is the whole problem.

## Goal / success criteria

1. **Divergences → ~0** and **max R̂ < 1.01** on the same 18 cells at the study's
   budget (or a documented reduction if a residual remains — with the fix-τ
   diagnostic showing the mechanism was the rotation).
2. **The prior over SFHs is unchanged** — bit-exact, not "close." This is a
   reparameterization of the sampler's coordinates, never a change to the modeled
   distribution. Non-negotiable (the linear-time DRW physics of #865/#874 stands).
3. **No worse, ideally cheaper.** The replacement is O(n), versus the current O(n³)
   dense Cholesky, and drops the `_DRW_CHOLESKY_JITTER` positive-definiteness hack.

## The mathematical constraint that picks the approach

**No single τ-independent basis can diagonalize the DRW covariance family on the
log-age grid.** The kernels `K(τ)` at different τ do not commute, so they share no
eigenbasis; every *exact* square root of `K(τ)` — Cholesky or symmetric — must rotate
with τ. So "exact same prior" and "zero τ-rotation" cannot both be maximal on this
grid. The two ways out:

- **Change the representation** (uniform linear-time grid → Toeplitz → Fourier
  diagonalizes for all τ) buys zero rotation but changes the prior (circulant
  boundary ≠ finite-domain OU). → **deferred to #1333.**
- **Change the square root** to the **OU state-space (innovations) recursion**: DRW
  *is* an exact first-order Markov process, so `s` can be generated by a recursion
  whose τ-dependence is a **per-step scalar** `ρᵢ = exp(−Δtᵢ/τ)` — banded and local,
  not a dense rotation — while reproducing `K(σ,τ)` **exactly.** → **this design.**

## Approach A — OU innovations (the replacement)

Generate the field by the exact OU forward recursion on the time-sorted grid
(`t = 10^u` is ascending because the log-age grid is ascending):

```
s₀   = √var · ξ₀
sᵢ   = ρᵢ · sᵢ₋₁ + √(var (1 − ρᵢ²)) · ξᵢ,     ρᵢ = exp(−Δtᵢ / τ),  Δtᵢ = tᵢ − tᵢ₋₁
var  = (σ ln10)²,      ξ ~ N(0, I)
```

Bias correction `k0_half = var/2` is unchanged (marginal variance is identical).

**Verified same-prior.** A standalone numerical check (job scratch
`verify_ou_innovations.py`) confirms the innovations-implied covariance equals the
exact `K(σ,τ)` to **~1e-16** (machine epsilon) across `n_grid ∈ {16,32,64}`,
`σ ∈ {0.3,0.8}`, `τ ∈ {1e7,1e8,5e8}` yr on the real non-uniform physical-time grid.
This is the bit-exact-same-prior proof the "replace the default" decision requires.

**Why the geometry improves.** τ enters `ξ → SFH` only through the scalar per-step
correlations `ρᵢ(τ)`; the coupling between ξ and τ is sequential/banded, not a dense
rotation. A global mass matrix tracks a banded target far better. This is the
standard non-centered parameterization for AR(1)/OU latent GPs.

**Behavior at the grid extremes** (matches the current model's documented
behavior): young ages `Δtᵢ ≪ τ` → `ρᵢ → 1`, strong correlation; old ages `Δtᵢ ≫ τ`
→ `ρᵢ → 0`, `s` becomes independent draws of variance `var` — "effectively diagonal
at old ages," exactly what the Cholesky path's docstring already notes.

**JIT/grad/vmap.** Implemented with `jax.lax.scan` (O(n) time, O(1) memory,
differentiable). Gradients w.r.t. σ, τ, ξ are all finite (checked analytically at
both `ρ→1` and `ρ→0` limits; a `clip(..., 0, None)` floor inside the `√` guards
against float round-off yielding a tiny negative). No dense matrix ⇒ no Cholesky,
no jitter.

### Non-goal / fallback (measured, not assumed)

Innovations reduces the τ-coupling from a dense rotation to a *banded* one — it does
not drive it to exactly zero. If the 18-cell divergence rate does not collapse under
A, the escalation is **#1333** (linear-time Fourier, zero rotation, at the cost of a
characterized prior change). The decision gate is empirical (below), never assumed.

## Implementation surface

Small and centralized — one dispatch point feeds both inference and mocks.

1. **`src/tengri/components/stellar/sfh/gp_sfh.py`** — add
   `drw_innovations_gp_from_xi(xi, psd_sigma_dex, psd_tau_yr, log_age_grid)` with the
   *same signature and `(gp_x, k0_half)` return* as `drw_linear_gp_from_xi`, using
   the `lax.scan` recursion. Keep `drw_linear_gp_from_xi` (dense Cholesky) in place
   as the **reference oracle** for the equivalence test and the future #1333
   comparison — it is not deleted, just no longer the default path.
2. **`src/tengri/components/stellar/sfh/registry.py`** — route the `field_model ==
   "drw"` branch of `compute_field_gp` (registry.py:1957) to the innovations
   function. This is the single dispatch point used by `forward/sed_model.py`
   (forward pass) and `components/stellar/component.py` (mock generation), so both
   inference and mocks pick up the new coordinates automatically.
3. Retire `_DRW_CHOLESKY_JITTER` usage from the `drw` default path (it remains only
   in the retained-oracle `drw_linear_gp_from_xi`).

## Tests (the same-prior proof lives here)

- **New — covariance equality (the bit-exact guard).** For a grid of `(n_grid, σ,
  τ)`, assert the innovations map's implied covariance equals the dense-Cholesky
  `K(σ,τ)` to ≤1e-10 (marker `regression_bug`). This is the numerical anchor of
  "replace, don't change the prior." Build the implied covariance the cheap way
  (propagate the recursion on the identity columns, or compare `AAᵀ`).
- **Existing #865 property suite must stay green as-is**
  (`tests/regression/bug/test_bug_865_linear_time_drw_field.py`): σ-in-dex,
  linear-time decorrelation, σ→0, ξ=0→gp_x=0, differentiability. These pin the
  *distribution*, which is unchanged. Confirm, do not edit. If any assertion moves,
  that is a signal the swap was not same-prior — investigate, do not "fix the test."
- **New — JIT/grad/vmap smoke** on the innovations function directly:
  `jax.jit`, `jax.grad` (finite, nonzero w.r.t. σ; finite w.r.t. τ), `jax.vmap` over
  a batch of ξ.
- **Neuter check:** revert the registry route (point `drw` back at Cholesky) and
  confirm the covariance-equality test still passes (it compares two square roots of
  the same K, so it must) but the *empirical geometry* comparison below is what
  actually distinguishes them — the covariance test is a same-prior guard, not a
  geometry guard. Keep that distinction explicit in the test docstrings.

## Empirical validation (the geometry proof — run, don't assume)

Run under the OOM watchdog, `PYTHONPATH=<wt>/src`, `JAX_PLATFORMS=cpu`, one fit per
process, `tengri.clear_cache()` between:

1. **Fix-τ diagnostic (the issue's request).** Pin `psd_tau` on the current Cholesky
   default and re-measure the divergence rate on a study cell. If it collapses, the
   rotation is confirmed as the cause — recorded either way.
2. **A vs current, across all recovery cases (per user).** Not a single cell — the
   full recovery matrix, innovations vs dense Cholesky, same seeds/budget per cell:
   - **dimension:** `n_grid ∈ {16, 32, 64}` (and a `128` spot-check if budget allows);
   - **prior regime:** low/high burstiness `σ ∈ {~0.3, ~0.8}` × short/long `τ`;
   - **observable arm:** photometry-only, photometry+emission-lines (the divergent
     arm), and full-spectrum — the three arms the recovery study already exercises.
   For each cell record divergence rate, max R̂, and SFH per-bin 68/95% coverage vs
   truth. Success = divergences → ~0 and R̂ < 1.01 **across the cases**, with recovery
   coverage no worse than the Cholesky baseline. Run sequentially, one fit/process,
   OOM-guarded; the heavy end goes in the companion sweep script, not a notebook loop.
3. **Shrinkage unchanged-or-better.** Per-node `1 − w_post/w_prior` (prior predictive
   must vary *every* parameter the posterior marginalizes over — pinning the SFH
   backbone gives a spurious −10 at old ages) to confirm the reparameterization
   preserved constraint rather than moving width around.

## Migration / docs

- **ξ labeling changes.** The `ξ → SFH` bijection differs (a different square root
  of the same K), so a stored ξ vector maps to a different SFH. SFHs and posteriors
  are unaffected *in distribution*; the field is fit fresh each time and users read
  the SFH, not ξ. Note in the changelog and the `compute_field_gp` docstring; no
  saved-artifact migration is needed.
- Update `drw_linear_gp_from_xi` / `compute_field_gp` docstrings and the field-SFH
  design note to describe the innovations default and cross-link #1333.
- Update the CLAUDE.md gotcha ("field non-centering rotates with psd_tau") to
  "resolved via OU innovations (#1301); zero-rotation variant tracked in #1333."

## Risks

- **Banded ≠ zero coupling.** A may not fully collapse divergences → #1333. Mitigated
  by making the empirical gate explicit and #1333 already filed.
- **Scan cost.** Negligible at `n_grid ≤ 256`; O(n) beats the retired O(n³) Cholesky.
- **vmap over the study.** The recursion vmaps over ξ cleanly (scan carry is
  per-sample); confirmed in the smoke test.
