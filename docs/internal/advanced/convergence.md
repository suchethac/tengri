# Convergence Diagnostics

How to know whether your inference results are trustworthy: the standard
diagnostics (Vehtari et al. 2021; Stan / ArviZ / BlackJAX conventions),
tengri's built-in tools, and when to worry.

## Standard thresholds

| Diagnostic | Threshold | Applies to |
|-----------|-----------|------------|
| ESS (bulk) | > 100 per parameter, > 400 total | Ray Tracing, NUTS |
| Divergences | 0 ideal; > 5% = serious | NUTS only |
| RT acceptance rate | 30--70% ideal; > 90% = barely moving | Ray Tracing only |
| NUTS acceptance rate | ~80% | NUTS only |
| chi2/dof | ~1.0 | All methods |

These thresholds follow Vehtari et al. (2021), "Rank-normalization, folding, and
localization," *Bayesian Analysis* 16(2):667--718 --- the same standards used by
Stan, ArviZ, and BlackJAX.

## Effective sample size (ESS)

ESS measures effective independent samples. Autocorrelated chains yield many
samples but little new information.

- **ESS > 100 per parameter**: minimum for reliable percentiles (median, 68% CI).
- **ESS > 400 total**: minimum for stable summaries across runs.
- **ESS < 50**: unreliable. Run longer or switch methods.

```python
result = fitter.run("mcmc_raytrace", n_steps=2000, n_burnin=200)

# Per-parameter ESS
ess = result.effective_sample_size()
for name, val in ess.items():
    if not name.startswith("psd_xi"):  # skip GP latent vector
        print(f"  {name}: ESS = {val:.0f}")
```

:::{tip}
The GP latent vector `psd_xi` has many components with individually low ESS ---
that is expected. Focus on the physical parameters when assessing convergence.
:::

## Divergences (NUTS only)

Divergent transitions signal high-curvature regions the leapfrog integrator
cannot traverse accurately, indicating potentially unreliable posteriors in
those regions.

- **0 divergences**: ideal.
- **< 5%**: investigate; may still be usable. Check involved parameters.
- **> 5%**: serious. Sampler systematically misses part of posterior.
  Reparametrize or switch to Ray Tracing.

```python
result = fitter.run("mcmc_nuts", n_warmup=500, n_burnin=50)
n_div = result.diagnostics.get("n_divergent", 0)
n_total = result.diagnostics.get("n_samples", 1)
print(f"Divergences: {n_div}/{n_total} ({100 * n_div / n_total:.1f}%)")
```

## Acceptance rate

### Ray Tracing

The acceptance rate measures how often proposals are accepted in the
Metropolis-Hastings step.

| Range | Interpretation | Action |
|-------|---------------|--------|
| < 20% | Stuck --- step_size too large | Reduce `step_size` |
| 30--70% | Good | None needed |
| > 90% | Barely moving --- step_size too small | Increase `step_size` |

:::{warning}
For stochastic SFH models (D ~ 137), there is a sharp viability cliff at
`step_size ~ 0.06` where acceptance drops from ~98% to 0%. Use `step_size=0.05`
with `n_leapfrog_steps=50` and compensate with more samples (`n_steps=2000`).
:::

### NUTS

NUTS auto-tunes the step size during warmup to target ~80% acceptance. If the
reported acceptance rate deviates significantly, the warmup may have failed.

## Known difficult parameters

`dust_tau_bc`, `dust_tau_diff`, and `met_logzsol` consistently show low ESS
across all samplers due to age-dust-metallicity degeneracy (physical
limitation, not a bug).

Banana-shaped correlations create highly curved posteriors, making efficient sampling difficult:

- Require 2–5× more samples than others to reach ESS > 100.
- geoVI handles the shape better than MGVI.
- If only these parameters have low ESS while others look good, results are
  likely usable — report wider uncertainties.

## Using the diagnostic functions

tengri provides two functions in `notebooks/_plot_style.py` for standardized
convergence checking.

### `convergence_check()` --- single result

```python
from _plot_style import convergence_check

result = fitter.run("mcmc_raytrace", n_steps=2000, n_burnin=200)
info = convergence_check(result, method_name="RT")
```

This prints a formatted report:

```
============================================================
  Convergence diagnostics: RT  [CONVERGED]
============================================================
  ESS (min / median): 142 / 387
  Acceptance rate:    52.3%
  All diagnostics passed.
============================================================
```

The returned dict contains:
- `info["converged"]` --- boolean, `True` if all checks pass
- `info["warnings"]` --- list of warning strings
- `info["ess_min"]`, `info["ess_median"]` --- ESS summary
- `info["n_params_low_ess"]` --- count of parameters below ESS threshold

### `convergence_table()` --- comparing methods

```python
from _plot_style import convergence_table

results = {
    "RT": result_rt,
    "geoVI": result_geovi,
    "NUTS": result_nuts,
}
all_info = convergence_table(results)
```

Output:

```
Method          ESS min  ESS med   Accept   Diverg     Status
---------------------------------------------------------------
RT                  142      387      52%        —         OK
geoVI                —        —        —        —         OK
NUTS                 89      312      81%        0       WARN

Warnings:
  [NUTS] Low ESS: 2/7 params below 100 (min ESS = 89 for dust_tau_bc)
```

## geoVI / MGVI diagnostics

Variational methods don't produce MCMC chains, so ESS and acceptance don't apply. Check:

1. **KL convergence**: loss should decrease monotonically and plateau.
   Oscillation or increase signals instability.

   ```python
   result = fitter.run("vi", n_iterations=15)
   print(result.diagnostics.get("loss_history"))
   ```

2. **Multi-seed agreement**: with the native backends
   (`native_vi_nonlinear`, `native_vi_linear`), pass `n_seeds > 1`
   (auto-set to 5 when omitted). The NIFTy `"vi"` driver has no
   `n_seeds` parameter — `run("vi", n_seeds=...)` raises `TypeError`;
   rerun it with different `key=` values instead. > 10% Hamiltonian
   disagreement suggests multimodality or insufficient iterations.

3. **MCMC comparison**: run Ray Tracing or NUTS and compare posteriors.
   Agreement validates the variational approximation.

:::{note}
Check KL convergence and compare geoVI to RT posteriors when possible for reliable validation.
:::

## Chi-squared per degree of freedom

The most basic goodness-of-fit diagnostic, available for all methods:

```python
result = fitter.run("vi", n_iterations=15)
print(f"chi2/dof = {result.diagnostics['chi2_dof']:.2f}")
```

| chi2/dof | Interpretation |
|----------|---------------|
| ~1.0 | Good fit --- residuals consistent with noise |
| < 0.5 | Overfitting or overestimated noise |
| 2--5 | Mediocre fit, may need more iterations or wider priors |
| > 5 | Poor fit --- model mismatch or bad initialization |

## Parameters at prior bounds

The fitter checks if any parameter's posterior median is within 2% of its prior
bounds. This often indicates the data prefers values outside the allowed range.

```python
# Printed automatically when verbose=True:
# "Parameters near bounds: dust_tau_bc, met_logzsol. Consider widening the prior."
```

If you see this warning, widen the prior in your `Parameters` and re-fit.

## When to worry vs. when it is OK

**Likely fine:**
- 1–2 parameters (especially dust/metallicity) have ESS slightly below 100;
  others well above. Age-dust-metallicity degeneracy.
- RT acceptance rate 45% instead of "ideal" 50%. Any value in 30–70% is good.
- chi2/dof = 1.3 for 5-band photometry. Photometric calibration uncertainty inflates chi2 slightly.

**Investigate:**
- ESS < 50 for *any* physical parameter.
- RT acceptance < 20% or > 90%.
- chi2/dof > 5 --- the model is not fitting the data.
- geoVI loss oscillates or increases in the last few iterations.

**Serious:**
- NUTS divergences > 5%.
- Multiple parameters piled up at prior bounds.
- geoVI and RT posteriors disagree qualitatively (different modes).

## Posterior predictive check

Generate predictions from posterior samples and verify they bracket the data:

```python
for i in range(10):
    sample = {k: v[i] for k, v in result.samples.items()}
    pred = model.predict_photometry(sample)
    # pred should bracket data within noise
```

Systematic misses in certain bands suggest missing physics (e.g., AGN, nebular emission).

## References

- Vehtari, A. et al. (2021). "Rank-normalization, folding, and localization."
  *Bayesian Analysis* 16(2):667--718.
- Behroozi, P. (2025). "Ray Tracing Sampler." Apache 2.0 license.
- Frank, P., Leike, R., Ensslin, T.A. (2021). "Geometric Variational Inference."
  *Entropy* 23(7):853.

Worked examples with convergence checks: [`00_quickstart.py`](https://github.com/suchethac/tengri/blob/main/notebooks/00_quickstart.py),
[`11_population.py`](https://github.com/suchethac/tengri/blob/main/notebooks/11_population.py).
