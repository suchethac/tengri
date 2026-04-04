# Convergence Diagnostics

How to know whether your inference results are trustworthy. This page covers
industry-standard diagnostics (Vehtari et al. 2021; Stan/ArviZ/BlackJAX conventions),
tengri's built-in diagnostic tools, and practical guidance for when to worry.

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

ESS measures how many independent samples your chain is worth. Autocorrelated chains
produce many samples but little new information.

- **ESS > 100 per parameter**: minimum for reliable percentile estimates (median, 68% CI).
- **ESS > 400 total**: minimum for stable posterior summaries across repeated runs.
- **ESS < 50**: posterior summaries are unreliable. Run longer chains or switch methods.

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

Divergent transitions occur when NUTS encounters regions of high curvature that
the leapfrog integrator cannot traverse accurately. They signal that the posterior
may be unreliable in those regions.

- **0 divergences**: ideal.
- **< 5%**: investigate but results may still be usable. Check which parameters are
  involved.
- **> 5%**: serious problem. The sampler is systematically missing part of the posterior.
  Consider reparametrization or switching to Ray Tracing.

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
across all samplers. This is a **physical limitation**, not a sampler bug.

The age-dust-metallicity degeneracy creates banana-shaped correlations in these
parameters. The posterior is highly curved, making it hard for any sampler to
explore efficiently. Practical consequences:

- These parameters need 2--5x more samples than others to reach ESS > 100.
- geoVI handles the banana shape better than MGVI (which assumes a Gaussian).
- If only these parameters have low ESS and everything else looks good,
  the results are likely still usable --- just report wider uncertainties.

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

Variational methods do not produce MCMC chains, so ESS and acceptance rate do not
directly apply. Instead, check:

1. **KL convergence**: the Hamiltonian (loss) should decrease monotonically and
   plateau. Oscillation or increase indicates instability.

   ```python
   result = fitter.run("vi", n_iterations=15)
   print(result.diagnostics.get("loss_history"))
   ```

2. **Multi-seed agreement**: run with `n_seeds > 1`. Seeds that disagree in
   Hamiltonian by > 10% suggest multimodality or insufficient iterations.

3. **Comparison to MCMC**: when validating, run Ray Tracing or NUTS and compare
   posteriors. If geoVI and RT agree, the variational approximation is adequate.

:::{note}
For geoVI, check KL convergence across iterations and compare to RT posteriors
when possible. This is the most reliable way to validate variational results.
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
- 1--2 parameters (especially dust/metallicity) have ESS slightly below 100,
  but all others are well above. The age-dust-metallicity degeneracy is the cause.
- RT acceptance rate is 45% instead of the "ideal" 50%. Any value in 30--70% is good.
- chi2/dof is 1.3 for a 5-band photometric fit. Photometric calibration uncertainty
  often inflates chi2 slightly.

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

Generate model predictions from posterior samples and verify they bracket the data:

```python
for i in range(10):
    sample = {k: v[i] for k, v in result.samples.items()}
    pred = model.predict_photometry(sample)
    # pred should bracket the data within noise
```

If posterior predictions systematically miss certain bands, the model may be
missing physics (e.g., AGN contribution, nebular emission).

## References

- Vehtari, A. et al. (2021). "Rank-normalization, folding, and localization."
  *Bayesian Analysis* 16(2):667--718.
- Behroozi, P. (2025). "Ray Tracing Sampler." Apache 2.0 license.
- Frank, P., Leike, R., Ensslin, T.A. (2021). "Geometric Variational Inference."
  *Entropy* 23(7):853.

See {doc}`../demonstrations/index` for worked examples that include convergence checks.
