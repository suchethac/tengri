# Standardized Model Redesign

**Date:** 2026-03-15
**Status:** Design proposal
**Goal:** One elegant solution for sampling correlated fields jointly with their kernel parameters, supporting arbitrary user-specified priors, and unifying individual + hierarchical inference.

---

## Problem

The current code has three separate inference paths:
1. `Fitter._build_loss_fn` — builds `½χ² + ½ξᵀξ + prior_penalties(θ)` with special cases for Gaussian/LogUniform priors
2. `HierarchicalFitter._run_geovi_cfm` — uses NIFTy's CorrelatedFieldMaker with a different generative model
3. `HierarchicalFitter._run_raytrace` — flat parameter vector with yet another loss construction

Each has different handling of PSD parameters, prior terms, and transforms. Making PSD params free requires switching code paths. Adding a new prior distribution requires editing the loss builder.

## Design Principle

**Full standardization:** Every parameter — physical, PSD, and latent field — is represented by a standardized latent variable `ξ ~ N(0, I)`. The user's chosen prior distribution defines the **transform** `ξ → θ`, not a penalty term. The loss is always:

```
H(ξ) = ½ χ²(data, f(ξ)) + ½ ξᵀξ
```

No special prior terms. No per-distribution cases. The physics and priors are absorbed into `f(ξ)`.

---

## User-Facing API

### Defining priors (unchanged)

```python
spec = ParamSpec(
    sfh_alpha        = Uniform(0.5, 3.0),
    sfh_beta         = Uniform(0.3, 2.0),
    sfh_tau_peak_gyr = Gaussian(4.0, 1.0, lo=0.5, hi=10.0),
    sfh_peak_sfr     = LogUniform(0.1, 50.0),
    met_logzsol      = Gaussian(-0.5, 0.3, lo=-2.0, hi=0.0),
    dust_tau_bc      = Uniform(0.0, 2.0),
    dust_tau_diff    = 0.3,        # Fixed
    dust_slope       = -0.7,       # Fixed
    psd_sigma        = Uniform(0.1, 4.0),   # FREE — will be inferred
    psd_tau_myr      = LogUniform(1.0, 300.0),  # FREE — will be inferred
    redshift         = 0.1,        # Fixed
    stochastic       = True,
    n_grid           = 128,
)
```

The user specifies priors using the same Distribution objects as before. Each distribution defines both the sampling behavior AND the standardized transform.

### Building the model

```python
model = Model(spec, ssp_data, filters=filters)
```

No change. Model knows which params are free vs fixed from the spec.

### Sampling from the model (new)

```python
# Sample from the prior (for mock generation) — already exists
params = spec.sample(key)
mock = model.mock(params, snr=20.0, key=noise_key)

# Sample from the POSTERIOR (new unified interface)
posterior = model.sample_posterior(
    data=mock.flux_obs,
    noise=mock.noise,
    method="raytrace",       # or "nuts", "geovi", "mgvi", "map"
    data_type="photometry",
    n_burnin=100,
    n_steps=300,
    key=key,
)

# Same interface for hierarchical — just pass multiple galaxies
posterior = model.sample_posterior(
    data=[gal1.flux_obs, gal2.flux_obs, ...],
    noise=[gal1.noise, gal2.noise, ...],
    method="geovi",
    n_iterations=20,
    key=key,
)
# Automatically detects list → hierarchical mode
# Shared params: whatever the spec has as free PSD params
# Per-galaxy params: everything else (SFH, dust, ξ_field)
```

### Accessing results

```python
# Physical parameter posteriors
posterior.summary()           # median ± 68% CI for all params including PSD
posterior.samples["psd_sigma"]  # array of σ_PSD samples
posterior.samples["psd_tau_myr"]  # array of τ_PSD samples

# Derived quantities
posterior.derived["stellar_mass"]

# SFH prediction
model.plot_sfh_posterior(posterior, true_params=true_params)

# Diagnostics
posterior.effective_sample_size()
posterior.diagnostics_summary()
```

---

## Distribution → Standardized Transform Mapping

Each `Distribution` subclass defines a `standardize` and `unstandardize` method:

```python
class Uniform(Distribution):
    """Uniform(lo, hi): ξ → lo + (hi - lo) · sigmoid(ξ)"""

    def standardize(self, theta):
        """Physical → standardized (for initialization)."""
        u = (theta - self.lo) / (self.hi - self.lo)
        return jnp.log(u / (1 - u))  # logit

    def unstandardize(self, xi):
        """Standardized → physical (inside forward model)."""
        return self.lo + (self.hi - self.lo) * jax.nn.sigmoid(xi)


class Gaussian(Distribution):
    """Gaussian(μ, σ, lo, hi): ξ → clip(μ + σ·ξ, lo, hi)"""

    def unstandardize(self, xi):
        return jnp.clip(self.mu + self.sigma * xi, self.lo, self.hi)


class LogUniform(Distribution):
    """LogUniform(lo, hi): ξ → exp(log(lo) + (log(hi)-log(lo)) · sigmoid(ξ))"""

    def unstandardize(self, xi):
        log_lo, log_hi = jnp.log(self.lo), jnp.log(self.hi)
        return jnp.exp(log_lo + (log_hi - log_lo) * jax.nn.sigmoid(xi))


class LogNormal(Distribution):
    """LogNormal(μ, σ): ξ → exp(μ + σ·ξ)"""
    # NEW distribution — natural for PSD amplitude

    def unstandardize(self, xi):
        return jnp.exp(self.mu + self.sigma * xi)


class Fixed(Distribution):
    """No ξ needed — returns constant."""
    is_fixed = True
```

**Important:** Because the prior is absorbed into the transform, the loss function needs **no special per-distribution terms**. The implicit prior induced by the transform + N(0,1) on ξ matches the user's specified distribution.

For `Uniform`: sigmoid(N(0,1)) is approximately uniform — close enough for inference, exact in the limit. This is what we already do.

For `Gaussian`: μ + σ·N(0,1) = N(μ,σ²) — exact.

For `LogUniform`: sigmoid transform in log space — matches.

For `LogNormal`: exp(μ + σ·N(0,1)) — exact.

---

## Internal Architecture

### StandardizedForwardModel

The core new class. Takes a `ParamSpec` and `Model`, builds the complete `ξ → observables` mapping:

```python
class StandardizedForwardModel:
    """Maps ξ ~ N(0, I) → predicted observables.

    The ENTIRE prior structure is absorbed into this mapping.
    Loss is always H(ξ) = ½χ² + ½ξᵀξ.
    """

    def __init__(self, model: Model):
        self.model = model
        self.spec = model.spec

        # Build the ξ → params mapping
        self._free_names = self.spec.free_params  # includes PSD if free
        self._fixed_values = self.spec.get_fixed_values()
        self._transforms = {}  # name → Distribution (for unstandardize)

        for name in self._free_names:
            self._transforms[name] = self.spec.get_distribution(name)

        self._stochastic = self.spec.stochastic
        self._n_grid = self.spec.n_grid
        self._psd_free = ("psd_sigma" in self._free_names
                          or "psd_tau_myr" in self._free_names)

    @property
    def domain(self) -> dict:
        """Standardized parameter domain.

        Returns dict of {name: shape} for all ξ components.
        Compatible with NIFTy's ShapeWithDtype if needed.
        """
        d = {}
        for name in self._free_names:
            d[name] = ()  # scalar
        if self._stochastic:
            d["psd_xi"] = (self._n_grid,)  # field latent
        return d

    def xi_to_params(self, xi: dict) -> dict:
        """Map standardized latents → physical parameters.

        This is the core transform. Each ξ_name is mapped through
        the corresponding distribution's unstandardize method.
        The PSD enters the correlated field through √P(σ,τ)·ξ_field.
        """
        params = {}

        # Physical params: ξ → bounded via distribution transform
        for name in self._free_names:
            if name == "psd_xi":
                continue
            params[name] = self._transforms[name].unstandardize(xi[name])

        # Fixed params
        params.update(self._fixed_values)

        # Correlated field: if PSD is free, √P depends on current (σ,τ)
        if self._stochastic:
            sigma = params.get("psd_sigma", self._fixed_values.get("psd_sigma"))
            tau = params.get("psd_tau_myr", self._fixed_values.get("psd_tau_myr"))

            # Build √P from current PSD params
            # Then: x(t) = IFFT(√P(σ,τ) · ξ_field)
            # This is where the PSD-field coupling lives
            sqrt_P = compute_sqrt_power_drw(sigma, tau * 1e6, self._n_grid, ...)
            x_field = jnp.fft.irfft(sqrt_P * xi["psd_xi"], n=self._n_grid)

            # Store the CORRELATED field, not the white noise
            params["_correlated_field"] = x_field
            # Model uses this directly instead of recomputing √P·ξ

        return params

    def __call__(self, xi: dict) -> jnp.ndarray:
        """Full forward model: ξ → predicted observables."""
        params = self.xi_to_params(xi)
        return self.model.predict_photometry(params)

    def params_to_xi(self, params: dict) -> dict:
        """Inverse: physical params → standardized (for initialization)."""
        xi = {}
        for name in self._free_names:
            if name in params:
                xi[name] = self._transforms[name].standardize(params[name])
        if self._stochastic and "psd_xi" in params:
            xi["psd_xi"] = params["psd_xi"]  # already standardized
        return xi
```

### Unified loss function

```python
def build_loss(smodel: StandardizedForwardModel, data, noise):
    """Build the standardized loss. Always the same form."""

    def loss_fn(xi_flat):
        xi = unravel(xi_flat)
        predicted = smodel(xi)
        chi2 = jnp.sum(((data - predicted) / noise) ** 2)
        prior = jnp.sum(xi_flat ** 2)  # ALWAYS just ½ξᵀξ
        return 0.5 * chi2 + 0.5 * prior

    return loss_fn
```

That's it. No special cases. No per-distribution penalty terms. Works for any combination of free/fixed parameters and any distribution type.

### Hierarchical version

```python
def build_hierarchical_loss(smodel, galaxies):
    """Same loss, multiple galaxies, shared PSD ξ components."""

    def loss_fn(xi_flat):
        xi = unravel(xi_flat)  # includes shared + per-galaxy components

        total_chi2 = 0.0
        for i, gal in enumerate(galaxies):
            # Build per-galaxy xi: shared PSD + galaxy's own field + phys
            xi_i = {
                **{k: xi[k] for k in shared_names},  # σ_PSD, τ_PSD
                **{k: xi[f"g{i}_{k}"] for k in per_galaxy_names},  # ξ_field, phys
            }
            predicted = smodel(xi_i)
            total_chi2 += jnp.sum(((gal.data - predicted) / gal.noise) ** 2)

        prior = jnp.sum(xi_flat ** 2)  # ONE prior over ALL latents
        return 0.5 * total_chi2 + 0.5 * prior

    return loss_fn
```

---

## Model Changes Required

The `Model` class needs one small change: accept a pre-computed correlated field instead of always generating its own.

```python
class Model:
    def predict_photometry(self, params):
        # ...
        if "_correlated_field" in params:
            # Use pre-computed field (from StandardizedForwardModel)
            x_field = params["_correlated_field"]
        else:
            # Legacy: compute from psd_xi using fixed PSD (old path)
            x_field = gp_from_xi(params["psd_xi"], self._sqrt_power)
        # ... rest unchanged
```

This is backward-compatible. Existing code that passes `psd_xi` still works. The new standardized path passes `_correlated_field` which bypasses the internal GP generation.

---

## Sampler Integration

Every sampler operates on the same `loss_fn(xi_flat)`:

```python
# MAP
xi_opt = adam_minimize(loss_fn, xi_init)

# Ray Tracing
chain, ll, ap = sample_raytrace(key, xi_init, lambda x: -loss_fn(x), ...)

# NUTS (BlackJAX)
state = blackjax.nuts(lambda x: -loss_fn(x), ...).step(...)

# geoVI (NIFTy) — use StandardizedForwardModel directly as jft.Model
nifty_model = jft.Model(smodel, domain=smodel.domain)
likelihood = jft.Gaussian(data, noise_inv).amend(nifty_model)
samples = jft.optimize_kl(likelihood, ...)

# MGVI — same as geoVI but sample_mode="linear_resample"
```

The `Fitter` class becomes a thin wrapper:

```python
class Fitter:
    def __init__(self, model, data, noise, data_type="photometry"):
        self.smodel = StandardizedForwardModel(model)
        self.loss_fn = build_loss(self.smodel, data, noise)

    def run(self, method, **kwargs):
        if method == "map":
            return self._run_map(**kwargs)
        elif method == "raytrace":
            return self._run_raytrace(**kwargs)
        # ... etc, all use self.loss_fn
```

---

## What Changes vs Current Code

| Component | Current | New |
|-----------|---------|-----|
| Loss function | `½χ² + ½ξᵀξ + Σ prior_penalty(θ_k)` | `½χ² + ½ξᵀξ` (always) |
| Prior handling | Special cases per distribution type | Transform absorbed into forward model |
| PSD free/fixed | Different code paths | Same code, same loss |
| Hierarchical | Separate `HierarchicalFitter` class | Same `StandardizedForwardModel`, shared ξ |
| NIFTy integration | Separate `signal_response` builder | `StandardizedForwardModel` IS the jft.Model |
| Adding new distribution | Edit loss builder + add penalty | Just implement `unstandardize()` |

## What Stays the Same

- `ParamSpec` API — unchanged
- `Model` API — unchanged (one small addition: accept `_correlated_field`)
- `Posterior` API — unchanged
- Distribution classes — add `standardize`/`unstandardize`, keep everything else
- All existing tests — backward compatible
- Notebooks — no changes needed

---

## New Distribution: LogNormal

For PSD amplitude, a LogNormal prior is more natural than Uniform:

```python
spec = ParamSpec(
    psd_sigma = LogNormal(mu=0.0, sigma=0.8),  # centered on exp(0)=1, spread ~0.8 dex
    psd_tau_myr = LogNormal(mu=3.9, sigma=1.0),  # centered on exp(3.9)≈50 Myr
    ...
)
```

This maps directly to the NIFTy CorrelatedFieldMaker convention where `fluctuations` is lognormal.

---

## Implementation Order

1. Add `standardize()` / `unstandardize()` to all Distribution classes
2. Implement `StandardizedForwardModel`
3. Add `_correlated_field` path to `Model`
4. Refactor `Fitter` to use `StandardizedForwardModel` + simplified loss
5. Refactor `HierarchicalFitter` to reuse the same components
6. Add `model.sample_posterior()` convenience method
7. Add `LogNormal` distribution
8. Run all tests, verify backward compatibility
9. Update notebooks and paper

---

## Custom Transforms and Flexibility

### Any-prior protocol

Users can provide ANY prior by implementing the `Distribution` protocol — just two methods:

```python
class MyCustomPrior(Distribution):
    """Example: truncated Cauchy prior."""

    def __init__(self, loc, scale, lo, hi):
        self.loc, self.scale = loc, scale
        self.lo, self.hi = lo, hi
        self.bounds = (lo, hi)
        self.is_fixed = False

    def unstandardize(self, xi):
        """ξ ~ N(0,1) → θ in physical space.

        This is the ONLY method needed for inference.
        Must be JAX-differentiable.
        """
        # Cauchy quantile function applied to Gaussian CDF
        u = jax.nn.sigmoid(xi)  # (0, 1)
        theta = self.loc + self.scale * jnp.tan(jnp.pi * (u - 0.5))
        return jnp.clip(theta, self.lo, self.hi)

    def sample(self, key):
        """Draw from the prior. Used for mock generation."""
        xi = jax.random.normal(key)
        return self.unstandardize(xi)

    def log_prob(self, x):
        """Log probability. Used for diagnostics only (not inference)."""
        return -jnp.log(self.scale * (1 + ((x - self.loc) / self.scale)**2))

# Usage — works with any Distribution:
spec = ParamSpec(
    sfh_alpha=MyCustomPrior(1.5, 0.5, 0.5, 3.0),  # custom prior
    dust_tau_bc=Uniform(0.0, 2.0),                  # built-in
    met_logzsol=Gaussian(-0.5, 0.3, lo=-2.0, hi=0.0),  # built-in
    ...
)
```

The key contract: **`unstandardize(xi)` must be a differentiable JAX function** that maps N(0,1) → physical parameter space. That's it. The loss function doesn't need to know what the transform is.

### What about non-standardizable priors?

Some priors can't be exactly represented as transforms of N(0,1). For example:
- A delta-function mixture (bimodal)
- A prior with discontinuities
- A prior known only as samples (empirical)

For these, the user can still provide an approximate `unstandardize` that captures the main features, or use a more flexible approach:

```python
class EmpiricalPrior(Distribution):
    """Prior from samples — uses quantile interpolation."""

    def __init__(self, samples, lo, hi):
        self.quantiles = jnp.sort(jnp.array(samples))
        self.lo, self.hi = lo, hi

    def unstandardize(self, xi):
        u = jax.nn.sigmoid(xi)  # map to (0,1)
        # Linear interpolation through empirical quantiles
        idx = u * (len(self.quantiles) - 1)
        idx_lo = jnp.floor(idx).astype(int)
        frac = idx - idx_lo
        return self.quantiles[idx_lo] * (1 - frac) + self.quantiles[idx_lo + 1] * frac
```

### SED model flexibility guarantee

The standardized approach does NOT constrain the SED model in any way:

1. **Any SFH parametrization** — the mean SFH (DPL, delayed-τ, non-parametric bins) is controlled by physical params with arbitrary priors
2. **Any dust model** — Charlot & Fall, Calzetti, or custom, parametrized however the user wants
3. **Any PSD model** — DRW, Matérn, broken power law, or user-defined P(ω). The correlated field generation `x = IFFT(√P · ξ)` works with any PSD function
4. **Any SSP library** — FSPS, BPASS, BC03, ProGeny. Just load different HDF5 templates
5. **Any observation type** — photometry, spectroscopy, joint. The `data_type` parameter selects the forward model branch
6. **Free or fixed anything** — any parameter can be free (with any prior) or fixed. PSD params, redshift, metallicity — all treated uniformly

The standardized model is a **wrapper** around the existing Model, not a replacement. It adds the ξ → params mapping on top. The Model's predict_photometry / predict_spectrum / predict_sfh methods are unchanged.

### Composable transforms

For power users, transforms can be composed:

```python
# Hierarchical: population mean + per-galaxy scatter
class HierarchicalGaussian(Distribution):
    """θ_i = μ + σ · ξ_i, where μ and σ are also free."""

    def __init__(self, mu_prior, sigma_prior):
        self.mu_prior = mu_prior      # Distribution for population mean
        self.sigma_prior = sigma_prior  # Distribution for population scatter

    def unstandardize(self, xi_dict):
        mu = self.mu_prior.unstandardize(xi_dict["mu"])
        sigma = self.sigma_prior.unstandardize(xi_dict["sigma"])
        return mu + sigma * xi_dict["individual"]
```

This enables fully hierarchical models where the prior itself has free parameters — going beyond the simple shared-PSD case.

---

## Why This Is Elegant

1. **One loss function** for everything: individual, hierarchical, any sampler
2. **Prior = transform**, not penalty: adding a new distribution means implementing one function
3. **PSD coupling is natural**: `√P(ξ_σ, ξ_τ) · ξ_field` — the kernel depends on other latents through the forward model
4. **Hierarchical is trivial**: share some ξ components, sum χ² over galaxies
5. **Any sampler works**: the latent space is always N(0,I)
6. **User API is clean**: specify priors with Distribution objects, call `model.sample_posterior()`
7. **NIFTy-compatible**: `StandardizedForwardModel` doubles as a `jft.Model`
