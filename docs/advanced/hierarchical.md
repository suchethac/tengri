# Hierarchical Inference

Population-level inference with shared PSD parameters. Instead of fitting each
galaxy independently, `PopulationFitter` constrains the burstiness prior
(PSD amplitude and timescale) jointly across a population, returning a
`PopulationPosterior`.

## What is hierarchical PSD inference?

In tengri's stochastic SFH model, each galaxy's SFH is a Gaussian process
governed by two PSD parameters:

- **psd_sigma**: amplitude of SFH fluctuations (burstiness strength)
- **psd_tau_myr**: characteristic timescale in Myr

Single-galaxy photometry poorly constrains PSD parameters — the data cannot
distinguish "high sigma, short tau" from "low sigma, long tau."

Hierarchical inference solves this by sharing PSD parameters across N
galaxies. Each galaxy retains its own dust, metallicity, and SFH
realization, but the burstiness prior is learned from the ensemble.
Constraint improves as sqrt(N).

## Usage

```python
from tengri import Model, Parameters, PopulationFitter, Observation, Photometry
import jax

obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))

spec = Parameters(
    redshift=0.1,
    sfh="field",
    dust=True,
)

ssp = ...  # load SSP data

hfitter = PopulationFitter(
    model_factory=lambda sigma, tau: Model(
        Parameters(redshift=0.1, sfh="field", dust=True),
        ssp,
        observation=obs,
        psd_sigma=sigma,
        psd_tau_myr=tau,
    ),
    galaxies=[
        {"flux_obs": flux_i, "noise": noise_i}
        for flux_i, noise_i in zip(catalog_flux, catalog_noise)
    ],
    psd_sigma_prior=(0.1, 4.0),   # uniform prior bounds
    psd_tau_prior=(1.0, 300.0),   # Myr
)

result = hfitter.run("vi", key=jax.random.PRNGKey(0), n_iterations=25)
```

### Available methods

| Method | Description |
|--------|-------------|
| `vi` | geoVI with CorrelatedFieldMaker (recommended) |
| `vi_linear` | MGVI --- faster per iteration, for very large N |
| `mcmc_raytrace` | Ray Tracing on flat parameter vector |

## The block Gibbs structure

Joint optimization is inefficient because shared PSD couples to every
galaxy. Instead, use a three-block Gibbs scheme:

**Block 1 — Shared PSD** (2 parameters): `psd_sigma`, `psd_tau_myr`. Set
prior for every galaxy's SFH. Nonlinear resampling, 6 samples.

**Block 2 — Per-galaxy physical** (N × ~7 params): dust, metallicity, SFH
backbone. geoVI handles age-dust-metallicity banana.

**Block 3 — Per-galaxy SFH fields** (N × n_grid params): GP latent vectors.
Nearly Gaussian conditional; MGVI (linear) suffices.

Each outer cycle updates all three blocks in sequence with standard resample+update.

## Total parameter dimension

```
D_total = D_shared + N_gal * (D_phys + D_xi)
```

For 100 galaxies with a stochastic SFH model:
- D_shared = 2 (PSD parameters)
- D_phys = 7 (dust, metallicity, SFH backbone per galaxy)
- D_xi = 130 (GP latent vector per galaxy)
- **D_total = 2 + 100 * 137 = 13,702**

The block Gibbs structure makes this tractable by never optimizing more than
~13,000 parameters simultaneously, and using cheap MGVI for the
highest-dimensional block.

## Convergence with population size

The constraint on shared PSD parameters improves as sqrt(N):

| N galaxies | PSD sigma uncertainty | PSD tau uncertainty |
|-----------|----------------------|-------------------|
| 1 | ~100% (prior-dominated) | ~100% |
| 10 | ~30% | ~35% |
| 50 | ~15% | ~18% |
| 100 | ~10% | ~12% |

These are approximate --- actual constraints depend on data quality (S/N, number
of bands) and the true PSD values.

:::{tip}
With N > 50 galaxies, the shared PSD parameters are typically well-constrained
even from photometry alone. Below N ~ 10, consider whether hierarchical
inference adds value over independent fits with a fixed PSD prior.
:::

## Two-population separation

A key validation test: generate two populations with distinct PSD parameters
(e.g., quiescent vs. star-forming) and verify that the hierarchical fitter
recovers both sets of PSD parameters when run on each population separately.

```python
# Population A: quiescent (low sigma, long tau)
hfitter_a = PopulationFitter(
    model_factory=factory,
    galaxies=quiescent_galaxies,
    psd_sigma_prior=(0.1, 4.0),
    psd_tau_prior=(1.0, 300.0),
)
result_a = hfitter_a.run("vi", n_iterations=25)

# Population B: bursty (high sigma, short tau)
hfitter_b = PopulationFitter(
    model_factory=factory,
    galaxies=bursty_galaxies,
    psd_sigma_prior=(0.1, 4.0),
    psd_tau_prior=(1.0, 300.0),
)
result_b = hfitter_b.run("vi", n_iterations=25)

# Compare recovered PSD parameters
print(f"Pop A: sigma={result_a.psd_sigma:.2f}, tau={result_a.psd_tau_myr:.1f} Myr")
print(f"Pop B: sigma={result_b.psd_sigma:.2f}, tau={result_b.psd_tau_myr:.1f} Myr")
```

## When to use hierarchical vs. individual fitting

**Use hierarchical inference when:**
- You want to learn the burstiness prior from data.
- You have a homogeneous population (same redshift, similar masses).
- N > 10 and you care about population-level SFH properties.

**Use `fit_batch` when:**
- You want quick per-galaxy results without population constraints.
- Your sample is heterogeneous (mixed redshifts, mass ranges).
- N < 10 or you already have a well-motivated PSD prior.

## Expected performance

For N = 100 galaxies, D_total = 13,702:

| Component | Dimension | Per-iteration cost |
|-----------|-----------|-------------------|
| Block 1 (shared PSD) | 2 | ~0.01 ms |
| Block 2 (physical) | 700 | ~0.5 ms |
| Block 3 (SFH fields) | 13,000 | ~2 ms |
| **Total per outer cycle** | | **~2.5 ms** |

With 25 outer iterations: ~190 ms runtime (after ~60s one-time compilation).

:::{warning}
Compilation time scales with the number of galaxies because the vmapped forward
model must be traced for N galaxies. For N > 100, expect compile times of several
minutes. The XLA cache at `~/.cache/tengri_jax_cache` persists this across sessions.
:::

## References

- Frank, P., Leike, R., Ensslin, T.A. (2021). "Geometric Variational Inference."
  *Entropy* 23(7):853.
- Edenhofer, G. et al. (2024). "Re-envisioning Numerical Information Field Theory
  (NIFTy.re)." arXiv:2402.16683

See [`11_population.py`](https://github.com/suchethac/tengri/blob/main/notebooks/11_population.py) for `PopulationFitter` with convergence checks and population recovery plots.
