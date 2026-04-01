# How tengri works

A concise explanation of the three ideas at the heart of tengri, without any code.

## 1. The star formation history is a latent field

In a conventional SED fitting code you choose a parametric shape for the star formation history
— an exponential, a delayed-τ, a double power-law — and fit its few parameters to the data.
This is fast but brittle: if the true SFH has a burst or a quench that the parametric form
cannot represent, the fitter forces the data into the wrong shape and returns biased posteriors
for every derived quantity (stellar mass, sSFR, quenching time).

tengri treats the SFH as a continuous 1D random field drawn from an Information Field Theory
(IFT) prior. The prior is defined by a **power spectral density** (PSD) with two parameters:
amplitude σ (how large the fluctuations are) and correlation time τ (how quickly they vary).
Smooth galaxies have small σ; starbursting galaxies have large σ and small τ. The IFT prior
is flexible enough to represent both without committing to either.

The field itself — a vector of ~130 latent GP coefficients, ξ ~ N(0, I) — is the high-dimensional
object that inference must explore. This is why the inference problem has D ≈ 137 free parameters
for a stochastic model, versus D ≈ 7 for a smooth parametric one.

## 2. The forward model maps the field to observables

Given a realization of the SFH field (plus dust, metallicity, AGN, and redshift parameters),
the forward model computes a predicted SED at any wavelength:

1. **SFH → stellar mass formed at each age**: The IFT field is transformed into a SFR(t) via
   an inverse Fourier transform, then multiplied by a smooth mean SFH envelope and integrated
   into stellar mass weights.

2. **Weights × SSP spectra → composite stellar SED**: DSPS multiplies the mass weights by the
   pre-computed simple stellar population (SSP) spectra and sums, giving the stellar continuum.

3. **Dust attenuation**: The two-component Charlot & Fall (2000) model attenuates young stars
   (birth cloud) and all stars (diffuse ISM) separately, using a choice of attenuation curve.

4. **Nebular, AGN, IR emission**: Optional physical components are added to the SED.

5. **Observation model**: The SED is redshifted, convolved with filter transmission curves
   or resampled to a spectroscopic pixel grid, and returned as predicted photometry or a spectrum.

The entire pipeline is written in pure JAX. Every step is differentiable, so gradients of the
log-likelihood with respect to all ~137 parameters flow back from photometry to PSD parameters
via automatic differentiation.

## 3. Inference inverts this mapping

Given observed photometry or a spectrum and its noise, inference finds the posterior distribution
over all physical parameters. Because the forward model is differentiable, gradient-based methods
work across the full high-dimensional space:

- **geovi / native_geovi** (default): Geometric Variational Inference (Frank et al. 2021) fits a
  nonlinear Gaussian approximation in the whitened latent space. Fast and reliable for
  catalog-scale fitting.
- **Ray Tracing** (Behroozi 2025): Exact MCMC with radiance tracking. The gold standard for
  high-dimensional posteriors when compute allows.
- **NUTS**: Hamiltonian Monte Carlo via BlackJAX. Best for low-dimensional validation (D ≲ 30).

The **PSD parameters σ and τ** can be treated as free per-galaxy parameters, or inferred
hierarchically across a population. Hierarchical inference (HierarchicalFitter) recovers the
shared PSD of a galaxy population with uncertainty that shrinks as 1/√N — quantifying
population-level burstiness from photometric data alone.

For a schematic diagram of the full pipeline, see Figure 1 of the tengri paper.
