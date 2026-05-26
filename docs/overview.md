# Overview

Galaxy SEDs carry the cumulative record of stellar mass assembly,
chemical enrichment, and dust processing. Recovering stellar mass,
star-formation history, dust, and metallicity from broadband
photometry and spectra is one of the oldest inference problems in
extragalactic astronomy, and the data volumes coming over the next
decade make the problem harder than the codes built in the 2010s
were designed for.

Tengri is a panchromatic galaxy SED inference library written in
[JAX](https://jax.readthedocs.io) on top of
[DSPS](https://github.com/ArgonneCPAC/dsps). A single forward model
covers stellar populations, dust attenuation and emission, nebular
gas, AGN, the IGM, radio, and X-ray, all driven by one shared set of
physical parameters. The samplers we ship — MAP, Laplace,
Pathfinder, NUTS, ray-tracing MCMC, nested sampling, geoVI, and
hierarchical population fits — all talk to that same model. We
never maintain a fast template-marginalised path next to a slow
Bayesian one, and we never derive a gradient by hand.

This is a community effort. The codebase is open, the physics
modules are being independently human-verified, and contributions —
new SFH families, dust laws, AGN templates, observation modes,
samplers — are welcome. The repository will move to the
`tengri-project` organisation on GitHub shortly, which is where
collaborative development and issue tracking will live going
forward.

## Philosophy

The shape of the codebase is driven by what survey-scale,
high-dimensional galaxy SED inference needs.

Catalogue inference is expensive. Nested sampling typically wants
$\sim 10^6$ likelihood calls per galaxy, so $10^6$ galaxies is
$\sim 10^{12}$ forward evaluations. That is many CPU-years even with
a neural emulator, and the emulator brings its own training cost,
approximation error, and dimensionality ceiling. JIT-compiling the
real physical model gets us down to tens of microseconds per call on
a single core, with no surrogate in the loop.

High-dimensional posteriors need exact gradients. Bursty SFHs
modelled as correlated random fields live in $\sim 100$-parameter
spaces, and hierarchical population fits add more on top. HMC,
variational inference, and Laplace approximation make those regimes
reachable, but only if the forward model is differentiable end to
end. Finite differences are too noisy once interpolation gets
involved, and hand-derived gradients become unmaintainable as soon
as you compose a dust law with a non-parametric SFH and an emulated
nebular spectrum.

The physical models keep changing. New SFH families, new dust laws,
new AGN templates, new nebular grids show up every year. A code that
entangles physics with sampling forces its users to re-derive
sampler-specific quantities every time something in the physics
moves. We want adding a new component to be one file, with no edits
to the inference engine.

That gives the codebase a consistent shape:

The forward model is the artifact. Inference is a thin shell on top
of it. Switching from a MAP point estimate to NUTS to geoVI to a
nested-sampling evidence calculation is a one-line change.

Physics lives in components, instruments live in observation. A dust
law is physics. A fiber aperture correction is an instrument. You
should not have to think about aperture corrections when you are
choosing between Calzetti and Charlot & Fall, and we wrote the layer
boundaries to make sure you don't have to.

We try hard to keep the user-facing surface astronomer-readable. A
new contributor should be able to add a dust attenuation law in one
sitting, in one file, without learning the inference layer. Building
a model in tengri (`SEDModel.build(sfh={'type': 'dpl', '*': FREE,
'beta': Uniform(1, 3)}, dust=..., neb=...)`) reads like a recipe
rather than a configuration file, and `model.spec.summary()` will
tell you which parameters you set, which are free, and which fell
back to library defaults.

## What's modular

| Layer | What you can swap |
|---|---|
| Stellar SSP | BC03, BPASS, FSPS, ProGeny — any HDF5 file in the DSPS schema |
| SFH | parametric (about 15 families), non-parametric (Leja+ continuity, Dirichlet), and stochastic (IFT correlated fields with PSD-governed burstiness) |
| Dust attenuation | Calzetti, Cardelli, Charlot & Fall two-component, Salim, Kriek & Conroy, others |
| Dust emission | Draine & Li, Dale, THEMIS |
| Nebular | `baked_in`, `cue` (emulator), `cloudy_grid`, `cb19` |
| AGN | accretion disc, SKIRTOR torus, BLR/NLR through Cue, optional X-ray and radio |
| IGM | Madau, Inoue |
| Inference | MAP, Laplace, Pathfinder, NUTS, ray-tracing MCMC, nested sampling, geoVI, hierarchical population fits |

`tengri.summary()` prints the live count for every registry, and
new components register themselves; nothing in this table is updated
by hand.

## Architecture

The forward model is layered. Each layer does one thing and talks to
the layer below it through one small Protocol.

```text
┌─────────────────────────────────────────────────────────────┐
│  Inference        MAP · Laplace · Pathfinder · NUTS · NSS   │
│                   geoVI · Population (hierarchical)         │
│       ↳ asks ForwardModel for log p(data | params)          │
└─────────────────────────────────────────────────────────────┘
                  │  knows nothing about physics
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Likelihood       Gaussian · StudentT · Composite           │
│       ↳ scores Observation.predict(params) against data     │
└─────────────────────────────────────────────────────────────┘
                  ▲
                  │  prediction dict
┌─────────────────────────────────────────────────────────────┐
│  ForwardModel     populations (SEDModel + SpatialModel)     │
│                   owns Parameters, threads ForwardState     │
│       ↳ .predict(params) → dict ──────────► Observation     │
└─────────────────────────────────────────────────────────────┘
                  ▲
                  │  threads state through components
┌─────────────────────────────────────────────────────────────┐
│  Components       Stellar · SFH · Dust attn. · Dust IR ·    │
│                   Nebular · AGN · IGM · Radio · X-ray       │
│       each: parameters, precompute, apply(state)            │
└─────────────────────────────────────────────────────────────┘
                  ▲
                  │
┌─────────────────────────────────────────────────────────────┐
│  Observation      Photometry · Spectroscopy ·               │
│                   FiberSpectroscopy · Imaging · Joint       │
│       ↳ reads ForwardState, returns prediction dict         │
└─────────────────────────────────────────────────────────────┘
```

The rule the whole stack follows is the one above: physics in
components, instruments in observation. A fiber aperture correction
lives in `FiberSpectroscopyObservation`, not in any component. A
per-age spatial profile lives in a `SpatialComponent`, not in
`ImagingObservation`. Keeping that line straight is what makes the
codebase still legible after eight or nine layers of physics get
stacked into a single fit.

### Components

A component is the atomic unit of physics. Each one declares its
parameters, what state it reads from upstream components, what state
it publishes to downstream ones, and how to evaluate itself on a
wavelength grid. Adding a new dust attenuation law, a new SFH family,
or a new AGN torus library is one file in this layer; the parameter
registry, the `reads`/`publishes` contract, and the JIT plumbing all
discover the component automatically.

The Stellar component is the entry point on the spectral side. It
reads the SFH, weights the SSP grid by age, and writes the
intrinsic rest-frame $L_\nu$ into the running state. Dust attenuation
then reads $L_\nu$ and writes both the attenuated SED and the
absorbed luminosity $L_{\rm absorbed}$. Dust IR emission reads
$L_{\rm absorbed}$ and adds an IR component back, so energy balance
is enforced by the contract rather than by a separate calibration
step. Nebular, AGN, IGM, radio, and X-ray are layered on top, each
declaring its own reads and publishes.

The same pattern handles morphology. A `SpatialComponent` declares
its parameters (effective radius, Sérsic index, axis ratio, position
angle, …) and writes a 2D profile into the state. Sérsic,
exponential, flat-slab, and bulge+disk profiles are built in;
correlated-field spatial priors plug in the same way.

### Sub-models

Sub-models compose components into a complete spectral or spatial
chain. They are thin: an `SEDModel` is a list of SED components plus
the validator that checks the `reads`/`publishes` graph is closed; a
`SpatialModel` is the same on the spatial side. A `SpatialSEDModel`
just runs both in sequence, with the SED chain first so that spatial
components can read mass, age, or colour state if they need it.
There is no special-casing for stellar atmospheres, hierarchical
SFHs, or per-age morphology — those are all components or
sub-models, slotted in through the same interface.

### ForwardModel

`ForwardModel` is what inference actually talks to. It owns the
`Parameters` object, holds one or more populations (each a sub-model
plus an optional spatial model), wires everything to the
`Observation`, and exposes a single `.predict(params)` that returns
the predicted data. Whether there is one population or three, whether
each population has a spatial counterpart, and whether the
observation is photometry, spectroscopy, fiber spectroscopy, imaging,
or some joint product, is invisible to the sampler.

Multi-population is supported from day one. A fit with an AGN point
source, a Sérsic bulge, and an exponential disc is three populations
under one `ForwardModel`, with parameter names namespaced by
population so nothing collides.

### Observation

The observation layer is where instruments live. `Photometry`
integrates the predicted SED through filter curves. `Spectroscopy`
resamples and applies a line-spread function. `FiberSpectroscopy`
adds an aperture correction that integrates the spatial profile
through the fiber footprint, which is what makes physically correct
joint photometry + fiber-spectrum fits a one-liner at the
`ForwardModel` level rather than a hack inside an SED component.
`Imaging` and joint observations follow the same pattern.

The whole architecture turns the most common galaxy SED inference setup —
broadband photometry of the whole galaxy plus a fiber spectrum of
the inner core — into a clean composition rather than a slab
approximation. The fiber sees the Sérsic core; the broadband sees
core plus envelope; the spatial component models the profile, the
observation integrates it through the fiber footprint, and only
then are predictions compared to data.

### Likelihood and inference

The likelihood scores `Observation.predict(params)` against the data
through whichever model is appropriate (Gaussian, Student-t, or a
composite of several). Inference sits one layer above, asking the
forward model for $\log p(\text{data} \mid \text{params})$ and
nothing else. Switching samplers does not change the forward model,
the components, the observation layer, or the likelihood. Each layer
keeps its own concerns.

## Why JAX

JAX is the substrate that lets us deliver speed, gradients, and
modularity in one codebase. None of the alternatives we considered
let us hit all three at once.

Automatic differentiation gives us $\partial \log p / \partial\theta$
for every parameter at roughly the cost of one extra forward pass. It
stays correct under interpolation, table lookups, and emulator
evaluations, where finite differences silently break. Once you start
composing a Charlot & Fall dust screen on top of a non-parametric SFH
on top of a Cue nebular emulator, hand-derived gradients stop being
maintainable; autodiff is what keeps the math honest.

JIT compilation makes the forward model competitive with hand-written
C, and `vmap` vectorises across a galaxy sample without writing the
loop yourself. The same source file runs on CPU, GPU, and TPU, which
is the only reason hierarchical population fits over thousands of
galaxies are practically affordable.

The discipline JAX imposes — pure functions, declared array shapes,
no in-place mutation inside a JIT, no Python `if` on a traced value —
takes some getting used to. We have tried to absorb most of that cost
in the library, so that a user calling `fitter.run(...)` does not have
to think about traced arrays.
