# Overview

Galaxy SEDs encode stellar mass assembly, chemical enrichment, and dust processing. Tengri recovers these from broadband photometry and spectra using one forward model that spans stellar populations through X-ray, all driven by shared parameters.

Built on [JAX](https://jax.readthedocs.io) and [DSPS](https://github.com/ArgonneCPAC/DSPS). Inference backends (optimizers, samplers, variational inference) plug in as registrations. `tengri.summary()` prints the live count for every registry; new components register themselves.

## Philosophy

The forward model is the artifact. Inference is a thin shell, so backend swaps are one-line changes. Physics lives in components; instruments in observation. Adding a dust law or sampler is a registration, not a rewrite.

## What's modular

| Layer | What you can swap |
|---|---|
| Stellar SSP | BC03, BPASS, FSPS, ProGeny: any HDF5 file in the DSPS schema |
| SFH | parametric (about 15 families), non-parametric (Leja+ continuity, Dirichlet), and stochastic (IFT correlated fields with PSD-governed burstiness) |
| Dust attenuation | Calzetti, Cardelli, Charlot & Fall two-component, Salim, Kriek & Conroy, others |
| Dust emission | Draine & Li, Dale, THEMIS |
| Nebular | `ssp` (baked into the SSP grid), `cue` (emulator), `cloudy`, `cb19` |
| AGN | accretion disc, SKIRTOR torus, BLR/NLR through Cue, optional X-ray and radio |
| IGM | Madau, Inoue |
| Inference | optimizers from `optax`, samplers from `BlackJAX`, variational inference from `NIFTy.re`, plus hierarchical / population extensions on top |

`tengri.summary()` prints the live count for every registry, and
new components register themselves; nothing in this table is updated
by hand.

## Architecture

The forward model is layered. Each layer does one thing and talks to
the layer below it through one small Protocol.

```text
┌─────────────────────────────────────────────────────────────┐
│  Inference        optax · BlackJAX · NIFTy.re · Population  │
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

Each component declares parameters, reads from upstream, publishes to downstream, and evaluates on a wavelength grid. A new dust law or AGN library is one file; the registry and JIT plumbing auto-discover it.

Energy balance is enforced by contract: Stellar emits L_ν, Dust attenuation publishes L_absorbed, Dust IR emission consumes it. Nebular, AGN, IGM, radio, and X-ray layer on top. Spatial components (Sérsic, exponential, bulge+disk) follow the same pattern.

### Sub-models

An `SEDModel` is components plus a validator that closes the reads/publishes graph. A `SpatialModel` does the same spatially. Hierarchical SFHs, per-age morphology, and stellar atmospheres are components or sub-models, not special cases.

### ForwardModel

What inference talks to. Owns `Parameters`, holds populations (sub-model + optional spatial model), wires to `Observation`, exposes `.predict(params)`. Multi-population from day one: AGN point source, Sérsic bulge, exponential disc are three populations with namespaced parameters.

### Observation

Instruments live here. `Photometry` integrates through filter curves. `Spectroscopy` resamples + LSF. `FiberSpectroscopy` corrects aperture via spatial profile integration—making joint photo + fiber fits one-liners rather than hacks in components. `Imaging` and joint observations follow the same pattern.

### Likelihood and inference

Likelihood (Gaussian, Student-t, composite) scores predictions against data. Inference asks for $\log p(\text{data} \mid \text{params})$ and nothing else. Backends plug in unchanged.

## Why JAX

Automatic differentiation: $\partial \log p / \partial\theta$ for free at the cost of one extra forward pass. Stays correct under interpolation and emulators, where finite differences fail.

JIT compilation makes the forward model competitive with C. `vmap` vectorizes across galaxy samples. One source runs on CPU, GPU, TPU.

JAX's discipline (pure functions, declared shapes, no mutation, no Python `if` on traced values) costs upfront. We absorb that cost in the library.
