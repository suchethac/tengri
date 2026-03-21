# tengri

Differentiable SED fitting with IFT star formation history priors.

**tengri** models galaxy star formation histories as continuous correlated fields
governed by a power spectral density, then fits observed photometry and spectra
via gradient-based inference in JAX. The PSD parameters encode the amplitude
and timescale of burstiness — physically interpretable priors that propagate
through a fully differentiable forward model.

## Getting started

Start with the {doc}`quickstart tutorial <tutorials/index>` to fit a galaxy SED
in under a minute.

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Learn tengri step by step — from installation to your first fit.
:::

:::{grid-item-card} Demonstrations
:link: demonstrations/index
:link-type: doc

Science workflows: spectroscopy, photometric catalogs, hierarchical inference.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

Physics deep-dives: PSD models, dust, AGN, nebular emission, noise.
:::

:::{grid-item-card} Observation Guide
:link: observation/index
:link-type: doc

Unified Observation API — photometry, spectroscopy, joint fitting.
:::

:::{grid-item-card} Performance
:link: performance/index
:link-type: doc

Benchmarks, optimization strategies, and profiling your fits.
:::

:::{grid-item-card} Advanced
:link: advanced/index
:link-type: doc

Convergence diagnostics, batch fitting, extending tengri, hierarchical inference.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Auto-generated from docstrings.
:::

:::{grid-item-card} Developer Guide
:link: developer/index
:link-type: doc

Architecture, contributing, internals.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

install
tutorials/index
demonstrations/index
reference/index
observation/index
performance/index
advanced/index
api/index
developer/index
changelog
```
