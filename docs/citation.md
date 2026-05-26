# Citing tengri

If tengri shows up in a publication, please cite the methods paper
(in preparation) and the upstream codes whose physics, grids, and
samplers we are building on. `tengri.cite_all()` returns a BibTeX
block for every model, SSP, and inference backend that actually ran
in your fit, so the acknowledgement list stays honest as you change
components.

## The tengri methods paper

```bibtex
@ARTICLE{tengri-paper-i,
  author  = {Cooray, S. and collaborators},
  title   = {{tengri}: A Differentiable Framework for High-Dimensional
             Bayesian Inference from Galaxy Spectral Energy Distributions},
  journal = {in preparation},
  year    = {2026}
}
```

This citation will be updated when Paper I is on arXiv. Paper II
(stochastic SFHs with IFT correlated-field priors, fit through geoVI)
follows.

## Acknowledgements

Tengri is built on a long stack of open-source astronomy and Bayesian
inference projects, and we are grateful to the authors and
maintainers of each one. They shape what tengri can do; we have just
glued them together.

**SED physics and stellar populations**

- [DSPS](https://github.com/ArgonneCPAC/dsps) — differentiable stellar
  population synthesis. Hearin et al. (2023),
  [arXiv:2112.08423](https://arxiv.org/abs/2112.08423).
- [FSPS](https://github.com/cconroy20/fsps), BC03, BPASS, ProGeny —
  SSP grids re-formatted into the DSPS schema.
- [Cue](https://github.com/yi-jia-li/cue) — neural emulator for
  nebular emission spectra.
- [SKIRTOR](https://sites.google.com/site/skirtorus/) — AGN torus
  templates (Stalevski et al. 2016).
- [THEMIS](https://www.ias.u-psud.fr/themis/), Draine & Li,
  Dale & Helou — dust IR emission templates.

**Inference and probabilistic programming**

- [JAX](https://github.com/google/jax) — autodiff, JIT, vectorisation.
- [NIFTy.re](https://gitlab.mpcdf.mpg.de/ift/nifty) — information
  field theory, geoVI / MGVI. Edenhofer et al. (2024),
  [arXiv:2402.16683](https://arxiv.org/abs/2402.16683).
- [BlackJAX](https://github.com/blackjax-devs/blackjax) — NUTS, HMC,
  MCLMC and friends.
- [optax](https://github.com/google-deepmind/optax) — MAP optimisers.
- Ray Tracing sampler — Behroozi (2025),
  [arXiv:2504.20029](https://arxiv.org/abs/2504.20029).
- Pathfinder — Zhang et al. (2022),
  [arXiv:2108.03782](https://arxiv.org/abs/2108.03782).
- Nested Slice Sampling — Yallup, Kroupa & Handley (2026),
  [arXiv:2601.23252](https://arxiv.org/abs/2601.23252).
- Elliptical Slice Sampling — Murray, Adams & MacKay (2010),
  [arXiv:1001.0175](https://arxiv.org/abs/1001.0175).

**Comparison and reproducibility**

- [CIGALE](https://cigale.lam.fr/) — reference panchromatic SED
  fitting code; tengri reproduces its main physics paths for
  cross-validation. See [Reproduction → CIGALE](reproduction/cigale).

## Inheriting credit automatically

```python
import tengri
fitter = ...
result = fitter.run("mcmc_nuts")
print(tengri.cite_all(result))   # BibTeX for every component that ran
```

`cite_all` walks the model and inference call graph and emits BibTeX
for everything that contributed, including the specific SSP grid you
loaded. It is the easiest way to keep the acknowledgement section of
a paper in sync with the fit.
