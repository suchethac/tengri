# CIGALE

[CIGALE](https://cigale.lam.fr/) (Boquien et al. 2019) is the
reference code for panchromatic galaxy and AGN SED inference.
Its energy-balance treatment of dust, its nebular module, and the
AGN block in the x-cigale fork are what most people in the field
expect a panchromatic fit to look like.

Tengri's AGN, dust, and nebular layers are written to track CIGALE
where the physics overlaps. Same disc + torus + BLR/NLR structure,
same energy balance between attenuated stellar light and re-emitted
dust luminosity, same X-ray photoelectric + Compton absorption.
Running both codes on the same data, the integrated quantities (M⋆,
SFR, A_V, L_AGN) should agree to within the noise, and the SED
templates should overlay wherever the two codes implement the same
physics.

This page will hold the side-by-side run: configuration recipes,
mock-recovery comparison on shared inputs, and a discussion of the
deliberate departures.

```{note}
The companion **CIGALE reproduction notebook is in the works** and
will land at `notebooks/repro_cigale.py` (Jupytext source) →
`docs/reproduction/cigale_notebook.ipynb`. It will fit a shared mock
and a shared real-galaxy SED with both codes side by side and report
parameter recovery, residuals, and runtime. Until then, the table
below is the contract the notebook is being written against.
```

## Scope of the comparison

| Layer | tengri | CIGALE / x-cigale | Expected agreement |
|---|---|---|---|
| Stellar SSP | DSPS BC03 / FSPS | BC03 / M2005 | < 0.05 dex on M⋆ |
| SFH | delayed-τ, dpl, non-parametric | delayed, periodic | shape-matched on integrals |
| Nebular | `baked_in` / `cue` | nebular module | line ratios within Cloudy grid spacing |
| Dust attenuation | Calzetti, Charlot & Fall | Calzetti, modified | A_V within prior width |
| Dust emission | Draine & Li, Dale, THEMIS | Draine & Li, Dale, THEMIS | identical templates → identical IR |
| AGN | disc + SKIRTOR + Cue BLR/NLR | x-cigale (Yang+ 2020) | L_AGN, AGN fraction within MCMC width |
| X-ray | photoelectric + Compton (N_H) | x-cigale X-ray | identical attenuation curve |
| IGM | Madau, Inoue | Meiksin | sub-percent above 1216 Å rest |

Where tengri intentionally departs from CIGALE, the departure and
its implications will be discussed inline. The main two are the SFH
prior (CIGALE uses grid templates; tengri samples continuous priors,
including stochastic fields) and the inference layer (CIGALE is
Bayesian on a grid; tengri is differentiable and gradient-based).
