# Experimental

Research demonstrations rather than tutorials. They explore features that are
still moving and may use APIs that change between releases, so they sit outside
the supported sequence — read them for what the method can do, not as a
template to copy. Each states its own caveats up front.

- **{doc}`stochastic_sfh_recovery`**: which parts of a bursty star formation
  history a fit can actually measure. tengri models burstiness as a smooth
  backbone times a Gaussian-process field with a PSD-governed prior, which adds
  a free parameter per age bin — so a fit will always return a bursty history
  whether or not the data said anything about it. A known history is injected
  and recovered to separate the measurement from the prior talking back.
- **{doc}`multimodel_bma_candels`**: Bayesian model averaging over modeling
  choices the photometry alone does not pin down. Seven CANDELS GOODS-South
  galaxies at $z \sim 1$ are fit under four configurations that vary the SFH
  family, stellar isochrone, dust law, and nebular treatment, then combined by
  evidence weight — putting a number on how much the choice of model
  contributes to the error budget.

```{toctree}
:maxdepth: 1

stochastic_sfh_recovery
multimodel_bma_candels
```
