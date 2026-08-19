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
- **{doc}`jwst_nonparametric_fits`**: a Prospector-style continuity SFH fit to
  JWST NIRCam broad plus medium band photometry at $z = 1.5$, run in individual
  and catalog mode with the cost of each stage measured. The operational
  finding is a warning about how that cost is read: the two settings that make
  the correlated bin-ratio fit finish sooner, a dense mass matrix and
  fixed-length HMC, are the two that stop it sampling, so wall time ranks the
  samplers in the wrong order and seconds per effective sample is the number to
  compare.
- **{doc}`apple_mps`**: running the forward model and fits on the Apple GPU via
  the community `jax-mps` backend. Needs a JAX version tengri does not pin and
  runs in pure float32, so it is a feasibility study rather than a
  recommendation — and it concludes against the GPU for single-galaxy work,
  which is the useful part. Measured on an M4 Pro: CPU wins by 59x on a
  gradient and 86x on a MAP fit, batches of a few hundred reach parity, and the
  run-to-run spread on MPS exceeds the differences between workloads.

```{toctree}
:maxdepth: 1

stochastic_sfh_recovery
multimodel_bma_candels
jwst_nonparametric_fits
apple_mps
```
