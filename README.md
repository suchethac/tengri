# diffsed

Differentiable SED fitting with Information Field Theory star formation history priors.

Fully differentiable JAX pipeline: PSD-governed GP → SFH → DSPS SED → photometry/spectroscopy.
Inference via geoVI (NIFTy.re), NUTS (BlackJAX), or Adam (optax).
