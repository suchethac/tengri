"""Inference backends: MAP optimization, MCMC sampling, variational inference.

All backends share the same interface:
    result = fit_*(forward_model, data, noise, prior_config, **kwargs)

Each returns an InferenceResult with posterior samples/point estimates,
diagnostics, and timing information.
"""
