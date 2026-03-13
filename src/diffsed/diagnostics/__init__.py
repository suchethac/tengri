"""Diagnostics enabled by full differentiability.

These tools are only possible because the entire pipeline
θ → SFH → SSP → dust → observables → likelihood
is differentiable end-to-end. They provide:

1. Fisher Information Matrix (FIM) — parameter constraints from data
2. Gradient SEDs (saliency) — ∂flux/∂θ across wavelength
3. Laplace approximation — cheap posteriors from the Hessian at MAP
"""
