# SPDX-License-Identifier: BSD-3-Clause
"""Likelihood primitives — single source of truth for χ² and friends.

The submodules expose:

- :mod:`tengri.inference.likelihoods.gaussian`: diagonal-Gaussian
  χ² used by both legacy ``build_loglikelihood_fn`` and the new
  :class:`tengri.pipeline.PhotometryLikelihood` Protocol adapter.

"""

from __future__ import annotations

from tengri.inference.likelihoods.gaussian import (
    diag_gaussian_chi2,
    diag_gaussian_log_prob,
)
from tengri.inference.likelihoods.marginalized import (
    CalibrationELineMarginalizedLikelihood,
    CalibrationMarginalizedLikelihood,
    CloudyELineMarginalizedLikelihood,
    ELineFittedLikelihood,
    ELineMarginalizedLikelihood,
)
from tengri.inference.likelihoods.protocol import (
    CensoredLikelihood,
    GaussianLikelihood,
    MultivariateGaussianLikelihood,
    StudentTLikelihood,
)

__all__ = [
    "CalibrationELineMarginalizedLikelihood",
    "CalibrationMarginalizedLikelihood",
    "CensoredLikelihood",
    "CloudyELineMarginalizedLikelihood",
    "ELineFittedLikelihood",
    "ELineMarginalizedLikelihood",
    "GaussianLikelihood",
    "MultivariateGaussianLikelihood",
    "StudentTLikelihood",
    "diag_gaussian_chi2",
    "diag_gaussian_log_prob",
]
