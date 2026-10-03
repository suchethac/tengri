# SPDX-License-Identifier: BSD-3-Clause
"""NIFTy Gaussian likelihood with per-datum upper/lower-limit censoring.

Imports ``nifty8.re`` at module level; import it only after the optional
dependency is known to be present.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import nifty8.re as jft

from tengri.observation.noise import censored_neg_log_likelihood

__all__ = ["CensoredGaussian"]


class CensoredGaussian(jft.Gaussian):
    """``jft.Gaussian`` whose energy scores limit bands with the Gaussian CDF.

    The energy is :func:`tengri.observation.noise.censored_neg_log_likelihood`
    at fixed noise (detection ``0.5 r^2 + ln sigma``, upper limit
    ``-ln Phi((F - m)/sigma)``, lower limit ``-ln Phi((m - F)/sigma)``).
    ``metric``, ``left_sqrt_metric`` and ``transformation`` are inherited
    unchanged, so a limit band enters the geoVI/MGVI metric as a detection at
    its limit value (see :mod:`tengri.inference._censoring` for why).

    Parameters
    ----------
    data : array, shape (n_data,)
        Observed values; a limit band holds the limit value [flux units].
    noise : array, shape (n_data,)
        Observed 1-sigma uncertainties [flux units].
    mask : array, shape (n_data,)
        Censoring flags: 0 detected, 1 upper limit, -1 lower limit.
    noise_cov_inv, noise_std_inv : callable
        Noise operators, as for ``jft.Gaussian``.
    """

    noise: Any = dataclasses.field(metadata=dict(static=False))
    mask: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, data, noise, mask, noise_cov_inv=None, noise_std_inv=None):
        self.noise = noise
        self.mask = mask
        super().__init__(data, noise_cov_inv=noise_cov_inv, noise_std_inv=noise_std_inv)

    def energy(self, primals):
        """Censored negative log-likelihood of ``primals`` (the model prediction)."""
        return censored_neg_log_likelihood(self.data, self.noise, primals, self.mask)
