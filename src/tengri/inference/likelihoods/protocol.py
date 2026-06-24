# SPDX-License-Identifier: BSD-3-Clause
"""Channel-parameterized :class:`Likelihood` Protocol adapters.

Design rule
-----------
**One class per *math type*, parameterized by which prediction channel
to read.** Don't duplicate the same Gaussian χ² for each observation
type — pin a different ``channel`` string instead.

This module ships four base adapters:

- :class:`GaussianLikelihood` — diagonal Gaussian (the workhorse;
  covers photometry, spectroscopy, line fluxes, spectral indices,
  equivalent widths, anything with diagonal Gaussian errors).
- :class:`StudentTLikelihood` — heavy-tailed alternative for outlier
  tolerance.
- :class:`CensoredLikelihood` — handles upper/lower limits via the
  normal CDF.
- :class:`MultivariateGaussianLikelihood` — correlated noise with a
  pre-inverted covariance matrix.

To add a new observation channel (e.g. ``"line_fluxes"``,
``"indices"``, ``"imaging_fnu_pixel"``, ``"fiber_spec_fnu"``), the
user does NOT need a new class — they instantiate
``GaussianLikelihood(channel="line_fluxes", obs=..., err=...)`` and
compose with :class:`CompositeLikelihood`.

Why not factory functions?
~~~~~~~~~~~~~~~~~~~~~~~~~~
The convenience names :class:`PhotometryLikelihood` and
:class:`SpectroscopyLikelihood` (sister modules) are real subclasses,
not factories — preserves :func:`isinstance` semantics, autocomplete
discoverability, and identifiable :func:`repr`. They each pin
``channel`` to the standard string and inherit :meth:`log_prob`
unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass

import jax.numpy as jnp

from tengri.inference.likelihoods.gaussian import diag_gaussian_log_prob
from tengri.observation.noise import (
    censored_neg_log_likelihood as _censored_neg_log_lik,
    variable_noise_hamiltonian as _student_t_neg_log_lik,
)

__all__ = [
    "CensoredLikelihood",
    "GaussianLikelihood",
    "MultivariateGaussianLikelihood",
    "StudentTLikelihood",
]


# ─────────────────────────────────────────────────────────────────────
# 1. Diagonal Gaussian — the workhorse
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GaussianLikelihood:
    r"""Diagonal Gaussian over any single prediction channel.

    Parameters
    ----------
    obs : ndarray
        Observed values (shape matches the prediction at ``channel``).
    err : ndarray
        1-σ uncertainties.
    channel : str, keyword-only
        Which prediction-dict key to read. Examples: ``"phot_fnu"``,
        ``"spec_fnu"``, ``"line_fluxes"``, ``"indices"``,
        ``"imaging_fnu_pixel"``, ``"fiber_spec_fnu"``.
    sigma_floor : float, keyword-only
        Fractional floor: ``σ_total² = err² + (sigma_floor·obs)²``.
    name : str, keyword-only
        Diagnostic identifier.

    Notes
    -----
    **JIT-compatible**: yes — pure JAX via
    :func:`diag_gaussian_log_prob`.
    """

    obs: jnp.ndarray
    err: jnp.ndarray
    _: KW_ONLY
    channel: str = "phot_fnu"
    sigma_floor: float = 0.0
    name: str = "gaussian"

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        del params
        return diag_gaussian_log_prob(
            prediction[self.channel], self.obs, self.err, sigma_floor=self.sigma_floor
        )

    def declared_parameters(self):
        return []


# ─────────────────────────────────────────────────────────────────────
# 2. Student-t — heavy-tailed
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StudentTLikelihood:
    r"""Student-t over any single prediction channel.

    Use when outliers (cosmic rays, unmodeled features, calibration
    glitches) would over-weight a Gaussian fit.

    Parameters
    ----------
    obs, err : ndarray
        Same as :class:`GaussianLikelihood`.
    dof : float, keyword-only
        Degrees of freedom. Heavy-tailed values: 2 (Alsing+2022),
        4 (moderate). ``None`` recovers Gaussian.
    f_cal : float, keyword-only
        Fractional calibration uncertainty added in quadrature to
        ``err`` before evaluating the t-density.
    channel : str, keyword-only
        Which prediction-dict key to read.

    Notes
    -----
    **JIT-compatible**: yes — wraps
    :func:`tengri.observation.noise.variable_noise_hamiltonian`
    (sign-flipped: that function returns *energy*, this returns
    log-probability).
    """

    obs: jnp.ndarray
    err: jnp.ndarray
    _: KW_ONLY
    dof: float | None = None
    f_cal: float = 0.0
    f_cal_param: str | None = None
    channel: str = "phot_fnu"
    name: str = "student_t"

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        # When ``f_cal_param`` is set, the calibration uncertainty is a
        # free parameter the inference engine fits — read it from the
        # params dict each call. Otherwise fall back to the static
        # ``f_cal`` constant.
        if self.f_cal_param is not None and params is not None and self.f_cal_param in params:
            f_cal = params[self.f_cal_param]
        else:
            f_cal = self.f_cal
        return -_student_t_neg_log_lik(
            data=self.obs,
            noise_obs=self.err,
            predicted=prediction[self.channel],
            f_cal=f_cal,
            dof=self.dof,
        )

    def declared_parameters(self):
        return []


# ─────────────────────────────────────────────────────────────────────
# 3. Censored — upper / lower limits
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CensoredLikelihood:
    r"""Diagonal Gaussian + per-point CDF for upper / lower limits.

    Parameters
    ----------
    obs, err : ndarray
        Observed values and 1-σ uncertainties. For censored points,
        ``obs`` carries the limit value.
    mask : ndarray, dtype int
        Per-point flag: ``0`` = detected (Gaussian),
        ``1`` = upper limit (CDF), ``-1`` = lower limit (CDF).
    f_cal : float, keyword-only
        Fractional calibration uncertainty for detected points only.
    dof : float | None, keyword-only
        If set, detected points use a Student-t instead of a Gaussian.
    channel : str, keyword-only
        Which prediction-dict key to read.

    Notes
    -----
    **JIT-compatible**: yes — wraps
    :func:`tengri.observation.noise.censored_neg_log_likelihood`
    (sign-flipped).
    """

    obs: jnp.ndarray
    err: jnp.ndarray
    mask: jnp.ndarray
    _: KW_ONLY
    f_cal: float = 0.0
    f_cal_param: str | None = None
    dof: float | None = None
    channel: str = "phot_fnu"
    name: str = "censored"

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        if self.f_cal_param is not None and params is not None and self.f_cal_param in params:
            f_cal = params[self.f_cal_param]
        else:
            f_cal = self.f_cal
        return -_censored_neg_log_lik(
            data=self.obs,
            noise_obs=self.err,
            predicted=prediction[self.channel],
            mask=self.mask,
            f_cal=f_cal,
            dof=self.dof,
        )

    def declared_parameters(self):
        return []


# ─────────────────────────────────────────────────────────────────────
# 4. Multivariate Gaussian — correlated noise
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MultivariateGaussianLikelihood:
    r"""Correlated Gaussian via a pre-inverted covariance matrix.

    Parameters
    ----------
    obs : ndarray, shape (n,)
        Observed values.
    cov_inv : ndarray, shape (n, n)
        Inverse of the noise covariance matrix. Pre-inverted at
        construction so :meth:`log_prob` is a single matrix-vector
        product per call.
    channel : str, keyword-only
        Which prediction-dict key to read.

    Notes
    -----
    **JIT-compatible**: yes — pure JAX.

    Drops the normalization constant
    :math:`-\tfrac{1}{2}\log\det(2\pi\Sigma)`. Add it back if you need
    a true log-evidence.

    For a GP-correlated spectrum, build ``cov_inv`` via the existing
    :func:`tengri.observation.noise.gp_noise_covariance` helper and
    invert once before construction.
    """

    obs: jnp.ndarray
    cov_inv: jnp.ndarray
    _: KW_ONLY
    channel: str = "spec_fnu"
    name: str = "mvn_gaussian"

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        del params
        diff = self.obs - prediction[self.channel]
        return -0.5 * (diff @ self.cov_inv @ diff)

    def declared_parameters(self):
        return []
