# SPDX-License-Identifier: BSD-3-Clause
"""Analytically-marginalised :class:`Likelihood` adapters.

Wraps two existing math primitives:

- :class:`CalibrationMarginalisedLikelihood` —
  :func:`tengri.observation.calibration.marginalize_calibration`
  (Chebyshev polynomial calibration integrated out analytically).
- :class:`ELineMarginalisedLikelihood` —
  :func:`tengri.observation.eline_marginalization.marginalize_emission_lines`
  (linear emission-line amplitudes integrated out analytically).

Both are diagonal Gaussian under the hood; the marginalisation is
analytic via standard Gaussian-conjugate algebra. They differ from
the simple :class:`GaussianLikelihood` only in that they integrate
out a nuisance vector before scoring the residuals, so they are
distinct *math types* (not just channel renames).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass

import jax.numpy as jnp

from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2
from tengri.observation.calibration import (
    marginalize_calibration as _marginalize_calibration,
)
from tengri.observation.eline_marginalization import (
    marginalize_emission_lines as _marginalize_emission_lines,
)

__all__ = [
    "CalibrationMarginalisedLikelihood",
    "CloudyELineMarginalisedLikelihood",
    "ELineFittedLikelihood",
    "ELineMarginalisedLikelihood",
]


# ─────────────────────────────────────────────────────────────────────
# Calibration polynomial — Chebyshev marginalisation
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationMarginalisedLikelihood:
    r"""Spectroscopy likelihood with the calibration polynomial
    integrated out analytically.

    Implements the Prospector approach (Johnson+2021): the model
    spectrum is treated as :math:`C(\lambda) m(\lambda)` with
    :math:`C(\lambda)` a Chebyshev polynomial of order ``n_poly``,
    and the polynomial coefficients are integrated under a Gaussian
    prior :math:`\mathcal{N}(0, \sigma_{\rm prior}^2 \mathbb{I})`.
    Reduces the sampled parameter space by ``n_poly`` dimensions.

    Parameters
    ----------
    fnu_obs : ndarray, shape (n_pixels,)
        Observed spectrum [erg/s/cm²/Hz].
    fnu_err : ndarray, shape (n_pixels,)
        1-σ uncertainties.
    wavelength : ndarray, shape (n_pixels,)
        Wavelength grid [Å]. Required for the Chebyshev basis; held
        as a Python attribute (not a free parameter).
    n_poly : int, keyword-only
        Number of polynomial coefficients (T_1 through T_n_poly).
        T_0 = 1 is implicit. Default 3.
    prior_sigma : float, keyword-only
        Standard deviation of the Gaussian prior on each coefficient.
        Default 1.0.
    channel : str, keyword-only
        Prediction-dict key. Default ``"spec_fnu"``.

    Notes
    -----
    **JIT-compatible**: yes — ``n_poly`` is a static argument of the
    underlying primitive.

    Returns the *positive* marginal log-likelihood (data term only;
    the analytic Gaussian normalisation in
    :func:`marginalize_calibration` is included in its return value
    by construction). Sign matches the rest of the
    :class:`Likelihood` cohort (higher = better fit).
    """

    fnu_obs: jnp.ndarray
    fnu_err: jnp.ndarray
    wavelength: jnp.ndarray
    _: KW_ONLY
    n_poly: int = 3
    prior_sigma: float = 1.0
    channel: str = "spec_fnu"
    name: str = "calibration_marginalised"

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        del params
        log_lik, _c_hat, _c_err = _marginalize_calibration(
            model_flux=prediction[self.channel],
            obs_flux=self.fnu_obs,
            obs_err=self.fnu_err,
            wavelength=self.wavelength,
            n_poly=self.n_poly,
            prior_sigma=self.prior_sigma,
        )
        return log_lik

    def declared_parameters(self):
        return []


# ─────────────────────────────────────────────────────────────────────
# Emission-line amplitudes — analytic marginalisation
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ELineMarginalisedLikelihood:
    r"""Spectroscopy likelihood with linear emission-line amplitudes
    integrated out analytically.

    Each emission line contributes a known *shape* (Gaussian profile
    centred at the rest-frame line wavelength) but with an *unknown
    amplitude*. The amplitudes form a linear nuisance vector under a
    Gaussian prior; the marginal likelihood is computed in closed form
    by Gaussian conjugacy.

    Parameters
    ----------
    fnu_obs : ndarray, shape (n_pixels,)
        Observed spectrum [erg/s/cm²/Hz].
    fnu_err : ndarray, shape (n_pixels,)
        1-σ uncertainties.
    design_matrix : ndarray, shape (n_pixels, n_lines)
        Per-line shape evaluated at every pixel — built once via
        :func:`tengri.observation.eline_marginalization.build_eline_design_matrix`
        (or the broad-line / doublet-constrained variants). Held on
        ``self`` because line wavelengths and LSF widths are fixed.
    prior_variance : ndarray | None, keyword-only
        Per-line prior variance on the amplitude. ``None`` → flat
        (1e10) — the original primitive's default.
    channel : str, keyword-only
        Prediction-dict key. Default ``"spec_fnu"``.

    Notes
    -----
    **JIT-compatible**: yes.

    The continuum prediction
    (:attr:`prediction[channel]`) is the model SED *without* lines;
    the residual ``data - model`` is what the design matrix is fit
    to. Returns the marginal log-likelihood directly.
    """

    fnu_obs: jnp.ndarray
    fnu_err: jnp.ndarray
    _: KW_ONLY
    design_matrix: jnp.ndarray | None = None
    design_matrix_builder: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray] | None = None
    prior_variance: jnp.ndarray | None = None
    channel: str = "spec_fnu"
    name: str = "eline_marginalised"

    def __post_init__(self) -> None:
        # Exactly one of (static design matrix) | (per-call builder)
        # must be provided. Both or neither is a usage error.
        has_static = self.design_matrix is not None
        has_builder = self.design_matrix_builder is not None
        if has_static == has_builder:
            raise ValueError(
                "ELineMarginalisedLikelihood: provide exactly one of "
                "`design_matrix` (static) or `design_matrix_builder` "
                "(per-call). Got "
                f"design_matrix={'set' if has_static else 'None'}, "
                f"design_matrix_builder={'set' if has_builder else 'None'}."
            )

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        # When the line wavelengths shift with redshift (or any other
        # free parameter), the design matrix must be rebuilt every
        # evaluation. The builder closure typically wraps
        # :func:`tengri.observation.eline_marginalization.build_eline_design_matrix`
        # with the model's wave_obs grid + LSF baked in.
        if self.design_matrix_builder is not None:
            design_matrix = self.design_matrix_builder(params or {})
        else:
            design_matrix = self.design_matrix
        residual = self.fnu_obs - prediction[self.channel]
        ln_l_marg, _a_hat, _a_cov = _marginalize_emission_lines(
            residual=residual,
            noise=self.fnu_err,
            design_matrix=design_matrix,
            prior_variance=self.prior_variance,
        )
        return ln_l_marg

    def declared_parameters(self):
        return []


# ─────────────────────────────────────────────────────────────────────
# Emission-line amplitudes — Cloudy-prior marginalisation
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CloudyELineMarginalisedLikelihood:
    r"""E-line marginalisation with a Cloudy-photoionisation prior.

    Same conjugate-Gaussian marginalisation as
    :class:`ELineMarginalisedLikelihood`, but the per-line prior
    means come from a Cloudy grid evaluated at the current
    ``(log_z, neb_logU)`` — both read from ``params`` at log-prob
    time, so the prior shifts with the sampler. ``prior_width_dex``
    sets the prior width around the Cloudy expectation.

    Parameters
    ----------
    fnu_obs, fnu_err : ndarray, shape (n_pixels,)
        Observed spectrum and 1-σ uncertainties [erg/s/cm²/Hz].
    design_matrix_builder : callable, keyword-only
        Closure that takes the params dict and returns a fresh
        ``(n_pixels, n_lines)`` design matrix — typically wraps
        :func:`tengri.inference.loss_functions._build_eline_G_eff`.
        Per-call rebuild is required because line wavelengths shift
        with redshift.
    line_wavelengths : ndarray, shape (n_lines,)
        Rest-frame line wavelengths in vacuum [Å]. Used by the Cloudy
        prior to look up expected line ratios.
    prior_width_dex : float, keyword-only
        Prior σ on log10 amplitude around the Cloudy expectation
        [dex]. Default 0.5.
    channel : str, keyword-only
        Prediction-dict key. Default ``"spec_fnu"``.

    Notes
    -----
    **JIT-compatible**: yes. Reads ``met_logzsol`` and ``neb_logU``
    from ``params`` (with sentinel defaults 0.0 and -3.0 to match the
    legacy fall-through that this adapter supersedes).
    """

    fnu_obs: jnp.ndarray
    fnu_err: jnp.ndarray
    _: KW_ONLY
    design_matrix_builder: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray]
    line_wavelengths: jnp.ndarray
    prior_width_dex: float = 0.5
    channel: str = "spec_fnu"
    name: str = "eline_marginalised_cloudy"

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        from tengri.observation.eline_priors import marginalize_emission_lines_cloudy

        params = params or {}
        design_matrix = self.design_matrix_builder(params)
        residual = self.fnu_obs - prediction[self.channel]
        log_z = params.get("met_logzsol", 0.0)
        neb_logU = params.get("neb_logU", -3.0)
        ln_l, _a_hat, _a_err = marginalize_emission_lines_cloudy(
            residual,
            self.fnu_err,
            design_matrix,
            log_z=log_z,
            neb_logU=neb_logU,
            line_wavelengths=self.line_wavelengths,
            prior_width_dex=self.prior_width_dex,
        )
        return ln_l

    def declared_parameters(self):
        return []


# ─────────────────────────────────────────────────────────────────────
# Emission-line amplitudes — explicit free-parameter fitting
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ELineFittedLikelihood:
    r"""Spectroscopy likelihood with explicit per-line amplitude params.

    Counterpart to :class:`ELineMarginalisedLikelihood`: instead of
    integrating the line amplitudes out analytically, they are
    explicit free parameters that the inference engine samples.
    Reads ``params[name]`` for each ``name`` in ``amplitude_names``,
    multiplies through the design matrix, and scores the residual
    with a diagonal Gaussian.

    Parameters
    ----------
    fnu_obs, fnu_err : ndarray, shape (n_pixels,)
        Observed spectrum and 1-σ uncertainties [erg/s/cm²/Hz].
    design_matrix_builder : callable, keyword-only
        Per-call closure (same shape as
        :class:`CloudyELineMarginalisedLikelihood.design_matrix_builder`).
    amplitude_names : tuple of str, keyword-only
        Param keys to read for each line amplitude. Order must match
        the columns of the design matrix.
    channel : str, keyword-only
        Prediction-dict key. Default ``"spec_fnu"``.

    Notes
    -----
    **JIT-compatible**: yes — the amplitude lookup is a simple list
    comprehension over a static tuple.
    """

    fnu_obs: jnp.ndarray
    fnu_err: jnp.ndarray
    _: KW_ONLY
    design_matrix_builder: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray]
    amplitude_names: tuple[str, ...] = ()
    channel: str = "spec_fnu"
    name: str = "eline_fitted"

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        params = params or {}
        design_matrix = self.design_matrix_builder(params)
        amplitudes = jnp.array([params[nm] for nm in self.amplitude_names])
        pred_with_lines = prediction[self.channel] + design_matrix @ amplitudes
        chi2 = diag_gaussian_chi2(pred_with_lines, self.fnu_obs, self.fnu_err)
        return -0.5 * chi2

    def declared_parameters(self):
        return list(self.amplitude_names)
