# SPDX-License-Identifier: BSD-3-Clause
"""Analytically-marginalized :class:`Likelihood` adapters.

Wraps two existing math primitives:

- :class:`CalibrationMarginalizedLikelihood`,
  :func:`tengri.observation.calibration.marginalize_calibration`
  (Chebyshev polynomial calibration integrated out analytically).
- :class:`ELineMarginalizedLikelihood`,
  :func:`tengri.observation.eline_marginalization.marginalize_emission_lines`
  (linear emission-line amplitudes integrated out analytically).

Both are diagonal Gaussian under the hood; the marginalization is
analytic via standard Gaussian-conjugate algebra. They differ from
the simple :class:`GaussianLikelihood` only in that they integrate
out a nuisance vector before scoring the residuals, so they are
distinct *math types* (not just channel renames).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from tengri._cache_keys import KeyPolicy, content, derive_key, exclude, shape
from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2
from tengri.inference.likelihoods.protocol import resolve_channel_data
from tengri.observation.banded import BandedMatrix
from tengri.observation.calibration import (
    _ERR_FLOOR,
    calibration_basis,
    marginalize_calibration as _marginalize_calibration,
    marginalize_calibration_segments,
)
from tengri.observation.eline_marginalization import (
    marginalize_emission_lines as _marginalize_emission_lines,
    model_line_prior,
    unit_line_prior,
    unit_model_line_prior,
)
from tengri.observation.linear_marginal import marginalize_linear_nuisance
from tengri.utils.scale import representable_floor as _representable_floor

__all__ = [
    "CalibrationMarginalizedLikelihood",
    "CloudyELineMarginalizedLikelihood",
    "ELineFittedLikelihood",
    "ELineMarginalizedLikelihood",
    "ModelELineMarginalizedLikelihood",
]


# ─────────────────────────────────────────────────────────────────────
# Calibration polynomial, Chebyshev marginalization
# ─────────────────────────────────────────────────────────────────────


def _resolve_chol(chol_data, chol_key, data_args):
    """Return the banded factor bands read through ``data_args``, or ``None`` without a factor."""
    if chol_data is None:
        return None
    return resolve_channel_data(chol_data, chol_key, None, data_args)


@dataclass(frozen=True)
class CalibrationMarginalizedLikelihood:
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
    segments, n_segments : keyword-only
        Per-pixel camera segment id and the segment count; with them the calibration
        multiplier carries one free scale per segment. Default: none.
    cal_prior : {"gaussian", "scale_free"}, keyword-only
        Prior on the calibration coefficients (see
        :func:`~tengri.observation.calibration.marginalize_calibration_segments`).
    q_window : ndarray, shape (n_pixels,), keyword-only
        0/1 window of the mean multiplier ``q`` when ``has_q``.
    has_q : bool, keyword-only
        Condition on ``q = exp(params["cal_log_q"])``; the value then includes the
        coefficient prior's own marginal of ``ln q``. The log-normal prior on
        ``ln q`` is the declared prior of ``cal_log_q``. Default False.

    Notes
    -----
    **JIT-compatible**: yes, ``n_poly`` is a static argument of the
    underlying primitive.

    Returns the marginal log-likelihood, including the Gaussian normalization,
    from :func:`marginalize_calibration`, which evaluates the closed form of
    :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance` on the
    Chebyshev design. Sign matches the rest of the :class:`Likelihood` cohort
    (higher = better fit).
    """

    fnu_obs: jnp.ndarray
    fnu_err: jnp.ndarray
    wavelength: jnp.ndarray
    _: KW_ONLY
    n_poly: int = 3
    prior_sigma: float = 1.0
    channel: str = "spec_fnu"
    name: str = "calibration_marginalized"
    fnu_obs_key: str | None = None
    fnu_err_key: str | None = None
    data_slice: tuple[int, int] | None = None
    segments: jnp.ndarray | None = None
    n_segments: int = 0
    cal_prior: str = "gaussian"
    q_window: jnp.ndarray | None = None
    has_q: bool = False
    chol_data: jnp.ndarray | None = None
    chol_offsets: tuple[int, ...] | None = None
    chol_key: str | None = None

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
        data_args: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        obs_flux = resolve_channel_data(self.fnu_obs, self.fnu_obs_key, self.data_slice, data_args)
        obs_err = resolve_channel_data(self.fnu_err, self.fnu_err_key, self.data_slice, data_args)
        if self.n_segments > 0 or self.cal_prior != "gaussian" or self.has_q:
            return marginalize_calibration_segments(
                prediction[self.channel],
                obs_flux,
                obs_err,
                self.wavelength,
                _segments_or_zeros(self.segments, self.wavelength),
                n_segments=self.n_segments,
                n_poly=self.n_poly,
                prior_sigma=self.prior_sigma,
                cal_prior=self.cal_prior,
                log_q=params["cal_log_q"] if self.has_q else None,
                q_window=self.q_window,
                q_prior=True,
                chol_data=_resolve_chol(self.chol_data, self.chol_key, data_args),
                chol_offsets=self.chol_offsets,
            )[0]
        log_lik, _c_hat, _c_err = _marginalize_calibration(
            model_flux=prediction[self.channel],
            obs_flux=obs_flux,
            obs_err=obs_err,
            wavelength=self.wavelength,
            n_poly=self.n_poly,
            prior_sigma=self.prior_sigma,
            chol_data=_resolve_chol(self.chol_data, self.chol_key, data_args),
            chol_offsets=self.chol_offsets,
        )
        return log_lik

    def declared_parameters(self):
        return []

    def cache_key(self) -> tuple:
        """Return a hashable cache key: ``fnu_obs``/``fnu_err`` by shape, the rest by content.

        See :meth:`~tengri.inference.likelihoods.protocol.GaussianLikelihood.cache_key`
        for the rationale.
        """
        return derive_key(self, _CALIBRATION_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY)


_CALIBRATION_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY: KeyPolicy = {
    "fnu_obs": shape("per-galaxy data; program shape depends on length, not values"),
    "fnu_err": shape("per-galaxy data; program shape depends on length, not values"),
    "wavelength": content("wavelength grid is baked into the Chebyshev basis"),
    "n_poly": content("polynomial order changes the design matrix shape"),
    "prior_sigma": content("calibration prior width is baked into the marginal likelihood"),
    "channel": content("which prediction-dict key to read determines the graph"),
    "name": content("diagnostic identifier, cheap and harmless to key"),
    "fnu_obs_key": content("data_args routing key changes which traced input is read"),
    "fnu_err_key": content("data_args routing key changes which traced input is read"),
    "data_slice": content("channel slice changes which part of the traced input is read"),
    "segments": content("segment ids add calibration columns to the marginal"),
    "n_segments": content("segment count changes the design matrix shape"),
    "cal_prior": content("scale-free vs Gaussian coefficient prior changes the likelihood"),
    "q_window": content("the window sets which pixels define the mean multiplier"),
    "has_q": content("the fiber-fraction latent conditions the calibration block"),
    "chol_data": shape("banded factor bands; only their shape fixes the program"),
    "chol_offsets": content("diagonal offsets of the factor are static program structure"),
    "chol_key": content("data_args routing key changes which traced input is read"),
}


def _segments_or_zeros(segments, wavelength):
    """Return the segment ids, or an all-zero id array when the grid has no segments."""
    return jnp.zeros(wavelength.shape[0], dtype=int) if segments is None else segments


# ─────────────────────────────────────────────────────────────────────
# Emission-line amplitudes, analytic marginalization
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ELineMarginalizedLikelihood:
    r"""Spectroscopy likelihood with linear emission-line amplitudes
    integrated out analytically.

    Each emission line contributes a known *shape* (Gaussian profile
    centered at the rest-frame line wavelength) but with an *unknown
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
        Per-line shape evaluated at every pixel, built once via
        :func:`tengri.observation.eline_marginalization.build_eline_design_matrix`
        (or the broad-line / doublet-constrained variants). Held on
        ``self`` because line wavelengths and LSF widths are fixed.
    prior_variance : ndarray | None, keyword-only
        Per-line prior variance on the amplitude. ``None`` → flat
        (1e10), the original primitive's default.
    prior_sigma : ndarray | None, keyword-only
        Per-line prior width on the amplitude, given instead of
        ``prior_variance`` when the variance would underflow the working precision.
    channel : str, keyword-only
        Prediction-dict key. Default ``"spec_fnu"``.

    Notes
    -----
    **JIT-compatible**: yes.

    The continuum prediction
    (:attr:`prediction[channel]`) is the model SED *without* lines;
    the residual ``data - model`` is what the design matrix is fit
    to. Returns the marginal log-likelihood directly.

    The value is the exact marginal likelihood of the residual under the Gaussian
    line prior, with the normalization of
    :func:`~tengri.observation.eline_marginalization.marginalize_emission_lines`.
    When ``presence_key`` names an entry of ``data_args``, that entry is a pixel
    mask (sliced by ``data_slice``) and a masked pixel is absent from every term.
    """

    fnu_obs: jnp.ndarray
    fnu_err: jnp.ndarray
    _: KW_ONLY
    design_matrix: jnp.ndarray | None = None
    design_matrix_builder: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray] | None = None
    prior_variance: jnp.ndarray | None = None
    prior_sigma: jnp.ndarray | None = None
    channel: str = "spec_fnu"
    name: str = "eline_marginalized"
    fnu_obs_key: str | None = None
    fnu_err_key: str | None = None
    data_slice: tuple[int, int] | None = None
    presence_key: str | None = None
    chol_data: jnp.ndarray | None = None
    chol_offsets: tuple[int, ...] | None = None
    chol_key: str | None = None

    def __post_init__(self) -> None:
        # Exactly one of (static design matrix) | (per-call builder)
        # must be provided. Both or neither is a usage error.
        has_static = self.design_matrix is not None
        has_builder = self.design_matrix_builder is not None
        if self.prior_variance is not None and self.prior_sigma is not None:
            raise ValueError(
                "ELineMarginalizedLikelihood: provide prior_variance or prior_sigma, not both."
            )
        if has_static == has_builder:
            raise ValueError(
                "ELineMarginalizedLikelihood: provide exactly one of "
                "`design_matrix` (static) or `design_matrix_builder` "
                "(per-call). Got "
                f"design_matrix={'set' if has_static else 'None'}, "
                f"design_matrix_builder={'set' if has_builder else 'None'}."
            )

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
        data_args: Mapping[str, jnp.ndarray] | None = None,
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
        fnu_obs = resolve_channel_data(self.fnu_obs, self.fnu_obs_key, self.data_slice, data_args)
        fnu_err = resolve_channel_data(self.fnu_err, self.fnu_err_key, self.data_slice, data_args)
        residual = fnu_obs - prediction[self.channel]
        pix_mask = None
        if (
            data_args is not None
            and self.presence_key is not None
            and self.presence_key in data_args
        ):
            pix_mask = data_args[self.presence_key]
            if self.data_slice is not None:
                pix_mask = pix_mask[self.data_slice[0] : self.data_slice[1]]
        ln_l_marg, _a_hat, _a_cov = _marginalize_emission_lines(
            residual=residual,
            noise=fnu_err,
            design_matrix=design_matrix,
            prior_variance=self.prior_variance,
            prior_sigma=self.prior_sigma,
            pix_mask=pix_mask,
            chol_data=_resolve_chol(self.chol_data, self.chol_key, data_args),
            chol_offsets=self.chol_offsets,
        )
        return ln_l_marg

    def declared_parameters(self):
        return []

    def cache_key(self) -> tuple:
        """Return a hashable cache key: ``fnu_obs``/``fnu_err`` by shape, the rest by content.

        ``design_matrix_builder`` is excluded: a closure's qualname is
        identical regardless of what it captured (``baked()`` cannot see
        inside it), and its content -- which lines, wavelengths, doublet
        constraints -- is already keyed content on ``Fitter`` via every
        ``_eline_*`` row in ``ENGINE_POLICY``. Keying it directly would
        either raise (no ``cache_key``/dataclass/custom ``__repr__``) or,
        worse, alias two different design matrices under one key.
        """
        return derive_key(self, _ELINE_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY)


_ELINE_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY: KeyPolicy = {
    "fnu_obs": shape("per-galaxy data; program shape depends on length, not values"),
    "fnu_err": shape("per-galaxy data; program shape depends on length, not values"),
    "design_matrix": content("static per-line shape; a per-model structural input"),
    "design_matrix_builder": exclude(
        "a closure: content already covered by the Fitter's _eline_* ENGINE_POLICY "
        "rows (names/wavelengths/constraint matrix); keying the closure itself would "
        "alias two different design matrices under one key (baked() cannot see what "
        "it captured)"
    ),
    "prior_variance": content("per-line amplitude prior variance is baked into the marginal"),
    "prior_sigma": content("per-line amplitude prior width is baked into the marginal"),
    "channel": content("which prediction-dict key to read determines the graph"),
    "name": content("diagnostic identifier, cheap and harmless to key"),
    "fnu_obs_key": content("data_args routing key changes which traced input is read"),
    "fnu_err_key": content("data_args routing key changes which traced input is read"),
    "data_slice": content("channel slice changes which part of the traced input is read"),
    "chol_data": shape("banded factor bands; only their shape fixes the program"),
    "chol_offsets": content("diagonal offsets of the factor are static program structure"),
    "chol_key": content("data_args routing key changes which traced input is read"),
}


# ─────────────────────────────────────────────────────────────────────
# Emission-line amplitudes, Cloudy-prior marginalization
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CloudyELineMarginalizedLikelihood:
    r"""E-line marginalization with a Cloudy-photoionization prior.

    Same conjugate-Gaussian marginalization as
    :class:`ELineMarginalizedLikelihood`, but the per-line prior
    means come from a Cloudy grid evaluated at the current
    ``(log_z, neb_logU)``, both read from ``params`` at log-prob
    time, so the prior shifts with the sampler. ``prior_width_dex``
    sets the prior width around the Cloudy expectation.

    Parameters
    ----------
    fnu_obs, fnu_err : ndarray, shape (n_pixels,)
        Observed spectrum and 1-σ uncertainties [erg/s/cm²/Hz].
    design_matrix_builder : callable, keyword-only
        Closure that takes the params dict and returns a fresh
        ``(n_pixels, n_lines)`` design matrix, typically wraps
        ``tengri.inference.likelihood._build_eline_G_eff``.
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
    name: str = "eline_marginalized_cloudy"
    fnu_obs_key: str | None = None
    fnu_err_key: str | None = None
    data_slice: tuple[int, int] | None = None
    chol_data: jnp.ndarray | None = None
    chol_offsets: tuple[int, ...] | None = None
    chol_key: str | None = None

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
        data_args: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        from tengri.observation.eline_priors import marginalize_emission_lines_cloudy

        params = params or {}
        design_matrix = self.design_matrix_builder(params)
        fnu_obs = resolve_channel_data(self.fnu_obs, self.fnu_obs_key, self.data_slice, data_args)
        fnu_err = resolve_channel_data(self.fnu_err, self.fnu_err_key, self.data_slice, data_args)
        residual = fnu_obs - prediction[self.channel]
        log_z = params.get("met_logzsol", 0.0)
        neb_logU = params.get("neb_logU", -3.0)
        ln_l, _a_hat, _a_err = marginalize_emission_lines_cloudy(
            residual,
            fnu_err,
            design_matrix,
            log_z=log_z,
            neb_logU=neb_logU,
            line_wavelengths=self.line_wavelengths,
            prior_width_dex=self.prior_width_dex,
            chol_data=_resolve_chol(self.chol_data, self.chol_key, data_args),
            chol_offsets=self.chol_offsets,
        )
        return ln_l

    def declared_parameters(self):
        return []

    def cache_key(self) -> tuple:
        """Return a hashable cache key: ``fnu_obs``/``fnu_err`` by shape, the rest by content.

        See :meth:`ELineMarginalizedLikelihood.cache_key` for why
        ``design_matrix_builder`` is excluded.
        """
        return derive_key(self, _CLOUDY_ELINE_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY)


_CLOUDY_ELINE_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY: KeyPolicy = {
    "fnu_obs": shape("per-galaxy data; program shape depends on length, not values"),
    "fnu_err": shape("per-galaxy data; program shape depends on length, not values"),
    "design_matrix_builder": exclude(
        "a closure; see ELineMarginalizedLikelihood's row for the full reason"
    ),
    "line_wavelengths": content(
        "redundant with _eline_wavelengths (already content in ENGINE_POLICY), kept "
        "content here too for defense in depth: cheap and harmless"
    ),
    "prior_width_dex": content("Cloudy prior width in dex is baked into the marginal"),
    "channel": content("which prediction-dict key to read determines the graph"),
    "name": content("diagnostic identifier, cheap and harmless to key"),
    "fnu_obs_key": content("data_args routing key changes which traced input is read"),
    "fnu_err_key": content("data_args routing key changes which traced input is read"),
    "data_slice": content("channel slice changes which part of the traced input is read"),
    "chol_data": shape("banded factor bands; only their shape fixes the program"),
    "chol_offsets": content("diagonal offsets of the factor are static program structure"),
    "chol_key": content("data_args routing key changes which traced input is read"),
}


@dataclass(frozen=True)
class ModelELineMarginalizedLikelihood:
    r"""E-line marginalization with a Gaussian prior centred on the model's line prediction.

    Each line amplitude has the prior :math:`\mathcal{N}(\mu_l, (w\,\mu_l)^2)` with
    :math:`\mu_l` the attenuated model line luminosity projected onto the spectrum's
    flux-density scale (:func:`~tengri.observation.eline_marginalization.model_line_prior`).
    A catalog line the model does not predict keeps a flat prior.

    Parameters
    ----------
    fnu_obs, fnu_err : ndarray, shape (n_pixels,)
        Observed spectrum and 1-sigma uncertainties [erg/s/cm^2/Hz].
    design_matrix_builder : callable, keyword-only
        Closure returning the ``(n_pixels, n_lines)`` design matrix from ``params``.
    prior_inputs_builder : callable, keyword-only
        Closure returning ``(redshift, dl_cm, aperture_frac, transmission)`` from
        ``params`` [dimensionless, cm, dimensionless, dimensionless].
    match_index, matched : ndarray, shape (n_lines,), keyword-only
        Static catalog-to-model line map from
        :func:`~tengri.observation.eline_marginalization.match_model_lines`.
    line_wavelengths : ndarray, shape (n_lines,), keyword-only
        Rest-frame vacuum wavelengths of the independent catalog lines [Angstrom].
    prior_width_dex : float, keyword-only
        Prior width :math:`\Delta` in dex. Default 0.3.
    channel : str, keyword-only
        Prediction-dict key of the model spectrum. Default ``"spec_fnu"``.

    Notes
    -----
    **JIT-compatible**: yes. The prediction must carry
    ``"log_line_lums_attenuated"`` [dex]; the static map is a NumPy array.

    **Gradient-safe**: yes, with respect to the model spectrum, the line luminosities,
    the aperture fraction and the redshift.
    """

    fnu_obs: jnp.ndarray
    fnu_err: jnp.ndarray
    _: KW_ONLY
    design_matrix_builder: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray]
    prior_inputs_builder: Callable[[Mapping[str, jnp.ndarray]], tuple]
    match_index: jnp.ndarray
    matched: jnp.ndarray
    line_wavelengths: jnp.ndarray
    prior_width_dex: float = 0.3
    channel: str = "spec_fnu"
    name: str = "eline_marginalized_model"
    fnu_obs_key: str | None = None
    fnu_err_key: str | None = None
    data_slice: tuple[int, int] | None = None

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
        data_args: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        params = params or {}
        design = self.design_matrix_builder(params)
        fnu_obs = resolve_channel_data(self.fnu_obs, self.fnu_obs_key, self.data_slice, data_args)
        fnu_err = resolve_channel_data(self.fnu_err, self.fnu_err_key, self.data_slice, data_args)
        model_spec = prediction[self.channel]
        redshift, dl_cm, aperture_frac, transmission = self.prior_inputs_builder(params)
        prior = model_line_prior(
            prediction["log_line_lums_attenuated"],
            self.match_index,
            self.matched,
            self.line_wavelengths,
            redshift,
            dl_cm,
            aperture_frac,
            self.prior_width_dex,
            transmission=transmission,
        )
        unit = unit_model_line_prior(design, prior, noise=fnu_err)
        result = marginalize_linear_nuisance(
            fnu_obs,
            model_spec,
            unit.design,
            sigma=fnu_err,
            prior_mean=unit.prior_mean,
            prior_precision=unit.prior_precision,
            scale=unit.scale,
        )
        n_pix = fnu_obs.shape[0]
        return result.log_like + 0.5 * n_pix * math.log(2.0 * math.pi) + jnp.sum(jnp.log(fnu_err))

    def declared_parameters(self):
        return []

    def cache_key(self) -> tuple:
        """Return a hashable cache key: data by shape, the rest by content.

        ``design_matrix_builder`` and ``prior_inputs_builder`` are closures and are
        excluded, as in :meth:`ELineMarginalizedLikelihood.cache_key`.
        """
        return derive_key(self, _MODEL_ELINE_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY)


_MODEL_ELINE_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY: KeyPolicy = {
    "fnu_obs": shape("per-galaxy data; program shape depends on length, not values"),
    "fnu_err": shape("per-galaxy data; program shape depends on length, not values"),
    "design_matrix_builder": exclude("a closure; see ELineMarginalizedLikelihood's row"),
    "prior_inputs_builder": exclude("a closure over the model and fixed values"),
    "match_index": content("static catalog-to-model map is baked into the prior"),
    "matched": content("static flat/Gaussian split is baked into the prior"),
    "line_wavelengths": content("catalog wavelengths enter the prior's observed lines"),
    "prior_width_dex": content("prior width in dex is baked into the marginal"),
    "channel": content("which prediction-dict key to read determines the graph"),
    "name": content("diagnostic identifier, cheap and harmless to key"),
    "fnu_obs_key": content("data_args routing key changes which traced input is read"),
    "fnu_err_key": content("data_args routing key changes which traced input is read"),
    "data_slice": content("channel slice changes which part of the traced input is read"),
}


# ─────────────────────────────────────────────────────────────────────
# Emission-line amplitudes, explicit free-parameter fitting
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ELineFittedLikelihood:
    r"""Spectroscopy likelihood with explicit per-line amplitude params.

    Counterpart to :class:`ELineMarginalizedLikelihood`: instead of
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
        :class:`CloudyELineMarginalizedLikelihood.design_matrix_builder`).
    amplitude_names : tuple of str, keyword-only
        Param keys to read for each line amplitude. Order must match
        the columns of the design matrix.
    channel : str, keyword-only
        Prediction-dict key. Default ``"spec_fnu"``.

    Notes
    -----
    **JIT-compatible**: yes, the amplitude lookup is a simple list
    comprehension over a static tuple.
    """

    fnu_obs: jnp.ndarray
    fnu_err: jnp.ndarray
    _: KW_ONLY
    design_matrix_builder: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray]
    amplitude_names: tuple[str, ...] = ()
    channel: str = "spec_fnu"
    name: str = "eline_fitted"
    fnu_obs_key: str | None = None
    fnu_err_key: str | None = None
    data_slice: tuple[int, int] | None = None

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
        data_args: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        params = params or {}
        design_matrix = self.design_matrix_builder(params)
        amplitudes = jnp.array([params[nm] for nm in self.amplitude_names])
        pred_with_lines = prediction[self.channel] + design_matrix @ amplitudes
        fnu_obs = resolve_channel_data(self.fnu_obs, self.fnu_obs_key, self.data_slice, data_args)
        fnu_err = resolve_channel_data(self.fnu_err, self.fnu_err_key, self.data_slice, data_args)
        chi2 = diag_gaussian_chi2(pred_with_lines, fnu_obs, fnu_err)
        return -0.5 * chi2

    def declared_parameters(self):
        return list(self.amplitude_names)

    def cache_key(self) -> tuple:
        """Return a hashable cache key: ``fnu_obs``/``fnu_err`` by shape, the rest by content.

        See :meth:`ELineMarginalizedLikelihood.cache_key` for why
        ``design_matrix_builder`` is excluded.
        """
        return derive_key(self, _ELINE_FITTED_LIKELIHOOD_CACHE_KEY_POLICY)


_ELINE_FITTED_LIKELIHOOD_CACHE_KEY_POLICY: KeyPolicy = {
    "fnu_obs": shape("per-galaxy data; program shape depends on length, not values"),
    "fnu_err": shape("per-galaxy data; program shape depends on length, not values"),
    "design_matrix_builder": exclude(
        "a closure; see ELineMarginalizedLikelihood's row for the full reason"
    ),
    "amplitude_names": content(
        "which latent parameters carry the line amplitudes determines the unravel "
        "structure; redundant with _eline_amplitude_names, kept content here too"
    ),
    "channel": content("which prediction-dict key to read determines the graph"),
    "name": content("diagnostic identifier, cheap and harmless to key"),
    "fnu_obs_key": content("data_args routing key changes which traced input is read"),
    "fnu_err_key": content("data_args routing key changes which traced input is read"),
    "data_slice": content("channel slice changes which part of the traced input is read"),
}


# ─────────────────────────────────────────────────────────────────────
# Combined: calibration polynomial + emission-line amplitudes
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationELineMarginalizedLikelihood:
    r"""Spectroscopy likelihood with the calibration polynomial and the emission-line
    amplitudes marginalized jointly and exactly.

    The calibrated model is
    :math:`m(\lambda)\,[1 + \sum_k c_k T_k(\lambda)] + \sum_j a_j G_j(\lambda)`.
    The calibration block multiplies the model and the line block adds to it, so
    both blocks are jointly linear in :math:`(c, a)`. The value is the marginal
    likelihood of the observed spectrum with every coefficient integrated out under
    its Gaussian prior, and every log-determinant term of the joint is kept.

    Parameters
    ----------
    fnu_obs, fnu_err : ndarray, shape (n_pixels,)
        Observed spectrum and 1-σ uncertainties [erg/s/cm²/Hz].
    wavelength : ndarray, shape (n_pixels,)
        Wavelength grid for the Chebyshev calibration basis [Å].
    design_matrix_builder : callable, keyword-only
        Per-call closure rebuilding the line design matrix. Required,
        line wavelengths shift with redshift. See
        :class:`ELineMarginalizedLikelihood`.
    n_poly, prior_sigma : keyword-only
        Calibration polynomial: order and per-coefficient prior σ.
    eline_prior_type : str, keyword-only
        ``"flat"`` (default, Gaussian with per-line variance
        ``eline_prior_sigma**2``), ``"cloudy"`` (Cloudy-grid prior
        evaluated at ``params["met_logzsol"]`` /
        ``params["neb_logU"]``) or ``"model"`` (Gaussian centred on the
        model's own attenuated line prediction, see
        :class:`ModelELineMarginalizedLikelihood`).
    eline_prior_sigma : float, keyword-only
        Per-line prior σ for the flat case [erg/s/cm²/Hz]. Default 1e10.
    eline_line_wavelengths : ndarray | None, keyword-only
        Rest-frame line wavelengths (Cloudy case only) [Å].
    eline_prior_width_dex : float, keyword-only
        Cloudy or model prior width [dex]. Default 0.5.
    prior_inputs_builder, match_index, matched : keyword-only
        Model prior only: the closure returning
        ``(redshift, dl_cm, aperture_frac, transmission)`` from ``params`` and the static
        catalog-to-model line map, as in
        :class:`ModelELineMarginalizedLikelihood`. The prediction must carry
        ``"log_line_lums_attenuated"`` [dex].
    channel : str, keyword-only
        Prediction-dict key. Default ``"spec_fnu"``.
    segments, n_segments : keyword-only
        Per-pixel camera segment id and the segment count; with them the calibration
        multiplier carries one free scale per segment. Default: none.
    cal_prior : {"gaussian", "scale_free"}, keyword-only
        Prior on the calibration coefficients (see
        :func:`~tengri.observation.calibration.marginalize_calibration_segments`).
    q_window : ndarray, shape (n_pixels,), keyword-only
        0/1 window of the mean multiplier ``q`` when ``has_q``.
    has_q : bool, keyword-only
        Condition on ``q = exp(params["cal_log_q"])``; the value then includes the
        coefficient prior's own marginal of ``ln q``. The log-normal prior on
        ``ln q`` is the declared prior of ``cal_log_q``. Default False.

    Notes
    -----
    **JIT-compatible**: yes; ``eline_prior_type`` is static at
    construction time.

    The joint marginal is one call to
    :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance` with the
    design :math:`X = [m\,T \mid G]`, where :math:`T` is the calibration basis, the
    model offset :math:`m_0 = m`, and the block prior precision
    :math:`\mathrm{blockdiag}(I/\sigma_{\rm prior}^2,\ \Lambda^{-1})`. For the Cloudy
    prior the line block carries the Cloudy-shifted prior means. The value omits the
    constant :math:`-\tfrac12 N \ln 2\pi` and :math:`-\sum_i \ln\sigma_i` of the
    normalized Gaussian, as :class:`ELineMarginalizedLikelihood` does.
    """

    fnu_obs: jnp.ndarray
    fnu_err: jnp.ndarray
    wavelength: jnp.ndarray
    _: KW_ONLY
    design_matrix_builder: Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray]
    n_poly: int = 3
    prior_sigma: float = 1.0
    eline_prior_type: str = "flat"
    eline_prior_sigma: float = 1e10
    eline_line_wavelengths: jnp.ndarray | None = None
    eline_prior_width_dex: float = 0.5
    prior_inputs_builder: Callable[[Mapping[str, jnp.ndarray]], tuple] | None = None
    match_index: jnp.ndarray | None = None
    matched: jnp.ndarray | None = None
    channel: str = "spec_fnu"
    name: str = "calibration_eline_marginalized"
    fnu_obs_key: str | None = None
    fnu_err_key: str | None = None
    data_slice: tuple[int, int] | None = None
    segments: jnp.ndarray | None = None
    n_segments: int = 0
    cal_prior: str = "gaussian"
    q_window: jnp.ndarray | None = None
    has_q: bool = False
    chol_data: jnp.ndarray | None = None
    chol_offsets: tuple[int, ...] | None = None
    chol_key: str | None = None

    def _line_prior(self, params: Mapping[str, jnp.ndarray], n_lines: int):
        """Return the line prior mean and width (standard deviation) for this parameter set."""
        if self.eline_prior_type == "cloudy":
            from tengri.observation.eline_priors import cloudy_line_priors

            means, sigmas = cloudy_line_priors(
                log_z=params.get("met_logzsol", 0.0),
                neb_logU=params.get("neb_logU", -3.0),
                line_wavelengths=self.eline_line_wavelengths,
                prior_width_dex=self.eline_prior_width_dex,
            )
            return means, sigmas
        return jnp.zeros(n_lines), jnp.full(n_lines, self.eline_prior_sigma)

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
        data_args: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        params = params or {}
        model_spec = prediction[self.channel]
        fnu_obs = resolve_channel_data(self.fnu_obs, self.fnu_obs_key, self.data_slice, data_args)
        fnu_err = resolve_channel_data(self.fnu_err, self.fnu_err_key, self.data_slice, data_args)
        design_lines = self.design_matrix_builder(params)
        n_lines = design_lines.shape[1]
        if self.n_segments > 0 or self.cal_prior != "gaussian" or self.has_q:
            chol_data = _resolve_chol(self.chol_data, self.chol_key, data_args)
            return self._segment_log_prob(
                model_spec, fnu_obs, fnu_err, design_lines, params, chol_data
            )
        basis = calibration_basis(
            self.wavelength, self.wavelength.min(), self.wavelength.max(), self.n_poly
        )
        chol_data = _resolve_chol(self.chol_data, self.chol_key, data_args)
        if chol_data is None:
            sigma = jnp.maximum(fnu_err, _representable_floor(_ERR_FLOOR))
            whitening = {"sigma": sigma}
        else:
            sigma = chol_data[0]
            whitening = {
                "chol_lower": BandedMatrix(np.asarray(self.chol_offsets, dtype=int), chol_data)
            }
        if self.eline_prior_type == "model":
            redshift, dl_cm, aperture_frac, transmission = self.prior_inputs_builder(params)
            model_prior = model_line_prior(
                prediction["log_line_lums_attenuated"],
                self.match_index,
                self.matched,
                self.eline_line_wavelengths,
                redshift,
                dl_cm,
                aperture_frac,
                self.eline_prior_width_dex,
                transmission=transmission,
            )
            line = unit_model_line_prior(design_lines, model_prior, noise=sigma)
        else:
            line_mean, line_width = self._line_prior(params, n_lines)
            line = unit_line_prior(design_lines, line_width, noise=sigma, prior_mean=line_mean)
        design = jnp.concatenate([model_spec[:, None] * basis, line.design], axis=1)
        prior_mean = jnp.concatenate([jnp.zeros(self.n_poly), line.prior_mean])
        prior_precision = block_diag(
            jnp.eye(self.n_poly) / self.prior_sigma**2, line.prior_precision
        )
        result = marginalize_linear_nuisance(
            fnu_obs,
            model_spec,
            design,
            **whitening,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            scale=line.scale,
        )
        n_pix = fnu_obs.shape[0]
        return result.log_like + 0.5 * n_pix * math.log(2.0 * math.pi) + jnp.sum(jnp.log(sigma))

    def _segment_log_prob(self, model_spec, fnu_obs, fnu_err, design_lines, params, chol_data):
        """Joint marginal with segment scales, a scale-free prior or the fiber-fraction latent."""
        err = jnp.maximum(fnu_err, _representable_floor(_ERR_FLOOR))
        noise = err if chol_data is None else chol_data[0]
        line_mean, line_width = self._line_prior(params, design_lines.shape[1])
        line = unit_line_prior(design_lines, line_width, noise=noise, prior_mean=line_mean)
        marginal = marginalize_calibration_segments(
            model_spec,
            fnu_obs,
            err,
            self.wavelength,
            _segments_or_zeros(self.segments, self.wavelength),
            n_segments=self.n_segments,
            n_poly=self.n_poly,
            prior_sigma=self.prior_sigma,
            cal_prior=self.cal_prior,
            lines=line,
            log_q=params["cal_log_q"] if self.has_q else None,
            q_window=self.q_window,
            q_prior=True,
            chol_data=chol_data,
            chol_offsets=self.chol_offsets,
        )[0]
        n_pix = fnu_obs.shape[0]
        return marginal + 0.5 * n_pix * math.log(2.0 * math.pi) + jnp.sum(jnp.log(noise))

    def declared_parameters(self):
        return []

    def cache_key(self) -> tuple:
        """Return a hashable cache key: ``fnu_obs``/``fnu_err`` by shape, the rest by content.

        See :meth:`ELineMarginalizedLikelihood.cache_key` for why
        ``design_matrix_builder`` is excluded.
        """
        return derive_key(self, _CALIBRATION_ELINE_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY)


_CALIBRATION_ELINE_MARGINALIZED_LIKELIHOOD_CACHE_KEY_POLICY: KeyPolicy = {
    "fnu_obs": shape("per-galaxy data; program shape depends on length, not values"),
    "fnu_err": shape("per-galaxy data; program shape depends on length, not values"),
    "wavelength": content("wavelength grid is baked into the Chebyshev basis"),
    "design_matrix_builder": exclude(
        "a closure; see ELineMarginalizedLikelihood's row for the full reason"
    ),
    "n_poly": content("calibration polynomial order changes the design matrix shape"),
    "prior_sigma": content("calibration prior width is baked into the marginal likelihood"),
    "eline_prior_type": content("flat vs Cloudy line-amplitude prior changes the likelihood"),
    "eline_prior_sigma": content("flat-prior per-line sigma is baked into the marginal"),
    "eline_line_wavelengths": content(
        "redundant with _eline_wavelengths (already content in ENGINE_POLICY), kept "
        "content here too for defense in depth: cheap and harmless"
    ),
    "eline_prior_width_dex": content("Cloudy prior width in dex is baked into the marginal"),
    "prior_inputs_builder": exclude("a closure over the model and fixed values"),
    "match_index": content("static catalog-to-model map is baked into the prior"),
    "matched": content("static flat/Gaussian split is baked into the prior"),
    "channel": content("which prediction-dict key to read determines the graph"),
    "name": content("diagnostic identifier, cheap and harmless to key"),
    "fnu_obs_key": content("data_args routing key changes which traced input is read"),
    "fnu_err_key": content("data_args routing key changes which traced input is read"),
    "data_slice": content("channel slice changes which part of the traced input is read"),
    "segments": content("segment ids add calibration columns to the marginal"),
    "n_segments": content("segment count changes the design matrix shape"),
    "cal_prior": content("scale-free vs Gaussian coefficient prior changes the likelihood"),
    "q_window": content("the window sets which pixels define the mean multiplier"),
    "has_q": content("the fiber-fraction latent conditions the calibration block"),
    "chol_data": shape("banded factor bands; only their shape fixes the program"),
    "chol_offsets": content("diagonal offsets of the factor are static program structure"),
    "chol_key": content("data_args routing key changes which traced input is read"),
}
