# SPDX-License-Identifier: BSD-3-Clause
"""Standardized forward model — the canonical inference path.

This module implements the **standardized parameterization** (ξ ~ N(0, I))
that every backend in :mod:`tengri.inference` ultimately consumes. It is
the principled formulation of single-Hamiltonian inference: priors live
inside per-parameter coordinate transforms so the loss collapses to
:math:`\\mathcal{H}(\\xi) = \\tfrac{1}{2}\\chi^2 + \\tfrac{1}{2}\\xi^\\top\\xi`,
independent of which distribution each free parameter was given.

Production inference currently routes through
:class:`tengri.Fitter` + :func:`tengri.inference.loss_functions.build_loss_fn`,
which implements the same standardization inline (the
``tengri.inference.loss_functions._unstandardize_parameters`` helper).
:class:`StandardizedForwardModel` here is the standalone, sampler-agnostic
encapsulation — used by the unit tests as the canonical reference for the
ξ-mapping math, and by future backends (Ray Tracing, vanilla MAP/Pathfinder)
that want a coordinate-mapping object without the full Fitter envelope.

Maps ξ ~ N(0, I) → predicted observables, absorbing ALL prior
structure into the forward model. The loss is always:

    H(ξ) = ½ χ²(data, f(ξ)) + ½ ξᵀξ

No separate prior terms. Works with any sampler (MAP, Ray Tracing,
NUTS, geoVI, MGVI) and unifies individual + hierarchical inference.

Usage::

    smodel = StandardizedForwardModel(model)
    loss = build_standardized_loss(smodel, data, noise)

    # Any sampler:
    chain = sample_raytrace(key, xi_init, lambda x: -loss(x), ...)
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.components.stellar.sfh.gp_sfh import compute_sqrt_power_drw
from tengri.inference.likelihoods.gaussian import standardized_residual
from tengri.utils.grid import make_log_age_grid


class StandardizedForwardModel:
    """Maps standardized latents ξ ~ N(0,I) → predicted observables.

    Each free parameter's prior distribution defines a differentiable
    transform ξ_k → θ_k via Distribution.unstandardize(). The correlated
    field couples PSD hyperparameters to the GP field through √P(σ,τ)·ξ.

    Parameters
    ----------
    model : Model
        The forward model (predict_photometry, predict_spectrum, etc.).
    psd_model : callable, optional
        Function(sigma, tau_yr, n_grid, log_ages) → sqrt_power array.
        Default: DRW. User can provide Extended Regulator, Flex-PSD, etc.
    """

    def __init__(self, model, psd_model: Callable | None = None):
        self.model = model
        self.spec = model.spec

        # Separate free and fixed parameters
        self._free_names = list(self.spec.free_params)
        self._fixed_values = self.spec.get_fixed_values()

        # Build transform registry: name → Distribution
        self._transforms = {}
        for name in self._free_names:
            self._transforms[name] = self.spec.get_distribution(name)

        self._stochastic = self.spec.stochastic
        self._n_grid = self.spec.n_grid

        # PSD model: default DRW, user can swap
        if psd_model is not None:
            self._psd_model = psd_model
        else:
            self._psd_model = self._default_drw_sqrt_power

        # Check if PSD params are free (accept both full and short names)
        self._psd_sigma_free = any(
            n in self._free_names for n in ("sfh_field_psd_sigma", "psd_sigma")
        )
        self._psd_tau_free = any(
            n in self._free_names for n in ("sfh_field_psd_tau_myr", "psd_tau_myr")
        )

        # Pre-compute log-age grid for GP
        if self._stochastic:
            self._log_ages = make_log_age_grid(self._n_grid)

    def _default_drw_sqrt_power(self, sigma, tau_yr, n_grid, log_ages):
        """Default DRW √P computation."""
        return compute_sqrt_power_drw(
            sigma,
            tau_yr,
            n_grid,
            log_ages,
        )

    # ── Domain: what latent variables exist ───────────────────────

    @property
    def domain(self) -> dict:
        """Standardized parameter domain mapping name to shape.

        Returns
        -------
        dict
            Mapping of parameter name to shape tuple (e.g. ``()`` for scalars,
            ``(n_grid,)`` for the GP field).
        """
        d = {}
        for name in self._free_names:
            d[name] = ()  # scalar
        if self._stochastic:
            d["sfh_field_xi"] = (self._n_grid,)  # GP white noise
        return d

    @property
    def n_latent(self) -> int:
        """Total number of latent dimensions.

        Returns
        -------
        int
            Number of scalar latent variables (free params + GP field if stochastic).
        """
        n = len(self._free_names)
        if self._stochastic:
            n += self._n_grid
        return n

    # ── Core transforms ───────────────────────────────────────────

    def xi_to_params(self, xi: dict) -> dict:
        """Map standardized latents → physical parameters.

        For the correlated field: if PSD params are free, √P depends
        on the current (σ, τ) values. The field x(t) = IFFT(√P · ξ_field)
        is computed here so the PSD-field coupling is natural.

        Parameters
        ----------
        xi : dict
            Standardized latent variables. Keys match self.domain.

        Returns
        -------
        dict
            Physical parameters ready for model.predict_*().
        """
        params = {}

        # Physical params: ξ → θ via each distribution's unstandardize
        for name in self._free_names:
            params[name] = self._transforms[name].unstandardize(xi[name])

        # Fixed params
        for name, val in self._fixed_values.items():
            params[name] = jnp.asarray(val)

        # Correlated field: build √P from current PSD params, apply to ξ
        xi_key = "sfh_field_xi" if "sfh_field_xi" in xi else "psd_xi"
        if self._stochastic and xi_key in xi:
            # Get sigma from whichever name is in params
            sigma = params.get(
                "sfh_field_psd_sigma",
                params.get(
                    "psd_sigma",
                    jnp.asarray(
                        self._fixed_values.get(
                            "sfh_field_psd_sigma", self._fixed_values.get("psd_sigma", 1.0)
                        )
                    ),
                ),
            )
            tau_myr = params.get(
                "sfh_field_psd_tau_myr",
                params.get(
                    "psd_tau_myr",
                    jnp.asarray(
                        self._fixed_values.get(
                            "sfh_field_psd_tau_myr", self._fixed_values.get("psd_tau_myr", 50.0)
                        )
                    ),
                ),
            )

            # Build √P from current PSD params (differentiable!)
            sqrt_power = self._psd_model(
                sigma,
                tau_myr * 1e6,  # convert Myr → yr
                self._n_grid,
                self._log_ages,
            )

            # x(t) = IFFT(√P · ξ_field) — the correlated field
            xi_field = xi[xi_key]
            x_field = jnp.fft.irfft(
                sqrt_power[: len(xi_field) // 2 + 1] * jnp.fft.rfft(xi_field),
                n=self._n_grid,
            )

            params["_correlated_field"] = x_field

            # Store xi under the key the Model expects
            params["sfh_field_xi"] = xi[xi_key]

        return params

    def params_to_xi(self, params: dict) -> dict:
        """Map physical parameters to standardized latents (for initialization).

        Parameters
        ----------
        params : dict
            Physical parameter dictionary, e.g. from a prior sample.

        Returns
        -------
        dict
            Standardized latent variables with keys matching ``self.domain``.
        """
        xi = {}
        for name in self._free_names:
            if name in params:
                xi[name] = self._transforms[name].standardize(jnp.asarray(params[name]))
            else:
                xi[name] = jnp.array(0.0)

        if self._stochastic:
            xi["sfh_field_xi"] = params.get(
                "sfh_field_xi", params.get("psd_xi", jnp.zeros(self._n_grid))
            )

        return xi

    # ── Forward pass ──────────────────────────────────────────────

    def predict(self, xi: dict, data_type: str = "photometry", wave_obs=None) -> jnp.ndarray:
        """Map standardized latents to predicted observables.

        Parameters
        ----------
        xi : dict
            Standardized latent variables with keys matching ``self.domain``.
        data_type : str
            Observation type: ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
        wave_obs : array_like or None
            Observed-frame wavelengths required for spectroscopy. [Angstrom]

        Returns
        -------
        ndarray
            Predicted flux vector matching the data layout. [erg/s/Hz]
        """
        params = self.xi_to_params(xi)

        if data_type == "photometry":
            return self.model.predict_photometry(params)
        elif data_type == "spectroscopy":
            return self.model.predict_spectrum(params, wave_obs)
        elif data_type == "joint":
            phot = self.model.predict_photometry(params)
            spec = self.model.predict_spectrum(params, wave_obs)
            return jnp.concatenate([phot, spec])
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def __call__(self, xi: dict) -> jnp.ndarray:
        """Shortcut: ξ → photometry."""
        return self.predict(xi, data_type="photometry")


# ── Loss function builders ────────────────────────────────────────


def build_standardized_loss(
    smodel: StandardizedForwardModel,
    data: jnp.ndarray,
    noise: jnp.ndarray,
    data_type: str = "photometry",
    wave_obs=None,
    data_mask: jnp.ndarray | None = None,
) -> Callable:
    """Build the unified loss function H(ξ) for a single galaxy.

    When no noise model is active (default):
        H(ξ) = ½ Σ_k ((d_k - m_k(ξ))/σ_k)² + ½ ξᵀξ

    When noise model is active (noise_frac_cal is free):
        H(ξ) = ½ Σ_k ((d_k - m_k)/σ_eff,k)² + Σ_k log(σ_eff,k) + ½ ξᵀξ

    where σ²_eff,k = σ²_obs,k + (f_cal · m_k)² and the log-determinant
    term prevents the trivial solution σ → ∞.

    When ``data_mask`` is provided, censored bands use the normal CDF
    likelihood instead of chi-squared.  Mask values: 0 = detected,
    1 = upper limit, -1 = lower limit.

    No prior penalty terms beyond ½ξᵀξ. The prior is absorbed into the
    transforms.

    Parameters
    ----------
    smodel : StandardizedForwardModel
        Standardized forward model wrapping a ``SEDModel``.
    data : array_like, shape (n_obs,)
        Observed fluxes or spectral values. [erg/s/Hz or consistent units]
    noise : array_like, shape (n_obs,)
        Observational uncertainties (1-sigma). Same units as ``data``.
    data_type : str
        Observation type: ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
    wave_obs : array_like or None
        Observed-frame wavelengths, required when ``data_type != "photometry"``.
        [Angstrom]
    data_mask : array_like or None
        Integer mask for censored data (0=detected, 1=upper limit, -1=lower limit).
        When provided, upper/lower limits use CDF likelihood instead of chi-squared.

    Returns
    -------
    loss_fn : callable
        Function mapping a flat latent vector ``xi_flat`` to a scalar loss value.
    unravel_fn : callable
        Function that reconstructs the structured ``xi`` dict from ``xi_flat``.
    """
    from tengri.observation.noise import (
        censored_neg_log_likelihood,
        get_noise_dof,
        has_noise_model,
        uses_student_t,
        variable_noise_hamiltonian,
    )

    use_variable_noise = has_noise_model(smodel.spec)
    noise_dof = get_noise_dof(smodel.spec) if uses_student_t(smodel.spec) else None
    use_censored = data_mask is not None

    xi_template = {}
    for name, shape in smodel.domain.items():
        if shape == ():
            xi_template[name] = jnp.array(0.0)
        else:
            xi_template[name] = jnp.zeros(shape)

    _, unravel_fn = ravel_pytree(xi_template)

    if use_censored:
        mask_arr = jnp.asarray(data_mask)

        def loss_fn(xi_flat: jnp.ndarray) -> jnp.ndarray:
            """Censored likelihood + prior: handles upper/lower limits in masked bands."""
            xi = unravel_fn(xi_flat)
            predicted = smodel.predict(xi, data_type=data_type, wave_obs=wave_obs)
            params = smodel.xi_to_params(xi)
            f_cal = params.get("noise_frac_cal", 0.0)

            e_lh = censored_neg_log_likelihood(
                data, noise, predicted, mask_arr, f_cal=f_cal, dof=noise_dof
            )
            prior = jnp.sum(xi_flat**2)
            return e_lh + 0.5 * prior

    elif use_variable_noise:

        def loss_fn(xi_flat: jnp.ndarray) -> jnp.ndarray:
            """Variable noise likelihood + prior: includes noise calibration uncertainty."""
            xi = unravel_fn(xi_flat)
            predicted = smodel.predict(xi, data_type=data_type, wave_obs=wave_obs)
            params = smodel.xi_to_params(xi)
            f_cal = params.get("noise_frac_cal", 0.0)

            e_lh = variable_noise_hamiltonian(data, noise, predicted, f_cal, dof=noise_dof)
            prior = jnp.sum(xi_flat**2)
            return e_lh + 0.5 * prior

    else:

        def loss_fn(xi_flat: jnp.ndarray) -> jnp.ndarray:
            """Gaussian likelihood + prior: standard chi-squared + isotropic prior."""
            xi = unravel_fn(xi_flat)
            predicted = smodel.predict(xi, data_type=data_type, wave_obs=wave_obs)

            chi2 = jnp.sum(standardized_residual(data, predicted, noise) ** 2)
            prior = jnp.sum(xi_flat**2)

            return 0.5 * chi2 + 0.5 * prior

    return loss_fn, unravel_fn


def build_hierarchical_loss(
    smodel: StandardizedForwardModel, galaxies: list, shared_names: list | None = None
) -> Callable:
    """Build hierarchical loss with shared parameters across a galaxy population.

    Supports variable noise model: when ``noise_frac_cal`` is free,
    the effective noise includes a calibration floor and the
    log-determinant penalty.

    Parameters
    ----------
    smodel : StandardizedForwardModel
        Standardized forward model for a single galaxy.
    galaxies : list of dict
        List of galaxy data dicts, each with keys ``"flux_obs"`` and ``"noise"``.
    shared_names : list of str or None
        Parameter names shared across all galaxies (e.g. PSD hyperparameters).
        Defaults to all PSD-prefixed free parameters.

    Returns
    -------
    loss_fn : callable
        Function mapping a flat hierarchical latent vector to the total loss.
    unravel_fn : callable
        Function reconstructing the structured latent dict from the flat vector.
    """
    from tengri.observation.noise import compute_effective_noise, has_noise_model

    use_variable_noise = has_noise_model(smodel.spec)
    n_gal = len(galaxies)

    if shared_names is None:
        # Default: share PSD params
        shared_names = [
            n
            for n in smodel._free_names
            if (n.startswith("psd_") or n.startswith("sfh_field_psd_"))
            and n not in ("psd_xi", "sfh_field_xi")
        ]

    per_galaxy_names = [n for n in smodel.domain if n not in shared_names]

    xi_template = {}
    for name in shared_names:
        xi_template[name] = jnp.array(0.0)
    for i in range(n_gal):
        for name in per_galaxy_names:
            shape = smodel.domain[name]
            if shape == ():
                xi_template[f"g{i}_{name}"] = jnp.array(0.0)
            else:
                xi_template[f"g{i}_{name}"] = jnp.zeros(shape)

    _, unravel_fn = ravel_pytree(xi_template)

    all_data = [jnp.asarray(g["flux_obs"]) for g in galaxies]
    all_noise = [jnp.asarray(g["noise"]) for g in galaxies]

    def loss_fn(xi_flat: jnp.ndarray) -> jnp.ndarray:
        """Compute total negative log-likelihood over all galaxies from a flat parameter vector."""
        xi_all = unravel_fn(xi_flat)

        total_lh = 0.0
        for i in range(n_gal):
            xi_i = {}
            for name in shared_names:
                xi_i[name] = xi_all[name]
            for name in per_galaxy_names:
                xi_i[name] = xi_all[f"g{i}_{name}"]

            predicted = smodel.predict(xi_i, data_type="photometry")

            if use_variable_noise:
                params_i = smodel.xi_to_params(xi_i)
                f_cal = params_i.get("noise_frac_cal", 0.0)
                sigma_eff = compute_effective_noise(all_noise[i], predicted, f_cal)
                total_lh += jnp.sum(standardized_residual(all_data[i], predicted, sigma_eff) ** 2)
                total_lh += 2.0 * jnp.sum(jnp.log(sigma_eff))
            else:
                total_lh += jnp.sum(
                    standardized_residual(all_data[i], predicted, all_noise[i]) ** 2
                )

        prior = jnp.sum(xi_flat**2)
        return 0.5 * total_lh + 0.5 * prior

    return loss_fn, unravel_fn
