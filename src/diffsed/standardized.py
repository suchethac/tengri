"""Standardized forward model for unified inference.

Maps ξ ~ N(0, I) → predicted observables, absorbing ALL prior
structure into the forward model. The loss is always:

    H(ξ) = ½ χ²(data, f(ξ)) + ½ ξᵀξ

No separate prior terms. Works with any sampler (MAP, Ray Tracing,
NUTS, geoVI, MGVI) and unifies individual + hierarchical inference.

Usage:
    smodel = StandardizedForwardModel(model)
    loss = build_standardized_loss(smodel, data, noise)

    # Any sampler:
    chain = sample_raytrace(key, xi_init, lambda x: -loss(x), ...)
"""

from __future__ import annotations

from typing import Optional, Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from diffsed.distributions import Distribution, Fixed
from diffsed.models.sfh.gp_sfh import compute_sqrt_power_drw
from diffsed.utils.grid import make_log_age_grid


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

    def __init__(self, model, psd_model: Optional[Callable] = None):
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

        # Check if PSD params are free
        self._psd_sigma_free = "psd_sigma" in self._free_names
        self._psd_tau_free = "psd_tau_myr" in self._free_names

        # Pre-compute log-age grid for GP
        if self._stochastic:
            self._log_ages = make_log_age_grid(self._n_grid)

    def _default_drw_sqrt_power(self, sigma, tau_yr, n_grid, log_ages):
        """Default DRW √P computation."""
        return compute_sqrt_power_drw(
            sigma, tau_yr, n_grid, log_ages,
        )

    # -------------------------------------------------------------------
    # Domain: what latent variables exist
    # -------------------------------------------------------------------

    @property
    def domain(self) -> dict:
        """Standardized parameter domain {name: shape}.

        Returns dict compatible with NIFTy's ShapeWithDtype.
        """
        d = {}
        for name in self._free_names:
            d[name] = ()  # scalar
        if self._stochastic:
            d["psd_xi"] = (self._n_grid,)  # GP white noise
        return d

    @property
    def n_latent(self) -> int:
        """Total number of latent dimensions."""
        n = len(self._free_names)
        if self._stochastic:
            n += self._n_grid
        return n

    # -------------------------------------------------------------------
    # Core transforms
    # -------------------------------------------------------------------

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
        if self._stochastic and "psd_xi" in xi:
            sigma = params.get(
                "psd_sigma",
                jnp.asarray(self._fixed_values.get("psd_sigma", 1.0))
            )
            tau_myr = params.get(
                "psd_tau_myr",
                jnp.asarray(self._fixed_values.get("psd_tau_myr", 50.0))
            )

            # Build √P from current PSD params (differentiable!)
            sqrt_power = self._psd_model(
                sigma, tau_myr * 1e6,  # convert Myr → yr
                self._n_grid, self._log_ages,
            )

            # x(t) = IFFT(√P · ξ_field) — the correlated field
            xi_field = xi["psd_xi"]
            x_field = jnp.fft.irfft(
                sqrt_power[:len(xi_field) // 2 + 1] * jnp.fft.rfft(xi_field),
                n=self._n_grid,
            )

            # Store as _correlated_field for Model to use directly
            params["_correlated_field"] = x_field

        return params

    def params_to_xi(self, params: dict) -> dict:
        """Inverse: physical params → standardized (for initialization).

        Parameters
        ----------
        params : dict
            Physical parameter values.

        Returns
        -------
        dict
            Standardized latent variables.
        """
        xi = {}
        for name in self._free_names:
            if name in params:
                xi[name] = self._transforms[name].standardize(
                    jnp.asarray(params[name])
                )
            else:
                xi[name] = jnp.array(0.0)

        if self._stochastic:
            xi["psd_xi"] = params.get("psd_xi", jnp.zeros(self._n_grid))

        return xi

    # -------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------

    def predict(self, xi: dict, data_type: str = "photometry",
                wave_obs=None) -> jnp.ndarray:
        """Full forward model: ξ → predicted observables.

        Parameters
        ----------
        xi : dict
            Standardized latent variables.
        data_type : str
            "photometry", "spectroscopy", or "joint".
        wave_obs : array, optional
            Observed wavelengths for spectroscopy.

        Returns
        -------
        array
            Predicted observables.
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


# -------------------------------------------------------------------
# Loss function builders
# -------------------------------------------------------------------

def build_standardized_loss(smodel: StandardizedForwardModel,
                            data: jnp.ndarray, noise: jnp.ndarray,
                            data_type: str = "photometry",
                            wave_obs=None) -> Callable:
    """Build the unified loss function.

    The loss is ALWAYS:
        H(ξ) = ½ Σ_k ((d_k - m_k(ξ))/σ_k)² + ½ ξᵀξ

    No prior penalty terms. The prior is absorbed into the transforms.

    Parameters
    ----------
    smodel : StandardizedForwardModel
    data : array
        Observed data.
    noise : array
        1-sigma uncertainties.
    data_type : str
    wave_obs : array, optional

    Returns
    -------
    callable
        loss_fn(xi_flat) → scalar. Takes a flat 1D array.
    """
    # Build the unravel function from a template
    xi_template = {}
    for name, shape in smodel.domain.items():
        if shape == ():
            xi_template[name] = jnp.array(0.0)
        else:
            xi_template[name] = jnp.zeros(shape)

    _, unravel_fn = ravel_pytree(xi_template)

    def loss_fn(xi_flat: jnp.ndarray) -> jnp.ndarray:
        xi = unravel_fn(xi_flat)
        predicted = smodel.predict(xi, data_type=data_type, wave_obs=wave_obs)

        chi2 = jnp.sum(((data - predicted) / noise) ** 2)
        prior = jnp.sum(xi_flat ** 2)  # ALWAYS just ½ξᵀξ

        return 0.5 * chi2 + 0.5 * prior

    return loss_fn, unravel_fn


def build_hierarchical_loss(smodel: StandardizedForwardModel,
                            galaxies: list,
                            shared_names: list = None) -> Callable:
    """Build hierarchical loss with shared parameters.

    Parameters
    ----------
    smodel : StandardizedForwardModel
    galaxies : list of dict
        Each has 'flux_obs' and 'noise'.
    shared_names : list of str, optional
        Parameter names to share across galaxies.
        Default: PSD params if free.

    Returns
    -------
    callable
        loss_fn(xi_flat) → scalar.
    """
    n_gal = len(galaxies)

    if shared_names is None:
        # Default: share PSD params
        shared_names = [n for n in smodel._free_names
                        if n.startswith("psd_") and n != "psd_xi"]

    per_galaxy_names = [n for n in smodel.domain
                        if n not in shared_names]

    # Build the flat vector template
    xi_template = {}
    # Shared params
    for name in shared_names:
        xi_template[name] = jnp.array(0.0)
    # Per-galaxy params
    for i in range(n_gal):
        for name in per_galaxy_names:
            shape = smodel.domain[name]
            if shape == ():
                xi_template[f"g{i}_{name}"] = jnp.array(0.0)
            else:
                xi_template[f"g{i}_{name}"] = jnp.zeros(shape)

    _, unravel_fn = ravel_pytree(xi_template)

    # Precompute data
    all_data = [jnp.asarray(g["flux_obs"]) for g in galaxies]
    all_noise = [jnp.asarray(g["noise"]) for g in galaxies]

    def loss_fn(xi_flat: jnp.ndarray) -> jnp.ndarray:
        xi_all = unravel_fn(xi_flat)

        total_chi2 = 0.0
        for i in range(n_gal):
            # Build per-galaxy xi: shared + galaxy's own
            xi_i = {}
            for name in shared_names:
                xi_i[name] = xi_all[name]
            for name in per_galaxy_names:
                xi_i[name] = xi_all[f"g{i}_{name}"]

            predicted = smodel.predict(xi_i, data_type="photometry")
            total_chi2 += jnp.sum(
                ((all_data[i] - predicted) / all_noise[i]) ** 2
            )

        prior = jnp.sum(xi_flat ** 2)
        return 0.5 * total_chi2 + 0.5 * prior

    return loss_fn, unravel_fn
