"""Parameter specification for diffsed models.

ParamSpec defines all model parameters: their names, distributions (or fixed
values), and physical bounds. A single ParamSpec is used for both mock
generation (sampling from priors) and inference (defining the prior).

Usage:
    from diffsed import ParamSpec, Uniform, Gaussian, Fixed

    spec = ParamSpec(
        sfh_alpha        = Uniform(0.5, 3.0),
        sfh_beta         = Uniform(0.3, 2.0),
        sfh_tau_peak_gyr = Uniform(0.5, 10.0),
        sfh_peak_sfr     = Uniform(0.1, 50.0),
        met_logzsol      = Gaussian(-0.3, 0.2, lo=-2.0, hi=0.2),
        dust_tau_bc      = Uniform(0.0, 4.0),
        dust_tau_diff    = 0.3,          # fixed
        dust_slope       = -0.7,         # fixed
        redshift         = 0.1,          # fixed
        stochastic       = False,
    )

    params = spec.sample(jax.random.PRNGKey(0))
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diffsed.distributions import (
    Distribution,
    Fixed,
    Uniform,
    resolve_shorthand,
)


# ---------------------------------------------------------------------------
# Parameter registry: known parameters with physical bound constraints
# ---------------------------------------------------------------------------

# (param_name, description, bound_check_fn, bound_error_message)
_PARAM_REGISTRY = {
    "sfh_alpha":         ("DPL falling slope",          lambda lo, hi: lo > 0,  "must have lo > 0"),
    "sfh_beta":          ("DPL rising slope",            lambda lo, hi: lo > 0,  "must have lo > 0"),
    "sfh_tau_peak_gyr":  ("DPL turnover time (Gyr)",     lambda lo, hi: lo > 0,  "must have lo > 0"),
    "sfh_peak_sfr":      ("Peak SFR (Msun/yr)",          lambda lo, hi: lo > 0,  "must have lo > 0"),
    "psd_sigma":         ("PSD amplitude (dex)",          lambda lo, hi: lo >= 0, "must have lo >= 0"),
    "psd_tau_myr":       ("PSD timescale (Myr)",          lambda lo, hi: lo > 0,  "must have lo > 0"),
    "met_logzsol":       ("log10(Z/Zsun)",                lambda lo, hi: True,    ""),
    "dust_tau_bc":       ("Birth cloud optical depth",    lambda lo, hi: lo >= 0, "must have lo >= 0"),
    "dust_tau_diff":     ("Diffuse ISM optical depth",    lambda lo, hi: lo >= 0, "must have lo >= 0"),
    "dust_slope":        ("Dust power-law index",         lambda lo, hi: True,    ""),
    "redshift":          ("Source redshift",               lambda lo, hi: lo >= 0, "must have lo >= 0"),
}

VALID_PARAM_NAMES = frozenset(_PARAM_REGISTRY.keys())
SETTINGS_KEYS = frozenset({"stochastic", "n_grid", "mean_sfh_type"})


# ---------------------------------------------------------------------------
# ParamSpec class
# ---------------------------------------------------------------------------

class ParamSpec:
    """Parameter specification defining model parameters and their priors.

    Parameters are specified as keyword arguments. Each can be:
    - A scalar (int/float) → Fixed value
    - A tuple (lo, hi) → Uniform prior
    - A Distribution object (Uniform, Gaussian, LogUniform, Fixed)

    Settings (not parameters):
    - stochastic (bool): Enable GP stochastic SFH component. Default: False.
    - n_grid (int): GP grid size. Only used if stochastic=True. Default: 256.
    - mean_sfh_type (str): Mean SFH model. Default: "double_powerlaw".
    """

    def __init__(self, **kwargs):
        # Separate settings from parameters
        self._stochastic = bool(kwargs.pop("stochastic", False))
        self._n_grid = int(kwargs.pop("n_grid", 256))
        self._mean_sfh_type = str(kwargs.pop("mean_sfh_type", "double_powerlaw"))

        # Validate parameter names
        for name in kwargs:
            if name not in VALID_PARAM_NAMES:
                raise ValueError(
                    f"Unknown parameter '{name}'. "
                    f"Valid parameters: {sorted(VALID_PARAM_NAMES)}"
                )

        # Resolve shorthands and store distributions
        self._distributions: dict[str, Distribution] = {}
        for name in sorted(VALID_PARAM_NAMES):
            if name in kwargs:
                self._distributions[name] = resolve_shorthand(kwargs[name])
            else:
                # Parameters not specified get sensible defaults
                self._distributions[name] = self._default_distribution(name)

        # Validate physical bounds
        self._validate_bounds()

        # Validate stochastic consistency
        if self._stochastic:
            psd_sigma = self._distributions.get("psd_sigma")
            psd_tau = self._distributions.get("psd_tau_myr")
            if psd_sigma is None or psd_tau is None:
                raise ValueError(
                    "stochastic=True requires psd_sigma and psd_tau_myr"
                )

    def _default_distribution(self, name: str) -> Distribution:
        """Default distributions for parameters not explicitly specified."""
        defaults = {
            "sfh_alpha":         Uniform(0.1, 5.0),
            "sfh_beta":          Uniform(0.1, 3.0),
            "sfh_tau_peak_gyr":  Uniform(0.1, 12.0),
            "sfh_peak_sfr":      Uniform(0.01, 200.0),
            "psd_sigma":         Fixed(0.0),
            "psd_tau_myr":       Fixed(50.0),
            "met_logzsol":       Uniform(-2.0, 0.2),
            "dust_tau_bc":       Uniform(0.0, 4.0),
            "dust_tau_diff":     Uniform(0.0, 3.0),
            "dust_slope":        Fixed(-0.7),
            "redshift":          Fixed(0.1),
        }
        return defaults[name]

    def _validate_bounds(self):
        """Check that distribution bounds respect physical constraints."""
        for name, dist in self._distributions.items():
            if dist.is_fixed:
                lo = hi = dist.bounds[0]
            else:
                lo, hi = dist.bounds

            desc, check_fn, err_msg = _PARAM_REGISTRY[name]
            if not check_fn(lo, hi):
                raise ValueError(
                    f"Parameter '{name}' ({desc}): bounds ({lo}, {hi}) "
                    f"violate physical constraint: {err_msg}"
                )

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def stochastic(self) -> bool:
        return self._stochastic

    @property
    def n_grid(self) -> int:
        return self._n_grid

    @property
    def mean_sfh_type(self) -> str:
        return self._mean_sfh_type

    @property
    def all_params(self) -> list[str]:
        """All parameter names (sorted, excludes settings)."""
        return sorted(self._distributions.keys())

    @property
    def free_params(self) -> list[str]:
        """Names of free (non-fixed) parameters."""
        return sorted(k for k, d in self._distributions.items() if not d.is_fixed)

    @property
    def fixed_params(self) -> list[str]:
        """Names of fixed parameters."""
        return sorted(k for k, d in self._distributions.items() if d.is_fixed)

    @property
    def n_free(self) -> int:
        """Number of free parameters (excludes psd_xi)."""
        return len(self.free_params)

    # -------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------

    def get_distribution(self, name: str) -> Distribution:
        """Get the distribution object for a parameter."""
        if name not in self._distributions:
            raise KeyError(f"Unknown parameter '{name}'")
        return self._distributions[name]

    def get_fixed_values(self) -> dict[str, float]:
        """Get a dict of {name: value} for all fixed parameters."""
        return {
            name: float(dist.bounds[0])
            for name, dist in self._distributions.items()
            if dist.is_fixed
        }

    def sample(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        """Draw one sample from all parameter distributions.

        Fixed parameters return their fixed value.
        If stochastic=True, also generates psd_xi ~ N(0, I) of shape (n_grid,).

        Parameters
        ----------
        key : PRNGKey
            Random key.

        Returns
        -------
        dict
            Parameter name → sampled value.
        """
        keys = jax.random.split(key, len(self._distributions) + 1)
        params = {}
        for i, name in enumerate(sorted(self._distributions.keys())):
            params[name] = self._distributions[name].sample(keys[i])

        if self._stochastic:
            params["psd_xi"] = jax.random.normal(keys[-1], shape=(self._n_grid,))

        return params

    def sample_batch(self, key: jax.Array, n: int) -> dict[str, jnp.ndarray]:
        """Draw n samples from all parameter distributions.

        Parameters
        ----------
        key : PRNGKey
            Random key.
        n : int
            Number of samples.

        Returns
        -------
        dict
            Parameter name → array of shape (n,) or (n, n_grid) for psd_xi.
        """
        keys = jax.random.split(key, n)
        # vmap over the sample function
        return jax.vmap(self.sample)(keys)

    def validate(self, params: dict[str, jnp.ndarray]) -> None:
        """Check that parameter values are within bounds.

        Parameters
        ----------
        params : dict
            Parameter name → value.

        Raises
        ------
        ValueError
            If any parameter is out of bounds.
        """
        for name, dist in self._distributions.items():
            if name not in params:
                continue
            val = float(params[name])
            lo, hi = dist.bounds
            if not dist.is_fixed and (val < lo or val > hi):
                raise ValueError(
                    f"Parameter '{name}' = {val} is outside bounds [{lo}, {hi}]"
                )

    def __repr__(self) -> str:
        lines = ["ParamSpec("]
        for name in sorted(self._distributions.keys()):
            dist = self._distributions[name]
            lines.append(f"    {name:20s} = {dist!r},")
        lines.append(f"    {'stochastic':20s} = {self._stochastic},")
        if self._stochastic:
            lines.append(f"    {'n_grid':20s} = {self._n_grid},")
        lines.append(")")
        return "\n".join(lines)
