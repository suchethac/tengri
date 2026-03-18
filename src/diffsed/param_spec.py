"""Parameter specification for diffsed models.

ParamSpec defines all model parameters: their names, distributions (or fixed
values), and physical bounds. A single ParamSpec is used for both mock
generation (sampling from priors) and inference (defining the prior).

The parameter set is dynamically determined by ``mean_sfh_type``, which
selects SFH model(s) from the registry. Non-SFH parameters (metallicity,
dust, redshift) are always present.

Usage
-----
Default (tsnorm + GP field)::

    spec = ParamSpec(
        sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
        sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
        sfh_tsnorm_width_gyr=Uniform(0.5, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        sfh_field_psd_sigma=Uniform(0.01, 1.0),
        sfh_field_psd_tau_myr=Uniform(10, 500),
        met_logzsol=Gaussian(-0.3, 0.2),
        dust_tau_bc=Uniform(0, 4),
        redshift=0.1,
    )

Parametric only (no GP)::

    spec = ParamSpec(
        mean_sfh_type = "tsnorm",
        sfh_tsnorm_log_peak_sfr = Uniform(-1, 2),
        ...
    )

Legacy DPL (backward compatible)::

    spec = ParamSpec(
        mean_sfh_type = "dpl",
        sfh_dpl_alpha    = Uniform(0.5, 3.0),
        sfh_dpl_beta     = Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr  = Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr = Uniform(-1, 2),
        ...
    )
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
from diffsed.models.sfh.registry import resolve_sfh

# ---------------------------------------------------------------------------
# Non-SFH parameter registry: always present regardless of mean_sfh_type
# ---------------------------------------------------------------------------

_NON_SFH_PARAMS = {
    "met_logzsol": (
        "log10(Z/Zsun)",
        lambda lo, hi: True,
        "",
        Uniform(-2.0, 0.2),
    ),
    "dust_tau_bc": (
        "Birth cloud optical depth",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Uniform(0.0, 4.0),
    ),
    "dust_tau_diff": (
        "Diffuse ISM optical depth",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Uniform(0.0, 3.0),
    ),
    "dust_slope": (
        "Dust power-law index",
        lambda lo, hi: True,
        "",
        Fixed(-0.7),
    ),
    "redshift": (
        "Source redshift",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Fixed(0.1),
    ),
    "noise_frac_cal": (
        "Fractional calibration noise floor (added in quadrature with obs noise)",
        lambda lo, hi: lo >= 0,
        "noise_frac_cal bounds must have lo >= 0",
        Fixed(0.0),
    ),
    "noise_dof": (
        "Student-t degrees of freedom for outlier robustness (0=Gaussian)",
        lambda lo, hi: lo >= 0,
        "noise_dof bounds must have lo >= 0",
        Fixed(0.0),
    ),
}

# ---------------------------------------------------------------------------
# Legacy parameter name aliases (old API → new API)
# ---------------------------------------------------------------------------

_LEGACY_PARAM_ALIASES = {
    "sfh_alpha": "sfh_dpl_alpha",
    "sfh_beta": "sfh_dpl_beta",
    "sfh_tau_peak_gyr": "sfh_dpl_tau_gyr",
    # NOTE: sfh_peak_sfr (linear) has NO alias to sfh_dpl_log_peak_sfr (log10).
    # These have different units. Users must migrate to log10 manually.
    "psd_sigma": "sfh_field_psd_sigma",
    "psd_tau_myr": "sfh_field_psd_tau_myr",
}

# Legacy mean_sfh_type aliases
_LEGACY_SFH_TYPE_ALIASES = {
    "double_powerlaw": "dpl",
}

# Settings keys that are not model parameters
SETTINGS_KEYS = frozenset({"stochastic", "n_grid", "mean_sfh_type"})


# ---------------------------------------------------------------------------
# Build parameter registry dynamically
# ---------------------------------------------------------------------------


def _build_param_registry(mean_sfh_type):
    """Build the parameter registry for a given mean_sfh_type.

    Returns
    -------
    registry : dict
        param_name -> (description, bound_check, bound_error)
    defaults : dict
        param_name -> default Distribution
    """
    _, sfh_params, _, _ = resolve_sfh(mean_sfh_type)

    registry = {}
    defaults = {}

    # SFH params from registry
    for pname, pdef in sfh_params.items():
        registry[pname] = (pdef.description, pdef.bound_check, pdef.bound_error)
        defaults[pname] = pdef.default

    # Non-SFH params (always present)
    for pname, (desc, check, err, default) in _NON_SFH_PARAMS.items():
        registry[pname] = (desc, check, err)
        defaults[pname] = default

    return registry, defaults


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
    - mean_sfh_type (str or list[str]): SFH model(s). Default: ["tsnorm", "field"].
    - stochastic (bool): DEPRECATED. Use mean_sfh_type with/without "field" instead.
    - n_grid (int): GP grid size. Only used if "field" in mean_sfh_type. Default: 256.
    """

    def __init__(self, **kwargs):
        # --- Extract settings ---
        raw_sfh_type = kwargs.pop("mean_sfh_type", None)
        explicit_stochastic = kwargs.pop("stochastic", None)
        n_grid = int(kwargs.pop("n_grid", 256))

        # --- Resolve legacy parameter aliases ---
        resolved_kwargs = {}
        detected_models = set()
        for name, val in kwargs.items():
            new_name = _LEGACY_PARAM_ALIASES.get(name, name)
            resolved_kwargs[new_name] = val
            # Auto-detect model from param name prefixes
            if new_name.startswith("sfh_dpl_"):
                detected_models.add("dpl")
            elif new_name.startswith("sfh_tsnorm_"):
                detected_models.add("tsnorm")
            elif new_name.startswith("sfh_snorm_"):
                detected_models.add("snorm")
            elif new_name.startswith("sfh_norm_"):
                detected_models.add("norm")
            elif new_name.startswith("sfh_lnorm_"):
                detected_models.add("lnorm")
            elif new_name.startswith("sfh_const_"):
                detected_models.add("const")
            elif new_name.startswith("sfh_exp_"):
                detected_models.add("exp")
            elif new_name.startswith("sfh_dexp_"):
                detected_models.add("dexp")
            elif new_name.startswith("sfh_burst_"):
                detected_models.add("burst")
            elif new_name.startswith("sfh_field_"):
                detected_models.add("field")

        # --- Resolve mean_sfh_type ---
        # Auto-detect model from parameter name prefixes if no explicit type given
        if raw_sfh_type is None and detected_models:
            raw_sfh_type = sorted(detected_models)

        mean_sfh_type = self._resolve_sfh_type(raw_sfh_type, explicit_stochastic, detected_models)

        # Normalize to list
        if isinstance(mean_sfh_type, str):
            mean_sfh_type = [mean_sfh_type]

        self._mean_sfh_type = mean_sfh_type
        self._n_grid = n_grid

        # --- Build dynamic parameter registry ---
        self._param_registry, self._defaults = _build_param_registry(mean_sfh_type)
        self._valid_param_names = frozenset(self._param_registry.keys())

        # --- Validate parameter names ---
        # Drop field params if field was removed (e.g., stochastic=False
        # with legacy psd_sigma/psd_tau_myr that are Fixed)
        resolved_kwargs = {
            name: val
            for name, val in resolved_kwargs.items()
            if name in self._valid_param_names or not name.startswith("sfh_field_")
        }
        for name in resolved_kwargs:
            if name not in self._valid_param_names:
                valid_sorted = sorted(self._valid_param_names)
                raise ValueError(
                    f"Unknown parameter '{name}' for mean_sfh_type={mean_sfh_type}. "
                    f"Valid parameters: {valid_sorted}"
                )

        # --- Resolve shorthands and store distributions ---
        self._distributions: dict[str, Distribution] = {}
        for name in sorted(self._valid_param_names):
            if name in resolved_kwargs:
                self._distributions[name] = resolve_shorthand(resolved_kwargs[name])
            else:
                self._distributions[name] = self._defaults[name]

        # --- Validate physical bounds ---
        self._validate_bounds()

    @staticmethod
    def _resolve_sfh_type(raw_sfh_type, explicit_stochastic, detected_models=None):
        """Determine mean_sfh_type from user inputs.

        Priority:
        1. Explicit ``mean_sfh_type`` kwarg (highest)
        2. Auto-detected from parameter name prefixes
        3. ``stochastic`` kwarg (adds/removes "field")
        4. Default: ``["dpl", "field"]``
        """
        if detected_models is None:
            detected_models = set()

        if raw_sfh_type is not None:
            if isinstance(raw_sfh_type, str):
                raw_sfh_type = _LEGACY_SFH_TYPE_ALIASES.get(raw_sfh_type, raw_sfh_type)
                result = [raw_sfh_type]
            else:
                result = [_LEGACY_SFH_TYPE_ALIASES.get(s, s) for s in raw_sfh_type]

            # Honor stochastic kwarg
            if explicit_stochastic is True and "field" not in result:
                result.append("field")
            elif explicit_stochastic is False and "field" in result:
                result = [s for s in result if s != "field"]

            return result

        # No explicit mean_sfh_type and no auto-detected models
        # Use stochastic flag or default
        if explicit_stochastic is True:
            return ["dpl", "field"]
        elif explicit_stochastic is False:
            return ["dpl"]
        else:
            # Default: dpl + field
            return ["dpl", "field"]

    def _validate_bounds(self):
        """Check that distribution bounds respect physical constraints."""
        for name, dist in self._distributions.items():
            if dist.is_fixed:
                lo = hi = dist.bounds[0]
            else:
                lo, hi = dist.bounds

            desc, check_fn, err_msg = self._param_registry[name]
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
        """Whether the model includes a GP field (backward-compat property)."""
        return "field" in self._mean_sfh_type

    @property
    def n_grid(self) -> int:
        """GP grid size (only relevant when stochastic=True)."""
        return self._n_grid

    @property
    def mean_sfh_type(self) -> list[str]:
        """SFH model type(s) as a list of strings."""
        return list(self._mean_sfh_type)

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
        """Number of free parameters (excludes sfh_field_xi)."""
        return len(self.free_params)

    @property
    def valid_param_names(self) -> frozenset:
        """Set of valid parameter names for this model configuration."""
        return self._valid_param_names

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
        If "field" in mean_sfh_type, also generates sfh_field_xi ~ N(0,I).

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

        if self.stochastic:
            params["sfh_field_xi"] = jax.random.normal(keys[-1], shape=(self._n_grid,))

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
            Parameter name → array of shape (n,) or (n, n_grid) for xi.
        """
        keys = jax.random.split(key, n)
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
                raise ValueError(f"Parameter '{name}' = {val} is outside bounds [{lo}, {hi}]")

    def __repr__(self) -> str:
        lines = [f"ParamSpec(mean_sfh_type={self._mean_sfh_type},"]
        for name in sorted(self._distributions.keys()):
            dist = self._distributions[name]
            lines.append(f"    {name:30s} = {dist!r},")
        if self.stochastic:
            lines.append(f"    {'n_grid':30s} = {self._n_grid},")
        lines.append(")")
        return "\n".join(lines)
