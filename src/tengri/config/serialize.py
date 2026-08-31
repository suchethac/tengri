# SPDX-License-Identifier: BSD-3-Clause
"""Serialization and deserialization of tengri model configurations.

Provides functions to convert model configurations (nested dicts of groups,
sentinels, and distributions) to/from JSON and YAML format, enabling
version-controlled, diffable model specifications.

The serialization format encodes:
- Sentinels (FREE, FIXED) as strings: "FREE", "FIXED"
- Fixed values as: {"__fixed__": value}  or bare scalars
- Distributions as: {"__prior__": "ClassName", "param1": value, ...}
- Structural strings (type names, law names) pass through unchanged

Round-trip invariant: ``model.config -> to_yaml() -> from_yaml() -> model.config``
should produce an equivalent specification.

Non-portable values (e.g., absolute file paths) are handled by:
- Storing relative paths where available
- Omitting build-time-resolved values from configs

Examples
--------
>>> from tengri import SEDModel, FREE, FIXED, Uniform, Fixed, load_ssp_data
>>> ssp = load_ssp_data("data/ssp.h5")
>>>
>>> # Create a model
>>> model = SEDModel.build(
...     ssp_data=ssp,
...     sfh={"type": "dpl", "all_params": FREE, "beta": Uniform(1, 3)},
...     dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
...     dust_emission={"type": "dale2014", "all_params": FIXED},
...     redshift=Fixed(0.1),
... )
>>>
>>> # Serialize to YAML
>>> config_yaml = model.to_yaml()
>>>
>>> # Deserialize and rebuild
>>> model2 = SEDModel.from_yaml(config_yaml, ssp_data=ssp)
"""

from __future__ import annotations

import difflib
import inspect
import json
import pathlib
from typing import Any, TypeAlias

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from tengri.config.exceptions import ConfigError
from tengri.parameters.priors import Distribution
from tengri.parameters.sentinels import FIXED, FREE, _Sentinel

__all__ = [
    "deserialize_config",
    "dict_to_distribution",
    "distribution_to_dict",
    "serialize_config",
]

#: Type alias for the serialized config format (nested dicts)
ConfigDict: TypeAlias = dict[str, Any]


# ── Serialization (Python → JSON/YAML) ──


def serialize_config(config: ConfigDict) -> ConfigDict:
    """Recursively serialize a model config for JSON/YAML output.

    Converts:
    - Sentinel objects (FREE, FIXED) → strings "FREE", "FIXED"
    - Fixed(value) → {"__fixed__": value} or bare scalar
    - Distribution(...) → {"__prior__": "ClassName", ...}
    - Other dicts/lists → recursed
    - Scalars → unchanged

    Parameters
    ----------
    config : dict
        Nested configuration dict (groups with type, priors, values).

    Returns
    -------
    dict
        Serialized config ready for JSON/YAML encoding.

    Notes
    -----
    This is the inverse of :func:`deserialize_config`. Round-trip is
    invariant: ``deserialize_config(serialize_config(x)) == x``.
    """
    return _serialize_recursive(config, path=[])


def _serialize_recursive(obj: Any, path: list[str]) -> Any:
    """Recursively serialize an object."""
    if isinstance(obj, _Sentinel):
        return obj.name  # "FREE" or "FIXED"
    elif isinstance(obj, Distribution):
        return distribution_to_dict(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_recursive(v, [*path, k]) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_recursive(item, path) for item in obj]
    else:
        # Scalars pass through unchanged
        return obj


def distribution_to_dict(dist: Distribution) -> dict[str, Any]:
    """Convert a Distribution instance to a serializable dict.

    Encodes the distribution class name and its construction parameters.

    Parameters
    ----------
    dist : Distribution
        A prior distribution (Uniform, Gaussian, Fixed, etc.).

    Returns
    -------
    dict
        Serialized distribution with "__prior__" key and param dict.

    Examples
    --------
    >>> from tengri import Uniform, Fixed
    >>> distribution_to_dict(Uniform(0, 1))
    {'__prior__': 'Uniform', 'lo': 0.0, 'hi': 1.0}
    >>> distribution_to_dict(Fixed(0.5))
    {'__fixed__': 0.5}
    """
    if dist.is_fixed:
        # Fixed distributions serialize as a single value or tagged dict
        value = dist.value
        # For simple scalars, bare value is simpler; for clarity, use __fixed__
        return {"__fixed__": value}

    class_name = type(dist).__name__
    params = _extract_distribution_params(dist)
    return {"__prior__": class_name, **params}


def _extract_distribution_params(dist: Distribution) -> dict[str, Any]:
    """Extract the construction parameters of a Distribution.

    Uses introspection on __init__ to recover lo, hi, mu, sigma, etc.
    """
    sig = inspect.signature(dist.__init__)
    result = {}

    for param_name in sig.parameters:
        if param_name in ("self", "description", "units"):
            continue

        # Get the value from the instance
        if hasattr(dist, param_name):
            value = getattr(dist, param_name)
            result[param_name] = value
        elif hasattr(dist, f"_{param_name}"):
            value = getattr(dist, f"_{param_name}")
            result[param_name] = value

    return result


# ── Deserialization (JSON/YAML → Python) ──


def deserialize_config(config: ConfigDict) -> ConfigDict:
    """Recursively deserialize a model config from JSON/YAML input.

    Converts:
    - Strings "FREE", "FIXED" (case-insensitive) → Sentinel objects
    - Dicts with "__fixed__" key → Fixed(value)
    - Dicts with "__prior__" key → Distribution instance
    - Other dicts/lists → recursed
    - Scalars → unchanged

    Parameters
    ----------
    config : dict
        Serialized configuration from JSON/YAML.


    Returns
    -------
    dict
        Deserialized config with Sentinel and Distribution objects restored.

    Raises
    ------
    ConfigError
        If unknown distribution type or malformed prior dict is encountered.

    Notes
    -----
    This is the inverse of :func:`serialize_config`.
    """
    return _deserialize_recursive(config, path=[])


def _deserialize_recursive(obj: Any, path: list[str]) -> Any:
    """Recursively deserialize an object."""
    if isinstance(obj, str):
        # Check for sentinels
        upper = obj.upper()
        if upper == "FREE":
            return FREE
        elif upper == "FIXED":
            return FIXED
        else:
            return obj  # Regular string
    elif isinstance(obj, dict):
        # Check for special dicts (__fixed__, __prior__)
        if "__fixed__" in obj:
            if len(obj) != 1:
                _raise_config_error(
                    path, f"__fixed__ dict must have exactly one key, got {len(obj)}"
                )
            value = obj["__fixed__"]
            # Import here to avoid circular import
            from tengri.parameters.priors import Fixed

            return Fixed(value)
        elif "__prior__" in obj:
            return dict_to_distribution(obj, path=path)
        else:
            # Regular dict, recurse
            return {k: _deserialize_recursive(v, [*path, k]) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_deserialize_recursive(item, path) for item in obj]
    else:
        # Scalars pass through unchanged
        return obj


def dict_to_distribution(d: dict[str, Any], *, path: list[str] | None = None) -> Distribution:
    """Convert a serialized dict to a Distribution instance.

    Parameters
    ----------
    d : dict
        Serialized distribution with "__prior__" key and param dict.
    path : list of str, optional
        Breadcrumb for error messages (path to this dict in the config).

    Returns
    -------
    Distribution
        The reconstructed prior (Uniform, Gaussian, Fixed, etc.).

    Raises
    ------
    ConfigError
        If "__prior__" is missing/unknown or parameters are invalid.

    Examples
    --------
    >>> dict_to_distribution({"__prior__": "Uniform", "lo": 0, "hi": 1})
    Uniform(lo=0, hi=1)
    """
    if path is None:
        path = []

    if "__prior__" not in d:
        _raise_config_error(path, "dict with priors must have '__prior__' key")

    prior_name = d["__prior__"]
    params = {k: v for k, v in d.items() if k != "__prior__"}

    # Import all distribution classes
    from tengri.parameters.priors import (
        Fixed,
        Gaussian,
        Laplace,
        LogNormal,
        LogUniform,
        StudentT,
        Uniform,
    )

    dist_map = {
        "Uniform": Uniform,
        "Gaussian": Gaussian,
        "LogUniform": LogUniform,
        "LogNormal": LogNormal,
        "StudentT": StudentT,
        "Laplace": Laplace,
        "Fixed": Fixed,
    }

    if prior_name not in dist_map:
        # Suggest close matches
        suggestions = difflib.get_close_matches(prior_name, dist_map.keys(), n=3, cutoff=0.6)
        msg = f"unknown prior '{prior_name}'"
        if suggestions:
            msg += f". Did you mean {suggestions[0]!r}?"
        _raise_config_error(path, msg)

    dist_class = dist_map[prior_name]

    # Validate and reconstruct
    try:
        # Check that all provided params are valid for this class
        sig = inspect.signature(dist_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        invalid = set(params.keys()) - valid_params
        if invalid:
            msg = f"prior '{prior_name}' got unexpected parameter(s): {invalid}"
            _raise_config_error(path, msg)

        # Reconstruct
        return dist_class(**params)
    except TypeError as e:
        _raise_config_error(
            path,
            f"{prior_name}: parameter error: {e}",
        )


def _format_path(path: list[str]) -> str:
    """Format a path list as 'group.subkey.param' for error messages."""
    if not path:
        return "<root>"
    return ".".join(path)


def _raise_config_error(path: list[str], message: str) -> None:
    """Raise a ConfigError with path context."""
    full_msg = f"{_format_path(path)}: {message}"
    raise ConfigError(full_msg)


# ── File I/O ──


def load_config_from_file(path: str | pathlib.Path) -> ConfigDict:
    """Load a serialized config from a JSON or YAML file.

    Parameters
    ----------
    path : str or Path
        Path to config file (.json, .yaml, .yml).

    Returns
    -------
    dict
        Deserialized config dict.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ConfigError
        If YAML parsing fails or config is malformed.

    Notes
    -----
    File type is inferred from extension: .json → JSON, .yaml/.yml → YAML.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        content = f.read()

    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in {path}: {e}") from e
    elif suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ConfigError(
                "YAML support requires 'pyyaml' package. Install with: pip install pyyaml"
            )
        try:
            raw = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    else:
        raise ConfigError(f"Unknown file type {suffix}. Expected .json, .yaml, or .yml")

    return deserialize_config(raw)


def save_config_to_file(
    config: ConfigDict, path: str | pathlib.Path, *, format: str = "yaml"
) -> None:
    """Save a serialized config to a JSON or YAML file.

    Parameters
    ----------
    config : dict
        Configuration dict (must be already serialized).
    path : str or Path
        Output file path.
    format : str, optional
        Output format: "json" or "yaml" (default: "yaml").

    Raises
    ------
    ConfigError
        If format is not recognized or YAML is unavailable.

    Notes
    -----
    File extension is inferred from path if format is not specified.
    """
    path = pathlib.Path(path)
    serialized = serialize_config(config)

    if format == "json" or path.suffix.lower() == ".json":
        with open(path, "w") as f:
            json.dump(serialized, f, indent=2)
    elif format in ("yaml", "yml") or path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise ConfigError(
                "YAML support requires 'pyyaml' package. Install with: pip install pyyaml"
            )
        with open(path, "w") as f:
            yaml.dump(serialized, f, default_flow_style=False, sort_keys=False)
    else:
        raise ConfigError(f"Unknown format: {format}")
