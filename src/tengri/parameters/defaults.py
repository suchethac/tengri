# SPDX-License-Identifier: BSD-3-Clause
"""Load default model configuration from defaults.toml.

Search order (first found wins):

1. ``~/.tengri/defaults.toml``: user-level override
2. ``<package root>/defaults.toml``: package defaults shipped with tengri

Copy the package defaults file to ``~/.tengri/defaults.toml`` and edit it
there to change defaults without touching the package itself.

Usage
-----
::

    from tengri.parameters.defaults import get_from_config_defaults, load_defaults

    # High-level API defaults (the [from_config] section)
    fc = get_from_config_defaults()
    sfh_type = fc["sfh"]  # "dense_basis" unless overridden

    # Full config tree
    raw = load_defaults()
    n_grid = raw["sfh"]["n_grid"]  # 64 unless overridden
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

# ── Sentinel ─────────────────────────────────────────────────────────────────


class _UnsetType:
    """Singleton sentinel distinguishing 'not provided' from any real value."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET = _UnsetType()
"""Sentinel for 'caller did not supply this argument'.

Use as a default in function signatures where ``None`` is a valid user value
and you still need to detect "not provided"::

    def foo(x=UNSET):
        if x is UNSET:
            x = get_from_config_defaults()["x"]
"""

# ── Paths ─────────────────────────────────────────────────────────────────────

_PACKAGE_DEFAULTS: Path = Path(__file__).parent.parent / "defaults.toml"
_USER_DEFAULTS: Path = Path.home() / ".tengri" / "defaults.toml"

# Fields whose empty-string TOML value should become Python None
_NULLABLE: dict[str, tuple[str, ...]] = {
    "from_config": ("nebular", "agn"),
    "dust": ("law_diff", "emission"),
}

# Hardcoded fallback in case the TOML file is missing or unreadable
_FALLBACK_FROM_CONFIG: dict[str, Any] = {
    "sfh": "tsnorm",
    "dust": "charlot_fall",
    "nebular": None,
    "agn": None,
    "redshift": 0.1,
}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _active_path() -> Path:
    """Return the defaults.toml path to use, preferring the user override."""
    return _USER_DEFAULTS if _USER_DEFAULTS.exists() else _PACKAGE_DEFAULTS


def _resolve_nulls(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert "" sentinel values to None in known optional string fields."""
    for section, keys in _NULLABLE.items():
        if section in raw:
            for key in keys:
                if key in raw[section] and raw[section][key] == "":
                    raw[section][key] = None
    return raw


# ── Public API ────────────────────────────────────────────────────────────────


def load_defaults() -> dict[str, Any]:
    """Load the active defaults.toml as a nested dict.

    Resolves ``""`` sentinel values to ``None`` for optional string fields
    (e.g. ``dust.emission``, ``from_config.nebular``).

    Returns
    -------
    dict
        Nested dict matching the TOML section structure.

    Raises
    ------
    ImportError
        If neither ``tomllib`` (stdlib, Python ≥ 3.11) nor ``tomli`` is
        available.
    FileNotFoundError
        If the package defaults file is missing (should never happen in a
        correctly installed package).
    """
    if tomllib is None:
        raise ImportError(
            "TOML parsing is unavailable.  Install tomli for Python < 3.11:\n    pip install tomli"
        )

    path = _active_path()
    with open(path, "rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)

    return _resolve_nulls(raw)


def get_from_config_defaults() -> dict[str, Any]:
    """Return the ``[from_config]`` section with safe hardcoded fallbacks.

    Never raises; falls back silently if the TOML file is unreadable.

    Returns
    -------
    dict with keys: sfh, dust, nebular, agn, redshift
    """
    try:
        data = load_defaults()
    except (ImportError, FileNotFoundError, OSError, tomllib.TOMLDecodeError):  # type: ignore[attr-defined]
        # ImportError: tomllib unavailable
        # FileNotFoundError: defaults.toml missing
        # OSError: permission denied or other I/O failure
        # TOMLDecodeError: malformed TOML syntax
        return dict(_FALLBACK_FROM_CONFIG)
    return {**_FALLBACK_FROM_CONFIG, **data.get("from_config", {})}


def get_inference_defaults(method: str | None = None) -> dict[str, Any]:
    """Return inference defaults from the ``[inference]`` section.

    Parameters
    ----------
    method : str or None
        If given, return only the sub-section for that method (e.g. ``"vi"``,
        ``"mcmc_raytrace"``).  If None, return the full ``[inference]`` dict.

    Returns
    -------
    dict
        Empty dict if the section or method is absent.
    """
    try:
        data = load_defaults()
    except (ImportError, FileNotFoundError, OSError, tomllib.TOMLDecodeError):  # type: ignore[attr-defined]
        # ImportError: tomllib unavailable
        # FileNotFoundError: defaults.toml missing
        # OSError: permission denied or other I/O failure
        # TOMLDecodeError: malformed TOML syntax
        return {}
    section = data.get("inference", {})
    if method is None:
        return section
    return section.get(method, {})
