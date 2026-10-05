# SPDX-License-Identifier: BSD-3-Clause
"""Single source of truth for the BMA model space: axes, ids, keys, weight sets.

The evidence runner writes one cell per (galaxy, model) under ``<model_key>.json``
and the combiner recovers each cell's weight sets from that same key. Both sides
import this module, so the key format, the grid mappings and the weight-set
membership exist in exactly one place. It is plain Python (no jax, no tengri, no
``configs`` import) so the combiner, the runner's dry-run and the tests can use it
in a process that never loads a JAX backend.

Key formats (every key matches ``^[A-Za-z0-9_.-]+$``):

- grid configuration:  ``config-I`` .. ``config-V``
- X-like model:        ``xlike-cigale_like`` etc.
- factorial model:     ``sfh-<sfh>__ssp-<grid>__att-<att>__ir-<ir>__nebular-<neb>``

A named key carries only the id: the configuration's other fields are display
strings ("Kriek+13, 2-comp") that are not filename safe and live in
``config_metadata``.

Weight sets: ``named_grid`` = I..V; ``named_all`` = I..V + X-like; ``factorial`` =
the 100-model Cartesian product. A model may belong to several sets.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from .config_metadata import CONFIGS, SSP_FOR_CONFIG

FILENAME_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")

# --- factorial axes -------------------------------------------------------

SFH_TYPES: tuple[str, ...] = ("continuity", "dirichlet", "delayed", "dpl", "lnorm")
ATTENUATION_TYPES: tuple[str, ...] = ("calzetti", "smc", "kriek_conroy_2c", "cf00_2c")
DUST_EMISSION = "dl14"
NEBULAR = "cue"

#: Axis order inside a factorial key and inside its component dict.
AXES: tuple[str, ...] = ("sfh", "ssp", "attenuation", "dust_emission", "nebular")

#: Axes derived from the ``ssp`` label (isochrone set and spectral library).
DERIVED_AXES: tuple[str, ...] = ("isochrone", "spectral_library")

# Key prefix per axis, in AXES order.
_AXIS_PREFIX = {
    "sfh": "sfh",
    "ssp": "ssp",
    "attenuation": "att",
    "dust_emission": "ir",
    "nebular": "nebular",
}
_PREFIX_AXIS = {prefix: axis for axis, prefix in _AXIS_PREFIX.items()}
_KEY_SEP = "__"


class SspEntry(NamedTuple):
    """One value of the ``ssp`` axis.

    ``grid`` is the registered SSP grid name (the label used in keys), ``config``
    the grid configuration that uses it, and ``isochrone`` / ``spectral_library``
    its derived-axis values.
    """

    grid: str
    config: str
    isochrone: str
    spectral_library: str


#: Grid configurations on the factorial axes. VI is deliberately excluded.
GRID_IDS: tuple[str, ...] = ("I", "II", "III", "IV", "V")

#: Derived-axis values (isochrone set, spectral library) per registered grid name.
_SSP_DERIVED = {
    "fsps_mist_c3k_a_chabrier": ("mist", "c3k"),
    "fsps_prsc_c3k_a_chabrier": ("prsc", "c3k"),
    "fsps_mist_miles_chabrier": ("mist", "miles"),
    "fsps_prsc_miles_chabrier": ("prsc", "miles"),
    "bpss_stars_c3k_a_chabrier": ("bpass", "c3k"),
}

SSP_AXIS: tuple[SspEntry, ...] = tuple(
    SspEntry(SSP_FOR_CONFIG[c], c, *_SSP_DERIVED[SSP_FOR_CONFIG[c]]) for c in GRID_IDS
)
SSP_LABELS: tuple[str, ...] = tuple(e.grid for e in SSP_AXIS)

# --- named ids ------------------------------------------------------------

XLIKE_IDS: tuple[str, ...] = (
    "cigale_like",
    "prospector_like",
    "bagpipes_like",
    "beagle_like",
    "dense_basis_like",
)

#: Structural fields of each grid configuration on the factorial axes, read from
#: the configuration table (``config_metadata.CONFIGS``), never restated.
_GRID_SFH = {c: CONFIGS[c]["sfh_type"] for c in GRID_IDS}
_GRID_ATTENUATION = {c: CONFIGS[c]["attenuation_axis"] for c in GRID_IDS}

_CONFIG_PREFIX = "config-"
_XLIKE_PREFIX = "xlike-"

WEIGHT_SETS: tuple[str, ...] = ("named_grid", "named_all", "factorial")


def ssp_entry(label: str) -> SspEntry:
    """Return the ``ssp`` axis entry for a registered grid name.

    Raises
    ------
    ValueError
        If ``label`` is not a value of the ssp axis.
    """
    for entry in SSP_AXIS:
        if entry.grid == label:
            return entry
    raise ValueError(f"Unknown ssp label {label!r}; axis values are {list(SSP_LABELS)}")


def ssp_grid_for_config(config: str) -> str:
    """Registered SSP grid name of grid configuration ``config`` (I..V)."""
    for entry in SSP_AXIS:
        if entry.config == config:
            return entry.grid
    raise ValueError(f"Unknown grid configuration {config!r}; expected one of {list(GRID_IDS)}")


def derived_axis_value(axis: str, ssp_label: str) -> str:
    """Isochrone or spectral-library value implied by an ssp label."""
    if axis not in DERIVED_AXES:
        raise ValueError(f"{axis!r} is not a derived axis {list(DERIVED_AXES)}")
    return getattr(ssp_entry(ssp_label), axis)


# --- keys -----------------------------------------------------------------


def model_key(components: dict[str, str]) -> str:
    """Filename-safe model key for a component dict.

    A dict with a ``"config"`` entry is a named model: ``config-<id>`` for a grid
    configuration, ``xlike-<id>`` for an X-like model. Otherwise every field of
    ``AXES`` is required and the factorial key is built in axis order.

    Raises
    ------
    ValueError
        For an unknown config id, a missing axis, or a value that would not be
        filename safe.
    """
    if "config" in components:
        cfg = components["config"]
        if cfg in GRID_IDS:
            key = f"{_CONFIG_PREFIX}{cfg}"
        elif cfg in XLIKE_IDS:
            key = f"{_XLIKE_PREFIX}{cfg}"
        else:
            raise ValueError(
                f"Unknown config id {cfg!r}: not a grid configuration {list(GRID_IDS)} "
                f"or an X-like id {list(XLIKE_IDS)}"
            )
        return key
    missing = [axis for axis in AXES if axis not in components]
    if missing:
        raise ValueError(f"Factorial components missing axes {missing}")
    key = _KEY_SEP.join(f"{_AXIS_PREFIX[axis]}-{components[axis]}" for axis in AXES)
    if not FILENAME_SAFE.match(key):
        raise ValueError(f"Model key {key!r} is not filename safe")
    return key


def parse_model_key(key: str) -> dict[str, str]:
    """Inverse of ``model_key``.

    Returns ``{"config": id}`` for a named key, or the five factorial axis values.

    Raises
    ------
    ValueError
        If ``key`` is not in either format.
    """
    if key.startswith(_CONFIG_PREFIX):
        cfg = key[len(_CONFIG_PREFIX) :]
        if cfg in GRID_IDS:
            return {"config": cfg}
    elif key.startswith(_XLIKE_PREFIX):
        cfg = key[len(_XLIKE_PREFIX) :]
        if cfg in XLIKE_IDS:
            return {"config": cfg}
    else:
        parts = key.split(_KEY_SEP)
        parsed: dict[str, str] = {}
        for part in parts:
            prefix, sep, value = part.partition("-")
            if not sep or prefix not in _PREFIX_AXIS or not value:
                break
            parsed[_PREFIX_AXIS[prefix]] = value
        else:
            if tuple(parsed) == AXES and len(parts) == len(AXES):
                return parsed
    raise ValueError(f"Not a BMA model key: {key!r}")


# --- enumeration ----------------------------------------------------------


def enumerate_factorial() -> list[dict[str, str]]:
    """The 100 factorial component dicts (sfh outer, then ssp, then attenuation)."""
    return [
        {
            "sfh": sfh,
            "ssp": ssp,
            "attenuation": att,
            "dust_emission": DUST_EMISSION,
            "nebular": NEBULAR,
        }
        for sfh in SFH_TYPES
        for ssp in SSP_LABELS
        for att in ATTENUATION_TYPES
    ]


def enumerate_named_grid() -> list[dict[str, str]]:
    """Grid configurations I..V as component dicts carrying a ``config`` id.

    Fields are the structural axes (sfh, ssp, attenuation); the configuration's
    dust-emission and nebular fields live in ``config_metadata``.
    """
    return [
        {
            "config": cfg,
            "sfh": _GRID_SFH[cfg],
            "ssp": ssp_grid_for_config(cfg),
            "attenuation": _GRID_ATTENUATION[cfg],
        }
        for cfg in GRID_IDS
    ]


def enumerate_named_all() -> list[dict[str, str]]:
    """Grid configurations I..V followed by the X-like models."""
    return enumerate_named_grid() + [{"config": key} for key in XLIKE_IDS]


_ENUMERATORS = {
    "named_grid": enumerate_named_grid,
    "named_all": enumerate_named_all,
    "factorial": enumerate_factorial,
}


def enumerate_set(set_name: str) -> list[dict[str, str]]:
    """Component dicts of one weight set."""
    if set_name not in _ENUMERATORS:
        raise ValueError(f"Unknown weight set {set_name!r}; expected one of {list(WEIGHT_SETS)}")
    return _ENUMERATORS[set_name]()


def expected_keys(set_name: str) -> list[str]:
    """Model keys a weight set is expected to hold, in enumeration order."""
    return [model_key(c) for c in enumerate_set(set_name)]


def set_membership(key: str) -> set[str]:
    """Names of the weight sets that hold ``key`` (empty for an unknown key)."""
    return {name for name in WEIGHT_SETS if key in _keys_of(name)}


def model_set_kind(key: str) -> str | None:
    """``"named"`` or ``"factorial"`` for a recognized key, else ``None``."""
    membership = set_membership(key)
    if "factorial" in membership:
        return "factorial"
    if membership:
        return "named"
    return None


_KEY_CACHE: dict[str, frozenset[str]] = {}


def _keys_of(set_name: str) -> frozenset[str]:
    if set_name not in _KEY_CACHE:
        _KEY_CACHE[set_name] = frozenset(expected_keys(set_name))
    return _KEY_CACHE[set_name]


# --- axes, marginals and prior mass ---------------------------------------


def axis_values(axis: str) -> tuple[str, ...]:
    """Ordered values of a factorial (or derived) axis."""
    if axis == "sfh":
        return SFH_TYPES
    if axis == "ssp":
        return SSP_LABELS
    if axis == "attenuation":
        return ATTENUATION_TYPES
    if axis == "dust_emission":
        return (DUST_EMISSION,)
    if axis == "nebular":
        return (NEBULAR,)
    if axis in DERIVED_AXES:
        return tuple(dict.fromkeys(getattr(e, axis) for e in SSP_AXIS))
    raise ValueError(f"Unknown axis {axis!r}")


def factorial_prior_mass() -> dict[str, dict[str, float]]:
    """Prior mass per axis value under a flat prior over the factorial models.

    Counts each axis value across ``enumerate_factorial()``, so an unbalanced
    axis (isochrone: mist 0.4, prsc 0.4, bpass 0.2) reads off the enumeration.
    """
    models = enumerate_factorial()
    n = len(models)
    out: dict[str, dict[str, float]] = {}
    for axis in AXES:
        out[axis] = {v: 0.0 for v in axis_values(axis)}
        for m in models:
            out[axis][m[axis]] += 1.0 / n
    for axis in DERIVED_AXES:
        out[axis] = {v: 0.0 for v in axis_values(axis)}
        for m in models:
            out[axis][derived_axis_value(axis, m["ssp"])] += 1.0 / n
    return out


# --- display labels -------------------------------------------------------

#: Short reader-facing label per axis value (matplotlib text, so mathtext is fine).
#: ``named`` holds the grid configuration ids and the X-like ids. A figure labels
#: every axis value and named model through ``display_label``, so a value that is
#: not listed here is an error rather than a raw registry name on a page.
DISPLAY_LABELS: dict[str, dict[str, str]] = {
    "sfh": {
        "continuity": "Continuity",
        "dirichlet": "Dirichlet",
        "delayed": r"Delayed-$\tau$",
        "dpl": "Double power law",
        "lnorm": "Log-normal",
    },
    "ssp": {
        "fsps_mist_c3k_a_chabrier": "MIST + C3K",
        "fsps_prsc_c3k_a_chabrier": "PARSEC + C3K",
        "fsps_mist_miles_chabrier": "MIST + MILES",
        "fsps_prsc_miles_chabrier": "PARSEC + MILES",
        "bpss_stars_c3k_a_chabrier": "BPASS + C3K",
    },
    "attenuation": {
        "calzetti": "Calzetti",
        "smc": "SMC",
        "kriek_conroy_2c": "Kriek & Conroy, 2-comp.",
        "cf00_2c": "Charlot & Fall, 2-comp.",
    },
    "dust_emission": {"dl14": "DL14"},
    "nebular": {"cue": "Cue"},
    "isochrone": {"mist": "MIST", "prsc": "PARSEC", "bpass": "BPASS"},
    "spectral_library": {"c3k": "C3K", "miles": "MILES"},
    "named": {
        "I": "I",
        "II": "II",
        "III": "III",
        "IV": "IV",
        "V": "V",
        "cigale_like": "CIGALE-like",
        "prospector_like": "Prospector-like",
        "bagpipes_like": "BAGPIPES-like",
        "beagle_like": "BEAGLE-like",
        "dense_basis_like": "Dense Basis-like",
    },
}


def display_label(axis: str, value: str) -> str:
    """Reader label of an axis value or, for ``axis="named"``, of a named model id.

    Raises
    ------
    ValueError
        If the axis or the value has no label.
    """
    if axis not in DISPLAY_LABELS:
        raise ValueError(f"No display labels for axis {axis!r}; axes are {list(DISPLAY_LABELS)}")
    try:
        return DISPLAY_LABELS[axis][value]
    except KeyError:
        raise ValueError(
            f"No display label for {axis} value {value!r}; labeled values are "
            f"{list(DISPLAY_LABELS[axis])}"
        ) from None
