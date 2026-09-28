"""Model space for Bayesian model averaging over SED configurations.

The factorial model space is a Cartesian product of component axes:
- sfh: continuity, dirichlet, delayed, dpl, lnorm (5 options)
- ssp: 5 stellar population grids from configurations I-V
- attenuation: calzetti, smc, kriek_conroy_2c, cf00_2c (4 options)
- dust_emission: dl14 (fixed)
- nebular: cue (fixed)

Total: 5 * 5 * 4 * 1 * 1 = 100 factorial models.

Named sets:
- named_grid: configurations I-V (6th configuration excluded from BMA)
- named_all: configurations I-V + X-like models (cigale_like, prospector_like, etc.)

The model_key function generates deterministic strings like:
  "sfh-continuity__ssp-mist_c3k__att-kriek_conroy_2c__ir-dl14__neb-cue"

Component dicts are extracted from configs.py to guarantee exact parity;
no copy-paste of priors or parameters.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import tengri
from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.cosmology import age_at_z

from config_metadata import CONFIGS, SSP_FOR_CONFIG
from configs import (
    _continuity_sfh,
    _kriek_conroy_two_component,
    met_prior_for,
)

LOG10_ZSUN = -1.848
MET_EDGE_INSET_DEX = 0.02


def _calzetti_single_component() -> dict:
    """Calzetti single-screen attenuation (from config_II)."""
    return {
        "type": "single_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_v": Uniform(0.0, 3.0),
    }


def _smc_single_component() -> dict:
    """SMC single-screen attenuation (from config_V)."""
    return {
        "type": "single_component",
        "law": "smc",
        "all_params": Fixed(DEFAULT),
        "tau_v": Uniform(0.0, 3.0),
    }


def _charlot_fall_two_component() -> dict:
    """Charlot & Fall (2000) two-component attenuation (from config_III)."""
    return {
        "type": "two_component",
        "law_bc": "power_law",
        "slope_bc": -1.3,
        "law_diff": "power_law",
        "slope_diff": -0.7,
        "all_params": Fixed(DEFAULT),
        "tau_bc": Uniform(0.0, 3.0),
        "tau_diff": Uniform(0.5, 3.0),
    }


def _dirichlet_sfh(ssp_data: tengri.SSPData, z: float) -> dict:
    """Dirichlet SFH (from config_IV)."""
    N_SFH_BINS = 7
    sfh = {
        "type": "dirichlet",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
        "bin_edges_gyr": tengri.make_agebins_from_zred(z, n_bins=N_SFH_BINS),
    }
    for i in range(N_SFH_BINS - 1):
        sfh[f"z_{i}"] = FREE
    return sfh


def _delayed_sfh(ssp_data: tengri.SSPData, z: float) -> dict:
    """Delayed-tau SFH (from config_III)."""
    return {
        "type": "delayed",
        "all_params": Fixed(DEFAULT),
        "tau_gyr": Uniform(0.1, 20.0),
        "age_gyr": Uniform(1.0, age_at_z(z)),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
    }


def _dpl_sfh(ssp_data: tengri.SSPData, z: float) -> dict:
    """Double power-law SFH (from config_II)."""
    tau_upper = age_at_z(z)
    return {
        "type": "dpl",
        "all_params": Fixed(DEFAULT),
        "alpha": Uniform(0.5, 5.0),
        "beta": Uniform(0.3, 3.0),
        "tau_gyr": Uniform(0.5, tau_upper),
        "age_gyr": Uniform(1.0, age_at_z(z)),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
    }


def _lnorm_sfh(ssp_data: tengri.SSPData, z: float) -> dict:
    """Log-normal SFH (from config_V)."""
    return {
        "type": "lnorm",
        "all_params": Fixed(DEFAULT),
        "peak_gyr": Uniform(0.1, age_at_z(z)),
        "width_gyr": Uniform(0.1, 5.0),
        "age_gyr": Uniform(1.0, age_at_z(z)),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": met_prior_for(ssp_data),
    }


# Axes for the factorial model space
FACTORIAL_SFH_TYPES = ["continuity", "dirichlet", "delayed", "dpl", "lnorm"]
FACTORIAL_SSP_KEYS = ["I", "II", "III", "IV", "V"]
FACTORIAL_ATTENUATION_TYPES = ["calzetti", "smc", "kriek_conroy_2c", "cf00_2c"]
FIXED_DUST_EMISSION = "dl14"
FIXED_NEBULAR = "cue"

# Mapping from axis names to dict builders
_SFH_BUILDERS = {
    "continuity": _continuity_sfh,
    "dirichlet": _dirichlet_sfh,
    "delayed": _delayed_sfh,
    "dpl": _dpl_sfh,
    "lnorm": _lnorm_sfh,
}

_ATTENUATION_BUILDERS = {
    "calzetti": _calzetti_single_component,
    "smc": _smc_single_component,
    "kriek_conroy_2c": _kriek_conroy_two_component,
    "cf00_2c": _charlot_fall_two_component,
}


def model_key(components: dict[str, str]) -> str:
    """Generate a deterministic model key from component names.

    Args:
        components: dict with keys "sfh", "ssp", "attenuation", "dust_emission",
            "nebular", optionally "config" (for named sets).

    Returns:
        String like "sfh-continuity__ssp-mist_c3k__att-kriek_conroy_2c__ir-dl14__neb-cue"
    """
    parts = []
    for key in ["sfh", "ssp", "attenuation", "dust_emission", "nebular"]:
        if key == "dust_emission":
            parts.append(f"ir-{components[key]}")
        elif key == "attenuation":
            parts.append(f"att-{components[key]}")
        else:
            parts.append(f"{key}-{components[key]}")
    return "__".join(parts)


def parse_model_key(key: str) -> dict[str, str]:
    """Inverse of model_key: parse a model key into components.

    Args:
        key: String like "sfh-continuity__ssp-mist_c3k__att-kriek_conroy_2c__ir-dl14__neb-cue"

    Returns:
        dict with keys "sfh", "ssp", "attenuation", "dust_emission", "nebular"
    """
    components = {}
    for part in key.split("__"):
        axis, value = part.split("-", 1)
        if axis == "ir":
            components["dust_emission"] = value
        elif axis == "att":
            components["attenuation"] = value
        else:
            components[axis] = value
    return components


def enumerate_factorial() -> list[dict[str, str]]:
    """Enumerate all 100 factorial models in deterministic order.

    Returns:
        List of component dicts, one per model. Order is:
        sfh (outer loop) -> ssp -> attenuation -> dust_emission, nebular (fixed).
    """
    models = []
    for sfh_type in FACTORIAL_SFH_TYPES:
        for ssp_key in FACTORIAL_SSP_KEYS:
            ssp_name = SSP_FOR_CONFIG[ssp_key]
            for att_type in FACTORIAL_ATTENUATION_TYPES:
                components = {
                    "sfh": sfh_type,
                    "ssp": ssp_name,
                    "attenuation": att_type,
                    "dust_emission": FIXED_DUST_EMISSION,
                    "nebular": FIXED_NEBULAR,
                }
                models.append(components)
    return models


def enumerate_named_grid() -> list[dict[str, str]]:
    """Enumerate the five grid configurations I-V as named models (excluding VI).

    Returns:
        List of 5 component dicts. Each includes a "config" key.
    """
    models = []
    for cfg_key in ["I", "II", "III", "IV", "V"]:
        cfg = CONFIGS[cfg_key]
        components = {
            "sfh": cfg["sfh_type"],
            "ssp": SSP_FOR_CONFIG[cfg_key],
            "attenuation": cfg["attenuation"],  # plain string like "Kriek+13, 2-comp"
            "dust_emission": cfg["dust_ir"],  # plain string
            "nebular": cfg["nebular"],  # plain string
            "config": cfg_key,
        }
        models.append(components)
    return models


def _load_xlike_builders() -> dict[str, callable]:
    """Load xlike config builders if the module exists.

    Raises:
        ImportError: If xlike_configs.py exists but fails to import (don't swallow).

    Returns:
        Dict mapping xlike keys to builder functions, or empty dict if absent.
    """
    here = Path(__file__).resolve().parent
    xlike_module_path = here / "xlike_configs.py"

    if not xlike_module_path.is_file():
        return {}

    # If file exists, import must succeed (don't swallow errors)
    try:
        import sys
        sys.path.insert(0, str(here))
        try:
            import xlike_configs as module
        finally:
            if str(here) in sys.path:
                sys.path.remove(str(here))
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        # If the file exists but import fails, raise (don't silently drop X-like)
        raise ImportError(f"Failed to import xlike_configs from {xlike_module_path}: {e}") from e

    builders = {}
    if hasattr(module, "XLIKE_BUILDERS"):
        builders.update(module.XLIKE_BUILDERS)
    return builders


def enumerate_named_all() -> list[dict[str, str]]:
    """Enumerate grid I-V plus X-like models if available.

    Returns:
        List of 5+ component dicts. Each includes a "config" key naming the
        configuration or xlike key. Returns just the grid if xlike module absent.
    """
    models = enumerate_named_grid()
    xlike_builders = _load_xlike_builders()
    for key in sorted(xlike_builders.keys()):
        components = {
            "config": key,
            # Other fields are not populated here; they depend on per-galaxy SSP
        }
        models.append(components)
    return models


def build_model(
    components: dict[str, str],
    ssp_data: tengri.SSPData,
    observation: tengri.Observation,
    z: float,
) -> SEDModel:
    """Build a SEDModel from component names.

    Args:
        components: dict with "sfh", "ssp", "attenuation", "dust_emission",
            "nebular" keys (all required for factorial models).
        ssp_data: Preloaded SSP data.
        observation: Observation with photometry.
        z: Redshift.

    Returns:
        Compiled SEDModel ready for inference.

    Raises:
        ValueError: if a component name is unrecognized.
    """
    sfh_type = components["sfh"]
    att_type = components["attenuation"]
    dust_em_type = components["dust_emission"]
    neb_type = components["nebular"]

    if sfh_type not in _SFH_BUILDERS:
        raise ValueError(f"Unknown sfh type: {sfh_type}")
    if att_type not in _ATTENUATION_BUILDERS:
        raise ValueError(f"Unknown attenuation type: {att_type}")
    if dust_em_type != FIXED_DUST_EMISSION:
        raise ValueError(
            f"Only {FIXED_DUST_EMISSION!r} dust emission is supported; got {dust_em_type!r}"
        )
    if neb_type != FIXED_NEBULAR:
        raise ValueError(f"Only {FIXED_NEBULAR!r} nebular is supported; got {neb_type!r}")

    sfh_builder = _SFH_BUILDERS[sfh_type]
    sfh = sfh_builder(ssp_data, z)

    att_builder = _ATTENUATION_BUILDERS[att_type]
    dust_attenuation = att_builder()

    dust_emission = {"type": FIXED_DUST_EMISSION, "all_params": Fixed(DEFAULT)}
    neb = {"type": FIXED_NEBULAR, "all_params": Fixed(DEFAULT), "neb_logU": Fixed(-2.5)}

    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh=sfh,
        dust_attenuation=dust_attenuation,
        dust_emission=dust_emission,
        neb=neb,
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )
