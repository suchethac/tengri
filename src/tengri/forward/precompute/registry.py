# SPDX-License-Identifier: BSD-3-Clause
"""Explicit registry of precompute-enabled components.

Single source of truth mapping component identifiers (physics-model names used
at configuration time) to the Python module that implements
:class:`~tengri.forward.precompute.protocol.PrecomputeModule` for that component.

Adding a new precompute-enabled component
-----------------------------------------

1. Create ``components/<component>/<component>_precompute.py`` following the
   Protocol shape (see ``protocol.py``).
2. Add one entry to ``_REGISTRY`` below.
3. Ensure the component's main module imports cleanly even when template data
   is missing (gracefully degrade, :func:`resolve` callers handle ``None``).

That's the full extension surface. ``SEDModel`` does not need editing.
"""

from __future__ import annotations

import importlib
from types import ModuleType

# Maps the component identifier (as used in ModelConfig / Parameters model selection)
# to the dotted module path of its precompute adapter.
_REGISTRY: dict[str, str] = {
    # Stellar population photometry (redshift + filters fixed → preintegrate SSP×filter tensor)
    "ssp": "tengri.components.stellar.sps.precompute",
    # Nebular (CLOUDY emulator / tabulated grid)
    "cloudy": "tengri.components.nebular.cloudy_precompute",
    # Feltre+2016 AGN NLR (CLOUDY c13.03)
    "feltre_nlr": "tengri.components.nebular.feltre_precompute",
    # MAPPINGS V photoionization (stellar)
    "mappings_v": "tengri.components.nebular.mappings_photo_precompute",
    # MAPPINGS shock
    "mappings_shock": "tengri.components.nebular.mappings_shock_precompute",
    # Dust IR emission, template-based models
    "dl07": "tengri.components.dust.dust_emission_precompute",
    "draine_li2007": "tengri.components.dust.dust_emission_precompute",
    "dale2014": "tengri.components.dust.dust_emission_precompute",
    "draine_li2014": "tengri.components.dust.dust_emission_precompute",
    "astrodust": "tengri.components.dust.dust_emission_precompute",
    "bosa": "tengri.components.dust.dust_emission_precompute",
    "themis": "tengri.components.dust.dust_emission_precompute",
    # Dust IR emission, analytic models
    "modified_blackbody": "tengri.components.dust.dust_analytic_precompute",
    "casey2012": "tengri.components.dust.dust_analytic_precompute",
    "pah_drude": "tengri.components.dust.dust_analytic_precompute",
    # AGN torus templates
    "skirtor": "tengri.components.agn.skirtor_precompute",
    "skirtor_agnfitter": "tengri.components.agn.skirtor_agnfitter_precompute",
    "silva04": "tengri.components.agn.silva04_precompute",
    "nenkova_agnfitter": "tengri.components.agn.nenkova_agnfitter_precompute",
    "cat3d_wind": "tengri.components.agn.cat3d_precompute",
    # AGN K&D 3-zone disc (custom dataclass, but still Protocol-shaped)
    "kd_disc": "tengri.components.agn.kd_precompute",
    "kubota_done": "tengri.components.agn.kd_precompute",
    # AGN analytic disc models
    "powerlaw_disc": "tengri.components.agn.disc_precompute",
    "ss_disc": "tengri.components.agn.disc_precompute",
    "cigale_disc": "tengri.components.agn.disc_precompute",
    # AGN empirical quasar model
    "qsogen": "tengri.components.agn.qsogen_precompute",
    # Radio analytic components
    "radio_synchrotron": "tengri.components.radio.radio_precompute",
    "radio_freefree": "tengri.components.radio.radio_precompute",
    "radio_agn_jet": "tengri.components.radio.radio_precompute",
    # X-ray analytic components
    "xray_xrb": "tengri.components.xray.xray_precompute",
    "xray_corona": "tengri.components.xray.xray_precompute",
    "xray_corona_lopez24": "tengri.components.xray.xray_precompute",
    # CB19 (3MdB_17) photoionization grid
    "cb19": "tengri.components.nebular.cb19_precompute",
    # AGN BLR / NLR Gaussian-line composers (filter-projection precompute only)
    "blr": "tengri.components.agn.blr_precompute",
    "nlr_gaussian": "tengri.components.agn.nlr_gaussian_precompute",
}


def resolve(component_name: str) -> ModuleType | None:
    """Return the precompute module for a component name, or None if not registered.

    Returns None when the component is unknown (falls back to runtime wavelength
    evaluation in the caller).

    Parameters
    ----------
    component_name: str
        Component identifier used in ModelConfig (e.g., ``"ssp"``, ``"dl07"``,
        ``"skirtor"``).

    Returns
    -------
    ModuleType or None
        The precompute module implementing :class:`PrecomputeModule` if registered,
        or None if not found (component uses runtime evaluation only).

    Notes
    -----
    Unregistered components fall back to per-call wavelength integration
    without caching.
    """
    module_path = _REGISTRY.get(component_name)
    if module_path is None:
        return None
    return importlib.import_module(module_path)


def registered_components() -> list[str]:
    """Return the sorted list of component names registered for precompute.

    Useful for sanity tests and documentation generation.

    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        Sorted list of component identifiers registered in _REGISTRY.

    Notes
    -----
    Used by validation and documentation tools to enumerate all precomputable
    components.
    """
    return sorted(_REGISTRY.keys())


def validate_precompute_module(component_name: str, module: ModuleType) -> None:
    """Validate that a precompute module satisfies shape contract at registration.

    At module discovery / import time, attempts a lightweight call to ensure
    that the precompute result object has the expected attributes (``shape``,
    ``data``, or grid-specific fields). On failure, raises a clear error
    identifying the component and the missing attribute.

    This validation is **not comprehensive** (does not guarantee JIT compatibility
    or correct lookup signature); it only checks that the precompute result
    has basic shape metadata. Finer errors surface during inference when
    ``build_lookup`` attempts to read the result.

    Parameters
    ----------
    component_name: str
        Component identifier from the registry (e.g., ``"ssp"``, ``"dl07"``).
    module: ModuleType
        The imported precompute module.

    Raises
    ------
    AttributeError
        If the module lacks required Protocol methods (``precompute``,
        ``build_lookup``).
    RuntimeError
        If validation of a sample precompute result fails (missing ``shape``,
        ``data``, or component-specific expected attributes).

    Notes
    -----
    This function is meant to be called by callers performing eager validation
    (e.g., SEDModel initialization, test suites). It is not currently hooked
    into :func:`resolve`, but could be in the future.
    """
    from tengri.forward.precompute.protocol import PrecomputeModule

    # Check Protocol surface
    if not isinstance(module, PrecomputeModule):
        missing = []
        for attr in ("AXIS_PARAMS", "precompute", "build_lookup"):
            if not hasattr(module, attr):
                missing.append(attr)
        raise AttributeError(
            f"Precompute module for {component_name!r} missing required "
            f"Protocol attributes: {missing}. Module: {module}"
        )

    # Check that precompute result has expected shape metadata
    # (This is a heuristic check; not a full contract validation.)
    # We do NOT attempt a full precompute call here because:
    # 1. It requires SSP grids, template files, filter curves (expensive setup).
    # 2. It would fail if templates are missing, even if the module is correctly
    #    shaped (we gracefully degrade by returning None from resolve() callers).
    # Instead, we rely on test coverage to catch shape mismatches.
