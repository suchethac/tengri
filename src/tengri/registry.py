"""Public introspection façade — list and describe everything available in tengri.

Users can call these from a notebook to discover models, components, and
inference methods without leaving the REPL.
"""

from __future__ import annotations

from typing import Any


def _entry_to_dict(name: str, entry: Any, *, kind: str) -> dict:
    """Normalize a registry entry (varied dataclass shapes) into a uniform dict."""
    return {
        "name": name,
        "kind": kind,
        "status": getattr(entry, "status", "production"),
        "citation": getattr(entry, "citation", ""),
        "short_doc": getattr(entry, "short_doc", ""),
    }


def list_agn_models(*, status: str | None = None) -> list[dict]:
    """List all registered AGN SED models.

    Parameters
    ----------
    status : str or None
        Filter by status: "production", "experimental", "demo", or "deprecated".
        Default: None (all statuses).

    Returns
    -------
    list of dict
        Sorted by name. Each dict has keys: name, kind, status, citation, short_doc.
    """
    from tengri.components.agn.unified import AGN_MODELS

    out = [_entry_to_dict(n, e, kind="agn_model") for n, e in AGN_MODELS.items()]
    if status:
        out = [m for m in out if m["status"] == status]
    return sorted(out, key=lambda m: m["name"])


def list_dust_laws(*, status: str | None = None) -> list[dict]:
    """List all registered dust attenuation laws.

    Parameters
    ----------
    status : str or None
        Filter by status: "production", "experimental", "demo", or "deprecated".
        Default: None (all statuses).

    Returns
    -------
    list of dict
        Sorted by name. Each dict has keys: name, kind, status, citation, short_doc.
    """
    from tengri.components.dust.attenuation import DUST_LAWS

    out = [_entry_to_dict(n, e, kind="dust_law") for n, e in DUST_LAWS.items()]
    if status:
        out = [m for m in out if m["status"] == status]
    return sorted(out, key=lambda m: m["name"])


def list_sfh_models(*, status: str | None = None) -> list[dict]:
    """List all registered star formation history models.

    Parameters
    ----------
    status : str or None
        Filter by status: "production", "experimental", "demo", or "deprecated".
        Default: None (all statuses).

    Returns
    -------
    list of dict
        Sorted by name. Each dict has keys: name, kind, status, citation, short_doc.
    """
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    out = [_entry_to_dict(n, e, kind="sfh_model") for n, e in SFH_REGISTRY.items()]
    if status:
        out = [m for m in out if m["status"] == status]
    return sorted(out, key=lambda m: m["name"])


def list_nebular_backends() -> list[dict]:
    """List all available nebular emission backends.

    Notes
    -----
    Hard-coded list (nebular has no decorator registry yet). Each dict has
    keys: name, kind, status, citation, short_doc.

    Returns
    -------
    list of dict
        Available nebular backends.
    """
    return [
        {
            "name": "baked_in",
            "kind": "nebular_backend",
            "status": "production",
            "citation": "DSPS / FSPS SSP-internal",
            "short_doc": "Emission baked into SSP grid; zero free params",
        },
        {
            "name": "cue",
            "kind": "nebular_backend",
            "status": "production",
            "citation": "Li+2024 (CUE neural emulator)",
            "short_doc": "Neural-network Cloudy emulator",
        },
        {
            "name": "cloudy_grid",
            "kind": "nebular_backend",
            "status": "production",
            "citation": "Byler+2017 grids",
            "short_doc": "Trilinear interp on Cloudy grid",
        },
        {
            "name": "cb19",
            "kind": "nebular_backend",
            "status": "experimental",
            "citation": "Charlot & Bruzual 2019",
            "short_doc": "Precomputed CB19 nebular grid",
        },
    ]


def list_components() -> list[dict]:
    """List the SEDComponent adapters currently wired into the forward model.

    Returns
    -------
    list of dict
        Component info. Each dict has keys: name, kind, status, module, short_doc.
        status="broken" indicates an import failed.
    """
    components = [
        ("stellar", "tengri.components.stellar.component"),
        ("dust", "tengri.components.dust.component"),
        ("agn", "tengri.components.agn.component"),
        ("nebular", "tengri.components.nebular.component"),
        ("radio", "tengri.components.radio.component"),
        ("igm", "tengri.components.igm.component"),
        ("xray", "tengri.components.xray.component"),
    ]
    out = []
    for name, module_path in components:
        try:
            __import__(module_path)
            out.append(
                {
                    "name": name,
                    "kind": "component",
                    "status": "production",
                    "module": module_path,
                    "short_doc": "",
                }
            )
        except Exception as e:
            out.append(
                {
                    "name": name,
                    "kind": "component",
                    "status": "broken",
                    "module": module_path,
                    "short_doc": f"import failed: {e!r}",
                }
            )
    return out


def list_inference_methods(*, tier: str | None = None) -> list[dict]:
    """List all registered inference methods.

    Parameters
    ----------
    tier : str or None
        Filter by tier: "primary" or "experimental". Default: None (all tiers).

    Returns
    -------
    list of dict
        Each dict has keys: name, kind, tier, status, short_doc, requires.
        requires is a list of optional dependency names.
    """
    from tengri.inference._backend_registry import all_backends

    out = []
    for entry in all_backends():
        out.append(
            {
                "name": entry.name,
                "kind": "inference_method",
                "tier": entry.tier,
                "status": "primary" if entry.tier == "primary" else "experimental",
                "short_doc": entry.short_doc,
                "requires": list(entry.requires),
            }
        )
    if tier:
        out = [m for m in out if m["tier"] == tier]
    return out


def describe(name: str) -> dict:
    """Universal lookup across every menu.

    Parameters
    ----------
    name : str
        Name of a model, method, or component to describe.

    Returns
    -------
    dict
        Metadata for the matched entry.

    Raises
    ------
    KeyError
        If name is not found in any registry.
    """
    for fn, _kind in [
        (list_inference_methods, "inference_method"),
        (list_agn_models, "agn_model"),
        (list_dust_laws, "dust_law"),
        (list_sfh_models, "sfh_model"),
        (list_nebular_backends, "nebular_backend"),
        (list_components, "component"),
    ]:
        for entry in fn():
            if entry["name"] == name:
                return entry
    raise KeyError(
        f"Unknown name '{name}'. Try one of "
        "list_inference_methods(), list_agn_models(), list_dust_laws(), "
        "list_sfh_models(), list_nebular_backends(), list_components()."
    )


def list_all() -> dict:
    """Return everything available — useful for a single notebook cell overview.

    Returns
    -------
    dict
        Keys: components, inference_methods, agn_models, dust_laws, sfh_models,
        nebular_backends. Each value is a list of dicts from the corresponding
        list_*() function.
    """
    return {
        "components": list_components(),
        "inference_methods": list_inference_methods(),
        "agn_models": list_agn_models(),
        "dust_laws": list_dust_laws(),
        "sfh_models": list_sfh_models(),
        "nebular_backends": list_nebular_backends(),
    }
