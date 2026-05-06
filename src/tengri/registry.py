"""Public introspection façade — list and describe everything available in tengri.

This is the first thing a new user should reach for. Open a notebook, run::

    import tengri

    tengri.help()  # the cheatsheet
    tengri.summary()  # one-line counts of every menu

then narrow down with::

    tengri.list_agn_models()  # all 12 AGN models, as a pretty table
    tengri.list_dust_laws(status="production")
    tengri.list_inference_methods(tier="primary")
    tengri.describe("skirtor")  # full metadata for any name

Returned values are normal Python lists, but they print as readable tables
in a notebook or REPL — no need to wrap them in `pprint`.
"""

from __future__ import annotations

from typing import Any


# ──────────────────────────────────────────────────────────────────
# Pretty-printed return types (still real lists/dicts — just nicer repr)
# ──────────────────────────────────────────────────────────────────


class _RegistryTable(list):
    """A `list[dict]` that prints as a column-aligned table.

    Behaves identically to a list otherwise — indexing, iteration,
    JSON serialisation, etc. all work as usual.
    """

    _PREFERRED_COLS = ("name", "tier", "status", "citation", "short_doc")

    def __repr__(self) -> str:
        if not self:
            return "(empty)"
        # Column order: preferred fields first, then any extras.
        all_keys = list(self[0].keys())
        cols = [k for k in self._PREFERRED_COLS if k in all_keys]
        cols += [k for k in all_keys if k not in cols and k not in ("kind", "module", "requires")]

        # Truncate very long fields for readability.
        def _cell(v: Any) -> str:
            s = "" if v is None else str(v)
            return s if len(s) <= 60 else s[:57] + "..."

        widths = {k: max(len(k), *(len(_cell(d.get(k, ""))) for d in self)) for k in cols}
        header = "  ".join(k.ljust(widths[k]) for k in cols)
        sep = "  ".join("─" * widths[k] for k in cols)
        rows = "\n".join(
            "  ".join(_cell(d.get(k, "")).ljust(widths[k]) for k in cols) for d in self
        )
        kind = self[0].get("kind", "entry")
        footer = f"\n[{len(self)} result{'s' if len(self) != 1 else ''} — {kind}]"
        return f"{header}\n{sep}\n{rows}{footer}"

    def _repr_html_(self) -> str:
        """Jupyter HTML repr — renders as a real HTML table in notebooks."""
        if not self:
            return "<i>(empty)</i>"
        all_keys = list(self[0].keys())
        cols = [k for k in self._PREFERRED_COLS if k in all_keys]
        cols += [k for k in all_keys if k not in cols and k not in ("kind", "module", "requires")]
        head = "".join(f"<th style='text-align:left'>{k}</th>" for k in cols)
        body = "".join(
            "<tr>"
            + "".join(f"<td style='text-align:left'>{d.get(k, '')}</td>" for k in cols)
            + "</tr>"
            for d in self
        )
        kind = self[0].get("kind", "entry")
        return (
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
            f"<i>{len(self)} result{'s' if len(self) != 1 else ''} — {kind}</i>"
        )


class _DescribeRecord(dict):
    """A `dict` that prints as a labelled block. Plain dict otherwise."""

    def __repr__(self) -> str:
        if not self:
            return "(empty)"
        width = max(len(k) for k in self) + 2
        lines = [f"  {k.ljust(width)}{v}" for k, v in self.items()]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><th style='text-align:left'>{k}</th><td>{v}</td></tr>" for k, v in self.items()
        )
        return f"<table>{rows}</table>"


# ──────────────────────────────────────────────────────────────────
# Internal entry-normalisation
# ──────────────────────────────────────────────────────────────────


def _entry_to_dict(name: str, entry: Any, *, kind: str) -> dict:
    """Normalize a registry entry (varied dataclass shapes) into a uniform dict."""
    return {
        "name": name,
        "kind": kind,
        "status": getattr(entry, "status", "production"),
        "citation": getattr(entry, "citation", ""),
        "short_doc": getattr(entry, "short_doc", ""),
    }


# ──────────────────────────────────────────────────────────────────
# Listing functions
# ──────────────────────────────────────────────────────────────────


def list_agn_models(*, status: str | None = None) -> _RegistryTable:
    """List all registered AGN SED models.

    Parameters
    ----------
    status : str, optional
        Filter by ``"production"``, ``"experimental"``, ``"demo"``, or
        ``"deprecated"``.

    Returns
    -------
    _RegistryTable
        List of metadata dicts. Prints as a table in a notebook.
    """
    from tengri.components.agn.unified import AGN_MODELS

    out = [_entry_to_dict(n, e, kind="agn_model") for n, e in AGN_MODELS.items()]
    if status:
        out = [m for m in out if m["status"] == status]
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_dust_laws(*, status: str | None = None) -> _RegistryTable:
    """List all registered dust **attenuation** laws.

    Attenuation describes how UV/optical photons are absorbed/scattered
    by dust along the line of sight (Calzetti, Cardelli, Charlot-Fall, …).
    For dust **emission** templates (DL07, Dale, MBB, …), see
    :func:`list_dust_emission_models`.
    """
    from tengri.components.dust.attenuation import DUST_LAWS

    out = [_entry_to_dict(n, e, kind="dust_attenuation") for n, e in DUST_LAWS.items()]
    if status:
        out = [m for m in out if m["status"] == status]
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


# Known dust emission template families. The runtime
# ``DUST_EMISSION_MODELS`` dict starts empty and is populated lazily
# when ``register_*_tabulated(grid_path)`` is called with a data file.
# We advertise the full menu so users can see what's *available* even
# before loading templates.
_DUST_EMISSION_MENU: tuple[dict[str, str], ...] = (
    {
        "name": "dl07",
        "status": "production",
        "citation": "Draine & Li 2007 (ApJ 657, 810)",
        "short_doc": "Diffuse + PAH grain mixture, Umin/Umax/qpah",
    },
    {
        "name": "dl14",
        "status": "production",
        "citation": "Draine et al. 2014 (ApJ 780, 172)",
        "short_doc": "Updated DL with extended PAH and silicate features",
    },
    {
        "name": "dale2014",
        "status": "production",
        "citation": "Dale et al. 2014 (ApJ 784, 83)",
        "short_doc": "SFR-driven empirical IR template family (alpha_sf)",
    },
    {
        "name": "astrodust",
        "status": "experimental",
        "citation": "Hensley & Draine 2023 (ApJ 948, 55)",
        "short_doc": "Astrodust + PAH unified grain model",
    },
    {
        "name": "themis",
        "status": "experimental",
        "citation": "Jones et al. 2017 (A&A 602, A46)",
        "short_doc": "THEMIS amorphous-carbon grain model",
    },
    {
        "name": "bosa",
        "status": "experimental",
        "citation": "Boquien et al. 2019 (CIGALE BOSA grids)",
        "short_doc": "BOSA dust SED templates",
    },
    {
        "name": "mbb",
        "status": "production",
        "citation": "Casey 2012 (MNRAS 425, 3094)",
        "short_doc": "Single-temperature modified blackbody (analytic)",
    },
)


def list_dust_emission_models(*, status: str | None = None) -> _RegistryTable:
    """List all available dust **emission** template families.

    Emission templates describe the IR re-radiation of energy absorbed by
    dust (DL07, DL14, Dale+2014, THEMIS, MBB, …). For UV/optical
    **attenuation** laws, see :func:`list_dust_laws`.

    Notes
    -----
    Templates are loaded lazily from data files via
    :func:`register_dl07_tabulated` etc.; this function shows the menu
    of *available* template families regardless of whether they have
    been loaded into the runtime ``DUST_EMISSION_MODELS`` dict.
    """
    out = [{**entry, "kind": "dust_emission"} for entry in _DUST_EMISSION_MENU]
    if status:
        out = [m for m in out if m["status"] == status]
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_sfh_models(*, status: str | None = None) -> _RegistryTable:
    """List all registered star formation history models."""
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    out = [_entry_to_dict(n, e, kind="sfh_model") for n, e in SFH_REGISTRY.items()]
    if status:
        out = [m for m in out if m["status"] == status]
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_nebular_backends() -> _RegistryTable:
    """List all available nebular emission backends."""
    return _RegistryTable(
        [
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
    )


def list_components() -> _RegistryTable:
    """List the SEDComponent adapters currently wired into the forward model."""
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
    return _RegistryTable(out)


def list_inference_methods(*, tier: str | None = None) -> _RegistryTable:
    """List all registered inference methods.

    Parameters
    ----------
    tier : str, optional
        Filter by ``"primary"`` (recommended for new users) or
        ``"experimental"``.
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
    return _RegistryTable(out)


# ──────────────────────────────────────────────────────────────────
# Describe + overview
# ──────────────────────────────────────────────────────────────────


def describe(name: str) -> _DescribeRecord:
    """Universal lookup across every menu.

    Parameters
    ----------
    name : str
        Name of a model, method, or component.

    Returns
    -------
    _DescribeRecord
        Metadata dict; prints as a labelled block.

    Raises
    ------
    KeyError
        If the name is not registered anywhere.
    """
    for fn in (
        list_inference_methods,
        list_agn_models,
        list_dust_laws,
        list_dust_emission_models,
        list_sfh_models,
        list_nebular_backends,
        list_components,
    ):
        for entry in fn():
            if entry["name"] == name:
                return _DescribeRecord(entry)
    raise KeyError(
        f"Unknown name '{name}'.  Try tengri.summary() for a menu of every "
        "AGN model, dust law, SFH variant, nebular backend, component, or "
        "inference method that exists."
    )


def list_all() -> dict[str, _RegistryTable]:
    """Return everything available — useful for a single notebook cell overview.

    Returns
    -------
    dict
        Keys: components, inference_methods, agn_models, dust_laws, sfh_models,
        nebular_backends. Each value is a `_RegistryTable` (list[dict]) that
        prints as a table.
    """
    return {
        "components": list_components(),
        "inference_methods": list_inference_methods(),
        "agn_models": list_agn_models(),
        "dust_laws": list_dust_laws(),
        "dust_emission_models": list_dust_emission_models(),
        "sfh_models": list_sfh_models(),
        "nebular_backends": list_nebular_backends(),
    }


# ──────────────────────────────────────────────────────────────────
# The two functions a new user calls first
# ──────────────────────────────────────────────────────────────────


def summary() -> None:
    """Print a one-line count of every menu in tengri.

    Notes
    -----
    This is what to run right after ``import tengri`` to see what's available.
    """
    counts = [
        (len(list_components()), "physics components", "list_components()"),
        (len(list_agn_models()), "AGN models", "list_agn_models()"),
        (len(list_dust_laws()), "dust attenuation laws", "list_dust_laws()"),
        (
            len(list_dust_emission_models()),
            "dust emission templates",
            "list_dust_emission_models()",
        ),
        (len(list_sfh_models()), "SFH models", "list_sfh_models()"),
        (len(list_nebular_backends()), "nebular backends", "list_nebular_backends()"),
        (
            len(list_inference_methods(tier="primary")),
            "primary inference methods",
            "list_inference_methods(tier='primary')",
        ),
        (len(list_inference_methods()), "total inference methods", "list_inference_methods()"),
    ]
    print("\ntengri — what's available:\n")
    width = max(len(label) for _, label, _ in counts)
    for n, label, call in counts:
        print(f"  {n:>4}  {label.ljust(width)}    tengri.{call}")
    print("\n  Look up any name:                          tengri.describe('skirtor')")
    print("  Curated cheatsheet for new users:          tengri.help()\n")


def help() -> None:  # noqa: A001 — intentional shadow inside the tengri namespace only
    """Print a curated cheatsheet covering the entry points new users need.

    This shadows :func:`builtins.help` only when accessed as ``tengri.help``;
    the global ``help()`` builtin is unaffected.
    """
    text = """
tengri — differentiable galaxy SED fitting in JAX

────────────────────────────────────────────────────────────────────
1.  See what's available
────────────────────────────────────────────────────────────────────
    tengri.summary()                      one-line counts of every menu
    tengri.list_agn_models()              12 AGN models (SKIRTOR, K&D, …)
    tengri.list_dust_laws()               21 attenuation curves (UV/optical)
    tengri.list_dust_emission_models()    7 IR emission templates (DL07, Dale, …)
    tengri.list_sfh_models()              34 SFH variants
    tengri.list_nebular_backends()        BakedIn / CUE / CloudyGrid / CB19
    tengri.list_inference_methods(tier="primary")
    tengri.describe("skirtor")            full metadata for any name

────────────────────────────────────────────────────────────────────
2.  A minimal fit
────────────────────────────────────────────────────────────────────
    obs        = tengri.Observation(photometry=tengri.Photometry.from_names([...]))
    parameters = tengri.Parameters(...)        # priors + fixed values
    model      = tengri.SEDModel(parameters, ssp_data, observation=obs)
    fitter     = tengri.Fitter(model, data, noise)
    posterior  = fitter.run("map")             # or "nuts", "vi", …

    See examples/ for runnable scripts.

────────────────────────────────────────────────────────────────────
3.  Contribute a new physics alternative
────────────────────────────────────────────────────────────────────
    Copy examples/contrib/example_new_agn_torus.py.  It registers a new
    AGN torus model with metadata (citation, status), then exercises the
    introspection API end-to-end.  See CONTRIBUTING.md for the 5-step recipe.

────────────────────────────────────────────────────────────────────
4.  Cite us
────────────────────────────────────────────────────────────────────
    tengri.print_citations()          acknowledgements for your paper
"""
    print(text)
