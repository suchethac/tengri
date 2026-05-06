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

    _PREFERRED_COLS = ("name", "kind", "tier", "status", "citation", "short_doc", "use")
    _ALWAYS_HIDDEN = ("module", "requires", "params")  # surfaced via describe()

    def _columns(self) -> list[str]:
        """Decide which columns to render.

        - ``kind`` is shown only when results span more than one kind
          (e.g. cross-menu search) so single-menu tables stay narrow.
        - ``use`` (the call-site hint) is shown only on cross-menu /
          search tables — single-menu tables expose it via describe()
          to keep the row width readable.
        """
        kinds = {d.get("kind") for d in self}
        hidden = set(self._ALWAYS_HIDDEN)
        if len(kinds) <= 1:
            hidden.add("kind")
            hidden.add("use")
        all_keys = list(self[0].keys())
        cols = [k for k in self._PREFERRED_COLS if k in all_keys and k not in hidden]
        cols += [k for k in all_keys if k not in cols and k not in hidden]
        return cols

    def __repr__(self) -> str:
        if not self:
            return "(empty)"
        cols = self._columns()

        # Truncate very long fields for readability.
        def _cell(v: Any) -> str:
            s = "" if v is None else str(v)
            return s if len(s) <= 80 else s[:77] + "..."

        widths = {k: max(len(k), *(len(_cell(d.get(k, ""))) for d in self)) for k in cols}
        header = "  ".join(k.ljust(widths[k]) for k in cols)
        sep = "  ".join("─" * widths[k] for k in cols)
        rows = "\n".join(
            "  ".join(_cell(d.get(k, "")).ljust(widths[k]) for k in cols) for d in self
        )
        # Footer "kind" reads "mixed" for cross-menu (search) results.
        kinds = {d.get("kind", "entry") for d in self}
        kind_label = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        footer = f"\n[{len(self)} result{'s' if len(self) != 1 else ''} — {kind_label}]"
        # Hint at how to actually use any row — only shown on single-kind
        # tables (where the `use` column is hidden) so search tables don't
        # repeat themselves.
        if len(kinds) == 1 and self and "use" in self[0]:
            example = next((d.get("use", "") for d in self if d.get("use")), "")
            if example:
                footer += f"\n  Use:  tengri.describe({self[0]['name']!r})  →  {example}"
        return f"{header}\n{sep}\n{rows}{footer}"

    def _repr_html_(self) -> str:
        """Jupyter HTML repr — renders as a real HTML table in notebooks."""
        if not self:
            return "<i>(empty)</i>"
        cols = self._columns()
        head = "".join(f"<th style='text-align:left'>{k}</th>" for k in cols)
        body = "".join(
            "<tr>"
            + "".join(f"<td style='text-align:left'>{d.get(k, '')}</td>" for k in cols)
            + "</tr>"
            for d in self
        )
        kinds = {d.get("kind", "entry") for d in self}
        kind_label = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        return (
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
            f"<i>{len(self)} result{'s' if len(self) != 1 else ''} — {kind_label}</i>"
        )


class _DescribeRecord(dict):
    """A `dict` that prints as a labelled block. Plain dict otherwise."""

    def __repr__(self) -> str:
        if not self:
            return "(empty)"
        width = max(len(k) for k in self) + 2
        lines = []
        for k, v in self.items():
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                # Multi-line list rendering: one item per row, indented under the key.
                if len(v) == 0:
                    lines.append(f"  {k.ljust(width)}(none)")
                elif len(v) <= 4:
                    lines.append(f"  {k.ljust(width)}{', '.join(v)}")
                else:
                    lines.append(f"  {k.ljust(width)}{v[0]}")
                    for item in v[1:]:
                        lines.append(f"  {' ' * width}{item}")
            else:
                lines.append(f"  {k.ljust(width)}{v}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        def _fmt(v):
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                return "<br>".join(v)
            return str(v)

        _row_tpl = (
            "<tr><th style='text-align:left'>{k}</th><td style='text-align:left'>{v}</td></tr>"
        )
        rows = "".join(_row_tpl.format(k=k, v=_fmt(v)) for k, v in self.items())
        return f"<table>{rows}</table>"


# ──────────────────────────────────────────────────────────────────
# Internal entry-normalisation
# ──────────────────────────────────────────────────────────────────


def _extract_params(entry: Any, kind: str) -> list[str]:
    """Best-effort free-parameter list for a registry entry.

    AGN entries → introspect callable signature, keep names starting with ``agn_``.
    SFH entries → ``callable.params`` (an ``SFHModelSpec`` field) → key list.
    Dust laws / others → empty (parameters come from the caller, not the
    registered function).
    """
    import inspect

    if kind == "sfh_model":
        spec = getattr(entry, "callable", None)
        params = getattr(spec, "params", None)
        if isinstance(params, dict):
            return list(params)

    if kind == "agn_model":
        fn = getattr(entry, "callable", None)
        if fn is None:
            return []
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return []
        return [p.name for p in sig.parameters.values() if p.name.startswith("agn_")]

    return []


def _usage_hint(name: str, kind: str) -> str:
    """Return a copy-pasteable one-liner showing how to actually use this
    entry — the missing piece between "I found a thing called skirtor"
    and "now what?"

    Patterns are based on the canonical spec/filter/fitter call sites
    used in ``docs/spine/00_quickstart.py``.
    """
    if kind == "filter":
        return f'Photometry.from_names(["{name}"])'
    if kind == "agn_model":
        return f'Parameters(..., agn_model="{name}")'
    if kind == "dust_attenuation":
        return f'Parameters(..., dust_law="{name}")'
    if kind == "dust_emission":
        return f'Parameters(..., dust_emission="{name}")'
    if kind == "sfh_model":
        return f'Parameters(..., mean_sfh_type="{name}")'
    if kind == "nebular_backend":
        return f'Parameters(..., nebular_backend="{name}")'
    if kind == "inference_method":
        return f'fitter.run("{name}")'
    if kind == "component":
        return f"tengri.{name}  (or tengri.list_{name}_models() / tengri.list_{name}_laws() if it has alternatives)"
    return ""


def _entry_to_dict(name: str, entry: Any, *, kind: str) -> dict:
    """Normalize a registry entry (varied dataclass shapes) into a uniform dict."""
    out = {
        "name": name,
        "kind": kind,
        "status": getattr(entry, "status", "production"),
        "citation": getattr(entry, "citation", ""),
        "short_doc": getattr(entry, "short_doc", ""),
        "use": _usage_hint(name, kind),
    }
    params = _extract_params(entry, kind)
    if params:
        out["params"] = params
    return out


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
def _emission_entry(d: dict[str, str]) -> dict:
    """Inject `use` field into a dust-emission menu row."""
    return {**d, "use": _usage_hint(d["name"], "dust_emission")}


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


_DUST_EMISSION_MENU = tuple(_emission_entry(d) for d in _DUST_EMISSION_MENU)


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
    raw = [
        (
            "baked_in",
            "production",
            "DSPS / FSPS SSP-internal",
            "Emission baked into SSP grid; zero free params",
        ),
        ("cue", "production", "Li+2024 (CUE neural emulator)", "Neural-network Cloudy emulator"),
        ("cloudy_grid", "production", "Byler+2017 grids", "Trilinear interp on Cloudy grid"),
        ("cb19", "experimental", "Charlot & Bruzual 2019", "Precomputed CB19 nebular grid"),
    ]
    return _RegistryTable(
        [
            {
                "name": n,
                "kind": "nebular_backend",
                "status": st,
                "citation": cit,
                "short_doc": doc,
                "use": _usage_hint(n, "nebular_backend"),
            }
            for (n, st, cit, doc) in raw
        ]
    )


_COMPONENT_DOCS: tuple[tuple[str, str, str], ...] = (
    (
        "stellar",
        "tengri.components.stellar.component",
        "SSP integration over SFH (DSPS), metallicity history, mass remaining",
    ),
    (
        "dust",
        "tengri.components.dust.component",
        "Two-component attenuation (BC + diffuse) — 21 laws available",
    ),
    (
        "agn",
        "tengri.components.agn.component",
        "Disc + torus + polar dust + BLR/NLR — 12 models available",
    ),
    (
        "nebular",
        "tengri.components.nebular.component",
        "Emission lines + continuum — BakedIn / CUE / CloudyGrid / CB19",
    ),
    (
        "radio",
        "tengri.components.radio.component",
        "SF synchrotron + AGN jets + free-free continuum",
    ),
    ("igm", "tengri.components.igm.component", "IGM transmission (Inoue+2014) + DLA absorption"),
    ("xray", "tengri.components.xray.component", "X-ray binaries + AGN corona"),
)


def list_filters() -> _RegistryTable:
    """List every filter curve bundled with tengri.

    Filter files live in ``data/filters/`` (relative to the install root)
    and follow the SVO naming convention ``Survey_Instrument_Band.dat``.

    Returns
    -------
    _RegistryTable
        One row per filter, with columns ``name`` (file stem),
        ``survey``, ``instrument``, ``band``. Prints as a table, also
        renders as HTML in Jupyter.

    Notes
    -----
    Counts and groupings are computed from the live filesystem on each
    call — no hardcoding. Add a ``Survey_Instrument_Band.dat`` file to
    ``data/filters/`` and it appears here automatically.
    """
    import os

    # Resolve filter directory: prefer repo-local ``data/filters/``,
    # fall back to package-local if installed wheel-style.
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "filters"),
        os.path.join(os.path.dirname(__file__), "data", "filters"),
        os.path.join(os.getcwd(), "data", "filters"),
    ]
    filter_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if filter_dir is None:
        return _RegistryTable([])
    out = []
    for fname in sorted(os.listdir(filter_dir)):
        if not fname.endswith(".dat"):
            continue
        stem = fname[:-4]
        parts = stem.split("_", 2)
        survey = parts[0] if len(parts) >= 1 else ""
        instr = parts[1] if len(parts) >= 2 else ""
        band = parts[2] if len(parts) >= 3 else ""
        out.append(
            {
                "name": stem,
                "kind": "filter",
                "survey": survey,
                "instrument": instr,
                "band": band,
                "use": _usage_hint(stem, "filter"),
            }
        )
    return _RegistryTable(out)


def list_components() -> _RegistryTable:
    """List the SEDComponent adapters currently wired into the forward model."""
    out = []
    for name, module_path, short_doc in _COMPONENT_DOCS:
        try:
            __import__(module_path)
            out.append(
                {
                    "name": name,
                    "kind": "component",
                    "status": "production",
                    "module": module_path,
                    "short_doc": short_doc,
                    "use": _usage_hint(name, "component"),
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
                    "use": "",
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
        # Inference methods use ``tier`` (primary | experimental) instead of
        # ``status``; status is omitted to avoid a redundant column.
        out.append(
            {
                "name": entry.name,
                "kind": "inference_method",
                "tier": entry.tier,
                "short_doc": entry.short_doc,
                "requires": list(entry.requires),
                "use": _usage_hint(entry.name, "inference_method"),
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
        list_filters,
    ):
        for entry in fn():
            if entry["name"] == name:
                return _DescribeRecord(entry)
    raise KeyError(
        f"Unknown name '{name}'.  Try tengri.summary() for a menu of every "
        "AGN model, dust law, SFH variant, nebular backend, component, "
        "filter, or "
        "inference method that exists."
    )


def search(query: str) -> _RegistryTable:
    """Cross-menu fuzzy search by name, short_doc, citation, or status.

    Walks every menu — components, inference methods, AGN models, dust
    attenuation laws, dust emission templates, SFH models, nebular
    backends — and returns every entry whose name, short_doc, citation,
    or status (case-insensitively) contains ``query``.

    Parameters
    ----------
    query : str
        Substring to match (case-insensitive).

    Returns
    -------
    _RegistryTable
        Matching entries from every menu, with a ``kind`` column so
        you can tell them apart. Prints as a table in the REPL.

    Examples
    --------
    >>> tengri.search("torus")  # find every torus model anywhere
    >>> tengri.search("pah")  # find every PAH-related thing
    >>> tengri.search("Leja")  # find everything Leja-cited
    """
    q = query.lower()
    hits: list[dict] = []
    for fn in (
        list_components,
        list_inference_methods,
        list_agn_models,
        list_dust_laws,
        list_dust_emission_models,
        list_sfh_models,
        list_nebular_backends,
        list_filters,
    ):
        for entry in fn():
            # Search every string-valued field — covers name, short_doc,
            # citation, status, plus filter-specific survey/instrument/band.
            haystack = " ".join(str(v) for v in entry.values() if isinstance(v, str)).lower()
            if q in haystack:
                hits.append(entry)
    return _RegistryTable(hits)


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
        "filters": list_filters(),
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
        (len(list_filters()), "photometric filters", "list_filters()"),
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


_TOPIC_HELP: dict[str, tuple[str, callable]] = {
    "agn": ("AGN models", lambda: list_agn_models()),
    "dust": (
        "dust attenuation + emission",
        lambda: _RegistryTable(list(list_dust_laws()) + list(list_dust_emission_models())),
    ),
    "sfh": ("SFH models", lambda: list_sfh_models()),
    "nebular": ("nebular backends", lambda: list_nebular_backends()),
    "components": ("physics components", lambda: list_components()),
    "inference": ("inference methods", lambda: list_inference_methods()),
    "filters": ("photometric filters", lambda: list_filters()),
}


def help(topic: str | None = None) -> None:
    """Print a curated cheatsheet covering the entry points new users need.

    Parameters
    ----------
    topic : str, optional
        If given, narrow the cheatsheet to one menu. Recognised topics:
        ``"agn"``, ``"dust"``, ``"sfh"``, ``"nebular"``, ``"components"``,
        ``"inference"``, ``"filters"``. Without a topic the full
        cheatsheet is printed.

    Notes
    -----
    Counts are read live from the registries — adding a new model or
    inference backend updates the cheatsheet immediately, no edit needed.

    This shadows :func:`builtins.help` only when accessed as ``tengri.help``;
    the global ``help()`` builtin is unaffected.
    """
    if topic is not None:
        return _help_topic(topic)

    # ── Live counts so that the cheatsheet is never stale. ──
    n_agn = len(list_agn_models())
    n_atte = len(list_dust_laws())
    n_emis = len(list_dust_emission_models())
    n_sfh = len(list_sfh_models())
    n_neb = len(list_nebular_backends())
    n_inf = len(list_inference_methods(tier="primary"))
    try:
        from tengri import list_filters  # avoid circular import on cold load

        n_filt = len(list_filters())
    except Exception:
        n_filt = "?"

    text = f"""
tengri — differentiable galaxy SED fitting in JAX

────────────────────────────────────────────────────────────────────
1.  See what's available
────────────────────────────────────────────────────────────────────
    tengri.summary()                      one-line counts of every menu
    tengri.list_agn_models()              {n_agn} AGN models
    tengri.list_dust_laws()               {n_atte} attenuation curves (UV/optical)
    tengri.list_dust_emission_models()    {n_emis} IR emission templates
    tengri.list_sfh_models()              {n_sfh} SFH variants
    tengri.list_nebular_backends()        {n_neb} nebular backends
    tengri.list_inference_methods(tier="primary")  {n_inf} primary methods
    tengri.list_filters()                 {n_filt} filter curves
    tengri.describe("skirtor")            full metadata for any name
    tengri.search("torus")                cross-menu fuzzy search
    tengri.doctor()                       env / install / SSP health check
    tengri.help("dust")                   topical cheatsheet for one menu

────────────────────────────────────────────────────────────────────
2.  A minimal fit
────────────────────────────────────────────────────────────────────
    obs        = tengri.Observation(photometry=tengri.Photometry.from_names([...]))
    parameters = tengri.Parameters(...)        # priors + fixed values
    model      = tengri.SEDModel(parameters, ssp_data, observation=obs)
    fitter     = tengri.Fitter(model, data, noise)
    posterior  = fitter.run("map")             # or "nuts", "vi", …
    posterior.summary()                        # median ± 68% CI per param

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


def _help_topic(topic: str) -> None:
    """Topical help for one menu — used by ``help(topic=…)``."""
    topic_l = topic.lower()
    if topic_l not in _TOPIC_HELP:
        valid = sorted(_TOPIC_HELP)
        raise ValueError(
            f"Unknown help topic '{topic}'.  Valid topics: {valid}.  "
            "Or call tengri.help() with no argument for the full cheatsheet."
        )
    label, fetch = _TOPIC_HELP[topic_l]
    entries = fetch()
    print(f"\ntengri.help('{topic_l}') — {label}: {len(entries)} available\n")
    print(entries)
    print()
