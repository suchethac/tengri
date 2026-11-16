# SPDX-License-Identifier: BSD-3-Clause
"""Resolve ``docs/model_wheel.yaml`` against the live tengri registries.

One resolver, two callers: ``docs/conf.py`` writes the model catalog page data
from it at build time, and ``tools/check_model_wheel.py`` checks coverage with
it. Sharing it means the guard checks exactly what ships.

Model sources in the YAML:

* ``lister:name`` or ``lister:category/name`` — a row in ``tengri.<lister>()``;
  its ``citation``, ``status`` and ``use`` are taken from the registry.
* ``ssp:<regex>`` — matched against ``tengri.list_known_ssps()`` names.
* ``param:<text>`` — a build option with no registry row; the text is its ``use``.
"""
from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path

import yaml

WHEEL_YAML = Path(__file__).resolve().parents[1] / "docs" / "model_wheel.yaml"
SSP_AXES = ("sps_codes", "isochrones", "spectral_libraries", "imf", "multiplicity")
_SKIP_LISTERS = ("list_all", "list_known_ssps", "list_filters")
_EMPTY_CITES = ("", "-", None)


def load_wheel(path: Path = WHEEL_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def registry_rows() -> dict[str, dict]:
    """Every registry row keyed as ``lister:name`` or ``lister:category/name``."""
    import tengri

    rows: dict[str, dict] = {}
    listers = (n for n in dir(tengri) if n.startswith("list_") and n not in _SKIP_LISTERS)
    for lister in listers:
        for row in getattr(tengri, lister)():
            if not isinstance(row, dict) or "name" not in row:
                rows[f"{lister}:{row}"] = {}
                continue
            cat = row.get("category")
            key = f"{lister}:{cat}/{row['name']}" if cat else f"{lister}:{row['name']}"
            rows[key] = row
    return rows


def ssp_names() -> list[str]:
    import tengri

    return [r["name"] for r in tengri.list_known_ssps()]


def _resolve_model(model: dict, rows: dict, ssps: list[str], where: str) -> dict:
    """One wheel model -> its JSON record. Raises on a source that does not resolve."""
    first_row, grids, use = None, [], ""
    for src in model["src"]:
        if src.startswith("ssp:"):
            hits = [s for s in ssps if re.search(src[4:], s)]
            if not hits:
                raise LookupError(f"{where}: {src} matches no SSP grid")
            grids.extend(h for h in hits if h not in grids)
        elif src.startswith("param:"):
            use = use or src[6:]
        elif src in rows:
            first_row = first_row or rows[src]
        else:
            raise LookupError(f"{where}: {src} is not a registry entry")
    row = first_row or {}
    reg_cite = row.get("citation")
    cite = reg_cite if reg_cite not in _EMPTY_CITES else (model.get("cite") or "")
    return {
        "label": model["label"],
        "short": model.get("short", ""),
        "cite": cite,
        "cite_status": "" if reg_cite not in _EMPTY_CITES else model.get("cite_status", ""),
        "status": row.get("status") or "",
        "use": row.get("use") or use,
        "src": list(model["src"]),
        "grids": grids,
    }


def resolve(wheel: dict | None = None) -> dict:
    """The wheel tree with every model resolved; the page renders this as-is."""
    wheel = wheel or load_wheel()
    rows, ssps = registry_rows(), ssp_names()
    modules = []
    for m in wheel["modules"]:
        groups = []
        for g in m["groups"]:
            models = [
                _resolve_model(md, rows, ssps, f"{m['id']}/{g['id']}/{md['label']}")
                for md in g["models"]
            ]
            groups.append({"id": g["id"], "label": g["label"], "short": g.get("short", ""), "models": models})
        modules.append(
            {"id": m["id"], "label": m["label"], "short": m.get("short", ""), "color": m["color"], "groups": groups}
        )
    return {"modules": modules}


def coverage_problems(wheel: dict | None = None) -> tuple[list[str], list[str], int]:
    """(registry problems, SSP-axis problems, number of registry entries drawn)."""
    wheel = wheel or load_wheel()
    rows, ssps = registry_rows(), ssp_names()
    used: set[str] = set()
    reg_errors: list[str] = []
    axis_errors: list[str] = []
    for m in wheel["modules"]:
        for g in m["groups"]:
            axis_hit: set[str] = set()
            for model in g["models"]:
                for src in model["src"]:
                    where = f"{m['id']}/{g['id']}/{model['label']}"
                    if src.startswith("ssp:"):
                        hits = [s for s in ssps if re.search(src[4:], s)]
                        if not hits:
                            reg_errors.append(f"{where}: {src} matches no SSP grid")
                        axis_hit.update(hits)
                    elif src.startswith("param:"):
                        continue
                    elif src not in rows:
                        reg_errors.append(f"{where}: {src} is not a registry entry")
                    else:
                        used.add(src)
            if m["id"] == "stellar" and g["id"] in SSP_AXES:
                missed = sorted(set(ssps) - axis_hit)
                if missed:
                    axis_errors.append(f"stellar/{g['id']}: {len(missed)} SSP grids unassigned: {missed}")
    excluded = wheel.get("excluded", {}).get("entries", [])
    for key in sorted(set(rows) - used):
        if not any(fnmatch.fnmatch(key, pat) for pat in excluded):
            reg_errors.append(f"registry entry not drawn and not excluded: {key}")
    return reg_errors, axis_errors, len(used)


def to_json(data: dict) -> str:
    """JSON safe to inline inside a ``<script type="application/json">`` block."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
