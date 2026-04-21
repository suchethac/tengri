"""Split mcmc/common.py god-file into per-sampler modules.

Reads the original common.py from git (commit 18948ad), extracts each top-level
function group by line range, and writes the correct per-sampler files.
Shared infrastructure goes to _shared.py; common.py becomes a thin re-export hub.

Run from repo root:
    python scripts/split_mcmc_common.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MCMC_DIR = REPO_ROOT / "src" / "tengri" / "inference" / "backends" / "mcmc"
ORIGINAL_COMMIT = "18948ad"

# Scan functions each sampler needs from _shared.py
_SAMPLER_SCAN_IMPORTS: dict[str, list[str]] = {
    "raytrace":    [],
    "nuts":        ["_nuts_burnin_scan", "_nuts_sample_scan"],
    "hmc":         ["_hmc_burnin_scan", "_hmc_sample_scan"],
    "dynamic_hmc": ["_dynamic_hmc_burnin_scan", "_dynamic_hmc_sample_scan"],
    "ghmc":        ["_ghmc_burnin_scan", "_ghmc_sample_scan"],
    "mclmc":       ["_adjusted_mclmc_sample_scan", "_mclmc_sample_scan"],
}

# Kernel getters each sampler needs
_SAMPLER_KERNEL_IMPORTS: dict[str, list[str]] = {
    "raytrace":    [],
    "nuts":        [],
    "hmc":         ["_get_hmc_kernel"],
    "dynamic_hmc": ["_get_dynamic_hmc_kernel"],
    "ghmc":        ["_get_ghmc_kernel"],
    "mclmc":       [],
}

_BASE_IMPORTS = """\
from __future__ import annotations

import logging
import time
import warnings

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri.inference._model_cache import get_model_cache
from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical
"""


def _sampler_imports(module_name: str) -> str:
    """Build the full import block for a per-sampler file."""
    shared_names = sorted(
        ["_get_cached_adaptation", "_get_flat_logdensity", "_set_cached_adaptation"]
        + _SAMPLER_SCAN_IMPORTS.get(module_name, [])
        + _SAMPLER_KERNEL_IMPORTS.get(module_name, [])
    )
    shared_import = (
        "from tengri.inference.backends.mcmc._shared import (\n"
        + "".join(f"    {n},\n" for n in shared_names)
        + ")\n"
    )
    return _BASE_IMPORTS + shared_import + "\nlogger = logging.getLogger(__name__)\n"


def read_original() -> str:
    result = subprocess.run(
        ["git", "show", f"{ORIGINAL_COMMIT}:src/tengri/inference/backends/mcmc/common.py"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return result.stdout


def top_level_spans(src: str) -> dict[str, tuple[int, int]]:
    """Return {name: (start_line, end_line)} for top-level functions (1-indexed, inclusive)."""
    tree = ast.parse(src)
    total = len(src.splitlines())
    fns: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.col_offset == 0:
            fns[node.name] = node.lineno
    ordered = sorted(fns.items(), key=lambda x: x[1])
    spans: dict[str, tuple[int, int]] = {}
    for i, (name, start) in enumerate(ordered):
        end = ordered[i + 1][1] - 1 if i + 1 < len(ordered) else total
        spans[name] = (start, end)
    return spans


def extract(src: str, start: int, end: int) -> str:
    """Lines start..end (1-indexed, inclusive), trailing whitespace stripped."""
    return "\n".join(src.splitlines()[start - 1 : end]).rstrip()


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    n = len(content.splitlines())
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({n} lines)")


def main() -> None:
    src = read_original()
    spans = top_level_spans(src)
    print(f"Parsed {len(src.splitlines())} lines, {len(spans)} top-level functions found")

    # ── _shared.py — all infrastructure before run_* ──────────────────────
    shared_fns = [
        "_get_nuts_kernel", "_get_hmc_kernel", "_get_dynamic_hmc_kernel", "_get_ghmc_kernel",
        "_nuts_sample_scan", "_nuts_burnin_scan",
        "_hmc_sample_scan", "_hmc_burnin_scan",
        "_dynamic_hmc_sample_scan", "_dynamic_hmc_burnin_scan",
        "_ghmc_sample_scan", "_ghmc_burnin_scan",
        "_mclmc_sample_scan", "_adjusted_mclmc_sample_scan",
        "_get_flat_logdensity", "_get_cached_adaptation", "_set_cached_adaptation",
    ]
    shared_bodies = []
    for fn in shared_fns:
        if fn not in spans:
            print(f"  WARNING: {fn} not in spans", file=sys.stderr)
            continue
        s, e = spans[fn]
        shared_bodies.append(extract(src, s, e))

    shared_content = (
        '"""Shared MCMC infrastructure: kernel getters, scan functions, logdensity helpers.\n\n'
        "Internal — imported by per-sampler modules. Not part of the public API.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "import functools\n"
        "import warnings\n\n"
        "import jax\n"
        "import jax.numpy as jnp\n"
        "from jax.flatten_util import ravel_pytree\n\n"
        "from tengri.inference._model_cache import get_model_cache\n"
        "from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical\n"
        "\n\n"
        + "\n\n\n".join(shared_bodies)
        + "\n"
    )
    write_file(MCMC_DIR / "_shared.py", shared_content)

    # ── per-sampler files ──────────────────────────────────────────────────
    sampler_map: dict[str, list[str]] = {
        "nuts":        ["run_nuts"],
        "hmc":         ["run_hmc"],
        "dynamic_hmc": ["run_dynamic_hmc"],
        "ghmc":        ["run_ghmc"],
        "mclmc":       ["run_mclmc", "run_adjusted_mclmc"],
    }
    # raytrace.py already exists with the low-level sample_raytrace implementation;
    # we append run_raytrace (the high-level Fitter wrapper) rather than overwriting.
    # run_elliptical_slice is already in elliptical_slice.py — just re-exported.

    sampler_labels = {
        "nuts":        "NUTS (No-U-Turn Sampler) via BlackJAX",
        "hmc":         "Standard Hamiltonian Monte Carlo via BlackJAX",
        "dynamic_hmc": "Dynamic HMC via BlackJAX",
        "ghmc":        "Generalized HMC via BlackJAX",
        "mclmc":       "MCLMC and Adjusted MCLMC via BlackJAX",
    }

    # ── Append run_raytrace to existing raytrace.py ───────────────────────
    # raytrace.py already has from __future__ + imports at the top; we only
    # add the new imports that aren't already there.
    raytrace_path = MCMC_DIR / "raytrace.py"
    existing_raytrace = raytrace_path.read_text(encoding="utf-8")
    if "def run_raytrace" not in existing_raytrace:
        # Imports needed by run_raytrace that may not be in the existing file
        rt_extra_imports = (
            "import logging\n"
            "import time\n\n"
            "from tengri.inference._model_cache import get_model_cache\n"
            "from tengri.inference._sample_utils import _maybe_map_init, _mean_params, _vmap_samples_to_physical\n"
            "from tengri.inference.backends.mcmc._shared import (\n"
            "    _get_cached_adaptation,\n"
            "    _get_flat_logdensity,\n"
            "    _set_cached_adaptation,\n"
            ")\n\n"
            "logger = logging.getLogger(__name__)\n"
        )
        s, e = spans["run_raytrace"]
        rt_body = extract(src, s, e)
        raytrace_path.write_text(
            existing_raytrace.rstrip()
            + "\n\n\n# ── Fitter interface ─────────────────────────────────\n\n"
            + rt_extra_imports
            + "\n\n"
            + rt_body
            + "\n",
            encoding="utf-8",
        )
        n = len(raytrace_path.read_text().splitlines())
        print(f"  appended run_raytrace to raytrace.py  ({n} lines total)")

    for module_name, fn_names in sampler_map.items():
        bodies = []
        for fn_name in fn_names:
            if fn_name not in spans:
                print(f"  WARNING: {fn_name} not in spans", file=sys.stderr)
                continue
            s, e = spans[fn_name]
            bodies.append(extract(src, s, e))

        content = (
            f'"""{sampler_labels[module_name]}.\n\n'
            "Extracted from mcmc/common.py. Import via ``tengri.inference.backends.mcmc.common``.\n"
            '"""\n\n'
            + _sampler_imports(module_name)
            + "\n\n"
            + "\n\n\n".join(bodies)
            + "\n"
        )
        write_file(MCMC_DIR / f"{module_name}.py", content)

    # ── thin re-export hub: common.py ─────────────────────────────────────
    public_fns = (
        sorted(fn for fns in sampler_map.values() for fn in fns)
        + ["run_elliptical_slice", "run_raytrace"]
    )
    hub_lines = [
        '"""MCMC samplers — re-export hub preserving the original public API.',
        "",
        "Samplers live in individual modules:",
        "  _shared.py     — kernel getters, scan functions, logdensity helpers",
        "  raytrace.py    — run_raytrace",
        "  nuts.py        — run_nuts",
        "  hmc.py         — run_hmc",
        "  dynamic_hmc.py — run_dynamic_hmc",
        "  ghmc.py        — run_ghmc",
        "  mclmc.py       — run_mclmc, run_adjusted_mclmc",
        "  elliptical_slice.py — run_elliptical_slice",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from tengri.inference.backends.mcmc.dynamic_hmc import run_dynamic_hmc",
        "from tengri.inference.backends.mcmc.elliptical_slice import run_elliptical_slice",
        "from tengri.inference.backends.mcmc.ghmc import run_ghmc",
        "from tengri.inference.backends.mcmc.hmc import run_hmc",
        "from tengri.inference.backends.mcmc.mclmc import run_adjusted_mclmc, run_mclmc",
        "from tengri.inference.backends.mcmc.nuts import run_nuts",
        "from tengri.inference.backends.mcmc.raytrace import run_raytrace",
        "",
        "__all__ = [",
    ]
    for name in sorted(public_fns):
        hub_lines.append(f'    "{name}",')
    hub_lines.extend(["]", ""])
    write_file(MCMC_DIR / "common.py", "\n".join(hub_lines))

    print("\nDone. Run:")
    print("  .venv/bin/ruff check --fix src/tengri/inference/backends/mcmc/")


if __name__ == "__main__":
    main()
