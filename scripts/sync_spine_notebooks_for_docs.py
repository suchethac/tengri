#!/usr/bin/env python3
"""Sync spine Jupytext sources into docs/spine/*.ipynb for Sphinx + nbsphinx.

Steps (per notebook):
1. ``jupytext --sync`` on ``notebooks/<slug>.py`` (creates/updates the paired
   ``notebooks/<slug>.ipynb``; that path is gitignored at repo root).
2. Copy the synced ``.ipynb`` into ``docs/spine/<slug>.ipynb``.

If the paired file is missing after sync, falls back to:

   jupytext notebooks/<slug>.py --to ipynb -o docs/spine/<slug>.ipynb

3. Run ``jupyter nbconvert --to notebook --inplace`` on each ``docs/spine/*.ipynb``
   (format check; nbsphinx uses nbconvert when building HTML).

4. Normalize markdown headings: exactly one H1 ``# <slug>`` (file stem); any other
   top-level ``#`` lines become ``##`` so Sphinx does not list them in the sidebar.

Run from repository root::

    python scripts/sync_spine_notebooks_for_docs.py

Or via ``make -C docs spine-ipynb``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SPINE_SLUGS = [
    "00_quickstart",
    "01_why_jax",
    "02_sed_anatomy",
    "03_fitting_photometry",
    "04_fitting_spectra",
    "05_joint_photometry_spectroscopy",
    "06_inference_methods",
    "07_degeneracies",
    "08_sfh_advanced",
    "09_dust_emission",
    "10_agn_advanced",
    "11_population",
    "12_diagnostics",
    "13_extending_tengri",
    "14_stochastic_sfh",
    "15_vi_inference",
    "16_simulation_interface",
    "17_emission_line_measurements",
]

# Human-readable titles for Sphinx sidebar (keyed by slug).
SPINE_TITLES: dict[str, str] = {
    "00_quickstart": "Quickstart",
    "01_why_jax": "Why JAX",
    "02_sed_anatomy": "SED Anatomy",
    "03_fitting_photometry": "Fitting Photometry",
    "04_fitting_spectra": "Fitting Spectra",
    "05_joint_photometry_spectroscopy": "Joint Photometry + Spectroscopy",
    "06_inference_methods": "Inference Methods",
    "07_degeneracies": "Age-Dust-Metallicity Degeneracies",
    "08_sfh_advanced": "Advanced Star Formation Histories",
    "09_dust_emission": "Dust Emission",
    "10_agn_advanced": "Advanced AGN Models",
    "11_population": "Population Inference",
    "12_diagnostics": "Model Diagnostics",
    "13_extending_tengri": "Extending tengri",
    "14_stochastic_sfh": "Stochastic SFH (Paper II preview)",
    "15_vi_inference": "VI Inference (Paper II preview)",
    "16_simulation_interface": "Simulation Interface",
    "17_emission_line_measurements": "Emission Line Measurements",
}


def _demote_extra_h1_lines(text: str) -> str:
    """Turn ``# foo`` into ``## foo`` except the first line (already the doc title)."""
    lines = text.split("\n")
    fixed: list[str] = []
    for i, line in enumerate(lines):
        if i == 0:
            fixed.append(line)
            continue
        stripped = line.lstrip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            indent = line[: len(line) - len(stripped)]
            fixed.append(indent + "#" + stripped)
        else:
            fixed.append(line)
    return "\n".join(fixed)


def normalize_markdown_headings(ipynb_path: Path, slug: str) -> None:
    """One H1 title in the first cell; demote other top-level ``#`` lines to ``##``."""
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    title = SPINE_TITLES.get(slug, slug)
    title_heading = f"# {title}"
    first_md = True
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if first_md:
            m = re.search(r"^#", text, flags=re.MULTILINE)
            if m:
                text = text[m.start() :]
            new_text, n_sub = re.subn(
                r"^#[^\n]*\n+",
                title_heading + "\n\n",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if n_sub == 0:
                new_text = title_heading + "\n\n" + text.lstrip("\n")
            cell["source"] = _demote_extra_h1_lines(new_text)
            first_md = False
        else:
            lines = text.split("\n")
            fixed: list[str] = []
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    indent = line[: len(line) - len(stripped)]
                    fixed.append(indent + "#" + stripped)
                else:
                    fixed.append(line)
            cell["source"] = "\n".join(fixed)
    ipynb_path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    nb_root = root / "notebooks"
    spine_out = root / "docs" / "spine"
    spine_out.mkdir(parents=True, exist_ok=True)

    for slug in SPINE_SLUGS:
        py_path = nb_root / f"{slug}.py"
        paired_ipynb = nb_root / f"{slug}.ipynb"
        out_ipynb = spine_out / f"{slug}.ipynb"

        if not py_path.is_file():
            print(f"error: missing {py_path}", file=sys.stderr)
            return 1

        subprocess.run(
            ["jupytext", "--sync", str(py_path)],
            check=True,
            cwd=str(root),
        )

        if paired_ipynb.is_file():
            shutil.copy2(paired_ipynb, out_ipynb)
        else:
            subprocess.run(
                [
                    "jupytext",
                    str(py_path),
                    "--to",
                    "ipynb",
                    "-o",
                    str(out_ipynb),
                ],
                check=True,
                cwd=str(root),
            )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--inplace",
                str(out_ipynb),
            ],
            check=True,
            cwd=str(root),
        )

        normalize_markdown_headings(out_ipynb, slug)

        print(f"synced {slug} -> {out_ipynb.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
