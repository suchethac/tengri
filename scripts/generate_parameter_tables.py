#!/usr/bin/env python
"""Generate per-domain parameter tables from component registries.

This script generates RST reference tables documenting all configurable
parameters for each physics component. Tables are written to
docs/_generated/ for inclusion in the published API reference.

The tables are organized by domain (sfh, dust_attenuation, neb, etc.) and
list each parameter's:
- Name (both short and full prefixed)
- Default value / free prior
- Units
- Description

Tables are regenerated on each `make html` via conf.py's setup hook.

Usage
-----
python scripts/generate_parameter_tables.py

Output
------
docs/_generated/parameter_tables.rst
    RST file with all per-domain parameter tables, ready for inclusion
    in the reference documentation via ``.. include::``.
"""

from __future__ import annotations

import pathlib


def structural_key_reference() -> str:
    """Render structural keys across all domains.

    Reads tengri.parameters.groups._GROUP_STRUCTURAL_KEYS and emits
    a reference table per domain. The internal '*' wildcard is rendered
    as 'all_params' (the user-facing name); '*' itself is never displayed.

    Returns
    -------
    str
        RST-formatted section with tables for each domain's structural keys.
    """
    from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS

    sections = []

    # Domain-specific structural key documentation
    STRUCTURAL_KEY_DOCS = {
        # Universal
        "type": "Physics variant (e.g., 'dpl', 'calzetti', 'cue'). Required.",
        "all_params": (
            "Wildcard: set all parameters to FREE or FIXED. Only user-facing wildcard (not '*')."
        ),
        # SFH
        "age_kernel": "Integration method: 'cic' (default, cloud-in-cell) or 'dsps' (histogram).",
        "bin_edges_gyr": "Non-parametric bin edges (Gyr). Only for type='histogram'.",
        "field_centering": "Field draw centering. Only for type='field'.",
        # Met (none beyond type/all_params)
        # Dust attenuation
        "law": "Dust law (e.g., 'calzetti', 'ccm89'). Required for single_component.",
        "law_bc": "Birth-cloud dust law. Required with law_diff on two_component.",
        "law_diff": "Diffuse dust law. Required with law_bc on two_component.",
        "law_neb": "Nebular dust law (reddens birth-cloud continuum only).",
        "dust_curve": "WG00 attenuation curve selector.",
        "geometry": "WG00 geometry ('slab', 'sphere', etc.).",
        "structure": "WG00 structure ('clumpy', 'homogeneous', etc.).",
        "slope_bc": "Birth-cloud law parameter override (two_component).",
        "slope_diff": "Diffuse law parameter override (two_component).",
        "slope_neb": "Nebular law parameter override (two_component).",
        "bump_strength_bc": "Birth-cloud bump strength override (two_component).",
        "bump_strength_diff": "Diffuse bump strength override (two_component).",
        "bump_strength_neb": "Nebular bump strength override (two_component).",
        "Rv_bc": "Birth-cloud RV override (two_component).",
        "Rv_diff": "Diffuse RV override (two_component).",
        "Rv_neb": "Nebular RV override (two_component).",
        "delta_bc": "Birth-cloud delta override (two_component).",
        "delta_diff": "Diffuse delta override (two_component).",
        "delta_neb": "Nebular delta override (two_component).",
        "lyman_cutoff": "Zero attenuation below 912 Å (Lyman limit). Two-component only.",
        "lyc_absorb_all": (
            "Absorb all ionizing photons (FSPS/CIGALE style) vs young-only. Two-component only."
        ),
        "eb_include_lyc": "Include ionizing luminosity in dust energy-balance integral.",
        # Dust emission
        "spinning_dust": "Include small spinning dust grains.",
        "f_cnm": "Cold neutral medium fraction.",
        "eta_balance": "Energy-balance coupling parameter or prior.",
        # Nebular
        "full_catalog": "Line catalog scope (bool).",
        "grid": "CLOUDY grid specification (dict).",
        # Shock
        "abundance": "Abundance mode: 'solar', 'lmc', etc.",
        # IGM
        "patchy": "Picket-fence vs smooth IGM (bool).",
        "dla": "Damped Lyman alpha sub-block. Omit or provide {'type': ...}.",
        # Radio / X-ray (no additional structural keys beyond type/all_params/subblocks)
        # AGN
        "norm": "Cross-component normalization: 'cigale_joint' (default) or 'independent'.",
        # AGN sub-blocks (named in _GROUP_STRUCTURAL_KEYS)
        "disc": "AGN accretion disk sub-block.",
        "torus": "AGN infrared torus sub-block.",
        "nlr": "Narrow-line region sub-block.",
        "blr": "Broad-line region sub-block.",
        "feii": "Iron emission sub-block.",
        "atten": "AGN attenuation sub-block.",
        "lines": "Deprecated: expands to (nlr, blr).",
        "sf": "Star-formation radio sub-block.",
        "agn": "AGN radio sub-block.",
        # Foreground (no type, only structural keys)
        "ebmv_mw": "Milky Way E(B-V) reddening (mag).",
        "rv": "Dust RV parameter for foreground reddening.",
    }

    # Top-level groups only (no '.' in the name)
    top_groups = sorted([g for g in _GROUP_STRUCTURAL_KEYS if "." not in g])

    sections.append("Structural keys by domain")
    sections.append("=" * len("Structural keys by domain"))
    sections.append("")

    for group in top_groups:
        keys = _GROUP_STRUCTURAL_KEYS[group]
        # Filter out internal '*'; show 'all_params' instead
        display_keys = []
        for k in sorted(keys):
            if k == "*":
                display_keys.append(
                    (
                        "all_params",
                        STRUCTURAL_KEY_DOCS.get(
                            "all_params", "(Wildcard: set all parameters FREE/FIXED)"
                        ),
                    )
                )
            else:
                display_keys.append((k, STRUCTURAL_KEY_DOCS.get(k, "")))

        if display_keys:
            sections.append("")
            sections.append(group)
            sections.append("-" * len(group))
            sections.append("")
            for key_name, key_doc in display_keys:
                sections.append(f"- ``'{key_name}'`` — {key_doc}")

    return "\n".join(sections)


def _write_parameter_tables(output_dir: pathlib.Path) -> None:
    """Write parameter tables to docs/_generated/parameter_tables.rst."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # For now, generate a placeholder that will be expanded later if needed
    # The structural keys are the main reference; per-parameter details live in
    # the component docstrings (and can be auto-extracted via Sphinx autodoc).
    content = f"""Parameter reference
====================

This page documents the configurable structure of the model grammar.
For per-component parameter details (defaults, units, descriptions),
see :doc:`components` or call ``tengri.describe(component_name)``.

{structural_key_reference()}

Round-trip serialization
========================

A model's configuration can be inspected and edited via serialization:

.. code-block:: python

    model = SEDModel.build(...)
    config = model.spec.to_groups()  # dict with all groups and parameters
    model.spec.summary()             # print with provenance tags

For per-parameter documentation, see the component reference or:

.. code-block:: python

    import tengri
    tengri.describe("cue")     # nebular backend "Cue"
    tengri.describe("calzetti")  # dust law "Calzetti 2000"
"""

    output_file = output_dir / "parameter_tables.rst"
    output_file.write_text(content, encoding="utf-8")
    print(f"Wrote {output_file}")


def main() -> int:
    """Generate parameter tables and write to docs/_generated/."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    output_dir = repo_root / "docs" / "_generated"

    try:
        _write_parameter_tables(output_dir)
        return 0
    except Exception as e:
        print(f"Error generating parameter tables: {e}", flush=True)
        raise


if __name__ == "__main__":
    exit(main())
