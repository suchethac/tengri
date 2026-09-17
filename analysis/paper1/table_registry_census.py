#!/usr/bin/env python3
"""Registry census generator for Table 1 of the tengri paper.

Uses tengri.list_all() as the single authoritative source.
Generates a LaTeX table and JSON summary of all registered components.

Counting rules:
- Available: status='production' (or tier='primary' for inference)
- Other: all other statuses/tiers except 'none' and pure aliases
- Excluded: 'none' sentinels + pure aliases + refusing entries
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import tengri


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters: _ & % #"""
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


# SFH classification: explicit, verified name sets
SFH_NONPARAMETRIC = {
    "bursty_continuity",
    "continuity",
    "continuity_flex",
    "dense_basis",
    "dense_basis_pure",
    "dirichlet",
    "prospector_beta",
}
SFH_STOCHASTIC = {"field", "field_sfr_noise"}

# Aliases: name -> canonical
ALIASES = {
    "salim_sbl18": "salim",
    "const_exp": "constant_then_exponential",
    "mbb": "modified_blackbody",
    "dl07_tabulated": "dl07",
    "db": "dense_basis",
    "dbp": "dense_basis_pure",
    "psb": "psb_wild2020",
}

# Sentinels to exclude
SENTINELS = {"none"}

# Refusing entries (with footnote)
REFUSING_ENTRIES = {"mappings_photo_stellar", "mappings_photo_agn"}


def count_by_status(table, status_filter="production"):
    """Count entries by status, excluding sentinels and aliases."""
    available = 0
    other = 0
    excluded = 0

    for entry in table:
        name = entry.get("name")
        if name in SENTINELS or name in ALIASES:
            excluded += 1
            continue
        if name in REFUSING_ENTRIES:
            excluded += 1
            continue

        status = entry.get("status")
        if status == status_filter:
            available += 1
        else:
            other += 1

    return available, other, excluded, len(table)


def count_by_tier(table):
    """Count inference methods by tier."""
    available = 0
    other = 0
    excluded = 0

    for entry in table:
        tier = entry.get("tier")
        if tier == "primary":
            available += 1
        elif tier == "broken":
            excluded += 1
        else:
            other += 1

    return available, other, excluded, len(table)


def classify_sfh(name):
    """Classify SFH: 'parametric', 'nonparametric', 'stochastic', or 'alias'."""
    if name in ALIASES:
        return "alias"
    if name in SFH_NONPARAMETRIC:
        return "nonparametric"
    if name in SFH_STOCHASTIC:
        return "stochastic"
    return "parametric"


def generate_latex_table(census, git_sha_short, date_str):
    """Generate LaTeX table for the paper."""
    lines = [
        r"\begin{table*}",
        (
            f"\\caption{{Registry census generated from the tengri registries at commit "
            f"{git_sha_short} ({date_str}). Available = production status (inference: primary tier); "
            f"Other = experimental or unvalidated; aliases and the \\texttt{{none}} sentinel "
            f"are excluded. The generated component reference in the documentation "
            f"lists every entry.}}"
        ),
        r"\label{tab:registry_census}",
        r"\begin{tabular}{lrrll}",
        r"\toprule",
        r"Component block & Available & Other & Build key & Example \\",
        r"\midrule",
    ]

    # SSP grids
    ssp = census["ssp"]
    lines.append(
        rf"SSP grid format & {ssp['available']} & {ssp['other']} "
        rf"& \texttt{{(shipped)}} & \texttt{{{latex_escape(ssp['example'])}}} \\"
    )

    # SFH
    for key, label in [
        ("sfh_parametric", "SFH -- parametric"),
        ("sfh_nonparametric", "SFH -- nonparametric"),
        ("sfh_stochastic", "SFH -- stochastic (field)"),
    ]:
        data = census[key]
        lines.append(
            rf"{label} & {data['available']} & {data['other']} "
            rf"& \texttt{{{latex_escape(data['build_key'])}}} & \texttt{{{latex_escape(data['example'])}}} \\"
        )

    # Metallicity
    met = census["metallicity"]
    lines.append(
        rf"Metallicity modes & {met['available']} & {met['other']} "
        rf"& \texttt{{{latex_escape(met['build_key'])}}} & \texttt{{{latex_escape(met['example'])}}} \\"
    )

    # Dust
    for key, label in [
        ("dust_attenuation", "Dust attenuation laws"),
        ("dust_emission", "Dust emission models"),
        (
            "dust_geometries",
            "Dust attenuation structure (screen / two-component / WG00 radiative transfer)",
        ),
    ]:
        data = census[key]
        build_key = latex_escape(data.get("build_key", "type="))
        example = latex_escape(data["example"])
        lines.append(
            rf"{label} & {data['available']} & {data['other']} "
            rf"& \texttt{{{build_key}}} & \texttt{{{example}}} \\"
        )

    # Nebular
    neb = census["nebular"]
    lines.append(
        rf"Nebular emission (photoionization) & {neb['available']} & {neb['other']} "
        rf"& \texttt{{{latex_escape(neb['build_key'])}}} & \texttt{{{latex_escape(neb['example'])}}} \\"
    )

    # Shock
    shock = census["shock"]
    lines.append(
        rf"Shock (MAPPINGS V) + DIG mixing & {shock['available']} & {shock['other']} "
        rf"& \texttt{{{latex_escape(shock['build_key'])}}} & \texttt{{{latex_escape(shock['example'])}}} \\"
    )

    # AGN
    for key, label in [
        ("agn_disc", "AGN disc"),
        ("agn_torus", "AGN torus"),
        ("agn_nlr", "AGN NLR"),
        ("agn_blr", "AGN BLR"),
        ("agn_feii", "AGN Fe II"),
        ("agn_atten", "AGN attenuation"),
    ]:
        data = census[key]
        lines.append(
            rf"{label} & {data['available']} & {data['other']} "
            rf"& \texttt{{{latex_escape(data['build_key'])}}} & \texttt{{{latex_escape(data['example'])}}} \\"
        )

    # Radio, X-ray, IGM
    for key, label in [
        ("radio", "Radio"),
        ("xray", "X-ray"),
        ("igm", "IGM"),
    ]:
        data = census[key]
        lines.append(
            rf"{label} & {data['available']} & {data['other']} "
            rf"& \texttt{{{latex_escape(data['build_key'])}}} & \texttt{{{latex_escape(data['example'])}}} \\"
        )

    # Observation
    obs_phot = census["observation_photometry"]
    lines.append(
        rf"Photometry: bundled filter curves + custom & {obs_phot['available']} "
        rf"& {obs_phot['other']} & \texttt{{(observation=)}} & \texttt{{photometry}} \\"
    )

    obs_spec = census["observation_spectroscopy"]
    lines.append(
        rf"Spectroscopy: LSF + calibration polynomial & {obs_spec['available']} "
        rf"& {obs_spec['other']} & \texttt{{(observation=)}} & \texttt{{spectroscopy}} \\"
    )

    obs_lines = census["observation_emission_lines"]
    lines.append(
        rf"Emission-line fluxes (rest \& obs frame) & {obs_lines['available']} "
        rf"& {obs_lines['other']} & \texttt{{(observation=)}} & \texttt{{line\_list}} \\"
    )

    # Inference
    inference = census["inference"]
    lines.append(
        rf"Inference backends (primary, experimental, broken) & {inference['available']} "
        rf"& {inference['other']} & \texttt{{method=}} & \texttt{{mcmc\_nuts}} \\"
    )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate registry census for Table 1.")
    parser.add_argument("--verbose", action="store_true", help="Print all entries")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent, help="Output directory"
    )
    args = parser.parse_args()

    verbose = args.verbose
    output_dir = args.output_dir

    if verbose:
        print("=== Registry Census from tengri.list_all() ===\n")

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"

    all_data = tengri.list_all()
    census = {"_git_sha_short": git_sha[:7], "_date_str": datetime.now().strftime("%Y-%m-%d")}
    all_tables = {}

    # SSP grids
    known_ssps = all_data.get("known_ssps", [])
    libraries = set()
    for ssp in known_ssps:
        lib = ssp["name"].split("_")[0]
        libraries.add(lib)
    if verbose:
        print(f"known_ssps: {len(libraries)} libraries shipped in {len(known_ssps)} grids\n")
    census["ssp"] = {
        "available": len(libraries),
        "other": 0,
        "build_key": "(shipped)",
        "example": f"{len(libraries)} libraries",
    }
    all_tables["known_ssps"] = known_ssps

    # Dust laws
    dust_laws = all_data.get("dust_laws", [])
    avail, other, excl, total = count_by_status(dust_laws)
    if verbose:
        print("dust_laws:")
        for d in dust_laws:
            name = d["name"]
            if name in ALIASES:
                print(f"  {name:40} -> {ALIASES[name]:30} [alias]")
            else:
                print(f"  {name:40} -> {name}")
        # Print use string for one entry
        dust_law_use = next((d.get("use") for d in dust_laws if d.get("name") == "calzetti"), None)
        if dust_law_use:
            print(f"  dust_laws use string (calzetti): {dust_law_use}")
        print(f"dust_laws: available={avail} other={other} excluded={excl} of {total} rows\n")
    census["dust_attenuation"] = {
        "available": avail,
        "other": other,
        "build_key": "dust_attenuation['law']",
        "example": "calzetti",
    }
    all_tables["dust_laws"] = dust_laws

    # Dust emission
    dust_emission = all_data.get("dust_emission_models", [])
    avail, other, excl, total = count_by_status(dust_emission)
    if verbose:
        print("dust_emission_models:")
        for d in dust_emission:
            name = d["name"]
            status = d.get("status", "unknown")
            if name in ALIASES:
                print(f"  {name:40} -> {ALIASES[name]:30} [alias] [{status}]")
            else:
                print(f"  {name:40} -> {name} [{status}]")
        # Print use string for one entry
        dust_emis_use = next(
            (d.get("use") for d in dust_emission if d.get("name") == "dl07"), None
        )
        if dust_emis_use:
            print(f"  dust_emission use string (dl07): {dust_emis_use}")
        print(
            f"dust_emission_models: available={avail} other={other} excluded={excl} of {total} rows\n"
        )
    census["dust_emission"] = {
        "available": avail,
        "other": other,
        "build_key": "dust_emission['type']",
        "example": "dl07",
    }
    all_tables["dust_emission_models"] = dust_emission

    # Dust geometries (dust_models)
    dust_models = all_data.get("dust_models", [])
    avail, other, excl, total = count_by_status(dust_models)
    if verbose:
        print("dust_models (geometries):")
        for d in dust_models:
            name = d["name"]
            use_str = d.get("use", "")
            print(f"  {name:40} [production]")
            if use_str:
                print(f"    use: {use_str}")
        print(f"dust_models: available={avail} other={other} excluded={excl} of {total} rows\n")
    census["dust_geometries"] = {
        "available": avail,
        "other": other,
        "build_key": "dust_attenuation['type']",
        "example": "two_component",
    }
    all_tables["dust_models"] = dust_models

    # Nebular backends
    nebular = all_data.get("nebular_backends", [])
    avail, other, excl, total = count_by_status(nebular)
    refusing_neb = []
    if verbose:
        print("nebular_backends:")
        for n in nebular:
            name = n["name"]
            status = n.get("status", "unknown")
            if name in SENTINELS:
                print(f"  {name:40} -> EXCLUDED (sentinel)")
            elif name in REFUSING_ENTRIES:
                print(f"  {name:40} -> EXCLUDED (refuses) [{status}]")
                refusing_neb.append(name)
            else:
                print(f"  {name:40} -> {name} [{status}]")
        print(
            f"nebular_backends: available={avail} other={other} excluded={excl} of {total} rows\n"
        )
    if verbose and nebular:
        neb_entry = next((n for n in nebular if n.get("name") == "cue"), None)
        if neb_entry:
            print(f"  nebular use string (cue): {neb_entry.get('use', 'N/A')}\n")

    census["nebular"] = {
        "available": avail,
        "other": other,
        "build_key": "neb['type']",
        "example": "cue",
    }
    all_tables["nebular_backends"] = nebular

    # Shock models
    shock_models = all_data.get("shock_models", [])
    avail, other, excl, total = count_by_status(shock_models)
    if verbose:
        print("shock_models:")
        for s in shock_models:
            name = s["name"]
            status = s.get("status", "unknown")
            if name in SENTINELS:
                print(f"  {name:40} -> EXCLUDED (sentinel)")
            else:
                print(f"  {name:40} -> {name} [{status}]")
        print(f"shock_models: available={avail} other={other} excluded={excl} of {total} rows\n")
    if verbose and shock_models:
        shock_entry = next((s for s in shock_models if s.get("name") == "mappings"), None)
        if shock_entry:
            print(f"  shock use string (mappings): {shock_entry.get('use', 'N/A')}\n")

    census["shock"] = {
        "available": avail,
        "other": other,
        "build_key": "shock['type']",
        "example": "mappings",
    }
    all_tables["shock_models"] = shock_models

    # Metallicity modes
    met_modes = all_data.get("metallicity_modes", [])
    avail, other, excl, total = count_by_status(met_modes)
    if verbose:
        print(
            f"metallicity_modes: available={avail} other={other} excluded={excl} of {total} rows\n"
        )
        # Print use string for delta (the default)
        delta_entry = next((m for m in met_modes if m.get("name") == "delta"), None)
        if delta_entry:
            print(f"  metallicity use string (delta/default): {delta_entry.get('use', 'N/A')}\n")
    census["metallicity"] = {
        "available": avail,
        "other": other,
        "build_key": "met['type']",
        "example": "delta",
    }
    all_tables["metallicity_modes"] = met_modes

    # SFH models
    sfh_models = all_data.get("sfh_models", [])
    sfh_para, sfh_npara, sfh_stoch = [], [], []
    sfh_alias_found = []

    for m in sfh_models:
        name = m["name"]
        sfh_type = classify_sfh(name)

        if verbose and sfh_type == "alias":
            print(f"sfh_models: {name:40} -> {ALIASES[name]:30} [alias] [{m.get('status')}]")

        if sfh_type == "nonparametric":
            sfh_npara.append(m)
        elif sfh_type == "stochastic":
            sfh_stoch.append(m)
        elif sfh_type != "alias":
            sfh_para.append(m)
        else:
            sfh_alias_found.append((name, ALIASES[name]))

    para_avail = len([m for m in sfh_para if m.get("status") == "production"])
    para_other = len([m for m in sfh_para if m.get("status") != "production"])
    npara_avail = len([m for m in sfh_npara if m.get("status") == "production"])
    npara_other = len([m for m in sfh_npara if m.get("status") != "production"])
    stoch_avail = len([m for m in sfh_stoch if m.get("status") == "production"])
    stoch_other = len([m for m in sfh_stoch if m.get("status") != "production"])

    if verbose:
        print(f"sfh_models (parametric): available={para_avail} other={para_other}")
        print(f"sfh_models (nonparametric): available={npara_avail} other={npara_other}")
        print(f"sfh_models (stochastic): available={stoch_avail} other={stoch_other}")
        print(f"sfh_models (aliases): {len(sfh_alias_found)}")
        print(
            f"sfh_models: available={para_avail + npara_avail + stoch_avail} "
            f"other={para_other + npara_other + stoch_other} "
            f"excluded={len(sfh_alias_found)} of {len(sfh_models)} rows\n"
        )

    if verbose and sfh_models:
        first_sfh = sfh_models[0]
        print(f"  sfh use string (first): {first_sfh.get('use', 'N/A')}\n")

    census["sfh_parametric"] = {
        "available": para_avail,
        "other": para_other,
        "build_key": "sfh['type']",
        "example": "dpl",
    }
    census["sfh_nonparametric"] = {
        "available": npara_avail,
        "other": npara_other,
        "build_key": "sfh['type']",
        "example": "continuity",
    }
    census["sfh_stochastic"] = {
        "available": stoch_avail,
        "other": stoch_other,
        "build_key": "sfh['type']",
        "example": "field",
    }
    all_tables["sfh_models"] = sfh_models

    # AGN blocks (by category)
    agn_blocks = all_data.get("agn_blocks", [])
    agn_cats = {}
    for block in agn_blocks:
        name = block["name"]
        cat = block.get("category")
        status = block.get("status", "unknown")

        if name not in SENTINELS:
            if cat not in agn_cats:
                agn_cats[cat] = {"available": [], "other": []}

            if status == "production":
                agn_cats[cat]["available"].append(name)
            else:
                agn_cats[cat]["other"].append(name)

            if verbose:
                print(f"agn_blocks: {name:40} ({cat}) [{status}]")

    if verbose:
        for cat in sorted(agn_cats.keys()):
            avail = len(agn_cats[cat]["available"])
            other = len(agn_cats[cat]["other"])
            excl = 1 if cat in agn_cats else 0  # 'none' sentinel
            total = avail + other + excl
            print(
                f"agn_blocks[{cat}]: available={avail} other={other} excluded={excl} of {total} rows"
            )
        print()

    # Print use strings for AGN blocks
    if verbose:
        agn_print_cats = {
            "disc": "disc",
            "torus": "torus",
            "nlr": "nlr",
            "blr": "blr",
            "feii": "feii",
            "attenuation": "atten",
        }
        for cat_key in agn_print_cats:
            first_block = next(
                (
                    b
                    for b in agn_blocks
                    if b.get("category") == cat_key and b.get("status") == "production"
                ),
                None,
            )
            if first_block:
                print(
                    f"  agn_{cat_key} use string ({first_block.get('name')}): {first_block.get('use', 'N/A')}"
                )
        print()

    # AGN attenuation uses 'atten' not 'attenuation' (verified from registry)
    for cat_key, cat_label, build_key in [
        ("disc", "agn_disc", "agn.disc="),
        ("torus", "agn_torus", "agn.torus="),
        ("nlr", "agn_nlr", "agn.nlr="),
        ("blr", "agn_blr", "agn.blr="),
        ("feii", "agn_feii", "agn.feii="),
        ("attenuation", "agn_atten", "agn.atten="),
    ]:
        data = agn_cats.get(cat_key, {"available": [], "other": []})
        # Pick sensible examples: skirtor for disc/torus, first production for others
        if cat_key == "disc" or cat_key == "torus":
            example = "skirtor"
        else:
            example = data["available"][0] if data["available"] else "none"

        census[cat_label] = {
            "available": len(data["available"]),
            "other": len(data["other"]),
            "build_key": build_key,
            "example": example,
        }

    all_tables["agn_blocks"] = agn_blocks

    # Radio, X-ray, IGM
    for table_key, label, build_key in [
        ("radio_models", "radio", "radio="),
        ("xray_models", "xray", "xray="),
        ("igm_models", "igm", "igm="),
    ]:
        table = all_data.get(table_key, [])
        avail, other, excl, total = count_by_status(table)
        if verbose:
            print(
                f"{table_key}: available={avail} other={other} excluded={excl} of {total} rows\n"
            )
            # Print use strings
            for entry in table:
                if entry.get("status") == "production" and entry.get("name") != "none":
                    print(
                        f"  {table_key} use string ({entry.get('name')}): {entry.get('use', 'N/A')}"
                    )
                    break  # Just print the first one
            if table_key == "xray_models":
                yang20 = next((e for e in table if e.get("name") == "yang20"), None)
                if yang20:
                    print(
                        f"  (Note: yang20 is Yang+2020 key, listed as: {yang20.get('short_doc', '')})"
                    )
            if table_key == "igm_models":
                inoue14 = next((e for e in table if e.get("name") == "inoue14"), None)
                if inoue14:
                    print(
                        f"  (Note: inoue14 is Inoue+2014, listed as: {inoue14.get('short_doc', '')})"
                    )
            print()

        prod_entries = [e["name"] for e in table if e.get("status") == "production"]

        # Pick examples: condon92 for radio, yang20 for xray, inoue14 for igm
        if label == "radio":
            example = "condon92"
        elif label == "xray":
            # Look for Yang+2020 key (yang20 or simple)
            if "yang20" in prod_entries:
                example = "yang20"
            elif "simple" in prod_entries:
                example = "simple"
            else:
                example = next(iter(prod_entries)) if prod_entries else "none"
        elif label == "igm":
            # Look for Inoue+2014 key
            if "inoue14" in prod_entries:
                example = "inoue14"
            else:
                example = next(iter(prod_entries)) if prod_entries else "none"
        else:
            example = next(iter(prod_entries)) if prod_entries else "none"

        census[label] = {
            "available": avail,
            "other": other,
            "build_key": build_key,
            "example": example,
        }
        all_tables[table_key] = table

    # Observation channels
    filters = all_data.get("filters", [])
    if verbose:
        print(f"filters: {len(filters)} rows\n")

    census["observation_photometry"] = {
        "available": len(filters),
        "other": 0,
        "build_key": "(observation=)",
        "example": "photometry",
    }
    all_tables["filters"] = filters

    census["observation_spectroscopy"] = {
        "available": 1,
        "other": 0,
        "build_key": "(observation=)",
        "example": "spectroscopy",
    }
    census["observation_emission_lines"] = {
        "available": 1,
        "other": 0,
        "build_key": "(observation=)",
        "example": "emission_line_fluxes",
    }

    # Inference methods (by tier)
    inference = all_data.get("inference_methods", [])
    avail, other, excl, total = count_by_tier(inference)
    primary_names = [m["name"] for m in inference if m.get("tier") == "primary"]
    broken_names = [m["name"] for m in inference if m.get("tier") == "broken"]

    if verbose:
        print("inference_methods:")
        for m in inference:
            name = m["name"]
            tier = m.get("tier", "unknown")
            print(f"  {name:40} [{tier}]")
        # Print use string for one entry
        first_primary = next((m for m in inference if m.get("tier") == "primary"), None)
        if first_primary:
            print(
                f"  inference use string ({first_primary.get('name')}): {first_primary.get('use', 'N/A')}"
            )
        print(
            f"inference_methods: available={avail} other={other} excluded={excl} of {total} rows\n"
        )

    census["inference"] = {
        "available": avail,
        "other": other,
        "build_key": "method=",
        "example": primary_names[0] if primary_names else "none",
    }
    all_tables["inference_methods"] = inference

    # Extract values for LaTeX generation
    git_sha_short = census.pop("_git_sha_short")
    date_str = census.pop("_date_str")

    # Generate LaTeX
    latex_table = generate_latex_table(census, git_sha_short, date_str)

    # Write LaTeX
    latex_file = output_dir / "tables" / "table_registry_census.tex"
    latex_file.parent.mkdir(parents=True, exist_ok=True)
    latex_file.write_text(latex_table)
    print(f"Written LaTeX table to {latex_file}")

    # Write JSON
    results = {
        "date": datetime.now().isoformat(),
        "git_sha": git_sha,
        "census": census,
        "alias_map": ALIASES,
        "sfh_classifications": {
            "parametric": "All SFH not in nonparametric/stochastic",
            "nonparametric": list(SFH_NONPARAMETRIC),
            "stochastic": list(SFH_STOCHASTIC),
        },
        "refusing_entries": list(REFUSING_ENTRIES),
        "all_registries": all_tables,
    }

    json_file = output_dir / "results" / "registry_census.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"Written JSON results to {json_file}")

    # Print LaTeX
    print("\n" + "=" * 80)
    print("LATEX TABLE:")
    print("=" * 80)
    print(latex_table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
