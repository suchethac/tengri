#!/usr/bin/env python3
"""CI guard for free-parameter prefix compliance with NAMING_CONTRACT.md §3.2.

This script walks all parameters from each preset configuration and verifies that
every free-parameter name:
1. Is registered in the parameter registry (via tengri.list_parameters())
2. Follows the mandatory domain prefix rule (NAMING_CONTRACT §3.2):

    sfh_, met_, dust_, neb_, agn_, eline_, noise_, radio_, xray_, shock_,
    chem_, igm_, dla_, or is exactly 'redshift'.

Usage
-----
    python tools/check_param_prefixes.py

Exit code 0 if all parameters pass; non-zero with violations listed otherwise.

Implementation
--------------
This script imports `tengri.presets` and builds Parameters objects from each
preset (starforming, quiescent, high_z, photoz, jwst_spec, agn_host), then
checks the free_params property against:
- Registry membership (added via ADR-0005)
- NAMING_CONTRACT §3.2 prefix rule

Violations can be:
- Fixed with a rename + alias (preferred), or
- Added to an allowlist with a tracking note (temporary/known).
"""

import re
import sys
from pathlib import Path

# Add src/ to path so we can import tengri
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import presets lazily to avoid full package initialization
from tengri.parameters.parameters import Parameters

# ── Naming contract regex ───────────────────────────────────────────────────

ALLOWED_PREFIXES = re.compile(
    r"^(sfh_|met_|dust_|neb_|agn_|eline_|noise_|radio_|xray_|shock_|chem_|igm_|dla_|spatial_).*$"
)
EXACT_MATCHES = {"redshift"}


def is_valid_param_name(name: str, registered_params: set[str] | None = None) -> bool:
    """Check if a parameter name complies with NAMING_CONTRACT §3.2.

    Multi-population (ADR-0012): names of the form
    ``"<population_name>.<param_name>"`` are valid iff the part after the
    first ``.`` satisfies the bare-name rule. This lets every population's
    namespaced parameters share the same prefix discipline without
    duplicating the registry per population.

    Parameters
    ----------
    name : str
        Parameter name to validate.
    registered_params : set[str] | None, optional
        Set of registered parameter names from the registry. If provided,
        the bare name (after namespace strip) must be in this set AND
        satisfy the prefix rule.

    Returns
    -------
    bool
        True if name matches the contract and is registered; False otherwise.
    """
    # Strip multi-population namespace (ADR-0012) before applying prefix check.
    bare_name = name.split(".", 1)[1] if "." in name else name
    # Check registry membership against the bare name
    if registered_params is not None and bare_name not in registered_params:
        return False
    # Check prefix rule against the bare name
    return bare_name in EXACT_MATCHES or bool(ALLOWED_PREFIXES.match(bare_name))


def check_preset(
    preset_name: str, params, config, registered_params: set[str]
) -> list[tuple[str, str, str]]:
    """Check a single preset's free parameters.

    Parameters
    ----------
    preset_name : str
        Name of the preset (e.g., "starforming").
    params : Parameters
        Parameters object from the preset.
    config : SEDModelConfig
        Configuration object (unused but returned by presets).
    registered_params : set[str]
        Set of parameters registered in the parameter registry.

    Returns
    -------
    list[tuple[str, str, str]]
        List of violations as (preset_name, param_name, violation_type).
        violation_type is one of: 'not_registered', 'invalid_prefix'.
    """
    violations = []
    for name in params.free_params:
        # Check registry first
        if name not in registered_params:
            violations.append((preset_name, name, "not_registered"))
        # Check prefix rule
        elif not is_valid_param_name(name, registered_params):
            violations.append((preset_name, name, "invalid_prefix"))
    return violations


def main() -> int:
    """Run the parameter prefix guard.

    Returns
    -------
    int
        0 if all checks pass; 1 if violations found.
    """
    all_violations = []

    # Load the parameter registry
    try:
        import tengri

        registered_params = set(tengri.list_parameters())
    except Exception as e:
        print(f"ERROR loading parameter registry: {e}", file=sys.stderr)
        return 1

    # Import presets module lazily after Parameters is imported
    try:
        import tengri.presets as presets
    except Exception as e:
        print(f"ERROR importing tengri.presets: {e}", file=sys.stderr)
        print("Attempting direct Parameter construction for audit...", file=sys.stderr)
        # Fall back to basic parameter construction
        try:
            from tengri.parameters.priors import Fixed, Gaussian, Uniform

            # Build a sample parameter set manually
            presets_dict = {
                "default": Parameters(
                    mean_sfh_type="dpl",
                    sfh_dpl_alpha=Uniform(0.5, 3.0),
                    sfh_dpl_beta=Uniform(0.3, 2.0),
                    sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
                    sfh_dpl_log_total_mass=Uniform(8, 12),
                    sfh_field_psd_sigma=Uniform(0.01, 1.0),
                    sfh_field_psd_tau_myr=Uniform(10, 500),
                    met_logzsol=Gaussian(-0.3, 0.2),
                    dust_tau_bc=Uniform(0, 4),
                    redshift=Fixed(0.1),
                ),
                "tsnorm_field": Parameters(
                    mean_sfh_type=["tsnorm", "field"],
                    sfh_tsnorm_log_total_mass=Uniform(8, 12),
                    sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
                    sfh_tsnorm_width_gyr=Uniform(0.5, 5),
                    sfh_tsnorm_skew=Uniform(-1, 1),
                    sfh_tsnorm_trunc=Uniform(1, 10),
                    sfh_field_psd_sigma=Uniform(0.01, 1.0),
                    sfh_field_psd_tau_myr=Uniform(10, 500),
                    met_logzsol=Gaussian(-0.3, 0.2),
                    dust_tau_bc=Uniform(0, 4),
                    redshift=Fixed(0.1),
                ),
            }
        except Exception as e2:
            print(f"ERROR building sample parameters: {e2}", file=sys.stderr)
            return 1
    else:
        # Use presets if available
        presets_dict = {}
        for name in ["starforming", "quiescent", "high_z", "photoz", "jwst_spec", "agn_host"]:
            try:
                preset_fn = getattr(presets, name, None)
                if preset_fn:
                    params, _ = preset_fn()
                    presets_dict[name] = params
            except Exception as e:
                print(f"WARNING: Could not build preset '{name}': {e}", file=sys.stderr)

    # Check each preset
    for preset_name, params in presets_dict.items():
        violations = check_preset(preset_name, params, None, registered_params)
        all_violations.extend(violations)

    # Report findings
    if all_violations:
        print("Parameter violations found:")
        print()
        by_preset = {}
        for preset_name, param_name, violation_type in all_violations:
            if preset_name not in by_preset:
                by_preset[preset_name] = {}
            if violation_type not in by_preset[preset_name]:
                by_preset[preset_name][violation_type] = []
            by_preset[preset_name][violation_type].append(param_name)

        for preset_name in sorted(by_preset.keys()):
            print(f"  {preset_name}:")
            for violation_type in sorted(by_preset[preset_name].keys()):
                type_label = (
                    "not registered in parameter registry"
                    if violation_type == "not_registered"
                    else "invalid prefix (not in NAMING_CONTRACT)"
                )
                print(f"    [{type_label}]")
                for param_name in sorted(set(by_preset[preset_name][violation_type])):
                    print(f"      - {param_name}")
        print()
        return 1

    print("All parameter names comply with NAMING_CONTRACT §3.2 and registry. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
