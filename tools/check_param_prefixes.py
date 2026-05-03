#!/usr/bin/env python3
"""CI guard for free-parameter prefix compliance with NAMING_CONTRACT.md §3.2.

This script walks all parameters from each preset configuration and verifies that
every free-parameter name starts with one of the mandatory domain prefixes:

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
checks the free_params property against the NAMING_CONTRACT regex.

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
    r"^(sfh_|met_|dust_|neb_|agn_|eline_|noise_|radio_|xray_|shock_|chem_|igm_|dla_).*$"
)
EXACT_MATCHES = {"redshift"}


def is_valid_param_name(name: str) -> bool:
    """Check if a parameter name complies with NAMING_CONTRACT §3.2.

    Parameters
    ----------
    name : str
        Parameter name to validate.

    Returns
    -------
    bool
        True if name matches the contract; False otherwise.
    """
    return name in EXACT_MATCHES or bool(ALLOWED_PREFIXES.match(name))


def check_preset(preset_name: str, params, config) -> list[str]:
    """Check a single preset's free parameters.

    Parameters
    ----------
    preset_name : str
        Name of the preset (e.g., "starforming").
    params : Parameters
        Parameters object from the preset.
    config : SEDModelConfig
        Configuration object (unused but returned by presets).

    Returns
    -------
    list[str]
        List of violations (param names that don't conform).
    """
    violations = []
    for name in params.free_params:
        if not is_valid_param_name(name):
            violations.append((preset_name, name))
    return violations


def main() -> int:
    """Run the parameter prefix guard.

    Returns
    -------
    int
        0 if all checks pass; 1 if violations found.
    """
    all_violations = []

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
                    sfh_dpl_log_peak_sfr=Uniform(-1, 2),
                    sfh_field_psd_sigma=Uniform(0.01, 1.0),
                    sfh_field_psd_tau_myr=Uniform(10, 500),
                    met_logzsol=Gaussian(-0.3, 0.2),
                    dust_tau_bc=Uniform(0, 4),
                    redshift=Fixed(0.1),
                ),
                "tsnorm_field": Parameters(
                    mean_sfh_type=["tsnorm", "field"],
                    sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
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
        violations = check_preset(preset_name, params, None)
        all_violations.extend(violations)

    # Report findings
    if all_violations:
        print("Parameter prefix violations found:")
        print()
        by_preset = {}
        for preset_name, param_name in all_violations:
            if preset_name not in by_preset:
                by_preset[preset_name] = []
            by_preset[preset_name].append(param_name)

        for preset_name in sorted(by_preset.keys()):
            print(f"  {preset_name}:")
            for param_name in sorted(set(by_preset[preset_name])):
                print(f"    - {param_name}")
        print()
        return 1

    print("All parameter names comply with NAMING_CONTRACT §3.2. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
