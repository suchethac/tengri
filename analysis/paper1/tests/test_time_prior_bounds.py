# SPDX-License-Identifier: BSD-3-Clause
"""Test that all time-like priors respect cosmic age at sample redshift.

Validates that parameters with time-like units (tau_gyr, age_gyr, peak_gyr, etc.)
never have priors whose upper bounds exceed the age of the universe at the sample's
maximum redshift. Such unbounded turnover times or positions render the parameter
unobservable and can degrade sampler conditioning.

See sfh_tau_conditioning.py for the measurement that motivates this constraint.
"""

from __future__ import annotations

import pytest

from analysis.paper1 import configs
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.unit


# The CANDELS sample spans z=1.012 to z=1.097; use the max as the binding constraint.
Z_SAMPLE_MAX = 1.097
AGE_AT_Z_MAX = age_at_z(Z_SAMPLE_MAX)


#: Stellar libraries this file may use for parameter extraction, most preferred
#: first. Both are tracked in the repository, so the guards run from a fresh
#: clone. The choice does not affect any bound this file checks.
_PROBE_LIBRARIES = ("fsps_prsc_miles_chabrier", "bpss_stars_c3k_a_chabrier")


def _available_probe_library() -> str:
    """The first tracked probe library that actually loads, or skip.

    Skipping here is honest: no tracked library present means the environment
    cannot build any configuration, which is different from a guard failing.
    """
    import tengri

    for name in _PROBE_LIBRARIES:
        try:
            tengri.load_ssp(name)
        except Exception:
            continue
        return name
    pytest.skip(f"no tracked stellar library available; tried {list(_PROBE_LIBRARIES)}")
    raise AssertionError("unreachable")  # pragma: no cover


def _extract_time_bounds(config_builder, z: float) -> dict[str, tuple[float, float]]:
    """Extract all time-like prior bounds from a configuration.

    Iterates over the SEDModel build arguments to find parameters with 'gyr'
    in their name (excluding 'bin_edges_gyr', which is handled separately).

    Parameters
    ----------
    config_builder : callable
        A config function like config_I, config_II, etc.
    z : float
        Redshift for the configuration (used to resolve age_at_z).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from parameter name to (lower, upper) bounds.
    """
    import tengri

    # Any stellar library will do for THIS question. The bounds being checked
    # are time-like (tau, age, peak); the only prior that reads the library is
    # met_logzsol, which this test ignores. So the library is chosen for
    # availability rather than for fidelity to each configuration.
    #
    # It must be one that is TRACKED IN THE REPOSITORY, or this whole file can
    # only run on a machine that happens to have the grid. Of the five grids the
    # six configurations name, only fsps_prsc_miles_chabrier and
    # bpss_stars_c3k_a_chabrier are tracked; the previous choice here,
    # fsps_mist_c3k_a_chabrier, is not, so a fresh clone could not run any of
    # these guards.
    ssp = tengri.load_ssp(_available_probe_library())
    obs = tengri.Observation(photometry=tengri.Photometry.from_names(["hst_f160w"]))

    # Build model to inspect the spec
    try:
        model = config_builder(ssp, obs, z)
    except Exception:
        # Configuration may fail if required grids are missing (e.g., Cloudy grid).
        # Skip extraction if we cannot build it.
        return {}

    time_bounds = {}

    # Get the spec's parameters
    spec = model.spec

    # Iterate over all parameters and extract those with time-like units
    for param_name in spec.all_params:
        # Skip parameters that don't suggest time semantics
        if "gyr" not in param_name.lower() and "yr" not in param_name.lower():
            continue

        # Skip bin_edges which are arrays, not scalar priors
        if param_name.endswith("_gyr") and "bin_edges" in param_name:
            continue

        # Get the distribution for this parameter
        try:
            param_dist = spec.get_distribution(param_name)
        except (KeyError, AttributeError):
            continue

        # Extract bounds for Uniform priors (the common case)
        if hasattr(param_dist, "lo") and hasattr(param_dist, "hi"):
            time_bounds[param_name] = (float(param_dist.lo), float(param_dist.hi))

    return time_bounds


#: Time-like priors that may exceed the cosmic age, each with the measurement
#: that earned the exemption. The bound is not "is this below the cosmic age"
#: but "does this direction go flat once it leaves the galaxy's history" -- and
#: that is not knowable from a parameter's name, so it is recorded here.
#:
#: Ratio below is max |dF/F| across the CANDELS bands with the parameter inside
#: the age window over the same outside it, from analysis/paper1 probes:
#:
#:   dpl tau (turnover)      467 - 121000   FLAT outside -> capped, not exempt
#:   delayed tau (e-folding)      64 - 79   well conditioned -> exempt
#:   lnorm peak (mode)          2.96 - 3.26 best conditioned of the three -> exempt
MEASURED_EXEMPT = {
    # An e-folding time keeps shaping the history at any value: 3.9-4.4% band
    # sensitivity outside the window, ratio 64-79.
    ("III", "sfh_delayed_tau_gyr"),
    # A log-normal peak beyond the observed epoch is not a flat direction -- it
    # encodes a star formation rate still RISING there, which the data
    # distinguish: 0.48-2.70% outside, ratio 2.96-3.26. Capping would delete
    # rising-SFH solutions the fit can constrain.
    ("V", "sfh_lnorm_peak_gyr"),
}


@pytest.mark.parametrize("config_key", configs.CONFIG_KEYS)
def test_time_prior_bounds_respect_cosmic_age(config_key):
    """Test that time-like priors in a configuration don't exceed cosmic age at z_max.

    Each configuration is built at the sample's maximum redshift. All discovered
    time-like priors (tau_gyr, age_gyr, peak_gyr, etc.) must have upper bounds
    <= the cosmic age at that redshift.

    Parameters listed in ``MEASURED_EXEMPT`` are allowed to exceed the cosmic
    age. Each entry carries the measurement that earned it; an exemption without
    one is a silenced failure rather than a decision, so do not add a name here
    without a number beside it.

    Parameters
    ----------
    config_key : str
        Configuration key (I, II, III, IV, V, VI).
    """
    config_builder = getattr(configs, f"config_{config_key}")

    time_bounds = _extract_time_bounds(config_builder, Z_SAMPLE_MAX)

    # If no time-like priors were found, skip with a note (may happen if
    # a configuration uses only fixed parameters).
    if not time_bounds:
        pytest.skip(f"Config {config_key}: no time-like priors extracted")

    # Check each time-like prior
    violations = []
    for param_name, (lo, hi) in sorted(time_bounds.items()):
        if (config_key, param_name) in MEASURED_EXEMPT:
            continue

        if hi > AGE_AT_Z_MAX:
            violations.append(
                f"{param_name}: [{lo:.3f}, {hi:.3f}] exceeds cosmic age "
                f"{AGE_AT_Z_MAX:.3f} Gyr at z={Z_SAMPLE_MAX:.3f}"
            )

    assert not violations, (
        f"Config {config_key} has time-like priors with bounds exceeding cosmic age:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_extractor_actually_finds_the_known_time_priors():
    """The guard above must not pass by extracting nothing.

    ``test_time_prior_bounds_respect_cosmic_age`` skips a configuration when no
    time-like prior is discovered, which is the honest outcome for the rows
    whose star formation history is a set of bin edges rather than scalar
    timescales. But a broken extractor produces exactly the same outcome for
    EVERY row, and the suite then reports skips and green -- an absence of
    results reading as a pass.

    This pins the configurations that are known to carry scalar time-like
    priors, so the extractor going silent is a failure rather than a skip.
    """
    expected = {
        "II": "sfh_dpl_tau_gyr",
        "III": "sfh_delayed_tau_gyr",
        "V": "sfh_lnorm_peak_gyr",
    }
    missing = []
    for config_key, param_name in expected.items():
        builder = getattr(configs, f"config_{config_key}")
        found = _extract_time_bounds(builder, Z_SAMPLE_MAX)
        if param_name not in found:
            missing.append(f"config {config_key}: expected {param_name}, got {sorted(found)}")
    assert not missing, (
        "the time-prior extractor found nothing where it should have:\n"
        + "\n".join(f"  {m}" for m in missing)
    )
