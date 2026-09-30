# SPDX-License-Identifier: BSD-3-Clause
"""Every SFH family's support is bounded to [0, age(z)]; mass is conserved (#2521).

Before this fix, ``_renormalize_to_mass`` (mean_sfh.py) rescaled each family's
SFH shape to ``10**log_total_mass`` by integrating over the FULL SSP age grid
(effectively [0, ~13.8 Gyr], today's cosmic age), independent of the source
redshift. A runtime clamp added for #683 then zeroed SFR at lookback ages
older than ``age_at_z(z)`` -- AFTER that normalization -- so whatever the
clamp removed was simply lost: formed mass fell below ``10**log_total_mass``
with no warning past the eager-only ``SFHBeforeBigBangWarning`` threshold.

This test enumerates every onset/age/peak-time-bearing family in
``SFH_REGISTRY`` (the same 21 families ``sweep_S3_report.md`` Sec. 4
identifies) x z in {0.5, 2.5, 6.0} x onset at {0.5, 1.2, 2.0} x age(z): 189
cells. Before the fix, 149/189 fail (measured in the sweep); the physical fix
makes formed mass equal the declared ``10**log_total_mass`` BY CONSTRUCTION
at every cell, and the SSP-age-bin mass distribution is exactly zero for any
bin older than ``age_at_z(z)``.

The mass-conservation correction (``_mass_conserving_total``, component.py)
sits at the ONE point both SSP age kernels' total_mass converges to, so
``age_kernel`` is swept alongside the onset grid: the histogram ("dsps")
kernel used by every GP-field build, and any explicit ``age_kernel='dsps'``
choice, must conserve formed mass exactly the same way the default
cloud-in-cell ("cic") kernel does. A composite (list) ``sfh_model`` sums
multiple additive members' shapes into one array before
``_mass_conserving_total`` ever sees it, so the correction can only restore
the AGGREGATE mass (the sum of every member's ``sfh_*_log_total_mass``), not
each member's own share individually; this per-family sweep therefore covers
only single-family builds. ``test_composite_sfh_mass_conserved_in_aggregate``
covers the composite case, pinning aggregate-exact conservation instead.

References
----------
.. [1] Issue #2521: "z-cap all SFH onset/age/peak-time parameters"
.. [2] Issue #683: the original before-Big-Bang runtime clamp
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import Fixed

pytestmark = pytest.mark.conservation


def _log_total_mass_public_name(family: str) -> str:
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    spec = SFH_REGISTRY[family]
    for pub, (internal, _scale, _offset) in spec.internal_param_map.items():
        if internal == "log_total_mass":
            return pub
    raise AssertionError(f"family {family!r} has no log_total_mass parameter")


# Family -> its public onset/age/peak-time parameter name (sweep_S3_report.md Sec. 4).
ONSET_PARAMS_BY_FAMILY = {
    "const": "sfh_const_start_gyr",
    "dexp": "sfh_dexp_start_gyr",
    "exp": "sfh_exp_start_gyr",
    "declining_exp": "sfh_declining_exp_age_gyr",
    "delayed": "sfh_delayed_age_gyr",
    "delayed_bq": "sfh_delayed_bq_age_main_gyr",
    "dpl": "sfh_dpl_age_gyr",
    "dpl_lookback": "sfh_dpl_lookback_age_gyr",
    "lnorm": "sfh_lnorm_age_gyr",
    "norm": "sfh_norm_peak_lbt_gyr",
    "periodic": "sfh_periodic_age_gyr",
    "psb": "sfh_psb_age_gyr",
    "sfh2exp": "sfh_sfh2exp_age_gyr",
    "snorm": "sfh_snorm_peak_lbt_gyr",
    "snorm_burst": "sfh_snorm_burst_peak_lbt_gyr",
    "trunc_exp": "sfh_trunc_exp_age_gyr",
    "tsnorm": "sfh_tsnorm_peak_lbt_gyr",
    "tsnorm_burst": "sfh_tsnorm_burst_peak_lbt_gyr",
    "const_exp": "sfh_cexp_age_gyr",
    "psb_flex": "sfh_psb_flex_tflex_gyr",
    "psb_suess2022": "sfh_psb2022_tflex_gyr",
}

Z_VALUES = (0.5, 2.5, 6.0)
MULTIPLIERS = (0.5, 1.2, 2.0)
LOG_TOTAL_MASS = 9.0
#: Wide enough to hold 2.0 x age_at_z(0.5) (~17.4 Gyr) plus headroom.
_OVERRIDE_CEIL_GYR = 100.0


#: ``_sfh_type_prefixes`` only ever tries the literal ``sfh_<registry_key>_``
#: candidate, so it misses families whose registry key and internal param
#: prefix genuinely differ (kept for backward compatibility): ``const_exp``'s
#: params are ``sfh_cexp_*``, ``psb_suess2022``'s are ``sfh_psb2022_*``.
_PREFIX_OVERRIDES = {
    "const_exp": "sfh_cexp_",
    "psb_suess2022": "sfh_psb2022_",
}


def _short_key(family, public_name, prefixes):
    override = _PREFIX_OVERRIDES.get(family)
    if override is not None and public_name.startswith(override):
        return public_name[len(override) :]
    prefix = next(p for p in prefixes if public_name.startswith(p))
    return public_name[len(prefix) :]


def _onset_override_floor(family, onset_name):
    """Reuse the registry's own declared floor for this onset param.

    Keeps whatever ordering/bound_check invariant the registry already
    satisfies (e.g. psb_flex/psb_suess2022's tflex_gyr floor must sit at or
    above tlast_gyr's ceiling) while widening the ceiling far enough to sweep
    onset values up to 2x age_at_z(0.5).
    """
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY
    from tengri.parameters.priors import Uniform as _UniformPrior

    pdef = SFH_REGISTRY[family].params[onset_name]
    for candidate in (pdef.free_prior, pdef.default):
        if isinstance(candidate, _UniformPrior):
            return float(candidate.bounds[0])
    return 1e-3


def _build_model(ssp, family, onset_name, z, age_kernel="cic"):
    from tengri import DEFAULT, Fixed, SEDModel, Uniform
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY
    from tengri.parameters.groups import _sfh_type_prefixes

    prefixes = _sfh_type_prefixes(frozenset(SFH_REGISTRY))
    onset_short = _short_key(family, onset_name, prefixes)
    mass_short = _short_key(family, _log_total_mass_public_name(family), prefixes)
    floor = _onset_override_floor(family, onset_name)

    sfh_group = {
        "type": family,
        "all_params": Fixed(DEFAULT),
        onset_short: Uniform(floor, _OVERRIDE_CEIL_GYR),
        mass_short: Fixed(LOG_TOTAL_MASS),
        "age_kernel": age_kernel,
    }
    if family == "delayed_bq":
        # Default age_bq_gyr (0.5 Gyr, the burst/quench onset) can exceed a
        # swept age_main_gyr at the smallest multiplier (0.5 x age(z) at
        # z=6 is 0.467 Gyr) -- a physically nonsensical "quench before
        # formation" combination unrelated to this test's own onset sweep.
        # Free it too so it can be pinned below age_main_gyr per cell.
        bq_name = "sfh_delayed_bq_age_bq_gyr"
        bq_short = _short_key(family, bq_name, prefixes)
        bq_floor = _onset_override_floor(family, bq_name)
        sfh_group[bq_short] = Uniform(bq_floor, _OVERRIDE_CEIL_GYR)

    return SEDModel.build(
        ssp_data=ssp,
        sfh=sfh_group,
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(z),
    )


AGE_KERNELS = ("cic", "dsps")


def _cells():
    for family, onset_name in sorted(ONSET_PARAMS_BY_FAMILY.items()):
        for age_kernel in AGE_KERNELS:
            for z in Z_VALUES:
                for mult in MULTIPLIERS:
                    yield pytest.param(
                        family,
                        onset_name,
                        z,
                        mult,
                        age_kernel,
                        id=f"{family}-{age_kernel}-z{z}-m{mult}",
                    )


@pytest.mark.parametrize(("family", "onset_name", "z", "mult", "age_kernel"), list(_cells()))
def test_formed_mass_conserved_and_sfr_zero_before_big_bang(
    family, onset_name, z, mult, age_kernel, synthetic_ssp_wide
):
    from tengri.utils.cosmology import age_at_z

    age_gyr = float(age_at_z(z))
    onset_value = mult * age_gyr
    model = _build_model(synthetic_ssp_wide, family, onset_name, z, age_kernel=age_kernel)
    params = {onset_name: onset_value}
    if family == "delayed_bq":
        params["sfh_delayed_bq_age_bq_gyr"] = min(0.3, 0.3 * onset_value)

    state = model.predict_state(params)
    joint_weights = np.asarray(state.derived["joint_weights"])
    total_mass = float(np.asarray(state.derived["log_mstar_formed"]))
    total_mass = 10.0**total_mass
    ssp_ages_yr = (10.0 ** np.asarray(model.ssp_data.ssp_lg_age_gyr)) * 1e9

    # (1) formed mass == 10**log_total_mass by construction, on BOTH age
    # kernels: ``_mass_conserving_total`` sits at the one point every
    # kernel's total_mass converges to, after the "cic" and "dsps" branches
    # have already produced their own (possibly z-truncated) integral.
    ratio = total_mass / 10.0**LOG_TOTAL_MASS
    assert ratio == pytest.approx(1.0, abs=1e-4), (
        f"{family} {age_kernel} z={z} mult={mult}: formed mass ratio {ratio} != 1"
    )

    if age_kernel != "cic":
        return

    # (2) zero mass at any SSP age bin older than age(z), except the single
    # bin immediately bracketing age(z): the CIC age kernel (#964) splits
    # each mass parcel between the two SSP nodes bracketing its own age with
    # linear-in-log-age weights, so a parcel at lookback just younger than
    # age(z) legitimately deposits part of its (already age(z)-masked) mass
    # onto the nearest older template node -- a fixed grid-resolution
    # artifact of that interpolation, not lookback-time-uncapped SFR. Any
    # node TWO OR MORE steps older than age(z) can only be reached by a
    # parcel whose own age already exceeds age(z), which the CIC mask
    # (`_cic_parcels`) zeros out before the deposit -- so mass there must be
    # exactly zero. The "dsps" histogram kernel bins by log-midpoint edges
    # instead and is not held to this per-bin contract here -- only to the
    # formed-mass contract in (1).
    age_marginal = joint_weights.sum(axis=0) * total_mass  # (n_age,) Msun
    too_old_idx = np.where((ssp_ages_yr / 1e9) > age_gyr)[0]
    if too_old_idx.size > 1:
        strictly_before_bb = too_old_idx[1:]  # excludes the boundary-adjacent node
        mass_before_bb = float(np.sum(age_marginal[strictly_before_bb]))
        assert mass_before_bb / total_mass < 1e-6, (
            f"{family} {age_kernel} z={z} mult={mult}: {mass_before_bb:.3e} Msun "
            f"({mass_before_bb / total_mass:.2%}) formed strictly before the Big Bang "
            f"(excluding the boundary-adjacent CIC node)"
        )


def test_gp_field_formed_mass_conserved(synthetic_ssp_wide):
    """A GP-field build forces ``age_kernel='dsps'`` (#964); mass must still conserve.

    The field draw lives on the coarse lookback grid with no dense CIC
    integrand, so every field build reaches the histogram ("dsps") kernel
    branch of ``_mass_conserving_total``, not the "cic" one.
    """
    from tengri import DEFAULT, Fixed, SEDModel

    z = 2.5
    from tengri.utils.cosmology import age_at_z

    age_gyr = float(age_at_z(z))
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "age_gyr": Fixed(2.0 * age_gyr),
            "log_total_mass": Fixed(LOG_TOTAL_MASS),
            "field": True,
        },
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(z),
    )
    pred = model.predict({})
    ratio = float(pred.stellar_mass) / 10.0**LOG_TOTAL_MASS
    assert ratio == pytest.approx(1.0, abs=1e-4), f"GP-field z={z}: formed mass ratio {ratio} != 1"


#: Four two-member additive composites, each pairing a family whose onset is
#: fixed to 2 x age(z) (heavily truncated) with one fixed comfortably inside
#: [0, age(z)] (untruncated), so the pooled truncation genuinely differs
#: per member -- the regime ``_mass_conserving_total``'s composite branch
#: pins to the AGGREGATE total, not per member (see its own docstring).
_COMPOSITE_PAIRS: tuple[tuple[str, dict, str, dict], ...] = (
    (
        "dpl",
        {"sfh_dpl_alpha": Fixed(1.5), "sfh_dpl_beta": Fixed(1.0), "sfh_dpl_tau_gyr": Fixed(3.0)},
        "const",
        {"sfh_const_start_gyr": Fixed(1.0), "sfh_const_end_gyr": Fixed(0.0)},
    ),
    (
        "exp",
        {"sfh_exp_tau_gyr": Fixed(2.0)},
        "delayed",
        {"sfh_delayed_tau_gyr": Fixed(1.0), "sfh_delayed_age_gyr": Fixed(0.9)},
    ),
    (
        "declining_exp",
        {"sfh_declining_exp_tau_gyr": Fixed(2.0)},
        "dexp",
        {"sfh_dexp_tau_gyr": Fixed(0.5), "sfh_dexp_start_gyr": Fixed(0.9)},
    ),
    (
        "norm",
        {"sfh_norm_width_gyr": Fixed(0.5)},
        "const",
        {"sfh_const_start_gyr": Fixed(1.0), "sfh_const_end_gyr": Fixed(0.0)},
    ),
)

#: Onset/age/peak-time public param name per family used above, so the
#: truncated member's own boundary can be set to 2 x age(z).
_ONSET_PARAM = {
    "dpl": "sfh_dpl_age_gyr",
    "exp": "sfh_exp_start_gyr",
    "declining_exp": "sfh_declining_exp_age_gyr",
    "norm": "sfh_norm_peak_lbt_gyr",
}


@pytest.mark.parametrize("truncated,extra_t,untruncated,extra_u", _COMPOSITE_PAIRS)
@pytest.mark.parametrize("z", (0.5, 2.5, 6.0))
def test_composite_sfh_mass_conserved_in_aggregate(
    synthetic_ssp_wide, truncated, extra_t, untruncated, extra_u, z
):
    """A composite's pooled formed mass equals the sum of its members' requests.

    ``_mass_conserving_total``'s composite branch (#2521) pins the pooled
    CIC/DSPS-measured mass to the sum of every ``sfh_*_log_total_mass`` key
    present, which is exact for the AGGREGATE total but not necessarily for
    each member individually when members are truncated by different
    fractions (see that function's own docstring for why full per-member
    exactness needs the CIC/DSPS kernel to integrate each additive member
    separately -- a larger change not attempted here). This test pins the
    aggregate guarantee only.
    """
    from tengri import DEFAULT, Fixed, SEDModel
    from tengri.utils.cosmology import age_at_z

    age_gyr = float(age_at_z(z))
    onset_param = _ONSET_PARAM[truncated]
    sfh_group = {
        "type": [truncated, untruncated],
        f"sfh_{truncated}_log_total_mass": Fixed(LOG_TOTAL_MASS),
        onset_param: Fixed(2.0 * age_gyr),
        f"sfh_{untruncated}_log_total_mass": Fixed(LOG_TOTAL_MASS),
        **extra_t,
        **extra_u,
    }
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh=sfh_group,
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(z),
    )
    pred = model.predict({})
    requested = 2.0 * 10.0**LOG_TOTAL_MASS
    ratio = float(pred.stellar_mass) / requested
    assert ratio == pytest.approx(1.0, abs=1e-4), (
        f"composite ['{truncated}', '{untruncated}'] z={z}: pooled formed mass ratio {ratio} != 1"
    )
