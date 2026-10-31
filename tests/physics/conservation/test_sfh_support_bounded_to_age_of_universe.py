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
``SFH_REGISTRY`` (21 families) x z in {0.5, 2.5, 6.0} x onset at {0.5, 1.2, 2.0} x age(z): 189
cells. Before the fix, 149/189 cells failed; the physical fix
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
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.conservation


def _log_total_mass_public_name(family: str) -> str:
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    spec = SFH_REGISTRY[family]
    for pub, (internal, _scale, _offset) in spec.internal_param_map.items():
        if internal == "log_total_mass":
            return pub
    raise AssertionError(f"family {family!r} has no log_total_mass parameter")


# Family -> its public onset/age/peak-time parameter name (the fraction of age(z)).
ONSET_PARAMS_BY_FAMILY = {
    "const": "sfh_const_start_frac",
    "dexp": "sfh_dexp_start_frac",
    "exp": "sfh_exp_start_frac",
    "declining_exp": "sfh_declining_exp_age_frac",
    "delayed": "sfh_delayed_age_frac",
    "delayed_bq": "sfh_delayed_bq_age_main_frac",
    "dpl": "sfh_dpl_age_frac",
    "dpl_lookback": "sfh_dpl_lookback_age_frac",
    "lnorm": "sfh_lnorm_age_frac",
    "norm": "sfh_norm_peak_lbt_frac",
    "periodic": "sfh_periodic_age_frac",
    "psb": "sfh_psb_age_frac",
    "sfh2exp": "sfh_sfh2exp_age_frac",
    "snorm": "sfh_snorm_peak_lbt_frac",
    "snorm_burst": "sfh_snorm_burst_peak_lbt_frac",
    "trunc_exp": "sfh_trunc_exp_age_frac",
    "tsnorm": "sfh_tsnorm_peak_lbt_frac",
    "tsnorm_burst": "sfh_tsnorm_burst_peak_lbt_frac",
    "const_exp": "sfh_cexp_age_frac",
    "psb_flex": "sfh_psb_flex_tflex_frac",
    "psb_suess2022": "sfh_psb2022_tflex_frac",
}

Z_VALUES = (0.5, 2.5, 6.0)
MULTIPLIERS = (0.5, 1.2, 2.0)
LOG_TOTAL_MASS = 9.0
#: Upper bound of the swept onset fractions: wide enough to hold 2.0 x age(z) at
#: every z (the fraction of age(z) is at most 2.0 here), in units of age(z=0).
_OVERRIDE_CEIL_FRAC = 100.0 / float(age_at_z(0.0))
#: Upper bound of the free delayed_bq age_bq [Gyr]: a Gyr name, not a fraction. Its
#: swept values are at most 0.3 Gyr, and a bound above 20 Gyr is refused as implausible.
_OVERRIDE_CEIL_GYR = 10.0


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
        onset_short: Uniform(floor, _OVERRIDE_CEIL_FRAC),
        mass_short: Fixed(LOG_TOTAL_MASS),
        "age_kernel": age_kernel,
    }
    if family == "delayed_bq":
        # Default age_bq_gyr (0.5 Gyr, the burst/quench onset) can exceed a
        # swept age_main at the smallest multiplier (0.5 x age(z) at
        # z=6 is 0.467 Gyr) -- a physically nonsensical "quench before
        # formation" combination unrelated to this test's own onset sweep.
        # Free it too so it can be pinned below age_main per cell.
        bq_name = "sfh_delayed_bq_age_bq_gyr"
        bq_short = _short_key(family, bq_name, prefixes)
        bq_floor = _onset_override_floor(family, bq_name)
        sfh_group[bq_short] = Uniform(bq_floor, _OVERRIDE_CEIL_GYR)

    if family in ("psb_flex", "psb_suess2022"):
        # The two quench times are both fractions of age(z) or both Gyr (#2748). The
        # sweep moves tflex only, so tlast stays at its absolute Gyr default.
        tlast_gyr = onset_name.replace("tflex_frac", "tlast_gyr")
        tlast_short = _short_key(family, onset_name.replace("tflex", "tlast"), prefixes)
        tlast_value = float(SFH_REGISTRY[family].params[tlast_gyr].default.value)
        tlast_frac = tlast_value / float(age_at_z(z))
        sfh_group[tlast_short] = Fixed(tlast_frac)
        # The swept tflex must stay above tlast, so its prior starts there.
        sfh_group[onset_short] = Uniform(max(floor, tlast_frac), _OVERRIDE_CEIL_FRAC)

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
    age_gyr = float(age_at_z(z))
    # The onset is a fraction of age(z), so mult x age(z) Gyr at this redshift.
    model = _build_model(synthetic_ssp_wide, family, onset_name, z, age_kernel=age_kernel)
    params = {onset_name: mult}
    if family == "delayed_bq":
        params["sfh_delayed_bq_age_bq_gyr"] = min(0.3, 0.3 * mult * age_gyr)

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
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "age_frac": Fixed(2.0),
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
#: Gyr values of the untruncated members. A timing given in Gyr is written as its
#: ``*_frac`` twin at the test redshift, capped below 1 where the universe is younger
#: than the Gyr value (the 1.0 Gyr start at z = 6), so the member stays inside [0, age(z)].
_COMPOSITE_PAIRS: tuple[tuple[str, dict, str, dict], ...] = (
    (
        "dpl",
        {"sfh_dpl_alpha": 1.5, "sfh_dpl_beta": 1.0, "sfh_dpl_tau_gyr": 3.0},
        "const",
        {"sfh_const_start_gyr": 1.0, "sfh_const_end_gyr": 0.0},
    ),
    (
        "exp",
        {"sfh_exp_tau_gyr": 2.0},
        "delayed",
        {"sfh_delayed_tau_gyr": 1.0, "sfh_delayed_age_gyr": 0.9},
    ),
    (
        "declining_exp",
        {"sfh_declining_exp_tau_gyr": 2.0},
        "dexp",
        {"sfh_dexp_tau_gyr": 0.5, "sfh_dexp_start_gyr": 0.9},
    ),
    (
        "norm",
        {"sfh_norm_width_gyr": 0.5},
        "const",
        {"sfh_const_start_gyr": 1.0, "sfh_const_end_gyr": 0.0},
    ),
)

#: Onset/age/peak-time public param name per family used above, so the
#: truncated member's own boundary can be set to 2 x age(z) (a fraction of age(z)).
_ONSET_PARAM = {
    "dpl": "sfh_dpl_age_frac",
    "exp": "sfh_exp_start_frac",
    "declining_exp": "sfh_declining_exp_age_frac",
    "norm": "sfh_norm_peak_lbt_frac",
}

#: Largest onset fraction of age(z) a Gyr value is converted to (#2748).
_MAX_ONSET_FRAC = 0.99


def _member_fixed(family: str, extra: dict, z: float) -> dict:
    """Fixed overrides of one composite member at redshift ``z``.

    A ``*_gyr`` timing whose ``*_frac`` twin the registry declares becomes that
    twin, valued at its Gyr value over age(z) and capped at ``_MAX_ONSET_FRAC``.
    Every other entry is a Fixed value unchanged.
    """
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    age = float(age_at_z(z))
    out = {}
    for name, value in extra.items():
        frac = name[: -len("_gyr")] + "_frac" if name.endswith("_gyr") else None
        if frac is not None and frac in SFH_REGISTRY[family].params:
            out[frac] = Fixed(min(value / age, _MAX_ONSET_FRAC))
        else:
            out[name] = Fixed(value)
    return out


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

    onset_param = _ONSET_PARAM[truncated]
    sfh_group = {
        "type": [truncated, untruncated],
        f"sfh_{truncated}_log_total_mass": Fixed(LOG_TOTAL_MASS),
        onset_param: Fixed(2.0),
        f"sfh_{untruncated}_log_total_mass": Fixed(LOG_TOTAL_MASS),
        **_member_fixed(truncated, extra_t, z),
        **_member_fixed(untruncated, extra_u, z),
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


@pytest.mark.parametrize(
    "family,z,mult,age_kernel",
    [
        (fam, z, mult, kernel)
        for fam in ("dpl", "delayed", "declining_exp", "periodic")
        for z in (0.1, 2.0)
        for mult in (0.5, 2.0)  # untruncated and truncated onsets
        for kernel in AGE_KERNELS
    ],
)
def test_sfr_history_integral_equals_formed_mass(family, z, mult, age_kernel, synthetic_ssp_wide):
    """∫ sfr_history d(lookback) inside the support == 10**log_mstar_formed (#2640).

    ``mult`` sweeps both the untruncated (0.5 x age(z), onset entirely within
    the support) and truncated (2.0 x age(z), onset exceeds age(z): the
    #2640 regression case) regimes, on both age kernels.

    The published history ends at ``age(z)`` and carries one factor, so its
    PLAIN trapezoid over the published grid equals the formed mass to float64
    precision.
    """
    age_gyr = float(age_at_z(z))
    onset_name = ONSET_PARAMS_BY_FAMILY[family]
    model = _build_model(synthetic_ssp_wide, family, onset_name, z, age_kernel=age_kernel)
    params = {onset_name: mult}
    if family == "delayed_bq":
        params["sfh_delayed_bq_age_bq_gyr"] = min(0.3, 0.3 * mult * age_gyr)

    st = model.predict_state(params)
    lbt_yr = np.asarray(st.derived["sfh_grid_lbt_yr"])
    sfr_history = np.asarray(st.derived["sfr_history"])
    formed_mass = 10.0 ** float(st.derived["log_mstar_formed"])

    trapz_support = np.trapezoid(sfr_history, lbt_yr)

    np.testing.assert_allclose(
        trapz_support,
        formed_mass,
        rtol=1e-8,
        err_msg=(
            f"{family} {age_kernel} z={z} mult={mult}: "
            f"∫sfr_history (plain, published grid) = {trapz_support:.6e}, "
            f"formed = {formed_mass:.6e}, ratio = {trapz_support / formed_mass:.8f}"
        ),
    )


def test_sfr_history_integral_equals_formed_mass_composite_field(synthetic_ssp_wide):
    """Same identity for the literal #2640 reproducer mechanism: dpl + field.

    The GP field forces the "dsps" kernel and modulates the published history
    multiplicatively AFTER the base shape's own normalization; both the
    unmodulated (zero field) and a bursty (random field) realization must
    satisfy the identity once rescaled.
    """
    import jax

    from tengri import DEFAULT, SEDModel

    z = 0.1
    n_grid = 256
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={
            "type": ["dpl", "field"],
            "sfh_dpl_age_frac": Fixed(2.0),  # onset exceeds age(z): truncated
            "sfh_dpl_log_total_mass": Fixed(LOG_TOTAL_MASS),
            "sfh_dpl_alpha": Fixed(1.0),
            "sfh_dpl_beta": Fixed(1.5),
            "sfh_dpl_tau_gyr": Fixed(3.0),
            "sfh_field_psd_sigma": Fixed(2.0),
            "sfh_field_psd_tau_myr": Fixed(50.0),
        },
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(z),
        n_grid=n_grid,
    )

    for label, xi in (
        ("zero", np.zeros(n_grid)),
        ("random", np.asarray(jax.random.normal(jax.random.PRNGKey(42), shape=(n_grid,)))),
    ):
        st = model.predict_state({"sfh_field_xi": xi})
        lbt_yr = np.asarray(st.derived["sfh_grid_lbt_yr"])
        sfr_history = np.asarray(st.derived["sfr_history"])
        formed_mass = 10.0 ** float(st.derived["log_mstar_formed"])

        trapz_support = np.trapezoid(sfr_history, lbt_yr)

        np.testing.assert_allclose(
            trapz_support,
            formed_mass,
            rtol=1e-6,
            err_msg=(
                f"dpl+field ({label} field): published "
                f"∫sfr_history = {trapz_support:.6e}, formed = {formed_mass:.6e}, "
                f"ratio = {trapz_support / formed_mass:.8f}"
            ),
        )
