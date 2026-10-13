# SPDX-License-Identifier: BSD-3-Clause
"""At z = 0 each fractional onset reproduces its absolute Gyr onset (#2748).

``sfh_<family>_<stem>_frac = 1`` maps to ``1 * age(z=0)``, the unrounded age of
the universe. The reference is the same build with the absolute Gyr name fixed
at that age, so the comparison is within one build and needs no main-branch
baseline. The comparison runs for one representative of each onset family.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.stellar.sfh.registry import _AGE_UNIV_GYR, SFH_REGISTRY
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.contract

_LOG_TOTAL_MASS = 10.0

#: One representative per onset family: (SFH type, onset stem).
FAMILIES = (
    ("dpl", "age"),
    ("delayed", "age"),
    ("exp", "start"),
    ("declining_exp", "age"),
    ("psb", "age"),
    ("delayed_bq", "age_main"),
    ("lnorm", "age"),
    ("tsnorm", "peak_lbt"),
    ("snorm", "peak_lbt"),
    ("snorm_burst", "peak_lbt"),
    ("tsnorm_burst", "peak_lbt"),
    ("norm", "peak_lbt"),
)


def _sed_at_z0(ssp, family, sfh, params=None):
    model = SEDModel.build(
        ssp_data=ssp,
        neb={"type": "none"},
        sfh={
            "type": family,
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Fixed(_LOG_TOTAL_MASS),
            **sfh,
        },
        redshift=Fixed(0.0),
    )
    return model.predict_state(params or {}).sed_intrinsic


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def _gyr_default(family, stem):
    """The registry default of the absolute Gyr onset, as a float [Gyr]."""
    dist = SFH_REGISTRY[family].params[f"sfh_{family}_{stem}_gyr"].default
    value = dist.value if isinstance(dist, Fixed) else dist.default
    return float(value)


@pytest.mark.parametrize(("family", "stem"), FAMILIES, ids=[f for f, _ in FAMILIES])
def test_frac_one_matches_gyr_at_age_of_universe(synthetic_ssp_wide, family, stem):
    """A sampled frac = 1 and a Gyr onset of age(z=0) give the same SED to 1e-10 relative.

    The fraction must be sampled: a Fixed(1) fraction is the unsampled shim, so
    the Gyr default stands.
    """
    age0 = float(age_at_z(0.0))
    via_frac = _sed_at_z0(
        synthetic_ssp_wide,
        family,
        {f"{stem}_frac": Uniform(0.01, 1.0)},
        params={f"sfh_{family}_{stem}_frac": 1.0},
    )
    via_gyr = _sed_at_z0(synthetic_ssp_wide, family, {f"{stem}_gyr": Fixed(age0)})
    rel = _rel(via_frac, via_gyr)
    assert rel <= 1e-10, f"{family}: relative SED difference {rel:.3e} > 1e-10"


@pytest.mark.parametrize(("family", "stem"), FAMILIES, ids=[f for f, _ in FAMILIES])
def test_frac_default_matches_gyr_default_at_z0(synthetic_ssp_wide, family, stem):
    """A sampled fraction at the Gyr default's ratio reproduces the Gyr default at z = 0.

    A default that is an exact ratio of the unrounded age (5 Gyr) matches to
    1e-10. The default ``_AGE_UNIV_GYR`` is the age rounded to three decimals,
    so its fraction is 1 and the SED differs by that rounding alone; the
    tolerance for that case is the measured bound, not 1e-10.
    """
    age0 = float(age_at_z(0.0))
    gyr_default = _gyr_default(family, stem)
    frac_default = min(1.0, gyr_default / age0)
    via_gyr = _sed_at_z0(synthetic_ssp_wide, family, {})
    via_frac = _sed_at_z0(
        synthetic_ssp_wide,
        family,
        {f"{stem}_frac": Uniform(0.01, 1.0, default=frac_default)},
        params={f"sfh_{family}_{stem}_frac": frac_default},
    )
    rel = _rel(via_frac, via_gyr)
    tol = 1e-4 if gyr_default == _AGE_UNIV_GYR else 1e-10
    assert rel <= tol, f"{family}: relative SED difference {rel:.3e} > {tol:g}"


@pytest.mark.parametrize(("family", "stem"), FAMILIES, ids=[f for f, _ in FAMILIES])
def test_unsampled_frac_leaves_gyr_value_in_force(synthetic_ssp_wide, family, stem):
    """An unsampled Fixed fraction does not replace the absolute Gyr onset (#2748).

    The Fixed fraction of 1 is the shim present in every merged dict. A Gyr onset
    of 5 Gyr must give a different SED from one at age(z=0), so the fraction must
    not overwrite it. Before the routing in ``merge_fixed_params``, both gave the
    age of the universe.
    """
    age0 = float(age_at_z(0.0))
    at_5 = _sed_at_z0(synthetic_ssp_wide, family, {f"{stem}_gyr": Fixed(5.0)})
    at_age0 = _sed_at_z0(synthetic_ssp_wide, family, {f"{stem}_gyr": Fixed(age0)})
    assert _rel(at_5, at_age0) > 1e-2, f"{family}: the Gyr onset is ignored"


_PSB_TYPES = ("psb_suess2022", "psb_flex")


@pytest.mark.parametrize("family", _PSB_TYPES)
def test_psb_fraction_pair_at_gyr_defaults_matches_gyr_at_z0(synthetic_ssp_wide, family):
    """The PSB quench-time fractions at their Gyr defaults' ratio reproduce the Gyr build at z = 0.

    Both fractions are sampled, since a pair given in Gyr for one time and as a
    fraction for the other is refused. The reference is the registry default in
    Gyr; the frac build uses the same two timings as fractions of the same age.
    """
    from tengri.components.stellar.sfh.registry import _spec_public_prefix

    age0 = float(age_at_z(0.0))
    prefix = _spec_public_prefix(SFH_REGISTRY[family])
    tlast_default = _gyr_default_named(family, f"{prefix}tlast_gyr")
    tflex_default = _gyr_default_named(family, f"{prefix}tflex_gyr")
    via_gyr = _sed_at_z0(synthetic_ssp_wide, family, {})
    via_frac = _sed_at_z0(
        synthetic_ssp_wide,
        family,
        {
            "tlast_frac": Uniform(0.01 / age0, 1.0 / age0, default=tlast_default / age0),
            "tflex_frac": Uniform(1.0 / age0, 5.0 / age0, default=tflex_default / age0),
        },
        params={
            f"{prefix}tlast_frac": tlast_default / age0,
            f"{prefix}tflex_frac": tflex_default / age0,
        },
    )
    rel = _rel(via_frac, via_gyr)
    assert rel <= 1e-10, f"{family}: relative SED difference {rel:.3e} > 1e-10"


def _gyr_default_named(family, name):
    """The registry default of one absolute-Gyr parameter of ``family``, as a float [Gyr]."""
    dist = SFH_REGISTRY[family].params[name].default
    value = dist.value if isinstance(dist, Fixed) else dist.default
    return float(value)
