# SPDX-License-Identifier: BSD-3-Clause
"""d(SED)/d(sfh_<family>_<stem>_frac) is finite and nonzero near the age of the universe.

Each fraction is a dimensionless onset in (0, 1] of age(z). At z = 2 the
gradient is taken at frac = 0.99, just inside the age of the universe, where
the onset is close to the Big Bang and the mass-conserving normalization is
most sensitive to it (#2748). One representative of each onset family is
checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.gradient

_Z = 2.0
_FRAC = 0.99
_LOG_TOTAL_MASS = 10.0

_LNORM_NAN = pytest.mark.xfail(
    strict=True,
    reason=(
        "the lnorm age gradient is NaN on the base commit with the Gyr parameter "
        "too (pre-existing, outside the #2748 fraction work)"
    ),
)

#: One representative per onset family: (SFH type, onset stem).
FAMILIES = (
    ("dpl", "age"),
    ("delayed", "age"),
    ("exp", "start"),
    ("declining_exp", "age"),
    ("psb", "age"),
    ("delayed_bq", "age_main"),
    pytest.param("lnorm", "age", marks=_LNORM_NAN),
    ("tsnorm", "peak_lbt"),
    ("snorm", "peak_lbt"),
    ("snorm_burst", "peak_lbt"),
    ("tsnorm_burst", "peak_lbt"),
    ("norm", "peak_lbt"),
)


def _model(ssp, family, stem):
    return SEDModel.build(
        ssp_data=ssp,
        neb={"type": "none"},
        sfh={
            "type": family,
            "all_params": Fixed(DEFAULT),
            f"{stem}_frac": Uniform(0.5, 1.0, default=_FRAC),
            "log_total_mass": Fixed(_LOG_TOTAL_MASS),
        },
        redshift=Fixed(_Z),
    )


@pytest.mark.parametrize(("family", "stem"), FAMILIES)
def test_sed_and_gradient_wrt_frac_are_finite_and_nonzero(synthetic_ssp_wide, family, stem):
    """The SED sum and its frac-gradient are finite and nonzero at frac = 0.99, z = 2."""
    model = _model(synthetic_ssp_wide, family, stem)
    name = f"sfh_{family}_{stem}_frac"

    def sed_sum(frac):
        return jnp.sum(model.predict_state({name: frac}).sed_intrinsic)

    value = sed_sum(_FRAC)
    grad = jax.grad(sed_sum)(_FRAC)

    assert bool(jnp.isfinite(value)) and float(value) > 0.0
    assert bool(jnp.isfinite(grad)), f"d(SED)/d({name}) is not finite"
    assert float(grad) != 0.0, f"d(SED)/d({name}) is exactly zero at frac={_FRAC}, z={_Z:g}"


_PSB_TYPES = ("psb_suess2022", "psb_flex")


@pytest.mark.parametrize("family", _PSB_TYPES)
def test_psb_tflex_frac_gradient_is_finite_and_nonzero(synthetic_ssp_wide, family):
    """d(SED)/d(tflex_frac) is finite and nonzero at frac = 0.99, z = 2 for both PSB families.

    The tlast fraction is sampled alongside, as a pair in fractions is required.
    Its ceiling is the tflex floor, so the zone stays non-negative over the draw.
    """
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY, _spec_public_prefix

    age0 = float(age_at_z(0.0))
    prefix = _spec_public_prefix(SFH_REGISTRY[family])
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        neb={"type": "none"},
        sfh={
            "type": family,
            "all_params": Fixed(DEFAULT),
            "tlast_frac": Uniform(0.01 / age0, 1.0 / age0, default=0.1 / age0),
            "tflex_frac": Uniform(0.5, 1.0, default=_FRAC),
            "log_total_mass": Fixed(_LOG_TOTAL_MASS),
        },
        redshift=Fixed(_Z),
    )
    name = f"{prefix}tflex_frac"
    partner = {f"{prefix}tlast_frac": 0.1 / age0}

    def sed_sum(frac):
        return jnp.sum(model.predict_state({name: frac, **partner}).sed_intrinsic)

    value = sed_sum(_FRAC)
    grad = jax.grad(sed_sum)(_FRAC)

    assert bool(jnp.isfinite(value)) and float(value) > 0.0
    assert bool(jnp.isfinite(grad)), f"d(SED)/d({name}) is not finite"
    assert float(grad) != 0.0, f"d(SED)/d({name}) is exactly zero at frac={_FRAC}, z={_Z:g}"
