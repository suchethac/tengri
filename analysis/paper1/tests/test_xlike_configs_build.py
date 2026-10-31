# SPDX-License-Identifier: BSD-3-Clause
"""Real-build smoke test for the five X-like configurations.

Pins that each X-like builder (cigale_like, prospector_like, bagpipes_like,
beagle_like, dense_basis_like) actually builds a real :class:`SEDModel`
against its declared SSP grid and predicts finite, positive photometry for a
prior sample -- the check that matters before committing an X-like key to
100 NUTS fits on real galaxies. ``test_xlike_configs.py`` only mocks
``SEDModel.build`` to check the argument dicts a builder constructs, which
cannot catch a key the build grammar rejects (see xlike_configs.py's
``load_ssp_for_xlike`` and ``dense_basis_like`` fixes, both found by
running this test for real).

Also pins three code-parity invariants on the built model's resolved spec:
- beagle_like has no dust-emission (IR re-emission) component.
- cigale_like's two-component attenuation resolves to leitherer02 on both
  the birth-cloud and diffuse screens.
- dense_basis_like's reference age for the time quantiles is ``age_at_z(z)``,
  derived by tengri from the model's redshift, and the build leaves the
  dense_basis registry entry untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1 import xlike_configs
from paper1.config_metadata import XLIKE_KEYS

from tengri import SFH_REGISTRY, Observation, Photometry
from tengri.components.stellar.component import age_universe_kwargs
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.contract

#: Fixed for every build here: the paper's CANDELS z~1 galaxy sample, and
#: the seed configs.py's own __main__ smoke check uses.
Z = 1.0
SEED = 0

#: Same 13-filter test set as configs.py's __main__ block, so this smoke
#: test builds against the observation shape the paper actually fits.
TEST_FILTERS = [
    "hst_f435w",
    "hst_f606w",
    "hst_f775w",
    "hst_f814w",
    "hst_f850lp",
    "hst_f105w",
    "hst_f125w",
    "hst_f160w",
    "vista_ks",
    "irac_36",
    "irac_45",
    "irac_58",
    "irac_80",
]


def _observation() -> Observation:
    return Observation(photometry=Photometry.from_names(TEST_FILTERS))


def _load_ssp_or_fail(key: str):
    """Load the X-like key's SSP grid, failing loudly (not skipping) if absent."""
    try:
        return xlike_configs.load_ssp_for_xlike(key)
    except FileNotFoundError as exc:
        pytest.fail(f"{key}: SSP grid not on disk, cannot smoke-test a real build.\n{exc}")


@pytest.mark.parametrize("key", XLIKE_KEYS)
def test_xlike_config_builds_and_predicts(key):
    """Every X-like key builds against its real SSP grid and predicts a
    finite, positive photometry vector for a prior sample."""
    ssp_data = _load_ssp_or_fail(key)
    observation = _observation()

    model = xlike_configs.XLIKE_BUILDERS[key](ssp_data, observation, z=Z)

    free_params = model.spec.free_params
    print(f"\n{key}: {len(free_params)} free parameters: {free_params}")
    assert len(free_params) > 0, f"{key} built with no free parameters"

    sample = model.spec.sample(key=jax.random.PRNGKey(SEED))
    pred = model.predict_photometry(sample)

    assert pred.shape == (len(TEST_FILTERS),), f"{key}: wrong photometry shape {pred.shape}"
    assert jnp.all(jnp.isfinite(pred)), f"{key}: non-finite photometry {pred}"
    assert jnp.all(pred > 0), f"{key}: non-positive photometry {pred}"


def test_beagle_like_has_no_dust_emission_component():
    """beagle_like's dust_emission={'type': 'none'} resolves to no IR
    re-emission component, matching Pacifici+2023 Table 1 (BEAGLE: dust_em
    = 'No')."""
    ssp_data = _load_ssp_or_fail("beagle_like")
    model = xlike_configs.XLIKE_BUILDERS["beagle_like"](ssp_data, _observation(), z=Z)
    assert model.spec.dust_emission is None, (
        f"beagle_like resolved a dust emission component: {model.spec.dust_emission!r}"
    )


def test_cigale_like_attenuation_resolves_to_leitherer02_both_screens():
    """cigale_like's two-component attenuation resolves to leitherer02 on
    both the birth-cloud and diffuse screens."""
    ssp_data = _load_ssp_or_fail("cigale_like")
    model = xlike_configs.XLIKE_BUILDERS["cigale_like"](ssp_data, _observation(), z=Z)
    assert model.spec.dust_law_bc == "leitherer02"
    assert model.spec.dust_law_diff == "leitherer02"


def test_dense_basis_like_age_universe_resolves_to_age_at_z():
    """dense_basis_like's time-quantile reference age is age_at_z(z).

    tengri derives it from the model's redshift
    (``tengri.components.stellar.component.age_universe_kwargs``), so the
    build must leave the dense_basis registry settings untouched and the
    derived reference age must equal the age of the universe at ``Z`` [yr],
    not the z=0 value.
    """
    ssp_data = _load_ssp_or_fail("dense_basis_like")
    settings_before = dict(SFH_REGISTRY["dense_basis"].settings)

    model = xlike_configs.XLIKE_BUILDERS["dense_basis_like"](ssp_data, _observation(), z=Z)

    assert dict(SFH_REGISTRY["dense_basis"].settings) == settings_before
    assert model._get_redshift({}) == pytest.approx(Z)

    derived = age_universe_kwargs("dense_basis", model._get_redshift({}))
    expected_yr = float(age_at_z(Z)) * 1e9
    assert float(derived["age_universe_yr"]) == pytest.approx(expected_yr)
    assert expected_yr != pytest.approx(13.47e9), (
        "age_at_z(Z) must differ from the z=0 age, otherwise this cannot tell the "
        "redshift-derived age from a constant"
    )
