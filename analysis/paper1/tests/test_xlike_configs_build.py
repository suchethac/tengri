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
- dense_basis_like's cosmic-age cutoff resolves to ``age_at_z(z)``, not the
  registry's z=0 default of 13.47 Gyr (see xlike_configs.dense_basis_like
  for why this is read off ``tengri.SFH_REGISTRY`` rather than off the
  model itself).
"""

from __future__ import annotations

import contextlib
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


@contextlib.contextmanager
def _dense_basis_registry_restored():
    """Restore SFH_REGISTRY['dense_basis']'s age-of-universe setting on exit.

    ``dense_basis_like`` deliberately leaves this registry entry mutated
    (see its docstring): the component that reads it does so lazily, at the
    first prediction after build, so restoring inside the builder itself
    would make the override inert. A test that builds it for real must
    restore the setting itself so it does not leak into whatever else runs
    in this pytest session.
    """
    original = SFH_REGISTRY["dense_basis"].settings.get("sfh_db_age_universe_gyr")
    try:
        yield
    finally:
        if original is None:
            SFH_REGISTRY["dense_basis"].settings.pop("sfh_db_age_universe_gyr", None)
        else:
            SFH_REGISTRY["dense_basis"].settings["sfh_db_age_universe_gyr"] = original


@pytest.mark.parametrize("key", XLIKE_KEYS)
def test_xlike_config_builds_and_predicts(key):
    """Every X-like key builds against its real SSP grid and predicts a
    finite, positive photometry vector for a prior sample."""
    ssp_data = _load_ssp_or_fail(key)
    observation = _observation()

    with contextlib.ExitStack() as stack:
        if key == "dense_basis_like":
            stack.enter_context(_dense_basis_registry_restored())

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
    """dense_basis_like's cosmic-age cutoff resolves to age_at_z(z) -- the
    unit the component reads (Gyr), via
    SFH_REGISTRY["dense_basis"].settings["sfh_db_age_universe_gyr"] (see
    src/tengri/components/stellar/component.py:2360 and
    src/tengri/components/stellar/sfh/registry.py:1977-2010) -- not the
    registry's z=0 default of 13.47 Gyr."""
    ssp_data = _load_ssp_or_fail("dense_basis_like")

    with _dense_basis_registry_restored():
        xlike_configs.XLIKE_BUILDERS["dense_basis_like"](ssp_data, _observation(), z=Z)

        expected_age_gyr = float(age_at_z(Z))
        resolved_age_gyr = SFH_REGISTRY["dense_basis"].settings["sfh_db_age_universe_gyr"]
        assert resolved_age_gyr == pytest.approx(expected_age_gyr)
        assert resolved_age_gyr != pytest.approx(13.47), (
            "age_at_z(1.0) must not coincide with the z=0 registry default -- "
            "otherwise this test cannot tell the override from a no-op"
        )
