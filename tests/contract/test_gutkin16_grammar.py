# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``neb={'type': 'gutkin16'}`` is registered end to end (#983).

Registration only: the forward path (``forward/*``, ``component.py``) is wired by
the next task, so nothing here builds an ``SEDModel``. Every assertion goes through
the menu (``list_nebular_backends``), the grammar (``parse_groups``), the flat
``Parameters`` object, the cache key, the grid-support table and the citation
association table.
"""

from __future__ import annotations

import warnings

import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed
from tengri._data_setup import package_or_env_data_path
from tengri.components.grid_support import grid_support
from tengri.config.exceptions import ParameterError
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

pytestmark = pytest.mark.contract

_GRID = str(package_or_env_data_path("gutkin16_grid.h5"))


def _neb(**neb):
    return parse_groups(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "gutkin16", "all_params": Fixed(DEFAULT), **neb},
        redshift=Fixed(0.1),
    )


def _free(**groups) -> set[str]:
    return set(parse_groups(**{"redshift": Fixed(0.1), **groups}).free_params)


def test_list_nebular_backends_has_gutkin16():
    names = {row["name"] for row in tengri.list_nebular_backends()}
    assert "gutkin16" in names


def test_translate_sets_nebular_mode():
    params = _neb()
    assert params.nebular_mode == "gutkin16"


def test_structural_keys_round_trip_to_groups():
    params = _neb(mup=300, continuum="none", grid=_GRID)
    assert params.nebular_gutkin16_mup == 300.0
    assert params.nebular_gutkin16_continuum == "none"
    assert params.nebular_gutkin16_grid_path == _GRID
    emitted = params.to_groups()["neb"]
    assert emitted["mup"] == 300.0
    assert emitted["continuum"] == "none"
    assert emitted["grid"] == _GRID


def test_structural_defaults_are_not_emitted():
    emitted = _neb().to_groups()["neb"]
    assert not {"mup", "continuum", "grid"} & set(emitted)


def test_mup_250_refused():
    with pytest.raises(ParameterError, match="mup"):
        _neb(mup=250)


def test_mup_key_refused_for_cb19():
    with pytest.raises(ValueError, match="mup"):
        parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={"type": "cb19", "mup": 100, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )


# Pinned: the nebular knobs gutkin16 declares, plus the shared always-declared
# nebular parameters the wildcard frees alongside them (printed and pinned).
_GUTKIN16_FREE = {
    "neb_logU",
    "neb_logZ_gas",
    "neb_log_nH",
    "neb_xi_d",
    "neb_co_ratio",
}


def test_all_params_free_frees_exactly_the_declared_axes():
    free = _free(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "gutkin16", "all_params": FREE},
    )
    neb_free = {name for name in free if name.startswith("neb_")}
    assert neb_free == _GUTKIN16_FREE | _GUTKIN16_SHARED_FREE
    assert "neb_co" not in neb_free
    assert "neb_dno" not in neb_free
    assert "neb_hbfrac" not in neb_free


# Shared nebular names the wildcard also frees (the always-declared nebular knobs,
# printed from a run of this fixture and pinned here).
_GUTKIN16_SHARED_FREE = {
    "neb_dig_delta_logU",
    "neb_dig_frac",
    "neb_eline_sigma_kms",
    "neb_fdust_frac",
    "neb_fesc",
    "neb_fesc_lya",
}


def test_cache_key_differs_by_mup_and_continuum():
    base = Parameters(mean_sfh_type="dpl", nebular="gutkin16", redshift=Fixed(0.1))
    mup = Parameters(
        mean_sfh_type="dpl", nebular="gutkin16", nebular_gutkin16_mup=300.0, redshift=Fixed(0.1)
    )
    cont = Parameters(
        mean_sfh_type="dpl",
        nebular="gutkin16",
        nebular_gutkin16_continuum="none",
        redshift=Fixed(0.1),
    )
    assert base.cache_key() != mup.cache_key()
    assert base.cache_key() != cont.cache_key()


def test_grid_support_narrows_log_nh_to_2_3_and_refuses_prior_past_grid():
    lo, hi = grid_support("neb", "gutkin16")["neb_log_nH"]
    assert (lo, hi) == (2.0, 3.0)
    with (
        warnings.catch_warnings(),
        pytest.raises(ParameterError, match=r"neb_log_nH"),
    ):
        warnings.simplefilter("ignore")
        Parameters(
            mean_sfh_type="dpl",
            nebular="gutkin16",
            neb_log_nH=Uniform(2.5, 3.5),
            redshift=Fixed(0.1),
        )


def test_citation_resolves():
    from tengri.citations.registry import cite
    from tengri.citations.resolve import association_keys_for

    keys = association_keys_for("gutkin16")
    assert "gutkin2016" in keys
    assert "cloudy" in keys
    for key in keys:
        assert cite(key) is not None, f"citation key {key!r} does not resolve"
