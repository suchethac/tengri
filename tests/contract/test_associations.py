# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :mod:`tengri.citations.associations`.

Pins the component-value → citation-key mappings so a future rename in
``references.bib`` is caught immediately rather than silently dropping a
paper from every user's bibliography.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract
from tengri.citations import (
    BACKEND_CITATIONS,
    CORE_CITATIONS,
    DUST_LAW_CITATIONS,
    NEBULAR_BACKEND_CITATIONS,
    REGISTRY,
    cites,
)
from tengri.citations.associations import (
    AGN_BLR_CITATIONS,
    AGN_DISC_CITATIONS,
    AGN_TORUS_CITATIONS,
    DUST_EMISSION_CITATIONS,
    DUST_MODEL_CITATIONS,
    FUNCTION_CITATIONS,
    IGM_CITATIONS,
    register_function_citations,
)

# ---------------------------------------------------------------------------
# Every key referenced in an association table must exist in the registry.
# ---------------------------------------------------------------------------


def _all_keys_from_table(table: dict) -> set[str]:
    keys: set[str] = set()
    for values in table.values():
        for k in values:
            keys.add(k)
    return keys


ALL_TABLES = (
    ("CORE_CITATIONS", {"_core": CORE_CITATIONS}),
    ("DUST_LAW_CITATIONS", DUST_LAW_CITATIONS),
    ("DUST_MODEL_CITATIONS", DUST_MODEL_CITATIONS),
    ("DUST_EMISSION_CITATIONS", DUST_EMISSION_CITATIONS),
    ("NEBULAR_BACKEND_CITATIONS", NEBULAR_BACKEND_CITATIONS),
    ("IGM_CITATIONS", IGM_CITATIONS),
    ("AGN_DISC_CITATIONS", AGN_DISC_CITATIONS),
    ("AGN_TORUS_CITATIONS", AGN_TORUS_CITATIONS),
    ("AGN_BLR_CITATIONS", AGN_BLR_CITATIONS),
    ("BACKEND_CITATIONS", BACKEND_CITATIONS),
)


@pytest.mark.parametrize("name,table", ALL_TABLES)
def test_all_association_keys_registered(name: str, table: dict | list):
    """Every citation key named in an association table must exist in REGISTRY.

    This is the test that catches a silent-drop refactor: if someone renames
    ``calzetti2000`` → ``calzetti_2000`` in ``references.bib`` without
    updating ``associations.py``, this assertion fails.
    """
    if name == "CORE_CITATIONS":
        keys = set(table["_core"])
    else:
        keys = _all_keys_from_table(table)

    missing = keys - set(REGISTRY.keys())
    assert not missing, f"{name} references unregistered keys: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Specific contract tests — component value → expected citation keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "law,expected",
    [
        ("calzetti", "calzetti2000"),
        ("kriek_conroy", "kriek_conroy2013"),
        ("smc", "gordon2003_smc"),
        ("cardelli", "cardelli1989"),
        ("salim", "salim2018"),
        ("reddy15", "reddy2015"),
        ("li08", "li2008_ext"),
    ],
)
def test_dust_law_mapping(law: str, expected: str):
    assert expected in DUST_LAW_CITATIONS[law], f"law_bc={law!r} should trigger {expected}"


@pytest.mark.parametrize(
    "emission,expected",
    [
        ("casey2012", "casey2012"),
        # modified_blackbody = Hildebrand 1983 (with da Cunha 2013 CMB-heating
        # correction applied automatically at z > 0); see the registry entry
        # in tengri/citations/associations.py.
        ("modified_blackbody", "hildebrand1983"),
        # greybody = Casey 2012 (with da Cunha 2013 CMB-heating correction).
        ("greybody", "casey2012"),
        ("dale2014", "dale2014"),
        ("draine_li2007", "draine_li2007"),
        ("draine_li2014", "draine2014"),
    ],
)
def test_dust_emission_mapping(emission: str, expected: str):
    assert expected in DUST_EMISSION_CITATIONS[emission]


@pytest.mark.parametrize(
    "backend,expected",
    [("cue", "cue"), ("cloudy", "cloudy")],
)
def test_nebular_backend_mapping(backend: str, expected: str):
    assert expected in NEBULAR_BACKEND_CITATIONS[backend]


@pytest.mark.parametrize(
    "igm_model,expected",
    [
        ("inoue", "inoue2014"),
        ("inoue2014", "inoue2014"),
        ("madau", "madau1995"),
    ],
)
def test_igm_mapping(igm_model: str, expected: str):
    assert expected in IGM_CITATIONS[igm_model]


@pytest.mark.parametrize(
    "disc,expected",
    [
        ("multicolor", "shakura_sunyaev1973"),
        ("kubota_done", "kubota_done2018"),
        ("adaf", "mahadevan1997"),
    ],
)
def test_agn_disc_mapping(disc: str, expected: str):
    assert expected in AGN_DISC_CITATIONS[disc]


@pytest.mark.parametrize(
    "torus,expected_any",
    [
        ("skirtor", "skirtor"),
        ("stalevski", "skirtor_2012"),
        ("clumpy", "clumpy_nenkova2008"),
        ("nenkova", "clumpy_nenkova2008"),
    ],
)
def test_agn_torus_mapping(torus: str, expected_any: str):
    assert expected_any in AGN_TORUS_CITATIONS[torus]


@pytest.mark.parametrize(
    "blr,expected",
    [
        ("temple", "temple2021_qsogen"),
        ("vanden_berk", "vandenberk2001"),
    ],
)
def test_agn_blr_mapping(blr: str, expected: str):
    assert expected in AGN_BLR_CITATIONS[blr]


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("mcmc_nuts", "blackjax"),
        ("pathfinder", "pathfinder"),
        ("mcmc_raytrace", "raytrace_behroozi"),
        ("evidence", "nss"),
        ("ess", "ess_murray2010"),
        ("vi", "nifty"),
    ],
)
def test_inference_backend_mapping(backend: str, expected: str):
    assert expected in BACKEND_CITATIONS[backend]


# ---------------------------------------------------------------------------
# @cites decorator
# ---------------------------------------------------------------------------


def test_cites_decorator_annotates_function():
    @cites("calzetti2000", "charlot_fall2000")
    def my_dust_law(wave, av):
        """A dust law."""
        return wave * av

    assert hasattr(my_dust_law, "_tengri_cites")
    assert "calzetti2000" in my_dust_law._tengri_cites
    assert "charlot_fall2000" in my_dust_law._tengri_cites


def test_cites_decorator_registers_in_function_citations():
    @cites("dsps")
    def my_sps_helper():
        pass

    qual = f"{my_sps_helper.__module__}.{my_sps_helper.__qualname__}"
    assert qual in FUNCTION_CITATIONS
    assert "dsps" in FUNCTION_CITATIONS[qual]


def test_cites_is_transparent():
    """The decorator must return the function unchanged."""

    @cites("tengri")
    def identity(x):
        return x

    assert identity(42) == 42


def test_register_function_citations_direct():
    register_function_citations("my.custom.path.func", ["tengri"])
    assert "tengri" in FUNCTION_CITATIONS["my.custom.path.func"]
