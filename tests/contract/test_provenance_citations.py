# SPDX-License-Identifier: BSD-3-Clause
"""SSP/IMF/nebular/dust provenance citations + the can't-infer warning.

Locks that ``collect_citations`` surfaces the full ingredient provenance from
an SSP grid's filename tokens (SPS code / isochrone / spectral library / IMF),
that the WavePrecomp precomputation method cites Zacharegkas+2025, and that an
unrecognisable filename fires a provenance warning (per user request).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from tengri.citations.collect import _ssp_provenance_keys
from tengri.components.stellar.sps.dsps_wrapper import SSPData

pytestmark = pytest.mark.contract

_Z = jnp.zeros(1)


def _ssp(source: str, imf: str = "chabrier") -> SSPData:
    return SSPData(_Z, _Z, _Z, _Z, None, None, imf=imf, source=source)


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "fsps_prsc_miles_chabrier",
            {"fsps2009", "fsps", "parsec", "miles", "chabrier2003", "aringer2009", "villaume2015"},
        ),
        ("fsps_mist_c3k_kroupa", {"mist", "mist_dotter2016", "fsps2009", "kroupa2001"}),
        ("bc03_pdva_stelib_salpeter", {"bc03", "padova", "stelib", "salpeter1955"}),
        ("fsps_bsti_basel_chabrier", {"basti", "basel", "chabrier2003"}),
        ("pgny_mist_c3k_chabrier", {"progeny", "mist", "chabrier2003"}),
    ],
)
def test_provenance_tokens_map_to_citations(source, expected):
    imf = source.rsplit("_", 1)[-1]
    keys = set(_ssp_provenance_keys(_ssp(source, imf=imf)))
    assert expected <= keys, f"missing {expected - keys} for {source!r}"


def test_unrecognised_filename_warns():
    """A filename that doesn't match <code>_<isochrone>_<library>_<imf> warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ssp_provenance_keys(_ssp("my_custom_grid_v2", imf="unknown"))
    assert any("infer" in str(w.message).lower() for w in caught)


def test_recognised_filename_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ssp_provenance_keys(_ssp("fsps_prsc_miles_chabrier"))
    assert not caught


def test_bpass_internal_isochrone_does_not_warn():
    """BPASS handles isochrones internally ('stars' token) — must not false-warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        keys = set(_ssp_provenance_keys(_ssp("bpss_stars_c3k_chabrier")))
    assert not caught
    assert "bpass" in keys


def test_wne_grid_cites_byler_nebular():
    """``wNE`` (with nebular emission) grids cite Byler+2017 + Cloudy."""
    keys = set(_ssp_provenance_keys(_ssp("ssp_prsc_miles_chabrier_wNE_logGasU-3.0")))
    assert {"byler2017", "cloudy"} <= keys


def test_dust_emission_citations_cover_models():
    """Every dust-emission model maps to a registered citation (THEMIS → Jones)."""
    from tengri.citations.associations import DUST_EMISSION_CITATIONS
    from tengri.citations.registry import REGISTRY

    assert DUST_EMISSION_CITATIONS["themis"] == ["jones2013", "jones2017"]
    for keys in DUST_EMISSION_CITATIONS.values():
        for k in keys:
            assert k in REGISTRY, f"dust-emission citation {k} not registered"


def test_imf_keys_present_in_registry():
    """Chabrier / Kroupa / Salpeter IMF citations are registered."""
    from tengri.citations.registry import REGISTRY

    for key in ("chabrier2003", "kroupa2001", "salpeter1955"):
        assert key in REGISTRY, f"{key} missing from citation registry"


def test_fsps_grids_cite_baked_in_agb_ingredients():
    """FSPS grids inherit Aringer+2009 (C-star library) and Villaume+2015 (AGB dust).

    Both are part of how FSPS generates every SSP grid (carbon-star spectra
    redward of K; circumstellar AGB dust via ``add_agb_dust_model``, on by
    default), so they must fire for any ``fsps_*`` source regardless of the
    isochrone / library / IMF chosen. Closes the SSP-provenance checkboxes of #560.
    """
    keys = set(_ssp_provenance_keys(_ssp("fsps_mist_miles_chabrier")))
    assert {"aringer2009", "villaume2015"} <= keys


def test_non_fsps_grids_omit_fsps_agb_ingredients():
    """The FSPS-baked AGB ingredients must NOT attach to non-FSPS SPS codes."""
    keys = set(_ssp_provenance_keys(_ssp("bc03_pdva_stelib_salpeter", imf="salpeter")))
    assert not ({"aringer2009", "villaume2015"} & keys)


def test_fsps_agb_ingredient_keys_present_in_registry():
    """Aringer+2009 and Villaume+2015 resolve to real references.bib entries."""
    from tengri.citations.registry import REGISTRY

    for key in ("aringer2009", "villaume2015"):
        assert key in REGISTRY, f"{key} missing from citation registry"
