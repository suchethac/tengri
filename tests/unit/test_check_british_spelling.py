# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the American-English spelling guard (tools/check_british_spelling.py).

Guards the detector against the two failure modes that matter:
1. False negatives — British spellings (incl. snake_case / camelCase / ALL_CAPS
   subwords) must be flagged.
2. False positives — American-invariant words, the matplotlib ``Greys`` colormap,
   and allowlisted external-data-contract keys must NOT be flagged.
"""

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_british_spelling.py"
_spec = importlib.util.spec_from_file_location("check_british_spelling", _TOOL)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _flagged(text):
    return {british.lower() for _, _, british, _ in mod.scan_text(text)}


@pytest.mark.parametrize(
    "british, american",
    [
        ("colour", "color"),
        ("behaviour", "behavior"),
        ("normalise", "normalize"),
        ("normalisation", "normalization"),
        ("marginalised", "marginalized"),
        ("catalogue", "catalog"),
        ("catalogued", "cataloged"),
        ("centre", "center"),
        ("finalise", "finalize"),
        ("photoionised", "photoionized"),
        ("modelling", "modeling"),
        ("analyse", "analyze"),
        ("grey", "gray"),
    ],
)
def test_british_words_are_flagged(british, american):
    assert mod.to_american(british) == american
    assert british in _flagged(f"the {british} value")


@pytest.mark.parametrize(
    "word",
    [
        "noise",
        "raise",
        "raising",
        "exercise",
        "precise",
        "surprise",
        "revise",
        "otherwise",
        "piecewise",
        "advertise",
        "analyses",
        "greys",
        "color",
        "normalize",
        "catalog",  # already American
    ],
)
def test_invariant_and_american_words_not_flagged(word):
    assert mod.to_american(word) is None
    assert not _flagged(f"the {word} value")


def test_subword_detection_in_identifiers():
    # snake_case, camelCase, and ALL_CAPS subwords are all caught.
    assert "normalise" in _flagged("def normalise_flux(x): ...")
    assert "marginalised" in _flagged("class CalibrationMarginalisedLikelihood: ...")
    assert "catalogue" in _flagged("SSP_CATALOGUE_URL = '...'")


def test_allowlisted_data_contract_key_not_flagged():
    key = "log10_specific_ionising_luminosity"
    assert key in mod.ALLOWED_TOKENS
    assert not _flagged(f'f["{key}"]["HI"]')
    # but the same British root elsewhere IS flagged
    assert "ionising" in _flagged("the ionising photon rate")


def test_fix_text_is_case_preserving():
    assert mod.fix_text("Colour and colour and COLOUR") == "Color and color and COLOR"
    assert mod.fix_text("CalibrationMarginalisedLikelihood") == "CalibrationMarginalizedLikelihood"
    # allowlisted key is left untouched by --fix
    key = "log10_specific_ionising_luminosity"
    assert mod.fix_text(f'f["{key}"]') == f'f["{key}"]'
