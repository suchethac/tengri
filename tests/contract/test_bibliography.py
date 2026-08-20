# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :class:`tengri.citations.Bibliography` — the live citation container."""

from __future__ import annotations

import pytest

from tengri.citations import Bibliography

pytestmark = pytest.mark.contract

# ---------------------------------------------------------------------------
# Basic mutation / query
# ---------------------------------------------------------------------------


def test_empty_bibliography_is_falsy():
    bib = Bibliography()
    assert not bib
    assert len(bib) == 0
    assert bib.keys == []
    assert list(bib) == []


def test_add_chainable_and_deduplicates():
    bib = Bibliography().add("tengri").add("jax", "dsps").add("tengri")
    assert bib.keys == ["tengri", "jax", "dsps"]
    assert len(bib) == 3


def test_add_ignores_empty_strings():
    bib = Bibliography().add("", "tengri", "")
    assert bib.keys == ["tengri"]


def test_extend_from_bibliography():
    a = Bibliography().add("tengri")
    b = Bibliography().add("jax", "dsps")
    a.extend(b)
    assert a.keys == ["tengri", "jax", "dsps"]


def test_extend_from_iterable():
    bib = Bibliography().add("tengri")
    bib.extend(["jax", "dsps", "tengri"])
    assert bib.keys == ["tengri", "jax", "dsps"]


def test_remove():
    bib = Bibliography().add("tengri", "jax", "dsps")
    bib.remove("jax")
    assert bib.keys == ["tengri", "dsps"]
    bib.remove("not_present")  # silent no-op
    assert bib.keys == ["tengri", "dsps"]


def test_contains_membership():
    bib = Bibliography().add("tengri")
    assert "tengri" in bib
    assert "jax" not in bib


def test_iter_yields_citation_records():
    from tengri.citations import Citation

    bib = Bibliography().add("tengri", "dsps")
    for c in bib:
        assert isinstance(c, Citation)


def test_iter_skips_unknown_keys():
    bib = Bibliography().add("tengri", "not_a_real_key", "dsps")
    cites = list(bib)
    assert len(cites) == 2  # missing key silently skipped


def test_str_preview():
    bib = Bibliography().add("tengri", "jax", "dsps", "calzetti2000", "nifty")
    s = str(bib)
    assert "Bibliography" in s
    assert "tengri" in s
    assert "+1 more" in s  # 5 keys, preview caps at 4


def test_str_empty():
    assert str(Bibliography()) == "Bibliography(empty)"


# ---------------------------------------------------------------------------
# Category grouping
# ---------------------------------------------------------------------------


def test_by_category_returns_mapping_of_lists():
    bib = Bibliography().add("tengri", "dsps", "calzetti2000", "charlot_fall2000")
    groups = bib.by_category()
    # Every value is a list of Citation
    for _cat, cites in groups.items():
        assert isinstance(cites, list)
        assert cites  # not empty

    assert "framework" in groups
    assert "ssp" in groups
    assert "dust_attenuation" in groups
    assert {c.key for c in groups["dust_attenuation"]} == {
        "calzetti2000",
        "charlot_fall2000",
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_report_grouped_default():
    bib = Bibliography().add("tengri", "dsps", "calzetti2000")
    report = bib.report()
    assert "Please cite the following" in report
    assert "Framework & theory" in report
    assert "Stellar populations" in report
    assert "Dust attenuation" in report
    assert "Calzetti" in report


def test_report_flat():
    bib = Bibliography().add("tengri", "dsps")
    flat = bib.report(group_by_category=False)
    # Numbered list, no category headings
    assert "[1]" in flat
    assert "[2]" in flat
    assert "Framework & theory" not in flat


def test_report_empty():
    assert "No citations" in Bibliography().report()


def test_to_bibtex_roundtrip():
    bib = Bibliography().add("calzetti2000")
    bibtex = bib.to_bibtex()
    assert "@article" in bibtex
    assert "Calzetti" in bibtex
    assert "10.1086/308692" in bibtex


def test_to_bibtex_multiple_separated_by_blank_line():
    bib = Bibliography().add("calzetti2000", "charlot_fall2000")
    bibtex = bib.to_bibtex()
    assert bibtex.count("@article") == 2
    assert "\n\n" in bibtex  # separator


# ---------------------------------------------------------------------------
# from_config classmethod
# ---------------------------------------------------------------------------


def test_from_config_none_yields_core_only():
    bib = Bibliography.from_config(None)
    assert bib.keys == ["tengri", "jax", "dsps"]


def test_from_config_calzetti_two_component():
    from tengri.config.settings import DustConfig, SEDModelConfig

    mc = SEDModelConfig(
        dust_attenuation=DustConfig(model="two_component", law_bc="calzetti", law_diff="power_law")
    )
    bib = Bibliography.from_config(mc)
    assert "charlot_fall2000" in bib.keys  # from two_component
    assert "calzetti2000" in bib.keys  # from law_bc


def test_from_config_source_label_preserved():
    bib = Bibliography.from_config(None, source="my model")
    assert bib.source == "my model"


# ---------------------------------------------------------------------------
# add_backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend,expected_key",
    [
        ("mcmc_nuts", "blackjax"),
        ("nuts", "blackjax"),
        ("pathfinder", "pathfinder"),
        ("mcmc_raytrace", "raytrace_behroozi"),
        ("evidence", "nss"),
        ("ess", "ess_murray2010"),
        ("elliptical_slice", "ess_murray2010"),
        ("vi", "nifty"),
    ],
)
def test_add_backend_maps_correctly(backend: str, expected_key: str):
    bib = Bibliography()
    bib.add_backend(backend)
    assert expected_key in bib.keys, f"{backend} should add {expected_key}; got {bib.keys}"


def test_add_backend_none_is_noop():
    bib = Bibliography()
    bib.add_backend(None)
    assert bib.keys == []


def test_add_backend_unknown_is_noop():
    bib = Bibliography()
    bib.add_backend("some_unknown_backend")
    assert bib.keys == []


# ---------------------------------------------------------------------------
# from_object classmethod
# ---------------------------------------------------------------------------


def test_from_object_returns_existing_bibliography_verbatim():
    """If an object already has a Bibliography, return it — don't rebuild."""

    class Fake:
        pass

    obj = Fake()
    custom = Bibliography().add("tengri", "some_sentinel_key")
    obj.bibliography = custom

    result = Bibliography.from_object(obj)
    assert result is custom


def test_from_object_falls_back_to_model_config():
    from tengri.config.settings import DustConfig, ModelConfig

    class Fake:
        model_config = ModelConfig(dust_attenuation=DustConfig(law_bc="calzetti"))
        _last_backend = None
        preset_name = "starforming"

    bib = Bibliography.from_object(Fake())
    assert "calzetti2000" in bib.keys
    assert "Fake(preset=starforming)" in bib.source


def test_from_object_picks_up_backend_citation():
    from tengri.config.settings import SEDModelConfig

    class Fake:
        model_config = SEDModelConfig()
        _last_backend = "mcmc_nuts"

    bib = Bibliography.from_object(Fake())
    assert "blackjax" in bib.keys


def test_from_object_honors_include_backend_false():
    from tengri.config.settings import SEDModelConfig

    class Fake:
        model_config = SEDModelConfig()
        _last_backend = "mcmc_nuts"

    bib = Bibliography.from_object(Fake(), include_backend=False)
    assert "blackjax" not in bib.keys
