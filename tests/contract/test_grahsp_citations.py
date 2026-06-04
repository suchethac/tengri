# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP citation wiring — registry entries + config-aware component keys.

Regression for the post-#649 polish: the GRAHSP component shipped citing the
QSOgen key (``temple2021_qsogen``, a copy-paste leftover) and the six GRAHSP
sub-model papers were absent from ``references.bib``. These tests pin the
verified registry entries and the config-driven ``citations()`` contract so the
component advertises Buchner+2024 (and only the sub-models it actually uses).

Bibliographic data verified against authoritative sources (ADS / publisher),
not memory — see the registry entries in ``references.bib``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

# (registry_key, expected year, a distinctive title fragment) — all verified.
GRAHSP_KEYS = [
    ("buchner2024", 2024, "Genuine Retrieval of the AGN Host Stellar Population"),
    ("grandi1982", 1982, "3000"),
    ("mor_netzer2012", 2012, "Hot graphite dust"),
    ("veron_cetty2004", 2004, "I Zw 1"),
    ("netzer_trakhtenbrot2014", 2014, "slim accretion discs"),
    ("bruhweiler_verner2008", 2008, "Fe II"),
]


@pytest.mark.parametrize("key, year, title_frag", GRAHSP_KEYS)
def test_grahsp_keys_resolve(key, year, title_frag):
    """Each GRAHSP sub-model key resolves to a populated registry entry."""
    from tengri.citations.registry import cite

    c = cite(key)  # raises KeyError if missing
    assert c.key == key
    assert c.year == year
    assert title_frag in c.title, f"{key}: {title_frag!r} not in {c.title!r}"
    assert c.short and c.role


def test_grahsp_component_citations_default():
    """Default config cites GRAHSP + Balmer (Grandi) + FeII (Bruhweiler)."""
    from tengri.components.agn.grahsp.component import GRAHSPSEDComponent

    keys = GRAHSPSEDComponent().citations()
    assert keys == ("buchner2024", "grandi1982", "bruhweiler_verner2008")
    # The QSOgen copy-paste leftover must not reappear.
    assert "temple2021_qsogen" not in keys


def test_grahsp_component_citations_config_aware():
    """Selecting MN12 torus + Netzer disc + Véron-Cetty FeII swaps the keys."""
    from tengri.components.agn.grahsp.component import (
        GRAHSPSEDComponent,
        GRAHSPSEDComponentConfig,
    )

    cfg = GRAHSPSEDComponentConfig(
        torus_model="mn12", disc_model="netzer", feii_template="veroncetty2004"
    )
    keys = set(GRAHSPSEDComponent(config=cfg).citations())
    assert keys == {
        "buchner2024",
        "grandi1982",
        "veron_cetty2004",
        "mor_netzer2012",
        "netzer_trakhtenbrot2014",
    }
    assert "bruhweiler_verner2008" not in keys  # FeII template switched


def test_grahsp_component_citations_minimal():
    """Disabling every sub-model still cites the GRAHSP paper itself."""
    from tengri.components.agn.grahsp.component import (
        GRAHSPSEDComponent,
        GRAHSPSEDComponentConfig,
    )

    cfg = GRAHSPSEDComponentConfig(include_balmer=False, include_feii=False, include_torus=False)
    assert GRAHSPSEDComponent(config=cfg).citations() == ("buchner2024",)


def test_grahsp_component_citation_keys_all_resolve():
    """Every key the component can emit exists in the registry."""
    from tengri.citations.registry import cite
    from tengri.components.agn.grahsp.component import (
        GRAHSPSEDComponent,
        GRAHSPSEDComponentConfig,
    )

    cfg = GRAHSPSEDComponentConfig(
        torus_model="mn12", disc_model="netzer", feii_template="veroncetty2004"
    )
    for key in set(GRAHSPSEDComponent().citations()) | set(
        GRAHSPSEDComponent(config=cfg).citations()
    ):
        cite(key)  # raises KeyError if any emitted key is unregistered
