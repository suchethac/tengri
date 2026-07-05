# SPDX-License-Identifier: BSD-3-Clause
"""Guard test: every concrete SEDComponent implements citations().

The ``SEDComponent.citations()`` Protocol method was made optional by commit
8c06e142 to work around a runtime_checkable subtlety. This test prevents
that regression — every concrete component must explicitly implement
citations() returning a tuple[str, ...], even if empty. The empty tuple is
the documented signal "this wrapper has no structurally-mandatory paper;
per-config citations come from tengri.citations.associations".

Why this matters: silently allowing a missing citations() encourages
components to ship without provenance. The Protocol contract — even when
the return value is () — forces every adapter to make an explicit choice.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.unit]


def _instantiate(cls):
    """Best-effort no-arg instantiation; some components need a Config."""
    try:
        return cls()
    except TypeError:
        # Component requires a Config positional/kw arg; rely on its default_factory
        return cls()


def test_every_concrete_component_implements_citations() -> None:
    """Every shipped concrete SEDComponent must have a callable citations()
    returning a tuple of str. Empty tuple is allowed (and meaningful).
    """
    from tengri.components.agn.component import AGNSEDComponent
    from tengri.components.agn.grahsp.component import GRAHSPSEDComponent
    from tengri.components.dust.component import DustAttenuationSEDComponent
    from tengri.components.dust.emission.analytic.modified_blackbody import (
        ModifiedBlackbodyIRSEDComponent,
    )
    from tengri.components.dust.two_component import DustSEDComponent
    from tengri.components.igm.component import IGMSEDComponent
    from tengri.components.nebular.component import NebularSEDComponent
    from tengri.components.radio.component import RadioSEDComponent
    from tengri.components.stellar.component import StellarSEDComponent
    from tengri.components.xray.component import XRaySEDComponent

    concrete = (
        AGNSEDComponent,
        DustAttenuationSEDComponent,
        ModifiedBlackbodyIRSEDComponent,
        DustSEDComponent,
        GRAHSPSEDComponent,
        IGMSEDComponent,
        NebularSEDComponent,
        RadioSEDComponent,
        StellarSEDComponent,
        XRaySEDComponent,
    )

    for cls in concrete:
        inst = _instantiate(cls)
        assert hasattr(inst, "citations"), f"{cls.__name__}: missing citations()"
        cits = inst.citations()
        assert isinstance(cits, tuple), (
            f"{cls.__name__}.citations() returned {type(cits).__name__}, expected tuple"
        )
        assert all(isinstance(k, str) for k in cits), (
            f"{cls.__name__}.citations() must contain only str bib keys"
        )


def test_protocol_advertises_citations_attribute() -> None:
    """The SEDComponent Protocol must declare citations()."""
    from tengri.protocols.component import SEDComponent

    assert hasattr(SEDComponent, "citations"), (
        "SEDComponent Protocol no longer declares citations() — "
        "regression of commit 8c06e142. Restore the required method."
    )


def test_known_structural_citations_present() -> None:
    """Components with always-required structural papers actually return them."""
    from tengri.components.dust.two_component import DustSEDComponent
    from tengri.components.stellar.component import StellarSEDComponent

    assert "charlot_fall2000" in DustSEDComponent().citations(), (
        "DustSEDComponent structurally implements Charlot & Fall (2000)"
    )
    assert "dsps" in StellarSEDComponent().citations(), (
        "StellarSEDComponent is structurally built on DSPS"
    )
