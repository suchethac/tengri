# SPDX-License-Identifier: BSD-3-Clause
"""Contract: a component's declared citations reach ``citations()`` (#1777).

``SEDModelComponent.__init_subclass__`` built the tuple from one spelling and
then wrote the result to the other::

    citations_attr = vars(cls).get("citations", ())
    cls._citations_tuple = citations_attr  # unconditional

So a subclass that declared ``_citations_tuple = (...)`` in its class body had
its keys overwritten with ``()`` at class creation — silently, because the
attribute it wrote is exactly the one that line clobbers. **Thirteen of the
fifteen components that surface citations used that spelling**, which is every
dust emission backend in the library: ``astrodust``, ``bosa``, ``casey2012``,
``dale2014``, ``dale2014_cigale``, ``dh02_ce01``, ``draine_li2007``,
``draine_li2014``, ``modified_blackbody``, ``pah_drude``, ``schreiber2016``,
``schreiber2018``, ``themis``. ``component.citations()`` returned nothing for
all of them.

The two that worked (``two_component``, ``wg00``) override the ``citations()``
*method*, a third mechanism that never went through the clobbered attribute —
which is why nothing looked wrong from the outside.

Un-breaking it surfaced a second layer: **seven of the newly-live keys did not
resolve**, because a key nobody ever looks up cannot be found wrong.
``da_cunha2013`` (three components) is spelled ``dacunha2013`` in
``references.bib``; ``draine_li2014`` is ``draine2014``; and
``boquien_salim2021``, ``schreiber2018`` and ``draine2011`` had no entry at
all. Both layers are pinned below, because fixing only the first would have
turned silent attribution loss into visible "(no bib entry)" comments.
"""

from __future__ import annotations

import pytest

import tengri.components.dust.emission  # noqa: F401  (registers the backends)
from tengri.citations import cite
from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _declaring_components():
    return sorted((name, cls) for name, cls in _REGISTRY.items() if tuple(cls().citations()))


class TestBothSpellingsSurvive:
    """Pins the seam itself, so no census can be too narrow for it."""

    def test_the_documented_spelling_survives(self):
        class _ViaCitations(SEDModelComponent):
            name = "_probe_via_citations"
            citations = ("calzetti2000",)

        assert tuple(_ViaCitations().citations()) == ("calzetti2000",)

    def test_the_private_spelling_survives(self):
        """The one that was dropped. Reverting the fix fails exactly here."""

        class _ViaTuple(SEDModelComponent):
            name = "_probe_via_tuple"
            _citations_tuple = ("calzetti2000",)

        assert tuple(_ViaTuple().citations()) == ("calzetti2000",), (
            "a class-body _citations_tuple was overwritten at class creation; "
            "13 components declared citations this way and surfaced none."
        )

    def test_declaring_nothing_still_yields_empty(self):
        class _Silent(SEDModelComponent):
            name = "_probe_silent"

        assert tuple(_Silent().citations()) == ()


class TestEveryDeclaredKeyResolves:
    @pytest.mark.parametrize("name", [n for n, _ in _declaring_components()])
    def test_the_keys_are_real(self, name):
        for key in _REGISTRY[name]().citations():
            try:
                cite(key)
            except KeyError:  # pragma: no cover - the assertion is the report
                pytest.fail(
                    f"component {name!r} cites {key!r}, which is not in "
                    f"references.bib. A key nobody looks up cannot be found "
                    f"wrong, so this only became visible once the declarations "
                    f"stopped being discarded."
                )


class TestTheCensusIsComplete:
    def test_the_dust_emission_backends_all_cite_something(self):
        """The population that was entirely dark.

        Derived from the menu rather than listed, so a new backend joins it.
        """
        import tengri
        from tengri.forward.component_factory import _EMISSION_TYPE_ALIASES

        # Resolve aliases. Without this the census silently skips any menu name
        # whose registry entry is spelled differently — draine2021_pah ->
        # draine2021_pah_ir is exactly that case, and it was the one component
        # this test failed to cover on its first draft.
        advertised = {
            _EMISSION_TYPE_ALIASES.get(r["name"], r["name"])
            for r in tengri.list_dust_emission_models()
        }
        silent = sorted(
            n for n in advertised if n in _REGISTRY and not tuple(_REGISTRY[n]().citations())
        )
        assert not silent, (
            f"{silent} are advertised dust emission models that cite nobody. "
            f"Every one of them repackages a published template library."
        )

    def test_the_sweep_is_not_vacuous(self):
        declaring = _declaring_components()
        assert len(declaring) >= 13, (
            f"only {len(declaring)} components report citations; this file was "
            f"written when 15 did, 13 of them newly un-dropped."
        )
