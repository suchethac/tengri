# SPDX-License-Identifier: BSD-3-Clause
"""Parity contract: dust validator must accept every registered law / emission model.

Sister to ``test_valid_sfh_types_parity.py``. The dust attenuation curves
in ``DUST_LAWS`` and the IR emission models in ``DUST_EMISSION_MODELS``
register themselves via decorators / direct dict assignment at module
import time; the dict-grammar validator must derive its accepted set
from those live registries, not maintain a parallel hand-edited copy
(see ADR-0005 / ADR-0008).

Before the registry-derived refactor: 13 of 21 registered dust laws and
2 of 11 registered emission models were silently rejected by the
validator while still working through the legacy ``Parameters(...)``
path. This test pins down that drift.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


def test_valid_dust_laws_mirrors_registry_keys() -> None:
    from tengri.components.dust.attenuation import DUST_LAWS
    from tengri.parameters.groups import _valid_dust_laws

    assert set(_valid_dust_laws()) == set(DUST_LAWS.keys()), (
        "groups._valid_dust_laws() drifted from DUST_LAWS. Per ADR-0005, "
        "the @register_dust_law decorator is the canonical source; the "
        "validator must derive from it."
    )


def test_valid_dust_emission_types_covers_registry_and_lazy_names() -> None:
    """Accepted set = live ``DUST_EMISSION_MODELS`` keys ∪ lazy-registered names.

    ``dl07_tabulated`` and ``draine2021_pah`` register lazily via
    ``register_dl07_tabulated`` / ``register_draine2021_pah_tabulated``
    helpers and won't appear in the live dict until those run. The
    validator must still accept them as legal type values.
    """
    from tengri.components.dust.emission import DUST_EMISSION_MODELS
    from tengri.parameters.groups import _LAZY_DUST_EMISSION_TYPES, _valid_dust_emission_types

    accepted = set(_valid_dust_emission_types())

    # Every eagerly-registered model must be accepted.
    assert set(DUST_EMISSION_MODELS.keys()) <= accepted, (
        "_valid_dust_emission_types() rejects names that DUST_EMISSION_MODELS "
        "already holds. This is the drift the registry-derived refactor "
        "fixes — every eagerly-registered emission model must be accepted."
    )
    # And every lazy name must also be accepted (even before the loader fires).
    assert set(_LAZY_DUST_EMISSION_TYPES) <= accepted


def test_dust_validator_still_rejects_unknown_law_with_suggestion() -> None:
    """The 'did you mean ...?' UX must survive the derive-from-registry change."""
    from tengri.parameters.groups import _translate_dust_attenuation

    with pytest.raises(ValueError, match="Unknown dust law 'calzeti'"):
        _translate_dust_attenuation({"type": "two_component", "law": "calzeti"}, {})  # typo


def test_dust_validator_still_rejects_unknown_emission_with_suggestion() -> None:
    from tengri.parameters.groups import _translate_dust_emission

    with pytest.raises(ValueError, match="Unknown dust_emission type 'modified_blakbody'"):
        _translate_dust_emission(
            {
                "type": "modified_blakbody",  # typo
            },
            {},
        )


def test_dust_law_name_as_type_points_to_law_bc() -> None:
    """A law name used as a dust 'type' must raise a law-vs-type hint (#664).

    The deleted SEDModelComponents (calzetti/salim18/mw/smc) used to
    make ``dust={'type': 'calzetti'}`` a silent no-op. After their removal the
    type is rejected; this pins the *targeted* message that redirects the common
    mistake to the correct ``law_bc``/``law_diff`` grammar instead of a bare
    "Unknown dust type".
    """
    from tengri.parameters.groups import _translate_dust_attenuation

    for law in ("calzetti", "smc", "salim_sbl18"):
        with pytest.raises(ValueError, match=rf"'{law}' is a dust attenuation \*law\*"):
            _translate_dust_attenuation({"type": law, "tau_v": 2.0}, {})


def test_previously_rejected_dust_laws_now_accepted() -> None:
    """Before this refactor, 13 registered laws were silently rejected.

    Spot-check a few that we know to be in the registry but were absent
    from the old hand-maintained ``_VALID_DUST_LAWS`` set, to make sure
    the validator now accepts them end-to-end.
    """
    from tengri.parameters.groups import _translate_dust_attenuation

    for law in ("prevot_smc", "lmc", "wd01_mwrv31", "vw07_bc"):
        result: dict = {}
        # Should not raise.
        _translate_dust_attenuation({"type": "two_component", "law": law}, result)
        assert result["dust_law_bc"] == law


# ── Building blocks: accepted by the parser, refused by the builder ───


def test_standalone_set_is_the_valid_set_minus_the_building_blocks() -> None:
    """The two sets must differ, and differ only by declared building blocks.

    Equality would mean the refusal has no subject; an unexplained difference
    would mean a real model quietly stopped being selectable. Both halves are
    read off ``EmissionComponent.energy_balanced``, so the guard fails rather
    than drifts when a backend changes its declaration.
    """
    from tengri.parameters.groups import (
        _dust_emission_component_class,
        _standalone_dust_emission_types,
        _valid_dust_emission_types,
    )

    valid = _valid_dust_emission_types()
    standalone = _standalone_dust_emission_types()

    assert standalone < valid, (
        "no dust_emission type is refused as a building block, so the refusal "
        "SEDModel.build performs has no subject and every test of it is vacuous"
    )
    for name in valid - standalone:
        cls = _dust_emission_component_class(name)
        assert cls is not None and cls.energy_balanced is False, (
            f"{name!r} is excluded from the standalone set but its component does not "
            "declare energy_balanced = False — the exclusion is not derived from anything"
        )


def test_pah_drude_is_the_declared_building_block() -> None:
    """Pin the specimen, so the guards above cannot go quiet by losing it.

    ``pah_drude`` carries the Smith+2007 aromatic-feature forest with no thermal
    continuum: scaled by ``L_ir``, never renormalized to it. Measured standalone
    at z = 0, ``|int sed_dust_ir dnu| / L_ir = 1.8925e-04``.
    """
    from tengri.parameters.groups import (
        _standalone_dust_emission_types,
        _valid_dust_emission_types,
    )

    assert "pah_drude" in _valid_dust_emission_types(), (
        "pah_drude must stay a *valid* type: the loader cache holds it, the analytic "
        "precompute grid builds it, and the builders menu enumerates every valid name "
        "through the grammar at import time"
    )
    assert "pah_drude" not in _standalone_dust_emission_types()


def test_the_parser_still_accepts_a_building_block() -> None:
    """The refusal must not move into ``parse_groups``.

    ``tengri.builders.dust.emission`` enumerates every accepted type through the
    grammar at *import* time to build its factory menu, so a refusal inside the
    parser makes ``import tengri`` raise. It belongs at ``SEDModel.build``,
    which is also what leaves the flat ``Parameters(...)`` escape hatch — the
    documented way to compose a custom model — working.
    """
    from tengri.parameters.groups import _translate_dust_emission

    result: dict = {}
    _translate_dust_emission({"type": "pah_drude"}, result)
    assert result["dust_emission"] == "pah_drude"


def test_build_refuses_a_building_block_and_says_why(synthetic_ssp_wide) -> None:
    """The user-facing half: the message must name the defect and the way out."""
    from tengri import DEFAULT, Fixed, SEDModel
    from tengri.config.exceptions import ParameterError

    with pytest.raises(ParameterError) as excinfo:
        SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "pah_drude", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )

    message = str(excinfo.value)
    assert "building block" in message
    # The size of the hole, not just the fact of the refusal.
    assert "1.8925e-04" in message
    # ...and at least one thing the user can type instead.
    assert "dale2014" in message
    # A ParameterError is a ValueError, so the existing grammar handlers catch it.
    assert isinstance(excinfo.value, ValueError)
