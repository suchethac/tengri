# SPDX-License-Identifier: BSD-3-Clause
"""``SEDModel.from_config(dust=...)`` only named the birth-cloud screen (#2021).

``build_model_from_config`` did ``if dust != "charlot_fall": spec_kwargs["dust_law_bc"]
= dust`` and never touched ``dust_law_diff`` or ``dust_model``. Because the model
stays ``dust_model="two_component"`` (the ``Parameters`` default), the
low-level inheritance of #1989 filled in ``dust_law_diff`` from
``dust_law_bc`` -- so a one-string ``dust=`` request happened to name both
screens by accident of a default that ``from_config`` never set on purpose.
The diffuse-ISM screen's law was never named explicitly, and the docstring
conflated a *model* name (``"charlot_fall"``) with *law* names
(``"calzetti"``, ``"kl04"``, ...) as though they were the same kind of thing.

``dust=`` is renamed to ``dust_attenuation_law=`` (one law, explicitly applied
to both screens via :func:`tengri.parameters._dust_laws.resolve_dust_screen_laws`,
the same resolver #2224 introduced for the flat ``Parameters(...)`` path) with
a deprecated ``dust=`` alias that warns and forwards. ``"charlot_fall"``
(the shipped default) is documented as an alias for ``"power_law"`` on both
screens, not a law-registry name in its own right.
"""

from __future__ import annotations

import pytest

from tengri.parameters._dust_laws import resolve_from_config_dust_law
from tengri.parameters.defaults import get_from_config_defaults

pytestmark = pytest.mark.regression_bug


class _Captured(Exception):
    """Raised by the stub ``Parameters`` to hand its kwargs back to the test."""

    def __init__(self, kwargs):
        self.kwargs = kwargs
        super().__init__("captured Parameters(**kwargs)")


class _StubParameters:
    """Stand-in for ``tengri.parameters.parameters.Parameters``.

    Raises immediately with the kwargs it was called with, so the test can
    inspect exactly what ``build_model_from_config`` resolved without
    building a real (SSP-backed) ``Parameters`` instance.
    """

    def __init__(self, **kwargs):
        raise _Captured(kwargs)


class _MarkerSSPData:
    """Stand-in for ``SSPData`` -- an ``isinstance`` target, never read."""


@pytest.fixture
def _no_real_construction(monkeypatch):
    """Patch the two things ``build_model_from_config`` would otherwise need.

    Both are imported LOCALLY inside ``build_model_from_config`` (``from
    tengri.parameters.parameters import Parameters`` and ``from
    tengri.components.stellar.sps.dsps_wrapper import SSPData,
    load_ssp_data``), executed fresh on every call, so patching the module
    attributes here is visible to the function without needing to patch its
    own namespace.
    """
    monkeypatch.setattr("tengri.parameters.parameters.Parameters", _StubParameters)
    monkeypatch.setattr("tengri.components.stellar.sps.dsps_wrapper.SSPData", _MarkerSSPData)
    return _MarkerSSPData()


def _build(ssp, **kwargs):
    from tengri.forward.convenience import build_model_from_config

    return build_model_from_config(model_cls=object, ssp=ssp, **kwargs)


class TestResolveFromConfigDustLaw:
    """Pure string resolution, no tengri imports beyond the module itself."""

    def test_charlot_fall_is_an_alias_for_power_law(self):
        assert resolve_from_config_dust_law("charlot_fall") == "power_law"

    def test_other_names_pass_through_unchanged(self):
        assert resolve_from_config_dust_law("calzetti") == "calzetti"
        assert resolve_from_config_dust_law("power_law") == "power_law"
        assert resolve_from_config_dust_law("kl04") == "kl04"


class TestFromConfigNamesBothScreens:
    """``build_model_from_config`` must pass equal, explicit bc/diff laws."""

    def test_dust_attenuation_law_names_both_screens(self, _no_real_construction):
        ssp = _no_real_construction
        with pytest.raises(_Captured) as exc_info:
            _build(ssp, dust_attenuation_law="calzetti")
        kwargs = exc_info.value.kwargs
        assert kwargs["dust_law_bc"] == "calzetti"
        assert kwargs["dust_law_diff"] == "calzetti"

    def test_default_resolves_to_power_law_both_screens(self, _no_real_construction):
        """Unset (``_UNSET``) falls back to the TOML default, itself an alias."""
        ssp = _no_real_construction
        with pytest.raises(_Captured) as exc_info:
            _build(ssp)
        kwargs = exc_info.value.kwargs
        assert kwargs["dust_law_bc"] == "power_law"
        assert kwargs["dust_law_diff"] == "power_law"

    def test_legacy_dust_kwarg_warns_once_and_resolves_identically(self, _no_real_construction):
        ssp = _no_real_construction
        with pytest.warns(DeprecationWarning) as record, pytest.raises(_Captured) as exc_info:
            _build(ssp, dust="calzetti")
        deprecation_warnings = [
            w for w in record.list if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 1
        kwargs = exc_info.value.kwargs
        assert kwargs["dust_law_bc"] == "calzetti"
        assert kwargs["dust_law_diff"] == "calzetti"

    def test_legacy_and_new_kwarg_disagreeing_raises(self, _no_real_construction):
        ssp = _no_real_construction
        with pytest.raises(ValueError) as exc_info:
            _build(ssp, dust="calzetti", dust_attenuation_law="smc")
        message = str(exc_info.value)
        assert "calzetti" in message
        assert "smc" in message


class TestFromConfigDefaults:
    def test_default_key_is_dust_attenuation_law(self):
        defs = get_from_config_defaults()
        assert defs["dust_attenuation_law"] == "charlot_fall"

    def test_dust_key_is_gone(self):
        defs = get_from_config_defaults()
        assert "dust" not in defs
