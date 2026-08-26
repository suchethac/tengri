# SPDX-License-Identifier: BSD-3-Clause
"""Contract conformance for the additive ``SEDModelComponent``s (radio, X-ray).

Every row drives one component through the contract in
``docs/dev/sed-model-components.md``: it is registered under the name it
reports, its declared parameters carry its prefix and units, and ``predict``
called with exactly what it declares publishes exactly what it promises.

Replaces two files that were one skeleton
-----------------------------------------

``test_radio_component.py`` (125 lines) and ``test_xray_component.py`` (124)
shared nine of eleven test names and differed only in the class, the prefix,
the expected output key and a hand-written parameter dict. Both are gone.

The radio predict test was passing on an all-zero SED
------------------------------------------------------

Its parameter dict supplied the six declared parameters and nothing else.
``RadioPowerLawSEDComponent`` declares ``optional_inputs`` of ``L_ir``,
``L_agn_bol`` and ``log_mstar``; with none of them the component falls back to
zero. Its two assertions -- ``sed_out.shape == wave.shape`` and ``"sed_radio"
in published`` -- are both true of an array of zeros, so nothing reported it.

Measured: ``max|sed_out|`` is exactly ``0.0`` with the old dict, and
``1.74e+29`` once the declared optional inputs are supplied. Every row here
supplies them, from ``optional_inputs()`` rather than a hand-written list, and
asserts the result is not identically zero.

Two assertions that could not fail, removed
-------------------------------------------

* ``test_has_required_methods`` was twelve ``hasattr``/``callable`` pairs on
  ``declared_parameters``, ``inputs``, ``outputs``, ``precompute``, ``apply``
  and ``predict``. Every one is defined on ``SEDModelComponent``, so all twelve
  are implied by ``issubclass`` and cannot distinguish any subclass from any
  other. One ``issubclass`` assertion replaces them.
* ``test_has_no_required_inputs`` was named and docstring'd for a property --
  "no required inputs (all are optional with fallbacks)" -- that its body never
  asserted. It checked ``isinstance(inputs_tuple, tuple)`` and stopped. The
  claim is asserted now, along with its other half: the optional inputs *are*
  non-empty, which is what makes the components work at all.

Parameters come from the declarations, not from a copy
-------------------------------------------------------

``predict`` is called with each declared parameter at ``prior.default``, keyed
by the declared name with the prefix stripped -- the dict shape
``SEDModelComponent`` documents. Two things follow from reading the
declarations rather than copying them:

* a renamed parameter either still works or fails here, and never silently
  drifts -- #1738 is that failure in the X-ray coronae, which declared
  ``xray_gamma`` against a parameter named ``xray_gamma_agn``
* the value comes from the declaration too, so it cannot fall outside the
  support the way a hand-picked number can. That is the failure
  ``tools/check_param_defaults.py`` exists for: nine AGN entry points shipped
  ``agn_log_lbol=45.0``, the log10(erg/s) magnitude, against a declaration in
  log10(L/Lsun).

Not covered here
----------------

``radio`` and ``xray`` are the bare-``SEDComponent`` base classes and have no
``predict`` at all -- they are the Protocol style CLAUDE.md reserves for models
that do not fit ``predict(p, sed_in, wave, **inputs)``. ``agn_xray_corona``
declares **no** parameters yet its ``predict`` reads ``p["gamma_agn"]``, so the
driver here raises ``KeyError`` on it; whether that is a defect or an
undocumented sub-block convention where the sibling X-ray component supplies
the value is not decidable from the test side, so it is named rather than
claimed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent
from tengri.protocols.component import DerivedKey, ParamDeclaration

pytestmark = pytest.mark.contract

#: A short SSP age axis for the array-valued optional inputs.
_N_AGE = 10

#: Values for declared ``optional_inputs``. Supplied by name, and
#: ``_optional_inputs_for`` refuses to run if a component declares one this
#: table does not cover -- a missing input is exactly how these components fall
#: back to zero, and a zero SED is what the predecessors were asserting against.
_OPTIONAL_INPUT_VALUES = {
    "L_ir": jnp.array(1e44),  # erg/s
    "L_agn_bol": jnp.array(1e45),  # erg/s
    "log_mstar": jnp.array(10.0),  # log10(M/Msun)
    "L_2500_intrinsic": jnp.array(1e30),  # erg/s/Hz
    "L_2500_30deg": jnp.array(1e30),  # erg/s/Hz
    "sfr": jnp.array(1.0),  # Msun/yr
    "age_weights": jnp.full(_N_AGE, 1e8),  # Msun per SSP age bin
    "ssp_ages_yr": jnp.logspace(6.0, 10.0, _N_AGE),  # yr
    "log_metallicity_history": jnp.full(_N_AGE, -1.848),  # log10(Z) absolute
}

#: (registry key, wavelength grid, minimum declared parameters, published key).
_COMPONENTS = [
    pytest.param("radio_powerlaw", (3, 8), 6, "sed_radio", id="radio_powerlaw"),
    pytest.param("radio_dpl", (3, 8), 6, "sed_radio", id="radio_dpl"),
    pytest.param("xray_aird", (0, 4), 4, "sed_xray", id="xray_aird"),
]


def _wave(decades):
    lo, hi = decades
    return jnp.logspace(lo, hi, 512)


def _representative_value(prior):
    """A value inside the prior's declared support.

    ``prior.default`` is the right source rather than a midpoint computed here:
    every prior carries one, and ``tools/check_param_defaults.py`` is a CI guard
    that it lies inside the support. Reading it means this file cannot drift
    from the declaration the way a hand-picked number would -- which is the same
    failure ``check_param_defaults.py`` exists for (nine AGN entry points shipped
    ``agn_log_lbol=45.0``, the log10(erg/s) magnitude, against a declaration in
    log10(L/Lsun)).
    """
    default = getattr(prior, "default", None)
    if default is not None:
        return float(default)
    lo, hi = prior.bounds
    return 0.5 * (float(lo) + float(hi))


def _params_from_declarations(comp):
    """The dict ``predict`` expects: declared names, prefix stripped, defaults."""
    prefix = comp.parameter_prefix
    params = {
        d.name.removeprefix(prefix): jnp.array(_representative_value(d.prior))
        for d in comp.declared_parameters()
    }
    params["redshift"] = jnp.array(0.1)
    return params


def _optional_inputs_for(comp):
    """Every declared optional input, by name, from the table above."""
    names = [i.name if hasattr(i, "name") else str(i) for i in comp.optional_inputs()]
    missing = [n for n in names if n not in _OPTIONAL_INPUT_VALUES]
    assert not missing, (
        f"{comp.name} declares optional inputs with no value in "
        f"_OPTIONAL_INPUT_VALUES: {missing}. Add them -- leaving one out makes "
        f"the component fall back to zero, which is what this file exists to catch."
    )
    return {n: _OPTIONAL_INPUT_VALUES[n] for n in names}


@pytest.mark.parametrize(("key", "decades", "min_params", "published_key"), _COMPONENTS)
class TestAdditiveComponentContract:
    """One component per row, driven entirely from its own declarations."""

    def test_registry_entry_reports_its_own_name(self, key, decades, min_params, published_key):
        """The class is registered under the name it reports, and is a component.

        The name/registry agreement is what the resolver depends on, and neither
        predecessor checked it -- both asserted ``comp.name == "..."`` against a
        literal, which cannot notice the registry key drifting away from it.

        The ``issubclass`` here replaces twelve ``hasattr``/``callable``
        assertions: every method they checked is defined on the base class.
        """
        cls = _REGISTRY[key]
        assert issubclass(cls, SEDModelComponent)
        assert cls().name == key, (
            f"registered under {key!r} but reports {cls().name!r}; the resolver "
            f"looks up by registry key"
        )

    def test_declared_parameters_carry_the_prefix_and_units(
        self, key, decades, min_params, published_key
    ):
        """Every declared parameter is a ParamDeclaration, prefixed, with units."""
        comp = _REGISTRY[key]()
        declared = comp.declared_parameters()

        assert len(declared) >= min_params, (
            f"{key} declares {len(declared)} parameters, fewer than the {min_params} "
            f"this contract expects; a parameter disappearing is a silent API break"
        )
        for d in declared:
            assert isinstance(d, ParamDeclaration), f"{d!r} is not a ParamDeclaration"
            assert d.name.startswith(comp.parameter_prefix), (
                f"{d.name} does not start with {comp.parameter_prefix!r} (NAMING_CONTRACT 3.2)"
            )
            assert d.units is not None, f"{d.name} has no units"
            assert d.description, f"{d.name} has no description"

    def test_inputs_are_empty_and_optional_inputs_are_not(
        self, key, decades, min_params, published_key
    ):
        """Required inputs: none. Optional inputs: some.

        The predecessor asserted the first half's *type* and neither half's
        content, under a name that promised both.
        """
        comp = _REGISTRY[key]()

        required = comp.inputs()
        assert isinstance(required, tuple)
        assert required == (), (
            f"{key} now has required inputs {required}; this component family is "
            f"documented as taking only optional ones, with fallbacks"
        )
        assert comp.optional_inputs(), (
            f"{key} declares no optional inputs, so nothing can drive it and its "
            f"SED can only be zero"
        )

    def test_declared_outputs_are_all_published(self, key, decades, min_params, published_key):
        """predict() publishes every key outputs() declares, and a real SED.

        ``outputs()`` is a promise the pipeline relies on to wire components
        together; a declared key that ``predict`` never publishes is a
        ``KeyError`` in whichever component consumes it downstream.
        """
        comp = _REGISTRY[key]()
        wave = _wave(decades)

        sed_out, published = comp.predict(
            _params_from_declarations(comp),
            jnp.zeros_like(wave),
            wave,
            **_optional_inputs_for(comp),
        )

        declared = {o.name for o in comp.outputs()}
        assert declared, f"{key} declares no outputs"
        missing = declared - set(published)
        assert not missing, f"{key} declares {sorted(missing)} in outputs() but never publishes it"
        assert published_key in published, f"{key} should publish {published_key}"

        arr = np.asarray(sed_out)
        assert arr.shape == wave.shape, f"{key}: SED shape {arr.shape} != wave shape {wave.shape}"
        assert np.all(np.isfinite(arr)), f"{key}: SED holds non-finite values"
        assert np.max(np.abs(arr)) > 0.0, (
            f"{key}: SED is identically zero even with every declared optional "
            f"input supplied. The predecessor asserted only shape and key "
            f"presence, both of which a zero array satisfies."
        )

    def test_published_sed_declares_luminosity_units(
        self, key, decades, min_params, published_key
    ):
        """The published SED key is declared in erg/s/Hz.

        CLAUDE.md: all SED components return erg/s/Hz (standardized
        2026-04-08). A component publishing a flux here would be wrong by the
        cosmological dimming factor and nothing downstream would notice.
        """
        comp = _REGISTRY[key]()
        declared = {o.name: o for o in comp.outputs()}

        assert published_key in declared, f"{key} does not declare {published_key} in outputs()"
        out = declared[published_key]
        assert isinstance(out, DerivedKey)
        assert out.units == "erg/s/Hz", (
            f"{key}: {published_key} declares units {out.units!r}, not 'erg/s/Hz'"
        )

    def test_precompute_returns_a_named_state(self, key, decades, min_params, published_key):
        """precompute() returns a state carrying this component's name."""
        comp = _REGISTRY[key]()
        state = comp.precompute()

        assert state.name == key, (
            f"{key}.precompute() returned a state named {state.name!r}; the "
            f"pipeline keys component state by name"
        )
