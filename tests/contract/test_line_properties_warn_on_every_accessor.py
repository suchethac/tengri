# SPDX-License-Identifier: BSD-3-Clause
"""A NaN line luminosity must say why, whichever accessor you reach it through.

``BakedInBackend`` and the shock backends publish no per-line luminosity
catalog, so all 17 ``lines`` properties come back NaN. #361 added a warning
that names the fix (``neb={'type': 'cue'}`` …) — but only on ``pred.lines.*``.
Measured on a ``neb={'type': 'none'}`` model:

===========================================  ======  =======
accessor                                     value   warned
===========================================  ======  =======
``pred.lines.halpha``                        NaN     yes
``pred.properties["halpha"]``                NaN     **no**
``predict_properties(names=("halpha",))``    NaN     **no**
===========================================  ======  =======

The two silent ones include ``predict_properties`` — the surface the naming
contract calls *"the ONE jit/vmap surface for derived quantities"*, i.e. the
one a fitting loop or a mock-catalog script actually uses. A NaN there
propagates into a likelihood or a catalog column with nothing said.

One helper, ``warn_if_lines_are_unavailable``, is now called by all three, and
the inline copy in ``_ensure_lines`` is gone — with two warnings for one
question, whichever got edited next would drift from the other.

The set of "line properties" is read off the registry (``group == "lines"``),
not listed here, so a new diagnostic is covered the moment it is registered.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
from tengri.forward.properties import PROPERTY_REGISTRY, line_property_names

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_WARNING_MARKER = "per-line luminosity catalog"


@pytest.fixture(scope="module")
def no_lines_model(ssp_data_fsps):
    """A model whose nebular backend publishes no per-line catalog."""
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )


def _line_warnings(fn) -> list[str]:
    """Run ``fn`` and return only the per-line-catalog warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if _WARNING_MARKER in str(w.message)]


_ACCESSORS = {
    "pred.lines.halpha": lambda m, p: m.predict(p).lines.halpha,
    "pred.properties['halpha']": lambda m, p: m.predict(p).properties["halpha"],
    "predict_properties(names=('halpha',))": lambda m, p: m.predict_properties(
        p, names=("halpha",)
    )["halpha"],
}


class TestTheCensus:
    def test_the_registry_defines_the_line_group(self):
        """A hand-written list would go stale; this reads the registry."""
        names = line_property_names()
        assert len(names) >= 10, (
            f"only {len(names)} properties are in the 'lines' group — the "
            f"group key or the registry changed, and this file's premise with it."
        )
        assert "halpha" in names and "stellar_mass" not in names

    def test_every_line_property_really_is_nan_here(self, no_lines_model):
        """The warning is only owed because the values are unusable."""
        params = dict(no_lines_model.spec.get_fixed_values())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            props = no_lines_model.predict(params).properties
            values = {n: float(props[n]) for n in sorted(line_property_names())}
        finite = {n: v for n, v in values.items() if np.isfinite(v)}
        assert not finite, (
            f"{sorted(finite)} are finite on a backend with no line catalog, so "
            f"warning about them would be wrong. Re-check the fixture."
        )


class TestEveryAccessorWarns:
    @pytest.mark.parametrize("label", sorted(_ACCESSORS))
    def test_it_says_why_the_value_is_nan(self, label, no_lines_model):
        params = dict(no_lines_model.spec.get_fixed_values())
        messages = _line_warnings(lambda: _ACCESSORS[label](no_lines_model, params))
        assert messages, (
            f"{label} returns NaN and says nothing. A silent NaN reaching a "
            f"likelihood or a catalog column is the failure mode #361 exists for."
        )

    @pytest.mark.parametrize("label", sorted(_ACCESSORS))
    def test_it_names_a_backend_that_would_fix_it(self, label, no_lines_model):
        params = dict(no_lines_model.spec.get_fixed_values())
        messages = _line_warnings(lambda: _ACCESSORS[label](no_lines_model, params))
        assert any("cue" in m for m in messages), (
            f"{label} warns without naming a backend that produces lines: {messages}"
        )

    @pytest.mark.parametrize("label", sorted(_ACCESSORS))
    def test_it_warns_once_not_twice(self, label, no_lines_model):
        """Two warnings for one question is how the two copies drifted."""
        params = dict(no_lines_model.spec.get_fixed_values())
        messages = _line_warnings(lambda: _ACCESSORS[label](no_lines_model, params))
        assert len(messages) == 1, (
            f"{label} raised {len(messages)} per-line-catalog warnings; there is "
            f"one helper, so there should be one warning: {messages}"
        )


class TestItStaysQuietWhenItShould:
    def test_a_non_line_property_does_not_warn(self, no_lines_model):
        params = dict(no_lines_model.spec.get_fixed_values())
        messages = _line_warnings(
            lambda: no_lines_model.predict_properties(params, names=("stellar_mass",))
        )
        assert not messages, f"asking for stellar_mass warned about emission lines: {messages}"

    def test_the_default_everything_call_does_not_warn(self, no_lines_model):
        """``names=None`` means "whatever this model has", not a line request.

        Warning there would fire on every default ``predict_properties`` call
        for every model without a line backend — noise that trains users to
        filter the warning that matters.
        """
        params = dict(no_lines_model.spec.get_fixed_values())
        messages = _line_warnings(lambda: no_lines_model.predict_properties(params))
        assert not messages, (
            f"the default predict_properties() call warned about lines: {messages}"
        )

    def test_a_backend_with_a_line_catalog_does_not_warn(self, no_lines_model):
        """Guard the converse, so the warning cannot be satisfied by always firing."""
        from tengri.forward.properties import warn_if_lines_are_unavailable

        class _HasLines:
            def predict_nebular_line_luminosities(self):  # pragma: no cover - probe
                return None

        class _Model:
            _nebular_backend = _HasLines()

        messages = _line_warnings(
            lambda: warn_if_lines_are_unavailable(_Model(), ("halpha", "hbeta"))
        )
        assert not messages, (
            f"a backend that publishes a line catalog still triggered the warning: {messages}"
        )


class TestTheInlineCopyIsGone:
    def test_prediction_does_not_warn_on_its_own(self):
        """One question, one warning site.

        ``_ensure_lines`` kept its own copy of this message. All 17 of its call
        sites read ``properties[...]`` on the next line, so the shared helper
        covers them; leaving the copy in place fired it twice and gave the next
        edit two places to land.
        """
        import inspect

        from tengri.forward.prediction import Prediction

        source = inspect.getsource(Prediction._ensure_lines)
        assert _WARNING_MARKER not in source or "warnings.warn" not in source, (
            "Prediction._ensure_lines still raises its own per-line-catalog "
            "warning; route it through warn_if_lines_are_unavailable instead."
        )

    def test_the_helper_is_reachable_from_the_registry_module(self):
        from tengri.forward import properties

        assert hasattr(properties, "warn_if_lines_are_unavailable")
        assert "warn_if_lines_are_unavailable" in properties.__all__
        assert PROPERTY_REGISTRY, "an empty registry would make this file vacuous"
