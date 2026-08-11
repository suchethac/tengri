# SPDX-License-Identifier: BSD-3-Clause
"""Contract: a warning that reports a computed number must carry it (#1645).

A warn site that formats a quantity into prose and discards the value forces
every consumer to regex-parse the message — and to accept whatever the format
spec rounded it to. ``SFHBeforeBigBangWarning`` rendered ``{frac:.0%}``, so
"69%" was anything in 0.685-0.695, and the mock builder that needed the number
could not get it.

An AST census over ``src/tengri`` found **12** warn sites with that shape, so the
fix is one mechanism plus a guard, not twelve patches:

* :func:`tengri.config.exceptions.warn_measured` attaches the exact values to the
  warning instance, under a uniform ``measurements`` dict and as attributes;
* ``tools/check_warning_payloads.py`` fails when a warn site renders a rounded
  number without carrying it.

The census had to be AST-based. ``warnings.warn(`` and its f-string sit on
different lines in every multi-line call, so the obvious line-based grep reports
**zero** hits and looks like a clean codebase.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.config.exceptions import measurements_of, warn_measured

pytestmark = pytest.mark.contract


class _Probe(UserWarning):
    pass


def _emit(**kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_measured("something happened", _Probe, **kwargs)
    return caught[0].message


class TestItCarriesTheValues:
    def test_the_value_is_exact_not_the_rendered_precision(self):
        """The whole point: the message may round, the payload may not."""
        w = _emit(fraction=0.6916830115613221)
        assert w.fraction == 0.6916830115613221

    def test_values_are_reachable_uniformly(self):
        """A consumer that does not know the attribute name still gets them."""
        w = _emit(fraction=0.25, grad_norm=1234.5)
        assert measurements_of(w) == {"fraction": 0.25, "grad_norm": 1234.5}

    def test_the_category_and_message_are_unchanged(self):
        """Attaching data must not alter what a user sees or filters on."""
        w = _emit(fraction=0.1)
        assert isinstance(w, _Probe)
        assert str(w) == "something happened"

    def test_no_measurements_is_allowed(self):
        """Not every warning reports a number; the helper stays usable."""
        assert measurements_of(_emit()) == {}


class TestItReadsAnyWarning:
    def test_measurements_of_a_plain_warning_is_empty(self):
        """Consumers must not need to know whether a site was migrated yet."""
        assert measurements_of(UserWarning("plain")) == {}

    def test_measurements_of_a_non_warning_is_empty(self):
        assert measurements_of(None) == {}
        assert measurements_of("not a warning") == {}


class TestItRefusesToCorruptTheWarning:
    @pytest.mark.parametrize("reserved", ["args", "with_traceback", "measurements"])
    def test_a_reserved_name_raises(self, reserved):
        """Silently shadowing ``args`` would break ``str(w)`` and pickling. A
        measurement name that collides must fail loudly at the raise site."""
        with pytest.raises(ValueError, match="reserved"):
            _emit(**{reserved: 1.0})

    def test_a_non_numeric_measurement_raises(self):
        """The payload is for numbers a consumer will compare or threshold.
        Prose belongs in the message, where it is already."""
        with pytest.raises(TypeError, match="numeric"):
            _emit(fraction="69%")


class TestTheGuardCatchesRegressions:
    """``tools/check_warning_payloads.py`` is what keeps site 13 from appearing.

    It reported 12 before this migration and 0 after; these pin that it can
    still go red, so it cannot quietly become decoration.
    """

    @staticmethod
    def _violations(source):
        import ast
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "guard", Path(__file__).resolve().parents[2] / "tools" / "check_warning_payloads.py"
        )
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        tree = ast.parse(source)
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and guard._is_bare_warn(node) and node.args:
                found.extend(guard._rounded_placeholders(node.args[0]))
        return found

    def test_a_bare_warn_with_a_rounded_number_is_flagged(self):
        source = 'warnings.warn(f"lost {frac:.0%} of it", SomeWarning, stacklevel=2)'
        assert self._violations(source) == [".0%"]

    def test_warn_measured_is_not_flagged(self):
        """The remedy must not itself trip the guard, or migration is impossible."""
        source = 'warn_measured(f"lost {frac:.0%}", SomeWarning, truncated_fraction=frac)'
        assert self._violations(source) == []

    def test_a_message_with_no_number_is_not_flagged(self):
        """Most warnings report no quantity; they must stay untouched."""
        source = 'warnings.warn("something qualitative happened", SomeWarning)'
        assert self._violations(source) == []

    def test_a_placeholder_without_rounding_is_not_flagged(self):
        """``{n}`` loses nothing, so there is no discarded precision to carry."""
        source = 'warnings.warn(f"{n} bands affected", SomeWarning)'
        assert self._violations(source) == []


class TestStacklevel:
    def test_the_helpers_own_frame_is_skipped(self):
        """``stacklevel`` counts from the caller exactly as ``warnings.warn``
        counts it, so a site migrating from ``warn(..., stacklevel=N)`` keeps
        the same number AND the same attribution. Here ``stacklevel=1`` must
        blame this file; if the helper's frame leaked, every migrated warning
        would point at ``exceptions.py`` instead of the physics that raised it."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_measured("x", _Probe, stacklevel=1, fraction=0.5)
        assert caught[0].filename == __file__

    def test_a_higher_stacklevel_points_further_out(self):
        """The default of 2 blames the caller's caller, which from a test
        function is pytest itself — proof the offset is applied, not ignored."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_measured("x", _Probe, fraction=0.5)
        assert caught[0].filename != __file__
