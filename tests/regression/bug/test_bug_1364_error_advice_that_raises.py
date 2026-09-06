# SPDX-License-Identifier: BSD-3-Clause
"""Regression: shipped messages must not instruct the user to write a call that raises.

Five messages told the user to do something that does not work, and two of them sit on
the migration path *off* a deprecated API — so the user met them precisely while doing
what the message asked (#1364).

1. ``lines=LineList([...])`` — advertised in the ``Observation(line_fluxes=...)``
   deprecation warning and in the ``ForwardModel`` "lines not declared" error.
   ``LineList`` is a frozen record of parallel arrays; following the advice raises
   ``TypeError: missing 7 required positional arguments``. The lookup constructor is
   ``LineList.from_names``.
2. ``pass ionizing_source_warning='suppress'`` / ``continuum_warning='suppress'`` —
   advertised by ``BakedInBackend`` and twice by ``CB19Backend``. These are real
   *backend constructor* arguments, but the warnings fire from ``SEDModel.build``, and
   the build grammar does not forward them, so trying it there raises ``TypeError``.

These tests are written to resist going stale: rather than hard-coding the fixed text,
they take the advice **out of the message the code actually emits** and execute it. Edit
a message into something that no longer works and the corresponding test fails.
"""

from __future__ import annotations

import re
import warnings

import pytest

pytestmark = pytest.mark.regression_bug


def _advised_filter(message_text):
    """Pull the ``message='...'`` argument out of an emitted warning's own text.

    Returns the regex the message tells the user to pass to
    ``warnings.filterwarnings``. Fails the test if the message stopped advertising
    one, which is itself the regression.
    """
    m = re.search(r"message='([^']+)'", message_text)
    assert m, (
        "message no longer advertises a warnings.filterwarnings(message='...') route; "
        f"a user building via SEDModel.build has no reachable way to silence it:\n"
        f"{message_text}"
    )
    return m.group(1)


class TestLineListAdviceIsConstructible:
    """The advertised constructor must be the one that works."""

    def test_the_advised_form_actually_constructs(self):
        """LOAD-BEARING: run the advice, do not merely spell-check it.

        Neuter: change the advice back to ``LineList([...])`` in either source
        message and ``test_messages_advertise_the_working_constructor`` below fails.
        """
        from tengri import LineList

        got = LineList.from_names(["Halpha"])
        assert got.names == ("Halpha",)

        # And the form the messages used to advise really does raise -- so the two
        # tests together pin a genuine difference, not a stylistic preference.
        with pytest.raises(TypeError):
            LineList(["Halpha"])

    @pytest.mark.parametrize(
        ("module", "needle"),
        [
            ("tengri.observation.observation", "lines=LineList.from_names(["),
            ("tengri.forward.forward_model", "lines=LineList.from_names(["),
        ],
    )
    def test_messages_advertise_the_working_constructor(self, module, needle):
        """The advice string in the source must name ``from_names``, not the raising form.

        This is a negative source assertion (checking that bad advice is absent),
        which is stronger than a positive one because the deleted construct
        (``LineList([...])`` directly) has a wrong answer (raises TypeError).
        Paired with test_the_advised_form_actually_constructs (positive: the good form
        works), this pins that the error message does not regress to re-advertising
        the bad construct.
        """
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(module))
        assert needle in src, f"{module} no longer advertises {needle!r}"
        assert "lines=LineList([" not in src, (
            f"{module} advertises lines=LineList([...]), which raises TypeError: "
            "LineList is a frozen record of parallel arrays, not a name list. "
            "Use LineList.from_names([...])."
        )


class TestAdvertisedSuppressionActuallySuppresses:
    """Each warning must name a route that works from where the user is standing."""

    def test_baked_in_nebular_warning(self):
        """LOAD-BEARING. The filter is read from the emitted message and executed."""
        from tengri.components.nebular.baked_in import BakedInBackend

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            BakedInBackend()
        assert len(rec) == 1, "probe setup failed: BakedInBackend() did not warn once"
        advice = _advised_filter(str(rec[0].message))

        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message=advice)
            BakedInBackend()
        assert len(rec2) == 0, (
            f"the message advertises filterwarnings(message={advice!r}) but that does "
            "not silence it -- the advice does not work"
        )

    @pytest.mark.parametrize("index", [0, 1])
    def test_cb19_warnings(self, index):
        """Both CB19 messages (ionizing source, missing continuum).

        Uses ``_emit_cb19_warnings`` rather than constructing ``CB19Backend``, which
        would need its CLOUDY grid files; this is the function that owns the text.
        """
        from tengri.components.nebular.cloudy_cb19 import _emit_cb19_warnings

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _emit_cb19_warnings("warn", "warn")
        assert len(rec) == 2, f"probe setup failed: expected 2 warnings, got {len(rec)}"
        advice = _advised_filter(str(rec[index].message))

        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message=advice)
            _emit_cb19_warnings("warn", "warn")
        assert len(rec2) == 1, (
            f"message {index} advertises filterwarnings(message={advice!r}); expected it "
            f"to silence exactly that one warning, leaving 1, but {len(rec2)} remain"
        )

    def test_the_constructor_kwarg_still_works_where_it_is_valid(self):
        """The messages still name the kwarg; it must remain true at the backend layer.

        The kwarg is not wrong — it is unreachable from ``SEDModel.build``. This pins
        that the message's parenthetical is accurate rather than folklore.
        """
        from tengri.components.nebular.baked_in import BakedInBackend
        from tengri.components.nebular.cloudy_cb19 import _emit_cb19_warnings

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            BakedInBackend(ionizing_source_warning="suppress")
            _emit_cb19_warnings("suppress", "suppress")
        assert len(rec) == 0, f"'suppress' no longer suppresses; {len(rec)} warning(s) left"
