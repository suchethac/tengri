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

Two routes are advertised today, and each test executes the one its own message
carries. ``BakedInBackend``'s advisory names the grammar spelling
``neb={'type': 'ssp'}`` -- R49 replaced its ``filterwarnings`` route with that, so the
advice to execute is now a ``SEDModel.build``, not a filter -- while the CB19 messages
still name ``warnings.filterwarnings(message=...)``.

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


def _advised_neb_group(message_text):
    """Pull the ``neb={'type': '...'}`` spelling out of an emitted warning's text.

    The grammar counterpart of :func:`_advised_filter`: R49 replaced this
    advisory's ``filterwarnings`` route with an explicit ``neb=`` spelling, so
    the executable advice is now a build keyword. Returns it as the dict to
    pass. Fails the test if the message stopped advertising one, which is
    itself the regression.
    """
    m = re.search(r"neb=\{'type': '([a-z_]+)'\}", message_text)
    assert m, (
        "message no longer advertises a neb={'type': '...'} route; a user "
        "building via SEDModel.build has no reachable way to silence it:\n"
        f"{message_text}"
    )
    return {"type": m.group(1)}


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

    def test_baked_in_nebular_warning(self, synthetic_ssp_wide):
        """LOAD-BEARING. The spelling is read from the emitted message and BUILT.

        R49 replaced this advisory's ``warnings.filterwarnings(message=...)``
        route with an explicit grammar spelling, ``neb={'type': 'ssp'}``, so
        the route to execute is a build, not a filter. Same purpose as before
        -- run the advice the message gives rather than spell-check it -- and
        the same shape: the spelling is parsed out of the text the code emits,
        so editing the message into something that does not work fails here.
        """
        from tengri import DEFAULT, Fixed, SEDModel
        from tengri.components.nebular.baked_in import BakedInBackend, BakedInNebularWarning

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            BakedInBackend()
        assert len(rec) == 1, "probe setup failed: BakedInBackend() did not warn once"
        neb_group = _advised_neb_group(str(rec[0].message))

        def build(**extra):
            return SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.1),
                **extra,
            )

        # The silent default still advises: exactly one advisory, else the
        # probe below proves nothing (a build that never warns would "pass"
        # the suppression assertion vacuously).
        with warnings.catch_warnings(record=True) as rec_default:
            warnings.simplefilter("always")
            build()
        baseline = [w for w in rec_default if issubclass(w.category, BakedInNebularWarning)]
        assert len(baseline) == 1, (
            f"probe setup failed: the silent default emitted {len(baseline)} "
            "BakedInNebularWarning, expected exactly 1 -- without it the "
            "suppression check below is vacuous"
        )

        # Now the route the message advertises, executed as written.
        with warnings.catch_warnings(record=True) as rec_advised:
            warnings.simplefilter("always")
            build(neb=neb_group)
        left = [w for w in rec_advised if issubclass(w.category, BakedInNebularWarning)]
        assert len(left) == 0, (
            f"the message advertises neb={neb_group!r} as the way to silence this "
            f"advisory when building via SEDModel.build, but {len(left)} "
            "BakedInNebularWarning remain -- the advice does not work"
        )

    def test_the_other_advertised_neb_spelling_also_suppresses(self, synthetic_ssp_wide):
        """The message offers ``{'type': 'none'}`` too; both must be real."""
        from tengri import DEFAULT, Fixed, SEDModel
        from tengri.components.nebular.baked_in import BakedInBackend, BakedInNebularWarning

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            BakedInBackend()
        text = str(rec[0].message)
        assert "{'type': 'none'}" in text, (
            f"message no longer advertises the 'none' spelling:\n{text}"
        )

        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                },
                neb={"type": "none"},
                redshift=Fixed(0.1),
            )
        left = [w for w in rec2 if issubclass(w.category, BakedInNebularWarning)]
        assert len(left) == 0, f"neb={{'type': 'none'}} leaves {len(left)} advisory(ies)"

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
