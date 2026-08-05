# SPDX-License-Identifier: BSD-3-Clause
"""Every shipped example must be *callable*, not merely spelled with real names.

``tests/contract/test_tutorials_teach_real_api.py`` resolves each
``receiver.attribute`` a tutorial mentions. That catches a phantom method, and
it is why ``posterior.plot_spectrum_fit()`` cannot come back. It cannot catch
the next failure along, because a name that exists still hands the reader a
``TypeError``:

* ``tengri.tutorial("first_fit")`` — the first thing a new user copies — opened
  with ``list_filters(instrument="2MASS")``. The parameter is ``survey``.
* the same tutorial then called ``generate_mock(model, key=..., snr=...)``
  while ``params`` is required *and positional*. Four tutorials repeated it.
* ``hierarchical`` built ``PopulationFitter(model=, galaxy_data=,
  shared_params=)``; the real signature is ``(model_factory, galaxies, ...)``
  and every one of those three keywords was rejected.

``hasattr`` is True in all three. ``inspect.Signature.bind`` is the one rule
that rejects all three, because it catches the unexpected keyword *and* the
missing required argument. Checking keyword names alone finds the first shape
and misses the second — which is how ``generate_mock``, the most-repeated
broken call in the file, survived two passes of this audit.

The rule is implemented once, in ``tools/check_doc_examples.py``, where it
also covers docstring examples and published pages. These tests pin the
tutorial surface specifically and, more importantly, prove the checker still
*rejects* — a bind check that silently matched nothing would pass forever.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

from check_doc_examples import bind_violations, public_api, tutorial_blocks

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


@pytest.fixture(scope="module")
def resolve():
    return public_api()


# ── guard the guard ──────────────────────────────────────────────────


class TestTheCheckerStillRejects:
    """A bind check that matched nothing would pass every tutorial forever."""

    def test_rejects_an_unexpected_keyword(self, resolve):
        """The exact shape of the first_fit defect."""
        bad = bind_violations('tengri.list_filters(instrument="2MASS")', {}, resolve)
        assert bad, "the checker no longer rejects an unexpected keyword argument"
        assert "instrument" in bad[0][1]

    def test_rejects_a_missing_required_positional(self, resolve):
        """The shape a keyword-name check cannot see — the generate_mock defect."""
        bad = bind_violations("tengri.generate_mock(model, key=k, snr=20.0)", {}, resolve)
        assert bad, "the checker no longer rejects a missing required argument"
        assert "params" in bad[0][1]

    def test_accepts_the_corrected_calls(self, resolve):
        """...and genuinely accepts, or it would just fail everything."""
        good = 'tengri.list_filters(survey="2MASS")\ntengri.generate_mock(model, truth, snr=20.0)'
        assert bind_violations(good, {}, resolve) == []

    def test_a_local_name_shadows_the_tengri_one(self, resolve):
        """Resolution is module-aware, or correct docstrings get flagged.

        There are two public ``list_filters``: ``tengri.list_filters(survey=)``
        and ``tengri.observation.filters.list_filters(instrument=)``. Resolving
        the bare name globally reports the second one's own correct docstring as
        a violation of the first one's signature.
        """
        import tengri.observation.filters as local

        ns = vars(local)
        assert bind_violations('list_filters(instrument="sdss")', ns, resolve) == []
        # ...and the same spelling is still wrong for the top-level function.
        assert bind_violations('tengri.list_filters(instrument="sdss")', {}, resolve)


# ── the surface itself ───────────────────────────────────────────────


def _tutorials():
    blocks = tutorial_blocks()
    assert blocks, "no tutorials discovered — these tests would prove nothing"
    return blocks


@pytest.mark.parametrize("name,code", _tutorials(), ids=[n for n, _ in _tutorials()])
def test_every_call_in_the_tutorial_binds(resolve, name, code):
    """A user copies this verbatim. It must not raise TypeError on arity."""
    bad = bind_violations(code, {}, resolve)
    assert not bad, "tutorial {!r} ships a call that cannot bind:\n{}".format(
        name,
        "\n".join(f"  {dotted}(...) — {err}" for dotted, err in bad),
    )


def test_the_sweep_is_not_vacuous(resolve):
    """Guard the parser: a chunker that stopped matching would pass silently."""
    blocks = tutorial_blocks()
    assert len(blocks) >= 10, f"only {len(blocks)} tutorials discovered"
    # Some tutorial must actually contain a call the checker can resolve,
    # otherwise every assertion above is vacuously true.
    resolvable = sum(
        1
        for _, code in blocks
        # a deliberately broken copy must be rejected -> proves calls are reached
        if bind_violations(code.replace("survey=", "instrument="), {}, resolve)
    )
    assert resolvable >= 1, (
        "no tutorial produced a violation even after corrupting its keywords — "
        "the chunk parser has stopped reaching tutorial call sites"
    )
