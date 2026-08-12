# SPDX-License-Identifier: BSD-3-Clause
"""``Instrument.list()`` built every instrument three times, on every lookup.

The row comprehension called the factory once per *column*::

    {"name": fac().name, "n_bands": len(fac().filter_names), ...}

so nine premade instruments cost twenty-seven constructions, and each one
loads that instrument's filter set. Measured: **160.6 ms** for nine rows.

That was survivable while nothing hot called it. Then ``list_instruments``
joined the menu census that ``describe()`` and ``search()`` sweep on *every*
lookup — the fix for the census drift — and it became 160.6 ms of a 161.7 ms
sweep: **99% of every** ``describe()`` **call, for 9 of 490 rows.**

Both properties are pinned structurally rather than by wall clock, because a
timing threshold is a flake generator and neither property is "fast today":

* one construction per instrument, not one per column — counted, not timed;
* the table is computed once per process — asserted on ``cache_info()``.

The rows must still be *fresh objects* each call: a cached mutable table
handed to a caller who mutates it corrupts every later lookup.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.observation import instrument as instrument_mod

pytestmark = [pytest.mark.regression_bug, pytest.mark.contract]


@pytest.fixture
def counted_factories(monkeypatch):
    """Swap in factories that count their own construction calls.

    Returns the tally dict. Both properties below are read off it, so neither
    test names the caching *mechanism* — only the observable behaviour.
    """
    calls: dict[str, int] = {}

    def counted(fac):
        def wrapper():
            calls[fac.__name__] = calls.get(fac.__name__, 0) + 1
            return fac()

        wrapper.__name__ = fac.__name__
        return wrapper

    monkeypatch.setattr(
        instrument_mod,
        "_PREMADE_FACTORIES",
        tuple(counted(f) for f in instrument_mod._PREMADE_FACTORIES),
    )
    instrument_mod._reset_premade_rows_cache()
    yield calls
    instrument_mod._reset_premade_rows_cache()


class TestBuiltOncePerInstrument:
    def test_each_factory_is_called_once_per_listing(self, counted_factories):
        """Three calls per row is the defect; one is the contract."""
        instrument_mod.Instrument.list()

        assert counted_factories, "no factory ran — the monkeypatch missed its target, not a pass"
        over = {name: n for name, n in counted_factories.items() if n > 1}
        assert not over, (
            f"these instruments were constructed more than once for one listing: {over}. "
            "Bind the instrument once and read every column off it."
        )


class TestComputedOncePerProcess:
    def test_repeat_listings_do_not_rebuild(self, counted_factories):
        """``describe()`` sweeps this menu on every lookup; rebuilding nine
        instruments per lookup is what made the sweep 160 ms."""
        for _ in range(5):
            tengri.list_instruments()

        assert counted_factories, "no factory ran — the monkeypatch missed its target"
        rebuilt = {name: n for name, n in counted_factories.items() if n > 1}
        assert not rebuilt, (
            f"five listings rebuilt these instruments {rebuilt} — the table is "
            "recomputed per call, so every describe() pays for it."
        )


class TestCachingDidNotLeakMutableState:
    def test_each_call_returns_independent_rows(self):
        """A cached mutable table is a lookup-corrupting footgun."""
        first = tengri.list_instruments()
        assert first, "no instruments listed — the census below would be vacuous"
        first[0]["name"] = "MUTATED-BY-CALLER"

        second = tengri.list_instruments()
        assert second[0]["name"] != "MUTATED-BY-CALLER", (
            "mutating one caller's row changed the next caller's table; the "
            "cache is handing out the same dict objects."
        )

    def test_the_rows_still_carry_what_describe_prints(self):
        rows = tengri.list_instruments()
        columns = ("name", "kind", "n_bands", "description", "use")
        missing = [c for c in columns if c not in rows[0]]
        assert not missing, f"caching dropped these columns: {missing}"
        assert all(r["n_bands"] > 0 for r in rows), (
            "an instrument reported zero bands — the build was skipped, not cached."
        )
