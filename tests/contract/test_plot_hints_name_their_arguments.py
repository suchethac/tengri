# SPDX-License-Identifier: BSD-3-Clause
"""A ``use:`` hint that is the name column plus an ellipsis tells nobody anything.

Every ``list_*`` row carries a ``use`` hint so a reader can copy a working
call. Across the twenty menus that have one, nineteen give something with
arguments in it::

    list_filters            Photometry.from_names(["2MASS_2MASS_H"])
    list_inference_methods  fitter.run("laplace")
    list_dust_laws          SEDModel.build(..., dust={'type': 'single_component', ...})
    list_recipes            recipes.agn_panchromatic() -> SEDModel.build(ssp_data=ssp, **recipe)

``list_plots`` was the exception: all ten rows read ``tengri.plot.<name>(...)``
— the ``name`` column again, plus a literal ellipsis. A reader who has the row
in front of them learns nothing from the column meant to tell them how to call
it, and ``plot_sfh`` needing ``(model, posterior)`` while ``setup_style`` needs
nothing is exactly the difference the column exists to convey.

The hint is now read off ``inspect.signature``, so a helper that gains or loses
a required argument cannot leave a stale hint behind.

These tests scan rather than pin strings: a hint is checked against the live
signature, so they stay true as the helpers change.
"""

from __future__ import annotations

import inspect
import re

import pytest

import tengri

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_CALL = re.compile(r"tengri\.plot\.(\w+)\(([^)]*)\)")


def _rows():
    return list(tengri.list_plots())


class TestTheCensus:
    def test_there_are_plots_to_check(self):
        assert len(_rows()) >= 5, "the plot menu shrank; this file would be near-vacuous"

    def test_no_hint_is_a_bare_placeholder(self):
        """The defect, stated as a property rather than a list of names."""
        placeholders = [r["name"] for r in _rows() if r["use"] == f"tengri.plot.{r['name']}(...)"]
        assert not placeholders, (
            f"these rows advertise only their own name plus an ellipsis: "
            f"{placeholders}. The use: column exists to say how to call the "
            f"helper; every other menu's does."
        )


class TestEveryHintMatchesItsSignature:
    @pytest.mark.parametrize("row", _rows(), ids=[r["name"] for r in _rows()])
    def test_the_hint_names_exactly_the_required_arguments(self, row):
        name, hint = row["name"], row["use"]
        match = _CALL.fullmatch(hint.strip())
        assert match, f"{name}: hint is not a tengri.plot call: {hint!r}"
        assert match.group(1) == name, f"{name}: hint names {match.group(1)!r}"

        advertised = [a.strip() for a in match.group(2).split(",") if a.strip()]
        fn = getattr(tengri.plot, name)
        required = [
            p.name
            for p in inspect.signature(fn).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        assert advertised == required, (
            f"{name} advertises {advertised} but its signature requires "
            f"{required}. The hint is derived from the signature, so a "
            f"mismatch means the derivation broke."
        )

    @pytest.mark.parametrize("row", _rows(), ids=[r["name"] for r in _rows()])
    def test_the_advertised_call_would_bind(self, row):
        """Names alone are not enough — the shape has to be acceptable.

        Binds sentinels rather than calling: rendering a real figure needs a
        fitted posterior, which is not this guard's question.
        """
        name, hint = row["name"], row["use"]
        args = [a.strip() for a in _CALL.fullmatch(hint.strip()).group(2).split(",") if a.strip()]
        inspect.signature(getattr(tengri.plot, name)).bind(*([object()] * len(args)))


class TestTheOtherMenusStillCarryArguments:
    def test_plots_is_not_the_only_menu_with_a_use_column(self):
        """Anti-vacuity: the comparison this file is premised on must exist."""
        with_use = 0
        for attr in dir(tengri):
            if not attr.startswith("list_"):
                continue
            fn = getattr(tengri, attr, None)
            if not callable(fn):
                continue
            try:
                rows = fn()
            except Exception:
                continue
            if (
                isinstance(rows, list)
                and rows
                and isinstance(rows[0], dict)
                and any(str(r.get("use", "")).strip() for r in rows)
            ):
                with_use += 1
        assert with_use >= 10, f"only {with_use} menus carry a use: column"
