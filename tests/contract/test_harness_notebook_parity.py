# SPDX-License-Identifier: BSD-3-Clause
"""The benchmark fixtures still build the models they claim to (#2096).

``bench/scripts/benchmark_notebook_sampler.py`` mirrors the notebooks' models by
copying them into Python functions, and nothing enforced that the copy stayed a
copy. Two had drifted: nb05 by #1989 (``law_bc="calzetti"`` rewritten to
``law="calzetti"``, which sets a second screen and moves R-hat 1.0043 -> 1.1426
on the notebook's own committed fit) and nb00 by #2044 (tsnorm + two-component
at D=7 rewritten to dpl + single-component + nebular at D=6). Both kept
labeling their rows with the notebook's name.

``tools/check_harness_parity.py`` holds the fixtures to their declared
provenance; this file runs it, and -- as importantly -- checks the guard still
fails when it should. A guard exercised only on a clean tree passes just as well
when it is blind, which is the lesson ``tests/unit/test_check_notebook_pairing.py``
already records for the notebook-pairing guard.

Both polarities, and three groups:

* **structural** -- no model building, so it runs anywhere: every fixture
  declares a provenance block, every ``mirrors`` names a notebook that exists,
  every ``historical`` anchor chain terminates at a ``mirrors``.
* **parity** -- the real check, one test per fixture. Needs the SSP grids,
  which are tracked in ``data/``.
* **negative** -- injected drift of each kind the guard exists to catch.

No fit is run: building the models is the whole check, and it is what makes this
affordable to gate on. A single nb05 NUTS fit costs 200-500 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_harness_parity as parity

pytestmark = pytest.mark.contract

#: The bare-stellar and wNE grids the fixtures build against. Both are tracked
#: in git, so this skip should never fire on a complete checkout -- it exists so
#: a shallow or partial one says why rather than erroring in an SSP loader.
_GRIDS = (
    REPO_ROOT / "data" / "fsps_prsc_miles_chabrier.h5",
    REPO_ROOT / "data" / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
)
requires_grids = pytest.mark.skipif(
    not all(p.is_file() for p in _GRIDS),
    reason=f"needs the SSP grids at {', '.join(p.name for p in _GRIDS)} (tracked in git)",
)


def _fixtures():
    return parity.registry()


FIXTURE_NAMES = sorted(_fixtures())


# ---------------------------------------------------------------------------
# structural -- builds nothing
# ---------------------------------------------------------------------------


def test_registry_is_not_empty():
    assert FIXTURE_NAMES, "the fixture registry is empty; the guard would pass vacuously"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_fixture_declares_provenance(name):
    """No fixture may join the registry without saying what it is a copy of.

    This is the defect #2096 reports, stated as a rule: ``_build_nb00`` never
    declared which quickstart it mirrored, so when #2044 moved the quickstart
    there was nothing to contradict.
    """
    parity.provenance(name, _fixtures()[name])


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_declared_notebook_exists(name):
    par = parity.provenance(name, _fixtures()[name])
    if par["kind"] != "mirrors":
        pytest.skip(f"{name} is {par['kind']}")
    assert (REPO_ROOT / par["notebook"]).is_file(), f"{name}: {par['notebook']} does not exist"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_historical_anchor_chain_grounds_in_a_notebook(name):
    """A historical fixture is exempt from notebook parity but not from reality.

    ``05pre`` is not checked against ``05_fitting_photometry.py`` -- it is not
    supposed to match it. It is checked against ``05``, which IS checked against
    the notebook. If that chain ended nowhere, "historical" would just be a way
    to opt out.
    """
    fixtures = _fixtures()
    par = parity.provenance(name, fixtures[name])
    if par["kind"] != "historical":
        pytest.skip(f"{name} is {par['kind']}")
    chain = parity.anchor_chain(name, fixtures)
    assert parity.provenance(chain[-1], fixtures[chain[-1]])["kind"] == "mirrors"


def test_at_least_one_fixture_mirrors_each_spine_notebook():
    """The three notebooks #2096 names each have a live fixture tracking them.

    Before #2096 nothing in the tree tracked ``00_quickstart.py`` at all: both
    nb00 fixtures were pre-#2044. ``00now`` was added so this assertion can hold.
    """
    mirrored = {
        par["notebook"]
        for cfg in _fixtures().values()
        if (par := cfg.get("parity", {})).get("kind") == "mirrors"
    }
    for expected in (
        "notebooks/00_quickstart.py",
        "notebooks/01_why_jax.py",
        "notebooks/05_fitting_photometry.py",
    ):
        assert expected in mirrored, f"no fixture mirrors {expected}"


# ---------------------------------------------------------------------------
# parity -- the real check
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models():
    """Every fixture's model, built once. Shared because an SSP load is 67 MB."""
    return parity.build_models(_fixtures(), FIXTURE_NAMES)


@requires_grids
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_matches_its_declared_provenance(name, models):
    """The fixture builds the model it says it builds.

    On failure the message names the differing spec keys and what to do. The
    short version: the harness follows the notebook, and a published measurement
    is never edited to make a fixture agree with it.
    """
    parity.check_fixture(name, _fixtures(), models)


# ---------------------------------------------------------------------------
# negative -- the guard must fail when it should
# ---------------------------------------------------------------------------


def test_missing_provenance_block_is_rejected():
    with pytest.raises(parity.ParityError, match="no parity= provenance block"):
        parity.provenance("fake", {"build": None})


@pytest.mark.parametrize(
    "par,match",
    [
        ({"kind": "invented"}, "not one of"),
        ({"kind": "mirrors"}, "needs notebook="),
        (
            {"kind": "historical", "differs_in": (), "superseded_by": "#1"},
            "needs anchor=",
        ),
        (
            {"kind": "historical", "anchor": "x", "superseded_by": "#1"},
            "needs differs_in",
        ),
        ({"kind": "historical", "anchor": "x", "differs_in": ()}, "needs superseded_by"),
        ({"kind": "standalone"}, "needs why="),
    ],
)
def test_malformed_provenance_blocks_are_rejected(par, match):
    with pytest.raises(parity.ParityError, match=match):
        parity.provenance("fake", {"parity": par})


def test_anchor_chain_rejects_a_cycle():
    fixtures = {
        "a": {
            "parity": dict(kind="historical", anchor="b", differs_in=("x",), superseded_by="#1")
        },
        "b": {
            "parity": dict(kind="historical", anchor="a", differs_in=("x",), superseded_by="#1")
        },
    }
    with pytest.raises(parity.ParityError, match="anchor cycle"):
        parity.anchor_chain("a", fixtures)


def test_anchor_chain_rejects_grounding_in_a_standalone():
    """ "Historical, anchored to something that mirrors nothing" is an escape hatch."""
    fixtures = {
        "a": {
            "parity": dict(kind="historical", anchor="b", differs_in=("x",), superseded_by="#1")
        },
        "b": {"parity": dict(kind="standalone", why="a control")},
    }
    with pytest.raises(parity.ParityError, match="must ground out"):
        parity.anchor_chain("a", fixtures)


def test_anchor_chain_rejects_an_unknown_anchor():
    fixtures = {
        "a": {
            "parity": dict(
                kind="historical", anchor="ghost", differs_in=("x",), superseded_by="#1"
            )
        }
    }
    with pytest.raises(parity.ParityError, match="is not a fixture"):
        parity.anchor_chain("a", fixtures)


@requires_grids
def test_a_drifted_mirror_is_caught(models):
    """The #1989 failure, injected: make ``05`` build the pre-#1989 dust law.

    This is the exact change that went unnoticed for ten days, and it is a
    three-key spec difference -- so if this test ever passes-by-not-failing, the
    guard has stopped seeing a dust-law swap.
    """
    drifted = dict(models)
    drifted["05"] = models["05pre"]
    with pytest.raises(parity.ParityError) as exc:
        parity.check_fixture("05", _fixtures(), drifted)
    message = str(exc.value)
    assert "no longer mirrors notebooks/05_fitting_photometry.py" in message
    assert "dust_attenuation.law_diff" in message
    assert "harness follows the notebook" in message


@requires_grids
def test_a_historical_fixture_repaired_into_a_duplicate_is_caught(models):
    """``05pre`` silently "fixed" to today's model would erase why it exists.

    The published 2026-08-17 row reproduces only under the pre-#1989 dust law.
    A well-meaning repair that made ``05pre`` match ``05`` would leave that
    report unreproducible with nothing raised, so it raises here.
    """
    duplicated = dict(models)
    duplicated["05pre"] = models["05"]
    with pytest.raises(parity.ParityError, match="same model as its anchor"):
        parity.check_fixture("05pre", _fixtures(), duplicated)


@requires_grids
def test_an_undeclared_difference_from_the_anchor_is_caught(models):
    """A historical fixture is exempt only in the ways it declares.

    Narrowing ``05pre``'s ``differs_in`` to two of its three keys must fail:
    otherwise ``differs_in`` would be decorative and a historical fixture could
    drift arbitrarily behind the word "historical".
    """
    fixtures = dict(_fixtures())
    narrowed = dict(fixtures["05pre"])
    narrowed["parity"] = dict(narrowed["parity"])
    narrowed["parity"]["differs_in"] = ("dust_attenuation.law", "dust_attenuation.law_bc")
    fixtures["05pre"] = narrowed
    with pytest.raises(parity.ParityError, match="undeclared differences"):
        parity.check_fixture("05pre", fixtures, models)


@requires_grids
def test_prediction_check_sees_what_the_spec_cannot(models):
    """The second half of the check is not redundant with the first.

    ``to_groups()`` carries no SSP grid, so two fixtures with identical specs on
    different grids are a spec match and a prediction mismatch. #964 and #1777
    were both structural losses from ``to_groups()``; the prediction comparison
    is what covers that class.
    """
    a, b = models["05"], models["05pre"]
    assert parity.max_relative_flux_difference(a, a) == 0.0
    assert parity.max_relative_flux_difference(a, b) > parity.FLUX_RTOL
