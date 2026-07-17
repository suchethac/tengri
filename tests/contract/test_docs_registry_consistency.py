# SPDX-License-Identifier: BSD-3-Clause
"""Names printed on the front page must be names the library actually accepts.

The README and ``docs/overview.md`` advertised two nebular backends that never
existed — ``baked_in`` and ``cloudy_grid`` — while the registry names them
``ssp`` and ``cloudy``. Both wrong spellings are the *implementation* names
(``BakedInBackend``, ``CloudyGridBackend``), which is why they read as plausible
and survived so long: a newcomer copying either off the front page got a
``ValueError`` on their first build.

These tests close the loop in both directions:

- every backend the published docs name must resolve in the live registry, so
  the docs cannot drift ahead of the code again;
- the "did you mean" hint map must stay coherent with the registry, so it
  cannot rot into suggesting a name that no longer exists (or shadowing one
  that has since become real).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import tengri
from tengri.components.nebular import NEBULAR_MODELS
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, UNVALIDATED_SFH_TYPES
from tengri.parameters.groups import _NEBULAR_TYPE_HINTS

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Pages a first-time user reads before they read anything else.
_PUBLISHED_PAGES = ("README.md", "docs/overview.md")


def _page_text(rel: str) -> str:
    path = _REPO_ROOT / rel
    if not path.is_file():
        pytest.skip(f"{rel} not present in this checkout")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("rel", _PUBLISHED_PAGES)
@pytest.mark.parametrize("stale", sorted(_NEBULAR_TYPE_HINTS))
def test_published_pages_do_not_advertise_stale_nebular_names(rel: str, stale: str) -> None:
    """The front page must not name a backend that ``SEDModel.build`` rejects."""
    text = _page_text(rel)
    hits = [
        line.strip() for line in text.splitlines() if re.search(rf"`{re.escape(stale)}`", line)
    ]
    assert not hits, (
        f"{rel} advertises nebular backend `{stale}`, which SEDModel.build rejects. "
        f"Use `{_NEBULAR_TYPE_HINTS[stale]}` instead. Offending lines: {hits}"
    )


def test_hint_map_points_at_real_backends() -> None:
    """Every 'did you mean' target must be a name the builder accepts."""
    valid = set(NEBULAR_MODELS)
    for stale, suggestion in _NEBULAR_TYPE_HINTS.items():
        assert suggestion in valid, (
            f"hint {stale!r} -> {suggestion!r} suggests a backend that is not "
            f"registered; known backends: {sorted(valid)}"
        )


def test_sfh_discovery_matches_what_the_builder_accepts() -> None:
    """``list_sfh_models(status='production')`` must equal the buildable set.

    Discovery advertised the raw registry (every registered SFH type) while
    ``SEDModel.build`` accepted only the DSPS-validated subset, so a newcomer
    following the documented discovery path could pick a model that hard-errors
    at build. The two surfaces must name the same set.
    """
    production = {r["name"] for r in tengri.list_sfh_models(status="production")}
    buildable = set(tengri.builders.sfh.available())
    assert production == buildable, (
        "discovery and the builder disagree: "
        f"listed-not-buildable={sorted(production - buildable)}, "
        f"buildable-not-listed={sorted(buildable - production)}"
    )


def test_sfh_unvalidated_types_are_still_discoverable() -> None:
    """The unvalidated types stay visible — flagged, not hidden.

    Dropping them from discovery would trade a confusing error for a silent
    omission; they are really registered, and someone is mid-way through
    validating them.
    """
    listed = {r["name"] for r in tengri.list_sfh_models()}
    assert listed == set(SFH_REGISTRY), "list_sfh_models() no longer lists the full registry"
    flagged = {r["name"] for r in tengri.list_sfh_models(status="unvalidated")}
    assert flagged == set(UNVALIDATED_SFH_TYPES)


def test_hint_map_does_not_shadow_a_real_backend() -> None:
    """A hint key must stay *invalid* — otherwise the hint is dead code.

    If a spelling in the hint map is ever registered for real, the validator
    accepts it before the hint is consulted, and the entry silently stops
    meaning anything. Fail loudly so the map gets cleaned up instead.
    """
    valid = set(NEBULAR_MODELS)
    shadowed = sorted(set(_NEBULAR_TYPE_HINTS) & valid)
    assert not shadowed, (
        f"{shadowed} are now real nebular backends, so their entries in "
        f"_NEBULAR_TYPE_HINTS are unreachable — remove them."
    )


# ── Stale inference-method names (round 2 of the same failure mode) ──────
# The published performance page taught ``vi_native`` for months after the
# backend was renamed ``native_vi_nonlinear`` — a reader copying the row got
# ``KeyError`` from ``fitter.run("vi_native")``. Same shape as the nebular
# map above: the old name reads plausible, so it survives review. The exact
# backtick-token regex keeps legitimate composites (the ``vi_native_vs_nifty``
# benchmark suite, ``benchmark_vi_native_vs_nifty.py``) out of scope.

_STALE_INFERENCE_NAMES = {"vi_native": "native_vi_nonlinear"}

_METHOD_PAGES = (
    *_PUBLISHED_PAGES,
    "docs/performance/index.md",
    "docs/performance/memory.md",
    "docs/method_selection.md",
    "docs/advanced/convergence.md",
    "docs/inference/index.md",
)


@pytest.mark.parametrize("rel", _METHOD_PAGES)
@pytest.mark.parametrize("stale", sorted(_STALE_INFERENCE_NAMES))
def test_pages_do_not_teach_stale_inference_methods(rel: str, stale: str) -> None:
    """No user-facing page may name an inference method run() rejects."""
    text = _page_text(rel)
    hits = [
        line.strip() for line in text.splitlines() if re.search(rf"`{re.escape(stale)}`", line)
    ]
    assert not hits, (
        f"{rel} teaches inference method `{stale}`, which fitter.run() rejects. "
        f"Use `{_STALE_INFERENCE_NAMES[stale]}` instead. Offending lines: {hits}"
    )


def test_stale_inference_map_points_at_real_methods() -> None:
    """Every replacement the stale map prescribes must resolve in the registry."""
    live = {row["name"] for row in tengri.list_inference_methods()}
    for stale, current in _STALE_INFERENCE_NAMES.items():
        assert current in live, f"map prescribes {current!r} for {stale!r}, not in registry"
        assert stale not in live, (
            f"{stale!r} is a real method again — its stale-map entry is wrong"
        )
