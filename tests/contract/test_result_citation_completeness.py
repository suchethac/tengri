# SPDX-License-Identifier: BSD-3-Clause
"""Regression contract: citing a *fit result* must cite everything that ran.

Fresh-user audit (2026-07). ``docs/index.md`` promises

    Every physics block, SSP grid, and inference backend carries its own
    citation, so the BibTeX for a fit is assembled from what actually ran

and teaches ``print_components_bibtex(result)``. It emitted three of twenty-one
entries. Two independent delegation hops were missing:

* a :class:`~tengri.inference.posterior.Posterior` stores its model under
  ``_model``, but the collectors probed ``obj.model``; and
* a :class:`~tengri.forward.forward_model.ForwardModel` exposes its SED only
  through ``_inner_sed_for_delegation()``.

Every probe was a ``getattr(..., None)`` that failed open, so the result path
degraded silently to the core keys instead of raising. A user following the
docs would have published without citing their dust law, nebular model, SSP
grid, or IMF.

A third defect made the sampler unrecoverable on its own: ``Posterior.method``
is a display string (``"NUTS (BlackJAX)"``, ``"MAP (ADAM, 5 restarts)"``), never
a registry key, so ``Fitter.run`` now stamps the canonical ``_backend_key``.

These tests assert **coverage**, not non-emptiness. The pre-existing guard in
``test_citations.py`` asserted only ``"@" in out``, which three-of-twenty-one
still satisfies — that is why this shipped.
"""

from __future__ import annotations

import pytest

from tengri import FIXED, Fixed, ForwardModel, SEDModel
from tengri.citations.associations import BACKEND_CITATIONS
from tengri.citations.collect import _backend_from, collect_citations
from tengri.inference.posterior import Posterior

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _keys(obj) -> set[str]:
    return {c.key for c in collect_citations(obj)}


@pytest.fixture
def sed(synthetic_ssp_wide, synthetic_tophat_obs):
    """A model with enough physics that dropped citations are visible."""
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        redshift=Fixed(0.1),
    )


@pytest.fixture
def forward(sed, synthetic_tophat_obs):
    return ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)


def _posterior_for(model) -> Posterior:
    """A Posterior shaped like one ``Fitter.run`` returns, without paying for a fit."""
    return Posterior(
        samples=None,
        params={},
        method="MAP (ADAM)",  # deliberately the display string, not a key
        wall_time_s=0.0,
        diagnostics={},
        _model=model,
    )


def test_forward_model_wrapper_does_not_hide_component_citations(sed, forward):
    """Hop 2: the wrapper must not shrink the citation set of what it wraps."""
    inner, outer = _keys(sed), _keys(forward)
    assert inner <= outer, f"ForwardModel dropped {sorted(inner - outer)}"


def test_result_cites_everything_the_model_does(sed, forward):
    """THE BUG. ``print_components_bibtex(result)`` is the documented per-fit
    surface; it must not report less than the model it was fit with."""
    result = _posterior_for(forward)
    model_keys, result_keys = _keys(sed), _keys(result)
    assert model_keys <= result_keys, (
        f"citing the result dropped {sorted(model_keys - result_keys)} — "
        "a user following the docs would publish without these"
    )


def test_named_physics_citations_survive_the_result_path(sed, forward):
    """Name the entries explicitly: a set-comparison alone would pass if both
    sides collapsed to the core keys."""
    result_keys = _keys(_posterior_for(forward))
    for key in ("calzetti2000", "dsps", "tengri"):
        assert key in result_keys, f"{key!r} missing from the fit's BibTeX"
    assert len(result_keys) > len({"tengri", "jax", "dsps"}), (
        "result collapsed to the core citations — the delegation chain is broken"
    )


def test_backend_key_is_the_registry_key_not_the_display_string(forward):
    """``Posterior.method`` is human-readable and cannot be matched against
    ``BACKEND_CITATIONS``; the canonical key is stamped separately."""
    result = _posterior_for(forward)
    assert result.method not in BACKEND_CITATIONS, (
        "test premise broken: method is supposed to be a display string"
    )
    assert _backend_from(result) is None, "no key stamped yet -> no backend"

    result._backend_key = "mcmc_nuts"
    assert _backend_from(result) == "mcmc_nuts"


def test_sampler_citation_reaches_the_result(forward):
    """A NUTS fit must cite BlackJAX; a MAP fit must not gain a sampler cite."""
    result = _posterior_for(forward)

    result._backend_key = "mcmc_nuts"
    assert "blackjax" in _keys(result), "NUTS fit did not cite its sampler"

    result._backend_key = "map"
    assert "blackjax" not in _keys(result), (
        "MAP gained a sampler citation — the fix is over-citing, not delegating"
    )


def test_cite_components_resolves_spec_through_the_chain(sed, forward):
    """``cite_components`` is a *second, independent* collector (registry.py),
    reached by the documented ``print_components_bibtex``. It resolved the spec
    with ``getattr(obj, "spec", obj)`` and its comment claimed "SEDModel /
    Posterior expose .spec" — but Posterior does not, so it fell back to the
    Posterior and found no components.

    Fixing only ``collect_citations`` would leave the documented surface broken,
    which is why this asserts the second path separately.
    """
    import tengri

    assert (
        not hasattr(forward, "spec") or getattr(_posterior_for(forward), "spec", None) is None
    ), "test premise: a Posterior must not expose .spec directly"
    model_rows = len(tengri.cite_components(sed))
    result_rows = len(tengri.cite_components(_posterior_for(forward)))
    assert result_rows >= model_rows, (
        f"cite_components(result) found {result_rows} components, "
        f"cite_components(model) found {model_rows} — the spec hop is broken"
    )


def test_documented_bibtex_call_emits_the_physics(sed, forward, capsys):
    """The exact call ``docs/index.md`` teaches must emit the physics entries,
    not just the framework ones. Asserting on named entries rather than
    ``"@" in out`` — the latter passes with 3 of 21 and is why this shipped."""
    import tengri

    tengri.print_components_bibtex(_posterior_for(forward))
    out = capsys.readouterr().out
    entries = [line.split("{")[1].rstrip(",") for line in out.splitlines() if line.startswith("@")]
    assert "Calzetti_2000" in entries, (
        f"the fit's dust law is missing from its BibTeX; got {sorted(entries)}"
    )
    framework_only = {"Cooray_2026", "Hearin_2023", "Jamesbradbury_2018"}
    assert set(entries) - framework_only, (
        "only framework citations emitted — no physics component reached the BibTeX"
    )


def test_fitter_run_stamps_the_backend_key(forward):
    """The stamp is applied at the single dispatch funnel in ``Fitter.run``,
    so it holds for every backend rather than per-runner."""
    import inspect

    from tengri.inference import fitter as fitter_mod

    src = inspect.getsource(fitter_mod.Fitter.run)
    assert "_backend_key" in src, (
        "Fitter.run no longer stamps _backend_key; sampler citations will be lost"
    )
