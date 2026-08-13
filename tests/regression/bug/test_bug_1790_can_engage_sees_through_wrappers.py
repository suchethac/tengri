# SPDX-License-Identifier: BSD-3-Clause
"""#1790: ``fast_nebular_can_engage`` answered ``True`` for every ``ForwardModel``.

The predicate reads ``_build_component_chain``, which is defined on
:class:`SEDModel` and nowhere else. A bare ``getattr`` therefore finds it only
when the object *is* an ``SEDModel`` — and inference is canonically through
:class:`ForwardModel` (#211), which holds ``populations[i].sed`` instead. The
miss fell through to a permissive ``return True``, so every ``ForwardModel``
was told the fast nebular grid could engage, dusty or not.

**Scope.** Since #1770 this predicate answers the **photometry** question and
only that: serving photometry from the per-Q_H grid requires zeroing
``sed_nebular``, and ``DustSEDComponent`` reads it, so a dusty model cannot
(#1748, measured bit-identical). Whether a *line-flux* fit gets the LUT is a
separate question — dust does not disarm that half, and #1770 measured 4.77x on
a dusty line fit. These tests therefore assert only the photometry gate;
``test_bug_1770_line_lut_survives_dust.py`` owns the other one.

**The fix is the unwrap, and only the unwrap.** ``_component_chains`` now walks
``populations[i].sed``, and requires *every* population to be clear rather than
reading ``populations[0]`` — picking one arbitrarily is the failing-open shape
``ForwardModel._single_inner_sed`` already refuses (#1271).

The no-chain fallback stays permissive, which is a decision rather than an
oversight. On the cost asymmetry it ought to fail closed — a wrong ``False``
forfeits a speedup, a wrong ``True`` attaches a config with no effect *and*
changes ``compile_signature()``, so the fit buys a second compiled kernel for
nothing. That flip was tried and measured: 5 failures, every one a
``_StubModel`` exposing neither a chain nor populations. Those are objects that
are not models, not models that cannot be read — and once the unwrap is in
place every real surface resolves to a chain, so the flip has no demonstrated
effect on any production path. Bundling a stub rewrite into the fix for a
different bug is how scope creeps; #1790 carries it as a follow-up with that
blast radius attached.

The dust-free arm is asserted first and deliberately: a guard that only checks
"a dusty model is refused" also passes when the predicate is broken to always
return ``False``, which would silently disable the real 30x speedup that #1748
measured on a dust-free model.
"""

from __future__ import annotations

import warnings

import pytest

from tengri import FIXED, FREE, Fixed, ForwardModel, SEDModel, Uniform
from tengri.inference.fitter import fast_nebular_can_engage

pytestmark = pytest.mark.regression_bug


def _sed(ssp, obs, *, dust: bool):
    dust_group = (
        {
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 2.0),
        }
        if dust
        else {"type": "none"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust=dust_group,
            neb={"type": "none"},
            redshift=Fixed(0.1),
            approx=None,
        )


def _forward(sed, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ForwardModel.build(sed=sed, observation=obs)


def test_a_dust_free_model_can_still_engage(synthetic_ssp_wide, synthetic_tophat_obs):
    """The control, asserted first — the fast path must remain reachable.

    Without this, a predicate hard-wired to ``False`` would satisfy every other
    assertion in this file while silently withdrawing the speedup #1748 measured
    at 30.63x on a dust-free model.
    """
    sed = _sed(synthetic_ssp_wide, synthetic_tophat_obs, dust=False)
    assert fast_nebular_can_engage(sed), (
        "a dust-free SEDModel must still be able to engage the fast nebular grid"
    )


def test_the_wrapper_agrees_with_the_sed_it_wraps(synthetic_ssp_wide, synthetic_tophat_obs):
    """A ForwardModel must answer whatever its inner SED answers, both ways.

    This is the bug in one line: before the fix the dusty pair disagreed —
    ``SEDModel`` said ``False`` and the ``ForwardModel`` wrapping that very same
    SED said ``True``.
    """
    for dust in (True, False):
        sed = _sed(synthetic_ssp_wide, synthetic_tophat_obs, dust=dust)
        forward = _forward(sed, synthetic_tophat_obs)
        assert fast_nebular_can_engage(forward) == fast_nebular_can_engage(sed), (
            f"ForwardModel and its inner SEDModel disagree for dust={dust}; the "
            "wrapper is not being unwrapped, so the guard is a no-op on the "
            "canonical inference path (#1790)"
        )


def test_a_dusty_forward_model_is_refused(synthetic_ssp_wide, synthetic_tophat_obs):
    """The headline: the canonical path must refuse the inert config."""
    sed = _sed(synthetic_ssp_wide, synthetic_tophat_obs, dust=True)
    forward = _forward(sed, synthetic_tophat_obs)
    assert not fast_nebular_can_engage(forward), (
        "a dusty ForwardModel was told the fast nebular grid can engage; dust "
        "reads sed_nebular, so FeaturePrecomp is bit-identical there (#1748)"
    )


def test_an_object_with_no_chain_at_all_stays_permissive():
    """The fallback is unchanged, and that is a decision rather than an oversight.

    On the asymmetry alone this should fail closed — see the module docstring.
    Flipping it was tried and measured: 5 failures, every one a ``_StubModel``
    exposing neither a chain nor populations. Those are objects that are not
    models, not models that cannot be read, and once the unwrap above is in
    place every real surface resolves to a chain — so the flip has no
    demonstrated effect on any production path.

    Pinned so the fallback cannot be flipped silently: changing it means
    rewriting those five tests, which is a decision to take deliberately (#1790).
    """

    class _Opaque:
        """Neither an SEDModel nor a populations container."""

    assert fast_nebular_can_engage(_Opaque()), (
        "the no-chain fallback changed; that flip breaks 5 stub-based tests and "
        "needs to be taken deliberately, not as a side effect (#1790)"
    )
