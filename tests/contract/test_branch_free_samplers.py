# SPDX-License-Identifier: BSD-3-Clause
"""The first-order samplers compile to a smaller program than NUTS.

That is the reason ``mcmc_barker`` and ``mcmc_mala`` exist.
``bench/reports/2026-08-30_mclmc_tuning.md`` measured **75% of a cold NUTS fit
going to XLA** -- 189.4 s cold against 46.8 s warm -- and MCLMC's fixed-length
scan compiling **14x cheaper**, and named the cause: NUTS compiles a ragged
tree-doubling ``while`` loop, MCLMC compiles a straight-line step. So the
compiled artifact's size is a speed property, and this file pins it.

**A correction is baked into this file, and it is the point of
:func:`test_a_bare_scan_already_lowers_to_a_while`.** The first version of this
test asserted that Barker's lowered program contains **zero**
``stablehlo.while``. It failed immediately: Barker lowered to 7, MALA to 6.
``lax.scan`` itself lowers to a ``stablehlo.while`` with a constant trip count,
so **the number of ``while`` ops is a proxy for ragged control flow and a bad
one** -- it cannot tell a fixed-length scan from a tree search. The assertion
that a scan-only program still shows ``while`` ops is kept so the zero-claim
cannot come back: it is the calibration that makes the other numbers readable.

What survives is the comparison, and it is measured rather than assumed. On one
toy target, lowering the same way, at the same scan lengths (2026-08-31):

===========================  ==========  =========  ==========
sampler                       HLO lines  while ops  compile s
===========================  ==========  =========  ==========
``nuts`` (max_doublings=10)        1895         14        3.07
``hmc L=10`` diagonal               907          8        1.41
``hmc L=10`` low-rank              1438          9        2.45
``barker``                          959          7        1.95
``mala``                            748          6        1.44
===========================  ==========  =========  ==========

The target is a plain anisotropic Gaussian rather than a tengri model, on
purpose: the property under test belongs to the sampler's control flow, and a
real forward model contributes its own loops (CLAUDE.md's ``age_kernel`` note
counts 6-14) which would make every number a statement about the model. The
model-side figure is ``bench/scripts/benchmark_sampler_compile.py``, and the
gap there is far larger because NUTS's tree body carries the whole forward
model.

Bounds below are deliberately loose -- roughly 1.5x the measured margin -- so
this fails on a structural change and not on a BlackJAX point release.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri.inference.backends.mcmc import _shared

pytestmark = pytest.mark.contract

#: Dimension of the toy target. The lowered control flow does not depend on it.
N_DIM = 4

#: Scan lengths. Both are loop *bounds* in the lowered program, not unrolled
#: bodies, so they move the line count by a constant and the ``while`` count
#: not at all.
N_WARMUP = 20
N_CHAIN = 20

#: Largest ``while``-op count a first-order sampler may lower to. Measured 6
#: (MALA) and 7 (Barker) against NUTS's 14; 9 leaves room for a BlackJAX
#: refactor without leaving room for a tree search.
MAX_FIRST_ORDER_WHILE_OPS = 9

#: Largest StableHLO program a first-order sampler may lower to, as a fraction
#: of NUTS's on the same target. Measured 0.51 (Barker) and 0.39 (MALA).
MAX_FIRST_ORDER_LINE_FRACTION = 0.75

#: Extra ``while`` ops the low-rank warmup may add over diagonal window
#: adaptation. Measured exactly 1, and it is in the ADAPTATION, not in the
#: sampling kernel -- ``blackjax.hmc`` is unchanged, which is what makes the
#: head-to-head against ``mcmc_hmc`` isolate the mass matrix.
MAX_LOW_RANK_EXTRA_WHILE_OPS = 3


def _target(position, data_args):
    """Anisotropic Gaussian log-density, in the ``(position, data_args)`` shape."""
    return -0.5 * jnp.sum((position / data_args) ** 2)


def _lowered_text(fn, *args):
    """StableHLO text of one jitted callable, lowered but not compiled."""
    return fn.lower(*args).as_text()


def _n_while(text):
    """How many ``while`` ops the lowered program contains (scans included)."""
    return text.count("stablehlo.while")


def _n_lines(text):
    """How many StableHLO lines the lowered program contains."""
    return len(text.splitlines())


@pytest.fixture(scope="module")
def call_args():
    """Position, keys and per-coordinate scales shared by every lowering here."""
    scales = jnp.asarray([10.0**i for i in range(N_DIM)])
    return (
        jnp.zeros(N_DIM),
        jax.random.PRNGKey(0),
        jax.random.split(jax.random.PRNGKey(1), N_CHAIN),
        scales,
    )


@pytest.fixture(scope="module")
def nuts_text(call_args):
    """NUTS's lowered program: the reference every other row is read against."""
    init, wkey, keys, scales = call_args
    return _lowered_text(
        _shared._nuts_full_scan,
        init,
        wkey,
        keys,
        _target,
        scales,
        N_WARMUP,
        10,
        False,
        0.8,
        False,
    )


def test_a_bare_scan_already_lowers_to_a_while():
    """The calibration, and the corrected mistake.

    A ``lax.scan`` of a trivial body -- statically known trip count, no
    branch, nothing adaptive -- still lowers to a ``stablehlo.while``. So a
    ``while`` count of zero is unreachable for any sampler in this codebase,
    and an assertion that a first-order sampler has none is checking a proxy
    that does not measure what it claims. This test exists so that claim
    cannot be reintroduced: if this ever passes with 0, the counting method
    changed and every bound below has to be re-derived.
    """

    def _trivial(carry, x):
        return carry + x, carry

    lowered = jax.jit(lambda xs: jax.lax.scan(_trivial, 0.0, xs)).lower(jnp.arange(8.0))
    assert _n_while(lowered.as_text()) > 0, (
        "a bare lax.scan lowered to zero while ops, so the while count in this "
        "file is measuring something other than what its bounds assume."
    )


@pytest.mark.parametrize("proposal", ["barker", "mala"])
def test_first_order_program_is_smaller_than_nuts(call_args, nuts_text, proposal):
    """Barker and MALA lower to a smaller program than NUTS on the same target.

    Both halves are checked. The line count is the quantity XLA's compile time
    tracks; the ``while`` count is the structural half -- NUTS's extra ones are
    its tree-doubling search, whose trip count depends on the trajectory rather
    than on a constant, which is also what makes a vmapped batch of NUTS chains
    run at the speed of its deepest tree.
    """
    init, wkey, keys, scales = call_args
    text = _lowered_text(
        _shared._first_order_full_scan,
        init,
        wkey,
        keys,
        _target,
        scales,
        N_WARMUP,
        proposal,
        _shared.FIRST_ORDER_TARGET_ACCEPT_RATE,
    )
    assert _n_while(text) <= MAX_FIRST_ORDER_WHILE_OPS, (
        f"{proposal} lowered to {_n_while(text)} while ops against NUTS's "
        f"{_n_while(nuts_text)}. A first-order sampler has one scan for warmup "
        "and one for sampling and nothing that searches."
    )
    assert _n_while(text) < _n_while(nuts_text)
    fraction = _n_lines(text) / _n_lines(nuts_text)
    assert fraction <= MAX_FIRST_ORDER_LINE_FRACTION, (
        f"{proposal}'s program is {_n_lines(text)} lines against NUTS's "
        f"{_n_lines(nuts_text)} ({fraction:.2f} of it). The backend's speed "
        "claim is that it is the cheaper program to compile."
    )


def test_low_rank_hmc_does_not_add_a_search(call_args):
    """The low-rank mass matrix must not turn the sampling kernel into a search.

    ``mcmc_hmc_lowrank``'s claim is that whatever it buys, it buys at
    ``mcmc_hmc``'s per-draw cost, because the kernel is ``blackjax.hmc``
    unchanged and only the warmup differs. Measured, it adds exactly **one**
    ``while`` op and ~530 HLO lines, both in the adaptation. A larger increase
    would mean the sampling half had changed too, and the head-to-head against
    ``mcmc_hmc`` would no longer isolate the mass matrix.
    """
    init, wkey, keys, scales = call_args
    diag = _lowered_text(
        _shared._hmc_full_scan, init, wkey, keys, _target, scales, N_WARMUP, 10, False, 0.85
    )
    low_rank = _lowered_text(
        _shared._hmc_low_rank_full_scan,
        init,
        wkey,
        keys,
        _target,
        scales,
        N_WARMUP,
        10,
        2,
        0.85,
    )
    extra = _n_while(low_rank) - _n_while(diag)
    assert extra <= MAX_LOW_RANK_EXTRA_WHILE_OPS, (
        f"low-rank HMC lowered to {_n_while(low_rank)} while ops against "
        f"diagonal HMC's {_n_while(diag)}, {extra} more. Only the warmup is "
        "supposed to differ."
    )


@pytest.mark.parametrize("name", ["mcmc_barker", "mcmc_mala", "mcmc_hmc_lowrank"])
def test_survey_backends_stay_experimental(name):
    """No tier was promoted by the survey that added these."""
    from tengri.inference._backend_registry import get_backend

    assert get_backend(name).tier == "experimental"


def test_first_order_reports_acceptance_not_divergences():
    """The diagnostics contract: no ``n_divergent`` key, and there must not be.

    Barker and MALA are Metropolis-corrected, so an over-large step is
    *rejected*, not flagged: there is no energy threshold and therefore no
    divergence to count. Reporting ``0`` would claim a mechanism the sampler
    does not have -- the error ``bench/reports/2026-08-30_mclmc_tuning.md``
    names for unadjusted samplers, where a zero would read as "none were found"
    while the truth is "none could be".
    """
    import inspect

    from tengri.inference.backends.mcmc.first_order import run_first_order

    body = inspect.getsource(run_first_order).split("diagnostics={", 1)[1]
    assert '"acceptance_rate"' in body
    assert '"n_divergent"' not in body, (
        "run_first_order reports a divergence count. Barker and MALA have no "
        "divergence mechanism; a count there is a claim about something that "
        "does not exist."
    )


def test_barker_and_mala_share_one_adaptation():
    """The control is a control: one code path, one warmup, one identity mass.

    Barker's published claim is robustness to a step size wrong for one
    direction's scale, and it is only testable against a sampler identical in
    every other respect. ``bench/reports/2026-08-31_catalog_preconditioning.md``
    Finding 5 is what an uncontrolled comparison costs: 40% of an apparent
    sampler deficit turned out to be a mass matrix one arm had and the other
    did not. If the two proposals ever stop sharing ``_first_order_full_scan``,
    that guarantee is gone and the comparison stops meaning anything.
    """
    import inspect

    from tengri.inference.backends.mcmc.first_order import (
        _forward,
        run_barker,
        run_first_order,
        run_mala,
    )

    # Both wrappers route through the one shared forwarder, which routes to the
    # one shared runner. Checked on ``_forward`` rather than on each wrapper's
    # own text because the wrappers spell their signatures out (so
    # ``accepts_precondition=True`` is verifiable by introspection) and no
    # longer name ``run_first_order`` directly.
    assert "run_first_order(context, proposal=proposal" in inspect.getsource(_forward)
    for fn in (run_barker, run_mala):
        assert "_forward(" in inspect.getsource(fn)

    # The following source assertions check run_first_order's internal
    # composition. These are legitimately about *structure*, not naming:
    # they verify that the function composition (full_scan for warmup,
    # chain_scan for sampling) is present. The control flow is not
    # observable at runtime without reading the implementation in detail.
    source = inspect.getsource(run_first_order)
    assert "_first_order_full_scan(" in source, (
        "run_first_order must compose _first_order_full_scan for the warmup phase"
    )
    assert "_first_order_chain_scan(" in source, (
        "run_first_order must compose _first_order_chain_scan for the sampling phase"
    )
