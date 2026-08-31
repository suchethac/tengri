# SPDX-License-Identifier: BSD-3-Clause
"""#231 -- ``pathfinder`` was quarantined for a crash nobody classified.

``scripts/validate_backends_231.py`` ran each backend in its own subprocess and
built its verdict like this::

    subprocess.run(cmd, timeout=TIMEOUT[backend], check=False)
    if out_json.exists():
        r = json.loads(out_json.read_text())
    else:
        r = {..., "error_type": "SegfaultOrAbort",
             "error_msg": "child died without writing JSON"}

**The return code is never read.** Every childless death -- a signal, an
uncaught exception, an OOM kill -- became the string ``"SegfaultOrAbort"``, and
that string became ``short_doc`` and ``tier="broken"`` for three backends. The
stored evidence does not even support the claim uniformly: in
``scripts/_backend_validation_results.json`` the ``native_vi_*`` rows on the dpl
mock are ``status: "timeout"``, not a crash at all, while their short_doc says
"segfaults".

Two real defects existed on that code path, both since fixed, and **neither is a
segfault**:

1. **blackjax >= 1.4 API drift** (fixed in ``4c1002ae7``, 2026-07-01). The call
   was ``blackjax.pathfinder(logdensity).approximate(...)``; blackjax made
   ``blackjax.pathfinder(...)`` return a ``VIAlgorithm`` that has no
   ``.approximate``. That is an ``AttributeError`` -- caught by
   :func:`test_the_instance_form_renamed_approximate_to_init` below, which is why the
   backend calls the module-level functions.
2. **Uncapped ELBO draws** (#1028, fixed in ``8807c838d``, 2026-07-10). blackjax
   defaults ``num_samples`` -- its *ELBO*-draw count, one full forward model
   each -- to 200, and nothing set it. #1029 measured that fixture at 25.65 GB
   and an OOM kill. An OOM kill is ``SIGKILL``, and it looks exactly like
   "child died without writing JSON". Pinned by
   :func:`test_the_elbo_draw_count_is_passed_explicitly`.

These are structural assertions, deliberately. A test that fitted a real
pathfinder posterior would belong to the slow tier and would prove only that one
fixture works; what wants pinning is that the two named defects cannot come back.
Measured behavior is in ``bench/reports/2026-08-31_vi_speed_evaluation.md``.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.regression_bug


def test_the_instance_form_renamed_approximate_to_init():
    """Defect 1: the instance method was **renamed**, not removed.

    Worth stating precisely, because the imprecise version ("the instance form
    is dead") is wrong and would mislead the next person. blackjax's own
    Pathfinder page still shows the instance style and it still works on 1.6.2 --
    under a different name::

        pf = blackjax.pathfinder(logdensity_fn)
        state, _ = pf.init(approx_key, w0, ftol=1e-4)  # was .approximate(...)
        samples, _ = pf.sample(sample_key, state, 5_000)

    So the 2026-05 call site was one rename away from working, and it raised
    ``AttributeError`` rather than crashing. The backend uses the module-level
    functions anyway: they take ``logdensity_fn`` explicitly and kept a stable
    signature across 1.3 -> 1.6.
    """
    blackjax = pytest.importorskip("blackjax")

    algorithm = blackjax.pathfinder(lambda x: -0.5 * (x**2).sum())
    assert not hasattr(algorithm, "approximate"), (
        "blackjax.pathfinder(logdensity) grew back an .approximate attribute. "
        "The backend deliberately uses the module-level blackjax.pathfinder."
        "approximate / .sample; re-check 4c1002ae7 before changing it."
    )
    assert hasattr(algorithm, "init"), "the instance form's replacement for .approximate"
    assert hasattr(blackjax.pathfinder, "approximate")
    assert hasattr(blackjax.pathfinder, "sample")


def test_the_two_namespaces_are_the_same_function():
    """``blackjax.pathfinder.approximate`` IS ``blackjax.vi.pathfinder.approximate``.

    ``_shared.py`` relies on the converse for the warm-start cap -- the two names
    are distinct *namespaces* holding one function object, so patching the API
    instance would not affect what ``pathfinder_adaptation`` resolves. This pins
    the identity that makes both halves of that sentence true, and it is also
    what makes tengri's call site the same entry point blackjax's own page uses
    in its module-level example.
    """
    blackjax = pytest.importorskip("blackjax")
    import blackjax.vi.pathfinder as module

    assert blackjax.pathfinder.approximate is module.approximate


def test_blackjax_still_defaults_the_elbo_draws_to_200():
    """The number the OOM diagnosis rests on, read off the installed signature.

    If upstream ever lowers this, the cap below stops being load-bearing and the
    story in #1029 needs re-reading rather than re-citing.
    """
    blackjax = pytest.importorskip("blackjax")
    import blackjax.vi.pathfinder as module

    assert inspect.signature(module.approximate).parameters["num_samples"].default == 200


def test_the_elbo_draw_count_is_passed_explicitly():
    """Defect 2: the ELBO-draw count must be ours, never blackjax's 200.

    ``n_elbo_draws`` is a **memory** knob: the draws are vmapped through the full
    forward model at every L-BFGS iterate, so peak memory goes as
    ``maxiter * n_elbo_draws * cost(one SED)``. Stan uses 25 and so do we.
    """
    from tengri.inference.backends.pathfinder import run_pathfinder

    params = inspect.signature(run_pathfinder).parameters
    assert "n_elbo_draws" in params, (
        "run_pathfinder no longer exposes n_elbo_draws; without it blackjax's "
        "default of 200 returns and with it a 26 GB peak (#1028)."
    )
    assert params["n_elbo_draws"].default == 25

    source = inspect.getsource(run_pathfinder)
    assert "num_samples=n_elbo_draws" in source, (
        "run_pathfinder must pass num_samples=n_elbo_draws to "
        "blackjax.pathfinder.approximate. blackjax calls both the posterior draws "
        "and the ELBO draws 'num_samples'; the one this caps is the expensive one."
    )


def test_the_warmstart_path_caps_the_same_draws():
    """The NUTS warm-start reaches ``approximate`` through blackjax, not through us.

    ``pathfinder_adaptation`` calls ``vi.pathfinder.approximate(...)`` with no
    ``num_samples`` and forwards its ``**extra_parameters`` to the *sampler*, so
    there is no supported override -- the cap has to be applied by rebinding the
    module attribute that ``pathfinder_adaptation`` resolves. Patching
    ``blackjax.pathfinder`` (the API instance) instead of ``blackjax.vi.pathfinder``
    (the module) is a silent no-op, which is the mistake this pins.
    """
    from tengri.inference.backends.mcmc._shared import _bounded_pathfinder_elbo_draws

    source = inspect.getsource(_bounded_pathfinder_elbo_draws)
    assert "vi.pathfinder.approximate" in source, (
        "the ELBO cap must rebind blackjax.vi.pathfinder.approximate -- the MODULE "
        "attribute pathfinder_adaptation resolves at call time."
    )


def test_the_cap_reaches_the_multipathfinder_namespace():
    """CAUSE 1 alone: the import-time binding must be rebound.

    ``blackjax/vi/multipathfinder.py`` does ``from blackjax.vi.pathfinder import
    ... approximate ...`` at *import* time, so it holds its own module-global
    binding and ``multi_approximate`` calls that bare name. A cap that rebinds
    only ``vi.pathfinder.approximate`` never reaches it.

    Deliberately split from the clamp test below. A single test asserting the
    final draw count would pass if *either* cause were fixed, and "one check that
    passes for two reasons" is the exact shape of the bug this file is about.
    """
    pytest.importorskip("blackjax")
    multipathfinder = pytest.importorskip("blackjax.vi.multipathfinder")

    from tengri.inference.backends.mcmc._shared import _bounded_pathfinder_elbo_draws

    before = multipathfinder.approximate
    with _bounded_pathfinder_elbo_draws(7):
        during = multipathfinder.approximate
    after = multipathfinder.approximate

    assert during is not before, (
        "blackjax.vi.multipathfinder.approximate was NOT rebound inside the cap. "
        "It holds an import-time binding, so patching vi.pathfinder alone leaves "
        "the multi-path route (effective_n_paths >= 2) running at 200 ELBO draws "
        "per path."
    )
    assert after is before, "the cap must restore every namespace it patched"


def test_the_cap_clamps_a_positional_num_samples():
    """CAUSE 2 alone: a positional 200 must be clamped, not merely defaulted past.

    ``multi_approximate`` forwards ``num_samples`` as the 4th POSITIONAL argument,
    so a cap expressed as a parameter default is overridden even in the right
    namespace. Asserted through the real wrapper, on one namespace only, so this
    fails if and only if the clamp regresses.
    """
    pytest.importorskip("blackjax")
    import blackjax.vi.pathfinder as pathfinder_module

    from tengri.inference.backends.mcmc._shared import _bounded_pathfinder_elbo_draws

    seen = []

    def _spy(rng_key, logdensity_fn, initial_position, num_samples=200, **kwargs):
        seen.append(num_samples)
        return "state", "info"

    original = pathfinder_module.approximate
    pathfinder_module.approximate = _spy
    try:
        with _bounded_pathfinder_elbo_draws(7):
            pathfinder_module.approximate(None, None, None, 200)  # positional, as upstream
            pathfinder_module.approximate(None, None, None)  # PATH A, no num_samples
            pathfinder_module.approximate(None, None, None, 3)  # caller wants fewer
    finally:
        pathfinder_module.approximate = original

    assert seen == [7, 7, 3], (
        f"ELBO draws reaching blackjax were {seen}, expected [7, 7, 3]: an explicit "
        "200 clamped down, an absent value capped, and a smaller request honored."
    )
