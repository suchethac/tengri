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


def test_the_elbo_draw_count_is_passed_explicitly(monkeypatch):
    """Defect 2: the ELBO-draw count must be ours, never blackjax's 200.

    ``n_elbo_draws`` is a **memory** knob: the draws are vmapped through the full
    forward model at every L-BFGS iterate, so peak memory goes as
    ``maxiter * n_elbo_draws * cost(one SED)``. Stan uses 25 and so do we.

    Behavioral form: verifies that the pathfinder backend actually receives and
    forwards its n_elbo_draws parameter. The forwarding is checked by replacing
    the backend runner with a stub that records what n_elbo_draws value it was
    called with, then comparing to the caller's input.
    """
    import dataclasses
    from contextlib import suppress

    pytest.importorskip("blackjax")
    from tengri.inference._backend_registry import _BACKENDS
    from tengri.inference.backends.pathfinder import run_pathfinder
    from tengri.inference.fitter import Fitter

    params = inspect.signature(run_pathfinder).parameters
    assert "n_elbo_draws" in params, (
        "run_pathfinder no longer exposes n_elbo_draws; without it blackjax's "
        "default of 200 returns and with it a 26 GB peak (#1028)."
    )
    assert params["n_elbo_draws"].default == 25

    # Stub that records what n_elbo_draws value it receives.
    recorded = {}

    class _Reached(Exception):
        pass

    def _stub_runner(*args, **kwargs):
        recorded.update(kwargs)
        raise _Reached

    class _Nothing:
        def __call__(self, *args, **kwargs):
            return None

        def __bool__(self):
            return False

    class _Spec:
        stochastic = False
        free_params = ("param_0",)
        all_params = ("param_0",)
        n_grid = 8
        n_free = 1

        def __getattr__(self, name):
            return None

    class _StubFitter:
        spec = _Spec()
        _lut_bias_checked = True

        def __getattr__(self, name):
            return _Nothing()

    # Replace the pathfinder backend with our stub.
    entry = _BACKENDS["pathfinder"]
    monkeypatch.setitem(_BACKENDS, "pathfinder", dataclasses.replace(entry, runner=_stub_runner))

    stub = _StubFitter()
    sentinel_n_elbo_draws = 42

    with suppress(_Reached):
        Fitter.run(stub, "pathfinder", key=0, n_elbo_draws=sentinel_n_elbo_draws)

    assert "n_elbo_draws" in recorded, (
        "pathfinder backend runner was never called with n_elbo_draws, or the "
        "parameter was dropped before reaching the runner"
    )
    assert recorded["n_elbo_draws"] == sentinel_n_elbo_draws, (
        f"Fitter.run was asked for n_elbo_draws={sentinel_n_elbo_draws} but the "
        f"pathfinder runner received n_elbo_draws={recorded['n_elbo_draws']!r} -- "
        f"the parameter is not being forwarded correctly"
    )


def test_the_warmstart_path_caps_the_same_draws():
    """The NUTS warm-start reaches ``approximate`` through blackjax, not through us.

    ``pathfinder_adaptation`` calls ``vi.pathfinder.approximate(...)`` with no
    ``num_samples`` and forwards its ``**extra_parameters`` to the *sampler*, so
    there is no supported override -- the cap has to be applied by rebinding the
    module attribute that ``pathfinder_adaptation`` resolves. Patching
    ``blackjax.pathfinder`` (the API instance) instead of ``blackjax.vi.pathfinder``
    (the module) is a silent no-op, which is the mistake this pins.

    Kept as source-code assertion: this pins the critical architectural choice
    that the warmstart path must patch the MODULE-level function, not the instance,
    because pathfinder_adaptation resolves the name at call time and reads what
    the module has at that moment. Testing this behaviorally would require
    constructing a full sampling path, which would be fragile; the source assertion
    directly verifies the mechanism.
    """
    from tengri.inference.backends.mcmc._shared import _bounded_pathfinder_elbo_draws

    source = inspect.getsource(_bounded_pathfinder_elbo_draws)
    assert "vi.pathfinder.approximate" in source, (
        "the ELBO cap must rebind blackjax.vi.pathfinder.approximate -- the MODULE "
        "attribute pathfinder_adaptation resolves at call time."
    )
