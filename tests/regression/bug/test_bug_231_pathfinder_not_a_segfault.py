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
   :func:`test_the_instance_form_has_no_approximate` below, which is why the
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


def test_the_instance_form_has_no_approximate():
    """Defect 1: ``blackjax.pathfinder(f).approximate`` does not exist.

    If this ever starts passing, the instance form became viable again -- which
    is information, not permission. The module-level call works on both.
    """
    blackjax = pytest.importorskip("blackjax")

    algorithm = blackjax.pathfinder(lambda x: -0.5 * (x**2).sum())
    assert not hasattr(algorithm, "approximate"), (
        "blackjax.pathfinder(logdensity) grew back an .approximate attribute. "
        "The backend deliberately uses the module-level blackjax.pathfinder."
        "approximate / .sample because those kept a stable signature across the "
        "1.3 -> 1.6 window; re-check 4c1002ae7 before changing it."
    )
    assert hasattr(blackjax.pathfinder, "approximate")
    assert hasattr(blackjax.pathfinder, "sample")


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
