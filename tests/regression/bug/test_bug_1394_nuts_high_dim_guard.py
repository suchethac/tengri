# SPDX-License-Identifier: BSD-3-Clause
"""One NUTS high-dimension advisory, shared by all three fitting surfaces.

``method="auto"`` and ``method="mcmc"`` already switch away from NUTS above
D=20, but an explicit ``method="mcmc_nuts"`` overrides nothing — the caller has
chosen, so the cost was simply paid in silence, sometimes as an OOM.

The guard lives in one module and is called from ``Fitter.run``,
``CatalogFitter.run`` and the hierarchical flat-MCMC path. These tests pin both
the behavior and the single-implementation property: three copies of an
identical rule are what produced the two stale defaults in #1394.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from tengri.inference._dimension_guard import (
    NUTS_LIKE,
    NUTS_WARN_D,
    warn_if_nuts_high_dim,
)

pytestmark = pytest.mark.regression_bug


def _warns(method, n_dim, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fired = warn_if_nuts_high_dim(method, n_dim, surface="T", **kw)
    assert fired == (len(rec) == 1), "return value must match whether a warning was emitted"
    return fired, rec


def test_threshold_is_strict_greater_than():
    """At exactly the threshold there is no warning; one above, there is."""
    assert _warns("mcmc_nuts", NUTS_WARN_D)[0] is False
    assert _warns("mcmc_nuts", NUTS_WARN_D + 1)[0] is True


def test_only_nuts_like_samplers_warn():
    """Samplers without the O(D^2) mass matrix stay quiet.

    ``mcmc_hmc`` moved out of this list in #1454. It runs the same warmup and
    adapts the same mass matrix — it differs from NUTS in trajectory length,
    not in mass-matrix adaptation — so excluding it meant the advisory was
    unreachable for the method CLAUDE.md recommends as the remedy for this
    exact OOM. See ``test_hmc_warns_because_it_shares_the_mass_matrix`` below.
    """
    for method in ("mcmc_raytrace", "vi", "vi_nonlinear_fast", "map", "nss"):
        assert _warns(method, 10_000)[0] is False, f"{method} must not warn"


def test_hmc_warns_because_it_shares_the_mass_matrix():
    """``mcmc_hmc`` carries the same O(D^2) cost and must be reachable (#1454)."""
    assert "mcmc_hmc" in NUTS_LIKE
    assert _warns("mcmc_hmc", 10_000)[0] is True, "mcmc_hmc must warn at high D"


def test_ghmc_stays_quiet_because_it_is_always_diagonal():
    """``mcmc_ghmc`` looks like it belongs here and does not (#1454).

    Its signature carries ``dense_mass_matrix=True``, but the flag is inert:
    GHMC's momentum generator treats ``momentum_inverse_scale`` as a diagonal
    vector, so ``ghmc.py`` pins ``adapt_key = ("hmc", True)`` — always diagonal
    — regardless. It never pays O(D^2), so warning would advertise a cost that
    does not exist. Reading the signature instead of the code gives the wrong
    answer here, which is why this case is pinned explicitly.
    """
    assert "mcmc_ghmc" not in NUTS_LIKE
    assert _warns("mcmc_ghmc", 10_000)[0] is False


def test_mcmc_auto_does_not_warn():
    """``mcmc`` auto-switches to ray tracing above D=20, so it is not NUTS by then.

    Warning here would be actively wrong: it would advise the user away from a
    sampler they were never going to get.
    """
    assert "mcmc" not in NUTS_LIKE
    assert _warns("mcmc", 10_000)[0] is False


def test_unknown_dimension_is_silent():
    """A wrong D is worse than no D, so ``None`` emits nothing."""
    assert _warns("mcmc_nuts", None)[0] is False


def test_message_names_the_surface_and_the_dimension():
    """The reader must be able to tell WHICH fit and HOW big without guessing."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_if_nuts_high_dim("mcmc_nuts", 137, surface="Fitter.run")
    msg = str(rec[0].message)
    assert "Fitter.run" in msg
    assert "D=137" in msg
    # It must offer a way out, not just a complaint.
    assert "dense_mass_matrix=False" in msg
    assert "mcmc_hmc" in msg


def test_the_advice_it_gives_is_real_api():
    """Guard against advice that raises — the #1364 defect class.

    Every knob the message recommends must actually exist, or the warning sends
    the reader into a TypeError.
    """
    from tengri.inference.backends.mcmc.nuts import run_nuts

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_if_nuts_high_dim("mcmc_nuts", 99, surface="T")
    msg = str(rec[0].message)

    if "dense_mass_matrix" in msg:
        params = inspect.signature(run_nuts).parameters
        assert "dense_mass_matrix" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ), "the message recommends dense_mass_matrix=, which run_nuts does not accept"

    from tengri.inference._backend_registry import _BACKENDS

    for named in ("mcmc_hmc", "mcmc_raytrace"):
        if named in msg:
            assert named in _BACKENDS, f"message names {named!r}, which is not registered"


@pytest.mark.parametrize("surface", ["fitter", "catalog"])
def test_every_surface_calls_the_shared_guard_not_its_own_copy(surface):
    """One implementation, N call sites — never N implementations.

    The NUTS high-dimension advisory is a single guard checked from all surfaces.
    Verify both that each surface calls it and that there is no local redefinition.
    """
    if surface == "fitter":
        from tengri.inference import fitter as mod

        src = inspect.getsource(mod.Fitter.run)
    else:
        from tengri.inference import catalog_fitter as mod

        src = inspect.getsource(mod._CatalogFitterOriginal.run)

    assert "_warn_if_nuts_high_dim(" in src, f"{surface} must call the shared guard"

    # Verify no locally re-implemented threshold comparison.
    # The guard lives in _dimension_guard and has a single threshold (NUTS_WARN_D).
    # Hardcoding "> 30" or "30 <" would be a duplicate implementation.
    assert "> 30" not in src and "30 <" not in src, (
        f"{surface} appears to hardcode its own threshold instead of using the shared one"
    )
