# SPDX-License-Identifier: BSD-3-Clause
"""One high-dimension advisory for NUTS, shared by every fitting surface.

``Fitter``, ``CatalogFitter`` and ``PopulationFitter`` all let a caller ask for
``mcmc_nuts`` explicitly, and all three can be handed a problem far larger than
NUTS warmup comfortably fits. The advice is identical in each case, so it lives
here once rather than as three copies that drift, the exact failure mode behind
the two stale defaults fixed in #1394.

Relationship to the *other* D thresholds
----------------------------------------
``fitter._AUTO_D_THRESHOLD`` and ``fitter._MCMC_AUTO_D_THRESHOLD`` (both 20)
decide what ``method="auto"`` / ``method="mcmc"`` **silently pick**: at or below
20 free parameters they select NUTS, above it they switch to geoVI / ray
tracing. Those never fire for an explicit ``method="mcmc_nuts"``, because the
caller has already made the choice.

:data:`NUTS_WARN_D` is that missing case: the caller asked for NUTS by name, so
nothing overrides them, they just get told what it will cost. The two numbers
are deliberately different. Auto-switching is a decision made *for* the user and
should be conservative; a warning is information handed *to* the user and should
not cry wolf in the 20-30 band where NUTS is expensive but still routine.

Why 30
------
Warmup memory is dominated by the mass matrix and the per-step trajectory
buffer. With ``dense_mass_matrix=True`` the mass matrix alone is O(D^2), so the
cost curve steepens rather than shifting: measured peaks of 3-6 GB on small
photometry problems (D <= 7) reach 20+ GB by D ~ 8 with a dense_basis mean SFH,
and a 137-D stochastic problem is firmly in ray-tracing territory. 30 sits above
the routine band and below the range where NUTS reliably needs a machine most
users do not have.
"""

from __future__ import annotations

import warnings

#: Free-parameter count above which an explicit NUTS request draws an advisory.
#: Override per-project via the ``nuts_warn_d`` inference default.
NUTS_WARN_D: int = 30

#: Methods that run a NUTS-style warmup and therefore carry the O(D^2)
#: mass-matrix cost.
#:
#: ``"mcmc"`` is absent deliberately: it auto-switches to ray tracing above
#: ``_MCMC_AUTO_D_THRESHOLD``, so by the time D is large it is no longer NUTS.
#:
#: ``"mcmc_ghmc"`` is absent for a different reason, and the distinction matters
#: because its signature says otherwise. GHMC's momentum generator treats
#: ``momentum_inverse_scale`` as a diagonal vector, so ``ghmc.py`` pins
#: ``adapt_key = ("hmc", True)``, always diagonal, regardless of the
#: ``dense_mass_matrix=True`` default in its signature. That default is inert;
#: GHMC never allocates a dense mass matrix and never pays O(D^2). Adding it
#: here would emit an advisory for a cost the method does not incur (#1454).
#:
#: ``"mcmc_hmc"`` IS included: fixed-length HMC differs from NUTS in trajectory
#: length, not in mass-matrix adaptation.
NUTS_LIKE: frozenset[str] = frozenset({"mcmc_nuts", "mcmc_hmc"})


def _threshold() -> int:
    """The configured warning threshold, falling back to :data:`NUTS_WARN_D`."""
    try:
        from tengri.parameters.defaults import get_inference_defaults

        return int(get_inference_defaults().get("nuts_warn_d", NUTS_WARN_D))
    except Exception:
        return NUTS_WARN_D


def warn_if_nuts_high_dim(method, n_dim, *, surface, stacklevel=3):
    """Advise, but do not refuse, when NUTS is asked for at high dimension.

    Parameters
    ----------
    method : str
        Resolved (canonical) method name. Only members of :data:`NUTS_LIKE`
        trigger anything, so this is safe to call unconditionally on any
        dispatch path.
    n_dim : int or None
        Free-parameter count for the problem actually being fitted. ``None``
        means the surface could not determine it cheaply, in which case no
        warning is emitted, a wrong D would be worse than none.
    surface : str
        Human name of the calling entry point, e.g. ``"Fitter.run"``. Appears in
        the message so the reader knows which of the three fits is being
        described.
    stacklevel : int
        Passed to :func:`warnings.warn`. Default 3 points at the user's own
        ``run()`` call rather than at this helper.

    Returns
    -------
    bool
        Whether a warning was emitted. Returned for tests; callers ignore it.

    Notes
    -----
    This is advisory by design. NUTS at D > 30 is expensive, not wrong, and it
    remains the right answer when a user needs exact posterior samples and has
    the memory for them. Refusing would make the honest choice unreachable; see
    the ``tier="broken"`` backends, which stay selectable for the same reason.
    """
    if method not in NUTS_LIKE or n_dim is None:
        return False
    threshold = _threshold()
    if n_dim <= threshold:
        return False
    warnings.warn(
        f"{surface}: method='{method}' at D={n_dim} free parameters "
        f"(> {threshold}). NUTS warmup memory is dominated by the mass matrix, "
        f"which is O(D^2) when dense_mass_matrix=True, problems this size have "
        f"been measured at 20+ GB and can OOM the process. Options: pass "
        f"dense_mass_matrix=False to drop to a diagonal metric, use "
        f"method='mcmc_hmc' for a fixed trajectory length, or "
        f"method='mcmc_raytrace' / a VI method, which is what method='mcmc' "
        f"selects automatically above D=20. Proceeding with NUTS as requested.",
        UserWarning,
        stacklevel=stacklevel,
    )
    return True
