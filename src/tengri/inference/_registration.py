"""Backend registry initialization for ``tengri.inference``.

This module is imported for its side effects — every
``@register_backend(...)`` decorator at module load time inserts a
``BackendEntry`` into ``_BACKENDS`` so that ``Fitter.run(method=...)``
can dispatch by name.

It lives in its own module (rather than at the bottom of ``fitter.py``)
to keep that file focused on the ``Fitter`` class itself. Mirrors the
``forward/_kernels`` layout (ADR-0004): adapter registration is a
package-level concern, not an orchestrator-level one.

Adding a new backend
--------------------
1. Implement ``def run_X(context, *, key, init_from=None, ...)`` in
   ``backends/X.py``. The context is an :class:`InferenceContext`;
   read forward-model state through its accessors.
2. Add an entry below: ``register_backend("X", tier="experimental",
   short_doc=..., legacy_fitter=False)(run_X)``.
3. Add a conformance smoke in ``tests/unit/inference/`` (the
   parametrised suite in ``test_backend_conformance.py`` will pick it
   up automatically).

See ADR-0010 for the full Protocol contract.
"""

from __future__ import annotations

from tengri.inference._backend_registry import register_backend
from tengri.inference.backends.evidence import run_nss as _ctx_run_nss
from tengri.inference.backends.map_dispatch import (
    run_laplace as _ctx_run_laplace,
    run_map as _ctx_run_map,
    run_pathfinder as _ctx_run_pathfinder,
)
from tengri.inference.backends.mcmc import (
    run_adjusted_mclmc as _ctx_run_adjusted_mclmc,
    run_dynamic_hmc as _ctx_run_dynamic_hmc,
    run_ghmc as _ctx_run_ghmc,
    run_hmc as _ctx_run_hmc,
    run_mclmc as _ctx_run_mclmc,
    run_nuts as _ctx_run_nuts,
    run_raytrace as _ctx_run_raytrace,
)
from tengri.inference.backends.mcmc.elliptical_slice import (
    run_elliptical_slice_fitter as _ctx_run_elliptical_slice,
)
from tengri.inference.backends.vi.native import run_native_vi as _ctx_run_native_vi
from tengri.inference.backends.vi.nifty import (
    run_nifty_fast_vi as _ctx_run_nifty_fast_vi,
    run_nifty_vi as _ctx_run_nifty_vi,
)


def _mcmc_auto_pick(context, *, key, init_from=None, **kw):
    """``mcmc`` auto-dispatcher: NUTS for low-D, raytrace for high-D.

    Threshold is looked up at call time (not import time) so this
    module has no import dependency on ``fitter.py`` — keeps the
    package import graph one-way and lets ``inference/__init__.py``
    rely on plain alphabetical import ordering.
    """
    from tengri.inference.fitter import _MCMC_AUTO_D_THRESHOLD

    if context.spec.n_free <= _MCMC_AUTO_D_THRESHOLD:
        return _ctx_run_nuts(context, key=key, init_from=init_from, **kw)
    return _ctx_run_raytrace(context, key=key, init_from=init_from, **kw)


# ── Primary backends ─────────────────────────────────────────────────────
register_backend(
    "map",
    tier="primary",
    short_doc="Adam MAP optimization",
    requires=("optax",),
    legacy_fitter=False,
)(_ctx_run_map)

# NIFTy geoVI/MGVI — sample_mode flags select geoVI vs MGVI.
register_backend(
    "vi",
    tier="primary",
    short_doc="NIFTy geoVI variational inference",
    aliases=("vi_nonlinear",),
    requires=("nifty8",),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_nifty_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "nonlinear_resample", **kw},
    )
)

register_backend(
    "vi_nonlinear_fast",
    tier="primary",
    short_doc="NIFTy geoVI without Python logging",
    requires=("nifty8",),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_nifty_fast_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "nonlinear_resample", **kw},
    )
)

register_backend(
    "vi_linear",
    tier="experimental",
    short_doc="NIFTy MGVI standard with logging",
    requires=("nifty8",),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_nifty_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "linear_resample", **kw},
    )
)

register_backend(
    "vi_linear_fast",
    tier="experimental",
    short_doc="NIFTy MGVI without Python logging",
    requires=("nifty8",),
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_nifty_fast_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "linear_resample", **kw},
    )
)

register_backend(
    "native_vi_nonlinear",
    tier="experimental",
    short_doc="Pure JAX geoVI variational inference",
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_native_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "geovi", **kw},
    )
)

register_backend(
    "native_vi_linear",
    tier="experimental",
    short_doc="Pure JAX MGVI via lax.while_loop",
    legacy_fitter=False,
)(
    lambda context, *, key, init_from=None, **kw: _ctx_run_native_vi(
        context,
        key=key,
        init_from=init_from,
        **{"sample_mode": "linear", **kw},
    )
)

# ``mcmc`` is an auto-dispatcher (NUTS for low-D, raytrace for high-D).
# It picks the concrete runner from ``context.spec.n_free``.
register_backend(
    "mcmc",
    tier="primary",
    short_doc="Auto MCMC: NUTS for low-D, raytrace for high-D",
    requires=("blackjax",),  # NUTS branch needs it; raytrace branch is pure JAX
    legacy_fitter=False,
)(_mcmc_auto_pick)

register_backend(
    "mcmc_nuts",
    tier="primary",
    short_doc="No-U-Turn Sampler",
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_nuts)

register_backend(
    "mcmc_raytrace",
    tier="primary",
    short_doc="Ray-tracing ensemble sampler (high-D)",
    legacy_fitter=False,
)(_ctx_run_raytrace)

# ── Experimental backends ────────────────────────────────────────────────
register_backend(
    "mcmc_hmc",
    tier="experimental",
    short_doc="Hamiltonian Monte Carlo",
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_hmc)

register_backend(
    "mcmc_dynamic_hmc",
    tier="experimental",
    short_doc="Dynamic HMC with adaptive step size",
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_dynamic_hmc)

register_backend(
    "mcmc_ghmc",
    tier="experimental",
    short_doc="Generalized HMC",
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_ghmc)

register_backend(
    "mcmc_mclmc",
    tier="experimental",
    short_doc="Microcanonical Langevin Monte Carlo",
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_mclmc)

register_backend(
    "mcmc_adjusted_mclmc",
    tier="experimental",
    short_doc="Adjusted microcanonical Langevin sampler",
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_adjusted_mclmc)

register_backend(
    "mcmc_ess",
    tier="experimental",
    short_doc="Elliptical slice sampling",
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_elliptical_slice)

register_backend(
    "nss",
    tier="experimental",
    short_doc="Nested sampling for model comparison",
    legacy_fitter=False,
)(_ctx_run_nss)

register_backend(
    "laplace",
    tier="experimental",
    short_doc="Laplace approximation",
    legacy_fitter=False,
)(_ctx_run_laplace)

register_backend(
    "pathfinder",
    tier="experimental",
    short_doc="Pathfinder variational inference",
    requires=("blackjax",),
    legacy_fitter=False,
)(_ctx_run_pathfinder)
