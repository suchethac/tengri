# SPDX-License-Identifier: BSD-3-Clause
"""The analytic metric on the batched catalog path: how it crosses the seam.

Phase 3 measured catalog ChEES against catalog HMC and found ChEES 2.5x slower
and converging on 4 galaxies of 64 against 15. It also named the cause: the
catalog engine could not thread ``precondition=``, so catalog ChEES ran with an
**identity** ``inverse_mass_matrix`` -- exactly the configuration Phase 2 had
already shown clears nothing. See
``bench/reports/2026-08-31_catalog_batched_samplers.md`` Finding 4.

What blocked it was not an oversight but a shape mismatch, and these tests pin
the resolution rather than the prose:

* The metric is ``J^T N^-1 J + I`` built at a galaxy's MAP from that galaxy's
  noise, so **it has a galaxy axis**. It cannot be hoisted out of the ``lax.map``
  as a shared constant, and one built from galaxy 0 and broadcast would silently
  whiten every galaxy against the wrong geometry.
* ``prepare_preconditioning`` returns a Python **closure** over one concrete
  matrix, which is a static value to JAX. There is no shape for a per-lane static
  value to take. The transform has to be a *traced* argument.
* The scan cores in ``_shared`` take ``logdensity_fn_2arg`` as a **static**
  argument and only ``init_flat``, the keys and ``data_args`` as traced ones. So
  the transform rides ``data_args``, and the wrapper that reads it must be built
  once and cached -- a fresh wrapper per fit would re-trace the whole sampler
  every call and turn every warm run back into a cold one.

Contract tests: they assert the shape of the seam. Whether the metric *helps* is
measured in ``bench/reports/2026-08-31_catalog_preconditioning.md``, not asserted
here.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract


def _is_docstring(node) -> bool:
    """Whether an AST statement is a bare string expression."""
    import ast

    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


class TestTheArgumentReachesTheEngine:
    def test_the_catalog_sampler_accepts_precondition(self):
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal

        sig = inspect.signature(_CatalogFitterOriginal._run_native_mcmc)
        assert "precondition" in sig.parameters
        assert sig.parameters["precondition"].default is None, (
            "preconditioning must stay opt-in (#1397): whitening is not a default"
        )

    def test_the_engine_builder_accepts_it_too(self):
        from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

        sig = inspect.signature(build_catalog_mcmc_engine)
        assert "precondition" in sig.parameters
        assert sig.parameters["precondition"].default is None

    def test_it_is_forwarded_rather_than_swallowed(self):
        """``**kwargs`` on ``run`` would accept the word and drop it silently.

        The failure this guards is the one ``run_ghmc``'s ``target_accept_rate``
        records: a surface that takes an argument its kernel never sees.
        """
        import tengri.inference.catalog_fitter as cf

        src = inspect.getsource(cf._CatalogFitterOriginal._run_native_mcmc)
        assert "precondition=precondition" in src


class TestTheMetricIsPerGalaxyAndTraced:
    def test_the_traced_builder_is_what_the_engine_calls(self):
        """Not ``prepare_preconditioning``, which cannot be traced at all."""
        import ast

        import tengri.inference.backends.mcmc.catalog as cat

        # Code only: the docstring legitimately *names* the non-traced entry
        # point in explaining why it cannot be used here.
        tree = ast.parse(inspect.getsource(cat.build_catalog_mcmc_engine).lstrip())
        body = tree.body[0].body
        code = "\n".join(ast.unparse(node) for node in body if not _is_docstring(node))
        assert "traced_preconditioner" in code
        assert "prepare_preconditioning" not in code

    def test_the_transform_rides_the_traced_pytree_not_a_closure(self):
        """``(A, data_args)``, so ``A`` batches with the data.

        A tuple rather than an extra ``data_args`` dict key on purpose: every
        function in ``_shared`` treats ``data_args`` as opaque and only forwards
        it to the log-density, whereas a new dict key would reach the model's own
        jitted log-density and change the pytree it was built for.
        """
        from tengri.inference.backends.mcmc.catalog import _preconditioned_logdensity

        def base(xi, data_args):
            return float(xi[0]) + 0.0 * data_args["d"]

        wrapped = _preconditioned_logdensity(base, 0.5)
        params = list(inspect.signature(wrapped).parameters)
        assert len(params) == 2
        src = inspect.getsource(wrapped)
        assert "matrix, data_args = precond_args" in src

    def test_the_wrapper_is_cached_so_the_warm_path_stays_warm(self):
        """A new function object per fit would re-trace the sampler every call.

        The scan cores key their compilation on ``logdensity_fn_2arg``'s
        *identity* (it is a ``static_argnums`` entry), which is why
        ``_get_flat_logdensity`` caches the base function on the Model. A wrapper
        that did not do the same would make every "warm" number a cold one, and
        nothing would raise.
        """
        from tengri.inference.backends.mcmc.catalog import _preconditioned_logdensity

        def base(xi, data_args):
            return 0.0

        assert _preconditioned_logdensity(base, 0.5) is _preconditioned_logdensity(base, 0.5)

    def test_a_different_strength_is_a_different_wrapper(self):
        """A density wrapped at one strength is not the one another gives (#1442).

        Sharing a cache entry across strengths would sample the wrong basis at a
        step size dual-averaged for the other one, silently.
        """
        from tengri.inference.backends.mcmc.catalog import _preconditioned_logdensity

        def base(xi, data_args):
            return 0.0

        assert _preconditioned_logdensity(base, 0.5) is not _preconditioned_logdensity(base, 1.0)


class TestOffIsOff:
    def test_none_leaves_the_base_logdensity_untouched(self):
        """The unpreconditioned program must be the one that compiled before.

        ``strength is None`` is a **trace-time** branch, resolved from a concrete
        Python value at build time, so a caller who did not ask for whitening
        gets byte-for-byte the graph they had. That is what lets the Phase 3 rows
        stay directly comparable.
        """
        from tengri.inference.preconditioning import _resolve_whitening_strength

        assert _resolve_whitening_strength(None, 3) is None
        assert _resolve_whitening_strength(False, 3) is None
        assert _resolve_whitening_strength(0.0, 3) is None, (
            "zero strength is the identity transform: enabling it would build a "
            "Hessian, factorize it and multiply by I -- all cost, no effect"
        )

    def test_true_resolves_to_half_not_full_whitening(self):
        """#1442: full whitening amplifies a misspecified metric without bound."""
        from tengri.inference.preconditioning import (
            DEFAULT_WHITENING_STRENGTH,
            _resolve_whitening_strength,
        )

        assert DEFAULT_WHITENING_STRENGTH == 0.5
        assert _resolve_whitening_strength(True, 3) == 0.5


class TestTheRefusalNowPointsSomewhere:
    def test_chees_dense_mass_refusal_names_precondition_as_the_route(self):
        """It used to say the catalog path "does not yet thread" the metric.

        It now does, so the message must send the caller there rather than to a
        different sampler. A refusal whose advice is stale is worse than none.
        """
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal

        cat = _CatalogFitterOriginal.__new__(_CatalogFitterOriginal)
        with pytest.raises(ValueError, match="precondition=True") as exc:
            _CatalogFitterOriginal._run_native_mcmc(
                cat, "mcmc_chees", key=None, dense_mass_matrix=True
            )
        assert "does not yet thread" not in str(exc.value)


class TestNoTierMoved:
    def test_chees_is_still_experimental(self):
        from tengri.inference._backend_registry import get_backend

        assert get_backend("mcmc_chees").tier == "experimental"

    def test_the_quarantine_held(self):
        from tengri.inference._backend_registry import get_backend
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal

        for name in ("mcmc_ghmc", "mcmc_mclmc"):
            assert get_backend(name).tier == "broken"
            assert name not in _CatalogFitterOriginal._MCMC_VMAPPABLE
