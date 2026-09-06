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
        """``precondition`` is declared on both ends and named in the handoff.

        The failure this guards is the one ``run_ghmc``'s ``target_accept_rate``
        records: a surface that takes an argument its kernel never sees.

        **Part behavioral, part source-spelling — deliberately, and here is why.**
        The behavioral form would monkeypatch ``build_catalog_mcmc_engine`` with a
        recorder and assert the caller's value arrives. That works on the
        single-galaxy ``Fitter``, whose runner is reachable through
        ``_BACKENDS[name].runner``; it does **not** work here. ``_run_native_mcmc``
        does not dispatch through the registry, and reaching its engine-builder
        call means surviving the catalog data pipeline first: a synthetic fitter
        gets as far as ``ValueError: could not convert <stub> to a NumPy dtype``,
        because real per-galaxy flux and noise arrays are converted before the
        builder is ever called. Instrumenting it needs a real catalog fixture,
        which does not belong in the fast contract tier.

        So this asserts the two things that *are* checkable without one:

        1. the parameter is declared on the surface (not absorbed by ``**kwargs``,
           which is the regression that would let a caller pass it into a void), and
        2. the name appears in the **code** of the handoff — AST-parsed with
           docstrings stripped, so a mention in prose cannot satisfy it.

        What stays unproven is the assignment itself: code that writes
        ``precondition=None`` beside a read of the argument satisfies both. That
        limit is the reason to prefer a recorder wherever one is reachable.
        """
        import ast

        import tengri.inference.catalog_fitter as cf

        sig = inspect.signature(cf._CatalogFitterOriginal._run_native_mcmc)
        assert "precondition" in sig.parameters, (
            "precondition must be a named parameter, not absorbed by **kwargs"
        )
        assert sig.parameters["precondition"].default is None, (
            "preconditioning must stay opt-in (#1397)"
        )

        tree = ast.parse(inspect.getsource(cf._CatalogFitterOriginal._run_native_mcmc).lstrip())
        code = "\n".join(
            ast.unparse(node) for node in tree.body[0].body if not _is_docstring(node)
        )
        assert "precondition" in code, (
            "_run_native_mcmc names precondition nowhere in its body: the surface "
            "accepts the argument and its kernel never sees it"
        )


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
        """``(A, data_args)`` unpacked inside, so ``A`` batches with the data.

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
        assert params == ["zeta", "precond_args"], (
            "parameters must be zeta and precond_args; precond_args carries the "
            "(matrix, data_args) tuple"
        )

        # The signature alone cannot show the tuple is unpacked in the right
        # order, and "it did not raise" cannot either -- a swapped unpack that
        # happens to broadcast would pass both. So vary the MATRIX half and
        # require the value to move: that is only true if the first element is
        # read as the matrix and actually applied.
        import numpy as np

        zeta = np.array([1.0])
        data_args = {"d": np.array([1.0])}

        one = wrapped(zeta, (np.eye(1), data_args))
        three = wrapped(zeta, (3.0 * np.eye(1), data_args))

        assert np.isfinite(one) and np.isfinite(three), (
            f"the preconditioned log-density must stay finite; got {one} and {three}"
        )
        assert one != three, (
            "scaling the preconditioner left the log-density unchanged: the matrix "
            "half of precond_args is being ignored, or the tuple is unpacked as "
            "(data_args, matrix) and the matrix never reaches the transform"
        )

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


class TestTheMassMatrixControlIsReachable:
    """Both arms of the HMC-vs-ChEES confound must be settable from a call.

    The batched path always gives ``mcmc_nuts``/``mcmc_hmc`` a warmup-estimated
    mass matrix while ChEES's ``inverse_mass_matrix`` stays at ones, so a
    head-to-head between them compares **two** differences: the trajectory
    length, and a second adaptation only one arm has. Neither knob was reachable
    from ``CatalogFitter.run`` before, which made the confound unmeasurable
    rather than merely unmeasured.
    """

    def test_the_catalog_sampler_accepts_chees_mass_matrix_estimation(self):
        """Declared on the surface, opt-in by default, and named in the handoff.

        Consolidated: a second test asserted the same signature and default under
        the name ``test_it_is_forwarded_rather_than_swallowed`` and added only a
        call that swallowed every exception, so it could not fail for the reason
        its name gave. See the ``precondition`` test above for why the recorder
        form is unavailable on this surface -- ``_run_native_mcmc`` does not
        dispatch through ``_BACKENDS``, and its engine-builder call sits behind
        the catalog data pipeline, which rejects a synthetic fitter while
        converting per-galaxy arrays.
        """
        import ast

        import tengri.inference.catalog_fitter as cf

        sig = inspect.signature(cf._CatalogFitterOriginal._run_native_mcmc)
        assert "mass_matrix_estimation" in sig.parameters, (
            "mass_matrix_estimation must be a named parameter, not absorbed by **kwargs"
        )
        assert sig.parameters["mass_matrix_estimation"].default is None, (
            "the analytic metric stays the default geometry; the ensemble "
            "estimate is an ablation (run_chees's own warning)"
        )

        tree = ast.parse(inspect.getsource(cf._CatalogFitterOriginal._run_native_mcmc).lstrip())
        code = "\n".join(
            ast.unparse(node) for node in tree.body[0].body if not _is_docstring(node)
        )
        assert "mass_matrix_estimation" in code, (
            "_run_native_mcmc names mass_matrix_estimation nowhere in its body: "
            "the surface accepts the argument and its kernel never sees it"
        )

    def test_dense_mass_matrix_is_still_refused_for_chees(self):
        """The new knob must not have opened a door the old refusal closed."""
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal

        cat = _CatalogFitterOriginal.__new__(_CatalogFitterOriginal)
        with pytest.raises(ValueError, match="dense_mass_matrix"):
            _CatalogFitterOriginal._run_native_mcmc(
                cat, "mcmc_chees", key=None, dense_mass_matrix=True
            )


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
