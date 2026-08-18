# SPDX-License-Identifier: BSD-3-Clause
"""PopulationSpecView — batched-sample wrapper over a template :class:`Parameters`.

A thin view that the :class:`PopulationSEDModel` returns from its
``.spec`` property. It duck-types ``Parameters`` so the standard
:class:`Fitter` machinery doesn't need to know it's looking at a
hierarchical fit — the **only** observable difference vs the template
spec is that :meth:`sample` returns per-galaxy params with shape
``(N,)`` (or ``(N, n_grid)`` for stochastic-SFH ``psd_xi``).

See the implementation plan
``docs/internal/plans/2026-05-22-population-sed-batched-forward.md``
for design rationale (Task 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from tengri.parameters.parameters import Parameters

__all__ = ["PopulationSpecView"]


class PopulationSpecView:
    """Batched-sample view over a template :class:`Parameters`.

    Implements the implicit :class:`Parameters` Protocol that
    :class:`Fitter` consumes:

    - ``free_params``: list of free-parameter names
    - ``all_params``: list of all parameter names
    - ``fixed_params``: list of fixed-parameter names
    - ``n_free``: number of free parameters
    - ``stochastic``: boolean flag
    - ``_distributions``: dict of name → Distribution
    - ``get_distribution(name)``
    - ``get_fixed_values()``
    - ``sample(key)``: **vmapped over N for per-galaxy params**

    Parameters
    ----------
    template : Parameters
        The SED template's spec. Provides the scalar Protocol surface.
    n_galaxies : int
        Population size. Determines the leading axis of batched samples.
    shared : tuple[str, ...]
        Names of parameters tied across the population. Samples of
        these stay scalar; samples of every other free parameter get
        a leading ``(N,)`` axis.

    Notes
    -----
    The wasteful path: :meth:`sample` is implemented as
    ``jax.vmap(template.sample)`` over a key split — which draws
    ``(N, ...)`` for *every* parameter — then overwrites shared
    parameter values with a single draw. The wasted draws for shared
    params (N-1 RNG calls per shared name) are acceptable because
    ``sample`` runs at init only, not in the inference hot loop.

    JIT: the underlying ``template.sample`` is JIT-safe; vmap composes
    cleanly.
    """

    def __init__(
        self,
        template: Parameters,
        n_galaxies: int,
        shared: tuple[str, ...],
    ) -> None:
        if n_galaxies < 1:
            raise ValueError(f"PopulationSpecView needs n_galaxies >= 1; got {n_galaxies}")
        self._template = template
        self._n_galaxies = n_galaxies
        self._shared = frozenset(shared)

    # ── Pass-throughs to the template ──────────────────────────────────

    @property
    def free_params(self) -> list[str]:
        """Names of free parameters (identical to the template)."""
        return self._template.free_params

    @property
    def all_params(self) -> list[str]:
        return self._template.all_params

    @property
    def fixed_params(self) -> list[str]:
        return self._template.fixed_params

    @property
    def n_free(self) -> int:
        return self._template.n_free

    @property
    def n_latent(self) -> int:
        """Flattened dimensionality of the batched free-parameter sample (#1408).

        Per-galaxy parameters count ``n_galaxies`` times (via
        :meth:`param_init_shape`) and shared parameters once, plus the
        stochastic field latent when present — the dimension hierarchical
        samplers actually operate in. Mirrors the accounting of
        ``Fitter._initialize_unbounded``; agreement with the engine's
        ``d_total`` is asserted in
        ``tests/inference/test_noise_broadcast_fix_1303.py``.
        """
        import numpy as np

        total = 0
        for name in self.free_params:
            shape = self.param_init_shape(name)
            total += int(np.prod(shape)) if shape else 1
        if self.stochastic:
            psd_shape = getattr(self, "psd_xi_init_shape", None) or (self._template.n_grid,)
            if callable(psd_shape):
                psd_shape = psd_shape()
            total += int(np.prod(psd_shape))
        return total

    @property
    def stochastic(self) -> bool:
        return self._template.stochastic

    @property
    def _distributions(self):
        """Underlying Distribution dict — Fitter sometimes reaches here."""
        return self._template._distributions

    @property
    def n_grid(self) -> int:
        """Stochastic-SFH grid size — delegates to the template's public ``n_grid``.

        Mirrors :attr:`tengri.parameters.parameters.Parameters.n_grid` so the
        population spec satisfies the same Parameters-protocol surface the
        scalar spec does. ``Fitter.compile_signature`` reads ``spec.n_grid``
        for stochastic fits regardless of topology; without this the
        hierarchical path raised ``AttributeError`` (suchethac/tengri#711, Gap 1).
        """
        return self._template.n_grid

    @property
    def _n_grid(self):
        """Private alias kept for back-compat — see the public :attr:`n_grid`."""
        return getattr(self._template, "_n_grid", None)

    def get_distribution(self, name: str):
        return self._template.get_distribution(name)

    def get_fixed_values(self) -> dict[str, float]:
        """Fixed values from the template (broadcast across galaxies)."""
        return self._template.get_fixed_values()

    def param_init_shape(self, name: str) -> tuple[int, ...]:
        """Initial-xi shape for one free parameter.

        For unbounded-space initialization in the Fitter (paper §2
        Standardized Inference): per-galaxy free parameters get a
        leading ``(N,)`` axis; shared parameters stay scalar.
        ``Parameters`` (the scalar template) implicitly returns
        ``()`` for every name — that fallback is provided in
        :meth:`Fitter._initialize_unbounded` via ``getattr``, so
        scalar specs don't need this method.
        """
        return () if name in self._shared else (self._n_galaxies,)

    @property
    def psd_xi_init_shape(self) -> tuple[int, ...]:
        """Initial-shape for the stochastic-SFH latent field.

        Hierarchical fits have one ``psd_xi`` realization per galaxy
        — shape ``(N_galaxies, n_grid)``. Scalar fits stay
        ``(n_grid,)``.
        """
        n_grid = self._n_grid
        if n_grid is None:
            return ()
        return (self._n_galaxies, int(n_grid))

    def resolve_mirrors(self, params: dict) -> dict:
        """Delegate mirror resolution to the template.

        Mirror parameters (target ← source value) are declared at the
        template level. Under batched sampling, the template's
        ``resolve_mirrors`` is applied to the per-galaxy and shared
        param dict alike — the assignment broadcasts naturally.
        """
        return self._template.resolve_mirrors(params)

    # ── Batched sampling ───────────────────────────────────────────────

    def sample(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        """Draw one batched sample from the population.

        Per-galaxy free parameters get a leading ``(N_galaxies,)`` axis;
        shared parameters stay scalar; fixed values are scalar.

        Implementation: ``jax.vmap(template.sample)`` over a split key
        produces ``(N, ...)`` for every name. Then overwrite shared
        parameter values with a single (scalar) draw from the same
        underlying distribution so they tie across the population.
        """
        key_per_galaxy, key_shared = jax.random.split(key, 2)
        per_galaxy_keys = jax.random.split(key_per_galaxy, self._n_galaxies)
        # Batched: one draw per galaxy → (N, ...) for every param.
        batched = jax.vmap(self._template.sample)(per_galaxy_keys)
        # Shared overwrite: single draw, broadcast scalar back.
        single = self._template.sample(key_shared)
        for name in self._shared:
            if name in batched and name in single:
                batched[name] = single[name]
        return batched

    # ── Identity helpers used by various callers ───────────────────────

    def __repr__(self) -> str:
        return (
            f"PopulationSpecView(n_galaxies={self._n_galaxies}, "
            f"n_free={self.n_free}, shared={sorted(self._shared)})"
        )
