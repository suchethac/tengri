# SPDX-License-Identifier: BSD-3-Clause
"""PopulationSpecView — batched-sample wrapper over a template :class:`Parameters`.

A thin view that the :class:`PopulationSEDModel` returns from its
``.spec`` property. It duck-types ``Parameters`` so the standard
:class:`Fitter` machinery doesn't need to know it's looking at a
hierarchical fit — the **only** observable difference vs the template
spec is that :meth:`sample` returns per-galaxy params with shape
``(N,)`` (or ``(N, n_grid)`` for stochastic-SFH ``psd_xi``).

See the implementation plan
``docs/superpowers/plans/2026-05-22-single-hamiltonian-fitter.md``
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

    - ``free_params``        — list of free-parameter names
    - ``all_params``         — list of all parameter names
    - ``fixed_params``       — list of fixed-parameter names
    - ``n_free``             — number of free parameters
    - ``stochastic``         — boolean flag
    - ``_distributions``     — dict of name → Distribution
    - ``get_distribution(name)``
    - ``get_fixed_values()``
    - ``sample(key)``        — **vmapped over N for per-galaxy params**

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
    def stochastic(self) -> bool:
        return self._template.stochastic

    @property
    def _distributions(self):
        """Underlying Distribution dict — Fitter sometimes reaches here."""
        return self._template._distributions

    @property
    def _n_grid(self):
        """Stochastic-SFH grid size on the template (if stochastic)."""
        return getattr(self._template, "_n_grid", None)

    def get_distribution(self, name: str):
        return self._template.get_distribution(name)

    def get_fixed_values(self) -> dict[str, float]:
        """Fixed values from the template (broadcast across galaxies)."""
        return self._template.get_fixed_values()

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
