# SPDX-License-Identifier: BSD-3-Clause
"""CompositeLikelihood: sum of per-channel Likelihood adapters.

The composition primitive for joint likelihoods. A user with photometry
+ spectroscopy + emission-line fluxes builds:

>>> from tengri.pipeline import (
...     PhotometryLikelihood,
...     SpectroscopyLikelihood,
...     CompositeLikelihood,
... )
>>> likelihood = CompositeLikelihood(
...     PhotometryLikelihood(phot_fnu, phot_err),
...     SpectroscopyLikelihood(spec_fnu, spec_err),
... )
>>> fitter = tengri.Fitter(model, data=..., noise=..., likelihood=likelihood)

The composite reads from each component's expected key in the
prediction dict — :class:`PhotometryLikelihood` reads ``"phot_fnu"``,
:class:`SpectroscopyLikelihood` reads ``"spec_fnu"`` — and sums their
log-probabilities. Adding a custom upper-limit or emission-line
likelihood to the joint set is one more constructor argument.

Why composition over a god-class
--------------------------------
A single ``JointLikelihood(phot_fnu, phot_err, spec_fnu, spec_err,
lines_flux, lines_err, ...)`` would grow a constructor parameter for
every channel a user might add. :class:`CompositeLikelihood` offloads
each channel to a single-purpose adapter and just sums the result.
This is the same pattern :data:`tengri.pipeline.run_components` uses
for SED forward-model components.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

__all__ = ["CompositeLikelihood"]


@dataclass(frozen=True)
class CompositeLikelihood:
    r"""Sum of log-probabilities from a list of per-channel Likelihoods.

    Parameters
    ----------
    *likelihoods : :class:`tengri.core.Likelihood`
        Concrete Likelihood objects. Order doesn't matter — sums are
        commutative — but is preserved for diagnostic ``name``
        construction.

    Notes
    -----
    **JIT-compatible**: yes if each constituent is JIT-compatible
    (every shipped adapter is).

    The composite owns no parameters of its own; the union of each
    constituent's :meth:`declared_parameters` is exposed via
    :meth:`declared_parameters`. Duplicate names across constituents
    raise at construction.
    """

    likelihoods: tuple[Any, ...] = field(default_factory=tuple)
    name: str = "composite"

    def __init__(self, *likelihoods: Any, name: str | None = None) -> None:
        # Variadic constructor for the natural call site
        # ``CompositeLikelihood(a, b, c)``. dataclass-generated init
        # would force a list/tuple — this matches the
        # :func:`run_components`-style ergonomics.
        object.__setattr__(self, "likelihoods", tuple(likelihoods))
        if name is None:
            constituent_names = "+".join(
                getattr(lk, "name", type(lk).__name__) for lk in likelihoods
            )
            name = f"composite[{constituent_names}]"
        object.__setattr__(self, "name", name)
        self._validate_no_duplicate_params()

    def _validate_no_duplicate_params(self) -> None:
        seen: dict[str, str] = {}
        for lk in self.likelihoods:
            decls = getattr(lk, "declared_parameters", lambda: [])()
            for decl in decls:
                pname = getattr(decl, "name", None) or decl[0]
                if pname in seen:
                    raise ValueError(
                        f"CompositeLikelihood: parameter {pname!r} declared "
                        f"by both {seen[pname]!r} and "
                        f"{getattr(lk, 'name', type(lk).__name__)!r}."
                    )
                seen[pname] = getattr(lk, "name", type(lk).__name__)

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        r"""Sum of log-probabilities across constituents.

        Each constituent reads only the prediction keys it needs —
        :class:`PhotometryLikelihood` ignores ``"spec_fnu"`` and vice
        versa.
        """
        total = jnp.asarray(0.0)
        for lk in self.likelihoods:
            total = total + lk.log_prob(prediction, params)
        return total

    def declared_parameters(self) -> list[Any]:
        """Union of constituents' declared nuisance parameters."""
        out: list[Any] = []
        for lk in self.likelihoods:
            decls = getattr(lk, "declared_parameters", lambda: [])()
            out.extend(decls)
        return out
