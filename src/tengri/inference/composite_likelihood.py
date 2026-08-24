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
prediction dict, :class:`PhotometryLikelihood` reads ``"phot_fnu"``,
:class:`SpectroscopyLikelihood` reads ``"spec_fnu"``, and sums their
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

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field

import jax.numpy as jnp

__all__ = ["CompositeLikelihood", "_check_channel_scales"]


@dataclass(frozen=True)
class CompositeLikelihood:
    r"""Sum of log-probabilities from a list of per-channel Likelihoods.

    Parameters
    ----------
    *likelihoods: :class:`tengri.protocols.Likelihood`
        Concrete Likelihood objects. Order doesn't matter, sums are
        commutative, but is preserved for diagnostic ``name``
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

    likelihoods: tuple = field(default_factory=tuple)
    name: str = "composite"

    def __init__(self, *likelihoods, name: str | None = None) -> None:
        # Variadic constructor for the natural call site
        # ``CompositeLikelihood(a, b, c)``. dataclass-generated init
        # would force a list/tuple, this matches the
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
        """Ensure no parameter names are declared by multiple likelihoods.

        All likelihoods must implement the Likelihood Protocol, which
        requires :meth:`declared_parameters` to return a list of parameter
        name strings (or an empty list). We validate that no name appears
        more than once across all constituents.
        """
        seen: dict[str, str] = {}
        for lk in self.likelihoods:
            decls = lk.declared_parameters()
            for pname in decls:
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
        data_args: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        r"""Sum of log-probabilities across constituents.

        Each constituent reads only the prediction keys it needs,
        :class:`PhotometryLikelihood` ignores ``"spec_fnu"`` and vice
        versa. ``data_args`` is forwarded to constituents that accept it
        (the built-in adapter cohort) so a shared compiled loss reads the
        current Fitter's data; user-supplied two-argument likelihoods
        keep working unchanged. The signature check runs at trace time,
        not per evaluation.
        """
        total = jnp.asarray(0.0)
        for lk in self.likelihoods:
            if data_args is not None and "data_args" in inspect.signature(lk.log_prob).parameters:
                total = total + lk.log_prob(prediction, params, data_args=data_args)
            else:
                total = total + lk.log_prob(prediction, params)
        return total

    def declared_parameters(self) -> list[str]:
        """Union of constituents' declared parameter names.

        Returns
        -------
        list[str]
            Concatenation of each constituent's :meth:`declared_parameters`
            list. Empty if all constituents are parameter-free.
        """
        out: list[str] = []
        for lk in self.likelihoods:
            out.extend(lk.declared_parameters())
        return out


def _check_channel_scales(likelihoods, prediction, params, data_args):
    """Eager pre-check that likelihood channels are on representable scales (#1495).

    Evaluates each channel's ``log_prob`` once, eagerly, at a reference
    parameter point, against the SAME prediction dict the loss builds, and
    raises when a channel's log-probability is non-finite or outside the
    float32 window (±3.4e38). A channel whose observations are supplied in
    the wrong units produces a chi-squared tens of orders of magnitude too
    large; summed with healthy channels it absorbs them completely through
    floating-point rounding, and the fit silently optimizes the broken
    channel alone. This guard converts that silent failure into a loud one
    naming the channel, before any sampling happens.

    Parameters
    ----------
    likelihoods: tuple of Likelihood
        The constituent likelihood objects to check (a composite's members,
        or a one-element tuple).
    prediction: dict
        The prediction dict the loss path builds (``phot_fnu`` / ``spec_fnu``
        / feature channels) at the reference parameters.
    params: dict
        The reference physical parameters the prediction was evaluated at.
    data_args: dict
        The fitter's concrete data arguments (same object the traced loss
        receives at call time).

    Raises
    ------
    ValueError
        If any channel's log_prob is non-finite or has magnitude > 3.4e38.

    Notes
    -----
    Runs once at loss build time, outside JIT; the traced ``log_prob`` is
    untouched. Not meant to detect every scale problem, only pathological
    ones (a ~29-order units mismatch), and it deliberately has NO fallback
    path: if a channel cannot be evaluated here, the same call fails inside
    the fit, so the error propagates instead of being swallowed.
    """
    if not likelihoods:
        return

    # float32 representable window: values beyond this are inf under pure f32.
    max_f32_magnitude = 3.4e38

    for i, lk in enumerate(likelihoods):
        lk_name = getattr(lk, "name", type(lk).__name__)

        # Evaluate the channel exactly the way the loss does (no try/except,
        # a failure here is a failure the fit would hit anyway; keep it loud).
        if "data_args" in inspect.signature(lk.log_prob).parameters:
            log_prob_val = float(lk.log_prob(prediction, params, data_args=data_args))
        else:
            log_prob_val = float(lk.log_prob(prediction, params))

        if jnp.isfinite(log_prob_val) and abs(log_prob_val) <= max_f32_magnitude:
            continue

        # Scale hints for the error message: the largest magnitudes on each
        # side of the channel's comparison usually expose the units mismatch.
        def _scale(x):
            try:
                arr = jnp.asarray(x)
                return float(jnp.max(jnp.abs(arr))) if arr.size else 0.0
            except (TypeError, ValueError):
                return float("nan")

        pred_scale = max((_scale(v) for v in prediction.values()), default=float("nan"))
        obs_scale = _scale(data_args.get("data"))
        raise ValueError(
            f"Likelihood channel {i} ({lk_name!r}) evaluated at the reference "
            f"parameters gives log_prob = {log_prob_val:.3e}, which is "
            + (
                "non-finite"
                if not jnp.isfinite(log_prob_val)
                else f"outside the float32 window (±{max_f32_magnitude:.1e})"
            )
            + f". Max |prediction| = {pred_scale:.3e}, max |data| = {obs_scale:.3e}. "
            "A chi-squared this large silently absorbs every other channel "
            "through floating-point rounding, check this channel's units "
            "against what the model predicts for it (#1495)."
        )
