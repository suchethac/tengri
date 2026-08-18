# SPDX-License-Identifier: BSD-3-Clause
"""Canonical test doubles for the :mod:`tengri.inference.loss_functions` seam.

The loss builder reads a small, growing set of attributes off ``fitter.spec``.
Five test modules each carried their own private copy of the same stand-in --
byte-identical ``_IdentityDist`` and ``_MockSpec`` classes, differing only in a
docstring -- so every widening of that contract had to be applied five times, by
hand, with nothing to say which copies had been missed.

That is not hypothetical. The eager channel-scale pre-check (#1495, merged via
#1905) began calling ``spec.sample(key)`` at loss-build time. #1931 added
``sample`` to the two copies under ``tests/contract/``, because those are the
copies the PR gate runs. The three under ``tests/inference/`` were not touched
and could not be seen: that tree is auto-marked ``slow`` and deselected from the
default run, so the gap only surfaced in the nightly schedule, where it read as
six unrelated ``AttributeError`` failures rather than as one missed edit.

One copy is the fix. When the loss builder next requires something new of a
spec, it is one edit here, and no tier can be silently behind another.

These doubles deliberately implement *only* what the seam reads. Do not reach
for ``unittest.mock.MagicMock`` in its place: a bare ``MagicMock`` answers
``hasattr`` affirmatively for every name it is asked about, so a model built
from one claims every optional capability the forward path probes for --
emission-line catalogs included -- and silently routes the test through
channels it never meant to exercise. A double that cannot say "no" cannot
describe the object it stands in for.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["FakeSpec", "IdentityDist"]


class IdentityDist:
    """Unbounded distribution whose ``unstandardize`` is the identity.

    Lets a test reason in physical parameter values without a prior transform
    standing between what it sets and what the loss sees.
    """

    bounds = (-jnp.inf, jnp.inf)

    def unstandardize(self, x):
        return x


class FakeSpec:
    """Stand-in for :class:`tengri.Parameters` over a list of free names.

    Parameters
    ----------
    free_names : sequence of str
        Names reported by :attr:`free_params`, and the keys returned by
        :meth:`sample`.
    stochastic : bool, optional
        Value of the ``stochastic`` flag the loss builder reads to decide
        whether a PSD ``xi`` prior term is added. Default False.
    """

    def __init__(self, free_names, *, stochastic: bool = False):
        self._free_names = list(free_names)
        self.stochastic = stochastic
        # has_noise_model() reads spec.all_params -- an empty iterable makes
        # the legacy path's noise detection short-circuit.
        self.all_params: list[str] = []

    @property
    def free_params(self):
        return self._free_names

    def get_distribution(self, name):
        return IdentityDist()

    def get_fixed_values(self):
        return {}

    def resolve_mirrors(self, params):
        return params

    def sample(self, key):
        """Reference draw, matching ``Parameters.sample``'s key -> dict shape.

        Returns a fixed 1.0 per free parameter rather than anything drawn from
        ``key``. The one caller that matters is the eager channel-scale
        pre-check, which needs a representative point to evaluate each
        likelihood channel at, not a random one -- and a deterministic draw
        keeps a scale failure reproducible instead of seed-dependent.
        """
        return {name: jnp.asarray(1.0) for name in self._free_names}
