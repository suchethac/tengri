# SPDX-License-Identifier: BSD-3-Clause
"""Shared mock helpers for test suite.

This module provides mock classes used across multiple test files to avoid
duplication. Fixtures should be imported via conftest; mocks are meant to
be imported directly here for use in test class definitions.
"""

import jax.numpy as jnp


class _IdentityDist:
    """Mock prior distribution that passes values through unchanged."""

    bounds = (-jnp.inf, jnp.inf)

    def unstandardize(self, x):
        return x


class MockSpec:
    """Lightweight stand-in for Parameters tracking free parameters.

    Used in tests that need a spec object but don't require full Parameter
    functionality (e.g., Fitter contract tests, loss function tests).

    Matches the minimal contract of tengri.Parameters:
    - stochastic: bool flag (False by default, may be overridden)
    - all_params: list of all parameter names (empty for mock)
    - free_params: list of free parameter names (provided at init)
    - get_distribution(name): returns a prior Distribution
    - get_fixed_values(): returns dict of fixed parameter values
    - resolve_mirrors(params): identity transformation
    - sample(key): draws samples from the prior
    """

    def __init__(self, free_names: list, stochastic: bool = False):
        """Initialize with a list of free parameter names.

        Parameters
        ----------
        free_names : list
            Names of free parameters
        stochastic : bool, optional
            Whether the spec is stochastic (default False)
        """
        self._free_names = free_names
        self.all_params = []
        self.stochastic = stochastic

    @property
    def free_params(self):
        return self._free_names

    def get_distribution(self, name: str):
        """Return the prior distribution for a parameter."""
        return _IdentityDist()

    def get_fixed_values(self):
        """Return dict of fixed parameter values (empty for mock)."""
        return {}

    def resolve_mirrors(self, params: dict) -> dict:
        """Identity transformation: return params unchanged."""
        return params

    def sample(self, key):
        """Return default sample values for all free parameters.

        Matches the contract of Parameters.sample: takes a PRNG key and returns
        a dict mapping parameter names to sampled values. For the mock, we return
        a default value (1.0) for each free parameter.
        """
        return {name: jnp.asarray(1.0) for name in self._free_names}
