# SPDX-License-Identifier: BSD-3-Clause
"""Hand-built stellar publications for tests that drive a dust component directly.

``DustSEDComponent.apply`` / ``AgeBinnedDustComponent.apply`` read the stellar
component's ``age_boundary_younger_fraction`` (and ``log_L_lyc_age`` for the
nebular weights) from ``state.derived``. A test that builds the derived dict by
hand supplies them here: the hard-step younger-than-boundary fraction evaluated
on SSP node ages (every node wholly young or wholly old, so the expected values
are the closed-form per-population transmissions) and no ionizing light
(``log_L_lyc_age = -inf``), which puts the nebular screen on its young-limit
(birth-cloud) form.
"""

from __future__ import annotations

import jax.numpy as jnp


def step_younger_fraction(ssp_ages_yr, boundaries_yr=(1.0e7,)):
    """Hard-step younger-than-boundary rows, shape ``(n_boundary, n_age)``."""
    ages = jnp.asarray(ssp_ages_yr)
    return jnp.stack([(ages < b).astype(ages.dtype) for b in boundaries_yr])


def hand_age_derived(ssp_ages_yr, boundaries_yr=(1.0e7,)):
    """The two stellar keys a hand-built ``state.derived`` needs, as a dict."""
    ages = jnp.asarray(ssp_ages_yr)
    return {
        "age_boundary_younger_fraction": step_younger_fraction(ages, boundaries_yr),
        "log_L_lyc_age": jnp.full(ages.shape, -jnp.inf),
    }
