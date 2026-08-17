# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for dust attenuation law k(5500) normalization (#1731).

Every dust attenuation law must be normalized such that k(5500 Å) = 1.0
(or equivalently, the attenuation at 5500 Å is zero). This ensures
consistent calibration across all laws and prevents scaling ambiguity.

The module docstring in `components/dust/attenuation.py` documents
the k(5500)=1 contract at line ~248.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


def test_all_dust_laws_normalized_at_5500():
    """Every dust law in the registry must have k(5500) = 1 to within 1e-6."""
    import jax.numpy as jnp

    from tengri.components.dust.laws._registry import DUST_LAWS

    # Evaluate k(5500) for each law
    wave_5500 = jnp.asarray(5500.0)
    failures = []

    for law_name, law_func in DUST_LAWS.items():
        try:
            k_5500 = float(law_func(wave_5500))
        except ImportError:
            # Skip laws that require optional dependencies (e.g., dust-extinction)
            continue
        except Exception as e:
            pytest.fail(f"Law '{law_name}' raised {type(e).__name__}: {e} when evaluating k(5500)")

        deviation = abs(k_5500 - 1.0)
        if deviation >= 1e-6:
            failures.append((law_name, k_5500, deviation))

    if failures:
        msg = "The following dust laws violate k(5500)=1 contract:\n"
        for law_name, k_5500, deviation in failures:
            msg += f"  {law_name:20s}: k(5500) = {k_5500:.9f} (error: {deviation:.2e})\n"
        pytest.fail(msg)
