# SPDX-License-Identifier: BSD-3-Clause
"""E-BFMI must be computed per chain, not over the flattened draw axis.

``run_nuts`` publishes the energy trace on the same flattened, burn-in-sliced
draw axis as ``samples`` and ``divergent_mask``, so the three can be joined
row-wise. That flattening is what makes the per-chain requirement easy to get
wrong: ``jnp.diff`` over the flat array silently includes one difference per
chain boundary, between two states that have nothing to do with each other.

Getting it wrong does not raise and does not produce an obviously silly number.
It produces a plausible one, in the direction that causes the most trouble: on
the fixture below, flattening takes E-BFMI from 3.2 to 0.33. The boundary adds
one term to the numerator but adds the between-chain spread to the denominator
once per draw, so the denominator wins and the answer is pushed DOWN, past the
~0.3 threshold at which E-BFMI is read as a warning. A flattened implementation
invents the funnel signature on healthy chains -- which, for the investigation
this diagnostic was added to serve, would have confirmed the hypothesis under
test. The first draft of this file asserted the opposite direction and the
fixture failed it; the direction here is measured, not reasoned.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.backends.mcmc.nuts import _ebfmi_per_chain


def _reference_ebfmi(trace):
    """E-BFMI of a single chain, written out longhand."""
    trace = np.asarray(trace, dtype=float)
    numerator = np.sum(np.diff(trace) ** 2)
    denominator = np.sum((trace - trace.mean()) ** 2)
    return numerator / denominator


class TestPerChain:
    def test_matches_a_longhand_per_chain_computation(self):
        chain_a = [10.0, 10.4, 9.7, 10.9, 10.1, 9.6]
        chain_b = [4.0, 4.9, 3.8, 4.2, 5.1, 4.4]
        got = _ebfmi_per_chain(jnp.asarray(chain_a + chain_b), n_chains=2, n_samples=6)
        assert got is not None
        assert float(got[0]) == pytest.approx(_reference_ebfmi(chain_a), rel=1e-9)
        assert float(got[1]) == pytest.approx(_reference_ebfmi(chain_b), rel=1e-9)

    def test_chain_boundary_is_not_a_difference(self):
        """The discriminating case: two chains at very different energies.

        Each chain wanders gently around its own level, so its honest E-BFMI is
        healthy. Flattening drags it below the warning threshold. A flattened
        implementation passes every other assertion in this file, which is the
        point of building this one.
        """
        low = [0.0, 0.3, -0.2, 0.1, -0.1, 0.2, -0.3, 0.15, 0.05, -0.15]
        high = [500.0, 500.3, 499.8, 500.1, 499.9, 500.2, 499.7, 500.15, 500.05, 499.85]
        flat = jnp.asarray(low + high)

        per_chain = _ebfmi_per_chain(flat, n_chains=2, n_samples=10)
        assert per_chain is not None
        honest = float(jnp.nanmin(per_chain))

        arr = np.asarray(flat, dtype=float)
        flattened_answer = np.sum(np.diff(arr) ** 2) / np.sum((arr - arr.mean()) ** 2)

        assert honest == pytest.approx(_reference_ebfmi(low), rel=1e-9)
        assert flattened_answer < honest, (
            "the fixture is not discriminating: with the chains separated in "
            "energy the flattened computation must report a SMALLER value than "
            "the per-chain truth, or this test cannot catch a flattened "
            "implementation"
        )
        # Far enough apart that no tolerance could confuse them, and across the
        # ~0.3 line where E-BFMI stops reading as healthy.
        assert flattened_answer < 0.3 < honest
        assert honest > 5.0 * flattened_answer

        # In the well-separated limit the flattened answer tends to 2/n_samples
        # and stops depending on the sampler at all: the numerator is dominated
        # by the single boundary term and the denominator by the between-chain
        # spread. Ten draws per chain therefore lands near 0.2 whatever the
        # chains did, which is the clearest statement of why flattening does not
        # merely bias the diagnostic but destroys it.
        assert flattened_answer == pytest.approx(2.0 / 10, rel=0.15)

    def test_single_chain_is_the_whole_trace(self):
        trace = [1.0, 1.5, 0.8, 1.2, 1.9]
        got = _ebfmi_per_chain(jnp.asarray(trace), n_chains=1, n_samples=5)
        assert got is not None
        assert float(got[0]) == pytest.approx(_reference_ebfmi(trace), rel=1e-9)


class TestRefusals:
    """It returns None rather than reshaping a layout it does not recognize."""

    def test_size_mismatch_refuses(self):
        assert _ebfmi_per_chain(jnp.arange(10.0), n_chains=3, n_samples=4) is None

    def test_too_few_draws_to_difference(self):
        assert _ebfmi_per_chain(jnp.asarray([1.0, 2.0]), n_chains=2, n_samples=1) is None

    def test_zero_chains_refuses(self):
        assert _ebfmi_per_chain(jnp.arange(6.0), n_chains=0, n_samples=6) is None

    def test_a_flat_chain_is_nan_not_an_exception(self):
        """Zero energy variance: undefined, not a division blow-up."""
        got = _ebfmi_per_chain(jnp.asarray([2.0, 2.0, 2.0, 2.0]), n_chains=1, n_samples=4)
        assert got is not None
        assert bool(jnp.isnan(got[0]))
