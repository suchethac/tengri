# SPDX-License-Identifier: BSD-3-Clause
"""Bin edges reach every binned member of a composite SFH (#3505).

``StellarSEDComponentConfig.sfh_bin_edges_gyr`` is bound onto each binned member of a
composite (list) SFH, in the exact forward, the SED-free fast path and the prior
sampler alike. Pinned as a value check against the members evaluated one by one with
the same edges, so an edge ladder that never arrives (the default ladder is used
instead) fails the comparison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.component import StellarSEDComponentConfig, evaluate_sfh
from tengri.components.stellar.sfh import sample_sfh_prior
from tengri.components.stellar.sfh.registry import SFH_REGISTRY

pytestmark = pytest.mark.regression_bug

Z_OBS = 1.0
N_GRID = 64
#: Seven bins, non-default: the default ladder ends at 13.7 Gyr and is log-spaced.
EDGES_GYR = np.array([0.0, 0.03, 0.15, 0.6, 1.5, 2.6, 3.6, 5.0])
#: Linear SSP-age grid [yr] that resolves every bin of EDGES_GYR.
AGES_YR = np.linspace(1.0e6, 5.0e9, 5000)


def _config(model, edges=EDGES_GYR):
    return StellarSEDComponentConfig(sfh_model=model, n_grid=N_GRID, sfh_bin_edges_gyr=edges)


def _sfr(model, edges=EDGES_GYR):
    out = evaluate_sfh(_config(model, edges), {"redshift": jnp.asarray(Z_OBS)}, AGES_YR, N_GRID)
    return np.asarray(out.sfr_on_ssp)


def test_composite_with_a_binned_member_builds_its_kwargs():
    """A list model no longer raises TypeError (unhashable list) when edges are set."""
    assert _config(["continuity", "burst"]).bin_edges_sfh_kwarg() == {}


def test_additive_composite_equals_the_sum_of_its_members_with_the_edges():
    """[continuity, delayed] with edges = continuity(edges) + delayed, member by member."""
    both = _sfr(["continuity", "delayed"])
    summed = _sfr("continuity") + _sfr("delayed", edges=None)
    np.testing.assert_allclose(both, summed, rtol=1e-12, atol=0.0)
    default_ladder = _sfr(["continuity", "delayed"], edges=None)
    assert np.max(np.abs(both - default_ladder)) > 1e-3 * np.max(np.abs(both)), (
        "the explicit edges changed nothing: the composite used the default ladder"
    )


def test_burst_composite_equals_the_mixture_of_continuity_with_the_edges():
    """[continuity, burst] = (1-f) smooth + f M B / int B, smooth built with the edges."""
    smooth = _sfr("continuity")
    burst_spec = SFH_REGISTRY["burst"]
    kw = {
        internal: np.asarray(p.default.default) * scale + offset
        for pub, (internal, scale, offset) in burst_spec.internal_param_map.items()
        if (p := burst_spec.params.get(pub)) is not None
    }
    f = 10.0 ** kw.pop("log_fburst")
    shape = np.asarray(burst_spec.fn(jnp.asarray(AGES_YR), **kw))
    m = np.trapezoid(smooth, AGES_YR)
    s = np.trapezoid(shape, AGES_YR)
    expected = (1.0 - f) * smooth + f * m * shape / s
    np.testing.assert_allclose(_sfr(["continuity", "burst"]), expected, rtol=1e-12, atol=0.0)


def test_prior_sampler_serves_a_composite_with_a_binned_member():
    """sample_sfh_prior(['continuity', 'burst']) was a TypeError on the branch."""
    _age, curves = sample_sfh_prior(
        ["continuity", "burst"], jax.random.PRNGKey(0), n=2, redshift=1.0
    )
    assert curves.shape == (2, 256)
    assert np.all(np.isfinite(np.asarray(curves)))
