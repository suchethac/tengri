# SPDX-License-Identifier: BSD-3-Clause
"""The photometric Green's function is vectorized over the age axis (#715).

The photometric branch used to loop over ``n_age`` in Python, unrolling one
interpolation and one integral per age into the trace. The branch must trace to
the same program whatever the age count, as the monochromatic branch already
does, and it must reproduce the per-age loop it replaces.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.analysis.diagnostics.green_functions import compute_green_function
from tengri.utils.filter_convention import FilterConvention, filter_weight as fw

pytestmark = pytest.mark.contract


def _ssp(n_age, n_wave=400, seed=0):
    key = jax.random.PRNGKey(seed)
    wave = jnp.linspace(1000.0, 20000.0, n_wave)
    flux = jnp.abs(jax.random.normal(key, (n_age, n_wave))) + 0.1
    return flux, wave


def _filter(n_filt=60):
    wave = jnp.linspace(4000.0, 6000.0, n_filt)
    trans = jnp.exp(-0.5 * ((wave - 5000.0) / 300.0) ** 2)
    return wave, trans


def _green_loop_reference(ssp_flux, ssp_wave, filt_wave, filt_trans):
    """The per-age loop the vectorized branch replaces, kept as the oracle."""
    weight = filt_trans * fw(filt_wave, FilterConvention.BESSELL)
    denom = jnp.trapezoid(weight, filt_wave)
    out = []
    for i in range(ssp_flux.shape[0]):
        on_filt = jnp.interp(filt_wave, ssp_wave, ssp_flux[i], left=0.0, right=0.0)
        out.append(jnp.trapezoid(on_filt * weight, filt_wave) / jnp.maximum(denom, 1e-30))
    return jnp.stack(out)


def test_matches_per_age_loop():
    flux, wave = _ssp(37)
    fw, ft = _filter()
    got = compute_green_function(flux, wave, filter_wave=fw, filter_trans=ft)
    want = _green_loop_reference(flux, wave, fw, ft)
    assert got.shape == (37,)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12)


def test_trace_size_independent_of_age_count():
    fw, ft = _filter()

    def n_eqns(n_age):
        flux, wave = _ssp(n_age)
        return len(
            jax.make_jaxpr(
                lambda f: compute_green_function(f, wave, filter_wave=fw, filter_trans=ft)
            )(flux).eqns
        )

    assert n_eqns(4) == n_eqns(40)
