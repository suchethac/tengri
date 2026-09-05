# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #290 / #298: predict at redshift=0 returned Inf / NaN.

`luminosity_distance(z=0)` returned 0 → `1 / (4π D_L²)` → ±Inf → propagated
to every observer-frame flux through `lnu_to_fnu`. Fix returns 10 pc
(optical absolute-magnitude convention) at z=0, giving finite flux values
consistent with the L_ν / (4π·(10pc)²) limit.

See PR #306.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def test_predict_photometry_z0_returns_finite_fluxes():
    pytest.importorskip("tengri")
    import tengri
    from tengri.observation.photometry import FilterCurve

    waves = [jnp.linspace(3500.0, 4500.0, 50)] * 3
    trans = [jnp.ones(50) * 0.5] * 3
    curves = tuple(
        FilterCurve(wave=w, trans=t, name=f"b{i}") for i, (w, t) in enumerate(zip(waves, trans))
    )
    obs = tengri.Observation(photometry=tengri.Photometry(filters=curves))
    try:
        ssp = tengri.load_ssp()
    except FileNotFoundError:
        pytest.skip("default wNE SSP not available")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_diff": 0.0,
                "tau_bc": 0.0,
            },
            redshift=tengri.Fixed(0.0),
        )

    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    fluxes = model.predict_photometry(params)
    assert jnp.all(jnp.isfinite(fluxes)), f"expected finite, got {fluxes}"
    assert jnp.all(fluxes > 0), f"expected positive, got {fluxes}"


def test_luminosity_distance_z0_is_ten_pc():
    pytest.importorskip("tengri")
    from tengri.utils.cosmology import luminosity_distance
    from tengri.utils.physics_constants import TEN_PC_CM

    dl = float(luminosity_distance(0.0))
    assert abs(dl - TEN_PC_CM) / TEN_PC_CM < 1e-6, f"expected ~{TEN_PC_CM:.3e} cm, got {dl:.3e}"
