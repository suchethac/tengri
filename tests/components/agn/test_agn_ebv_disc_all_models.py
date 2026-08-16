# SPDX-License-Identifier: BSD-3-Clause
"""Parametrized coverage: ``agn_ebv_disc`` works on every disc+torus model.

The Prévot-SMC disc-reddening helper ``_redden_disc`` is wired into every
registered AGN model whose forward pass produces an explicit disc SED
that is later summed with a torus. These tests exercise each registered
model with ``agn_ebv_disc = 0`` (no-op) and ``agn_ebv_disc > 0``
(meaningful UV suppression), and confirm that gradients flow through
the new parameter for inference.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.bounds

_DATA_DIR = Path(__file__).resolve().parents[4] / "data"
_SILVA04 = _DATA_DIR / "silva04_torus_grid.h5"
_CAT3D = _DATA_DIR / "cat3d_wind_torus_grid.h5"
_SKIRTOR = any(
    (_DATA_DIR / p).is_file() for p in ("skirtor_templates_v2.h5", "skirtor_templates_v3.h5")
)


# Each row: (model-name, kwargs required beyond defaults, grid-availability guard)
_CASES = [
    (
        "silva04",
        {"agn_log_nh_silva": 23.0},
        _SILVA04.is_file(),
        "Silva+04 grid not built",
    ),
    (
        "cat3d_wind",
        {"agn_cos_inc": 0.5, "agn_a_cat3d": -2.0, "agn_fwd_cat3d": 0.45},
        _CAT3D.is_file(),
        "CAT3D-Wind grid not built",
    ),
    (
        "adaf",
        {"agn_log_mbh": 8.0, "agn_log_ledd": -3.0},
        True,
        "ADAF is analytic; should always be available",
    ),
    (
        "kubota_done_full",
        {"agn_log_mbh": 8.0, "agn_log_ledd": -1.0},
        True,
        "Kubota-Done 3-zone + two-temperature torus is analytic",
    ),
    (
        "skirtor",
        {},
        _SKIRTOR,
        "SKIRTOR template grid not available",
    ),
]


@pytest.fixture(scope="module")
def wavelength() -> jnp.ndarray:
    return jnp.geomspace(1e3, 1e6, 256)


@pytest.mark.parametrize("model_name,extra_kwargs,available,why", _CASES)
def test_ebv_zero_is_noop(model_name, extra_kwargs, available, why, wavelength):
    if not available:
        pytest.skip(why)
    from tengri.components.agn.unified import resolve_agn_model

    fn = resolve_agn_model(model_name)
    base = fn(wavelength, agn_log_lbol=12.0, agn_lum_ratio=0.1, **extra_kwargs)
    with_zero = fn(
        wavelength, agn_log_lbol=12.0, agn_lum_ratio=0.1, agn_ebv_disc=0.0, **extra_kwargs
    )
    # Finite-precision tolerance: the helper multiplies by 10**0 which is
    # numerically 1.0, so we expect bit-for-bit or near-bit-for-bit equality.
    assert jnp.allclose(base, with_zero, rtol=1e-12, atol=0.0), (
        f"{model_name}: ebv_disc=0.0 changed the SED"
    )


@pytest.mark.parametrize("model_name,extra_kwargs,available,why", _CASES)
def test_ebv_positive_suppresses_uv(model_name, extra_kwargs, available, why, wavelength):
    if not available:
        pytest.skip(why)
    from tengri.components.agn.unified import resolve_agn_model

    fn = resolve_agn_model(model_name)
    unreddened = fn(
        wavelength, agn_log_lbol=12.0, agn_lum_ratio=0.1, agn_ebv_disc=0.0, **extra_kwargs
    )
    reddened = fn(
        wavelength, agn_log_lbol=12.0, agn_lum_ratio=0.1, agn_ebv_disc=0.3, **extra_kwargs
    )
    uv_mask = wavelength < 3000.0
    u_sum = float(unreddened[uv_mask].sum())
    r_sum = float(reddened[uv_mask].sum())
    if u_sum <= 0.0:
        pytest.skip(f"{model_name}: no UV disc flux to redden")
    rel_drop = 1.0 - r_sum / u_sum
    assert rel_drop > 0.01, (
        f"{model_name}: UV flux dropped by only {rel_drop:.2%} under E(B-V)=0.3"
    )


@pytest.mark.parametrize("model_name,extra_kwargs,available,why", _CASES)
def test_grad_flows_through_ebv(model_name, extra_kwargs, available, why, wavelength):
    if not available:
        pytest.skip(why)
    from tengri.components.agn.unified import resolve_agn_model

    fn = resolve_agn_model(model_name)

    def loss(ebv: float) -> float:
        sed = fn(
            wavelength,
            agn_log_lbol=12.0,
            agn_lum_ratio=0.1,
            agn_ebv_disc=ebv,
            **extra_kwargs,
        )
        return jnp.log1p(jnp.sum(sed))

    g = assert_grad_matches_fd(loss, 0.1)
    assert jnp.isfinite(g), f"{model_name}: non-finite gradient through agn_ebv_disc"
    assert float(g) != 0.0, f"{model_name}: inert agn_ebv_disc gradient"
