# SPDX-License-Identifier: BSD-3-Clause
"""Every AGN published luminosity is finite and accurate in pure float32 (#2842).

The published luminosity diagnostics of the standalone AGN components were ``erg/s`` values of
1e43 to 1e47, which are ``inf`` or NaN in float32 (max 3.4e38) before any physics runs
(NAMING_CONTRACT section 4c). They are published as ``log10`` [dex re erg/s], formed in log
space, and the specific luminosities (``erg/s/Hz``, a spectral density the contract reserves
that unit for) stay as they were.

The invariant, per component and per published key, over the declared parameter box (its
center, both edges of every parameter and random draws): the float32 value is finite wherever
the float64 value is (an absent component is ``-inf`` in both), and agrees with it to
``FLOAT32_RTOL`` in the luminosity. ``1e-4`` is the pointwise float32 tolerance the repository
holds the ADAF shape to (``tests/physics/agn/test_adaf_float32_shape.py``); the measured worst
error here is about 1.5e-5.

Float32 is selected with ``jax.enable_x64(False)`` inside the test; ``JAX_ENABLE_X64=0`` is
never set for pytest, which would turn the float64 reference arms into float32 runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.physics._agn_published import (
    SED_MODEL_COMPONENTS,
    box_points,
    build_component,
    declared_box,
    predict,
    relative_error,
)

pytestmark = pytest.mark.regression_bug

FLOAT32_RTOL = 1.0e-4

#: The emitted SED is held to the same tolerance, except the Kubota and Done disc: its three-zone
#: radius solve leaves an 8e-4 float32 error in the SED at some box points (unchanged by the
#: change this file pins; the published integrals of that SED agree to 1e-5).
_SED_RTOL = {"kd18": 2.0e-3}

#: Draws per component beyond the center and the box edges (SKIRTOR is the slow one).
_RANDOM_DRAWS = {"skirtor": 3}
_DEFAULT_DRAWS = 6


@pytest.fixture(scope="module", params=sorted(SED_MODEL_COMPONENTS))
def component(request):
    return request.param, build_component(request.param)


def test_declared_outputs_hold_no_integrated_erg_per_s_luminosity(component):
    """No published output is an integrated luminosity in ``erg/s``; luminosities are ``dex``.

    NAMING_CONTRACT section 4c: ``erg/s/Hz`` is reserved for spectral density, and an
    integrated luminosity of ~1e44 erg/s is ``inf`` in float32.
    """
    name, comp = component
    outputs = {key.name: key.units for key in comp.outputs()}
    bad = {k: u for k, u in outputs.items() if u not in ("dex", "erg/s/Hz")}
    assert not bad, f"{name} publishes non-float32-safe outputs {bad}"
    assert any(u == "dex" for u in outputs.values()), name


def test_published_keys_are_finite_and_accurate_in_float32(component):
    """Float32 published values equal float64 over the declared box (FLOAT32_RTOL)."""
    name, comp = component
    box = declared_box(comp)
    points = box_points(box, _RANDOM_DRAWS.get(name, _DEFAULT_DRAWS), seed=2842)
    worst = {}
    for point in points:
        pub64, _ = predict(comp, point, float32=False)
        pub32, _ = predict(comp, point, float32=True)
        assert set(pub32) == set(pub64) == {key.name for key in comp.outputs()}
        for key, v64 in pub64.items():
            err = relative_error(key, pub32[key], v64)
            worst[key] = max(worst.get(key, 0.0), err)
            assert err < FLOAT32_RTOL, (
                f"{name}.{key}: float32 {pub32[key]!r} vs float64 {v64!r} (rel {err:.2e}) "
                f"at {point}"
            )
    assert set(worst) == {key.name for key in comp.outputs()}


def test_emitted_sed_is_finite_and_accurate_in_float32(component):
    """The emitted SED is finite in float32 and follows float64 where it is not negligible."""
    name, comp = component
    box = declared_box(comp)
    for point in box_points(box, 2, seed=7):
        _, sed64 = predict(comp, point, float32=False)
        _, sed32 = predict(comp, point, float32=True)
        assert np.all(np.isfinite(sed32)), f"{name}: non-finite float32 SED at {point}"
        significant = (
            sed64 > 1.0e-3 * sed64.max() if sed64.max() > 0 else np.zeros_like(sed64, bool)
        )
        if significant.any():
            err = np.max(np.abs(sed32[significant] / sed64[significant] - 1.0))
            assert err < _SED_RTOL.get(name, FLOAT32_RTOL), (
                f"{name}: float32 SED rel {err:.2e} at {point}"
            )


def test_exactly_zero_power_publishes_minus_inf_in_both_precisions():
    """A component at zero luminosity fraction publishes ``-inf``, never NaN or a floor."""
    cases = {"kd18": "lum_ratio", "powerlaw": "lum_ratio", "skirtor_agnfitter": "torus_frac"}
    for name, key in cases.items():
        comp = build_component(name)
        point = {**{k: 0.5 * (lo + hi) for k, (lo, hi) in declared_box(comp).items()}, key: 0.0}
        for float32 in (False, True):
            pub, sed = predict(comp, point, float32=float32)
            for out_key, value in pub.items():
                if out_key.startswith("log_"):
                    assert value == -np.inf, (name, float32, out_key, value)
            assert np.all(sed == 0.0), (name, float32)


def test_skirtor_zero_torus_fraction_is_finite_in_float32():
    """SKIRTOR at ``torus_frac = 0`` (the box edge): no NaN SED, every log key ``-inf``."""
    comp = build_component("skirtor")
    point = {k: 0.5 * (lo + hi) for k, (lo, hi) in declared_box(comp).items()}
    point["torus_frac"] = 0.0
    for float32 in (False, True):
        pub, sed = predict(comp, point, float32=float32)
        assert np.all(sed == 0.0), float32
        assert all(v == -np.inf for k, v in pub.items() if k.startswith("log_")), (float32, pub)
        assert all(v == 0.0 for k, v in pub.items() if not k.startswith("log_")), (float32, pub)
