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

from tengri.utils.physics_constants import L_SUN
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
    bad = {k: u for k, u in outputs.items() if u not in ("dex", "Lsun", "erg/s/Hz")}
    assert not bad, f"{name} publishes non-float32-safe outputs {bad}"
    assert any(u == "dex" for u in outputs.values()), name


def test_every_log_luminosity_has_a_solar_luminosity_sibling(component):
    """``log_L_agn_<piece>`` is published beside ``L_agn_<piece>_lsun`` (NAMING_CONTRACT 4c).

    An AGN power of 1e8 to 1e14 Lsun is representable in float32, so the contract's first rule
    (luminosities in Lsun) applies and the value exists beside its exponent. The sibling equals
    ``10**(log - log10(L_sun))`` to 1e-12 in float64, is exactly 0 for an absent piece, and its
    name carries the unit because the bare ``L_agn_<piece>`` was the retired erg/s spelling.
    """
    name, comp = component
    declared = {key.name: key.units for key in comp.outputs()}
    logs = [k for k in declared if k.startswith("log_L_agn_")]
    assert logs, name
    point = {k: 0.5 * (lo + hi) for k, (lo, hi) in declared_box(comp).items()}
    pub, _ = predict(comp, point, float32=False)
    for log_key in logs:
        sibling = f"L_agn_{log_key[len('log_L_agn_') :]}_lsun"
        assert declared.get(sibling) == "Lsun", (name, sibling)
        expected = 0.0 if pub[log_key] == -np.inf else 10.0 ** (pub[log_key] - np.log10(L_SUN))
        assert pub[sibling] == pytest.approx(expected, rel=1e-12), (name, sibling)


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
            assert err < FLOAT32_RTOL, f"{name}: float32 SED rel {err:.2e} at {point}"


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
