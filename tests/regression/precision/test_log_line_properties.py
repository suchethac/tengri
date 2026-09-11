# SPDX-License-Identifier: BSD-3-Clause
r"""Emission-line luminosities need a float32-safe companion (#1534, #1206 item 3).

Line luminosities are ~1e40-1e42 erg/s, past float32's 3.4e38 ceiling, so eleven
public properties are ``inf`` in pure float32. #1206 item 3 proposes returning them
in ``L_sun``/``log10`` -- a **breaking** unit change. This ships the additive
alternative instead, following the ``log_q_h`` precedent that already shipped for the
same reason: a ``log_<name>`` companion beside each linear property. Nothing breaks,
no deprecation cycle, and the breaking change remains available later.

**The companion has to be computed upstream of the overflow or it is decorative.**
``log10_magnitude(inf)`` is ``+inf`` by contract (#1527), so wrapping the *linear*
accessor would faithfully report that it has nothing -- which is exactly how
``log_line_lums`` shipped broken the first time (#1534). These read
``derived["log_line_lums"]`` and never leave the log domain.

**The doublet sum is the difficulty.** ``extract_line_luminosity`` indexes *and sums*
-- [OII] is 3727+3730 -- and a sum is not a log-domain operation. Adding the logs, or
taking their max, would both be wrong. ``extract_log_line_luminosity`` uses a base-10
``logsumexp``, exact for it, and reduces to the stored value for a single-wavelength
target.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_PARAMS = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}

#: Doublets, whose companion must be a logsumexp rather than a passthrough.
_DOUBLETS = ("oii",)


def _model(ssp):
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_r", "wise_w1"])),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.0,
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        # The X-ray block is here so the pair sweep actually covers `l_x_*`. Without
        # it those properties are not published and every assertion below would pass
        # while saying nothing about them -- the same fixture gap that let
        # `log_line_lums` ship broken (#1534).
        xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=None,
    )


@pytest.fixture(scope="module")
def linear_and_log(ssp_bare):
    """(names, linear float64, log float64) for every erg/s line property."""
    with jax.enable_x64(True):
        model = _model(ssp_bare)
        names = [
            n
            for n in model.available_properties
            if f"log_{n}" in model.available_properties and not n.startswith("log_")
        ]
        lin = model.predict_properties(_PARAMS, names=tuple(names))
        log = model.predict_properties(_PARAMS, names=tuple(f"log_{n}" for n in names))
    return names, lin, log


def test_the_sweep_is_not_vacuous(linear_and_log):
    """Guard the guard: a sweep over an empty set passes without testing anything."""
    names, _, _ = linear_and_log
    assert names, (
        "no (X, log_X) property pairs are published by this model, so every sweep "
        "below is vacuously true -- fix the fixture, do not delete the sweep"
    )
    assert "halpha" in names, f"halpha lost its companion; pairs found: {names}"
    for doublet in _DOUBLETS:
        assert doublet in names, f"{doublet} (a doublet) is not covered: {names}"
    for xray in ("l_x_xrb", "l_x_total"):
        assert xray in names, (
            f"{xray} is not covered. Its linear form overflows float32 at ANY star "
            f"formation rate — the HMXB coefficient alone is 2.6e39. Pairs: {names}"
        )


def test_every_float32_unrepresentable_property_has_a_companion(ssp_bare):
    """The census: no published property may exceed float32's ceiling uncovered.

    Derived from the float64 magnitudes rather than a hand-written list, so a new
    erg/s property added later fails here instead of being remembered. Measuring
    "non-finite in float32" instead would be contaminated — Cue's pure-f32 forward
    is non-finite (#1719), so properties well inside float32 read NaN downstream
    for reasons that have nothing to do with units.
    """
    f32_max = float(np.finfo(np.float32).max)
    with jax.enable_x64(True):
        model = _model(ssp_bare)
        names = tuple(model.available_properties)
        values = model.predict_properties(_PARAMS, names=names)

    uncovered = [
        n
        for n in names
        if not n.startswith("log_")
        and np.isfinite(float(np.asarray(values[n]).ravel()[0]))
        and abs(float(np.asarray(values[n]).ravel()[0])) > f32_max
        and f"log_{n}" not in names
    ]
    assert not uncovered, (
        "these published properties exceed float32's ceiling with no log_ companion, "
        f"so a float32 user has no route to them: {uncovered}"
    )


def test_the_companion_carries_the_right_value_in_float64(linear_and_log):
    """``log_X == log10(X)`` wherever the linear form is finite and positive.

    The doublets are the load-bearing entries: a companion that returned the first
    matched component, or the sum of the logs, would pass every single-line check
    here and fail these two.
    """
    names, lin, log = linear_and_log
    for name in names:
        a = float(np.asarray(lin[name]))
        b = float(np.asarray(log[f"log_{name}"]))
        if not np.isfinite(a) or a <= 0.0:
            continue
        assert b == pytest.approx(np.log10(a), rel=1e-10), (
            f"log_{name} = {b} but log10({name}) = {np.log10(a)}; the companion is "
            "not the log of its linear sibling"
        )


def test_the_companion_is_finite_in_float32_where_the_linear_overflows(ssp_bare):
    """The point of the exercise: pure float32, linear ``inf``, companion finite.

    Asserting only that the companion is finite would pass for a constant. The
    float64 comparison above pins the value; this pins that it *survives*.
    """
    f32_max = float(np.finfo(np.float32).max)
    with jax.enable_x64(True):
        model64 = _model(ssp_bare)
        names = [
            n
            for n in model64.available_properties
            if f"log_{n}" in model64.available_properties and not n.startswith("log_")
        ]
        lin64 = model64.predict_properties(_PARAMS, names=tuple(names))

    overflowing = [n for n in names if abs(float(np.asarray(lin64[n]))) > f32_max]
    assert overflowing, (
        "no line property exceeds float32's ceiling on this fixture, so this test "
        "cannot observe the defect it exists for"
    )

    with jax.enable_x64(False):
        model32 = _model(ssp_bare)
        log32 = model32.predict_properties(_PARAMS, names=tuple(f"log_{n}" for n in overflowing))

    for name in overflowing:
        value = float(np.asarray(log32[f"log_{name}"]))
        assert np.isfinite(value), (
            f"log_{name} is {value} in pure float32 while {name} overflows "
            f"({float(np.asarray(lin64[name])):.4e} > {f32_max:.4e}). The companion is "
            "being computed after the overflow instead of from log_line_lums."
        )
        assert np.any(value != 0.0), (
            "`value` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )


def test_a_doublet_companion_is_a_logsumexp_not_a_passthrough():
    """Pin the arithmetic directly, since a fixture may not separate the two.

    For [OII] 3727 + 3730 with equal components, the sum is ``2L`` and the companion
    must be ``log10(L) + log10(2)``. A passthrough returns ``log10(L)``; adding the
    logs returns ``2 log10(L)``. All three differ.
    """
    from tengri.utils.sed_quantities import extract_log_line_luminosity

    waves = jnp.asarray([3727.12, 3730.12, 6564.61])
    log_lums = jnp.asarray([40.0, 40.0, 41.0])

    got = float(extract_log_line_luminosity(waves, log_lums, (3727.12, 3730.12)))
    assert got == pytest.approx(40.0 + np.log10(2.0), rel=1e-12), (
        f"doublet companion is {got}; expected {40.0 + np.log10(2.0)} "
        "(passthrough would give 40.0, summing the logs 80.0)"
    )

    single = float(extract_log_line_luminosity(waves, log_lums, (6564.61,)))
    assert single == pytest.approx(41.0, rel=1e-12), (
        f"a single-wavelength target must reduce to the stored value, got {single}"
    )
