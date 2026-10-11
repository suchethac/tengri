# SPDX-License-Identifier: BSD-3-Clause
"""#2683 follow-up: the ``dsps`` age kernel's time-parameter gradients are the physical ones.

The histogram kernel was fed an SFR table sampled at fixed lookbacks. For a
step-like history (``psb_flex``) the position of a bin edge never enters a table
value, so ``d(flux)/d(tflex)`` and ``d(flux)/d(tlast)`` were exactly zero; for a
smooth history with a moving onset the gradient was the slope between table-row
crossings, 1-5 % off the converged one (a staircase at the row spacing). The
kernel now integrates the same dense, edge-resolved integrand the cloud-in-cell
kernel does, so a moving edge moves mass continuously.

Reference: central finite difference of the cic kernel on a 256x refined
integrand (the converged quadrature, no kernel assumptions).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.stellar import component as stellar_component

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "sdss_u", "sdss_r", "2mass_h"]
TRUTH_FACTOR = 256

#: psb_flex with every ratio non-zero, so the flexible-zone edges carry real steps
#: (all-zero ratios make the history flat and its tflex derivative physically ~0).
PSB = {
    "type": "psb_flex",
    "ratio_young": Fixed(0.6),
    "ratio_flex_0": Fixed(0.5),
    "ratio_flex_1": Fixed(-0.4),
    "ratio_flex_2": Fixed(0.3),
    "ratio_flex_3": Fixed(-0.5),
    "ratio_old_0": Fixed(0.3),
    "ratio_old_1": Fixed(-0.2),
    "ratio_old_2": Fixed(0.2),
}


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def obs():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _model(ssp, obs, kernel, z, sfh):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(z),
        sfh={**sfh, "all_params": Fixed(DEFAULT), "age_kernel": kernel},
    )


def _grad(model, key, x):
    return np.asarray(jax.jacfwd(lambda v: model.predict_photometry({key: v}))(jnp.asarray(x)))


def _truth_fd(monkeypatch, ssp, obs, z, sfh, key, x, h):
    """Central difference of the cic kernel on the 256x integrand."""
    jax.clear_caches()  # compiled kernels are cached per function, not per module constant
    try:
        with monkeypatch.context() as m:
            m.setattr(stellar_component, "INTEGRAND_FACTOR_PARAMETRIC", TRUTH_FACTOR)
            m.setattr(stellar_component, "INTEGRAND_FACTOR_SAWTOOTH", TRUTH_FACTOR, raising=False)
            model = _model(ssp, obs, "cic", z, sfh)
            f = lambda v: np.asarray(model.predict_photometry({key: jnp.asarray(v)}))  # noqa: E731
            return (f(x + h) - f(x - h)) / (2.0 * h)
    finally:
        jax.clear_caches()


@pytest.mark.parametrize(
    ("key", "sweep", "points", "h"),
    [
        ("sfh_psb_flex_tflex_gyr", Uniform(0.5, 5.0), (2.03, 2.3, 3.1), 2e-3),
        ("sfh_psb_flex_tlast_gyr", Uniform(0.01, 1.0), (0.213, 0.3), 2e-4),
    ],
)
def test_a_psb_flex_edge_gradient_is_nonzero_and_converged(
    monkeypatch, ssp, obs, key, sweep, points, h
):
    """tflex/tlast gradient: was identically 0; now within 0.5 % of the converged one."""
    short = key.removeprefix("sfh_psb_flex_")
    model = _model(ssp, obs, "dsps", 0.0, {**PSB, short: sweep})
    for x in points:
        g = _grad(model, key, x)
        truth = _truth_fd(monkeypatch, ssp, obs, 0.0, {**PSB, short: sweep}, key, x, h)
        assert np.all(np.isfinite(g)) and float(np.max(np.abs(g))) > 0.0
        rel = np.abs(g / truth - 1.0)
        assert np.all(rel <= 0.005), f"{key}={x}: |grad/truth-1| = {np.round(rel * 100, 2)} %"


@pytest.mark.parametrize(
    ("family", "key", "sfh", "z", "points", "bound"),
    [
        (
            "delayed",
            "sfh_delayed_age_gyr",
            {"type": "delayed", "tau_gyr": Fixed(1.0), "log_total_mass": Fixed(10.0)},
            0.0,
            (3.07, 5.0, 5.15, 5.3),
            0.021,
        ),
        ("dpl", "sfh_dpl_age_gyr", {"type": "dpl"}, 0.1, (5.0, 5.15, 5.3), 0.008),
    ],
)
def test_b_onset_gradient_matches_converged(
    monkeypatch, ssp, obs, family, key, sfh, z, points, bound
):
    """Onset gradient between SSP nodes: was 1-5 % off in every band; now within 2.1 %."""
    full = {**sfh, "age_gyr": Uniform(1.0, 10.0)}
    model = _model(ssp, obs, "dsps", z, full)
    worst = 0.0
    for x in points:
        g = _grad(model, key, x)
        truth = _truth_fd(monkeypatch, ssp, obs, z, full, key, x, 1e-4)
        assert np.all(np.isfinite(g)) and float(np.max(np.abs(g))) > 0.0
        worst = max(worst, float(np.max(np.abs(g / truth - 1.0))))
    assert worst <= bound, f"{family}: worst |grad/truth-1| = {worst * 100:.2f} %"


def test_c_onset_flux_has_no_row_staircase(ssp, obs):
    """Fine onset scan: the dsps/cic flux ratio varies smoothly (no step above 5e-5)."""
    full = {
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "log_total_mass": Fixed(10.0),
        "age_gyr": Uniform(1.0, 10.0),
    }
    models = {k: _model(ssp, obs, k, 0.0, full) for k in ("cic", "dsps")}
    xs = np.linspace(5.0, 5.04, 81)
    ratio = np.array(
        [
            float(
                models["dsps"].predict_photometry({"sfh_delayed_age_gyr": jnp.asarray(x)})[2]
                / models["cic"].predict_photometry({"sfh_delayed_age_gyr": jnp.asarray(x)})[2]
            )
            for x in xs
        ]
    )
    step = float(np.max(np.abs(np.diff(ratio))))
    assert step <= 5e-5, f"largest step of dsps/cic in 0.5 Myr onset increments: {step:.2e}"


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
@pytest.mark.parametrize("tau", [0.1, 0.2])
def test_d_periodic_tau_gradient_where_a_burst_end_meets_an_onset(ssp, obs, kernel, tau):
    """Exponential periodic bursts at ``tau = k * delta``: was 16-230 % off the slope.

    The integrand took every burst end (``onset - tau``) as a knot, though only a
    rectangular burst is discontinuous there. At ``tau = k * delta`` (here
    ``delta`` = 0.1 Gyr) that knot coincided with a later onset knot and the tau
    gradient was 16-230 % off the central difference at h = 1e-4 across the four
    bands. Reference: the
    central difference of the same model, at a tau where no knot is crossed
    within h.
    """
    full = {"type": "periodic", "tau_bursts_gyr": Uniform(0.05, 0.25)}
    model = _model(ssp, obs, kernel, 0.1, full)
    key = "sfh_periodic_tau_bursts_gyr"
    g = _grad(model, key, tau)
    h = 1e-4
    f = lambda v: np.asarray(model.predict_photometry({key: jnp.asarray(v)}))  # noqa: E731
    fd = (f(tau + h) - f(tau - h)) / (2.0 * h)
    assert np.all(np.isfinite(g)) and np.all(g != 0.0), f"tau gradient {g}"
    rel = np.abs(g / fd - 1.0)
    assert np.all(rel <= 1e-3), f"tau={tau}: |grad/FD-1| = {rel}"
