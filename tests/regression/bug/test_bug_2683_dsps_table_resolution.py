# SPDX-License-Identifier: BSD-3-Clause
"""#2683: the ``dsps`` age kernel agrees with the dense ``cic`` kernel at the SFH onset.

The histogram kernel interpolates ``log10 M(<t)`` in ``log10 t`` and reads it at
log-midpoint bin edges. Fed one SFR row per SSP node, the edge inside the table
segment that holds the SFH onset reads ~zero mass and the node containing the
onset loses its whole weight (+2.3 % FUV on the delayed-tau fiducial, a flux that
jumps as the onset crosses a node, a staircase gradient). The kernel now
integrates the first-order dense integrand (see test_bug_2683_followup_*), which
is within 0.01 % of a converged quadrature for smooth histories; these tests pin
the onset behaviour on the delayed-tau fiducial.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "sdss_u", "sdss_r", "2mass_h"]
R_BAND = BANDS.index("sdss_r")
ONSET_KEY = "sfh_delayed_age_gyr"


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def obs():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _build(ssp, obs, kernel, **sfh):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={**sfh, "all_params": Fixed(DEFAULT), "age_kernel": kernel},
    )


def _delayed(onset):
    return {
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(onset),
        "log_total_mass": Fixed(10.0),
    }


def _age_weights_and_photometry(model):
    w = np.asarray(model.predict_state({}).derived["joint_weights"]).sum(0)
    return w / w.sum(), np.asarray(model.predict_photometry({}))


def _node_index(ssp, age_gyr):
    return int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(age_gyr))))


def test_a_onset_node_keeps_its_weight(ssp, obs):
    """Onset 5.0 Gyr, z=0: 5.012 Gyr node weight; cic 0.0383, dsps 0.0000 before and -18 % now."""
    j = _node_index(ssp, 5.012)
    w_cic, p_cic = _age_weights_and_photometry(_build(ssp, obs, "cic", **_delayed(5.0)))
    w_dsps, p_dsps = _age_weights_and_photometry(_build(ssp, obs, "dsps", **_delayed(5.0)))
    rel = abs(w_dsps[j] - w_cic[j]) / w_cic[j]
    assert rel <= 0.25, f"node {j}: cic {w_cic[j]:.4f} dsps {w_dsps[j]:.4f} (rel {rel:.3f})"
    err = np.abs(p_dsps / p_cic - 1)
    assert np.all(err <= 0.004), f"|dsps/cic-1| in {BANDS}: {np.round(err * 100, 3)} %"


def test_b_flux_does_not_swing_as_onset_crosses_a_node(ssp, obs):
    """Onset 5.0 -> 5.3 Gyr (7 steps): the bias was +2.3 % at 5.0 and -1.4 % at 5.3."""
    worst = 0.0
    for onset in np.linspace(5.0, 5.3, 7):
        p_cic = np.asarray(
            _build(ssp, obs, "cic", **_delayed(float(onset))).predict_photometry({})
        )
        p_dsps = np.asarray(
            _build(ssp, obs, "dsps", **_delayed(float(onset))).predict_photometry({})
        )
        worst = max(worst, float(np.max(np.abs(p_dsps / p_cic - 1))))
    assert worst <= 0.004, f"max |dsps/cic-1| over the scan and bands: {worst * 100:.3f} %"


@pytest.mark.parametrize(
    ("family", "tol"),
    [
        ("dpl", 0.004),
        ("lnorm", 0.004),
        ("exp", 0.004),
        ("continuity", 0.012),
        ("psb_flex", 0.012),
    ],
)
def test_c_default_families_agree(ssp, obs, family, tol):
    """Smooth defaults 0.07-0.19 % (was 0.6-2.3 %); step-like defaults 0.7 % (was 4.6 %)."""
    p_cic = np.asarray(_build(ssp, obs, "cic", type=family).predict_photometry({}))
    p_dsps = np.asarray(_build(ssp, obs, "dsps", type=family).predict_photometry({}))
    err = np.abs(p_dsps / p_cic - 1)
    assert np.all(err <= tol), f"{family}: |dsps/cic-1| in {BANDS}: {np.round(err * 100, 3)} %"


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
def test_d_formed_mass_is_the_declared_mass(ssp, obs, kernel):
    m = _build(ssp, obs, kernel, **_delayed(5.0))
    logm = float(m.predict_state({}).derived["log_mstar_formed"])
    assert abs(10.0**logm / 1e10 - 1.0) <= 1e-9, (
        f"{kernel}: formed mass / declared - 1 = {10.0**logm / 1e10 - 1.0:.3e}"
    )


def _flux_and_grad(ssp, obs, kernel):
    sfh = {**_delayed(5.0), "age_gyr": Uniform(1.0, 10.0)}
    m = _build(ssp, obs, kernel, **sfh)

    def r_flux(x):
        return m.predict_photometry({ONSET_KEY: x})[R_BAND]

    return jax.jit(r_flux), jax.jit(jax.grad(r_flux))


@pytest.mark.parametrize("onset", [5.0, 5.15])
def test_e_onset_gradient_matches_cic(ssp, obs, onset):
    """d(r-band flux)/d(onset): dsps was 21-48 % off cic (a staircase); now within 8 %.

    Fails with a node-only table (refine = 1): gradient 34 % and 25 % off at 5.0 and 5.15 Gyr.
    """
    g = {}
    for kernel in ("cic", "dsps"):
        _, grad = _flux_and_grad(ssp, obs, kernel)
        g[kernel] = float(grad(jnp.asarray(onset)))
    assert np.isfinite(g["cic"]) and np.isfinite(g["dsps"]) and g["cic"] != 0.0
    rel = abs(g["dsps"] - g["cic"]) / abs(g["cic"])
    assert rel <= 0.08, (
        f"onset {onset}: grad cic {g['cic']:.4e} dsps {g['dsps']:.4e} (rel {rel:.3f})"
    )


def test_f_onset_scan_has_no_staircase(ssp, obs):
    """61 onsets in 5.0-5.3 Gyr: the dsps/cic flux ratio has no step above 0.2 %.

    The physical flux is smooth in the onset and cic follows it, so the dsps/cic
    ratio must vary smoothly with it. (The node-only table biased the ratio by
    -1.4 to +2.3 % across this window without a jump larger than this threshold,
    so the onset-gradient test is the one that separates the two tables.)
    """
    flux = {k: _flux_and_grad(ssp, obs, k)[0] for k in ("cic", "dsps")}
    onsets = np.linspace(5.0, 5.3, 61)
    ratio = np.array(
        [float(flux["dsps"](jnp.asarray(x)) / flux["cic"](jnp.asarray(x))) for x in onsets]
    )
    step = np.abs(np.diff(ratio))
    assert step.max() <= 0.002, (
        f"largest step of dsps/cic {step.max() * 100:.3f} % at onset "
        f"{onsets[int(step.argmax())]:.3f} Gyr"
    )
