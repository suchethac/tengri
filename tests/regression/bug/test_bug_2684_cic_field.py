# SPDX-License-Identifier: BSD-3-Clause
"""#2684: ``age_kernel='cic'`` integrates a correlated-field history.

A field draw defines the SFR at its own lookback nodes; the history between
nodes is the linear interpolation of the draw. The cloud-in-cell kernel takes
the nodes as exact knots (the rule it already applies to a tabulated history)
and the ``dsps`` kernel samples the same interpolant on its refined table, so
the two kernels integrate one function and agree to the histogram kernel's
accuracy. The formed mass equals the declaration on both.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar import component as stellar_component
from tengri.components.stellar.component import DSPSUnresolvedHistoryWarning

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "sdss_r"]
FIELD_SFHS = {
    "delayed_field": {"type": "delayed", "field": True},
    "dpl_field": {"type": ["dpl", "field"]},
    "dpl_burst_field": {"type": ["dpl", "burst"], "field": True},
}
#: Total-variation bounds of the age weights against a dense-quadrature truth
#: (the cic integrand at 256x resolution): (cic, dsps). The delayed field is
#: smooth; the dpl draws carry field nodes ~3x finer than the SSP node spacing,
#: which the histogram kernel's one-node-per-parcel assignment cannot resolve
#: (a documented limitation: up to 16 % FUV / 9 % r flux vs cic).
TV_BOUNDS = {
    "delayed_field": (0.003, 0.04),
    "dpl_field": (0.03, 0.10),
    "dpl_burst_field": (0.03, 0.10),
}
TRUTH_FACTOR = 256


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def obs():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _build(ssp, obs, kernel, spec, **extra):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={**spec, "all_params": Fixed(DEFAULT), "age_kernel": kernel, **extra},
    )


def _age_marginal(model, params):
    w = np.asarray(model.predict_state(params).derived["joint_weights"]).sum(0)
    return w / w.sum()


@pytest.mark.parametrize("name", sorted(FIELD_SFHS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_both_kernels_build_conserve_mass_and_track_the_truth(ssp, obs, name, seed, monkeypatch):
    spec = FIELD_SFHS[name]
    marg, logm, phot = {}, {}, {}
    for kernel in ("cic", "dsps"):
        m = _build(ssp, obs, kernel, spec)
        p = m.spec.sample(jax.random.PRNGKey(seed))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            marg[kernel] = _age_marginal(m, p)
            logm[kernel] = float(m.predict_state(p).derived["log_mstar_formed"])
        phot[kernel] = np.asarray(m.predict_photometry(p))
        if kernel == "dsps":
            warned = any(isinstance(r.message, DSPSUnresolvedHistoryWarning) for r in rec)
    declared = 10.0 ** logm["cic"]
    assert abs(10.0 ** logm["dsps"] / declared - 1.0) <= 1e-9
    monkeypatch.setattr(stellar_component, "INTEGRAND_FACTOR_TABULATED", TRUTH_FACTOR)
    truth = _age_marginal(_build(ssp, obs, "cic", spec), p)
    tv = {k: 0.5 * float(np.abs(w - truth).sum()) for k, w in marg.items()}
    b_cic, b_dsps = TV_BOUNDS[name]
    assert tv["cic"] <= b_cic, f"{name} seed {seed}: cic TV vs truth {tv['cic']:.4f} > {b_cic}"
    assert tv["dsps"] <= b_dsps, (
        f"{name} seed {seed}: dsps TV vs truth {tv['dsps']:.4f} > {b_dsps}"
    )
    err = np.abs(phot["dsps"] / phot["cic"] - 1.0)
    if name == "delayed_field":
        assert np.all(err <= 0.005), f"|dsps/cic-1| in {BANDS}: {np.round(err * 100, 3)} %"
    if err.max() > 0.02:
        assert warned, f"{name} seed {seed}: {err.max() * 100:.1f} % difference without a warning"


def test_cic_field_gradient_wrt_a_field_latent_is_finite_and_matches_fd(ssp, obs):
    m = _build(ssp, obs, "cic", {"type": "delayed", "field": True})
    p = m.spec.sample(jax.random.PRNGKey(0))
    key = next(k for k in sorted(p) if "field" in k and np.ndim(p[k]) >= 1)
    idx = int(np.argmax(np.abs(np.asarray(p[key]))))

    def flux(x):
        q = dict(p)
        q[key] = p[key].at[idx].set(x)
        return m.predict_photometry(q)[1]

    x0 = p[key][idx]
    g = float(jax.grad(flux)(x0))
    h = 1e-4
    fd = float((flux(x0 + h) - flux(x0 - h)) / (2 * h))
    assert np.isfinite(g) and g != 0.0, f"grad wrt {key}[{idx}] = {g}"
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1e-30, f"{key}[{idx}]: grad {g:.6e} vs FD {fd:.6e}"
