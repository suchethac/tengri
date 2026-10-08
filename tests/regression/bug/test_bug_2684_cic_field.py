# SPDX-License-Identifier: BSD-3-Clause
"""#2684: ``age_kernel='cic'`` integrates a correlated-field history.

A field draw defines the SFR at its own lookback nodes; the history between
nodes is the linear interpolation of the draw. The cloud-in-cell kernel takes
the nodes as exact knots (the rule it already applies to a tabulated history);
``dsps`` names the same integration (#2683), so the two agree to round-off. The
formed mass equals the declaration on both.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar import component as stellar_component

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "sdss_r"]
FIELD_SFHS = {
    "delayed_field": {"type": "delayed", "field": True},
    "dpl_field": {"type": ["dpl", "field"]},
    "dpl_burst_field": {"type": ["dpl", "burst"], "field": True},
}
#: Total-variation bounds of the age weights against a dense-quadrature truth
#: (the cic integrand at 256x resolution). The delayed field is smooth; the dpl
#: draws carry field nodes ~3x finer than the SSP node spacing. Both kernel
#: names integrate the same function, so one bound serves both.
TV_BOUNDS = {
    "delayed_field": 0.003,
    "dpl_field": 0.03,
    "dpl_burst_field": 0.03,
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
        marg[kernel] = _age_marginal(m, p)
        logm[kernel] = float(m.predict_state(p).derived["log_mstar_formed"])
        phot[kernel] = np.asarray(m.predict_photometry(p))
    declared = 10.0 ** logm["cic"]
    assert abs(10.0 ** logm["dsps"] / declared - 1.0) <= 1e-9
    monkeypatch.setattr(stellar_component, "INTEGRAND_FACTOR_TABULATED", TRUTH_FACTOR)
    truth = _age_marginal(_build(ssp, obs, "cic", spec), p)
    tv = {k: 0.5 * float(np.abs(w - truth).sum()) for k, w in marg.items()}
    bound = TV_BOUNDS[name]
    for kernel in ("cic", "dsps"):
        assert tv[kernel] <= bound, (
            f"{name} seed {seed}: {kernel} TV vs truth {tv[kernel]:.4f} > {bound}"
        )
    err = np.abs(phot["dsps"] / phot["cic"] - 1.0)
    assert np.all(err <= 1e-9), f"|dsps/cic-1| in {BANDS}: {np.round(err * 100, 6)} %"


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
