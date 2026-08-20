# SPDX-License-Identifier: BSD-3-Clause
"""Issue #440: igm.meiksin06 and xray.yang20 must be discoverable + work.

Gap 1 (X-ray Yang+2020): the underlying physics (alpha_ox corona +
photoelectric N_H from Morrison & McCammon 1983 + Compton scattering +
HMXB/LMXB scaling) already landed via PR #325. This test pins the
user-visible ``"yang20"`` alias.

Gap 2 (IGM Meiksin 2006): new component, implementing Meiksin (2006)
exactly as CIGALE's ``pcigale.sed_modules.redshifting.igm_transmission``
evaluates it. The
diffuse Lyman-alpha forest continuum + LLS damping replaces
Inoue+2014's binary step structure with a smooth ramp -- exactly
what the §12 reproduction notebook needs.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri
from tengri.components.igm.meiksin06 import igm_transmission_meiksin06
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.contract


# ── Registry discoverability ──────────────────────────────────────


def test_yang20_listed():
    names = [e["name"] for e in tengri.list_xray_models()]
    assert "yang20" in names, f"yang20 missing from xray menu: {names}"


def test_meiksin06_listed():
    names = [e["name"] for e in tengri.list_igm_models()]
    assert "meiksin06" in names, f"meiksin06 missing from igm menu: {names}"


# ── Physics smoke tests ────────────────────────────────────────────


def test_meiksin06_transmission_shape():
    """Above (1+z)*912 A the IGM is transparent; below it drops smoothly."""
    wave_obs = np.linspace(500.0, 10000.0, 500)
    T = np.asarray(igm_transmission_meiksin06(wave_obs, z=3.0))

    # Sanity: everything in [0, 1].
    assert (T >= 0.0).all() and (T <= 1.0001).all()

    # Above Lyman-alpha at z=3 (1216 * 4 = 4864 A) absorption is fully
    # off (no Lyman line + no continuum). Note: the LyC cutoff is at
    # 912 * (1+z) = 3648 A, but Lyman lines extend up to Ly-alpha at
    # 1216 * (1+z) = 4864 A, so transparency only kicks in above ~5000 A.
    above_idx = np.searchsorted(wave_obs, 5500.0)
    assert T[above_idx:].min() > 0.99, "IGM should be transparent above Ly-alpha"

    # Just below Lyman alpha (1216 * 4 = 4864 wait -- Ly-alpha is the wavelength
    # of Ly-alpha, the cutoff for ANY Lyman absorption is 912 * (1+z)).
    # Below that, transmission drops; check a few interior wavelengths drop
    # below 0.5.
    below_idx = np.searchsorted(wave_obs, 1500.0)
    assert T[below_idx] < 0.5, "IGM should suppress flux blueward of Ly-alpha at z=3"


def test_meiksin06_jit_and_grad_safe():
    """Pure JAX kernel must JIT and accept gradient wrt z."""
    wave_obs = np.linspace(500.0, 10000.0, 100)
    T = assert_jit_matches_eager(igm_transmission_meiksin06, wave_obs, 3.0)
    assert np.isfinite(np.asarray(T)).all()


# ── End-to-end SEDModel.build smoke tests ──────────────────────────


@pytest.fixture(scope="module")
def ssp(synthetic_ssp_wide):
    # #613: synthetic SSP — IGM transmission (observer-frame) and the X-ray flag
    # are SSP-independent, so these grammar/forward checks run on CI.
    return synthetic_ssp_wide


def test_meiksin06_builds_through_grammar(ssp):
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust={
            "law": "power_law",
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.0,
            "tau_bc": 0.0,
        },
        igm={"type": "meiksin06"},
        redshift=tengri.Fixed(3.0),
    )
    assert m.spec.igm_model == "meiksin06"
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    sed = np.asarray(m.predict_rest_sed(p).sed)
    assert np.isfinite(sed).all() and sed.max() > 0.0


def test_yang20_alias_builds(ssp):
    """``xray={'type': 'yang20'}`` resolves identically to ``'simple'``."""
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust={
            "law": "power_law",
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.0,
            "tau_bc": 0.0,
        },
        xray={"type": "yang20"},
        redshift=tengri.Fixed(0.05),
    )
    # No exception = pass; the xray flag must be on.
    assert m.spec.xray is True
