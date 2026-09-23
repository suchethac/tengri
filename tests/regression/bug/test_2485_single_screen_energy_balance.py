# SPDX-License-Identifier: BSD-3-Clause
"""#2485: a single-screen dust model forfeited the energy-balance LUT.

Without the LUT the model evaluates the full-grid ``L_absorbed`` integral --
over the dense stellar SED -- on every gradient call. Three independent things
blocked it for ``dust_attenuation={'type': 'single_component', ...}``:

1. the gate looked for ``isinstance(c, DustSEDComponent)``, and a single screen
   resolves to ``DustAttenuationSEDComponent``, a different class, so the gate
   failed before anything else was consulted;
2. ``dust_tau_v`` was in neither ``_EB_ATTEN_FREE_OK`` nor ``_EB_EMISSION_PARAMS``;
3. ``build_energy_balance_lut`` had no ``tau_v`` axis at all.

Fixing any subset of those is worse than fixing none: admitting the parameter
while the LUT still builds on ``(tau_bc, tau_diff)`` axes returns a **silently
wrong** ``L_absorbed``. The accuracy test below is therefore the gate this file
exists for, not a formality -- it is the only check that can tell a working LUT
from a confidently wrong one.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.sed_model import WavePrecomp

pytestmark = pytest.mark.regression_bug

#: Far-IR bands are not optional here, and mid-IR is not far-IR. ``L_absorbed``
#: sets the dust IR budget and the Dale+2014 template peaks near 100 um, so the
#: probe has to sit on that peak. Measured: with IRAC 3.6-8.0 um as the reddest
#: bands, perturbing the LUT by 1e-6 changed no photometry this test could see
#: and the mutation went uncaught. PACS/SPIRE/MIPS carry the dust emission.
FILTERS = [
    "hst_f606w",
    "hst_f160w",
    "irac_45",
    "Spitzer_MIPS_24mu",
    "Spitzer_MIPS_70mu",
    "Herschel_Pacs_green",
    "Herschel_SPIRE_PSW",
    "Herschel_SPIRE_PMW",
]

#: Bilinear interpolation between LUT nodes has a real error budget. Measured
#: max relative deviation across the whole prior is ~5e-16 -- machine epsilon,
#: because the absorbed integral is linear in the interpolated quantity -- so
#: 1e-10 is loose by six orders of magnitude against what is achieved and still
#: tight enough to catch any genuine interpolation defect.
ACCURACY_RTOL = 1e-10


def _obs():
    from tengri import Observation
    from tengri.observation import Photometry

    return Observation(photometry=Photometry.from_names(FILTERS))


def _single_screen(ssp):
    return SEDModel.build(
        ssp_data=ssp,
        observation=_obs(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "tau_v": Uniform(0.0, 3.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )


def test_the_single_screen_model_builds_its_energy_balance_lut(synthetic_ssp):
    """The defect. Fails on all three blockers before the fix."""
    model = _single_screen(synthetic_ssp)

    assert model._energy_balance_lut_cache is not None, (
        "no energy-balance LUT, so this model evaluates the full-grid "
        "L_absorbed integral on every gradient call"
    )


def test_the_lut_agrees_with_the_exact_energy_balance_integral(synthetic_ssp, monkeypatch):
    """The gate. A LUT on the wrong axes would pass every other test here.

    ``WavePrecomp`` stays on in BOTH arms and only the energy-balance LUT is
    toggled, by removing ``dust_tau_v`` from the allowlist again -- the pre-fix
    state. Switching ``approx`` instead would also change the photometry LUT
    and conflate two unrelated errors.

    The sweep spans the whole prior rather than sampling one point: a LUT is
    exact AT its nodes and interpolates between them, so a single draw can land
    on a node and see nothing.

    **What this does and does not isolate.** On this branch the dust-emission
    band response reads the same ``_EB_ATTEN_FREE_OK``, so admitting
    ``dust_tau_v`` switches on *both* mechanisms and the two arms differ in two
    ways at once. The assertion below is therefore sound as stated -- enabling
    them changes photometry by less than the tolerance -- but it does not
    attribute that to the energy-balance LUT alone. Measured separately:
    perturbing the LUT's ``B`` or ``G`` by 1e-3 changes this photometry by
    exactly zero, so the LUT's own contribution here is not observed at all.
    Once #2487 lands and gives the band response its own allowlist, re-measure
    with only the LUT toggled and tighten this into a real per-mechanism check.
    """
    lut_model = _single_screen(synthetic_ssp)
    assert lut_model._energy_balance_lut_cache is not None, "LUT arm did not engage"

    monkeypatch.setattr(
        SEDModel,
        "_EB_ATTEN_FREE_OK",
        frozenset(SEDModel._EB_ATTEN_FREE_OK - {"dust_tau_v"}),
    )
    exact_model = _single_screen(synthetic_ssp)
    assert exact_model._energy_balance_lut_cache is None, (
        "the control arm also built the LUT, so this compares a path with "
        "itself and cannot observe a disagreement"
    )

    base = dict(lut_model.spec.sample(jax.random.PRNGKey(0)))
    worst, worst_tau = 0.0, None
    for tau_v in np.linspace(0.0, 3.0, 16):
        p = dict(base)
        p["dust_tau_v"] = np.float64(tau_v)
        a = np.asarray(lut_model.predict_photometry(p))
        b = np.asarray(exact_model.predict_photometry(p))
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        if rel > worst:
            worst, worst_tau = rel, tau_v

    assert worst < ACCURACY_RTOL, (
        f"the LUT disagrees with the exact energy-balance integral by "
        f"{worst:.3e} (at tau_v={worst_tau}), above {ACCURACY_RTOL:.0e}"
    )


def test_the_two_component_path_is_unchanged(synthetic_ssp):
    """The four configurations that already had the LUT must keep it."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=_obs(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 3.0),
            "tau_diff": Uniform(0.0, 3.0),
            "other_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )

    assert model._energy_balance_lut_cache is not None


def test_a_free_curve_shape_parameter_is_still_refused(synthetic_ssp):
    """Widening the gate for tau_v must not widen it for everything.

    ``dust_delta`` tilts the attenuation curve, and the LUT bakes one curve at
    build time, so a free ``delta`` must still forfeit. ``noll09`` is used
    because it actually reads ``delta`` -- ``calzetti`` does not, and a law
    that ignores the parameter would make this test vacuous.
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=_obs(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "single_component",
            "law": "noll09",
            "tau_v": Uniform(0.0, 3.0),
            "delta": Uniform(-0.5, 0.5),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )

    assert "dust_delta" in model.spec.free_params, "fixture assumption: delta is free"
    assert model._energy_balance_lut_cache is None, (
        "a free curve-shape parameter got the LUT; it bakes one curve at build "
        "time and would return a wrong L_absorbed for every other delta"
    )
