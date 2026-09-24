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

#: **The fixture matters.** ``synthetic_ssp`` spans 3000-10000 AA, so it never
#: reaches the Lyman limit and cannot observe the Lyman-continuum mask these
#: tests exist to pin -- on it the node agreement is 1.8e-2, which is the grid
#: being too coarse rather than the mapping being wrong. ``synthetic_ssp_wide``
#: spans 100 AA to 1 mm on 1600 points and does cross 912 AA. Measured on the
#: real ``fsps_prsc_c3k_a_chabrier`` grid, all 24 nodes agree to 5.9e-09.

#: **At a LUT node** the LUT reproduces the exact spectral integral by
#: construction, so agreement there is the real test of the mapping. Measured
#: on this configuration: 0 at tau_v = 0 and 7e-10 to 1.3e-09 at the next five
#: nodes. 1e-8 is loose against that and still tight enough that any systematic
#: error in the mapping fails it -- the pre-fix code, which baked a different
#: Lyman-continuum cutoff than the runtime used, was wrong by 2e-2 to 1e-1.
NODE_RTOL = 1e-8

#: **Between nodes** the LUT interpolates, and that error is real and much
#: larger. Measured at the first five midpoints of the 24-node linear grid over
#: [0, 3]: 6.4e-2, 1.7e-2, 8.4e-3, 4.9e-3, 3.1e-3 -- worst at small tau_v,
#: where the absorbed integral is most strongly curved in tau.
#:
#: **This is not specific to the single screen.** The shipped two-component
#: path, on the same grid, measures 6.0e-2, 1.4e-2 and 3.4e-3 at its first
#: three midpoints. So this bound documents a pre-existing property of the
#: energy-balance LUT rather than anything this change introduced, and it is
#: tracked separately. Asserting it here keeps it from growing unnoticed.
INTERP_RTOL = 0.10


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


def test_the_single_screen_model_builds_its_energy_balance_lut(synthetic_ssp_wide):
    """The defect. Fails on all three blockers before the fix."""
    model = _single_screen(synthetic_ssp_wide)

    assert model._energy_balance_lut_cache is not None, (
        "no energy-balance LUT, so this model evaluates the full-grid "
        "L_absorbed integral on every gradient call"
    )


def test_the_lut_agrees_with_the_exact_integral_at_its_nodes(synthetic_ssp_wide, monkeypatch):
    """The gate. A LUT on the wrong axes, or with a different Lyman-continuum
    cutoff baked in than the runtime uses, fails here and passes everything else.

    Sampled AT the grid nodes, because that is where the LUT is exact by
    construction: away from them any disagreement is confounded with ordinary
    interpolation error, which the next test bounds separately. Checking only a
    sweep of arbitrary tau_v conflates the two and cannot tell a broken mapping
    from a coarse grid -- the pre-fix code failed at 2.16e-2 and the true cause
    was invisible until the node/midpoint split was made.

    ``WavePrecomp`` stays on in BOTH arms and only the energy-balance LUT is
    toggled, by removing ``dust_tau_v`` from the allowlist -- the pre-fix state.
    Switching ``approx`` instead would also move the photometry LUT.
    """
    lut_model = _single_screen(synthetic_ssp_wide)
    assert lut_model._energy_balance_lut_cache is not None, "LUT arm did not engage"
    nodes = np.asarray(lut_model._energy_balance_lut_cache.tau_diff_grid)
    assert nodes.shape[0] > 1, "fixture assumption: tau_v is free, so the grid has nodes"

    monkeypatch.setattr(
        SEDModel,
        "_EB_ATTEN_FREE_OK",
        frozenset(SEDModel._EB_ATTEN_FREE_OK - {"dust_tau_v"}),
    )
    exact_model = _single_screen(synthetic_ssp_wide)
    assert exact_model._energy_balance_lut_cache is None, (
        "the control arm also built the LUT, so this compares a path with itself"
    )

    base = dict(lut_model.spec.sample(jax.random.PRNGKey(0)))
    worst, worst_tau = 0.0, None
    for tau_v in nodes:
        p = dict(base)
        p["dust_tau_v"] = np.float64(tau_v)
        a = np.asarray(lut_model.predict_photometry(p))
        b = np.asarray(exact_model.predict_photometry(p))
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        if rel > worst:
            worst, worst_tau = rel, float(tau_v)

    assert worst < NODE_RTOL, (
        f"at a LUT node the interpolation is exact, so this is the mapping "
        f"itself disagreeing: {worst:.3e} at tau_v={worst_tau}, above {NODE_RTOL:.0e}"
    )


def test_the_interpolation_error_between_nodes_stays_bounded(synthetic_ssp_wide, monkeypatch):
    """Bounds a pre-existing LUT property so it cannot grow unnoticed.

    Between nodes the LUT interpolates and the error is real -- up to ~6% at
    small tau_v on the 24-node linear grid. The shipped two-component path has
    the same behavior on the same grid, so this is not a defect of the single
    screen; it is the resolution of the grid both share.
    """
    lut_model = _single_screen(synthetic_ssp_wide)
    nodes = np.asarray(lut_model._energy_balance_lut_cache.tau_diff_grid)

    monkeypatch.setattr(
        SEDModel,
        "_EB_ATTEN_FREE_OK",
        frozenset(SEDModel._EB_ATTEN_FREE_OK - {"dust_tau_v"}),
    )
    exact_model = _single_screen(synthetic_ssp_wide)

    base = dict(lut_model.spec.sample(jax.random.PRNGKey(0)))
    worst, worst_tau = 0.0, None
    for i in range(min(8, nodes.shape[0] - 1)):
        tau_v = 0.5 * (nodes[i] + nodes[i + 1])
        p = dict(base)
        p["dust_tau_v"] = np.float64(tau_v)
        a = np.asarray(lut_model.predict_photometry(p))
        b = np.asarray(exact_model.predict_photometry(p))
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        if rel > worst:
            worst, worst_tau = rel, float(tau_v)

    assert worst < INTERP_RTOL, (
        f"interpolation error grew to {worst:.3e} at tau_v={worst_tau}, above "
        f"the {INTERP_RTOL:.2f} bound measured when this was written"
    )


def test_the_two_component_path_is_unchanged(synthetic_ssp_wide):
    """The four configurations that already had the LUT must keep it."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
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


def test_a_free_curve_shape_parameter_is_still_refused(synthetic_ssp_wide):
    """Widening the gate for tau_v must not widen it for everything.

    ``dust_delta`` tilts the attenuation curve, and the LUT bakes one curve at
    build time, so a free ``delta`` must still forfeit. ``noll09`` is used
    because it actually reads ``delta`` -- ``calzetti`` does not, and a law
    that ignores the parameter would make this test vacuous.
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
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


# ---------------------------------------------------------------------------
# An active nebular backend, and free parameters that interact with the LUT.
#
# The single screen attenuates the nebular continuum with the SAME curve as the
# stellar light, so the absorbed budget is the sum of both. The fast path took
# only the LUT's stellar term for a while, which silently dropped whatever the
# backend emitted: 7.7e-2 relative photometry error AT a LUT node, on the shape
# configurations II and V of the paper grid use.
# ---------------------------------------------------------------------------

CUE_NEB = {"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Fixed(-2.5)}

#: Module-level so it is not a call in a default argument (ruff B008).
Z_FIXED = Fixed(0.1)


def _with_neb(ssp, neb, redshift=Z_FIXED, law="calzetti"):
    from tengri import Observation
    from tengri.observation import Photometry

    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(FILTERS)),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "single_component",
            "law": law,
            "tau_v": Uniform(0.0, 3.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb=neb,
        redshift=redshift,
        approx=WavePrecomp(),
    )


def _worst_node_error(ssp, monkeypatch, **kw):
    """Max relative photometry difference over the LUT's own nodes."""
    on = _with_neb(ssp, **kw)
    if on._energy_balance_lut_cache is None:
        return None  # declined
    nodes = np.asarray(on._energy_balance_lut_cache.tau_diff_grid)
    monkeypatch.setattr(
        SEDModel,
        "_EB_ATTEN_FREE_OK",
        frozenset(SEDModel._EB_ATTEN_FREE_OK - {"dust_tau_v"}),
    )
    off = _with_neb(ssp, **kw)
    assert off._energy_balance_lut_cache is None, "control arm kept the LUT"
    base = dict(on.spec.sample(jax.random.PRNGKey(0)))
    worst = 0.0
    for tau_v in nodes:
        p = dict(base)
        p["dust_tau_v"] = np.float64(tau_v)
        a = np.asarray(on.predict_photometry(p))
        b = np.asarray(off.predict_photometry(p))
        worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))))
    return worst


#: The log-domain sum of a LUT stellar term and an integrated nebular term
#: loses a little precision against one direct integral. Measured 4.3e-07 with
#: Cue; 1e-5 is loose against that and still four orders below the 7.7e-2 the
#: dropped-nebular bug produced.
NEB_NODE_RTOL = 1e-5


def test_an_active_nebular_backend_is_absorbed_too(synthetic_ssp_wide, monkeypatch):
    """The defect: the fast path used to assign all absorption to the stellar term."""
    worst = _worst_node_error(synthetic_ssp_wide, monkeypatch, neb=CUE_NEB)
    assert worst is not None, "the LUT declined; this test needs it to engage"
    assert worst < NEB_NODE_RTOL, (
        f"nebular absorption is not reaching the dust budget: {worst:.3e} at a "
        "LUT node, where interpolation is exact"
    )


@pytest.mark.parametrize(
    "neb_override",
    [
        pytest.param({"neb_logU": Uniform(-3.5, -1.5)}, id="free_logU"),
        pytest.param({"neb_logZ_gas": Uniform(-1.0, 0.3)}, id="free_logZ_gas"),
    ],
)
def test_free_nebular_parameters_stay_correct(synthetic_ssp_wide, monkeypatch, neb_override):
    """The nebular term is integrated at runtime, so freeing its parameters is fine.

    Worth pinning rather than assuming: had the nebular contribution been baked
    into the LUT instead, a free parameter would have frozen it at its build-time
    value and the error would be invisible at the default.
    """
    neb = {**CUE_NEB, **neb_override}
    worst = _worst_node_error(synthetic_ssp_wide, monkeypatch, neb=neb)
    assert worst is not None, "the LUT declined; this test needs it to engage"
    assert worst < NEB_NODE_RTOL, f"free nebular parameter broke the LUT: {worst:.3e}"


def test_free_redshift_is_fine_when_the_law_does_not_read_it(synthetic_ssp_wide, monkeypatch):
    """Calzetti has no redshift dependence, so a baked curve stays valid."""
    worst = _worst_node_error(
        synthetic_ssp_wide, monkeypatch, neb=CUE_NEB, redshift=Uniform(0.05, 1.5)
    )
    assert worst is not None, "the LUT declined even though calzetti ignores redshift"
    assert worst < NEB_NODE_RTOL, f"free redshift broke the LUT: {worst:.3e}"


def test_free_redshift_with_a_redshift_dependent_law_is_refused(synthetic_ssp_wide):
    """A refusal that is not a workaround: the curve genuinely cannot be baked.

    ``narayanan_z`` reads redshift, so a free redshift means a different
    attenuation curve per sample and no single build-time table can represent
    it. Declining is correct; the exact integral still runs.
    """
    model = _with_neb(
        synthetic_ssp_wide, neb=CUE_NEB, redshift=Uniform(0.05, 1.5), law="narayanan_z"
    )
    assert model._energy_balance_lut_cache is None, (
        "a redshift-dependent law with free redshift got a LUT baked at one redshift"
    )
