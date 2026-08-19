# SPDX-License-Identifier: BSD-3-Clause
"""The IGM band factor averaged T alone, and the fix must not cost anything (#1135).

``igm_phot_factor`` (#1107) band-averages the transmission **unweighted by the
spectrum**, forming ``<S>*<T>`` where the flux needs ``<S*T>``. Across GALEX FUV at
z~0.8 the transmission runs from ~1 to ~0 *inside* the bandpass, so that covariance
term reached -9.5%. #1122's sub-band quadrature already carried the machinery to fix
it — evaluate T at the same nodes as the dust screen — but doing so *at runtime* took
the compiled kernel from 277k to 2.42M FLOPs and tripped the LUT's own cheapness
guard, so it was reverted.

The transmission has no free parameters (absent patchy reionization / DLAs), so the
node values are a BUILD-TIME constant. #1135 folds them into the sub-band weights
before the SSP metallicity contraction, which is exact and free.

Every accuracy assertion compares the LUT against the **exact path** (approx=None) —
never against another preintegral. A precompute-vs-preintegral gate shares the
machinery under test and is structurally blind to errors they have in common.
"""

import jax
import numpy as np
import pytest

import tengri
from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.igm.component import IGMSEDComponent, IGMSEDComponentConfig

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r"]
KEY = jax.random.PRNGKey(0)

#: The IGM's covariance with the SED is a rest-UV effect. FUV is the band where the
#: transmission actually sweeps through the bandpass at these redshifts.
FUV, NUV = 0, 1


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


def _build(ssp, approx, *, z=0.8, tau_diff=0.0, tau_bc=0.0, igm=True, **kw):
    extra = {} if igm else {"igm": {"type": "none"}}
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        sfh={"type": "dpl", "*": FREE},
        dust={"law": "power_law", "type": "two_component", "*": FIXED, "tau_diff": tau_diff, "tau_bc": tau_bc},
        redshift=Fixed(z),
        approx=approx,
        **extra,
        **kw,
    )


def _err_vs_exact(ssp, approx, **kw):
    """Signed per-band |LUT/exact - 1|. ONE param dict, sampled from the exact model
    and SHARED — ``spec.sample()`` returns FIXED params too, so a dict sampled from a
    differently-configured model would override this one's fixed dust."""
    m_exact = _build(ssp, None, **kw)
    p = dict(m_exact.spec.sample(KEY))
    exact = np.asarray(m_exact.predict_photometry(p))
    lut = np.asarray(_build(ssp, approx, **kw).predict_photometry(p))
    return lut / exact - 1.0


def _flops(model, p):
    f = jax.jit(lambda q: model.predict_photometry(q))
    analysis = f.lower(p).compile().cost_analysis()
    if isinstance(analysis, list):
        analysis = analysis[0]
    return int(analysis["flops"])


# ── 1. the covariance term itself ─────────────────────────────────────────────


@pytest.mark.parametrize("z", [0.8, 1.0, 1.5])
def test_igm_covariance_is_captured_at_the_quadrature_nodes(ssp, z):
    """GALEX FUV/NUV must track the exact path once T rides the quadrature nodes.

    Dust OFF, so the IGM term is isolated: any residual here is the band average's
    ``<S>*<T>`` gap and nothing else. Before #1135 this read -9.46% (FUV, z=0.8) and
    -10.76% (NUV, z=1.5).

    Neuter-check: drop the fold (return ``stellar_state`` unchanged from
    ``_fold_igm_into_subbands``) and these go to ~-9% / ~-11%.
    """
    err = _err_vs_exact(ssp, WavePrecomp(), z=z)
    assert abs(err[FUV]) < 0.01, f"GALEX FUV off by {err[FUV] * 100:+.2f}% at z={z}"
    assert abs(err[NUV]) < 0.01, f"GALEX NUV off by {err[NUV] * 100:+.2f}% at z={z}"


def test_the_igm_is_the_whole_fuv_floor(ssp):
    """With the IGM off, the LUT already tracked the exact path — which is how the
    FUV floor left over from #1122 was identified as IGM rather than as the dust
    quadrature or the z grid.

    Pins that attribution: if a future change re-introduces a dust-side FUV error,
    this goes red *without* the IGM being involved, and the next reader is not sent
    chasing the transmission again.
    """
    err = _err_vs_exact(ssp, WavePrecomp(), z=0.8, tau_diff=0.7, tau_bc=1.0, igm=False)
    assert np.max(np.abs(err)) < 0.005, f"IGM-free LUT off by {np.max(np.abs(err)) * 100:.2f}%"


# ── 2. the cost — the reason #1133's version was reverted ─────────────────────


def test_the_fold_is_free_at_runtime(ssp):
    """Folding T at build time must not move the compiled kernel appreciably.

    The reverted runtime implementation cost 2.42M FLOPs (277k before it) because it
    ran the Inoue+2014 evaluation on ~1900 node wavelengths on EVERY call. The fold
    is a build-time constant multiply, so the runtime einsum is the same shape it
    always was — the IGM term should cost order 1e5, not 1e6, FLOPs.

    Neuter-check: restore the runtime evaluation and this goes to ~2.0M.
    """
    m_exact = _build(ssp, None, tau_diff=0.7, tau_bc=1.0)
    p = dict(m_exact.spec.sample(KEY))

    with_igm = _flops(_build(ssp, WavePrecomp(), tau_diff=0.7, tau_bc=1.0), p)
    without_igm = _flops(_build(ssp, WavePrecomp(), tau_diff=0.7, tau_bc=1.0, igm=False), p)
    igm_cost = with_igm - without_igm

    assert igm_cost < 300_000, (
        f"the IGM term costs {igm_cost:,} FLOPs — it is supposed to be a build-time "
        f"constant folded into the sub-band weights, not a per-call evaluation "
        f"(the reverted runtime version cost ~2,030,000)"
    )
    # And the LUT as a whole must stay far cheaper than exact (#1107's guard).
    exact = _flops(m_exact, p)
    assert exact / with_igm >= 10.0, (
        f"LUT is only {exact / with_igm:.1f}x cheaper than exact; the IGM fold must "
        f"not pin the full-resolution grid alive"
    )


# ── 3. rest-frame photometry carries no IGM ───────────────────────────────────


def test_rest_frame_photometry_carries_no_igm(ssp):
    """``phot_rest_fnu`` is projected at z=0 and carries no IGM, by contract.

    The IGM-folded tensor is therefore published *alongside* the IGM-free one rather
    than replacing it. Replacing it — the obvious implementation — would silently
    redden rest-frame photometry, which is exactly the kind of quiet corruption the
    fast path is prone to.
    """
    p = dict(_build(ssp, None, tau_diff=0.7, tau_bc=1.0).spec.sample(KEY))
    rest_igm = np.asarray(
        _build(ssp, WavePrecomp(), tau_diff=0.7, tau_bc=1.0).predict_observables(p).phot_rest_fnu
    )
    rest_none = np.asarray(
        _build(ssp, WavePrecomp(), tau_diff=0.7, tau_bc=1.0, igm=False)
        .predict_observables(p)
        .phot_rest_fnu
    )
    np.testing.assert_allclose(rest_igm, rest_none, rtol=1e-12)


# ── 4. the metallicity axis — why the fold happens before the SSP contraction ──


def test_the_transmission_is_evaluated_at_the_per_metallicity_node(ssp):
    """T must be folded per-(met, age, filter, sub-band), not at a marginalized node.

    The node published at runtime is a metallicity-weighted average whose weights move
    with the FREE ``met_logzsol``, so T at "the node" is a function of (z, Z) — not of
    z alone. On the production SSP grid the node shifts by up to 68% of a sub-band
    width across metallicity and T there by up to 1.3% in GALEX FUV, so a table keyed
    on (z, age, filter, k) — which is what #1135 originally proposed — would be built
    at one metallicity and used at all of them, drifting silently as the sampler
    explores met.

    Recovers the implied transmission from the folded weights and checks it against
    ``igm_absorption`` evaluated independently AT THE PER-MET NODES. Folding at a
    marginalized node, or on a met-collapsed table, turns this red.

    Asserting the *wiring* rather than the size of the met effect: the magnitude is a
    property of the SSP grid (it vanishes on grids whose templates do not reach the
    rest-UV of these bands), so a threshold on it is not an invariant and would make
    this test depend on which SSP is installed.
    """
    from tengri.components.igm.igm import igm_absorption

    z = 0.8
    m = _build(ssp, WavePrecomp(), z=z)
    lut = m._cached_component_chain[0]._state.ssp_phot_lut
    assert lut.ssp_subband_phot_igm is not None, "the IGM fold did not happen"
    assert lut.ssp_subband_phot_igm.shape == lut.ssp_subband_phot.shape
    assert lut.ssp_subband_phot_igm.ndim == 4, "(n_met, n_age, n_filter, n_subbands)"

    # Recover T on the RATIO, not on the weights: Phi is a filter integral in small
    # L_sun/Hz units, so allclose's absolute tolerance would swamp a factor-of-0.85
    # attenuation outright and report a broken fold as fine.
    folded = np.asarray(lut.ssp_subband_phot_igm)
    bare = np.asarray(lut.ssp_subband_phot)
    live = bare > 0.0
    implied_T = folded[live] / bare[live]
    assert np.all(implied_T > 0.0) and np.all(implied_T <= 1.0 + 1e-9), (
        "the folded factor is not a transmission"
    )

    # The wiring invariant: T at each node, recomputed independently.
    nodes = np.asarray(lut.ssp_subband_waves_rest)
    expected = np.asarray(igm_absorption((nodes * (1.0 + z)).reshape(-1), z)).reshape(nodes.shape)[
        live
    ]
    np.testing.assert_allclose(
        implied_T,
        expected,
        rtol=1e-10,
        err_msg="the folded transmission was not evaluated at the per-metallicity node",
    )


def test_the_fold_tracks_metallicity(ssp):
    """Moving ``met_logzsol`` must move the IGM-attenuated flux the way the exact
    path does. This is what a (z, age, filter, k) table could not do.
    """
    m_exact = _build(ssp, None, z=0.8)
    m_lut = _build(ssp, WavePrecomp(), z=0.8)
    p = dict(m_exact.spec.sample(KEY))
    for met in (-1.5, -0.5, 0.0, 0.3):
        q = dict(p, met_logzsol=met)
        exact = np.asarray(m_exact.predict_photometry(q))
        lut = np.asarray(m_lut.predict_photometry(q))
        err = abs(lut[FUV] / exact[FUV] - 1.0)
        assert err < 0.015, f"GALEX FUV off by {err * 100:.2f}% at met_logzsol={met}"


# ── 5. the gate must fail SAFE ────────────────────────────────────────────────


@pytest.mark.parametrize("cfg", [{"igm_patchy": True}, {"use_dla": True}])
def test_free_parameter_configs_are_not_frozen(cfg):
    """Patchy reionization and DLAs read free parameters (``igm_x_HI``,
    ``dla_log_n_hi``, …), so T is not a function of (lambda, z) alone. Tabulating it
    at build time would silently pin a transmission the sampler is still moving —
    the BOSA failure mode from #1107, where an emitter whose shape tracked its own
    luminosity sailed through a response built at a single value and returned fluxes
    13% wrong.

    The gate must refuse rather than freeze.
    """
    comp = IGMSEDComponent(config=IGMSEDComponentConfig(**cfg))
    nodes = np.linspace(1000.0, 5000.0, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    assert comp.subband_node_transmission(nodes, [1.0]) is None


def test_shape_disagreement_refuses_rather_than_broadcasts():
    """A z grid that does not match the node table's leading axis is a wiring bug.
    Returning None (fall back to the live path) is safe; broadcasting something
    plausible into the forward model is not.
    """
    comp = IGMSEDComponent()
    nodes = np.linspace(1000.0, 5000.0, 7 * 2 * 3 * 4 * 5).reshape(7, 2, 3, 4, 5)
    assert comp.subband_node_transmission(nodes, np.linspace(0.1, 2.0, 5)) is None


# ── 6. free redshift — the fold rides the z-table's own grid ───────────────────


def test_free_redshift_folds_on_the_ztable_grid(ssp):
    """The folded table must live on the SAME z grid as the sub-band photometry it
    multiplies, so it rides the same triweight interpolation. A mismatched grid would
    interpolate T against the wrong redshifts — accurately, and wrongly.
    """
    m = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        sfh={"type": "dpl", "*": FREE},
        dust={"law": "power_law", "type": "two_component", "*": FIXED, "tau_diff": 0.7, "tau_bc": 1.0},
        redshift=Uniform(0.05, 1.6),
        approx=WavePrecomp(n_z=120),
    )
    zt = m._cached_component_chain[0]._state.ssp_phot_ztable
    assert zt.ssp_subband_phot_igm_table is not None, "free-z fold did not happen"
    assert zt.ssp_subband_phot_igm_table.shape == zt.ssp_subband_phot_table.shape
    assert zt.ssp_subband_phot_igm_table.shape[0] == zt.z_grid.shape[0]

    # And it must actually be accurate at a z away from the grid nodes.
    m_exact = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        sfh={"type": "dpl", "*": FREE},
        dust={"law": "power_law", "type": "two_component", "*": FIXED, "tau_diff": 0.7, "tau_bc": 1.0},
        redshift=Uniform(0.05, 1.6),
        approx=None,
    )
    p = dict(m_exact.spec.sample(KEY))
    p["redshift"] = 0.83
    exact = np.asarray(m_exact.predict_photometry(p))
    lut = np.asarray(m.predict_photometry(p))
    err = abs(lut[FUV] / exact[FUV] - 1.0)
    assert err < 0.02, f"free-z GALEX FUV off by {err * 100:.2f}%"
