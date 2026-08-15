# SPDX-License-Identifier: BSD-3-Clause
r"""The nebular bucket was screened at :math:`\lambda_{\rm eff}`, not through the band (#1738).

``predict_via_precomp`` reddened the whole nebular bucket by
:math:`A_{\rm diff}(\lambda_{\rm eff})\,A_{\rm bc}(\lambda_{\rm eff})` — one screen
sample per filter — while :meth:`~tengri.SEDModel.predict` integrates the reddened
nebular SED through the true transmission:

.. math::

    L_b^{\rm exact} = \frac{\int L_{\rm neb}(\lambda)\,A(\lambda)\,T(\lambda)\,w(\lambda)\,
                            {\rm d}\lambda}{\int T\,w\,{\rm d}\lambda}
    \;\neq\;
    A(\lambda_{\rm eff}) \cdot \frac{\int L_{\rm neb} T w\,{\rm d}\lambda}
                                    {\int T w\,{\rm d}\lambda}

The two agree only where :math:`A` is flat across the band. Nebular emission is
line-dominated, and a line sits where it sits — not at :math:`\lambda_{\rm eff}` — so
the sampled screen is wrong by the screen's own variation across the band. The stellar
continuum got the K-point sub-band quadrature in #1122; nebular was left behind, and
``predict_via_precomp`` said so in a comment:

    *"Still evaluated at λ_eff: the nebular component publishes no sub-band tensors,
    so the quadrature cannot reach it."*

**Why this is fixed exactly rather than by mirroring #1122's quadrature.** A sub-band
tensor would only *converge* toward the right answer (1/K²). It is unnecessary here:
on any model with dust the nebular continuum is already materialized —
``DustSEDComponent`` declares ``sed_nebular`` an input, which is what disarms the fast
nebular grid (#1281, #1748) — and the dust component already computes and republishes
its *reddened* form. Integrating that through the band is exact and reads a dense array
already live in the compiled graph. Where the dense continuum is **not** materialized
there is no dust consumer, hence no screen, hence nothing to fix.

**Measured on this fixture before the fix** (real FSPS SSP, SDSS *griz*, delayed SFH,
Cue nebular; max relative ``predict_photometry`` difference, precomp vs exact):

===========================  ==================  ==================
case                         stellar only        with nebular
===========================  ==================  ==================
tau_diff=0.5 tau_bc=1 z=.05  2.891e-04           1.298e-03
tau_diff=1.0 tau_bc=2 z=.05  6.360e-04           1.203e-03
tau_diff=2.0 tau_bc=2 z=1.0  7.843e-04           1.000e-03
tau_diff=0.3 tau_bc=.5 z=.5  3.244e-05           8.337e-04
===========================  ==================  ==================

Nebular is only 0.8–3.5 % of the band flux on this fixture yet contributes most of the
gap — up to a **26x** inflation over the stellar-only floor. That floor is the K=5
quadrature error of #1122 and is *not* removable by anything done here, which is why
the assertions below are written against the **measured stellar-only gap of the same
fixture** rather than a hardcoded constant: a fixed tolerance would silently encode
today's SSP grid, filter set and K.

**Do not quote 18 % / 46 % for this.** Those come from a synthetic SSP that trips Cue's
wNE guard and produces an absurd Q_H, so nebular dominates every band. It is a valid
stress case for localizing the mechanism and a wrong number for a science claim.

**The tau_bc = 0 control is not optional.** Without it "the paths agree" is equally
consistent with "the screen is integrated correctly" and with "this fixture has no
nebular emission to get wrong".
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

#: The published reddened term is a band integral of an array the pipeline already
#: holds — it must match a directly recomputed integral to float64 summation noise,
#: not merely "closely". Anything looser would pass on a screen still sampled at one
#: wavelength but scaled to look right.
_TERM_TOL = 1e-9

#: Headroom over the same fixture's stellar-only gap. The nebular channel is allowed
#: to be a minor contributor to the residual; it is not allowed to dominate it. Set
#: from the measured post-fix ratios with room for grid/filter drift, and deliberately
#: far below the 4.5x-26x inflation the bug produced.
_FLOOR_FACTOR = 1.5

#: Absolute slack so the assertion cannot be made vacuous by a case whose stellar-only
#: gap is near zero (tau_diff=0.3/z=0.5 measures 3.2e-05).
_FLOOR_SLACK = 1e-5

_CASES = [
    (0.5, 1.0, 0.05),
    (1.0, 2.0, 0.05),
    (2.0, 2.0, 1.0),
    (0.3, 0.5, 0.5),
]


def _build(ssp, approx, *, tau_diff, tau_bc, z, neb="cue", free_mass=False):
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(
            photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i", "sdss_z"])
        ),
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0) if free_mass else 10.0,
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_diff": tau_diff,
            "tau_bc": tau_bc,
        },
        neb={"type": neb, "all_params": FIXED},
        redshift=Fixed(z),
        approx=approx,
    )


def _max_rel(ssp, *, tau_diff, tau_bc, z, neb="cue"):
    """Max relative |precomp - exact| over the bands, on ``predict_photometry``."""
    from tengri import WavePrecomp

    kw = dict(tau_diff=tau_diff, tau_bc=tau_bc, z=z, neb=neb)
    exact = np.asarray(_build(ssp, None, **kw).predict_photometry({}))
    lut = np.asarray(_build(ssp, WavePrecomp(), **kw).predict_photometry({}))
    return float(np.max(np.abs(lut - exact) / np.maximum(np.abs(exact), 1e-300)))


def test_control_the_fixture_actually_has_nebular_emission(ssp_data_fsps):
    """Guard the guard: with no screen the paths agree, and the nebular bucket is live.

    This is what localizes every failure below to the screen rather than to the filter
    integration, and it fails loudly if the fixture stops emitting.
    """
    from tengri import WavePrecomp

    zeroed = _max_rel(ssp_data_fsps, tau_diff=0.0, tau_bc=0.0, z=0.05)
    assert zeroed < 1e-6, (
        f"exact and precomp disagree by {zeroed:.3e} with NO dust screen at all. The "
        "residual under test is the screen; a gap here means the filter integration "
        "itself diverged and every assertion below is misattributed."
    )

    state = _build(ssp_data_fsps, WavePrecomp(), tau_diff=0.0, tau_bc=0.0, z=0.05).predict_state(
        {}
    )
    neb = np.asarray(state.derived["nebular_phot_lnu_precomp"])
    assert np.all(neb > 0.0), (
        f"nebular photometry is not positive in every band ({neb}); this fixture "
        "cannot exercise the nebular screen."
    )


@pytest.mark.parametrize(("tau_diff", "tau_bc", "z"), _CASES)
def test_nebular_no_longer_dominates_the_precomp_gap(ssp_data_fsps, tau_diff, tau_bc, z):
    """Turning nebular on must not inflate the gap beyond the stellar quadrature floor.

    Written against the same fixture's ``neb='none'`` gap rather than a constant: the
    floor is the K=5 error of #1122 and moves with the SSP grid, the filter set and K,
    none of which this fix controls.
    """
    kw = dict(tau_diff=tau_diff, tau_bc=tau_bc, z=z)
    floor = _max_rel(ssp_data_fsps, neb="none", **kw)
    with_neb = _max_rel(ssp_data_fsps, neb="cue", **kw)

    budget = floor * _FLOOR_FACTOR + _FLOOR_SLACK
    assert with_neb <= budget, (
        f"nebular inflates the precomp gap {with_neb / max(floor, 1e-300):.1f}x over the "
        f"stellar-only floor at tau_diff={tau_diff}, tau_bc={tau_bc}, z={z} "
        f"({floor:.3e} -> {with_neb:.3e}, budget {budget:.3e}). The nebular screen is "
        "being sampled at one wavelength per band instead of integrated through it."
    )


@pytest.mark.parametrize(("tau_diff", "tau_bc", "z"), _CASES)
def test_published_reddened_nebular_term_is_the_exact_band_integral(
    ssp_data_fsps, tau_diff, tau_bc, z
):
    """The published term must BE the band integral of the reddened nebular SED.

    Recomputed here straight from the dense arrays the state carries, so this pins the
    quantity itself rather than its effect on a total that other channels also move.
    """
    from tengri import WavePrecomp
    from tengri.observation.photometry import lnu_filter_integral

    model = _build(ssp_data_fsps, WavePrecomp(), tau_diff=tau_diff, tau_bc=tau_bc, z=z)
    state = model.predict_state({})
    published = np.asarray(state.derived["nebular_phot_lnu_attenuated_precomp"])

    recomputed = np.asarray(
        [
            float(
                lnu_filter_integral(
                    state.derived["sed_nebular"], state.wave, f.wave, f.trans, redshift=z
                )
            )
            for f in model.observation.photometry.filters
        ]
    )

    np.testing.assert_allclose(
        published,
        recomputed,
        rtol=_TERM_TOL,
        err_msg=(
            "the published reddened nebular band term is not the band integral of the "
            "reddened nebular SED that sits beside it in the same state."
        ),
    )

    # A silent zero is the failure mode this fix could plausibly acquire: the exact
    # term is read from the dense continuum, and the fast nebular grid serves
    # photometry by ZEROING that continuum (#1281). Today a dust component is itself
    # what disarms the grid, so the two can never coexist — but if that invariant ever
    # breaks, every dusty model loses its nebular light with nothing raised. The
    # lambda_eff form is a bounded band-average of the same quantity, so it is a
    # legitimate order-of-magnitude witness.
    lameff = np.asarray(
        state.derived["dust_diff_attenuation_precomp"]
        * state.derived["dust_bc_attenuation_precomp"]
        * state.derived["nebular_phot_lnu_precomp"]
    )
    assert np.all(published > 0.0), (
        f"the reddened nebular band term is not positive in every band ({published}). "
        "A zero here means the dense nebular continuum was zeroed underneath this "
        "projection (the fast-grid path) and the nebular light has been silently lost."
    )
    np.testing.assert_allclose(
        published,
        lameff,
        rtol=0.5,
        err_msg=(
            "the exact band-integrated nebular term and its lambda_eff approximation "
            "differ by more than 50%. They are two estimates of one integral; a gap "
            "this size means one of them is not that integral at all."
        ),
    )


def test_the_lambda_eff_screen_is_still_demonstrably_wrong(ssp_data_fsps):
    """Pin the *mechanism*, so a future regression is diagnosed and not just detected.

    Reconstructs the old behavior from published state and asserts it is materially
    wrong. If this ever stops being wrong, the assertions above have become vacuous.
    """
    from tengri import WavePrecomp

    def lameff_error(tau_diff, tau_bc, z):
        d = (
            _build(ssp_data_fsps, WavePrecomp(), tau_diff=tau_diff, tau_bc=tau_bc, z=z)
            .predict_state({})
            .derived
        )
        exact_term = np.asarray(d["nebular_phot_lnu_attenuated_precomp"])
        lameff_term = np.asarray(
            d["dust_diff_attenuation_precomp"]
            * d["dust_bc_attenuation_precomp"]
            * d["nebular_phot_lnu_precomp"]
        )
        return float(
            np.max(np.abs(lameff_term - exact_term) / np.maximum(np.abs(exact_term), 1e-300))
        )

    mild = lameff_error(0.5, 1.0, 0.05)
    harsh = lameff_error(2.0, 2.0, 1.0)

    assert mild > 1e-3, (
        f"screening at lambda_eff is only {mild:.3e} off the band integral — the "
        "fixture no longer exercises the bug this test exists to pin."
    )
    assert harsh > mild, (
        f"the lambda_eff error did not grow with the screen ({mild:.3e} -> {harsh:.3e}); "
        "that ordering is the signature of a band-averaging error, and losing it means "
        "the residual now has some other cause."
    )


def test_rest_band_nebular_carries_the_same_screen(ssp_data_fsps):
    """The observed and rest-frame publishes are twins (#1148, #1665).

    Emitting the observed-band fix alone would leave every rest-frame consumer on the
    lambda_eff screen — the exact failure mode of #1665, which moved 13/13 spectral
    indices with nothing raised.
    """
    from tengri import WavePrecomp

    d = (
        _build(ssp_data_fsps, WavePrecomp(), tau_diff=1.0, tau_bc=2.0, z=0.3)
        .predict_state({})
        .derived
    )

    assert "nebular_restband_lnu_attenuated_precomp" in d, (
        "the observed band publishes an exact reddened nebular term but the rest band "
        "does not; the two are twins and a half-fix silently strands rest-frame consumers."
    )
    rb_exact = np.asarray(d["nebular_restband_lnu_attenuated_precomp"])
    rb_lameff = np.asarray(
        d["dust_diff_restband_attenuation_precomp"]
        * d["dust_bc_restband_attenuation_precomp"]
        * d["nebular_restband_lnu_precomp"]
    )
    assert np.all(np.isfinite(rb_exact)) and np.all(rb_exact > 0.0)
    gap = float(np.max(np.abs(rb_lameff - rb_exact) / np.maximum(np.abs(rb_exact), 1e-300)))
    assert gap > 1e-4, (
        f"the rest-band lambda_eff screen differs from its band integral by only "
        f"{gap:.3e}; this assertion is no longer pinning anything."
    )


def test_zero_tau_publishes_the_intrinsic_bucket_unchanged(ssp_data_fsps):
    """With no screen the reddened integral must reduce to the intrinsic one.

    A limit check: it catches a projection that silently applies some *other*
    normalization (a stray 1+z, a filter renormalization) which a comparison against
    the lambda_eff form at tau>0 would absorb.
    """
    from tengri import WavePrecomp

    d = (
        _build(ssp_data_fsps, WavePrecomp(), tau_diff=0.0, tau_bc=0.0, z=0.2)
        .predict_state({})
        .derived
    )
    np.testing.assert_allclose(
        np.asarray(d["nebular_phot_lnu_attenuated_precomp"]),
        np.asarray(d["nebular_phot_lnu_precomp"]),
        rtol=1e-9,
        err_msg=(
            "with tau=0 the reddened nebular band integral must equal the intrinsic "
            "one; a difference here is a projection/normalization error, not a screen."
        ),
    )


def test_wave_precomp_still_beats_the_exact_path_on_flops(ssp_data_fsps):
    """The fix must not claw back the WavePrecomp speedup (#1109).

    The point of the LUT is that branch 1 never references the dense *stellar* SED,
    letting XLA eliminate the full-resolution chain. Reading the *nebular* dense array
    is free only because a dusty model already materializes it; if that stops being
    true, this guard is what says so. FLOPs off the compiled HLO, never wall clock
    (#1696).
    """
    import jax

    from tengri import WavePrecomp

    def grad_flops(approx):
        model = _build(ssp_data_fsps, approx, tau_diff=0.5, tau_bc=1.0, z=0.1, free_mass=True)

        def loss(v):
            return jnp.sum(model.predict_photometry({"sfh_delayed_log_total_mass": v}))

        return int(
            jax.jit(jax.grad(loss)).lower(jnp.asarray(10.0)).compile().cost_analysis()["flops"]
        )

    lut = grad_flops(WavePrecomp())
    exact = grad_flops(None)
    assert lut < exact, (
        f"WavePrecomp gradient costs {lut:,} FLOPs vs the exact path's {exact:,}. The "
        "LUT is meant to be the cheap path; the nebular fix has made the dense chain "
        "live in a way that defeats the dead-code elimination it depends on."
    )
