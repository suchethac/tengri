# SPDX-License-Identifier: BSD-3-Clause
r"""Per-channel bound on the exact-vs-precomp divergence (#1738).

``predict_via_precomp`` is a **second implementation** of :meth:`SEDModel.predict`,
not a spelling of it, and deliberately so: it reaches its answer without referencing
the dense SED, which is what lets XLA eliminate the full-resolution chain (#1109).
The two are kept in sync by hand. That is a maintainable arrangement only while the
divergence is *bounded and attributed* — otherwise drift accumulates silently in
whichever channel nobody measured this quarter.

Before this file the only bound in the tree was a single
``assert rel.max() < 0.02`` on one synthetic configuration
(``test_precompute_accuracy_synthetic.py``). One number over one config cannot say
*which* channel moved, so a regression in a small channel hides inside the budget of
a large one — which is how #1122, #1148 and #1665 all reached main.

Each case below turns one channel on over a fixed baseline and bounds the whole-model
gap. The bounds are **measured ceilings with headroom, not targets**: raising one is
a decision to accept a larger divergence and belongs in review, not in a quiet edit.

Measured 2026-08-15 on the FSPS SSP through SDSS *gri*, delayed SFH, z as noted:

=============================  ======================  ===================
channel added                  measured max rel. gap   bound here
=============================  ======================  ===================
stellar only                   3.2e-05 - 7.8e-04       2e-03
+ nebular, two_component       3.1e-05 - 7.8e-04       2e-03
+ nebular, single_component    1.8e-03 - 2.0e-03       5e-03
+ shock (MAPPINGS V)           4.5e-02 - 3.8e-01       5e-01
=============================  ======================  ===================

**The two nebular rows differ because the #1738 fix reaches only one of them.** That
row split is the point: a single "nebular" row would have averaged an exact channel
together with a defective one and reported something true of neither. The
``predict_via_precomp`` docstring claimed nebular was exact *full stop* for about a
day before measurement contradicted it — the qualifier is load-bearing, not pedantry.

**The shock row pins a known defect, not an acceptable state** — and it is a *bigger*
defect than the tree's own comments say. ``predict_via_precomp`` documents shock at
"2.4 % / 5.1 % / 8.3 %" and attributes it to band-averaging, fixable with a sub-band
LUT. Measured here on a real SSP it reaches **37.7 %** at
:math:`\tau=2`, :math:`z=1`, and band-averaging is not the mechanism:

* ``two_component.py`` never reads ``sed_shock``. It forms
  ``non_stellar_other = non_stellar_pre_dust - sed_neb`` and adds that bucket
  **unattenuated**, so the exact path applies *no* stellar dust screen to shock.
* Measured: in the exact path, shock's photometric contribution divided by its
  intrinsic ``shock_phot_lnu_precomp`` is **1.66e-55 regardless of τ** — the bare
  cosmology factor, with no attenuation term. ``predict_via_precomp`` meanwhile
  multiplies shock by ``a_diff·a_bc``, which is 0.025-0.109 at :math:`\tau_{\rm bc}=2`.

So the two paths disagree about **whether shock is attenuated at all**, not about how
finely the screen is sampled — and no quadrature closes a factor-of-40 gap. Which
behavior is right is a physics decision (shocked gas plausibly sits behind some
dust), so it is bounded here rather than silently changed. Verified pre-existing by
direct A/B: 3.770e-01 both with and without the #1738 nebular fix.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.contract

_STELLAR_FLOOR_BOUND = 2e-3
_NEBULAR_BOUND = 2e-3
#: Known-defect ceiling for nebular under ``single_component`` dust, which the #1738
#: fix does not reach. Measured worst case 1.955e-03; not a target. See the module
#: docstring — closing it is sequenced after #1808.
_SINGLE_COMPONENT_NEBULAR_BOUND = 5e-3
#: Known-defect ceiling — see the module docstring. Not a target. Measured worst case
#: is 3.77e-01 (tau=2, z=1); the headroom here is for grid/filter drift only, and the
#: right way to change this number is downward, by fixing the channel.
_SHOCK_BOUND = 5e-1


def _build(ssp, approx, *, tau_diff, tau_bc, z, neb, shock, dust_type="two_component"):
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel

    shock_group = {"frac": 1.0, "all_params": FIXED} if shock else {"type": "none"}
    if dust_type == "single_component":
        # One screen over all stars; tau_v is its depth. tau_bc has no analog here.
        dust_group = {
            "type": "single_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_v": tau_diff,
        }
    else:
        dust_group = {
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_diff": tau_diff,
            "tau_bc": tau_bc,
        }
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": 10.0,
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust=dust_group,
        neb={"type": neb, "all_params": FIXED},
        shock=shock_group,
        redshift=Fixed(z),
        approx=approx,
    )


def _gap(ssp, *, tau_diff, tau_bc, z, neb, shock, dust_type="two_component"):
    from tengri import WavePrecomp

    kw = dict(tau_diff=tau_diff, tau_bc=tau_bc, z=z, neb=neb, shock=shock, dust_type=dust_type)
    exact = np.asarray(_build(ssp, None, **kw).predict_photometry({}))
    lut = np.asarray(_build(ssp, WavePrecomp(), **kw).predict_photometry({}))
    return float(np.max(np.abs(lut - exact) / np.maximum(np.abs(exact), 1e-300)))


_SCREENS = [(0.5, 1.0, 0.05), (1.0, 2.0, 0.05), (2.0, 2.0, 1.0)]


@pytest.mark.parametrize(("tau_diff", "tau_bc", "z"), _SCREENS)
def test_no_screen_means_no_divergence(ssp_data_fsps, tau_diff, tau_bc, z):
    """Every bound below is attributable only if tau=0 already agrees.

    Parametrized over the same screens so the identity of the fixture cannot drift
    between this control and the cases it licenses.
    """
    del tau_diff, tau_bc  # the control is defined by tau=0; z is what varies
    gap = _gap(ssp_data_fsps, tau_diff=0.0, tau_bc=0.0, z=z, neb="cue", shock=False)
    assert gap < 1e-6, (
        f"exact and precomp differ by {gap:.3e} at z={z} with NO screen. The screens "
        "are what the bounds below measure; a gap here means the filter integration "
        "itself diverged and every attribution in this file is wrong."
    )


@pytest.mark.parametrize(("tau_diff", "tau_bc", "z"), _SCREENS)
def test_stellar_channel_stays_at_the_quadrature_floor(ssp_data_fsps, tau_diff, tau_bc, z):
    """The K=5 sub-band quadrature of #1122 — the floor for the whole path."""
    gap = _gap(ssp_data_fsps, tau_diff=tau_diff, tau_bc=tau_bc, z=z, neb="none", shock=False)
    assert gap < _STELLAR_FLOOR_BOUND, (
        f"stellar-only precomp gap {gap:.3e} exceeds {_STELLAR_FLOOR_BOUND:.0e} at "
        f"tau_diff={tau_diff}, tau_bc={tau_bc}, z={z}. This is the floor every other "
        "channel is measured against; if it moved, re-measure the others before "
        "trusting them."
    )


@pytest.mark.parametrize(("tau_diff", "tau_bc", "z"), _SCREENS)
def test_nebular_channel_adds_nothing_above_the_floor(ssp_data_fsps, tau_diff, tau_bc, z):
    """Nebular is screened exactly since #1738 — it must not reintroduce a gap."""
    gap = _gap(ssp_data_fsps, tau_diff=tau_diff, tau_bc=tau_bc, z=z, neb="cue", shock=False)
    assert gap < _NEBULAR_BOUND, (
        f"nebular precomp gap {gap:.3e} exceeds {_NEBULAR_BOUND:.0e} at "
        f"tau_diff={tau_diff}, tau_bc={tau_bc}, z={z}. Nebular is supposed to be "
        "screened by an exact band integral; a gap this size means it has fallen back "
        "to the lambda_eff form."
    )


@pytest.mark.parametrize(("tau_diff", "tau_bc", "z"), _SCREENS)
def test_shock_channel_stays_within_its_known_defect_bound(ssp_data_fsps, tau_diff, tau_bc, z):
    """Shock is still screened at lambda_eff. Bounded so it cannot worsen unnoticed."""
    gap = _gap(ssp_data_fsps, tau_diff=tau_diff, tau_bc=tau_bc, z=z, neb="cue", shock=True)
    assert gap < _SHOCK_BOUND, (
        f"shock precomp gap {gap:.3e} exceeds the known-defect ceiling "
        f"{_SHOCK_BOUND:.2e} at tau_diff={tau_diff}, tau_bc={tau_bc}, z={z}."
    )


@pytest.mark.parametrize(("tau_diff", "tau_bc", "z"), _SCREENS)
def test_single_component_nebular_stays_within_its_known_defect_bound(
    ssp_data_fsps, tau_diff, tau_bc, z
):
    """The #1738 fix covers two-component dust only; bound what it does not reach.

    ``single_component`` reddens nebular through a screen applied to the
    already-summed ``sed_intrinsic``, so no separately reddened nebular SED exists
    for the band projection to consume and the lambda_eff form survives. Measured
    ~3x over the stellar floor — an order of magnitude better than the 26x removed on
    two-component, and still worth pinning so it cannot drift while it waits on #1808.
    """
    gap = _gap(
        ssp_data_fsps,
        tau_diff=tau_diff,
        tau_bc=tau_bc,
        z=z,
        neb="cue",
        shock=False,
        dust_type="single_component",
    )
    assert gap < _SINGLE_COMPONENT_NEBULAR_BOUND, (
        f"single_component nebular precomp gap {gap:.3e} exceeds the known-defect "
        f"ceiling {_SINGLE_COMPONENT_NEBULAR_BOUND:.2e} at tau_v={tau_diff}, z={z}."
    )


def test_the_two_component_fix_did_not_reach_single_component(ssp_data_fsps):
    """Pin the SCOPE of #1738, not just its effect.

    Two assertions that must both hold, because either alone is misleading. The
    two-component arm proves the exact band-integrated screen is engaging at all;
    the single-component arm proves this test is still measuring a real gap. If the
    single-component arm ever drops to the floor, the fix has been extended and this
    test plus the bound above are stale — tighten them in that change.
    """
    kw = dict(tau_diff=1.0, tau_bc=1.0, z=0.05, neb="cue", shock=False)
    floor = _gap(ssp_data_fsps, neb="none", tau_diff=1.0, tau_bc=1.0, z=0.05, shock=False)
    two_comp = _gap(ssp_data_fsps, dust_type="two_component", **kw)
    single = _gap(ssp_data_fsps, dust_type="single_component", **kw)

    assert two_comp <= floor * 1.5 + 1e-5, (
        f"two_component nebular ({two_comp:.3e}) is no longer at the stellar floor "
        f"({floor:.3e}); the exact band-integrated screen has stopped engaging."
    )
    assert single > 2.0 * two_comp, (
        f"single_component nebular ({single:.3e}) is no longer materially worse than "
        f"two_component ({two_comp:.3e}). Either the fix was extended — in which case "
        "tighten _SINGLE_COMPONENT_NEBULAR_BOUND and delete this test — or the "
        "two-component path regressed."
    )


def test_shock_is_still_the_worst_channel(ssp_data_fsps):
    """Pin the ordering, so the docstring's ranking cannot quietly become false.

    If shock ever stops dominating, either it was fixed — in which case its bound
    above is stale and must be tightened — or something else regressed past it. Both
    are things a reviewer must see.
    """
    kw = dict(tau_diff=1.0, tau_bc=2.0, z=0.05)
    stellar = _gap(ssp_data_fsps, neb="none", shock=False, **kw)
    nebular = _gap(ssp_data_fsps, neb="cue", shock=False, **kw)
    shock = _gap(ssp_data_fsps, neb="cue", shock=True, **kw)

    assert shock > 10 * nebular, (
        f"shock ({shock:.3e}) no longer dominates nebular ({nebular:.3e}). If shock "
        f"was given an exact screen, tighten _SHOCK_BOUND to match; this file's "
        "ranking is now wrong either way."
    )
    assert nebular < 3 * max(stellar, 1e-9), (
        f"nebular ({nebular:.3e}) has drifted above the stellar floor "
        f"({stellar:.3e}); the exact nebular screen of #1738 is not engaging."
    )
