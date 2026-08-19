# SPDX-License-Identifier: BSD-3-Clause
"""A composed ``approx`` must publish what each of its members publishes (#1665).

``WavePrecomp()`` alone is exact. ``FeaturePrecomp()`` alone is exact. Composed,
the pair silently dropped the nebular contribution from every rest-frame band:
13 of 13 spectral indices moved, worst ``HgA`` by **+1733%**, with no exception
and no warning.

Mechanism. ``NebularSEDComponent.apply`` takes its fast grid branch when
``grid_table.log_phot_per_qh`` is populated -- and that channel is only built
when ``WavePrecomp`` supplied filters. So the grid path fires for the *pair* and
for neither single, which is why single-member coverage could not see it. That
branch published ``nebular_phot_lnu_precomp`` (the observed band) and never its
rest-frame twin ``nebular_restband_lnu_precomp``, so ``measure_spectral_indices``
summed a rest band with the nebular emission missing. The five worst indices are
exactly the Balmer ones -- Hbeta, HgA, HgF, HdA, HdF -- whose windows sit on
nebular *emission* lines; the continuum-break and metal indices moved only by the
nebular *continuum*, a few tenths of a percent.

The rule pinned here is the general one, not the thirteen numbers: **the two
photometry publishes are twins**, so a path that emits one and not the other is
broken regardless of which index happens to expose it. ``test_restband_twin_*``
asserts that exactly, and needs no tolerance.

The same gate also skips the discrete line-catalog publish, which made
``predict_line_ratios`` raise a *misdirecting* error: a Cue model was told to
"Use Cue or CloudyGrid". ``test_line_ratio_refusal_*`` pins the refusal naming
the real cause -- an error that sends you to fix the one thing you did right is
worse than the raise it replaced.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_BARE_SSP_CANDIDATES = [
    "data/fsps_prsc_miles_chabrier.h5",
    "data/ssp_prsc_bc03_chabrier.h5",
]

# Every index the measurable registry exposes -- the defect hit all of them.
_INDEX_NAMES = [
    "Ca4227",
    "D4000",
    "Dn4000",
    "Fe4383",
    "Fe5270",
    "Fe5335",
    "Hbeta",
    "HdA",
    "HdF",
    "HgA",
    "HgF",
    "Mgb",
    "uv_slope_beta",
]

# Tolerance is *derived*, not read off the failure. The fast grid is an
# interpolant, so a correct pair still carries its interpolation error; the
# builder's own warning bounds that at ~1.3% worst-case for the
# collisionally-excited lines (uniform met axis) and ~0.5% node-snapped. 5%
# clears that ceiling with margin while sitting far below the smallest arm of
# the defect (+28.7% on HdA) -- so this threshold cannot be satisfied by the bug.
_GRID_INTERP_CEILING_PCT = 5.0


def _bare_ssp_path():
    return next((p for p in _BARE_SSP_CANDIDATES if Path(p).is_file()), None)


def _requirements():
    bare = _bare_ssp_path()
    if bare is None or not Path("data/cue_weights.npz").is_file():
        pytest.skip("No bare-stellar SSP / Cue weights available.")
    return bare


def _approx_arms():
    """(label, factory) for exact and each precompute combination."""
    from tengri.forward.sed_model import FeaturePrecomp, WavePrecomp

    return {
        "exact": lambda: None,
        "wave": lambda: WavePrecomp(),
        "feature": lambda: FeaturePrecomp(),
        "wave+feature": lambda: (WavePrecomp(), FeaturePrecomp()),
    }


def _build(approx, index_data):
    import warnings

    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, load_ssp_data

    ssp = load_ssp_data(_requirements())
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
        spectral_indices=index_data,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust={
                "law_diff": "calzetti",
                "type": "two_component",
                "law_bc": "calzetti",
                "all_params": FIXED,
            },
            neb={"type": "cue", "all_params": FIXED},
            redshift=Fixed(0.05),
            approx=approx,
        )


def _index_data():
    from tengri.observation.spectral_indices import SpectralIndexData

    return SpectralIndexData.from_names(
        _INDEX_NAMES,
        values=[1.0] * len(_INDEX_NAMES),
        errors=[0.05] * len(_INDEX_NAMES),
    )


@pytest.mark.parametrize("arm", ["wave", "feature", "wave+feature"])
def test_spectral_indices_agree_with_exact_or_refuse_loudly(arm):
    """No ``approx`` combination may *silently* move a spectral index.

    Two answers are acceptable and one is not. Returning a value binds that
    value to the exact path. Refusing is equally fine -- the fast-nebular grid
    deliberately deletes the Cue continuum forward, which is where its speedup
    comes from, so a rest-frame quantity genuinely cannot be served from it and
    ``predict_spectrum`` has refused on exactly this ground since #950.

    What is forbidden is the third outcome, which is what shipped: return a
    number, no exception, no warning, wrong by up to 1733%.

    Deliberately *not* asserted: which of the two acceptable answers the pair
    gives. Pinning "it raises" would turn a future correct-values implementation
    red for succeeding.
    """
    sid = _index_data()
    arms = _approx_arms()

    exact = np.asarray(_build(arms["exact"](), sid).predict_spectral_indices({}, sid.index_defs))
    model = _build(arms[arm](), sid)
    try:
        got = np.asarray(model.predict_spectral_indices({}, sid.index_defs))
    except ValueError as exc:
        assert "fast-nebular" in str(exc), (
            f"approx={arm!r} refused, but not with the fast-nebular explanation "
            f"a user can act on: {exc}"
        )
        return

    dev_pct = np.abs((got - exact) / np.abs(exact)) * 100.0
    worst = int(np.argmax(dev_pct))
    assert dev_pct[worst] < _GRID_INTERP_CEILING_PCT, (
        f"approx={arm!r} returned SILENTLY WRONG indices: {_INDEX_NAMES[worst]} "
        f"moved {dev_pct[worst]:.2f}% ({exact[worst]:.6f} -> {got[worst]:.6f}); "
        f"{int((dev_pct >= _GRID_INTERP_CEILING_PCT).sum())} of {len(_INDEX_NAMES)} "
        "indices exceed the grid-interpolation ceiling."
    )


@pytest.mark.parametrize("accessor", ["rest_sed", "obs_sed"])
def test_sed_accessors_agree_with_exact_or_refuse_loudly(accessor):
    """``pred.rest_sed()`` / ``pred.obs_sed()`` are rest-SED consumers too.

    They read ``state.sed_intrinsic``, the very array the fast grid zeroes the
    nebular contribution out of, so they belong to the same census as
    ``predict_spectrum``. Returning a nebular-free panchromatic SED to someone
    who asked for the model SED is the same defect wearing a different name.
    """
    from tengri.forward.sed_model import FeaturePrecomp, WavePrecomp

    sid = _index_data()
    exact = np.asarray(getattr(_build(None, sid).predict({}), accessor)())

    pred = _build((WavePrecomp(), FeaturePrecomp()), sid).predict({})
    try:
        got = np.asarray(getattr(pred, accessor)())
    except ValueError as exc:
        assert "fast-nebular" in str(exc), f"{accessor} refused unhelpfully: {exc}"
        return

    finite = np.isfinite(exact) & np.isfinite(got) & (np.abs(exact) > 0)
    dev_pct = np.abs((got[finite] - exact[finite]) / np.abs(exact[finite])) * 100.0
    assert dev_pct.max() < _GRID_INTERP_CEILING_PCT, (
        f"pred.{accessor}() returned SILENTLY WRONG values under the pair: "
        f"worst deviation {dev_pct.max():.2f}%."
    )


def test_sed_derived_properties_are_exact_on_the_fast_path():
    """Every nebular-continuum-dependent property must match the exact model.

    ``predict_properties`` reaches the rest SED by a different route from
    ``predict_spectral_indices`` (the quantities are built inside the forward
    pass), so it needs its own check. Under the pair, 13 of 43 properties moved
    -- worst ``l_tir`` by **30.07%**, the energy-balance family (irx 27%,
    l_dust_absorbed 19%) hardest, because deleting the nebular continuum removes
    reprocessed luminosity the dust budget balances against.

    That was fixed at the cause (#1673): ``predict_state`` materializes the
    nebular component, so these properties are served from a complete forward
    state. This asserts the stronger post-fix property -- **equality with the
    exact model** -- rather than the refusal that stood in for it, in the same
    spirit as the two ``_or_refuse_loudly`` tests above: never pin "it raises"
    when the values are available to compare.

    ``_FAST_NEBULAR_UNSAFE_PROPERTIES`` is now this census -- the properties that
    depend on the nebular continuum, and therefore the ones worth checking -- not
    a refusal list. ``stellar_mass`` is deliberately outside it and included here
    as the control: it was correct on the fast path even before the fix, so it
    cannot distinguish a real repair from a vacuous comparison.
    """
    from tengri.forward.sed_model import (
        _FAST_NEBULAR_UNSAFE_PROPERTIES,
        FeaturePrecomp,
        WavePrecomp,
    )

    sid = _index_data()
    names = (*sorted(_FAST_NEBULAR_UNSAFE_PROPERTIES), "stellar_mass")
    exact = _build(_approx_arms()["exact"](), sid).predict_properties({}, names=names)
    fast = _build((WavePrecomp(), FeaturePrecomp()), sid).predict_properties({}, names=names)

    assert "stellar_mass" not in _FAST_NEBULAR_UNSAFE_PROPERTIES

    worst_name, worst_pct = None, 0.0
    for name in names:
        ref, got = float(exact[name]), float(fast[name])
        assert np.isfinite(got), f"{name} is not finite on the fast path: {got}"
        pct = abs(got - ref) / max(abs(ref), 1e-300) * 100.0
        if pct > worst_pct:
            worst_name, worst_pct = name, pct
    assert worst_pct < _GRID_INTERP_CEILING_PCT, (
        f"{worst_name} moved {worst_pct:.3f}% between the fast pair and the exact "
        f"path ({float(exact[worst_name]):.6e} -> {float(fast[worst_name]):.6e}); the "
        "nebular continuum is not reaching predict_properties (#1665/#1673)."
    )


def test_line_ratios_agree_with_exact_or_refuse_without_misdirecting():
    """Line ratios need the discrete catalog the fast grid used to skip.

    Historically this refused, and the refusal itself was a second bug: a **Cue**
    model fell through to the generic backend message and was told *"Use Cue or
    CloudyGrid"* -- advice the user had already taken, naming a cause that was
    not theirs. An error that sends you to fix the one thing you did right is
    worse than the raise it replaced.

    Since #1673 the materialized state carries the discrete catalog, so the
    values are available and are compared against the exact model. The refusal
    branch is kept as an accepted outcome (same convention as the two
    ``_or_refuse_loudly`` tests above) -- but if it is taken, it must still not
    misdirect.
    """
    from tengri.forward.sed_model import FeaturePrecomp, WavePrecomp
    from tengri.observation import LineRatioData

    sid = _index_data()
    lrd = LineRatioData.from_dict({("Halpha", "Hbeta"): (2.86, 0.3)})
    exact = np.asarray(_build(_approx_arms()["exact"](), sid).predict_line_ratios({}, lrd))
    model = _build((WavePrecomp(), FeaturePrecomp()), sid)

    try:
        got = np.asarray(model.predict_line_ratios({}, lrd))
    except ValueError as exc:
        msg = str(exc)
        assert "fast-nebular" in msg, f"refusal does not name the fast path: {msg}"
        assert "Use Cue or CloudyGrid" not in msg, (
            "refusal still tells a Cue user to use Cue -- the pre-#1665 misdirection."
        )
        return

    dev_pct = np.abs((got - exact) / np.maximum(np.abs(exact), 1e-300)) * 100.0
    assert float(np.max(dev_pct)) < _GRID_INTERP_CEILING_PCT, (
        f"line ratios moved {float(np.max(dev_pct)):.3f}% between the fast pair and "
        f"the exact path ({exact} -> {got}); the discrete line catalog is not "
        "reaching predict_line_ratios (#1665/#1673)."
    )


@pytest.mark.parametrize("arm", ["wave", "feature", "wave+feature"])
def test_restband_twin_is_published_whenever_the_observed_band_is(arm):
    """``nebular_phot_lnu_precomp`` and its rest-frame twin ship together.

    This is the rule the thirteen indices were merely a symptom of, and it is an
    exact key-set assertion -- no tolerance, no SSP-grid sensitivity. A path that
    publishes the observed band while dropping the rest band is broken whether or
    not the caller happens to ask for an index.
    """
    sid = _index_data()
    state = _build(_approx_arms()[arm](), sid).predict_state({})

    if "nebular_phot_lnu_precomp" not in state.derived:
        pytest.skip(f"approx={arm!r} does not use the photometry LUT publish.")

    assert "nebular_restband_lnu_precomp" in state.derived, (
        f"approx={arm!r} published 'nebular_phot_lnu_precomp' but not "
        "'nebular_restband_lnu_precomp'. Every rest-frame consumer "
        "(spectral indices, rest-frame colors) then silently loses the "
        "nebular contribution -- see #1665."
    )
