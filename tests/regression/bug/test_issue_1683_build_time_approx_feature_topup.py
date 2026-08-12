# SPDX-License-Identifier: BSD-3-Clause
"""Naming ``approx=WavePrecomp()`` must not cost the feature LUT (#1683).

#1596 gave a photometry-only fit the ``FeaturePrecomp`` top-up — but only on
the branch where the model carries **no** build-time ``approx=``. The other
spelling stayed open on *both* fit surfaces: a model built
``approx=WavePrecomp()`` was returned untouched, so naming the wave LUT
explicitly *cost* the feature LUT and fit slower than passing nothing at all.

That is the same pathology ``_add_feature_precomp``'s own docstring already
recorded for a line channel — "naming WavePrecomp explicitly made a lines fit
slower than passing nothing at all" — left unfixed for the photometry-only
case #1596 added.

Measured cost, from the issue: **82,526,904 -> 3,737,260** likelihood-gradient
FLOPs (22x) on a 10-parameter Cue model, read from the compiled HLO. The batch
half is the worse one: ``PopulationFitter`` and ``CatalogFitter`` evaluate the
forward model the most, so a user who read the docs and was explicit opted
*out* of the dominant lever.

Verdict rule, fixed before the probe ran: *the gap is confirmed for a surface
iff the build-time-approx arm resolves WITHOUT ``FeaturePrecomp`` while the
no-approx arm resolves WITH it.* Both arms of both surfaces met it, so both
surfaces are pinned here — and the no-approx control is asserted in the same
test so the comparison cannot go vacuous.

Stubs, like the #1596 test: the question is which config the policy resolves,
which needs no SSP grid. What that config then computes is a different
question, owned by ``test_approx_channel_isolation.py``.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.forward.sed_model import FeaturePrecomp, WavePrecomp
from tengri.inference.fitter import Fitter, _resolve_batch_fit_approx

pytestmark = pytest.mark.regression_bug


class _StubApprox:
    def __init__(self, *, wave=False, feature=False, spectrum=False):
        self.wave_precomp = wave
        self.feature_precomp = feature
        self.spectrum_precomp = spectrum


class _StubObs:
    photometry = object()
    line_fluxes = None
    line_ratios = None
    spectral_indices = None


class _StubModel:
    """The minimum both resolvers read, with the build-time spelling as a knob."""

    def __init__(
        self,
        *,
        features_tabulate=True,
        build_time_wave=False,
        build_time_feature=False,
        observation=None,
    ):
        self.approx = _StubApprox(wave=build_time_wave, feature=build_time_feature)
        configs = ()
        if build_time_wave:
            configs += (WavePrecomp(),)
        if build_time_feature:
            configs += (FeaturePrecomp(),)
        self.approx_configs = configs
        self.features_tabulate = features_tabulate
        self.observation = observation if observation is not None else _StubObs()
        self.received_cfg = None
        self._build_time_wave = build_time_wave
        self._build_time_feature = build_time_feature

    def _has_modern_approx(self):
        return self._build_time_wave or self._build_time_feature

    def with_approx(self, cfg):
        cfgs = cfg if isinstance(cfg, tuple) else (cfg,)
        if any(isinstance(c, FeaturePrecomp) for c in cfgs) and not self.features_tabulate:
            # The real contract: a backend with nothing to tabulate raises
            # (measured: neb='none' -> ValueError); the raise IS the detection.
            raise ValueError("no emission lines to tabulate")
        new = _StubModel(
            features_tabulate=self.features_tabulate,
            build_time_wave=self._build_time_wave,
            build_time_feature=self._build_time_feature,
            observation=self.observation,
        )
        new.received_cfg = cfg
        return new


class _StubFitterSelf:
    data_type = "photometry"
    _line_flux_override = None
    _eline_marginalize = False
    _eline_fitted = False

    # ``_fits_line_fluxes`` delegates to ``_line_fluxes_for`` since #1691 (the
    # override and the observation attribute are one channel), so the stub has
    # to carry both — as a bound method, not a staticmethod.
    _line_fluxes_for = Fitter._line_fluxes_for
    _fits_line_fluxes = Fitter._fits_line_fluxes
    _fits_lines = Fitter._fits_lines
    _auto_approx_config = Fitter._auto_approx_config
    _add_feature_precomp = Fitter._add_feature_precomp
    _warn_lines_without_lut = Fitter._warn_lines_without_lut
    _resolve_fit_approx = Fitter._resolve_fit_approx

    def __init__(self, model):
        self.model = model


def _cfgs_of(resolved, original):
    """The precompute configs a resolver settled on, as a flat tuple.

    A resolver that returns the model untouched settled on the build-time
    configs — that is the defect's signature, not an error, so it is reported
    rather than raised on.
    """
    if resolved is original:
        return tuple(getattr(original, "approx_configs", ()))
    cfg = resolved.received_cfg
    return cfg if isinstance(cfg, tuple) else (cfg,)


def _single(model):
    return _cfgs_of(_StubFitterSelf(model)._resolve_fit_approx(model, "auto"), model)


def _batch(model):
    return _cfgs_of(_resolve_batch_fit_approx(model, "auto", "photometry"), model)


def _counts(cfgs):
    return (
        sum(isinstance(c, WavePrecomp) for c in cfgs),
        sum(isinstance(c, FeaturePrecomp) for c in cfgs),
    )


@pytest.mark.parametrize(("surface", "resolve"), [("single_galaxy", _single), ("batch", _batch)])
def test_a_build_time_wave_precomp_still_gets_the_feature_topup(surface, resolve):
    """The defect, on both surfaces, with its own control in the same test.

    The no-approx arm is asserted first: it is what #1596 fixed, and if it ever
    stopped resolving both LUTs the comparison below would pass for the wrong
    reason.
    """
    no_approx = _counts(resolve(_StubModel(build_time_wave=False)))
    assert no_approx == (1, 1), (
        f"the {surface} no-approx arm resolved {no_approx} (wave, feature) rather than "
        "(1, 1); #1596's fix is the control this comparison rests on, so fix that "
        "first — the assertion below cannot mean anything without it"
    )

    built = _counts(resolve(_StubModel(build_time_wave=True)))
    assert built[1] == 1, (
        f"the {surface} surface resolved {built} (wave, feature) for a model built with "
        "approx=WavePrecomp(). Naming the wave LUT explicitly cost the feature LUT — "
        "22x in likelihood-gradient FLOPs on a 10-parameter Cue model, and slower than "
        "passing nothing at all."
    )
    assert built[0] == 1, (
        f"the {surface} surface resolved {built[0]} WavePrecomp configs; the build-time "
        "one must be carried over, not re-appended"
    )


@pytest.mark.parametrize(("surface", "resolve"), [("single_galaxy", _single), ("batch", _batch)])
@pytest.mark.parametrize("channel", ["line_ratios", "spectral_indices"])
def test_a_line_adjacent_channel_still_blocks_the_topup(surface, resolve, channel):
    """#1659's exclusion must survive the new branch.

    The ratio term reads the backend's DISCRETE line catalog, which the
    feature-LUT path does not publish. Widening the top-up to the build-time
    spelling must not widen it past that limit — the top-up now fires from two
    branches, so the exclusion has to hold on both.
    """
    obs = _StubObs()
    setattr(obs, channel, object())
    cfgs = resolve(_StubModel(build_time_wave=True, observation=obs))
    assert not any(isinstance(c, FeaturePrecomp) for c in cfgs), (
        f"the {surface} top-up fired for a model carrying {channel}, which needs the "
        "discrete catalog the LUT path does not publish"
    )


@pytest.mark.parametrize(("surface", "resolve"), [("single_galaxy", _single), ("batch", _batch)])
def test_a_backend_that_cannot_tabulate_keeps_its_wave_lut_and_stays_silent(surface, resolve):
    """Failure is the expected case here, not an anomaly.

    A backend with nothing to tabulate raises, and that raise IS the detection.
    It must cost nothing — the wave LUT survives — and it must not warn: the
    line-channel warning names a 21x cost that does not apply to a
    photometry-only fit, and unactionable advice reads as a defect in the
    caller's model.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfgs = resolve(_StubModel(features_tabulate=False, build_time_wave=True))

    assert _counts(cfgs) == (1, 0), (
        f"the {surface} surface resolved {_counts(cfgs)} (wave, feature) for a backend "
        "that cannot tabulate; it must fall back to the wave LUT, never to the raw model"
    )
    messages = [str(w.message) for w in caught]
    assert not any("look-up table" in m or "21x" in m for m in messages), (
        f"the {surface} surface warned about a feature LUT on a photometry-only fit "
        f"whose backend simply has nothing to tabulate: {messages}"
    )


#: One row per ground on which the top-up may be granted or refused, all on a
#: model that already carries a build-time ``approx=`` — the branch both
#: surfaces share, and the one #1683 lived on.
_ELIGIBILITY_MATRIX = {
    "nothing_in_the_way": {},
    "already_topped_up": {"build_time_feature": True},
    "backend_cannot_tabulate": {"features_tabulate": False},
    "line_ratios": {"channel": "line_ratios"},
    "spectral_indices": {"channel": "spectral_indices"},
    "line_ratios_and_cannot_tabulate": {"channel": "line_ratios", "features_tabulate": False},
}


def _eligibility_model(*, channel=None, **kwargs):
    obs = _StubObs()
    if channel is not None:
        setattr(obs, channel, object())
    return _StubModel(build_time_wave=True, observation=obs, **kwargs)


def test_the_two_surfaces_cannot_disagree_about_feature_lut_eligibility():
    """The rule #1683 broke, pinned as a rule rather than as its two instances.

    #1683 was not "a missing top-up" so much as *two resolvers holding
    separate copies of one policy* — the batch surface returned above the
    top-up, the single-galaxy surface returned after granting it to line fits
    only, and neither knew what the other did. #1691 was the same shape one
    level down. Pinning only the two rows that were wrong would let the third
    disagreement ship exactly as those two did, so this walks every ground on
    which the top-up can be granted or refused and requires the surfaces to
    reach the same verdict on each.

    A fresh model per surface: the resolvers memoize their clones per model
    object, so sharing one would let the first call answer for the second.

    The spread assertion is the control. "Both surfaces agree" is also what a
    pair of resolvers that never top anything up would report, so the matrix
    has to be shown non-degenerate — some rows granting, some refusing —
    before the agreement means anything.
    """
    verdicts = {
        name: (
            _counts(_single(_eligibility_model(**kw))),
            _counts(_batch(_eligibility_model(**kw))),
        )
        for name, kw in _ELIGIBILITY_MATRIX.items()
    }

    disagreed = {n: v for n, v in verdicts.items() if v[0] != v[1]}
    assert not disagreed, (
        "the single-galaxy and batch resolvers reached different (wave, feature) "
        f"verdicts on {sorted(disagreed)}: {disagreed}. Both read one policy "
        "(_refuses_feature_precomp); a split here means one surface grew a rule the "
        "other does not have, which is how #1683 and #1691 each shipped"
    )

    granted = {n for n, v in verdicts.items() if v[0][1] == 1}
    refused = {n for n, v in verdicts.items() if v[0][1] == 0}
    assert granted and refused, (
        f"the eligibility matrix is degenerate — granted={sorted(granted)}, "
        f"refused={sorted(refused)}. With every row falling the same way the "
        "agreement above holds vacuously and pins nothing"
    )
