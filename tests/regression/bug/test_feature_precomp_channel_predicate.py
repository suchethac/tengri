# SPDX-License-Identifier: BSD-3-Clause
"""The LUT-attachment predicate must count every feature channel.

``Fitter._fits_lines`` gates whether ``FeaturePrecomp`` is attached, and counted
only the eline nuisance amplitudes and a measured line-flux channel. It omitted
**line ratios** and **spectral indices**, which are served from
``predict_state`` because they need the discrete line catalog the LUT does not
publish. A photometry + line-ratio fit therefore answered "no lines", was
classed photometry-only by the #1596 top-up, was handed the LUT, and then asked
``predict_line_ratios`` for a catalog that path does not produce.

Second occurrence of the same defect: ``_fits_lines``'s own docstring records
the predicate being found incomplete in 2026-07, when a measured line-flux
channel "silently stayed on the exact path at ~21x the per-gradient cost". That
fix added a case. Adding a case leaves the next omission just as available, and
here it could not even work -- ``_fits_lines`` is asked two *opposite*
questions (``_auto_approx_config`` appends the LUT when it is true, the #1596
top-up attaches it when it is false), so widening it would have fixed one call
site by breaking the other. Hence a separate predicate,
``_needs_full_forward_state``, gating both.

These are decision-table tests over the real ``Observation`` attributes, not
stubs: the failure mode is a predicate reading the wrong attribute, which a stub
defines away.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.forward.sed_model import FeaturePrecomp
from tengri.inference.fitter import Fitter
from tengri.observation.line_ratio_data import LineRatioData
from tengri.observation.spectral_indices import STANDARD_INDICES, SpectralIndexData

pytestmark = pytest.mark.regression_bug


class _Obs:
    """Only the attributes the predicates read, with the real objects in them."""

    def __init__(self, *, line_ratios=None, spectral_indices=None, line_fluxes=None):
        self.line_ratios = line_ratios
        self.spectral_indices = spectral_indices
        self.line_fluxes = line_fluxes
        self.data_type = "photometry"


class _Model:
    def __init__(self, observation):
        self.observation = observation


def _ratios():
    return LineRatioData.from_dict({("Halpha", "Hbeta"): (4.2, 0.3)})


def _indices():
    defs = tuple(STANDARD_INDICES[k] for k in list(STANDARD_INDICES)[:1])
    return SpectralIndexData(index_defs=defs, values=jnp.array([1.4]), errors=jnp.array([0.02]))


@pytest.mark.parametrize(
    ("label", "kwargs", "expected"),
    [
        ("photometry_only", {}, False),
        ("line_ratios", {"line_ratios": _ratios}, True),
        ("spectral_indices", {"spectral_indices": _indices}, True),
        ("both", {"line_ratios": _ratios, "spectral_indices": _indices}, True),
    ],
)
def test_needs_full_forward_state_counts_ratios_and_indices(label, kwargs, expected):
    """Every channel served from ``predict_state`` must be reported."""
    obs = _Obs(**{k: v() for k, v in kwargs.items()})
    got = Fitter._needs_full_forward_state(_Model(obs))
    assert got is expected, (
        f"_needs_full_forward_state said {got} for the {label} configuration. "
        "Line ratios and spectral indices need the discrete line catalog, which "
        "the emission-line LUT does not publish, so a fit carrying either must "
        "never be handed FeaturePrecomp."
    )


def test_the_predicate_reads_the_attributes_the_data_args_are_built_from():
    """Pin the attribute names, not just the behaviour.

    ``_build_data_args`` publishes ``line_ratio_obs`` / ``index_obs`` from
    ``observation.line_ratios`` / ``observation.spectral_indices``, and
    ``build_loss_fn`` derives ``needs_state`` from those keys. The predicate
    here must read the same two attributes or the two can disagree silently --
    which is the whole failure this file exists for. A renamed attribute makes
    ``getattr(..., None)`` return ``None`` forever, so the predicate would go
    quietly and permanently False.
    """
    from tengri.observation.observation import Observation

    assert hasattr(Observation, "__init__")
    fields = Observation.__init__.__code__.co_varnames
    for name in ("line_ratios", "spectral_indices"):
        assert name in fields, (
            f"Observation no longer takes {name!r}; Fitter._needs_full_forward_state "
            "reads it via getattr and would silently report False for every fit."
        )


def test_a_ratio_fit_is_not_classified_photometry_only():
    """The #1596 top-up must not claim a ratio fit as photometry-only.

    This is the defect end-to-end at the policy layer: the top-up fires on
    ``data_type == 'photometry' and not _fits_lines(model)``, and a ratio fit
    satisfied both halves.
    """
    model = _Model(_Obs(line_ratios=_ratios()))
    assert not Fitter._fits_lines(Fitter.__new__(Fitter), model), (
        "fixture no longer reproduces the misclassification: _fits_lines already "
        "counts ratios, so the guard below proves nothing"
    )
    assert Fitter._needs_full_forward_state(model), (
        "a line-ratio fit must be excluded from the photometry-only feature-LUT "
        "top-up by _needs_full_forward_state, since _fits_lines does not see it"
    )


def test_a_line_flux_plus_ratio_fit_is_also_excluded():
    """The mixed case, which the obvious fix does not reach.

    A fit carrying line fluxes *and* ratios answers ``_fits_lines`` true, so it
    is not caught by the photometry-only top-up at all — it is caught by the
    other two sites, which append the LUT precisely *because* lines are fit.
    Widening ``_fits_lines`` cannot help here: it is already true. Only a
    predicate about what the LUT can serve excludes it.
    """
    fitter = Fitter.__new__(Fitter)
    fitter.data_type = "photometry"
    fitter._eline_marginalize = False
    fitter._eline_fitted = False

    obs = _Obs(line_ratios=_ratios(), line_fluxes=object())
    model = _Model(obs)
    assert Fitter._fits_lines(fitter, model), "fixture must have _fits_lines true"

    cfg = Fitter._auto_approx_config(fitter, model)
    cfgs = cfg if isinstance(cfg, tuple) else (cfg,)
    assert not any(isinstance(c, FeaturePrecomp) for c in cfgs), (
        f"auto policy resolved {cfgs!r} for a line-flux + line-ratio fit. "
        "_fits_lines is true here, so this case is reachable only through "
        "_needs_full_forward_state."
    )


def test_no_warning_advises_a_lut_that_cannot_be_used():
    """Withholding the LUT deliberately must not emit the 21x-cost advice.

    The warning tells the caller to add ``FeaturePrecomp``. For a ratio/index
    fit that is not available, so the advice would name a remedy that cannot be
    applied — and unactionable advice reads as a defect in the caller's model.
    """
    import warnings

    fitter = Fitter.__new__(Fitter)
    fitter.data_type = "photometry"
    fitter._eline_marginalize = False
    fitter._eline_fitted = False

    model = _Model(_Obs(line_ratios=_ratios(), line_fluxes=object()))
    model.approx = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Fitter._warn_lines_without_lut(fitter, model)
    messages = [str(w.message) for w in caught]
    assert not any("FeaturePrecomp" in m for m in messages), (
        f"warned about a missing FeaturePrecomp on a fit that cannot use one: {messages}"
    )


def test_auto_approx_config_withholds_the_lut_from_ratio_fits():
    """The resolved config for a ratio fit must not contain FeaturePrecomp."""
    fitter = Fitter.__new__(Fitter)
    fitter.data_type = "photometry"
    fitter._eline_marginalize = False
    fitter._eline_fitted = False

    model = _Model(_Obs(line_ratios=_ratios()))
    cfg = Fitter._auto_approx_config(fitter, model)
    cfgs = cfg if isinstance(cfg, tuple) else (cfg,)
    assert not any(isinstance(c, FeaturePrecomp) for c in cfgs), (
        f"auto policy resolved {cfgs!r} for a photometry + line-ratio fit. The "
        "LUT does not publish the discrete catalog predict_line_ratios needs."
    )
