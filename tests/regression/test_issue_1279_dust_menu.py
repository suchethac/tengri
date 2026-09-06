# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1279: advertised dust emission types must *run*.

#1279 reported ``dh02_ce01`` (Dale & Helou 2002 + Chary & Elbaz 2001 cold dust)
as advertised in the build grammar but unusable. It was closed as
non-reproducing, and this file was the standing guard that agreed:

    Measurement shows it builds successfully via the lazy
    `DUST_EMISSION_MODELS` path even though absent from `_REGISTRY`, so the
    reported issue does not reproduce.

The report was right and the guard was wrong. Its last two lines were::

    # Actually call predict to trigger component chain building
    model.predict(params)

**That comment is false, and it is the whole defect.** ``model.predict`` is
lazy: it returns a prediction handle and resolves no components. Nothing is
computed until an observable accessor is called. So the sweep built 19 models
and evaluated none of them, and "every advertised dust emission type builds"
was true of a menu entry that cannot produce a number.

Both real channels raise for ``dh02_ce01``::

    >>> model.predict(params).photometry()      # exploration
    ValueError: dust_emission component 'dh02_ce01' (resolved from grammar
    type 'dh02_ce01') not found in registry.
    >>> model.predict_photometry(params)        # inference hot path
    ValueError: ...

The type was declared in ``_LAZY_DUST_EMISSION_TYPES`` — accepted by the
grammar and advertised by ``list_dust_emission_models()`` — but had no
``SEDModelComponent``, so it resolved only on the legacy
``DUST_EMISSION_MODELS`` path that no public prediction surface reaches.
``src/tengri/components/dust/emission/templates/dh02_ce01.py`` supplies the
missing component.

The sweep below therefore *evaluates* every advertised type, on both public
channels, and requires finite output — a NaN return is not a pass either.

The infrared bands in the fixture are a strengthening, **not** the bug: the
choice of dust emission model measurably changes the photometry even over the
old optical-only baseline (18 of 19 types differ from each other there), so
that fixture was live. A dust *emission* sweep that never looks at the infrared
is simply weaker than it should be.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def infrared_observation():
    """Three optical bands plus mid- and far-IR.

    A dust *emission* sweep should look where dust emits. This is a
    strengthening of the old optical-only fixture, not a fix for it — see the
    module docstring.
    """
    from tengri import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    spans = [
        (3500.0, 4500.0),
        (5000.0, 6500.0),
        (7500.0, 9000.0),
        (1.0e5, 1.4e5),  # ~10-14 um, mid-IR
        (8.0e5, 1.2e6),  # ~80-120 um, far-IR: where cold dust emits
    ]
    curves = tuple(
        FilterCurve(wave=jnp.linspace(lo, hi, 50), trans=jnp.ones(50) * 0.5, name=f"band_{i}")
        for i, (lo, hi) in enumerate(spans)
    )
    return Observation(photometry=Photometry(filters=curves))


def _advertised_types():
    """Every dust emission type that is a *model*, i.e. selectable on its own.

    The grammar accepts a second, smaller class — building blocks such as
    ``pah_drude``, which emit a real piece of the IR SED but never renormalize
    to ``L_ir`` — and ``SEDModel.build`` refuses those outright. Sweeping them
    here would assert the opposite of the contract; :func:`_building_block_types`
    and the test below check *their* contract instead, so neither set is
    unchecked.
    """
    from tengri.parameters.groups import _standalone_dust_emission_types

    return sorted(_standalone_dust_emission_types())


def _building_block_types():
    from tengri.parameters.groups import (
        _standalone_dust_emission_types,
        _valid_dust_emission_types,
    )

    return sorted(_valid_dust_emission_types() - _standalone_dust_emission_types())


@pytest.mark.parametrize("name", _advertised_types())
def test_every_advertised_dust_emission_type_runs_on_both_channels(
    name, synthetic_ssp, infrared_observation
):
    """Advertised, therefore usable — on the exploration *and* inference paths."""
    from tengri import DEFAULT, SEDModel

    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=infrared_observation,
        sfh={"type": "dpl"},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": name, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    params = {k: jnp.array(0.0) for k in model.spec.free_params}

    rich = model.predict(params)
    lean = np.asarray(model.predict_photometry(params))

    assert np.all(np.isfinite(lean)), (
        f"dust emission {name!r}: predict_photometry returned non-finite values "
        f"{lean}. A NaN is not a pass — an unguarded smoke test reads it as one."
    )
    assert np.all(np.isfinite(np.asarray(rich.photometry()))), (
        f"dust emission {name!r}: predict().photometry() returned non-finite values."
    )


def test_the_sweep_reaches_the_infrared(infrared_observation):
    """Assert the coverage rather than trusting the fixture's name."""
    reddest = max(float(jnp.max(f.wave)) for f in infrared_observation.photometry.filters)
    assert reddest > 1.0e5, (
        f"reddest band ends at {reddest:.3g} A, so this sweep of dust "
        f"*emission* models never looks at the wavelengths they exist to model."
    )


def test_predict_alone_would_compute_nothing(synthetic_ssp, infrared_observation):
    """Pin the reason the old guard passed, so it cannot be reintroduced.

    ``model.predict(params)`` is lazy. A sweep that stops there evaluates no
    model, which is exactly how an advertised-but-undispatched type went
    unnoticed. If ``predict`` ever became eager this test would fail, and the
    sweep above could then be simplified — but until it does, calling an
    accessor is load-bearing, not decoration.
    """
    from tengri import DEFAULT, SEDModel

    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=infrared_observation,
        sfh={"type": "dpl"},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    params = {k: jnp.array(0.0) for k in model.spec.free_params}
    handle = model.predict(params)
    assert not hasattr(handle, "__array__"), (
        "model.predict now returns something array-like; if it became eager, "
        "revisit the comment this test exists to correct."
    )
    assert np.all(np.isfinite(np.asarray(handle.photometry()))), (
        "the accessor, not predict() itself, is what runs the forward model"
    )


@pytest.mark.parametrize("name", _building_block_types())
def test_every_building_block_type_is_refused_with_its_reason(
    name, synthetic_ssp, infrared_observation
):
    """The other half of the sweep: a non-model type must refuse, and say why.

    Without this, narrowing :func:`_advertised_types` to the standalone set
    would be a silent shrinking of coverage — the excluded name would simply
    stop being tested and nothing would notice if it started building a model
    that discards 99.98% of the absorbed energy.
    """
    from tengri import DEFAULT, SEDModel
    from tengri.config.exceptions import ParameterError

    with pytest.raises(ParameterError, match="building block"):
        SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=infrared_observation,
            sfh={"type": "dpl"},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": name, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )


def test_the_sweep_is_not_empty():
    """If the menu derivation ever returned nothing, the sweep reports green."""
    assert len(_advertised_types()) >= 10
    # ...and the refusal sweep above is parametrized off a set that must not be
    # empty either, or it collects nothing and reports green having run zero
    # cases (the ``SKIPPED [1] got empty parameter set`` shape).
    assert len(_building_block_types()) >= 1
