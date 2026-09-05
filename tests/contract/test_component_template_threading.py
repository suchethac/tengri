# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the general ``TemplateThreading`` seam, shared by every component.

The AGN and dust-IR guards each pin one subsystem. This one pins the
*mechanism* they now share: any component that declares
``accepts_threaded_templates`` gets its library published by
``SEDModel._template_data_for_jit`` and handed to ``predict`` as a traced
argument, so it appears in the compiled graph as an XLA ``Parameter`` rather
than a ``Constant``.

Measured before the seam generalized (#1694), against a 0.05 MB bare-stellar
floor:

===================  =========  ==============
case                 exact      WavePrecomp
===================  =========  ==============
``shock=mappings``   3.61 MB    4.76 MB
``neb=cb19``         0.70 MB    1.85 MB
===================  =========  ==============

``shock`` baked because the publisher walked ``EmissionComponent`` only, so a
component outside the dust subsystem could not thread at all. ``cb19`` baked
for a subtler reason: its grid *was* published and *was* passed, but
``predict_nebular_line_luminosities`` ended its signature with ``**_kwargs``
and discarded it, while ``predict_nebular_sed`` — the other door into the same
method — never forwarded it. Every layer looked wired.
"""

from __future__ import annotations

import pathlib
import warnings

import jax
import numpy as np
import pytest

from tengri import DEFAULT, SEDModel, WavePrecomp
from tengri.components.sed_model_component import _REGISTRY
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.components.template_threading import TemplateThreading
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed

from ._jaxpr_consts import baked_mb

pytestmark = pytest.mark.contract

_SSP_DIR = pathlib.Path("data")

#: Exact-path budget. The bare-stellar floor is ~0.05 MB; the smallest offender
#: this file exists for (cb19, 0.70 MB) is comfortably above it.
_BAKED_BUDGET_MB = 0.5

#: WavePrecomp carries its own ~1.20 MB floor of padded filter curves, which is
#: unrelated to component libraries and is asserted separately by the control.
_PRECOMP_BUDGET_MB = 1.5


@pytest.fixture(scope="module")
def ssp():
    candidates = sorted(_SSP_DIR.glob("ssp_*.h5"))
    if not candidates:
        pytest.skip(f"no SSP grid available under {_SSP_DIR.resolve()}")
    return load_ssp_data(str(candidates[0].resolve()))


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"]))


def _build(ssp, obs, groups, approx=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
            approx=approx,
            **groups,
        )


def _traced_baked_mb(model):
    """Megabytes of constants baked into the traced forward graph."""
    from tengri.inference._model_cache import _default_owner

    model._get_or_build_predict_observables_jit()
    impl = _default_owner.get_structural_kernel(model.compile_signature())[
        "predict_observables_impl"
    ]
    params = model.spec.sample(jax.random.PRNGKey(0))
    return baked_mb(
        jax.make_jaxpr(impl)(
            params,
            model.spec.get_fixed_values(),
            model.ssp_data,
            model._template_data_for_jit(),
            model._ztable_data_for_jit(),
        )
    )


def _rest_sed(model):
    params = model.spec.sample(jax.random.PRNGKey(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(params).rest_sed())


# ``shock_frac`` defaults to 0, so a default-configured shock component adds
# EXACTLY nothing to the SED (measured: 0 pixels changed against a bare model).
# Every case below therefore pins a normalization that makes the component live;
# ``test_cases_are_live`` asserts that it stayed that way.
_CASES = {
    "shock_frac": {"shock": {"type": "mappings", "frac": Fixed(0.5)}},
    "shock_lhalpha": {
        "shock": {"type": "mappings", "norm": "lhalpha", "log_lhalpha": Fixed(41.0)}
    },
    "cb19": {"neb": {"type": "cb19", "all_params": Fixed(DEFAULT)}},
}


def _case_params():
    return [pytest.param(groups, id=name) for name, groups in _CASES.items()]


# ── The rule ──────────────────────────────────────────────────────


def _opted_in():
    """Registered components that declare the threading flag."""
    return [
        (name, cls)
        for name, cls in _REGISTRY.items()
        if getattr(cls, "accepts_threaded_templates", False)
    ]


def test_every_registered_component_exposes_the_threading_seam():
    """The registry is homogeneous with respect to threading.

    ``_REGISTRY`` carries two families by design: :class:`SEDModelComponent`
    subclasses that write a ``predict``, and eight bare-Protocol components
    registered by hand (``_REGISTRY["wg00"] = WG00AttenuationSEDComponent``)
    that own ``apply`` because their shape does not fit ``predict``. That split
    is deliberate (ADR-0009/0011) and is not what this asserts.

    What it asserts is that the split does not leak into *threading*. While the
    seam lived on ``SEDModelComponent`` alone, a bare-Protocol author who
    copied a sibling's ``accepts_threaded_templates = True`` got an
    ``AttributeError`` from the publisher: the flag was readable through
    ``getattr`` but ``templates_for_threading`` did not exist. The flag was
    answerable and the method was not — the worst shape for a seam to have.
    """
    orphans = [name for name, cls in _REGISTRY.items() if not issubclass(cls, TemplateThreading)]
    assert not orphans, (
        f"components {orphans} are registered but do not inherit "
        f"TemplateThreading, so they cannot opt into template threading and "
        f"setting the flag on them raises AttributeError in the publisher. "
        f"Add TemplateThreading as a base class."
    )


def test_every_opted_in_component_can_resolve_a_bundle():
    """A component that declares the flag must be able to produce a bundle.

    Pins the rule rather than today's instances: a component added later that
    sets ``accepts_threaded_templates`` without a working ``load`` would
    silently fall back to its in-``predict`` load and bake, and no case-based
    test would notice because no case would name it.
    """
    opted_in = _opted_in()
    assert opted_in, (
        "no registered component declares accepts_threaded_templates — the "
        "threading seam has been disconnected"
    )

    unresolvable = []
    for name, cls in opted_in:
        try:
            component = cls()
        except Exception:
            # Components needing constructor arguments are exercised by the
            # case tests below; this rule check covers the default-constructible
            # majority.
            continue
        if component.templates_for_threading() is None:
            unresolvable.append(name)

    assert not unresolvable, (
        f"components {unresolvable} declare accepts_threaded_templates but "
        f"templates_for_threading() returns None, so nothing is published and "
        f"they fall back to a trace-time load. Either implement load() or drop "
        f"the flag."
    )


def test_threading_namespaces_do_not_collide():
    """Two components sharing a namespace must also share the name-keyed layout.

    ``result["nebular"]`` holds a bare grid, not a name-keyed dict. A component
    declaring ``template_namespace = "nebular"`` would corrupt it, so the
    publisher raises — this pins that no shipped component does.
    """
    legacy_bare_namespaces = {"nebular"}
    offenders = [
        name
        for name, cls in _opted_in()
        if (getattr(cls, "template_namespace", "") or name) in legacy_bare_namespaces
    ]
    assert not offenders, (
        f"components {offenders} declare a template_namespace that already "
        f"holds a bare bundle rather than a name-keyed dict"
    )


# ── The cases ─────────────────────────────────────────────────────


def test_baseline_bakes_almost_nothing(ssp, obs):
    """Control: the floor the budgets below are measured against."""
    baked = _traced_baked_mb(_build(ssp, obs, {}))
    assert baked < _BAKED_BUDGET_MB, f"bare-stellar baseline already bakes {baked:.2f} MB"


@pytest.mark.parametrize("groups", _case_params())
def test_template_threads_on_the_exact_path(ssp, obs, groups):
    """No opted-in component may bake its library on the exact wave-grid path."""
    baked = _traced_baked_mb(_build(ssp, obs, groups))
    assert baked < _BAKED_BUDGET_MB, (
        f"{groups} bakes {baked:.2f} MB into the traced graph (budget "
        f"{_BAKED_BUDGET_MB} MB). The library must reach the component as a "
        f"traced argument via template_data, not from a module-level cache "
        f"read at trace time."
    )


@pytest.mark.parametrize("groups", _case_params())
def test_template_threads_under_waveprecomp(ssp, obs, groups):
    """...and on the LUT path, which is what an actual fit compiles.

    ``approx="auto"`` resolves to :class:`WavePrecomp` for every photometry
    fit, and ``ShockNebular`` overrides ``apply`` to take that branch over. A
    fix applied only to the inherited orchestration would leave the constant in
    place exactly where it is paid most: measured 4.76 MB here against 3.61 on
    the exact path.
    """
    baked = _traced_baked_mb(_build(ssp, obs, groups, approx=WavePrecomp()))
    assert baked < _PRECOMP_BUDGET_MB, (
        f"{groups} bakes {baked:.2f} MB under WavePrecomp (budget "
        f"{_PRECOMP_BUDGET_MB} MB, of which ~1.20 MB is the padded filter "
        f"curves the baseline also carries)."
    )


@pytest.mark.parametrize("groups", _case_params())
def test_cases_are_live(ssp, obs, groups):
    """Each case must change the SED, or the equivalence test below is vacuous.

    Guarding this explicitly because the obvious spelling of the shock case —
    ``{'type': 'mappings', 'all_params': FIXED}`` — contributes *nothing*:
    ``shock_frac`` defaults to 0. Its threading test still measured a real
    3.61 MB constant, so the perf assertion passed while the equivalence
    assertion compared a component against itself doing nothing.
    """
    bare = _rest_sed(_build(ssp, obs, {}))
    with_component = _rest_sed(_build(ssp, obs, groups))
    assert with_component.shape == bare.shape
    n_changed = int(np.sum(np.abs(with_component - bare) > 0))
    assert n_changed > 0, (
        f"{groups} leaves the SED bit-identical to a bare model, so it "
        f"contributes nothing and every other assertion about it is vacuous"
    )


@pytest.mark.parametrize("groups", _case_params())
def test_threading_is_numerically_inert(ssp, obs, groups):
    """Threading moves data, never physics — bit-identical, rtol=0, atol=0.

    The control neuters the publisher, which is exactly the pre-threading
    behavior: every backend falls back to its own module-level load.
    """
    control = _build(ssp, obs, groups)
    control._template_data_for_jit = lambda: None
    expected = _rest_sed(control)

    actual = _rest_sed(_build(ssp, obs, groups))

    assert np.array_equal(expected, actual), (
        f"threading changed the SED for {groups}: max relative deviation "
        f"{np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-300)):.3e}. "
        f"It must move the array from a Constant to a Parameter and nothing else."
    )
