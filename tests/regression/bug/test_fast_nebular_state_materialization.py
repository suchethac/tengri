# SPDX-License-Identifier: BSD-3-Clause
r"""The nebular grid's photometry shortcut must not reach the forward state.

``NebularSEDComponent.apply`` zeroes ``sed_nebular`` when the per-Q_H grid may
serve photometry, because the saving *is* skipping the Cue forward. Whether it
may is decided by ``components_consuming(chain, "sed_nebular")`` — a census of
the ADR-0009 component contract, so a dust law that declares the input disables
the shortcut and keeps its energy balance (b0a274806).

The census sees the component contract, and only that. Every reader that takes
a published key off ``state.derived`` *without* declaring an input is invisible
to it, and the forward state has several: ``state_to_sed_components`` (behind
both ``Posterior.sed_components`` and ``Prediction.sed.components``) and the
accumulated ``state.sed_intrinsic`` that ``pred.rest_sed()`` returns.

Measured on a dust-free Cue model at ``(WavePrecomp(), FeaturePrecomp())`` —
the config #1596 auto-resolves for exactly this kind of fit:

* ``pred.rest_sed()`` was wrong by **97.19 %** of its peak,
* ``sed_nebular`` was **exactly 0.0**, and ``sed_total`` short by the same
  97.19 % — the two agreeing to seven digits *because* the missing term is
  precisely ``sed_nebular``,
* while ``predict_photometry`` stayed right to 5.4e-08. The photometry channel
  is served from the grid, so only the state was ever wrong.

The fix is a second chain: ``predict_state`` materializes by default, and the
one caller that may take the shortcut — the JIT observables kernel, which reads
the projected observables and never the SED — opts in explicitly. So the
default is correct and the optimization is the thing that must be asked for.

Assertions are bit-exact. The dust-emitting arm already measured ``0.0``
between the two paths before this fix, so equality is the achievable bar and
there is no tolerance to widen later.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.components.nebular.component import NebularSEDComponent
from tengri.forward.component_factory import state_to_sed_components
from tengri.forward.sed_model import FeaturePrecomp, WavePrecomp

pytestmark = pytest.mark.regression_bug

_BASE = dict(
    sfh={
        "type": "delayed",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Uniform(9.0, 11.0),
        "tau_gyr": 1.0,
        "age_gyr": 5.0,
    },
    redshift=Fixed(0.1),
)

_CUE = {"type": "cue", "all_params": Fixed(DEFAULT)}

_DUST_EMIT = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_diff": Uniform(0.0, 1.5),
    "tau_bc": 0.0,
}

#: A peer group now. Only the ``dust_emitting`` row gets it -- the ``dust_free_*``
#: rows must stay emission-free, since that is what enables the shortcut they test.
_DUST_EMISSION = {"type": "dale2014", "all_params": Fixed(DEFAULT)}

#: ``dust_free_*`` are the exposed rows: with nothing declaring ``sed_nebular``
#: the shortcut is (correctly) enabled, which is what makes the state wrong.
#: ``dust_emitting`` is the control — b0a274806 already disables the shortcut
#: there, and it must stay bit-exact so a regression is distinguishable from
#: this defect.
_MODELS = {
    "dust_free_cue": dict(dust_attenuation={"type": "none"}, neb=_CUE),
    "dust_free_cue_shock": dict(dust_attenuation={"type": "none"}, neb=_CUE, shock={"frac": 0.1}),
    "dust_emitting_cue": dict(dust_attenuation=_DUST_EMIT, dust_emission=_DUST_EMISSION, neb=_CUE),
}

_BANDS = ["sdss_g", "sdss_r", "wise_w1", "herschel_250"]

_FAST = (WavePrecomp(), FeaturePrecomp())


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(_BANDS))


def _build(ssp, obs, groups, approx):
    kw = dict(ssp_data=ssp, observation=obs, **_BASE, **groups)
    if approx is not None:
        kw["approx"] = approx
    return SEDModel.build(**kw)


def _at_prior_center(model):
    return {
        n: float(model.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
        for n in model.spec.free_params
    }


def _full_params(model):
    return {**model.spec.get_fixed_values(), **_at_prior_center(model)}


def _nebular(chain):
    return [c for c in chain if isinstance(c, NebularSEDComponent)]


def _decomposition_and_sed(ssp, obs, groups, approx):
    """The two state-derived surfaces, via the paths users actually reach them by.

    ``state_to_sed_components(model.predict_state(...))`` is literally the body
    of ``Posterior.sed_components``; ``pred.rest_sed()`` is the accumulated
    ``state.sed_intrinsic``.
    """
    model = _build(ssp, obs, groups, approx)
    params = _full_params(model)
    comp = state_to_sed_components(model.predict_state(params))
    comp = {k: np.asarray(v, dtype=np.float64) for k, v in comp.items()}
    sed = np.asarray(model.predict(params).rest_sed(), dtype=np.float64)
    return comp, sed


def _rel(a, b):
    """Relative deviation, for the failure message only -- never the assertion."""
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.abs(b).max()
        return float(np.abs(a - b).max() / denom) if denom > 0 else float("nan")


@pytest.mark.parametrize("composition", sorted(_MODELS))
def test_the_fast_grid_returns_the_same_forward_state_as_the_exact_path(
    ssp_data_fsps, obs, composition
):
    """Every state-derived surface must be identical with and without the LUT.

    Stated over the whole decomposition rather than over ``sed_nebular`` alone:
    the mechanism is "a published key nobody *declares* may be dropped", and
    ``sed_shock`` is exposed the same way. Asking about every key means a second
    component acquiring the same shortcut is caught by this test as written.
    """
    groups = _MODELS[composition]
    with jax.enable_x64(True):
        exact_comp, exact_sed = _decomposition_and_sed(ssp_data_fsps, obs, groups, None)
        fast_comp, fast_sed = _decomposition_and_sed(ssp_data_fsps, obs, groups, _FAST)

    for key in sorted(exact_comp):
        assert np.array_equal(fast_comp[key], exact_comp[key]), (
            f"{key} differs by {_rel(fast_comp[key], exact_comp[key]):.4e} between the "
            f"fast and exact paths on the {composition} model. The precompute is a "
            "speed knob for the photometry channel; the forward state it hands back "
            "must not depend on it."
        )

    assert np.array_equal(fast_sed, exact_sed), (
        f"pred.rest_sed() differs by {_rel(fast_sed, exact_sed):.4e} on the "
        f"{composition} model. The SED accessor reads state.sed_intrinsic, which "
        "accumulates sed_nebular — zeroing it to serve photometry from the grid "
        "silently removes the nebular continuum from the published SED."
    )


def test_the_dust_free_fixture_really_does_take_the_shortcut(ssp_data_fsps, obs):
    """Guard the guard: without the shortcut enabled the test above is vacuous.

    The defect needs *both* halves — a grid carrying a photometry table, and no
    declared ``sed_nebular`` consumer to veto its use. A fixture that lost
    either would pass the equality test while proving nothing.
    """
    model = _build(ssp_data_fsps, obs, _MODELS["dust_free_cue"], _FAST)
    neb = _nebular(model._cached_component_chain)
    assert neb, "no nebular component in the chain — fixture no longer tests anything"
    table = neb[0].grid_table
    assert table is not None and getattr(table, "log_phot_per_qh", None) is not None, (
        "the per-Q_H grid carries no photometry table, so apply() never takes the "
        "shortcut branch and the equality test above cannot detect its absence"
    )
    assert neb[0].must_materialize_sed is False, (
        "the observables chain no longer takes the shortcut, so the equality test "
        "above is satisfied trivially rather than by materializing on demand"
    )


def test_a_correct_state_is_not_bought_by_disabling_the_shortcut(ssp_data_fsps, obs):
    """The optimization must survive the fix, on the path that has it.

    Correctness bought by switching the fast path off everywhere trades a silent
    physics error for a silent ~4x performance one (#1596). The two chains carry
    opposite flags: the observables kernel keeps the shortcut, the forward state
    materializes.
    """
    model = _build(ssp_data_fsps, obs, _MODELS["dust_free_cue"], _FAST)
    observables_chain = model._cached_component_chain
    state_chain = model._full_state_chain()

    assert _nebular(observables_chain)[0].must_materialize_sed is False, (
        "the JIT observables kernel lost the per-Q_H photometry shortcut; #1596 "
        "measured that cost at ~4x on a photometry-only Cue fit"
    )
    assert _nebular(state_chain)[0].must_materialize_sed is True, (
        "the forward-state chain still skips the Cue continuum forward, so "
        "predict_state hands back a zeroed sed_nebular"
    )


def test_materialization_is_asked_of_every_component_not_hard_coded_for_nebular(
    ssp_data_fsps, obs
):
    """The hook is a contract, not a special case.

    ``materialized_chain`` asks each component for a publishing variant and
    leaves the rest untouched by identity. Pinned so the next component that
    acquires a skip-optimization implements one method instead of editing a
    branch in ``predict_state`` — the shape that let this defect exist.
    """
    from tengri.forward.orchestrator import materialized_chain

    model = _build(ssp_data_fsps, obs, _MODELS["dust_free_cue"], _FAST)
    chain = model._cached_component_chain
    full = materialized_chain(chain)

    assert len(full) == len(chain), "materialized_chain must preserve chain length and order"
    for before, after in zip(chain, full, strict=True):
        if isinstance(before, NebularSEDComponent):
            assert after is not before and after.must_materialize_sed is True, (
                "the nebular component must return a publishing variant"
            )
        else:
            assert after is before, (
                f"{type(before).__name__} has no publication shortcut, so "
                "materialized_chain must return it unchanged rather than "
                "reconstructing it"
            )
