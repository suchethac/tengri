# SPDX-License-Identifier: BSD-3-Clause
"""The derived sweep must select exactly what the hardcoded tuple selected.

Replacing ``for _emitter in ("xray", "radio")`` with a census derived from the
component chain is a refactor, so the obligation is to show it changes nothing
for models that exist today. ``radio`` and ``xray`` are the only two components
in the tree that implement ``emission_terms``, so the derived census must come
back as exactly ``["radio", "xray"]`` for a chain carrying both -- the old
tuple, sorted.

Naming the emitters is not the same as optimizing them, so the census assertion
is paired with one that the band responses were actually built. A census that
returned the right names while every response silently failed would pass the
first check and defeat the purpose of the mechanism.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.forward.sed_model import WavePrecomp, _chain_implements_emission_terms

pytestmark = pytest.mark.contract

#: The tuple this refactor replaced, sorted. Written out rather than imported:
#: the point is to pin the old behavior independently of the new code path.
HARDCODED_BEFORE = ["radio", "xray"]


def _model_with_radio_and_xray(ssp, obs):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "other_params": Fixed(DEFAULT),
        },
        # No dust_emission: #1970 refuses Dale+2014 beside SF synchrotron,
        # because the template carries its own embedded radio tail and the two
        # would double-count. Dust is not what this file is about.
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "dpl"}, "all_params": Fixed(DEFAULT)},
        xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )


def test_derived_census_equals_the_hardcoded_tuple_it_replaced(
    synthetic_ssp, synthetic_tophat_obs
):
    """The no-op proof: same emitters in, same emitters out."""
    model = _model_with_radio_and_xray(synthetic_ssp, synthetic_tophat_obs)
    chain = model._cached_component_chain
    assert chain is not None, "fixture assumption: the chain is built at construction"

    assert _chain_implements_emission_terms(chain) == HARDCODED_BEFORE


def test_the_responses_are_actually_built_not_merely_listed(synthetic_ssp, synthetic_tophat_obs):
    """A census naming the emitters proves nothing if no response was built."""
    model = _model_with_radio_and_xray(synthetic_ssp, synthetic_tophat_obs)

    for name in HARDCODED_BEFORE:
        cached = getattr(model, f"_{name}_term_response_cache", "unset")
        assert cached != "unset", (
            f"_{name}_term_response_cache was never assigned, so the derived sweep "
            f"did not reach {name} at all"
        )
        assert cached is not None, (
            f"{name} was named by the census but its band response came back None, "
            "so the emitter silently fell back to the exact per-call integral"
        )


def test_a_model_without_those_emitters_predicts_finitely(synthetic_ssp, synthetic_tophat_obs):
    """The sweep is empty here; the model must still build and predict."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
    )
    assert _chain_implements_emission_terms(model._cached_component_chain) == []

    params = {n: jnp.asarray(model.spec.param_dict[n]) for n in model.spec.free_params}
    phot = model.predict_photometry(params)
    assert jnp.all(jnp.isfinite(phot))
    assert phot.shape[0] > 0


def test_the_build_loop_actually_consults_the_derived_census(
    synthetic_ssp, synthetic_tophat_obs, monkeypatch
):
    """The wiring, not the helper.

    Every other test here calls ``_chain_implements_emission_terms`` directly, so
    all of them keep passing if the build loop is quietly reverted to the literal
    ``("xray", "radio")`` -- measured: a faithful revert leaves the whole file
    green. That revert is the one edit that undoes this change, and a suite that
    cannot see it is guarding the helper rather than the behavior.

    So replace the census with one naming an emitter that is in no chain and in
    no hardcoded tuple, and require the build to have tried it. Reaching the
    emitter is what leaves ``_phantom_term_response_cache`` behind; a loop
    iterating a literal tuple never creates that attribute.
    """
    from tengri.forward import sed_model as sed_model_module

    monkeypatch.setattr(
        sed_model_module,
        "_chain_implements_emission_terms",
        lambda chain: ["phantom"],
    )
    model = _model_with_radio_and_xray(synthetic_ssp, synthetic_tophat_obs)

    assert hasattr(model, "_phantom_term_response_cache"), (
        "the build never attempted the emitter the census named, so the loop is "
        "not reading the census"
    )
