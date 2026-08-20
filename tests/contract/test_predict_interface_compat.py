# SPDX-License-Identifier: BSD-3-Clause
"""Contract locks for the SEDModel predict-surface diet (cleanup PR-2).

Three guarantees:

1. Every deprecated ``predict_*`` shim emits DeprecationWarning AND returns
   bit-exact the same values as its replacement (shims are pass-through).
2. The promoted per-component surface ``pred.sed.components`` decomposes the
   SAME published arrays as ``state_to_sed_components`` /
   ``Posterior.sed_components`` (one shared helper).
3. Uses the synthetic wide SSP (#613) so it runs on CI without data files.
"""

from __future__ import annotations

import warnings

import chex
import jax.numpy as jnp
import pytest

from tengri import Fixed, SEDModel

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    """Small all-fixed model with dust so attenuated != intrinsic."""
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl"},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 0.3,
            "tau_diff": 0.2,
        },
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


def _no_dep_warnings():
    ctx = warnings.catch_warnings()
    warnings.simplefilter("ignore", DeprecationWarning)
    return ctx


class TestTwinShims:
    """Migration-era ``*_components`` twins: warn + bit-exact pass-through."""

    def test_photometry_twin_warns_and_matches(self, model):
        with pytest.warns(DeprecationWarning, match="predict_photometry"):
            via_shim = model.predict_photometry_components({})
        chex.assert_trees_all_close(via_shim, model._photometry_via_state({}), rtol=0)

    def test_sfh_quantities_twin_warns_and_matches(self, model):
        from tengri.forward import state_to_sfh_quantities

        with pytest.warns(DeprecationWarning, match="state_to_sfh_quantities"):
            via_shim = model.predict_sfh_quantities_components({})
        direct = state_to_sfh_quantities(model.predict_state({}))
        chex.assert_trees_all_close(
            jnp.asarray(via_shim.stellar_mass), jnp.asarray(direct.stellar_mass), rtol=0
        )

    def test_sed_quantities_twin_warns_and_matches_canonical(self, model):
        with pytest.warns(DeprecationWarning, match="predict_sed_quantities"):
            via_shim = model.predict_sed_quantities_components({})
        with _no_dep_warnings():
            canonical = model.predict_sed_quantities({})
        chex.assert_trees_all_close(
            jnp.asarray(via_shim.l_bol), jnp.asarray(canonical.l_bol), rtol=0
        )


class TestTailShims:
    """Zero-caller interactive getters: warn, behavior unchanged."""

    @pytest.mark.parametrize(
        "method",
        [
            "predict_luminosity",
            "predict_ionizing_quantities",
            "predict_radio_quantities",
            "predict_xray_quantities",
        ],
    )
    def test_tail_method_warns(self, model, method):
        with pytest.warns(DeprecationWarning, match="model.predict"):
            getattr(model, method)({})

    def test_emission_lines_warns_before_backend_error(self, model):
        """The shim warns even when the no-nebular model then raises."""
        with (
            pytest.warns(DeprecationWarning, match="model.predict"),
            pytest.raises(NotImplementedError, match="BakedIn"),
        ):
            model.predict_emission_lines({})


class TestComponentsPromotion:
    """pred.sed.components — the one per-component decomposition surface."""

    def test_components_matches_state_helper(self, model):
        from tengri.forward import state_to_sed_components

        pred = model.predict({})
        comp = pred.sed.components
        direct = state_to_sed_components(model.predict_state({}))
        assert set(comp) == set(direct)
        for key in comp:
            chex.assert_trees_all_close(comp[key], direct[key], rtol=0)

    def test_components_keys_cover_posterior_contract(self, model):
        from tengri.inference.posterior import Posterior

        comp = model.predict({}).sed.components
        assert set(Posterior._COMPONENT_KEYS) <= set(comp)
        assert "wavelength" in comp

    def test_components_shapes_and_finiteness(self, model):
        comp = model.predict({}).sed.components
        n_wave = comp["wavelength"].shape[0]
        for arr in comp.values():
            chex.assert_shape(arr, (n_wave,))
        chex.assert_tree_all_finite(comp["sed_total"])

    def test_dust_makes_attenuated_differ_from_intrinsic(self, model):
        comp = model.predict({}).sed.components
        assert not bool(jnp.allclose(comp["sed_attenuated"], comp["sed_intrinsic"]))

    def test_no_extra_forward_pass(self, model):
        """components reuses the Prediction's cached state object."""
        pred = model.predict({})
        _ = pred.sfh.stellar_mass  # populates the cache
        state_before = pred._cache["_state"]
        _ = pred.sed.components
        assert pred._cache["_state"] is state_before


class TestPredictStateIsNotAdvertisedAsPublic:
    """``predict_state`` must not contradict the prediction contract (#1736).

    Two authoritative sources disagreed. ``NAMING_CONTRACT.md`` §4b names the
    public prediction surfaces and ``predict_state`` is not among them, while
    the method's own docstring opened with "This is the public bridge ...".
    Docstrings *are* the API reference here — ``docs/api/*.rst`` are autodoc
    stubs — so the two readings were equally authoritative, and two reviewers
    auditing the published notebooks reached opposite conclusions about whether
    to teach it to beginners. The quickstart nearly shipped a paragraph doing so.

    Resolved as internal, because the pull toward it was a missing *public*
    accessor for per-component SEDs, and that accessor exists:
    ``pred.sed.components`` (covered by the class above). These tests pin the
    resolution so the wording cannot drift back.
    """

    def test_docstring_does_not_claim_to_be_public(self):
        doc = SEDModel.predict_state.__doc__ or ""
        assert "public bridge" not in doc, (
            "predict_state's docstring calls itself 'the public bridge' again. "
            "NAMING_CONTRACT §4b names the public prediction surfaces and this "
            "is not one of them; a docstring is the API reference here, so this "
            "wording is the contract disagreeing with itself (#1736)."
        )

    def test_docstring_does_not_read_as_deprecated(self):
        """Internal is not the same status as deprecated, and a substring decides.

        ``test_predict_surface_classification.py`` derives the deprecated label
        with ``"deprecat" in doc.lower()``, so prose in this docstring *is* the
        classification. The first draft of this fix wrote "may change without a
        deprecation cycle" -- accurate English, and it moved ``predict_state``
        out of ``UNSANCTIONED_PREDICT_METHODS`` into the deprecated set and took
        the contract shard red.

        The distinction is real, not bookkeeping: deprecated means scheduled for
        removal with a named successor, while ``predict_state`` has production
        callers (``predict_observables_jit``, ``ForwardModel.predict_observables``)
        and is going nowhere. It is unsanctioned, which is the bucket for "live,
        un-deprecated, outside the sanctioned three".
        """
        doc = (SEDModel.predict_state.__doc__ or "").lower()
        assert "deprecat" not in doc, (
            "predict_state's docstring contains 'deprecat', which reclassifies it "
            "as a deprecated method in test_predict_surface_classification.py and "
            "fails the contract shard. It is UNSANCTIONED, not deprecated -- it has "
            "production callers. Say 'no stability guarantee' instead."
        )

    def test_docstring_points_at_the_public_replacement(self):
        doc = SEDModel.predict_state.__doc__ or ""
        assert "components" in doc, (
            "predict_state no longer names the supported alternative. Telling a "
            "reader a surface is internal without saying what replaces it is why "
            "the notebooks reached for it: the component-decomposition figures "
            "had no other documented path. Name pred.sed.components (#1736)."
        )

    def test_contract_document_states_the_resolution(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        contract = (root / "docs" / "dev" / "NAMING_CONTRACT.md").read_text(encoding="utf-8")
        assert "predict_state" in contract, (
            "NAMING_CONTRACT.md does not mention predict_state. Silence is what "
            "let the docstring and the contract disagree for as long as they did "
            "-- a reader checking the contract found nothing either way (#1736)."
        )
        assert "pred.sed.components" in contract, (
            "NAMING_CONTRACT.md marks predict_state non-public without naming the "
            "public per-component surface, which leaves the decomposition figures "
            "with no compliant path -- the exact gap that caused #1736."
        )
