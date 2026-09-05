# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``sfh={'field_centering': a}`` reaches the field the model builds.

The math primitive shipped in #1399 — :func:`drw_partial_gp_from_zeta`,
:func:`drw_latent_log_prior`, and ``compute_field_gp(centering=...)`` — with
tests that call it *directly*. Nothing else ever passed the argument: all four
call sites (``stellar/component.py``, ``sed_model.py`` twice,
``inference/population/reconstruct.py``) took the ``centering=1.0`` default, and
no ``field_centering`` existed on any user surface. So the primitive was green
and unreachable, which is the shape #1488 names — selectable is not enabled.

That matters more here than for a physics block. #1355's whole purpose is an
A/B over ``a``: a knob that silently stays at 1.0 does not fail, it returns the
non-centered answer three times and reports a null result.

These tests pin the four guarantees the ``age_kernel`` axis established
(#1508/#1532), plus the one specific to a reparameterization:

1. the default is **bit-identical** to explicit ``a = 1``;
2. ``a < 1`` genuinely changes the field, end-to-end through
   ``predict_photometry`` — not merely in ``predict_state``;
3. the two settings get **distinct compile signatures**, so the shared kernel
   cache cannot hand one model the other's photometry;
4. the axis is discoverable and rejects bad input loudly;
5. **the prior travels with the map.** At ``a < 1`` the latent prior is
   ``N(0, sigma_s^(2-2a) I)``, not ``N(0, I)``. A change of coordinates that
   moves the map without moving the prior is not a reparameterization — it is a
   different posterior, and it is invisible: the sampler runs clean and the
   recovered sigma drifts with a knob that was meant to change nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.contract


SFH_FIELD = {
    "type": ["dpl", "field"],
    "all_params": Fixed(DEFAULT),
}


def _build(ssp, obs=None, **sfh_extra):
    """A minimal dust-free, fixed-z GP-field model differing only in the sfh group."""
    kwargs = {}
    if obs is not None:
        kwargs["observation"] = obs
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh=dict(SFH_FIELD, **sfh_extra),
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.05),
        n_grid=16,
        **kwargs,
    )


def _xi(n=16, seed=0):
    import jax.numpy as jnp

    return jnp.asarray(np.random.default_rng(seed).normal(size=n))


def _field_params(model, seed=0):
    """The latent plus whatever else the spec leaves free."""
    return {"sfh_field_xi": _xi(16, seed)}


def _age_marginal(model, params):
    state = model.predict_state(params)
    return np.asarray(state.derived["joint_weights"]).sum(axis=0)


class TestFieldCenteringSelection:
    def test_default_is_bit_identical_to_explicit_one(self, synthetic_ssp_wide):
        """Leaving field_centering unset must not perturb a single existing model."""
        p = _field_params(None)
        w_default = _age_marginal(_build(synthetic_ssp_wide), p)
        w_one = _age_marginal(_build(synthetic_ssp_wide, field_centering=1.0), p)
        assert np.array_equal(w_default, w_one)

    @pytest.mark.parametrize("a", [0.0, 0.5])
    def test_partial_centering_changes_the_field(self, synthetic_ssp_wide, a):
        """a < 1 must actually change the map — a knob that no-ops is the bug."""
        p = _field_params(None)
        w_one = _age_marginal(_build(synthetic_ssp_wide, field_centering=1.0), p)
        w_a = _age_marginal(_build(synthetic_ssp_wide, field_centering=a), p)
        assert np.abs(w_one - w_a).sum() > 1e-6, (
            f"field_centering={a} produced the same age weights as a=1; the knob "
            f"never reached compute_field_gp"
        )


class TestFieldCenteringReachesTheHotPath:
    """The knob must reach predict_photometry, not just predict_state.

    ``predict_photometry`` is the inference hot path and goes through the cached
    compiled observables closure; ``predict_state`` does not. A knob verified
    only on ``predict_state`` can be wholly inert where the A/B would read it.
    """

    def test_photometry_differs_between_centerings(self, synthetic_ssp_wide, synthetic_tophat_obs):
        p = _field_params(None)
        m_one = _build(synthetic_ssp_wide, synthetic_tophat_obs, field_centering=1.0)
        m_zero = _build(synthetic_ssp_wide, synthetic_tophat_obs, field_centering=0.0)
        p_one = np.asarray(m_one.predict_photometry(p))
        p_zero = np.asarray(m_zero.predict_photometry(p))
        assert np.all(np.isfinite(p_one)) and np.all(np.isfinite(p_zero))
        assert not np.array_equal(p_one, p_zero), (
            "predict_photometry is identical across centerings — the knob does "
            "not reach the compiled observables closure"
        )

    def test_photometry_is_build_order_independent(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Whichever centering is built first must not win for the other (#1450)."""
        p = _field_params(None)

        def phot(a, other_first):
            if other_first is not None:
                _build(
                    synthetic_ssp_wide, synthetic_tophat_obs, field_centering=other_first
                ).predict_photometry(p)
            m = _build(synthetic_ssp_wide, synthetic_tophat_obs, field_centering=a)
            return np.asarray(m.predict_photometry(p))

        assert np.array_equal(phot(0.0, None), phot(0.0, 1.0))
        assert np.array_equal(phot(1.0, None), phot(1.0, 0.0))

    def test_compile_signature_distinguishes_centerings(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """The cache key itself must separate the settings."""
        m_one = _build(synthetic_ssp_wide, synthetic_tophat_obs, field_centering=1.0)
        m_zero = _build(synthetic_ssp_wide, synthetic_tophat_obs, field_centering=0.0)
        assert m_one.compile_signature() != m_zero.compile_signature()


class TestFieldCenteringRejectsBadInput:
    # NOTE: these match the *range* wording, not the bare name. Matching only
    # "field_centering" passes vacuously against the grammar's own
    # "Unknown key 'field_centering'" — which is what these asserted first, and
    # they went green before a line of the knob existed.
    @pytest.mark.parametrize("bad", [1.5, -0.1])
    def test_out_of_range_raises(self, synthetic_ssp_wide, bad):
        with pytest.raises(ValueError, match=r"field_centering.*between 0 and 1"):
            _build(synthetic_ssp_wide, field_centering=bad)

    def test_non_numeric_raises(self, synthetic_ssp_wide):
        with pytest.raises((ValueError, TypeError), match=r"field_centering.*between 0 and 1"):
            _build(synthetic_ssp_wide, field_centering="centered")

    def test_without_a_field_an_explicit_request_raises(self, synthetic_ssp_wide):
        """An explicit request the model cannot serve must not silently no-op.

        A parametric-only SFH has no GP field to reparameterize, so
        ``field_centering`` has nothing to act on. Accepting it there would be
        the #1488 pattern exactly: the value is taken, and nothing happens.
        """
        with pytest.raises(ValueError, match=r"field_centering.*field"):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
                sfh={
                    "type": "delayed",
                    "tau_gyr": Fixed(1.0),
                    "age_gyr": Fixed(5.0),
                    "log_total_mass": Fixed(10.0),
                    "field_centering": 0.5,
                    "all_params": Fixed(DEFAULT),
                },
                dust_attenuation={
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.05),
            )


class TestTheLatentPriorTravelsWithTheMap:
    """#1355's silent-failure mode, and the reason this knob is not just plumbing.

    ``standardized_neg_log_prior`` penalizes every latent as ``N(0, I)``. That is
    correct only at ``a = 1``. At ``a < 1`` the field latent's prior is
    ``N(0, sigma_s^(2-2a) I)`` and carries a ``-n(1-a) log sigma_s`` normalizer
    that depends on a *sampled* parameter — so dropping it does not shift the
    posterior by a constant, it changes the sigma marginal.
    """

    def test_prior_differs_between_centerings_at_fixed_latent(self):
        """The same zeta must not score the same under two different a."""
        import jax.numpy as jnp

        from tengri.inference.loss_functions import standardized_neg_log_prior

        p = {"psd_xi": jnp.ones(16)}
        one = float(standardized_neg_log_prior(p, (), stochastic=True, centering=1.0))
        zero = float(
            standardized_neg_log_prior(p, (), stochastic=True, centering=0.0, psd_sigma_dex=0.8)
        )
        assert one != zero, (
            "the penalty is identical at a=1 and a=0 — the fit would target the "
            "non-centered prior with the partially-centered map, which is a "
            "different posterior at every a and reports nothing"
        )

    def test_the_normalizer_couples_the_prior_to_sigma(self):
        """At a < 1 the penalty must MOVE with sigma; at a = 1 it must not.

        This is the whole hazard in one assertion. The ``-n(1-a) log sigma_s``
        term is the only thing tying the latent prior to a sampled parameter,
        and dropping it leaves a sampler that runs clean while the recovered
        sigma drifts with a knob meant to change nothing.
        """
        import jax.numpy as jnp

        from tengri.inference.loss_functions import standardized_neg_log_prior

        p = {"psd_xi": jnp.ones(16)}

        def at(sigma, a):
            return float(
                standardized_neg_log_prior(
                    p, (), stochastic=True, centering=a, psd_sigma_dex=sigma
                )
            )

        assert at(0.4, 0.0) != at(0.9, 0.0), "a=0 penalty does not depend on sigma"
        # a = 1 ignores sigma entirely — the standardized case, unchanged.
        assert at(0.4, 1.0) == at(0.9, 1.0)

    def test_missing_sigma_raises_instead_of_silently_standardizing(self):
        """The failure mode here is silence, so the guard must be loud."""
        import jax.numpy as jnp

        from tengri.inference.loss_functions import standardized_neg_log_prior

        with pytest.raises(ValueError, match=r"psd_sigma_dex"):
            standardized_neg_log_prior({"psd_xi": jnp.ones(4)}, (), stochastic=True, centering=0.5)

    def test_prior_is_scalar_at_every_centering(self):
        """#1551's guarantee must survive the new branch, batched included."""
        import jax.numpy as jnp

        from tengri.inference.loss_functions import standardized_neg_log_prior

        for a in (1.0, 0.5, 0.0):
            value = standardized_neg_log_prior(
                {"psd_xi": jnp.ones((4, 16))},
                (),
                stochastic=True,
                centering=a,
                psd_sigma_dex=0.8,
            )
            assert jnp.ndim(value) == 0, f"a={a} returned shape {jnp.shape(value)}"

    def test_both_consumers_thread_the_centering(self):
        """Pinned on the source, as #1551 pins the shared helper itself.

        Two call sites read this prior — the objective and the context
        accessor. A knob threaded into one of them is the drift that test
        exists to prevent.
        """
        import inspect

        from tengri.inference import context as context_module, loss_functions

        assert "centering=field_centering" in inspect.getsource(loss_functions.build_loss_fn)
        assert "centering=field_centering" in inspect.getsource(
            context_module.InferenceContext.log_prior_fn.fget
        )


class TestReconstructionUsesTheSameMap:
    """The fourth ``compute_field_gp`` call site — posterior reconstruction.

    ``centered_fields`` turns stored latents back into a field. Its own
    docstring says it delegates rather than reimplementing because "two
    implementations of one transform is how a reconstruction silently stops
    matching the fit that produced it" — which is precisely what a hardcoded
    ``a = 1`` would reintroduce for any fit run at ``a < 1``.
    """

    def test_reconstruction_follows_the_centering(self):
        import jax.numpy as jnp

        from tengri.inference.population.reconstruct import centered_fields

        grid = jnp.asarray(np.linspace(6.0, 10.14, 16))
        xi = jnp.asarray(np.random.default_rng(3).normal(size=16))

        one = np.asarray(centered_fields(xi, 0.8, 1.5e8, grid))
        zero = np.asarray(centered_fields(xi, 0.8, 1.5e8, grid, centering=0.0))

        assert not np.allclose(one, zero), (
            "centered_fields ignores centering — an a<1 fit would be "
            "reconstructed with the non-centered map"
        )

    def test_default_reconstruction_is_unchanged(self):
        """Existing callers pass four arguments and must be untouched."""
        import jax.numpy as jnp

        from tengri.inference.population.reconstruct import centered_fields

        grid = jnp.asarray(np.linspace(6.0, 10.14, 16))
        xi = jnp.asarray(np.random.default_rng(4).normal(size=16))
        assert np.array_equal(
            np.asarray(centered_fields(xi, 0.8, 1.5e8, grid)),
            np.asarray(centered_fields(xi, 0.8, 1.5e8, grid, centering=1.0)),
        )


class TestFieldCenteringIsDiscoverable:
    """A builder-accepted axis named by no menu is undiscoverable (#1446).

    This axis is a **continuous** setting, so it gets no ``list_*`` menu of its
    own — inventing one for a float would be a new idiom, and a new idiom is the
    tell of a point fix. It is instead named in the row of the model it acts on,
    which is where a user looking at the field SFH will actually be.
    """

    def test_search_finds_the_axis(self):
        import tengri

        hits = tengri.search("field_centering")
        assert len(hits) > 0, "search('field_centering') returns nothing — the axis is unfindable"

    def test_the_field_sfh_row_names_the_knob(self):
        """Discoverable from the menu the user is already reading."""
        import tengri

        row = next(r for r in tengri.list_sfh_models() if r["name"] == "field")
        assert "field_centering" in row["short_doc"]
