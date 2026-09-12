# SPDX-License-Identifier: BSD-3-Clause
"""Policy-ledger tests for the four inference-side cache keys (#2163 E.5).

Covers ``Fitter._engine_cache_key()`` / ``_data_fingerprint()`` (both derived
from the complementary ``tengri.inference._engine_policy`` ledgers),
``adaptation_method_key()`` / ``_ADAPT_IRRELEVANT`` (the per-backend MCMC
adaptation cache), and ``PreconditionedProblem.cache_key``.

Representative fitters reuse the tiny, fast ``SSPData`` fixture pattern from
``tests/regression/bug/test_1972_baked_spec_state_collisions.py`` (a
real-enough-to-compile 8x15x200 grid, orders of magnitude cheaper than loading
an on-disk SSP file) and ``tests/contract/_signature_builds.py`` for a build
that goes through the full ``SEDModel.build`` grammar.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fitter, Fixed, Parameters, SEDModel, Uniform, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.inference._engine_policy import ENGINE_POLICY, FINGERPRINT_POLICY
from tengri.inference._sample_utils import _data_fingerprint
from tengri.inference.backends.mcmc._shared import (
    _ADAPT_IRRELEVANT,
    _adaptation_cache_key,
    adaptation_method_key,
)
from tengri.inference.backends.mcmc.chees import run_chees
from tengri.inference.backends.mcmc.dynamic_hmc import run_dynamic_hmc
from tengri.inference.backends.mcmc.first_order import run_first_order
from tengri.inference.backends.mcmc.ghmc import run_ghmc
from tengri.inference.backends.mcmc.hmc import run_hmc
from tengri.inference.backends.mcmc.mclmc import run_adjusted_mclmc, run_mclmc
from tengri.inference.backends.mcmc.nuts import run_nuts
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineList
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Representative Fitters
# ---------------------------------------------------------------------------


def _small_ssp() -> SSPData:
    """A real-enough-to-compile SSP grid, cheap to build (test_1972's fixture)."""
    n_met, n_age, n_wave = 8, 15, 200
    rng = np.random.default_rng(0)
    return SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.asarray(rng.uniform(0.5, 1.5, (n_met, n_age, n_wave)), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )


def _base_spec(**overrides) -> Parameters:
    kwargs = dict(redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0))
    kwargs.update(overrides)
    return Parameters(**kwargs)


def fitter_photometry() -> Fitter:
    """Representative A: plain photometry."""
    ssp = _small_ssp()
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    model = SEDModel(_base_spec(), ssp, observation=obs)
    return Fitter(model, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")


def fitter_spectroscopy_eline_calibration() -> Fitter:
    """Representative B: spectroscopy with eline marginalization + calibration."""
    ssp = _small_ssp()
    wave_obs = np.linspace(3800.0, 9000.0, 300)
    spec_cfg = Spectroscopy(
        wave_obs=wave_obs, eline_mode="marginalized", eline_catalog=LineList.default_13()
    )
    obs = Observation(spectroscopy=spec_cfg)
    model = SEDModel(_base_spec(), ssp, observation=obs)
    return Fitter(
        model,
        jnp.ones(300),
        jnp.ones(300) * 0.1,
        data_type="spectroscopy",
        calibration_marginalize=True,
        approx=None,
    )


def fitter_joint_photometry_line_fluxes() -> Fitter:
    """Representative C: joint photometry + spectroscopy with a line-flux override (#1599)."""
    ssp = _small_ssp()
    wave_obs = np.linspace(3800.0, 9000.0, 300)
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
        spectroscopy=Spectroscopy(wave_obs=wave_obs),
    )
    model = SEDModel(_base_spec(), ssp, observation=obs)
    line_flux = LineFluxData(
        names=("Halpha",),
        fluxes=jnp.array([1e-16]),
        errors=jnp.array([1e-17]),
        wavelengths=jnp.array([6564.61]),
    )
    data = jnp.concatenate([jnp.ones(3), jnp.ones(300)])
    noise = jnp.concatenate([jnp.ones(3) * 0.1, jnp.ones(300) * 0.1])
    return Fitter(model, data, noise, data_type="joint", line_flux_data=line_flux, approx=None)


def fitter_with_data_mask() -> Fitter:
    """Representative D: photometry with a censoring mask."""
    ssp = _small_ssp()
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    model = SEDModel(_base_spec(), ssp, observation=obs)
    return Fitter(
        model,
        jnp.ones(3),
        jnp.ones(3) * 0.1,
        data_type="photometry",
        data_mask=jnp.array([0, 0, 1]),
    )


def fitter_with_presence() -> Fitter:
    """Representative E: photometry with per-band presence (batched-catalog shape)."""
    ssp = _small_ssp()
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    model = SEDModel(_base_spec(), ssp, observation=obs)
    return Fitter(
        model,
        jnp.ones(3),
        jnp.ones(3) * 0.1,
        data_type="photometry",
        presence=jnp.array([1.0, 1.0, 0.0]),
    )


def fitter_after_a_map_multistart_run() -> Fitter:
    """Representative F: a Fitter that has RUN a MAP multistart fit.

    Found the hard way (#2163 E.5): ``_memo_batch_kernel`` lazily attaches
    ``_map_multistart_kernel_cache`` to ``self.__dict__`` the first time
    ``run("map", n_restarts>1)`` executes (map_dispatch.py's
    ``_run_map_multistart``); it is ABSENT on a freshly-constructed Fitter,
    so representatives A-E alone missed it entirely, and it defaulted to
    ``content`` (embedding the whole model + a huge closure repr) --
    silently breaking ``_data_fingerprint`` reproducibility across a
    MAP-then-sample workflow
    (``tests/regression/test_sampler_seed_reproducibility.py::
    test_explicit_map_seeds_the_sampler_instead_of_a_second_map``).
    ``_batch_map_kernel_cache`` / ``_batch_mcmc_kernel_cache`` /
    ``_blackjax_draw_kernel_cache`` / ``_batch_adapt_kernel_cache``
    (fit_batch's vmapped kernels) and ``_native_vi_nonlinear_engine``
    (vi/native.py) are the same class of lazy memo, found by source review
    rather than by exercising them here: fit_batch needs a batched
    ForwardModel/CatalogFitter and native VI is ``tier=broken`` (CLAUDE.md),
    neither a good fit for a fast contract test.
    """
    from tests.contract._signature_builds import resolve_ssp_data

    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i", "sdss_z"]))
    model = SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={
            "type": "dpl",
            "alpha": Uniform(0.5, 4.0),
            "beta": Uniform(0.3, 3.0),
            "other_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.1),
    )
    data = jnp.array([1.2, 0.8, 0.5, 0.9])
    noise = jnp.array([0.1, 0.08, 0.06, 0.07])
    fitter = Fitter(model, data, noise, data_type="photometry", compile_modes=None)
    # Both multistart memos: the default optimizer (lbfgs) attaches
    # no memo attribute (sequential scipy); Adam attaches the optax sibling.
    fitter.run("map", n_restarts=2, n_steps=5, verbose=False)
    fitter.run("map", n_restarts=2, n_steps=5, optimizer="adam", verbose=False)
    return fitter


REPRESENTATIVE_FITTER_BUILDERS = (
    ("A-photometry", fitter_photometry),
    ("B-spectroscopy-eline-calibration", fitter_spectroscopy_eline_calibration),
    ("C-joint-line-fluxes", fitter_joint_photometry_line_fluxes),
    ("D-data-mask", fitter_with_data_mask),
    ("E-presence", fitter_with_presence),
    ("F-after-map-multistart", fitter_after_a_map_multistart_run),
)


@pytest.fixture(scope="module")
def representative_fitters() -> list[Fitter]:
    return [build() for _name, build in REPRESENTATIVE_FITTER_BUILDERS]


# ---------------------------------------------------------------------------
# Policy completeness + partition (replaces the retired `fitter_sig == 18`
# ratchet and `test_engine_cache_key_matches_compile_signature_fields` in
# tests/contract/test_compile_signature_invariants.py)
# ---------------------------------------------------------------------------


#: Lazy memo caches ``_memo_batch_kernel`` / ``vi/native.py`` attach to a
#: Fitter's ``__dict__`` post-construction, under conditions this fast
#: contract test does not exercise (a batched fit_batch vmap path, or the
#: tier=broken native-VI-nonlinear backend). ``_map_multistart_kernel_cache``
#: -- the sibling that IS exercised, by representative F -- found the real
#: bug (#2163 E.5): before it was classified, it defaulted to ``content``
#: and broke ``_data_fingerprint`` reproducibility across a MAP-then-sample
#: workflow. These five are excluded in ENGINE_POLICY/FINGERPRINT_POLICY for
#: the identical reason (found by source review, see _engine_policy.py's
#: comments), so ``assert_policy_complete``'s "stale" check -- which looks
#: for a policy row with nothing backing it -- is told to allow them rather
#: than either skipping the check or paying for a batched/native-VI fixture.
_LAZY_MEMO_ATTRS_NOT_EXERCISED_HERE = frozenset(
    {
        "_batch_map_kernel_cache",
        "_batch_mcmc_kernel_cache",
        "_batch_adapt_kernel_cache",
        "_blackjax_draw_kernel_cache",
        "_native_vi_nonlinear_engine",
    }
)


def _assert_policy_complete_allowing_known_lazy_memos(fitters, policy) -> None:
    """Like :func:`assert_policy_complete`, tolerating the lazy memos above."""
    all_attrs: set[str] = set()
    for fitter in fitters:
        all_attrs.update(vars(fitter))
    policy_attrs = set(policy)
    unclassified = sorted(all_attrs - policy_attrs)
    assert not unclassified, f"unclassified attributes: {unclassified}"
    stale = sorted(policy_attrs - all_attrs - _LAZY_MEMO_ATTRS_NOT_EXERCISED_HERE)
    assert not stale, f"stale policy entries: {stale}"


def test_engine_policy_complete(representative_fitters):
    """ENGINE_POLICY classifies every attribute on every representative Fitter."""
    _assert_policy_complete_allowing_known_lazy_memos(representative_fitters, ENGINE_POLICY)


def test_fingerprint_policy_complete(representative_fitters):
    """FINGERPRINT_POLICY classifies every attribute on every representative Fitter."""
    _assert_policy_complete_allowing_known_lazy_memos(representative_fitters, FINGERPRINT_POLICY)


def test_engine_and_fingerprint_ledgers_partition_fitter_attributes(representative_fitters):
    """Every Fitter attribute is content in AT MOST one ledger, never lost from both.

    The engine's four ``shape`` rows (``data``, ``noise``, ``data_mask``,
    ``presence``) are exactly the fingerprint's ``content`` rows -- by
    design, NOT a conflict: ``shape`` only ever hashes shape/dtype, so
    pairing it with the fingerprint's full-value ``content`` row is the
    whole point of the split (structure vs. data, on the SAME attribute).
    ``_runtime_redshift`` is the fingerprint's fifth content row despite
    being excluded from the engine key (#1316). What must never happen is
    an attribute hashed by full VALUE (``content``) in BOTH ledgers at once
    -- that is exactly the #2163 E.5 hazard this module's own
    ``_user_likelihood`` fix closed (see ``GaussianLikelihood.cache_key``) --
    or an attribute excluded from BOTH without a memo/threading/tail/structure
    reason (its content must be captured somewhere: the engine key's tail,
    or the model's own signature).
    """
    all_attrs: set[str] = set()
    for fitter in representative_fitters:
        all_attrs.update(vars(fitter))

    for name in sorted(all_attrs):
        engine_mode, engine_reason = ENGINE_POLICY[name]
        fp_mode, fp_reason = FINGERPRINT_POLICY[name]

        assert not (engine_mode == "content" and fp_mode == "content"), (
            f"{name!r} is hashed by full VALUE (content) in BOTH ledgers -- "
            f"a per-galaxy data leak into the engine key, or a structural "
            f"attribute needlessly duplicated into the fingerprint"
        )

        if engine_mode == "exclude" and fp_mode == "exclude":
            combined = f"{engine_reason} | {fp_reason}"
            assert any(
                kw in combined for kw in ("memo", "thread", "tail", "structure", "scheduling")
            ), (
                f"{name!r} is excluded from BOTH ledgers without a "
                f"memo/threading/tail/structure reason: {combined}"
            )
        else:
            assert engine_mode != "exclude" or fp_mode != "exclude", f"{name!r} unreachable branch"


def test_engine_and_fingerprint_shape_content_rows_agree(representative_fitters):
    """The engine's shape rows are exactly the fingerprint's content rows (+ redshift)."""
    engine_shape_rows = {name for name, (mode, _) in ENGINE_POLICY.items() if mode == "shape"}
    fingerprint_content_rows = {
        name for name, (mode, _) in FINGERPRINT_POLICY.items() if mode == "content"
    }
    assert engine_shape_rows == {"data", "noise", "data_mask", "presence"}
    assert fingerprint_content_rows == engine_shape_rows | {"_runtime_redshift"}


# ---------------------------------------------------------------------------
# Engine key: equal content -> equal key; data values do not move it
# ---------------------------------------------------------------------------


def test_equal_content_fitters_share_an_engine_key():
    """Two independently-built Fitters with equal structure get equal engine keys."""
    ssp = _small_ssp()
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    model_a = SEDModel(_base_spec(), ssp, observation=obs)
    model_b = SEDModel(_base_spec(), ssp, observation=obs)
    fa = Fitter(model_a, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")
    fb = Fitter(model_b, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")
    assert fa._engine_cache_key() == fb._engine_cache_key()


def test_data_values_do_not_move_the_engine_key_but_do_move_the_fingerprint():
    """Two Fitters differing only in the DATA VALUES: same engine key, different fingerprint."""
    ssp = _small_ssp()
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    model = SEDModel(_base_spec(), ssp, observation=obs)
    fa = Fitter(model, jnp.array([1.0, 2.0, 3.0]), jnp.ones(3) * 0.1, data_type="photometry")
    fb = Fitter(model, jnp.array([9.0, 9.0, 9.0]), jnp.ones(3) * 0.1, data_type="photometry")
    assert fa._engine_cache_key() == fb._engine_cache_key(), (
        "data VALUES must not move the engine key -- only shape does"
    )
    assert _data_fingerprint(fa) != _data_fingerprint(fb), (
        "data VALUES must move the fingerprint -- that is its entire job"
    )


def test_data_length_does_move_the_engine_key():
    """Two Fitters differing in data LENGTH get different engine keys (shape row)."""
    ssp = _small_ssp()
    obs3 = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    obs2 = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))
    model3 = SEDModel(_base_spec(), ssp, observation=obs3)
    model2 = SEDModel(_base_spec(), ssp, observation=obs2)
    fa = Fitter(model3, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")
    fb = Fitter(model2, jnp.ones(2), jnp.ones(2) * 0.1, data_type="photometry")
    assert fa._engine_cache_key() != fb._engine_cache_key()


# ---------------------------------------------------------------------------
# Runtime redshift: moves the fingerprint only (#1316)
# ---------------------------------------------------------------------------


def test_runtime_redshift_moves_the_fingerprint_only(synthetic_ssp_wide, synthetic_tophat_obs):
    """A per-galaxy routed redshift shares one engine but fingerprints as distinct targets."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.3),
        approx=WavePrecomp(catalog_z_range=(0.05, 0.9), n_z=32),
    )
    params = model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(model.predict_photometry(params))
    noise = 0.05 * np.abs(flux)

    f1 = Fitter(model, flux, noise, data_type="photometry", params_override={"redshift": 0.2})
    f2 = Fitter(model, flux, noise, data_type="photometry", params_override={"redshift": 0.7})

    assert f1._engine_cache_key() == f2._engine_cache_key(), (
        "distinct runtime redshifts must share one engine (spec #1320 Section 9.4)"
    )
    assert _data_fingerprint(f1) != _data_fingerprint(f2), (
        "distinct runtime redshifts must fingerprint as distinct targets (#1316); "
        "otherwise a catalog loop reuses one galaxy's MAP/adaptation for another"
    )


# ---------------------------------------------------------------------------
# Adaptation method key: per-backend perturbation
# ---------------------------------------------------------------------------

_ADAPTATION_BACKENDS = (
    ("nuts", run_nuts, dict(n_warmup=100, target_accept_rate=0.85)),
    ("hmc", run_hmc, dict(n_warmup=100, target_accept_rate=0.85)),
    ("dynamic_hmc", run_dynamic_hmc, dict(n_warmup=100, target_accept_rate=0.85)),
    ("chees", run_chees, dict(n_warmup=100, target_accept_rate=0.651)),
    ("ghmc", run_ghmc, dict(n_warmup=100, alpha=0.8)),
    ("mclmc", run_mclmc, dict(n_warmup=100, desired_energy_var=5e-4)),
    ("adjusted_mclmc", run_adjusted_mclmc, dict(n_warmup=100, target_accept_rate=0.65)),
    ("first_order", run_first_order, dict(n_warmup=100, target_accept_rate=0.574)),
)


_ADAPTATION_BACKEND_IDS = [b[0] for b in _ADAPTATION_BACKENDS]


@pytest.mark.parametrize("name,fn,base_kwargs", _ADAPTATION_BACKENDS, ids=_ADAPTATION_BACKEND_IDS)
def test_excluded_parameters_do_not_move_the_adaptation_key(name, fn, base_kwargs):
    """Every name in _ADAPT_IRRELEVANT that is also a real parameter of `fn` is inert."""
    params = inspect.signature(fn).parameters
    baseline = adaptation_method_key(name, fn, base_kwargs)
    for excluded_name in _ADAPT_IRRELEVANT:
        if excluded_name not in params:
            continue
        perturbed = dict(base_kwargs)
        # A value plausible for any parameter: strings and bools are valid
        # Python objects regardless of the parameter's own type, and baked()
        # only needs to see that it is DIFFERENT from the default, not that
        # it is physically sensible (the runner itself is never called here).
        perturbed[excluded_name] = "___perturbed___"
        key = adaptation_method_key(name, fn, perturbed)
        assert key == baseline, (
            f"{name}: excluded parameter {excluded_name!r} moved the adaptation key"
        )


@pytest.mark.parametrize("name,fn,base_kwargs", _ADAPTATION_BACKENDS, ids=_ADAPTATION_BACKEND_IDS)
def test_non_excluded_parameters_move_the_adaptation_key(name, fn, base_kwargs):
    """Every real parameter of `fn` NOT in _ADAPT_IRRELEVANT changes the key when perturbed."""
    sig = inspect.signature(fn)
    baseline = adaptation_method_key(name, fn, base_kwargs)
    for pname, param in sig.parameters.items():
        if pname in _ADAPT_IRRELEVANT or param.default is inspect.Parameter.empty:
            continue
        # ghmc deliberately excludes target_accept_rate on top of the shared
        # ledger (MEADS does not read it) -- not itself part of _ADAPT_IRRELEVANT.
        if name == "ghmc" and pname == "target_accept_rate":
            continue
        perturbed = dict(base_kwargs)
        perturbed[pname] = "___perturbed___"
        key = adaptation_method_key(name, fn, perturbed)
        assert key != baseline, (
            f"{name}: non-excluded parameter {pname!r} did not move the adaptation key"
        )


def test_init_from_and_n_samples_are_excluded():
    assert "init_from" in _ADAPT_IRRELEVANT
    assert "n_samples" in _ADAPT_IRRELEVANT


def test_n_warmup_and_target_accept_rate_are_not_excluded():
    assert "n_warmup" not in _ADAPT_IRRELEVANT
    assert "target_accept_rate" not in _ADAPT_IRRELEVANT


# ---------------------------------------------------------------------------
# Adaptation cache key: two fitters on different data get different keys;
# the same data reloaded gets an equal key.
# ---------------------------------------------------------------------------


def test_adaptation_cache_key_differs_across_data_and_matches_on_reload():
    ssp = _small_ssp()
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    model = SEDModel(_base_spec(), ssp, observation=obs)

    f1 = Fitter(model, jnp.array([1.0, 2.0, 3.0]), jnp.ones(3) * 0.1, data_type="photometry")
    f2 = Fitter(model, jnp.array([9.0, 9.0, 9.0]), jnp.ones(3) * 0.1, data_type="photometry")
    f1_reloaded = Fitter(
        model, jnp.array([1.0, 2.0, 3.0]), jnp.ones(3) * 0.1, data_type="photometry"
    )

    method_key = adaptation_method_key("nuts", run_nuts, dict(n_warmup=100))
    k1 = _adaptation_cache_key(f1, method_key)
    k2 = _adaptation_cache_key(f2, method_key)
    k1_reloaded = _adaptation_cache_key(f1_reloaded, method_key)

    assert k1 != k2, "two fitters on different data must get different adaptation keys"
    assert k1 == k1_reloaded, "the same data reloaded must get an equal adaptation key"


# ---------------------------------------------------------------------------
# Preconditioning
# ---------------------------------------------------------------------------


def test_preconditioned_problem_cache_key_is_hashable_and_strength_only():
    from tengri.inference.preconditioning import (
        _PRECONDITIONED_PROBLEM_CACHE_KEY_POLICY,
        PreconditionedProblem,
    )

    content_rows = {
        name
        for name, (mode, _) in _PRECONDITIONED_PROBLEM_CACHE_KEY_POLICY.items()
        if mode == "content"
    }
    assert content_rows == {"strength"}
    assert set(_PRECONDITIONED_PROBLEM_CACHE_KEY_POLICY) == set(
        PreconditionedProblem.__dataclass_fields__
    )
