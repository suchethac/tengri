# SPDX-License-Identifier: BSD-3-Clause
r"""#2262: DIG mixing is skipped at build time when the spec pins the fraction at zero.

https://github.com/suchethac/tengri/issues/2262

``_mix_dig_backend_evaluations``'s Python-float short-circuit
(``isinstance(neb_dig_frac, (int, float)) and neb_dig_frac == 0.0``) never
fires through ``SEDModel.build``: every parameter value ``NebularSEDComponent.apply``
receives is a JAX array or tracer, for the declared ``Fixed(0.0)`` default exactly
as for any other pin. The fix resolves "DIG active" once at build time
(``_dig_may_be_active(spec)``, #2222's predicate) into a frozen
``NebularSEDComponentConfig.dig_active`` field, threaded through
``build_components`` and read at every mixing call site, so a model whose spec
pins ``neb_dig_frac`` at 0.0 evaluates the nebular backend once per channel
instead of two, with the second evaluation not merely zero-weighted but never
invoked.

The first version of this file (commit 53b9a236e) built its fixture with
``ssp_data=None`` and ``stellar={"sps": "bc03"}`` -- the retired flat
``stellar=`` spelling (NAMING_CONTRACT, #1720; the current grammar's
metallicity group is ``met=``) -- and ``observation={"photometry": {"bands":
[]}, ...}`` (a plain dict where ``Observation``/``Photometry`` instances
belong). Every one of those raises inside ``SEDModel.build``, and a bare
``except (Exception,): pytest.skip(...)`` around the build (and again around
each ``predict_photometry`` / ``predict`` call) turned that raise into
"1 passed, 4 skipped" -- exactly the shape
``tools/check_test_skip_handlers.py`` forbids. Rewritten here against the
real #2195 fixture (``fsps_prsc_miles_chabrier.h5`` + ``cue_weights.npz``,
gated by :mod:`tests._data_skip`, no ``except`` around any build or predict
call), the production call counts below are measured, not guessed.

**Call counts** (backend/reconstruction calls per channel, one
``predict_photometry`` + one ``predict(...).lines.all_lums`` on the exact
path; one ``predict_photometry`` + one ``predict_line_fluxes`` on the grid
path):

=====================  ===========  ==========
condition               exact path   grid path
=====================  ===========  ==========
``neb_dig_frac`` Fixed(0.0) default   1           1
``neb_dig_frac`` Fixed(0.3)           2           2
``neb_dig_frac`` FREE                 2           2
=====================  ===========  ==========

**FLOP guard.** Gradient FLOPs of ``predict_photometry`` (``_grad_flops``,
mirrors ``test_bug_1748_feature_precomp_effect.py``) on the #2195 fixture,
declared default (``dig_active=False``) versus the same spec with
``neb_dig_frac`` FREE (``dig_active=True``, forcing the comparison build --
not by changing the runtime value fed to ``predict_photometry``, which stays
a Python literal either way, but by making ``_dig_may_be_active`` see a
non-degenerate declaration): measured 147,434,528 (default) versus
159,926,608 (forced), a 1.085x reduction. This is well short of the ~50 %
an earlier draft of this fix estimated without measuring it: the DIG
evaluation this ticket removes is a small fraction of a photometry
gradient's total cost once dust attenuation and dust emission are in the
graph, so "one fewer Cue forward" does not mean "half the FLOPs". The
guard asserts the direction (default strictly lower), not a magnitude.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_platforms", "cpu")

from tengri import (
    DEFAULT,
    FREE,
    FeaturePrecomp,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.components.nebular import nebular_grid_precompute as _ngp
from tengri.components.nebular.cue import CueBackend
from tests._data_skip import CUE_WEIGHTS, DATA_DIR, requires_cue_weights

pytestmark = pytest.mark.regression_bug

_SSP_PATH = DATA_DIR / "fsps_prsc_miles_chabrier.h5"

#: Bands and redshift of the #2195 fixture, reused here (exact path).
_FILTERS = ("galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i")
_REDSHIFT = 0.1

#: Lines tabulated by the grid-path fixture (Hb, [OIII], Ha, [NII]), matching
#: ``test_bug_2195_dig_delta_logu_reaches_cue.py``'s grid test.
_LINES = jnp.asarray([4862.68, 5008.24, 6564.61, 6585.27])

_CONDITIONS = (
    pytest.param({}, 1, id="default_fixed_0"),
    pytest.param({"dig_frac": Fixed(0.3)}, 2, id="fixed_0.3"),
    pytest.param({"dig_frac": FREE}, 2, id="free"),
)


@pytest.fixture(scope="module")
def _cue_fixture_available():
    if not _SSP_PATH.is_file() or not CUE_WEIGHTS.is_file():
        pytest.skip(f"needs {_SSP_PATH} and {CUE_WEIGHTS}")


def _params_for(neb_extra: dict) -> dict:
    """Explicit value for ``neb_dig_frac`` when it is FREE; empty otherwise.

    A Fixed value never needs a params entry (the build merges it in); a FREE
    one does, or ``predict_photometry`` has nothing to sample it at.
    """
    if neb_extra.get("dig_frac") is FREE:
        return {"neb_dig_frac": 0.5}
    return {}


def _build_exact(neb_extra: dict):
    """The #2195 fixture (cue backend, dust on, exact path -- no ``approx``)."""
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
    from tengri.observation.filters import load_filter

    obs = Observation(photometry=Photometry(filters=tuple(load_filter(n) for n in _FILTERS)))
    return SEDModel.build(
        ssp_data=load_ssp_data(str(_SSP_PATH)),
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), **neb_extra},
        redshift=Fixed(_REDSHIFT),
    )


def _build_grid(neb_extra: dict):
    """The grid-path counterpart: dust off (#1748 -- dust disarms the photometry
    shortcut), ``neb_logU`` free (an ordinary ionization axis, so the table
    carries a photometry/restband channel regardless of DIG activity), built
    with ``approx=(WavePrecomp(), FeaturePrecomp(...))`` the way
    ``tests/components/nebular/test_nebular_grid_precompute.py`` builds its
    ``_wave_model`` fixture: documented behavior, not a new finding --
    ``reconstruct_nebular_phot`` / ``reconstruct_nebular_restband``'s own
    docstrings (``nebular_grid_precompute.py``) state the table must be
    "built from a ``WavePrecomp`` model so ``log_phot_per_qh`` is populated";
    ``FeaturePrecomp`` alone serves the line channel only.
    """
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
    from tengri.observation.filters import load_filter

    obs = Observation(photometry=Photometry(filters=tuple(load_filter(n) for n in _FILTERS)))
    approx = (WavePrecomp(), FeaturePrecomp(lines=_LINES, n_grid=4))
    return SEDModel.build(
        ssp_data=load_ssp_data(str(_SSP_PATH)),
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Uniform(-4.0, -1.0),
            **neb_extra,
        },
        redshift=Fixed(_REDSHIFT),
        approx=approx,
    )


def _count_class_method(monkeypatch, cls, name):
    """Wrap ``cls.name`` (a bound-method attribute) with a call counter.

    Patches the CLASS, not the instance: ``compile_signature()`` derives a
    cache key from ``vars(instance)`` (``tengri._cache_keys.classify``), which
    never sees a class-level method. Patching the instance instead (setting
    ``backend.predict_nebular_sed = wrapper``) puts the wrapper in
    ``vars(backend)``, and baking a wrapper whose closure captures the
    *original bound method* recurses back into ``backend.cache_key()``
    (measured: ``RecursionError`` via ``CueBackend.cache_key -> derive_key ->
    baked(wrapper) -> baked(original bound method) -> baked(backend) ->
    backend.cache_key()`` ad infinitum) -- an instance-level patch never
    reaches a real assertion, it just crashes ``predict_photometry`` itself.
    """
    original = getattr(cls, name)
    counter = {"n": 0}

    def wrapped(self, *a, **kw):
        counter["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(cls, name, wrapped)
    return counter


def _count_module_function(monkeypatch, module, name):
    """Wrap a module-level function with a call counter (grid-path reconstructors).

    These are looked up by a local ``from ... import name`` inside
    ``NebularSEDComponent.apply`` / ``SEDModel.predict_line_fluxes``, executed
    fresh on every call, so patching the module attribute (not a cached
    reference) is what the call site actually sees.
    """
    original = getattr(module, name)
    counter = {"n": 0}

    def wrapped(*a, **kw):
        counter["n"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(module, name, wrapped)
    return counter


@requires_cue_weights
@pytest.mark.parametrize("neb_extra,expected", _CONDITIONS)
def test_dig_shortcircuit_call_count_exact_path(
    _cue_fixture_available, monkeypatch, neb_extra, expected
):
    """Exact path: ``predict_nebular_sed`` / ``predict_nebular_line_luminosities``
    are called once per channel at the declared default, twice at Fixed(0.3)
    or FREE. Attaches counters to ``CueBackend`` (class-level, see
    ``_count_class_method``) after the model is built, then runs
    ``predict_photometry`` once (SED channel) and ``predict(...).lines.all_lums``
    once (line channel), eager.
    """
    model = _build_exact(neb_extra)
    params = _params_for(neb_extra)

    sed_calls = _count_class_method(monkeypatch, CueBackend, "predict_nebular_sed")
    model.predict_photometry(params)
    assert sed_calls["n"] == expected, (
        f"predict_nebular_sed called {sed_calls['n']} times for {neb_extra}; "
        f"expected {expected} (#2262)"
    )

    line_calls = _count_class_method(monkeypatch, CueBackend, "predict_nebular_line_luminosities")
    pred = model.predict(params)
    _ = pred.lines.all_lums
    assert line_calls["n"] == expected, (
        f"predict_nebular_line_luminosities called {line_calls['n']} times for "
        f"{neb_extra}; expected {expected} (#2262)"
    )


@requires_cue_weights
@pytest.mark.parametrize("neb_extra,expected", _CONDITIONS)
def test_dig_shortcircuit_call_count_grid_path(
    _cue_fixture_available, monkeypatch, neb_extra, expected
):
    """Grid path: ``reconstruct_nebular_phot`` / ``reconstruct_nebular_restband``
    (via ``predict_photometry``) and ``reconstruct_nebular_line_log_lums`` (via
    ``predict_line_fluxes``) follow the same 1/2/2 pattern as the exact path.
    """
    model = _build_grid(neb_extra)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    params.update(_params_for(neb_extra))

    phot_calls = _count_module_function(monkeypatch, _ngp, "reconstruct_nebular_phot")
    restband_calls = _count_module_function(monkeypatch, _ngp, "reconstruct_nebular_restband")
    model.predict_photometry(params)
    assert phot_calls["n"] == expected, (
        f"reconstruct_nebular_phot called {phot_calls['n']} times for {neb_extra}; "
        f"expected {expected} (#2262)"
    )
    assert restband_calls["n"] == expected, (
        f"reconstruct_nebular_restband called {restband_calls['n']} times for "
        f"{neb_extra}; expected {expected} (#2262)"
    )

    line_calls = _count_module_function(monkeypatch, _ngp, "reconstruct_nebular_line_log_lums")
    model.predict_line_fluxes(params, target_wavelengths=_LINES, redden=False)
    assert line_calls["n"] == expected, (
        f"reconstruct_nebular_line_log_lums called {line_calls['n']} times for "
        f"{neb_extra}; expected {expected} (#2262)"
    )


@requires_cue_weights
def test_dig_shortcircuit_gradient_flops(_cue_fixture_available):
    """Gradient FLOPs at the declared default are strictly lower than a build
    forced active (``neb_dig_frac`` FREE), on the #2195 fixture. Modeled on
    ``_grad_flops`` in ``test_bug_1748_feature_precomp_effect.py``. Measured
    pair: 147,434,528 (default) vs 159,926,608 (forced) -- see the module
    docstring for why this is not the ~50 % an earlier estimate claimed.
    """

    def _grad_flops(model):
        free = list(model.spec.free_params)
        defaults = {k: 10.0 if "log_total_mass" in k else 0.5 for k in free}

        def loss(v):
            params = dict(defaults)
            params["sfh_delayed_log_total_mass"] = v
            return jnp.sum(model.predict_photometry(params))

        return int(
            jax.jit(jax.grad(loss)).lower(jnp.asarray(10.0)).compile().cost_analysis()["flops"]
        )

    default_model = _build_exact({})
    forced_model = _build_exact({"dig_frac": FREE})

    flops_default = _grad_flops(default_model)
    flops_forced = _grad_flops(forced_model)
    assert flops_default < flops_forced, (
        f"declared-default gradient FLOPs ({flops_default:,}) are not lower than "
        f"a forced-active build ({flops_forced:,}); the build-time short-circuit "
        "is not reaching the compiled graph (#2262)"
    )


@requires_cue_weights
def test_dig_active_config_field_defaults_true(_cue_fixture_available):
    """``NebularSEDComponentConfig.dig_active`` exists and defaults to ``True``.

    The frozen field the fix threads from ``SEDModel.build`` through
    ``build_components``. ``True`` -- not ``False`` -- is the bare-dataclass
    default: the same defensive-fallback shape as ``cue_full_catalog``, so a
    direct caller of ``build_components``/``NebularSEDComponentConfig`` who
    does not set this field never has declared DIG physics silently dropped
    (worst case: one avoidable extra evaluation, never a wrong answer). A
    real model built through ``SEDModel.build`` always resolves this field
    explicitly via ``_dig_may_be_active(spec)`` and ignores the default.
    """
    from tengri.components.nebular.component import NebularSEDComponentConfig

    config = NebularSEDComponentConfig()
    assert hasattr(config, "dig_active"), "NebularSEDComponentConfig missing dig_active (#2262)"
    assert config.dig_active is True
