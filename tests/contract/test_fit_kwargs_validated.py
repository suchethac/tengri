# SPDX-License-Identifier: BSD-3-Clause
"""#1469: every fit entry that forwards ``**kwargs`` toward a runner refuses an
unknown name with a ``TypeError`` that names the user's call.

Before the fix, ``Catalog.fit(lines=...)`` reached a runner and raised
``TypeError: <runner>() got an unexpected keyword argument 'lines'``: an
internal function the caller never named. ``check_unknown_kwargs`` answers this
at the dispatch seam for ``Fitter.run``; these tests pin that every entry which
reaches a runner without passing through it gives the same answer, and that a
correctly spelled runner kwarg still passes through.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.contract

_UNKNOWN = {"lines": {"Halpha": 1.0}}

# One sequential engine (per-galaxy Fitter.run) and one batched engine
# (CatalogFitter._run_native_mcmc, whose signature differs from the registry runner).
# Each entry: the valid runner kwarg to pass through, and a name the runner must list.
_METHODS = {
    "map": ({"n_steps": 5}, "n_steps"),
    "mcmc_nuts": ({"n_warmup": 5, "n_samples": 5}, "n_warmup"),
}

_FLUX = np.array([1.0, 3.0, 5.0])
_NOISE = np.array([0.1, 0.2, 0.3])


@pytest.fixture
def sed_3band(synthetic_ssp_wide, simple_observation):
    """3-band SEDModel, free DPL SFH, with catalog_z_range so the batched engine runs."""
    from tengri import DEFAULT, FREE, Fixed, SEDModel, WavePrecomp

    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.5,
        },
        neb={"type": "none"},
        redshift=Fixed(DEFAULT),
        approx=WavePrecomp(catalog_z_range=(0.05, 1.5)),
    )


@pytest.fixture
def fwd_3band(sed_3band, simple_observation):
    from tengri import ForwardModel

    return ForwardModel.build(sed=sed_3band, observation=simple_observation)


@pytest.fixture
def table_3band():
    return {
        "band_0": np.array([1.0, 2.0, 1.5]),
        "band_0_err": np.array([0.1, 0.1, 0.1]),
        "band_1": np.array([3.0, 4.0, 3.5]),
        "band_1_err": np.array([0.2, 0.2, 0.2]),
        "band_2": np.array([5.0, 6.0, 5.5]),
        "band_2_err": np.array([0.3, 0.3, 0.3]),
        "z": np.array([0.1, 0.5, 0.3]),
    }


def _catalog_fit(sed, fwd, table, **kwargs):
    from tengri import Catalog

    cat = Catalog(fwd, table, flux_unit="cgs_fnu", redshift_col="z")
    return cat.fit(key=jax.random.PRNGKey(0), **kwargs)


def _catalog_fitter_run(sed, fwd, table, **kwargs):
    from tengri.inference.catalog_fitter import _CatalogFitterOriginal

    galaxies = [
        {
            "flux_obs": np.array([table[f"band_{b}"][i] for b in range(3)]),
            "noise": np.array([table[f"band_{b}_err"][i] for b in range(3)]),
        }
        for i in range(3)
    ]
    fitter = _CatalogFitterOriginal(fwd, galaxies, data_type="photometry")
    return fitter.run(key=jax.random.PRNGKey(0), **kwargs)


def _forward_model_fit(sed, fwd, table, **kwargs):
    return fwd.fit(_FLUX, _NOISE, key=jax.random.PRNGKey(0), **kwargs)


def _sed_model_fit(sed, fwd, table, **kwargs):
    return sed.fit(_FLUX, _NOISE, key=jax.random.PRNGKey(0), **kwargs)


ENTRIES = {
    "Catalog.fit": _catalog_fit,
    "CatalogFitter.run": _catalog_fitter_run,
    "ForwardModel.fit": _forward_model_fit,
    "SEDModel.fit": _sed_model_fit,
}


@pytest.mark.parametrize("method", list(_METHODS))
@pytest.mark.parametrize("entry", list(ENTRIES))
def test_unknown_kwarg_names_the_call_not_the_runner(
    entry, method, sed_3band, fwd_3band, table_3band
):
    fit = ENTRIES[entry]
    _, must_list = _METHODS[method]
    with pytest.raises(TypeError) as excinfo:
        fit(sed_3band, fwd_3band, table_3band, method=method, **_UNKNOWN)
    msg = str(excinfo.value)
    assert "lines" in msg
    assert must_list in msg, "the accepted-kwargs list must name the runner's parameters"
    for internal in ("run_map(", "run_nuts(", "_run_native_mcmc(", "_run_sequential("):
        assert internal not in msg, f"internal runner name leaked: {msg!r}"


@pytest.mark.parametrize("method", list(_METHODS))
@pytest.mark.parametrize("entry", list(ENTRIES))
def test_runner_kwarg_still_passes_through(entry, method, sed_3band, fwd_3band, table_3band):
    fit = ENTRIES[entry]
    kwargs, _ = _METHODS[method]
    post = fit(sed_3band, fwd_3band, table_3band, method=method, **kwargs)
    assert post is not None


# PopulationFitter takes a hierarchical SED model factory and a population of
# galaxies; building one is out of scope for a contract test. The validation
# runs before any model state is read, so a bare instance exercises it exactly.
_POP_METHODS = ("vi_nonlinear_fast", "vi_linear_fast", "mcmc_raytrace")


def _population_fitter():
    from tengri.inference.hierarchical import PopulationFitter

    return object.__new__(PopulationFitter)


@pytest.mark.parametrize("method", _POP_METHODS)
def test_population_unknown_kwarg_names_the_call_not_the_runner(method):
    pop = _population_fitter()
    with pytest.raises(TypeError) as excinfo:
        pop.run(method, key=jax.random.PRNGKey(0), **_UNKNOWN)
    msg = str(excinfo.value)
    assert "lines" in msg
    assert "_run_geovi(" not in msg and "_run_raytrace(" not in msg, msg


@pytest.mark.parametrize(
    ("method", "cfm_method", "knob"),
    [
        ("vi_nonlinear_fast", "geovi", "n_iterations"),
        ("vi_linear_fast", "geovi", "n_iterations"),
        ("mcmc_raytrace", "raytrace", "n_steps"),
    ],
)
def test_population_runner_kwarg_is_not_refused(method, cfm_method, knob):
    """The engine's own knob passes the check (the engine itself is not run here)."""
    pop = _population_fitter()
    assert pop._refuse_unknown_kwargs(method, cfm_method, {knob: 1}) is None


def test_batched_mcmc_engine_accepts_init_from_but_not_unknown_names(fwd_3band, table_3band):
    """``init_from`` is an engine parameter, not an unknown name (#2961 refused it).

    The check runs at the dispatch seam, so no fit is needed: the engine's own
    ``init_from`` passes while a misspelled sibling is still refused by name.
    """
    from tengri.inference.catalog_fitter import _CatalogFitterOriginal

    galaxies = [
        {
            "flux_obs": np.array([table_3band[f"band_{b}"][i] for b in range(3)]),
            "noise": np.array([table_3band[f"band_{b}_err"][i] for b in range(3)]),
        }
        for i in range(3)
    ]
    fitter = _CatalogFitterOriginal(fwd_3band, galaxies, data_type="photometry")
    fitter._refuse_unknown_kwargs("mcmc_hmc", {"init_from": "prior", "n_samples": 5})
    with pytest.raises(TypeError, match="init_form") as excinfo:
        fitter._refuse_unknown_kwargs("mcmc_hmc", {"init_form": "prior"})
    listed = str(excinfo.value).split("It takes:")[1].split("]")[0]
    assert "init_from" in listed, "the engine takes init_from, so the list must offer it"
    assert "method_tag" not in listed, "internal dispatch plumbing is not a user option"
    # The native VI engine takes no init_from: it is still refused there.
    with pytest.raises(TypeError, match="init_from"):
        fitter._refuse_unknown_kwargs("native_vi_linear", {"init_from": "prior"})
