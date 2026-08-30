# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for the CANDELS photometry pipeline of the framework paper (#2089).

Bug: ``fit_one.extract_photometry`` carried a private AB zero point of
3.63e-23 (3.63 Jy: 1000x too small) and a private column map whose five ACS
keys were spelled ``WFC3_*``; a ``continue`` guard hid the mismatch, so
galaxy 13097 fit 8 bands of fluxes 1000x too faint and every NUTS transition
diverged.

Mutation checks (each test names the mutant that kills it):
- zero point back to 3.63e-23: ``test_ab_mag_to_fnu_matches_the_ab_definition``,
  ``test_extract_photometry_13097_returns_every_usable_band``.
- ``ACS_F435W`` key renamed back to ``WFC3_F435W``:
  ``test_the_map_is_exactly_the_documented_one``,
  ``test_every_value_is_a_tengri_filter_and_every_key_a_catalog_column``,
  ``test_extract_photometry_13097_returns_every_usable_band``.
- ``raise KeyError`` -> ``continue``:
  ``test_a_missing_mapped_column_raises_instead_of_dropping_the_band``.
- drop the ``ks_taken`` rule: ``test_one_ks_band_isaac_first_hawki_only_as_fallback``.
- the derived dust key back to ``"dust_tau_v"``, or ``filter_names`` back to
  ``dtype=object``: ``test_save_fit_outputs_writes_a_consistent_npz_for_every_configuration``.
- drop the collision guard in ``build_npz_payload``:
  ``test_the_npz_payload_refuses_a_derived_key_that_shadows_a_parameter``.
- ``retune_settings`` toggles ``dense_mass_matrix`` on attempt 2, or attempt 3
  keeps 0.95 instead of raising the target to 0.99:
  ``test_retune_policy_raises_target_accept_then_lengthens_warmup_and_never_toggles_dense``.
- ``select_best_attempt`` ranks by ``rhat_max`` alone, ignoring divergences:
  ``test_run_fit_keeps_the_best_attempt_when_the_bar_is_missed``.
- the interim ``save_best_so_far`` call is dropped from ``run_fit``'s loop:
  ``test_run_fit_writes_the_best_so_far_after_each_missed_attempt``.
- ``cell_is_adopted`` inverted: ``test_only_missing_skips_adopted_cells``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

REPO = Path(__file__).resolve().parents[2]
PAPER1 = REPO / "analysis" / "paper1"
# fit_one.py does a bare ``from candels_io import ...``: the directory must be importable.
if str(PAPER1) not in sys.path:
    sys.path.insert(0, str(PAPER1))

import candels_io
import fit_one

import tengri
from tengri.utils import physics_constants

EXPECTED_MAP = {
    "ACS_F435W": "hst_f435w",
    "ACS_F606W": "hst_f606w",
    "ACS_F775W": "hst_f775w",
    "ACS_F814W": "hst_f814w",
    "ACS_F850LP": "hst_f850lp",
    "WFC3_F098M": "hst_f098m",
    "WFC3_F105W": "hst_f105w",
    "WFC3_F125W": "hst_f125w",
    "WFC3_F160W": "hst_f160w",
    "ISAAC_KS": "vista_ks",
    "HAWKI_KS": "vista_ks",
    "IRAC_CH1": "irac_36",
    "IRAC_CH2": "irac_45",
    "IRAC_CH3": "irac_58",
    "IRAC_CH4": "irac_80",
}


@pytest.fixture(scope="module")
def catalog():
    try:
        return candels_io.load_candels_z1()
    except FileNotFoundError:
        pytest.skip("CANDELS catalog not found")


@pytest.fixture(scope="module")
def registry_names():
    return {row["alias"] for row in tengri.list_filters()}


def test_ab_mag_to_fnu_matches_the_ab_definition():
    # The script's private copy of the zero point is the library's, exactly.
    assert candels_io.AB_ZERO_POINT_ERG == physics_constants.MAGGIES_ZP_CGS

    fnu, err = candels_io.ab_mag_to_fnu(21.342, 0.001)
    expected = 3.631e-20 * 10 ** (-21.342 / 2.5)
    assert float(fnu) == pytest.approx(expected, rel=1e-3, abs=0)
    assert float(fnu) == pytest.approx(1.055e-28, rel=2e-3, abs=0)
    assert float(err) == pytest.approx(expected * np.log(10) / 2.5 * 0.001, rel=1e-3, abs=0)


def test_ab_mag_to_fnu_is_elementwise():
    fnu, err = candels_io.ab_mag_to_fnu(np.array([20.0, 25.0]), np.array([0.01, 0.1]))
    assert fnu.shape == (2,) and err.shape == (2,)
    assert fnu[0] / fnu[1] == pytest.approx(100.0)


def test_the_map_is_exactly_the_documented_one():
    assert candels_io.CANDELS_TO_TENGRI == EXPECTED_MAP
    assert list(candels_io.CANDELS_TO_TENGRI) == list(EXPECTED_MAP)
    assert candels_io.KS_COLUMNS == ("ISAAC_KS", "HAWKI_KS")
    assert "CTIO_U" not in candels_io.CANDELS_TO_TENGRI
    assert "VIMOS_U" not in candels_io.CANDELS_TO_TENGRI


def test_every_value_is_a_tengri_filter_and_every_key_a_catalog_column(catalog, registry_names):
    for column, name in candels_io.CANDELS_TO_TENGRI.items():
        assert name in registry_names, name
        assert column in catalog["header"], column
        assert f"e{column}" in catalog["header"], column


def test_load_candels_z1_carries_the_data_matrix(catalog):
    assert catalog["data"].shape == (len(catalog["id"]), len(catalog["header"]))


def _row_from(mags: dict[str, tuple[float, float]]):
    header = ["ID"] + [x for c in candels_io.CANDELS_TO_TENGRI for x in (c, f"e{c}")]
    row = np.zeros(len(header))
    for column in candels_io.CANDELS_TO_TENGRI:
        mag, err = mags.get(column, (22.0, 0.05))
        row[header.index(column)] = mag
        row[header.index(f"e{column}")] = err
    return header, row


def test_a_missing_mapped_column_raises_instead_of_dropping_the_band():
    header, row = _row_from({})
    i = header.index("ACS_F435W")
    header_without = header[:i] + header[i + 1 :]
    row_without = np.delete(row, i)
    with pytest.raises(KeyError, match="ACS_F435W"):
        candels_io.photometry_for_row(header_without, row_without)


def test_one_ks_band_isaac_first_hawki_only_as_fallback():
    header, row = _row_from({"ISAAC_KS": (21.0, 0.01), "HAWKI_KS": (22.0, 0.01)})
    names, fnu, _ = candels_io.photometry_for_row(header, row)
    assert names.count("vista_ks") == 1
    isaac_flux = float(candels_io.ab_mag_to_fnu(21.0, 0.01)[0])
    assert fnu[names.index("vista_ks")] == pytest.approx(isaac_flux, rel=1e-12, abs=0)

    header, row = _row_from({"ISAAC_KS": (98.992, -99.0), "HAWKI_KS": (22.0, 0.01)})
    names, fnu, _ = candels_io.photometry_for_row(header, row)
    assert names.count("vista_ks") == 1
    hawki_flux = float(candels_io.ab_mag_to_fnu(22.0, 0.01)[0])
    assert fnu[names.index("vista_ks")] == pytest.approx(hawki_flux, rel=1e-12, abs=0)


def test_photometry_for_row_skips_sentinels_and_orders_by_the_map():
    header, row = _row_from({"WFC3_F098M": (98.992, -99.0)})
    names, fnu, err = candels_io.photometry_for_row(header, row)
    assert "hst_f098m" not in names
    assert names[:5] == ["hst_f435w", "hst_f606w", "hst_f775w", "hst_f814w", "hst_f850lp"]
    assert len(names) == len(fnu) == len(err) == 13
    assert np.all(err > 0)


def test_extract_photometry_13097_returns_every_usable_band(catalog, registry_names):
    z = float(catalog["z"][np.where(catalog["id"] == 13097)[0][0]])
    gal_id, z_out, names, fnu, err = fit_one.extract_photometry(13097, catalog, z)
    assert gal_id == 13097 and z_out == z
    # Every usable band: 15 map keys -> 14 distinct filters (both Ks columns share
    # ``vista_ks``) minus ``WFC3_F098M``, a sentinel for this galaxy. The buggy
    # private map returned 8 (the five ACS bands were spelled ``WFC3_*``).
    assert len(names) == 13
    assert "hst_f435w" in names
    assert "hst_f098m" not in names  # sentinel in the catalog for this galaxy
    assert names.count("vista_ks") == 1
    assert set(names) <= registry_names
    assert np.all((fnu > 5e-30) & (fnu < 5e-28)), fnu
    assert np.all(err > 0) and np.all(err < fnu)


def test_thin_samples_and_iter_draws_use_flattened_draws():
    rng = np.random.default_rng(0)
    samples = {"a": rng.normal(size=9000), "b": rng.normal(size=9000)}
    thin = fit_one.thin_samples(samples)
    assert thin["a"].ndim == 1
    assert 0 < thin["a"].shape[0] <= fit_one.MAX_SAVED_DRAWS
    assert thin["a"][1] == samples["a"][3]  # 9000 draws -> every third one
    assert fit_one.thin_samples({"a": np.arange(2400.0)})["a"].shape == (2400,)

    draws = list(fit_one.iter_draws(thin, {"fixed": 1.5}, 5))
    assert len(draws) == 5
    assert draws[0] == {"fixed": 1.5, "a": float(thin["a"][0]), "b": float(thin["b"][0])}
    assert list(fit_one.iter_draws({"a": np.arange(3.0)}, {}, 10)) == [
        {"a": 0.0},
        {"a": 1.0},
        {"a": 2.0},
    ]


def test_iter_draws_strides_across_every_chain_not_the_first_draws():
    # Flattened draws are chain-major, so the first ``n_draws`` of a 4-chain record
    # are chain 0's warm-up-adjacent draws alone (#2089). ``iter_draws`` strides.
    record = {"a": np.arange(9000.0), "b": np.arange(9000.0) * -1.0}
    drawn = list(fit_one.iter_draws(record, {"fixed": 2.5}, 5))
    assert [d["a"] for d in drawn] == [0.0, 2250.0, 4500.0, 6749.0, 8999.0]
    assert [d["b"] for d in drawn] == [-0.0, -2250.0, -4500.0, -6749.0, -8999.0]
    assert all(d["fixed"] == 2.5 for d in drawn)

    # ``n_draws >= n_available``: every draw, in order, no repeats.
    every = [d["a"] for d in fit_one.iter_draws({"a": np.arange(7.0)}, {}, 99)]
    assert every == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_driver_timeout_covers_a_retune():
    import inspect

    import run_candels_fits

    assert run_candels_fits.DEFAULT_FIT_TIMEOUT_S >= 21600
    default = inspect.signature(run_candels_fits.run_fit_subprocess).parameters["timeout"].default
    assert default == run_candels_fits.DEFAULT_FIT_TIMEOUT_S


def test_posterior_api_names_used_by_the_scripts():
    from tengri.inference.posterior import Posterior

    # The names the scripts call.
    assert hasattr(Posterior, "effective_sample_size")
    assert hasattr(Posterior, "rhat")
    # The names they used to guess at. ``Posterior.__getattr__`` raises for both,
    # so a ``hasattr`` guard on them is permanently False and silently drops ESS.
    assert not hasattr(Posterior, "ess")
    assert not hasattr(Posterior, "covariance")

    for script in ("fit_one.py", "run_backend_sweep.py"):
        source = (PAPER1 / script).read_text()
        assert "hasattr(posterior," not in source, script
        assert "posterior.ess()" not in source, script


def test_backend_sweep_shares_fit_one_photometry_and_runs_the_owner_list(catalog):
    import run_backend_sweep

    assert run_backend_sweep.extract_photometry is fit_one.extract_photometry
    assert run_backend_sweep.apply_systematic_error_floor is fit_one.apply_systematic_error_floor
    assert run_backend_sweep.SWEEP_METHODS == (
        "map",
        "laplace",
        "mcmc",
        "mcmc_nuts",
        "mcmc_hmc",
        "mcmc_raytrace",
    )

    z = float(catalog["z"][np.where(catalog["id"] == 13097)[0][0]])
    a = fit_one.extract_photometry(13097, catalog, z)
    b = run_backend_sweep.extract_photometry(13097, catalog, z)
    assert a[2] == b[2]
    np.testing.assert_array_equal(a[3], b[3])
    np.testing.assert_array_equal(a[4], b[4])


def test_every_sweep_method_is_a_registered_backend_name():
    import run_backend_sweep

    from tengri.inference._backend_registry import get_backend
    from tengri.inference.fitter import _CANONICAL_METHODS

    # ``get_backend`` is the registry's own lookup (single source of truth for
    # fitter.run dispatch); ``"mcmc"`` is checked against the fitter's accepted
    # canonical-method-name set instead, since the brief calls it "resolved in
    # Fitter.run, not the registry" (in fact it is registered too, as the auto
    # NUTS/raytrace dispatcher, but the canonical-name check is the one the
    # brief pins down and it is also correct).
    for method in run_backend_sweep.SWEEP_METHODS:
        if method == "mcmc":
            assert method in _CANONICAL_METHODS, method
        else:
            assert get_backend(method).name == method


# The 13 bands galaxy 13097 is detected in (``WFC3_F098M`` is a sentinel for it, and
# the two Ks columns share ``vista_ks``), mirroring ``test_paper1_configs.py``.
CANDELS_13097_FILTERS = [
    "hst_f435w",
    "hst_f606w",
    "hst_f775w",
    "hst_f814w",
    "hst_f850lp",
    "hst_f105w",
    "hst_f125w",
    "hst_f160w",
    "vista_ks",
    "irac_36",
    "irac_45",
    "irac_58",
    "irac_80",
]
CANDELS_13097_Z = 1.097
#: Draws in the hand-built posterior of the save-path test. 120 was the brief's
#: figure; every draw costs one ``predict`` plus one ``predict_state`` (~0.36 s
#: for configuration I on this machine), which put configuration I alone near 45 s
#: and configurations II/III past the 60 s per-configuration budget, so the record
#: is 40 draws. The save path is indifferent to the count.
SAVE_PATH_DRAWS = 40
SAVE_PATH_SFH_GRID = 100  # ``np.logspace(6, 10.1, 100)`` in ``save_fit_outputs``


def test_the_npz_payload_refuses_a_derived_key_that_shadows_a_parameter():
    # Configuration I samples ``dust_tau_v``; the derived dust array used to be
    # written under that same name, and ``np.savez(**samples, **derived)`` died
    # with "got multiple values for keyword argument 'dust_tau_v'" (#2089) --
    # after a 1463 s fit that had already passed the adoption bar.
    samples = {"dust_tau_v": np.zeros(3), "met_logzsol": np.ones(3)}
    payload = fit_one.build_npz_payload(samples, {"dust_tau": np.full(3, 0.5)})
    assert sorted(payload) == ["dust_tau", "dust_tau_v", "met_logzsol"]

    with pytest.raises(ValueError, match="dust_tau_v"):
        fit_one.build_npz_payload(samples, {"dust_tau_v": np.full(3, 0.5)})
    with pytest.raises(ValueError, match="met_logzsol"):
        fit_one.build_npz_payload(samples, {"dust_tau": np.zeros(3)}, {"met_logzsol": np.zeros(3)})


@pytest.mark.parametrize("config_key", ["I", "II", "III"])
def test_save_fit_outputs_writes_a_consistent_npz_for_every_configuration(config_key, tmp_path):
    """The save path runs to completion for I, II and III and writes one schema.

    Exercises the block that killed the first real grid cell: the fit itself was
    fine, the NPZ write was not, and only configuration I collided -- an unfixed
    grid would have produced a mixed NPZ schema (#2089).
    """
    import configs

    from tengri.inference.posterior import Posterior

    try:
        ssp = configs.load_ssp_for(config_key)
    except FileNotFoundError:
        pytest.skip(f"SSP grid for config {config_key} not found")

    photometry = tengri.Photometry.from_names(CANDELS_13097_FILTERS)
    observation = tengri.Observation(photometry=photometry)
    sed_model = getattr(configs, f"config_{config_key}")(ssp, observation, CANDELS_13097_Z)
    free_params = list(sed_model.spec.free_params)

    # Prior draws: physically valid for ``predict`` by construction.
    batch = sed_model.spec.sample_batch(jax.random.PRNGKey(0), SAVE_PATH_DRAWS)
    samples = {k: np.asarray(batch[k], dtype=float) for k in free_params}
    assert all(v.shape == (SAVE_PATH_DRAWS,) for v in samples.values())

    posterior = Posterior(
        samples=samples,
        params={k: float(v[0]) for k, v in samples.items()},
        method="mcmc_nuts",
        wall_time_s=1.0,
        diagnostics={"n_divergent": 0, "n_samples": SAVE_PATH_DRAWS, "n_chains": 1},
        _model=sed_model,
    )
    diagnostics = {
        "gal_id": 13097,
        "config": config_key,
        "n_free": len(free_params),
        "divergences": 0,
        "rhat_max": 1.0031,
        "ess_min": 118.0,
        "wall_time_s": 1.0,
        "adoption_pass": True,
    }
    obs_fnu = np.full(len(CANDELS_13097_FILTERS), 1e-28)
    obs_sigma = obs_fnu * 0.05

    npz_path, json_path = fit_one.save_fit_outputs(
        posterior,
        diagnostics,
        [dict(diagnostics)],
        [],
        sed_model,
        config_key,
        13097,
        tmp_path,
        obs_fnu=obs_fnu,
        obs_sigma=obs_sigma,
        filter_names=CANDELS_13097_FILTERS,
    )
    assert npz_path == tmp_path / f"13097_{config_key}.npz"
    assert json_path == tmp_path / f"13097_{config_key}.json"

    # allow_pickle=False, and every array actually read: an object array (the old
    # ``filter_names``) raises here rather than at some later reader's expense.
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}

    # Not one sampled parameter was overwritten by a derived quantity.
    for name, draws in samples.items():
        assert name in arrays, name
        np.testing.assert_array_equal(arrays[name], draws)

    expected_dust_param = "dust_tau_v" if config_key == "I" else "dust_tau_diff"
    assert expected_dust_param in free_params
    assert str(arrays["dust_tau_name"]) == expected_dust_param
    # The derived dust array is that parameter's draws, strided over the record.
    np.testing.assert_array_equal(arrays["dust_tau"], samples[expected_dust_param])

    derived_keys = {"stellar_mass", "sfr_100myr", "sfr_10myr", "dust_tau", "dust_tau_name"}
    # A real ``Posterior.samples`` carries EVERY parameter, not only the free ones
    # (the real configuration II NPZ has 41 parameter arrays for 8 free parameters),
    # so the derived and grid keys must miss the whole spec, not just ``free_params``.
    all_params = set(free_params) | set(sed_model.spec.get_fixed_values())
    assert len(all_params) > len(free_params)
    grid_keys = {
        "sfh_lookback_time_yr",
        "sfh_sfr_median",
        "sfh_sfr_p16",
        "sfh_sfr_p84",
        "model_photometry_median",
        "obs_fnu",
        "obs_sigma",
        "filter_names",
    }
    assert not derived_keys & all_params, "a derived key shadows a model parameter"
    assert not grid_keys & all_params, "a grid key shadows a model parameter"
    for name in ("stellar_mass", "sfr_100myr", "sfr_10myr", "dust_tau"):
        assert arrays[name].shape == (SAVE_PATH_DRAWS,), name
        assert np.isfinite(arrays[name]).all(), name

    for name in ("sfh_lookback_time_yr", "sfh_sfr_median", "sfh_sfr_p16", "sfh_sfr_p84"):
        assert arrays[name].shape == (SAVE_PATH_SFH_GRID,), name
        assert np.isfinite(arrays[name]).all(), name
    assert (arrays["sfh_sfr_p16"] <= arrays["sfh_sfr_median"] + 1e-30).all()
    assert (arrays["sfh_sfr_median"] <= arrays["sfh_sfr_p84"] + 1e-30).all()

    n_bands = len(CANDELS_13097_FILTERS)
    assert arrays["model_photometry_median"].shape == (n_bands,)
    assert np.isfinite(arrays["model_photometry_median"]).all()
    np.testing.assert_array_equal(arrays["obs_fnu"], obs_fnu)
    np.testing.assert_array_equal(arrays["obs_sigma"], obs_sigma)
    assert list(arrays["filter_names"]) == CANDELS_13097_FILTERS

    # The NPZ holds nothing else: the schema is the same for every configuration.
    assert set(arrays) == set(free_params) | derived_keys | grid_keys

    payload = json.loads(json_path.read_text())
    assert payload["attempts"] == [diagnostics]
    assert payload["retune_history"] == []
    for name, value in diagnostics.items():
        assert payload[name] == value


def test_retune_policy_raises_target_accept_then_lengthens_warmup_and_never_toggles_dense():
    """A retune moves the step size twice, then the warmup -- never the mass matrix.

    Measured on grid cell 13097/II (600 warmup + 4x600 draws, D = 8): attempt 1
    with a diagonal mass matrix gave 3/2400 divergences at max R-hat 1.0014, and
    the old retune's ``dense_mass_matrix=True`` turned that into 79/2400 at
    1.023 (#2089). Cell 13097/III (D = 11) then missed on 77/2400 divergences at
    max R-hat 1.012 after 5741 s. Percent-level divergences are a step-size
    problem, so the target rises to 0.95 and then to 0.99 -- both at the base
    warmup, the cheap end of the ladder -- before any attempt pays for a longer
    warmup.
    """
    import inspect

    base = {
        "n_warmup": 600,
        "n_samples": 600,
        "n_chains": 4,
        "dense_mass_matrix": False,
        "target_accept_rate": 0.85,
    }
    frozen = dict(base)

    assert fit_one.retune_settings(1, base) == base
    assert fit_one.retune_settings(2, base) == {**base, "target_accept_rate": 0.95}
    # Attempt 3 raises the target again on the SAME warmup: a second step-size
    # attempt costs one base-warmup run, a longer warmup costs two.
    assert fit_one.retune_settings(3, base) == {**base, "target_accept_rate": 0.99}
    assert fit_one.retune_settings(4, base) == {
        **base,
        "target_accept_rate": 0.99,
        "n_warmup": 1200,
    }
    assert fit_one.retune_settings(5, base) == {
        **base,
        "target_accept_rate": 0.99,
        "n_warmup": 2400,
    }

    # The mass matrix is diagonal in every attempt, and ``base`` is never mutated:
    # each call returns a new dict.
    for attempt in (1, 2, 3, 4, 5):
        settings = fit_one.retune_settings(attempt, base)
        assert settings["dense_mass_matrix"] is False, attempt
        assert settings is not base
    assert base == frozen

    assert fit_one.DEFAULT_TARGET_ACCEPT == 0.85
    assert fit_one.RETUNE_TARGET_ACCEPT_1 == 0.95
    assert fit_one.RETUNE_TARGET_ACCEPT_2 == 0.99
    assert fit_one.DEFAULT_RETUNE_ATTEMPTS == 3
    assert (
        inspect.signature(fit_one.run_fit).parameters["retune_attempts"].default
        == fit_one.DEFAULT_RETUNE_ATTEMPTS
    )
    # The policy is the one the fit loop actually uses, not a parallel definition.
    assert "retune_settings(" in inspect.getsource(fit_one.run_fit)


def test_run_fit_keeps_the_best_attempt_when_the_bar_is_missed():
    """The best attempt is the one with fewest divergences, then lowest max R-hat.

    Cell 13097/II produced a near-passing attempt 1 (3 divergences, R-hat 1.0014)
    and a much worse retune, and the two-attempt cap discarded both (#2089).
    """
    import inspect

    attempts = [
        {"divergences": 3, "rhat_max": 1.0014, "retune_attempt": 1},
        {"divergences": 79, "rhat_max": 1.023, "retune_attempt": 2},
        {"divergences": 3, "rhat_max": 1.0020, "retune_attempt": 3},
    ]
    assert fit_one.select_best_attempt(attempts) == 0
    # Order-independent, and a tie on divergences is broken by R-hat.
    assert fit_one.select_best_attempt(list(reversed(attempts))) == 2
    assert fit_one.select_best_attempt([attempts[2], attempts[0]]) == 1
    assert fit_one.select_best_attempt([attempts[1]]) == 0

    # Divergences come first: an attempt that diverges 50 times does not win on
    # R-hat alone. This is the case that separates the ruled key from ``rhat_max``.
    diverging = [
        {"divergences": 0, "rhat_max": 1.05, "retune_attempt": 1},
        {"divergences": 50, "rhat_max": 1.001, "retune_attempt": 2},
    ]
    assert fit_one.select_best_attempt(diverging) == 0

    with pytest.raises(ValueError, match="no attempts"):
        fit_one.select_best_attempt([])

    # The selection is the one the fit loop actually uses, not a parallel
    # definition: ``run_fit`` reaches it through the one save-the-best helper,
    # which is also what writes the interim NPZ after every missed attempt.
    assert "save_best_so_far(" in inspect.getsource(fit_one.run_fit)
    assert "select_best_attempt(" in inspect.getsource(fit_one.save_best_so_far)


#: Draws in each stubbed posterior of the save-after-every-attempt test. The real
#: save path runs three times there and costs one ``predict`` plus one
#: ``predict_state`` per draw (~0.36 s for configuration I), so the record is short;
#: the save path is indifferent to the count.
BEST_SO_FAR_DRAWS = 8
#: Divergences of the three stubbed attempts. Attempt 2 is the best of the three,
#: and it is neither the first nor the last -- so a saved NPZ that merely held the
#: newest or the oldest posterior would be caught.
BEST_SO_FAR_DIVERGENCES = (50, 20, 30)


def test_run_fit_writes_the_best_so_far_after_each_missed_attempt(catalog, tmp_path, monkeypatch):
    """A completed attempt is on disk before the next one starts (#2089).

    ``save_fit_outputs`` used to run once, after the loop. Cell 13097/III spent
    5741 s on attempt 1 and missed the bar on 77/2400 divergences; had the
    per-cell timeout killed the process during a retune, those hours would have
    left nothing but the per-attempt JSON. The sampler seam (``ForwardModel.fit``)
    is stubbed, so no NUTS runs, while the model, the retune loop and the whole
    save path are real.
    """
    import configs

    from tengri.inference.posterior import Posterior

    try:
        ssp = configs.load_ssp_for("I")
    except FileNotFoundError:
        pytest.skip("SSP grid for config I not found")

    photometry = tengri.Photometry.from_names(CANDELS_13097_FILTERS)
    observation = tengri.Observation(photometry=photometry)
    sed_model = configs.config_I(ssp, observation, CANDELS_13097_Z)
    free_params = list(sed_model.spec.free_params)

    # One set of prior draws per attempt, all different, so the NPZ on disk
    # identifies WHICH attempt was saved.
    draws = [
        {
            k: np.asarray(v, dtype=float)
            for k, v in sed_model.spec.sample_batch(
                jax.random.PRNGKey(100 + i), BEST_SO_FAR_DRAWS
            ).items()
            if k in free_params
        }
        for i in range(len(BEST_SO_FAR_DIVERGENCES))
    ]

    class _StubPosterior:
        """The surface ``run_fit`` and ``save_fit_outputs`` read off a posterior."""

        def __init__(self, samples, n_divergent):
            self.samples = samples
            self.diagnostics = {"n_divergent": n_divergent}

        def rhat(self):
            return {"dust_tau_v": 1.02}

        def effective_sample_size(self):
            return {"dust_tau_v": 120.0}

    # ``samples`` and ``diagnostics`` are pinned against the real class by the
    # save-path test, which builds a real ``Posterior``; the two methods here.
    for name in ("rhat", "effective_sample_size"):
        assert hasattr(Posterior, name), name

    fit_kwargs: list[dict] = []

    def stub_fit(self, data, **kwargs):
        index = len(fit_kwargs)
        fit_kwargs.append(dict(kwargs))
        return _StubPosterior(draws[index], BEST_SO_FAR_DIVERGENCES[index])

    monkeypatch.setattr(fit_one.ForwardModel, "fit", stub_fit)

    # The wrapper records the attempt each save was handed and then calls the real
    # save path, so the assertions below are about files that were really written.
    real_save = fit_one.save_fit_outputs
    saved_attempts: list[int] = []
    on_disk_attempts: list[int] = []
    npz_path = tmp_path / "13097_I.npz"
    json_path = tmp_path / "13097_I.json"

    def recording_save(posterior, diagnostics, attempts, retune_history, *args, **kwargs):
        saved_attempts.append(diagnostics["best_attempt"])
        result = real_save(posterior, diagnostics, attempts, retune_history, *args, **kwargs)
        on_disk_attempts.append(json.loads(json_path.read_text())["best_attempt"])
        return result

    monkeypatch.setattr(fit_one, "save_fit_outputs", recording_save)

    payload = fit_one.run_fit(
        13097, "I", "mcmc_nuts", tmp_path, n_warmup=4, n_samples=4, n_chains=1
    )

    # THE POINT: attempt 1's posterior reached the NPZ before attempt 2 started,
    # and attempt 2 replaced it. Without the interim save this is ``[2]``.
    assert saved_attempts == [1, 2, 2]
    assert on_disk_attempts == [1, 2, 2]

    # The loop really climbed the ladder while doing it.
    assert [c["target_accept_rate"] for c in fit_kwargs] == [0.85, 0.95, 0.99]
    assert [c["n_warmup"] for c in fit_kwargs] == [4, 4, 4]
    assert [c["dense_mass_matrix"] for c in fit_kwargs] == [False, False, False]

    assert npz_path.exists()
    final = json.loads(json_path.read_text())
    assert final["best_attempt"] == 2
    assert final["adoption_pass"] is False
    assert len(final["attempts"]) == 3
    assert [a["divergences"] for a in final["attempts"]] == list(BEST_SO_FAR_DIVERGENCES)
    assert payload["best_attempt"] == 2
    assert payload["adoption_pass"] is False

    # The NPZ holds the BEST attempt's draws, not the newest ones.
    with np.load(npz_path, allow_pickle=False) as npz:
        for name, values in draws[1].items():
            np.testing.assert_array_equal(npz[name], values)


def test_only_missing_skips_adopted_cells(tmp_path):
    """``--only-missing`` skips a cell whose JSON says it cleared the adoption bar."""
    import run_candels_fits

    adopted = tmp_path / "13097_II.json"
    adopted.write_text(json.dumps({"gal_id": 13097, "config": "II", "adoption_pass": True}))
    missed = tmp_path / "15336_III.json"
    missed.write_text(json.dumps({"gal_id": 15336, "config": "III", "adoption_pass": False}))

    assert run_candels_fits.cell_is_adopted(adopted) is True
    assert run_candels_fits.cell_is_adopted(missed) is False
    # A cell that was never run has no JSON at all.
    assert run_candels_fits.cell_is_adopted(tmp_path / "24497_I.json") is False
    # A truncated JSON (a per-cell timeout killing the process mid-write) is not adopted.
    broken = tmp_path / "13097_I.json"
    broken.write_text('{"adoption_pass": true')
    assert run_candels_fits.cell_is_adopted(broken) is False
    # Neither is a JSON that predates the key.
    legacy = tmp_path / "15336_I.json"
    legacy.write_text(json.dumps({"gal_id": 15336, "config": "I"}))
    assert run_candels_fits.cell_is_adopted(legacy) is False

    # The flag is opt-in: without it the driver runs every cell, as today.
    assert run_candels_fits.parse_args([]).only_missing is False
    assert run_candels_fits.parse_args(["--only-missing"]).only_missing is True
