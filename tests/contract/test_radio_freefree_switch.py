# SPDX-License-Identifier: BSD-3-Clause
"""Contract: radio grammar ``freefree`` structural switch reaches RadioSEDComponentConfig.

The ``radio.sf.freefree`` key gates thermal free-free emission inclusion via
``RadioSEDComponentConfig.include_freefree``. This contract verifies:

- Grammar wiring: ``radio={'sf': {'type': 'bell2003', 'freefree': False}}``
  reaches ``model.spec.radio_include_freefree`` and, through it, the live
  ``RadioSEDComponentConfig`` in ``model._component_configs``.
- Physics: explicit ``False`` drops the Murphy+2011 thermal term, leaving
  ``sed_radio`` bit-exact with :func:`tengri.radio.radio_sfr_bell2003` in the
  radio band; the default build carries measurably more flux there.
- Defaults: an absent ``freefree`` key resolves to ``spec.radio_include_freefree
  is None``, which the component itself resolves to ``True`` for ``bell2003``.
- Errors: ``freefree=True`` with ``sfr_mode='bell2003_split'`` raises
  ``ConfigError`` (the double-counting guard, ruling R19); a non-bool value
  raises ``TypeError`` both at ``RadioSEDComponentConfig`` directly and at the
  public grammar.
- Round trip: ``freefree`` survives ``to_groups()`` and ``spec.summary()``.
- Compile identity: the switch is structural, not a parameter -- two models
  differing only in ``freefree`` must not share a ``compile_signature()``.

Uses a synthetic SSP (module-level ``_synthetic_ssp``, mirroring
``test_pipeline_wiring.py``) reaching the radio band, plus
``synthetic_tophat_obs`` from ``conftest.py`` -- both CI-runnable without the
gitignored ``data/ssp_*.h5`` grids, unlike the ``ssp_data_bc03`` fixture (skips
when the file is absent, which it is in CI).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.radio.component import RadioSEDComponentConfig
from tengri.config.exceptions import ConfigError
from tengri.radio import radio_sfr_bell2003

pytestmark = pytest.mark.contract

#: Matches ``tengri.components.radio.radio._RADIO_WAVE_MIN_AA`` (1 mm): the
#: SED is zero shortward of this by construction, so any comparison must be
#: restricted to the radio band or a false "extra term" would show up as a
#: mismatch in the non-radio wavelength range where BOTH sides are already
#: (correctly) zero for an unrelated reason.
_RADIO_WAVE_MIN_AA = 1.0e7


def _synthetic_ssp(log_wave_max: float = 11.0, n_wave: int = 900):
    """Smooth synthetic SSP reaching the radio band (default: 100 A - 1e11 A)."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(2.0, log_wave_max, n_wave)
    ages_gyr = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


@pytest.fixture(scope="module")
def synthetic_radio_ssp():
    return _synthetic_ssp()


def _build_radio_model(ssp, obs, freefree=None, *, sf_type="bell2003"):
    """A dusty, radio-emitting model with the given ``sf.freefree`` setting.

    Mirrors ``test_pipeline_wiring._radio_model``: const SFH, two-component
    Calzetti attenuation, Draine & Li (2014) dust re-emission (the source of
    the ``L_ir`` the FIR-radio correlation scales against), and Bell (2003)
    synchrotron + power-law AGN radio. ``radio_q_ir``/``radio_alpha_sf`` are
    pinned so the direct-formula comparison below is exact rather than
    sample-dependent; ``freefree`` is left off the ``sf`` sub-dict entirely
    when ``None``, matching the grammar's "absent key" case.
    """
    sf_dict = {"type": sf_type}
    if freefree is not None:
        sf_dict["freefree"] = freefree
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.0),
            sfh={"type": "const", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti"},
            dust_emission={"type": "draine_li2014"},
            radio={
                "sf": sf_dict,
                "agn": {"type": "powerlaw"},
                "all_params": FREE,
                "q_ir": Fixed(2.5),
                "alpha_sf": Fixed(0.8),
            },
        )


def _radio_config(model) -> RadioSEDComponentConfig:
    """The live ``RadioSEDComponentConfig`` off ``model._component_configs``."""
    for name, config in model._component_configs:
        if name == "radio":
            return config
    raise AssertionError("radio component not found in model._component_configs")


# ── Grammar -> spec -> component wiring ────────────────────────────
def test_freefree_false_reaches_spec_and_component(synthetic_radio_ssp, synthetic_tophat_obs):
    """Grammar ``freefree: False`` reaches ``model.spec`` and the live component config."""
    model = _build_radio_model(synthetic_radio_ssp, synthetic_tophat_obs, freefree=False)

    assert model.spec.radio_include_freefree is False, (
        "spec.radio_include_freefree must be False when passed via grammar"
    )
    config = _radio_config(model)
    assert isinstance(config, RadioSEDComponentConfig)
    assert config.include_freefree is False, "config.include_freefree must be False"


def test_freefree_absent_key_resolves_to_true_for_bell2003(
    synthetic_radio_ssp, synthetic_tophat_obs
):
    """Absent ``freefree`` key leaves ``spec`` at ``None``; the component resolves it."""
    model = _build_radio_model(synthetic_radio_ssp, synthetic_tophat_obs, freefree=None)

    assert model.spec.radio_include_freefree is None, (
        "spec.radio_include_freefree must be None when the freefree key is absent"
    )
    config = _radio_config(model)
    assert config.include_freefree is True, (
        "RadioSEDComponentConfig must resolve an unset include_freefree to True "
        "for sfr_mode='bell2003'"
    )


# ── Physics ─────────────────────────────────────────────────────────
def test_freefree_false_drops_the_thermal_term(synthetic_radio_ssp, synthetic_tophat_obs):
    """``freefree=False`` leaves ``sed_radio`` bit-exact with the synchrotron-only formula."""
    model_no_ff = _build_radio_model(synthetic_radio_ssp, synthetic_tophat_obs, freefree=False)
    params_no_ff = model_no_ff.spec.sample(jax.random.PRNGKey(0))
    state_no_ff = model_no_ff.predict_state(params_no_ff)

    model_default = _build_radio_model(synthetic_radio_ssp, synthetic_tophat_obs, freefree=None)
    params_default = model_default.spec.sample(jax.random.PRNGKey(0))
    state_default = model_default.predict_state(params_default)

    L_ir = float(state_no_ff.derived["L_ir"])
    # radio_q_ir / radio_alpha_sf are Fixed (see _build_radio_model), so they
    # are absent from spec.sample()'s free-only output (#2296); read the
    # pinned values directly.
    fixed_no_ff = model_no_ff.spec.get_fixed_values()
    q_ir = float(fixed_no_ff["radio_q_ir"])
    alpha_sf = float(fixed_no_ff["radio_alpha_sf"])
    wave = np.asarray(state_no_ff.wave)
    sed_radio_no_ff = np.asarray(state_no_ff.derived["sed_radio"])

    synchrotron_only = np.asarray(radio_sfr_bell2003(wave, L_ir, q_ir, alpha_sf))

    radio_mask = wave > _RADIO_WAVE_MIN_AA
    assert np.any(radio_mask), "no radio-band wavelength points in the model's grid"
    np.testing.assert_allclose(
        sed_radio_no_ff[radio_mask],
        synchrotron_only[radio_mask],
        rtol=1e-8,
        err_msg="freefree=False must produce synchrotron only (no thermal term)",
    )

    sed_radio_default = np.asarray(state_default.derived["sed_radio"])
    ratio = sed_radio_default[radio_mask] / sed_radio_no_ff[radio_mask]
    assert np.all(ratio >= 1.0), "the default build must produce SED >= freefree=False"
    assert np.mean(ratio) > 1.01, (
        "the default build should have noticeably more flux than freefree=False"
    )


# ── Errors ────────────────────────────────────────────────────────
def test_freefree_true_with_split_raises_config_error(synthetic_radio_ssp, synthetic_tophat_obs):
    """Explicit ``freefree=True`` with ``bell2003_split`` raises ``ConfigError``."""
    with pytest.raises(ConfigError, match="bell2003_split"):
        _build_radio_model(
            synthetic_radio_ssp,
            synthetic_tophat_obs,
            freefree=True,
            sf_type="bell2003_split",
        )


def test_freefree_non_bool_raises_at_component():
    """Non-bool ``include_freefree`` raises ``TypeError`` at the config dataclass."""
    with pytest.raises(TypeError):
        RadioSEDComponentConfig(sfr_mode="bell2003", include_freefree="false")
    with pytest.raises(TypeError):
        RadioSEDComponentConfig(sfr_mode="bell2003", include_freefree=1)
    with pytest.raises(TypeError):
        RadioSEDComponentConfig(sfr_mode="bell2003", include_freefree=[False])


def test_freefree_non_bool_raises_at_grammar(synthetic_radio_ssp, synthetic_tophat_obs):
    """Non-bool ``radio.sf.freefree`` raises ``TypeError`` from ``SEDModel.build``."""
    with pytest.raises(TypeError):
        _build_radio_model(synthetic_radio_ssp, synthetic_tophat_obs, freefree="false")


# ── Round trip ──────────────────────────────────────────────────────
def test_freefree_round_trips_through_to_groups_and_summary(
    synthetic_radio_ssp, synthetic_tophat_obs
):
    """``freefree`` survives ``to_groups()`` and ``spec.summary()`` runs cleanly."""
    model = _build_radio_model(synthetic_radio_ssp, synthetic_tophat_obs, freefree=False)

    groups = model.spec.to_groups()
    assert "radio" in groups, "radio group must be in to_groups() output"
    assert "sf" in groups["radio"], "radio.sf sub-block must be in to_groups() output"
    assert groups["radio"]["sf"]["freefree"] is False, (
        "freefree=False must round-trip through to_groups()"
    )

    model.spec.summary()


# ── Compile identity ────────────────────────────────────────────────
def test_freefree_changes_the_compile_signature(synthetic_radio_ssp, synthetic_tophat_obs):
    """Two models differing only in ``freefree`` must not share a compiled kernel."""
    model_true = _build_radio_model(synthetic_radio_ssp, synthetic_tophat_obs, freefree=True)
    model_false = _build_radio_model(synthetic_radio_ssp, synthetic_tophat_obs, freefree=False)

    assert model_true.compile_signature() != model_false.compile_signature(), (
        "freefree=True and freefree=False must compile to different signatures -- "
        "sharing one would hand one model's kernel to the other"
    )


# ── Nebular backend continuum rule (#2346) ────────────────────────────────
def _build_cue_radio_model(ssp, obs, freefree=None):
    """A Cue nebular + radio model with the given ``sf.freefree`` setting.

    Like ``_build_radio_model``, but with ``neb={'type': 'cue', ...}`` to test
    the factory-level auto rule (#2346): when the declared nebular backend
    carries a free-free continuum (Cue, CloudyGrid), ``include_freefree=None``
    resolves to ``False`` at construction, so the radio block does not add a
    second thermal term on top of the nebular continuum.

    Requires the Cue weights file and the git-tracked bare-stellar FSPS PRSC SSP.
    """
    from tengri._data_setup import find_data_str

    cue_weights = find_data_str("cue_weights.npz")
    if cue_weights is None:
        pytest.skip("Cue weights file not found")

    sf_dict = {"type": "bell2003"}
    if freefree is not None:
        sf_dict["freefree"] = freefree
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.0),
            sfh={"type": "const", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti"},
            dust_emission={"type": "draine_li2014"},
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            radio={
                "sf": sf_dict,
                "agn": {"type": "powerlaw"},
                "all_params": FREE,
                "q_ir": Fixed(2.5),
                "alpha_sf": Fixed(0.8),
            },
        )


def test_freefree_absent_key_resolves_to_false_with_cue_continuum(
    ssp_data_fsps, synthetic_tophat_obs
):
    """When Cue (free-free bearing nebular) is declared, None → False automatically."""
    model = _build_cue_radio_model(ssp_data_fsps, synthetic_tophat_obs, freefree=None)

    assert model.spec.radio_include_freefree is None, (
        "spec.radio_include_freefree must be None when the freefree key is absent"
    )
    config = _radio_config(model)
    assert config.include_freefree is False, (
        "RadioSEDComponentConfig must resolve None to False when the declared "
        "nebular backend carries a free-free continuum (Cue carries one)"
    )


def test_freefree_absent_key_stays_true_for_cb19(synthetic_radio_ssp, synthetic_tophat_obs):
    """CB19 has no continuum, so None stays True (no nebular free-free to conflict).

    Asserts the predicate directly that CB19 carries no nebular continuum,
    and verifies the model-level check skips when the resolved CB19 grid is
    the flat placeholder (git-tracked data/cb19_templates.h5).
    """
    from tengri._data_setup import find_data_str
    from tengri.components.nebular._models import nebular_backend_carries_freefree
    from tengri.components.nebular.cloudy_cb19 import CB19DegenerateGridError

    cb19_grid = find_data_str("cb19_templates.h5")
    if cb19_grid is None:
        pytest.skip("CB19 templates file not found")

    assert nebular_backend_carries_freefree("cb19") is False, (
        "cb19 publishes no nebular continuum, so it must not switch the radio free-free term off"
    )

    sf_dict = {"type": "bell2003"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SEDModel.build(
                ssp_data=synthetic_radio_ssp,
                observation=synthetic_tophat_obs,
                redshift=Fixed(0.0),
                sfh={"type": "const", "all_params": FREE},
                dust_attenuation={"type": "two_component", "law": "calzetti"},
                dust_emission={"type": "draine_li2014"},
                neb={"type": "cb19", "all_params": Fixed(DEFAULT)},
                radio={
                    "sf": sf_dict,
                    "agn": {"type": "powerlaw"},
                    "all_params": FREE,
                    "q_ir": Fixed(2.5),
                    "alpha_sf": Fixed(0.8),
                },
            )
        except CB19DegenerateGridError as exc:
            assert "flat placeholder" in str(exc), (
                f"Unexpected CB19DegenerateGridError (not the placeholder): {str(exc)[:120]}"
            )
            pytest.skip(f"cb19 grid resolved to the flat placeholder: {str(exc)[:120]}")

    assert model.spec.radio_include_freefree is None
    config = _radio_config(model)
    assert config.include_freefree is True, (
        "RadioSEDComponentConfig must resolve None to True when the nebular "
        "backend does NOT carry a free-free continuum (CB19 publishes zeros)"
    )


def test_explicit_freefree_true_overrides_cue_auto_rule(ssp_data_fsps, synthetic_tophat_obs):
    """Explicit ``freefree=True`` overrides the Cue auto-rule; SED differs from False."""
    model_explicit_true = _build_cue_radio_model(
        ssp_data_fsps, synthetic_tophat_obs, freefree=True
    )
    model_explicit_false = _build_cue_radio_model(
        ssp_data_fsps, synthetic_tophat_obs, freefree=False
    )

    config_true = _radio_config(model_explicit_true)
    config_false = _radio_config(model_explicit_false)

    assert config_true.include_freefree is True
    assert config_false.include_freefree is False

    # Verify that the SED differs above 1 mm (radio band starts at 1 mm)
    params_true = model_explicit_true.spec.sample(jax.random.PRNGKey(0))
    state_true = model_explicit_true.predict_state(params_true)

    params_false = model_explicit_false.spec.sample(jax.random.PRNGKey(0))
    state_false = model_explicit_false.predict_state(params_false)

    wave = np.asarray(state_true.wave)
    sed_radio_true = np.asarray(state_true.derived["sed_radio"])
    sed_radio_false = np.asarray(state_false.derived["sed_radio"])

    radio_mask = wave > _RADIO_WAVE_MIN_AA
    assert np.any(radio_mask), "no radio-band wavelength points in the model's grid"
    assert not np.allclose(sed_radio_true[radio_mask], sed_radio_false[radio_mask], rtol=1e-8), (
        "explicit freefree=True must produce a different sed_radio than freefree=False"
    )


def test_one_thermal_term_contract_with_cue(ssp_data_fsps, synthetic_tophat_obs):
    """Cue model absent freefree key = Cue model explicit False; exactly one thermal term."""
    model_absent = _build_cue_radio_model(ssp_data_fsps, synthetic_tophat_obs, freefree=None)
    model_explicit_false = _build_cue_radio_model(
        ssp_data_fsps, synthetic_tophat_obs, freefree=False
    )

    params_absent = model_absent.spec.sample(jax.random.PRNGKey(0))
    state_absent = model_absent.predict_state(params_absent)

    params_explicit_false = model_explicit_false.spec.sample(jax.random.PRNGKey(0))
    state_explicit_false = model_explicit_false.predict_state(params_explicit_false)

    sed_radio_absent = np.asarray(state_absent.derived["sed_radio"])
    sed_radio_explicit = np.asarray(state_explicit_false.derived["sed_radio"])

    sed_nebular_absent = np.asarray(state_absent.derived["sed_nebular"])
    sed_nebular_explicit = np.asarray(state_explicit_false.derived["sed_nebular"])

    np.testing.assert_array_equal(
        sed_radio_absent,
        sed_radio_explicit,
        err_msg="absent freefree key and explicit freefree=False must produce identical sed_radio",
    )
    np.testing.assert_array_equal(
        sed_nebular_absent,
        sed_nebular_explicit,
        err_msg=(
            "absent freefree key and explicit freefree=False must produce identical sed_nebular"
        ),
    )
