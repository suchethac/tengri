# SPDX-License-Identifier: BSD-3-Clause
"""Composable shock nebular emission: grammar wiring + composition (#851).

Guards that the top-level ``shock={...}`` grammar group reaches the live
forward model and composes *additively* with any photoionized nebular backend
— the surface is not the silent no-op it was before #851, where
``Parameters(shock=True)`` registered ``shock_*`` params that ``predict()``
ignored entirely (only the legacy ``nonstell`` kernel — dead code removed in
#922 — composed shock).

Covers:

* **Physics equivalence** — the composable component's ``norm="frac"`` path
  reproduces :func:`tengri.forward.emission_helpers.shock_emission` bit-for-bit
  (the legacy path it supersedes).
* **Both knobs** — relative ``frac`` (scales the galaxy Hα) and absolute
  ``log_lhalpha`` (decoupled from the SFR).
* **Composition** — shock and photoionized emission are independent and
  additive: toggling shock leaves ``sed_nebular`` untouched and vice versa.
* **Grammar wiring** — ``shock={...}`` builds; ``type="none"`` disables; the
  normalization/abundance/component knobs enter the ``compile_signature`` so
  the kernel cache does not color-leak.
* **Path equivalence** — the grammar surface and the low-level
  ``Parameters(shock=True)`` escape hatch produce the same SED.

Uses the synthetic wide SSP + the hardcoded Allen+2008 shock fallback (active
when ``data/mappings_templates.h5`` is absent), so the structural + composition
tests run in default CI. Tests that need the *full* MAPPINGS grid are marked
separately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.nebular.shock_model import ShockNebular, ShockNebularConfig
from tengri.forward.emission_helpers import shock_emission
from tengri.parameters.parameters import Parameters
from tengri.utils.physics_constants import C_AA

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


# ─────────────────────────────────────────────────────────────────────
# Component-level: physics equivalence + both knobs (no SSP needed)
# ─────────────────────────────────────────────────────────────────────


def test_frac_path_matches_legacy_shock_emission_bit_exact():
    """``norm="frac"`` reproduces the legacy ``shock_emission`` helper exactly.

    Both use the same order-of-magnitude Hα proxy
    (``L(Hα) ~ 1e-3 L_bol`` of ``sed_in``) and the same ``compute_shock_sed``
    template, so the component-chain path introduces zero physics drift.
    """
    wave = np.geomspace(1.0e3, 1.0e7, 4000)
    sed_in = np.geomspace(1.0e28, 1.0e20, 4000)
    kw = dict(shock_velocity=350.0, shock_log_density=0.5, shock_b_over_sqrt_n=1.0)

    legacy = np.asarray(
        shock_emission(jnp.asarray(wave), jnp.asarray(sed_in), shock_frac=0.7, **kw)
    )
    comp = ShockNebular()  # default norm="frac"
    p = {
        "frac": jnp.asarray(0.7),
        "log_lhalpha": jnp.asarray(41.0),
        "velocity": jnp.asarray(350.0),
        "log_density": jnp.asarray(0.5),
        "b_over_sqrt_n": jnp.asarray(1.0),
    }
    sed_out, published = comp.predict(p, jnp.asarray(sed_in), jnp.asarray(wave))
    mine = np.asarray(published["sed_shock"])

    assert np.array_equal(mine, legacy)
    assert np.allclose(np.asarray(sed_out), sed_in + mine)


def test_absolute_knob_is_independent_of_sed_in():
    """``norm="lhalpha"`` sets the Hα anchor absolutely, ignoring ``sed_in``."""
    wave = np.geomspace(1.0e3, 1.0e7, 2000)
    comp = ShockNebular()
    comp.config = ShockNebularConfig(norm="lhalpha")
    p = {
        "frac": jnp.asarray(0.0),
        "log_lhalpha": jnp.asarray(41.5),
        "velocity": jnp.asarray(300.0),
        "log_density": jnp.asarray(0.0),
        "b_over_sqrt_n": jnp.asarray(1.0),
    }
    _, pub_lo = comp.predict(p, jnp.full(wave.shape, 1.0e28), jnp.asarray(wave))
    _, pub_hi = comp.predict(p, jnp.full(wave.shape, 1.0e30), jnp.asarray(wave))
    assert np.allclose(np.asarray(pub_lo["sed_shock"]), np.asarray(pub_hi["sed_shock"]))
    assert bool(np.any(np.asarray(pub_lo["sed_shock"]) > 0))


def test_frac_knob_scales_with_galaxy_luminosity():
    """``norm="frac"`` ties the shock anchor to the galaxy L_bol (``sed_in``)."""
    wave = np.geomspace(1.0e3, 1.0e7, 2000)
    comp = ShockNebular()  # norm="frac"
    p = {
        "frac": jnp.asarray(0.5),
        "log_lhalpha": jnp.asarray(41.0),
        "velocity": jnp.asarray(300.0),
        "log_density": jnp.asarray(0.0),
        "b_over_sqrt_n": jnp.asarray(1.0),
    }
    _, pub_lo = comp.predict(p, jnp.full(wave.shape, 1.0e28), jnp.asarray(wave))
    _, pub_hi = comp.predict(p, jnp.full(wave.shape, 1.0e30), jnp.asarray(wave))
    lo = float(np.nanmax(np.asarray(pub_lo["sed_shock"])))
    hi = float(np.nanmax(np.asarray(pub_hi["sed_shock"])))
    # 100× brighter galaxy → 100× brighter shock (linear in L_bol).
    assert hi > lo
    assert np.isclose(hi / lo, 100.0, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────
# Grammar wiring (synthetic SSP; no full grid needed)
# ─────────────────────────────────────────────────────────────────────


def _build(ssp, obs, shock=None, neb=None):
    return tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
        neb=neb if neb is not None else {"type": "none"},
        shock=shock,
        redshift=tengri.Fixed(0.1),
    )


def test_grammar_activates_shock(synthetic_ssp_wide, synthetic_tophat_obs):
    m = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        shock={
            "norm": "frac",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "frac": tengri.Fixed(0.3),
        },
    )
    assert m.spec.shock is True
    assert m.spec.shock_norm == "frac"
    shock_params = {p for p in m.spec.all_params if p.startswith("shock_")}
    assert {"shock_frac", "shock_log_lhalpha", "shock_velocity"} <= shock_params


def test_grammar_explicit_priors_free_shock_params(synthetic_ssp_wide, synthetic_tophat_obs):
    m = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        shock={
            "norm": "frac",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "frac": tengri.Uniform(0.0, 1.0),
            "velocity": tengri.Uniform(150.0, 800.0),
        },
    )
    free = {p for p in m.spec.free_params if p.startswith("shock_")}
    assert free == {"shock_frac", "shock_velocity"}


def test_grammar_type_none_disables(synthetic_ssp_wide, synthetic_tophat_obs):
    m = _build(synthetic_ssp_wide, synthetic_tophat_obs, shock={"type": "none"})
    assert m.spec.shock is False


def test_norm_enters_compile_signature(synthetic_ssp_wide, synthetic_tophat_obs):
    """The normalization mode changes the SED, so it must be part of the
    structural fingerprint — otherwise a ``frac`` model would silently reuse a
    cached ``lhalpha`` kernel (kernel-cache color-leak)."""
    m_frac = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        shock={
            "norm": "frac",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "frac": tengri.Fixed(0.3),
        },
    )
    m_abs = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        shock={
            "norm": "lhalpha",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "log_lhalpha": tengri.Fixed(41.0),
        },
    )
    assert m_frac.compile_signature() != m_abs.compile_signature()


def test_invalid_norm_raises(synthetic_ssp_wide, synthetic_tophat_obs):
    with pytest.raises(ValueError, match="shock norm"):
        _build(
            synthetic_ssp_wide,
            synthetic_tophat_obs,
            shock={"norm": "bogus", "all_params": tengri.Fixed(tengri.DEFAULT)},
        )


def test_shock_group_round_trips(synthetic_ssp_wide, synthetic_tophat_obs):
    """``to_groups()`` emits the shock group (type + norm + priors) and rebuilds
    to the same active shock config."""
    m = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        shock={
            "norm": "lhalpha",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "log_lhalpha": tengri.Uniform(38.0, 44.0),
            "velocity": tengri.Fixed(350.0),
        },
    )
    groups = m.spec.to_groups()
    assert groups["shock"]["type"] == "mappings"
    assert groups["shock"]["norm"] == "lhalpha"

    rebuilt = tengri.SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        **{k: v for k, v in groups.items() if k != "redshift"},
        redshift=tengri.Fixed(0.1),
    )
    assert rebuilt.spec.shock is True
    assert rebuilt.spec.shock_norm == "lhalpha"
    assert "shock_log_lhalpha" in rebuilt.spec.free_params


# ─────────────────────────────────────────────────────────────────────
# Composition through the live forward model (synthetic SSP + shock fallback)
# ─────────────────────────────────────────────────────────────────────


def _rest(model, params):
    out = model.predict_rest_sed(params)
    return np.asarray(out.wavelength), np.asarray(out.sed)


def test_shock_composes_in_component_chain(synthetic_ssp_wide, synthetic_tophat_obs):
    """The regression that #851 fixes: ``shock_frac`` is NOT a no-op in the
    live component-chain. Toggling it changes ``predict_rest_sed`` output."""
    m = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        neb={"type": "none"},
        shock={
            "norm": "frac",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "frac": tengri.Fixed(0.8),
            "velocity": tengri.Fixed(400.0),
        },
    )
    theta = dict(m.spec.sample(jax.random.PRNGKey(0)))
    _, sed_off = _rest(m, {**theta, "shock_frac": 0.0})
    _, sed_on = _rest(m, {**theta, "shock_frac": 0.8})
    assert not np.allclose(sed_off, sed_on)
    assert float(np.nanmax(np.abs(sed_on - sed_off))) > 0.0


def test_shock_and_photoionized_are_independent(synthetic_ssp_wide, synthetic_tophat_obs):
    """Composition: the shock contribution does not perturb the photoionized
    continuum, and both are present when composed."""
    m = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        neb={"type": "ssp"},  # baked-in photoionized (no external grid needed)
        shock={
            "norm": "frac",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "frac": tengri.Fixed(0.8),
            "velocity": tengri.Fixed(400.0),
        },
    )
    theta = dict(m.spec.sample(jax.random.PRNGKey(0)))
    st_off = m.predict_state({**theta, "shock_frac": 0.0})
    st_on = m.predict_state({**theta, "shock_frac": 0.8})
    neb_off = np.asarray(st_off.derived.get("sed_nebular"))
    neb_on = np.asarray(st_on.derived.get("sed_nebular"))
    sh_off = np.asarray(st_off.derived.get("sed_shock"))
    sh_on = np.asarray(st_on.derived.get("sed_shock"))
    # Photoionized part is untouched by the shock toggle.
    assert np.allclose(neb_off, neb_on)
    # Shock is off at frac=0, on at frac=0.8.
    assert float(np.nanmax(np.abs(sh_off))) == 0.0
    assert float(np.nanmax(np.abs(sh_on))) > 0.0


def test_grammar_matches_low_level_shock_flag(synthetic_ssp_wide, synthetic_tophat_obs):
    """The ``shock={...}`` grammar surface and the low-level
    ``Parameters(shock=True)`` escape hatch produce the same SED."""
    m_grammar = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        neb={"type": "none"},
        shock={
            "norm": "frac",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "frac": tengri.Fixed(0.6),
            "velocity": tengri.Fixed(350.0),
        },
    )
    spec = Parameters(
        mean_sfh_type="const",
        redshift=0.1,
        dust_model="off",
        shock=True,
        shock_norm="frac",
        shock_frac=0.6,
        shock_velocity=350.0,
    )
    m_lowlevel = tengri.SEDModel(spec, synthetic_ssp_wide, observation=synthetic_tophat_obs)

    theta = dict(m_grammar.spec.sample(jax.random.PRNGKey(1)))
    _, sed_g = _rest(m_grammar, theta)
    _, sed_l = _rest(m_lowlevel, theta)
    assert np.allclose(sed_g, sed_l, rtol=1e-10, atol=0.0)


def test_absolute_knob_composes_via_grammar(synthetic_ssp_wide, synthetic_tophat_obs):
    m = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        neb={"type": "none"},
        shock={
            "norm": "lhalpha",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "log_lhalpha": tengri.Fixed(41.0),
            "velocity": tengri.Fixed(400.0),
        },
    )
    theta = dict(m.spec.sample(jax.random.PRNGKey(0)))
    st = m.predict_state(theta)
    shock = np.asarray(st.derived.get("sed_shock"))
    assert float(np.nanmax(np.abs(shock))) > 0.0
    # Total shock luminosity exceeds the single Hα anchor (multi-line spectrum).
    nu = C_AA / np.asarray(st.wave)
    L_shock = float(np.abs(np.trapezoid(shock, nu)))
    assert L_shock > 10.0**41.0
