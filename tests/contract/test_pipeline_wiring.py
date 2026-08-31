# SPDX-License-Identifier: BSD-3-Clause
"""Every physics parameter a user can set must move the model output.

This file exists to catch **silent parameter drops** — a user sets
``agn_a_spin=0.7``, the pipeline ignores it, and the function's own default is
used instead.  The fit converges, the posterior is reported, and that
parameter's marginal is the prior.

Until 2026-08 the whole file tested that claim by grepping the pipeline
*source text* for ``params["agn_a_spin"]``.  A source scan cannot fail in the
direction that matters:

* It goes **red on a harmless refactor.**  That is #1403: the sole match for
  the qualified spelling was ``SEDModel._get_non_stellar_kwargs``, a method
  nothing called.  Deleting it as dead code turned these assertions red while
  the wiring they describe was, and remained, entirely intact.
* It stays **green on the actual bug.**  A pipeline that reads
  ``p["gamma_hmxb"]`` and then hands ``xray_total`` a hardcoded default matches
  every pattern above.  So does a read inside a dead branch, and so does the
  parameter name appearing in a comment.

The scan had already been caught matching dead code three separate times
(``emission_helpers.agn_emission``, ``_get_non_stellar_kwargs``, and the DL14
copy), and each time the repair was to broaden the pattern.  Two polar-dust
tests broke the cycle and did the right thing instead — *"vary the angle and
require the output to move"*.  This file now applies that rule to all of it.

**What replaced it.**  One measurement per parameter: build a model in which
the parameter is free, evaluate ``predict_state`` at each end of its declared
range, and require the published state to change.  A dropped parameter gives
exactly ``0.0`` and fails; every live one measured between 7.8e-4 and 1.0.

Two disciplines make that measurement trustworthy, and neither is optional:

1. **A positive control per family.**  A parameter known to matter is swept
   first, and the family is refused if *it* does not move.  Without this a
   broken fixture — an unbuilt component, a params dict that never reaches the
   model — reports every parameter as live, silently.
2. **Per-array relative change, never a global sum.**  The first version of
   this probe summed ``|x|`` across the derived state and reported four
   parameters as dead.  All four were fine: a 1e72 stellar luminosity sat in
   the same sum, and at 16 significant digits a 1e50 component cannot be seen.
   "No response" and "swamped" are the same number.

And one domain rule, learned the same way.  ``radio_T_e`` first measured
exactly flat, which reads as a dropped parameter.  It is not: the FIR-radio
correlation multiplies ``L_ir``, so with no dust component in the build the
whole radio SED is zero and there is nothing for ``T_e`` to change.  Flatness
measured outside a knob's active regime says nothing about the knob.  The
radio family here therefore carries dust, and a wavelength grid that reaches
10 m so radio has somewhere to live.  (That the zero itself is silent rather
than refused — where the AGN ``fracAGN`` analog raises ``ConfigError`` — is
#2106, not this file's subject.)
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, SEDModel, Uniform

pytestmark = pytest.mark.contract


# ── The claim under test ──────────────────────────────────────────
#: ``(param, family, lo, hi)`` for every parameter this file asserts is
#: forwarded.  ``lo``/``hi`` sit inside the declared prior (see
#: ``tools/check_param_ranges.py``); the sweep is a response test, not a
#: physics-value test, so the endpoints only need to be far enough apart to
#: move a live parameter.
_FORWARDED: tuple[tuple[str, str, float, float], ...] = (
    # AGN — disc (Kubota & Done 2018 three-zone) and clumpy torus (SKIRTOR)
    ("agn_a_spin", "agn", 0.0, 0.9),
    ("agn_cos_inc", "agn", 0.1, 0.99),
    ("agn_tau_skirtor", "agn", 3.0, 11.0),
    ("agn_p_skirtor", "agn", 0.0, 1.5),
    ("agn_q_skirtor", "agn", 0.0, 1.5),
    ("agn_oa_skirtor", "agn", 20.0, 60.0),
    ("agn_f_hard", "agn", 0.0, 0.1),
    ("agn_gamma_warm", "agn", 2.0, 3.0),
    ("agn_kt_warm", "agn", 0.1, 1.0),
    ("agn_gamma_hard", "agn", 1.5, 2.5),
    ("agn_kt_hot", "agn", 50.0, 300.0),
    ("agn_r_warm_ratio", "agn", 1.0, 5.0),
    ("agn_polar_oa", "agn", 10.0, 80.0),
    # X-ray binaries
    ("xray_gamma_hmxb", "xray", 1.5, 2.5),
    ("xray_gamma_lmxb", "xray", 1.2, 2.0),
    ("xray_E_cut", "xray", 100.0, 500.0),
    # Radio free-free
    ("radio_T_e", "radio", 5000.0, 20000.0),
    ("radio_alpha_ff", "radio", -0.3, 0.0),
    # Dust IR
    ("dust_alpha_dl14", "dust", 1.0, 3.0),
)

#: Per family, a parameter that certainly moves the output.  If the control is
#: flat the fixture is broken and every result from it is meaningless, so the
#: whole family errors rather than reporting a wall of false positives.
_CONTROLS: dict[str, tuple[str, float, float]] = {
    "agn": ("agn_log_lbol", 10.0, 13.0),
    "xray": ("sfh_const_log_total_mass", 9.0, 11.0),
    "radio": ("radio_q_ir", 1.8, 3.0),
    "dust": ("dust_tau_diff", 0.05, 2.0),
}

#: A live parameter must clear this.  Chosen against the measured spread: the
#: weakest live response is ``radio_T_e`` at 7.8e-4 and the strongest
#: ``dust_alpha_dl14`` at 1.0, while a dropped parameter is identically 0.0.
#: Five orders of margin below the weakest live signal, so this fails only for
#: a parameter that genuinely does nothing.
_MIN_RESPONSE = 1e-9


# ── Measurement ───────────────────────────────────────────────────
def _published(state) -> dict[str, np.ndarray]:
    """Every finite float array the forward pass publishes, by name."""
    out: dict[str, np.ndarray] = {}
    derived = state.derived
    for attr in dir(derived):
        if attr.startswith("_"):
            continue
        value = getattr(derived, attr, None)
        if value is None or callable(value):
            continue
        try:
            arr = np.asarray(jnp.asarray(value))
        except (TypeError, ValueError):
            continue
        if arr.dtype.kind in "fc" and arr.size:
            out[attr] = arr

    # Named, not probed. A try/except around a guessed attribute is how the
    # first version of this helper silently dropped the SED entirely: the
    # state has no ``.sed``, and nothing said so.
    for attr in ("sed_intrinsic", "sed_attenuated", "sed_observed"):
        value = getattr(state, attr)
        if value is not None:
            out[attr] = np.asarray(jnp.asarray(value))
    return out


def _response(model, base: dict, name: str, lo: float, hi: float) -> tuple[float, str]:
    """Largest **per-array** relative change over the published state.

    Per-array and relative, never a global sum — see the module docstring: a
    single dominant luminosity in a summed metric hides every smaller
    component beneath float64's 16 digits.
    """
    low, high = dict(base), dict(base)
    low[name] = jnp.array(lo)
    high[name] = jnp.array(hi)
    before = _published(model.predict_state(low))
    after = _published(model.predict_state(high))

    best, where = 0.0, "<nothing published>"
    for key in sorted(before.keys() & after.keys()):
        x, y = before[key], after[key]
        if x.shape != y.shape:
            return float("inf"), key
        scale = max(np.max(np.abs(x)), np.max(np.abs(y)))
        if not np.isfinite(scale) or scale == 0.0:
            continue
        rel = float(np.max(np.abs(y - x)) / scale)
        if rel > best:
            best, where = rel, key
    return best, where


# ── Model fixtures, one per family ────────────────────────────────
def _synthetic_ssp(log_wave_max: float = 7.0, n_wave: int = 600):
    """Smooth synthetic SSP; ``log_wave_max`` extends the grid for radio."""
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


def _build(**groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(redshift=0.1, **groups)


def _agn_model(ssp, obs):
    return _build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const"},
        dust_attenuation={"type": "two_component", "law": "calzetti"},
        agn={
            "type": "composable",
            "disc": {"type": "kubota_done", "all_params": FREE},
            "torus": {"type": "skirtor", "all_params": FREE},
            "atten": {"type": "polar_dust", "all_params": FREE},
            "all_params": FREE,
        },
    )


def _xray_model(ssp, obs):
    return _build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": FREE},
        xray={
            "type": "yang20",
            "gamma_hmxb": Uniform(1.5, 2.5),
            "gamma_lmxb": Uniform(1.2, 2.0),
            "E_cut": Uniform(100.0, 500.0),
        },
    )


def _radio_model(ssp, obs):
    # Dust is not decoration here: the FIR-radio correlation normalizes against
    # L_ir, so without it sed_radio is identically zero and every radio
    # parameter measures flat for a reason that has nothing to do with wiring.
    return _build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": FREE},
        dust_attenuation={"type": "two_component", "law": "calzetti"},
        dust_emission={"type": "draine_li2014"},
        radio={
            "sf": {"type": "bell2003"},
            "agn": {"type": "powerlaw"},
            "all_params": FREE,
            "T_e": Uniform(5000.0, 20000.0),
            "alpha_ff": Uniform(-0.3, 0.0),
        },
    )


def _dust_model(ssp, obs):
    return _build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const"},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
        dust_emission={"type": "draine_li2014", "alpha_dl14": Uniform(1.0, 3.0)},
    )


_BUILDERS = {
    "agn": _agn_model,
    "xray": _xray_model,
    "radio": _radio_model,
    "dust": _dust_model,
}


@pytest.fixture(scope="module")
def responses(synthetic_tophat_obs):
    """Sweep every parameter in :data:`_FORWARDED`, once, per family.

    Module-scoped and computed in one pass: each family's model is built once
    and each parameter swept once, so the per-parameter tests below only assert
    on numbers that already exist.  Sweeping inside each test would rebuild
    four models nineteen times.
    """
    ssp_optical = _synthetic_ssp()
    # Radio lives longward of 1 mm; on the optical grid there is nothing to see.
    ssp_radio = _synthetic_ssp(log_wave_max=11.0, n_wave=900)

    measured: dict[str, tuple[float, str]] = {}
    controls: dict[str, tuple[float, str]] = {}
    free_params: dict[str, set[str]] = {}

    for family, builder in _BUILDERS.items():
        ssp = ssp_radio if family == "radio" else ssp_optical
        model = builder(ssp, synthetic_tophat_obs)
        base = model.spec.sample(jax.random.PRNGKey(0))
        free_params[family] = set(base)

        controls[family] = _response(model, base, *_CONTROLS[family])

        for name, fam, lo, hi in _FORWARDED:
            if fam != family or name not in base:
                continue
            measured[name] = _response(model, base, name, lo, hi)

    return {"measured": measured, "controls": controls, "free": free_params}


# ── The forwarding contract ───────────────────────────────────────
@pytest.mark.parametrize("family", sorted(_BUILDERS))
def test_the_positive_control_moves_the_output(responses, family):
    """Refuse the family's results unless a known-live parameter registers.

    This is the test that makes every other one in the file mean something.  A
    fixture that quietly built the wrong model, or a params dict the model
    never reads, would otherwise report all nineteen parameters as forwarded.
    """
    name, lo, hi = _CONTROLS[family]
    rel, where = responses["controls"][family]
    assert rel > _MIN_RESPONSE, (
        f"positive control {name} ({lo} -> {hi}) moved the {family} model by {rel:.3e} "
        f"(largest change on {where!r}). The fixture is blind, so no result from this "
        "family can be believed — fix the build before reading the failures below."
    )


@pytest.mark.parametrize(
    ("param", "family", "lo", "hi"), _FORWARDED, ids=[row[0] for row in _FORWARDED]
)
def test_parameter_reaches_the_physics(responses, param, family, lo, hi):
    """A user-set value must change what the model predicts.

    Failure means the parameter is declared, accepted, sampled — and dropped
    somewhere between the params dict and the physics function, which is
    invisible to every other kind of test: the fit still converges and the
    posterior still reports a marginal, which is the prior.
    """
    assert param in responses["free"][family], (
        f"{param} is not free in the {family} fixture, so this test measured nothing. "
        "Either the build grammar no longer exposes it or the group changed name."
    )
    rel, where = responses["measured"][param]
    assert rel > _MIN_RESPONSE, (
        f"{param} swept {lo} -> {hi} changed the published state by {rel:.3e} "
        f"(largest change on {where!r}) — it is being dropped before the physics call. "
        "The parameter is still declared and still sampled, so a fit using it will "
        "converge and report its prior back as a posterior."
    )


# ── Declaration, which is a different claim ───────────────────────
@pytest.mark.parametrize("param", sorted({row[0] for row in _FORWARDED}))
def test_parameter_is_declared_in_the_registry(param):
    """Forwarding is not declaration: a parameter can move the SED and still be
    missing from the registry that priors, summaries and the CI range guards
    read."""
    from tengri.parameters.registry import registry

    record = registry().get(param)
    assert record is not None, (
        f"{param} must be in the parameter registry "
        "(canonical source: the owning component's _params.PARAMS)"
    )
    assert record.prior is not None, f"{param} is registered without a prior"


@pytest.mark.parametrize(
    ("bucket", "param"),
    [("_AGN_PARAMS", row[0]) for row in _FORWARDED if row[1] == "agn"]
    + [("_XRAY_PARAMS", row[0]) for row in _FORWARDED if row[1] == "xray"]
    + [("_RADIO_PARAMS", row[0]) for row in _FORWARDED if row[1] == "radio"]
    + [("shock", "shock_b_over_sqrt_n")],
)
def test_parameter_is_in_its_declaring_bucket(bucket, param):
    from tengri.parameters._builders import _resolve_lazy_bucket

    if bucket == "shock":
        bucket = "_SHOCK_PARAMS"
    assert param in _resolve_lazy_bucket(bucket), (
        f"{param} must be declared in {bucket} "
        "(canonical source: the owning component's _params.PARAMS)"
    )


# ── Model-level switches ──────────────────────────────────────────
def test_radio_model_attributes_carry_the_configured_values(
    synthetic_tophat_obs,
):
    """``sfr_mode`` and ``include_freefree`` are model attributes, not params.

    Previously ``assert "_radio_sfr_mode" in _model_src()`` — a substring
    search over the whole of ``sed_model.py``, which a comment mentioning the
    name satisfies.  Read the attributes off a built model instead, and require
    the configured block name rather than merely a truthy value.
    """
    model = _radio_model(_synthetic_ssp(log_wave_max=11.0, n_wave=900), synthetic_tophat_obs)

    assert model._radio_sfr_mode == "bell2003", (
        f"radio sf block 'bell2003' must reach the model; got {model._radio_sfr_mode!r}"
    )
    assert model._radio_include_freefree is True, (
        "free-free must be enabled by default; a False here silently removes the "
        "thermal component that radio_T_e and radio_alpha_ff control"
    )


# ── Polar dust: the limit that has no free-parameter signature ────
def test_polar_dust_is_a_noop_at_zero_ebv():
    """Polar dust must do nothing when ``agn_polar_ebv == 0``.

    Previously asserted by grepping ``emission_helpers.agn_emission`` for an
    ``agn_polar_ebv > 0.0`` branch.  That helper was a dead duplicate, so the
    test passed on code nothing ran, and it pinned a design the live path had
    abandoned: ``polar_dust_total`` is branchless because
    ``exp(-0.921 * ebv * ...)`` is already the identity at ``ebv = 0`` (a
    Python-level branch would not survive ``jax.jit``).  Assert the invariant
    the guard existed to protect instead of any particular implementation.
    """
    from tengri.components.agn.polar_dust import polar_dust_total

    wave = jnp.linspace(1000.0, 30000.0, 128)
    l_nu_disc = jnp.ones_like(wave)
    atten, emis = polar_dust_total(l_nu_disc, wave, cos_inc=0.9, opening_angle_deg=40.0, ebv=0.0)

    np.testing.assert_allclose(np.asarray(atten), np.asarray(l_nu_disc), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(emis), 0.0, atol=0)


def test_polar_dust_opening_angle_changes_the_result():
    """The unit-level companion to the ``agn_polar_oa`` row in :data:`_FORWARDED`.

    The table row proves the parameter survives the pipeline; this proves the
    function it lands in is not itself ignoring the argument.  Both are needed:
    a live parameter reaching a function that discards it looks identical to a
    dropped one from the outside, and identical to correct wiring from inside.
    """
    from tengri.components.agn.polar_dust import polar_dust_total

    wave = jnp.linspace(1000.0, 30000.0, 128)
    disc = jnp.ones_like(wave)
    narrow, _ = polar_dust_total(disc, wave, cos_inc=0.5, opening_angle_deg=10.0, ebv=0.3)
    wide, _ = polar_dust_total(disc, wave, cos_inc=0.5, opening_angle_deg=80.0, ebv=0.3)

    assert not np.allclose(np.asarray(narrow), np.asarray(wide)), (
        "opening_angle_deg must change the polar-dust result — it is being ignored"
    )
