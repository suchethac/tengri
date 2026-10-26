# SPDX-License-Identifier: BSD-3-Clause
"""The composable AGN runner publishes an energy ledger that closes on the accretion power.

NAMING_CONTRACT section 4c, "Composable runner" (#3076). For each piece of the SED (disc,
torus, lines, polar dust) the runner publishes ``log_L_agn_<piece>`` (angle-integrated) and
``log_L_agn_<piece>_los`` (along ``agn_cos_inc``), the ``_unemitted`` powers of the disc, lines
and torus, ``log_L_agn_los``, ``log_L_agn_power``, every one with an ``L_agn_*_lsun`` sibling,
and ``agn_ledger_closed``. Checked here, with one tolerance for every block:

* C1: the angle-integrated torus of a library torus is ``f L_bol`` (its library anisotropy),
  any other torus and the polar dust carry the same power along every sightline, and the
  disc's sphere mean and unemitted share match an independent dense integral over ``cos i``,
  all to 1e-6;
* C2: on a conserving recipe the angle-integrated pieces plus the unemitted ones are
  ``agn_lum_ratio * 10**agn_log_lbol L_sun`` to 1e-6, once each block's own normalization is
  measured (the dense integral of the block alone, not a table), and ``agn_ledger_closed``
  is 1;
* C3: every ``_los`` key and ``log_L_agn_los`` equal the frequency integral of that piece of the
  SED on an independent dense grid to 1e-6. Both grids carry a node pair at every spectral
  edge of the selected blocks (``spectral_edges``), so both converge at second order;
* C4: the ``Lsun`` siblings are the dex keys in solar luminosities, and the ledger agrees between
  pure float32 and float64 to 1e-4 dex on several recipes;
* ``log_L_agn_power`` is the AGN dust power the torus is normalized on;
* the recipes documented as non-conserving publish ``agn_ledger_closed = 0`` and their build
  emits the rule-10 ``RecipeWarning``; the shipped AGN recipes publish the flag their policy
  implies;
* the ledger does not touch the SED: the SED and the sub-block SEDs agree with and without it
  (to rounding in one compiled graph; bit-identical through the model, which computes the
  ledger on request).
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks._ledger import (
    NON_CONSERVING_DISCS,
    NON_CONSERVING_TORI,
    ledger_policy_closes,
)
from tengri.components.agn.blocks._protocol import (
    AGN_BLOCKS,
    TORUS_ANISOTROPY_BLOCKS,
    collect_block_templates,
)
from tengri.components.agn.blocks.runner import (
    RecipeWarning,
    composable_agn_l_nu,
    validate_block_recipe,
)
from tengri.components.agn.blocks.spectral_edges import block_edges
from tengri.config.exceptions import TengriIOError
from tengri.utils.physics_constants import C_AA, L_SUN

pytestmark = pytest.mark.contract

_LOG_LBOL = 10.5
_LOG_LSUN = float(np.log10(L_SUN))
_WAVE = jnp.asarray(np.geomspace(100.0, 1.0e7, 400))
_PIECES = {"disc": "disc", "torus": "torus", "lines": "lines", "polar_dust": "polar"}
_UNEMITTED = ("disc", "torus", "lines")
#: The one tolerance of C1, C2 and C3, for every block.
_RTOL = 1.0e-6
#: Tori that bundle the disc in one self-normalized template: no debit of the disc.
_SELF_CONTAINED = ("none", "grahsp", "grahsp_mn12", "qsogen")


def _selectors(disc, torus, nlr="none", blr="none", feii="none", atten="none"):
    return {
        "agn_disc_block": disc,
        "agn_nlr_block": nlr,
        "agn_blr_block": blr,
        "agn_feii_block": feii,
        "agn_torus_block": torus,
        "agn_attenuation_block": atten,
    }


def _templates(sel):
    """The recipe's libraries, threaded as the AGN component threads them."""
    return collect_block_templates({k[len("agn_") : -len("_block")]: v for k, v in sel.items()})


def _evaluate(wave, sel, params, *, ledger, x64=True, strict=True):
    """The runner on ``wave`` (jitted, libraries threaded); ``strict=False`` as the component
    calls it, with the full declared parameter set."""
    templates = _templates(sel)

    def run(p, ts):
        return composable_agn_l_nu(
            wave,
            _LOG_LBOL,
            return_components=True,
            return_ledger=ledger,
            template_state=ts,
            strict_keys=strict,
            **sel,
            **p,
        )

    p = {k: v if isinstance(v, str) else jnp.asarray(v) for k, v in params.items()}
    static = {k: v for k, v in p.items() if isinstance(v, str)}
    traced = {k: v for k, v in p.items() if not isinstance(v, str)}
    with jax.enable_x64(x64):
        # The grid carries the evaluation dtype: the runner picks its float32 reference
        # luminosity off it.
        wave = jnp.asarray(np.asarray(wave), dtype=jnp.float64 if x64 else jnp.float32)
        fn = jax.jit(lambda q, ts: run({**q, **static}, ts))
        sed, comps = fn(traced, templates)
        return np.asarray(sed), {k: np.asarray(v) for k, v in comps.items()}


def _edge_pairs(sel):
    """A node pair at every registered spectral edge of the recipe [A]."""
    edges = np.asarray(block_edges(sel), dtype=np.float64)
    edges = edges[(edges > 1.0e-3) & (edges < 1.0e10)]
    return np.concatenate([edges * (1.0 - 1.0e-9), edges * (1.0 + 1.0e-9)])


def _dense_wave(sel, coarse: int = 256_000, window: int = 144_001) -> jnp.ndarray:
    """An independent dense grid: 8x the ledger's log density outside 900 A - 3e5 A and 3x
    inside it (one uniform piece: the line blocks conserve each line's flux over the cells they
    are given), with the edge pairs."""
    nodes = np.concatenate(
        [
            np.geomspace(1.0e-3, 1.0e10, coarse),
            np.geomspace(9.0e2, 3.0e5, window),
            _edge_pairs(sel),
        ]
    )
    return jnp.asarray(np.unique(nodes))


def _power(lnu, wave):
    """Frequency integral of ``L_nu`` on ``wave`` [erg/s] (trapezoid in ln nu)."""
    nu = C_AA / np.asarray(wave)
    return float(-np.trapezoid(np.asarray(lnu) * nu, np.log(nu)))


def _lin(comps, key):
    """``10**key`` [erg/s]; 0 for an absent piece."""
    return float(10.0 ** comps[key])


def _skip_without_library(sel):
    """Skip only when a selected block's library file is absent from this checkout."""
    try:
        _templates(sel)
    except FileNotFoundError as exc:
        pytest.skip(f"library data absent: {exc}")


@functools.cache
def _disc_normalization(disc: str) -> float:
    """The disc block's own power over ``10**agn_log_lbol L_sun``, minus one.

    The dense integral of the disc alone at ``cos i = 1/2``, where ``2 cos i D + H`` (and an
    isotropic disc) equals its sphere mean.
    """
    if disc == "none":
        return 0.0
    sel = _selectors(disc, "none")
    wave = _dense_wave(sel)
    _, comps = _evaluate(wave, sel, {"agn_norm": "conserving", "agn_cos_inc": 0.5}, ledger=False)
    return _power(comps["disc"], wave) / 10.0 ** (_LOG_LBOL + _LOG_LSUN) - 1.0


@functools.cache
def _torus_normalization(torus: str, frac: float) -> float:
    """A torus block's own power over ``f 10**agn_log_lbol L_sun``, minus one.

    Zero for a library torus, whose angle-integrated power is ``f L_bol`` by its anisotropy
    (C1); the dense integral of the torus alone otherwise (it is isotropic, C1).
    """
    if torus in _SELF_CONTAINED or torus in TORUS_ANISOTROPY_BLOCKS:
        return 0.0
    sel = _selectors("none", torus)
    wave = _dense_wave(sel)
    params = {"agn_norm": "conserving", "agn_cos_inc": 0.5, "agn_torus_frac": frac}
    _, comps = _evaluate(wave, sel, params, ledger=False)
    return _power(comps["torus"], wave) / (frac * 10.0 ** (_LOG_LBOL + _LOG_LSUN)) - 1.0


def _check_c3(sel, params, comps):
    """C3: each ``_los`` key is the dense-grid integral of its piece of the SED."""
    dense = _dense_wave(sel)
    _, dense_comps = _evaluate(dense, sel, params, ledger=False)
    total = 0.0
    for piece, key in _PIECES.items():
        p = _power(dense_comps[key], dense)
        total += p
        if p == 0.0:
            assert comps[f"log_L_agn_{piece}_los"] == -np.inf, piece
            continue
        assert _lin(comps, f"log_L_agn_{piece}_los") == pytest.approx(p, rel=_RTOL), piece
    assert _lin(comps, "log_L_agn_los") == pytest.approx(total, rel=_RTOL)


def _check_c2(sel, params, comps):
    """C2: angle-integrated pieces + unemitted = lum_ratio x the blocks' accretion power."""
    frac = float(params.get("agn_torus_frac", 0.0))
    torus = sel["agn_torus_block"]
    disc_share = 1.0 if torus in _SELF_CONTAINED else 1.0 - frac
    excess = disc_share * _disc_normalization(sel["agn_disc_block"])
    if torus not in _SELF_CONTAINED:
        excess += frac * _torus_normalization(torus, frac)
    budget = float(params.get("agn_lum_ratio", 1.0)) * 10.0 ** (_LOG_LBOL + _LOG_LSUN)
    total = sum(_lin(comps, f"log_L_agn_{p}") for p in _PIECES)
    total += sum(_lin(comps, f"log_L_agn_{p}_unemitted") for p in _UNEMITTED)
    assert total / budget - 1.0 == pytest.approx(excess, abs=_RTOL)
    assert float(comps["agn_ledger_closed"]) == 1.0


def _check_c4_siblings(comps):
    """C4: every ``L_agn_*_lsun`` is its dex key in solar luminosities."""
    for key, val in comps.items():
        if key.startswith("log_L_agn_") and key != "log_L_agn_direct_los":
            sib = f"{key[len('log_') :]}_lsun"
            assert float(comps[sib]) == pytest.approx(10.0 ** (float(val) - _LOG_LSUN), rel=1e-12)


def _check_c1_torus(sel, params, comps):
    """C1 for the torus: ``f L_bol`` for a library torus; equal to the sightline otherwise."""
    if sel["agn_torus_block"] == "none" or sel["agn_attenuation_block"] != "none":
        return
    if sel["agn_torus_block"] in TORUS_ANISOTROPY_BLOCKS:
        expected = (
            float(params["agn_torus_frac"])
            * float(params.get("agn_lum_ratio", 1.0))
            * 10.0 ** (_LOG_LBOL + _LOG_LSUN)
        )
        assert _lin(comps, "log_L_agn_torus") == pytest.approx(expected, rel=_RTOL)
    else:
        assert _lin(comps, "log_L_agn_torus") == pytest.approx(
            _lin(comps, "log_L_agn_torus_los"), rel=_RTOL
        )


_CONSERVING_TORI = sorted(set(AGN_BLOCKS["torus"]) - NON_CONSERVING_TORI)
_CONSERVING_DISCS = sorted(set(AGN_BLOCKS["disc"]) - NON_CONSERVING_DISCS)

#: Disc x torus pairs from the registry: every conserving torus behind the multicolor disc, and
#: every conserving disc in front of the Fritz torus (a library torus with anisotropy and a
#: wavelength-dependent sightline screen), at a Type-2-leaning sightline.
_PAIRS = sorted(
    {("multicolor", t) for t in _CONSERVING_TORI} | {(d, "fritz") for d in _CONSERVING_DISCS}
)

_BASE = {"agn_norm": "conserving", "agn_cos_inc": 0.35, "agn_torus_frac": 0.4}


@pytest.mark.parametrize(("disc", "torus"), _PAIRS, ids=[f"{d}-{t}" for d, t in _PAIRS])
def test_registry_pair_ledger_closes(disc, torus):
    """C1-C4 for every conserving disc x torus pair under ``agn_norm='conserving'``."""
    sel = _selectors(disc, torus)
    _skip_without_library(sel)
    _, comps = _evaluate(_WAVE, sel, _BASE, ledger=True)
    _check_c2(sel, _BASE, comps)
    _check_c3(sel, _BASE, comps)
    _check_c1_torus(sel, _BASE, comps)
    _check_c4_siblings(comps)


#: Recipes that exercise the other stages: lines, the polar dust, a foreground attenuation
#: block (with and without the EUV edge of the multicolor disc), ``cigale_joint`` with its tie
#: inactive, and ``agn_lum_ratio``.
_RECIPES = {
    "lines": (
        _selectors("kubota_done", "simple", "analytic", "analytic", "boroson_green"),
        {**_BASE, "agn_cos_inc": 0.2},
    ),
    "polar_dust": (
        _selectors("kubota_done", "skirtor", atten="polar_dust"),
        {**_BASE, "agn_polar_ebv": 0.2, "agn_cos_inc": 0.7},
    ),
    "polar_dust_lines": (
        _selectors("kubota_done", "two_temperature", "analytic", "analytic", atten="polar_dust"),
        {**_BASE, "agn_polar_ebv": 0.15, "agn_cos_inc": 0.9},
    ),
    "smc_screen": (
        _selectors("kubota_done", "two_temperature", atten="smc_prevot"),
        {**_BASE, "agn_ebv": 0.3},
    ),
    "smc_screen_multicolor": (
        _selectors("multicolor", "two_temperature", atten="smc_prevot"),
        {**_BASE, "agn_ebv": 0.3},
    ),
    "qsogen_screen": (
        _selectors("multicolor", "simple", atten="qsogen"),
        {**_BASE, "agn_ebv": 0.2},
    ),
    "disc_reddening": (
        _selectors("kubota_done", "simple"),
        {**_BASE, "agn_ebv_disc": 0.1},
    ),
    "cigale_joint_tie_off": (
        _selectors("kubota_done", "skirtor", atten="polar_dust"),
        {**_BASE, "agn_norm": "cigale_joint", "agn_ir_frac": 0.0, "agn_polar_ebv": 0.1},
    ),
    "cigale_joint_untied_torus": (
        _selectors("kubota_done", "nenkova_agnfitter"),
        {**_BASE, "agn_norm": "cigale_joint"},
    ),
    "lum_ratio": (
        _selectors("kubota_done", "fritz"),
        {**_BASE, "agn_lum_ratio": 0.3},
    ),
    "no_disc": (_selectors("none", "fritz"), _BASE),
}


@pytest.mark.parametrize("name", sorted(_RECIPES))
def test_stage_recipe_ledger_closes(name):
    """C1-C4 on recipes with lines, polar dust, attenuation, the inactive tie, lum_ratio."""
    sel, params = _RECIPES[name]
    _skip_without_library(sel)
    _, comps = _evaluate(_WAVE, sel, params, ledger=True)
    _check_c2(sel, params, comps)
    _check_c3(sel, params, comps)
    _check_c1_torus(sel, params, comps)
    _check_c4_siblings(comps)
    # The polar dust is re-emitted isotropically: its two powers are one.
    if np.isfinite(comps["log_L_agn_polar_dust"]):
        assert _lin(comps, "log_L_agn_polar_dust") == pytest.approx(
            _lin(comps, "log_L_agn_polar_dust_los"), rel=_RTOL
        )


def _richardson_sphere_mean(sel, params, wave, levels=(9, 10)):
    """The sphere mean of the emitted disc's power: trapezoid on ``2**k + 1`` uniform sightlines,
    Richardson-extrapolated across the two levels [erg/s]."""
    templates = _templates(sel)
    nu = jnp.asarray(C_AA / np.asarray(wave))
    log_nu = jnp.log(nu)

    def disc_power(mu, ts):
        _, c = composable_agn_l_nu(
            wave,
            _LOG_LBOL,
            return_components=True,
            template_state=ts,
            **sel,
            **{**params, "agn_cos_inc": mu},
        )
        return -jnp.trapezoid(c["disc"] * nu, log_nu)

    powers = jax.jit(lambda mus, ts: jax.lax.map(lambda m: disc_power(m, ts), mus, batch_size=16))
    means = []
    for k in levels:
        mus = np.linspace(0.0, 1.0, 2**k + 1)
        means.append(float(np.trapezoid(np.asarray(powers(jnp.asarray(mus), templates)), mus)))
    return means[1] + (means[1] - means[0]) / 3.0


_LIBRARY_TORI = ("skirtor", "fritz", "nenkova_agnfitter", "skirtor_agnfitter_1p")


@pytest.mark.parametrize("torus", _LIBRARY_TORI)
def test_disc_sphere_mean_matches_a_dense_independent_mu_integral(torus):
    """C1 for the disc behind each library torus: ``log_L_agn_disc`` and
    ``log_L_agn_disc_unemitted`` against 1025 uniform sightlines (Richardson), 1e-6.

    The frequency grid is built here (the ledger's density, independently), so the check
    isolates the sightline quadrature: the library inclination nodes and the Type-1/2 sigmoid
    are where a uniform quadrature loses its order. The unemitted reference is the disc's own
    sphere-mean power (its dense integral at ``cos i = 1/2``) debited by ``1 - f``, minus the
    emitted mean.
    """
    sel = _selectors("kubota_done", torus)
    _skip_without_library(sel)
    params = {"agn_norm": "conserving", "agn_torus_frac": 0.4}
    wave = jnp.asarray(
        np.unique(
            np.concatenate(
                [
                    np.geomspace(1.0e-3, 1.0e10, 32_000),
                    np.geomspace(9.0e2, 3.0e5, 48_001),
                    _edge_pairs(sel),
                ]
            )
        )
    )
    emitted = _richardson_sphere_mean(sel, params, wave)
    intrinsic = (1.0 - params["agn_torus_frac"]) * 10.0 ** (_LOG_LBOL + _LOG_LSUN)
    intrinsic *= 1.0 + _disc_normalization("kubota_done")
    _, comps = _evaluate(_WAVE, sel, {**params, "agn_cos_inc": 0.5}, ledger=True)
    assert _lin(comps, "log_L_agn_disc") == pytest.approx(emitted, rel=_RTOL)
    assert _lin(comps, "log_L_agn_disc_unemitted") == pytest.approx(intrinsic - emitted, rel=_RTOL)


#: Float32 recipes: a library torus behind polar dust, lines behind a screened library torus,
#: the CIGALE tie, a foreground attenuation block.
_FLOAT32_RECIPES = ("polar_dust", "lines_fritz", "cigale_joint_tie_on", "smc_screen")


def _float32_recipe(name):
    if name == "lines_fritz":
        return _selectors("kubota_done", "fritz", "analytic", "analytic"), _BASE
    if name == "cigale_joint_tie_on":
        return _selectors("kubota_done", "fritz"), {
            **_BASE,
            "agn_norm": "cigale_joint",
            "agn_ir_frac": 0.3,
        }
    return _RECIPES[name]


@pytest.mark.parametrize("name", _FLOAT32_RECIPES)
def test_float32_ledger_matches_float64(name):
    """C4: the ledger is finite in pure float32 and agrees with float64 to 1e-4 dex.

    Holds where the float32 SED is finite: an anisotropic template torus is not (#3101).
    """
    sel, params = _float32_recipe(name)
    _skip_without_library(sel)
    _, c64 = _evaluate(_WAVE, sel, params, ledger=True)
    sed32, c32 = _evaluate(_WAVE, sel, params, ledger=True, x64=False)
    assert np.all(np.isfinite(sed32))
    for key, val in c64.items():
        # ``log_L_agn_direct_los`` is not a ledger key: it is the pre-#3076 value, unchanged,
        # NaN in float32 for the kubota_done and multicolor discs on main as well (#3104).
        if key.startswith("log_L_agn_") and key != "log_L_agn_direct_los" and np.isfinite(val):
            assert np.isfinite(c32[key]), key
            assert float(c32[key]) == pytest.approx(float(val), abs=1e-4), key


def test_power_is_the_torus_normalization():
    """``log_L_agn_power``: ``f L_bol`` for a conserving torus; under the active CIGALE tie the
    torus's sightline power, which the emitted torus then carries (no polar dust here)."""
    sel = _selectors("kubota_done", "simple")
    _, comps = _evaluate(_WAVE, sel, _BASE, ledger=True)
    f_lbol = _BASE["agn_torus_frac"] * 10.0 ** (_LOG_LBOL + _LOG_LSUN)
    assert _lin(comps, "log_L_agn_power") == pytest.approx(f_lbol, rel=_RTOL)
    sel = _selectors("multicolor", "skirtor")
    _skip_without_library(sel)
    params = {**_BASE, "agn_norm": "cigale_joint", "agn_ir_frac": 0.3}
    _, tied = _evaluate(_WAVE, sel, params, ledger=True)
    assert np.isfinite(tied["log_L_agn_power"])
    assert float(tied["agn_ledger_closed"]) == 0.0
    assert _lin(tied, "log_L_agn_torus_los") == pytest.approx(
        _lin(tied, "log_L_agn_power"), rel=_RTOL
    )
    _, untied = _evaluate(_WAVE, _selectors("multicolor", "none"), _BASE, ledger=True)
    assert untied["log_L_agn_power"] == -np.inf


@pytest.mark.parametrize("name", sorted(_RECIPES))
def test_ledger_leaves_the_sed_unchanged(name):
    """The ledger is publication only: the SED and sub-block SEDs do not move.

    To rounding: compiled in one graph with the ledger, the compiler may fuse the SED's
    arithmetic differently (one recipe moves by 3.4e-16). The model computes the ledger in a
    graph of its own, on request, so its SED is bit-identical
    (``test_bug_3076_los_is_the_emitted_sum``).
    """
    sel, params = _RECIPES[name]
    _skip_without_library(sel)
    sed_off, comps_off = _evaluate(_WAVE, sel, params, ledger=False)
    sed_on, comps_on = _evaluate(_WAVE, sel, params, ledger=True)
    np.testing.assert_allclose(sed_on, sed_off, rtol=1e-15, atol=0.0)
    for key in ("disc", "torus", "lines", "polar", "log_L_agn_direct_los"):
        np.testing.assert_allclose(comps_on[key], comps_off[key], rtol=1e-15, atol=0.0)


#: Recipes documented as non-conserving (decision 3 of the 2026-10-13 rulings) and the
#: ``cigale_joint`` bookkeeping that adds the lines on top or normalizes on ``agn_power``.
_NON_CONSERVING = {
    "grahsp_torus": (_selectors("multicolor", "grahsp"), _BASE, True),
    "grahsp_mn12_torus": (_selectors("multicolor", "grahsp_mn12"), _BASE, True),
    "qsogen_torus": (_selectors("multicolor", "qsogen"), _BASE, True),
    "grahsp_disc": (_selectors("grahsp_sbpl", "simple"), _BASE, True),
    "independent": (_selectors("multicolor", "fritz"), {**_BASE, "agn_norm": "independent"}, True),
    "cigale_joint_lines": (
        _selectors("multicolor", "simple", "analytic"),
        {**_BASE, "agn_norm": "cigale_joint"},
        False,
    ),
    "polar_dust_library_torus": (
        _selectors("kubota_done", "nenkova_agnfitter", atten="polar_dust"),
        {**_BASE, "agn_polar_ebv": 0.2, "agn_cos_inc": 0.8},
        False,
    ),
    "cigale_joint_tie_on": (
        _selectors("multicolor", "skirtor"),
        {**_BASE, "agn_norm": "cigale_joint", "agn_ir_frac": 0.3},
        False,
    ),
}


@pytest.mark.parametrize("name", sorted(_NON_CONSERVING))
def test_non_conserving_recipe_flags_and_notes(name):
    """``agn_ledger_closed`` is 0; decision-3 recipes also emit the rule-10 build note."""
    sel, params, noted = _NON_CONSERVING[name]
    _skip_without_library(sel)
    _, comps = _evaluate(_WAVE, sel, params, ledger=True)
    assert float(comps["agn_ledger_closed"]) == 0.0
    _check_c4_siblings(comps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        issues = validate_block_recipe(
            **sel, agn_norm=params["agn_norm"], params=None, param_support=None
        )
    notes = [i for i in issues if "energy ledger does not close" in i]
    assert bool(notes) is noted, issues
    assert any(issubclass(w.category, RecipeWarning) for w in caught) or not noted


def _recipe_agn(name):
    """Selectors and parameter values of a shipped recipe's AGN (its declared defaults)."""
    from tengri import recipes
    from tengri.parameters.groups import parse_groups

    groups = {k: v for k, v in getattr(recipes, name)().items() if k != "approx"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = parse_groups(**groups)
    if spec.agn_model != "composable":
        return None
    sel = {k: getattr(spec, k) for k in _selectors("none", "none")}
    values = {"agn_norm": spec.agn_norm}
    for key, dist in spec._distributions.items():
        if not key.startswith("agn_") or key == "agn_log_lbol":
            continue
        default = dist.value if dist.is_fixed else dist.default
        if default is None:
            lo, hi = dist.bounds
            default = 0.5 * (lo + hi)
        values[key] = float(default)
    return sel, values


def _agn_recipe_names():
    from tengri import recipes

    return [n for n in recipes.__all__ if "agn" in n]


@pytest.mark.parametrize("name", _agn_recipe_names())
def test_shipped_agn_recipe_ledger(name):
    """Each shipped AGN recipe publishes the flag its policy implies; C2 holds where it is 1."""
    found = _recipe_agn(name)
    if found is None:
        pytest.skip(f"{name} has no composable AGN")
    sel, params = found
    try:
        _, comps = _evaluate(_WAVE, sel, params, ledger=True, strict=False)
    except (FileNotFoundError, TengriIOError) as exc:
        pytest.skip(f"library data absent: {exc}")
    closes = ledger_policy_closes(**sel, agn_norm=params["agn_norm"])
    tie_on = (
        params["agn_norm"] == "cigale_joint"
        and sel["agn_torus_block"] in ("skirtor", "fritz")
        and params.get("agn_ir_frac", 0.0) > 0.0
    )
    assert float(comps["agn_ledger_closed"]) == float(closes and not tie_on)
    if closes and not tie_on:
        _check_c2(sel, params, comps)
    if tie_on:
        assert np.isfinite(comps["log_L_agn_power"])
    _check_c4_siblings(comps)
