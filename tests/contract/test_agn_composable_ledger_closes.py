# SPDX-License-Identifier: BSD-3-Clause
"""The composable AGN runner publishes an energy ledger that closes on the accretion power.

NAMING_CONTRACT section 4c, "Composable runner" (#3076). For each piece of the SED (disc,
torus, lines, polar dust) the runner publishes ``log_L_agn_<piece>`` (angle-integrated) and
``log_L_agn_<piece>_los`` (along ``agn_cos_inc``), the ``_unemitted`` powers of the disc, lines
and torus, ``log_L_agn_los``, every one with an ``L_agn_*_lsun`` sibling, and
``agn_ledger_closed``. Checked here:

* C1: the angle-integrated torus of a library torus is ``f L_bol`` exactly (its library
  anisotropy), any other torus and the polar dust carry the same power along every sightline,
  and the disc's sphere mean matches an independent 201-point trapezoid in ``cos i``;
* C2: on a conserving recipe the angle-integrated pieces plus the unemitted ones are
  ``agn_lum_ratio * 10**agn_log_lbol L_sun``, to the accuracy of the blocks' own normalization
  (measured per block below), and ``agn_ledger_closed`` is 1;
* C3: every ``_los`` key and ``log_L_agn_los`` equal the frequency integral of that piece of the
  SED on an independent dense grid, to 1e-6;
* C4: the ``Lsun`` siblings are the dex keys in solar luminosities, and the ledger agrees between
  pure float32 and float64;
* the recipes documented as non-conserving publish ``agn_ledger_closed = 0`` and their build
  emits the rule-10 ``RecipeWarning``; the shipped AGN recipes publish the flag their policy
  implies;
* the ledger does not touch the SED: the SED and the sub-block SEDs are bit-identical with and
  without it.
"""

from __future__ import annotations

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
from tengri.config.exceptions import TengriIOError
from tengri.utils.physics_constants import C_AA, L_SUN

pytestmark = pytest.mark.contract

_LOG_LBOL = 10.5
_LOG_LSUN = float(np.log10(L_SUN))
_WAVE = jnp.asarray(np.geomspace(100.0, 1.0e7, 400))
_PIECES = {"disc": "disc", "torus": "torus", "lines": "lines", "polar_dust": "polar"}
_UNEMITTED = ("disc", "torus", "lines")

_C3_RTOL = 1.0e-6
#: Dense reference grid: 16x the ledger's density outside the window, 3x inside it.
DENSE_PER_DEX = 9840
DENSE_WINDOW = 144_001
#: C1: a library torus's angle-integrated power is its sightline power (the ledger-grid
#: integral, within 3.4e-7 dex of ``f L_bol`` measured) minus its anisotropy.
_C1_TORUS_ATOL_DEX = 1.0e-6
#: C2: the pieces add up to the accretion power to the accuracy of the blocks' own
#: normalization on the ledger grid. Measured, then tripled: kubota_done, adaf, schartmann2005
#: and the tori within 3e-6; the others below.
_C2_RTOL = 1.0e-5
_C2_RTOL_BY_BLOCK = {
    "multicolor": 2.1e-4,  # 6.8e-5: its spectrum is cut at 30 A
    "powerlaw": 9.0e-5,  # 2.8e-5
    "relagn": 2.1e-5,  # 6.7e-6
    "richards2006": 2.5e-4,  # 8.2e-5
    "skirtor": 4.1e-4,  # 1.4e-4 (the CIGALE disc)
    "slone_netzer": 9.7e-4,  # 3.2e-4
    "kd18_agnfitter": 3.3e-5,  # 1.1e-5
    "kd18_agnfitter_warmindex": 3.3e-5,  # 1.1e-5
    "nenkova": 1.3e-4,  # 4.3e-5 beside the multicolor disc's -4.1e-5
}
#: C3 for the disc pieces: a disc with a hard spectral edge (the multicolor cut at 30 A and
#: kink at 82 A; the template discs' grid ends) or a disc behind a library torus (whose
#: transmission steps at the library's first wavelength node) converges at first order on
#: any fixed grid, so its line-of-sight power is the dense integral only to the edge's
#: share. Measured over the sweep (worst sightline), then tripled.
_C3_DISC_RTOL_BY_BLOCK = {
    "multicolor": 3.0e-3,  # 6.6e-5 unscreened, 1.0e-3 behind the Fritz torus
    "relagn": 4.2e-4,  # 1.4e-4
    "kd18_agnfitter": 3.3e-4,  # 1.1e-4
    "kd18_agnfitter_warmindex": 3.3e-4,  # 1.1e-4
    "powerlaw": 1.4e-4,  # 4.7e-5
    "richards2006": 1.4e-4,  # 4.5e-5
    "skirtor": 1.2e-4,  # 4.0e-5
    "slone_netzer": 2.8e-5,  # 9.3e-6
    "kubota_done": 3.0e-5,  # 1.0e-5 behind the Fritz torus, 9.0e-6 behind SKIRTOR
    "adaf": 4.0e-6,  # 1.3e-6
}


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


def _evaluate(wave, sel, params, *, ledger, x64=True):
    """The runner on ``wave`` (jitted, libraries threaded), as the component calls it."""
    templates = _templates(sel)

    def run(p, ts):
        return composable_agn_l_nu(
            wave,
            _LOG_LBOL,
            return_components=True,
            return_ledger=ledger,
            template_state=ts,
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


def _dense_wave(per_dex: int = DENSE_PER_DEX, window: int = DENSE_WINDOW) -> jnp.ndarray:
    """An independent dense grid, uniform in ln lambda on each of three pieces.

    The line blocks conserve each line's flux over the cells they are given, so the window is
    one uniform piece (interleaving nodes inside it moves the line integral by 3e-6).
    """
    lo, hi = 9.0e2, 3.0e5
    nodes = np.concatenate(
        [
            np.geomspace(1.0e-3, lo, int(per_dex * np.log10(lo / 1.0e-3)) + 1)[:-1],
            np.geomspace(lo, hi, window),
            np.geomspace(hi, 1.0e10, int(per_dex * np.log10(1.0e10 / hi)) + 1)[1:],
        ]
    )
    return jnp.asarray(nodes)


def _power(lnu, wave):
    """Frequency integral of ``L_nu`` on ``wave`` [erg/s] (trapezoid in ln nu)."""
    nu = C_AA / np.asarray(wave)
    return float(-np.trapezoid(np.asarray(lnu) * nu, np.log(nu)))


def _lin(comps, key):
    """``10**key`` [erg/s]; 0 for an absent piece."""
    return float(10.0 ** comps[key])


def _has_lines(sel):
    return any(sel[k] != "none" for k in ("agn_nlr_block", "agn_blr_block", "agn_feii_block"))


def _c2_rtol(sel):
    return max(
        [_C2_RTOL]
        + [_C2_RTOL_BY_BLOCK.get(sel[k], 0.0) for k in ("agn_disc_block", "agn_torus_block")]
    )


def _skip_without_library(sel):
    """Skip only when a selected block's library file is absent from this checkout."""
    try:
        _templates(sel)
    except FileNotFoundError as exc:
        pytest.skip(f"library data absent: {exc}")


def _check_c3(sel, params, comps):
    """C3: each ``_los`` key is the dense-grid integral of its piece of the SED."""
    dense = _dense_wave()
    _, dense_comps = _evaluate(dense, sel, params, ledger=False)
    total = 0.0
    for piece, key in _PIECES.items():
        p = _power(dense_comps[key], dense)
        total += p
        if p == 0.0:
            assert comps[f"log_L_agn_{piece}_los"] == -np.inf, piece
            continue
        rtol = _C3_RTOL
        if piece == "disc":
            rtol = _C3_DISC_RTOL_BY_BLOCK.get(sel["agn_disc_block"], _C3_RTOL)
        assert _lin(comps, f"log_L_agn_{piece}_los") == pytest.approx(p, rel=rtol), piece
    disc_share = _power(dense_comps["disc"], dense) / total
    rtol_total = max(_C3_RTOL, disc_share * _C3_DISC_RTOL_BY_BLOCK.get(sel["agn_disc_block"], 0.0))
    assert _lin(comps, "log_L_agn_los") == pytest.approx(total, rel=rtol_total)


def _check_c2(sel, params, comps):
    """C2: angle-integrated pieces + unemitted = lum_ratio x accretion power."""
    budget = float(params.get("agn_lum_ratio", 1.0)) * 10.0 ** (_LOG_LBOL + _LOG_LSUN)
    total = sum(_lin(comps, f"log_L_agn_{p}") for p in _PIECES)
    total += sum(_lin(comps, f"log_L_agn_{p}_unemitted") for p in _UNEMITTED)
    assert total / budget - 1.0 == pytest.approx(0.0, abs=_c2_rtol(sel))
    assert float(comps["agn_ledger_closed"]) == 1.0


def _check_c4_siblings(comps):
    """C4: every ``L_agn_*_lsun`` is its dex key in solar luminosities."""
    for key, val in comps.items():
        if key.startswith("log_L_agn_") and key != "log_L_agn_direct_los":
            sib = f"{key[len('log_') :]}_lsun"
            assert float(comps[sib]) == pytest.approx(10.0 ** (float(val) - _LOG_LSUN), rel=1e-12)


def _check_c1_torus(sel, params, comps):
    """C1 for the torus: ``f L_bol`` for a library torus; equal to the sightline otherwise."""
    if sel["agn_torus_block"] == "none":
        return
    if sel["agn_attenuation_block"] != "none":
        return
    if sel["agn_torus_block"] in TORUS_ANISOTROPY_BLOCKS:
        expected = (
            _LOG_LBOL
            + _LOG_LSUN
            + np.log10(float(params["agn_torus_frac"]))
            + np.log10(float(params.get("agn_lum_ratio", 1.0)))
        )
        assert float(comps["log_L_agn_torus"]) == pytest.approx(expected, abs=_C1_TORUS_ATOL_DEX)
    else:
        assert float(comps["log_L_agn_torus"]) == pytest.approx(
            float(comps["log_L_agn_torus_los"]), abs=1e-6
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
#: block, ``cigale_joint`` with its tie inactive, and ``agn_lum_ratio``.
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
        assert float(comps["log_L_agn_polar_dust"]) == pytest.approx(
            float(comps["log_L_agn_polar_dust_los"]), abs=1e-6
        )


def test_disc_sphere_mean_matches_an_independent_quadrature():
    """C1 for the disc: 201-point trapezoid in cos i of the emitted disc, 2e-4."""
    sel = _selectors("multicolor", "fritz")
    _skip_without_library(sel)
    _, comps = _evaluate(_WAVE, sel, _BASE, ledger=True)
    mus = np.linspace(0.0, 1.0, 201)
    wave = jnp.asarray(np.geomspace(1.0e-3, 1.0e10, 8000))
    templates = _templates(sel)

    def disc_power(mu):
        _, c = composable_agn_l_nu(
            wave,
            _LOG_LBOL,
            return_components=True,
            template_state=templates,
            **sel,
            **{**_BASE, "agn_cos_inc": mu},
        )
        nu = C_AA / wave
        return -jnp.trapezoid(c["disc"] * nu, jnp.log(nu))

    powers = np.asarray(jax.jit(jax.vmap(disc_power))(jnp.asarray(mus)))
    expected = np.trapezoid(powers, mus)
    assert _lin(comps, "log_L_agn_disc") == pytest.approx(expected, rel=2e-4)


def test_float32_ledger_matches_float64():
    """C4: the ledger is finite in pure float32 and agrees with float64."""
    sel, params = _RECIPES["polar_dust"]
    _skip_without_library(sel)
    _, c64 = _evaluate(_WAVE, sel, params, ledger=True)
    _, c32 = _evaluate(_WAVE, sel, params, ledger=True, x64=False)
    for key, val in c64.items():
        # ``log_L_agn_direct_los`` is not a ledger key: it is the pre-#3076 value, unchanged,
        # and NaN in float32 for this disc on main as well.
        if key.startswith("log_L_agn_") and key != "log_L_agn_direct_los" and np.isfinite(val):
            assert np.isfinite(c32[key]), key
            # 1e-4 dex: the unemitted disc is a difference of two float32 sums (3.8e-5 measured).
            assert float(c32[key]) == pytest.approx(float(val), abs=1e-4), key


@pytest.mark.parametrize("name", sorted(_RECIPES))
def test_ledger_leaves_the_sed_bit_identical(name):
    """The ledger is publication only: the SED and sub-block SEDs do not move by one bit."""
    sel, params = _RECIPES[name]
    _skip_without_library(sel)
    sed_off, comps_off = _evaluate(_WAVE, sel, params, ledger=False)
    sed_on, comps_on = _evaluate(_WAVE, sel, params, ledger=True)
    np.testing.assert_array_equal(sed_on, sed_off)
    for key in ("disc", "torus", "lines", "polar", "log_L_agn_direct_los"):
        np.testing.assert_array_equal(comps_on[key], comps_off[key])


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
        _, comps = _evaluate(_WAVE, sel, params, ledger=True)
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
    _check_c4_siblings(comps)
