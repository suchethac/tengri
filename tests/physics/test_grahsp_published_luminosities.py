# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP published ``L_agn_bol``, ``L_agn_torus`` and ``L_agn_absorbed`` are the integrals of
the SED the component emits.

Each published luminosity is the integral of the SED the component emits, under the key's
documented definition (Buchner+ 2024 section 2.1.4-2.1.5, arXiv:2405.19297):

* ``L_agn_bol``: intrinsic BBB + lines + FeII + Balmer continuum above upstream's lumBolBBB
  cut (91.1753 nm, owner ruling 16), before the bi-attenuation;
* ``L_agn_bol_ionizing``: the same BBB side below that cut, from the X-ray floor (12.4 nm);
* ``L_agn_torus``: intrinsic torus continuum + Si feature over its full support;
* ``L_agn_absorbed``: intrinsic minus attenuated AGN-side power, over the full support.

The reference is ``tests/physics/_grahsp_reference.py``: segmented Gauss-Legendre in
ln(lambda) from 1 nm to 1e15 nm of the component's own emitted SED, with a break at every hard
edge, template node, narrow feature and clip crossing. It reaches the power-law tail of the
big blue bump and shares no rule, node count or closed form with the component.

Invariants tested: each key agrees across a dense, a 40-point and a 12-point caller grid; it
equals the reference at the declared defaults, at the box corners, where the Si clip binds,
and over 200 random draws of the declared box for every option combination (the default
configuration, the MN12 torus, the Netzer disc, the Veron-Cetty FeII, all three together and a
type 2 AGN); and the SBPL tail up to the prior edge is inside the published value.

The tests call no private helper of the component: the attenuation screen is
``attenuation_factors``, and amplitudes are read off the emitted SED.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.grahsp.bolometric import BBB_BOL_CUT_NM
from tests.physics._agn_publication import PUB_COARSE, PUB_DENSE, PUB_FILTERS
from tests.physics._grahsp_reference import (
    CONFIGS,
    KEYS,
    Reference,
    default_params,
    draw_params,
)

pytestmark = pytest.mark.limit

_RTOL = 1.0e-6
_CORNER_RTOL = 1.0e-7
_N_DRAWS = 200
_SEED = 20261012


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def reference(request):
    """(name, Reference) for each option combination; evaluators are jitted once per module."""
    return request.param, Reference(**CONFIGS[request.param])


def _rel(pub, ref):
    return {k: pub[k] / ref[k] - 1.0 for k in KEYS}


def test_published_L_independent_of_caller_grid(reference):
    """Each L_* is the same on a dense, a 40-point and a 12-point caller grid (1e-6)."""
    name, ref = reference
    rng = np.random.default_rng(_SEED)
    for params in (default_params(ref.full), draw_params(ref.full, rng)):
        dense = ref.published(params, PUB_DENSE)
        for label, grid in (("coarse-40", PUB_COARSE), ("filter-12", PUB_FILTERS)):
            other = ref.published(params, grid)
            for key in KEYS:
                err = abs(other[key] / dense[key] - 1.0)
                assert err < _RTOL, f"GRAHSP[{name}] {key} {label}: rel diff {err:.3e}"


def test_published_L_equals_dense_integral_over_the_declared_box(reference):
    """200 random draws of the declared box (plus the defaults): every key within 1e-6.

    The draws cover every free parameter at once, so a piece whose publication depends on
    another piece's option (the torus under the Netzer disc, say) shows as a miss.
    """
    name, ref = reference
    rng = np.random.default_rng(_SEED)
    points = [default_params(ref.full)] + [draw_params(ref.full, rng) for _ in range(_N_DRAWS)]
    worst = dict.fromkeys(KEYS, 0.0)
    n_binding = 0
    for params in points:
        expected, n_roots = ref.values(params)
        n_binding += int(n_roots > 0)
        for key, err in _rel(ref.published(params, PUB_FILTERS), expected).items():
            worst[key] = max(worst[key], abs(err))
    assert max(worst.values()) < _RTOL, f"GRAHSP[{name}] worst rel {worst}"
    if name in ("default", "netzer_disc", "veroncetty_feii", "sy2"):
        assert n_binding >= 5, f"vacuous draw set: {n_binding} draws where the Si clip binds"


@pytest.mark.parametrize("name", ["default", "all_options"])
def test_published_L_identity_with_the_structural_knobs_drawn(name):
    """60 draws that also vary the Balmer strength, the line width and the BBB cutoff (1e-6)."""
    ref = Reference(**CONFIGS[name])
    rng = np.random.default_rng(_SEED + 1)
    worst = dict.fromkeys(KEYS, 0.0)
    for _ in range(60):
        params = draw_params(ref.full, rng, extended=True)
        expected, _ = ref.values(params)
        for key, err in _rel(ref.published(params, PUB_FILTERS), expected).items():
            worst[key] = max(worst[key], abs(err))
    assert max(worst.values()) < _RTOL, f"GRAHSP[{name}] worst rel {worst}"


@pytest.mark.parametrize("bend_width", [0.1, 10.0])
@pytest.mark.parametrize("bend_loc_nm", [50.0, 150.0])
def test_published_L_bol_includes_the_power_law_tail_at_the_prior_edge(bend_width, bend_loc_nm):
    """At plslope = -1 (the prior edge) the BBB tail is inside L_agn_bol and L_agn_absorbed (1e-6).

    Beyond the 10 um cutoff the big blue bump falls as lambda**(plslope - 1), so at the shallow
    edge of the prior it carries tens of ppm of the luminosity above 1e8 nm. The reference
    reaches 1e15 nm.
    """
    ref = Reference()
    params = default_params(ref.full)
    params["agn_grahsp_plslope"] = jnp.asarray(-1.0)
    params["agn_grahsp_plbendwidth"] = jnp.asarray(bend_width)
    params["agn_grahsp_plbendloc_nm"] = jnp.asarray(bend_loc_nm)
    expected, _ = ref.values(params)
    errs = _rel(ref.published(params, PUB_FILTERS), expected)
    for key in ("L_agn_bol", "L_agn_absorbed"):
        assert abs(errs[key]) < _RTOL, f"{key} at plslope=-1: rel {errs[key]:+.3e}"


def test_sbpl_disc_block_normalization_includes_the_power_law_tail():
    """The SBPL disc block's bolometric normalization holds to 1e-6 at plslope = -1.

    With ``agn_grahsp_log_l5100`` unset the block scales its unit spectrum so that the
    integral above 91.1753 nm is ``10**agn_log_lbol`` L_sun. The integral of the block's own
    output, taken to 1e15 nm, must reproduce that target.
    """
    from tengri.components.agn.blocks.grahsp_blocks import grahsp_sbpl_disc_block
    from tengri.utils.physics_constants import L_SUN

    n = (20 * 13) + 1
    breaks = np.geomspace(10.0 * BBB_BOL_CUT_NM, 1.0e16, n)  # Angstrom, from lumBolBBB's cut
    x, w = np.polynomial.legendre.leggauss(16)
    lo, hi = np.log(breaks[:-1]), np.log(breaks[1:])
    half = 0.5 * (hi - lo)[:, None]
    lam_aa = np.exp(lo[:, None] + half * (x[None, :] + 1.0))
    weight = half * w[None, :] * lam_aa
    for plslope in (-1.0, -1.6):
        out = grahsp_sbpl_disc_block(
            jnp.asarray(lam_aa.ravel()),
            45.0,
            agn_grahsp_plslope=plslope,
            agn_grahsp_plbendwidth=10.0,
        )
        total = float(np.sum(weight.ravel() * np.asarray(out)))
        err = total / (10.0**45.0 * L_SUN) - 1.0
        assert abs(err) < _RTOL, f"block normalization at plslope={plslope}: rel {err:+.3e}"


def test_published_L_identity_at_the_attenuation_and_amplitude_corners():
    """L_agn_bol and L_agn_absorbed match the reference at every box corner (1e-7).

    The corners are those of the parameters that enter the attenuation (E(B-V), E(B-V)_AGN)
    and the line and FeII amplitudes (A_lines, A_FeII); the rest sit at their declared
    defaults, and L5100 scales every piece alike. The lines, FeII forest and continua are
    linear in the amplitudes, so each piece is evaluated once at unit amplitude on the
    reference nodes and combined per corner with the attenuation of that corner.
    """
    from tengri.components.agn.grahsp.attenuation import attenuation_factors

    ref = Reference()
    unit = default_params(ref.full)
    unit["agn_grahsp_a_lines"] = jnp.asarray(1.0)
    unit["agn_grahsp_a_feii"] = jnp.asarray(1.0)
    lam, w, _ = ref.nodes(unit)
    only = {
        "include_bbb": False,
        "include_lines": False,
        "include_feii": False,
        "include_torus": False,
        "include_si": False,
        "include_balmer": False,
        "apply_attenuation": False,
    }
    piece = {
        name: ref.evaluator(**{**only, **on})(lam, unit)
        for name, on in (
            ("bbb", {"include_bbb": True}),
            ("lines", {"include_lines": True}),
            ("feii", {"include_feii": True}),
            ("torus", {"include_torus": True, "include_si": True}),
        )
    }
    bounds = {d.name: d.prior.bounds for d in ref.full.declared_parameters()}
    names = ("agn_grahsp_ebv", "agn_grahsp_ebv_agn", "agn_grahsp_a_feii", "agn_grahsp_a_lines")
    above_lyman = lam >= 91.1753
    worst = {"L_agn_bol": 0.0, "L_agn_absorbed": 0.0}
    for corner in itertools.product(*(bounds[n] for n in names)):
        params = dict(unit)
        params.update({n: jnp.asarray(v) for n, v in zip(names, corner)})
        a_feii, a_lines = corner[2], corner[3]
        screen = np.asarray(attenuation_factors(jnp.asarray(lam), corner[0], corner[1])[1])
        bbb_side = piece["bbb"] + a_lines * (piece["lines"] + a_feii * piece["feii"])
        ref_bol = float(np.sum(w * np.where(above_lyman, bbb_side, 0.0)))
        ref_abs = float(np.sum(w * (bbb_side + piece["torus"]) * (1.0 - screen)))
        pub = ref.published(params, PUB_FILTERS)
        worst["L_agn_bol"] = max(worst["L_agn_bol"], abs(pub["L_agn_bol"] / ref_bol - 1.0))
        worst["L_agn_absorbed"] = max(
            worst["L_agn_absorbed"], abs(pub["L_agn_absorbed"] / ref_abs - 1.0)
        )
    for key, err in worst.items():
        assert err < _CORNER_RTOL, f"GRAHSP {key}: worst corner rel {err:.3e}"


_BINDING = {
    "agn_grahsp_si": -4.0,
    "agn_grahsp_cool_lam_um": 30.0,
    "agn_grahsp_cool_width": 0.2,
    "agn_grahsp_hot_lam_um": 1.0,
    "agn_grahsp_hot_width": 0.2,
    "agn_grahsp_hot_fcov": 0.04,
}


def test_published_L_identity_where_si_clip_binds():
    """At a corner where the emitted max(si, -torus) binds, every key matches the reference.

    The clip is a kink of the emitted torus term; the published values carry its correction
    over the Si windows. The corner is one where the emitted torus plus Si reaches zero.
    """
    ref = Reference()
    params = default_params(ref.full)
    params.update({k: jnp.asarray(v) for k, v in _BINDING.items()})
    expected, n_roots = ref.values(params)
    assert n_roots >= 2, "the corner does not bind the clip"
    for key, err in _rel(ref.published(params, PUB_FILTERS), expected).items():
        assert abs(err) < _RTOL, f"{key} at the binding corner: rel {err:+.3e}"


def test_torus_clip_gate_never_skips_a_binding_clip():
    """Across random draws of the declared box, L_agn_torus equals the dense integral (1e-8).

    The Si clip correction is skipped when a rigorous bound shows ``max(si, -torus)`` cannot
    bind. The draws cover both classes (bound and unbound); the expectation is the reference
    integral of the emitted torus term, shared with no closed form of the component.
    """
    ref = Reference(
        include_bbb=False,
        include_lines=False,
        include_feii=False,
        include_balmer=False,
        apply_attenuation=False,
    )
    rng = np.random.default_rng(_SEED + 2)
    n_bound = n_unbound = 0
    for _ in range(80):
        params = draw_params(ref.full, rng)
        expected, n_roots = ref.values(params)
        n_bound += int(n_roots > 0)
        n_unbound += int(n_roots == 0)
        err = ref.published(params, PUB_FILTERS)["L_agn_torus"] / expected["L_agn_torus"] - 1.0
        assert abs(err) < 1.0e-8, f"L_agn_torus {err:+.2e} ({n_roots} crossings)"
    assert n_bound >= 5 and n_unbound >= 5, f"vacuous: bound {n_bound}, unbound {n_unbound}"


def test_torus_absorbed_power_is_exact_over_torus_shapes_at_high_extinction():
    """L_agn_absorbed of the torus alone, random torus shapes x E(B-V) in [3, 10]: 1e-7.

    At high E(B-V) the weight 1 - attenuation varies over the log-normal torus by orders of
    magnitude and has a kink at 1100 nm, so the fixed-node expectation over the torus is
    stressed by both. The tolerance is the accuracy the order of that rule is built for; a
    rule with half the nodes misses it.
    """
    ref = Reference(
        include_bbb=False,
        include_lines=False,
        include_feii=False,
        include_balmer=False,
    )
    rng = np.random.default_rng(_SEED + 3)
    worst = 0.0
    for _ in range(_N_DRAWS):
        params = draw_params(ref.full, rng)
        params["agn_grahsp_ebv"] = jnp.asarray(np.exp(rng.uniform(np.log(3.0), np.log(10.0))))
        expected, _ = ref.values(params)
        pub = ref.published(params, PUB_FILTERS)["L_agn_absorbed"]
        worst = max(worst, abs(pub / expected["L_agn_absorbed"] - 1.0))
    assert worst < 1.0e-7, f"torus L_agn_absorbed worst rel {worst:.3e}"


def test_feii_absorbed_piece_matches_independent_quadrature_at_the_corners():
    """The FeII x (1 - attenuation) piece equals 24-point Gauss-Legendre per segment (1e-10).

    The reference is a different rule at a different resolution from the one it checks (two
    points per linear segment), so it does not share that rule's error. The corners are those
    of the parameters that enter the attenuation and the FeII amplitude. The amplitude is read
    off the emitted FeII spectrum at a template node (the template is the only shape).
    """
    from tengri.components.agn.grahsp.attenuation import attenuation_factors

    ref = Reference(
        include_bbb=False,
        include_lines=False,
        include_torus=False,
        include_si=False,
        include_balmer=False,
    )
    wave = np.asarray(ref.tpl.feii_wave_nm, dtype=float)
    lumin = np.asarray(ref.tpl.feii_lumin, dtype=float)
    x, wt = np.polynomial.legendre.leggauss(24)
    mid, half = 0.5 * (wave[:-1] + wave[1:]), 0.5 * (wave[1:] - wave[:-1])
    lam = mid[:, None] + half[:, None] * x[None, :]
    lum = (
        lumin[:-1, None]
        + (lumin[1:] - lumin[:-1])[:, None]
        * (lam - wave[:-1, None])
        / (wave[1:] - wave[:-1])[:, None]
    )
    bounds = {d.name: d.prior.bounds for d in ref.full.declared_parameters()}
    names = ("agn_grahsp_ebv", "agn_grahsp_ebv_agn", "agn_grahsp_a_feii", "agn_grahsp_a_lines")
    unattenuated = ref.evaluator(apply_attenuation=False)
    k = int(np.argmax(lumin))
    for corner in itertools.product(*(bounds[n] for n in names)):
        params = default_params(ref.full)
        params.update({n: jnp.asarray(v) for n, v in zip(names, corner)})
        amplitude = float(unattenuated(wave[k : k + 1], params)[0]) / lumin[k]
        screen = np.asarray(attenuation_factors(jnp.asarray(lam.ravel()), corner[0], corner[1])[1])
        expected = amplitude * float(
            np.sum(half[:, None] * wt[None, :] * lum * (1.0 - screen).reshape(lam.shape))
        )
        pub = ref.published(params, PUB_FILTERS)["L_agn_absorbed"]
        assert abs(pub / expected - 1.0) < 1.0e-10, (
            f"FeII at {corner}: rel {pub / expected - 1:+.2e}"
        )


def test_absorbed_lines_at_wide_lines_match_the_dense_integral():
    """L_agn_absorbed of the emission lines alone at FWHM 2e4 km/s, E(B-V) in {0.5, 1}: 1e-6.

    At this width the 1083 nm line sits about half a sigma from the 1100 nm attenuation kink,
    so the kink lies inside the window of that line and of its neighbors. The line rule cuts
    the window at the kink. A fixed-node Gauss-Hermite rule across the kink converges only
    algebraically: measured at 13 points it is off by 1.2e-6 here, and the error is largest at
    moderate E(B-V), where the attenuation is still steep at the kink.
    """
    ref = Reference(
        include_bbb=False,
        include_feii=False,
        include_torus=False,
        include_si=False,
        include_balmer=False,
    )
    rng = np.random.default_rng(_SEED + 4)
    worst = 0.0
    for ebv in (0.5, 1.0):
        for _ in range(20):
            params = draw_params(ref.full, rng)
            params["agn_grahsp_linewidth_kms"] = jnp.asarray(2.0e4)
            params["agn_grahsp_ebv"] = jnp.asarray(ebv)
            expected, _ = ref.values(params)
            pub = ref.published(params, PUB_FILTERS)["L_agn_absorbed"]
            worst = max(worst, abs(pub / expected["L_agn_absorbed"] - 1.0))
    assert worst < _RTOL, f"lines L_agn_absorbed at 2e4 km/s worst rel {worst:.3e}"
