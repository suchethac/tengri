# SPDX-License-Identifier: BSD-3-Clause
"""The ``(tau_V, mu)`` build path: energy-balance LUT hull and free ``dust_mu`` (#3176).

``dust_parameterization='tauv_mu'`` declares ``dust_tau_v`` and ``dust_mu`` in place of
the birth-cloud / diffuse pair, so the energy-balance LUT has no ``dust_tau_bc`` /
``dust_tau_diff`` prior to take its axes from. Its (tau_bc, tau_diff) hull is the image
of the (tau_V, mu) box. These tests compare the LUT path against the exact path (the
LUT is disabled by removing ``dust_tau_v`` / ``dust_mu`` from the allowed sets), the
same comparison the single-screen sibling in ``test_2485_single_screen_energy_balance``
makes, and check the gradients, precision and the blue-band warning path.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import tengri
from tengri import DEFAULT, Fixed, Observation, SEDModel, Uniform
from tengri.forward.sed_model import WavePrecomp
from tengri.observation import Photometry

pytestmark = pytest.mark.regression_bug

#: Same filter set as the single-screen sibling (far-IR carries the dust emission).
FILTERS = [
    "hst_f606w",
    "hst_f160w",
    "irac_45",
    "Spitzer_MIPS_24mu",
    "Spitzer_MIPS_70mu",
    "Herschel_Pacs_green",
    "Herschel_SPIRE_PSW",
    "Herschel_SPIRE_PMW",
]

#: Between LUT nodes the energy-balance LUT interpolates; the sibling measured up to
#: ~6% at small tau on the 24-node grid and bounds it at 10%. The same bound applies here.
INTERP_RTOL = 0.10

#: At a LUT node the interpolation is exact by construction (sibling: 1e-8).
NODE_RTOL = 1e-8

# Off-node ratchets. These small-depth bounds are ratchets on the linear-node LUT (#3199):
# the grid places its 24 nodes linearly in tau, and between nodes the linear interpolant
# error is set by the node spacing at small depth, not by the (tau_V, mu) mapping. The
# native parameterization carries the same error (see the shared-error test below). They
# must be tightened when the geometric node placement (#3199) lands.
OFF_NODE_RATCHET = {
    ("a", 0.05): 1e-1,
    ("a", 1.0): 5e-3,
    ("a", 3.9): 2e-4,
    ("b", 0.01, 0.05): 2e-3,
    ("b", 0.01, 0.95): 0.25,
    ("b", 4.0, 0.05): 5e-2,
    ("b", 4.0, 0.95): 1e-7,
    ("b", 2.0, 0.5): 1e-3,
    ("e", 0.05): 0.4,
}

#: The LUT's per-node depth axis size (``_grid`` in ``SEDModel``).
N_NODES = 24

TAU_V_BOUNDS = (0.01, 4.0)
MU_BOUNDS = (0.05, 0.95)


def _obs(filters=FILTERS):
    return Observation(photometry=Photometry.from_names(filters))


def _tauv_mu_model(ssp, *, tau_v, mu, law_diff="calzetti", filters=FILTERS, approx=None):
    """A two-screen model declared with the ``(tau_V, mu)`` parameterization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=_obs(filters),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "parameterization": "tauv_mu",
                "law_bc": "calzetti",
                "law_diff": law_diff,
                "tau_v": tau_v,
                "mu": mu,
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
            approx=WavePrecomp() if approx is None else approx,
        )


def _exact_model(ssp, monkeypatch, *, drop, **kwargs):
    """A model with the LUT disabled for the named free dust parameters (the pre-fix gate).

    The patch is scoped to this build, so the LUT arm of a later call stays unpatched.
    """
    with monkeypatch.context() as patch:
        patch.setattr(
            SEDModel,
            "_EB_ATTEN_FREE_OK",
            frozenset(SEDModel._EB_ATTEN_FREE_OK - set(drop)),
        )
        return _tauv_mu_model(ssp, **kwargs)


def _rel_err(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _lut_vs_exact(ssp, monkeypatch, *, tau_v, mu, drop, law_diff="calzetti", points):
    """Relative photometry error of the LUT path against the exact path at each point."""
    lut_model = _tauv_mu_model(ssp, tau_v=tau_v, mu=mu, law_diff=law_diff)
    assert lut_model._energy_balance_lut_cache is not None, "LUT arm did not engage"
    exact_model = _exact_model(ssp, monkeypatch, drop=drop, tau_v=tau_v, mu=mu, law_diff=law_diff)
    assert exact_model._energy_balance_lut_cache is None, "control arm built the LUT"

    base = dict(lut_model.spec.sample(jax.random.PRNGKey(0)))
    errors = []
    for point in points:
        p = {**base, **point}
        a = np.asarray(lut_model.predict_photometry(p))
        b = np.asarray(exact_model.predict_photometry(p))
        errors.append((point, _rel_err(a, b)))
    return lut_model, errors


# ---------------------------------------------------------------------------
# (a) free tau_V, fixed mu: the LUT hull and LUT-vs-exact at three depths
# ---------------------------------------------------------------------------


def test_free_tau_v_lut_hull_is_the_tau_bc_tau_diff_image_of_the_box(synthetic_ssp_wide):
    """The axes span the image of the (tau_V, mu) box, not [0, 1] or the prior of a missing key."""
    model = _tauv_mu_model(synthetic_ssp_wide, tau_v=Uniform(*TAU_V_BOUNDS), mu=Fixed(0.4))
    lut = model._energy_balance_lut_cache
    assert lut is not None
    v_lo, v_hi = TAU_V_BOUNDS
    m = 0.4
    assert_allclose(float(lut.tau_bc_grid[0]), (1.0 - m) * v_lo, rtol=1e-12)
    assert_allclose(float(lut.tau_bc_grid[-1]), (1.0 - m) * v_hi, rtol=1e-12)
    assert_allclose(float(lut.tau_diff_grid[0]), m * v_lo, rtol=1e-12)
    assert_allclose(float(lut.tau_diff_grid[-1]), m * v_hi, rtol=1e-12)
    assert lut.tau_bc_grid.shape[0] == N_NODES
    assert lut.tau_diff_grid.shape[0] == N_NODES


def test_free_tau_v_lut_matches_exact_path(synthetic_ssp_wide, monkeypatch):
    """LUT vs exact at tau_V = 0.05, 1.0, 3.9 (between nodes) and at the grid's own nodes."""
    v_nodes = np.linspace(*TAU_V_BOUNDS, N_NODES)
    interior = [{"dust_tau_v": v} for v in (0.05, 1.0, 3.9)]
    # With mu fixed, tau_bc = (1-m) tau_V and tau_diff = m tau_V land on the grid's nodes
    # whenever tau_V does, so these points are exact to NODE_RTOL.
    at_nodes = [{"dust_tau_v": float(v_nodes[k])} for k in (1, 8, 23)]
    _, interior_err = _lut_vs_exact(
        synthetic_ssp_wide,
        monkeypatch,
        tau_v=Uniform(*TAU_V_BOUNDS),
        mu=Fixed(0.4),
        drop={"dust_tau_v"},
        points=interior,
    )
    _, node_err = _lut_vs_exact(
        synthetic_ssp_wide,
        monkeypatch,
        tau_v=Uniform(*TAU_V_BOUNDS),
        mu=Fixed(0.4),
        drop={"dust_tau_v"},
        points=at_nodes,
    )
    for point, err in interior_err:
        tau_v = point["dust_tau_v"]
        print(f"(a) interior tau_v={tau_v}: rel err {err:.3e}")
        bound = OFF_NODE_RATCHET[("a", tau_v)]
        assert err < bound, f"interior LUT error {err:.3e} above ratchet {bound:g} at {point}"
    for point, err in node_err:
        print(f"(a) node tau_v={point['dust_tau_v']}: rel err {err:.3e}")
        assert err < NODE_RTOL, f"node LUT error {err:.3e} at {point}"


# ---------------------------------------------------------------------------
# (b) free tau_V and free mu: the four corners and the center of the box
# ---------------------------------------------------------------------------


def test_free_tau_v_and_mu_lut_matches_exact_at_box_corners_and_center(
    synthetic_ssp_wide, monkeypatch
):
    """A free dust_mu may take the LUT (#3176): corners and center agree with the exact path."""
    corners = [{"dust_tau_v": v, "dust_mu": m} for v in TAU_V_BOUNDS for m in MU_BOUNDS]
    center = {"dust_tau_v": 2.0, "dust_mu": 0.5}
    _, errors = _lut_vs_exact(
        synthetic_ssp_wide,
        monkeypatch,
        tau_v=Uniform(*TAU_V_BOUNDS),
        mu=Uniform(*MU_BOUNDS),
        drop={"dust_tau_v", "dust_mu"},
        points=[*corners, center],
    )
    for point, err in errors:
        key = ("b", point["dust_tau_v"], point["dust_mu"])
        print(f"(b) tau_v={key[1]:.3f} mu={key[2]:.3f}: rel err {err:.3e}")
        bound = OFF_NODE_RATCHET[key]
        assert err < bound, f"box LUT error {err:.3e} above ratchet {bound:g} at {point}"


# ---------------------------------------------------------------------------
# (c) gradients wrt tau_V and mu through the LUT
# ---------------------------------------------------------------------------


def test_gradient_wrt_tau_v_and_mu_is_finite_and_nonzero(synthetic_ssp_wide):
    """jax.grad of a band flux wrt dust_tau_v and dust_mu through the LUT, at an interior point."""
    model = _tauv_mu_model(
        synthetic_ssp_wide, tau_v=Uniform(*TAU_V_BOUNDS), mu=Uniform(*MU_BOUNDS)
    )
    assert model._energy_balance_lut_cache is not None
    base = dict(model.spec.sample(jax.random.PRNGKey(0)))

    def total(v, m):
        return jnp.sum(model.predict_photometry({**base, "dust_tau_v": v, "dust_mu": m}))

    d_v, d_m = jax.grad(total, argnums=(0, 1))(jnp.asarray(1.0), jnp.asarray(0.4))
    assert np.isfinite(float(d_v)) and np.isfinite(float(d_m))
    assert float(d_v) != 0.0
    assert float(d_m) != 0.0


# ---------------------------------------------------------------------------
# (d) float32 against float64
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:  # pragma: no cover - depends on checkout
        pytest.skip(f"default bare-stellar SSP not available: {exc}")


def test_float32_matches_float64_on_the_tauv_mu_lut(ssp):
    """The float32 build agrees with float64 to 1e-4 (the chevallard13 float32 tolerance).

    Uses the shipped SSP. The synthetic wide grid gives NaN in float32 for the native
    two-component model too, so it cannot carry this comparison.
    """
    point = {"dust_tau_v": 1.0, "dust_mu": 0.4}
    model64 = _tauv_mu_model(ssp, tau_v=Uniform(*TAU_V_BOUNDS), mu=Uniform(*MU_BOUNDS))
    base = dict(model64.spec.sample(jax.random.PRNGKey(0)))
    a = np.asarray(model64.predict_photometry({**base, **point}))
    with jax.enable_x64(False):
        model32 = _tauv_mu_model(ssp, tau_v=Uniform(*TAU_V_BOUNDS), mu=Uniform(*MU_BOUNDS))
        b = np.asarray(model32.predict_photometry({**base, **point}))
    assert_allclose(b, a, rtol=1e-4)


# ---------------------------------------------------------------------------
# (e) chevallard13 as law_diff: each node's curve is evaluated at its own depth
# ---------------------------------------------------------------------------


def test_chevallard13_diffuse_lut_matches_exact_at_three_tau_v(synthetic_ssp_wide, monkeypatch):
    """The LUT with a depth-dependent diffuse law agrees with the exact path.

    A LUT that evaluated every node's curve at one depth would miss this by far more
    than the interpolation bound.
    """
    v_nodes = np.linspace(*TAU_V_BOUNDS, N_NODES)
    # Three node depths, where the LUT is exact: a per-node curve at the wrong depth fails.
    at_nodes = [{"dust_tau_v": float(v_nodes[k])} for k in (1, 8, 23)]
    _, errors = _lut_vs_exact(
        synthetic_ssp_wide,
        monkeypatch,
        tau_v=Uniform(*TAU_V_BOUNDS),
        mu=Fixed(0.4),
        drop={"dust_tau_v"},
        law_diff="chevallard13",
        points=at_nodes,
    )
    for point, err in errors:
        print(f"(e) chevallard13 node tau_v={point['dust_tau_v']:.4f}: rel err {err:.3e}")
        assert err < NODE_RTOL, f"chevallard13 node LUT error {err:.3e} at {point}"
    _, off_node = _lut_vs_exact(
        synthetic_ssp_wide,
        monkeypatch,
        tau_v=Uniform(*TAU_V_BOUNDS),
        mu=Fixed(0.4),
        drop={"dust_tau_v"},
        law_diff="chevallard13",
        points=[{"dust_tau_v": 0.05}],
    )
    for point, err in off_node:
        print(f"(e) chevallard13 off-node tau_v={point['dust_tau_v']}: rel err {err:.3e}")
        bound = OFF_NODE_RATCHET[("e", point["dust_tau_v"])]
        assert err < bound, f"chevallard13 off-node error {err:.3e} above ratchet {bound:g}"


def _native_model(ssp, *, tau_bc, tau_diff):
    """A native two-screen model with free tau_BC and tau_diff over the given priors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=_obs(),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "calzetti",
                "tau_bc": tau_bc,
                "tau_diff": tau_diff,
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
            approx=WavePrecomp(),
        )


def test_native_and_tauv_mu_share_the_linear_node_error(synthetic_ssp_wide):
    """Same nodes, same interpolant: the (tau_V, mu) hull adds no error of its own.

    With mu = 0.4 the (tau_V, mu) hull is tau_BC in [0.006, 2.4] and tau_diff in
    [0.004, 1.6]. Native priors over exactly that hull give the same 24 nodes, so the
    native LUT and the tauv_mu LUT interpolate identically at (tau_BC, tau_diff) =
    (0.03, 0.02) = (0.6, 0.4) x tau_V = 0.05.
    """
    native = _native_model(
        synthetic_ssp_wide,
        tau_bc=Uniform(0.006, 2.4),
        tau_diff=Uniform(0.004, 1.6),
    )
    tauv_mu = _tauv_mu_model(synthetic_ssp_wide, tau_v=Uniform(*TAU_V_BOUNDS), mu=Fixed(0.4))
    assert native._energy_balance_lut_cache is not None
    assert tauv_mu._energy_balance_lut_cache is not None
    a = np.asarray(
        native.predict_photometry(
            {
                **dict(native.spec.sample(jax.random.PRNGKey(0))),
                "dust_tau_bc": 0.03,
                "dust_tau_diff": 0.02,
            }
        )
    )
    b = np.asarray(
        tauv_mu.predict_photometry(
            {**dict(tauv_mu.spec.sample(jax.random.PRNGKey(0))), "dust_tau_v": 0.05}
        )
    )
    assert_allclose(b, a, rtol=1e-6)


# ---------------------------------------------------------------------------
# (f) the blue-band warning path does not crash on a tauv_mu model
# ---------------------------------------------------------------------------


def test_blue_band_warning_path_runs_for_tauv_mu_with_free_tau_v(synthetic_ssp_wide):
    """The n_subbands=0 warning reads dust_tau_v's bound, so a GALEX-like band must not crash."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        model = _tauv_mu_model(
            synthetic_ssp_wide,
            tau_v=Uniform(*TAU_V_BOUNDS),
            mu=Fixed(0.4),
            filters=["galex_fuv", "sdss_r"],
            approx=WavePrecomp(n_subbands=0),
        )
    assert model is not None
    assert model._max_plausible_tau("dust_tau_v") == pytest.approx(TAU_V_BOUNDS[1])
