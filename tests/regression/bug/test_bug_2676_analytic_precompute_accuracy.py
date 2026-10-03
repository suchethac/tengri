# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2676: accuracy of the analytic dust precompute lookup.

``dust_analytic_precompute`` preintegrates the thermal-continuum models
(``modified_blackbody``, ``graybody``, ``casey2012``) and the ``pah_drude`` template
through observed-frame filters. Three limits are pinned here against a converged
quadrature of the closed-form spectrum, written in this file and independent of the package:

* off-node error: at the default node grids the triweight lookup was 4-10 % off the band
  integral between nodes; the lookup is now a cubic spline of ``ln(flux)`` on nodes spanning the
  declared prior ranges and is accurate to ``TABLE_RTOL`` off-node (``TOL`` below);
* the Wien tail: a 1500-point rest grid, linearly interpolated, overestimated the band flux of a
  cold spectrum by ~1 %;
* outside the rest grid: a band beyond 0.01 um - 10 mm, or ``pah_drude`` beyond 31.6 um, read 0.

Reference: with ``T_eff`` the CMB-heated temperature of da Cunha et al. (2013), the closures return
``S(lambda) = shape(lambda) / I * (1 - B(T_cmb) / B(T_eff))`` per unit absorbed luminosity, with
``I`` the frequency integral of ``shape`` over 0.01 um - 10 mm. Here ``I`` is a composite
Gauss-Legendre rule in ``ln lambda`` (cross-checked with ``scipy.integrate.quad``) and the band
flux is

.. math::

    \\Phi = \\frac{\\int S(\\lambda_{obs}/(1+z))\\, T(\\lambda_{obs})\\, \\lambda_{obs}^{-1}
    d\\lambda_{obs}}{\\int T(\\lambda_{obs})\\, \\lambda_{obs}^{-1} d\\lambda_{obs}},

with ``T`` linear between the filter's nodes: the convention of
``tengri.utils.grid_interp.preintegrate_grid``. Flux values below ``TABLE_FLUX_FLOOR`` (1e-150 per
unit absorbed luminosity) are outside the accuracy domain: the closures clip ``h nu / k T`` at 500
there.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from tengri.components.dust import dust_analytic_precompute as adapter
from tengri.components.dust._params import PARAMS
from tengri.components.dust.drude_profiles import SMITH2007_PAH_FEATURES
from tengri.components.dust.emission import DUST_EMISSION_MODELS as M
from tengri.utils.grid_interp import preintegrate_grid
from tengri.utils.physics_constants import C_CGS, H_PLANCK, K_BOLTZ

pytestmark = pytest.mark.regression_bug

TOL = 1.0e-3
FLOOR = 1.0e-150
REDSHIFTS = (0.0, 0.5, 2.0, 6.0)
_HC_K_CM = H_PLANCK * C_CGS / K_BOLTZ  # cm K
_T_CMB0 = 2.725
_CONTINUUM = ("modified_blackbody", "graybody", "casey2012")
_ALL_MODELS = tuple(adapter.AXIS_PARAMS)
_GRID_KEYWORD = {
    "dust_T": "T_grid",
    "dust_beta_ir": "beta_grid",
    "dust_alpha_mir": "alpha_mir_grid",
    "dust_lambda_0_um": "lambda_0_um_grid",
}


# ── Filters ───────────────────────────────────────────────────────────────────────────────────


def _tophat(lo_um, hi_um, n=24):
    inner = np.geomspace(lo_um * 1e4, hi_um * 1e4, n)
    wave = np.concatenate([[inner[0] * (1 - 1e-6)], inner, [inner[-1] * (1 + 1e-6)]])
    return wave, np.concatenate([[0.0], np.ones(n), [0.0]])


def _gauss(center_um, frac=0.12, n=48):
    wave = np.linspace(center_um * (1 - 3 * frac), center_um * (1 + 3 * frac), n) * 1e4
    return wave, np.exp(-0.5 * ((wave / 1e4 - center_um) / (frac * center_um)) ** 2)


#: Mid-infrared to millimeter: narrow, broad (Wien-side) and Gaussian bands.
_FILTERS = (
    _tophat(7.0, 9.0),
    _tophat(8.0, 24.0),
    _gauss(24.0),
    _tophat(60.0, 90.0),
    _gauss(160.0),
    _tophat(250.0, 500.0),
    _gauss(850.0),
    _tophat(1000.0, 1300.0),
    _tophat(2500.0, 3500.0),
)
_FILTER_WAVES = [f[0] for f in _FILTERS]
_FILTER_TRANS = [f[1] for f in _FILTERS]


# ── Closed-form references ────────────────────────────────────────────────────────────────────


def _bose(x):
    return np.exp(-x) / -np.expm1(-x)


def _log_expm1(x):
    return x + np.log1p(-np.exp(-x))


def _t_eff(t_dust, beta, z):
    k = 4.0 + beta
    return (max(t_dust, 1.0) ** k + (_T_CMB0 * (1 + z)) ** k - _T_CMB0**k) ** (1.0 / k)


def _shape(model, lam_cm, t_eff, p):
    """Unnormalized spectrum (constants common to every term dropped)."""
    nu = C_CGS / lam_cm
    x = _HC_K_CM / (lam_cm * t_eff)
    if model == "modified_blackbody":
        return (nu / (C_CGS / 250.0e-4)) ** p["dust_beta_ir"] * nu**3 * _bose(x)
    tau = (p["dust_lambda_0_um"] * 1e-4 / lam_cm) ** p["dust_beta_ir"]
    gray = -np.expm1(-tau) * (1.0 / lam_cm) ** 3 * _bose(x)
    if model == "graybody":
        return gray
    a = p["dust_alpha_mir"]
    denom = (26.68 + 6.246 * a) ** -2.0 + (1.905e-4 + 7.243e-5 * a) * t_eff
    lam_c = 0.75e3 / denom * 1e-7
    xc = _HC_K_CM / (lam_c * t_eff)
    tau_c = (p["dust_lambda_0_um"] * 1e-4 / lam_c) ** p["dust_beta_ir"]
    n_pl = -np.expm1(-tau_c) * (1.0 / lam_c) ** 3 * _bose(xc)
    return gray + n_pl * (lam_c**-a * lam_cm**a) * np.exp(-((lam_cm / lam_c) ** 2))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def _norm_integral(model, t_eff, p):
    """Frequency integral of the shape over 0.01 um - 10 mm (composite Gauss-Legendre)."""
    edges = np.linspace(np.log(1e-6), np.log(1.0), 2401)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ln_lam = (half[:, None] * _GL_X[None, :] + mid[:, None]).ravel()
    w = (half[:, None] * _GL_W[None, :]).ravel()
    lam = np.exp(ln_lam)
    return np.sum(w * _shape(model, lam, t_eff, p) * C_CGS / lam)


def _spectrum(model, p, z):
    """Closed-form ``S(lambda_aa)`` [1/Hz per unit absorbed luminosity] at redshift ``z``."""
    t_eff = _t_eff(p["dust_T"], p["dust_beta_ir"], z)
    t_cmb = _T_CMB0 * (1 + z)
    norm = _norm_integral(model, t_eff, p)

    def s(lam_aa):
        lam = np.asarray(lam_aa) * 1e-8
        ratio = np.exp(_log_expm1(_HC_K_CM / (lam * t_eff)) - _log_expm1(_HC_K_CM / (lam * t_cmb)))
        return _shape(model, lam, t_eff, p) / norm * (1.0 - ratio)

    return s, t_eff


def _pah_spectrum(lam_aa):
    lam_um = np.asarray(lam_aa) * 1e-4
    total = np.zeros_like(lam_um)
    for f in SMITH2007_PAH_FEATURES:
        x = lam_um / f.wave_um - f.wave_um / lam_um
        total += f.strength * f.gamma**2 / (x**2 + f.gamma**2)
    return total * (lam_aa * 1e-8) ** 2 / C_CGS


def _band_nodes(fw, ft, z, slope):
    """Composite Gauss-Legendre nodes/weights of one filter (10 points, slope x width <= 0.5)."""
    gx, gw = np.polynomial.legendre.leggauss(10)
    live = (fw[1:] > fw[:-1]) & ((ft[:-1] > 0) | (ft[1:] > 0))
    lo, hi = fw[:-1][live], fw[1:][live]
    n_sub = np.maximum(1, np.ceil(np.log(hi / lo) * slope(0.5 * (lo + hi) / (1 + z)) / 0.5))
    nodes, weights = [], []
    for a, b, n in zip(lo, hi, n_sub.astype(int), strict=True):
        edges = np.linspace(a, b, n + 1)
        half = 0.5 * np.diff(edges)
        mid = 0.5 * (edges[1:] + edges[:-1])
        x = (half[:, None] * gx[None, :] + mid[:, None]).ravel()
        nodes.append(x)
        weights.append((half[:, None] * gw[None, :]).ravel() * np.interp(x, fw, ft) / x)
    x, w = np.concatenate(nodes), np.concatenate(weights)
    return x, w / w.sum()


def _band(spec, fw, ft, z, t_eff):
    """Band flux of spectrum ``spec`` through one filter, ``Phi``."""
    slope = lambda lam_aa: np.minimum(_HC_K_CM / (lam_aa * 1e-8 * t_eff), 500.0) + 10.0  # noqa: E731
    x, w = _band_nodes(fw, ft, z, slope)
    return float(np.sum(w * spec(x / (1 + z))))


def _reference(model, p, z, filters=None):
    filters = _FILTERS if filters is None else filters
    spec, t_eff = _spectrum(model, p, z)
    return np.array([_band(spec, fw, ft, z, t_eff) for fw, ft in filters])


def _pah_reference(z, filters=None):
    filters = _FILTERS if filters is None else filters
    slope = lambda lam_aa: np.full_like(lam_aa, 2.0 / 0.012 + 10.0)  # noqa: E731
    out = []
    for fw, ft in filters:
        x, w = _band_nodes(fw, ft, z, slope)
        out.append(float(np.sum(w * _pah_spectrum(x / (1 + z)))))
    return np.array(out)


def _declared_range(name):
    """Declared free-prior range of a parameter, read from the dust declarations."""
    return tuple(float(v) for v in next(p for p in PARAMS if p.name == name).free_prior.bounds)


# ── Tables, samples ───────────────────────────────────────────────────────────────────────────


@functools.cache
def _table(model, z):
    result = adapter.precompute(_FILTER_WAVES, _FILTER_TRANS, z, None, model=model)
    return result, adapter.build_lookup(result, model=model)


def _sample(model, n=40, seed=2676):
    """Random off-node points of the declared ranges plus points 2 % inside each bound."""
    names = adapter.AXIS_PARAMS[model]
    rng = np.random.default_rng(seed)
    ranges = [_declared_range(nm) for nm in names]
    pts = [[rng.uniform(lo, hi) for lo, hi in ranges] for _ in range(n)]
    for corner in range(2 ** len(names)):
        pts.append(
            [
                lo + (0.02 if (corner >> i) & 1 == 0 else 0.98) * (hi - lo)
                for i, (lo, hi) in enumerate(ranges)
            ]
        )
    return [dict(zip(names, row, strict=True)) for row in pts]


def _call(lookup, p, names):
    return np.asarray(lookup(1.0, *[p[n] for n in names]), dtype=float).ravel()


# ── (a) off-node accuracy over the declared ranges, redshifts and filters ─────────────────────


@pytest.mark.parametrize("z", REDSHIFTS)
@pytest.mark.parametrize("model", _CONTINUUM)
def test_off_node_lookup_matches_converged_quadrature(model, z):
    """Lookup / quadrature within TOL on random off-node points of the declared prior ranges."""
    _, lookup = _table(model, z)
    names = adapter.AXIS_PARAMS[model]
    worst, checked = 0.0, 0
    for p in _sample(model):
        ref = _reference(model, p, z)
        got = _call(lookup, p, names)
        ok = ref > FLOOR
        checked += int(ok.sum())
        worst = max(worst, float(np.max(np.abs(got[ok] / ref[ok] - 1.0), initial=0.0)))
    assert checked > 0.5 * len(_FILTERS) * len(_sample(model)), "most of the sample must be live"
    assert worst <= TOL, f"{model} z={z}: max |lookup/quadrature - 1| = {worst:.2e} > {TOL}"


@pytest.mark.parametrize("z", (0.0, 2.0))
@pytest.mark.parametrize("model", _CONTINUUM)
def test_reference_agrees_with_scipy_quad_and_package_exact_path(model, z):
    """The converged reference equals ``scipy.quad`` and the package's own band integral.

    The package path is the closure on a 60000-point rest grid through ``preintegrate_grid``; it
    is compared where the band's Wien exponent at the blue edge is below 25, where that grid
    resolves the spectrum.
    """
    names = adapter.AXIS_PARAMS[model]
    p = _sample(model, n=3, seed=7)[1]
    ref = _reference(model, p, z)
    spec, t_eff = _spectrum(model, p, z)
    for i in (1, 3, 5):
        fw, ft = _FILTERS[i]
        total = 0.0
        denom = 0.0
        for a, b, ta, tb in zip(fw[:-1], fw[1:], ft[:-1], ft[1:], strict=True):
            if ta == 0.0 and tb == 0.0:
                continue
            kw = dict(epsabs=0.0, epsrel=1e-11, limit=400)
            tr = functools.partial(np.interp, xp=fw, fp=ft)
            total += quad(lambda x, tr=tr: spec(x / (1 + z)) * tr(x) / x, a, b, **kw)[0]
            denom += quad(lambda x, tr=tr: tr(x) / x, a, b, **kw)[0]
        assert total / denom == pytest.approx(ref[i], rel=1e-7)

    wave = np.geomspace(1e2, 1e8, 60000)
    sed = np.asarray(M[model](jnp.asarray(wave), 1.0, **p, redshift=z))
    pkg = np.asarray(
        preintegrate_grid(
            templates=sed[None, :],
            wave_rest=wave,
            filter_waves=_FILTER_WAVES,
            filter_trans=_FILTER_TRANS,
            redshift=z,
            dl_cm=1.0,
        ).phot
    ).ravel()
    blue_x = np.array([_HC_K_CM / (fw[1] / (1 + z) * 1e-8 * t_eff) for fw, _ in _FILTERS])
    live = blue_x < 25.0
    assert live.sum() >= 4
    np.testing.assert_allclose(pkg[live], ref[live], rtol=2e-3)
    _, lookup = _table(model, z)
    np.testing.assert_allclose(_call(lookup, p, names)[live], pkg[live], rtol=3e-3)


@pytest.mark.parametrize("z", REDSHIFTS)
def test_pah_drude_matches_converged_quadrature(z):
    _, lookup = _table("pah_drude", z)
    got = np.asarray(lookup(1.0)).ravel()
    ref = _pah_reference(z)
    np.testing.assert_allclose(got, ref, rtol=TOL)


def test_every_registered_builder_is_covered():
    """The sweeps above parametrize over ``AXIS_PARAMS``; the four builders are all in it."""
    assert set(adapter.AXIS_PARAMS) == {*_CONTINUUM, "pah_drude"}
    for names in adapter.AXIS_PARAMS.values():
        for name in names:
            assert name in adapter.AXIS_TRANSFORM_KEYS
            assert name in adapter.DEFAULT_AXIS_NODES
            lo, hi = adapter.declared_range(name)
            axis = adapter.default_axis(name)
            assert axis.size == adapter.DEFAULT_AXIS_NODES[name]
            assert axis[0] == pytest.approx(lo) and axis[-1] == pytest.approx(hi)


# ── (b) the Wien tail ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ("modified_blackbody", "graybody"))
def test_wien_tail_band_at_cold_temperature(model):
    """8-24 um at 15 K (nodes 12-20 K) and at the coldest default node region, z = 0."""
    names = adapter.AXIS_PARAMS[model]
    fw, ft = _tophat(8.0, 24.0, n=400)
    p = {"dust_T": 15.3, "dust_beta_ir": 1.5, "dust_lambda_0_um": 200.0}
    grids = {
        "T_grid": np.linspace(12.0, 20.0, 9),
        "beta_grid": np.array([1.0, 1.5, 2.0]),
    }
    if model == "graybody":
        grids["lambda_0_um_grid"] = np.array([100.0, 200.0, 400.0])
    result = adapter.precompute([fw], [ft], 0.0, None, model=model, **grids)
    got = float(_call(adapter.build_lookup(result, model=model), p, names)[0])
    ref = _reference(model, p, 0.0, [(fw, ft)])[0]
    assert got / ref == pytest.approx(1.0, abs=TOL)
    _, default_lookup = _table(model, 0.0)
    p_cold = {**p, "dust_T": 20.7}
    ref_cold = _reference(model, p_cold, 0.0)
    got_cold = _call(default_lookup, p_cold, names)
    # the 8-24 um band is index 1
    assert got_cold[1] / ref_cold[1] == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("model", ("modified_blackbody", "graybody", "casey2012"))
def test_wien_tail_band_on_a_coarse_filter_table(model):
    """A filter tabulated at its two edges only spans a factor 3 in wavelength per segment;
    at the coldest declared temperature the Wien slope across it is resolved by the
    sub-division, not by the table."""
    names = adapter.AXIS_PARAMS[model]
    fw, ft = _tophat(8.0, 24.0, n=2)
    p = {"dust_T": 20.5, "dust_beta_ir": 1.5, "dust_alpha_mir": 2.0, "dust_lambda_0_um": 200.0}
    result = adapter.precompute([fw], [ft], 0.0, None, model=model)
    got = float(_call(adapter.build_lookup(result, model=model), p, names)[0])
    ref = _reference(model, p, 0.0, [(fw, ft)])[0]
    assert got / ref == pytest.approx(1.0, abs=TOL)


def test_filter_tabulated_from_zero_wavelength_is_finite():
    """A filter whose first node is at 0 A is cut at 100 A rest frame, not divided by zero."""
    z, model = 0.5, "modified_blackbody"
    wave = np.linspace(0.0, 6.0e5, 64)
    trans = np.exp(-0.5 * ((wave - 3.0e5) / 1.0e5) ** 2)
    p = {"dust_T": 40.0, "dust_beta_ir": 1.8}
    result = adapter.precompute([wave], [trans], z, None, model=model)
    got = float(_call(adapter.build_lookup(result, model=model), p, adapter.AXIS_PARAMS[model])[0])
    cut = np.concatenate([[100.0 * (1.0 + z)], wave[1:]])
    ref = _reference(model, p, z, [(cut, trans)])[0]
    assert np.isfinite(got) and got > 0.0
    assert got / ref == pytest.approx(1.0, abs=TOL)


# ── (c) outside the normalization grid ────────────────────────────────────────────────────────


_OUTSIDE = (
    ("15-30 mm", _tophat(15e3, 30e3)),
    ("12-20 mm", _tophat(12e3, 20e3)),
    ("30-90 A", _tophat(0.003, 0.009)),
)


@pytest.mark.parametrize("z", (0.0, 6.0))
@pytest.mark.parametrize("model", _CONTINUUM)
def test_filters_outside_the_rest_grid_read_the_exact_integral(model, z):
    """A band beyond either end of 0.01 um - 10 mm equals the quadrature, never a silent zero."""
    names = adapter.AXIS_PARAMS[model]
    filters = [f for _, f in _OUTSIDE]
    result = adapter.precompute(
        [f[0] for f in filters], [f[1] for f in filters], z, None, model=model
    )
    lookup = adapter.build_lookup(result, model=model)
    for p in _sample(model, n=3, seed=11)[:3]:
        ref = _reference(model, p, z, filters)
        got = _call(lookup, p, names)
        assert np.all(np.isfinite(got)) and np.all(got > 0.0)
        live = ref > FLOOR
        assert live[:2].all(), "the millimeter bands carry flux"
        np.testing.assert_allclose(got[live], ref[live], rtol=TOL)
        assert np.all(got[~live] <= FLOOR), "a negligible band is tiny, not O(1)"


@pytest.mark.parametrize("z", (0.0, 2.0))
def test_pah_drude_beyond_31_6_um(z):
    """40-70 um and 3-10 mm bands, where the Drude wings are nonzero, equal the quadrature."""
    filters = [_tophat(40.0, 70.0), _tophat(3000.0, 10000.0), _tophat(15e3, 30e3)]
    result = adapter.precompute(
        [f[0] for f in filters], [f[1] for f in filters], z, None, model="pah_drude"
    )
    got = np.asarray(adapter.build_lookup(result, model="pah_drude")(1.0)).ravel()
    ref = _pah_reference(z, filters)
    assert np.all(ref > 0.0) and np.all(got > 0.0)
    np.testing.assert_allclose(got, ref, rtol=TOL)


# ── (d) gradients ─────────────────────────────────────────────────────────────────────────────

_GRAD_TOL = 2e-2  # on d ln(flux) / d ln(parameter)


def _dlnphi_reference(model, p, names, z, i_filter, rel_step=1e-4):
    out = []
    for n in names:
        up, dn = dict(p), dict(p)
        h = rel_step * p[n]
        up[n], dn[n] = p[n] + h, p[n] - h
        fu = _reference(model, up, z, [_FILTERS[i_filter]])[0]
        fd = _reference(model, dn, z, [_FILTERS[i_filter]])[0]
        out.append(p[n] * (np.log(fu) - np.log(fd)) / (2 * h))
    return np.array(out)


@pytest.mark.parametrize("z", (0.0, 2.0))
@pytest.mark.parametrize("model", _CONTINUUM)
def test_gradient_is_finite_and_matches_the_exact_flux(model, z):
    """``jax.grad`` of a band flux is finite on the whole sample and equals the reference slope."""
    _, lookup = _table(model, z)
    names = adapter.AXIS_PARAMS[model]
    jac = jax.jit(jax.jacrev(lambda v: lookup(1.0, *v)))
    worst = 0.0
    for p in _sample(model, n=8):
        v = jnp.asarray([p[n] for n in names])
        j = np.asarray(jac(v))
        assert np.all(np.isfinite(j)), f"non-finite gradient at {p}"
        assert np.any(j != 0.0), f"identically zero gradient at {p}"
        phi = _call(lookup, p, names)
        ref_phi = _reference(model, p, z)
        for i in (1, 3, 5, 7):
            if ref_phi[i] < 1e-100:
                continue
            fd = _dlnphi_reference(model, p, names, z, i)
            mine = j[i] / phi[i] * np.array([p[n] for n in names])
            worst = max(worst, float(np.max(np.abs(mine - fd))))
    assert worst <= _GRAD_TOL, f"max |dln(flux)/dln(p) - reference| = {worst:.2e}"


@pytest.mark.parametrize("model", _CONTINUUM)
def test_gradient_is_continuous_across_nodes(model):
    """First and second derivatives of ``ln(flux)`` agree on both sides of an interior node."""
    result, lookup = _table(model, 0.5)
    names = adapter.AXIS_PARAMS[model]
    mid = {n: 0.5 * sum(_declared_range(n)) + 1.234e-3 for n in names}
    for k, name in enumerate(names):
        axis = np.asarray(result["axes"][k])
        node = float(axis[axis.size // 2])

        def lnphi(x, k=k):
            args = [jnp.asarray(mid[n]) for n in names]
            args[k] = x
            return jnp.log(lookup(1.0, *args)[5])

        eps = 1e-7 * node
        g1, g2 = jax.grad(lnphi), jax.grad(jax.grad(lnphi))
        left, right = g1(node - eps), g1(node + eps)
        assert abs(left - right) <= 1e-5 * max(abs(left), 1e-3), f"{name}: C1 jump"
        h_left, h_right = g2(node - eps), g2(node + eps)
        assert abs(h_left - h_right) <= 1e-4 * max(abs(h_left), 1e-2), f"{name}: C2 jump"


# ── (e) float32 ───────────────────────────────────────────────────────────────────────────────

_F32_RTOL = 2e-3
_F32_MIN_NORMAL = 1.2e-38


@pytest.mark.parametrize("z", (0.0, 6.0))
@pytest.mark.parametrize("model", _CONTINUUM)
def test_float32_lookup_agrees_with_float64(model, z):
    """A float32 query reads the log table in float32: within 2e-3 where float32 can hold it."""
    _, lookup = _table(model, z)
    names = adapter.AXIS_PARAMS[model]
    corners = [{n: _declared_range(n)[j] for n in names} for j in (0, 1)]
    jac32 = jax.jit(jax.jacrev(lambda v: lookup(jnp.float32(1.0), *v)))
    for p in [*_sample(model, n=8), *corners]:
        ref = _call(lookup, p, names)
        out32 = lookup(jnp.float32(1.0), *[jnp.float32(p[n]) for n in names])
        assert out32.dtype == jnp.float32
        got = np.asarray(out32, dtype=float)
        assert np.all(np.isfinite(got)) and np.all(got >= 0.0)
        live = ref > 1e3 * _F32_MIN_NORMAL
        np.testing.assert_allclose(got[live], ref[live], rtol=_F32_RTOL)
        assert np.all(got[ref < 1e-45] < 1e-37)
        grad32 = jac32(jnp.asarray([p[n] for n in names], dtype=jnp.float32))
        assert np.all(np.isfinite(np.asarray(grad32))), f"non-finite float32 gradient at {p}"
        assert np.any(np.asarray(grad32) != 0.0), f"identically zero float32 gradient at {p}"


# ── (f) node values ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", _CONTINUUM)
def test_node_values_are_exact_and_match_the_quadrature(model):
    """The lookup returns the table value at a node and the table value is the band integral."""
    result, lookup = _table(model, 2.0)
    names = adapter.AXIS_PARAMS[model]
    axes = [np.asarray(a) for a in result["axes"]]
    grid = np.asarray(result["grid_phot"])
    for idx in [
        tuple(a.size // 2 for a in axes),
        tuple(0 for _ in axes),
        tuple(a.size - 1 for a in axes),
    ]:
        p = {n: float(ax[i]) for n, ax, i in zip(names, axes, idx, strict=True)}
        got = _call(lookup, p, names)
        np.testing.assert_allclose(got, grid[idx], rtol=1e-12)
        ref = _reference(model, p, 2.0)
        live = ref > FLOOR
        np.testing.assert_allclose(got[live], ref[live], rtol=TOL)


@pytest.mark.parametrize("model", _CONTINUUM)
def test_collapsed_lookup_equals_the_full_lookup(model):
    """Pinning axes evaluates the same spline: collapsed and full lookups agree at the pin."""
    from unittest.mock import MagicMock

    names = adapter.AXIS_PARAMS[model]
    _, full_lookup = _table(model, 0.5)
    p = _sample(model, n=1, seed=3)[0]
    fixed = {names[0]: p[names[0]], names[-1]: p[names[-1]]}
    params = MagicMock()
    params.get_fixed_values.return_value = fixed
    params.free_params = []
    result = adapter.precompute(_FILTER_WAVES, _FILTER_TRANS, 0.5, params, model=model)
    lookup = adapter.build_lookup(result, model=model)
    free = [n for n in names if n not in fixed]
    got = np.asarray(lookup(1.0, *[p[n] for n in free]))
    np.testing.assert_allclose(got, _call(full_lookup, p, names), rtol=1e-12)
    assert len(result["axes"]) == len(free)


def test_unrepresentable_table_is_refused():
    """A filter with no transmission has no band flux: refuse instead of tabulating zeros."""
    wave = np.array([1e5, 2e5, 3e5])
    with pytest.raises(ValueError, match="transmission"):
        adapter.precompute([wave], [np.zeros(3)], 0.0, None, model="modified_blackbody")
