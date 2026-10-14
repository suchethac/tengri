# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR's energy budget counts the disc over its native template range (#2851).

``agn_log_lbol`` is a physical accretion power (#2678), so the one scale that fixes the
templates must make the sphere mean of EVERYTHING the radiative transfer carries equal to
``torus_frac 10**agn_log_lbol L_sun``. The library's disc runs from 10 A to 1e8 A, but the
analytic disc the component emits (the CIGALE ``skirtor2016`` piecewise power law, 8 nm to
1e6 nm) is zero below 80 A. The template disc carries 9 to 58 % (sphere mean, per cell) more
power below 80 A than above it: obscured, the torus absorbs it and re-emits it as dust;
escaping, it leaves. Counting the disc only above 80 A drops that power from the budget of the
escaping rays while the dust that it heated stays in, and shifts the absolute disc and dust
scale by -4 to +9 % with the opening angle.

The budget therefore counts the disc over its native range, and the escaping disc outside the
emitted support is published as ``log_L_agn_disc_unemitted`` (not in the SED):

.. math:: \\bar P_{\\rm disc} + \\bar P_{\\rm torus} + P_{\\rm polar} + \\bar P_{\\rm unemitted}
          = f\\,10^{\\log L_{\\rm bol}} L_\\odot .

The references are independent of the component: ``RawSkirtor`` reads
``skirtor_templates_v3.h5`` with ``h5py``, interpolates with SciPy's PCHIP, integrates the power
laws in closed form, and sets the scale from the raw sphere mean; the emitted pieces are
rebuilt from those records on a dense grid.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from tengri._data_setup import find_data
from tengri.utils.physics_constants import L_SUN
from tests.physics._agn_publication import PUB_COARSE, ln_nu_integral
from tests.physics._agn_published import build_component, declared_box
from tests.physics._skirtor_reference import (
    DISC_SUPPORT_AA,
    RawSkirtor,
    cone_absorbed,
    emitted_pieces,
    loglog_power,
)

pytestmark = pytest.mark.regression_bug

_POINT_KEYS = ("tau_skirtor", "p_skirtor", "q_skirtor", "oa_skirtor", "radius_ratio")
_WIDE = np.unique(
    np.concatenate(
        [
            np.geomspace(1.0, 1.0e10, 40_000),
            [x for e in DISC_SUPPORT_AA for x in (e * (1 - 1e-9), e, e * (1 + 1e-9))],
        ]
    )
)


@pytest.fixture(scope="module")
def skirtor():
    comp = build_component("skirtor")
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in declared_box(comp).items()}
    return comp, mid


@pytest.fixture(scope="module")
def raw_library():
    path = find_data("skirtor_templates_v3.h5")
    if path is None:
        pytest.skip("SKIRTOR v3 templates unavailable")
    return RawSkirtor(str(path))


def _published(comp, params, wave=PUB_COARSE):
    w = jnp.asarray(wave)
    _, pub = comp.predict(
        {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}, jnp.zeros_like(w), w
    )
    return {k: float(v) for k, v in pub.items()}


def _ledger(pub):
    """Escaping disc, torus, polar re-emission and unemitted disc [erg/s] from the keys."""
    keys = ("disc", "torus", "polar_dust", "disc_unemitted")
    return np.array([10.0 ** pub[f"log_L_agn_{k}"] for k in keys])


@pytest.mark.parametrize("polar_ebv", [0.0, 0.2])
@pytest.mark.parametrize("delta", [0.0, 0.6])
@pytest.mark.parametrize("cos_inc", [0.05, 0.9])
@pytest.mark.parametrize("frac", [1.0, 0.4])
def test_published_ledger_with_the_unemitted_disc_closes(skirtor, frac, cos_inc, delta, polar_ebv):
    """Sphere-mean disc + torus + polar + unemitted disc = ``torus_frac 10**log_lbol L_sun``."""
    comp, mid = skirtor
    pub = _published(
        comp,
        {**mid, "torus_frac": frac, "cos_inc": cos_inc, "delta": delta, "polar_ebv": polar_ebv},
    )
    expected = frac * 10.0 ** mid["log_lbol"] * L_SUN
    err = abs(_ledger(pub).sum() / expected - 1.0)
    assert err < 1.0e-12, f"ledger / (f L) - 1 = {err:.2e}"
    lsun = pub["L_agn_disc_unemitted_lsun"] * L_SUN
    assert lsun == pytest.approx(10.0 ** pub["log_L_agn_disc_unemitted"], rel=1e-12)


@pytest.mark.parametrize("disk_type", [1, 2])
@pytest.mark.parametrize("polar_ebv", [0.0, 0.2])
def test_other_disc_types_close_the_ledger_with_the_unemitted_disc(disk_type, polar_ebv):
    """Schartmann (1) and ADAF (2) discs: the four-term ledger closes to 1e-12."""
    from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig

    path = find_data("skirtor_templates_v3.h5")
    if path is None:
        pytest.skip("SKIRTOR v3 templates unavailable")
    comp = SKIRTORTorus(config=SKIRTORTorusConfig(grid_path=str(path), disk_type=disk_type))
    object.__setattr__(comp, "data", comp.load(None))
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in declared_box(comp).items()}
    wave = jnp.asarray(PUB_COARSE)
    publish = jax.jit(lambda p: comp.predict(p, jnp.zeros_like(wave), wave)[1])
    expected = 10.0 ** mid["log_lbol"] * L_SUN
    for delta in (-1.0, 0.7):
        for oa, tau, cos_inc in ((15.0, 3.0, 0.9), (75.0, 10.0, 0.2)):
            params = {
                **mid,
                "delta": delta,
                "oa_skirtor": oa,
                "tau_skirtor": tau,
                "cos_inc": cos_inc,
                "polar_ebv": polar_ebv,
                "torus_frac": 1.0,
            }
            pub = {
                k: float(v)
                for k, v in publish(
                    {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
                ).items()
            }
            err = abs(_ledger(pub).sum() / expected - 1.0)
            assert err < 1.0e-12, f"disk_type={disk_type} delta={delta} oa={oa}: {err:.2e}"


@pytest.mark.parametrize("cos_inc", [0.05, 0.4, 0.95])
@pytest.mark.parametrize(
    "extra",
    [{}, {"oa_skirtor": 23.0, "tau_skirtor": 9.5, "radius_ratio": 14.0, "delta": 0.4}],
    ids=["center", "closed_torus"],
)
def test_unemitted_disc_equals_the_raw_records_outside_the_support(
    skirtor, raw_library, extra, cos_inc
):
    """``log_L_agn_disc_unemitted`` (sphere mean and line of sight) from the raw records (1e-6).

    The reference reads the file with ``h5py``, interpolates with SciPy's PCHIP, integrates
    the power laws below 80 A and above 1e7 A in closed form, and scales by the one factor
    that makes the raw native sphere mean (order 48) equal ``f L``. The key must not depend on
    ``delta`` or on the polar dust: the emitted disc does not enter it.
    """
    comp, mid = skirtor
    params = {**mid, **extra, "polar_ebv": 0.2, "cos_inc": cos_inc}
    point = tuple(params[k] for k in _POINT_KEYS)
    scale = raw_library.scale(params)
    mean = raw_library.sphere_mean(lambda m: raw_library.unemitted_power(point, m))
    pub = _published(comp, params)
    got = 10.0 ** pub["log_L_agn_disc_unemitted"]
    assert abs(got / (scale * mean) - 1.0) < 1.0e-6
    got_los = 10.0 ** pub["log_L_agn_disc_unemitted_los"]
    ref_los = scale * raw_library.unemitted_power(point, cos_inc)
    assert abs(got_los / ref_los - 1.0) < 1.0e-8
    other = _published(comp, {**params, "delta": -0.7, "polar_ebv": 0.0})
    assert other["log_L_agn_disc_unemitted"] == pytest.approx(
        pub["log_L_agn_disc_unemitted"], abs=1e-12
    )
    assert 0.0 < got < 10.0 ** mid["log_lbol"] * L_SUN * float(params["torus_frac"])


@pytest.mark.parametrize("polar_ebv", [0.0, 0.2])
@pytest.mark.parametrize("delta", [0.0, 0.6])
@pytest.mark.parametrize("cos_inc", [0.1, 0.5, 0.95])
def test_every_emitted_piece_equals_the_integral_of_the_sed(
    skirtor, raw_library, cos_inc, delta, polar_ebv
):
    """Each emitted piece's key equals the integral of its piece of the SED (1e-6); the sum too.

    The pieces are rebuilt from the raw records on a wide grid (``emitted_pieces``): the disc
    after the polar dust's Type-1 attenuation, the dust, and the polar re-emission, whose power
    is the cone integral of the absorbed disc. The unemitted disc is not in the SED, so the
    whole SED integrates to the three emitted keys without it.
    """
    comp, mid = skirtor
    params = {**mid, "cos_inc": cos_inc, "delta": delta, "polar_ebv": polar_ebv}
    pieces = emitted_pieces(raw_library, params, _WIDE)
    pub = _published(comp, params)
    disc = ln_nu_integral(pieces["disc_attenuated"], _WIDE)
    dust = ln_nu_integral(pieces["dust"], _WIDE)
    assert abs(10.0 ** pub["log_L_agn_disc_los"] / disc - 1.0) < 1.0e-6
    assert abs(10.0 ** pub["log_L_agn_torus_los"] / dust - 1.0) < 1.0e-6
    polar = cone_absorbed(raw_library, params, _WIDE, order=16) if polar_ebv > 0.0 else 0.0
    if polar_ebv > 0.0:
        assert abs(10.0 ** pub["log_L_agn_polar_dust"] / polar - 1.0) < 1.0e-6
    else:
        assert pub["log_L_agn_polar_dust"] == -np.inf
    wide = jnp.asarray(_WIDE)
    sed, _ = comp.predict(
        {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()},
        jnp.zeros_like(wide),
        wide,
    )
    emitted = ln_nu_integral(np.asarray(sed), _WIDE)
    published = disc + dust + polar
    assert abs(emitted / published - 1.0) < 1.0e-6


def test_the_unemitted_disc_is_not_in_the_emitted_sed(skirtor):
    """Below 80 A the emitted SED is the dust alone, while the template disc there is not zero."""
    comp, mid = skirtor
    lam = jnp.asarray(np.geomspace(10.0, 79.0, 60))
    p = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in {**mid, "polar_ebv": 0.0}.items()}
    sed_full, _ = comp.predict(p, jnp.zeros_like(lam), lam)
    from tengri.components.agn.skirtor import create_skirtor_components_from_grid

    template = create_skirtor_components_from_grid(comp.config.grid_path)(
        wavelength=lam,
        agn_log_lbol=mid["log_lbol"],
        agn_tau_skirtor=mid["tau_skirtor"],
        agn_p_skirtor=mid["p_skirtor"],
        agn_q_skirtor=mid["q_skirtor"],
        agn_oa_skirtor=mid["oa_skirtor"],
        agn_radius_ratio=mid["radius_ratio"],
        agn_cos_inc=mid["cos_inc"],
        frac_agn=mid["torus_frac"],
    )
    # The SED below 80 A is the dust alone: the template disc there is what the key reports.
    np.testing.assert_allclose(np.asarray(sed_full), np.asarray(template.dust), rtol=1e-12)
    assert float(np.max(np.asarray(template.disk))) > 0.0


@pytest.mark.parametrize("polar_ebv", [0.0, 0.2])
def test_anisotropy_has_the_sign_of_the_radiative_transfer(skirtor, polar_ebv):
    """Face-on disc LOS > sphere mean > edge-on disc LOS; the torus is brighter face-on.

    The escaping disc peaks along the polar axis and is hidden edge-on; the dust the torus
    re-emits is anisotropic the other way round from the disc's, but still brighter face-on in
    the library's records.
    """
    comp, mid = skirtor
    base = {**mid, "polar_ebv": polar_ebv}
    face = _published(comp, {**base, "cos_inc": 0.99})
    edge = _published(comp, {**base, "cos_inc": 0.01})
    assert face["log_L_agn_disc_los"] > face["log_L_agn_disc"] > edge["log_L_agn_disc_los"]
    assert face["log_L_agn_torus_los"] > edge["log_L_agn_torus_los"]


def _native_cell_budget(raw, cos_edges_order=24, resolve_cone=False):
    """Sphere mean of the native escaping disc + dust of every library cell [library units].

    PCHIP of the stored (norm-scaled) spectra in cos i, Gauss-Legendre of order 24 on each
    interval between the inclination nodes, log-log power-law integral in wavelength. With
    ``resolve_cone`` the disc inside the polar cone (``cos i > sin oa``) is the SKIRTOR
    direct-light law ``mu (1 + 2 mu)`` (Stalevski et al. 2012, Eq. 1) fitted to the nodes inside
    the cone, integrated analytically, and the obscured part is PCHIP over the nodes at or
    below the cone edge.
    """
    mu = raw.axes[-1]
    shape = raw.cubes["disk"].shape[:5]
    n_cell = int(np.prod(shape))
    disc = raw.cubes["disk"].reshape(n_cell, mu.size, -1)
    dust = raw.cubes["dust"].reshape(n_cell, mu.size, -1)
    oa = np.broadcast_to(raw.axes[3][None, None, None, :, None], shape).reshape(-1)
    edges = np.concatenate([[0.0], mu[1:]])
    x, w = np.polynomial.legendre.leggauss(cos_edges_order)
    mq = np.concatenate([0.5 * (b - a) * x + 0.5 * (a + b) for a, b in itertools.pairwise(edges)])
    wq = np.concatenate([0.5 * (b - a) * w for a, b in itertools.pairwise(edges)])
    node_disc = loglog_power(raw.wave, disc)
    total = np.zeros(n_cell)
    for c0 in range(0, n_cell, 240):
        sl = slice(c0, c0 + 240)
        if not resolve_cone:
            d = PchipInterpolator(mu, disc[sl], axis=1)(mq)
            total[sl] += loglog_power(raw.wave, d) @ wq
        u = PchipInterpolator(mu, dust[sl], axis=1)(mq)
        total[sl] += loglog_power(raw.wave, u) @ wq
    if resolve_cone:
        xg, wg = np.polynomial.legendre.leggauss(48)
        for i in range(n_cell):
            edge = np.sin(np.radians(oa[i]))
            inside = mu > edge + 1e-9
            k = np.mean(node_disc[i][inside] / (mu * (1 + 2 * mu))[inside])
            cone = k * ((1 - edge**2) / 2 + 2 * (1 - edge**3) / 3)
            below = mu <= edge + 1e-9
            obscured = 0.0
            if below.sum() >= 2:
                f = PchipInterpolator(mu[below], node_disc[i][below])
                knots = np.concatenate([[0.0], mu[below][1:]])
                for a, b in itertools.pairwise(knots):
                    obscured += f(0.5 * (b - a) * xg + 0.5 * (a + b)) @ (0.5 * (b - a) * wg)
            total[i] += cone + obscured
    return total, oa


def _supported_disc_budget(raw):
    """The same sphere mean with the disc counted only inside the emitted support (80 A to 1e7)."""
    mu = raw.axes[-1]
    n_cell = int(np.prod(raw.cubes["disk"].shape[:5]))
    disc = raw.cubes["disk"].reshape(n_cell, mu.size, -1)
    dust = raw.cubes["dust"].reshape(n_cell, mu.size, -1)
    edges = np.concatenate([[0.0], mu[1:]])
    x, w = np.polynomial.legendre.leggauss(24)
    mq = np.concatenate([0.5 * (b - a) * x + 0.5 * (a + b) for a, b in itertools.pairwise(edges)])
    wq = np.concatenate([0.5 * (b - a) * w for a, b in itertools.pairwise(edges)])
    total = np.zeros(n_cell)
    for c0 in range(0, n_cell, 240):
        sl = slice(c0, c0 + 240)
        d = PchipInterpolator(mu, disc[sl], axis=1)(mq)
        u = PchipInterpolator(mu, dust[sl], axis=1)(mq)
        lo, hi = DISC_SUPPORT_AA
        total[sl] = loglog_power(raw.wave, d, lower=lo, upper=hi) @ wq
        total[sl] += loglog_power(raw.wave, u) @ wq
    return total


def test_native_budget_scatter_is_the_librarys_own_residual(raw_library):
    """The raw-record budget scatter across the 1920 cells, and what removed its trend.

    SKIRTOR's records conserve energy only to the radiative-transfer run's own accuracy: the
    sphere-mean escaping disc plus dust differs from cell to cell. Counting the disc over its
    native range (10 A to 1e8 A) leaves 2.5 % rms with no trend in opening angle (measured
    2.48 %, correlation +0.11); counting it only inside the emitted support (above 80 A)
    left 5.1 % rms with a +0.88 correlation, because 9 to 58 % of the face-on-weighted disc power
    lies below 80 A and the fraction rises with the torus closing. Resolving the cone edge
    (10 inclination nodes cannot follow the disc's step at ``cos i = sin(oa)``) leaves 1.6 % rms,
    the library's own residual (Monte Carlo noise and the finite photon budget of the
    radiative-transfer run). Each number is a bound on a quantity measured from the file with
    h5py and SciPy alone.
    """
    native, oa = _native_cell_budget(raw_library)
    resolved, _ = _native_cell_budget(raw_library, resolve_cone=True)
    supported = _supported_disc_budget(raw_library)

    def stats(total):
        rel = total / total.mean() - 1.0
        return rel.std(), np.corrcoef(rel, oa)[0, 1]

    rms_native, corr_native = stats(native)
    rms_resolved, _ = stats(resolved)
    rms_supported, corr_supported = stats(supported)
    assert rms_native < 0.03 and abs(corr_native) < 0.2, (rms_native, corr_native)
    assert rms_resolved < 0.02, rms_resolved
    assert rms_supported > 0.045 and corr_supported > 0.8, (rms_supported, corr_supported)
