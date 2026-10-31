# SPDX-License-Identifier: BSD-3-Clause
"""AGNfitter-rX torus reductions keep their anisotropy where its provenance is clear (#2851).

The AGNfitter-rX pickles (Martinez-Ramirez et al. 2024, Sect. 3.3) average the original
libraries over all but one to three parameters. Whether a reduction keeps the original's
flux ratio between inclinations decides whether the torus is normalized once over the
sphere (restored) or once per sightline (kept):

* ``SKIRTOR_mean_1p``: restored. Its stored power ratios between inclinations match the
  SKIRTOR v3 records (``skirtor_templates_v3.h5``, multiplied back by their ``norm``)
  averaged over tau, p, q, opening angle and radius ratio to 0.4 %, the same radiative
  transfer averaged the same way.
* ``NK0_mean_1p``/``2p``/``3p``: restored. CLUMPY fluxes are normalized to the bolometric
  AGN flux ``F_AGN = L/4 pi D^2`` in every direction (Nenkova et al. 2008, ApJ 685, 160,
  Sect. 2 and 3.6), so the records share one scale; the 3p reduction averaged over its
  optical-depth axis is the 2p reduction to 5e-4, so the reductions are linear means on that
  scale, and the powers fall monotonically from face-on to edge-on, by a factor below 3 for
  1p as the paper's Fig. 11 bounds it.
* ``SKIRTOR_mean_2p``/``3p`` and ``CAT3D_mean_3p``: kept per sightline. The 2p and 3p power
  ratios do not follow the v3 records averaged the same way (3p: scattered 0.2 to 3.9 about
  1), and no CAT3D-WIND source states the scale of its records across inclination.

References are independent of the components: ``RawTorus`` and the SKIRTOR v3 average read
the files with ``h5py`` and integrate the records' power laws in closed form.
"""

from __future__ import annotations

import importlib

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data
from tengri.utils.physics_constants import L_SUN
from tests.physics._agn_publication import ln_nu_integral
from tests.physics._skirtor_reference import loglog_power
from tests.physics._template_torus_reference import LIBRARIES, RawTorus

pytestmark = pytest.mark.regression_bug

_WIDE = np.geomspace(5.0, 1.0e8, 20000)
_KW = {"agn_log_lbol": 10.0, "agn_torus_frac": 0.7}
_L_TORUS = 10.0 ** _KW["agn_log_lbol"] * L_SUN * _KW["agn_torus_frac"]

#: Restored library -> (point on the other axes, kwargs of the SED kernel at that point).
_RESTORED = {
    "nenkova_agnfitter": ((), {}),
    "nenkova_agnfitter_2p": ((42.5,), {"agn_oa_nenkova": 42.5}),
    "nenkova_agnfitter_3p": ((42.5, 70.0), {"agn_oa_nenkova": 42.5, "agn_tv_nenkova": 70.0}),
    "skirtor_agnfitter_1p": ((), {}),
}


def _path(filename):
    found = find_data(filename)
    if found is None:
        pytest.skip(f"template data unavailable ({filename})")
    return str(found)


@pytest.fixture(scope="module", params=sorted(_RESTORED))
def library(request):
    name = request.param
    mod = importlib.import_module(f"tengri.components.agn.{name}")
    path = _path(LIBRARIES[name][0])
    return name, mod, getattr(mod, f"load_{name}_grid")(path), RawTorus(name, path)


def _emitted_power(name, mod, grid, mu):
    kwargs = dict(_RESTORED[name][1], **_KW, agn_cos_inc=mu)
    sed = getattr(mod, f"{name}_sed_from_grid")(grid, jnp.asarray(_WIDE), **kwargs)
    return ln_nu_integral(np.asarray(sed), _WIDE)


@pytest.mark.parametrize("mu", [0.0, 0.17, 0.5, 0.77, 0.94, 1.0])
def test_restored_sightline_is_the_raw_record_over_its_sphere_mean(library, mu):
    """Emitted power along ``mu`` is ``f L P(mu) / mean P`` of the raw records (2e-5)."""
    name, mod, grid, raw = library
    point = _RESTORED[name][0]
    expected = _L_TORUS * raw.power(point, mu) / raw.sphere_mean(point)
    assert _emitted_power(name, mod, grid, mu) == pytest.approx(float(expected), rel=2e-5)


@pytest.mark.parametrize("mu", [0.13, 0.48, 0.91])
def test_restored_shape_is_the_per_sightline_shape(library, mu):
    """The emitted torus is the per-sightline library times one scalar (1e-12).

    The anisotropy is a scalar on the library's axes, so the spectral shape off-node, between
    inclination and other-axis nodes alike, is that of the records normalized per sightline.
    """
    from tengri.components.agn._template_grid import torus_lnu_from_grid

    name, mod, grid, _ = library
    kwargs = dict(_RESTORED[name][1], **_KW, agn_cos_inc=mu)
    if name == "skirtor_agnfitter_1p":
        # The library's axis is the inclination from the polar axis [deg] (#3085).
        coords = (float(np.degrees(np.arccos(mu))),)
    else:
        coords = (mu, *_RESTORED[name][1].values())
    wave = jnp.asarray(_WIDE)
    sed = np.asarray(getattr(mod, f"{name}_sed_from_grid")(grid, wave, **kwargs))
    shape = np.asarray(torus_lnu_from_grid(grid, wave, coords, **_KW))
    sig = shape > 0.0
    ratio = sed[sig] / shape[sig]
    assert np.max(np.abs(ratio / np.median(ratio) - 1.0)) < 1e-12


def test_restored_face_on_is_brighter_than_edge_on(library):
    """The face-on torus carries more power than the edge-on one (it was equal before)."""
    name, mod, grid, _ = library
    ratio = _emitted_power(name, mod, grid, 1.0) / _emitted_power(name, mod, grid, 0.0)
    assert 1.2 < ratio < 5.0, f"{name}: face-on/edge-on power {ratio:.3f}"


def _fnu_power(path, group):
    with h5py.File(path, "r") as f:
        g = f[group]
        wave = np.asarray(g["wavelength"][()], dtype=float)
        table = np.asarray(g["template"][()], dtype=float)
    return loglog_power(wave, table * 2.99792458e18 / wave**2)


def test_skirtor_mean1p_scale_is_the_v3_radiative_transfer_averaged():
    """``SKIRTOR_mean_1p`` P(i)/P(90) equals the norm-restored v3 dust mean (0.4 %)."""
    mean1p = _fnu_power(_path("skirtor_mean1p_torus_grid.h5"), "skirtor_mean1p")
    with h5py.File(_path("skirtor_templates_v3.h5"), "r") as f:
        wave = np.asarray(f["wavelength"][()], dtype=float)
        dust = np.asarray(f["spectra/dust_emission"][()], dtype=float)
        norm = np.asarray(f["spectra/norm"][()], dtype=float)
        cos_i = np.asarray(f["grid/cos_inclination"][()], dtype=float)
    v3 = loglog_power(wave, (dust * norm[..., None]).mean(axis=(0, 1, 2, 3, 4)))
    v3 = v3[np.argsort(-cos_i)]  # 0 deg first, as mean1p
    np.testing.assert_allclose(mean1p / mean1p[-1], v3 / v3[-1], rtol=5e-3)
    assert mean1p[0] / mean1p[-1] > 2.0


def test_skirtor_mean3p_scale_is_not_the_v3_radiative_transfer():
    """``SKIRTOR_mean_3p`` does not carry the v3 inclination ratio, so it stays per sightline."""
    p3 = _fnu_power(_path("skirtor_mean3p_torus_grid.h5"), "skirtor_mean3p")  # (oa, inc, tv)
    ratio = p3[:, 0, :] / p3[:, -1, :]
    assert ratio.min() < 0.5 and ratio.max() > 3.0


def test_nenkova_reductions_are_means_on_one_scale():
    """NK0 3p averaged over its tau axis is the 2p reduction (5e-4): one common scale."""
    p2 = _fnu_power(_path("nenkova_agnfitter_2p_torus_grid.h5"), "nenkova_agnfitter_2p")
    p3 = _fnu_power(_path("nenkova_agnfitter_3p_torus_grid.h5"), "nenkova_agnfitter_3p")
    np.testing.assert_allclose(p3.mean(axis=2), p2, rtol=5e-4)
    assert np.all(np.diff(p2[2:], axis=0) < 0.0), "power must fall from 20 deg to edge-on"


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("skirtor_agnfitter", {"agn_oa_skirtor": 40.0, "agn_tv_skirtor": 7.0}),
        ("skirtor_agnfitter_2p", {"agn_oa_skirtor": 40.0}),
    ],
)
@pytest.mark.parametrize("incl", [0.0, 35.0, 90.0])
def test_kept_skirtor_reductions_carry_f_lbol_along_every_sightline(name, kwargs, incl):
    """The kept reductions are normalized per sightline: every sightline carries f L (1e-6)."""
    mod = importlib.import_module(f"tengri.components.agn.{name}")
    files = {
        "skirtor_agnfitter": "skirtor_mean3p_torus_grid.h5",
        "skirtor_agnfitter_2p": "skirtor_mean2p_torus_grid.h5",
    }
    grid = getattr(mod, f"load_{name}_grid")(_path(files[name]))
    sed = getattr(mod, f"{name}_sed_from_grid")(
        grid, jnp.asarray(_WIDE), agn_cos_inc=float(np.cos(np.radians(incl))), **kwargs, **_KW
    )
    assert ln_nu_integral(np.asarray(sed), _WIDE) == pytest.approx(_L_TORUS, rel=1e-6)
