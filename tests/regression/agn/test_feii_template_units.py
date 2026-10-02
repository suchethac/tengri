# SPDX-License-Identifier: BSD-3-Clause
"""FeII pseudo-continuum: unit convention, template handling, normalisation.

The PyQSOFit templates tabulate ``log10(lambda)`` and **F_lambda**
[erg/s/cm^2/A].  The pseudo-continuum must carry the template's F_lambda
*shape* into L_lambda, scaled so that

    integral_{4434}^{4684 A} L_lambda,FeII d(lambda) = R_Fe * L(H-beta)

(R_Fe = ``agn_fe2_strength``, the Boroson & Green 1992 window).  Earlier
code treated the F_lambda column as L_nu and then multiplied by c/lambda^2,
imprinting a spurious lambda^-2 tilt; it also kept negative template nodes
through the broadening and resampled log-log.  These tests are independent
of the implementation: the reference is built here with NumPy straight from
the data files.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.agn.blocks.feii import boroson_green_feii_block
from tengri.components.agn.blr import _blr_l_hbeta, _fe2_pseudo_continuum, compute_blr_sed
from tengri.utils.physics_constants import C_AA, C_KM_S

DATA = Path(tengri.__file__).parent / "data" / "agn_fe2"
WIN = (4434.0, 4684.0)
FWHM = 5000.0


def _native(name: str):
    d = np.genfromtxt(DATA / name, comments="#")
    return 10.0 ** d[:, 0], d[:, 1]


def _ref_template(wave: np.ndarray) -> np.ndarray:
    """Clip at zero, linear-in-lambda resample, UV below 3500 A (no broadening)."""
    uw, uf = _native("fe_uv_pyqsofit.txt")
    ow, of = _native("fe_optical_pyqsofit.txt")
    uv = np.interp(wave, uw, np.maximum(uf, 0.0), left=0.0, right=0.0)
    op = np.interp(wave, ow, np.maximum(of, 0.0), left=0.0, right=0.0)
    return np.where(wave < 3500.0, uv, op)


def _ref_broadened(wave: np.ndarray, fwhm: float) -> np.ndarray:
    """Independent reference: dense wide grid, velocity-Gaussian, zero padded."""
    dense = np.arange(1000.0, 8000.0, 0.25)
    t = _ref_template(dense)
    out = np.empty_like(wave)
    for i, w in enumerate(wave):
        sig = w * fwhm / 2.3548 / C_KM_S
        k = np.exp(-0.5 * ((dense - w) / sig) ** 2)
        out[i] = np.sum(t * k) / np.sum(k)
    return out


def _native_integral(w, f, lo, hi):
    """Trapezoid integral of the zero-clipped native nodes over [lo, hi], exact at the ends."""
    m = (w > lo) & (w < hi)
    ew = np.concatenate([[lo], w[m], [hi]])
    ef = np.interp(ew, w, np.maximum(f, 0.0))
    return np.trapezoid(ef, ew)


def _slope(wave, f, lo, hi, floor_frac=0.05):
    band = (wave >= lo) & (wave <= hi)
    m = band & (f > floor_frac * f[band].max())
    return np.polyfit(np.log(wave[m]), np.log(f[m]), 1)[0]


def _window_integral(wave, f):
    m = (wave >= WIN[0]) & (wave <= WIN[1])
    return np.trapezoid(f[m], wave[m])


def _block_lambda(wave, strength=1.0, fwhm=FWHM):
    return np.asarray(
        boroson_green_feii_block(
            jnp.asarray(wave),
            0.0,
            jnp.asarray(1e44),
            agn_fe2_strength=strength,
            agn_blr_fwhm_kms=fwhm,
        )
    )


@pytest.mark.parametrize(("lo", "hi"), [(4000.0, 6000.0), (2200.0, 3000.0)])
def test_block_log_slope_matches_template_flambda(lo, hi):
    """The L_lambda of the block has the template's F_lambda shape (no lambda^-2 tilt)."""
    wave = np.arange(lo - 100.0, hi + 100.0, 1.0)
    got = _block_lambda(wave)
    ref = _ref_broadened(wave, FWHM)
    assert abs(_slope(wave, got, lo, hi) - _slope(wave, ref, lo, hi)) < 0.01


def test_negative_template_nodes_are_clipped_at_load():
    from tengri.components.agn import blr

    _uw, uf, _ow, of = blr._load_fe2_templates()
    assert uf.min() >= 0.0
    assert of.min() >= 0.0
    # the raw files do contain negative nodes (this guards the test premise)
    assert np.genfromtxt(DATA / "fe_optical_pyqsofit.txt", comments="#")[:, 1].min() < 0.0


@pytest.mark.parametrize(
    ("name", "lo", "hi"),
    [("fe_optical_pyqsofit.txt", 4200.0, 5400.0), ("fe_uv_pyqsofit.txt", 2200.0, 3300.0)],
)
def test_resampling_conserves_template_integral(name, lo, hi):
    """Unbroadened pseudo-continuum integral equals the native-node integral (1e-4).

    The grid always contains the R_Fe window (the normalisation needs it); the
    integral is taken over ``[lo, hi]`` only, and compared with the trapezoid
    integral of the clipped native nodes, both divided by the native window
    integral.
    """
    seg = np.linspace(lo, hi, int(round((hi - lo) / 0.2)) + 1)
    win = np.arange(WIN[0], WIN[1] + 0.25, 0.25)
    wave = seg if lo < WIN[0] < hi else np.concatenate([seg, win])
    tiny = np.asarray(_fe2_pseudo_continuum(jnp.asarray(wave), 1e-4, 1.0))
    ow, of = _native("fe_optical_pyqsofit.txt")
    norm = _native_integral(ow, of, *WIN)
    nw, nf = _native(name)
    native = _native_integral(nw, nf, lo, hi) / norm
    sel = (wave >= lo) & (wave <= hi)
    got = np.trapezoid(tiny[sel], wave[sel])
    assert got == pytest.approx(native, rel=1e-4)


def test_unbroadened_shape_is_the_clipped_linear_template():
    wave = np.arange(4200.0, 5600.0, 0.5)
    got = np.asarray(_fe2_pseudo_continuum(jnp.asarray(wave), 1e-4, 1.0))
    ref = _ref_template(wave)
    scale = got.max() / ref.max()
    np.testing.assert_allclose(got, ref * scale, rtol=1e-6, atol=1e-6 * got.max())


def test_window_normalisation_is_r_fe_times_lhbeta():
    """Energy in 4434-4684 A relative to L(Hbeta) equals agn_fe2_strength."""
    wave = np.arange(4300.0, 4800.0, 0.25)
    l5100, cf, eff, fbol = 1e44, 0.1, 0.08, 9.0
    l_hb = float(_blr_l_hbeta(l5100 * fbol, cf, eff))
    for r_fe in (0.5, 1.7):
        lam = np.asarray(
            boroson_green_feii_block(
                jnp.asarray(wave),
                0.0,
                jnp.asarray(l5100),
                agn_fe2_strength=r_fe,
                agn_blr_fwhm_kms=FWHM,
                agn_blr_f_bol=fbol,
                agn_blr_cf=cf,
                agn_blr_line_efficiency=eff,
            )
        )
        assert _window_integral(wave, lam) / l_hb == pytest.approx(r_fe, rel=1e-3)


def test_compute_blr_sed_feii_is_flambda_shape_with_same_normalisation():
    """The monolithic BLR path (returns L_nu) carries the same FeII L_lambda."""
    wave = np.arange(4000.0, 6000.0, 1.0)
    l_bol, cf, eff, r_fe = 1e45, 0.1, 0.08, 1.3
    kw = dict(covering_fraction=cf, fwhm_kms=FWHM, line_efficiency=eff)
    with_fe = compute_blr_sed(jnp.asarray(wave), l_bol, agn_fe2_strength=r_fe, **kw)
    no_fe = compute_blr_sed(jnp.asarray(wave), l_bol, agn_fe2_strength=0.0, **kw)
    lam = np.asarray(with_fe - no_fe) * C_AA / wave**2  # L_nu -> L_lambda
    ref = _ref_broadened(wave, FWHM)
    assert abs(_slope(wave, lam, 4100.0, 5900.0) - _slope(wave, ref, 4100.0, 5900.0)) < 0.01
    l_hb = float(_blr_l_hbeta(l_bol, cf, eff))
    assert _window_integral(wave, lam) / l_hb == pytest.approx(r_fe, rel=1e-3)


@pytest.mark.parametrize("name", ["fe_uv_pyqsofit.txt", "fe_optical_pyqsofit.txt"])
def test_feii_provenance_sha256_matches_shipped_file(name):
    """PROVENANCE.md records the real sha256 of each shipped FeII template."""
    import hashlib
    import re

    text = (DATA / "PROVENANCE.md").read_text()
    m = re.search(rf"\*\*File\*\*: `{re.escape(name)}`.*?\*\*SHA256\*\*: `([0-9a-f]{{64}})`", text, re.S)
    assert m is not None, f"no SHA256 recorded for {name} in PROVENANCE.md"
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == m.group(1)
