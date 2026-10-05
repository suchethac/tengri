# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2597: ``schreiber2016`` is the tabulated Schreiber+2018 library.

``dust_emission={'type': 'schreiber2016'}`` used to evaluate an analytic
modified-blackbody (beta = 1.5) plus six Drude profiles, whose band powers sit
0.000-8.4x from the library CIGALE ships under that module name, while the
tabulated component that did carry the library mixed unit-normalised templates
(a PAH *power* fraction) and had no public spelling.

The library (Schreiber et al. 2018, A&A 609, A30, arXiv:1710.10276) tabulates
one kilogram of dust per temperature node as a dust continuum and a PAH
template; the paper defines ``f_PAH`` as the PAH *mass* fraction (Sect. 3.2,
eq. 14), so the two per-kg templates mix as ``(1 - f) cont + f pah`` and the
mixture is renormalised to the absorbed luminosity.

Every expected value below is mixed in the test from the arrays of
``data/schreiber2016_templates.h5`` (per-kg templates, ``L_lambda`` times
``lambda^2 / c`` to ``L_nu``), never read back from the model.
"""

from __future__ import annotations

import importlib.util
import inspect
import warnings
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contract.test_dust_ir_emission_precompute_parity import (
    assert_precompute_matches_exact,
    synthetic_ssp,  # noqa: F401  (fixture)
)

pytestmark = pytest.mark.regression_bug

_REPO = Path(__file__).resolve().parents[3]
_H5 = _REPO / "data" / "schreiber2016_templates.h5"
_C_AA = 2.99792458e18  # speed of light [Angstrom/s]
_BANDS_UM = ((3, 8), (8, 24), (24, 70), (70, 160), (160, 500), (500, 1000))

if not _H5.is_file():
    pytest.skip(f"template file {_H5} not available", allow_module_level=True)


def _model():
    """The registry closure, resolved through the public registry."""
    from tengri.components.dust.emission import DUST_EMISSION_MODELS

    return DUST_EMISSION_MODELS["schreiber2016"]


def _arrays() -> dict[str, np.ndarray]:
    with h5py.File(_H5, "r") as f:
        return {
            k: np.asarray(f[k][:]) for k in ("wavelength_aa", "tdust_grid", "continuum", "pah")
        }


def _per_kg_mix_lnu(T: float, f: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference ``(wavelength_aa, L_nu)``: per-kg templates, T-interpolated, mass-mixed."""
    a = _arrays()
    t = a["tdust_grid"]
    i = int(np.clip(np.searchsorted(t, T) - 1, 0, t.size - 2))
    ft = (T - t[i]) / (t[i + 1] - t[i])
    cont = (1 - ft) * a["continuum"][i] + ft * a["continuum"][i + 1]
    pah = (1 - ft) * a["pah"][i] + ft * a["pah"][i + 1]
    w = a["wavelength_aa"]
    return w, ((1 - f) * cont + f * pah) * w**2 / _C_AA


def _power(w: np.ndarray, lnu: np.ndarray, lo_um: float = 1.0, hi_um: float = 1000.0) -> float:
    m = (w >= lo_um * 1e4) & (w <= hi_um * 1e4)
    return float(-np.trapezoid(lnu[m], _C_AA / w[m]))


def _band_fractions(w: np.ndarray, lnu: np.ndarray) -> np.ndarray:
    total = _power(w, lnu)
    return np.array([_power(w, lnu, a, b) / total for a, b in _BANDS_UM])


def _model_lnu(w: np.ndarray, T: float, f: float) -> np.ndarray:
    return np.asarray(_model()(jnp.asarray(w), 1.0, dust_T=T, dust_f_pah=f))


@pytest.mark.parametrize(("T", "f"), [(20.0, 0.05), (35.0, 0.2), (50.0, 0.5)])
def test_band_powers_equal_the_per_kg_library_mix(T, f):
    """Band fractions of the registry model equal the library mixed per kg, to 2e-3."""
    w, ref = _per_kg_mix_lnu(T, f)
    got = _band_fractions(w, _model_lnu(w, T, f))
    want = _band_fractions(w, ref)
    np.testing.assert_allclose(got, want, rtol=2e-3, err_msg=f"T={T} f_pah={f}")


@pytest.mark.parametrize(("T", "f"), [(20.0, 0.05), (35.0, 0.2), (50.0, 0.5)])
def test_f_pah_is_the_mass_fraction_not_the_power_fraction(T, f):
    """The SED is ``(1 - w) S(0) + w S(1)``, ``w = f R / (1 - f + f R)``, R the per-kg ratio."""
    a = _arrays()
    i = int(np.searchsorted(a["tdust_grid"], T))
    wave = a["wavelength_aa"]
    to_lnu = wave**2 / _C_AA
    # R over the whole tabulated range: that is what the model renormalises on.
    r = _power(wave, a["pah"][i] * to_lnu, 0.0, 1e6) / _power(
        wave, a["continuum"][i] * to_lnu, 0.0, 1e6
    )
    w_pow = f * r / (1 - f + f * r)
    s0, s1 = _model_lnu(wave, T, 0.0), _model_lnu(wave, T, 1.0)
    got = _model_lnu(wave, T, f)
    np.testing.assert_allclose(got, (1 - w_pow) * s0 + w_pow * s1, rtol=1e-6, atol=0.0)
    # R ~ 3: a mass fraction of f carries MORE than f of the power.
    assert w_pow > f


def test_interpolated_temperature_is_between_the_two_node_mixes():
    """T = 35.5 K is the per-kg template halfway between the 35 K and 36 K nodes."""
    w, ref = _per_kg_mix_lnu(35.5, 0.2)
    got = _model_lnu(w, 35.5, 0.2)
    ref_norm = ref / _power(w, ref)
    got_norm = got / _power(w, got)
    np.testing.assert_allclose(got_norm, ref_norm, rtol=1e-9, atol=0.0)
    lo = _band_fractions(w, _per_kg_mix_lnu(35.0, 0.2)[1])
    hi = _band_fractions(w, _per_kg_mix_lnu(36.0, 0.2)[1])
    mid = _band_fractions(w, got)
    assert np.all(mid >= np.minimum(lo, hi) - 1e-12) and np.all(mid <= np.maximum(lo, hi) + 1e-12)


def test_the_stand_in_is_gone():
    """No analytic ``schreiber2016`` closure, component, or ``schreiber2016_ir`` module remains."""
    from tengri.components.dust.emission.analytic import _closures

    assert not hasattr(_closures, "schreiber2016"), "analytic stand-in closure still defined"
    for module in (
        "tengri.components.dust.emission.analytic.schreiber2016",
        "tengri.components.dust.schreiber2016_ir",
    ):
        assert importlib.util.find_spec(module) is None, f"{module} still importable"


def _fixed_model(ssp, T: float | None, f: float | None):
    from tengri import DEFAULT, Fixed, SEDModel

    emission = {"type": "schreiber2016", "all_params": Fixed(DEFAULT)}
    if T is not None:
        emission["T"] = Fixed(T)
    if f is not None:
        emission["f_pah"] = Fixed(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "const", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
            dust_attenuation={"law": "calzetti", "type": "single_component", "tau_v": Fixed(0.5)},
            dust_emission=emission,
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )


@pytest.mark.parametrize("f", [0.2, 0.35, 0.5])
def test_build_peaks_on_the_76_micron_pah_complex(synthetic_ssp, f):  # noqa: F811
    """Through ``SEDModel.build`` the nu L_nu peak of the dust IR sits at the library's 7.6 um."""
    model = _fixed_model(synthetic_ssp, 35.0, f)
    state = model.predict_state(dict(model.spec.sample(jax.random.PRNGKey(0))))
    wave = np.asarray(state.wave)
    sed = np.asarray(state.derived["sed_dust_ir"])
    peak_um = wave[np.argmax(sed * _C_AA / wave)] / 1e4

    w, ref = _per_kg_mix_lnu(35.0, f)
    ref_peak_um = w[np.argmax(ref * _C_AA / w)] / 1e4
    assert 7.5 < ref_peak_um < 7.7, "library reference peak moved off the 7.7 um complex"
    assert abs(peak_um / ref_peak_um - 1.0) < 0.02, (peak_um, ref_peak_um)


def test_defaults_are_declared_once(synthetic_ssp):  # noqa: F811
    """Closure signature, component declaration and a built ``Fixed(DEFAULT)`` model agree."""
    from tengri.components.dust.emission.templates.schreiber2016 import (
        Schreiber2016IRSEDComponent,
    )
    from tengri.components.dust.emission_templates import create_schreiber2016_from_grid

    closure = create_schreiber2016_from_grid(str(_H5))
    sig = inspect.signature(closure).parameters
    declared = {
        d.name: float(d.prior.default) for d in Schreiber2016IRSEDComponent().declared_parameters()
    }
    fixed = _fixed_model(synthetic_ssp, None, None).spec.get_fixed_values()
    for knob, kw in (("dust_T", "dust_T"), ("dust_f_pah", "dust_f_pah")):
        values = {sig[kw].default, declared[knob], float(fixed[knob])}
        assert len(values) == 1, f"{knob}: closure/component/build disagree: {values}"


def test_float32_is_finite_and_tracks_float64():
    """In float32 the SED is finite and within 1e-3 of the float64 result at every node."""
    from tengri.components.dust.emission_templates import create_schreiber2016_from_grid

    w, _ = _per_kg_mix_lnu(30.0, 0.1)
    want = _model_lnu(w, 30.0, 0.1)
    with jax.enable_x64(False):
        fn32 = create_schreiber2016_from_grid(str(_H5))
        got = np.asarray(fn32(jnp.asarray(w, dtype=jnp.float32), 1.0, dust_T=30.0, dust_f_pah=0.1))
    assert got.dtype == np.float32 and np.all(np.isfinite(got)) and np.all(got >= 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-6 * want.max())


@pytest.mark.parametrize(("knob", "x0"), [("dust_T", 35.5), ("dust_f_pah", 0.23)])
def test_gradient_is_finite_nonzero_and_equals_the_reference(knob, x0):
    """``jax.grad`` of the 8-24 um power fraction w.r.t. T and f_pah matches the in-test mix."""
    w, _ = _per_kg_mix_lnu(35.5, 0.23)
    wj = jnp.asarray(w)
    band = np.flatnonzero((w >= 8e4) & (w <= 24e4))
    full = np.flatnonzero((w >= 1e4) & (w <= 1e7))
    nu = jnp.asarray(_C_AA / w)

    def frac(lnu):
        return jnp.trapezoid(lnu[band], nu[band]) / jnp.trapezoid(lnu[full], nu[full])

    def objective(x):
        T, f = (x, 0.23) if knob == "dust_T" else (35.5, x)
        return frac(_model()(wj, 1.0, dust_T=T, dust_f_pah=f))

    grad = float(jax.grad(objective)(x0))
    assert np.isfinite(grad) and grad != 0.0

    h = 1e-4

    def reference(x):
        T, f = (x, 0.23) if knob == "dust_T" else (35.5, x)
        _, lnu = _per_kg_mix_lnu(T, f)
        return _power(w, lnu, 8, 24) / _power(w, lnu)

    fd = (reference(x0 + h) - reference(x0 - h)) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-3)


def test_wave_precomp_matches_the_exact_path(synthetic_ssp):  # noqa: F811
    """WavePrecomp photometry of a built ``schreiber2016`` model matches the exact path (0.5%)."""
    from tengri import DEFAULT, Fixed
    from tests.contract.test_dust_ir_emission_precompute_parity import _tophat

    filters = [_tophat(c) for c in (8.0e4, 2.4e5, 7.0e5, 1.6e6)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_precompute_matches_exact(
            synthetic_ssp,
            filters,
            "schreiber2016",
            dust_config={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": 0.5,
                "emission": {
                    "type": "schreiber2016",
                    "all_params": Fixed(DEFAULT),
                    "T": Fixed(35.0),
                    "f_pah": Fixed(0.3),
                },
            },
            tolerance=0.005,
        )


def test_missing_template_file_raises_with_path_and_recipe(tmp_path, monkeypatch):
    """No analytic fallback: a missing file raises, naming the path and how to regenerate it."""
    from tengri.components.dust.emission.templates.schreiber2016 import (
        Schreiber2016IRSEDComponent,
    )
    from tengri.components.dust.emission_templates import load_schreiber2016_templates

    missing = tmp_path / "schreiber2016_templates.h5"
    with pytest.raises(FileNotFoundError, match="regenerate_schreiber2016_from_cigale") as exc:
        load_schreiber2016_templates(str(missing))
    assert str(missing) in str(exc.value)

    monkeypatch.setattr("tengri._data_setup.find_data_str", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError, match="regenerate_schreiber2016_from_cigale"):
        Schreiber2016IRSEDComponent().load(None)


def test_provenance_points_at_the_one_paper():
    """h5 attrs and the bib agree on A&A 609, A30 = arXiv:1710.10276; the A35 entry is gone."""
    with h5py.File(_H5, "r") as f:
        attrs = {k: str(v) for k, v in f.attrs.items()}
    assert attrs["arxiv"] == "1710.10276"
    assert attrs["doi"] == "10.1051/0004-6361/201731506"
    assert "609" in attrs["paper"] and "A30" in attrs["paper"]
    assert "1606.00841" not in " ".join(attrs.values())

    from tengri.citations import associations

    bib = (_REPO / "src" / "tengri" / "citations" / "references.bib").read_text()
    assert "registry_key  = {schreiber2016}" not in bib
    assert "1601.02642" not in bib
    entry = bib[bib.index("@article{Schreiber_2018,") :].split("\n}\n", 1)[0]
    assert "1710.10276" in entry and "registry_key  = {schreiber2018}" in entry
    assert associations.DUST_EMISSION_CITATIONS["schreiber2016"] == ["schreiber2018"]
