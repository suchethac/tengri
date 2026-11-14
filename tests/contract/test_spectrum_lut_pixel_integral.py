# SPDX-License-Identifier: BSD-3-Clause
"""Pixel-integrated spectrum LUT serves the flux-conserving resample (#3413).

With ``resample="conserving"`` the exact path broadens the model on its own grid
(velocity dispersion, then the instrument line-spread function) and takes the mean
over each pixel's edges last. ``SpectrumPrecomp()`` must reproduce that per pixel,
for the flux and for its Jacobian with respect to the free parameters.

The metric is the largest per-pixel relative error. The denominator is floored at
1e-6 of the spectrum peak (flux) or of the column peak (Jacobian) so that pixels at
the edge of the model range, where both values vanish, do not divide by zero.
"""

import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import SpectrumPrecomp

pytestmark = pytest.mark.regression_paper

_TOL = 1e-4
_FLOOR = 1e-6

_BARE_SSP = "data/fsps_prsc_miles_chabrier.h5"
_CUE_WEIGHTS = "data/cue_weights.npz"

_SDSS = np.linspace(3800.0, 9200.0, 200)  # 27 Angstrom pixels
_DESI = np.linspace(6000.0, 6048.0, 61)  # 0.8 Angstrom pixels
_PRISM = np.geomspace(6000.0, 9000.0, 40)  # constant R_pix ~ 100
_IR = np.geomspace(2.0e4, 2.0e5, 60)  # 3.9 % pixels across the Calzetti clip at 3.115 um rest

_TWO_COMP_FREE = ("sfh_dpl_alpha", "dust_tau_bc", "dust_tau_diff")
_SINGLE_FREE = ("sfh_dpl_alpha", "dust_tau_v")


def _dust(kind):
    from tengri import DEFAULT, Fixed, Uniform

    if kind == "single":
        return {
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Uniform(0.0, 1.5, "dust_tau_v"),
        }
    return {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_bc": Uniform(0.0, 2.0, "dust_tau_bc"),
        "tau_diff": Uniform(0.0, 1.5, "dust_tau_diff"),
    }


def _model(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue, approx):
    """Exact (``approx=None``) or LUT model on the conserving resample.

    ``resolution`` is a scalar R, ``None``, or ``("banded", R)`` for a Gaussian banded
    resolution matrix (DESI-like, #3413).
    """
    from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
    from tengri.observation.banded import gaussian_resolution_bands

    spectro = {"resample": "conserving", "wave_obs": jnp.asarray(wave_obs)}
    if isinstance(resolution, tuple):
        spectro["resolution_matrix"] = gaussian_resolution_bands(
            jnp.asarray(wave_obs), resolution[1]
        )
    else:
        spectro["resolution"] = resolution
    obs = Observation(spectroscopy=Spectroscopy(**spectro))
    neb = {"type": "cue", "all_params": Fixed(DEFAULT)} if with_cue else {"type": "none"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": "dpl",
                "alpha": Uniform(0.1, 5.0, "sfh_dpl_alpha"),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation=_dust(dust_kind),
            neb=neb,
            redshift=Fixed(0.1),
        )
        spec = base.spec.merge_observation_params(sigma_v_kms=Fixed(float(sigma_v)))
        return type(base)(spec, ssp, observation=obs, approx=approx)


def _pixel_rel(a, b):
    """Largest per-pixel relative error, the denominator floored at a fraction of the peak."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    floor = _FLOOR * np.max(np.abs(b), axis=0, keepdims=True)
    denom = np.maximum(np.abs(b), floor)
    return float(np.max(np.abs(a - b) / np.where(denom > 0, denom, 1.0)))


def _flux_and_jacobian(model, names, base):
    def f(theta):
        p = dict(base)
        for i, name in enumerate(names):
            p[name] = theta[i]
        return model.predict_observables(p).spec_fnu

    theta0 = jnp.asarray([base[n] for n in names])
    return np.asarray(f(theta0)), np.asarray(jax.jacfwd(f)(theta0))


def pair_errors(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue):
    """Return ``(flux_err, jacobian_err)`` of the LUT against the exact conserving path."""
    names = _SINGLE_FREE if dust_kind == "single" else _TWO_COMP_FREE
    exact = _model(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue, None)
    lut = _model(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue, SpectrumPrecomp())
    base = {
        "sfh_dpl_alpha": jnp.asarray(1.5),
        "dust_tau_bc": jnp.asarray(0.5),
        "dust_tau_diff": jnp.asarray(0.3),
        "dust_tau_v": jnp.asarray(0.5),
    }
    f_e, j_e = _flux_and_jacobian(exact, names, base)
    f_l, j_l = _flux_and_jacobian(lut, names, base)
    return _pixel_rel(f_l, f_e), _pixel_rel(j_l, j_e)


_CASES = [
    pytest.param(
        _SDSS, None, 0.0, "two", False, id="sdss-2comp-nolines-sv0"
    ),
    pytest.param(
        _SDSS, 2000.0, 150.0, "two", False, id="sdss-2comp-R2000-sv150"
    ),
    pytest.param(
        _SDSS, 2000.0, 150.0, "single", True, id="sdss-1screen-cue-R2000-sv150"
    ),
    pytest.param(
        _DESI, 3000.0, 150.0, "two", False, id="desi-2comp-R3000-sv150"
    ),
    pytest.param(
        _PRISM, 100.0, 0.0, "two", False, id="prism-2comp-R100-sv0"
    ),
    pytest.param(
        _SDSS, None, 150.0, "two", False, id="sdss-2comp-nores-sv150"
    ),
    pytest.param(
        _SDSS, ("banded", 2000.0), 0.0, "two", False, id="sdss-2comp-banded-R2000-sv0"
    ),
    pytest.param(
        _SDSS, ("banded", 2000.0), 150.0, "two", False, id="sdss-2comp-banded-R2000-sv150"
    ),
    pytest.param(
        _DESI, ("banded", 3000.0), 150.0, "two", False, id="desi-2comp-banded-R3000-sv150"
    ),
    pytest.param(
        _SDSS, None, 0.0, "single", True, id="sdss-1screen-cue-nores-sv0"
    ),
    pytest.param(
        _SDSS, None, 150.0, "single", True, id="sdss-1screen-cue-nores-sv150"
    ),
    pytest.param(
        _SDSS, ("banded", 2000.0), 0.0, "two", True, id="sdss-2comp-cue-banded-R2000-sv0"
    ),
    pytest.param(
        _DESI, ("banded", 3000.0), 150.0, "two", True, id="desi-2comp-cue-banded-R3000-sv150"
    ),
    pytest.param(
        _SDSS, None, 0.0, "two", True, id="sdss-2comp-cue-nores-sv0"
    ),
    pytest.param(
        _SDSS, None, 150.0, "two", True, id="sdss-2comp-cue-nores-sv150"
    ),
    pytest.param(_IR, None, 0.0, "two", False, id="ir-2comp-nores-sv0"),
    pytest.param(_IR, None, 150.0, "two", False, id="ir-2comp-nores-sv150"),
    pytest.param(_IR, None, 0.0, "single", False, id="ir-1screen-nores-sv0"),
    pytest.param(_IR, 100.0, 150.0, "two", False, id="ir-2comp-R100-sv150"),
    pytest.param(
        _SDSS, 2000.0, 0.0, "two", True, id="sdss-2comp-cue-R2000-sv0"
    ),
]


def _require_data(with_cue):
    if not os.path.isfile(_BARE_SSP):
        pytest.skip("bare-stellar SSP not available under data/")
    if with_cue and not os.path.isfile(_CUE_WEIGHTS):
        pytest.skip("Cue weights not available under data/")


@pytest.mark.parametrize(
    ("wave_obs", "resolution", "sigma_v", "dust_kind", "with_cue"), _CASES
)
def test_pixel_integral_lut_matches_conserving_exact(
    wave_obs, resolution, sigma_v, dust_kind, with_cue
):
    """SpectrumPrecomp under resample='conserving' matches the exact conserving path."""
    from tengri import load_ssp_data

    _require_data(with_cue)
    ssp = load_ssp_data(_BARE_SSP)
    flux_err, jac_err = pair_errors(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue)
    print(f"\nflux max rel err = {flux_err:.3e}   gradient max rel err = {jac_err:.3e}")
    assert flux_err <= _TOL, f"flux max per-pixel rel err {flux_err:.3e} > {_TOL:g}"
    assert jac_err <= _TOL, f"gradient max per-pixel rel err {jac_err:.3e} > {_TOL:g}"


#: Pixels the table computes exactly per draw must match the exact path to rounding, on the
#: flux and on the Jacobian (#3413).
_FLAGGED_TOL = 1e-10

_FLAGGED_CASES = [
    pytest.param(_SDSS, None, 0.0, "two", False, id="sdss-2comp-6300A-jump"),
    pytest.param(_SDSS, 2000.0, 150.0, "single", True, id="sdss-1screen-cue-R2000-sv150"),
    pytest.param(_IR, None, 0.0, "two", False, id="ir-2comp"),
    pytest.param(_IR, None, 0.0, "single", False, id="ir-1screen"),
    pytest.param(_IR, 100.0, 150.0, "two", False, id="ir-2comp-R100-sv150"),
]


@pytest.mark.parametrize(
    ("wave_obs", "resolution", "sigma_v", "dust_kind", "with_cue"), _FLAGGED_CASES
)
def test_flagged_pixels_are_exact(wave_obs, resolution, sigma_v, dust_kind, with_cue):
    """Pixels next to a dust-law breakpoint or curvature are the exact path's, to rounding."""
    from tengri import load_ssp_data

    _require_data(with_cue)
    ssp = load_ssp_data(_BARE_SSP)
    names = _SINGLE_FREE if dust_kind == "single" else _TWO_COMP_FREE
    exact = _model(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue, None)
    lut = _model(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue, SpectrumPrecomp())
    layout = lut._spectrum_flagged_pixels()
    assert layout is not None and layout.pixels.size > 0
    base = {
        "sfh_dpl_alpha": jnp.asarray(1.5),
        "dust_tau_bc": jnp.asarray(0.5),
        "dust_tau_diff": jnp.asarray(0.3),
        "dust_tau_v": jnp.asarray(0.5),
    }
    f_e, j_e = _flux_and_jacobian(exact, names, base)
    f_l, j_l = _flux_and_jacobian(lut, names, base)
    px = layout.pixels
    flux_err = float(np.max(np.abs(f_l - f_e)[px] / np.abs(f_e)[px]))
    floor = _FLOOR * np.max(np.abs(j_e), axis=0, keepdims=True)
    jac_err = float(
        np.max((np.abs(j_l - j_e) / np.maximum(np.abs(j_e), floor))[px])
    )
    print(f"\nflagged {px.tolist()}: flux {flux_err:.3e}  jacobian {jac_err:.3e}")
    assert flux_err <= _FLAGGED_TOL, f"flagged flux rel err {flux_err:.3e}"
    assert jac_err <= _FLAGGED_TOL, f"flagged Jacobian rel err {jac_err:.3e}"


def _family_model(
    ssp, family, approx, *, sigma_v_override=None, resolution_override=None, wave_obs=_SDSS
):
    """Two-component dust, a Fixed-z conserving spectrum, plus one family (#3413)."""
    from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform

    spectro = {"wave_obs": jnp.asarray(wave_obs), "resample": "conserving"}
    sigma_v = 0.0
    extra = {"neb": {"type": "none"}}
    if family in ("banded", "banded-point"):
        from tengri.observation.banded import gaussian_resolution_bands

        spectro["resolution_matrix"] = gaussian_resolution_bands(jnp.asarray(_SDSS), 2000.0)
        if family == "banded-point":
            spectro["resample"] = "point"
        sigma_v = 150.0
    elif family == "banded-cue":
        from tengri.observation.banded import gaussian_resolution_bands

        spectro["resolution_matrix"] = gaussian_resolution_bands(jnp.asarray(_SDSS), 2000.0)
        extra["neb"] = {"type": "cue", "all_params": Fixed(DEFAULT)}
        sigma_v = 150.0
    elif family == "cue-R2000-sv150":
        extra["neb"] = {"type": "cue", "all_params": Fixed(DEFAULT)}
        spectro["resolution"] = 2000.0
        sigma_v = 150.0
    elif family == "agn":
        extra["agn"] = {
            "type": "composable",
            "disc": {"type": "powerlaw"},
            "all_params": Fixed(DEFAULT),
        }
    elif family == "dust-emission":
        extra["dust_emission"] = {"type": "modified_blackbody", "all_params": Fixed(DEFAULT)}
    elif family == "radio":
        extra["radio"] = {"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}}
    elif family == "xray":
        extra["xray"] = {"type": "simple", "delta_alpha_ox": Fixed(-1.5)}
    elif family == "igm":
        extra["igm"] = {"type": "inoue"}
    if sigma_v_override is not None:
        sigma_v = sigma_v_override
    if resolution_override is not None:
        spectro["resolution"] = resolution_override
    obs = Observation(spectroscopy=Spectroscopy(**spectro))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation=_dust("two"),
            redshift=Fixed(0.1),
            **extra,
        )
    spec = base.spec.merge_observation_params(sigma_v_kms=Fixed(sigma_v))
    # Outside the suppression above, so the approx resolution's own warnings reach the caller.
    return type(base)(spec, ssp, observation=obs, approx=approx)


_STATIC_REFUSED = ["igm", "dust-emission"]
_PROJECTED = ["agn", "radio", "xray"]


#: Each projected family on a band where it is non-negligible: AGN on the optical SDSS grid,
#: radio at 1-10 cm, X-ray at 1-60 Angstrom (#3413). Dust emission is refused (see
#: ``_STATIC_REFUSED``): on a mid-IR band the stellar pixel moments alone miss by ~1e-4.
_PROJECTED_BANDS = {
    "radio": np.geomspace(1.0e8, 1.0e9, 60),
    "xray": np.geomspace(1.0, 60.0, 60),
}


def _family_pair_errors(ssp, family, sigma_v, resolution, wave_obs=_SDSS):
    """LUT vs exact conserving errors for one projected family (flux, Jacobian)."""
    names = ("dust_tau_bc", "dust_tau_diff")
    kw = {"sigma_v_override": sigma_v, "resolution_override": resolution, "wave_obs": wave_obs}
    exact = _family_model(ssp, family, None, **kw)
    lut = _family_model(ssp, family, SpectrumPrecomp(), **kw)
    base = {"dust_tau_bc": jnp.asarray(0.5), "dust_tau_diff": jnp.asarray(0.3)}
    f_e, j_e = _flux_and_jacobian(exact, names, base)
    f_l, j_l = _flux_and_jacobian(lut, names, base)
    return _pixel_rel(f_l, f_e), _pixel_rel(j_l, j_e)


@pytest.mark.parametrize("resolution", [None, 2000.0], ids=["nores", "R2000"])
def test_projected_agn_matches_conserving_exact(resolution):
    """AGN continuum light is served per draw, matching the exact conserving path (#3413)."""
    from tengri import load_ssp_data

    _require_data(False)
    ssp = load_ssp_data(_BARE_SSP)
    flux_err, jac_err = _family_pair_errors(ssp, "agn", 150.0, resolution)
    print(f"\nagn: flux max rel err = {flux_err:.3e}   gradient max rel err = {jac_err:.3e}")
    assert flux_err <= _TOL, f"flux max per-pixel rel err {flux_err:.3e} > {_TOL:g}"
    assert jac_err <= _TOL, f"gradient max per-pixel rel err {jac_err:.3e} > {_TOL:g}"


@pytest.mark.parametrize("family", sorted(_PROJECTED_BANDS))
def test_projected_family_matches_conserving_exact(family):
    """Dust-emission, radio and X-ray light is served per draw on its own band (#3413)."""
    from tengri import load_ssp_data

    _require_data(False)
    ssp = load_ssp_data(_BARE_SSP)
    flux_err, jac_err = _family_pair_errors(ssp, family, 150.0, None, _PROJECTED_BANDS[family])
    print(f"\n{family}: flux max rel err = {flux_err:.3e}   gradient max rel err = {jac_err:.3e}")
    assert flux_err <= _TOL, f"flux max per-pixel rel err {flux_err:.3e} > {_TOL:g}"
    assert jac_err <= _TOL, f"gradient max per-pixel rel err {jac_err:.3e} > {_TOL:g}"


@pytest.mark.parametrize("family", _PROJECTED)
def test_projected_family_routes_to_table_under_auto(family):
    """A projected family is served: auto takes the pixel-moment table (#3413)."""
    from tengri import load_ssp_data
    from tengri.inference.fitter import _spectrum_lut_for_auto

    _require_data(False)
    ssp = load_ssp_data(_BARE_SSP)
    assert _spectrum_lut_for_auto(_family_model(ssp, family, None)) is not None


@pytest.mark.parametrize("family", _STATIC_REFUSED)
def test_refused_family_routes_exact_under_auto(family):
    """A family the table still refuses is refused statically: auto goes exact, explicit raises."""
    from tengri import load_ssp_data
    from tengri.inference.fitter import _spectrum_lut_for_auto

    _require_data(family.startswith("cue"))
    ssp = load_ssp_data(_BARE_SSP)
    assert _spectrum_lut_for_auto(_family_model(ssp, family, None)) is None
    with pytest.raises(ValueError, match="pixel integral"):
        _family_model(ssp, family, SpectrumPrecomp())


def test_nebular_scalar_resolution_routes_to_table_under_auto():
    """A nebular family under a scalar resolution is served: auto takes the table (#3413)."""
    from tengri import load_ssp_data
    from tengri.inference.fitter import _spectrum_lut_for_auto

    _require_data(True)
    ssp = load_ssp_data(_BARE_SSP)
    assert _spectrum_lut_for_auto(_family_model(ssp, "cue-R2000-sv150", None)) is not None


def test_banded_nebular_routes_to_table_under_auto():
    """A nebular family under a banded matrix on fixed-z pixel means is served (#3413)."""
    from tengri import load_ssp_data
    from tengri.inference.fitter import _spectrum_lut_for_auto

    _require_data(True)
    ssp = load_ssp_data(_BARE_SSP)
    assert _spectrum_lut_for_auto(_family_model(ssp, "banded-cue", None)) is not None


def test_banded_resolution_routes_to_table_under_auto():
    """A banded resolution matrix on fixed-z pixel means is served: auto takes the table."""
    from tengri import load_ssp_data
    from tengri.inference.fitter import _spectrum_lut_for_auto

    _require_data(False)
    ssp = load_ssp_data(_BARE_SSP)
    assert _spectrum_lut_for_auto(_family_model(ssp, "banded", None)) is not None


def test_banded_resolution_on_point_samples_falls_back_to_exact():
    """A banded matrix on point samples is not served: auto goes exact, explicit falls back."""
    from tengri import load_ssp_data
    from tengri.inference.fitter import _spectrum_lut_for_auto

    _require_data(False)
    ssp = load_ssp_data(_BARE_SSP)
    assert _spectrum_lut_for_auto(_family_model(ssp, "banded-point", None)) is None
    with pytest.warns(UserWarning, match="banded resolution matrix"):
        _family_model(ssp, "banded-point", SpectrumPrecomp())


def _rest_fnu_pair(ssp, family, sigma_v, resolution, wave_obs):
    """Rest-frame F_nu at 10 pc: the table's output and the exact rest SED, same edges (#3413).

    The exact reference projects ``sed_intrinsic`` (every family, no instrument) on the
    pixel edges ``wave_obs / (1 + z)``, the rest frame the table integrates over.
    """
    from tengri.observation.spectrum import project_spectrum
    from tengri.utils.physics_constants import TEN_PC_CM

    kw = {"sigma_v_override": sigma_v, "resolution_override": resolution, "wave_obs": wave_obs}
    exact = _family_model(ssp, family, None, **kw)
    lut = _family_model(ssp, family, SpectrumPrecomp(), **kw)
    params = {"dust_tau_bc": jnp.asarray(0.5), "dust_tau_diff": jnp.asarray(0.3)}
    exact_state = exact.predict_state(params)
    rest_edges = np.asarray(wave_obs) / 1.1
    ref = project_spectrum(
        exact_state.sed_intrinsic,
        exact_state.wave,
        jnp.asarray(rest_edges),
        0.0,
        TEN_PC_CM,
        conserving=True,
    )
    lut_state = lut.predict_state(params)
    full = {**dict(lut.spec.get_fixed_values()), **params}
    out = lut.observation.predict_spectrum_via_precomp(
        lut_state, full, observables_type=None, sigma_lib_kms=0.0, lsf_scale=1.0, lsf_n_bins=16
    )
    return np.asarray(out["spec_rest_fnu"]), np.asarray(ref)


_REST_CASES = [
    pytest.param("agn", 150.0, 2000.0, _SDSS, id="agn-sv150-R2000"),
    pytest.param("radio", 0.0, None, _PROJECTED_BANDS["radio"], id="radio-1-10cm"),
    pytest.param("xray", 0.0, None, _PROJECTED_BANDS["xray"], id="xray-1-60A"),
    pytest.param("cue-R2000-sv150", 150.0, 2000.0, _SDSS, id="cue-sv150-R2000"),
]


def test_calzetti_clip_zero_is_a_breakpoint():
    """The Calzetti IR branch is clipped at zero near 3.115 um rest; the table splits there (#3413)."""
    from tengri.components.dust.attenuation import calzetti, law_breakpoints

    assert any(abs(bp - 31149.76) < 1.0 for bp in law_breakpoints("calzetti"))
    assert float(calzetti(jnp.asarray(31160.0))) == 0.0
    assert float(calzetti(jnp.asarray(31100.0))) > 0.0


def test_segment_expansion_point_lies_in_its_segment():
    """A pixel that straddles a breakpoint expands each part about a point inside that part (#3413)."""
    from tengri.observation.spectrum import segment_centres

    centres = np.array([30500.0, 31400.0])
    cs = segment_centres(centres, [31149.76])
    assert cs.shape == (2, 2)
    # Pixel 0 (centre below the break) keeps its centre in segment 0.
    assert cs[0, 0] == 30500.0
    # Pixel 1 (centre above the break) keeps its centre in segment 1, and its segment 0 part
    # is expanded about the midpoint of the pixel's part below the break, not the break itself.
    assert cs[1, 1] == 31400.0
    assert 30000.0 < cs[1, 0] < 31149.76


@pytest.mark.parametrize(("family", "sigma_v", "resolution", "wave_obs"), _REST_CASES)
def test_rest_frame_output_matches_exact_rest_sed(family, sigma_v, resolution, wave_obs):
    """spec_rest_fnu under the table equals the exact rest SED, every served family (#3413)."""
    from tengri import load_ssp_data

    _require_data(family.startswith("cue"))
    ssp = load_ssp_data(_BARE_SSP)
    got, ref = _rest_fnu_pair(ssp, family, sigma_v, resolution, wave_obs)
    assert np.all(np.isfinite(got))
    err = _pixel_rel(got, ref)
    print(f"\n{family}: rest-frame max rel err = {err:.3e}")
    assert err <= _TOL, f"rest-frame max per-pixel rel err {err:.3e} > {_TOL:g}"
