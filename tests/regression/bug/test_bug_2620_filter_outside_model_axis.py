# SPDX-License-Identifier: BSD-3-Clause
"""#2620: a photometric filter beyond the model wavelength axis is refused at build.

The photometry integral zero-fills outside the rest-frame axis and divides by
the whole filter weight, so an uncovered band returned 0 and a partly covered
one returned the covered fraction, with no warning. ``SEDModel`` now measures,
at build time and after every component has declared its axis extension, the
fraction of each band's bandpass-weighted transmission outside the redshifted
axis, and raises ``ConfigError`` above ``FILTER_COVERAGE_TOLERANCE`` (1e-3).

Top-hat filters are built here on a dense geometric grid, so the trapezoid
quadrature of the helper reproduces the analytic integrals of ``1/lambda``
(photon counting) and ``1/lambda**2`` (energy) to well below 1e-6.

Masking is known only at fit time (``Fitter(presence=...)``,
``CatalogFitter(missing='mask')``), so the build-time check covers every filter
of the observation; there is no masked-band cell by design.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.config.exceptions import ConfigError
from tengri.observation import Observation
from tengri.observation.photometry import (
    FILTER_COVERAGE_TOLERANCE,
    FilterConvention,
    FilterCurve,
    filter_coverage_fraction,
)
from tengri.observation.photometry_config import Photometry
from tengri.parameters.priors import Gaussian
from tengri.utils.cosmology import luminosity_distance

pytestmark = pytest.mark.regression_bug

UM = 1.0e4  # Angstrom per micron
AXIS_END_UM = 160.0  # the truncated SSP's red edge (a BC03-type axis)
AXIS_MIN_AA = 0.0091 * UM
AXIS_MAX_AA = AXIS_END_UM * UM


def _hat(a_um, b_um, name="hat", n=4001):
    """Top-hat T = 1 on [a, b] um, geometric nodes, zero-transmission end nodes."""
    a, b = a_um * UM, b_um * UM
    wave = np.geomspace(a, b, n)
    return FilterCurve(
        wave=jnp.asarray(np.concatenate([[a * (1 - 1e-9)], wave, [b * (1 + 1e-9)]])),
        trans=jnp.asarray(np.concatenate([[0.0], np.ones(n), [0.0]])),
        name=name,
    )


def _uncovered_photon(a, b, x):
    """Analytic uncovered fraction of a 1/lambda-weighted top-hat [a, b] above x."""
    if x >= b:
        return 0.0
    if x <= a:
        return 1.0
    return float(np.log(b / x) / np.log(b / a))


def _uncovered_energy(a, b, x):
    """Analytic uncovered fraction of a 1/lambda**2-weighted top-hat above x."""
    if x >= b:
        return 0.0
    if x <= a:
        return 1.0
    return float((1 / x - 1 / b) / (1 / a - 1 / b))


def _frac(filt, z, convention=FilterConvention.BESSELL):
    return filter_coverage_fraction(filt.wave, filt.trans, AXIS_MIN_AA, AXIS_MAX_AA, z, convention)


# ── (a) the helper against analytic top-hats ────────────────────────────────

_BANDS = [
    # (a_um, b_um, z): the model axis ends at 160 (1 + z) um in the observed frame
    (100.0, 140.0, 0.0),
    (140.0, 180.0, 0.0),
    (200.0, 300.0, 0.0),
    (200.0, 300.0, 1.0),
    (280.0, 360.0, 1.0),
    (400.0, 500.0, 1.0),
]


@pytest.mark.parametrize("a_um, b_um, z", _BANDS)
@pytest.mark.parametrize(
    "convention, analytic",
    [(FilterConvention.BESSELL, _uncovered_photon), (FilterConvention.ENERGY, _uncovered_energy)],
)
def test_coverage_fraction_matches_analytic_tophat(a_um, b_um, z, convention, analytic):
    expected = analytic(a_um, b_um, AXIS_END_UM * (1.0 + z))
    got = _frac(_hat(a_um, b_um), z, convention)
    assert got == pytest.approx(expected, abs=1e-6)


def test_coverage_fraction_inside_is_exactly_zero_and_outside_exactly_one():
    assert _frac(_hat(100.0, 140.0), 0.0) == 0.0
    assert _frac(_hat(200.0, 300.0), 0.0) == 1.0
    assert _frac(_hat(0.001, 0.005), 0.0) == 1.0  # wholly blueward of the axis


def test_coverage_fraction_node_on_axis_edge():
    edge, lo = AXIS_MAX_AA, AXIS_MIN_AA
    flat = [1.0, 1.0, 1.0]
    # a filter ending exactly on the red edge is covered; one starting there is not
    assert filter_coverage_fraction([100 * UM, 130 * UM, edge], flat, lo, edge, 0.0) == 0.0
    assert filter_coverage_fraction([edge, 180 * UM, 200 * UM], flat, lo, edge, 0.0) == 1.0
    # the same at the blue edge
    assert filter_coverage_fraction([lo, 2 * lo, 3 * lo], flat, lo, edge, 0.0) == 0.0
    assert filter_coverage_fraction([lo / 3, lo / 2, lo], flat, lo, edge, 0.0) == 1.0


def test_coverage_fraction_zero_transmission_tails_do_not_count():
    # transmission lives on 100-140 um; zero-weight nodes run far past the axis end
    wave = np.array([20.0, 99.0, 100.0, 140.0, 141.0, 900.0]) * UM
    trans = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert filter_coverage_fraction(wave, trans, AXIS_MIN_AA, AXIS_MAX_AA, 0.0) == 0.0


def test_coverage_fraction_does_not_assume_a_monotonic_table():
    filt = _hat(140.0, 180.0, n=401)
    order = np.random.default_rng(0).permutation(filt.wave.shape[0])
    shuffled = filter_coverage_fraction(
        np.asarray(filt.wave)[order], np.asarray(filt.trans)[order], AXIS_MIN_AA, AXIS_MAX_AA, 0.0
    )
    assert shuffled == _frac(filt, 0.0)
    assert shuffled == pytest.approx(_uncovered_photon(140.0, 180.0, 160.0), abs=1e-6)


@pytest.mark.parametrize(
    "wave, trans",
    [([200 * UM], [1.0]), ([100 * UM, 300 * UM], [0.0, 0.0])],
    ids=["single_node", "zero_transmission"],
)
def test_coverage_fraction_filter_without_weight_has_nothing_to_lose(wave, trans):
    assert filter_coverage_fraction(wave, trans, AXIS_MIN_AA, AXIS_MAX_AA, 0.0) == 0.0


# ── model builders ──────────────────────────────────────────────────────────


@functools.cache
def _short_ssp():
    """The tracked SSP cut at 160 um: the red edge of a BC03-type axis."""
    ssp = tengri.load_ssp()
    keep = np.asarray(ssp.ssp_wave) <= AXIS_MAX_AA
    return ssp._replace(ssp_wave=ssp.ssp_wave[keep], ssp_flux=ssp.ssp_flux[..., keep])


_DUST = {
    "type": "two_component",
    "law_bc": "calzetti",
    "law_diff": "calzetti",
    "tau_bc": Fixed(0.0),
    "tau_diff": Fixed(0.3),
    "all_params": Fixed(DEFAULT),
}
_RADIO = {"sf": {"type": "bell2003"}, "agn": {"type": "none"}, "all_params": Fixed(DEFAULT)}
_TORUS = {
    "type": "composable",
    "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
    "torus": {"type": "simple", "all_params": Fixed(DEFAULT)},
    "blr": {"type": "none"},
    "nlr": {"type": "none"},
    "feii": {"type": "none"},
    "atten": {"type": "none"},
}
_MBB = {"type": "modified_blackbody", "all_params": Fixed(DEFAULT)}
_DL14 = {
    "type": "draine_li2014",
    "dust_qpah": Fixed(2.5),
    "dust_umin": Fixed(1.0),
    "dust_gamma_dl": Fixed(0.1),
    "dust_alpha_dl14": Fixed(2.0),
    "all_params": Fixed(DEFAULT),
}


def _build(filters, redshift=None, dust_attenuation=None, convention="bessell", **blocks):
    redshift = Fixed(0.5) if redshift is None else redshift
    return SEDModel.build(
        _short_ssp(),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_attenuation or {"type": "none"},
        neb={"type": "none"},
        redshift=redshift,
        observation=Observation(
            photometry=Photometry(filters=tuple(filters), convention=convention)
        ),
        **blocks,
    )


def _straddle(z, uncovered, a_um=100.0):
    """Top-hat from ``a_um (1+z)`` with ``uncovered`` of its 1/lambda weight beyond the axis."""
    x = AXIS_END_UM * (1.0 + z)
    a = a_um * (1.0 + z)
    b = (x / a**uncovered) ** (1.0 / (1.0 - uncovered))
    return _hat(a, b, name="straddling_band")


# ── (b) the build refuses; the message carries band, fraction, axis, redshift ─


@pytest.mark.parametrize("z", [0.0, 0.5, 1.0])
def test_build_refuses_a_band_straddling_the_axis_end(z):
    filt = _straddle(z, 0.4)
    with pytest.raises(ConfigError) as err:
        _build([filt], redshift=Fixed(z))
    msg = str(err.value)
    assert "straddling_band" in msg
    assert f"{_frac(filt, z):.4g}" in msg and f"{_frac(filt, z):.4g}".startswith("0.4")
    assert f"[0.0091, {AXIS_END_UM:.4g}] um" in msg
    assert f"z = {z:.4g}" in msg
    assert "#2620" in msg


# ── (c) control: covered bands build and match an independent integral ──────


@pytest.mark.parametrize("z", [0.0, 0.5])
@pytest.mark.parametrize("band_um", [(30.0, 100.0), (1.0, 3.0)])
def test_covered_band_builds_and_matches_numpy_integral(z, band_um):
    filt = _hat(*band_um, name="covered")
    model = _build([filt], redshift=Fixed(z))
    state = model.predict_state({})
    assert _numpy_flux(state, filt, z) == pytest.approx(
        float(model.predict_photometry({})[0]), rel=1e-6
    )


def _numpy_flux(state, filt, z):
    """Independent numpy band flux: trapezoid of L_nu T / lambda on the union grid."""
    wave = np.asarray(state.wave) * (1.0 + z)
    lnu = np.asarray(state.sed_intrinsic)
    fw, ft = np.asarray(filt.wave), np.asarray(filt.trans)
    grid = np.unique(np.concatenate([wave, fw]))
    weight = np.interp(grid, fw, ft, left=0.0, right=0.0) / grid
    num = np.trapezoid(np.interp(grid, wave, lnu, left=0.0, right=0.0) * weight, grid)
    mean_lnu = num / np.trapezoid(weight, grid)
    return mean_lnu * (1.0 + z) / (4.0 * np.pi * float(luminosity_distance(z)) ** 2)


# ── tolerance boundary: what an accepted band can lose ──────────────────────


@pytest.mark.parametrize("uncovered", [5e-4, 9e-4])
def test_band_just_inside_tolerance_builds_and_loses_exactly_its_fraction(uncovered):
    """A band 0 < f <= 1e-3 uncovered is accepted; its flux is the zero-filled integral.

    The zero-filled flux is ``A / W_tot`` and the covered-renormalized flux is
    ``A / W_cov``; their ratio is ``1 - f``. So the tolerance admits an
    under-estimate of at most 0.1 % (1.1 mmag), and the model equals the
    independent numpy integral to 1e-6.
    """
    assert uncovered <= FILTER_COVERAGE_TOLERANCE
    z = 0.0
    filt = _straddle(z, uncovered)
    assert _frac(filt, z) == pytest.approx(uncovered, rel=1e-3)
    model = _build([filt], redshift=Fixed(z))
    state = model.predict_state({})
    flux = float(model.predict_photometry({})[0])
    assert _numpy_flux(state, filt, z) == pytest.approx(flux, rel=1e-6)
    renormalized = flux / (1.0 - _frac(filt, z))
    assert 1.0 - flux / renormalized == pytest.approx(uncovered, rel=1e-3)
    assert 1.0 - flux / renormalized <= FILTER_COVERAGE_TOLERANCE


def test_band_just_outside_tolerance_is_refused():
    with pytest.raises(ConfigError):
        _build([_straddle(0.0, 1.5e-3)], redshift=Fixed(0.0))


# ── (e) components that extend the axis turn the refusal into a build ───────

_EXTENDS = [
    # id, blocks, band [um], refused, axis max [um]
    ("stellar", {}, (150.0, 250.0), True, 160.0),
    (
        "xray",
        {"xray": {"type": "simple", "all_params": Fixed(DEFAULT)}},
        (150.0, 250.0),
        True,
        160.0,
    ),
    ("torus_simple", {"agn": _TORUS}, (150.0, 250.0), True, 160.0),
    ("modified_blackbody", {"dust_emission": _MBB}, (150.0, 250.0), False, 1.0e4),
    ("draine_li2014", {"dust_emission": _DL14}, (150.0, 250.0), False, 1.0e4),
    ("mbb_band_past_1cm", {"dust_emission": _MBB}, (2.0e4, 3.0e4), True, 1.0e4),
    ("radio", {"radio": _RADIO}, (150.0, 250.0), False, 3.0e7),
    ("radio_band_past_1cm", {"radio": _RADIO}, (2.0e4, 3.0e4), False, 3.0e7),
]


@pytest.mark.parametrize(
    "blocks, band, refused, axis_max_um",
    [c[1:] for c in _EXTENDS],
    ids=[c[0] for c in _EXTENDS],
)
def test_axis_extension_by_component(blocks, band, refused, axis_max_um):
    filt = _hat(*band, name="far_ir")
    kwargs = {"dust_attenuation": _DUST, **blocks}
    if refused:
        with pytest.raises(ConfigError, match="far_ir"):
            _build([filt], redshift=Fixed(0.0), **kwargs)
        return
    model = _build([filt], redshift=Fixed(0.0), **kwargs)
    wave = np.asarray(model.predict_state({}).wave)
    assert wave.max() / UM == pytest.approx(axis_max_um, rel=1e-6)
    flux = float(model.predict_photometry({})[0])
    assert np.isfinite(flux) and flux > 0.0


@pytest.mark.parametrize(
    "blocks, axis_min_um",
    [({}, 0.0091), ({"xray": {"type": "simple", "all_params": Fixed(DEFAULT)}}, 4.13e-6)],
    ids=["stellar", "xray"],
)
def test_axis_blue_end_by_component(blocks, axis_min_um):
    model = _build([_hat(30.0, 100.0, name="covered")], redshift=Fixed(0.0), **blocks)
    wave = np.asarray(model.predict_state({}).wave)
    assert wave.min() / UM == pytest.approx(axis_min_um, rel=1e-3)


# ── (d) free redshift: both ends of the prior support ───────────────────────


@pytest.mark.parametrize(
    "band, prior, failing_z, end",
    [
        ((150.0, 250.0), Uniform(0.0, 1.0), 0, "lower"),  # covered at z = 1 only
        ((0.0100, 0.0120), Uniform(0.0, 1.0), 1, "upper"),  # covered at z = 0 only
        ((150.0, 250.0), Gaussian(0.5, 0.2, lo=0.0, hi=1.0), 0, "lower"),
        ((150.0, 250.0), Gaussian(0.5, 0.2, lo=0.0), 0, "lower"),
    ],
    ids=[
        "red_band_fails_low_z",
        "blue_band_fails_high_z",
        "truncated_gaussian",
        "half_open_gaussian",
    ],
)
def test_free_redshift_checks_both_ends_and_names_the_failing_redshift(
    band, prior, failing_z, end
):
    with pytest.raises(ConfigError) as err:
        _build([_hat(*band, name="moving_band")], redshift=prior)
    msg = str(err.value)
    assert "moving_band" in msg
    assert f"z = {failing_z} ({end} end of the redshift prior)" in msg


def test_free_redshift_covered_at_both_ends_builds():
    _build([_hat(100.0, 150.0, name="ok")], redshift=Uniform(0.0, 1.0))


def test_unbounded_prior_end_exposes_no_support_and_is_not_tested():
    # The half-open Gaussian has no upper bound, so only z = 0 is tested; the blue
    # band that fails at z = 1 is therefore not seen. The spec gives no support there.
    _build([_hat(0.0100, 0.0120, name="blue")], redshift=Gaussian(0.0, 0.2, lo=0.0))


# ── catalog / z-table / rebuilt models reach the same check ─────────────────


def test_catalog_z_range_checks_both_range_ends_even_for_a_fixed_redshift():
    filt = _hat(150.0, 250.0, name="catalog_band")
    _build([filt], redshift=Fixed(1.0))  # alone, the fixed z = 1 covers it
    with pytest.raises(ConfigError, match=r"z = 0 \(lower end of catalog_z_range\)"):
        _build([filt], redshift=Fixed(1.0), approx=WavePrecomp(catalog_z_range=(0.0, 1.5)))


def test_catalog_z_range_covered_at_both_ends_builds():
    _build([_hat(30.0, 100.0, name="ok")], approx=WavePrecomp(catalog_z_range=(0.0, 1.5)))


def test_wave_precomp_fixed_redshift_refuses():
    with pytest.raises(ConfigError, match="precomp_band"):
        _build(
            [_hat(150.0, 250.0, name="precomp_band")], redshift=Fixed(0.0), approx=WavePrecomp()
        )


def test_with_fixed_redshift_rebuild_hits_the_check():
    model = _build([_hat(150.0, 250.0, name="rebuilt_band")], redshift=Fixed(1.0))
    assert model.with_fixed_redshift(1.5) is not model
    with pytest.raises(ConfigError, match="rebuilt_band"):
        model.with_fixed_redshift(0.0)


def test_energy_convention_uses_the_observation_weight():
    """The check weights each band as the photometry does: 1/lambda**2 under ENERGY.

    A wide band uncovered by 1.3e-3 in photon counting is under 1e-3 in
    energy weighting (the energy weight favors the covered, bluer part).
    """
    filt = _straddle(0.0, 1.3e-3, a_um=10.0)
    assert _frac(filt, 0.0, FilterConvention.BESSELL) > FILTER_COVERAGE_TOLERANCE
    assert _frac(filt, 0.0, FilterConvention.ENERGY) < FILTER_COVERAGE_TOLERANCE
    with pytest.raises(ConfigError, match="straddling_band"):
        _build([filt], redshift=Fixed(0.0))
    _build([filt], redshift=Fixed(0.0), convention=FilterConvention.ENERGY)
