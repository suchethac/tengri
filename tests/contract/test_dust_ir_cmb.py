# SPDX-License-Identifier: BSD-3-Clause
"""Contract: opt-in CMB heating and contrast for the tabulated dust models (#2766).

``dust_emission={'type': ..., 'cmb': True}`` applies da Cunha et al. (2013),
ApJ, 766, 13 (arXiv:1302.0844) to a tabulated emission model:

* Eq. 12, the CMB-heated dust temperature, for the single-temperature libraries
  (``schreiber2016``, ``schreiber2018``), together with the luminosity boost the
  paper states in Sec. 2.2 (point ii);
* Eq. 18, the contrast against the CMB, for every tabulated model.

Every expected value below is written out in the test from the paper's formula
(a Planck function in numpy, Eq. 12 as a one-line power law), never read back
from the code under test. Default off must be bit-identical.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri._data_setup import find_data_str
from tengri.components.dust.emission._cmb import (
    CMB_FAR_IR_MIN_AA,
    CMB_TEMPLATE_BETA_IR,
    cmb_heating_luminosity_boost,
    template_far_ir_temperature,
)
from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.components.dust.emission._physics import planck_bnu
from tengri.components.sed_model_component import _REGISTRY
from tengri.forward.component_factory import _resolve_registry_component
from tengri.parameters.groups import parse_groups
from tengri.utils.physics_constants import AA_TO_CM, C_CGS, H_PLANCK, K_BOLTZ

pytestmark = pytest.mark.contract

#: The CMB temperature at z = 0 the package uses for every dust model [K].
T_CMB0 = 2.725
BETA = CMB_TEMPLATE_BETA_IR
UM = 1.0e4  # Angstrom per micron


def _paper_heated_temperature(t0, z, beta=BETA, t_cmb0=T_CMB0):
    """Eq. 12 of da Cunha et al. (2013), written out."""
    return (t0 ** (4 + beta) + t_cmb0 ** (4 + beta) * ((1 + z) ** (4 + beta) - 1)) ** (
        1.0 / (4 + beta)
    )


def _planck_nu(nu, t):
    """B_nu(T) [erg/s/cm2/Hz/sr] written as e^-x / (1 - e^-x)."""
    x = H_PLANCK * nu / (K_BOLTZ * t)
    return 2 * H_PLANCK * nu**3 / C_CGS**2 * np.exp(-x) / (-np.expm1(-x))


def _paper_contrast(wave_aa, t_dust_z, z, t_cmb0=T_CMB0):
    """Eq. 18: 1 - B_nu(T_CMB(z)) / B_nu(T_dust(z)), with a numpy Planck function."""
    nu = C_CGS / (np.asarray(wave_aa, dtype=float) * AA_TO_CM)
    x_cmb = H_PLANCK * nu / (K_BOLTZ * t_cmb0 * (1 + z))
    x_dust = H_PLANCK * nu / (K_BOLTZ * t_dust_z)
    # B(T_cmb)/B(T_dust) = (e^x_dust - 1)/(e^x_cmb - 1), formed so that neither
    # exponential overflows or underflows at 0.1 micron.
    ratio = np.exp(x_dust - x_cmb) * (-np.expm1(-x_dust)) / (-np.expm1(-x_cmb))
    return 1.0 - ratio


def _need(*files: str) -> None:
    missing = [f for f in files if find_data_str(f) is None]
    if missing:
        pytest.skip(f"template file(s) not available: {missing}")


def _component(name: str, *, cmb: bool):
    comp = _resolve_registry_component("dust_emission", name, config=None)
    comp.cmb = cmb
    return comp


@pytest.fixture(scope="module")
def wave():
    return jnp.geomspace(1.0e3, 3.0e7, 4000)  # 0.1 micron to 3 mm


def _emit(comp, params, wave, templates=None):
    sed, published = comp.predict(
        params,
        jnp.zeros_like(wave),
        wave,
        L_ir=1.0,
        **({"templates": templates} if templates else {}),
    )
    return sed, published


# ── The equations, pinned against the paper ────────────────────────────────


class TestPaperEquations:
    def test_cmb_temperature_at_z6_is_the_papers_19_1_K(self):
        # Sec. 2.1: "at z = 6, where the CMB temperature has increased from
        # 2.73 K to 19.1 K".
        assert round(T_CMB0 * (1 + 6.0), 1) == 19.1

    @pytest.mark.parametrize("t0", [18.0, 25.0, 35.0])
    @pytest.mark.parametrize("z", [0.0, 2.0, 4.0, 6.0, 10.0])
    def test_heated_temperature_is_eq_12(self, t0, z):
        from tengri.components.dust.emission._physics import cmb_corrected_temperature

        got = float(cmb_corrected_temperature(t0, z, BETA))
        assert got == pytest.approx(_paper_heated_temperature(t0, z), rel=1e-12)

    def test_heating_is_negligible_below_z4_for_18K_dust(self):
        # Fig. 1 / Sec. 2.1: for T_d(z=0) = 18 K "the effect of dust heating by
        # the CMB becomes non-negligible at z ~ 4". Fixed here as the paper's
        # own statement: < 0.5% at z = 2, several percent at z = 4, tens of
        # percent by z = 6.
        t0 = 18.0
        rise = {z: _paper_heated_temperature(t0, z, 2.0) / t0 - 1 for z in (2.0, 4.0, 6.0)}
        assert rise[2.0] < 0.005
        assert 0.02 < rise[4.0] < 0.05
        assert rise[6.0] > 0.10

    def test_z0_heating_is_the_identity(self):
        for t0 in (15.0, 35.0, 80.0):
            t_eff = _paper_heated_temperature(t0, 0.0)
            assert t_eff == pytest.approx(t0, rel=1e-14)
            boost = cmb_heating_luminosity_boost(jnp.asarray(t0), jnp.asarray(t_eff), BETA)
            assert float(boost) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("t0,z", [(35.0, 4.0), (25.0, 6.0), (18.0, 8.0)])
    def test_luminosity_boost_is_the_T_to_4_plus_beta_ratio(self, t0, z):
        # Sec. 2.2 (ii): L grows by [T_d(z)/T_d(0)]^(4+beta) because
        # int nu^beta B_nu(T) dnu is proportional to T^(4+beta) (Eq. 11).
        # Check the proportionality by quadrature rather than restating it.
        t_eff = _paper_heated_temperature(t0, z)
        w = np.geomspace(1.0e2, 1.0e9, 60000)  # Angstrom, wide enough for 4 < T < 100 K
        nu = C_CGS / (w * AA_TO_CM)

        def power(t):
            return -np.trapezoid(nu**BETA * _planck_nu(nu, t), nu)

        by_quadrature = power(t_eff) / power(t0)
        boost = float(cmb_heating_luminosity_boost(jnp.asarray(t0), jnp.asarray(t_eff), BETA))
        assert boost == pytest.approx(by_quadrature, rel=2e-4)
        assert boost == pytest.approx(
            1 + (T_CMB0 / t0) ** (4 + BETA) * ((1 + z) ** (4 + BETA) - 1), rel=1e-12
        )

    @pytest.mark.parametrize("t0,z", [(35.0, 4.0), (35.0, 6.0), (25.0, 4.0), (25.0, 6.0)])
    def test_contrast_is_eq_18_at_three_wavelengths(self, t0, z):
        from tengri.components.dust.emission._cmb import cmb_observed_emission

        lam = jnp.asarray([250.0 * UM, 500.0 * UM, 1000.0 * UM])
        t_z = _paper_heated_temperature(t0, z)
        got = np.asarray(cmb_observed_emission(lam, jnp.ones_like(lam), jnp.asarray(z), t_z))
        np.testing.assert_allclose(got, _paper_contrast(lam, t_z, z), rtol=1e-10)

    def test_contrast_limits(self):
        lam = np.asarray([10.0 * UM, 250.0 * UM, 1000.0 * UM, 5000.0 * UM])
        c = _paper_contrast(lam, 35.0, 4.0)
        assert np.all(np.diff(c) < 0) and np.all(c <= 1)  # worse at longer wavelength
        assert c[0] == pytest.approx(1.0, abs=1e-12)  # short wavelength: all detected
        # Dust at the CMB temperature: nothing detectable (Sec. 2.3).
        assert _paper_contrast(lam, T_CMB0 * 5.0, 4.0) == pytest.approx(0.0, abs=1e-12)


# ── The template temperature estimate ──────────────────────────────────────


class TestTemplateTemperature:
    @staticmethod
    def _mbb(wave, t, beta=BETA):
        nu = C_CGS / (wave * AA_TO_CM)
        return (nu / 1e12) ** beta * planck_bnu(wave, t)

    @pytest.mark.parametrize("t", [15.0, 25.0, 35.0, 60.0, 100.0])
    def test_recovers_the_temperature_of_a_modified_blackbody(self, wave, t):
        got = float(template_far_ir_temperature(wave, self._mbb(wave, t)))
        assert got == pytest.approx(t, rel=2e-3)

    def test_scale_free(self, wave):
        base = self._mbb(wave, 30.0)
        small = float(template_far_ir_temperature(wave, base * 1e-30))
        large = float(template_far_ir_temperature(wave, base * 1e30))
        assert small == pytest.approx(large, rel=1e-12)

    def test_emission_below_the_band_edge_is_ignored(self, wave):
        base = self._mbb(wave, 30.0)
        blob = jnp.exp(-0.5 * ((wave - 8.0 * UM) / (1.0 * UM)) ** 2) * jnp.max(base)
        assert jnp.all(wave[blob > 1e-12 * jnp.max(blob)] < CMB_FAR_IR_MIN_AA)
        with_blob = float(template_far_ir_temperature(wave, base + blob))
        assert with_blob == pytest.approx(
            float(template_far_ir_temperature(wave, base)), rel=1e-12
        )

    def test_no_far_ir_power_is_finite_and_neutral(self, wave):
        zero = float(template_far_ir_temperature(wave, jnp.zeros_like(wave)))
        assert np.isfinite(zero) and zero == pytest.approx(250.0)

    def test_gradient_is_finite(self, wave):
        g = jax.grad(lambda s: template_far_ir_temperature(wave, self._mbb(wave, 30.0) * s))(1.0)
        assert np.isfinite(float(g))


class TestFloat32:
    def test_estimate_and_contrast_stay_finite_and_close_in_float32(self, wave):
        from tengri.components.dust.emission._cmb import cmb_observed_emission

        def run(x64):
            with jax.enable_x64(x64):
                dtype = jnp.float64 if x64 else jnp.float32
                w = wave.astype(dtype)
                nu = C_CGS / (w * AA_TO_CM)
                lnu = (
                    (nu / 1e12) ** BETA * planck_bnu(w, 30.0) * 1e30
                )  # an erg/s/Hz-scale template
                t = template_far_ir_temperature(w, lnu)
                out = cmb_observed_emission(w, lnu, jnp.asarray(5.0, dtype), t)
                return np.asarray(t, dtype=float), np.asarray(out, dtype=float)

        t64, out64 = run(True)
        t32, out32 = run(False)
        assert np.isfinite(out32).all() and np.isfinite(t32)
        assert t32 == pytest.approx(t64, rel=1e-3)
        keep = out64 > 1e-6 * out64.max()
        np.testing.assert_allclose(out32[keep], out64[keep], rtol=2e-3)


# ── Components ─────────────────────────────────────────────────────────────

ALL_TABULATED = (
    "astrodust",
    "bosa",
    "dale2014",
    "dale2014_cigale",
    "dh02_ce01",
    "draine_li2007",
    "draine_li2014",
    "schreiber2016",
    "schreiber2018",
    "themis",
)


def test_exactly_the_tabulated_models_offer_cmb():
    offered = {
        n
        for n, c in _REGISTRY.items()
        if isinstance(c, type) and issubclass(c, EmissionComponent) and c.cmb_supported
    }
    assert offered == set(ALL_TABULATED)
    # the analytic models apply it unconditionally and must not offer a second switch
    for name in ("modified_blackbody", "graybody", "casey2012"):
        assert not _REGISTRY[name].cmb_supported


class TestSchreiber:
    """Single-temperature library: heating (Eq. 12) and contrast (Eq. 18), both exact."""

    P: ClassVar[dict] = {"T": 35.0, "f_pah": 0.05}

    @pytest.fixture(scope="class")
    def lib(self):
        _need("schreiber2016_templates.h5")
        comp = _component("schreiber2016", cmb=False)
        return comp.load(None)

    def _run(self, wave, lib, *, cmb, z, t=35.0, name="schreiber2016"):
        comp = _component(name, cmb=cmb)
        return _emit(comp, {**self.P, "T": t, "redshift": z}, wave, lib)[0]

    @pytest.mark.parametrize("z", [4.0, 6.0])
    def test_ratio_to_the_heated_uncorrected_sed_is_boost_times_eq_18(self, wave, lib, z):
        on = self._run(wave, lib, cmb=True, z=z)
        t_eff = _paper_heated_temperature(35.0, z)
        off_at_heated_T = self._run(wave, lib, cmb=False, z=z, t=t_eff)
        boost = 1 + (T_CMB0 / 35.0) ** (4 + BETA) * ((1 + z) ** (4 + BETA) - 1)
        for lam in (250.0 * UM, 500.0 * UM, 1000.0 * UM):
            i = int(jnp.argmin(jnp.abs(wave - lam)))
            expected = boost * float(_paper_contrast(wave[i], t_eff, z))
            assert float(on[i] / off_at_heated_T[i]) == pytest.approx(expected, rel=1e-12)

    def test_contrast_matches_the_issue_table_for_a_35K_galaxy_at_z4(self, wave, lib):
        # #2766: contrast 0.938 / 0.825 / 0.729 at rest 250 / 500 / 1000 micron.
        z = 4.0
        on = self._run(wave, lib, cmb=True, z=z)
        t_eff = _paper_heated_temperature(35.0, z)
        off_at_heated_T = self._run(wave, lib, cmb=False, z=z, t=t_eff)
        boost = (t_eff / 35.0) ** (4 + BETA)
        got = []
        for lam in (250.0 * UM, 500.0 * UM, 1000.0 * UM):
            i = int(jnp.argmin(jnp.abs(wave - lam)))
            got.append(float(on[i] / off_at_heated_T[i]) / boost)
        np.testing.assert_allclose(got, [0.938, 0.825, 0.729], atol=6e-4)

    def test_z0_heating_is_the_identity_and_only_the_contrast_remains(self, wave, lib):
        on = self._run(wave, lib, cmb=True, z=0.0)
        off = self._run(wave, lib, cmb=False, z=0.0)
        expected = off * _paper_contrast(np.asarray(wave), 35.0, 0.0)
        np.testing.assert_allclose(np.asarray(on), np.asarray(expected), rtol=1e-12, atol=0.0)

    def test_schreiber2018_takes_the_same_path(self, wave):
        _need("schreiber2018_templates.h5")
        lib = _component("schreiber2018", cmb=False).load(None)
        on = _emit(
            _component("schreiber2018", cmb=True),
            {"T": 30.0, "f_pah": 0.05, "redshift": 5.0},
            wave,
            lib,
        )[0]
        t_eff = _paper_heated_temperature(30.0, 5.0)
        off = _emit(
            _component("schreiber2018", cmb=False),
            {"T": t_eff, "f_pah": 0.05, "redshift": 5.0},
            wave,
            lib,
        )[0]
        i = int(jnp.argmin(jnp.abs(wave - 500.0 * UM)))
        boost = (t_eff / 30.0) ** (4 + BETA)
        assert float(on[i] / off[i]) == pytest.approx(
            boost * float(_paper_contrast(wave[i], t_eff, 5.0)), rel=1e-12
        )

    def test_observed_power_stays_close_to_the_starlight_budget(self, wave, lib):
        # The point of the luminosity boost (Sec. 2.2 ii): the CMB energy the dust
        # absorbs is the energy the contrast removes again, so what is detected
        # against the CMB stays near the unit starlight budget. Without the boost
        # the detected power would fall short by the full contrast loss.
        from tengri.components.dust.emission._physics import integrate_lnu_over_nu

        on = self._run(wave, lib, cmb=True, z=4.0)
        detected = float(integrate_lnu_over_nu(on, wave))
        t_eff = _paper_heated_temperature(35.0, 4.0)
        unboosted = detected / (t_eff / 35.0) ** (4 + BETA)
        assert abs(detected - 1.0) < abs(unboosted - 1.0)
        assert detected == pytest.approx(1.0, abs=0.02)


class TestRadiationFieldLibraries:
    """No dust temperature: contrast only, at the heated template temperature."""

    CASES: ClassVar[dict] = {
        "draine_li2007": ("dl07_templates.h5", {"umin": 2.0, "gamma_dl": 0.05, "qpah": 2.5}),
        "dale2014": ("dale2014_templates.h5", {"alpha_dale": 2.0}),
    }

    @pytest.fixture(params=sorted(CASES))
    def case(self, request, wave):
        fname, params = self.CASES[request.param]
        _need(fname)
        comp = _component(request.param, cmb=False)
        lib = comp.load(None) if comp.accepts_threaded_templates else None
        return request.param, params, lib

    @pytest.mark.parametrize("z", [4.0, 6.0])
    def test_spectrum_is_scaled_by_eq_18_at_the_heated_template_temperature(self, wave, case, z):
        name, params, lib = case
        p = {**params, "redshift": z}
        off = _emit(_component(name, cmb=False), p, wave, lib)[0]
        on = _emit(_component(name, cmb=True), p, wave, lib)[0]
        t_template = float(template_far_ir_temperature(wave, off))
        t_z = _paper_heated_temperature(t_template, z)
        ratio = np.asarray(on / jnp.where(off > 0, off, 1.0))
        expected = _paper_contrast(np.asarray(wave), t_z, z)
        sel = np.asarray(off > 1e-30 * float(jnp.max(off)))
        np.testing.assert_allclose(ratio[sel], expected[sel], rtol=1e-10)
        assert 5.0 < t_template < 60.0  # a physical cold-dust temperature

    def test_the_spectrum_is_not_otherwise_reshaped_and_loses_power(self, wave, case):
        from tengri.components.dust.emission._physics import integrate_lnu_over_nu

        name, params, lib = case
        p = {**params, "redshift": 5.0}
        off = _emit(_component(name, cmb=False), p, wave, lib)[0]
        on = _emit(_component(name, cmb=True), p, wave, lib)[0]
        assert bool(jnp.all(on <= off * (1 + 1e-12)))
        # mid-IR (PAH) emission is untouched: the CMB cannot hide a 10 micron photon
        i = int(jnp.argmin(jnp.abs(wave - 10.0 * UM)))
        assert float(on[i] / off[i]) == pytest.approx(1.0, abs=1e-12)
        assert float(integrate_lnu_over_nu(on, wave)) < float(integrate_lnu_over_nu(off, wave))

    def test_default_off_is_bit_identical_to_the_unwrapped_predict(self, wave, case):
        name, params, lib = case
        comp = _component(name, cmb=False)
        p = {**params, "redshift": 5.0}
        wrapped = _emit(comp, p, wave, lib)
        raw = type(comp).predict.__wrapped__(
            comp, p, jnp.zeros_like(wave), wave, L_ir=1.0, **({"templates": lib} if lib else {})
        )
        assert jnp.array_equal(wrapped[0], raw[0])


class TestJitAndGradients:
    @pytest.fixture(scope="class")
    def schreiber(self):
        _need("schreiber2016_templates.h5")
        return _component("schreiber2016", cmb=True).load(None)

    def test_schreiber_jit_matches_eager_and_grad_wrt_redshift_is_finite_nonzero(
        self, wave, schreiber
    ):
        comp = _component("schreiber2016", cmb=True)

        def flux_500(z, t):
            sed, _ = comp.predict(
                {"T": t, "f_pah": 0.05, "redshift": z},
                jnp.zeros_like(wave),
                wave,
                L_ir=1.0,
                templates=schreiber,
            )
            return jnp.interp(500.0 * UM, wave, sed)

        eager = flux_500(4.0, 35.0)
        assert float(jax.jit(flux_500)(4.0, 35.0)) == pytest.approx(float(eager), rel=1e-12)
        g_z, g_t = jax.grad(flux_500, argnums=(0, 1))(4.0, 35.0)
        assert np.isfinite(float(g_z)) and float(g_z) != 0.0
        assert np.isfinite(float(g_t)) and float(g_t) != 0.0

    def test_radiation_field_grad_wrt_redshift_is_finite_nonzero(self, wave):
        _need("dl07_templates.h5")
        comp = _component("draine_li2007", cmb=True)
        lib = comp.load(None)

        def flux_500(z):
            sed, _ = comp.predict(
                {"umin": 2.0, "gamma_dl": 0.05, "qpah": 2.5, "redshift": z},
                jnp.zeros_like(wave),
                wave,
                L_ir=1.0,
                templates=lib,
            )
            return jnp.interp(500.0 * UM, wave, sed)

        g = float(jax.grad(flux_500)(4.0))
        assert np.isfinite(g) and g != 0.0

    def test_missing_redshift_raises_instead_of_assuming_zero(self, wave, schreiber):
        comp = _component("schreiber2016", cmb=True)
        with pytest.raises(KeyError, match="redshift"):
            comp.predict(
                {"T": 35.0, "f_pah": 0.05},
                jnp.zeros_like(wave),
                wave,
                L_ir=1.0,
                templates=schreiber,
            )


# ── Grammar ────────────────────────────────────────────────────────────────


class TestGrammar:
    @staticmethod
    def _spec(**emission):
        return parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_emission={"all_params": Fixed(DEFAULT), **emission},
            redshift=Fixed(4.0),
        )

    def test_default_is_off_and_the_key_is_not_emitted(self):
        spec = self._spec(type="schreiber2016")
        assert spec.dust_ir_cmb is False
        assert "cmb" not in spec.to_groups()["dust_emission"]

    def test_on_round_trips(self):
        spec = self._spec(type="schreiber2016", cmb=True)
        assert spec.dust_ir_cmb is True
        assert spec.to_groups()["dust_emission"]["cmb"] is True
        assert parse_groups(**spec.to_groups()).dust_ir_cmb is True

    @pytest.mark.parametrize("name", ["modified_blackbody", "graybody", "casey2012"])
    def test_analytic_models_refuse_it_because_they_already_apply_it(self, name):
        with pytest.raises(ValueError, match="no CMB option"):
            self._spec(type=name, cmb=True)

    def test_off_is_accepted_everywhere(self):
        assert self._spec(type="modified_blackbody", cmb=False).dust_ir_cmb is False

    def test_requires_a_type(self):
        with pytest.raises(ValueError, match="requires a dust_emission type"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_emission={"cmb": True},
                redshift=Fixed(4.0),
            )

    def test_must_be_a_bool(self):
        with pytest.raises(ValueError, match="True or False"):
            self._spec(type="schreiber2016", cmb="yes")


# ── Through the model: default-off identity and WavePrecomp parity ─────────

IR_BANDS = ("herschel_70", "herschel_160", "herschel_250", "herschel_350", "herschel_500")


def _ir_filters():
    from tengri.observation.filters import load_filter_set

    return tuple(load_filter_set(list(IR_BANDS))[2])


def _build(ssp, emission, *, redshift, approx=None):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry(filters=_ir_filters())),
        approx=approx,
        dust_attenuation={
            "type": "two_component",
            "law_bc": "power_law",
            "law_diff": "power_law",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(2.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission=emission,
        neb={"type": "none"},
        redshift=redshift,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(0.8),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
    )


@pytest.fixture(scope="module")
def schreiber_available():
    _need("schreiber2016_templates.h5")


class TestThroughTheModel:
    def test_default_off_and_explicit_false_are_bit_identical(
        self, synthetic_ssp_wide, schreiber_available
    ):
        emission = {"type": "schreiber2016", "all_params": Fixed(DEFAULT)}
        absent = _build(synthetic_ssp_wide, emission, redshift=Fixed(4.0))
        off = _build(synthetic_ssp_wide, {**emission, "cmb": False}, redshift=Fixed(4.0))
        a = np.asarray(absent.predict_photometry({}))
        b = np.asarray(off.predict_photometry({}))
        assert np.array_equal(a, b)

    def test_on_lowers_the_submillimeter_bands_and_keeps_the_mid_ir(
        self, synthetic_ssp_wide, schreiber_available
    ):
        emission = {"type": "schreiber2016", "all_params": Fixed(DEFAULT)}
        off = np.asarray(
            _build(synthetic_ssp_wide, emission, redshift=Fixed(4.0)).predict_photometry({})
        )
        on = np.asarray(
            _build(
                synthetic_ssp_wide, {**emission, "cmb": True}, redshift=Fixed(4.0)
            ).predict_photometry({})
        )
        ratio = on / off
        assert np.all(np.isfinite(ratio))
        # observed 70 / 160 micron sit at rest 14 / 32 micron: untouched by the CMB
        # (the heated template is slightly brighter there, by the luminosity boost)
        assert ratio[0] == pytest.approx(1.0, abs=0.02)
        # observed 500 micron is rest 100 micron: suppressed relative to 250 micron
        assert ratio[4] < ratio[2] <= ratio[1] * 1.001

    @pytest.mark.parametrize(
        "redshift,name",
        [
            (Fixed(4.0), "fixed z"),
            (Fixed(5.3), "fixed z, second value"),
        ],
    )
    def test_wave_precomp_matches_the_exact_path(
        self, synthetic_ssp_wide, schreiber_available, redshift, name
    ):
        emission = {"type": "schreiber2016", "all_params": Fixed(DEFAULT), "cmb": True}
        exact = _build(synthetic_ssp_wide, emission, redshift=redshift)
        lut = _build(synthetic_ssp_wide, emission, redshift=redshift, approx=WavePrecomp())
        pe = np.asarray(exact.predict_photometry({}))
        pl = np.asarray(lut.predict_photometry({}))
        np.testing.assert_allclose(pl, pe, rtol=5e-3, err_msg=name)

    def test_wave_precomp_follows_a_free_redshift(self, synthetic_ssp_wide, schreiber_available):
        # The correction makes the template depend on z, so the per-filter band
        # response must be tabulated over redshift and read at the evaluation
        # redshift: a LUT built at one z would be flat across the range.
        emission = {"type": "schreiber2016", "all_params": Fixed(DEFAULT), "cmb": True}
        redshift = Uniform(3.0, 6.0)
        exact = _build(synthetic_ssp_wide, emission, redshift=redshift)
        lut = _build(synthetic_ssp_wide, emission, redshift=redshift, approx=WavePrecomp())
        by_z = {}
        for z in (3.3, 4.4, 5.6):
            pe = np.asarray(exact.predict_photometry({"redshift": z}))
            pl = np.asarray(lut.predict_photometry({"redshift": z}))
            np.testing.assert_allclose(pl, pe, rtol=5e-3, err_msg=f"z={z}")
            by_z[z] = pl
        # the LUT is not flat in z: the 500 micron band moves by far more than the tolerance
        assert abs(by_z[5.6][4] / by_z[3.3][4] - 1.0) > 0.05

    def test_photometry_gradient_wrt_redshift_is_finite_through_the_lut(
        self, synthetic_ssp_wide, schreiber_available
    ):
        emission = {"type": "schreiber2016", "all_params": Fixed(DEFAULT), "cmb": True}
        lut = _build(
            synthetic_ssp_wide, emission, redshift=Uniform(3.0, 6.0), approx=WavePrecomp()
        )
        g = jax.grad(lambda z: lut.predict_photometry({"redshift": z})[4])(4.4)
        assert np.isfinite(float(g)) and float(g) != 0.0


@pytest.mark.parametrize("name", ["modified_blackbody", "graybody", "casey2012"])
def test_direct_parameters_path_refuses_cmb_on_an_analytic_type(synthetic_ssp_wide, name):
    # Bypassing the grammar (flat Parameters kwargs) must not double-apply the CMB.
    from tengri.parameters.parameters import Parameters

    spec = Parameters(
        dust_emission=name,
        dust_model="two_component",
        dust_ir_cmb=True,
        redshift=Fixed(4.0),
    )
    with pytest.raises(ValueError, match="refusing to apply it twice"):
        SEDModel(spec, ssp_data=synthetic_ssp_wide)
