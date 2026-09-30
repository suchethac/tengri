# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for Lyα escape cancellation (#2531) and Q_H unit contract (#2532).

Physics validation:
- Lyα must scale with the same escape/dust factor as every other recombination line,
  for every nebular backend, dust-attenuation geometry, and escape/dust budget.
- Q_H takes L☉/Hz input (not erg/s/Hz), with L_SUN offset added in log space for
  float32 safety.
"""


# Inline helpers for model building

import numpy as np
import pytest
from jax import numpy as jnp

from tengri import DEFAULT, Fixed, SEDModel
from tengri._data_setup import find_data


def _get_ssp():
    return __import__("tengri").load_ssp("fsps_prsc_miles_chabrier", download=False)


def _neb_group(backend, fesc=None, fdust_frac=None, grid_path=None):
    g = {"type": backend, "all_params": Fixed(DEFAULT)}
    if backend in ("cue", "cloudy", "cb19", "mappings"):
        if fesc is not None:
            g["fesc"] = Fixed(fesc)
        if fdust_frac is not None:
            g["fdust_frac"] = Fixed(fdust_frac)
    if backend == "cloudy":
        g["grid"] = str(grid_path)
    return g


def _dust_atten_group(kind, law="calzetti", tau_v=0.5, tau_bc=0.5, tau_diff=0.3):
    """Build a ``dust_attenuation`` group dict for ``kind``.

    ``kind`` is one of "none" (no screen), "single_component" (uniform
    foreground screen), or "two_component" (birth-cloud + diffuse-ISM
    geometry). All three are real, distinct structural choices in the
    public dict grammar (`docs/dev/model-construction.md`) — see
    ``TestLyaEscapeCancellation.test_dust_kind_changes_attenuated_continuum``
    for the guard that this function is not accidentally a no-op.
    """
    if kind == "none":
        return {"type": "none"}
    if kind == "single_component":
        return {
            "type": "single_component",
            "law": law,
            "tau_v": Fixed(tau_v),
            "other_params": Fixed(DEFAULT),
        }
    if kind == "two_component":
        return {
            "type": "two_component",
            "law": law,
            "tau_bc": Fixed(tau_bc),
            "tau_diff": Fixed(tau_diff),
            "other_params": Fixed(DEFAULT),
        }
    raise ValueError(f"unknown dust_kind {kind!r}")


def build_model(neb_backend="cloudy", neb_fesc=None, neb_fdust_frac=None, dust_kind="none"):
    """Build a minimal SEDModel for the Lyα / Q_H regression checks.

    Raises on any build failure except a missing CLOUDY grid file. A missing
    grid is a checkout/data-availability gap (#2011), not a code regression:
    catching every exception and relabeling it as a skip would hide real
    defects in the model-construction path behind "not applicable here".
    """
    grid_path = None
    if neb_backend == "cloudy":
        grid_path = find_data("cloudy_grid_mist.h5")
        if grid_path is None:
            pytest.skip("CLOUDY grid not shipped in this checkout (#2011)")

    ssp = _get_ssp()
    kwargs = dict(
        sfh={
            "type": "const",
            "log_total_mass": Fixed(9.0),
            "start_gyr": Fixed(0.05),
            "end_gyr": Fixed(0.0),
            "other_params": Fixed(DEFAULT),
        },
        met={"type": "delta", "all_params": Fixed(DEFAULT)},
        neb=_neb_group(neb_backend, neb_fesc, neb_fdust_frac, grid_path=grid_path),
        dust_attenuation=_dust_atten_group(dust_kind),
        redshift=Fixed(0.1),
    )
    spec = __import__("tengri").parse_groups(**kwargs)
    return SEDModel(spec, ssp, observation=None)


from tengri.components.nebular._shared import (
    apply_lya_escape,
    compute_qh,
    compute_qh_log10,
)
from tengri.utils.physics_constants import C_CGS, H_PLANCK, L_SUN as LSUN_ERG_S

# Lyman limit [Angstrom] — same value as tengri.components.nebular._constants._LYMAN_LIMIT.
LYMAN_LIMIT_AA = 911.76

pytestmark = pytest.mark.bounds

# Dust-attenuation geometries and escape/dust budgets swept by every invariance
# test below (#2531 blast radius: the buggy division canceled the general
# suppression regardless of backend or dust geometry, so the sweep must cover
# both, not just the single reproducer cell).
_DUST_KINDS = ["none", "single_component", "two_component"]
_ESCAPE_DUST_CELLS = [
    (0.0, 0.0),
    (0.5, 0.0),
    (0.99, 0.0),
    (0.0, 0.5),
    (0.5, 0.3),
    (0.3, 0.69),
]


class TestLyaEscapeCancellation:
    """Test suite for issue #2531: Lyα escape factor correction."""

    @pytest.fixture(scope="class")
    def ssp(self, ssp_data_fsps):
        return ssp_data_fsps

    @pytest.mark.parametrize("backend", ["cue", "cloudy"])
    @pytest.mark.parametrize("dust_kind", _DUST_KINDS)
    @pytest.mark.parametrize("fesc,fdust", _ESCAPE_DUST_CELLS)
    def test_lya_hbeta_ratio_invariant_at_neb_fesc_lya_zero(self, backend, dust_kind, fesc, fdust):
        """With neb_fesc_lya=0, Lyα/Hβ ratio must be invariant across escape/dust budgets.

        This is the primary reproducer from #2531: the buggy code multiplies Lyα by
        (1 - neb_fesc_lya) / k_factor, canceling the general suppression (measured fail
        before fix: 25.8 -> 67 -> 4103 for Cloudy). Fixed code: ratio invariant to
        rtol 1e-6, for every (backend, dust_kind) pair swept here, not only the
        dust-free cell the original reproducer used.
        """
        model = build_model(
            neb_backend=backend, neb_fesc=fesc, neb_fdust_frac=fdust, dust_kind=dust_kind
        )
        fluxes = model.predict_line_fluxes({}, target_wavelengths=[1215.67, 4862.68])
        lya, hbeta = fluxes

        # Reference ratio from the (fesc=0, fdust=0) cell, same backend/dust_kind.
        ref_model = build_model(
            neb_backend=backend, neb_fesc=0.0, neb_fdust_frac=0.0, dust_kind=dust_kind
        )
        ref_fluxes = ref_model.predict_line_fluxes({}, target_wavelengths=[1215.67, 4862.68])
        ref_lya, ref_hbeta = ref_fluxes
        ref_ratio = ref_lya / ref_hbeta

        # At neb_fesc_lya=0, ratio must be INVARIANT (not match a fixed value).
        lya_hbeta_ratio = lya / hbeta
        np.testing.assert_allclose(
            lya_hbeta_ratio,
            ref_ratio,
            rtol=1e-6,
            err_msg=(
                f"{backend}/{dust_kind}/fesc={fesc}/fdust={fdust}: "
                f"Lyα/Hβ = {lya_hbeta_ratio:.6f} deviates from reference "
                f"{ref_ratio:.6f} (rtol 1e-6 violated)"
            ),
        )

        # Case-B window [15, 40] only holds for the intrinsic (dust-free) ratio:
        # a wavelength-dependent dust screen moves Lyα/Hβ away from the pure
        # recombination-cascade value by construction, so this check applies
        # only to dust_kind="none".
        if dust_kind == "none":
            assert 15 <= ref_ratio <= 40, (
                f"{backend}: reference Lyα/Hβ = {ref_ratio:.6f} outside Case-B window [15, 40]"
            )

    @pytest.mark.parametrize("backend", ["cue", "cloudy"])
    @pytest.mark.parametrize("dust_kind", _DUST_KINDS)
    @pytest.mark.parametrize("fesc,fdust", _ESCAPE_DUST_CELLS)
    def test_halpha_hbeta_ratio_invariant_zero_fesc_lya(self, backend, dust_kind, fesc, fdust):
        """Hα/Hβ ratio invariant across escape/dust budgets (non-Lyα baseline)."""
        model = build_model(
            neb_backend=backend, neb_fesc=fesc, neb_fdust_frac=fdust, dust_kind=dust_kind
        )
        fluxes = model.predict_line_fluxes({}, target_wavelengths=[6562.79, 4862.68])
        halpha, hbeta = fluxes

        ref_model = build_model(
            neb_backend=backend, neb_fesc=0.0, neb_fdust_frac=0.0, dust_kind=dust_kind
        )
        ref_fluxes = ref_model.predict_line_fluxes({}, target_wavelengths=[6562.79, 4862.68])
        ref_halpha, ref_hbeta = ref_fluxes
        ref_ratio = ref_halpha / ref_hbeta

        # Hα/Hβ must be INVARIANT across escape/dust budgets
        halpha_hbeta_ratio = halpha / hbeta
        np.testing.assert_allclose(
            halpha_hbeta_ratio,
            ref_ratio,
            rtol=1e-6,
            err_msg=(
                f"{backend}/{dust_kind}/fesc={fesc}/fdust={fdust}: "
                f"Hα/Hβ = {halpha_hbeta_ratio:.6f} deviates from reference "
                f"{ref_ratio:.6f} (rtol 1e-6 violated)"
            ),
        )

    def test_dust_kind_changes_attenuated_continuum(self):
        """``dust_kind`` must be a real structural choice, not a no-op.

        Regression guard: the Lyα/Hβ invariance tests above are blind to a
        dust_attenuation dict grammar that silently fails to attenuate — an
        invariant ratio across dust_kind values would look identical whether
        or not the dust screen actually engaged. Confirm each dusty geometry
        produces a genuinely different rest-frame continuum from the
        dust-free build, and from each other, at one representative UV
        wavelength (2000 A, inside every screen law's coverage).
        """
        wave_probe = np.array([2000.0])

        model_none = build_model(neb_backend="cue", dust_kind="none")
        sed_none = float(model_none.predict({}).rest_sed(wave_probe)[0])

        seds = {"none": sed_none}
        for kind in ("single_component", "two_component"):
            model_dusty = build_model(neb_backend="cue", dust_kind=kind)
            sed_dusty = float(model_dusty.predict({}).rest_sed(wave_probe)[0])
            seds[kind] = sed_dusty
            assert not np.isclose(sed_dusty, sed_none, rtol=1e-3), (
                f"dust_kind={kind!r} did not change the attenuated continuum at "
                f"2000 A relative to dust_kind='none' (none={sed_none:.6e}, "
                f"{kind}={sed_dusty:.6e}): the dict grammar built for this "
                "dust_kind is a no-op."
            )

        # The two dusty geometries are physically distinct (uniform screen vs.
        # birth-cloud + diffuse-ISM) and must not collapse onto each other.
        assert not np.isclose(seds["single_component"], seds["two_component"], rtol=1e-3), (
            "single_component and two_component produced the same attenuated "
            f"continuum at 2000 A ({seds['single_component']:.6e} vs "
            f"{seds['two_component']:.6e}): the two dust_kind geometries are "
            "not actually distinct in the built model."
        )

    def test_apply_lya_escape_helper(self):
        """Unit test of the shared apply_lya_escape helper on synthetic lines.

        Verifies that the helper correctly identifies the Lyα row and applies
        the (1 - neb_fesc_lya) factor without the division.
        """
        # Synthetic line array: 5 lines at 912 (LyC edge), 1215.67 (Lyα), 4862 (Hβ),
        # 5007 (O III), 6564 (Hα) [Angstrom]
        wavelengths = jnp.array([912.0, 1215.67, 4862.68, 5006.84, 6563.77])
        line_lum = jnp.array([1.0, 23.3, 1.0, 2.5, 1.5])  # arbitrary units

        neb_fesc_lya = 0.3

        result = apply_lya_escape(line_lum, wavelengths, neb_fesc_lya)

        # Lyα (index 1) should be multiplied by (1 - 0.3) = 0.7
        expected = jnp.array([1.0, 23.3 * (1.0 - neb_fesc_lya), 1.0, 2.5, 1.5])

        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_apply_lya_escape_no_mutation(self):
        """Verify that apply_lya_escape does not mutate the input array."""
        wavelengths = jnp.array([912.0, 1215.67, 4862.68])
        line_lum = jnp.array([1.0, 23.3, 1.0])
        line_lum_orig = jnp.copy(line_lum)

        _ = apply_lya_escape(line_lum, wavelengths, 0.5)

        assert jnp.allclose(line_lum, line_lum_orig)


class TestComputeQhUnitContract:
    """Test suite for issue #2532: compute_qh unit contract and internal constants.

    The contract: compute_qh_log10 takes L☉/Hz as input (per M☉), not erg/s/Hz.
    The L_SUN conversion is added in log space to avoid float32 overflow.
    """

    @pytest.fixture(scope="class")
    def ssp(self, ssp_data_fsps):
        return ssp_data_fsps

    @pytest.fixture(scope="class")
    def solar_zero_age_node(self, ssp):
        """Locate the solar-metallicity, ~1 Myr node BY VALUE.

        ``ssp.ssp_flux`` has shape ``(n_met=15, n_age=93, n_wave)``. Indexing
        by a hardcoded ``[0, 9, :]`` silently selects the wrong physical node
        if the grid's axis order or spacing ever changes — that indexing
        actually resolved to lgmet=-4.000 at 0.8913 Myr, not solar, which is
        exactly the axis-swap failure mode this fixture is built to catch
        loudly instead: it asserts the selected node's physical coordinates.
        """
        lgmet = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
        lg_age_gyr = np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64)

        i_met = int(np.argmin(np.abs(lgmet - np.log10(0.0142))))
        i_age = int(np.argmin(np.abs(lg_age_gyr - (-3.0))))  # -3.0 dex Gyr = 1 Myr

        assert abs(lgmet[i_met] - (-1.854)) < 0.01, (
            f"Selected metallicity node lgmet={lgmet[i_met]:.4f} is not solar "
            "(-1.854 +/- 0.01): axis swap or grid change suspected."
        )
        age_myr = 10 ** lg_age_gyr[i_age] * 1000.0
        assert abs(age_myr - 1.0) / 1.0 < 0.01, (
            f"Selected age node = {age_myr:.4f} Myr is not within 1% of 1 Myr: "
            "axis swap or grid change suspected."
        )
        return i_met, i_age

    @staticmethod
    def _independent_log10_qh(wave, flux_lsun_hz):
        """Q_H = integral_{lambda < 911.76 A} L_nu / (h nu) d nu, float64 numpy.

        Computed independently of ``compute_qh_log10`` / ``compute_qh`` — not
        read off the code under test — as a direct numpy trapezoid quadrature
        of the same physical integral, matching the Lyman-continuum photon
        rate definition (e.g. Osterbrock & Ferland 2006, Ch. 2).
        """
        wave64 = np.asarray(wave, dtype=np.float64)
        flux64 = np.asarray(flux_lsun_hz, dtype=np.float64)
        nu = C_CGS / (wave64 * 1e-8)  # Hz
        mask = wave64 < LYMAN_LIMIT_AA
        l_nu_erg_s_hz = flux64 * float(LSUN_ERG_S)  # L_sun/Hz -> erg/s/Hz
        integrand = np.where(mask, l_nu_erg_s_hz / (H_PLANCK * nu), 0.0)  # photons/s/Hz
        order = np.argsort(nu)
        q_h = np.trapezoid(integrand[order], nu[order])
        return np.log10(q_h)

    def test_compute_qh_log10_raw_lsun_hz_input(self, ssp, solar_zero_age_node):
        """compute_qh_log10 and compute_qh must accept L☉/Hz input.

        Input: raw flux as stored in SSP grid (L☉/Hz per M☉), at the
        solar-metallicity ~1 Myr node selected by value (see
        ``solar_zero_age_node``).

        The expected Q_H is computed independently in this test (float64
        numpy trapezoid quadrature, ``_independent_log10_qh``), not read off
        the code. Literature bracket: for a zero-age solar-metallicity
        instantaneous-burst population, log10(Q_H / Msun) is expected in
        [46.5, 47.0] (Leitherer et al. 1999, Starburst99: an instantaneous
        burst of 1e6 Msun gives log N(H0) ~ 52.7 photons/s, i.e.
        log10(Q_H/Msun) ~ 52.7 - 6 = 46.7).
        """
        i_met, i_age = solar_zero_age_node
        wave = ssp.ssp_wave
        flux_raw_lsun_hz = ssp.ssp_flux[i_met, i_age, :]

        q_log10_numpy = self._independent_log10_qh(wave, flux_raw_lsun_hz)
        assert 46.5 <= q_log10_numpy <= 47.0, (
            f"Independent numpy Q_H = {q_log10_numpy:.4f} dex falls outside the "
            "Starburst99 literature bracket [46.5, 47.0] for a zero-age solar-"
            "metallicity population (Leitherer et al. 1999)."
        )

        # Test log10 form against the independent reference.
        q_log10 = float(compute_qh_log10(wave, flux_raw_lsun_hz))
        assert abs(q_log10 - q_log10_numpy) < 0.01, (
            f"compute_qh_log10: Q_H = {q_log10:.4f} dex vs independent numpy "
            f"{q_log10_numpy:.4f} dex (diff {abs(q_log10 - q_log10_numpy):.4f} dex "
            "> 0.01 dex tolerance). Not using the L☉/Hz contract."
        )

        # Test linear form — FLOAT64 ONLY. compute_qh returns
        # pow10(compute_qh_log10(...)); Q_H here is ~5.9e46, which is inf under
        # float32 (ceiling ~3.4e38). This suite runs under x64 by default
        # (tests/conftest.py); compute_qh's own float32 behavior is exercised
        # directly in _shared.py's docstring, not re-asserted here.
        q_linear = float(compute_qh(wave, flux_raw_lsun_hz))
        q_linear_dex = np.log10(q_linear)
        assert abs(q_linear_dex - q_log10_numpy) < 0.01, (
            f"compute_qh: Q_H = {q_linear_dex:.4f} dex vs independent numpy "
            f"{q_log10_numpy:.4f} dex. Not consistent with compute_qh_log10."
        )

    def test_compute_qh_log10_with_lsun_conversion_offset(self, ssp, solar_zero_age_node):
        """Verify offset when flux is pre-multiplied by L_SUN (erg/s/Hz input).

        If caller mistakenly converts flux to erg/s/Hz before calling
        compute_qh_log10, the result shifts by log10(L_SUN) ~ 33.583 dex.
        Uses the same value-selected node as the test above.
        """
        i_met, i_age = solar_zero_age_node
        wave = ssp.ssp_wave
        flux_raw_lsun_hz = ssp.ssp_flux[i_met, i_age, :]

        # Correct call: raw L☉/Hz input
        q_raw = float(compute_qh_log10(wave, flux_raw_lsun_hz))

        # Mistaken call: convert to erg/s/Hz first (multiplying by L_SUN)
        flux_erg_hz = flux_raw_lsun_hz * LSUN_ERG_S
        q_erg = float(compute_qh_log10(wave, flux_erg_hz))

        # Verify offset is exactly log10(L_SUN) ≈ 33.584 dex
        log10_lsun = np.log10(LSUN_ERG_S)
        offset = q_erg - q_raw
        assert abs(offset - log10_lsun) < 0.01, (
            f"Offset = {offset:.3f} dex, expected log10(L_SUN) = {log10_lsun:.3f}. "
            "The float32-safe conversion in log space is not working correctly."
        )
