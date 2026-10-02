# SPDX-License-Identifier: BSD-3-Clause
"""Crossval: tengri GRAHSP vs upstream GRAHSP (commit 45054ddf), 38 parameter sets.

Each known upstream physics error is applied explicitly as documented conversion:
- Emission lines: upstream integrate to √2× unit-area → scale tengri by √2
- Torus log-Gaussian width: upstream exp(−x²/W²); tengri exp(−x²/(2W²))
  → W_tengri = W_upstream/√2 (exact parametrisation)
- Balmer continuum: compare integral only (upstream coarse FeII nodes)
- Attenuation: upstream (intrinsic+atten)/intrinsic vs tengri attenuated/intrinsic
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tests._data_skip import requires_grahsp


def _map_feii_template(name: str) -> str:
    return {"BruhweilerVerner08": "bruhweiler2008", "Veron-Cetty04": "veroncetty2004"}[name]


PARAM_MAPPING = {
    "plslope": ("plslope", lambda x: x),
    "uvslope": ("uvslope", lambda x: x),
    "plbendloc": ("plbendloc_nm", lambda x: x),
    "plbendwidth": ("plbendwidth", lambda x: x),
    "cutoff": ("cutoff_nm", lambda x: x),
    "afeii": ("a_feii", lambda x: x),
    "alines": ("a_lines", lambda x: x),
    "linewidth": ("linewidth_kms", lambda x: x),
    "type": ("agn_type", lambda x: int(x)),
    "fcov": ("fcov", lambda x: x),
    "si": ("si", lambda x: x),
    "coollam": ("cool_lam_um", lambda x: x),
    "coolwidth": ("cool_width", lambda x: x),
    "hotlam": ("hot_lam_um", lambda x: x),
    "hotwidth": ("hot_width", lambda x: x),
    "hotfcov": ("hot_fcov", lambda x: x),
    "ebv": ("ebv", lambda x: x),
    "ebv_agn": ("ebv_agn", lambda x: x),
    "abc": ("a_bc", lambda x: x),
    "feii": ("feii_template", _map_feii_template),
}


@pytest.fixture
def ref_file():
    path = Path(__file__).resolve().parents[2] / "data" / "grahsp_upstream_reference.h5"
    if not path.exists():
        pytest.skip(f"Reference file not found: {path}")
    return h5py.File(path, "r")


@pytest.fixture
def templates():
    return load_grahsp_templates()


@pytest.fixture
def torus_template_wave():
    """Native upstream torus wavelength grid (506 points, 360-100000 nm)."""
    with h5py.File(Path(__file__).resolve().parents[2] / "data" / "grahsp/grahsp_templates.h5", "r") as f:
        return f["torus"]["wave_nm"][:]


class _RefParams(NamedTuple):
    name: str
    params_dict: dict
    wavelength_nm: np.ndarray
    contributions: dict[str, np.ndarray]


def _load_ref_params(ref_file: h5py.File, group_name: str) -> _RefParams:
    grp = ref_file[group_name]
    params_dict = {k[6:]: v for k, v in grp.attrs.items() if k.startswith("param_")}
    wavelength_nm = ref_file["wavelength_nm"][:]
    contributions = {k: grp[k][:] for k in grp.keys()}
    return _RefParams(group_name, params_dict, wavelength_nm, contributions)


def _upstream_to_tengri_params(ref_params: _RefParams) -> GRAHSPParams:
    upstream = ref_params.params_dict
    kwargs = {"l5100": 1.0}
    for upstream_name, (tengri_name, converter) in PARAM_MAPPING.items():
        if upstream_name in upstream:
            kwargs[tengri_name] = converter(upstream[upstream_name])
    return GRAHSPParams(**kwargs)


def _dex_max(tengri: np.ndarray, upstream: np.ndarray, threshold: float = 1e-6) -> tuple[float, int]:
    """Max dex diff where both > threshold. Returns (max_dex, count_above_threshold)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        dex_diff = np.abs(np.log10(tengri + 1e-50) - np.log10(upstream + 1e-50))
        sig = (tengri > threshold) & (upstream > threshold)
        dex_diff = np.where(sig, dex_diff, np.nan)
    return float(np.nanmax(dex_diff)) if np.any(~np.isnan(dex_diff)) else 0.0, int(np.sum(sig))


@requires_grahsp
class TestGRAHSPCrossval:
    """38-point crossval against upstream GRAHSP commit 45054ddf."""

    @pytest.mark.parametrize("param_set_name", [
        "fiducial", "plslope_-2_p0", "plslope_-1_p5", "plslope_-1_p0",
        "uvslope_-0_p5", "uvslope_0_p5", "plbendloc_80_p0", "plbendloc_120_p0",
        "plbendwidth_0_p5", "plbendwidth_2_p0", "afeii_0", "afeii_2",
        "alines_0_p5", "alines_2", "linewidth_1000_p0", "linewidth_10000_p0",
        "abc_0_p3", "abc_1_p0", "fcov_0_p2", "fcov_0_p8",
        "si_-1_p0", "si_0_p0", "coollam_15_p0", "coollam_19_p0",
        "coolwidth_0_p35", "coolwidth_0_p55", "hotlam_1_p5", "hotlam_2_p5",
        "hotwidth_0_p4", "hotwidth_0_p6", "hotfcov_0_p5", "hotfcov_2_p0",
        "ebv_0_p05", "ebv_0_p1", "ebv_agn_0_p1", "ebv_agn_0_p2",
        "feii_Veron-Cetty04", "type_2",
    ])
    def test_bbb_no_conversions(self, ref_file, templates, param_set_name):
        """BBB disc: expect <=1e-6 dex (no known physics differences)."""
        ref_params = _load_ref_params(ref_file, param_set_name)
        tengri_params = _upstream_to_tengri_params(ref_params)

        wave_nm = jnp.asarray(ref_params.wavelength_nm)
        tengri_sed = evaluate_grahsp_agn(wave_nm, tengri_params, templates)

        tengri_bbb = np.asarray(tengri_sed.bbb)
        upstream_disc = ref_params.contributions.get("agn.activate_Disk", np.zeros_like(ref_params.wavelength_nm))

        max_dex, n_sig = _dex_max(tengri_bbb, upstream_disc, threshold=1e-30)
        assert max_dex <= 1e-6, f"[{param_set_name}] BBB: {max_dex:.3e} dex"

    @pytest.mark.parametrize("param_set_name", [
        "fiducial", "plslope_-2_p0", "plslope_-1_p5", "plslope_-1_p0",
        "uvslope_-0_p5", "uvslope_0_p5", "plbendloc_80_p0", "plbendloc_120_p0",
        "plbendwidth_0_p5", "plbendwidth_2_p0", "afeii_0", "afeii_2",
        "alines_0_p5", "alines_2", "linewidth_1000_p0", "linewidth_10000_p0",
        "abc_0_p3", "abc_1_p0", "fcov_0_p2", "fcov_0_p8",
        "si_-1_p0", "si_0_p0", "coollam_15_p0", "coollam_19_p0",
        "coolwidth_0_p35", "coolwidth_0_p55", "hotlam_1_p5", "hotlam_2_p5",
        "hotwidth_0_p4", "hotwidth_0_p6", "hotfcov_0_p5", "hotfcov_2_p0",
        "ebv_0_p05", "ebv_0_p1", "ebv_agn_0_p1", "ebv_agn_0_p2",
        "feii_Veron-Cetty04", "type_2",
    ])
    def test_feii_no_conversions(self, ref_file, templates, param_set_name):
        """FeII: expect <=1e-6 dex where > 1e-6 of peak."""
        ref_params = _load_ref_params(ref_file, param_set_name)
        tengri_params = _upstream_to_tengri_params(ref_params)

        wave_nm = jnp.asarray(ref_params.wavelength_nm)
        tengri_sed = evaluate_grahsp_agn(wave_nm, tengri_params, templates)

        tengri_feii = np.asarray(tengri_sed.feii)
        upstream_feii = ref_params.contributions.get("agn.activate_FeLines", np.zeros_like(ref_params.wavelength_nm))

        peak_upstream = np.max(np.abs(upstream_feii))
        threshold = max(1e-6, 1e-6 * peak_upstream)

        max_dex, n_sig = _dex_max(tengri_feii, upstream_feii, threshold=threshold)
        assert max_dex <= 1e-6, f"[{param_set_name}] FeII: {max_dex:.3e} dex ({n_sig} points)"

    @pytest.mark.parametrize("param_set_name", [
        "fiducial", "plslope_-2_p0", "plslope_-1_p5", "plslope_-1_p0",
        "uvslope_-0_p5", "uvslope_0_p5", "plbendloc_80_p0", "plbendloc_120_p0",
        "plbendwidth_0_p5", "plbendwidth_2_p0", "afeii_0", "afeii_2",
        "alines_0_p5", "alines_2", "linewidth_1000_p0", "linewidth_10000_p0",
        "abc_0_p3", "abc_1_p0", "fcov_0_p2", "fcov_0_p8",
        "si_-1_p0", "si_0_p0", "coollam_15_p0", "coollam_19_p0",
        "coolwidth_0_p35", "coolwidth_0_p55", "hotlam_1_p5", "hotlam_2_p5",
        "hotwidth_0_p4", "hotwidth_0_p6", "hotfcov_0_p5", "hotfcov_2_p0",
        "ebv_0_p05", "ebv_0_p1", "ebv_agn_0_p1", "ebv_agn_0_p2",
        "feii_Veron-Cetty04", "type_2",
    ])
    def test_si_with_width_conversion(self, ref_file, templates, param_set_name):
        """Si: expect <=1e-6 dex after width reparametrisation W_ten = W_up/√2."""
        ref_params = _load_ref_params(ref_file, param_set_name)
        upstream = ref_params.params_dict
        tengri_params = _upstream_to_tengri_params(ref_params)

        # Apply width reparametrisation
        w_cool_adj = float(upstream.get("coolwidth", 0.45)) / np.sqrt(2)
        w_hot_adj = float(upstream.get("hotwidth", 0.5)) / np.sqrt(2)

        tengri_params_adj = GRAHSPParams(
            l5100=tengri_params.l5100,
            plslope=tengri_params.plslope,
            uvslope=tengri_params.uvslope,
            plbendloc_nm=tengri_params.plbendloc_nm,
            plbendwidth=tengri_params.plbendwidth,
            cutoff_nm=tengri_params.cutoff_nm,
            a_lines=tengri_params.a_lines,
            a_feii=tengri_params.a_feii,
            linewidth_kms=tengri_params.linewidth_kms,
            agn_type=tengri_params.agn_type,
            fcov=tengri_params.fcov,
            si=tengri_params.si,
            cool_lam_um=tengri_params.cool_lam_um,
            cool_width=w_cool_adj,
            hot_lam_um=tengri_params.hot_lam_um,
            hot_width=w_hot_adj,
            hot_fcov=tengri_params.hot_fcov,
            ebv=tengri_params.ebv,
            ebv_agn=tengri_params.ebv_agn,
            a_bc=tengri_params.a_bc,
        )

        wave_nm = jnp.asarray(ref_params.wavelength_nm)
        tengri_sed_adj = evaluate_grahsp_agn(wave_nm, tengri_params_adj, templates)

        tengri_si = np.asarray(tengri_sed_adj.si)
        upstream_si = ref_params.contributions.get("agn.activate_Torus_Si", np.zeros_like(ref_params.wavelength_nm))

        # Compare in torus window 0.4-99 µm
        wave_um = ref_params.wavelength_nm / 1000.0
        in_window = (wave_um >= 0.4) & (wave_um <= 99.0)

        max_dex, n_sig = _dex_max(tengri_si[in_window], upstream_si[in_window], threshold=1e-6)
        assert max_dex <= 1e-6, f"[{param_set_name}] Si: {max_dex:.3e} dex ({n_sig} points)"

    @pytest.mark.parametrize("param_set_name", [
        "fiducial", "plslope_-2_p0", "plslope_-1_p5", "plslope_-1_p0",
        "uvslope_-0_p5", "uvslope_0_p5", "plbendloc_80_p0", "plbendloc_120_p0",
        "plbendwidth_0_p5", "plbendwidth_2_p0", "afeii_0", "afeii_2",
        "alines_0_p5", "alines_2", "linewidth_1000_p0", "linewidth_10000_p0",
        "abc_0_p3", "abc_1_p0", "fcov_0_p2", "fcov_0_p8",
        "si_-1_p0", "si_0_p0", "coollam_15_p0", "coollam_19_p0",
        "coolwidth_0_p35", "coolwidth_0_p55", "hotlam_1_p5", "hotlam_2_p5",
        "hotwidth_0_p4", "hotwidth_0_p6", "hotfcov_0_p5", "hotfcov_2_p0",
        "ebv_0_p05", "ebv_0_p1", "ebv_agn_0_p1", "ebv_agn_0_p2",
        "feii_Veron-Cetty04", "type_2",
    ])
    def test_lines_with_sqrt2_scaling(self, ref_file, templates, param_set_name):
        """Broad+narrow lines: expect <=1e-6 dex after √2 scaling (upstream integrate √2× unit-area)."""
        ref_params = _load_ref_params(ref_file, param_set_name)
        tengri_params = _upstream_to_tengri_params(ref_params)

        wave_nm = jnp.asarray(ref_params.wavelength_nm)
        tengri_sed = evaluate_grahsp_agn(wave_nm, tengri_params, templates)

        tengri_broad = np.asarray(tengri_sed.broad_lines)
        tengri_narrow = np.asarray(tengri_sed.narrow_lines)
        tengri_lines_scaled = (tengri_broad + tengri_narrow) * np.sqrt(2)

        upstream_bl = ref_params.contributions.get("agn.activate_EmLines_BL", np.zeros_like(ref_params.wavelength_nm))
        upstream_nl = ref_params.contributions.get("agn.activate_EmLines_NL", np.zeros_like(ref_params.wavelength_nm))
        upstream_lines = upstream_bl + upstream_nl

        peak_upstream = np.max(np.abs(upstream_lines))
        threshold = max(1e-6, 1e-6 * peak_upstream)

        max_dex, n_sig = _dex_max(tengri_lines_scaled, upstream_lines, threshold=threshold)
        assert max_dex <= 1e-6, f"[{param_set_name}] Lines: {max_dex:.3e} dex ({n_sig} points)"

    @pytest.mark.parametrize("param_set_name", [
        "abc_0_p3", "abc_1_p0",  # Only ABC > 0
    ])
    def test_balmer_continuum_integral(self, ref_file, templates, param_set_name):
        """Balmer continuum (ABC>0 only): integral over 210-355 nm, measured tolerance."""
        ref_params = _load_ref_params(ref_file, param_set_name)
        upstream = ref_params.params_dict
        abc_val = float(upstream.get("abc", 0.0))

        if abc_val <= 0:
            pytest.skip("ABC=0, no Balmer continuum")

        tengri_params = _upstream_to_tengri_params(ref_params)

        wave_nm = jnp.asarray(ref_params.wavelength_nm)
        tengri_sed = evaluate_grahsp_agn(wave_nm, tengri_params, templates)

        tengri_bc = np.asarray(tengri_sed.balmer)
        upstream_bc = ref_params.contributions.get("agn.activate_BC", np.zeros_like(ref_params.wavelength_nm))

        # Integrate over 210-355 nm window
        in_window = (ref_params.wavelength_nm >= 210.0) & (ref_params.wavelength_nm <= 355.0)

        if not np.any(in_window):
            pytest.skip("No data in 210-355 nm window")

        tengri_integral = np.trapz(tengri_bc[in_window], ref_params.wavelength_nm[in_window])
        upstream_integral = np.trapz(upstream_bc[in_window], ref_params.wavelength_nm[in_window])

        if abs(upstream_integral) > 1e-30:
            rel_diff = abs(tengri_integral - upstream_integral) / abs(upstream_integral)
        else:
            rel_diff = 0.0 if abs(tengri_integral) < 1e-30 else 1.0

        # Measured tolerance for BC integral: ~0.01 (smoothing differences)
        assert rel_diff <= 0.01, f"[{param_set_name}] BC integral: rel_diff={rel_diff:.3e}"

    @pytest.mark.parametrize("param_set_name", [
        "ebv_0_p05", "ebv_0_p1", "ebv_agn_0_p1", "ebv_agn_0_p2",  # E(B-V) > 0
    ])
    def test_attenuation_factors(self, ref_file, templates, param_set_name):
        """Attenuation factors (E(B-V)>0 only): expect <=1e-6 dex where intrinsic significant."""
        ref_params = _load_ref_params(ref_file, param_set_name)
        upstream = ref_params.params_dict
        ebv = float(upstream.get("ebv", 0.0))
        ebv_agn = float(upstream.get("ebv_agn", 0.0))

        if ebv <= 0 and ebv_agn <= 0:
            pytest.skip("No attenuation")

        tengri_params = _upstream_to_tengri_params(ref_params)

        wave_nm = jnp.asarray(ref_params.wavelength_nm)
        tengri_sed = evaluate_grahsp_agn(wave_nm, tengri_params, templates)

        # BBB attenuation
        tengri_bbb_total = (
            np.asarray(tengri_sed.bbb) + np.asarray(tengri_sed.broad_lines) +
            np.asarray(tengri_sed.narrow_lines) + np.asarray(tengri_sed.feii) +
            np.asarray(tengri_sed.balmer)
        )
        tengri_bbb_atten = np.asarray(tengri_sed.bbb_attenuated)

        upstream_disc = ref_params.contributions.get("agn.activate_Disk", np.zeros_like(ref_params.wavelength_nm))
        upstream_atten_disc = ref_params.contributions.get("attenuation.agn.activate_Disk", np.zeros_like(ref_params.wavelength_nm))
        upstream_disc_atten = upstream_disc + upstream_atten_disc

        # Compare factors where intrinsic significant
        peak_upstream = np.max(np.abs(upstream_disc))
        threshold = max(1e-6, 1e-6 * peak_upstream)
        sig = upstream_disc > threshold

        if np.any(sig):
            with np.errstate(divide='ignore', invalid='ignore'):
                tengri_factor = tengri_bbb_atten[sig] / (tengri_bbb_total[sig] + 1e-50)
                upstream_factor = upstream_disc_atten[sig] / (upstream_disc[sig] + 1e-50)

                dex_diff = np.abs(np.log10(tengri_factor + 1e-50) - np.log10(upstream_factor + 1e-50))
                max_dex = np.nanmax(dex_diff[~np.isinf(dex_diff)]) if np.any(~np.isinf(dex_diff)) else 0.0

            assert max_dex <= 1e-6, f"[{param_set_name}] Attenuation factor: {max_dex:.3e} dex"
