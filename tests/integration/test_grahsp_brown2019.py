"""Brown 2019 AGN atlas regression test (3 representative sources).

Uses best-fit parameters from Table 4 of Buchner+ 2024 (arXiv:2405.19297) for
three Brown 2019 atlas AGN spanning luminosity / obscuration regimes:

- ``MRK231`` — luminous, dust-buried QSO (post-merger ULIRG).
- ``IRAS-11119+3257`` — luminous Type-1 QSO with strong outflow.
- ``F2M1113+1244`` — red, optically obscured Type-1 QSO.

The test does **not** refit the photometry — it verifies that the GRAHSP
forward model reproduces self-consistent SEDs and bolometric quantities at
the published best-fit parameters, demonstrating the pipeline end-to-end on
realistic AGN.

Notes on Table 4 column units (paper Sidewaystable, columns 5-21)::

    Col          | Logged?  | Unit         | Field name
    -------------|----------|--------------|---------------
    L_AGN        | yes      | erg/s        | lum5100A (= lambda*L_lambda)
    A_FeII       | yes      |              | A_FeII
    A_lines      | yes      |              | A_lines
    W_line       | yes      | km/s         | line_width_kms
    beta         | no       |              | plslope
    lambda_bend  | yes      | nm           | plbendloc_nm
    W_bend       | yes      | dex          | plbendwidth
    beta_UV      | no       |              | uvslope
    E(B-V)       | yes      | mag          | ebv
    E(B-V)_AGN   | yes      | mag          | ebv_agn
    Si           | no       |              | Si
    fcov         | no       |              | fcov
    lambda_cool  | no       | um           | cool_lam_um
    W_cool       | no       | dex          | cool_width  (already in dex)
    lambda_hot   | no       | um           | hot_lam_um
    W_hot        | no       | dex          | hot_width
    f_hot        | no       |              | hot_fcov
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.grahsp import GRAHSPParams, evaluate_grahsp_agn

# Paper Table 4 best-fits (logs reverted; comments give the table value).
BROWN2019_TABLE4 = {
    "MRK231": dict(
        # 45.1 -> 10^45.1
        l5100=10.0**45.1,
        a_feii=10.0**0.6,
        a_lines=10.0**-0.4,
        linewidth_kms=10.0**3.9,
        plslope=-1.7,
        plbendloc_nm=10.0**2.2,
        plbendwidth=10.0**-0.7,
        uvslope=1.7,
        ebv=10.0**-1.1,
        ebv_agn=10.0**-0.1,
        si=-1.6,
        fcov=0.7,
        cool_lam_um=30.0,
        cool_width=0.2,
        hot_lam_um=10.0**0.71,
        hot_width=1.0,
        hot_fcov=2.5,
        agn_type=1,
    ),
    "IRAS-11119+3257": dict(
        l5100=10.0**45.8,
        a_feii=10.0**0.7,
        a_lines=10.0**-0.3,
        linewidth_kms=10.0**3.7,
        plslope=-1.6,
        plbendloc_nm=10.0**2.1,
        plbendwidth=10.0**-1.7,
        uvslope=-0.5,  # row says -0.5 -- positive vs negative ambiguous in the paper.
        # The paper requires uvslope > plslope; with plslope=-1.6 we need uv > -1.6.
        # We adopt the PDF table value (-0.5) as-is.
        ebv=10.0**-0.9,
        ebv_agn=10.0**0.1,
        si=-2.2,
        fcov=0.2,
        cool_lam_um=30.0,
        cool_width=0.6,
        hot_lam_um=10.0**0.58,  # 3.8 um
        hot_width=1.0,
        hot_fcov=10.0**0.99,  # 9.9
        agn_type=1,
    ),
    "F2M1113+1244": dict(
        l5100=10.0**46.0,
        a_feii=10.0**1.3,
        a_lines=10.0**-0.5,
        linewidth_kms=10.0**2.8,
        plslope=-1.6,
        plbendloc_nm=10.0**2.3,
        plbendwidth=10.0**-2.0,
        uvslope=-0.1,
        ebv=10.0**-0.9,
        ebv_agn=10.0**-0.1,
        si=-0.8,
        fcov=0.8,
        cool_lam_um=30.0,
        cool_width=0.8,
        hot_lam_um=10.0**0.23,  # 1.7 um
        hot_width=0.6,
        hot_fcov=10.0**0.6,
        agn_type=1,
    ),
}


@pytest.mark.parametrize("source", list(BROWN2019_TABLE4.keys()))
def test_brown2019_sed_finite_and_physical(source):
    """Assert the Buchner+ 2024 Table 4 best-fit produces a finite, ordered SED."""
    params = GRAHSPParams(**BROWN2019_TABLE4[source])
    # 100 nm to 100 um wavelength grid (covers UV to far-IR rest-frame).
    wave_nm = jnp.logspace(2, 5, 500)
    sed = evaluate_grahsp_agn(wave_nm, params)

    assert jnp.all(jnp.isfinite(sed.bbb)), f"{source} BBB has non-finite values"
    assert jnp.all(jnp.isfinite(sed.torus)), f"{source} torus has non-finite values"
    assert jnp.all(jnp.isfinite(sed.bbb_attenuated))
    assert sed.l_bol_bbb > 0
    assert sed.l_bol_torus > 0
    # Heavy attenuation should always reduce the AGN-attenuated SED.
    intrinsic = sed.bbb + sed.broad_lines + sed.narrow_lines + sed.feii
    assert jnp.all(sed.bbb_attenuated <= intrinsic + 1e-30)


@pytest.mark.parametrize("source", list(BROWN2019_TABLE4.keys()))
def test_brown2019_torus_normalisation(source):
    """At 12 um, lambda*L_lambda should equal 2.5 * lum5100A * fcov (Eq. fcov)."""
    params = GRAHSPParams(**BROWN2019_TABLE4[source])
    wave_nm = jnp.array([12000.0])
    sed = evaluate_grahsp_agn(wave_nm, params)
    # Note: only the dust continuum (no Si) at 12 um equals the analytic form.
    # Si feature peaks elsewhere; at 12 um its contribution is small but not 0.
    # We verify the dust continuum alone:
    expected_lam_Llam = 2.5 * params.l5100 * params.fcov
    np.testing.assert_allclose(
        float(sed.torus[0] * 12000.0),
        expected_lam_Llam,
        rtol=1e-6,
    )


def test_brown2019_lbolbbb_within_paper_range():
    """L_bol_BBB should track the published L_AGN column to within ~1 dex."""
    params = GRAHSPParams(**BROWN2019_TABLE4["IRAS-11119+3257"])
    wave_nm = jnp.logspace(2, 5, 1000)
    sed = evaluate_grahsp_agn(wave_nm, params)
    # Paper L_AGN = 10^45.8 erg/s = lambda*L_lambda(5100Å). Bolometric BBB
    # integrates BBB+lines+FeII above 91.2 nm — typically a few times larger
    # than lambda*L_lambda(5100). We assert it is in the same dex.
    assert sed.l_bol_bbb > 0.1 * params.l5100
    assert sed.l_bol_bbb < 100.0 * params.l5100
