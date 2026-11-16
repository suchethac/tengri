# SPDX-License-Identifier: BSD-3-Clause
"""#3547 D4: ``load_catalog`` output goes straight into ``Catalog`` (no fitting here).

The table load_catalog returns is a flat column table. This test builds the
``Catalog`` the fitter consumes and checks the arrays it hands over: a nodata cell
is absent, an upper-limit cell is censored at n_sigma times its error with sigma =
err, and detected rows are unchanged.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.contract

pytest.importorskip("pandas")

BANDS = ("band_0", "band_1", "band_2")
NODATA = -9999.0
UPPER = -99.0


@pytest.fixture
def fwd(synthetic_ssp_wide, simple_observation):
    """Small 3-band ForwardModel with a catalog redshift range (as in test_catalog_class)."""
    from tengri import DEFAULT, FREE, Fixed, ForwardModel, SEDModel, WavePrecomp

    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.5,
        },
        neb={"type": "none"},
        redshift=Fixed(DEFAULT),
        approx=WavePrecomp(catalog_z_range=(0.01, 2.0)),
    )
    return ForwardModel.build(sed=sed, observation=simple_observation)


@pytest.fixture
def pipeline_csv(tmp_path: Path) -> Path:
    """Three galaxies: detected; a nodata cell in band_0; an upper limit in band_1."""
    p = tmp_path / "pipeline.csv"
    p.write_text(
        "id,redshift,band_0,band_0_err,band_1,band_1_err,band_2,band_2_err\n"
        f"gal1,0.5,1.0,0.1,2.0,0.2,3.0,0.3\n"
        f"gal2,0.6,{NODATA},0.2,2.5,0.25,3.5,0.35\n"
        f"gal3,0.7,1.5,0.15,{UPPER},0.4,2.5,0.25\n"
    )
    return p


def _load(path):
    from tengri.io import load_catalog

    return load_catalog(
        path,
        flux_unit="mJy",
        filter_mapping={b: b for b in BANDS},
        sentinels={NODATA: "nodata", UPPER: ("upper_limit", 3.0)},
    )


def _censor_cols():
    return {b: f"{b}_censor" for b in BANDS}


def test_loaded_table_builds_a_catalog_directly(pipeline_csv, fwd):
    from tengri import Catalog

    table = _load(pipeline_csv)
    cat = Catalog(
        fwd,
        table,
        flux_unit="cgs_fnu",
        redshift_col="redshift",
        censor_cols=_censor_cols(),
        missing="mask",
    )
    ca = cat._catalog_arrays

    # nodata cell: absent, so it leaves the likelihood
    assert not ca.presence[1, 0]
    # upper limit cell: flux = 3 * err, sigma = err, censor +1
    assert ca.censor[2, 1] == 1
    np.testing.assert_allclose(ca.flux[2, 1], 3.0 * 0.4)
    np.testing.assert_allclose(ca.noise[2, 1], 0.4)
    # detected rows unchanged
    np.testing.assert_allclose(ca.flux[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(ca.noise[0], [0.1, 0.2, 0.3])
    assert ca.censor[0].tolist() == [0, 0, 0]
    assert bool(ca.presence[0].all())
    np.testing.assert_allclose(ca.redshift, [0.5, 0.6, 0.7])


def test_nodata_without_missing_mask_is_refused_with_guidance(pipeline_csv, fwd):
    from tengri import Catalog

    table = _load(pipeline_csv)
    with pytest.raises(ValueError, match="missing='mask'"):
        Catalog(
            fwd,
            table,
            flux_unit="cgs_fnu",
            redshift_col="redshift",
            censor_cols=_censor_cols(),
        )
