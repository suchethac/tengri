# SPDX-License-Identifier: BSD-3-Clause
r"""Rounded line labels land on the line list's vacuum centers at ingestion (#3113).

Cloudy's [O II] 7320/7330 blend labels (7323, 7332, 7332.15) are names, not air
wavelengths, and PyNeb/MAPPINGS labels (``H1r_6563A``) round the air wavelength
to the nearest Angstrom. Converting either as an exact air wavelength left the
rendered lines 0.2 to 3.55 Angstrom red or blue of the vacuum lines of
``observation/line_list.py``: Cue's [O II] 7320 and 7330 sat +3.55 and +1.81
Angstrom red of 7321.47 and 7332.21.

Every ingestion path that takes such labels is swept: Cue, the Flury+2024
stellar and AGN grids, and the 3MdB shock templates. The expected line for each
label is written out here, not derived from the ingestion rule.

Bound: the line list quotes its vacuum wavelengths to 0.01 Angstrom, so a line
that sits on its counterpart agrees to well under 0.01; 0.1 Angstrom still
separates the defects it is about (0.16 Angstrom and up; the smaller rounding
residuals pass before and after) and is below the 0.2
Angstrom the DESI fit centers are accurate to.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from tengri.components.nebular._line_ingest import (
    CLOUDY_BLEND_LABEL_LINES,
    cloudy_labels_to_vacuum,
    rounded_labels_to_vacuum,
)
from tengri.observation.line_list import LineList
from tests._data_skip import CUE_WEIGHTS, DATA_DIR, requires_cue_weights

pytestmark = pytest.mark.regression_bug

_TOL_AA = 0.1
_HALF_WIDTH_AA = 0.5
#: Air label in the gap between H8 and [Ne III] 3968, more than the half-width from every line.
_FREE_AIR_AA = 3920.0
#: 0.048 Angstrom from H8 (air 3889.048) and 0.36 from He I 3889: the nearest line wins.
_SNAP_AIR_AA = 3889.0
_FLURY = DATA_DIR / "flury2024_grids.h5"
_LINES = LineList.default_optical()
_CATALOG = dict(zip(_LINES.names, np.asarray(_LINES.wavelengths, dtype=float)))

#: Cue (Cloudy) label [Angstrom, air] -> line-list line it is the label of.
_CUE_COUNTERPARTS = {
    7323.0: "OII_7320",
    7332.0: "OII_7330",
    7332.15: "OII_7330",
    6562.80: "Halpha",
    4861.32: "Hbeta",
    5006.84: "OIII_5007",
    4958.91: "OIII_4959",
    6583.45: "NII_6584",
    6548.05: "NII_6548",
    6300.30: "OI_6300",
    6716.44: "SII_6717",
    6730.82: "SII_6731",
    3726.03: "OII_3726",
    3728.81: "OII_3729",
    4340.46: "Hgamma",
    4101.73: "Hdelta",
    4363.21: "OIII_4363",
    9068.62: "SIII_9069",
    9530.62: "SIII_9532",
}

#: PyNeb integer label of the Flury+2024 grids -> line-list line it rounds.
_ROUNDED_COUNTERPARTS = {
    "Ne5_3426A": "NeV_3426",
    "O2_3726A": "OII_3726",
    "O2_3729A": "OII_3729",
    "H1r_4102A": "Hdelta",
    "H1r_4340A": "Hgamma",
    "O3_4363A": "OIII_4363",
    "H1r_4861A": "Hbeta",
    "O3_4959A": "OIII_4959",
    "O3_5007A": "OIII_5007",
    "N2_5755A": "NII_5755",
    "He1r_5876A": "HeI_5876",
    "O1_6300A": "OI_6300",
    "S3_6312A": "SIII_6312",
    "N2_6548A": "NII_6548",
    "H1r_6563A": "Halpha",
    "N2_6583A": "NII_6584",
    "S2_6716A": "SII_6717",
    "S2_6731A": "SII_6731",
    "Ar3_7136A": "ArIII_7135",
    "S3_9069A": "SIII_9069",
    "S3_9531A": "SIII_9532",
}

#: The 3MdB templates spell the same lines their own way, and two of their
#: labels name the whole [O II] blend.
_SHOCK_COUNTERPARTS = {
    "OII_3726A": "OII_3726",
    "OII_3729A": "OII_3729",
    "Hg_4341A": "Hgamma",
    "Hb_4861A": "Hbeta",
    "O3_4959A": "OIII_4959",
    "O3_5007A": "OIII_5007",
    "HeI_5876A": "HeI_5876",
    "OI_6300A": "OI_6300",
    "HA_6563A": "Halpha",
    "NII_6548A": "NII_6548",
    "NII_6583A": "NII_6584",
    "SII_6716A": "SII_6717",
    "SII_6731A": "SII_6731",
    "ArIII_7135A": "ArIII_7135",
    "OII_7320A": "OII_7320",
    "OII_7330A": "OII_7330",
    "SIII_9069A": "SIII_9069",
    "SIII_9532A": "SIII_9532",
}


def _assert_on_catalog(rendered, expected_names, where):
    for got, name in zip(rendered, expected_names):
        assert abs(got - _CATALOG[name]) < _TOL_AA, (where, name, float(got), _CATALOG[name])


def _counterpart(label):
    return next((v for k, v in _CUE_COUNTERPARTS.items() if abs(label - k) < 1e-3), None)


@requires_cue_weights
@pytest.mark.parametrize(
    ("key", "attr", "n_min"),
    [
        ("sorted_line_wavelength", "sorted_line_wav", 18),
        ("nn_line_wavelength", "nn_line_wav", 3),
    ],
)
def test_cue_labels_land_on_the_line_list(key, attr, n_min):
    from tengri.components.nebular.cue import load_cue_weights

    weights = load_cue_weights(str(CUE_WEIGHTS))
    raw = np.asarray(np.load(CUE_WEIGHTS)[key], dtype=float)
    rendered = np.asarray(getattr(weights, attr), dtype=float)
    pairs = [(i, _counterpart(r)) for i, r in enumerate(raw) if _counterpart(r) is not None]
    assert len(pairs) >= n_min
    _assert_on_catalog([rendered[i] for i, _ in pairs], [c for _, c in pairs], f"cue.{attr}")


def test_cloudy_blend_labels_serve_the_synthesizer_path_too():
    """The Synthesizer NLR loader ingests through this function (grid not in the repo)."""
    out = cloudy_labels_to_vacuum(np.array([7323.0, 7332.0, 7332.15, 6562.80]))
    _assert_on_catalog(out, ["OII_7320", "OII_7330", "OII_7330", "Halpha"], "synthesizer")
    assert set(CLOUDY_BLEND_LABEL_LINES.values()) == {"OII_7320", "OII_7330"}


def _sweep_flury(grid_waves, group):
    with h5py.File(_FLURY, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f[group]["line_names"][:]]
    pairs = [
        (i, _ROUNDED_COUNTERPARTS[n]) for i, n in enumerate(names) if n in _ROUNDED_COUNTERPARTS
    ]
    assert len(pairs) == len(_ROUNDED_COUNTERPARTS)
    waves = np.asarray(grid_waves, dtype=float)
    _assert_on_catalog([waves[i] for i, _ in pairs], [c for _, c in pairs], group)


@pytest.mark.skipif(not _FLURY.is_file(), reason="flury2024_grids.h5 not present")
@pytest.mark.parametrize("density", ["cdn", "cpr"])
def test_flury_agn_grid_lines_land_on_the_line_list(density):
    from tengri.components.nebular.mappings_photo import _load_agn_grid

    grid = _load_agn_grid(_FLURY, density)
    _sweep_flury(grid.line_wavelengths, f"agn_oxaf/{density}")


@pytest.mark.skipif(not _FLURY.is_file(), reason="flury2024_grids.h5 not present")
@pytest.mark.parametrize("model", ["sb99", "bpass"])
@pytest.mark.parametrize("density", ["cdn", "cpr"])
def test_flury_stellar_grid_lines_land_on_the_line_list(model, density):
    from tengri.components.nebular.mappings_photo import _load_stellar_grid

    grid = _load_stellar_grid(_FLURY, model, density)[0]
    _sweep_flury(grid.line_wavelengths, f"{model}/{density}")


def test_shock_template_lines_land_on_the_line_list():
    from tengri.components.nebular.shock import _load_mappings_grids

    grids = _load_mappings_grids()
    if grids is None:
        pytest.skip("mappings_templates.h5 not present")
    g = grids["mappings5"]
    waves = np.asarray(g["line_wavelengths_aa"], dtype=float)
    pairs = [
        (i, _SHOCK_COUNTERPARTS[n])
        for i, n in enumerate(g["line_names"])
        if n in _SHOCK_COUNTERPARTS
    ]
    assert len(pairs) == len(_SHOCK_COUNTERPARTS)
    _assert_on_catalog([waves[i] for i, _ in pairs], [c for _, c in pairs], "shock")


def test_counterpart_free_label_is_farther_than_the_half_width_from_every_line():
    from tengri.utils.air_vacuum import vac_to_air

    air = np.asarray(vac_to_air(np.asarray(_LINES.wavelengths, dtype=float)), dtype=float)
    assert np.min(np.abs(air - _FREE_AIR_AA)) > _HALF_WIDTH_AA


def test_labels_without_a_line_list_counterpart_keep_the_plain_conversion():
    """Vacuum UV labels and air labels far from any line-list line are not snapped."""
    from tengri.utils.air_vacuum import air_to_vac

    raw = np.array([1549.0, _FREE_AIR_AA, 7319.0, 7331.0])
    out = rounded_labels_to_vacuum(raw, half_width_aa=_HALF_WIDTH_AA)
    np.testing.assert_allclose(out, air_to_vac(raw), rtol=0, atol=1e-9)


def test_label_within_the_half_width_of_a_listed_line_snaps_to_its_vacuum_centre():
    out = rounded_labels_to_vacuum(np.array([_SNAP_AIR_AA]), half_width_aa=_HALF_WIDTH_AA)
    np.testing.assert_allclose(out, [_CATALOG["H8"]], rtol=0, atol=1e-9)
