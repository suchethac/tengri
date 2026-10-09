"""Value pins for every shipped surviving-mass table (one per Z row).

The loader tests compare a loaded table with the same companion file, so they cannot
see a change to the file's content. These pins fix the bytes and the 10 Gyr column of
every table in ``src/tengri/data/ssp_mass_remaining``: a sha256 of the ``mass_remaining``
dataset and, for each metallicity row, the value at the oldest node with
``log10_age_yr <= 10``.

A pin changes only when a table is deliberately rebuilt. To regenerate after such a
rebuild, run from the repository root:

    .venv/bin/python tests/physics/test_mass_remaining_table_pins.py

and paste the printed ``PINS`` literal over the one below, with the rebuild recorded in
the table's PROVENANCE.md.
"""

from __future__ import annotations

import glob
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

TABLE_DIR = Path(__file__).resolve().parents[2] / "src" / "tengri" / "data" / "ssp_mass_remaining"
TEN_GYR_LOG10 = 10.0
VALUE_TOL = 1e-12

# name -> (sha256 of mass_remaining as contiguous float64, index of the 10 Gyr node,
#          its log10 age, the 10 Gyr column (one value per Z row))
PINS = {
    "mass_remaining_bc03pdva94_chabrier.h5": (
        "61caa31586239efd8fb16d4a0552d5c0b6e908e65ef64d5e3005976812a0c057",
        179,
        10.0,
        (0.48537, 0.4948, 0.50279, 0.50537, 0.51206, 0.50813),
    ),
    "mass_remaining_bsti_chabrier.h5": (
        "f8670699e0f7c604156cf2ea41a0555df8e66de5e1bda99182cca64036563683",
        84,
        10.0,
        (
            0.5534674981472293,
            0.5541202207352737,
            0.5550327871733562,
            0.5569414908705667,
            0.5602694642125424,
            0.5852025192071615,
            0.5873517038421538,
            0.5745015756166929,
            0.5774677243814663,
            0.5786284914363636,
        ),
    ),
    "mass_remaining_bsti_kroupa.h5": (
        "13c0b0ab66c2eed9dc3fb880a11d4c2afaff8b81a334cd788766217ce5ebf0f4",
        84,
        10.0,
        (
            0.5796323701055692,
            0.5802759112596084,
            0.5811768355422186,
            0.5830452230957708,
            0.586273999096905,
            0.6125165762752702,
            0.6145468947921071,
            0.5997487724647784,
            0.6025395246900573,
            0.603630240662744,
        ),
    ),
    "mass_remaining_bsti_salpeter.h5": (
        "6904fb26050a0e93d14890432024c736208a8a706e1b99b01292df31b3bfea90",
        84,
        10.0,
        (
            0.7227184134282182,
            0.7231840274152599,
            0.7238355893276518,
            0.7251857569910358,
            0.727515539612305,
            0.8187893391103005,
            0.8202568338382432,
            0.7372174810161869,
            0.7391913392631144,
            0.7399706429825116,
        ),
    ),
    "mass_remaining_mist_chabrier.h5": (
        "8146b65e8321a7bfca27678979aee22ed570adfafc911dc4363fe1a162ed3edc",
        100,
        10.0,
        (
            0.5544191368937145,
            0.5546472206626318,
            0.555040829465646,
            0.5553474559770268,
            0.5562476116301301,
            0.5577281981643266,
            0.560127777237928,
            0.5635425826919641,
            0.5682277431756388,
            0.5742288899065908,
            0.580243970414251,
            0.5832663412166346,
        ),
    ),
    "mass_remaining_mist_kroupa.h5": (
        "e720786269624826c3f36df0efbc03648472b5f8b197806ed2e37af8b9c2fcc0",
        100,
        10.0,
        (
            0.580795281729464,
            0.5810221417808977,
            0.5814123796257971,
            0.5817175713409053,
            0.5826076624111082,
            0.5840658530089835,
            0.5864118273367206,
            0.5897076733573478,
            0.5941763634597965,
            0.5998257402750355,
            0.6054741633781429,
            0.6083232611067289,
        ),
    ),
    "mass_remaining_mist_salpeter.h5": (
        "95c8e3395c1d85822b61bd7858abe7559c5dc7547121a4d404472a59f8b2d2af",
        100,
        10.0,
        (
            0.7331103931439635,
            0.733261888646792,
            0.7335578312869356,
            0.7337565746141895,
            0.7343867947206437,
            0.7354293192389547,
            0.7371428519006712,
            0.7394110894301802,
            0.742694856949523,
            0.7468195153186102,
            0.7503162390261359,
            0.752789099675378,
        ),
    ),
    "mass_remaining_pdva_chabrier.h5": (
        "9706346815e49f02b7b68260afcda135169a3c373d685d2faf1e074b0d911a78",
        90,
        10.0,
        (
            0.5552038903967417,
            0.5554540840499607,
            0.555660125379862,
            0.5566355959059167,
            0.5569614418306514,
            0.5570980375101591,
            0.556979706681144,
            0.5576781765773068,
            0.5588250515636126,
            0.5597111463395517,
            0.560545790292744,
            0.5612636863211419,
            0.5618868260544988,
            0.5656259371144908,
            0.5676000174052979,
            0.5666119241544717,
            0.569921451377311,
            0.5729557632652597,
            0.5737977354020659,
            0.5736262466297902,
            0.5743016967553617,
            0.5749229385228347,
        ),
    ),
    "mass_remaining_pdva_kroupa.h5": (
        "9424917abc45e88735ff15e27b6cd4a083e672242df2123f3f19aad36b109ca5",
        90,
        10.0,
        (
            0.5813260911003351,
            0.581571697896963,
            0.5817737806504403,
            0.5827278224971811,
            0.5830450345328392,
            0.5831778088960736,
            0.5830675442017121,
            0.5837495440511017,
            0.584862751082288,
            0.5857187859442541,
            0.5865244090405792,
            0.5872147023586992,
            0.5878124097900395,
            0.5913704297471971,
            0.5932364927554296,
            0.5923157254684616,
            0.5954091318821048,
            0.5982629898666719,
            0.5990579806170894,
            0.5989032652164132,
            0.5995375695672012,
            0.6001142951339854,
        ),
    ),
    "mass_remaining_pdva_salpeter.h5": (
        "781d5931fb8ce877197acb450dc4c816b673881e6de8a193fed1e219f5fa47c1",
        90,
        10.0,
        (
            0.7239419541874216,
            0.7241195027017236,
            0.7242655610995041,
            0.7249548409106823,
            0.7251839476892332,
            0.725279817698637,
            0.7252006316523175,
            0.7256930561234524,
            0.7264963944423427,
            0.7271138124138211,
            0.7276946213954355,
            0.7281920309989874,
            0.728622589636528,
            0.731182515874436,
            0.7325235499609448,
            0.731862230430526,
            0.7340904394455919,
            0.7361361558885846,
            0.7367166211033812,
            0.7366096669468218,
            0.7370543569624917,
            0.7374533102416704,
        ),
    ),
    "mass_remaining_pgny_mist_chabrier.h5": (
        "fb4cb8f4627eb7f3379b28f2efde8a496a067c46d3806c547ac9da22c9e86f89",
        100,
        10.0,
        (
            0.5635842152250559,
            0.5684481054427595,
            0.5826271768008852,
            0.5959028425533364,
            0.5539485055819544,
            0.5544406649465989,
            0.5550629179626769,
            0.552842699814599,
            0.5529528416357178,
            0.5513782595602652,
            0.5509176410732501,
            0.5540443760215871,
            0.5560827504689817,
            0.5873298310628683,
            0.5614754790744688,
        ),
    ),
    "mass_remaining_prsc_chabrier.h5": (
        "cc1d9dbeea8975c729cdaab9b49f32e779b25d1842c3809efa5f9cffba5b7af8",
        90,
        10.0,
        (
            0.554085881299938,
            0.5529689119311548,
            0.5537787116142862,
            0.5549503419924678,
            0.5570648780495655,
            0.5607077410869453,
            0.5637331212799023,
            0.5661641794070829,
            0.5682021729634469,
            0.5716566345408063,
            0.5733905721890296,
            0.5748875505612733,
            0.5789024304575906,
            0.5791399449296607,
            0.5815277565085162,
        ),
    ),
    "mass_remaining_prsc_kroupa.h5": (
        "35a0c9587ae76e2034a3fab12411d5d78d92faa398a5c99addf54ded44170fd9",
        90,
        10.0,
        (
            0.5804982455015953,
            0.5791872357604959,
            0.5799943562905224,
            0.5811457833950904,
            0.5832241366396713,
            0.5867681283490668,
            0.5896588510594565,
            0.5919737662749489,
            0.5938991226955701,
            0.5971421932698913,
            0.5987801046391081,
            0.6001833589273157,
            0.6041936110327462,
            0.6044226300931633,
            0.6074412056044447,
        ),
    ),
    "mass_remaining_prsc_salpeter.h5": (
        "feaaced09897598e391dee40fd5bd658bbeb4331dd806acdd3f8db0c9bc4d833",
        90,
        10.0,
        (
            0.732193448556431,
            0.72509291417613,
            0.7257187618799279,
            0.7263950208493017,
            0.7278630596545524,
            0.7305801698800011,
            0.7325025875734963,
            0.7342664874042808,
            0.7357734361153824,
            0.7378784675804603,
            0.7391355575158435,
            0.7400572371954254,
            0.7490724352612668,
            0.7494366928805505,
            0.77191515923204,
        ),
    ),
}


def _read(path: Path):
    with h5py.File(path, "r") as f:
        table = np.ascontiguousarray(f["mass_remaining"][()], dtype=np.float64)
        ages = np.asarray(f["log10_age_yr"][()], dtype=np.float64)
    return table, ages


def _pin_for(path: Path):
    """Return the pin tuple for one table, computed from the committed file."""
    table, ages = _read(path)
    digest = hashlib.sha256(table.tobytes()).hexdigest()
    idx = int(np.flatnonzero(ages <= TEN_GYR_LOG10 + 1e-9)[-1])
    return digest, idx, float(ages[idx]), tuple(float(v) for v in table[:, idx])


def _shipped_tables():
    return sorted(Path(p) for p in glob.glob(str(TABLE_DIR / "mass_remaining_*.h5")))


@pytest.mark.regression_paper
def test_every_shipped_table_has_a_pin():
    """A new table must add its pin; a removed table must drop it."""
    assert {p.name for p in _shipped_tables()} == set(PINS)


@pytest.mark.regression_paper
@pytest.mark.parametrize("path", _shipped_tables(), ids=lambda p: p.name)
def test_table_bytes_match_the_pinned_sha256(path):
    digest, _, _, _ = _pin_for(path)
    assert digest == PINS[path.name][0], f"{path.name}: table bytes changed"


@pytest.mark.regression_paper
@pytest.mark.parametrize("path", _shipped_tables(), ids=lambda p: p.name)
def test_ten_gyr_column_matches_every_z_row(path):
    """Each metallicity row carries its own pinned value at the 10 Gyr node (1e-12)."""
    idx, age, column = PINS[path.name][1:]
    _, got_idx, got_age, got_column = _pin_for(path)
    assert got_idx == idx and got_age == age, f"{path.name}: 10 Gyr node moved"
    assert len(got_column) == len(column)
    for iz, (got, want) in enumerate(zip(got_column, column)):
        assert abs(got - want) <= VALUE_TOL, f"{path.name}: row {iz} {got!r} != {want!r}"


if __name__ == "__main__":
    for shipped in _shipped_tables():
        digest, idx, age, column = _pin_for(shipped)
        print(f'    "{shipped.name}": (')
        print(f'        "{digest}",')
        print(f"        {idx},")
        print(f"        {age!r},")
        print("        (" + ", ".join(repr(v) for v in column) + "),")
        print("    ),")
