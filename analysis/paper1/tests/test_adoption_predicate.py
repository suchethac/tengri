"""``--only-missing`` must not skip a cell whose adopted flag rests on too few draws."""

import json

from analysis.paper1.fit_one import ESS_FLOOR
from analysis.paper1.run_candels_fits import cell_is_adopted


def _write(path, **payload):
    path.write_text(json.dumps(payload))
    return path


def test_adopted_flag_alone_is_not_enough(tmp_path):
    # 14099/V: adoption_pass true on ess_min 3 before ESS_FLOOR joined the bar.
    p = _write(tmp_path / "14099_V.json", adoption_pass=True, ess_min=3.0)
    assert cell_is_adopted(p) is False


def test_adopted_with_ess_at_floor_is_adopted(tmp_path):
    p = _write(tmp_path / "x.json", adoption_pass=True, ess_min=ESS_FLOOR)
    assert cell_is_adopted(p) is True


def test_missing_ess_is_not_adopted(tmp_path):
    p = _write(tmp_path / "y.json", adoption_pass=True)
    assert cell_is_adopted(p) is False


def test_not_adopted_and_missing_file(tmp_path):
    assert (
        cell_is_adopted(_write(tmp_path / "z.json", adoption_pass=False, ess_min=500.0)) is False
    )
    assert cell_is_adopted(tmp_path / "absent.json") is False
