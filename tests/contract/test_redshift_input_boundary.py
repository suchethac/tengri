# SPDX-License-Identifier: BSD-3-Clause
"""A negative redshift is refused where it enters, not evaluated as 10 pc.

Luminosity distance clamps z <= 0 to 10 pc (the absolute-magnitude convention), so a
negative redshift that reaches it silently produces a flux about 1e17 too bright. The
batched spectroscopic seam and a fixed single-galaxy redshift are both refused here. z = 0
stays valid (rest frame).

Markers: contract (surface protected: redshift input validation at the user boundary)
"""

from __future__ import annotations

import pytest

from tengri import Fixed
from tengri.observation.batched import build_spectro_batches
from tests.contract.test_batched_spectro_guards import _model, _record_for
from tests.contract.test_batched_spectrum_forward import _grid

pytestmark = pytest.mark.contract


def test_batched_negative_redshift_is_refused(synthetic_ssp):
    """A galaxy with z < 0 is rejected when its batch is built."""
    wave = _grid(300, 4000.0, 7000.0)
    template = _model(synthetic_ssp, wave)
    with pytest.raises(ValueError, match="redshift"):
        build_spectro_batches([_record_for(wave, z=-0.1)], model=template)


def test_batched_zero_redshift_is_accepted(synthetic_ssp):
    """z = 0 is the rest frame and must build."""
    wave = _grid(300, 4000.0, 7000.0)
    template = _model(synthetic_ssp, wave)
    batches = build_spectro_batches([_record_for(wave, z=0.0)], model=template)
    assert len(batches) == 1


def test_fixed_negative_redshift_is_refused_at_build(synthetic_ssp):
    """A Fixed redshift below zero is refused when the model is built."""
    wave = _grid(300, 4000.0, 7000.0)
    with pytest.raises(ValueError, match="redshift"):
        _model(synthetic_ssp, wave, redshift=Fixed(-0.1))
