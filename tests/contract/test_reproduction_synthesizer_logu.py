# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the Synthesizer reproduction runs Cue at the inner-face log U that matches
Synthesizer's grid (#2632).

Synthesizer's ``U`` (``calculate_U_from_Q``) is the Stromgren ionization parameter of
Gutkin et al. (2016) eq. 7 times three, the volume-averaged value. Cue's ``neb_logU``
is the inner-face value at R = 10^19 cm, U = Q / (4 pi R^2 n_H c). The fixed grid
pins U and n_H, so Q is fixed and the two are related by a constant. These tests pin
the mapping helper against both definitions and check that the page no longer fixes
Cue at the un-mapped log U = -2.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from reproduction.synthesizer._drivers.synthesizer_driver import (
    cue_logu_from_synthesizer_u,
)

from tengri.utils.physics_constants import C_CGS

pytestmark = pytest.mark.contract

_REPO = Path(__file__).resolve().parents[2]
_PAGE = _REPO / "reproduction" / "synthesizer" / "01_synthesizer.py"
_R_INNER_FACE = 1e19
_N_H = 100.0


@pytest.fixture
def calculate_q_from_u():
    """Synthesizer's own Q(U); the mapping is defined against it, so it skips without it.

    ``cosmos-synthesizer`` is an optional reproduction dependency, not in ``[dev]``,
    so a module-level import fails collection of the whole suite on a runner without it.
    """
    photoionisation = pytest.importorskip("synthesizer.photoionisation")
    return photoionisation.calculate_Q_from_U


def _inline_logu(calculate_q_from_u, u_syn: float) -> float:
    """Inner-face log U of the Q that Synthesizer's U implies, computed inline."""
    q = calculate_q_from_u(u_syn, _N_H)
    return math.log10(q / (4.0 * math.pi * _R_INNER_FACE**2 * _N_H * C_CGS))


def test_matched_value_matches_inline_definitions(calculate_q_from_u):
    got = cue_logu_from_synthesizer_u(0.01)
    assert got == pytest.approx(_inline_logu(calculate_q_from_u, 0.01), abs=1e-9)
    assert abs(got - (-2.35)) < 0.01


@pytest.mark.parametrize("u_syn", [1e-3, 1e-2, 1e-1, 1.0])
def test_helper_matches_inline_definition_across_u(calculate_q_from_u, u_syn):
    expected = _inline_logu(calculate_q_from_u, u_syn)
    assert cue_logu_from_synthesizer_u(u_syn) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("u_syn", [1e-3, 1e-2, 1e-1])
def test_round_trip_through_cue_inverse(calculate_q_from_u, u_syn):
    """The helper's output, fed to Cue's own inverse, returns Synthesizer's Q."""
    from tengri.components.nebular.cue import _logq_from_logu

    logu = cue_logu_from_synthesizer_u(u_syn)
    logq = float(_logq_from_logu(jnp.asarray(logu), jnp.asarray(math.log10(_N_H)), log_R=19.0))
    assert logq == pytest.approx(math.log10(calculate_q_from_u(u_syn, _N_H)), abs=1e-9)


def test_mapping_is_monotonic_in_synthesizer_u(calculate_q_from_u):
    grid = np.logspace(-3, 0, 25)
    mapped = np.array([cue_logu_from_synthesizer_u(float(u)) for u in grid])
    assert np.all(np.diff(mapped) > 0.0)


def test_page_no_longer_fixes_unmapped_logu():
    source = _PAGE.read_text(encoding="utf-8")
    assert '"logU": Fixed(-2.0)' not in source
    assert "LOGU_CUE_FIDUCIAL" in source
