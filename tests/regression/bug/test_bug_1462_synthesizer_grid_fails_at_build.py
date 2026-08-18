# SPDX-License-Identifier: BSD-3-Clause
"""A missing Synthesizer AGN grid fails the build, not the first predict (#1462).

``SEDModel.build`` already resolved the grid path at build time, to pre-warm the
``SynthesizerNLRBackend`` singleton before JIT (the #390 class of bug). But the
whole block sat inside ``contextlib.suppress(Exception)``, so a grid that was
simply not on disk was swallowed along with everything else: ``build`` returned
a model object and the error (now ``TengriIOError``, #1952) surfaced at the first
``predict_photometry``, far from the ``nlr='synthesizer'`` that caused it.

``recipes.unified_agn()`` is the reachable case — it selects
``nlr='synthesizer'``, and the grid ships via ``synthesizer-download
--agn-test-grids`` rather than with tengri.

The tests below pin **both** directions, because the narrowness is the whole
point. Only path *resolution* moved out of the suppress; pre-warming stayed
inside it, since that is a genuine optimization and the lazy path remains
correct if the singleton cannot be built. A fix that also propagated a corrupt
grid would break every environment whose grid file is present but unreadable,
which the suppress exists to tolerate.
"""

import re

import jax.numpy as jnp
import pytest

import tengri
from tengri import Observation, Photometry, SEDModel
from tengri.config.exceptions import TengriIOError
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

# Bare-stellar: the test below builds ``recipes.star_forming_photometry()``,
# which selects Cue, and Cue refuses a nebular-included grid (#1579). This read
# the wNE grid and only worked because conftest's TENGRI_ALLOW_WNE_CUE=1 --
# meant for the synthetic fixtures' unphysical Q_H -- also disabled the
# metadata check. Both grids are committed, so the skip guard is unchanged.
_SSP = "data/fsps_prsc_miles_chabrier.h5"


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def ssp():
    import pathlib

    if not pathlib.Path(_SSP).is_file():
        pytest.skip(f"{_SSP} not present")
    return tengri.load_ssp(_SSP)


@pytest.fixture
def obs():
    return Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (4800.0, 6200.0, 9000.0, 22000.0)))
    )


def test_a_missing_grid_raises_at_build(ssp, obs, monkeypatch, tmp_path):
    """The regression: point the resolver at an empty directory."""
    monkeypatch.setenv("TENGRI_SYNTHESIZER_AGN_GRID_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # so the data/ fallback cannot find a real grid

    # Escaped: the unescaped '.' would match any character, so the pattern would
    # pass on a filename this test does not mean to accept.
    with pytest.raises(TengriIOError, match=re.escape("test_grid_agn-nlr.hdf5")):
        SEDModel.build(ssp_data=ssp, observation=obs, **tengri.recipes.unified_agn())


def test_the_error_still_names_how_to_get_the_grid(ssp, obs, monkeypatch, tmp_path):
    """Moving a failure earlier must not cost the user the remedy.

    An error that fires at the right time but no longer says what to do is a
    worse trade than the late one it replaced.
    """
    monkeypatch.setenv("TENGRI_SYNTHESIZER_AGN_GRID_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TengriIOError) as excinfo:
        SEDModel.build(ssp_data=ssp, observation=obs, **tengri.recipes.unified_agn())

    message = str(excinfo.value)
    assert "TENGRI_SYNTHESIZER_AGN_GRID_DIR" in message
    assert "synthesizer-download" in message


def test_a_present_but_unreadable_grid_still_builds(ssp, obs, monkeypatch, tmp_path):
    """The other direction: the suppress must keep covering pre-warm failures.

    An empty file resolves (it exists) but cannot be read as HDF5. That is what
    ``contextlib.suppress`` is legitimately for — the lazy path is still
    correct — so the build must survive it. Without this test the obvious
    over-fix (propagating every exception) would pass the test above.
    """
    (tmp_path / "test_grid_agn-nlr.hdf5").write_bytes(b"")
    (tmp_path / "test_grid_agn-blr.hdf5").write_bytes(b"")
    monkeypatch.setenv("TENGRI_SYNTHESIZER_AGN_GRID_DIR", str(tmp_path))

    model = SEDModel.build(ssp_data=ssp, observation=obs, **tengri.recipes.unified_agn())
    assert model is not None


def test_recipes_without_a_grid_dependency_are_unaffected(ssp, obs, monkeypatch, tmp_path):
    """The guard must be scoped to the synthesizer blocks and nothing else."""
    monkeypatch.setenv("TENGRI_SYNTHESIZER_AGN_GRID_DIR", str(tmp_path))

    model = SEDModel.build(
        ssp_data=ssp, observation=obs, **tengri.recipes.star_forming_photometry()
    )
    assert model is not None
