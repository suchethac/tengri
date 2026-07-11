# SPDX-License-Identifier: BSD-3-Clause
"""A broken ``grid_path`` must fail loudly, never as zero AGN emission.

Background. :class:`SKIRTORTorus` has two distinct "no grid" situations and they
are not the same thing:

1. ``grid_path`` is **unset** — the component was constructed without a torus
   library. Returning ``None`` (and therefore zero emission) is the documented,
   intended behavior: nothing was asked for, nothing is delivered.

2. ``grid_path`` **is set but cannot be loaded** — the file is missing, is not
   readable, or lacks the expected datasets. The user explicitly asked for a
   torus and named a file. Silently degrading this to zero emission produces a
   fit in which the AGN torus contributes *exactly nothing* while every
   torus parameter (``agn_tau_skirtor``, ``agn_oa_skirtor``, …) is a no-op — a
   scientifically wrong answer with no error and no warning.

``load`` previously caught ``(FileNotFoundError, OSError, KeyError)`` and
returned ``None`` for **both** cases, collapsing (2) into (1). These tests pin
the distinction.

Note the composable build path (``SEDModel.build(agn={'torus': 'skirtor'})``)
was never affected: it resolves the grid through
``skirtor._find_skirtor_grid()``, which already raises. The silent path is
reachable only by constructing :class:`SKIRTORTorus` directly, which is the
expert escape hatch documented in its class docstring.
"""

import pytest

from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig

pytestmark = pytest.mark.contract


def test_unset_grid_path_returns_none():
    """Case 1: no grid requested -> None (zero emission) is intended."""
    torus = SKIRTORTorus(config=SKIRTORTorusConfig(grid_path=None))
    assert torus.load(None) is None


def test_missing_grid_file_raises_rather_than_zeroing(tmp_path):
    """Case 2: a named-but-absent grid must raise, not silently zero out."""
    missing = tmp_path / "no_such_skirtor_grid.h5"
    torus = SKIRTORTorus(config=SKIRTORTorusConfig(grid_path=str(missing)))

    with pytest.raises(FileNotFoundError) as excinfo:
        torus.load(None)

    # The message must name the offending path, so the user can spot a typo.
    assert str(missing) in str(excinfo.value)


def test_unreadable_grid_file_raises_rather_than_zeroing(tmp_path):
    """A file that exists but is not a usable grid must also raise."""
    junk = tmp_path / "not_really_a_grid.h5"
    junk.write_bytes(b"this is not an HDF5 file")
    torus = SKIRTORTorus(config=SKIRTORTorusConfig(grid_path=str(junk)))

    with pytest.raises((FileNotFoundError, OSError, KeyError)):
        torus.load(None)
