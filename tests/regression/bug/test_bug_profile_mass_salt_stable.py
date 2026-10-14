# SPDX-License-Identifier: BSD-3-Clause
"""The profiled-mass PRNG salt must not depend on the Python hash seed.

Bug: ``inference/mass_profile.py`` derived its ``fold_in`` salt from
``abs(hash("tengri.profile_mass"))``. ``str`` hashing is salted per process
(``PYTHONHASHSEED``), so the same PRNG key produced different mass draws across
runs. The salt is now a fixed ``zlib.crc32`` value.

The test runs the same draw in two interpreters with different hash seeds and
requires identical output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zlib

import pytest

pytestmark = pytest.mark.regression_bug

_SNIPPET = """
import jax
from tengri.inference.mass_profile import _PROFILE_MASS_SALT
key = jax.random.PRNGKey(7)
draws = jax.random.normal(jax.random.fold_in(key, _PROFILE_MASS_SALT), (4,))
print(_PROFILE_MASS_SALT, [repr(float(x)) for x in draws])
"""


def _run_with_hash_seed(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed, "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip().splitlines()[-1]


def test_salt_is_the_fixed_crc32_value():
    """The salt is crc32 of the tag, masked to the int32 domain fold_in expects."""
    from tengri.inference.mass_profile import _PROFILE_MASS_SALT

    assert zlib.crc32(b"tengri.profile_mass") & 0x7FFFFFFF == _PROFILE_MASS_SALT


def test_mass_draw_is_identical_across_hash_seeds():
    """Two interpreters with different PYTHONHASHSEED draw the same profiled-mass stream."""
    assert _run_with_hash_seed("1") == _run_with_hash_seed("2")
