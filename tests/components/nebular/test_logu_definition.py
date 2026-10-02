# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for `neb_logU` definition and documentation.

Tests validate that `neb_logU` is the inner-face ionization parameter at
R = 10^19 cm (not the Strömgren radius or volume-averaged ⟨U⟩ of Gutkin et al.
2016), and that docstring statements accurately reflect this definition.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.nebular.cue import _logq_from_logu
from tengri.utils.physics_constants import C_CGS

pytestmark = [pytest.mark.unit, pytest.mark.regression_bug]


def test_logq_from_logu_is_the_inner_face_relation():
    """Test that _logq_from_logu correctly implements U = Q_H / (4π R² n_H c).

    This test is a regression for issue #2632: it pins the definition that
    neb_logU is the ionization parameter at the inner face R = 10^19 cm.
    The test passes on main; it confirms the current definition is correctly
    implemented.
    """
    test_cases = [
        (-3, 2),
        (-2, 2),
        (-1, 3),
    ]

    for logU, log_n in test_cases:
        U = 10.0**logU
        n_H = 10.0**log_n
        R = 1e19  # cm
        log_R = 19.0  # log10(R / cm)

        # The relation: Q_H = 4π R² n_H c U
        Q_H_expected = 4.0 * np.pi * R**2 * n_H * C_CGS * U
        logQ_expected = np.log10(Q_H_expected)

        # Compute via _logq_from_logu: logQ = logU + log(4π·c) + 2·log(R) + log(n_H)
        # where log_R = log10(R) = 19 → 2·log10(R) = 38
        logQ_computed = _logq_from_logu(logU, log_n, log_R=log_R)

        np.testing.assert_allclose(
            logQ_computed,
            logQ_expected,
            rtol=1e-6,
            err_msg=f"Failed for logU={logU}, log_n={log_n}",
        )


def test_docstring_mapping_to_volume_averaged_u():
    """Assert the numeric mapping to Gutkin et al. (2016) ⟨U⟩ and that the docstring states it.

    Checks the arithmetic (log ⟨U⟩ = -2 -> neb_logU = -2.35; neb_logU = -2 ->
    log ⟨U⟩ = -1.88), then that the _logq_from_logu docstring quotes both
    numbers, says "inner-face" and does not mention the Strömgren radius.

    The docstring of _logq_from_logu must state the relation between the
    inner-face U (what neb_logU is) and the volume-averaged ⟨U⟩ used by
    Synthesizer. This test asserts the specific numbers cited in the issue:
    at n_H = 100 cm⁻³, log ⟨U⟩ = -2 corresponds to neb_logU = -2.35,
    and neb_logU = -2 corresponds to log ⟨U⟩ = -1.88.

    See Gutkin et al. (2016) Eq. 1: ⟨U⟩ = (3 Q n α_B² ε² / 4π c³)^(1/3)
    with ε = 1 (spherical).
    """
    n_H = 100.0  # cm^-3
    alpha_B = 2.59e-13  # cm^3 s^-1, recombination coefficient

    # Test case 1: log ⟨U⟩ = -2 → compute Q from Gutkin eq. 1, then neb_logU
    log_U_avg_1 = -2.0
    U_avg_1 = 10.0**log_U_avg_1

    # From Gutkin: Q = 4π c³ U_avg^3 / (3 n α_B²)
    Q_1 = (4.0 * np.pi * C_CGS**3 * U_avg_1**3) / (3.0 * n_H * alpha_B**2)
    logQ_1 = np.log10(Q_1)

    # Compute neb_logU from Q using inner-face relation
    R = 1e19  # cm
    U_inner_1 = Q_1 / (4.0 * np.pi * R**2 * n_H * C_CGS)
    logU_inner_1 = np.log10(U_inner_1)

    # Assert: logU_inner should be -2.35 to 0.01
    np.testing.assert_allclose(
        logU_inner_1, -2.35, atol=0.01, err_msg="log ⟨U⟩ = -2 should map to neb_logU ≈ -2.35"
    )

    # Test case 2: neb_logU = -2 → compute Q from inner relation, then log ⟨U⟩
    logU_inner_2 = -2.0
    U_inner_2 = 10.0**logU_inner_2

    # Inner-face: Q = 4π R² n c U
    Q_2 = 4.0 * np.pi * R**2 * n_H * C_CGS * U_inner_2
    logQ_2 = np.log10(Q_2)

    # Compute volume-averaged U from Q using Gutkin eq. 1
    U_avg_2 = (3.0 * Q_2 * n_H * alpha_B**2 / (4.0 * np.pi * C_CGS**3)) ** (1.0 / 3.0)
    log_U_avg_2 = np.log10(U_avg_2)

    # Assert: log ⟨U⟩ should be -1.88 to 0.01
    np.testing.assert_allclose(
        log_U_avg_2, -1.88, atol=0.01, err_msg="neb_logU = -2 should map to log ⟨U⟩ ≈ -1.88"
    )

    # Assert the docstring of _logq_from_logu accurately documents the inner-face definition.
    docstring = _logq_from_logu.__doc__ or ""

    # (a) The docstring should not mention "stromgren" or "strömgren" (case-insensitive).
    assert "stromgren" not in docstring.lower() and "strömgren" not in docstring.lower(), (
        "Docstring should not mention Strömgren radius; neb_logU is the inner-face parameter"
    )

    # (b) The docstring must quote the mapping numbers −2.35 and −1.88 (with Unicode minus).
    assert "−2.35" in docstring, (
        "Docstring must cite the mapping −2.35 (with Unicode minus) for log ⟨U⟩ = −2"
    )
    assert "−1.88" in docstring, (
        "Docstring must cite the mapping −1.88 (with Unicode minus) for neb_logU = −2"
    )

    # (c) The docstring must mention "inner-face" (case-insensitive).
    assert "inner-face" in docstring.lower(), (
        "Docstring should mention 'inner-face' to clarify the definition of neb_logU"
    )


def test_converter_describes_a_relative_metallicity_axis():
    """Assert the FSPS grid converter describes log_met as log10(Z / Z_sun).

    Regression for issue #2633 item 1: scripts/convert_fsps_cloudy_grid.py must
    not call the axis "absolute metallicity" and must mention Z_sun.
    """
    import pathlib
    import subprocess

    # Read the converter script as text
    # Use git to find the repo root
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    )
    repo_root = pathlib.Path(result.stdout.strip())
    converter_path = repo_root / "scripts" / "convert_fsps_cloudy_grid.py"

    with open(converter_path) as f:
        converter_text = f.read()

    # Parse for the description strings for log_met axis
    # Both (lines and continuum) should describe log_met as relative, not absolute
    assert "absolute metallicity" not in converter_text.lower(), (
        "Converter should not describe log_met as 'absolute metallicity'"
    )
    # Verify the converter describes Z/Z_sun at least once
    assert "Z / Z_sun" in converter_text or "Z_sun" in converter_text, (
        "Converter must describe the metallicity axis as log10(Z / Z_sun) or mention Z_sun"
    )
