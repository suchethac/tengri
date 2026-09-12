# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2295 — reversed-operand trapezoid integration.

Root cause: On Apple GPU via jax-mps under MLX compile, a reversed array combined
with a broadcast scalar is silently zeroed past element 0 (upstream jax-mps#232,
issue #2295). This affects ``jnp.trapezoid`` which multiplies by 0.5 internally.
The mathematical identity ``trapezoid(y[::-1], x[::-1]) == -trapezoid(y, x)``
holds exactly term-by-term; the fix is to integrate over the descending grid
and negate the result rather than reverse operands.

Four sites were affected:
  - AGN polar dust anisotropic luminosity: ``polar_dust.py``
  - AGN ADAF normalizations (float32 and float64 branches): ``adaf.py`` (2 sites)
  - AGN unified disc luminosity: ``unified.py``

https://github.com/suchethac/tengri/issues/2295
https://github.com/tillahoffmann/jax-mps/issues/232
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.agn.polar_dust import anisotropic_polar_luminosity

pytestmark = pytest.mark.regression_bug


def test_negated_descending_equals_reversed_identity():
    """Float64 identity: -trapezoid(y, x) == trapezoid(y[::-1], x[::-1]) exactly.

    nu is descending (frequency = c / wavelength). The mathematical identity
    holds exactly term-by-term. Float64 differs only by summation order.
    """
    # Synthetic wavelength grid (ascending, as is typical)
    wavelength = jnp.logspace(2, 8, 400)  # [Angstrom]

    # Frequency grid (descending because nu = c / lambda, lambda ascending)
    c_aa = 2.998e18  # [Angstrom / s]
    nu_desc = c_aa / wavelength  # descending

    # Synthetic SED (power law)
    y_desc = 1e28 * (wavelength / 1e4) ** -1.5

    # Test both orderings: descending as-is, and ascending (reversed)
    x_desc = nu_desc
    y_asc = y_desc[::-1]
    x_asc = nu_desc[::-1]

    # Identity: -trapz(y, x descending) == trapz(y reversed, x reversed)
    integral_negated = -jnp.trapezoid(y_desc, x_desc)
    integral_reversed = jnp.trapezoid(y_asc, x_asc)

    # Float64 comparison with rel tolerance and explicit abs floor
    magnitude = jnp.abs(integral_reversed)
    abs_floor = 1e-12 * magnitude
    np.testing.assert_allclose(
        integral_negated,
        integral_reversed,
        rtol=1e-12,
        atol=abs_floor,
        err_msg="Negated descending != reversed identity",
    )


def test_trapezoid_identity_across_magnitudes():
    """Trapezoid identity holds at different magnitude scales.

    Tests that -trapz(y, x descending) == trapz(y[::-1], x[::-1]) for spectra
    at ADAF-typical magnitudes (1e20 to 1e28 erg/s/Hz).
    """
    c_aa = 2.998e18
    wavelength = jnp.logspace(2, 8, 400)
    nu = c_aa / wavelength

    # Test at typical ADAF spectrum magnitudes
    for magnitude in [1e20, 1e24, 1e28]:
        y = magnitude * (wavelength / 1e4) ** -1.2

        # Compute both forms
        integral_negated = -jnp.trapezoid(y, nu)
        integral_reversed = jnp.trapezoid(y[::-1], nu[::-1])

        # Must match to float64 precision
        np.testing.assert_allclose(
            integral_negated,
            integral_reversed,
            rtol=1e-12,
            atol=1e-12 * jnp.abs(integral_reversed),
            err_msg=f"Identity failed at magnitude {magnitude}",
        )


def test_anisotropic_polar_luminosity_matches_reversed_reference():
    """The rewritten site equals the pre-#2295 reversed-operand form to 1e-12.

    The reference is the old form (reversed operands), exact on CPU; only
    MLX-compiled Apple GPU miscomputes it (jax-mps#232).
    """
    from tengri.components.agn.polar_dust import _C_AA

    wavelength = jnp.logspace(2, 7, 300)  # [Angstrom]
    l_nu_disk = 1e28 * (wavelength / 1e4) ** -0.5
    extinction_factor = jnp.clip(jnp.exp(-((1e4 / wavelength) ** 1.1)), 0.0, 1.0)
    opening_angle_deg = 40.0

    result = anisotropic_polar_luminosity(
        l_nu_disk=l_nu_disk,
        wavelength=wavelength,
        opening_angle_deg=opening_angle_deg,
        extinction_factor=extinction_factor,
    )

    # Pre-#2295 reference: reversed-operand trapezoid, same geometry factor.
    sin_oa = jnp.sin(jnp.radians(opening_angle_deg))
    aniso_factor = 7.0 / 18.0 - sin_oa**2 / 6.0 - (2.0 / 9.0) * sin_oa**3
    l_nu_absorbed = l_nu_disk * (1.0 - extinction_factor)
    nu = _C_AA / wavelength
    reference = jnp.maximum(aniso_factor * jnp.trapezoid(l_nu_absorbed[::-1], nu[::-1]), 0.0)

    np.testing.assert_allclose(
        float(result),
        float(reference),
        rtol=1e-12,
        atol=1e-12 * abs(float(reference)),
        err_msg="rewritten site diverged from the pre-#2295 reference form",
    )
    assert result > 0.0


def test_no_reversed_trapezoid_operands_in_src():
    """Source scan: no reversed-array operands passed to trapezoid().

    This test scans every .py file under the installed tengri package for
    trapezoid calls and asserts that neither operand contains [::-1],
    jnp.flip, np.flip, or lax.rev.

    Reversed operands are silently zeroed on Apple GPU (jax-mps#232, #2295).
    Use -trapezoid(y, x) instead to integrate over a descending grid.
    """
    import re

    # Locate the installed tengri package
    pkg_path = Path(tengri.__file__).resolve().parent

    # Find all .py files under the package
    py_files = list(pkg_path.rglob("*.py"))

    failed_sites = []

    for py_file in py_files:
        with open(py_file) as f:
            content = f.read()

        # Find all trapezoid calls (basic regex; assumes balanced parens)
        trapezoid_pattern = r"trapezoid\s*\("
        for match in re.finditer(trapezoid_pattern, content):
            start = match.start()

            # Extract the argument list by counting parentheses
            paren_depth = 0
            arg_start = match.end()
            arg_end = arg_start

            for i, char in enumerate(content[arg_start:]):
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    if paren_depth == 0:
                        arg_end = arg_start + i
                        break
                    paren_depth -= 1

            arg_span = content[arg_start:arg_end]

            # Check for forbidden patterns
            forbidden = ["[::-1]", "jnp.flip(", "np.flip(", "lax.rev"]
            violations = [p for p in forbidden if p in arg_span]

            if violations:
                # Find the line number for reporting
                line_num = content[:start].count("\n") + 1
                failed_sites.append(
                    (
                        str(py_file.relative_to(pkg_path.parent)),
                        line_num,
                        arg_span[:100],
                        violations,
                    )
                )

    if failed_sites:
        msg = (
            "Found reversed-operand trapezoid calls (jax-mps#232, #2295):\n"
            "Use -trapezoid(y, x) instead to integrate over a descending grid.\n\n"
        )
        for file, line, args, viols in failed_sites:
            msg += f"{file}:{line}\n  Arguments: {args}\n  Violations: {viols}\n"
        pytest.fail(msg)
