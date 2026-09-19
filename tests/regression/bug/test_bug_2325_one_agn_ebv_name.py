# SPDX-License-Identifier: BSD-3-Clause
r"""
Regression test for issue #2325: one E(B-V) name, agn_ebv; agn_attenuation_ebv retired.

This tests that the retirement of the duplicate parameter name is enforced,
and that the surviving agn_ebv parameter works across all blocks that previously
read the old name.
"""

import pytest

import jax.numpy as jnp
from tengri import SEDModel, Fixed, Uniform, Parameters


pytestmark = pytest.mark.regression_bug


def test_agn_attenuation_ebv_flat_form_refused(synthetic_ssp_wide, synthetic_tophat_obs):
    """Building with the retired agn_attenuation_ebv at top level raises.

    The error must name agn_ebv (the surviving parameter) and reference #2325.
    """
    with pytest.raises(ValueError, match=r"agn_attenuation_ebv.*agn_ebv.*2325"):
        SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            redshift=Fixed(0.0),
            agn={
                "type": "composable",
                "atten": {"law": "prevot_smc"},
                "agn_attenuation_ebv": Fixed(0.3),  # Old name at wrong level
            },
        )


def test_agn_attenuation_ebv_dict_form_atten_subblock_refused():
    """Building with the retired agn_attenuation_ebv in dict form (atten sub-block) raises.

    The error must name agn_ebv (the surviving parameter) and reference #2325.
    """
    with pytest.raises(ValueError, match=r"agn_attenuation_ebv.*agn_ebv.*2325"):
        Parameters(
            redshift=Fixed(0.0),
            agn={
                "type": "composable",
                "atten": {"type": "smc_prevot", "agn_attenuation_ebv": 0.3},
            },
        )


def test_agn_attenuation_ebv_dict_form_agn_level_refused():
    """Building with the retired agn_attenuation_ebv at agn top level raises.

    The error must name agn_ebv (the surviving parameter) and reference #2325.
    """
    with pytest.raises(ValueError, match=r"agn_attenuation_ebv.*agn_ebv.*2325"):
        Parameters(
            redshift=Fixed(0.0),
            agn={
                "type": "composable",
                "atten": "smc_prevot",
                "agn_attenuation_ebv": 0.3,  # Old name at wrong level
            },
        )


def test_agn_ebv_survives_smc_prevot_block(synthetic_ssp_wide, synthetic_tophat_obs):
    """Building with agn_ebv (the surviving name) works for smc_prevot block.

    The parameter value must reach the prediction and change the output.
    """
    # Build model with agn_ebv on smc_prevot block (composable atten)
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        redshift=Fixed(0.0),
        agn={
            "type": "composable",
            "atten": {
                "type": "smc_prevot",
                "agn_ebv": Fixed(0.0),
            },
        },
    )

    # Predict with agn_ebv=0.0
    params_0 = model.spec.default_params()
    sed_0 = model.predict_photometry(params_0)

    # Predict with agn_ebv=0.3
    params_3 = model.spec.default_params()
    params_3["agn_ebv"] = 0.3
    sed_3 = model.predict_photometry(params_3)

    # The SEDs must differ (reddening reduces short-wavelength flux)
    assert not jnp.allclose(sed_0, sed_3), (
        f"SED with agn_ebv=0.0 did not differ from agn_ebv=0.3. "
        f"Parameter had no effect on smc_prevot block prediction."
    )


def test_agn_ebv_survives_qsogen_block(synthetic_ssp_wide, synthetic_tophat_obs):
    """Building with agn_ebv (the surviving name) works for qsogen block.

    The parameter value must reach the prediction and change the output.
    """
    # Build model with agn_ebv on qsogen attenuation block
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        redshift=Fixed(0.0),
        agn={
            "type": "composable",
            "atten": {
                "type": "qsogen",
                "agn_ebv": Fixed(0.0),
            },
        },
    )

    # Predict with agn_ebv=0.0
    params_0 = model.spec.default_params()
    sed_0 = model.predict_photometry(params_0)

    # Predict with agn_ebv=0.3
    params_3 = model.spec.default_params()
    params_3["agn_ebv"] = 0.3
    sed_3 = model.predict_photometry(params_3)

    # The SEDs must differ (reddening reduces short-wavelength flux)
    assert not jnp.allclose(sed_0, sed_3), (
        f"SED with agn_ebv=0.0 did not differ from agn_ebv=0.3. "
        f"Parameter had no effect on qsogen block prediction."
    )


def test_agn_attenuation_ebv_absent_from_src():
    """Verify the retired name does not appear in any src/ consumer code.

    The census walks src/tengri/**/*.py, counts occurrences of the literal
    'agn_attenuation_ebv' outside of comments, and subtracts lines inside the
    retirement block (defined by _RETIRED_AGN_ATTEN_EBV and _agn_atten_ebv_retired_error).
    The remainder must be 0. Comments documenting the retirement are allowed.
    """
    from pathlib import Path

    # Locate the retirement block boundaries in groups.py
    groups_py = Path("src/tengri/parameters/groups.py")
    with open(groups_py, "r") as f:
        lines = f.readlines()

    # Find the block boundaries: from first _RETIRED_AGN_ATTEN_EBV to end of _agn_atten_ebv_retired_error
    retirement_block_start = None
    retirement_block_end = None

    for i, line in enumerate(lines):
        # Find FIRST occurrence of the frozenset definition (not later uses)
        if "_RETIRED_AGN_ATTEN_EBV" in line and "frozenset" in line and retirement_block_start is None:
            retirement_block_start = i
        # Find the function definition
        if "_agn_atten_ebv_retired_error" in line and "def " in line:
            # Found the function definition; find its end (next def or class at same indent)
            base_indent = len(line) - len(line.lstrip())
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip() and (next_line.startswith("def ") or next_line.startswith("class ")):
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= base_indent:
                        retirement_block_end = j
                        break
            if retirement_block_end is None:
                retirement_block_end = len(lines)

    # Walk src/tengri/**/*.py and count offenders
    offenders = []
    src_root = Path("src/tengri")
    for py_file in sorted(src_root.rglob("*.py")):
        with open(py_file, "r") as f:
            file_lines = f.readlines()

        for line_num, line_content in enumerate(file_lines, start=1):
            if "agn_attenuation_ebv" not in line_content:
                continue

            # Skip comment lines (documentation of retirement is allowed)
            # Find the first occurrence of '#' that marks a comment (not within strings)
            stripped = line_content.lstrip()
            if stripped.startswith("#"):
                # Entire line is a comment
                continue

            # Check if this line is inside the retirement block (only skip if in groups.py)
            if py_file.name == "groups.py" and retirement_block_start is not None:
                if retirement_block_start <= (line_num - 1) < retirement_block_end:
                    continue

            # Record the offender
            offenders.append(f"{py_file}:{line_num}: {line_content.rstrip()}")

    assert (
        not offenders
    ), f"Found {len(offenders)} unexpected references to agn_attenuation_ebv in src/:\n" + "\n".join(
        offenders
    )
