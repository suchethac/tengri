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
    """Verify the retired name does not appear in any src/ consumer.

    The census must show exactly 0 consumers of agn_attenuation_ebv in src/tengri,
    except for the retirement frozenset itself and the interception helper.
    """
    import subprocess
    import re

    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "agn_attenuation_ebv",
            "--",
            "src/",
        ],
        cwd="/Users/suchethacooray/Projects/tengri/.claude/worktrees/agent-a28a5d918234296f6",
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

    # Allowed: _RETIRED_AGN_ATTEN_EBV frozenset and the helper function
    allowed_patterns = [
        r"_RETIRED_AGN_ATTEN_EBV",
        r"_agn_atten_ebv_retired_error",
    ]

    filtered_lines = []
    for line in lines:
        if not line:
            continue
        # Skip lines that match allowed patterns (the retirement infrastructure)
        if any(re.search(pattern, line) for pattern in allowed_patterns):
            continue
        filtered_lines.append(line)

    assert (
        not filtered_lines
    ), f"Found {len(filtered_lines)} unexpected references to agn_attenuation_ebv in src/:\n" + "\n".join(
        filtered_lines
    )
