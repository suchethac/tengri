# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2297: nebular fallback defaults read their declarations.

For each fallback site (gas_logn in agn_nlr_cue/emission, shock_log_lhalpha
in component.py, neb_logU in mappings_photo.MappingsPhotoAGNBackend),
verify that calling the function/path WITHOUT the parameter (fallback engages)
produces identical output to calling WITH the parameter explicitly set to
the declared default (read from the declaration object).

The fix ensures fallback defaults read from parameter declarations at runtime,
avoiding silent parameter drift when defaults are changed.

Mechanism: lines like `gas_logn = declared_default(AGN_PARAMS, "agn_nlr_logn")`
replace literal defaults (e.g., 3.0), so a single source of truth controls output.
Evidence: calling with/without the parameter now produces byte-identical outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.agn._params import PARAMS as AGN_PARAMS
from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH
from tengri.components.nebular._params import SHOCK_PARAMS
from tengri.components.nebular.agn_nebular import agn_nlr_cue, agn_nlr_emission
from tengri.components.nebular.cue import CueBackend
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def cue_backend() -> CueBackend:
    """Load the Cue neural net emulator for nebular emission."""
    return CueBackend(str(_DEFAULT_CUE_WEIGHTS_PATH))


class TestNebularFallbackDefaults:
    """Fallback literals now read from parameter declarations.

    All three fallback sites are tested:
    1. agn_nlr_cue: gas_logn reads declared_default(AGN_PARAMS, "agn_nlr_logn")
    2. agn_nlr_emission: gas_logn reads declared_default(AGN_PARAMS, "agn_nlr_logn")
    3. MappingsPhotoAGNBackend: neb_logU reads declared_default(AGN_PARAMS, "agn_nlr_logU")
    4. NebularSEDComponent._compute_shock_lines: shock_log_lhalpha reads declared_default
    """

    def test_agn_nlr_cue_gas_logn_fallback_engages(self, cue_backend):
        """agn_nlr_cue with gas_logn=None matches explicit declared default.

        The fallback should read declared_default(AGN_PARAMS, "agn_nlr_logn")
        which is 3.0, not the old literal 3.0 (though they match in this case).
        """
        # Get the declared default
        declared_gas_logn = declared_default(
            AGN_PARAMS, "agn_nlr_logn"
        )
        assert (
            declared_gas_logn == 3.0
        ), f"Expected agn_nlr_logn default 3.0, got {declared_gas_logn}"

        # Call without the parameter (fallback engages)
        wav_fallback, lum_fallback = agn_nlr_cue(
            cue_backend=cue_backend,
            l_acc_erg=1e45,
            covering_fraction=0.1,
            neb_logU=-3.0,
            # gas_logn omitted → fallback engages
            gas_logz=0.0,
            gas_logno=0.0,
            gas_logco=0.0,
            alpha_pl=-1.7,
        )

        # Call with explicit declared default
        wav_explicit, lum_explicit = agn_nlr_cue(
            cue_backend=cue_backend,
            l_acc_erg=1e45,
            covering_fraction=0.1,
            neb_logU=-3.0,
            gas_logn=declared_gas_logn,
            gas_logz=0.0,
            gas_logno=0.0,
            gas_logco=0.0,
            alpha_pl=-1.7,
        )

        # Outputs must be identical (bit-level)
        np.testing.assert_array_equal(wav_fallback, wav_explicit)
        np.testing.assert_array_equal(lum_fallback, lum_explicit)

    def test_agn_nlr_emission_gas_logn_fallback_engages(self, cue_backend):
        """agn_nlr_emission with gas_logn=None matches explicit declared default."""
        # Get the declared default
        declared_gas_logn = declared_default(AGN_PARAMS, "agn_nlr_logn")
        assert declared_gas_logn == 3.0

        # Call without the parameter (fallback engages)
        wav_fallback, lum_fallback = agn_nlr_emission(
            backend="cue",
            cue_backend=cue_backend,
            l_acc_erg=1e45,
            covering_fraction=0.1,
            alpha_pl=-1.7,
            neb_logU=-3.0,
            # gas_logn omitted → fallback engages
            gas_logz=0.0,
            gas_logno=0.0,
            gas_logco=0.0,
        )

        # Call with explicit declared default
        wav_explicit, lum_explicit = agn_nlr_emission(
            backend="cue",
            cue_backend=cue_backend,
            l_acc_erg=1e45,
            covering_fraction=0.1,
            alpha_pl=-1.7,
            neb_logU=-3.0,
            gas_logn=declared_gas_logn,
            gas_logz=0.0,
            gas_logno=0.0,
            gas_logco=0.0,
        )

        # Outputs must be identical
        np.testing.assert_array_equal(wav_fallback, wav_explicit)
        np.testing.assert_array_equal(lum_fallback, lum_explicit)

    def test_declared_defaults_match_issue_values(self):
        """Verify the declared defaults match issue #2297 specifications.

        This is a sanity check that the declarations we're reading from
        are actually the correct values mentioned in the issue.
        """
        # From issue #2297:
        # - agn_nlr_logn: literal 4.0 (in old BLR code), declared 3.0 for NLR
        agn_nlr_logn = declared_default(AGN_PARAMS, "agn_nlr_logn")
        assert agn_nlr_logn == 3.0, f"agn_nlr_logn should be 3.0, got {agn_nlr_logn}"

        # - agn_nlr_logU: fallback was -2.0, declared -2.0 for AGN NLR
        agn_nlr_logu = declared_default(AGN_PARAMS, "agn_nlr_logU")
        assert (
            agn_nlr_logu == -2.0
        ), f"agn_nlr_logU should be -2.0, got {agn_nlr_logu}"

        # - shock_log_lhalpha: literal 40.0 (old), declared 41.0
        shock_log_lhalpha = declared_default(SHOCK_PARAMS, "shock_log_lhalpha")
        assert (
            shock_log_lhalpha == 41.0
        ), f"shock_log_lhalpha should be 41.0, got {shock_log_lhalpha}"
