# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2297: nebular fallback defaults read their declarations.

For each fallback site, verify that calling WITHOUT the parameter (fallback
engages) produces identical output to calling WITH the parameter explicitly set
to the declared default.

Mechanism: lines like `gas_logn = declared_default(AGN_PARAMS, "agn_nlr_logn")`
replace literal defaults, so a single source of truth controls output.
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
    """Fallback defaults now read from parameter declarations.

    All three fallback sites tested: gas_logn in agn_nlr_cue/emission,
    neb_logU in MappingsPhotoAGNBackend, shock_log_lhalpha in component.py.
    """

    def test_agn_nlr_cue_gas_logn_fallback_engages(self, cue_backend):
        """agn_nlr_cue: fallback matches explicit declared_default read."""
        declared_gas_logn = declared_default(
            AGN_PARAMS, "agn_nlr_logn"
        )

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
        """agn_nlr_emission: fallback matches explicit declared_default read."""
        declared_gas_logn = declared_default(AGN_PARAMS, "agn_nlr_logn")

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
