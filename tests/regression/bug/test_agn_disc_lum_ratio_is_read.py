# SPDX-License-Identifier: BSD-3-Clause
"""The declared ``lum_ratio`` drives the KD18 and power-law disc models.

Both disc classes declare ``lum_ratio`` as a free parameter, but their ``predict`` once
read a ``frac`` key that no caller supplied, so the public ``apply`` path raised
``KeyError`` for any configuration and a user's ``agn_lum_ratio`` could not reach the
model. The declared name must drive the disc: a different ``lum_ratio`` must change both
the spectrum and the published disc luminosity ``log_L_agn_disc`` [dex re erg/s].
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.kd18_disc_model import KD18Disc
from tengri.components.agn.powerlaw_disc_model import PowerLawDisc
from tengri.forward.orchestrator import default_params_dict
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.regression_bug

_WAVE = jnp.asarray(np.geomspace(1.0e3, 1.0e7, 400))


def _apply(component, lum_ratio):
    params = dict(default_params_dict([component]))
    params["agn_lum_ratio"] = jnp.asarray(lum_ratio)
    return component.apply(ForwardState(wave=_WAVE), params)


@pytest.mark.parametrize("component_cls", [KD18Disc, PowerLawDisc], ids=["kd18", "powerlaw"])
def test_changing_lum_ratio_changes_the_disc_output(component_cls):
    """lum_ratio 0.1 and 0.5 give different spectra and a larger published disc power."""
    low = _apply(component_cls(), 0.1)
    high = _apply(component_cls(), 0.5)

    log_l_low = float(low.derived["log_L_agn_disc"])
    log_l_high = float(high.derived["log_L_agn_disc"])
    assert np.isfinite(log_l_low) and np.isfinite(log_l_high)
    assert log_l_high > log_l_low + 1e-6, (
        f"{component_cls.__name__}: lum_ratio has no effect on log_L_agn_disc "
        f"({log_l_low:.6f} vs {log_l_high:.6f} dex re erg/s)"
    )
    assert not np.allclose(np.asarray(low.sed_intrinsic), np.asarray(high.sed_intrinsic))
