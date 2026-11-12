# SPDX-License-Identifier: BSD-3-Clause
"""A failed line-catalog publish and a missing PAH data file raise (#3284).

Both failures used to warn and carry on: the nebular line publish dropped
``line_waves`` / ``line_lums`` silently, and the Draine+2021 PAH component
returned no templates. A requested component whose data cannot be read is an
error, not an advisory.

Taxonomy marker: regression_bug
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.nebular._shared import NebularBackendDefaults
from tengri.config.exceptions import BackendError, TengriIOError

pytestmark = pytest.mark.regression_bug


class _PublishFailsBackend(NebularBackendDefaults):
    """Stand-in photoionization backend whose line catalog publish raises."""

    lsun_erg = 3.828e33

    def predict_nebular_sed(self, *args, **kwargs):
        """Return a zero continuum on the requested wavelength grid."""
        return jnp.zeros_like(kwargs["ssp_wave"])

    def predict_nebular_line_luminosities(self, *args, **kwargs):
        """Raise an arbitrary, non-documented failure."""
        raise RuntimeError("line table corrupt")


def test_failed_line_publish_raises_backend_error():
    """A line publish that raises propagates as BackendError naming the backend."""
    from tengri.components.nebular.component import (
        NebularSEDComponent,
        NebularSEDComponentConfig,
    )
    from tengri.protocols.component import ForwardState

    comp = NebularSEDComponent(
        config=NebularSEDComponentConfig(backend="mappings"),
        backend=_PublishFailsBackend(),
    )
    state = ForwardState(wave=jnp.linspace(1000.0, 9000.0, 64))
    state = state.with_(
        derived=state.derived.with_(
            ssp_ages_yr=jnp.array([1e6, 1e7, 1e8]),
            age_weights=jnp.array([1.0, 1.0, 1.0]),
        )
    )
    params = {
        d.name: d.prior.default
        for d in comp.declared_parameters()
        if getattr(d.prior, "default", None) is not None
    }
    params["met_logzsol"] = 0.0

    with pytest.raises(BackendError, match="'mappings'") as info:
        comp.apply(state, params)

    assert isinstance(info.value.__cause__, RuntimeError)
    assert "line table corrupt" in str(info.value.__cause__)


def test_missing_pahspec_grid_raises_tengri_io_error(monkeypatch, tmp_path):
    """A requested PAH component with no template file raises, naming the file."""
    from tengri.components.dust.draine2021_pah import PAHSPEC_PATH_ENV
    from tengri.components.dust.draine2021_pah_ir import (
        Draine2021PAHIRConfig,
        Draine2021PAHIRSEDComponent,
    )

    missing = tmp_path / "no_such_pahspec.h5"
    monkeypatch.setenv(PAHSPEC_PATH_ENV, str(missing))
    comp = Draine2021PAHIRSEDComponent(config=Draine2021PAHIRConfig(template_path=str(missing)))

    with pytest.raises(TengriIOError, match=r"no_such_pahspec\.h5") as info:
        comp.precompute(wave_grid=jnp.linspace(1000.0, 9000.0, 64))

    assert info.value.__cause__ is not None
