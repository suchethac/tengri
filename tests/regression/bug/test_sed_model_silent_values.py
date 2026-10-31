# SPDX-License-Identifier: BSD-3-Clause
"""SEDModel paths that returned a wrong number without raising.

Each test pins one site where a missing or mis-merged input fell back to a
default instead of failing. The shared symptom is a plausible float out of a
path that should have refused:

* ``predict_hbeta`` returned a fake 1 Lsun on any ``AttributeError``,
  ``TypeError`` or ``ValueError``, not only on the invalid-params case it was
  written for.
* The deprecated ``predict_obs_sed(free_params)`` did not merge Fixed values,
  so a Fixed DLA redshift silently became 0 and the absorber sat at the source
  redshift.
* ``_representative_redshift`` returned the placeholder redshift in catalog
  mode, where the real redshift is per galaxy and the catalog range is the
  honest summary.
* ``_response_z_nodes`` built a table on [0.001, 3] when the redshift prior
  was missing, and fed ``inf`` into the ``ln(1+z)`` axis when it was open.
  A catalog range wider than the table also went unreported.

References
----------
.. [1] tengri forward-model hardening, silent-value audit (sed_model.py).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.config.exceptions import ParameterError, ZTableCoverageWarning
from tengri.forward.sed_model import WavePrecomp
from tengri.parameters.priors import Gaussian

pytestmark = pytest.mark.regression_bug


def _build(ssp, obs, *, redshift, approx=None, igm=None):
    kwargs = dict(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
        neb={"type": "none"},
        redshift=redshift,
    )
    if approx is not None:
        kwargs["approx"] = approx
    if igm is not None:
        kwargs["igm"] = igm
    return SEDModel.build(**kwargs)


def test_predict_hbeta_propagates_internal_type_error(
    synthetic_ssp_wide, synthetic_tophat_obs, monkeypatch
):
    """A TypeError from the SFH path must reach the caller, not become 1 Lsun.

    The safe fallback exists for invalid params. A broken internal call is a
    programming error, and returning a constant hides it.
    """
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, redshift=Fixed(0.1))

    def _broken(_params):
        raise TypeError("simulated internal type error")

    monkeypatch.setattr(model, "_predict_sfh_quantities", _broken)
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="simulated"):
        model.predict_hbeta({})


def test_deprecated_predict_obs_sed_honors_fixed_dla_redshift(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """A Fixed dla_z must reach the absorber through the deprecated shim.

    With dla_z=2.0 and a source at z=3.0, the absorber sits at 2.0. Dropping the
    Fixed value puts it at the source redshift, a different SED with no error.
    """
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        redshift=Fixed(3.0),
        igm={"type": "inoue", "dla": {"log_n_hi": Fixed(21.0), "z": Fixed(2.0)}},
    )
    full = dict(model.spec.sample(jax.random.PRNGKey(1)))
    free = {k: full[k] for k in model.spec.free_params}

    with pytest.warns(DeprecationWarning):
        legacy = np.asarray(model.predict_obs_sed(free).sed)
    current = np.asarray(model.predict(full).obs_sed())
    np.testing.assert_allclose(legacy, current, rtol=1e-10)


def test_representative_redshift_is_catalog_midpoint(synthetic_ssp_wide, synthetic_tophat_obs):
    """In catalog mode the representative z is the catalog range midpoint.

    The Fixed(0.1) here is a placeholder; the real z arrives per galaxy.
    """
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        redshift=Fixed(0.1),
        approx=WavePrecomp(catalog_z_range=(0.01, 1.0)),
    )
    assert model._representative_redshift() == pytest.approx(0.505)


def test_gaussian_redshift_prior_without_z_max_raises(synthetic_ssp_wide, synthetic_tophat_obs):
    """An open redshift prior with no WavePrecomp z_max has no table range to build.

    The old fallback silently used [0.001, 3]; the fix must name the knob.
    """
    with pytest.raises(ParameterError, match="z_max"):
        _build(
            synthetic_ssp_wide,
            synthetic_tophat_obs,
            redshift=Gaussian(0.5, 0.2, lo=0.0),
            approx=WavePrecomp(z_min=0.01),
        )


def test_gaussian_redshift_prior_with_explicit_z_range_builds(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The same open prior builds once the table range is stated explicitly."""
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        redshift=Gaussian(0.5, 0.2, lo=0.0),
        approx=WavePrecomp(z_min=0.01, z_max=3.0),
    )
    assert model._response_z_nodes() is not None


def test_catalog_range_wider_than_table_warns(synthetic_ssp_wide, synthetic_tophat_obs):
    """A catalog range wider than the WavePrecomp table must warn.

    Before, the warning was suppressed whenever a catalog range was set, so a
    catalog could run past the table without a word.
    """
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        redshift=Fixed(0.1),
        approx=WavePrecomp(catalog_z_range=(0.01, 3.0), z_min=0.01, z_max=1.0),
    )
    with pytest.warns(ZTableCoverageWarning):
        model._response_z_nodes()


def test_uniform_prior_is_unchanged_by_the_range_guard(synthetic_ssp_wide, synthetic_tophat_obs):
    """A finite declared prior still builds without an explicit z range."""
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        redshift=Uniform(0.05, 1.0),
        approx=WavePrecomp(),
    )
    assert model._response_z_nodes() is not None
