# SPDX-License-Identifier: BSD-3-Clause
"""Per-galaxy emission-line limits at catalog scale (#1469).

A single galaxy can express a non-detection since #1460::

    Data(lines={"Halpha": (flux, err, "upper")})

A catalog could not. ``Catalog`` took ``line_cols`` / ``line_err_cols`` but no
censor column, and ``censor_cols`` is photometry-only -- it is validated
against the band axis. So a DESI catalog in which some galaxies detect Halpha
and others do not had to be fit as if every line were a detection, pulling
those fits toward flux the galaxies demonstrably do not have.

The Observation-level flag was never the answer: it is one flag for the whole
catalog, and non-detection is a per-galaxy property.

Convention matches the photometric ``censor_cols``: 0 detected, 1 upper limit,
-1 lower limit, booleans rejected.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

pytestmark = pytest.mark.regression_bug


def _catalog(**kw):
    return build_two_galaxy_catalog(halpha=(1.0, 4.0), **kw)


def test_catalog_accepts_line_censor_cols(synthetic_ssp_wide, synthetic_tophat_obs):
    """The column mapping is accepted and lands on the ingested arrays."""
    cat, _ = _catalog(
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
        line_censor=[0, 1],
        line_censor_cols=["halpha_limit"],
    )

    ca = cat._catalog_arrays
    assert ca.line_censor is not None, "line_censor_cols was passed but nothing was ingested"
    assert ca.line_censor.shape == (2, 1), f"expected (2, 1), got {ca.line_censor.shape}"
    assert int(ca.line_censor[0, 0]) == 0
    assert int(ca.line_censor[1, 0]) == 1


def test_upper_limit_changes_that_galaxys_fit(synthetic_ssp_wide, synthetic_tophat_obs):
    """Marking a line an upper limit must change the fit for that galaxy.

    Same key, same galaxy index, same flux value -- only the censor flag
    differs. A flag that reaches nothing would leave the fit bit-identical,
    which is how #1460 hid at the single-galaxy seam.
    """
    common = dict(ssp=synthetic_ssp_wide, obs_base=synthetic_tophat_obs)
    detected, _ = _catalog(**common, line_censor=[0, 0], line_censor_cols=["halpha_limit"])
    censored, _ = _catalog(**common, line_censor=[0, 1], line_censor_cols=["halpha_limit"])

    key = jax.random.PRNGKey(0)
    a = detected.fit(method="map", key=key, n_steps=40)
    b = censored.fit(method="map", key=key, n_steps=40)

    def params(post, i):
        return {k: float(np.asarray(v)) for k, v in post[i].params.items()}

    changed = max(abs(params(a, 1)[k] - params(b, 1)[k]) for k in params(a, 1))
    untouched = max(abs(params(a, 0)[k] - params(b, 0)[k]) for k in params(a, 0))

    assert changed > 1e-10, (
        "galaxy 1's fit is unchanged when its Halpha is marked an upper limit "
        f"(max delta {changed:g}) -- the censor flag reaches nothing and the "
        "non-detection is being fit as a measurement"
    )
    assert untouched == 0.0, (
        "galaxy 0 was flagged 'detected' in both catalogs but its fit moved "
        f"(max delta {untouched:g}) -- the censor flag is not per-galaxy"
    )


def test_batched_engine_refuses_rather_than_dropping_limits(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """A batched method must say it cannot honor the flags, not ignore them.

    The vmapped engines compile one loss for the whole catalog and carry no
    per-galaxy limit mask. Staying quiet would fit every non-detection as a
    measurement -- the exact failure #1460 and #1599 closed elsewhere.
    """
    cat, _ = _catalog(
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
        line_censor=[0, 1],
        line_censor_cols=["halpha_limit"],
    )

    with pytest.raises(NotImplementedError, match=r"(?i)line_censor_cols"):
        cat.fit(method="mcmc_nuts", key=jax.random.PRNGKey(0), n_warmup=4, n_samples=4)


@pytest.mark.parametrize("bad", [2, -2, 7])
def test_invalid_flag_values_are_refused(synthetic_ssp_wide, synthetic_tophat_obs, bad):
    """Only 0 / 1 / -1 are censor flags."""
    with pytest.raises(ValueError, match=r"(?i)flag"):
        _catalog(
            ssp=synthetic_ssp_wide,
            obs_base=synthetic_tophat_obs,
            line_censor=[0, bad],
            line_censor_cols=["halpha_limit"],
        )


def test_boolean_censor_column_is_refused(synthetic_ssp_wide, synthetic_tophat_obs):
    """A boolean column is an include-mask, not a censor flag.

    The photometric ``censor_cols`` already refuses this because ``True``
    silently means "upper limit" under ``astype(int)``, which launders an
    intended include-mask into censoring. The line axis must answer the same.
    """
    with pytest.raises(ValueError, match=r"(?i)bool"):
        _catalog(
            ssp=synthetic_ssp_wide,
            obs_base=synthetic_tophat_obs,
            line_censor=np.array([True, False]),
            line_censor_cols=["halpha_limit"],
        )


def test_count_mismatch_is_refused(synthetic_ssp_wide, synthetic_tophat_obs):
    """One censor column per line column."""
    with pytest.raises(ValueError, match=r"(?i)line_censor_cols"):
        _catalog(
            ssp=synthetic_ssp_wide,
            obs_base=synthetic_tophat_obs,
            line_censor=[0, 1],
            line_censor_cols=["halpha_limit", "spurious_second"],
        )
