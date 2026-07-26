# SPDX-License-Identifier: BSD-3-Clause
"""#1321: ``Data.censor`` must reject flag values outside ``{0, 1, -1}``.

``Data`` is the single-galaxy seam for measurement validation (spec §3.2),
and the catalog seam ``ingest_catalog(censor_cols=...)`` already rejects
garbage flags. ``Data.validate_against`` did not: it rejected boolean
arrays (a ``True`` would read as "upper limit") but accepted any numeric
value, so ``censor=[0, 2, 5]`` passed validation.

Downstream that is silent, not loud. ``censored_neg_log_likelihood``
(``observation/noise.py``) dispatches
``jnp.where(mask == 1, upper, jnp.where(mask == -1, lower, detected))``,
so every unrecognized flag falls through to the **detected** branch: a
mis-scaled or sentinel-coded censor column turns upper limits into
detections and biases the fit, with nothing raised and nothing logged.

The two seams must agree — same allowed set, same rejection.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def _obs(n=3):
    from tengri.observation import Observation, Photometry

    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"][:n]))


def _data(censor):
    from tengri import Data

    return Data(photometry=(jnp.ones(3), jnp.full(3, 0.1)), censor=censor)


@pytest.mark.parametrize(
    "censor,bad_value",
    [
        (jnp.array([0, 2, 5]), "2"),
        (jnp.array([0, -99, 1]), "-99"),
        (jnp.array([0.0, 0.5, 1.0]), "0.5"),
        (jnp.array([0.0, np.nan, 1.0]), "nan"),
    ],
)
def test_invalid_censor_flags_rejected(censor, bad_value):
    """Anything outside {0, 1, -1} raises, and the message names the value."""
    with pytest.raises(ValueError, match="censor") as excinfo:
        _data(censor).validate_against(_obs())
    assert bad_value in str(excinfo.value), (
        f"the error must name the offending flag {bad_value!r}: {excinfo.value}"
    )


@pytest.mark.parametrize("censor", [jnp.array([0, 1, -1]), jnp.array([0.0, 1.0, -1.0])])
def test_valid_censor_flags_accepted(censor):
    """The legal set passes, whether spelled as ints or as whole floats."""
    validated = _data(censor).validate_against(_obs())
    np.testing.assert_array_equal(np.asarray(validated.censor), np.asarray(censor))


def test_boolean_censor_still_rejected():
    """The pre-existing boolean guard is not weakened by the range check."""
    with pytest.raises(ValueError, match="bool"):
        _data(jnp.array([True, False, True])).validate_against(_obs())


def test_both_seams_reject_the_same_flag_set():
    """``Data`` and ``ingest_catalog`` must not disagree about what is legal.

    They validate the same channel for the same likelihood; a value one
    accepts and the other refuses means single-galaxy and catalog fits of
    the same measurements disagree.
    """
    from tengri.inference.catalog_ingest import ingest_catalog

    table = {
        "sdss_g": np.array([1.0]),
        "sdss_g_err": np.array([0.1]),
        "sdss_r": np.array([1.0]),
        "sdss_r_err": np.array([0.1]),
        "sdss_i": np.array([1.0]),
        "sdss_i_err": np.array([0.1]),
        "cg": np.array([0]),
        "cr": np.array([2]),  # illegal
        "ci": np.array([0]),
    }
    with pytest.raises(ValueError, match="censor"):
        ingest_catalog(
            table,
            photometry=_obs().photometry,
            flux_unit="cgs_fnu",
            censor_cols={"sdss_g": "cg", "sdss_r": "cr", "sdss_i": "ci"},
        )
    with pytest.raises(ValueError, match="censor"):
        _data(jnp.array([0, 2, 0])).validate_against(_obs())
