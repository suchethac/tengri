# SPDX-License-Identifier: BSD-3-Clause
"""``flux_cols`` must be able to name a real catalog column (#1458).

``ingest_catalog`` documented ``flux_cols`` as the column names in your table,
then required every entry to already **be** a band name from the observation.
So the parameter could only ever be a permutation of its own default — it could
not name a real catalog column, which is the only reason it exists.

``err_cols``, documented identically, had no such restriction. That asymmetry
is what identified the check as a mistake rather than a contract: no design
would validate one side against the observation and the other against nothing.

Column existence is validated against the **table** during extraction, which is
the right referent and already reports the actual column list.
"""

import numpy as np
import pytest

from tengri.inference.catalog_ingest import ingest_catalog
from tengri.observation import Photometry

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def phot():
    return Photometry.from_names(["sdss_g", "sdss_r"])


@pytest.fixture
def table():
    """A table whose columns look like a real survey's, not like band names."""
    return {
        "FLUX_G": np.array([1.0e-28, 2.0e-28]),
        "FLUX_R": np.array([1.5e-28, 2.5e-28]),
        "ERR_G": np.array([1.0e-29, 1.0e-29]),
        "ERR_R": np.array([1.0e-29, 1.0e-29]),
    }


def test_arbitrary_flux_column_names_are_accepted(phot, table):
    """The reproducer from the issue: survey-style names must ingest."""
    out = ingest_catalog(
        table,
        photometry=phot,
        flux_unit="cgs_fnu",
        flux_cols=["FLUX_G", "FLUX_R"],
        err_cols=["ERR_G", "ERR_R"],
    )
    assert np.asarray(out.flux).shape == (2, 2)


def test_the_binding_is_positional_and_reads_the_named_column(phot, table):
    """Not just "it does not raise" — the right column reaches the right band.

    A fix that accepted the names but ignored them, or that fell back to the
    band-name defaults, would pass a raises-or-not test.
    """
    out = ingest_catalog(
        table,
        photometry=phot,
        flux_unit="cgs_fnu",
        flux_cols=["FLUX_G", "FLUX_R"],
        err_cols=["ERR_G", "ERR_R"],
    )
    flux = np.asarray(out.flux)
    np.testing.assert_allclose(flux[:, 0], table["FLUX_G"])
    np.testing.assert_allclose(flux[:, 1], table["FLUX_R"])


def test_reordering_the_columns_reorders_the_bands(phot, table):
    """Positional binding must be real, since it is what the docstring warns of."""
    swapped = ingest_catalog(
        table,
        photometry=phot,
        flux_unit="cgs_fnu",
        flux_cols=["FLUX_R", "FLUX_G"],
        err_cols=["ERR_R", "ERR_G"],
    )
    flux = np.asarray(swapped.flux)
    np.testing.assert_allclose(flux[:, 0], table["FLUX_R"])
    np.testing.assert_allclose(flux[:, 1], table["FLUX_G"])


def test_a_column_absent_from_the_table_still_raises(phot, table):
    """Loosening the check must not remove the validation that matters.

    The referent moves from the observation to the table; it does not vanish.
    """
    with pytest.raises(ValueError) as excinfo:
        ingest_catalog(
            table,
            photometry=phot,
            flux_unit="cgs_fnu",
            flux_cols=["FLUX_G", "NOT_A_COLUMN"],
            err_cols=["ERR_G", "ERR_R"],
        )
    message = str(excinfo.value)
    assert "NOT_A_COLUMN" in message
    # The message must name the table's real columns, or the user cannot act.
    assert "FLUX_R" in message, message


def test_band_names_still_work_as_flux_cols(phot):
    """Guard against over-reaching: the old (only) usage must keep working."""
    band_table = {
        "sdss_g": np.array([1.0e-28]),
        "sdss_r": np.array([1.5e-28]),
        "sdss_g_err": np.array([1.0e-29]),
        "sdss_r_err": np.array([1.0e-29]),
    }
    out = ingest_catalog(
        band_table,
        photometry=phot,
        flux_unit="cgs_fnu",
        flux_cols=["sdss_g", "sdss_r"],
        err_cols=["sdss_g_err", "sdss_r_err"],
    )
    assert np.asarray(out.flux).shape == (1, 2)


def test_the_count_check_survives(phot, table):
    """One column per band remains required — that is the real constraint."""
    with pytest.raises(ValueError, match="flux_cols count"):
        ingest_catalog(
            table,
            photometry=phot,
            flux_unit="cgs_fnu",
            flux_cols=["FLUX_G"],
            err_cols=["ERR_G", "ERR_R"],
        )
