# SPDX-License-Identifier: BSD-3-Clause
"""#3547 D4: ``ingest_catalog`` refuses sentinels and negative errors unless declared.

Without ``lim_flag`` a sentinel flux (below the CIGALE invalid threshold -9990) or a
negative error used to be read silently as a detection or as noise. Both now raise,
naming the columns and pointing at ``lim_flag="full"``, the existing declaration.
"""

import numpy as np
import pytest


def _phot():
    from tengri.observation import Photometry

    return Photometry.from_names(["sdss_g", "sdss_r"])


def _table(**over):
    base = {
        "sdss_g": np.array([1.0, 2.0]),
        "sdss_g_err": np.array([0.1, 0.1]),
        "sdss_r": np.array([3.0, 4.0]),
        "sdss_r_err": np.array([0.2, 0.2]),
    }
    base.update(over)
    return base


def test_sentinel_flux_without_lim_flag_is_refused_naming_the_declaration():
    """-9999 is a CIGALE sentinel, not a detection: refuse it unless lim_flag is declared."""
    from tengri.inference.catalog_ingest import ingest_catalog

    t = _table(sdss_g=np.array([1.0, -9999.0]))
    with pytest.raises(ValueError, match="sdss_g") as exc:
        ingest_catalog(t, photometry=_phot(), flux_unit="mJy")
    assert 'lim_flag="full"' in str(exc.value)
    assert "1 row" in str(exc.value)


def test_negative_error_without_lim_flag_is_refused_naming_the_declaration():
    """A negative error is a CIGALE upper limit: refuse it unless lim_flag is declared."""
    from tengri.inference.catalog_ingest import ingest_catalog

    t = _table(sdss_r_err=np.array([0.2, -0.2]))
    with pytest.raises(ValueError, match="sdss_r_err") as exc:
        ingest_catalog(t, photometry=_phot(), flux_unit="mJy")
    assert 'lim_flag="full"' in str(exc.value)


def test_declared_lim_flag_full_reads_the_same_table():
    """The same table with lim_flag="full" is read: the sentinel drops, the limit is kept."""
    from tengri.inference.catalog_ingest import ingest_catalog

    t = _table(sdss_g=np.array([1.0, -9999.0]), sdss_r_err=np.array([0.2, -0.2]))
    ca = ingest_catalog(t, photometry=_phot(), flux_unit="mJy", lim_flag="full")
    assert not ca.presence[1, 0]
    assert ca.censor[1, 1] == 1
