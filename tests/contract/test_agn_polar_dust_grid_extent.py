# SPDX-License-Identifier: BSD-3-Clause
"""R66: a model grid that truncates the polar reference's integration range.

Under ``agn_norm='cigale_joint'`` with ``torus='skirtor'``, the polar dust's
absorbed-power reference is built on the SKIRTOR templates' NATIVE wavelength
axis (10 A - 1e8 A) -- that is what CIGALE's ``skirtor2016`` integrates over
(``x=AGN1.wl`` throughout) and what R60 Item B moved tengri onto. Getting
there, ``skirtor_disc_dust_ratio`` resamples the caller's disc array onto that
axis with ``left=0.0, right=0.0``, so wherever the model's own grid does not
reach, the disc is **zero-filled** and the unit-area shape is renormalized
over a truncated spectrum.

Measured on this branch (composable ``torus='skirtor'`` +
``atten='polar_dust'``, ``agn_norm='cigale_joint'``, ``agn_ir_frac=0.3``,
i=30, reading ``int(polar)/int(torus)`` over frequency):

=========================================  ================  =========
model grid                                 ``polar/torus``   vs covered
=========================================  ================  =========
8 A - 1e8 A, n=3000                             0.264063       1.0000
0.0413 A - 3e11 A, n=4000                       0.264063       1.0000  (bit-identical)
80 A - 1e7 A, n=3000                            0.284060       1.0757
500 A - 1e8 A, n=3000                           0.290990       1.1020
8 A - 1e6 A, n=3000                             0.263950       0.9996
=========================================  ================  =========

The 80 A - 1e7 A row is the one that settles what "cover" has to mean: it
spans the CIGALE piecewise disc's own declared breakpoints (8 - 1e6 nm) in
full and is still 7.6% off, because ``piecewise_powerlaw_disk`` extrapolates
its end segments rather than truncating -- those breakpoints hold only 86.99%
of the shape's integral -- and because the zero-fill happens on the TEMPLATE
axis, not on the breakpoints. So the required range is the SKIRTOR axis, read
off the grid file, and the guard says so.

A real ``SEDModel.build`` gets there for free: ``torus='skirtor'`` puts the
template axis into the master-grid union
(``forward.wavelength_extension._AGN_TORUS_TEMPLATES``), so the union spans
10 A - 1e8 A whatever the SSP grid does (measured: a 91 A - 1e8 A SSP gives
10 A - 1e8 A once the torus is attached). The guard is therefore a ratchet on
that union: it fires when the template contribution stops arriving, which is
what the tests below simulate, and which is exactly how the +10.2% would come
back silently.

References
----------
- Stalevski et al. 2016, MNRAS, 458, 2288 (SKIRTOR)
- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri import DEFAULT, Fixed

pytestmark = pytest.mark.contract


def _build(ssp, *, torus="skirtor", norm="cigale_joint", atten="polar_dust", ir_frac=0.3):
    import tengri

    agn = {
        "type": "composable",
        "norm": norm,
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": torus, "all_params": Fixed(DEFAULT)},
        "atten": {"type": atten, "all_params": Fixed(DEFAULT)},
        "all_params": Fixed(DEFAULT),
    }
    if ir_frac is not None:
        agn["agn_ir_frac"] = Fixed(ir_frac)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tengri.SEDModel.build(
            ssp,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dl07", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn=agn,
            redshift=Fixed(0.05),
        )


@pytest.fixture
def union_loses_the_skirtor_axis(monkeypatch):
    """Drop the SKIRTOR template axis from the master-grid union.

    The union is what keeps a real build covered, so removing the torus
    entry is how the truncation this guard refuses actually reaches a model.
    Patched at the registry, not at the collector, so the collector's own
    plumbing still runs.
    """
    from tengri.forward import wavelength_extension

    patched = {
        key: value
        for key, value in wavelength_extension._AGN_TORUS_TEMPLATES.items()
        if key != "skirtor"
    }
    monkeypatch.setattr(wavelength_extension, "_AGN_TORUS_TEMPLATES", patched)
    return patched


class TestPolarDustRefusesATruncatingGrid:
    def test_a_truncating_grid_raises(self, synthetic_ssp_wide, union_loses_the_skirtor_axis):
        """The grid no longer reaches the SKIRTOR axis, so the build is refused.

        ``synthetic_ssp_wide`` spans 100 A - 1e7 A, so with the template axis
        gone from the union the master grid misses BOTH ends of the
        10 A - 1e8 A range the polar reference is integrated over.
        """
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError, match=r"polar"):
            _build(synthetic_ssp_wide)

    def test_the_refusal_names_the_required_range_and_the_grid_it_got(
        self, synthetic_ssp_wide, union_loses_the_skirtor_axis
    ):
        """A refusal a caller cannot act on is half a guard (#1364)."""
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError) as exc:
            _build(synthetic_ssp_wide)
        msg = str(exc.value)
        for token in ("polar_dust", "skirtor", "10 A", "1e+08 A"):
            assert token in msg, f"the refusal does not name {token!r}: {msg}"
        assert "100 A" in msg, f"the refusal does not report the grid it got: {msg}"
        assert "10.20%" in msg, "the refusal must report what the truncation costs"

    def test_the_covering_default_build_is_untouched(self, synthetic_ssp_wide):
        """The control: unpatched, the union covers and nothing is refused."""
        model = _build(synthetic_ssp_wide)
        wave = np.asarray(model.wavelengths)
        assert wave.min() <= 10.0 and wave.max() >= 1.0e8, (
            f"probe setup failed: the master grid is {wave.min():.4g} - "
            f"{wave.max():.4g} A, which does not cover the SKIRTOR axis, so "
            "this control is not testing what it claims"
        )

    def test_a_non_skirtor_torus_is_not_refused(
        self, synthetic_ssp_wide, union_loses_the_skirtor_axis
    ):
        """Scope: without the R-tie there is no native-axis resampling.

        ``agn_norm='cigale_joint'`` only ties the disc when the torus is
        SKIRTOR (``blocks/runner.py``), so with ``torus='fritz'`` the polar
        reference is integrated on the model's own grid and no zero-fill
        happens. The guard must not reach that build -- refusing it would be
        a refusal with no measurement behind it.
        """
        model = _build(synthetic_ssp_wide, torus="fritz")
        assert model.spec.agn_model == "composable"

    def test_no_polar_dust_block_is_not_refused(
        self, synthetic_ssp_wide, union_loses_the_skirtor_axis
    ):
        """Scope: the reference only exists when the polar block is active."""
        model = _build(synthetic_ssp_wide, atten="none")
        assert model.spec.agn_model == "composable"

    def test_the_required_range_is_read_off_the_grid_file(self):
        """The bounds are the file's own axis, not a literal in the guard.

        A regenerated SKIRTOR grid with a different axis must move the
        requirement with it; a hard-coded 10 A / 1e8 A would not.
        """
        h5py = pytest.importorskip("h5py")
        from tengri.forward.sed_model import _polar_reference_required_extent_aa

        extent = _polar_reference_required_extent_aa("skirtor")
        assert extent is not None, "the skirtor torus declares no required extent"
        from tengri.components.agn.skirtor import _find_skirtor_grid

        with h5py.File(_find_skirtor_grid(), "r") as f:
            axis = np.asarray(f["wavelength"][:])
        np.testing.assert_allclose(
            np.asarray(extent), [axis.min(), axis.max()], rtol=1e-12, atol=0.0
        )

    def test_a_non_skirtor_torus_declares_no_required_extent(self):
        """Only the tie's native-axis resampling creates a requirement."""
        from tengri.forward.sed_model import _polar_reference_required_extent_aa

        for block in ("fritz", "two_temperature", "none", None):
            assert _polar_reference_required_extent_aa(block) is None, (
                f"{block!r} declares a required extent, but no R-tie resamples "
                "the disc onto a native axis for it"
            )
