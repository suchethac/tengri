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
i=30, reading ``int(polar)/int(torus)`` over frequency). Both CIGALE-lineage
disc shapes, because the size of the error is a property of the shape being
zero-filled and not of the guard:

=========================================  ====================  =============
model grid                                 ``polar/torus``       vs covered
                                           disc=skirtor          disc=skirtor
=========================================  ====================  =============
8 A - 1e8 A, n=3000                              0.264046724       1.000000
0.0413 A - 3e11 A, n=4000                        0.264046724       1.000000
1 A - 1e9 A, n=6000                              0.264046724       1.000000
8 A - 1e8 A, n=6000 (resolution control)         0.264046724       1.000000
80 A - 1e7 A, n=3000                             0.284060926       1.075798
500 A - 1e8 A, n=3000                            0.290982429       1.102011
8 A - 1e6 A, n=3000                              0.263936198       0.999581
100 A - 1e6 A, n=1500                            0.286324800       1.084372
=========================================  ====================  =============

Four grids that cover agree **bit-for-bit**, at two different resolutions, so
the effect is extent and not quadrature. With ``disc='schartmann2005'`` the
same grids give 0.257339472 (all four covering, again bit-identical),
1.001058, 0.970316, 0.999593 and 1.000584 -- same sign of failure, an order of
magnitude smaller, because that shape carries less of its integral outside
the model's grid.

The 80 A - 1e7 A row is the one that settles what "cover" has to mean: it
spans the CIGALE piecewise disc's own declared breakpoints (8 - 1e6 nm) in
full and is still 7.6% off, because ``piecewise_powerlaw_disk`` extrapolates
its end segments rather than truncating -- those breakpoints hold only 86.99%
of the ``skirtor`` shape's integral -- and because the zero-fill happens on
the TEMPLATE axis, not on the breakpoints. So the required range is the
SKIRTOR axis, read off the grid file, and the guard says so.

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
from typing import ClassVar

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


class TestComposablePrecomputeChoosesACoveringGrid:
    """R71: the precompute helper's own default grid covers the same range.

    ``blocks/composable_precompute.precompute`` defaults ``wave_rest`` to a
    grid it chooses itself, and that grid never becomes the model's
    ``_rest_wavelength`` -- so the build-time guard above cannot see it. The
    value it used to choose, ``np.logspace(2.0, 6.0, 1500)``, covers neither
    end of the SKIRTOR axis: measured with ``disc='skirtor'`` at i=30 and
    ``agn_ir_frac=0.3``, ``int(polar)/int(torus)`` came out **0.286324800**
    on it against **0.264046724** on a covering grid, +8.44%.

    The default is now derived from the same declared native support the
    guard requires, unioned with the legacy 100 A - 1e6 A span so nothing a
    caller had before is lost, and sampled at the legacy points-per-decade
    (374.75, floor 1500 points): 10 A - 1e8 A in 2625 points for a
    ``polar_dust`` + ``skirtor`` recipe, the legacy grid unchanged for a
    recipe with no such tie. A ``wave_rest`` the caller passes goes through
    the guard itself.
    """

    _FIDUCIAL: ClassVar[dict] = dict(
        agn_disc_block="skirtor",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="skirtor",
        agn_attenuation_block="polar_dust",
        agn_norm="cigale_joint",
        agn_tau_skirtor=7.0,
        agn_p_skirtor=1.0,
        agn_q_skirtor=1.0,
        agn_oa_skirtor=40.0,
        agn_polar_ebv=0.03,
        agn_polar_oa=40.0,
        agn_polar_T=100.0,
        agn_polar_beta=1.6,
    )

    @staticmethod
    def _recipe(attenuation="polar_dust", torus="skirtor"):
        from tengri.components.agn.blocks.recipe import Recipe

        return Recipe.from_selectors(
            disc="skirtor",
            torus=torus,
            attenuation=attenuation,
            axis_params=("agn_log_lbol",),
        )

    @classmethod
    def _share(cls, wave):
        import jax.numpy as jnp

        from tengri.components.agn.blocks.runner import compose_l_nu
        from tengri.utils.physics_constants import C_AA

        wave = jnp.asarray(np.asarray(wave, dtype=np.float64))
        nu = C_AA / wave
        order = jnp.argsort(nu)
        _sed, comps = compose_l_nu(
            wave,
            12.0,
            agn_ir_frac=0.3,
            agn_torus_frac=0.5,
            agn_cos_inc=float(np.cos(np.deg2rad(30.0))),
            return_components=True,
            **cls._FIDUCIAL,
        )
        integ = {
            key: float(jnp.abs(jnp.trapezoid(jnp.asarray(comps[key])[order], nu[order])))
            for key in ("polar", "torus")
        }
        return integ["polar"] / integ["torus"]

    def test_the_default_grid_gives_the_covering_grids_share(self):
        """The whole point: the derived default must not move the split."""
        from tengri.components.agn.blocks.composable_precompute import default_wave_rest
        from tengri.components.agn.skirtor import _load_raw_disk_dust_grid

        if _load_raw_disk_dust_grid() is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")
        covering = self._share(np.geomspace(8.0, 1.0e8, 3000))
        default = self._share(default_wave_rest(self._recipe()))
        assert default == pytest.approx(covering, rel=1e-6, abs=0.0), (
            f"the precompute's default grid gives int(polar)/int(torus) = "
            f"{default:.9f} against the covering grid's {covering:.9f} "
            f"({default / covering - 1.0:+.4%}). The hard-coded "
            "np.logspace(2.0, 6.0, 1500) it replaced gave 0.286324800, +8.44%."
        )

    def test_the_default_grid_covers_the_required_range(self):
        """Stated as an extent as well, so a density change cannot hide it."""
        from tengri.components.agn.blocks.composable_precompute import default_wave_rest
        from tengri.forward.sed_model import _polar_reference_required_extent_aa

        required = _polar_reference_required_extent_aa("skirtor")
        if required is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")
        grid = default_wave_rest(self._recipe())
        assert grid.min() <= required[0] and grid.max() >= required[1], (
            f"default grid {grid.min():.6g} - {grid.max():.6g} A does not cover the "
            f"required {required[0]:.6g} - {required[1]:.6g} A"
        )
        assert grid.size >= 1500, (
            f"the derived default must not be coarser than the legacy 1500-point "
            f"grid; got {grid.size}"
        )

    def test_a_recipe_with_no_tie_keeps_the_legacy_grid(self):
        """The control: nothing widens a grid that has nothing to cover."""
        from tengri.components.agn.blocks.composable_precompute import default_wave_rest

        grid = default_wave_rest(self._recipe(attenuation="none", torus="none"))
        assert grid.size == 1500
        assert grid.min() == pytest.approx(100.0, rel=1e-12, abs=0.0)
        assert grid.max() == pytest.approx(1.0e6, rel=1e-12, abs=0.0)

    def test_precompute_evaluates_on_the_derived_default(self, monkeypatch):
        """``precompute`` must actually use it, not just be able to compute it."""
        import tengri.components.agn.blocks.composable_precompute as cp
        from tengri.components.agn.skirtor import _load_raw_disk_dust_grid

        if _load_raw_disk_dust_grid() is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")
        seen = {}

        def _capture(wave_rest, recipe, axis_grids, fixed_values, agn_log_lbol_default):
            seen["wave_rest"] = np.asarray(wave_rest)
            raise _StopPrecompute

        monkeypatch.setattr(cp, "_evaluate_recipe_on_grid", _capture)
        with pytest.raises(_StopPrecompute):
            cp.precompute(
                [np.linspace(4000.0, 9000.0, 32)],
                [np.ones(32)],
                0.1,
                None,
                recipe=self._recipe(),
                axis_grids={"agn_log_lbol": np.linspace(11.0, 13.0, 2)},
            )
        grid = seen["wave_rest"]
        assert grid.min() <= 10.0 and grid.max() >= 1.0e8, (
            f"precompute evaluated on {grid.min():.6g} - {grid.max():.6g} A, which "
            "does not cover the SKIRTOR axis the polar reference is built on"
        )

    def test_a_truncating_user_grid_is_refused_with_the_same_message(self):
        """A caller's own ``wave_rest`` gets the guard, verbatim."""
        from tengri.components.agn.blocks import composable_precompute as cp
        from tengri.components.agn.skirtor import _load_raw_disk_dust_grid
        from tengri.config.exceptions import ConfigError

        if _load_raw_disk_dust_grid() is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")
        with pytest.raises(ConfigError) as exc:
            cp.precompute(
                [np.linspace(4000.0, 9000.0, 32)],
                [np.ones(32)],
                0.1,
                None,
                recipe=self._recipe(),
                axis_grids={"agn_log_lbol": np.linspace(11.0, 13.0, 2)},
                wave_rest=np.logspace(2.0, 6.0, 1500),
            )
        msg = str(exc.value)
        for token in ("10 A to 1e+08 A", "polar_dust", "cigale_joint", "+8.44%"):
            assert token in msg, f"the refusal does not name {token!r}: {msg}"

    def test_a_covering_user_grid_builds(self):
        """And an explicit grid that covers is left alone."""
        from tengri.components.agn.blocks import composable_precompute as cp
        from tengri.components.agn.skirtor import _load_raw_disk_dust_grid

        if _load_raw_disk_dust_grid() is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")
        out = cp.precompute(
            [np.linspace(4000.0, 9000.0, 32)],
            [np.ones(32)],
            0.1,
            None,
            recipe=self._recipe(),
            axis_grids={"agn_log_lbol": np.linspace(11.0, 13.0, 2)},
            wave_rest=np.geomspace(8.0, 1.0e8, 600),
        )
        assert "grid_phot" in out and out["grid_phot"].shape[0] == 2


class TestNodeCoincidenceSlackIsHonestAboutFloat32:
    """The node-coincidence slack must absorb a float32 rounding step, exactly.

    ``_check_polar_reference_grid_extent`` treats an endpoint within a small
    relative slack of the requirement as coincident, because
    ``resample_template`` zero-fills strictly outside the caller's span and
    an endpoint exactly on the requirement loses nothing. The slack has to be
    at least float32 machine epsilon (``2**-23``), the largest relative
    distance a float32 value can land from its true neighbor after a
    canonicalization round-trip -- a tighter slack refuses a grid whose
    endpoint is off by only one float32 ulp, and a much looser one would
    silently accept a genuinely truncated grid.
    """

    def _required_extent(self):
        from tengri.forward.sed_model import _polar_reference_required_extent_aa

        required = _polar_reference_required_extent_aa("skirtor")
        if required is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")
        return required

    def test_an_endpoint_one_float32_ulp_off_is_accepted(self):
        from tengri.forward.sed_model import _check_polar_reference_grid_extent

        lo_req, hi_req = self._required_extent()
        # Round-trip each endpoint through float32, then step one ulp toward
        # the interior -- the worst case the slack has to absorb.
        lo32 = np.float32(lo_req)
        hi32 = np.float32(hi_req)
        lo_perturbed = float(np.nextafter(lo32, np.float32(np.inf)))
        hi_perturbed = float(np.nextafter(hi32, np.float32(-np.inf)))
        assert lo_perturbed > lo_req and hi_perturbed < hi_req, (
            "probe setup failed: the perturbation did not narrow the grid"
        )
        wave = np.geomspace(lo_perturbed, hi_perturbed, 500)
        _check_polar_reference_grid_extent(
            wave,
            attenuation_block="polar_dust",
            agn_norm="cigale_joint",
            torus_block="skirtor",
        )  # must not raise

    def test_an_endpoint_far_beyond_the_tolerance_is_still_refused(self):
        from tengri.config.exceptions import ConfigError
        from tengri.forward.sed_model import _check_polar_reference_grid_extent

        lo_req, hi_req = self._required_extent()
        # 1e-4 relative is ~1000x float32 epsilon (1.19e-7) -- far beyond any
        # float32 canonicalization step, so this must still be refused.
        lo_far = lo_req * (1.0 + 1.0e-4)
        wave = np.geomspace(lo_far, hi_req, 500)
        with pytest.raises(ConfigError, match=r"polar"):
            _check_polar_reference_grid_extent(
                wave,
                attenuation_block="polar_dust",
                agn_norm="cigale_joint",
                torus_block="skirtor",
            )


class _StopPrecompute(Exception):
    """Sentinel: stop ``precompute`` once the grid it chose has been seen."""
