# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2610: filter convention attribution, pivot wavelength, facade pivots.

Frozen: compute_effective_wavelength returns the pivot wavelength sqrt(∫Tλ dλ / ∫T/λ dλ),
not the mean wavelength ∫Tλ dλ / ∫T dλ; convention text attributes BAGPIPES to photon-counting
(BESSELL) and CIGALE to energy-counting (ENERGY); Galaxy.plot computes each band's pivot from
the filter curve's wave/trans instead of reading a non-existent lambda_eff attribute.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tengri.observation.filters import compute_effective_wavelength
from tengri.observation.photometry import FilterConvention, list_filter_conventions

pytestmark = pytest.mark.regression_bug


class TestEffectiveWavelengthIsPivot:
    """compute_effective_wavelength returns the pivot, not mean wavelength."""

    @pytest.mark.parametrize(
        "filter_file",
        [
            "SLOAN_SDSS_g.dat",
            "SLOAN_SDSS_r.dat",
            "JWST_NIRCam_F277W.dat",
        ],
    )
    def test_effective_wavelength_is_the_pivot(self, filter_file):
        """Effective wavelength equals the pivot wavelength for shipped curves."""
        import os

        # Load the shipped curve from file
        data_dir = os.environ.get("TENGRI_DATA_DIR", "data")
        filter_path = os.path.join(data_dir, "filters", filter_file)

        if not os.path.exists(filter_path):
            pytest.skip(f"Filter file not found: {filter_path}")

        wave_aa, trans = np.loadtxt(filter_path, unpack=True)

        # Compute the effective wavelength
        computed = compute_effective_wavelength(wave_aa, trans)

        # Compute the pivot independently
        pivot = np.sqrt(
            np.trapezoid(trans * wave_aa, wave_aa) / np.trapezoid(trans / wave_aa, wave_aa)
        )

        # Check they match to rtol 1e-6 (0.0001%)
        np.testing.assert_allclose(computed, pivot, rtol=1e-6)

    def test_pivot_literature_values(self):
        """Check against literature pivot values from BAGPIPES for SDSS g band."""
        import os

        # SDSS g band should have pivot around 4702.5 Å according to BAGPIPES
        data_dir = os.environ.get("TENGRI_DATA_DIR", "data")
        filter_path = os.path.join(data_dir, "filters", "SLOAN_SDSS_g.dat")

        if not os.path.exists(filter_path):
            pytest.skip(f"Filter file not found: {filter_path}")

        wave_aa, trans = np.loadtxt(filter_path, unpack=True)

        computed = compute_effective_wavelength(wave_aa, trans)

        # BAGPIPES reference value is 4702.495 Å
        # Literature tolerance: within 1 Å
        assert abs(computed - 4702.5) < 1.0, (
            f"sdss_g pivot {computed:.1f} Å differs from literature value 4702.5 Å by > 1 Å"
        )


class TestConventionTextAttributionAccurate:
    """Convention documentation attributes codes correctly."""

    def test_convention_text_attributes_bagpipes_to_photon_counting(self):
        """BESSELL text mentions BAGPIPES; ENERGY does not."""
        conventions = list_filter_conventions()
        conv_dict = conventions.to_dict()

        bessell_doc = conv_dict["bessell"].lower()
        energy_doc = conv_dict["energy"].lower()

        # BESSELL should mention BAGPIPES (photon-counting)
        assert "bagpipes" in bessell_doc, (
            f"BESSELL convention docstring should mention BAGPIPES; got: {bessell_doc}"
        )

        # ENERGY should NOT mention BAGPIPES
        assert "bagpipes" not in energy_doc, (
            f"ENERGY convention docstring should not mention BAGPIPES; got: {energy_doc}"
        )

        # ENERGY should mention CIGALE
        assert "cigale" in energy_doc, (
            f"ENERGY convention docstring should mention CIGALE; got: {energy_doc}"
        )

    def test_convention_enum_docstring_accurate(self):
        """BAGPIPES sits in the BESSELL bullet of the enum docstring, not the ENERGY one."""
        doc = FilterConvention.__doc__
        bessell_start = doc.index("- ``BESSELL``")
        energy_start = doc.index("- ``ENERGY``")
        energy_end = doc.index("\n\n", energy_start)
        bessell_bullet = doc[bessell_start:energy_start].lower()
        energy_bullet = doc[energy_start:energy_end].lower()

        assert "bagpipes" in bessell_bullet
        assert "bagpipes" not in energy_bullet
        assert "cigale" in energy_bullet


class TestFacadeFilterMetadata:
    """Galaxy.plot computes the filter pivots from each curve's wave/trans."""

    def test_galaxy_plot_reaches_filter_pivot_without_lambda_eff(self, synthetic_ssp_wide):
        """``Galaxy.plot`` takes pivots from wave/trans; ``FilterCurve`` has no ``lambda_eff``."""
        import jax.numpy as jnp
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
        from tengri.facade import Galaxy
        from tengri.observation.photometry import FilterCurve

        def _tophat(center, frac=0.16, n=40):
            wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
            trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
            return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

        filters = tuple(_tophat(c) for c in (4800.0, 6200.0))
        assert not any(hasattr(f, "lambda_eff") for f in filters)
        obs = Observation(photometry=Photometry(filters=filters))
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8, 12)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.3),
        )
        galaxy = Galaxy(
            ssp=synthetic_ssp_wide,
            observation=obs,
            parameters=None,
            model_config=None,
            model=model,
        )
        galaxy._flux_obs = np.ones(2)
        galaxy._noise = np.full(2, 0.1)
        galaxy.result = SimpleNamespace(samples={"log_total_mass": jnp.linspace(9.0, 11.0, 8)})
        try:
            fig = galaxy.plot()
            xs = np.asarray(fig.axes[0].lines[0].get_xdata())
            expected = [
                compute_effective_wavelength(np.asarray(f.wave), np.asarray(f.trans))
                for f in filters
            ]
            np.testing.assert_allclose(xs, expected, rtol=1e-6)
        finally:
            plt.close("all")
