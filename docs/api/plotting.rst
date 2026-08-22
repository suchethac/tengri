Plotting
========

Publication-quality plotting utilities for SED fits, SFH recovery,
corner plot comparisons, convergence diagnostics, and parameter sweeps.
Designed for ApJ/MNRAS figures.

Style setup
-----------

.. autofunction:: tengri.analysis.plotting.setup_style

Posterior visualization
-----------------------

.. autofunction:: tengri.analysis.plotting.plot_1d_posterior

.. autofunction:: tengri.analysis.plotting.plot_autocorrelation

.. autofunction:: tengri.analysis.plotting.posterior_plot_sed

.. autofunction:: tengri.analysis.plotting.posterior_plot_sfh

SFH plots
---------

.. autofunction:: tengri.analysis.plotting.plot_sfh

.. autofunction:: tengri.analysis.plotting.plot_sfh_comparison

.. autofunction:: tengri.analysis.plotting.sfh_sed_comparison

.. autofunction:: tengri.analysis.plotting.add_sfh_inset

SED plots
---------

.. autofunction:: tengri.analysis.plotting.plot_sed_fit

.. autofunction:: tengri.analysis.plotting.plot_spectrum_fit

.. autofunction:: tengri.analysis.plotting.plot_calibration

Filters
-------

.. autofunction:: tengri.analysis.plotting.plot_filter_curves

.. autofunction:: tengri.analysis.plotting.plot_filter_coverage

.. autofunction:: tengri.analysis.plotting.compare_filter_sets

Corner plots
------------

.. autofunction:: tengri.analysis.plotting.safe_corner

.. autofunction:: tengri.analysis.plotting.plot_corner_comparison

Convergence diagnostics
-----------------------

.. autofunction:: tengri.analysis.plotting.convergence_check

.. autofunction:: tengri.analysis.plotting.convergence_table

Parameter sweeps and mock data
------------------------------

.. autofunction:: tengri.analysis.plotting.sweep_parameter

.. autofunction:: tengri.analysis.plotting.parameter_gallery

.. autofunction:: tengri.analysis.plotting.mock_plot

Diagnostics
-----------

.. autofunction:: tengri.analysis.plotting.diagnostics_table

Constants
---------

.. Document at the defining module (``styles``) rather than the package
   re-export: autodoc resolves ``#:`` doc-comments against the module named in
   the directive, and pointed at the package these rendered Python's builtin
   ``dict.__doc__`` instead of the palette description.

.. autodata:: tengri.analysis.plotting.styles.COLORS

.. autodata:: tengri.analysis.plotting.styles.SDSS_WAVE_EFF

.. autodata:: tengri.analysis.plotting.styles.SDSS_BANDS

.. autodata:: tengri.analysis.plotting.styles.SDSS_BAND_COLORS

.. autodata:: tengri.analysis.plotting.styles.SDSS_BAND_NAMES

.. autodata:: tengri.analysis.plotting.styles.SED_XLABEL

.. autodata:: tengri.analysis.plotting.styles.SED_XLIM

.. autodata:: tengri.analysis.plotting.styles.SED_XSCALE

.. autodata:: tengri.analysis.plotting.styles.SED_YLABEL

.. autodata:: tengri.analysis.plotting.styles.SFH_XLABEL

.. autodata:: tengri.analysis.plotting.styles.SFH_YLABEL

.. autodata:: tengri.analysis.plotting.styles.SPECTRAL_FEATURES
