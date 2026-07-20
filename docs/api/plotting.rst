Plotting
========

Publication-quality plotting utilities for SED fits, SFH recovery,
corner plot comparisons, and convergence diagnostics tables. Designed
for ApJ/MNRAS figures.

Style Setup
-----------

.. autofunction:: tengri.analysis.plotting.setup_style

SFH Plots
----------

.. autofunction:: tengri.analysis.plotting.plot_sfh

.. autofunction:: tengri.analysis.plotting.plot_sfh_comparison

SED Plots
---------

.. autofunction:: tengri.analysis.plotting.plot_sed_fit

.. autofunction:: tengri.analysis.plotting.plot_spectrum_fit

Corner Plots
------------

.. autofunction:: tengri.analysis.plotting.safe_corner

.. autofunction:: tengri.analysis.plotting.plot_corner_comparison

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

.. autodata:: tengri.analysis.plotting.styles.SPECTRAL_FEATURES
