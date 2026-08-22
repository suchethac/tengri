Diagnostics
===========

Tools enabled by end-to-end differentiability of the forward model.
These provide Fisher information analysis, gradient SEDs (saliency maps),
chain convergence diagnostics, and Green's function time-sensitivity analysis.

Fisher information
------------------

.. autofunction:: tengri.analysis.diagnostics.fisher.compute_jacobian

.. autofunction:: tengri.analysis.diagnostics.fisher.compute_fisher_matrix

.. autofunction:: tengri.analysis.diagnostics.fisher.fisher_parameter_errors

.. autofunction:: tengri.analysis.diagnostics.fisher.fisher_correlation_matrix

Chain convergence diagnostics
------------------------------

.. autofunction:: tengri.analysis.diagnostics.autocorrelation_time

.. autofunction:: tengri.analysis.diagnostics.autocorrelation_time_combined

.. autofunction:: tengri.analysis.diagnostics.effective_sample_size

.. autofunction:: tengri.analysis.diagnostics.check_chain_length

Saliency (gradient SEDs)
------------------------

.. autofunction:: tengri.analysis.diagnostics.saliency.compute_gradient_sed

.. autofunction:: tengri.analysis.diagnostics.saliency.compute_all_gradient_seds

.. autofunction:: tengri.analysis.diagnostics.saliency.compute_photometry_sensitivity

Emission-line measures
----------------------

.. autofunction:: tengri.analysis.diagnostics.compute_equivalent_widths

.. autofunction:: tengri.analysis.diagnostics.compute_line_fluxes

.. autofunction:: tengri.analysis.diagnostics.compute_line_moments

Green's functions
-----------------

.. autofunction:: tengri.analysis.diagnostics.green_functions.compute_green_function

.. autofunction:: tengri.analysis.diagnostics.green_functions.compute_window_function

.. autofunction:: tengri.analysis.diagnostics.green_functions.compute_time_sensitivity_matrix

Spectral indices and diagnostics
---------------------------------

.. autofunction:: tengri.analysis.diagnostics.dn4000

.. autofunction:: tengri.analysis.diagnostics.irx

.. autofunction:: tengri.analysis.diagnostics.uv_slope_beta

.. autofunction:: tengri.analysis.diagnostics.rest_frame_color

.. autofunction:: tengri.analysis.diagnostics.rest_frame_luminosity

.. autofunction:: tengri.analysis.diagnostics.dust_energy_balance

.. autofunction:: tengri.analysis.diagnostics.integrate_lnu_over_band
