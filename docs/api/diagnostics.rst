Diagnostics
===========

Tools enabled by end-to-end differentiability of the forward model.
These provide Fisher information analysis, gradient SEDs (saliency maps),
and Green's function time-sensitivity analysis.

Fisher Information
------------------

.. autofunction:: tengri.analysis.diagnostics.fisher.compute_jacobian

.. autofunction:: tengri.analysis.diagnostics.fisher.compute_fisher_matrix

.. autofunction:: tengri.analysis.diagnostics.fisher.fisher_parameter_errors

.. autofunction:: tengri.analysis.diagnostics.fisher.fisher_correlation_matrix

Saliency (Gradient SEDs)
-------------------------

.. autofunction:: tengri.analysis.diagnostics.saliency.compute_gradient_sed

.. autofunction:: tengri.analysis.diagnostics.saliency.compute_all_gradient_seds

.. autofunction:: tengri.analysis.diagnostics.saliency.compute_photometry_sensitivity

Green's Functions
-----------------

.. autofunction:: tengri.analysis.diagnostics.green_functions.compute_green_function

.. autofunction:: tengri.analysis.diagnostics.green_functions.compute_window_function

.. autofunction:: tengri.analysis.diagnostics.green_functions.compute_window_function_fourier

.. autofunction:: tengri.analysis.diagnostics.green_functions.compute_time_sensitivity_matrix
