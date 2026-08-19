:orphan:

.. _sphx_glr_auto_examples_advanced:

Advanced Topics
===============

Extension-point demonstration (SEDModelComponent), Fisher degeneracy, and validation techniques: gradient vs finite-difference, mass conservation, redshift-frame invariance, WavePrecomp accuracy.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The on-ramp for adding a custom physics block to tengri. Subclass SEDModelComponent, declare name, parameter_prefix, priors as class attributes, and implement predict(p, sed_in, wave). __init_subclass__ registers the new variant and auto-fills the inputs() / outputs() contracts.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_custom_attenuation_component_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_custom_attenuation_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Authoring a new physics block with SEDModelComponent</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="tengri is a differentiable JAX package. Every model gradient ∂L/∂θ computed via jax.grad() should numerically match a central finite-difference approximation. This diagnostic builds a star-forming model with several free parameters, defines a chi-squared loss, and compares autodiff vs FD gradients for each parameter. A mismatch (&gt;1e-3) indicates a non-differentiable operation.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_gradient_finite_difference_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_gradient_finite_difference`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Autodiff gradients vs. finite-difference derivatives: diagnostic verification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Internal consistency check: the cumulative SFR integral ∫₀ᵗ SFR(t) dt should equal the stellar mass returned by predict_properties(). This diagnostic varies the DPL SFH parameters and verifies that the two pathways (manual trapz of the trajectory vs library integration) agree to ~0.1%. Discrepancies &gt; 5% trigger a warning and would indicate a bug in either the SFH trajectory or the mass integration kernel.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_mass_conservation_sfh_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_mass_conservation_sfh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mass conservation in SFH: manual integration vs predict_properties</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The rest-frame SED depends only on intrinsic galaxy properties (SFH, dust, metallicity, nebular, AGN) and is independent of redshift. Redshift only enters via the observation (wavelength shift, distance dimming, IGM attenuation). This diagnostic verifies that Prediction.rest_sed returns bit-identical SEDs across a range of redshifts for identical intrinsic parameters. Age-of-the-Universe constraints at high-z may truncate the SFH legitimately, producing smooth variation; any non-smooth jump signals a coupling bug.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_redshift_rest_invariance_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_redshift_rest_invariance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rest-frame SED Redshift Invariance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WavePrecomp approximation pre-integrates SSP × filter LUTs and interpolates photometry through a redshift table, trading exact calculations for speed. This diagnostic compares exact-wave-grid photometry against WavePrecomp variants at different ztable densities n_z, showing how fractional errors decrease with finer redshift grids.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_waveprecomp_accuracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_waveprecomp_accuracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">WavePrecomp photometric accuracy across redshift grids</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cramér-Rao bound from the Fisher Information Matrix shows that SDSS 5-band photometry alone cannot separately constrain age, dust, and metallicity. Adding NIR or MIR bands breaks the degeneracy by factors of 2–5×, quantifying the information gain from multiwavelength coverage.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_fisher_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_fisher_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-Dust-Metallicity Degeneracy: Fisher Analysis</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/advanced/plot_custom_attenuation_component
   /auto_examples/advanced/plot_diag_gradient_finite_difference
   /auto_examples/advanced/plot_diag_mass_conservation_sfh
   /auto_examples/advanced/plot_diag_redshift_rest_invariance
   /auto_examples/advanced/plot_diag_waveprecomp_accuracy
   /auto_examples/advanced/plot_fisher_degeneracy

