:orphan:

.. _sphx_glr_auto_examples_inference:

Inference Methods
=================

Fitting posteriors and checking that they mean something — comparing
samplers and their convergence (split-R-hat, ESS, multi-chain speedups),
corner plots and prior-vs-posterior comparisons, prior- and
posterior-predictive checks, photo-z chi-square grids, and hierarchical
population fits.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We fit mock SDSS photometry using MAP (maximum a posteriori) optimization with Adam and recover the input star-formation history. The figure overlays the MAP-recovered SFH against the ground truth, demonstrating convergence on the morphology despite the nonconvex likelihood landscape.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_convergence_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_convergence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAP fit recovery: star-formation history from mock photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Galaxy photometry is degenerate in redshift and stellar mass — the same galaxy can look identical at different redshifts if the mass is adjusted. We mock a star-forming galaxy at z=2.5 with known stellar mass, observe it in ugrizYJHK bands at S/N=10, then compute χ² on a 2D grid of (z, M*) to show the classic photo-z degeneracy valley. The figure maps χ² as a heatmap with 1σ/2σ/3σ contours and overlays the true redshift.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_photoz_chi2_grid_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_photoz_chi2_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photo-z degeneracy: chi² landscape over redshift and stellar mass</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Before fitting, sample 200 draws from the prior and push each through the forward model. The envelope of predicted photometry is the prior predictive distribution — what the model can produce under our chosen priors, without any conditioning on observations.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_prior_predictive_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_prior_predictive`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Prior predictive check: what does the model predict before it sees data?</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Simulate a galaxy with a double-power-law (dpl) star formation history, mock SDSS photometry at S/N=20, and recover the SFH using MAP optimization. The figure compares the true and recovered SFH as a function of time, with the bottom panel showing photometric residuals normalized by noise. The recovery demonstrates how well the SED fitting posterior reconstructs SFH shape despite degeneracies with dust and metallicity.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_sfh_recovery_test_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_sfh_recovery_test`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SFH recovery with MAP: double power-law against mock photometry</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/inference/plot_convergence
   /auto_examples/inference/plot_photoz_chi2_grid
   /auto_examples/inference/plot_prior_predictive
   /auto_examples/inference/plot_sfh_recovery_test

