:orphan:

.. _sphx_glr_auto_examples_inference:

Inference Methods
=================

Method selection by dimensionality: `mcmc_nuts` for D ≤ 6, `mcmc_hmc` for D ~ 7–8, `mcmc_raytrace`/`vi` for D >~ 20, `laplace` for cheap intervals from MAP Hessian. `vi` and `native_vi_*` are not posterior-equivalent; both native backends are tier=broken and must never be taught in an example. Convergence diagnostics: split-R-hat, ESS, prior-vs-posterior comparisons, corner plots, posterior-predictive checks.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reference: Conroy+2013.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_convergence_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_convergence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAP fit recovery: star-formation history from mock photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Galaxy photometry is degenerate in redshift and stellar mass — the same galaxy can look identical at different redshifts if the mass is adjusted.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_photoz_chi2_grid_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_photoz_chi2_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photo-z degeneracy: chi² landscape over redshift and stellar mass</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The prior predictive envelope — which combinations of parameters the model can produce under the chosen priors, without data — reveals silent pathologies. Coverage: does the envelope contain the data? If not, the posterior will shift to prior boundaries. Width: narrow bands indicate parameters already constrained by the prior alone; data cannot improve estimates there.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_prior_predictive_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_prior_predictive`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Prior predictive check: what does the model predict before it sees data?</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reference: Conroy+2013.">

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

