

.. _sphx_glr_auto_examples_inference:

Inference Methods
=================

Comparing inference methods and convergence diagnostics.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Runs a quick fit and displays convergence diagnostics: ESS per parameter, summary table, and trace plots for a subset of parameters.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_convergence_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_convergence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Convergence Diagnostics</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Fits mock photometry and displays a corner plot with injected truth values marked. Uses tengri&#x27;s safe_corner utility.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_corner_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_corner`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Corner Plot with Truth Overlay</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hierarchical inference recovers the shared PSD parameters (σ, τ) of a galaxy population. The posterior width on σ scales as 1/√N_galaxies, while individual fits are far too uncertain. This illustrates why population-level inference is essential for measuring burstiness.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_hierarchical_convergence_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_hierarchical_convergence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Population PSD Recovery: 1/√N Convergence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compares point-estimate (MAP) and variational (vi/geoVI) inference on mock 5-band photometry. Overlays posteriors as a corner plot.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_method_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_method_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAP vs geoVI Posterior Comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Renders the wall-time / peak-memory / iteration scaling of tengri&#x27;s two pure-JAX population variational engines on a 5-band SDSS photometry catalog with a stochastic-SFH forward model:">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_population_scaling_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_population_scaling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Population VI scaling: time, memory, and convergence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A Bayesian fit uses prior distributions over parameters and refines them using observed data to obtain posteriors. This script shows how priors (dashed lines) and posteriors (histograms) differ for key physical parameters (stellar mass age, metallicity, dust optical depth) after fitting mock photometry.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_prior_posterior_compare_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_prior_posterior_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Prior vs Posterior: Parameter Constraints from Inference</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/inference/plot_convergence
   /auto_examples/inference/plot_corner
   /auto_examples/inference/plot_hierarchical_convergence
   /auto_examples/inference/plot_method_comparison
   /auto_examples/inference/plot_population_scaling
   /auto_examples/inference/plot_prior_posterior_compare

