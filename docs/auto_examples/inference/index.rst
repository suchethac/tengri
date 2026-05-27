:orphan:

.. _sphx_glr_auto_examples_inference:

Inference Methods
=================

Comparing inference methods and convergence diagnostics.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The convergence diagnostic shows how the negative log posterior (loss) decays across optimizer iterations. We fit mock photometry using MAP (maximum a posteriori) optimization with Adam and display the loss curve, showing when the optimizer has effectively converged. The right panel overlays the recovered SFH against the truth.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_convergence_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_convergence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAP fit convergence: loss decay across iterations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates parameter degeneracies and individual 1-D marginalized posteriors after fitting mock 5-band SDSS photometry. The corner plot shows the full 2-D covariance structure between parameters; blue lines mark the injected truth. Note: for demonstration scale; production runs use 10× more VI iterations and samples.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_corner_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_corner`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Posterior corner plot from variational inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Fit a population of 50 mock galaxies under a hierarchical prior on a shared population-level metallicity parameter. The hierarchical model pools information across galaxies to tighten constraints on the population-level mean — a key differentiator of tengri&#x27;s inference stack.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_hierarchical_population_fit_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_hierarchical_population_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hierarchical population fit with shared metallicity hyperprior</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates how adding rest-frame optical emission-line equivalent widths (H-alpha, [OIII]) to broadband photometry dramatically tightens parameter constraints and breaks the notorious photo-z/dust degeneracy. Two panels compare posterior widths: (a) redshift posterior with and without line constraints, and (b) dust attenuation posterior showing dramatic improvement.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_joint_photometry_line_fit_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_joint_photometry_line_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Joint photometry + emission-line fitting breaks photo-z and dust degeneracies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates convergence behavior of two inference methods: MAP (point-estimate via optimization) and pure-JAX geometric variational inference (native VI). Both are initialized from the same MAP fit, then evolve independently to show how they explore the posterior. The SFH panel on the right shows the recovered star-formation history from each method overlaid on the truth.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_method_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_method_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparing MAP and pure-JAX variational inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates the n_chains knob on the gradient-based MCMC backends. Each chain shares the cached step size and mass matrix (so this is only meaningful on the second call against the same model), then jax.vmap dispatches the chains in parallel across XLA SIMD lanes (CPU) or accelerator cores (GPU/TPU).">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_multichain_speedup_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_multichain_speedup`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-chain MCMC speedup via jax.vmap</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates MCMC parameter estimation and posterior covariance structure after fitting mock 5-band SDSS photometry with a double-power-law (dpl) star formation history. The corner plot visualizes all 1-D marginalized posteriors and 2-D joint distributions, with blue lines marking the injected truth values.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_posterior_corner_dpl_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_posterior_corner_dpl`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Posterior corner plot from MCMC: double-power-law SFH</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The posterior predictive check (PPC) is the gold-standard Bayesian goodness-of-fit diagnostic (Rubin 1984; Gelman et al. 1996). We fit mock SDSS photometry, draw 100 samples from the posterior, regenerate mock photometry for each sample, and overlay the band-by-band model envelope (16th, 50th, 84th percentiles) against observed data with residuals normalized by noise. For well-fit models, residuals cluster within ±2σ.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_posterior_predictive_check_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_posterior_predictive_check`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Posterior predictive check: Bayesian goodness-of-fit diagnostic</div>
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
   /auto_examples/inference/plot_corner
   /auto_examples/inference/plot_hierarchical_population_fit
   /auto_examples/inference/plot_joint_photometry_line_fit
   /auto_examples/inference/plot_method_comparison
   /auto_examples/inference/plot_multichain_speedup
   /auto_examples/inference/plot_photoz_chi2_grid
   /auto_examples/inference/plot_posterior_corner_dpl
   /auto_examples/inference/plot_posterior_predictive_check
   /auto_examples/inference/plot_prior_predictive
   /auto_examples/inference/plot_sfh_recovery_test

