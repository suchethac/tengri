

.. _sphx_glr_auto_examples_workflows:

Workflows
=========

End-to-end fitting workflows — BPT classification, dust Monte-Carlo
resampling, high-z LBG fits, method comparison, and post-starburst recovery.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates computing emission-line ratios for a mock galaxy catalog with mixed star-forming and AGN fractions. Plots the BPT diagram ([OIII]/Hβ vs [NII]/Hα) and overlays Kewley+2001 and Kauffmann+2003 demarcation lines to show how emission-line diagnostics separate ionization mechanisms.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_bpt_classification_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_bpt_classification`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Workflow: BPT Emission-Line Classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates quantifying observational uncertainties through posterior predictive resampling. A galaxy is fit with NUTS, then the posterior is resampled 200 times to generate a posterior predictive SED ensemble. Shows the SED with 1σ and 2σ confidence envelopes. This workflow illustrates how to propagate Bayesian posterior uncertainty into derived predictions for robust error budgets.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_dust_mc_resampling_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_dust_mc_resampling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Workflow: Dust Attenuation Uncertainty via Posterior Resampling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates fitting a z=4 Lyman-break galaxy with JWST/HST photometry. A young, dust-free star-forming galaxy&#x27;s SED shows a sharp UV dropout. This workflow shows how to recover age, dust, and redshift from the characteristic Lyman-break signature in broadband colors.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_high_z_lbg_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_high_z_lbg`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Workflow: High-z Lyman-Break Galaxy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compares three inference methods on identical mock data: MAP (point estimate), geoVI/VI (variational approximation), and NUTS (gold-standard MCMC). Shows how each method differs in capturing posterior shape and uncertainty. MAP underestimates uncertainty; VI approximates the shape; NUTS is the reference. This workflow demonstrates method choice tradeoffs for practitioners.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_method_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_method_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Workflow: Inference Method Comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates identifying post-starburst galaxies through model comparison. A post-starburst has a truncated SFH with a recent burst followed by quenching. When fit with a smooth tau model (incorrect), the fit poorly recovers the truth. This shows how model misspecification can bias SFH inference and why flexible models matter for interpreting star formation histories.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_post_starburst_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_post_starburst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Workflow: Post-Starburst (E+A) Galaxy Diagnosis</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/workflows/plot_workflow_bpt_classification
   /auto_examples/workflows/plot_workflow_dust_mc_resampling
   /auto_examples/workflows/plot_workflow_high_z_lbg
   /auto_examples/workflows/plot_workflow_method_comparison
   /auto_examples/workflows/plot_workflow_post_starburst

