:orphan:

.. _sphx_glr_auto_examples_workflows:

Workflows
=========

End-to-end fitting workflows — BPT classification, dust Monte-Carlo
resampling, high-z LBG fits, method comparison, and post-starburst recovery.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="If the data is informative the MAP estimate sits at the likelihood maximum and the choice of prior barely matters. If the data is uninformative the MAP slides toward the prior mode.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_prior_systematic_dust_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_prior_systematic_dust`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">How priors push the dust posterior — flat vs narrow prior on τ_diff</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates BPT ([OIII]/Hβ vs [NII]/Hα) line ratios computed directly from the model&#x27;s rest-frame SED via continuum-subtracted boxcar integration around each line center, swept across a stellar metallicity grid. The Kewley+2001 and Kauffmann+2003 demarcation lines are overlaid for context.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_bpt_classification_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_bpt_classification`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram: emission lines from the baked-in nebular SSP</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates dust attenuation effects and how fitting uncertainty propagates to the recovered SED. A galaxy with free dust parameters (tau_bc and tau_diff) is fit with MAP, showing the best-fit SED plus mock perturbation envelopes to illustrate the uncertainty range from photometric noise.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_dust_mc_resampling_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_dust_mc_resampling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation: uncertainty in SED from dust parameter estimation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Fits a z=4 young, dust-free star-forming galaxy using JWST (F150W/F200W/F277W) and HST (F814W) broadband photometry. The characteristic Lyman-break signature (sharp UV dropout at observed ~4 micron) constrains age and metallicity even with just 4 bands. Demonstrates recovery of the young starburst component from the dropout depth.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_high_z_lbg_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_high_z_lbg`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">High-redshift Lyman-break galaxy: Lyman dropout signatures in JWST/HST</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates the simplest inference workflow: building a flexible SFH model with free dust parameters, generating mock photometry at S/N = 20, then running MAP to recover the input star formation history and dust attenuation. The figure shows the recovered SFH (dashed) against the ground truth (solid).">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_method_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_method_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recovering a truncated-skew-normal SFH from SDSS photometry via MAP</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two galaxies with very different physical properties — a dusty star-forming galaxy at z=0.3 and an unobscured Lyman-break galaxy at z=3.5 — can produce nearly identical ugrizY broadband fluxes. The 4000 Å break of the dusty low-z galaxy and the Lyman break of the high-z galaxy land at the same observed wavelength, so without intermediate bands or IR coverage the photo-z is bimodal.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_photoz_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_photoz_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The photo-z degeneracy: dusty z≈0.3 vs unobscured z≈3.5</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A post-starburst galaxy shows a recent burst followed by quenching. When fit with smooth tau-model (incorrect), the fit biases the recovered SFH. This workflow compares two models on the same mock data to show how model flexibility directly impacts star formation history inference.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_post_starburst_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_post_starburst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Model misspecification: post-starburst galaxies reveal wrong SFH</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When did star formation in a galaxy stop? Optical-only color, the 4000 Å break, and Hα equivalent width respond on different timescales: NUV − r reddens within ~100 Myr of quenching (loss of O/B stars), D_n(4000) continues to rise over 1–3 Gyr as A stars evolve, and Hα EW drops fastest of all (within ~10 Myr) since it tracks only the youngest ionizing photons.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_quenching_diagnostics_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_quenching_diagnostics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Three diagnostics of quenching epoch in one figure</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A hydro simulation or semi-analytic model hands you two tables per galaxy — SFR(t) and the metallicity of the gas it formed from, Z(t). Catalog takes both as records and returns photometry for the whole population in one compile.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_simulation_seds_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_simulation_seds`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Predicting SEDs for a simulated population: what collapsing Z(t) costs</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/workflows/plot_prior_systematic_dust
   /auto_examples/workflows/plot_workflow_bpt_classification
   /auto_examples/workflows/plot_workflow_dust_mc_resampling
   /auto_examples/workflows/plot_workflow_high_z_lbg
   /auto_examples/workflows/plot_workflow_method_comparison
   /auto_examples/workflows/plot_workflow_photoz_degeneracy
   /auto_examples/workflows/plot_workflow_post_starburst
   /auto_examples/workflows/plot_workflow_quenching_diagnostics
   /auto_examples/workflows/plot_workflow_simulation_seds

