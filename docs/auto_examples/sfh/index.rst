:orphan:

.. _sphx_glr_auto_examples_sfh:

Star Formation Histories
========================

Parametric forms (DPL, delayed-exponential, lognormal) and non-parametric (PSD-governed stochastic). Quenching pathways, burst observability, and SFH form comparison.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed mean SFH and stellar mass, continuity (Leja+2019) and field (PSD-governed) priors yield strikingly different stochastic realizations: continuity produces smooth log-normal transitions; field produces controlled burstiness governed by σ_field.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_continuity_vs_bursty_psd_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_continuity_vs_bursty_psd`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Continuity Prior vs PSD-Governed Prior: Stochastic Structure at Fixed Mean</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing how the rising slope α (columns) and falling slope β (rows) together control the full SFH morphology. Early-time α determines assembly speed; late-time β sets the post-peak decay. The optical SED responds across each cell. Bottom panels show representative 1D sweeps: α alone (left, at fixed β) and β alone (right, at fixed α), illustrating how each parameter independently shapes the full UV-to-IR SED.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dpl_alpha_beta_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dpl_alpha_beta_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double power-law SFH parameter space: early growth α vs late quenching β</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing five stochastic-SFH realizations for each combination of amplitude σ (vertical axis) and damping timescale τ (horizontal axis). Larger σ produces more dramatic bursts; longer τ sustains those bursts. Each panel shows the mean smooth SFH (dashed) and colored realizations. Bottom panels show representative SEDs for σ alone (left) and τ alone (right), illustrating how each parameter independently shapes the UV continuum and optical colors.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_burstiness_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_burstiness`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD parameter space: amplitude σ and timescale τ control burstiness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare three star-formation histories representing distinct quenching scenarios: (1) Constantly star-forming (no quenching), (2) Slowly quenched exponential decay (tau=4 Gyr, peak 6 Gyr ago), and (3) Rapidly quenched post-starburst (truncated skew-normal, peak 2 Gyr ago, width 0.3 Gyr). The resulting rest-frame SEDs exhibit markedly different colors, equivalent widths (Hα), and spectral slopes, highlighting how quenching timescale imprints on observable photometry and spectroscopy.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_quenching_pathway_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_quenching_pathway_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quenching pathways: fast vs slow termination of star formation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="CIGALE&#x27;s sfh2exp star-formation history superposes an old, exponentially declining main population with a second, more recent exponential burst that contributes a fixed fraction f_burst of the total stellar mass formed. It is the classic parametrization for post-starburst and rejuvenated systems.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh2exp_main_plus_burst_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh2exp_main_plus_burst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">sfh2exp: double declining exponential (old population + recent burst)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each parametric SFH in tengri encodes a different prior on when a galaxy forms its stars. We overlay the SFR(t) shape of nine production-status forms at their default parameter values, all integrated to the same total stellar mass, so the differences are entirely in the shape — not the normalization.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_form_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_form_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parametric SFH form atlas</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The parametric SFH atlas (``plot_sfh_form_compare.py``) shows seven classical analytic SFH shapes. Beyond those, tengri ships three non-parametric families that bin the mass formed in successive lookback intervals — useful when the data resolve more than ~5 SFR bins and you want a flexible prior that doesn&#x27;t impose a strong shape.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_nonparametric_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_nonparametric_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Non-parametric SFH families compared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Generate stochastic star-formation histories using the Fourier-space GP correlated field model, governed by a damped-random-walk power spectrum. Left panel shows mild burstiness (σ=0.3, τ=300 Myr); right shows strong burstiness (σ=1.0, τ=100 Myr). Five realizations appear in each panel, with the smooth mean SFH overlaid.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_stochastic_sfh_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_stochastic_sfh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stochastic SFH samples from GP-correlated fields with different burstiness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How observable is an underlying ancient burst (10 Gyr ago) beneath a young (300 Myr) starburst? outshining problem in broadband photometry (Trager+ 2000, Renzini 2006): the young burst&#x27;s UV emission completely dominates over the ancient burst&#x27;s optical/IR, rendering the ancient population invisible to broadband SED fitting.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_two_burst_observability_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_two_burst_observability`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The outshining problem: young bursts eclipse ancient populations</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/sfh/plot_continuity_vs_bursty_psd
   /auto_examples/sfh/plot_dpl_alpha_beta_grid
   /auto_examples/sfh/plot_psd_burstiness
   /auto_examples/sfh/plot_quenching_pathway_compare
   /auto_examples/sfh/plot_sfh2exp_main_plus_burst
   /auto_examples/sfh/plot_sfh_form_compare
   /auto_examples/sfh/plot_sfh_nonparametric_compare
   /auto_examples/sfh/plot_stochastic_sfh
   /auto_examples/sfh/plot_two_burst_observability

