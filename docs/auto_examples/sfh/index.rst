

.. _sphx_glr_auto_examples_sfh:

Star Formation Histories
========================

Parametric and stochastic star formation history models.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four representative (σ, τ) pairs define burstiness regimes: Smooth (σ=0.3, τ=100 Myr), Moderate (σ=1.0, τ=50 Myr), Bursty (σ=2.0, τ=20 Myr), and Extreme (σ=3.0, τ=5 Myr). Each panel shows one forward-model draw with the smooth mean SFH overlaid, illustrating the range of morphologies that each regime produces before inference.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_bursty_recovery_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_bursty_recovery`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Four regimes of stochastic-SFH burstiness from smooth to extreme</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four perspectives on chemical evolution: (1) closed-box model with varying SFR timescales; (2) cumulative metallicity from different exponential SFHs; (3) leaky-box model showing how outflow rates suppress Z; and (4) age-metallicity relation across galactic radii. Together they show how star formation and galactic winds control the Z(t) history.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_chemical_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_chemical_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Chemical evolution: How SFH and outflows shape metal enrichment history</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The timescale τ of a delayed-exponential SFH sets how quickly star formation falls after its peak: short τ means rapid decline and old stars, long τ means a sustained tail and younger mean age. We vary τ across the prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dexp_tau_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dexp_tau_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Delayed-exponential timescale τ controls decay after peak SFR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing how the rising slope α (columns) and falling slope β (rows) together control the full SFH morphology. Early-time α determines assembly speed; late-time β sets the post-peak decay. The optical SED responds across each cell, revealing how parameter space maps to stellar age.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dpl_alpha_beta_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dpl_alpha_beta_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double power-law SFH parameter space: early growth α vs late quenching β</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The rising slope α of a double-power-law star formation history controls how rapidly the galaxy assembled its mass before the peak. Larger α means a more abrupt onset of star formation, leaving a younger O/B-star population at the time of observation and a steeper rest-frame UV slope. We vary α across the prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dpl_alpha_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dpl_alpha_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Early-time SFH slope α shapes the UV continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The falling slope β of a double-power-law SFH controls quenching after the peak. Large β means rapid quenching and an old stellar population; small β means a gentle tail and more mixed ages. We vary β across its prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dpl_beta_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dpl_beta_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Post-peak quenching slope β shapes stellar age distribution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The peak lookback time of a log-normal SFH controls when most stars formed, shifting the age structure and dramatically affecting UV slope, 4000 Å break strength, and NIR luminosity. We vary the peak time across its prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_lnorm_peak_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_lnorm_peak_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Log-normal peak lookback time shifts stellar age and SED morphology</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare all parametric SFH models available in tengri. Each is evaluated on a lookback-time grid with representative parameters, showing the range of morphologies from smooth exponentials to sharp truncations. No SSP data required.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_parametric_sfh_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_parametric_sfh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Catalog of parametric star-formation-history models</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three PSD models govern the frequency structure of stochastic SFHs: the default damped random walk (DRW), the Matern family (which includes DRW as a special case), and the extended regulator model. Plotted in frequency space at representative parameters. No SSP data required.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_alternatives_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_alternatives`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparison of power-spectral-density models for stochastic SFHs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing five stochastic-SFH realizations for each combination of amplitude σ (vertical axis) and damping timescale τ (horizontal axis). Larger σ produces more dramatic bursts; longer τ sustains those bursts. Each panel shows the mean smooth SFH (dashed) and colored realizations, revealing how the two PSD parameters together map to observable burstiness regimes.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_burstiness_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_burstiness`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD parameter space: amplitude σ (rows) and timescale τ (columns) control burstiness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The amplitude σ of the power spectral density sets how dramatically star formation fluctuates around the smooth trend: σ ≈ 0 means nearly constant SFR, large σ produces dramatic bursts that leave imprints in UV slope, optical colors, and stellar masses. We vary σ across its prior range with the timescale τ fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_sigma_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_sigma_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD amplitude σ controls burst magnitude in stochastic SFHs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The damping timescale τ (in Myr) of the power spectral density governs how long star-formation bursts persist. Short τ means rapid flickering; long τ means sustained episodes that leave their imprint on the SED. We vary τ across the prior range with the burst amplitude σ fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_tau_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_tau_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD timescale τ controls burst duration in stochastic SFHs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A galaxy with two separated bursts—one at 10 Gyr (old) and one at 0.3 Gyr (recent)— produces a SED that blends young hot and old cool stellar populations. Left panel shows the optical-to-NIR region in linear scale; right panel shows the full panchromatic SED in log-log, revealing the emission from both young and old stars.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_double_burst_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_double_burst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dual-epoch star formation: old and recent bursts leave distinct SED signatures</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each parametric SFH in tengri encodes a different prior on when a galaxy forms its stars. We overlay the SFR(t) shape of nine production-status forms at their default parameter values, all integrated to the same total stellar mass, so the differences are entirely in the shape — not the normalisation.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Four quenching scenarios—constant SFR, exponential decline, sharp truncation, and recent burst—produce distinct SED shapes. Constant SFR yields a young, blue galaxy; sharp quenching creates old red colors; a recent burst injects young stars atop an old population. The SED reveals the full assembly history.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_quenching_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_quenching_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quenching morphology sets the age mix and resulting SED colors</div>
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


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/sfh/plot_bursty_recovery
   /auto_examples/sfh/plot_chemical_evolution
   /auto_examples/sfh/plot_dexp_tau_sweep
   /auto_examples/sfh/plot_dpl_alpha_beta_grid
   /auto_examples/sfh/plot_dpl_alpha_sweep
   /auto_examples/sfh/plot_dpl_beta_sweep
   /auto_examples/sfh/plot_lnorm_peak_sweep
   /auto_examples/sfh/plot_parametric_sfh
   /auto_examples/sfh/plot_psd_alternatives
   /auto_examples/sfh/plot_psd_burstiness
   /auto_examples/sfh/plot_psd_sigma_sweep
   /auto_examples/sfh/plot_psd_tau_sweep
   /auto_examples/sfh/plot_sfh_double_burst
   /auto_examples/sfh/plot_sfh_form_compare
   /auto_examples/sfh/plot_sfh_nonparametric_compare
   /auto_examples/sfh/plot_sfh_quenching_compare
   /auto_examples/sfh/plot_stochastic_sfh

