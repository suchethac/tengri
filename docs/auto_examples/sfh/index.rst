:orphan:

.. _sphx_glr_auto_examples_sfh:

Star Formation Histories
========================

Default kernel: CIC (cloud-in-cell). `'dsps'` kernel available for cross-code parity but interpolates in log-space, annihilating the first SSP node (~3.8% mass loss) and shifting age gradients by 43%. Parametric forms (DPL, delayed-exponential, lognormal) and non-parametric (PSD-governed stochastic). Mismatch between true and assumed SFH form can produce unrecognizable posteriors.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The bursty continuity prior (Tacchella+2022, ApJ 926, 134) shares the piecewise-constant continuity SFH with Leja+2019 but doubles the Student-t scale on log-SFR ratios whose younger bin edge is recent (&lt; 1 Gyr lookback). The result is a prior that lets recent SFR variations swing by ~1 dex while keeping older history smooth (σ = 0.3 dex).">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_bursty_continuity_sigma_schedule_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_bursty_continuity_sigma_schedule`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bursty continuity prior: bin-edge-dependent σ schedule (Tacchella+2022)</div>
    </div>


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

    <div class="sphx-glr-thumbcontainer" tooltip="Six perspectives on chemical evolution: (1) closed-box model with varying SFR timescales; (2) cumulative metallicity from different exponential SFHs; (3) leaky-box model showing how outflow rates suppress Z; (4) age-metallicity relation across galactic radii; (5) three metallicity evolution scenarios (constant solar, linear ramp, two-step); (6) resulting integrated SEDs showing how Z(t) pathways alter optical/UV colors and absorption features. Together they show how star formation, galactic winds, and chemical enrichment control the Z(t) history and observable photometry.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_chemical_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_chemical_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Chemical evolution: How SFH and outflows shape metal enrichment history</div>
    </div>


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

    <div class="sphx-glr-thumbcontainer" tooltip="The timescale τ of a delayed-exponential SFH sets how quickly star formation falls after its peak: short τ means rapid decline and old stars, long τ means a sustained tail and younger mean age. We vary τ across the prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dexp_tau_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dexp_tau_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Delayed-exponential timescale τ controls decay after peak SFR</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduction of the star-formation-history overview figure of Buchner et al. (2024, GRAHSP): galaxy SEDs for a delayed-\tau SFH (\mathrm{SFR}\propto t\,e^{-t/\tau}, CIGALE sfh_delayed) whose cutoff timescale \tau is swept from 100 Myr (yellow; SFR truncates early, old-star-dominated) to 10 Gyr (dark blue; continuously rising, young, nebular- and dust-rich). Minimal attenuation E(B-V)=0.01 is applied. The inset shows the corresponding star-formation histories.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_grahsp_paper_sfh_tau_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_grahsp_paper_sfh_tau_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP SFH figure reproduction: delayed-tau galaxy SED sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The peak time of a log-normal SFH controls when most stars formed, shifting the age structure and dramatically affecting UV slope, 4000 Å break strength, and NIR luminosity. Following Carnall+2018 / BAGPIPES, the peak is measured in cosmic time since formation (T = age - lookback); larger peak times correspond to more recent star formation. We vary the peak across its prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_lnorm_peak_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_lnorm_peak_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Log-normal peak time shifts stellar age and SED morphology</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tengri ships the non-parametric SFH priors that appear most often in Prospector papers, all with the published prior on the SFR ratios:">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_prospector_priors_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_prospector_priors_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Prospector prior families: continuity vs bursty vs Dirichlet vs PSB</div>
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


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How observable is an underlying ancient burst (10 Gyr ago) beneath a young (300 Myr) starburst? outshining problem in broadband photometry (Trager+ 2000, Renzini 2006): the young burst&#x27;s UV emission completely dominates over the ancient burst&#x27;s optical/IR, rendering the ancient population invisible to broadband SED fitting.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_two_burst_observability_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_two_burst_observability`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The outshining problem: young bursts eclipse ancient populations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A common SED-fitting failure mode: pick a smooth parametric SFH (delayed exponential, tau-model, lognormal) for a galaxy whose true star-formation history has short-timescale bursts. The continuum-anchored bands (optical, NIR) absorb the mass and the fit looks plausible — but the UV bands, where young O/B stars dominate, carry the residual of the recent burst.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_wrong_model_trap_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_wrong_model_trap`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fitting a stochastic SFH with a smooth parametric prior leaves a UV residual</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/sfh/plot_bursty_continuity_sigma_schedule
   /auto_examples/sfh/plot_bursty_recovery
   /auto_examples/sfh/plot_chemical_evolution
   /auto_examples/sfh/plot_continuity_vs_bursty_psd
   /auto_examples/sfh/plot_dexp_tau_sweep
   /auto_examples/sfh/plot_dpl_alpha_beta_grid
   /auto_examples/sfh/plot_grahsp_paper_sfh_tau_sweep
   /auto_examples/sfh/plot_lnorm_peak_sweep
   /auto_examples/sfh/plot_prospector_priors_compare
   /auto_examples/sfh/plot_psd_burstiness
   /auto_examples/sfh/plot_quenching_pathway_compare
   /auto_examples/sfh/plot_sfh2exp_main_plus_burst
   /auto_examples/sfh/plot_sfh_form_compare
   /auto_examples/sfh/plot_sfh_nonparametric_compare
   /auto_examples/sfh/plot_sfh_quenching_compare
   /auto_examples/sfh/plot_stochastic_sfh
   /auto_examples/sfh/plot_two_burst_observability
   /auto_examples/sfh/plot_wrong_model_trap

