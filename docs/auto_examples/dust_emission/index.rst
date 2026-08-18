:orphan:

.. _sphx_glr_auto_examples_dust_emission:

Dust Emission
=============

Dust emission templates auto-load from ``data/``; analytic fallbacks are not suitable for science. PAH features in Draine & Li templates (q_PAH and U_min sweeps). Mid-IR PAH diagnostics distinguish star-forming, AGN, and composite systems. Temperature sweeps. Template libraries: BOSA, THEMIS, PAHspec, Astrodust (HD23).


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The BOSA infrared template library is parametrized jointly by total infrared luminosity log L_TIR and specific star formation rate log sSFR. Neither axis alone tells the full story: at fixed sSFR the FIR peak migrates with L_TIR (dust temperature), while at fixed L_TIR the PAH mid-IR forest brightens with sSFR. Three side-by-side panels at fixed sSFR overlay three L_TIR values each, making the 2-D dependence legible in a single figure rather than two skinny 1-D loops.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA library: PAH features and FIR peak depend on both sSFR and L_TIR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 2-D grid on the Draine &amp; Li 2007 template library: rows step through PAH mass fraction q_PAH (controls mid-IR PAH-feature strength), columns through the minimum radiation field U_min (sets the diffuse dust temperature, i.e. the FIR peak position). The two axes act nearly orthogonally — a surprise for anyone who would lump them together as &quot;PAH knobs.&quot;">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_qpah_umin_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_qpah_umin_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The q_PAH and U_min knobs move PAH amplitude and FIR peak independently</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All dust IR-emission libraries shipped in tengri, shown on two scales:">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_ir_library_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_ir_library_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR-emission library comparison: models and templates</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Casey 2012 modified blackbody dust SEDs across the canonical fitter&#x27;s two knobs — dust temperature T_dust and emissivity index β. Each curve in the top panel is a fixed β = 1.8 MBB swept in T; the bottom panel fixes T = 30 K and sweeps β. The peak shifts by ~40 μm per 10 K of warming; the sub-mm slope steepens by one power-law index per Δβ = 1.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_mbb_temperature_beta_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_mbb_temperature_beta_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modified blackbody: T_dust × β grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 5–30 μm rest-frame spectrum showcases distinct infrared tracers: dust polycyclic aromatic hydrocarbon (PAH) emission peaks at 6.2, 7.7, 8.6, 11.3, and 12.7 μm in star-forming galaxies, while silicate absorption (9.7 μm Si–O stretch) and AGN heating suppress PAH and introduce continuum growth in AGN-dominated systems. We model three templates: (a) pure starburst (no AGN), (b) pure AGN (no star formation), and (c) composite with AGN fraction = 0.5. the diagnostic power of mid-IR spectroscopy: PAH strength probes star formation rate, while continuum slope and silicate depth reveal AGN heating and dust temperature.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_mid_ir_pah_features_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_mid_ir_pah_features`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mid-IR PAH features in star-forming, AGN, and composite galaxies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep across the 13 published PAHspec starlight spectra (mMMP, m31bulge, BC03/BPASS SSPs) at fixed ionization parameter. Demonstrates strong dependence of PAH features on starlight hardness.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_pahspec_starlight_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_pahspec_starlight_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine+2021 PAHspec: starlight-spectrum sweep at fixed log U</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For Draine &amp; Li (2007) dust at fixed mass, raising the diffuse radiation field intensity U_min does two things at once: it shifts the SED peak blueward (warmer dust) and proportionally boosts the total far-IR luminosity (``L_IR`` ∝ U_min). The standard T_peak–``L_IR`` correlation seen in observations is the joint footprint of these two effects.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_tdust_vs_lir_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_tdust_vs_lir`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radiation field strength sets both dust peak temperature and L_IR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Jones et al. (2017) THEMIS dust model distributes grains over a range of starlight intensities U with a power law dU/dM \propto U^{-\alpha}. The slope alpha controls how much warm, intensely-illuminated dust contributes relative to the cold diffuse component: a smaller alpha puts more mass at high U, shifting the FIR peak blueward and filling in the mid-IR.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_alpha_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_alpha_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS dust IR: radiation-field slope (alpha)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust re-radiates absorbed starlight across a broad range of temperatures: colder dust (e.g., diffuse cirrus at ~20 K) peaks in the far-infrared (~250 μm), while warmer dust grains (e.g., starburst regions at ~40 K) peak at shorter wavelengths (~50–100 μm).">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_warm_cold_dust_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_warm_cold_dust_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR SED: Warm and cold dust decomposition</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/dust_emission/plot_bosa_grid
   /auto_examples/dust_emission/plot_dust_qpah_umin_grid
   /auto_examples/dust_emission/plot_ir_library_compare
   /auto_examples/dust_emission/plot_mbb_temperature_beta_grid
   /auto_examples/dust_emission/plot_mid_ir_pah_features
   /auto_examples/dust_emission/plot_pahspec_starlight_sweep
   /auto_examples/dust_emission/plot_tdust_vs_lir
   /auto_examples/dust_emission/plot_themis_alpha_sweep
   /auto_examples/dust_emission/plot_warm_cold_dust_decomposition

