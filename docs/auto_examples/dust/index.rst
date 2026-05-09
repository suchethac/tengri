

.. _sphx_glr_auto_examples_dust:

Dust Models
===========

Attenuation laws, two-component dust, and IR emission.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep log10 U over the published [-3, +6] range of the Hensley &amp; Draine 2023 Astrodust+PAH grid (91 lgU points, finer than Draine+2021 PAHspec&#x27;s 15-point grid).  Shows the FIR peak shifting blueward and the MIR PAH features rising as the radiation field intensifies.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_astrodust_hd23_lgU_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_astrodust_hd23_lgU_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hensley & Draine 2023 Astrodust+PAH: log U sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All major attenuation laws implemented in tengri evaluated at fixed τ_V = 1.0. Shows wavelength dependence (k(λ)) from UV through near-infrared, highlighting the UV bump (2175 Å) and the steepness differences between Milky Way, SMC, and starburst models.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_attenuation_law_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_attenuation_law_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Attenuation Law Comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep log10 L_TIR over the full published 41-point grid (8.5 to 12.5 dex in 0.1 dex steps) at fixed log10 sSFR = -9.6 (typical star-forming galaxy). Increasing L_TIR makes the dust hotter → FIR peak shifts blueward and PAH features become more prominent relative to the FIR continuum.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_bosa_ltir_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_bosa_ltir_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA: log L_TIR sweep at fixed log sSFR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep log10(sSFR / yr^-1) across the canonical 14-point BOSA grid (Boquien &amp; Salim 2021) at fixed log10 L_TIR = 11 (typical LIRG luminosity).  Higher sSFR → harder mid-IR colour and stronger PAH features; quiescent (low-sSFR) galaxies have a colder FIR peak.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_bosa_ssfr_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_bosa_ssfr_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA: log sSFR sweep at fixed log L_TIR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The characteristic temperature T of the modified blackbody dust emission controls the peak wavelength of far-infrared emission. Wien&#x27;s law: λ_peak ≈ 2900 μm·K / T. Hotter dust (higher T) peaks at shorter wavelengths (more mid-IR), cooler dust peaks further into the far-IR/submm.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_T_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_T_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modified Blackbody Dust Temperature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plot all available attenuation laws in tengri. Each curve k(lambda) describes the wavelength dependence of dust attenuation, normalized at 5500 A. No SSP data required.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_curves_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_curves`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust Attenuation Curves</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All tengri dust emission models evaluated at L_absorbed = 1e10 L☉ and T = 35 K. Template-based models (DL07, DL14, Dale+2014) are shown only when data files are present; they are skipped gracefully otherwise.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_emission_models_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_emission_models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust Emission Models: Overview</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare transmission curves for three dust geometries (Witt &amp; Gordon 2000): screen (foreground), mixed (slab), and clumpy (two-phase medium). At fixed optical depth τ_V = 1.0, geometry controls the spectral shape: screens are reddest, mixed intermediate, clumpy greyest.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_geometry_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_geometry_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust Geometry: Screen vs Mixed vs Clumpy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="2D grid of dust IR emission spectra showing how PAH mass fraction (q_PAH) and ISRF hardness (U_min) independently shape the mid- and far-infrared SED. Uses Draine &amp; Li 2007 templates. 3×3 panel grid covering q_PAH ∈ {0.5, 2.5, 4.5}% and U_min ∈ {0.5, 2, 10} (MW-like to very hard radiation field).">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_qpah_umin_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_qpah_umin_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR SED: q_PAH × U_min Grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The power-law slope δ modifies the Calzetti attenuation curve shape. Negative δ steepens UV attenuation; positive δ flattens it. This controls whether dust absorbs more or less light at short wavelengths relative to optical.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_slope_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_slope_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Attenuation Curve Slope (δ)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep the starlight intensity log10 U over the published [0, 7] range of the Draine, Li, Hensley et al. 2021 PAHspec library at a fixed (mMMP starlight, standard ionization, standard size distribution) configuration.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_pahspec_lgU_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_pahspec_lgU_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine+2021 PAHspec: log U sweep at fixed (starlight, ion, size)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep the categorical starlight config across the 13 published PAHspec choices (mMMP, m31bulge, BC03 / BPASS at various ages and metallicities) at fixed log10 U = 1 and the standard (ionization, size_distribution) defaults.  This shows that PAH features scale strongly with starlight hardness — the headline result of the paper.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_pahspec_starlight_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_pahspec_starlight_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine+2021 PAHspec: starlight-spectrum sweep at fixed log U</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The PAH mass fraction q_PAH controls the strength of polycyclic aromatic hydrocarbon (PAH) mid-infrared emission features at 3.3, 6.2, 7.7, 8.6, and 11.3 μm. Higher q_PAH → stronger PAH features. Range: 0.47–4.58 % for DL07, 0.47–7.32 % for DL14.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_qpah_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_qpah_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PAH Mass Fraction (q_PAH)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Birth-cloud dust optical depth τ_BC controls how much of the youngest stellar light escapes the cocoon. Higher τ_BC reddens the UV and suppresses nebular emission from embedded HII regions.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_tau_bc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_tau_bc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth Cloud Optical Depth (τ_BC)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The diffuse ISM attenuation affects all stellar light (not just young stars). Higher τ_diff reddens the optical continuum and weakens the 4000 Å break, a signature of aging stellar populations.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_tau_diff_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_tau_diff_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diffuse ISM Optical Depth (τ_diff)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep q_HAC (mass fraction of small a-C(:H) hydrocarbon grains &lt; 1.5 nm) across the canonical CIGALE-distributed Jones+2017 THEMIS grid at fixed U_min = 1 and alpha = 2.  The PAH-like mid-IR features at 3.3, 6.2, 7.7, 8.6, 11.3 μm strengthen with q_HAC; the FIR continuum is essentially unchanged.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_themis_qhac_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_themis_qhac_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: q_HAC sweep at fixed U_min</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep U_min over the full published 37-point CIGALE grid (0.10 to 80.0) at the fiducial q_HAC = 0.17 and alpha = 2. Higher U warms the dust → FIR peak shifts blueward and the MIR small-grain emission grows relative to the FIR cold peak.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_themis_umin_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_themis_umin_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: U_min sweep at fixed q_HAC</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Visualize the Charlot &amp; Fall (2000) two-component dust model: birth cloud attenuation affects only young stars (age &lt; ~10 Myr), while diffuse ISM attenuation affects all stars. The smooth sigmoid transition between components is shown as a function of stellar age.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_two_component_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_two_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Two-Component Dust SEDModel</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="U_min sets the intensity of the diffuse radiation field heating the dust. Higher U_min implies a hotter, more intense radiation field → hotter overall dust → shifted far-infrared peak to shorter wavelengths. Range: 0.1–25 for DL07, 0.1–50 for DL14.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_umin_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_umin_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine & Li Minimum Radiation Field (U_min)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 2175 Å UV bump in dust attenuation curves is usually attributed to PAHs or small graphite grains. Sweeping the bump amplitude from zero to MW-like takes the attenuation curve from a smooth power law to the characteristic MW shape.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_uv_bump_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_uv_bump_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV Bump Strength (dust_bump_strength)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/dust/plot_astrodust_hd23_lgU_sweep
   /auto_examples/dust/plot_attenuation_law_compare
   /auto_examples/dust/plot_bosa_ltir_sweep
   /auto_examples/dust/plot_bosa_ssfr_sweep
   /auto_examples/dust/plot_dust_T_sweep
   /auto_examples/dust/plot_dust_curves
   /auto_examples/dust/plot_dust_emission_models
   /auto_examples/dust/plot_dust_geometry_sweep
   /auto_examples/dust/plot_dust_qpah_umin_grid
   /auto_examples/dust/plot_dust_slope_sweep
   /auto_examples/dust/plot_pahspec_lgU_sweep
   /auto_examples/dust/plot_pahspec_starlight_sweep
   /auto_examples/dust/plot_qpah_sweep
   /auto_examples/dust/plot_tau_bc_sweep
   /auto_examples/dust/plot_tau_diff_sweep
   /auto_examples/dust/plot_themis_qhac_sweep
   /auto_examples/dust/plot_themis_umin_sweep
   /auto_examples/dust/plot_two_component
   /auto_examples/dust/plot_umin_sweep
   /auto_examples/dust/plot_uv_bump_sweep

