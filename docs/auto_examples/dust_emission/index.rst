

.. _sphx_glr_auto_examples_dust_emission:

Dust Emission
=============

How dust re-radiates absorbed starlight in the IR — PAH features and the
q_PAH / U_min sweeps of Draine & Li templates, modified-blackbody
temperature sweeps, and dives into the BOSA, THEMIS, PAHspec, and
Astrodust (HD23) template grids.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Per-H grain volume distribution (4\pi/3)\,a^3\,dn/d\ln a / n_{\rm H} versus grain radius for the Hensley &amp; Draine 2023 fiducial size distribution (MW high-latitude R_V=3.1 sightline), reading from the HDU 1 metadata embedded in tengri&#x27;s HDF5.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_01_size_distribution_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_01_size_distribution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH size distribution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plots \lambda I_\lambda / N_{\rm H} / U for several \log_{10} U values from the Hensley &amp; Draine 2023 grid. Dividing by U makes the U-dependence of the PAH-vs-FIR ratio visible: low-U curves stack atop each other in the FIR while the MIR features rise steeply with U.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_02_emission_vs_lgU_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_02_emission_vs_lgU`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH emission vs log U</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Astrodust vs PAH components at the fiducial U=1.6 (lgU=0.2).">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_03_components_at_fiducial_U_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_03_components_at_fiducial_U`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH per-component decomposition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Switch between dust IR templates with one config-field change.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_04_sedmodel_dust_emission_swap_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_04_sedmodel_dust_emission_swap`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DustEmissionSEDComponent — swap MBB / PAHspec / Astrodust</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="f_ion(a) and f_align(a) versus grain size — H&amp;D 2023 fiducials.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_05_ionization_alignment_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_05_ionization_alignment`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH ionization fraction and alignment</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Extinction, polarized extinction, and albedo — H&amp;D 2023 fiducial.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_06_extinction_and_scattering_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_06_extinction_and_scattering`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH extinction, scattering, and albedo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spinning dust microwave emission — H&amp;D 2023 fiducial.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_07_spinning_dust_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_07_spinning_dust`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH spinning-dust microwave emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Polarized emission and polarization fraction — H&amp;D 2023.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_08_polarized_emission_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_08_polarized_emission`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH polarized emission and polarization fraction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep log10 U over the published [-3, +6] range of the Hensley &amp; Draine 2023 Astrodust+PAH grid (91 lgU points, finer than Draine+2021 PAHspec&#x27;s 15-point grid).  Shows the FIR peak shifting blueward and the MIR PAH features rising as the radiation field intensifies.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_lgU_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_lgU_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hensley & Draine 2023 Astrodust+PAH: log U sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep log10 L_TIR over the full published 41-point grid (8.5 to 12.5 dex in 0.1 dex steps) at fixed log10 sSFR = -9.6 (typical star-forming galaxy). Increasing L_TIR makes the dust hotter → FIR peak shifts blueward and PAH features become more prominent relative to the FIR continuum.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_ltir_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_ltir_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA: log L_TIR sweep at fixed log sSFR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep log10(sSFR / yr^-1) across the canonical 14-point BOSA grid (Boquien &amp; Salim 2021) at fixed log10 L_TIR = 11 (typical LIRG luminosity).  Higher sSFR → harder mid-IR colour and stronger PAH features; quiescent (low-sSFR) galaxies have a colder FIR peak.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_ssfr_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_ssfr_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA: log sSFR sweep at fixed log L_TIR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The characteristic temperature T of the modified blackbody dust emission controls the peak wavelength of far-infrared emission. Wien&#x27;s law: λ_peak ≈ 2900 μm·K / T. Hotter dust (higher T) peaks at shorter wavelengths (more mid-IR), cooler dust peaks further into the far-IR/submm.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_T_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_T_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modified Blackbody Dust Temperature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All tengri dust emission models evaluated at L_absorbed = 1e10 L☉ and T = 35 K. Template-based models (DL07, DL14, Dale+2014) are shown only when data files are present; they are skipped gracefully otherwise.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_emission_models_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_emission_models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust Emission Models: Overview</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="2D grid of dust IR emission spectra showing how PAH mass fraction (q_PAH) and ISRF hardness (U_min) independently shape the mid- and far-infrared SED. Uses Draine &amp; Li 2007 templates. 3×3 panel grid covering q_PAH ∈ {0.5, 2.5, 4.5}% and U_min ∈ {0.5, 2, 10} (MW-like to very hard radiation field).">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_qpah_umin_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_qpah_umin_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR SED: q_PAH × U_min Grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep the starlight intensity log10 U over the published [0, 7] range of the Draine, Li, Hensley et al. 2021 PAHspec library at a fixed (mMMP starlight, standard ionization, standard size distribution) configuration.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_pahspec_lgU_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_pahspec_lgU_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine+2021 PAHspec: log U sweep at fixed (starlight, ion, size)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep the categorical starlight config across the 13 published PAHspec choices (mMMP, m31bulge, BC03 / BPASS at various ages and metallicities) at fixed log10 U = 1 and the standard (ionization, size_distribution) defaults.  This shows that PAH features scale strongly with starlight hardness — the headline result of the paper.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_pahspec_starlight_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_pahspec_starlight_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine+2021 PAHspec: starlight-spectrum sweep at fixed log U</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The PAH mass fraction q_PAH controls the strength of polycyclic aromatic hydrocarbon (PAH) mid-infrared emission features at 3.3, 6.2, 7.7, 8.6, and 11.3 μm. Higher q_PAH → stronger PAH features. Range: 0.47–4.58 % for DL07, 0.47–7.32 % for DL14.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_qpah_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_qpah_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PAH Mass Fraction (q_PAH)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep alpha (the power-law slope dU/dM \propto U^\alpha of the radiation-field distribution) over the full published 21-point CIGALE grid (1.0 to 3.0 in 0.1 dex steps) at the fiducial q_HAC = 0.17 and U_min = 1.0.  Lower \alpha puts more mass at high U → warmer SED with the FIR peak shifted blueward; higher \alpha collapses toward the single-U limit.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_alpha_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_alpha_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: power-law slope alpha sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep q_HAC (mass fraction of small a-C(:H) hydrocarbon grains &lt; 1.5 nm) across the canonical CIGALE-distributed Jones+2017 THEMIS grid at fixed U_min = 1 and alpha = 2.  The PAH-like mid-IR features at 3.3, 6.2, 7.7, 8.6, 11.3 μm strengthen with q_HAC; the FIR continuum is essentially unchanged.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_qhac_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_qhac_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: q_HAC sweep at fixed U_min</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep U_min over the full published 37-point CIGALE grid (0.10 to 80.0) at the fiducial q_HAC = 0.17 and alpha = 2. Higher U warms the dust → FIR peak shifts blueward and the MIR small-grain emission grows relative to the FIR cold peak.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_umin_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_umin_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: U_min sweep at fixed q_HAC</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="U_min sets the intensity of the diffuse radiation field heating the dust. Higher U_min implies a hotter, more intense radiation field → hotter overall dust → shifted far-infrared peak to shorter wavelengths. Range: 0.1–25 for DL07, 0.1–50 for DL14.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_umin_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_umin_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine & Li Minimum Radiation Field (U_min)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/dust_emission/plot_astrodust_hd23_01_size_distribution
   /auto_examples/dust_emission/plot_astrodust_hd23_02_emission_vs_lgU
   /auto_examples/dust_emission/plot_astrodust_hd23_03_components_at_fiducial_U
   /auto_examples/dust_emission/plot_astrodust_hd23_04_sedmodel_dust_emission_swap
   /auto_examples/dust_emission/plot_astrodust_hd23_05_ionization_alignment
   /auto_examples/dust_emission/plot_astrodust_hd23_06_extinction_and_scattering
   /auto_examples/dust_emission/plot_astrodust_hd23_07_spinning_dust
   /auto_examples/dust_emission/plot_astrodust_hd23_08_polarized_emission
   /auto_examples/dust_emission/plot_astrodust_hd23_lgU_sweep
   /auto_examples/dust_emission/plot_bosa_ltir_sweep
   /auto_examples/dust_emission/plot_bosa_ssfr_sweep
   /auto_examples/dust_emission/plot_dust_T_sweep
   /auto_examples/dust_emission/plot_dust_emission_models
   /auto_examples/dust_emission/plot_dust_qpah_umin_grid
   /auto_examples/dust_emission/plot_pahspec_lgU_sweep
   /auto_examples/dust_emission/plot_pahspec_starlight_sweep
   /auto_examples/dust_emission/plot_qpah_sweep
   /auto_examples/dust_emission/plot_themis_alpha_sweep
   /auto_examples/dust_emission/plot_themis_qhac_sweep
   /auto_examples/dust_emission/plot_themis_umin_sweep
   /auto_examples/dust_emission/plot_umin_sweep

