

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

    <div class="sphx-glr-thumbcontainer" tooltip="Per-H grain volume distribution versus grain radius for the Hensley &amp; Draine 2023 fiducial size distribution (MW high-latitude R_V=3.1 sightline).">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_01_size_distribution_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_01_size_distribution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH size distribution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Emission per H per ionization parameter U across the Hensley &amp; Draine 2023 grid. Dividing by U reveals its effect: PAH-to-FIR ratio plateaus in FIR (U-independent) but rises steeply with U in MIR.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_02_emission_vs_lgU_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_02_emission_vs_lgU`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH emission vs log U</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Per-component breakdown (Astrodust continuum, PAHs, spinning dust) at the Hensley &amp; Draine 2023 fiducial ionization parameter \log_{10} U = 0.2.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_03_components_at_fiducial_U_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_03_components_at_fiducial_U`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH per-component decomposition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare dust emission templates at fixed infrared luminosity. Shows how spectral shape changes across modified-blackbody, Draine+2021 PAHspec, and Hensley &amp; Draine 2023 Astrodust while bolometric output remains conserved.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_04_sedmodel_dust_emission_swap_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_04_sedmodel_dust_emission_swap`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DustEmissionSEDComponent — swap MBB / PAHspec / Astrodust</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Ionization fraction and alignment efficiency versus grain size for the Hensley &amp; Draine 2023 fiducial size distribution.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_05_ionization_alignment_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_05_ionization_alignment`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH ionization fraction and alignment</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Extinction opacity, polarized extinction, and single-scattering albedo for the Hensley &amp; Draine 2023 fiducial size distribution.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_06_extinction_and_scattering_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_06_extinction_and_scattering`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH extinction, scattering, and albedo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spinning dust microwave emission across 10–100 GHz, decomposed by grain (Astrodust/PAH) and phase (CNM/WNM), for the Hensley &amp; Draine 2023 fiducial.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_07_spinning_dust_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_07_spinning_dust`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH spinning-dust microwave emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Polarized emission and polarization fraction from Astrodust grains at the Hensley &amp; Draine 2023 fiducial ionization parameter.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_08_polarized_emission_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_08_polarized_emission`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH polarized emission and polarization fraction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep ionization parameter across the published [-3, +6] range of the Hensley &amp; Draine 2023 grid. Shows FIR peak shift toward shorter wavelengths and rising MIR PAH features as radiation field intensifies.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_lgU_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_lgU_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hensley & Draine 2023 Astrodust+PAH: log U sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep infrared luminosity across the BOSA grid at fixed specific star formation rate. Increasing L_TIR heats dust, shifting FIR peak blueward and enhancing PAH relative to continuum.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_ltir_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_ltir_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA: log L_TIR sweep at fixed log sSFR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep specific star formation rate across the BOSA grid at fixed infrared luminosity. Higher sSFR produces harder mid-IR colors and stronger PAH features; quiescent galaxies exhibit colder FIR peaks.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_ssfr_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_ssfr_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA: log sSFR sweep at fixed log L_TIR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust temperature T sets the far-infrared peak via Wien&#x27;s displacement law. Higher T shifts the peak blueward into the mid-IR; lower T shifts it redward toward the submillimeter.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_T_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_T_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modified Blackbody Dust Temperature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All six dust-emission ingredients shipped with tengri, called with the same absorbed bolometric luminosity (1e10 L_sun) and the same warm-dust temperature (35 K). Analytic models (modified BB, Casey 2012, energy-balance split) drop sharply blue-ward of the warm-dust peak; template-based libraries (DL07, DL14, Dale+2014) carry PAH features in the 3-20 μm window. Template models silently skip if the data files aren&#x27;t available.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_emission_models_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_emission_models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust-emission model family at fixed L_abs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The single most-used number an observer reads off a sub-mm SED is L_IR(8–1000 μm). Converting it to a dust mass requires assuming a dust temperature and emissivity; the standard analytic estimator is">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_mass_from_lir_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_mass_from_lir`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust mass inferred from L_IR and modified-blackbody temperature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="2D grid of dust IR emission showing how PAH mass fraction and radiation-field hardness independently shape the mid- and far-infrared SED. Uses Draine &amp; Li 2007 templates across parameter space.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_qpah_umin_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_qpah_umin_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR SED: q_PAH × U_min Grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All seven shipped dust IR-emission libraries reprocessing the same absorbed UV power into the IR, normalised so the integrated L_IR(8–1000 μm) is identical across curves. The differences then sit entirely in the SED shape — peak wavelength (T_dust proxy), PAH-feature amplitude in the 3–20 μm window, and how steeply the sub-mm tail falls.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_ir_library_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_ir_library_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR-emission library comparison at fixed L_dust</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep ionization parameter across the Draine+2021 PAHspec library at fixed starlight spectrum and size distribution. Low U: FIR-cooling regime; high U: mid-IR peak shift and PAH-feature strengthening.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_pahspec_lgU_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_pahspec_lgU_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine+2021 PAHspec: log U sweep at fixed (starlight, ion, size)</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="PAH mass fraction controls strength of polycyclic aromatic hydrocarbon mid-infrared emission features. Higher q_PAH produces stronger features at 3.3, 6.2, 7.7, 8.6, 11.3 μm. Range varies by dust model.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_qpah_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_qpah_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PAH Mass Fraction (q_PAH)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep radiation-field distribution slope across the THEMIS grid at fixed grain content and minimum intensity. Lower alpha shifts weight toward high U, warming dust and shifting FIR peak blueward; higher alpha approaches single-U.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_alpha_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_alpha_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: power-law slope alpha sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep hydrocarbon grain content across the THEMIS grid at fixed minimum radiation field strength. PAH-like mid-IR features strengthen with q_HAC while FIR continuum remains essentially unchanged.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_qhac_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_qhac_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: q_HAC sweep at fixed U_min</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep minimum radiation field strength across the THEMIS grid at fixed hydrocarbon grain content. Higher U warms dust, shifting FIR peak blueward and strengthening mid-IR grain emission relative to far-IR.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_umin_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_umin_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: U_min sweep at fixed q_HAC</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Minimum radiation field intensity U_min controls diffuse dust heating. Higher U_min implies hotter dust and FIR peak shifted blueward toward shorter wavelengths.">

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
   /auto_examples/dust_emission/plot_dust_mass_from_lir
   /auto_examples/dust_emission/plot_dust_qpah_umin_grid
   /auto_examples/dust_emission/plot_ir_library_compare
   /auto_examples/dust_emission/plot_mbb_temperature_beta_grid
   /auto_examples/dust_emission/plot_pahspec_lgU_sweep
   /auto_examples/dust_emission/plot_pahspec_starlight_sweep
   /auto_examples/dust_emission/plot_qpah_sweep
   /auto_examples/dust_emission/plot_themis_alpha_sweep
   /auto_examples/dust_emission/plot_themis_qhac_sweep
   /auto_examples/dust_emission/plot_themis_umin_sweep
   /auto_examples/dust_emission/plot_umin_sweep

