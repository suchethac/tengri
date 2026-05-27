:orphan:

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

    <div class="sphx-glr-thumbcontainer" tooltip="The BOSA infrared template library is parametrised jointly by total infrared luminosity log L_TIR and specific star formation rate log sSFR. Neither axis alone tells the full story: at fixed sSFR the FIR peak migrates with L_TIR (dust temperature), while at fixed L_TIR the PAH mid-IR forest brightens with sSFR. Three side-by-side panels at fixed sSFR overlay three L_TIR values each, making the 2-D dependence legible in a single figure rather than two skinny 1-D loops.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA library: PAH features and FIR peak depend on both sSFR and L_TIR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep infrared luminosity across the BOSA grid at fixed specific star formation rate. Increasing L_TIR heats dust, shifting FIR peak blueward and enhancing PAH relative to continuum. Library is normalised by ∫Lν dν=1; shape variation with L_TIR is intentionally small.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_ltir_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_ltir_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA: log L_TIR sweep at fixed log sSFR</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="A 2-D grid on the Draine &amp; Li 2007 template library: rows step through PAH mass fraction q_PAH (controls mid-IR PAH-feature strength), columns through the minimum radiation field U_min (sets the diffuse dust temperature, i.e. the FIR peak position). The two axes act nearly orthogonally — a surprise for anyone who would lump them together as &quot;PAH knobs.&quot;">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_qpah_umin_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_qpah_umin_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The q_PAH and U_min knobs move PAH amplitude and FIR peak independently</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The mid-infrared ionisation-parameter sensitivity is library-specific, but the FIR-peak migration with rising log U is a universal prediction. We overlay the Hensley &amp; Draine 2023 (Astrodust+PAH) and the Draine+2021 PAHspec libraries at the same three log U values to surface where the two agree (FIR peak position) and where they differ (MIR PAH-feature strength and the Astrodust silicate plateau near 18 microns).">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_logu_cross_library_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_logu_cross_library`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Two PAH libraries respond to log U with the same FIR-peak migration</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The Draine &amp; Li (2007) dust model naturally separates three emission regimes via its parameters. Varying q_PAH (PAH mass fraction) and U_min (minimum radiation-field intensity) traces three archetypal SED shapes:">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_pah_warm_cold_split_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_pah_warm_cold_split`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR SED: PAH / Warm grain / Cold grain decomposition</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The starlight intensity floor U_min sets the temperature of the diffuse-ISM component in template-based dust libraries. We compare the Draine &amp; Li 2007 grid (fixed q_PAH = 2.5%) and the THEMIS grid (fixed q_HAC = 0.17) at three matched U_min values to highlight that the FIR-peak position is remarkably consistent between the two grain-physics paradigms, while THEMIS predicts a stronger mid-IR continuum from its hydrogenated amorphous carbon component.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_umin_cross_library_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_umin_cross_library`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Minimum radiation field U_min: DL07 and THEMIS agree on the FIR peak</div>
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

   /auto_examples/dust_emission/plot_astrodust_hd23_01_size_distribution
   /auto_examples/dust_emission/plot_astrodust_hd23_02_emission_vs_lgU
   /auto_examples/dust_emission/plot_astrodust_hd23_03_components_at_fiducial_U
   /auto_examples/dust_emission/plot_astrodust_hd23_04_sedmodel_dust_emission_swap
   /auto_examples/dust_emission/plot_astrodust_hd23_05_ionization_alignment
   /auto_examples/dust_emission/plot_astrodust_hd23_06_extinction_and_scattering
   /auto_examples/dust_emission/plot_astrodust_hd23_07_spinning_dust
   /auto_examples/dust_emission/plot_astrodust_hd23_08_polarized_emission
   /auto_examples/dust_emission/plot_bosa_grid
   /auto_examples/dust_emission/plot_bosa_ltir_sweep
   /auto_examples/dust_emission/plot_dust_T_sweep
   /auto_examples/dust_emission/plot_dust_emission_models
   /auto_examples/dust_emission/plot_dust_qpah_umin_grid
   /auto_examples/dust_emission/plot_ir_library_compare
   /auto_examples/dust_emission/plot_logu_cross_library
   /auto_examples/dust_emission/plot_mbb_temperature_beta_grid
   /auto_examples/dust_emission/plot_pah_warm_cold_split
   /auto_examples/dust_emission/plot_pahspec_starlight_sweep
   /auto_examples/dust_emission/plot_qpah_sweep
   /auto_examples/dust_emission/plot_themis_alpha_sweep
   /auto_examples/dust_emission/plot_themis_qhac_sweep
   /auto_examples/dust_emission/plot_umin_cross_library
   /auto_examples/dust_emission/plot_warm_cold_dust_decomposition

