:orphan:

.. _sphx_glr_auto_examples_nebular:

Nebular Emission
================

Emission lines are vacuum throughout: Hα is 6564.61 Å, not the 6562.8 Å air
value. Mixing the two shifts every line centroid.

``neb={'type': ...}`` takes ``ssp``, ``cue``, ``cb19``, ``cloudy`` or ``none``.
The default, ``ssp``, uses the emission already baked into a with-nebular (wNE)
SSP grid. The live backends instead compute it, and expect a bare stellar grid.
Feed a bare grid to the baked-in path and both continuum and line fluxes come
out low, with no error raised.

Gas-phase metallicity is its own knob and does not follow the stellar one. Shock emission examples include line-ratio diagnostics and a composable shock-group sweep across shock parameters.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Emission lines are vacuum throughout: [OIII] = 5008.24 Å, [NII] = 6585.28 Å, Hα = 6564.61 Å, Hβ = 4862.68 Å. Overlays Kewley+2001 SF/AGN demarcation and Kauffmann+2003 SF/composite line.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_diagram_population_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_diagram_population`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram: star-forming galaxies, AGN, and shocks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cue (Li, Leja &amp; Speagle 2023) maps a four-dimensional HII region control space — ionization parameter log U, gas-phase metallicity log Z_gas, ionizing-spectrum shape, and dust-to-metal ratio — onto an emission-line spectrum. A two-dimensional sweep over the two knobs most users will turn (``log U`` and log Z_gas) is shown for four diagnostic line ratios.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_parameter_atlas_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_parameter_atlas`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comprehensive 2D sweep of ionization parameter and metallicity (Cue)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We sweep the ionizing-photon escape fraction f_esc from 0 to 1.0 at fixed log U and metallicity, showing both the broadband SED response and a zoomed view of the critical Lyman-continuum (912 A) region. The Lyman edge deepens as ionizing photons escape the ISM unabsorbed, suppressing optical line ratios simultaneously.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_fesc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_fesc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Escape fraction reshapes the SED from the Lyman continuum to optical lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Murphy+2011 SFR-Hα relation requires ionizing photons from stars younger than ~10 Myr. Constant-SFR models at ages 1–300 Myr show the calibration breaks at young (&lt;10 Myr; insufficient ionizing photons) and old (&gt;100 Myr; all stars too old to ionize) populations. We sweep stellar metallicity to show the calibration validity range is weakly sensitive to Z: higher Z reduces ionizing photon production, compressing the valid age window slightly toward older ages.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_halpha_sfr_calibration_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_halpha_sfr_calibration_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hα SFR calibration breaks at young ages, weakly dependent on metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Lyα rest-frame wavelength is 1216 Å (vacuum). EW peaks at 3–5 Myr when O-type stars dominate ionization, then decays past 10 Myr. Higher metallicity suppresses ionizing photon production, reducing peak EW.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_lyalpha_ew_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_lyalpha_ew_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha equivalent width peaks at young ages, varies with gas metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare four nebular emission backends on identical star-forming spectra:">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_nebular_backends_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_nebular_backends`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular backends: Cue, CloudyGrid, SSP-embedded, and BakedIn</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hydrogen-ionizing photon production (Q_H, photons/s per solar mass) depends critically on stellar population age. Young starbursts (age ≈ 3–5 Myr) produce ionizing photons at peak rates; by 100 Myr, Q_H drops by ~3 orders of magnitude. We show how this evolution varies across metallicity Z = [-1.0, -0.5, 0.0, +0.3] using FSPS bare-stellar (non-nebular) SSP templates, as ionizing photons are consumed by CLOUDY during wNE SSP generation and would appear suppressed.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_qh_vs_age_metallicity_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_qh_vs_age_metallicity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionizing photon production rate Q_H peaks sharply with stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Shock emission (MAPPINGS V models) can mimic AGN on the BPT diagram. We show how shock velocity, gas density, and magnetic field strength affect line ratios and diagnostic positions. Four-panel layout shows velocity and density sequences on BPT, line ratios vs velocity, and magnetic field strength.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_shock_emission_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_shock_emission`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAPPINGS V shocks: velocity, density, and magnetic field effects</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The shock group adds MAPPINGS V shock emission as an additive component to any photoionized nebular backend (Cue, CloudyGrid, CB19, or baked-in). Here we show how increasing shock contamination (shock_frac) moves a star-forming galaxy from the SF locus toward the LI(N)ER/shock region of the BPT diagnostic plane.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_shock_frac_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_shock_frac_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Shock-group sweep: composable shock contamination on the BPT diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Five widely-used optical strong-line metallicity diagnostics evaluated across the Cue logZ_gas prior. Each one carries a different systematic — Pettini &amp; Pagel 2004 O3N2 saturates at high Z, the R23 ratio is double-valued (Pagel+1979), N2 (Marino+2013) is monotonic but small dynamic range, the [O III]/[O II] diagnostic tracks ionization, and [S II]/[O II] is a low-ion proxy.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_strong_line_metallicity_diagnostics_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_strong_line_metallicity_diagnostics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Strong-line gas-phase metallicity diagnostics</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/nebular/plot_bpt_diagram_population
   /auto_examples/nebular/plot_cue_parameter_atlas
   /auto_examples/nebular/plot_fesc_sweep
   /auto_examples/nebular/plot_halpha_sfr_calibration_age
   /auto_examples/nebular/plot_lyalpha_ew_vs_age
   /auto_examples/nebular/plot_nebular_backends
   /auto_examples/nebular/plot_qh_vs_age_metallicity
   /auto_examples/nebular/plot_shock_emission
   /auto_examples/nebular/plot_shock_frac_sweep
   /auto_examples/nebular/plot_strong_line_metallicity_diagnostics

