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

Gas-phase metallicity is its own knob and does not follow the stellar one.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cue neural emulator responds to 12+ parameters. We show how each knob (ionization, metallicity, density, abundances, ionizing slope) moves a galaxy on the BPT-N plane log [OIII]/Hβ vs log [NII]/Hα. Each panel sweeps one parameter while holding fiducial values fixed. Kewley+2001 and Kauffmann+2003 demarcations shown for reference.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_cue_flexibility_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_cue_flexibility`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue nebular knobs affect BPT positions individually</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Show how the Cue neural emulator (Li+2025) maps the 2D parameter space (log U, log Z_gas) onto three classical BPT diagnostic diagrams. Lines of constant log U (varying metallicity) and constant log Z (varying ionization) show the full grid&#x27;s coverage and demarcation positions.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_cue_grid_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_cue_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue nebular grid on BPT diagram</div>
    </div>


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

    <div class="sphx-glr-thumbcontainer" tooltip="The Cue knobs fesc (ionizing-photon escape fraction) and logU (HII region ionization parameter) jointly govern the line spectrum of a star-forming galaxy: escape fraction sets how many ionizing photons reach the gas, logU shifts the resulting ionization balance of the gas they ionize. We map the response of three diagnostic lines/ratios on a 2-D grid.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_fesc_logu_atlas_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_fesc_logu_atlas`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue 2-D atlas: ionizing escape fraction × ionization parameter</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The Lyα-specific escape fraction f_esc_lya sets what fraction of Lyα photons can escape the ISM without scattering. Higher f_esc_lya suppresses the Lyα emission line while leaving other nebular lines unchanged.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_fesc_lya_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_fesc_lya_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyα escape fraction controls Lyman-alpha strength</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Nebular free-free, free-bound, and two-photon emission respond to gas-phase metallicity (``logZ_gas``) through changes in metal cooling efficiency and ionization balance. metallicity sensitivity of the nebular continuum at fixed ionization parameter.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_gas_z_continuum_effect_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_gas_z_continuum_effect`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas-phase metallicity effect on nebular continuum</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="A galaxy&#x27;s velocity dispersion sigma_v_kms broadens every spectral feature — including the nebular emission lines — from a few tens of km/s (dynamically cold disks) to several hundred km/s (dispersion-dominated spheroids and AGN narrow-line regions). The broadening is a forward-model convolution applied to the predicted spectrum, so it is only visible when the instrument line-spread function is finer than the velocity width: we therefore predict a spectrum on a high-resolution grid (R ~ 10000) around the [O III] λλ4959,5007 + Hβ region and sweep sigma_v_kms.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_line_sigma_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_line_sigma_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission line broadening traces gas kinematics</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 1-D log Z_gas sweep on the SED scale, complementing the 2-D atlas in plot_cue_parameter_atlas.py and the line-ratio projection in plot_strong_line_metallicity_diagnostics.py. Reader sees how every strong optical line moves together as Z_gas climbs, with [N II]/Hα and [O III]/Hbeta the textbook diagnostics.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logz_gas_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logz_gas_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas metallicity reshapes the optical nebular continuum and line forest</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Ionizing photon production declines rapidly with stellar population age (~t^-1). We show how nebular line strength evolves from young (50 Myr) to old (5 Gyr) populations.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_age_dependence_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_age_dependence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular emission fades with stellar population age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Young massive stars produce harder ionizing continua and drive the nebular emission toward higher [O III]/Hbeta. We sweep the SFH timescale tau_gyr from 0.1 to 2 Gyr on a single dual power-law model and plot the resulting line ratios against the Kewley+2001 / Kauffmann+2003 demarcation curves. The locus migrates from the star-forming wing into the composite region as the population ages — SFH timescale is the upstream knob behind the BPT ionization sequence.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_bpt_logu_grid_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_bpt_logu_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar-population age moves a galaxy on the BPT diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nebular gas density controls ionization balance and recombination rates, affecting emission line strengths. Higher density increases cooling efficiency, shifting line ratios through recombination rate changes.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_density_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_density_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular density affects recombination and cooling</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The optical [O III] 5007 / Hβ ratio is set primarily by the ionization parameter log U: more energetic Lyman continuum photons per H atom ionize more O+ to O++, while Hβ recombination depends mostly on the ionizing photon rate (``Q_H``) and is roughly insensitive to log U. The ratio therefore rises monotonically with log U at fixed gas metallicity.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_oiii_hbeta_logu_at_fixed_z_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_oiii_hbeta_logu_at_fixed_z`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">[O III]/Hβ vs ionization parameter at fixed gas metallicity</div>
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

   /auto_examples/nebular/plot_bpt_cue_flexibility
   /auto_examples/nebular/plot_bpt_cue_grid
   /auto_examples/nebular/plot_bpt_diagram_population
   /auto_examples/nebular/plot_cue_fesc_logu_atlas
   /auto_examples/nebular/plot_cue_parameter_atlas
   /auto_examples/nebular/plot_fesc_lya_sweep
   /auto_examples/nebular/plot_fesc_sweep
   /auto_examples/nebular/plot_gas_z_continuum_effect
   /auto_examples/nebular/plot_halpha_sfr_calibration_age
   /auto_examples/nebular/plot_line_sigma_sweep
   /auto_examples/nebular/plot_logz_gas_sweep
   /auto_examples/nebular/plot_lyalpha_ew_vs_age
   /auto_examples/nebular/plot_neb_age_dependence
   /auto_examples/nebular/plot_neb_bpt_logu_grid
   /auto_examples/nebular/plot_neb_density_sweep
   /auto_examples/nebular/plot_nebular_backends
   /auto_examples/nebular/plot_oiii_hbeta_logu_at_fixed_z
   /auto_examples/nebular/plot_qh_vs_age_metallicity
   /auto_examples/nebular/plot_shock_emission
   /auto_examples/nebular/plot_strong_line_metallicity_diagnostics

