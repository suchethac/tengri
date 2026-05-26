

.. _sphx_glr_auto_examples_nebular:

Nebular Emission
================

Nebular emission backends comparison.


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

    <div class="sphx-glr-thumbcontainer" tooltip="The BPT diagram ([OIII]/Hβ vs [NII]/Hα) separates ionizing sources. Shocks (MAPPINGS V, Allen+2008) trace a sequence from HII regions through composite regions into Seyfert regions as velocity increases. We plot shock models alongside the standard demarcation lines.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_diagnostics_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_diagnostics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram separates star formation from shocks and AGN</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Baldwin-Phillips-Terlevich (BPT) diagram ([OIII]/Hβ vs [NII]/Hα) separates ionization mechanisms: star formation, AGN, and composites.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_diagram_population_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_diagram_population`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram population with star-forming galaxies and AGN-like models</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Cue has six tuning knobs that control HII-region ionization and the diffuse ionized gas. This six-panel tour sweeps each knob individually and reports the L_Hα response relative to the baseline, in dex. A flat line means the parameter has no effect on Hα at fixed other knobs.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_flex_tour_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_flex_tour`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue knob flexibility: six dimensions of HII region control</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The ionization parameter logU controls the hardness of the ionizing radiation field and drives rapid changes in optical line ratios. We show how [OIII]/[OII] (O32) and [OIII]/Hβ respond to logU from -4 to -1 at fixed metallicity (Z/Zsun = -0.5), demonstrating the use of O32 as a logU diagnostic (Kewley &amp; Dolphin 2002). Cue (Li et al. 2024, 2025) samples the ionizing spectrum flexibility and provides smooth gradients through metallicity, density, and ionization parameters for joint SED fitting.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_logu_line_ratios_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_logu_line_ratios`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionization parameter (logU) controls emission-line diagnostics</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cue (Li, Leja &amp; Speagle 2023) maps a four-dimensional HII region control space — ionization parameter log U, gas-phase metallicity log Z_gas, ionizing-spectrum shape, and dust-to-metal ratio — onto an emission-line spectrum. A two-dimensional sweep over the two knobs most users will turn (``log U`` and log Z_gas) is shown for four diagnostic line ratios.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_parameter_atlas_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_parameter_atlas`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comprehensive sweep of the Cue nebular parameters</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diffuse ionized gas (DIG) has lower ionization parameter than HII regions, shifting galaxies toward the LINER region on the BPT diagram. We vary the DIG fraction from pure HII (0) to mixed gas (0.8).">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_dig_frac_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_dig_frac_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diffuse ionized gas suppresses strong optical lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Zoomed rest-frame spectrum of an ionised-gas-dominated SF galaxy with the strongest optical / near-UV emission lines labelled. Wavelengths are vacuum; line positions follow NIST/Atomic Line List.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_emission_line_atlas_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_emission_line_atlas`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optical emission-line atlas of a young star-forming galaxy</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="We sweep the ionising-photon escape fraction f_esc from 0 to 0.9 at fixed log U and metallicity, and read out the response in diagnostic-ratio space ([O III]/Hbeta etc.). Companion to plot_lyman_continuum_escape.py, which shows the same physics in SED space focused on the Lyman edge.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_fesc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_fesc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Escape fraction suppresses the optical line ratios, not just amplitudes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nebular free-free, free-bound, and two-photon emission respond to gas-phase metallicity (``logZ_gas``) through changes in metal cooling efficiency and ionization balance. This example demonstrates the metallicity sensitivity of the nebular continuum at fixed ionization parameter.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_gas_z_continuum_effect_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_gas_z_continuum_effect`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas-phase metallicity effect on nebular continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Murphy+2011 SFR-Hα relation requires ionizing photons from stars younger than ~10 Myr. Constant-SFR models at ages 1–300 Myr show the calibration breaks at young (&lt;10 Myr; insufficient ionizing photons) and old (&gt;100 Myr; all stars too old to ionize) populations.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_halpha_sfr_calibration_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_halpha_sfr_calibration_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hα SFR calibration breaks at young ages</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three classical strong-line metallicity diagnostics computed as a function of gas-phase metallicity (``logZ_gas``). The plot spans 12 + log(O/H) from ~7 to ~9 and illustrates key observational features: the saturation of [O III]/H-beta at high metallicity (Kewley &amp; Dopita 2002), the monotonic but small dynamic range of [N II]/H-alpha (Marino et al. 2013), and the famous double-valued R23 ratio which peaks near 12 + log(O/H) ≈ 8.3 (Pagel et al. 1979).">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_line_ratios_metallicity_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_line_ratios_metallicity_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optical line-ratio diagnostics along the metallicity gradient</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Emission line velocity dispersion broadens lines from a few km/s (narrow, kinematically resolved) to hundreds of km/s (unresolved at typical spectroscopic resolution). We show the [OIII] region broadened across the dynamical range.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_line_sigma_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_line_sigma_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission line broadening traces gas kinematics</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Varying log U from -4 to -1.5 on a young star-forming galaxy at fixed metallicity changes every strong optical line simultaneously — Hbeta, [O III], Halpha, [N II], [S II] all move together. We plot the full 4000-7500 A SED so the continuum context is visible alongside the line forest. Companion to plot_cue_logu_line_ratios.py, which projects the same sweep onto two-line diagnostic axes.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logu_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logu_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionisation parameter reshapes the full optical SED, not just line ratios</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 1-D log Z_gas sweep on the SED scale, complementing the 2-D atlas in plot_cue_parameter_atlas.py and the line-ratio projection in plot_strong_line_metallicity_diagnostics.py. Reader sees how every strong optical line moves together as Z_gas climbs, with [N II]/Halpha and [O III]/Hbeta the textbook diagnostics.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logz_gas_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logz_gas_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas metallicity reshapes the optical nebular continuum and line forest</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Lyman-alpha (Lyα) equivalent width (EW) traces stellar population age through the presence and strength of massive O stars. We construct a sequence of constant star-formation-rate (CSF) models with ages ranging from 1 Myr to 30 Myr at fixed metallicity (Z = Zsun; logZ = 0), compute the rest-frame Lyα emission line luminosity and the underlying continuum at 1216 Å, then derive EW(Lyα) = L(Lyα) / L_continuum.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_lyalpha_ew_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_lyalpha_ew_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha equivalent width peaks during O-star dominance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We zoom on the Lyman-continuum region (rest 800-1300 A) and sweep the escape fraction f_esc to show how the 912 A discontinuity deepens as more ionising photons leave the ISM unabsorbed. Companion to plot_fesc_sweep.py, which projects the same physics into optical line-ratio diagnostics.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_lyman_continuum_escape_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_lyman_continuum_escape`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-continuum escape fraction reshapes the SED around the 912 A edge</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Two nebular backends, same SFH, same dust, same metallicity. BakedIn pulls line ratios from the SSP grid (Conroy + Byler wNE templates); Cue (Li, Leja &amp; Speagle 2023) is a neural emulator over the CLOUDY parameter space, run here at log U = -3.0.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_backend_compare_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_backend_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular backends side-by-side: BakedIn vs Cue</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Young massive stars produce harder ionising continua and drive the nebular emission toward higher [O III]/Hbeta. We sweep the SFH timescale tau_gyr from 0.1 to 2 Gyr on a single dual power-law model and plot the resulting line ratios against the Kewley+2001 / Kauffmann+2003 demarcation curves. The locus migrates from the star-forming wing into the composite region as the population ages — SFH timescale is the upstream knob behind the BPT ionisation sequence.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Compare Cue (neural emulator; current recommended path) against traditional photoionization grids (CloudyGrid) and SSP-embedded nebular. Shows [OIII] and H-alpha regions on a young starburst.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_nebular_backends_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_nebular_backends`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue nebular emulator vs alternatives</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Four widely-used optical strong-line metallicity diagnostics evaluated across the Cue logZ_gas prior. Each one carries a different systematic — Pettini &amp; Pagel 2004 O3N2 saturates at high Z, the R23 ratio is double-valued, N2 (Marino+2013) is monotonic but small dynamic range, and the [Ne III]/[O II] diagnostic is weakly Z-dependent.">

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
   /auto_examples/nebular/plot_bpt_diagnostics
   /auto_examples/nebular/plot_bpt_diagram_population
   /auto_examples/nebular/plot_cue_fesc_logu_atlas
   /auto_examples/nebular/plot_cue_flex_tour
   /auto_examples/nebular/plot_cue_logu_line_ratios
   /auto_examples/nebular/plot_cue_parameter_atlas
   /auto_examples/nebular/plot_dig_frac_sweep
   /auto_examples/nebular/plot_emission_line_atlas
   /auto_examples/nebular/plot_fesc_lya_sweep
   /auto_examples/nebular/plot_fesc_sweep
   /auto_examples/nebular/plot_gas_z_continuum_effect
   /auto_examples/nebular/plot_halpha_sfr_calibration_age
   /auto_examples/nebular/plot_line_ratios_metallicity_evolution
   /auto_examples/nebular/plot_line_sigma_sweep
   /auto_examples/nebular/plot_logu_sweep
   /auto_examples/nebular/plot_logz_gas_sweep
   /auto_examples/nebular/plot_lyalpha_ew_vs_age
   /auto_examples/nebular/plot_lyman_continuum_escape
   /auto_examples/nebular/plot_neb_age_dependence
   /auto_examples/nebular/plot_neb_backend_compare
   /auto_examples/nebular/plot_neb_bpt_logu_grid
   /auto_examples/nebular/plot_neb_density_sweep
   /auto_examples/nebular/plot_nebular_backends
   /auto_examples/nebular/plot_qh_vs_age_metallicity
   /auto_examples/nebular/plot_shock_emission
   /auto_examples/nebular/plot_strong_line_metallicity_diagnostics

