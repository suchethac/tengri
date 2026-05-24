

.. _sphx_glr_auto_examples_nebular:

Nebular Emission
================

Nebular emission backends comparison.



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

    <div class="sphx-glr-thumbcontainer" tooltip="Escape fraction f_esc sets what fraction of ionizing photons reach the ISM. Higher f_esc suppresses all nebular emission lines since fewer photons remain to ionize gas.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_fesc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_fesc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionizing photon escape suppresses nebular emission</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Higher ionisation parameter log U drives stronger [OIII] and [NII] emission, steering the galaxy toward the Seyfert region on the BPT diagram. We vary log U across the typical range for star-forming galaxies.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logu_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logu_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionization parameter controls optical line strength</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Gas metallicity controls [NII]/Hα and [OIII]/Hβ ratios, the primary optical metallicity diagnostics. We vary nebular metallicity across the abundance range.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logz_gas_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logz_gas_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas metallicity shifts optical emission line ratios</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The BPT diagram ([OIII]/Hβ vs [NII]/Hα) classifies ionizing sources. We show how stellar population age controls ionization parameter: younger (hotter) populations move the locus toward higher [OIII]/Hβ, steering from star-forming toward composite/Seyfert regions.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_bpt_logu_grid_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_bpt_logu_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram ionization sequence from ages</div>
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
   /auto_examples/nebular/plot_cue_fesc_logu_atlas
   /auto_examples/nebular/plot_cue_flex_tour
   /auto_examples/nebular/plot_cue_parameter_atlas
   /auto_examples/nebular/plot_dig_frac_sweep
   /auto_examples/nebular/plot_emission_line_atlas
   /auto_examples/nebular/plot_fesc_lya_sweep
   /auto_examples/nebular/plot_fesc_sweep
   /auto_examples/nebular/plot_line_sigma_sweep
   /auto_examples/nebular/plot_logu_sweep
   /auto_examples/nebular/plot_logz_gas_sweep
   /auto_examples/nebular/plot_neb_age_dependence
   /auto_examples/nebular/plot_neb_backend_compare
   /auto_examples/nebular/plot_neb_bpt_logu_grid
   /auto_examples/nebular/plot_neb_density_sweep
   /auto_examples/nebular/plot_nebular_backends
   /auto_examples/nebular/plot_shock_emission
   /auto_examples/nebular/plot_strong_line_metallicity_diagnostics

