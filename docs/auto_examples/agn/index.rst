

.. _sphx_glr_auto_examples_agn:

AGN Models
==========

AGN disc and torus SED templates.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The X-ray corona response of an AGN depends jointly on bolometric luminosity (which sets the X-ray normalisation through the Lusso &amp; Risaliti L_X-L_UV correlation) and on the UV-to-X-ray slope alpha_OX (which sets the relative balance of UV and X-ray emission). Four panels at log L_bol = 44, 45, 46, 47 erg/s overlay three alpha_OX values each, showing that the absolute X-ray luminosity scales with L_bol while the X-ray-to-UV ratio is set independently by alpha_OX.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_alpha_ox_lbol_2d_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_alpha_ox_lbol_2d`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray corona shape across the alpha_OX vs log L_bol plane</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Shakura-Sunyaev thin disc model shows how the big blue bump (BBB) peak shifts to longer wavelengths as black-hole mass increases. At fixed Eddington ratio log(L_bol / L_Edd) = -1.0, the disc temperature scales as T_{\rm in} \propto (\dot{m} / m_\odot)^{1/4}, where the inner temperature determines the location of peak νLν. Higher mass → lower accretion rate → cooler disc → redder peak.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_bbb_mbh_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_bbb_mbh_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Big Blue Bump: multicolor disc temperature evolution with black-hole mass</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A single log L_bol = 12.5 composable AGN built up component by component — disc alone, +torus, +narrow lines, +broad lines — so the reader can see what each block contributes to the total spectrum. The bottom panel shows the same decomposition stacked so the layers add up to the full SED.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_components_breakdown_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_components_breakdown`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN composite SED: per-block decomposition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six accretion-disc backbones at fixed bolometric luminosity log L_bol = 12.5 (in log L_sun), evaluated in isolation with the host suppressed and no torus/lines/dust. The differences between the curves are entirely how each model partitions the disc power across wavelength: pure blackbody vs warm Comptonization, relativistic vs Newtonian potential, empirical-fit vs first-principles continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_disc_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_disc_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN disc continuum: model comparison at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four AGN configurations of increasing physical complexity at the same bolometric luminosity (log L_bol = 12.5 in L_sun units) — bare multicolour disc, +SKIRTOR torus, +NLR narrow-line forest, and an empirical QSOgen template that bundles all of the above. The reader sees which spectral feature each block introduces (mid-IR torus bump, optical narrow lines, broad UV continuum) and which are essentially universal across the modelling choice.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_hierarchy_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_hierarchy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Building up an AGN SED: disc, then torus, then lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A Seyfert galaxy SED is decomposed photometrically by varying the AGN contribution fraction agn_frac from 0 (pure host) to 1.0 (pure AGN) to 0.5 (composite). how to isolate the AGN contribution from the host galaxy using a single model and varying a structural parameter — useful for diagnosing photometric AGN contamination.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_host_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_host_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN host-galaxy decomposition: disentangling Seyfert contributions</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four production line backbones layered on top of the same disc + torus at fixed log L_bol = 12.5. The line backbone controls which optical/UV emission features the model produces — narrow-line region forbidden lines, broad-line permitted lines, or pre-canned empirical line lists.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_lines_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_lines_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN emission-line backbones compared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust-free quasar spectra are intrinsically blue in the UV and optical. Intrinsic dust reddening ebv (E(B−V)) reddens the continuum via extinction. Varying ebv from 0 to 0.4 shows the transition from unobscured type-1 QSO colours to moderately dust-enshrouded systems.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_qsogen_ebv_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_qsogen_ebv_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen disc: dust reddening tunes UV to optical colour</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The QSOgen model includes a UV/optical emission-line forest and broad Balmer continuum on top of the underlying disc. The relative strength of these line features with respect to the continuum controls the slope and colour of the UV–optical SED.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_qsogen_emline_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_qsogen_emline_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen lines: emission-line contributions vary with luminosity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six dusty-torus libraries reprocessing the same accretion-disc continuum at fixed log L_bol = 12.5 (in log L_sun) and standard inclination. The disc is held at multicolor (Kubota &amp; Done 2018) so the differences in the curves are entirely how each torus library geometrically distributes hot grains and re-emits the absorbed UV in the MIR.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_torus_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_torus_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN dusty torus: library comparison at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The unified AGN model attributes the Type-1 vs Type-2 dichotomy to geometry alone. Three inclinations of an identical disc + SKIRTOR torus + broad-line region (Type 1, face-on, cos i = 0.95), torus edge (intermediate, cos i = 0.5), and edge-on (Type 2, cos i = 0.1). The broad UV bump and BLR lines vanish behind the torus at high inclination; the mid-IR torus reprocessed emission stays.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_type12_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_type12`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Same AGN, different viewing angle: Type 1 to Type 2 by inclination</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The ultraviolet-to-X-ray spectral slope α_OX encodes the fundamental physics of accretion discs. At higher bolometric luminosities, discs shift toward cooler effective temperatures and steeper UV slopes, reducing the X-ray-to-UV flux ratio. We compute α_OX for 15 tengri AGN disc models (multicolor, no torus/lines) across log L_bol ∈ [10.5, 14.0], measuring at rest-frame 2500 Å (UV) and 2 keV (X-ray). The Lusso &amp; Risaliti 2016 fit α_OX = −0.166 log L_2500 + 4.74 captures the observational trend that luminous quasars are more UV-bright and X-ray-weak.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_alpha_ox_lusso_risaliti_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_alpha_ox_lusso_risaliti`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lusso & Risaliti 2016: α_OX – L_UV relation for AGN discs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduces the UV-to-X-ray connection panel from Yang et al. 2020 (X-CIGALE Fig. 3): the X-ray corona is normalised through the Just+07 alpha_OX-L_2500 relation, anchored at the disc-derived L_2500. Offsets delta_alpha_OX from -0.3 to +0.3 dex pivot the X-ray power-law about the 2500 A anchor — the disc UV stays fixed (single curve at log lam &gt; 1), only the X-ray normalisation moves.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_alpha_ox_uv_xray_connection_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_alpha_ox_uv_xray_connection`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">delta_alpha_OX pivots the X-ray spectrum about the disc UV anchor</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The agn.disc, agn.lines, agn.feii, agn.torus, agn.atten sub-blocks of SEDModel.build are composable: turning one on at a time and overlaying the all-on reference (dashed grey) shows which features each sub-block contributes. Five panels at fixed log L_bol = 12.0, all built via the public nested-dict grammar:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_composable_block_toggles_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_composable_block_toggles`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cumulative buildup of the GRAHSP AGN recipe, one sub-block at a time</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The composable AGN grammar (``agn.disc``, agn.torus, agn.lines, agn.feii, agn.atten) lets the user mix sub-blocks across model families. Same SEDModel.build call, three different physics tuples:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_composable_recipes_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_composable_recipes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Three AGN recipes built by swapping selectors, not call sites</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The collaborator workflow for adding a new AGN model. We define a toy single-temperature blackbody torus, register it with register_agn_model, confirm it is discoverable through tengri.list_agn_models and tengri.describe, then evaluate it on the public SEDModel.build path and plot it next to the production SKIRTOR torus at the same bolometric luminosity. The toy curve is a greybody; the SKIRTOR curve carries the silicate 9.7 micron feature and the inclination-dependent geometry the toy elides.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_custom_torus_extension_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_custom_torus_extension`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Registering a custom AGN torus model and using it through SEDModel.build</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Identical AGN configuration (multicolour disc + SKIRTOR torus at log L_bol = 12.5), one with the narrow-line region (FWHM ~ a few hundred km/s, characteristic Type-2 spectrum) and the other with the broad-line region (FWHM ~ thousands of km/s, Type-1). Side-by-side zooms on the UV (Ly-alpha, C IV) and the optical (Hbeta, [O III], Halpha) make the velocity-width contrast unmistakable while controlling for continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_nlr_blr_lines_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_nlr_blr_lines`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Narrow vs broad line region: a velocity-width contrast in two windows</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduces Figure 1 of Yang et al. 2020 (the X-CIGALE polar-dust introduction): SMC-law attenuation of the AGN disc by dust above the torus, plus an energy-conserving mid-IR greybody re-emission. Two panels at cos_inc = 0.95 (Type-1, face-on into the polar cone) and cos_inc = 0.10 (Type-2, edge-on view of the torus) for opening angle 40°. We sweep agn_polar_ebv from 0.00 to 0.30 — covering the empirical range Yang+2020 anchor against red quasars.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_polar_dust_ebv_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_polar_dust_ebv_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polar-dust E(B-V) sweep for Type 1 and Type 2 AGN sightlines (X-CIGALE)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Polar dust disc attenuation applies only to Type 1 (face-on) sightlines — the equatorial torus already screens the disc for Type 2. The bi-conical polar dust absorbs disc photons regardless of viewing angle, however, and re-emits them isotropically as a FIR greybody (Casey 2012). So both Type 1 and Type 2 sweeps show the FIR re-emission bump growing with E(B-V); only the UV/optical attenuation is gated by sightline.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_polar_dust_ebv_type12_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_polar_dust_ebv_type12_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polar dust E(B-V) reddens Type 1 & 2 AGN differently</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Temple, Hewett &amp; Banerji (2021) QSOgen empirical template, used as the agn.disc.type=&quot;qsogen&quot; selector. We sweep log L_bol from 10.0 to 13.5 (in L_sun units) at fixed redshift to show that the template&#x27;s spectral shape is approximately self-similar across the quasar luminosity function — the only knob that moves features (the Baldwin-effect drop in C IV/Ly-alpha equivalent width) is the bolometric normalisation.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_qsogen_spectrum_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_qsogen_spectrum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen empirical quasar SED across four decades of bolometric luminosity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In a relativistic accretion-disc model the inner boundary sits at the innermost stable circular orbit (ISCO). Higher spin shrinks the ISCO, raises the inner-disc temperature, and shifts disc power blueward — the UV spectral slope alpha (L_nu ~ nu^alpha across 912 to 3000 Å) hardens monotonically with spin. We sweep a_spin from 0 to 0.998 on the Kubota &amp; Done (2018) disc backbone, the public-API entry point for spin-sensitive disc physics in tengri, and report alpha alongside the SEDs.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_relagn_spin_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_relagn_spin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Black-hole spin hardens the UV slope through ISCO migration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute and plot the broad-line region (BLR) size-luminosity relation (R_BLR ∝ L^0.5) using tengri AGN models. how AGN continuum luminosity connects to reverberation mapping measurements of the BLR extent.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_reverberation_size_luminosity_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_reverberation_size_luminosity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN Reverberation Size-Luminosity Relation (Bentz+2013)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three accretion-disc backbones at the same bolometric anchor (log L_bol / L_sun = 12.5): the Richards et al. 2006 empirical mean Type-1 SDSS quasar template, the Temple, Hewett &amp; Banerji 2021 empirical QSOgen, and the Shakura-Sunyaev multicolour disc (the outer-disc component of Kubota &amp; Done 2018). Each is normalised to the same bolometric output so the differences are entirely in spectral shape — Richards+2006 is broader than QSOgen and carries the infrared bump from its host-galaxy-corrected composite, while the multicolour disc cuts off sharply on either side of the big blue bump.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_richards2006_template_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_richards2006_template`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Richards+2006 empirical Type-1 quasar template alongside physical discs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three distinct AGN types overlaid to show how AGN morphology and obscuration evolve with luminosity:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_seyfert_quasar_blazar_archetypes_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_seyfert_quasar_blazar_archetypes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN archetypes: Seyfert, quasar, and LIRG/Sy across bolometric luminosity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrate how the SKIRTOR clumpy radiative-transfer torus (Stalevski+2016) reprocesses the hot accretion disc as a function of viewing angle.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_inclination_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_inclination_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR AGN torus: inclination-dependent obscuration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The SKIRTOR clumpy torus model (Stalevski et al. 2016) emits thermal IR radiation that depends strongly on two parameters: viewing angle (inclination θ via cos_inc) and optical depth (``tau_97`` at 9.7 μm). Face-on systems show a smooth thermal continuum; edge-on systems develop deep 9.7 μm silicate absorption. Higher τ increases reprocessed flux.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_variants_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_variants`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: viewing angle and optical depth effects</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduces the SKIRTOR vs Fritz comparison from Yang et al. 2020 (X-CIGALE Fig. 2). Both libraries re-emit the same disc-absorbed luminosity in the mid-IR; the mid-IR peak amplitude differs by ~0.5 dex because SKIRTOR&#x27;s clumpy 3-D Stalevski+2016 RT redistributes heating more efficiently into the bright NIR-MIR continuum than a smooth-density torus. tengri does not ship Fritz+2006 directly; we substitute Silva+04 (template-based smooth torus, the closest contemporary analogue) — the qualitative contrast (clumpy bright MIR vs smooth fainter MIR) is preserved.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_vs_smooth_torus_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_vs_smooth_torus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR clumpy vs Silva+04 smooth-torus comparison (X-CIGALE Fig. 2)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The growth of a supermassive black hole (SMBH) traces a path through the (M_BH, L_bol) plane. Starting as a dormant low-mass hole, accretion during mergers builds both mass and luminosity. Peak luminosity occurs as a luminous QSO before accretion slows and the system fades. This example traces four key evolutionary stages and plots both the track on the (M_BH, L_bol) diagram and the corresponding SEDs.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_smbh_growth_track_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_smbh_growth_track`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN SMBH growth track: dormant → merger → QSO → fading</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Unified AGN models explain the Type 1/Type 2 dichotomy as a purely geometric effect — the same accretion disc + dusty torus system appears as:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_type1_type2_unified_model_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_type1_type2_unified_model`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Type 1 vs Type 2 AGN: Unified viewing-angle classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sanders et al. (1988) proposed that Ultra-Luminous Infrared Galaxies (ULIRGs) are the dust-shrouded precursors to optical QSOs. This sequence traces progressive unveiling of a buried AGN through five stages:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_ulirg_to_qso_transition_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_ulirg_to_qso_transition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">ULIRG→QSO evolutionary sequence: dust-obscured starburst to bare quasar</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Accretion disc reverberation mapping reveals how the hot UV-emitting inner disc responds to ionizing source changes. Fausnaugh+2016 observed NGC 5548 using HST multi-band photometry (UV, optical) and found that UV variations lead optical by τ(λ) — the light-crossing time across the effective emission radius at wavelength λ.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_variability_continuum_lag_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_variability_continuum_lag`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN UV→optical continuum reverberation: light-crossing time lags</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/agn/plot_agn_alpha_ox_lbol_2d
   /auto_examples/agn/plot_agn_bbb_mbh_sweep
   /auto_examples/agn/plot_agn_components_breakdown
   /auto_examples/agn/plot_agn_disc_compare
   /auto_examples/agn/plot_agn_hierarchy
   /auto_examples/agn/plot_agn_host_decomposition
   /auto_examples/agn/plot_agn_lines_compare
   /auto_examples/agn/plot_agn_qsogen_ebv_sweep
   /auto_examples/agn/plot_agn_qsogen_emline_sweep
   /auto_examples/agn/plot_agn_torus_compare
   /auto_examples/agn/plot_agn_type12
   /auto_examples/agn/plot_alpha_ox_lusso_risaliti
   /auto_examples/agn/plot_alpha_ox_uv_xray_connection
   /auto_examples/agn/plot_composable_block_toggles
   /auto_examples/agn/plot_composable_recipes
   /auto_examples/agn/plot_custom_torus_extension
   /auto_examples/agn/plot_nlr_blr_lines
   /auto_examples/agn/plot_polar_dust_ebv_sweep
   /auto_examples/agn/plot_polar_dust_ebv_type12_sweep
   /auto_examples/agn/plot_qsogen_spectrum
   /auto_examples/agn/plot_relagn_spin
   /auto_examples/agn/plot_reverberation_size_luminosity
   /auto_examples/agn/plot_richards2006_template
   /auto_examples/agn/plot_seyfert_quasar_blazar_archetypes
   /auto_examples/agn/plot_skirtor_inclination_sweep
   /auto_examples/agn/plot_skirtor_variants
   /auto_examples/agn/plot_skirtor_vs_smooth_torus
   /auto_examples/agn/plot_smbh_growth_track
   /auto_examples/agn/plot_type1_type2_unified_model
   /auto_examples/agn/plot_ulirg_to_qso_transition
   /auto_examples/agn/plot_variability_continuum_lag

