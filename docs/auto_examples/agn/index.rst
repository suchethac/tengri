

.. _sphx_glr_auto_examples_agn:

AGN Models
==========

AGN disc and torus SED templates.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 4-panel grid (one panel per log L_bol value) showing how the X-ray corona spectrum depends jointly on bolometric luminosity and the UV-to-X-ray slope alpha_ox. Both parameters affect the X-ray normalisation; only alpha_ox shifts the relative balance between UV and X-ray emission. Sweeps cover the canonical X-ray band 0.1–1000 keV.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_alpha_ox_lbol_2d_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_alpha_ox_lbol_2d`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray Corona: Spectral Index vs Bolometric Luminosity 2D Sweep</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The bolometric correction K_X = L_{\rm bol} / L_X relates the total AGN luminosity to the flux in a single observational band. For X-ray selected AGN, this is essential for converting observed X-ray fluxes back to total AGN power.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_bolometric_correction_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_bolometric_correction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN Bolometric Correction: K_X(L_bol) Across Four Bands</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The torus inclination angle determines how much cold dust emission we observe. Face-on (high cos_inc) views show a smooth thermal bump; edge-on (low cos_inc) views expose more reprocessed mid-infrared flux and can show silicate absorption features.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_cos_inc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_cos_inc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: viewing angle tunes IR profile shape</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Compare all AGN model tiers in tengri: from simple power-law disc + single torus (3 parameters) through the full unified NLR/BLR model (12+ parameters). Each tier adds physical complexity. No SSP data required.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_hierarchy_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_hierarchy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN SEDModel Hierarchy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A Seyfert galaxy SED is decomposed photometrically by varying the AGN contribution fraction agn_frac from 0 (pure host) to 1.0 (pure AGN) to 0.5 (composite). This demonstrates how to isolate the AGN contribution from the host galaxy using a single model and varying a structural parameter — useful for diagnosing photometric AGN contamination.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="The disc continuum normalisation tracks bolometric luminosity directly; the disc temperature shifts more subtly with the implied accretion rate. Varying agn_log_lbol from 10 to 14 (in log10 L_sun) sweeps four orders of magnitude in disc luminosity, comparable to typical Seyfert through bright-QSO regimes. The spectral shape (slope, peak position) remains nearly fixed.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_log_lbol_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_log_lbol_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen disc: bolometric luminosity controls overall flux</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The torus opening angle (``oa_skirtor``) sets how much of the central disc is visible. A narrower torus (smaller opening angle) hides the disc and relies on reprocessed torus emission; a more open torus exposes the hot disc continuum and shifts the SED blueward.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_oa_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_oa_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: opening angle controls exposed disc fraction</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The SKIRTOR clumpy torus has a radial dust-density profile with power-law index p. Steeper profiles (higher p) concentrate more dust closer to the disc, reducing the mid-IR peak temperature and shifting flux toward the far-IR. Flatter profiles distribute dust more uniformly and hotter on average.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_skirtor_p_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_skirtor_p_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: radial density profile tunes IR emission peak</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 9.7 μm optical depth tau_97 controls the strength of silicate dust absorption/emission in the mid-infrared. Thin tori (tau ~3) show weak features and more continuum; thick tori (tau ~11) develop deep absorption troughs or bright emission depending on viewing angle.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_tau_skirtor_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_tau_skirtor_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: optical depth governs silicate feature strength</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The unified model of AGN activity: the same physical system appears as Type 1 (broad-line, blue disc continuum visible) or Type 2 (narrow-line only, torus blocks the accretion disc) depending purely on viewing angle.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_type12_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_type12`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Type 1 vs Type 2 AGN: Geometric Unification</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The dimensionless spin parameter a determines the innermost stable circular orbit (ISCO). Higher spin pushes ISCO inward, raising peak disc temperature and shifting the UV bump bluer. This demonstrates the classic Kerr black hole effect on thin disc accretion: Schwarzschild (a=0) → near-extremal Kerr (a*=0.998).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_bh_spin_disc_continuum_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_bh_spin_disc_continuum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Black hole spin effect on accretion disc UV peak temperature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Single recipe (all-GRAHSP), but each pipeline stage rendered independently on top of the disc continuum. Demonstrates how the five blocks (``disc → lines → feii → torus → attenuation``) contribute to the total SED — useful for understanding which knob controls which feature.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_composable_block_toggles_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_composable_block_toggles`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Composable AGN: per-block contribution breakdown</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The composable AGN block subsystem (``agn_model=&quot;composable&quot;``) lets users pick one block per pipeline stage and combine across models. This example compares four recipes built from the registered block set:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_composable_recipes_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_composable_recipes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Composable AGN: mix-and-match recipes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The composable AGN runner exposes three evaluation modes (see docs/dev/three_evaluation_modes.md):">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_composable_three_modes_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_composable_three_modes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Composable AGN: three evaluation modes side-by-side</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed black hole mass M_BH = 10^8 M_sun, the accretion disc luminosity and spectral shape scale with Eddington ratio λ_Edd = L_bol / L_Edd. Here we sweep λ_Edd from 0.001 to 1.0 at five logarithmic steps and overlay the disc continuum (100–3000 Å) to show how lower accretion rates produce fainter discs with unchanged spectral shape (Shakura &amp; Sunyaev 1973).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_eddington_ratio_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_eddington_ratio_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Eddington ratio sweep: multicolor disc thermal scaling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The black hole mass (M_BH) and stellar bulge mass (M_) of galaxies follow a tight empirical scaling relation. This example builds 12 mock AGN-hosting galaxies sweeping log M_ from 9 to 12 M_☉, derives M_BH from the published Kormendy &amp; Ho (2013) and Reines &amp; Volonteri (2015) relations, and constrains the AGN bolometric luminosity via a random Eddington ratio (λ_Edd ∈ [0.001, 0.1]).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_mbh_mstar_relation_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_mbh_mstar_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">M_BH–M_* scaling relation: Kormendy & Ho 2013 and Reines & Volonteri 2015</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plot narrow-line (NLR, FWHM ~500 km/s) and broad-line region (BLR, FWHM ~5000 km/s) emission spectra. Shows how BLR vanishes at high inclination angles (Type 2 AGN) while NLR remains visible.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_nlr_blr_lines_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_nlr_blr_lines`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Narrow and Broad Line Region Emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plot QSOgen (Temple, Hewett &amp; Banerji 2021) empirical quasar SEDs. Shows how an empirically-trained surrogate matches observed quasar spectra across the UV through near-IR, with parametric control over redshift and luminosity.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_qsogen_spectrum_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_qsogen_spectrum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen Empirical Quasar Template</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrate the effect of BH spin on the relativistic outer-disc SED using the RELAGN model (Hagen &amp; Done 2023) with KYCONV Kerr-metric ray-tracing (Dovciak, Karas &amp; Yaqoob 2004).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_relagn_spin_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_relagn_spin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">RELAGN Spin Sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute and plot the broad-line region (BLR) size-luminosity relation (R_BLR ∝ L^0.5) using tengri AGN models. This demonstrates how AGN continuum luminosity connects to reverberation mapping measurements of the BLR extent.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_reverberation_size_luminosity_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_reverberation_size_luminosity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN Reverberation Size-Luminosity Relation (Bentz+2013)</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The 9.7 μm and 18 μm silicate bands are strong diagnostics of AGN torus orientation. When viewing the torus face-on (high cos_inc), dust emission dominates and silicates appear in emission. Edge-on views (low cos_inc) show silicates in absorption against the hot dust continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_silicate_features_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_silicate_features`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR Torus: Silicate features from face-on to edge-on</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The growth of a supermassive black hole (SMBH) traces a path through the (M_BH, L_bol) plane. Starting as a dormant low-mass hole, accretion during mergers builds both mass and luminosity. Peak luminosity occurs as a luminous QSO before accretion slows and the system fades. This example traces four key evolutionary stages and plots both the track on the (M_BH, L_bol) diagram and the corresponding SEDs.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_smbh_growth_track_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_smbh_growth_track`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN SMBH growth track: dormant → merger → QSO → fading</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The torus half-opening angle (OA, polar half-angle in degrees) controls the covering fraction and the relative strength of direct vs. re-processed AGN emission as a function of observer inclination. Smaller OA (narrow torus) covers a smaller solid angle, reducing the fraction of reprocessed emission visible face-on and increasing direct continuum. Larger OA (flared torus) increases covering, suppressing direct light and boosting thermal re-emission in the mid-infrared.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_torus_opening_angle_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_torus_opening_angle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus opening angle sweep: covering factor and MIR emission</div>
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
   /auto_examples/agn/plot_agn_bolometric_correction
   /auto_examples/agn/plot_agn_components_breakdown
   /auto_examples/agn/plot_agn_cos_inc_sweep
   /auto_examples/agn/plot_agn_disc_compare
   /auto_examples/agn/plot_agn_hierarchy
   /auto_examples/agn/plot_agn_host_decomposition
   /auto_examples/agn/plot_agn_lines_compare
   /auto_examples/agn/plot_agn_log_lbol_sweep
   /auto_examples/agn/plot_agn_oa_sweep
   /auto_examples/agn/plot_agn_qsogen_ebv_sweep
   /auto_examples/agn/plot_agn_qsogen_emline_sweep
   /auto_examples/agn/plot_agn_skirtor_p_sweep
   /auto_examples/agn/plot_agn_tau_skirtor_sweep
   /auto_examples/agn/plot_agn_torus_compare
   /auto_examples/agn/plot_agn_type12
   /auto_examples/agn/plot_alpha_ox_lusso_risaliti
   /auto_examples/agn/plot_bh_spin_disc_continuum
   /auto_examples/agn/plot_composable_block_toggles
   /auto_examples/agn/plot_composable_recipes
   /auto_examples/agn/plot_composable_three_modes
   /auto_examples/agn/plot_eddington_ratio_sweep
   /auto_examples/agn/plot_mbh_mstar_relation
   /auto_examples/agn/plot_nlr_blr_lines
   /auto_examples/agn/plot_qsogen_spectrum
   /auto_examples/agn/plot_relagn_spin
   /auto_examples/agn/plot_reverberation_size_luminosity
   /auto_examples/agn/plot_seyfert_quasar_blazar_archetypes
   /auto_examples/agn/plot_skirtor_inclination_sweep
   /auto_examples/agn/plot_skirtor_silicate_features
   /auto_examples/agn/plot_skirtor_variants
   /auto_examples/agn/plot_smbh_growth_track
   /auto_examples/agn/plot_torus_opening_angle
   /auto_examples/agn/plot_type1_type2_unified_model
   /auto_examples/agn/plot_ulirg_to_qso_transition
   /auto_examples/agn/plot_variability_continuum_lag

