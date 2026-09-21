:orphan:

.. _sphx_glr_auto_examples_agn:

AGN Models
==========

Torus models in `components/agn/torus.py` are toy models; SKIRTOR is the one for science. Disc continua (multicolor, KD18, relagn, qsogen), narrow-/broad-line and FeII emission, polar-dust and Type 1/2 attenuation. Cross-validated against CIGALE and AGNfitter.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All thirteen accretion-disc backbones registered under agn.disc.type, at fixed bolometric luminosity log L_bol = 12.5 (in log L_sun), evaluated in isolation with the host suppressed and no torus/lines/dust. The differences between the curves are entirely how each model partitions the disc power across wavelength: pure blackbody vs warm Comptonization, relativistic vs Newtonian potential, radiatively efficient thin disc vs inefficient ADAF, empirical composite vs first-principles continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_disc_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_disc_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN disc continuum: every registered model at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Iron pseudo-continuum (Fe II) emission in AGN produces characteristic humps in the near-UV and optical bands. The strength and shape are governed by the Fe II equivalent width and ionization state, parameterized in tengri by the agn_fe2_strength parameter relative to H-beta (Balmer lines).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_feii_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_feii_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fe II pseudo-continuum strength evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Until recently the agn_ parameters were declared with fixed* defaults and no prior range, so the build grammar&#x27;s FREE controls (``agn={&#x27;all_params&#x27;: FREE}``, recipes.agn_panchromatic()) silently resolved every AGN parameter to a constant — a fit would freeze the entire AGN sector with no error. The registry now gives each parameter a physically-motivated Uniform/``LogUniform`` prior (Nenkova+2008, Kubota &amp; Done 2018, Stalevski+2016 grid extents), so FREE actually frees them.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_free_param_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_free_param_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN parameters are free-able now — and every one moves the SED</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four AGN configurations of increasing physical complexity at the same bolometric luminosity (log L_bol = 12.5 in L_sun units) — bare multicolor disc, +SKIRTOR torus, +NLR narrow-line forest, and an empirical QSOgen template that bundles all of the above. The reader sees which spectral feature each block introduces (mid-IR torus bump, optical narrow lines, broad UV continuum) and which are essentially universal across the modeling choice.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_hierarchy_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_hierarchy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Building up an AGN SED: disc, then torus, then lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The renderable line backbones registered under the three composable line selectors — agn.nlr (narrow-line region), agn.blr (broad-line region), and agn.feii (iron pseudo-continuum) — each layered on the same disc + torus at fixed log L_bol = 12.5. The backbone controls which optical/UV features the model produces: narrow forbidden lines, broad permitted lines, or the blended Fe II forest.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_lines_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_lines_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN emission-line backbones compared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The QSOgen model (Temple+2021) includes empirical UV/optical emission-line forest and broad Balmer continuum. The relative strength of these line features with respect to the continuum obeys the Baldwin effect: luminous quasars show weaker equivalent-width emission lines (the line flux grows sublinearly with continuum). This sweep shows the Baldwin effect in the QSOgen template across six decades of bolometric luminosity (log L_bol = 9 to 13 L_sun), revealing the Ly-alpha + C IV feature cluster around 1000–1600 Å and optical hydrogen Balmer lines (Hα, Hβ).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_qsogen_emline_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_qsogen_emline_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen emission lines: Baldwin effect across AGN luminosity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All ten dusty-torus libraries registered under agn.torus.type, reprocessing the same accretion-disc continuum at fixed log L_bol = 12.5 (in log L_sun) and standard inclination. The disc is held at multicolor (Kubota &amp; Done 2018) so the differences in the curves are entirely how each torus library geometrically distributes hot grains and re-emits the absorbed UV in the MIR — clumpy radiative transfer (SKIRTOR, CLUMPY, CAT3D-WIND) vs smooth-dust grids (Fritz, Silva) vs phenomenological graybodies.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_torus_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_torus_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN dusty torus: library comparison at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The agn.disc, agn.lines, agn.feii, agn.torus, agn.atten sub-blocks of SEDModel.build are composable: turning one on at a time and overlaying the all-on reference (dashed gray) shows which features each sub-block contributes. Five panels at fixed log L_bol = 12.0, all built via the public nested-dict grammar:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_composable_block_toggles_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_composable_block_toggles`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cumulative buildup of the GRAHSP AGN recipe, one sub-block at a time</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A toy single-temperature blackbody torus implemented as a modern SEDModelComponent subclass, discoverable through SEDModel.build and composable with other AGN blocks. The SEDModelComponent pattern is the recommended path for any new SED physics — AGN, dust, or stellar.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_custom_torus_extension_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_custom_torus_extension`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Custom AGN torus model via SEDModelComponent and direct integration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Kubota &amp; Done (2018) three-zone accretion disc model shows how the big blue bump (BBB) peaks at different wavelengths depending on black-hole mass and Eddington ratio. Sweeping across the accretion-state plane from low-luminosity advection-dominated (ADAF-like) to high-Eddington thin-disc reveals the transition: high mass + low Eddington gives cool outer discs peaking in the NIR; low mass + high Eddington gives hot inner zones peaking in the FUV/UV.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_kd18_disc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_kd18_disc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kubota & Done 2018 disc: Accretion state effects on continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Polar dust disc attenuation applies only to Type 1 (face-on) sightlines — the equatorial torus already screens the disc for Type 2. The bi-conical polar dust absorbs disc photons regardless of viewing angle, however, and re-emits them isotropically as a FIR graybody (Casey 2012). So both Type 1 and Type 2 sweeps show the FIR re-emission bump growing with E(B-V); only the UV/optical attenuation is gated by sightline.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_polar_dust_ebv_type12_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_polar_dust_ebv_type12_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polar dust E(B-V) reddens Type 1 & 2 AGN differently</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="This is tengri&#x27;s full-grid SKIRTOR torus (Stalevski+2012, 2016), following the X-CIGALE skirtor2016 conventions: a 5-D clumpy two-phase library indexed by equatorial optical depth tau, radial and polar density gradients p / q, half-opening angle oa, and inclination cos i (plus an optional Casey-2012 polar-dust graybody). It is the science-grade counterpart to the parameter-averaged skirtor_agnfitter library — and, having the full grid, it responds strongly to its parameters.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_xcigale_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_xcigale_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus (full X-CIGALE grid): optical depth and inclination</div>
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


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/agn/plot_agn_disc_compare
   /auto_examples/agn/plot_agn_feii_sweep
   /auto_examples/agn/plot_agn_free_param_sensitivity
   /auto_examples/agn/plot_agn_hierarchy
   /auto_examples/agn/plot_agn_lines_compare
   /auto_examples/agn/plot_agn_qsogen_emline_sweep
   /auto_examples/agn/plot_agn_torus_compare
   /auto_examples/agn/plot_composable_block_toggles
   /auto_examples/agn/plot_custom_torus_extension
   /auto_examples/agn/plot_kd18_disc_sweep
   /auto_examples/agn/plot_polar_dust_ebv_type12_sweep
   /auto_examples/agn/plot_relagn_spin
   /auto_examples/agn/plot_skirtor_xcigale_sweep
   /auto_examples/agn/plot_type1_type2_unified_model
   /auto_examples/agn/plot_ulirg_to_qso_transition

