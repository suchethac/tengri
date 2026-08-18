:orphan:

.. _sphx_glr_auto_examples_xray:

X-ray Emission
==============

X-ray binaries (HMXB, LMXB) scaled with SFR and stellar mass. AGN coronae: luminosity, photon index γ, exponential cutoff E_cut, UV-to-X-ray slope α_ox. Model-family comparison included.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The CIGALE-faithful corona derives the X-ray normalization from L_2500 via the empirical alpha_OX-L_2500 correlation. tengri ships three published parametrizations:">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_alpha_ox_relations_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_alpha_ox_relations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Three empirical alpha_OX-L_2500 prescriptions diverge at the quasar peak</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The UV-to-X-ray spectral slope alpha_OX (defined as log F_X minus log F_UV divided by log nu_X minus log nu_UV) separates X-ray-loud quasars (alpha_OX around -1.2, strong X-ray relative to the UV continuum) from X-ray-quiet systems (alpha_OX around -1.8, suppressed X-ray). The CIGALE-faithful corona derives alpha_OX from L_2500 via the Just+2007 relation by default; here we sweep delta_alpha_ox to apply offsets from -0.4 to +0.4 around that empirical value, at fixed L_2500 (= L_bol = 1e45 erg/s through the standard Hopkins+2007 bolometric correction). More positive delta brightens the corona; more negative suppresses it.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_alpha_ox_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_alpha_ox_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN UV-to-X-ray power-law slope alpha_OX controls X-ray normalization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The X-CIGALE X-ray module (Yang et al. 2020) sums four physically distinct emitters: the AGN corona (a cut-off power law normalized through the α_OX–L_2500 relation), low- and high-mass X-ray binaries (LMXB ∝ M⋆, HMXB ∝ SFR; Lehmer et al. 2016 metallicity/age scalings), and a hot interstellar-gas term (∝ SFR). This reproduces Yang+2020 Figure 1 for a typical AGN host: L_2–10 keV = 10⁴³ erg s⁻¹, M⋆ = 10¹¹ M⊙, SFR = 10 M⊙ yr⁻¹, T = 1 Gyr, Z = 0.02.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_component_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_component_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray SED decomposition: AGN, LMXB, HMXB, hot gas</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every X-ray model tengri registers, on one host galaxy, with only the xray block changing. At default parameters they collapse onto two curves, and that is the point of the figure rather than a defect in it:">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_model_family_compare_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_model_family_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray model family: five names, two prescriptions</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The line-of-sight column density N_H reshapes the AGN X-ray spectrum in two regimes: photoelectric absorption (``zphabs``) suppresses the soft band roughly as \exp(-\sigma(E)\,N_H) with cross-section \sigma \propto E^{-3}, while Compton down-scattering (``cabs``) adds an energy-independent suppression \exp(-\sigma_T\,N_H) that becomes dominant once log N_H ≳ 24 (the Compton-thick boundary). A constant warm-electron scattered fraction (~1 % of the intrinsic continuum) is added back, which is the only flux observable in the soft band for nearly opaque columns and explains why Compton-thick AGN are still marginally detectable in soft-band stacks (Matsumoto et al. 2026 Fig. 11/12).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_nh_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_nh_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">N_H column density sweep: from unobscured to Compton-thick</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In Compton-thick AGN (log N_H ≳ 24 cm⁻²), the line-of-sight obscurer extinguishes the primary AGN corona below ~ 10 keV. What&#x27;s left is the reflected component — the fraction of corona photons that hit the cold accretion disc, Compton-scatter off bound electrons, and emerge along the line of sight without being photoelectrically absorbed. The resulting spectrum peaks around 30 keV (the famous Compton hump) and is the smoking-gun signature that NuSTAR / Swift-BAT surveys use to confirm buried supermassive black holes (Ricci+2017, Matsumoto+26).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_pexrav_compton_hump_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_pexrav_compton_hump`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compton hump in obscured AGN: pexrav reflection across log N_H</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="X-ray binaries (XRBs) are the dominant X-ray sources in star-forming galaxies once an AGN is excluded. High-mass XRBs trace the recent star-formation rate (Mineo+2012), while low-mass XRBs trace the integrated stellar mass (Lehmer+2019). The two scalings have different spectral shapes too: HMXBs are slightly harder, LMXBs slightly softer. Two side-by-side sweeps — SFR (left) at fixed M_star = 1e11 M☉, and M_star (right) at fixed SFR = 10 M☉/yr — separate the two channels on the same axes.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_sf_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_sf`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray binary luminosity scales with SFR (HMXB) and stellar mass (LMXB)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/xray/plot_alpha_ox_relations
   /auto_examples/xray/plot_alpha_ox_sweep
   /auto_examples/xray/plot_xray_component_decomposition
   /auto_examples/xray/plot_xray_model_family_compare
   /auto_examples/xray/plot_xray_nh_sweep
   /auto_examples/xray/plot_xray_pexrav_compton_hump
   /auto_examples/xray/plot_xray_sf

