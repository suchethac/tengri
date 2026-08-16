:orphan:

.. _sphx_glr_auto_examples_xray:

X-ray Emission
==============

X-ray binaries (HMXB, LMXB) scaled with SFR and stellar mass. AGN coronae: luminosity, photon index γ, exponential cutoff E_cut, UV-to-X-ray slope α_ox.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The X-ray power-law spectrum steepens above an exponential cutoff E_cut. Compact coronae with low optical depth have low E_cut (~100 keV); thick, optically-deep coronae extend to higher E_cut (~1 TeV). Variation of E_cut at fixed γ=1.8 and α_ox=−1.4 shows how the hard X-ray tail responds to changes in coronal geometry or magnetic field.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_E_cut_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_E_cut_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN X-ray hard-tail rollover: exponential cutoff E_cut governs high-energy turnover</div>
    </div>


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

    <div class="sphx-glr-thumbcontainer" tooltip="X-ray absorption in AGN undergoes a qualitative shift at N_H ≈ 1e24 cm⁻², where the cross-section for Compton scattering becomes comparable to photoelectric absorption. Below this threshold, soft photons (E &lt; 10 keV) are suppressed by the Thompson cross-section σ_T ≈ 0.66 Barn, creating a steep spectral curvature in the soft band. Above it, the entire 2–10 keV continuum is suppressed equally, flattening the spectrum and leaving only a scattered component (~1% of the intrinsic flux) observable.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_compton_thick_vs_thin_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_compton_thick_vs_thin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photoelectric vs. Compton-thick regimes: the N_H = 1e24 cm−2 transition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The AGN X-ray corona produces a cut-off power-law (photon index Gamma roughly 1.8, E_cut around 300 keV) normalized through the alpha_OX-L_2500 relation (Lusso &amp; Risaliti 2016). At fixed Gamma and alpha_OX, increasing bolometric luminosity shifts the whole spectrum upward but leaves the spectral shape nearly intact — the sub-linear alpha_OX relation only steepens the shape at the top of the quasar regime.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_agn_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_agn`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN corona: bolometric luminosity sets normalization, not shape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The CIGALE-faithful obscured-AGN spectral model combines two knobs that classification surveys often confound: delta_alpha_ox (offset from the empirical alpha_OX-L_2500 relation, controlling the intrinsic X-ray-to-UV ratio) and log N_H (line-of-sight column density, suppressing soft-band flux through zphabs × cabs). We compute the hardness ratio HR = (H - S) / (H + S) with S = 0.5–2 keV and H = 2–10 keV across the joint (delta_alpha_ox, log N_H) plane on a fixed L_2500 anchor (= L_bol = 1e45 erg/s through the Hopkins+2007 bolometric correction).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_alpha_ox_nh_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_alpha_ox_nh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hardness ratio across the alpha_OX vs log N_H plane</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The X-ray photon index γ controls how rapidly the AGN corona&#x27;s power-law spectrum falls off above a few keV. Flat spectra (low γ ~1.4) extend more photons to high energies; steep spectra (high γ ~2.4) drop quickly. We vary γ across its typical observational range at fixed bolometric luminosity.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_gamma_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_gamma_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN X-ray spectral hardness: photon index γ controls power-law steepness</div>
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


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed host (constant SFR = 3 M☉/yr, Mineo+12 HMXB contribution) we sweep the composable AGN&#x27;s bolometric luminosity agn_log_lbol from 9 to 13 (in log L_sun). The host XRB component is a flat power-law below ~10 keV; the AGN corona contributes a much harder power-law that dominates above log L_bol ≳ 11.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_vs_agn_lbol_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_vs_agn_lbol`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray SED response to AGN bolometric luminosity</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/xray/plot_E_cut_sweep
   /auto_examples/xray/plot_alpha_ox_relations
   /auto_examples/xray/plot_alpha_ox_sweep
   /auto_examples/xray/plot_compton_thick_vs_thin
   /auto_examples/xray/plot_xray_agn
   /auto_examples/xray/plot_xray_alpha_ox_nh
   /auto_examples/xray/plot_xray_component_decomposition
   /auto_examples/xray/plot_xray_gamma_sweep
   /auto_examples/xray/plot_xray_nh_sweep
   /auto_examples/xray/plot_xray_pexrav_compton_hump
   /auto_examples/xray/plot_xray_sf
   /auto_examples/xray/plot_xray_vs_agn_lbol

