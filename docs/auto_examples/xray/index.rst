

.. _sphx_glr_auto_examples_xray:

X-ray Emission
==============

Multi-wavelength X-ray components: X-ray binaries (HMXB + LMXB) and AGN coronae.

Star-Forming Galaxies
^^^^^^^^^^^^^^^^^^^^^

- ``plot_xray_sf.py`` — X-ray binary scaling with SFR and stellar mass

AGN Coronae
^^^^^^^^^^^

- ``plot_xray_agn.py`` — AGN X-ray coronae: luminosity sequence and spectral hardness
- ``plot_xray_gamma_sweep.py`` — Photon index γ controls power-law steepness
- ``plot_E_cut_sweep.py`` — Exponential cutoff E_cut governs hard-tail rollover
- ``plot_alpha_ox_sweep.py`` — UV-to-X-ray slope α_ox controls normalisation



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

    <div class="sphx-glr-thumbcontainer" tooltip="The UV-to-X-ray spectral slope α_ox (defined as log(F_X) − log(F_UV) / log(ν_X) − log(ν_UV)) separates &quot;X-ray loud&quot; quasars (α_ox ~ −1.2, strong X-ray relative to UV continuum) from &quot;X-ray quiet&quot; systems (α_ox ~ −1.8, suppressed X-ray). More negative α_ox suppresses the X-ray continuum and weakens the high-energy tail. We vary α_ox at fixed bolometric luminosity, showing the anticorrelation of X-ray strength and UV continuum slope.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_alpha_ox_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_alpha_ox_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN UV-to-X-ray power-law slope α_ox controls relative X-ray normalisation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="AGN coronae are compact hot regions where the hard X-ray power law (photon index ~1.7–2.0) is produced via Compton up-scattering of seed UV photons by hot electrons. xray_agn_corona models the primary continuum as a cut-off power-law normalised through the α_ox–L_2500 relation (Lusso &amp; Risaliti 2016), then attenuated by the zphabs(N_H) × cabs(N_H) line-of-sight obscurer with a 1 % warm-electron scattered fraction added back (Ricci+2017 spectral model adopted in Matsumoto+2026 Eq. B6).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_agn_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_agn`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN X-ray coronae: luminosity sequence</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="X-ray binaries (XRBs) are among the brightest X-ray sources in galaxies. High-mass XRBs (HMXBs) form copiously during starbursts and scale ~SFR, while low-mass XRBs (LMXBs) are long-lived remnants scaling with integrated stellar mass. This demo isolates the SFR and M_* dependencies separately on a starburst galaxy template to show how the XRB spectral luminosity responds to both the recent star formation rate and the accumulated stellar mass.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_sf_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_sf`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray binary scaling: HMXB traces current SFR, LMXB traces stellar mass</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed host (constant SFR = 3 M_sun/yr, Mineo+12 HMXB contribution) we sweep the composable AGN&#x27;s bolometric luminosity agn_log_lbol from 9 to 13 (in log L_sun). The host XRB component is a flat power-law below ~10 keV; the AGN corona contributes a much harder power-law that dominates above log L_bol ≳ 11.">

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
   /auto_examples/xray/plot_alpha_ox_sweep
   /auto_examples/xray/plot_xray_agn
   /auto_examples/xray/plot_xray_gamma_sweep
   /auto_examples/xray/plot_xray_nh_sweep
   /auto_examples/xray/plot_xray_sf
   /auto_examples/xray/plot_xray_vs_agn_lbol

