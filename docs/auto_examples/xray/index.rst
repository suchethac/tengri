

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

    <div class="sphx-glr-thumbcontainer" tooltip="AGN coronae are compact hot regions where the hard X-ray power law (photon index ~1.7–2.0) is produced via Compton scattering off hot electrons. The X-ray spectrum reflects the coronal temperature, optical depth, and geometry. This demo varies L_bol across six decades (10⁴²–10⁴⁶·⁵ erg/s) to show the gradual brightening of the X-ray continuum and the persistence of the power-law form across the luminosity sequence. A separate panel isolates key spectral features: soft excess (0.5–2 keV), hard continuum (2–10 keV), Compton reflection hump (10–100 keV), and the iron K-α line (6.4 keV).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_agn_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_agn`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN X-ray coronae: luminosity sequence and spectral hardness at high energies</div>
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
   /auto_examples/xray/plot_xray_sf
   /auto_examples/xray/plot_xray_vs_agn_lbol

