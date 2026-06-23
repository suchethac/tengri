:orphan:

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

    <div class="sphx-glr-thumbcontainer" tooltip="Iron pseudo-continuum (Fe II) emission in AGN produces characteristic humps in the near-UV and optical bands. The strength and shape are governed by the Fe II equivalent width and ionization state, parameterized in tengri by the agn_fe2_strength parameter relative to H-beta (Balmer lines).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_feii_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_feii_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fe II pseudo-continuum strength evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Until recently the agn_ parameters were declared with fixed defaults and no prior range, so the build grammar&#x27;s FREE controls (``agn={&#x27;&#x27;: FREE}``, recipes.agn_panchromatic()) silently resolved every AGN parameter to a constant — a fit would freeze the entire AGN sector with no error. The registry now gives each parameter a physically-motivated Uniform/``LogUniform`` prior (Nenkova+2008, Kubota &amp; Done 2018, Stalevski+2016 grid extents), so FREE actually frees them.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_free_param_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_free_param_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN parameters are free-able now — and every one moves the SED</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Dust-free quasar spectra are intrinsically blue in the UV and optical. Adding a polar-dust attenuation component reddens the accretion-disc continuum: increasing the polar-dust reddening agn_polar_ebv (E(B−V), [mag]) from 0 to 0.4 walks the SED from unobscured type-1 QSO colours to a moderately dust-reddened continuum, while the absorbed UV energy is re-radiated as a polar-dust infrared bump.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_qsogen_ebv_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_qsogen_ebv_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSO continuum: polar-dust reddening tunes UV to optical colour</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Seven dusty-torus libraries reprocessing the same accretion-disc continuum at fixed log L_bol = 12.5 (in log L_sun) and standard inclination. The disc is held at multicolor (Kubota &amp; Done 2018) so the differences in the curves are entirely how each torus library geometrically distributes hot grains and re-emits the absorbed UV in the MIR.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_torus_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_torus_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN dusty torus: library comparison at fixed L_bol</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="agn={&quot;feii&quot;: {&quot;type&quot;: &quot;boroson_green&quot;}} adds the empirical Boroson &amp; Green (1992) Fe II pseudo-continuum to a composable AGN — the broad iron-line blanket that fills the 2200–3000 Å and 4400–4700 Å windows of type-1 quasars. Its amplitude is set by agn_fe2_strength (the standard R_{\rm Fe} ratio of Fe II to broad H\ \beta).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_boroson_green_feii_template_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_boroson_green_feii_template`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Boroson & Green (1992) Fe II template: the optical/UV iron blanket</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The CAT3D-Wind torus (Hönig &amp; Kishimoto 2017) splits the circumnuclear dust into a mid-plane clumpy disc plus a polar outflow (&quot;wind&quot;). Its infrared reprocessing is controlled by three observables: the wind mass fraction fwd, the radial cloud-distribution index a, and the viewing angle cos i.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_cat3d_wind_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_cat3d_wind_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">CAT3D-Wind clumpy torus: wind fraction and viewing angle</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The GRAHSP AGN model (Buchner+ 2024) optionally adds a Balmer continuum following Grandi (1982): a 15,000 K blackbody truncated at the Balmer edge (3646 Å) and Gaussian-broadened by the line width. Together with the FeII forest it builds the &quot;small blue bump&quot; seen blueward of ~4000 Å in type-1 quasars.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_balmer_continuum_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_balmer_continuum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP Balmer continuum: building the small blue bump</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The GRAHSP big blue bump can be modelled two ways. The default is a smooth bending power-law (Ryde 1998 form) with free UV/optical slopes and a bend wavelength. The physical alternative is the Netzer accretion-disc grid (Netzer &amp; Trakhtenbrot 2014), tabulated over black-hole mass, spin and Eddington ratio — selected with disc_model=&quot;netzer&quot; plus disc_m / disc_a / disc_mdot.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_disc_vs_bbb_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_disc_vs_bbb`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP accretion disc: Netzer templates vs the bending power-law</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The iron pseudo-continuum (the &quot;FeII forest&quot;) is a defining feature of type-1 AGN optical/UV spectra. GRAHSP offers two templates: the photoionisation model of Bruhweiler &amp; Verner (2008) (the upstream default) and the empirical Veron-Cetty, Joly &amp; Veron (2004) template. They differ most in the relative strength and shape of the UV (2200-3000 Å) and optical (4400-5400 Å) multiplet blends.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_feii_templates_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_feii_templates`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP FeII forest: Bruhweiler+Verner 2008 vs Veron-Cetty 2004</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduction of Fig. 1 of Buchner et al. (2024, GRAHSP): how the individual model components sum to the total emission (black). The AGN side is the GRAHSP bending power-law disk/BBB (blue), iron + emission-line forest (red), and the dusty torus (yellow dashed), normalised so the disk has L_{5100\,\mathrm{\AA}}^{\rm AGN}=10^{44}\,\mathrm{erg\,s^{-1}} =10^{37}\,\mathrm{W} (blue square); the torus is anchored at 12 µm (yellow diamond). The host is a stellar population (purple) and its reprocessed dust emission (green).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_paper_fig1_overview_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_paper_fig1_overview`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP Fig. 1 reproduction: panchromatic AGN + host overview</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Faithful reproduction of Fig. 3 of Buchner et al. (2024, GRAHSP), showing how the 15 AGN parameters configure the spectrum in L_\lambda (arbitrary units). The bending power-law BBB (blue) is normalised at 5100 Å with optical slope \beta, bending at \lambda_{\rm bend} (width W_{\rm bend}) to the UV slope \beta_{\rm UV}. Emission lines (light red) of width W_{\rm lines} and an FeII forest (dark red) are scaled by A_{\rm lines} / A_{\rm FeII}. The log-Gaussian torus (dark yellow) has cool/hot components at \lambda_{\rm cool}/\lambda_{\rm hot} (widths W_{\rm cool}/W_{\rm hot}), peak ratio f_{\rm hot}, 12 µm normalisation f_{\rm cov}, and silicate depth Si (here −1, absorption; dotted). Component colours map to the GRAHSP/pcigale modules activatepl (BBB), activategtorus (torus), and activatelines (lines + FeII).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_paper_fig3_params_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_paper_fig3_params`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP Fig. 3 reproduction: AGN model parameter map</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Faithful reproduction of Fig. 9 of Buchner et al. (2024, GRAHSP): the AGN spectrum from intrinsic (blue, top) to strongly attenuated (red, bottom) as the AGN-only colour excess agn_grahsp_ebv_agn is swept from 0.01 to 1. GRAHSP attenuates the AGN side with an SMC/Prevot (1984) law (paper §2.1.5), which rises steeply into the UV — so the UV/optical continuum is suppressed far more than the near-IR, and the heaviest attenuation eventually bites into the torus too. The intrinsic torus component is overplotted dashed black.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_paper_fig9_agn_attenuation_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_paper_fig9_agn_attenuation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP Fig. 9 reproduction: attenuation of the AGN model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="GRAHSP ships two torus prescriptions. The default is an empirical log-Gaussian cool+hot dust continuum (``activategtorus``). The alternative is the Mor &amp; Netzer 2012 template torus (``activatetorus``), which interpolates between mean / 25th / 75th-percentile observed AGN mid-IR SEDs via agn_grahsp_tor_temp and applies a short-wavelength Gaussian cutoff at agn_grahsp_tor_cutoff_um.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_torus_modes_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_torus_modes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP torus: empirical log-Gaussian vs Mor & Netzer 2012 templates</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The Nenkova et al. (2008) CLUMPY library is the AGN dusty-torus model used by FSPS and Prospector. tengri ships the same templates (vendored from FSPS as data/nenkova08_torus_grid.h5) and interpolates them with a pure-JAX triweight kernel, so the equatorial optical depth agn_tau is a fully differentiable, fitted parameter — it can be sampled by NUTS, optimised by MAP, or marginalised by VI, just like in Prospector.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_nenkova_tau_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_nenkova_tau_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">CLUMPY torus (Nenkova+2008): optical depth as a fitted parameter</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The Silva, Maiolino &amp; Granato (2004) AGN torus templates are empirical reprocessed-dust SEDs binned by line-of-sight hydrogen column density agn_log_nh_silva. As the column rises from unobscured (Type-1-like, N_\mathrm{H} \sim 10^{22}\,\mathrm{cm^{-2}}) to Compton-thick (N_\mathrm{H} \sim 10^{25}\,\mathrm{cm^{-2}}):">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_silva04_nh_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_silva04_nh_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Silva+04 torus: Obscuration and the 9.7 micron silicate feature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The SKIRTOR_mean_3p torus is the Stalevski+2016 clumpy two-phase torus library averaged over its clumpiness parameters, as packaged by AGNfitter-rX. Three observables remain: the half-opening angle oa, the equatorial optical depth tau_V, and the inclination incl.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_agnfitter_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_agnfitter_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR_mean_3p torus: optical depth and inclination</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="SKIRTOR (Stalevski et al. 2016) is a clumpy radiative transfer torus model with a three-dimensional parameter space (half-opening angle, inclination, optical depth). Two different implementations exist in tengri:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_agnfitter_vs_cigale_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_agnfitter_vs_cigale`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: AGNfitter-averaged vs. X-CIGALE full grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrate how the SKIRTOR clumpy radiative-transfer torus (Stalevski+ 2012, 2016) reprocesses the hot accretion disc and dust as a function of viewing angle.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_inclination_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_inclination_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR AGN torus: inclination-dependent obscuration and silicate features</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="agn={&quot;type&quot;: &quot;skirtor_stalevski&quot;} returns the SKIRTOR 2016 SED exactly as Stalevski&#x27;s radiative transfer computed it — accretion disc + clumpy torus + scattered light, read straight from the full-coverage grid with no analytic-disc substitution and no re-normalisation. This is the faithful template that codes reading SKIRTOR directly (e.g. ProSpect&#x27;s SKIRTOR_interp) reproduce.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_stalevski_raw_template_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_stalevski_raw_template`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Raw SKIRTOR (Stalevski 2016): the published radiative-transfer template</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="This is tengri&#x27;s full-grid SKIRTOR torus (Stalevski+2012, 2016), following the X-CIGALE skirtor2016 conventions: a 5-D clumpy two-phase library indexed by equatorial optical depth tau, radial and polar density gradients p / q, half-opening angle oa, and inclination cos i (plus an optional Casey-2012 polar-dust greybody). It is the science-grade counterpart to the parameter-averaged skirtor_agnfitter library — and, having the full grid, it responds strongly to its parameters.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_xcigale_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_xcigale_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus (full X-CIGALE grid): optical depth and inclination</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Slone &amp; Netzer (2012) accretion-disc library (SN12, as packaged by AGNfitter-rX) tabulates the big-blue-bump continuum over black-hole mass and Eddington ratio. The disc&#x27;s characteristic temperature scales as T_\mathrm{max} \propto (\dot m / M_\mathrm{BH})^{1/4}, so the spectral peak walks across the UV/optical as those two knobs change:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_slone_netzer_disc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_slone_netzer_disc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Slone & Netzer 2012 disc: Black-hole mass and Eddington ratio</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The composable AGN runner sums disc + broad/narrow lines + FeII + torus, but a real dusty torus also obscures the central engine along edge-on sightlines while its own infrared emission is not re-extinguished by that same screen. tengri applies this inclination-dependent torus screen automatically whenever the torus is one of the two CIGALE production grids (``skirtor`` or fritz); it closes the &quot;disc + torus composed additively, no torus screen on disc&quot; gap (#294).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_torus_screen_disc_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_torus_screen_disc`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN unification: the torus screens the disc with inclination</div>
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
   /auto_examples/agn/plot_agn_feii_sweep
   /auto_examples/agn/plot_agn_free_param_sensitivity
   /auto_examples/agn/plot_agn_hierarchy
   /auto_examples/agn/plot_agn_host_decomposition
   /auto_examples/agn/plot_agn_lines_compare
   /auto_examples/agn/plot_agn_qsogen_ebv_sweep
   /auto_examples/agn/plot_agn_qsogen_emline_sweep
   /auto_examples/agn/plot_agn_torus_compare
   /auto_examples/agn/plot_alpha_ox_lusso_risaliti
   /auto_examples/agn/plot_alpha_ox_uv_xray_connection
   /auto_examples/agn/plot_boroson_green_feii_template
   /auto_examples/agn/plot_cat3d_wind_sweep
   /auto_examples/agn/plot_composable_block_toggles
   /auto_examples/agn/plot_composable_recipes
   /auto_examples/agn/plot_custom_torus_extension
   /auto_examples/agn/plot_grahsp_balmer_continuum
   /auto_examples/agn/plot_grahsp_disc_vs_bbb
   /auto_examples/agn/plot_grahsp_feii_templates
   /auto_examples/agn/plot_grahsp_paper_fig1_overview
   /auto_examples/agn/plot_grahsp_paper_fig3_params
   /auto_examples/agn/plot_grahsp_paper_fig9_agn_attenuation
   /auto_examples/agn/plot_grahsp_torus_modes
   /auto_examples/agn/plot_kd18_disc_sweep
   /auto_examples/agn/plot_nenkova_tau_sweep
   /auto_examples/agn/plot_nlr_blr_lines
   /auto_examples/agn/plot_polar_dust_ebv_sweep
   /auto_examples/agn/plot_polar_dust_ebv_type12_sweep
   /auto_examples/agn/plot_qsogen_spectrum
   /auto_examples/agn/plot_relagn_spin
   /auto_examples/agn/plot_reverberation_size_luminosity
   /auto_examples/agn/plot_richards2006_template
   /auto_examples/agn/plot_seyfert_quasar_blazar_archetypes
   /auto_examples/agn/plot_silva04_nh_sweep
   /auto_examples/agn/plot_skirtor_agnfitter_sweep
   /auto_examples/agn/plot_skirtor_agnfitter_vs_cigale
   /auto_examples/agn/plot_skirtor_inclination_sweep
   /auto_examples/agn/plot_skirtor_stalevski_raw_template
   /auto_examples/agn/plot_skirtor_variants
   /auto_examples/agn/plot_skirtor_vs_smooth_torus
   /auto_examples/agn/plot_skirtor_xcigale_sweep
   /auto_examples/agn/plot_slone_netzer_disc_sweep
   /auto_examples/agn/plot_smbh_growth_track
   /auto_examples/agn/plot_torus_screen_disc
   /auto_examples/agn/plot_type1_type2_unified_model
   /auto_examples/agn/plot_ulirg_to_qso_transition
   /auto_examples/agn/plot_variability_continuum_lag

