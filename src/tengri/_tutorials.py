# SPDX-License-Identifier: BSD-3-Clause
"""Interactive tutorials; runnable, copy-pasteable code recipes.

Each tutorial is a small self-contained snippet teaching one concrete
pattern (a fit, a custom model, a custom likelihood, an inference
swap, …).  Designed for the REPL workflow:

    >>> tengri.tutorial()  # list available topics
    >>> tengri.tutorial("first_fit")  # print the recipe
    >>> tengri.tutorial("register_a_model", run=True)  # actually execute it

Tutorials that require an SSP file or external data print the recipe
but skip the live execution with a clear note.  Tutorials that operate
purely on the registry (``register_a_model``, ``custom_likelihood``,
``swap_inference``) execute live and the user sees the introspection
results immediately.
"""

from __future__ import annotations

import textwrap
import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class _Tutorial:
    name: str
    title: str
    code: str
    runner: callable | None = None  # None = print-only
    needs_ssp: bool = False


# ──────────────────────────────────────────────────────────────────
# Recipe 1: first fit (synthetic)
# ──────────────────────────────────────────────────────────────────


_FIRST_FIT = _Tutorial(
    name="first_fit",
    title="Your first SED fit (mock galaxy → posterior)",
    needs_ssp=True,
    code=textwrap.dedent(
        """
        import jax
        import tengri

        # 1. Pick the bands and load a SSP grid
        photometry = tengri.Photometry.from_names(
            tengri.list_filters(survey="SDSS").names()
            + tengri.list_filters(survey="2MASS").names()
        )
        ssp = tengri.load_ssp_data("data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

        # 2. Build a Parameters spec (priors + fixed values).  See
        #    tengri.suggest_parameters(mean_sfh_type="dpl") for the full
        #    list of legal kwargs.
        spec = tengri.Parameters(
            mean_sfh_type="dpl",
            redshift=tengri.Fixed(0.05),
            sfh_dpl_alpha=tengri.Uniform(0.5, 3.0),
            sfh_dpl_beta=tengri.Uniform(0.5, 3.0),
            sfh_dpl_tau_gyr=tengri.Uniform(0.5, 8.0),
            sfh_dpl_log_total_mass=tengri.Uniform(8.0, 12.0),
            dust_tau_diff=tengri.Uniform(0.0, 2.0),
            met_logzsol=tengri.Uniform(-1.5, 0.2),
        )

        # 3. Generate mock photometry from a known truth point
        obs = tengri.Observation(photometry=photometry)
        model = tengri.SEDModel(spec, ssp, observation=obs)
        truth = spec.sample(jax.random.PRNGKey(0))   # the truth point to recover
        mock = tengri.generate_mock(model, truth, key=jax.random.PRNGKey(1), snr=20.0)

        # 4. Fit it back
        fitter = tengri.Fitter(model, data=mock["flux_obs"], noise=mock["noise"])
        posterior = fitter.run("map", n_steps=300, key=jax.random.PRNGKey(1))
        posterior.summary()
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 2: register a new AGN model alternative (LIVE)
# ──────────────────────────────────────────────────────────────────


def _run_register_a_model() -> None:
    """Live execution: register a toy torus, query it via the introspection
    API, demonstrate it appears in summary() / list_agn_models() / search()
    immediately."""
    import tengri
    from tengri.components.agn.unified import register_agn_model

    print(
        ">>> @register_agn_model('demo_torus', citation='Tutorial demo', "
        "status='experimental', short_doc='Tutorial: single-T graybody')"
    )
    print(">>> def demo_torus(wavelength, agn_log_lbol, **_kwargs): ...")
    print()

    @register_agn_model(
        "demo_torus",
        citation="Tutorial demo",
        status="experimental",
        short_doc="Tutorial: single-T graybody",
    )
    def _demo_torus(wavelength, agn_log_lbol=10.0, **_kwargs):
        # Trivial body; the point is registration, not physics.
        import jax.numpy as jnp

        return jnp.ones_like(wavelength) * (10.0**agn_log_lbol)

    print("--- list_agn_models(status='experimental') ---")
    print(tengri.list_agn_models(status="experimental"))
    print()
    print("--- describe('demo_torus') ---")
    print(tengri.describe("demo_torus"))
    print()
    print("--- search('demo_torus') ---")
    print(tengri.search("demo_torus"))


_REGISTER_A_MODEL = _Tutorial(
    name="register_a_model",
    title="Add a new AGN torus alternative; discoverable in 6 lines",
    runner=_run_register_a_model,
    code=textwrap.dedent(
        '''
        from tengri.components.agn.unified import register_agn_model
        import jax.numpy as jnp

        @register_agn_model(
            "my_torus",
            citation="Author & Year (ApJ 999, 1)",
            status="experimental",
            short_doc="One-line description for list_agn_models().",
        )
        def my_torus(wavelength, agn_log_lbol, agn_T_torus=300.0, **_kwargs):
            """Replace this body with your physics.  Returns L_nu in erg/s/Hz."""
            ...

        # Now your model is discoverable everywhere:
        tengri.list_agn_models(status="experimental")   # your row appears
        tengri.describe("my_torus")                      # full metadata
        tengri.search("my_torus")                        # cross-menu search

        # And usable via Parameters:
        spec = tengri.Parameters(mean_sfh_type="dpl",
                                 agn_model="my_torus",
                                 agn_log_lbol=tengri.Uniform(8, 12))
        '''
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 3: same model, swap inference (LIVE if SSP available)
# ──────────────────────────────────────────────────────────────────


_SWAP_INFERENCE = _Tutorial(
    name="swap_inference",
    title="Same model, three samplers: no other changes",
    needs_ssp=True,
    code=textwrap.dedent(
        """
        # Build the fitter once
        fitter = tengri.Fitter(model, data=mock["flux_obs"], noise=mock["noise"])

        # MAP: Adam, fast, point estimate
        map_result = fitter.run("map", n_steps=300)

        # NUTS: exact, ≲20-D, gold standard
        nuts_result = fitter.run("mcmc_nuts", init_from=map_result, n_warmup=500)

        # geoVI: variational, scales to high-D, cheaper than NUTS
        vi_result = fitter.run("vi", init_from=map_result)

        # "auto": picks NUTS for low-D, ray-tracing for high-D
        auto = fitter.run("mcmc", init_from=map_result)

        # See tengri.list_inference_methods(tier="primary") for all options.
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 4: custom likelihood
# ──────────────────────────────────────────────────────────────────


_CUSTOM_LIKELIHOOD = _Tutorial(
    name="custom_likelihood",
    title="Define a custom likelihood (Student-t, censored, censored+upper limits, …)",
    code=textwrap.dedent(
        """
        # The simplest path: pass a NoiseModel that already encodes the
        # likelihood you want.  These three are built-in:
        from tengri import NoiseModel

        gauss      = NoiseModel.gaussian(noise)                   # default
        student_t  = NoiseModel.student_t(noise, dof=4.0)         # heavy-tail
        with_floor = NoiseModel.gaussian(noise, frac_cal=0.05)    # 5% calibration floor

        fitter = tengri.Fitter(model, data=fluxes, noise=student_t)

        # For genuinely non-standard likelihoods, implement the Likelihood
        # Protocol from tengri.protocols.  Minimum surface:
        #
        #     class MyLikelihood:
        #         def log_prob(self, params, predicted, data) -> float: ...
        #
        # Pass it to the Fitter via:
        #     Fitter(model, data, noise=..., likelihood=my_likelihood)
        #
        # See src/tengri/inference/likelihood.py for the contract and
        # src/tengri/inference/likelihoods/ for the in-tree examples
        # (PhotometryLikelihood, SpectroscopyLikelihood, JointLikelihood,
        # CalibrationMarginalizedLikelihood).
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 5: Protocol-conforming custom component (LIVE)
# ──────────────────────────────────────────────────────────────────


def _run_register_a_component() -> None:
    """Live execution: register a tiny SEDComponent and verify
    its declared parameter shows up in the param map."""
    from dataclasses import dataclass, field

    import tengri
    from tengri.protocols.component import ParamDeclaration, SEDComponentConfig

    @dataclass(frozen=True)
    class _DemoConfig(SEDComponentConfig):
        name: str = "demo"

    @tengri.register_component
    @dataclass(frozen=True)
    class _DemoComponent:
        config: _DemoConfig = field(default_factory=_DemoConfig)
        name: str = "demo"
        parameter_prefix: str = "demo_"

        def declared_parameters(self):
            return [
                ParamDeclaration(
                    "demo_amp",
                    tengri.Uniform(0.0, 1.0),
                    "Tutorial demo amplitude",
                )
            ]

        def precompute(self, ssp_data=None, wave_grid=None):
            return None

        def apply(self, state, params):
            return state

    print(">>> @register_component   # appended to the SEDComponent registry")
    print()
    print("--- The new param now lives in the auto-derived param-map ---")
    from tengri.parameters.translate import _build_param_map

    m = _build_param_map(mean_sfh_type="dpl")
    if "demo_amp" in m:
        print(f"  demo_amp present in param-map: {m['demo_amp']}")
    else:
        print(
            "  (demo_amp not yet in param-map: only auto-injected once "
            "you build a Parameters with it)"
        )


_REGISTER_A_COMPONENT = _Tutorial(
    name="register_a_component",
    title="Register a Protocol-conforming SEDComponent: auto-derives translation",
    runner=_run_register_a_component,
    code=textwrap.dedent(
        """
        from dataclasses import dataclass, field
        from tengri.protocols.component import ParamDeclaration, SEDComponentConfig
        import tengri

        @dataclass(frozen=True)
        class MyConfig(SEDComponentConfig):
            name: str = "my_block"

        @tengri.register_component
        @dataclass(frozen=True)
        class MyAGNComponent:
            config: MyConfig = field(default_factory=MyConfig)
            name: str = "my_block"
            parameter_prefix: str = "my_block_"

            def declared_parameters(self):
                return [
                    ParamDeclaration("my_block_eddington",
                                     tengri.Uniform(-2.0, 0.5),
                                     "Eddington ratio log10(L/L_Edd)"),
                ]

            def precompute(self, ssp_data=None, wave_grid=None): ...
            def apply(self, state, params): ...

        # Now Parameters() accepts your new param without translate.py edits:
        spec = tengri.Parameters(mean_sfh_type="dpl",
                                 my_block_eddington=tengri.Uniform(-1, 0))
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 6: diagnose a bad fit
# ──────────────────────────────────────────────────────────────────


_DIAGNOSTICS = _Tutorial(
    name="diagnostics",
    title="Diagnose a fit: convergence, ESS, divergences",
    code=textwrap.dedent(
        """
        # After fitter.run("mcmc_nuts"):
        posterior.summary()                 # median ± 68% CI per param
        posterior.diagnostics_summary()     # ESS, R-hat, divergences, accept rate
        posterior.check_convergence()       # bool + per-param details

        # Per-param ESS: anything ≲ 100 is suspicious
        ess = posterior.effective_sample_size()
        rhat = posterior.rhat()             # should be ≲ 1.05 for all params

        # If divergences > 0:
        #   - your prior may be too tight (try wider Uniform)
        #   - parameters may be degenerate (try running with one fixed)
        #   - the sampler may need a smaller step size (n_warmup ≥ 1000)

        # Visual check:
        posterior.plot_corner(truths=truth_dict)   # parameter degeneracies
        posterior.plot_sed()                       # data vs model
        posterior.plot_sfh()                       # SFH(t) posterior band
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 7: population / hierarchical
# ──────────────────────────────────────────────────────────────────


_HIERARCHICAL = _Tutorial(
    name="hierarchical",
    title="Hierarchical / population fit: share priors across many galaxies",
    code=textwrap.dedent(
        """
        # Same model, N galaxies, per-galaxy free params + shared
        # population-level hyperparameters (e.g. PSD amplitude / timescale
        # for the IFT correlated-field SFH prior).

        import jax
        import tengri
        from tengri import PopulationFitter

        # What is shared is fixed by the model, not chosen per call: the PSD
        # amplitude and timescale of the correlated-field SFH prior. You supply
        # a factory that takes those two and returns a model built with them.
        def model_factory(psd_sigma, psd_tau_myr):
            spec = tengri.Parameters(
                mean_sfh_type=["dpl", "field"],
                sfh_field_psd_sigma=psd_sigma,
                sfh_field_psd_tau_myr=psd_tau_myr,
            )
            return tengri.SEDModel(spec, ssp_data, observation=obs)

        galaxies = [                           # one dict per galaxy
            {"flux_obs": g1_flux, "noise": g1_noise},
            {"flux_obs": g2_flux, "noise": g2_noise},
        ]

        pop = PopulationFitter(
            model_factory,
            galaxies,
            psd_sigma_prior=(0.1, 4.0),        # uniform prior on sigma_PSD
            psd_tau_prior=(1.0, 300.0),        # uniform prior on tau_PSD [Myr]
        )

        posterior = pop.run("vi", key=jax.random.PRNGKey(0))
        # → returns a PopulationPosterior with population + per-galaxy posteriors

        # See PopulationFitter docstring for the full hierarchical pattern.
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 8: design philosophy
# ──────────────────────────────────────────────────────────────────


_PHILOSOPHY = _Tutorial(
    name="philosophy",
    title="Design philosophy: why tengri is layered the way it is",
    code=textwrap.dedent(
        """
        ┌────────────────────────────────────────────────────────────────┐
        │   Parameters   →   SEDModel   →   Fitter   →   Posterior      │
        │   (priors)         (forward)      (loss+      (samples,        │
        │                    physics)       sampler)    diagnostics)     │
        └────────────────────────────────────────────────────────────────┘

        1.  Pure JAX, end to end.

            Every physics block (SFH, SPS, dust, AGN, IGM, radio, X-ray)
            is a pure function of (params, wavelength).  No side effects,
            no global state, no in-place mutation.  This buys us:

            • analytic gradients: every parameter gets ∂loss/∂param free
            • JIT compilation: the whole forward pass as one XLA graph
            • vmap: parallel evaluation of N galaxies for free
            • full sampler portfolio: MAP, NUTS, geoVI, ray-tracing,
              MCLMC, Pathfinder all share the same forward model

        2.  Standardized latent space.

            Every prior is absorbed into the forward model via differentiable
            transforms.  The sampler always sees ξ ~ N(0, I); the loss is
            always H(ξ) = ½·χ²(data, f(ξ)) + ½·ξᵀξ.  No special-case prior
            penalties, no per-distribution tricks.  This is what makes
            high-dimensional sampling tractable.

        3.  IFT correlated-field SFH (the headline science).

            log SFR(t) = IFFT(√P(σ, τ) · ξ),  ξ ~ N(0, I)

            • σ_burst (psd_sigma)        amplitude of SFR fluctuations [dex]
            • τ_burst (psd_tau_myr)      memory timescale [Myr]

            Different (σ, τ) capture stellar winds (~1-10 Myr), supernova
            feedback cycles (~20-50 Myr), or gas accretion (~100-300 Myr).
            Hierarchical fits share (σ, τ) across N galaxies.

        4.  Modular, swappable physics.

            Every block has a registry of alternatives: see:

                tengri.list_agn_models()        # AGN model families
                tengri.list_agn_blocks()        # composable AGN blocks
                tengri.list_dust_laws()         # attenuation curves
                tengri.list_dust_emission_models()  # IR emission templates
                tengri.list_sfh_models()        # SFH variants
                tengri.list_nebular_backends()  # nebular backends

            Counts live in tengri.summary(): this page does not repeat
            them, because a hand-written number goes stale the next time
            someone registers a model.

            Adding your own:  tengri.tutorial("register_a_model")

        5.  Inference is a separate concern from the model.

            One Fitter.run(method=) call swaps NUTS → geoVI → MCMC →
            ray-tracing without rebuilding the model.

                tengri.list_inference_methods(tier="primary")

            See:  tengri.tutorial("swap_inference")

        Further reading
        ───────────────
        • docs/dev/design_philosophy.md         architecture + IFT framework
        • docs/dev/NAMING_CONTRACT.md           naming conventions
        • docs/dev/docstring-standard.md        docstring tier rules
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 9: key classes
# ──────────────────────────────────────────────────────────────────


_KEY_CLASSES = _Tutorial(
    name="key_classes",
    title="Key classes: what each one does and how they compose",
    code=textwrap.dedent(
        """
        ┌── Parameters  ────────────────────────────────────────────────┐
        │ The PRIOR specification.  What's free, what's fixed, with what │
        │ distribution.  Built with **kwargs:                            │
        │                                                                │
        │   spec = tengri.Parameters(                                    │
        │       mean_sfh_type="dpl",       # structural choice            │
        │       redshift=tengri.Fixed(0.05),                              │
        │       sfh_dpl_alpha=tengri.Uniform(0.5, 3.0),  # free param     │
        │       ...                                                      │
        │   )                                                            │
        │   spec.summary()    # prints the table                         │
        │                                                                │
        │ Discovery:  tengri.suggest_parameters(mean_sfh_type="dpl")     │
        └────────────────────────────────────────────────────────────────┘

        ┌── Photometry / Spectroscopy / Observation  ──────────────────┐
        │ The OBSERVATION configuration: which bands, which              │
        │ wavelength grid.  Doesn't carry the fluxes; those go to Fitter. │
        │                                                                │
        │   phot = tengri.Photometry.from_names(["sdss_g","sdss_r",...])  │
        │   obs  = tengri.Observation(photometry=phot)                    │
        │                                                                │
        │ Discovery:  tengri.list_filters(survey="JWST")                  │
        └────────────────────────────────────────────────────────────────┘

        ┌── SEDModel  ─────────────────────────────────────────────────┐
        │ The FORWARD MODEL.  Wires Parameters + SSP + Observation into  │
        │ one JIT-compiled function: predict(params) → Prediction object.│
        │ Holds physics dispatch (which dust law, which SFH, AGN on/off).│
        │                                                                │
        │   model = tengri.SEDModel(spec, ssp_data, observation=obs)     │
        │   repr(model)                                                  │
        │   # SEDModel(sfh='dpl', dust='two_component', agn='off', ...)  │
        │                                                                │
        │ Methods: predict, predict_properties, predict_photometry, mock │
        └────────────────────────────────────────────────────────────────┘

        ┌── Fitter  ────────────────────────────────────────────────────┐
        │ The INFERENCE ENGINE.  Builds a loss from SEDModel + data +    │
        │ noise; dispatches to one of 19 backends.                       │
        │                                                                │
        │   fitter = tengri.Fitter(model, data=fluxes, noise=errors)    │
        │   posterior = fitter.run("nuts")                               │
        │   posterior = fitter.run("vi", init_from=map_result)           │
        │                                                                │
        │ Discovery:  tengri.list_inference_methods(tier="primary")     │
        └────────────────────────────────────────────────────────────────┘

        ┌── Posterior  ─────────────────────────────────────────────────┐
        │ The RESULT.  Samples (or MAP point), derived quantities (M*,   │
        │ SFR, sSFR), diagnostics (ESS, R-hat, divergences), persistence │
        │ (save / load / to_arviz), method chaining (refine).            │
        │                                                                │
        │   posterior.summary()                # parameter median ± 68%  │
        │   posterior.plot_corner(truths=...)  # parameter degeneracies  │
        │   posterior.plot_sed()               # data vs model band      │
        │   posterior.plot_sfh()               # SFH(t) posterior band   │
        │   posterior.properties["stellar_mass"]  # array (n_samples,)   │
        │   posterior.save("fit.h5")                                     │
        │   refined = posterior.refine("nuts") # method chaining          │
        └────────────────────────────────────────────────────────────────┘

        Further:  tengri.tutorial("philosophy")  for the layered design,
                  tengri.tutorial("first_fit")   for the end-to-end recipe.
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 10: common use cases
# ──────────────────────────────────────────────────────────────────


_USE_CASES = _Tutorial(
    name="use_cases",
    title="Common use cases: pick the recipe that matches your science",
    code=textwrap.dedent(
        """
        Use case 1  ──  ONE GALAXY, broadband photometry
        ─────────────────────────────────────────────────
        DPL SFH + two-component dust + Charlot-Fall attenuation.
        MAP gives a quick point estimate; NUTS costs more, gives the posterior.

            tengri.tutorial("first_fit")

        Use case 2  ──  ONE GALAXY, joint photometry + spectroscopy
        ───────────────────────────────────────────────────────────
        Same one-liner; declare both channels on the Observation, then pass
        each channel's arrays by name.

            obs = tengri.Observation(photometry=phot, spectroscopy=spec_data)
            model = tengri.SEDModel.build(ssp_data=ssp, observation=obs, ...)
            post = model.fit(photometry=(flux_p, err_p),
                             spectrum=(flux_s, err_s), method="mcmc_nuts")

        Use case 3  ──  CATALOG of N galaxies, independent fits
        ─────────────────────────────────────────────────────────
        One noun: Catalog. Table in, posteriors out. With redshift_col and a
        model built with approx=WavePrecomp(catalog_z_range=...), each row
        is fit at its own redshift on one shared LUT.

            cat = tengri.Catalog(fwd, table, flux_unit="mJy", redshift_col="z")
            post = cat.fit(method="map", key=jax.random.PRNGKey(0))
            post["stellar_mass"]                   # (N,) medians

        Use case 4  ──  HIERARCHICAL fit (e.g. shared SFH burstiness)
        ─────────────────────────────────────────────────────────────
        Per-galaxy free params + population-level shared hyperparameters.

            tengri.tutorial("hierarchical")

        Use case 5  ──  STOCHASTIC SFH (the IFT thesis)
        ────────────────────────────────────────────────
        DPL envelope + GP-correlated burstiness, governed by PSD prior.

            spec = tengri.Parameters(
                mean_sfh_type=["dpl", "field"],     # add the IFT field
                sfh_field_psd_sigma=tengri.LogUniform(0.05, 1.0),
                sfh_field_psd_tau_myr=tengri.LogUniform(10, 500),
                ...
            )
            posterior = model.fit(flux, err, method="vi")  # geoVI scales to high-D

        Use case 6  ──  MOCK RECOVERY (validation / Paper I)
        ─────────────────────────────────────────────────────
        Synthetic galaxy → fit it back → check truth in the posterior.

            truth = model.spec.sample(jax.random.PRNGKey(0))
            mock = tengri.generate_mock(model, truth, key=jax.random.PRNGKey(1), snr=20.0)
            posterior = model.fit(mock["flux_obs"], mock["noise"], method="mcmc_nuts")
            posterior.plot_corner(truths=mock["params"])

        Use case 7  ──  MODEL COMPARISON
        ─────────────────────────────────
        Same data, different physics; compare via Bayes factor / WAIC.

            for agn_model in (None, "skirtor", "kubota_done"):
                spec_i = tengri.Parameters(..., agn_model=agn_model)
                model_i = tengri.SEDModel(spec_i, ssp, observation=obs)
                results[agn_model] = model_i.fit(data, noise, method="nss")

        Use case 8  ──  CUSTOM PHYSICS
        ───────────────────────────────
        Add your own SED model alternative and have it appear in
        list_*() / describe() / search() automatically.

            tengri.tutorial("register_a_model")
            tengri.tutorial("register_a_component")
            tengri.tutorial("custom_likelihood")
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 11: mock recovery (validation pattern)
# ──────────────────────────────────────────────────────────────────


_MOCK_RECOVERY = _Tutorial(
    name="mock_recovery",
    title="Mock recovery: sanity-check a model on synthetic data",
    needs_ssp=True,
    code=textwrap.dedent(
        """
        # Standard validation: sample a true point, generate noisy mock photometry,
        # fit it back, confirm the truth lands inside the posterior.

        import jax
        import tengri

        # 1. Build the model exactly as you would for a real fit
        spec = tengri.Parameters(
            mean_sfh_type="dpl",
            redshift=tengri.Fixed(0.1),
            sfh_dpl_alpha=tengri.Uniform(0.5, 3.0),
            sfh_dpl_log_total_mass=tengri.Uniform(8, 12),
            dust_tau_diff=tengri.Uniform(0, 2),
            met_logzsol=tengri.Uniform(-1.5, 0.2),
        )
        obs = tengri.Observation(
            photometry=tengri.Photometry.from_names(
                tengri.list_filters(survey="SDSS").names()
                + tengri.list_filters(survey="2MASS").names()
            )
        )
        model = tengri.SEDModel(spec, ssp_data, observation=obs)

        # 2. Sample the truth point, then generate a mock at SNR=20.
        #    generate_mock returns flux_obs, noise, and params (the truth)
        truth = spec.sample(jax.random.PRNGKey(0))
        mock = tengri.generate_mock(model, truth, key=jax.random.PRNGKey(1), snr=20.0)

        # 3. Fit it back: MAP for speed, then NUTS for posterior
        fitter = tengri.Fitter(model, data=mock["flux_obs"], noise=mock["noise"])
        map_result = fitter.run("map", n_steps=300)
        posterior  = fitter.run("nuts", init_from=map_result, n_warmup=500)

        # 4. Check truth is inside the 68% credible interval
        posterior.summary()
        posterior.plot_corner(truths=truth)   # red markers = injected truth

        # 5. For Paper-I-style validation: repeat with N seeds, check coverage
        for seed in range(20):
            mock = tengri.generate_mock(model, truth, key=jax.random.PRNGKey(seed))
            ...
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 12: model comparison
# ──────────────────────────────────────────────────────────────────


_COMPARE_MODELS = _Tutorial(
    name="compare_models",
    title="Model comparison: same data, different physics, evidence-based ranking",
    code=textwrap.dedent(
        """
        # Same data, vary one structural choice (here: AGN model).
        # Compare via Bayesian evidence (nested sampling) or out-of-sample
        # WAIC / LOO if you have arviz.

        import tengri
        results = {}

        for agn_model in (None, "skirtor", "kubota_done", "cat3d_wind"):
            spec = tengri.Parameters(
                mean_sfh_type="dpl",
                agn_model=agn_model,                # the only thing that changes
                redshift=tengri.Fixed(0.1),
                sfh_dpl_alpha=tengri.Uniform(0.5, 3.0),
                # ... shared priors ...
            )
            model_i  = tengri.SEDModel(spec, ssp_data, observation=obs)
            fitter_i = tengri.Fitter(model_i, data=fluxes, noise=errors)
            results[agn_model] = fitter_i.run("nss")    # nested sampling → log Z

        # Bayes factors via log evidence
        import math
        baseline = results[None].log_evidence
        for name, post in results.items():
            log_BF = post.log_evidence - baseline
            print(f"  {name!r:25s}  log Z = {post.log_evidence:7.2f}  "
                  f"ΔlogBF vs no-AGN = {log_BF:+5.2f}")

        # Or via WAIC (if posterior samples available: needs arviz)
        for name, post in results.items():
            idata = post.to_arviz()
            print(f"  {name!r:25s}  WAIC = {idata.waic:.2f}")

        # See:
        #   tengri.describe('skirtor')  →  param details for each candidate
        #   tengri.list_agn_models()    →  the full menu
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 13: joint photometry + spectroscopy
# ──────────────────────────────────────────────────────────────────


_JOINT_PHOT_SPEC = _Tutorial(
    name="joint_phot_spec",
    title="Joint photometry + spectroscopy fit (with calibration marginalization)",
    needs_ssp=True,
    code=textwrap.dedent(
        """
        # Combine broadband photometry with a flux-calibrated spectrum.
        # The model fits both simultaneously; spectroscopic flux calibration
        # uncertainty is marginalized out via Chebyshev polynomials.

        import tengri
        import jax.numpy as jnp

        # 1. Photometry side (same as a phot-only fit)
        photometry = tengri.Photometry.from_names(
            tengri.list_filters(survey="SDSS").names()
            + tengri.list_filters(survey="2MASS").names()
        )

        # 2. Spectroscopy side: pass the spectral grid + LSF
        spectroscopy = tengri.Spectroscopy(
            wave_obs=spec_wavelengths,     # observed-frame [Angstrom]
            resolution=2000.0,             # resolving power R
        )

        # Masking is done on the *noise*, not declared on Spectroscopy:
        # masked pixels get noise = inf, which drops them from the likelihood.
        spec_errors = tengri.observation.apply_wavelength_mask(
            spec_errors, spec_wavelengths,
            mask_ranges=[(5560, 5590), (7580, 7680)],   # telluric A/B bands
        )

        # 3. Bundle into one Observation
        obs = tengri.Observation(photometry=photometry,
                                 spectroscopy=spectroscopy)

        # 4. Build the spec: emission lines marginalized analytically
        spec = tengri.Parameters(
            mean_sfh_type="dpl",
            redshift=tengri.Fixed(z_observed),
            nebular_backend="cue",        # ionization-parameter free
            eline_marginalize=True,       # closed-form line marginalization
        )
        model = tengri.SEDModel(spec, ssp_data, observation=obs)

        # 5. Fit: pass the *concatenated* (phot, spec) flux + noise
        joint_flux  = jnp.concatenate([phot_fluxes, spec_fluxes])
        joint_noise = jnp.concatenate([phot_errors, spec_errors])

        fitter = tengri.Fitter(
            model, data=joint_flux, noise=joint_noise,
            data_type="joint",
            calibration_marginalize=True,  # Chebyshev nuisance, Prospector-style
            cal_n_poly=3,
        )
        posterior = fitter.run("nuts")
        posterior.summary()
        posterior.plot_sed()               # photometry + spectrum, calibration applied
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 14: derived quantities / properties
# ──────────────────────────────────────────────────────────────────


_PROPERTIES = _Tutorial(
    name="properties",
    title="Derived quantities: the property catalog",
    code=textwrap.dedent(
        """
        # After a fit, every galaxy (synthetic or real) has the same
        # derived quantities: stellar mass, SFR, sSFR, age, emission
        # lines, SED shape indices, and more.  The catalog is topology
        # agnostic: same names on every topology, just with different
        # shapes.

        import jax
        import tengri

        # 1. DISCOVERY: see what properties are available
        tengri.list_properties()                  # all 49 properties
        tengri.list_properties(group="sfh")       # SFH group only
        tengri.describe_property("stellar_mass")  # full metadata

        # 2. SINGLE GALAXY: lazy access via model.predict()
        pred = model.predict(params)
        pred.stellar_mass              # attribute shorthand
        pred.properties["stellar_mass"]  # dict-like access
        pred.sfh.stellar_mass          # grouped (same value)
        pred.sed.dn4000                # SED shape quantities
        pred.lines.halpha              # emission line luminosity

        # 3. POSTERIOR: array of shape (n_samples,)
        posterior.properties["stellar_mass"]     # array
        lo, med, hi = posterior.properties.ci("stellar_mass")
        # → (lower 16%, median, upper 84%)

        posterior.properties.ci("stellar_mass", level=0.95)  # 95% CI

        # 4. BATCH / MOCK CATALOG: JIT-compatible vmap
        # Compute properties for many parameter sets at once (no fit needed)
        import jax
        params_batch = spec.sample_batch(jax.random.PRNGKey(0), n=10_000)

        # vmap-compatible; returns shape (10000,) arrays
        props = jax.vmap(model.predict_properties,
                        in_axes=(0, None),  # params_batch, names
                        )(params_batch, ("stellar_mass", "sfr_100myr"))
        props["stellar_mass"]  # shape (10_000,)

        # 5. FULL PROPERTY DICT: export names='all' or unspecified
        pred_dict = model.predict_properties(params)  # all 49
        pred_dict.to_dict(names=("stellar_mass", "dn4000"))  # select subset

        Notes
        ─────
        The topology rule: derived quantities follow the parameter topology.
        • Prediction (single galaxy): scalars
        • Posterior (MCMC/VI samples): shape (n_samples,)
        • Population / Catalog (many galaxies): shape (n_galaxies, ...)
        All use the same property names and discovery API.

        See:  tengri.tutorial("mock_catalog"): batch mock using vmap
              tengri.tutorial("approx_vs_exact"): exact vs precomputed paths
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 15: exact vs approximate photometry prediction
# ──────────────────────────────────────────────────────────────────


_APPROX_VS_EXACT = _Tutorial(
    name="approx_vs_exact",
    title="Exact by default, approximate by choice: photometry prediction paths",
    code=textwrap.dedent(
        """
        # Two prediction paths for photometry:
        #
        # EXACT: full SED integration (default)
        #   Always safe; valid for any filter and any redshift.
        #   Used in fitting.
        #
        # APPROX: lookup table (WavePrecomp)
        #   Precomputed at build time on the model's filters + z grid.
        #   Cheaper per call, but valid only for build-time filters + grid;
        #   arbitrary filters always use EXACT. Measure before relying on it.
        #   The keyword is `approx=` on both ends -- `approx=WavePrecomp(...)`
        #   at build, `approx=True` at call -- because it is one mechanism.
        #   Measured error: <0.1% at z~0, but 1.4% in sdss_g by z=3, and it is
        #   a bias, so it enters the gradient multiplied by SNR. See
        #   bench/reports/2026-08-17_wave_precomp_accuracy.md

        import jax
        import tengri

        # ── BUILD TIME ──

        # Default: uses exact photometry throughout
        model_exact = tengri.SEDModel.build(ssp_data=ssp, observation=obs, ...)

        # Opt into the approximate path: precompute a lookup table
        # (paid once at build time, saves at every later predict call)
        model_fast = tengri.SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            ...,
            approx=tengri.WavePrecomp(n_z=200, z_min=0.0, z_max=3.0),
        )

        # ── INFERENCE ──

        # Each model keeps its build-time choice. Fitting uses whatever
        # was chosen at build: no per-call control there.
        posterior_exact = fitter_exact.run("map")
        posterior_fast  = fitter_fast.run("map")

        # ── POST-FIT PREDICTION ──

        # Default: exact (always correct, never silently wrong)
        pred = model_exact.predict(params)
        photo_exact = pred.photometry()           # exact path

        # Opt into approx for post-fit batches on build-time filters
        photo_fast = pred.photometry(approx=True)

        # ── VMAP BATCH (MOCK CATALOG) ──

        # For many parameter sets on build-time filters, approx is right:
        params_batch = spec.sample_batch(jax.random.PRNGKey(0), n=10_000)
        fn_fast = jax.vmap(
            lambda p: model_fast.predict(p).photometry(approx=True)
        )
        batch_photos = fn_fast(params_batch)  # shape (10000, n_filters)

        Rules
        ─────
        1. EXACT is always safe; default in post-fit exploration.
        2. FAST requires:
           • build-time filters (model.observation.photometry.names)
           • wavelengths within the build-time z grid
        3. approx=True + arbitrary filters → ValueError
        4. Fitting always uses build-time choice (no override).
        5. approx=True is most useful for vmap'd batches on big n_samples.

        Why two paths?
        ───────────────
        A precomputed lookup table is an approximation (interpolant), not
        the true physics. The default (exact) must never silently change
        the result. approx=True is an explicit opt-in speed knob that users
        must think about: "Am I using build-time filters? Is this on the
        z grid?" If you answer no, the exact path is your only choice.

        See:  tengri.tutorial("mock_catalog"): batch examples
              tengri.list_properties()       : derived quantities
        """
    ).strip(),
)


# ──────────────────────────────────────────────────────────────────
# Recipe 16: mock catalog via vmap (no fit required)
# ──────────────────────────────────────────────────────────────────


_MOCK_CATALOG = _Tutorial(
    name="mock_catalog",
    title="Mock a catalog from your own parameters: vmap, no fit needed",
    code=textwrap.dedent(
        """
        # The SEDModel is a pure function: given params, compute observables.
        # Use jax.vmap to evaluate it over any parameter batch (no fit
        # or observed data required).  Useful for forecasting, validation,
        # ablation studies, or testing your model choices.

        import jax
        import jax.numpy as jnp
        import tengri

        # 1. BUILD THE MODEL (same as fitting)
        spec = tengri.Parameters(
            mean_sfh_type="dpl",
            redshift=tengri.Fixed(0.05),
            sfh_dpl_alpha=tengri.Uniform(0.5, 3.0),
            sfh_dpl_beta=tengri.Uniform(0.5, 3.0),
            sfh_dpl_tau_gyr=tengri.Uniform(0.5, 8.0),
            sfh_dpl_log_total_mass=tengri.Uniform(8, 12),
            dust_tau_diff=tengri.Uniform(0, 2),
            met_logzsol=tengri.Uniform(-1.5, 0.2),
        )
        obs = tengri.Observation(photometry=photometry)
        model = tengri.SEDModel(spec, ssp_data, observation=obs)

        # 2a. HAND-BUILT BATCH: dict of arrays
        params_batch = {
            "sfh_dpl_alpha": jnp.array([0.5, 1.0, 2.0]),      # shape (3,)
            "sfh_dpl_beta": jnp.array([1.0, 2.0, 3.0]),
            ...
            "redshift": jnp.array([0.05, 0.05, 0.05]),
        }

        # 2b. OR RANDOM BATCH: prior samples
        key = jax.random.PRNGKey(42)
        params_batch = spec.sample_batch(key, n=1_000)

        # 3. VMAP OVER PHOTOMETRY
        fn_phot = jax.vmap(
            lambda p: model.predict(p).photometry(approx=False)
        )
        batch_photos = fn_phot(params_batch)  # shape (n_params, n_filters)

        # 4. VMAP OVER PROPERTIES: jit-compatible subset
        fn_props = jax.vmap(
            lambda p: model.predict_properties(p)
        )
        batch_props = fn_props(params_batch)  # dict of shape (n_params,)
        batch_props["stellar_mass"]  # shape (1000,)

        # 5. VMAP OVER REST-FRAME SED (mock spectroscopy)
        fn_sed = jax.vmap(
            lambda p: model.predict(p).rest_sed()
        )
        batch_seds = fn_sed(params_batch)  # shape (n_params, n_wave)

        Statistics
        ───────────
        # Compute summary stats over your mock catalog
        stellar_mass_batch = batch_props["stellar_mass"]
        lo = jnp.percentile(stellar_mass_batch, 16)
        med = jnp.percentile(stellar_mass_batch, 50)
        hi = jnp.percentile(stellar_mass_batch, 84)
        print(f"M* = {med:.2e} [+{hi-med:.2e} -{med-lo:.2e}] Msun")

        Why vmap?
        ─────────
        • No Python loop overhead; pure JAX graph.
        • Automatic differentiation works through batches (if needed).
        • Scales to millions of parameter sets (GPU memory permitting).
        • Same binary as single-galaxy predict: one compile, many evals.

        Common uses
        ────────────
        • Forecast detection limits on a JWST catalog
        • Validate model physics (mock recovery without MCMC)
        • Ablation: "what if we exclude AGN?" (change agn_model, vmap)
        • Parametric study: "how does SFR depend on stellar mass?"
        • Generate training data for a neural network surrogate

        See:  tengri.tutorial("properties")   : derived quantities
              tengri.tutorial("approx_vs_exact"): exact vs precomputed paths
        """
    ).strip(),
)


# Master registry
_LEGACY_TUTORIAL_NAMES = {"fast_vs_exact": "approx_vs_exact"}
"""Renamed topics, kept resolvable for one release."""

_TUTORIALS: dict[str, _Tutorial] = {
    t.name: t
    for t in (
        _FIRST_FIT,
        _PHILOSOPHY,
        _KEY_CLASSES,
        _USE_CASES,
        _MOCK_RECOVERY,
        _COMPARE_MODELS,
        _JOINT_PHOT_SPEC,
        _REGISTER_A_MODEL,
        _REGISTER_A_COMPONENT,
        _SWAP_INFERENCE,
        _CUSTOM_LIKELIHOOD,
        _DIAGNOSTICS,
        _HIERARCHICAL,
        _PROPERTIES,
        _APPROX_VS_EXACT,
        _MOCK_CATALOG,
    )
}


def tutorial(name: str | None = None, *, run: bool = False) -> None:
    """Print (and optionally execute) a runnable tutorial recipe.

    Parameters
    ----------
    name: str, optional
        Tutorial name.  Pass without an argument to list all topics.
    run: bool
        If ``True`` and the tutorial supports live execution, run it
        after printing the recipe.  Tutorials that need an SSP file or
        external data print a "skipping live execution" note instead.

    Topics
    ------
    first_fit            Mock galaxy → posterior in 30 s
    register_a_model     Add a new AGN/dust/SFH alternative (live)
    register_a_component Protocol-conforming SEDComponent (live)
    swap_inference       Same model, three samplers
    custom_likelihood    Student-t, calibration floor, custom Protocol
    diagnostics          Convergence checking, ESS, divergences
    hierarchical         Population fit across many galaxies
    """
    if name is None:
        print("\ntengri.tutorial(name): copy-pasteable runnable recipes:\n")
        width = max(len(t.name) for t in _TUTORIALS.values())
        for t in _TUTORIALS.values():
            star = "  ●" if t.runner is not None else "   "
            print(f" {star} {t.name.ljust(width)}   {t.title}")
        print()
        print("  ● = supports live execution (run=True).  Others print only.\n")
        print("  Examples:")
        print("    tengri.tutorial('first_fit')             # print the recipe")
        print("    tengri.tutorial('register_a_model', run=True)  # actually execute")
        print()
        return

    if name in _LEGACY_TUTORIAL_NAMES:
        new = _LEGACY_TUTORIAL_NAMES[name]
        warnings.warn(
            f"tutorial('{name}') was renamed to tutorial('{new}') when the "
            f"runtime `fast=` keyword became `approx=`; the old name will be "
            f"removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        name = new

    if name not in _TUTORIALS:
        raise KeyError(f"Unknown tutorial '{name}'.  Run tengri.tutorial() for the menu.")

    t = _TUTORIALS[name]
    print()
    print(f"━━━  tengri.tutorial('{name}')  ━━━")
    print(f"     {t.title}")
    print("─" * 78)
    print()
    print(t.code)
    print()

    if run:
        if t.needs_ssp:
            print("─" * 78)
            print("  This tutorial needs an SSP file under data/ssp_*.h5.")
            print("  Copy the recipe above into a script / notebook and run it there.")
            print()
            return
        if t.runner is None:
            print("─" * 78)
            print("  This tutorial is print-only (the recipe needs your data / model).")
            print("  Copy it into a notebook to run.")
            print()
            return
        print("━━━  LIVE EXECUTION  ━━━")
        print()
        t.runner()
        print()


def explain(thing) -> None:
    """Print the architectural role of a built-in class or model.

    For built-in classes this prints the layered-architecture context
    (where the class fits in the Parameters → SEDModel → Fitter →
    Posterior pipeline) plus a pointer to the relevant tutorial. For
    registered names it delegates to :func:`tengri.describe`.

    Parameters
    ----------
    thing: type or str
        Either a tengri class (``tengri.Parameters``, ``tengri.SEDModel``,
        ``tengri.Fitter``, ``tengri.Posterior``, ``tengri.Observation``)
        or a registered name (``"skirtor"``, ``"calzetti"``, ``"dpl"``…).

    Examples
    --------
    >>> tengri.explain(tengri.SEDModel)
    >>> tengri.explain("skirtor")
    """
    # String → registry lookup
    if isinstance(thing, str):
        from tengri.registry import describe as _describe

        print(_describe(thing))
        return

    # Prefer ``__name__`` (for classes) but fall through to ``type(...)``
    # for instances: a user with ``model = SEDModel(...)`` should be
    # able to call ``tengri.explain(model)`` and get the SEDModel blurb.
    if isinstance(thing, type):
        name = thing.__name__
    else:
        name = type(thing).__name__
    blurbs: dict[str, str] = {
        "Parameters": (
            "PRIOR SPECIFICATION  (layer 3: user API)\n"
            "Built with **kwargs; structural choices (mean_sfh_type, "
            "agn_model, dust_emission, …) determine which params are legal.\n"
            "  Discovery:    tengri.suggest_parameters(mean_sfh_type='dpl')\n"
            "  Tutorial:     tengri.tutorial('key_classes')"
        ),
        "Photometry": (
            "OBSERVATION CONFIG  (filter set, *not* the fluxes)\n"
            "Constructed via Photometry.from_names([...]).  Fluxes go to Fitter.\n"
            "  Discovery:    tengri.list_filters(survey='SDSS')\n"
            "  Tutorial:     tengri.tutorial('first_fit')"
        ),
        "Spectroscopy": (
            "SPECTROSCOPIC OBSERVATION CONFIG\n"
            "Wraps wavelength + LSF + masking; pair with Photometry inside an Observation."
        ),
        "Observation": (
            "BUNDLE of Photometry / Spectroscopy / LineList passed to SEDModel.\n"
            "  obs = tengri.Observation(photometry=phot, spectroscopy=spec_data)"
        ),
        "SEDModel": (
            "FORWARD MODEL  (layer 2)\n"
            "Wires Parameters + SSP + Observation into one JIT-compiled function:\n"
            "  predict_photometry(params) → flux array.\n"
            "  Tutorial:     tengri.tutorial('key_classes')"
        ),
        "Fitter": (
            "INFERENCE ENGINE  (layer 3)\n"
            "Builds the loss from model + data + noise, dispatches to one of 19 samplers.\n"
            "  Discovery:    tengri.list_inference_methods(tier='primary')\n"
            "  Tutorial:     tengri.tutorial('swap_inference')"
        ),
        "PopulationFitter": (
            "HIERARCHICAL FITTER  (population-level)\n"
            "Fits N galaxies jointly with shared hyperparameters (e.g. SFH burstiness).\n"
            "  Tutorial:     tengri.tutorial('hierarchical')"
        ),
        "CatalogFitter": (
            "CATALOG FITTER  (vmap'd independent fits)\n"
            "Same model, N galaxies, no shared params: one compile, N posteriors."
        ),
        "Posterior": (
            "RESULT CONTAINER\n"
            "Samples + derived (M*, SFR, sSFR) + diagnostics + persistence.\n"
            "  Methods:      summary, save, load, to_arviz, refine, plot_corner\n"
            "  Tutorial:     tengri.tutorial('diagnostics')"
        ),
    }
    blurb = blurbs.get(name)
    if blurb is None:
        print(f"  {name}: no architectural blurb yet; try tengri.describe('{name}').")
        return
    print(f"\n{name}\n{'─' * len(name)}\n{blurb}\n")


def examples() -> None:
    """List every runnable example script under ``examples/``.

    Prints the script path, one-line description (from its module
    docstring), and the canonical command to run it.  Use this to
    discover what's already on disk before writing a new script from
    scratch.
    """
    import ast
    from pathlib import Path

    repo_root: Path | None = None
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / "examples").is_dir() and (p / "src" / "tengri").is_dir():
            repo_root = p
            break
    if repo_root is None:
        # Installed wheel: look near the package
        try:
            import tengri as _t  # type: ignore[import-not-found]

            for p in Path(_t.__file__).resolve().parents:
                if (p / "examples").is_dir():
                    repo_root = p
                    break
        except Exception:
            pass
    if repo_root is None or not (repo_root / "examples").is_dir():
        print("  No examples/ directory found.")
        return

    examples_dir = repo_root / "examples"
    rows: list[tuple[str, str]] = []
    for path in sorted(examples_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(repo_root))
        try:
            tree = ast.parse(path.read_text())
            doc = ast.get_docstring(tree) or ""
        except Exception:
            doc = ""
        first_line = (doc.strip().splitlines() or [""])[0][:90]
        rows.append((rel, first_line))

    if not rows:
        print("  No example scripts found in examples/.")
        return

    print(f"\ntengri runnable example scripts ({len(rows)} total):\n")
    width = max(len(p) for p, _ in rows)
    for path, doc in rows:
        print(f"  {path.ljust(width)}  {doc}")
    print()
    print("  Run any of them with:")
    print(f"    cd {repo_root}")
    print("    .venv/bin/python <path>")
    print()
