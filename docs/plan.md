# Plan: Differentiable coupled predictability and targeted observations

## Context
The goal is a perfect-model study of how predictable the coupled ocean–atmosphere system is at monthly to interannual leads. Targets are ENSO, PDO, PNA and intraseasonal oscillations (MJO/BSISO), and possibly marine heat waves (MHWs) and atmospheric rivers (ARs). The tools are a differentiable coupled Earth system model (NumericalEarth + SpeedyWeather + Oceananigans), differentiated with Reactant/Enzyme.

Two outcomes:
0. **Modular phenomenon framework.** Choose and filter targets (ENSO, PDO, NAM/SAM, PNA, seasonal regional outcomes, MJO, MHWs, ARs), then measure skill for each one. Coarse, well-resolved modes come first. How well the model resolves finer phenomena (e.g. the MJO), and how much that affects the coarse modes, is itself a result.
1. **Strongly coupled covariance.** Build the nature-run covariance C(τ) and the forecast covariance Σ(τ) with the full ocean–atmosphere cross-covariance. Derive modes of variability from them, and compare them with the dynamical modes of the propagator. (Notation: `docs/technical.md`, Notation.)
2. **Idealized targeting system.** Given the current state and an index J (Niño-3.4, PDO, …) at lead τ, predict where and in which season extra observations would most reduce the forecast variance of J. Validate the predictions with OSSEs.

The directory currently holds only `test_NumericalEarth.jl`, which has never run (SpeedyWeather and NumericalEarth are not installed). Hardware: M3 Pro with 36 GB, no CUDA. Develop on the laptop CPU at coarse resolution, and write code that can move to a GPU cluster through Reactant.

### What I found about the packages (as of Nov 2026)
- **NumericalEarth v0.9.3:** Coupled models are built with `EarthSystemModel(; atmosphere, ocean, sea_ice)`. The SpeedyWeather extension needs both `SpeedyWeather` (0.22–0.24) and `ConservativeRegridding` (0.2), and requires Oceananigans 0.117.x. `examples/global_climate_simulation.jl` and `test/test_speedy_coupling.jl` are working coupled templates.
- **AD:** No one has differentiated the coupled Speedy+Oceananigans model yet.
  - SpeedyWeather v0.24 has experimental Reactant gradients for `PrimitiveWetModel` (`SpectralGrid(architecture=ReactantDevice())`, `time_steps!(sim, n; checkpointing=Reactant.Binomial(...))`).
  - Oceananigans has gradient-tested Reactant support with `ExplicitFreeSurface` (`test/reactant/simulation_gradients.jl`). CATKE compiles slowly, and the FFT implicit free surface is unsupported.
  - NumericalEarth's gradient pattern to copy is in `examples/era5_forced_slab_land.jl`: `Enzyme.autodiff` over `@trace mincut=true checkpointing=true`.
- **Blocker:** The Speedy↔ocean exchanger builds ConservativeRegridding regridders on CPU only and rejects non-CPU architectures. Making it work under ReactantState is the core new engineering.
- **Versions checked 2026-11-15:**
  - **Target set:** Reactant 0.2.308, Enzyme 0.13.219, Oceananigans 0.117.2, NumericalEarth 0.9.3, SpeedyWeather 0.24.1, ConservativeRegridding 0.2.21, KrylovKit 0.10.4 and Checkpointing 0.13.0. These should resolve together.
  - **Oceananigans 0.118 is held back:** NumericalEarth PR #935 is open, and ClimaSeaIce caps Oceananigans at 0.117. We want 0.118 because it adds a device-valued `stop_iteration`, so one compiled gradient handles any number of steps. Upgrade when both caps lift.
  - **PRs to watch:**
    - NumericalEarth #845, a WIP coupled adjoint with Reactant (land, ocean and Breeze atmosphere) — the closest precedent;
    - NumericalEarth #882 and #892, Speedy coupling;
    - SpeedyWeather 0.25 (#1402, breaking).
- **Julia:** Pin **1.11.9**. Switch when any of these happens:
  - NumericalEarth's Reactant CI moves to 1.12 or later;
  - SpeedyWeather #1051 closes (plain Enzyme on 1.12);
  - we drop plain-Enzyme cross-checks.

  If we go Reactant-only, 1.12.x is reasonable, and 1.13.4 is plausible: the #60924 miscompile fix is in 1.13.4, and Oceananigans runs its AD tests on 1.13. NumericalEarth's Reactant CI runs on it, and it is the only version where plain Enzyme can cross-check Reactant gradients for SpeedyWeather. Julia 1.13 has a known miscompile.

## Key scientific design choice
At leads beyond about 2 weeks, the atmospheric tangent-linear model loses validity: chaos makes gradients explode. Two complementary tools follow from this:
- **Ensemble statistics** are the backbone at long leads: the forecast covariance Σ(τ), and ensemble sensitivity (Ancell & Hakim). They are robust and need no AD.
- **AD (adjoint/JVP)** has three roles:
  - exact sensitivities and singular vectors at leads where M is valid (days to about 1 month for the atmosphere; months for ocean-initialized perturbations);
  - **ensemble-averaged adjoints** at longer leads: averaging ∇J over ensemble members filters out the chaotic atmospheric growth;
  - chaining short tangent-linear windows.

Each AD estimate is cross-checked against its ensemble counterpart. Where they disagree, that is itself a result.

## Approach (phases)

### 0. Environment (first step, as requested)
- `juliaup add 1.11`, then `juliaup override set 1.11` in this directory.
- `julia --project=. -e 'using Pkg; Pkg.add([...])'` to create `Project.toml` and `Manifest.toml` with: NumericalEarth (0.9.3), Oceananigans (0.117.x), SpeedyWeather (0.24), ConservativeRegridding (0.2), Reactant, Enzyme, KrylovKit, LinearAlgebra, SparseArrays, Statistics, NCDatasets, PythonPlot, IJulia. Set `[compat]` bounds to the resolved versions.
- Update `test_NumericalEarth.jl` to use `EarthSystemModel`, and run a 5-day coupled smoke test on CPU.

### 1. Package skeleton: `PredictableOcean` (src/ + experiments/ + test/ + docs/)
- `src/model.jl`: `build_coupled_model(; resolution=:coarse, arch, physics=:default)`. Wraps `atmosphere_simulation` and `ocean_simulation` from NumericalEarth.
  - Coarse default: T31L8 Speedy, coupled to a 20-level Oceananigans ocean.
  - The ocean uses `ExplicitFreeSurface` (or split-explicit for forward-only runs) and a cheap closure, not CATKE, for AD.
  - **The ocean grid is refined in the equatorial waveguide.** It uses a `LatitudeLongitudeGrid` with stretched latitude spacing: about 0.5° or finer within ±10°, relaxing to 2° poleward. Refinement is a parameter, so it can be swept.
- `src/io.jl`: all output (nature run, ensembles, covariance factors, modes, gradients, singular vectors, targeting maps, OSSE results) goes to **NetCDF via NCDatasets**, with CF-style metadata. Each file records the configuration and the package versions from the Manifest, for provenance.
- `src/state.jl`: `CoupledState`, with `flatten`/`unflatten` between model fields and a vector `x`.
  - Atmosphere: spectral vor, div, T, q, ln ps. Ocean: T, S, u, v, η.
  - Metrics: total-energy norm (atmosphere plus ocean heat content and kinetic energy), area and volume weights, and masks.
- `src/phenomena/`: a **modular registry of phenomena**. Every analysis in phases 3–7 is written against a generic `Phenomenon` interface, so adding a new target is just a new definition.
  - **Each `Phenomenon` holds:**
    - the variables and region it needs;
    - a filter: time averaging or band-pass (e.g. seasonal means, 20–100-day band), spatial smoothing, and season or start-month selection;
    - a differentiable index functional `J(x)` for adjoints and singular vectors, plus a pattern for projecting covariances and ensembles;
    - metrics: ensemble skill (anomaly correlation, RMSE against spread, signal-to-noise, predictable component) and whether the nature run resolves the phenomenon.
  - **Patterns are learned from the model's own nature run** (EOF or regression patterns), not from observations. That keeps the perfect-model study self-consistent. A comparison with observed patterns is reported as a diagnostic.
  - **Tier 1, coarse modes likely to be resolved (do first):** ENSO (Niño-3.4, plus a coupled SST/thermocline index), PDO, NAM/AO, SAM, PNA, seasonal-mean surface temperature and precipitation in user-chosen regions (seasonal forecast outcomes).
  - **Tier 2, marginally resolved:** MJO/BSISO (RMM-like index from OLR/U850/U200 EOFs), NAO and blocking proxies.
  - **Tier 3, local and extreme:** MHW box SST, AR proxy (mean IVT over a box).
  - **Cross-tier result:** the registry makes it easy to ask how poorly resolved Tier 2–3 phenomena affect Tier 1 predictability, e.g. MJO influence on ENSO onset or on the PNA. Approaches: condition on the model's MJO activity, and compute Tier 1 sensitivities projected onto the Tier 2 patterns.
  - A user selects targets with a config. `run_analysis(phenomena=[:enso, :sam], leads=1:12, start_months=[2, 5, 8, 11])` dispatches to the covariance, predictability, targeting and OSSE modules.

### 2. Nature run and climatology
- Spin up the coupled model, then run decades at coarse resolution. Store snapshots monthly, plus daily for the atmosphere.
- **Gate, using the registry's `resolved` diagnostic:** Tier 1 phenomena must be present (realistic spectrum, pattern and seasonality) before predictability work starts. Tier 2–3 phenomena are assessed in the same way. A weak or absent MJO, AR or MHW is recorded as a finding, not a blocker.
- `experiments/01_nature_run.jl`, `02_variability_diagnostics.jl`.

### 3. Strongly coupled covariance and modes (outcome 1)
- `src/covariance.jl`: low-rank covariances X Xᵀ/(N−1), with ensemble or anomaly matrices stored as factors (never as the dense matrix).
  - **Nature-run covariance C(τ; month)**, seasonally varying, from nature-run anomalies.
  - **Forecast covariance Σ(τ; t)** from an ensemble of perturbed starts.
  - Optional ocean–atmosphere localization through a Schur product applied by matrix-free products. Cross-blocks are kept in full: strong coupling is the point.
- **Modes:**
  - coupled EOFs (leading eigenvectors of C(0) in the energy norm);
  - a linear inverse model (LIM) and principal oscillation patterns (POPs) from lag covariances C(τ)C(0)⁻¹;
  - singular vectors of M′(τ) weighted by Σ(0) (phase 5).
  - Compare the three: are C(0)'s ENSO and PDO modes the same as the optimal-growth modes?
- `experiments/03_coupled_covariance.jl`.

### 4. Differentiable propagator M(τ) (engineering core)
Work up in this order, with gradient tests at every step:
1. Atmosphere alone (Reactant, `PrimitiveWetModel`, prescribed SST).
2. Ocean alone (Reactant, prescribed fluxes).
3. **Port the exchanger using precomputed weights** (`src/regridding.jl`, `src/reactant_coupling.jl`):
   - For each pair of atmosphere and ocean grids, compute the ConservativeRegridding weights once on CPU, in both directions. Save them to NetCDF as a COO sparse matrix with grid hashes, and reload them instead of recomputing.
   - At compile time, pass the weights to the Reactant-compiled step as **constant sparse or banded operators**. Regridding is then a fixed linear map, which is trivially differentiable: the adjoint is the transpose. That removes all regridder construction, host loops and `@warn` from the traced `time_step!`.
   - Check that the weights conserve flux (row and column sums against cell areas) and that the transpose gives the adjoint.
   - Upstream the result to NumericalEarth if possible.
4. Coupled model: `jvp(x, δx; τ)` and `vjp(x, z; τ)` functions, compiled once with `Reactant.Binomial` checkpointing and the step count as a device number.
- Check against plain Enzyme on 1.11 for small cases.

### 4b. Physics trade-off between differentiability and nonlinear realism (science and engineering target)
SpeedyWeather's convection and large-scale condensation are simple threshold schemes. Their switches give non-smooth or NaN gradients, but they also produce the observed nonlinear behaviour (rectification, intermittency, ENSO asymmetry).
- `physics=` selects among options:
  - `:default`: the original schemes, forward-only;
  - `:smoothed`: softplus or sigmoid versions of the thresholds, with a sharpness parameter;
  - `:dry`: dynamics only.
- **Forward realism:** for each option, measure ENSO amplitude and asymmetry, the convectively coupled wave spectrum and the MJO, and precipitation climatology, using the same registry diagnostics as the nature run.
- **Gradient quality:** finite-difference agreement, how long the tangent-linear model stays valid, and gradient noise.
- **Hybrid option:** run the nature run with `:default` physics, and compute the tangent-linear and adjoint models with `:smoothed`. This is the usual approach for a simplified adjoint. Quantify the resulting error in ΔVar.

### 4c. Equatorial waveguide resolution (science and engineering target)
- Sweep the ocean's equatorial refinement, and the atmosphere truncation (T31 → T63), against:
  - equatorial Kelvin and Rossby waves and tropical instability waves;
  - ENSO period and amplitude;
  - the coupled singular vectors, and how fast they become stable;
  - compile time, memory and gradient cost.
- The output is a recommended resolution for the differentiable coupled model, documented as a result.

### 5. Predictability analysis
`src/predictability.jl`, `experiments/05_predictability.jl`. The analysis is generic over registered phenomena; run Tier 1 first, then Tiers 2–3. For each phenomenon J, lead τ (from weeks for NAM/SAM to 1–24 months for ENSO/PDO) and start month:
- Gradient g = M′(τ)ᵀ∇J, both single-member and ensemble-averaged.
- Linear forecast variance of J, gᵀ Σ(0) g, compared with ensemble spread.
- Σ(0)-weighted singular vectors using KrylovKit `svdsolve`/`eigsolve` on Σ(0)^{1/2} M′ᵀ W M′ Σ(0)^{1/2}.
- Predictable component and signal-to-noise as a function of lead and season, including the spring predictability barrier.

### 6. Targeted observations (outcome 2)
`src/targeting.jl`, `experiments/06_targeting.jl`. For a candidate observation set H with error R, the reduction in forecast variance of J is

ΔVar(J) = (Hb)ᵀ (H Σ(0) Hᵀ + R)⁻¹ (Hb),  where b = Σ(0) g.

- This needs one adjoint per (J, τ, start date). Every candidate observation is then cheap to evaluate.
- Candidate platforms: Argo-like T/S profile, TAO-like mooring, satellite SST box, radiosonde or dropsonde.
- Greedy selection of observation sets (serial ETKF-style update of Σ(0)).
- Outputs: maps of ΔVar by region and season, e.g. "deploy in the western Pacific in boreal spring to reduce variance of the December Niño-3.4 forecast". There is also an ensemble-only version (ensemble sensitivity / ETKF signal variance) to cross-check.

### 7. OSSE validation
`src/assimilation.jl`, `experiments/07_osse.jl`.
- Draw synthetic observations from the nature run under a baseline network, plus or minus the targeted additions.
- Assimilate with a strongly coupled EnKF/ETKF (the coupled ensemble covariance Σ from phase 3), then run ensemble forecasts.
- Check that the actual reduction in J's forecast error matches the predicted ΔVar, and that targeted observations beat random placement of the same number.
- 4D-Var using the Hessian is a later option.

## Risks
- Compile time and memory for months-long gradients on a 36 GB CPU. Mitigate with coarse resolution, Binomial checkpointing, and a cluster later.
- Moist physics and convection may give non-smooth or NaN gradients. Phase 4b handles this explicitly through the smoothed, dry and hybrid physics options.
- Stretched ocean grids under Reactant may hit untested code paths. Test them early with a small ocean-only gradient test.
- Coarse resolution limits MJO, MHWs and ARs. Treat these as stretch targets, possibly with a higher-resolution atmosphere for short leads.
- The exchanger port may need upstream changes in NumericalEarth.

## Verification
- `test/runtests.jl`:
  - `flatten`/`unflatten` round-trip;
  - adjoint dot-product test ⟨M δx, λ⟩ = ⟨δx, Mᵀ λ⟩;
  - JVP and VJP against centered finite differences over a few steps;
  - Reactant gradients against plain Enzyme;
  - ΔVar formula against a brute-force Kalman update on a small toy (e.g. a Lorenz-96 coupled to a slab ocean) where everything can be checked exactly.
- Regridding weights: conservation checks, and confirm the compiled coupled step matches the CPU NumericalEarth step to round-off.
- Experiment scripts write NetCDF. You run them and look at the plots; key numbers go in rerunnable script cells.

## Development workflow (decided 2026-11-15)
Reuse the patterns from `~/projects/monsoon-breeze` (heavy code in a precompiled module, run by driver scripts with a settings block, a Slurm job that takes a run directory) and from `~/projects/ekman-astz` (a `plot/` environment, reader and plotting libraries, notebooks).

### Layout
```
predictable-ocean/
  CLAUDE.md                project rules and who-reads-code; points to notes/STATUS.md and plots/CLAUDE.md
  Project.toml             driver environment: PredictableOcean (path source), Reactant, Enzyme, …
  PredictableOcean/     package: Project.toml with compat bounds, src/, test/; PrecompileTools workload
    src/toy/               ToyCoupled (Lorenz-96 + slab/recharge ocean), implements AbstractPropagator
    src/coupled/           model.jl, state.jl, regridding.jl, reactant_coupling.jl, physics options
    src/phenomena/         Phenomenon registry
    src/covariance.jl  predictability.jl  targeting.jl  assimilation.jl  io.jl (NetCDF)
  drivers/                 driver scripts, each with a `##### Settings` block (the whole run configuration)
  run/<name>/              a copy of the driver plus its output; git tracks only the driver copy
  plots/                   own environment (no model packages), CLAUDE.md, readers.jl, plotlib.jl, notebooks
  docs/                    plan.md, objectives.md, technical.md, results.md, tutorial_enzyme_reactant.md
  notes/STATUS.md          current state and next steps; read first when picking up work
  helpers/                 precompile_job.sh, run_job.sh (Slurm), check_julia.sh, graph_package.sh
  .claude/agents/          code-reader.md, upstream-scout.md
  .claude/skills/          julia-env, differentiable-code, add-phenomenon, netcdf-output
```

### Repository
- Repo `github.com/deszoeke/differentiable-earth`, cloned to `~/projects/differentiable-earth`. It has one commit (LICENSE); `test_NumericalEarth.jl` is untracked. Renamed to `predictable-ocean` (repo and clones) on 2026-11-18, after the package became `PredictableOcean`.
- HPC path: `/ceoas/deszoeks/projects/predictable-ocean`.

### Julia 1.11 pin (decided). Write up in `docs/julia_pin.md`
- **Mechanism:** `juliaup override set 1.11` in the repo (Mac), `julia +1.11` in every job script, and `julia = "~1.11"` in `[compat]` for the package and the driver environment. **Commit `Manifest.toml`** for the main environment, so the laptop and the HPC run the same Reactant/Enzyme versions. `plots/` stays unpinned, with an untracked Manifest, and gets its own Jupyter kernel.
- **Rationale:** plain-Enzyme cross-checks of Reactant gradients for SpeedyWeather work only on ≤1.11 (SpeedyWeather #1051), and NumericalEarth's Reactant CI runs on 1.11.9.
- **Cost:** a separate Julia version and depot cache from the user's other 1.13 projects, slower precompiles, and a dedicated Jupyter kernel.
- **Criteria to unpin** (the `upstream-scout` checks these at each phase start; `docs/julia_pin.md` keeps a dated log):
  1. SpeedyWeather #1051 is closed, or its Enzyme CI includes 1.12+.
  2. NumericalEarth's Reactant CI runs on 1.12+.
  3. We stop needing plain-Enzyme cross-checks, i.e. the Reactant gradients have been validated on the coupled model.
  4. A package we need drops 1.11 or requires 1.12+.

  When we unpin: bump `[compat]`, re-resolve, rerun the full gradient test suite, and record the result.

### References
- **DJ4Earth paper:** Moses et al. 2026, *JAMES*, doi:10.1029/2025ms005615. It shows Enzyme.jl + Reactant.jl with checkpointing giving correct, efficient gradients for ShallowWaters.jl, Oceananigans, DJUICE and SpeedyWeather, each separately and not coupled. Its repo, `github.com/DJ4Earth/differentiable-esm-components-2025`, has the reproduction scripts; `upstream-scout` summarizes what can be reused. Cite it in `docs/technical.md`.
- **DJ4Earth repo contents, checked 2026-11-15:**
  - **Channelanigans** (submodule): Oceananigans channel adjoint under Reactant, with `SplitExplicitFreeSurface` + WENO, `@trace mincut=true checkpointing=true`, and `@compile raise_first=true raise=true sync=true`. It takes gradients with respect to initial T/S and forcing. **This is the best template for the ocean adjoint.** It also shows that split-explicit + WENO differentiates, which Oceananigans' own tests don't cover. It is unpinned, so we must pin it ourselves.
  - **SpeedyWeather `checkpointed_sensitivity.jl`:** plain Enzyme + Checkpointing.jl Revolve at T32L8, written against Speedy 0.17's API, so it's stale. It is upstreamed as Speedy PR #876, which makes it a pattern for the plain-Enzyme cross-checks.
  - No coupled model, singular vectors or Krylov code.
- **Related work:** Gelbrecht et al. 2023 *GMD* (the case for differentiable ESMs); JCM v1.1 (JAX SPEEDY physics, 2026); FESOM2-JAX (a differentiable ocean–sea-ice model on GH200).

### Running
- **Laptop:** the driver runs `julia +1.11 --project=. drivers/x.jl` from its run directory.
- **HPC:** `sbatch helpers/run_job.sh run/<name>`, using the user-level `arch_env.sh` with `julia +1.11`. Partitions: CPU (`ceoas`), A100, and GH200 (aarch64). The HPC port comes later.
  - **Reactant GPU support, checked 2026-11-15:** Reactant_jll ships CUDA 12.9 and 13.1 builds for both aarch64 and x86_64, and Reactant is used on GH200 (open issues #2632 on memory limits and #1812 on unified memory). There is no official GH200 CI, so treat GH200 as working but less tested. A100 works.
  - **Driver:** Reactant needs a CUDA 12.x or newer driver on the nodes; check `nvidia-smi` when we port.
  - **Login nodes** have no `libcuda`, so set `[preferences.Reactant_jll] gpu = "cuda"` to stop the CPU build being chosen there.
  - **TF32:** XLA uses TF32 for Float32 matrix products by default. Use Float64, or turn TF32 off, for gradient checks.
  - **First target: A100** (x86_64, better tested). Then try GH200 for memory, at 96 GB or more.
- **JIT:** recompiling after a change is acceptable. Reactant `@compile` happens at run time inside the driver, and the compiled thunk is reused within a run.
- **Git:** GitHub repo, created by the user. Push and pull between the laptop and the HPC. Commit and push only when asked. Work on `main`; tag phase milestones; branch only for risky experiments (changed from one branch per phase on 2026-11-16).

### plots/ extension (improving on ekman-astz `plot/`)
- **Library: CairoMakie + GeoMakie (decided).** Every package in the stack plots through Makie, with no PythonPlot support:
  - `OceananigansMakieExt` (Field/FieldTimeSeries recipes);
  - SpeedyWeather `RingGridsMakieExt` (`heatmap(::AbstractField)`) and `RingGridsGeoMakieExt` (`globe`);
  - NumericalEarth's Makie helpers;
  - CairoMakie in all the examples, including Breeze's.

  So we get native plots of model fields for free. The plotting functions follow the user's PythonPlot habits where helpful: panel specs, rasterized maps, and returned handles.
- **`plots/readers.jl`:** reads the project's NetCDF conventions and returns plain arrays with their coordinates and metadata. It does no computation that hides physics.
- **`plots/plotlib.jl`:** **user-configurable plotting functions**. Each has keyword defaults and returns its handles (`(; fig, axs, artists, cbars)`) so notebook cells can restyle afterwards.
  - Generic building blocks: a panel-spec pattern like ekman `maps.jl` (`panel(x; kind=:map|:line|:hovmoller|:lead_season, …)`, `plot_panels`).
  - Standard figures dispatched on result types: skill against lead, lead–season maps, mode patterns, singular vectors, targeting ΔVar maps, OSSE verification.
- **`plots/CLAUDE.md`:** how to add a configurable plotting function (signature conventions, docstrings, dispatch rather than branching on keyword values, returning handles), and the scope of a separate plotting session.
- **Notebooks:** they call these functions and run on the Mac or on the HPC Jupyter service. Numbers that matter come from saved code, never from reading a plot.

### Improvements over the earlier projects, built in from the start
- **NetCDF provenance attributes:** git commit and dirty flag, the settings block, Julia and package versions, host and architecture.
- **Notebooks paired with `.jl` scripts using Jupytext** (percent format). The `.jl` file is committed and outputs are stripped.
- **A `PreToolUse` hook** that blocks Read/Grep on `*.ipynb`.
- **`/fewer-permission-prompts`** run for a read-only allowlist.
- **GitHub Actions CI** running the toy-model tests and the regridding checks on CPU.
- **Short CLAUDE.md files:** status goes in `notes/STATUS.md`, API details in docstrings.
- **Back-porting** these to ekman-astz and monsoon-breeze is a separate, later task, if wanted.

### Rules (for CLAUDE.md)
- Re-read files before every edit, and make small targeted edits.
- I run unit tests and smoke tests under about 2 minutes. You run the experiments. Don't re-run to verify unless asked.
- Results come from saved, rerunnable code, never from scratch scripts.
- **Explain first:** before implementing each Enzyme or Reactant piece, explain the approach in chat and in `docs/technical.md`. Ask questions rather than assume.
- **Code standards:** type-stable functions, no globals, dispatch over keyword branching, docstrings on the API, Blue style, Aqua/JET in the tests.
- **Reactant-safe traced code:** no host control flow, scalar indexing or logging; loops written with `@trace`; constants (such as the regridding weights) passed in.
- Output is NetCDF, and docs and `notes/STATUS.md` are updated when a phase or a run finishes.

### Who reads code
- File and line already known: read them directly.
- Otherwise send it to `code-reader`, which is **Haiku** and read-only. It covers this project and the installed package sources (NumericalEarth and its ext, Oceananigans and its Reactant/Enzyme ext, SpeedyWeather, Reactant). For hard questions (MLIR/Reactant internals, adjoint bugs) call it with `model: "sonnet"`.
- It can optionally query graphify graphs of the package sources, built with `helpers/graph_package.sh` as in monsoon-breeze.
- Judgment calls, planning and editing stay with the main session.

### Order of work
1. **Toy model first.** Build the analysis code on the toy: covariance, singular vectors, targeting, OSSE. Write the Enzyme/Reactant tutorial on the toy at the same time (`docs/tutorial_enzyme_reactant.md` plus `drivers/tutorial_ad.jl`).
2. Alongside step 1: the Julia 1.11 environment and a coupled-model smoke test.
3. The coupled model then becomes a second `AbstractPropagator`.

### Toy model and the per-phase loop
- `AbstractPropagator` defines `step`, `jvp`, `vjp` and `state`. `ToyCoupled`, a Lorenz-96 atmosphere coupled to slab or recharge-oscillator ocean, implements it in a few hundred lines. Build and test all of phases 3, 5, 6 and 7 against the toy, with exact answers. The expensive coupled model (phase 4) is then just a second implementation of the same interface.
- **Loop for each phase:**
  1. Write a short phase plan.
  2. The code-reader agent surveys the upstream APIs.
  3. Build small increments, each with tests.
  4. You run the experiments.
  5. Update `docs/results.md`.
  6. Run `/code-review`.
  7. Commit when you ask.
- **Code standards:**
  - type-stable functions, no globals, dispatch on `Phenomenon` and propagator types, docstrings on exported API;
  - JuliaFormatter in Blue style;
  - Aqua.jl, and JET.jl for the core, in tests;
  - Reactant-safe traced code: no host control flow or scalar indexing, no logging, and loops written with `@trace`.
- **Agents (decided):**
  - `code-reader`: Haiku by default, read-only (Read, Grep, Glob, read-only Bash). It greps and analyzes this project and `~/.julia/packages` sources, and reports `file:line` findings. It is escalated to Sonnet for hard problems, e.g. Reactant/MLIR internals or adjoint bugs. That is a per-call `model: "sonnet"` override, or a `code-reader-deep` variant.
  - `upstream-scout`: read-only, using web and `gh`. Checks builds, CI, releases and issues for Reactant, Enzyme, Oceananigans, SpeedyWeather and NumericalEarth before each phase and before any upgrade.
- **Teaching mode:** the user is new to Enzyme and Reactant. Before implementing each AD piece, explain the approach (what is traced or compiled, the activity annotations, checkpointing, why a test proves correctness) in `docs/technical.md` and in chat, and ask questions rather than assume.
- **Julia version:** start on 1.11.9 for now. Keep the code version-agnostic, with no 1.11-specific workarounds. Re-evaluate each time `upstream-scout` reports a relevant change, e.g. full Reactant coverage of the coupled model or plain-Enzyme support on 1.12+.
- **Skills (decided):** `julia-env`, `differentiable-code`, `add-phenomenon`, `netcdf-output`.

## Documentation (`docs/`, Markdown, updated as each phase finishes)
- `docs/objectives.md`: science questions, the perfect-model framing, the phenomenon tiers, and the targeting goal.
- `docs/technical.md`: environment and versions, model configurations, the state vector and norms, precomputed regridding, Reactant/AD setup and checkpointing, the physics options, the NetCDF file conventions, and how to add a new `Phenomenon`.
- `docs/results.md` (or `docs/results/<phase>.md`): scientific results for each phase. This includes:
  - which phenomena are resolved;
  - coupled modes of C;
  - skill and predictability by phenomenon, lead and season;
  - the physics and resolution trade-offs;
  - targeting maps and OSSE validation.
  
  Each result links to the NetCDF file and script that produced it.
- `README.md`: a short overview, with pointers to the files above.

## Design revision (2026-11-16): where AD is used
**Evidence.** The toy experiments are in `docs/tutorial_enzyme_reactant.md` §6 and `docs/technical.md` §4.1.
- The tangent-linear model has no saturation, so beyond the atmospheric predictability limit the single-trajectory derivative grows like e^(λT); in the toy, λ corresponds to a doubling time of about 2 days.
- Ocean perturbations reach the atmosphere through the coupling within days, and come back amplified. So even the sensitivities of ocean metrics such as Tₑ to the initial ocean state are dominated by chaotic paths.
- AD gives the *expected derivative*, but the target is the *derivative of the expected forecast*. In chaotic systems these differ, and averaging the derivative over an ensemble does not converge. The "ensemble-averaged adjoint at long leads" idea in this plan is **dropped**.

**Revised roles:**
1. **Coupled tangent-linear and adjoint models over short windows** (up to about 1–2 weeks, the TLM-valid window): exact sensitivities, short-range Σ(0)-weighted singular vectors, 4D-Var-style analysis, and targeting for weather and subseasonal targets (ARs, MJO-event onset, PNA/NAM regimes).
2. **Ocean-only adjoint (Oceananigans) under prescribed atmospheric forcing:** climatological plus stochastic, or a statistical atmosphere, with no dynamic atmosphere. A coarse-resolution ocean isn't chaotic, so the derivatives stay valid for months to years. This gives the sensitivity of ENSO/PDO metrics to the initial ocean state, validated against coupled ensembles.
3. **Long-lead coupled response from statistics:** the fluctuation–dissipation response operator G(τ) = C(τ) C(0)⁻¹, built from the strongly coupled lagged covariance (aim 1), with the LIM as one form. Long-lead targeting through ensemble-transform signal variance or ΔVar computed with G(τ) in place of the TLM.
4. **Hybrid / chaining:** a short-window adjoint, then a covariance-based response.
5. **Reserve:** least-squares shadowing (LSS/NILSS) for exact derivatives of long-time statistics, if ever needed.
6. **Jacobian about a fixed basic state (for linear and stochastic atmosphere models).** AD gives the exact tendency Jacobian A = ∂f/∂x evaluated at the nature-run mean state. Unlike the trajectory TLM, this is a time-invariant operator. With eddy damping (Farrell & Ioannou), A is stable and non-normal: disturbances grow transiently, then decay.
   - A driven by noise is a linear stochastic atmosphere. Its stochastic optimals and covariance can be compared with the coupled covariance C(0).
   - Gradients of statistics over short windows can calibrate the damping and the noise.
   - Caveat: x̄ is not a model solution, and the damping is a modelling choice.

**Unchanged:** the coupled model must still be differentiable, for roles 1 and 2 and for later calibration and hybrid machine learning. So phase 4 still goes ahead: precomputed regridding weights and Reactant ports of the atmosphere, the ocean and the coupler.

## Re-scoped aims and phases (2026-11-16)
This section supersedes the earlier phase list wherever they differ.

**Why.** The toy experiments and the AD design revision showed two things:
- Trajectory derivatives are valid only inside the predictability window, about 1–2 weeks for the atmosphere.
- Predictability at monthly to interannual leads lives in the **expectation** and **covariance** of the coupled system: predictability of the second kind.

So the project centres on **large ensembles of a low-resolution strongly coupled model and its coupled covariance**. It is organized as a **hierarchy of methods**, each scored against the one above on the same predictability metrics. In this perfect-model setting the true predictability is known from the ensembles, so every cheaper method's skill and spread can be checked.

**The hierarchy:**
1. **Physics ensembles:** large ensembles of the coupled model, NumericalEarth with SpeedyWeather and Oceananigans. These are the truth.
2. **The strongly coupled covariance:** lagged covariances with the full ocean–atmosphere cross terms. From them:
   - **POP/LIM eigenmodes** of G(τ) = C(τ) C(0)⁻¹. The **least-damped modes** are the candidate predictable patterns (ENSO, PDO, …), with their coupled atmospheric parts.
   - **Predictable component analysis** (Schneider & Griffies), a generalized eigenproblem between Σ(τ), the forecast-ensemble covariance, and C(0), the climatological covariance. It uses the nonlinear ensemble directly.
   - **Average predictability time** (DelSole & Tippett): predictability integrated over lead.
   - **Response operators** by the fluctuation–dissipation relation, G(τ) = C(τ) C(0)⁻¹.
   - **Optimal (transient) growth.** G is non-normal, so the least-damped mode is not necessarily the most important. Temporary constructive interference of more strongly damped, non-orthogonal modes can produce maximum amplification at a finite lead.
     - The **optimal initial structures** are the singular vectors of G(τ) = exp(Lτ), in a chosen norm, and their growth curve is max‖G(τ)‖ as a function of τ (Penland & Sardeshmukh 1995).
     - Predictable components and APT capture this automatically, because they are built from the forecast statistics rather than from eigenvalues.
   - **Comparisons:** does the least-damped POP equal the leading predictable component, and at which leads do they diverge?
3. **Linear baselines:** LIM, POP and PCA forecasts, and the predictability each implies.
4. **ML emulation with rigorous predictability.** Emulators trained deterministically (MSE) learn the conditional mean. They relax toward climatology and **under-predict error and spread**, the mirror image of the TLM's unbounded growth. Remedies:
   - **Probabilistic training:** proper scores (CRPS, energy score) on the physics ensembles.
   - **Split the state:**
     - an ML (nonlinear) drift on the slow, predictable subspace (the least-damped modes);
     - a stochastic residual whose covariance obeys the fluctuation–dissipation constraint Q = −(L C(0) + C(0) Lᵀ), so the emulator's climatological variance is correct by construction and its spread cannot collapse.
   - **Generative emulators,** such as score-based diffusion conditioned on the initial slow state, to produce ensembles directly.
   - **AD calibrates the emulator's noise.** The ML emulator is differentiable and much less chaotic than the physics, so AD can fit the structure and amplitude of its stochastic forcing directly. The method is the reparameterization trick: the noise is S(θ) ξ, with ξ sampled and held fixed during each gradient step.
     - The loss is a proper ensemble score (CRPS or energy score) plus penalties for mismatch with C(0) and with the lagged covariances of the physics ensembles, over windows where the rollout gradients stay stable.
     - The FDT relation supplies the starting value of Q; AD then refines it for the nonlinear drift.
     - Stochastic optimals, the forcing patterns that excite the most variance, come from the emulator's Jacobian.
   - **Verification against the physics ensembles:** spread–skill ratio, rank histograms, and spectra of the predictable components, the APT and the least-damped modes, all by lead and season.
   - **Emulator ensembles with the right error growth:** train a one-step emulator on values and the physical model's Jacobian products (Enzyme), so its ensembles grow errors correctly and saturate at the climate covariance; its Jacobian then gives the adjoint for sensitivity and targeting. First emulator: an SFNO for the SpeedyWeather atmosphere under prescribed SST, in Julia. See `docs/technical.md` §6.
5. **Targeted observations** (adaptive observing: choosing where and when to add observations such as floats, moorings, drifters, sondes or satellite sampling; then assimilating them in OSSEs to verify the forecast improvement): ΔVar from ensemble or covariance statistics (ETKF signal variance, or R(τ) in place of the TLM), validated with OSSEs. The **ocean-only adjoint** supplies interannual ocean-initial sensitivity at full ocean dimension, for siting Argo or TPOS-style observations.

**AD's supporting roles** (see the design revision above):
- the ocean-only adjoint under prescribed or stochastic atmospheric forcing;
- a short-window coupled TLM and adjoint for subseasonal targets, extending a little into weeks 2–4;
- basic-state Jacobians for linear and stochastic atmospheres;
- **end-to-end training** of ML or stochastic components over short windows, with losses on ensemble spread as well as the mean.

Reactant is kept mainly for **GPU performance portability of the large ensembles**.

**Revised phases:**
1. **Toy model:** done (tag `phase1-toy-done`). AD validated; the chaos limit and the failure of ensemble-mean adjoints shown; LIM and regression baselines.
2. **Coupled model, forward:**
   - smoke test: done;
   - regridding weights: done, saved to NetCDF, adjoint = transpose;
   - next: a multi-decade nature run at low resolution, the realism check for each phenomenon tier, and a timing study (CPU, and later GPU through Reactant).
3. **Large ensembles:** initialized from the nature run (perturbed atmosphere, perturbed ocean, or both), at several start seasons and leads up to 2 years. They supply forecast covariances Σ(τ) and, with the nature run, the nature-run covariance C(τ).
4. **Strongly coupled covariance and predictable modes:** POP/LIM, PCA, APT and response operators, by phenomenon (registry) and season.
5. **Targeting and OSSEs** from ensemble or covariance statistics, plus the ocean-only adjoint for ocean observations.
6. **ML emulation** of the slow subspace with calibrated stochastic residuals, verified against phases 3–4.
   - May start earlier, after the nature run: an SFNO one-step atmosphere emulator trained on values and Jacobian products, then degraded emulators to find which deficiencies bias spread, predictability and targeting (`docs/technical.md` §6.6–6.8) (`docs/technical.md` §6.1).
7. **AD extras as needed:**
   - Reactant ports and checks of the atmosphere and ocean. `drivers/reactant_*_check.jl` is written; it needs CUDA.jl, deferred, because Reactant compiles KernelAbstractions kernels through its CUDA extension even on CPU.
   - Short-window coupled adjoint.
   - End-to-end training.
