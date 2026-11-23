# Status

_Last updated 2026-11-23._

## Done
- Plan adopted (`docs/plan.md`).
- Julia 1.11.9 installed, with a juliaup override in this repo.
- Tools: `gh` (logged in), plus `jupytext` and `nbstripout` from the micromamba `nbtools` env, symlinked into `~/.local/bin`.
- Agents (`code-reader`, `upstream-scout`) and skills (`julia-env`, `differentiable-code`, `add-phenomenon`, `netcdf-output`) written.

- Main environment resolved on 1.11.9 at the target versions; Manifest committed (main, b856558).
- Phase 1 (tag `phase1-toy-done`):
  - `ToyCoupled` (Lorenz-96 + slab ocean + recharge oscillator) and the `AbstractPropagator` interface, with 11 passing tests;
  - `io.jl` (NetCDF with provenance);
  - `drivers/toy_nature_run.jl`;
  - `plots/` (readers, plotlib, `toy_nature_run` notebook).
- Toy ENSO tuned for 3× irregularity: R = −0.006, γ = 0.0156, σ = 0.03. Over 300 years the CV of cycle lengths is ≈ 0.3, against ≈ 0.1 before (`drivers/toy_enso_sweep.jl`; `docs/technical.md` §2).
- Toy nature run, `run/toy-nature/`, rerun with the new defaults (50 years). Numbers from `plots/toy_nature_run.jl`:
  - peak period 3.84 years;
  - 11 cycles, 4.3 ± 1.1 years, CV 0.265;
  - std(Tₑ) = 0.256, skewness +0.20;
  - e-folding times: X₁ 2 days, T₁ 45 days, Tₑ 244 days.

- Plotting: Pluto notebooks are the default (`plots/*_pluto.jl`); Jupyter is kept for the HPC. Heavy computation goes in drivers.
- AD layer (`src/ad.jl`: `jvp`, `vjp`, `gradient_J`, `Component`); `@trace` loop in `propagate` (copies its input first). Tests: 20 pass, including Reactant against Enzyme (`DE_TEST_REACTANT=true`).
- AD tutorial: `docs/tutorial_enzyme_reactant.md`, `drivers/tutorial_ad.jl` (run in `run/tutorial-ad/`), `plots/tutorial_ad_pluto.jl`. Findings:
  - the AD is correct at all leads;
  - the toy TLM is valid for only about 1 week; derivatives grow like e^(λT), with a doubling time of about 2 days;
  - single-trajectory adjoints are meaningless beyond about 2 weeks;
  - Reactant is 30–600× faster per gradient after compiling.

- Toy sensitivity comparison, first run (`run/toy-sensitivity/`; `docs/technical.md` §4.1; `plots/sensitivity_comparison_pluto.jl`):
  - adjoints are excellent at 2 weeks and fail beyond;
  - the ensemble-mean adjoint does not converge;
  - regression is poor;
  - the LIM keeps skill: Tₑ pattern correlation 0.64 at 1 year;
  - but the truth is noise-dominated beyond about 2 months, so the run needs a redesign.

- Phase 2 (tag `phase2-regrid-done`; all work now on `main`):
  - coupled smoke test passes (`run/coupled-smoke/`);
  - regridding weights extracted, checked (transpose 4e-15, constant-preserving, identical to `ConservativeRegridding.regrid!`) and saved (`assets/`), through the `PredictableOceanCoupledExt` extension.
- Reactant checks pass against finite differences:
  - **ocean:** Oceananigans on a LatitudeLongitudeGrid with buoyancy and Coriolis, 1e-12 (`run/reactant-ocean/`);
  - **atmosphere:** SpeedyWeather PrimitiveWet T32L8 Float64, 3e-8 (`run/reactant-atmosphere/`).
- Plain Enzyme on SpeedyWeather segfaults on Julia 1.11, so unpin criterion 3 is met (`docs/julia_pin.md`).
- **Plan re-scoped** (`docs/plan.md`, "Re-scoped aims and phases"): ensembles plus the strongly coupled covariance (least-damped modes, PCA/APT, response operators), ML emulation with rigorous predictability, AD in supporting roles.

- **HPC smoke test** (2026-11-16, job 7432005, `rime`, an aarch64 GH200 node, Grace CPU only, 8 threads), `run/coupled-smoke-hpc/`:
  - build 223 s; first step 37.5 s (compilation);
  - **≈ 0.39 s per 40-minute coupled step** (179 steps in about 70 s);
  - SST identical to the Mac run.
- **Implied cost:**
  - about 1.4 h per member-year, about 17 simulated years per day per 8-core job, about 150 SYPD per 72-core node if members scale independently (unmeasured);
  - 2-year ensembles of 30 members × 4 seasons × 20 years (4800 member-years) ≈ 32 node-days;
  - a decadal check of 5 starts × 20 members × 10 years ≈ 7 node-days;
  - a 500-year nature run takes about 30 days of wall time as one 8-core job, so single long runs need speed-ups.
  - Likely speed-ups: a longer coupled Δt with the atmosphere sub-stepped (the ocean currently steps every 40 min); thread scaling; GPU.
- **Ensemble sampling** (user's point, 2026-11-17):
  - pooled over 4 seasons × 20 years of starts, the 30-member ensembles give about 2300 degrees of freedom for the **noise** covariance Σ(τ) at every lead (29 per start × 80 starts). That is ample for strongly coupled cross-covariances, though seasonal and state dependence reduce it (about 580 per season).
  - The **signal** covariance and the climatological C(0) and C(τ) have only about 80 correlated ensemble-mean samples, so the effective number is the count of independent slow states. These need the long nature run.
  - The coupled state is 10⁵–10⁶ variables, so analysis works in a reduced space of about 100–300 EOFs.
- **Decadal predictability** (idea, not yet in the plan):
  - extrapolate with the LIM, G(τ) = C(τ) C(0)⁻¹ = exp(Lτ), Q = −(L C(0) + C(0) Lᵀ), with the τ-test;
  - diagnostic potential predictability from a several-century control run with realistic geometry;
  - check against a few 10-year ensembles. Caveat: AMOC and decadal ocean physics at 2°.
- **Literature:** `docs/novelty.md` (assessment) and `docs/references.md` (56 AMS-style references with DOIs). Consensus thread: https://consensus.app/search/literature-review-to-assess-the-novelty-of-a-plann/uihD0upqS4-DK0PLLx3Jsg/ . The Elicit connector is unavailable (no API access on the plan).

- **Emulator experiment docs:** `docs/technical.md` §6 (covariance, adjoint sensitivity, targeting, what an emulator must supply, §6.8 withholding experiment) and §7 (6 h map, derivatives, pipeline). `notes/reading_list.md`.
- **SpeedyWeather 0.25:** NumericalEarth pinned to main 83e1c68 for 0.25 compatibility. Return to a release ≥ 0.9.5 when one is out.
- **6 h map check** (`drivers/speedy_map_check.jl`, `run/speedy-map-check/`):
  - the map is bit-deterministic;
  - 0.06 s per map on Mac CPU;
  - the Reactant reverse pass compiles in 326 s and takes 0.54 s per VJP;
  - finite-difference JVPs are linear only for ε ≲ 3e-9, because physics switches flip at larger ε;
  - the dot-product test agrees to 1%, first attributed to kinks at q = 0 and q = q_sat. Revised 2026-11-23: the more likely cause is the radius scaling of vor and div in the reverse pass, fixed in R2, which now agrees to 1e-5 (`docs/technical.md` §7.2);
  - the zenith kernel is patched at runtime (Irrational/Int `ifelse` failed to raise in Reactant).
- **z₀ is hidden state:** momentum roughness carries across the 6 h restart, so the map is M(x, z₀). z₀ is stored per map in the nature run and in the restart file. Resuming is bit-identical.
- **Aquaplanet nature run on Mac** (`drivers/speedy_nature_run.jl`, `run/speedy-nature/`):
  - T31L8, 2-year spin-up, then 40 years: 0.06 s per map, about 87 s per year, 8.7 GB;
  - years 41–50 done from the restart file (2026-11-23): top 330.4–330.8 K, bottom 291.8–292.0 K; 50 files, 11 GB;
  - the bottom-layer global-mean T is steady at 291.9 K;
  - the top layer (σ = 0.017) warms from 318.6 K in year 1 to 330.6 K by year 10, then holds at 330.5–330.8 K. So years 1–10 count as spin-up.

## Next
- **R2, Jacobian samples** (`drivers/speedy_jacobian_samples.jl` + `.sbatch`):
  - the smoke test passed on the HPC (2026-11-23, `brizo01`): dot test 1e-5 after the radius-scaling fix, 8.5 s per state with 8 CPUs;
  - the full run was submitted as `--array=0-5` (about 1.3 h per task);
  - 2 CPUs were 15× slower because of thread contention (`docs/technical.md` §7.3).
- **Emulator track:** the training runs R2–R4 of `docs/technical.md` §7.3 (Jacobian samples, physics test ensembles, SFNO training). For the HPC nature rerun, use `NATURE_SPINUP_DAYS=3650`. At T31 a CPU core suffices: 40 years take about 1 h, so a GPU is not needed.

Coupled track:
1. A timing study: atmosphere and ocean cost per step; a longer coupled Δt with the atmosphere sub-stepped; thread scaling; later GPU. Then a multi-decade low-resolution coupled nature run.
2. Realism checks for the Tier 1 phenomena in the nature run.
3. Large ensembles, then the covariance and predictable modes.

## Open questions
- Unpin Julia 1.11 and move to 1.12 or 1.13? Criterion 3 is met; waiting for the user's decision.
- z₀ in the emulator (measured 2026-11-23, drivers/speedy_z0_check.jl, year 11, 20 states): the z₀ part of the 6 h change, relative to the change, is 3e-5 (vor, T, q, lnps) and 2.4e-4 (div) with z₀ recomputed from x; 2e-4 and 1.7e-3 with fixed z₀. Negligible: M̂ sees x only. A nature-run rerun sets `use_computed_roughness = false` (LearnedWindSlowdown), so no z₀ carries between steps and M depends on x only.
- Training data (decided 2026-11-23): the Mac nature run is canonical; the user copies it to the HPC.
- JVP training targets: the finite-difference JVPs agree with the VJPs to 1e-5 (R2 smoke test), so both can be used.
- Rerun the map check with the radius-scaling correction, to confirm that it removes the 1%.
- Report the zenith-kernel bug upstream? (Ask the user.)
- Rerun the package tests, the coupled smoke test and the Reactant atmosphere check after the 0.25 upgrade.
