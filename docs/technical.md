# Technical notes

Living document: the methods and implementation choices, written before each piece is built.
See `docs/plan.md` for the overall plan and `docs/julia_pin.md` for the Julia version pin.

References:
- Moses et al. (2026), *DJ4Earth: Differentiable, and performance-portable Earth system modeling via program transformations*, JAMES, doi:10.1029/2025ms005615.
- Reproduction code: github.com/DJ4Earth/differentiable-esm-components-2025.

## Notation (used in all project documents)
One symbol per quantity; time dependence is always an argument in parentheses (no subscript 0). Full references: `docs/references.md`.

**Times**
| Symbol | Meaning |
|---|---|
| t | absolute (model or calendar) time, e.g. a start date; month(t) is its season |
| τ | time separation: lag in nature-run statistics, and lead in forecasts (time since the reference time t) |
| ℓ | the fixed LIM fitting lag, as in L = (1/ℓ) log G(ℓ) |
| n | number of model time steps; Δt is the model time step, so τ = n Δt |
| Δτ | emulator time step (6 h), a whole number of model steps Δt; lead τ advances in steps of Δτ for emulator quantities (§6) |

**State, members, averages**
| Symbol | Meaning |
|---|---|
| x | anomaly state vector (departure from the nature run's seasonal climatology) |
| x_clim(t) | the nature run's seasonal climatology at time t; the full model state is x_clim(t) + x |
| x_k | ensemble member k = 1 … K; x̄ is the ensemble mean (average of x_k over k); x′_k = x_k − x̄ |
| ⟨·⟩ | an average; always stated "over t" (in the nature run) or "over k" (over members) |
| ᵀ | transpose |

**Covariances and operators**
| Symbol | Meaning |
|---|---|
| C(τ) | nature-run lag covariance ⟨x(t+τ) x(t)ᵀ⟩ over t (the model's own climate, not observations); C(0) is the nature-run covariance. Seasonal version: C(τ; month), averaged only over t in that calendar month |
| Σ(τ; t) | forecast (ensemble) covariance at lead τ for the ensemble started at t: ⟨(x_k(t+τ) − x̄(t+τ)) (x_k(t+τ) − x̄(t+τ))ᵀ⟩ over k. Σ(0; t) is the initial-uncertainty (analysis-error) covariance. The start argument t is dropped when it is clear |
| Σ(τ, 0) | ensemble cross-covariance between member deviations at lead τ and at the start |

**Same-time and lagged.** Σ(τ) is a *same-time* covariance: both factors are member deviations at lead τ, so it is symmetric and measures the forecast spread. C(τ) is *lagged*, x(t+τ) against x(t), and not symmetric for τ ≠ 0. The lagged ensemble quantity is Σ(τ, 0). Predictable component analysis compares two same-time covariances, Σ(τ) and C(0).

| Symbol | Meaning |
|---|---|
| S(τ) | signal covariance: covariance over starts t of the ensemble means x̄(t+τ). C(0) = ⟨Σ(τ; t)⟩ over t + S(τ) |
| M(τ) | the forecast map, written on anomalies: M(τ)(x) = (model run for lead τ from the full state x_clim(t) + x) − x_clim(t + τ). The model itself steps full states; differences between states, and the Jacobian M′, are the same for full states and anomalies |
| M′(τ) | tangent-linear propagator (Jacobian of M along a trajectory); M′(τ)ᵀ is the adjoint |
| M̂(Δτ), M̂′, x̂ | ML one-step emulator of M (nonlinear); its Jacobian; its single forecast from x̂(0) = x̄(0) (§6) |
| T(x) | tangent network, T(x) ≈ M′(Δτ; x) (§6) |
| X(τ) | d × K scaled member deviations [x′_1 … x′_K]/√(K − 1), so Σ(τ) = X(τ) X(τ)ᵀ (§6) |
| z | adjoint (cotangent) vector, as in z(0) = M′(τ)ᵀ z(τ) |
| L | LIM generator: dx/dt = L x + noise |
| G(τ) | propagator of the expected forecast, C(τ) C(0)⁻¹ = exp(Lτ); also estimated from ensembles, as G(τ; x) = Σ(τ, 0) Σ(0)⁻¹ conditioned on the start state x, or G(τ; month) seasonally |
| Q | noise covariance, −(L C(0) + C(0) Lᵀ); seasonal: Q(t) = dC/dt − L(t) C(t) − C(t) L(t)ᵀ |
| N(τ) | covariance added by noise since the start, C(0) − G(τ) C(0) G(τ)ᵀ, so Σ(τ) = G(τ) Σ(0) G(τ)ᵀ + N(τ) |
| A | tendency Jacobian ∂f/∂x evaluated at a fixed basic state (for linear stochastic models) |
| J | scalar forecast metric, e.g. Niño-3.4 at lead τ |
| g | sensitivity ∂J/∂x(0); for the expected forecast, g = ∂⟨J(M(x(0) + ε))⟩/∂x(0), averaged over ε ~ N(0, Σ(0)) |
| W | weighting matrix defining a norm |
| H, R | observation operator and observation-error covariance (targeting and assimilation) |
| s | climatological standard deviation of each variable, √diag C(0) |
| λ | Lyapunov exponent (error growth rate) |

Toy-model parameters (F, α, β, λ for slab damping, R, γ, σ, ϵ, r, a, μ, …) are named in §2 only and are not project notation.

## 1. The propagator interface
Every analysis is written against `AbstractPropagator` (`PredictableOcean/src/propagator.jl`). A propagator has:
- a flat state vector `x`;
- `advance(m, x)`, which takes one time step and returns a new vector without mutating `x`;
- `propagate(m, x, n)`, which takes n steps: the forecast map M(τ) for lead τ = n Δt;
- `state_slices(m)`, which names the parts of the state.

The toy model and, later, the coupled NumericalEarth model both implement this interface. Covariances, singular vectors, targeting and OSSEs therefore run unchanged on either.

## 2. The toy model: `ToyCoupled`
`PredictableOcean/src/toy/toy_coupled.jl`. A Lorenz-96 "atmosphere" of K = 36 sectors, coupled to a slab ocean under each sector and to a recharge-oscillator ENSO. It is meant to have the same three time scales as the real problem:

| Component | Variables | Time scale | Role |
|---|---|---|---|
| Lorenz-96 | X₁…X_K | days (chaotic; 1 model time unit ≈ 5 days) | weather noise, limits atmospheric predictability |
| slab ocean | T₁…T_K | ~90 days (λ⁻¹ = 18 units) | ocean memory, local SST persistence |
| recharge oscillator | Tₑ, h | years (period ≈ 3.7 years, irregularity CV ≈ 0.3) | ENSO-like predictable signal |

Equations:
- dX_k/dt = (X_{k+1} − X_{k−2}) X_{k−1} − X_k + F + α T_k + αₑ p_k Tₑ
- dT_k/dt = −λ T_k + β (X_k − X̄) + κ (T_{k+1} − 2T_k + T_{k−1})
- dTₑ/dt = R Tₑ + γ h + σ ξ − ϵ Tₑ³
- dh/dt = −r h − a (μ Tₑ + ξ)

where ξ is the mean of (X_k − X̄) over the "equatorial" sectors 1…n_eq (a wind-burst forcing) and p_k is a wavenumber-1 teleconnection pattern.

**Model statement: what the variables represent.** All variables are anomalies and nondimensional (nominal units in the NetCDF files); one model time unit ≈ 5 days.
- **X_k (k = 1…36):** a Lorenz-96 "atmosphere" variable in longitude sector k around a latitude circle, such as a wind or geopotential anomaly. X (no subscript) is the vector of all 36. X̄ = 2.3 is its approximate climatological mean, used to form anomalies. Sectors 1…n_eq (n_eq = 4) are labelled "equatorial".
- **T_k:** the surface (thermodynamic, slab-ocean mixed-layer) temperature anomaly under sector k. It is damped (λ), forced by an atmospheric heat flux β(X_k − X̄), diffused zonally (κ), and feeds back on the atmosphere through α T_k.
- **Tₑ:** the eastern equatorial Pacific SST anomaly, a Niño-3-like ENSO index, as in Jin's (1997) recharge oscillator. It is *not* a basin-mean temperature or heat content. It is the eastern Pacific surface temperature, set by **upwelling** of subsurface water whose temperature depends on the thermocline depth.
  - In the toy, the upwelling enters as the thermocline feedback γ h (a deeper thermocline means warmer upwelled water). R Tₑ represents the Bjerknes wind–SST feedback combined with damping.
  - The nonlinearity of upwelling is represented only crudely, by the symmetric cubic damping −ϵ Tₑ³. That term bounds the oscillation but gives no El Niño/La Niña asymmetry.
  - A quadratic nonlinear-upwelling term, such as a nonlinear dynamical-heating term ∝ Tₑ² or ∝ Tₑ h, would add the observed asymmetry if needed.
- **h:** the western equatorial Pacific thermocline-depth anomaly, i.e. the equatorial heat content to be discharged or recharged. It responds only to the basin-scale zonal wind stress τ = μ Tₑ + ξ (dh/dt = −r h − a τ). Tₑ is driven by the thermocline feedback γ h, the Bjerknes growth/damping R, the wind bursts σ ξ and cubic damping ϵ Tₑ³.
- **ξ:** the mean of (X_k − X̄) over the equatorial sectors: atmospheric "wind bursts", the stochastic forcing of ENSO.
- **Couplings.** atmosphere → ENSO through ξ; ENSO → atmosphere through the teleconnection αₑ p_k Tₑ (wavenumber 1); ocean ↔ atmosphere locally through α and β.
- **Simplifications to keep in mind:**
  - Tₑ is *not* the slab-ocean temperature of the equatorial sectors: T₁…T₄ and Tₑ are separate variables, linked only through the atmosphere.
  - The model has no zonal basin geometry: "eastern" and "western" are labels on Tₑ and h, not locations among the sectors.

The coupling runs both ways:
- the ocean forces the atmosphere through α T_k;
- ENSO forces it through αₑ p_k Tₑ;
- the atmosphere forces the ocean through β and ξ.

So the full covariance has ocean–atmosphere cross-terms, as the real one will.

Time stepping is RK4 with Δt = 0.05 (≈ 6 h). The code uses only whole-array operations (`circshift`, broadcasts, `sum`, `vcat`, and 1-element slices instead of scalar indexing), with no data-dependent branches. That makes it traceable by Enzyme and by Reactant.

Why a toy model first:
- The analysis code (covariances C and Σ, singular vectors, targeting ΔVar, ETKF/OSSE) can be checked against exact answers in seconds.
- The Enzyme/Reactant workflow can be learned on a model small enough that its full Jacobian fits in memory (74 × 74), so every AD result can be checked against a brute-force one.

Status: implemented and tested (`PredictableOcean/test/test_toy.jl`). The tests cover the Lorenz-96 limit, purity, boundedness and sensitive dependence on initial conditions. **ENSO tuning (2026-11-15).** The first guess was a self-sustained oscillation: R = +0.002, γ = 0.02, σ = 0.01. Its period was ≈ 3.8 years, but it was too regular: the CV of cycle lengths was ≈ 0.1 over 300 years.
- The defaults are now a damped, noise-driven oscillator: **R = −0.006, γ = 0.0156, σ = 0.03**.
- That gives a CV of ≈ 0.3 (0.32 and 0.27 for two seeds over 300 years), about 3× the first guess, close to observed ENSO.
- Period: 3.6–3.8 years. std(Tₑ): ≈ 0.28.
- Sweeps: `drivers/toy_enso_sweep.jl`, run in `run/toy-enso-*`.
- Damping shortens the period, so γ was lowered to restore it (period ∝ 1/√(γa)).

## 3. Automatic differentiation: the approach (proposed, not yet implemented)
APIs were checked against the installed Reactant 0.2.308 and Enzyme 0.13.219 on 2026-11-15.

### 3.1 What we need from AD
Write M(τ) = `propagate(m, ·, n)` for the forecast map over lead τ = n Δt, and M′(τ) = ∂M/∂x for its Jacobian along the trajectory from x(0). We never form M′ for the real model, which would be millions by millions. Instead we compute its products with vectors:
- **Tangent-linear model (TLM), a JVP:** δx(τ) = M′(τ) δx(0). This is how a small initial perturbation grows, to first order. In AD terms it is *forward mode*: one forward run carrying a tangent alongside the state.
- **Adjoint, a VJP:** z(0) = M′(τ)ᵀ z(τ). For a scalar forecast metric J(x(τ)), such as Tₑ at lead τ, the gradient g = ∂J/∂x(0) = M′(τ)ᵀ ∇J. In AD terms it is *reverse mode*: run forward, then propagate sensitivities backward from τ to 0. One reverse pass gives the sensitivity of J to **every** initial-state variable at once, which is why the targeting analysis is affordable.
- Singular vectors and the Σ(0)-weighted analyses call these two products repeatedly, inside KrylovKit's Lanczos iterations.

### 3.2 Plain Enzyme
Enzyme differentiates the compiled LLVM code of ordinary Julia functions:
- **JVP:** `Enzyme.autodiff(Forward, x -> propagate(m, x, n), Duplicated(x, v))[1]`. `Duplicated(x, v)` pairs the input with its tangent v, and the result is the output tangent M′ v.
- **VJP:** reverse mode needs a scalar output, so differentiate the scalar z·M(x). Its gradient is exactly M′ᵀz: `Enzyme.gradient(Reverse, x -> dot(z, propagate(m, x, n)), x)[1]`. For J itself, `Enzyme.gradient(Reverse, x -> J(propagate(m, x, n)), x)[1]`.
- **Activities:** `Duplicated` (differentiated array, with a shadow), `Const` (not differentiated, e.g. the model parameters), `Active` (a differentiated scalar return).
- **Cost:** reverse mode stores the forward trajectory, so memory grows with n. That's fine for the toy, but this is why checkpointing exists.

### 3.3 Reactant
Reactant **traces** a Julia function into a compiler graph (MLIR/StableHLO), optimizes it, and compiles it with XLA for CPU (here) or GPU (HPC). Reactant's gradient is Enzyme run on that graph (EnzymeMLIR), so it is the same AD applied one level down.
1. **Convert inputs:** `xr = Reactant.to_rarray(x)` gives a `ConcreteRArray`. Scalars become `ConcreteRNumber` only if asked.
2. **Compile once:** `vjp_c = @compile vjp(m, xr, zr)` traces and compiles. Calling `vjp_c(m, xr2, zr2)` then runs at compiled speed for new inputs. `Array(result)` brings the result back.
3. **Loops must be traced.** A plain Julia `for` loop would be unrolled n times at trace time. `@trace for _ in 1:n ... end` becomes one compiled `while` loop. `@trace` does nothing outside Reactant, so the package's `propagate` can carry it and serve plain Julia, plain Enzyme and Reactant from **one code path**.
4. **Checkpointing:** `@trace checkpointing=... mincut=true for ...`.
   - `checkpointing=true` stores √n states and recomputes the rest.
   - `Reactant.Periodic(k)` stores k states.
   - `Reactant.Binomial(b)` uses the revolve algorithm with a budget of b states; it is recommended when n is a traced number.
5. **Baked-in constants:** a plain Julia value passed to `@compile` (Δt, the model parameters, a Julia `Int` n) is frozen into the compiled code. Calling it later with a different value silently uses the old one. To vary n without recompiling, pass `n` as `ConcreteRNumber` and use `Binomial` checkpointing. Our default instead compiles one function per lead time, which is simpler and cheap for the toy.
6. **Rules for traced code:** no scalar indexing without `@allowscalar` (the toy uses 1-element slices), and no data-dependent Julia `if` (use `@trace if`, or smooth the switch).

### 3.4 How we know the derivatives are right
Run on the toy for every propagator; the coupled model later gets the first four (`differentiable-code` skill):
1. **Finite differences:** (J(x+εv) − J(x−εv))/2ε ≈ ∇J·v over ε = 1e-2 … 1e-8.
2. **Dot-product test:** ⟨M′δx, z⟩ = ⟨δx, M′ᵀz⟩ to round-off. This shows the adjoint really is the transpose of the TLM.
3. **Reactant equals plain Enzyme** to round-off, which catches tracing bugs.
4. **TLM validity against lead time:** compare M(x+δx) − M(x) with M′δx as τ grows. The atmosphere's part should fail within a few weeks, because of chaos. ENSO-related directions should stay linear much longer. This sets where adjoint results can be trusted, and motivates the ensemble-averaged gradients in the plan.
5. **Toy only:** the full Jacobian (`Enzyme.jacobian`, 74×74) against all of the above.

## 4. Sensitivities that survive chaos (toy comparison, phase 1)
**Why.** The tutorial (`docs/tutorial_enzyme_reactant.md` §6) showed that a single-trajectory adjoint grows like e^(λτ), with a doubling time of about 2 days. Beyond about 2 weeks it says nothing about the forecast we actually care about. For targeting at monthly-to-interannual leads, the right object is the sensitivity of the **expected** forecast to the **slow** initial state (the ocean T_k, Tₑ and h: 38 variables). The expectation is taken over the analysis uncertainty, which is dominated by the chaotic atmosphere:

  gᵢ = ∂⟨J(M(x(0) + ε))⟩ / ∂xᵢ(0), averaged over ε ~ N(0, Σ(0)), with i over the slow variables.

**Truth (exact ensemble answer).** For each slow variable i, centered finite differences of the ensemble mean, using paired ("common random number") ensembles:

  gᵢ ≈ ( ⟨J(M(x(0) + δᵢ eᵢ + ε_k))⟩ − ⟨J(M(x(0) − δᵢ eᵢ + ε_k))⟩ ) / 2δᵢ, averages over members k = 1…N.

The step is δᵢ = 0.5 sᵢ: large enough to beat the sampling noise, and smooth because of the ensemble average. Its standard error comes from the member-by-member differences. Cost: 38 × 2 × N forecasts per initial state. That's cheap for the toy and impossible for the real model; it exists to judge the estimators below.

**Estimators** (all affordable for the coupled model):
1. **Single-trajectory adjoint** ∂J/∂x(0) at x(0): one reverse pass. Expected to fail beyond about 2 weeks.
2. **Ensemble-mean adjoint:** the mean over members of ∂J/∂x(0) at x(0) + ε_k, with N reverse passes. Its expectation equals gᵢ, but in chaotic systems its variance grows like e^(2λτ), so it may not converge at long leads (Lea, Allen & Haine 2000, *Tellus*).
3. **Ensemble (regression) sensitivity:** regress J(τ) on the initial slow perturbations across an ensemble drawn from Σ(0) (Ancell & Hakim 2007). With a diagonal Σ(0) this estimates gᵢ directly, using N forward runs and no adjoint.
4. **LIM (linear inverse model):** fit dx/dt = L x on the slow variables from the nature run, with G(ℓ) = C(ℓ) C(0)⁻¹ and L = (1/ℓ) log G(ℓ). Then g = the row of exp(Lτ) for J. It is statistical and state-independent, uses no model runs, and is the basis of the "modes from the covariance" analysis.

**Initial uncertainty Σ(0) (diagonal):** atmosphere 0.1 s; ocean T_k, Tₑ and h 0.05 s, where s is the climatological standard deviation of each variable. (Earlier text and the tutorial notebook called it σ or s_clim; the drivers' NetCDF variables keep the names `sigma_clim` and `s_clim`.)

**Leads:** 2 weeks, 2 months, 6 months, 1 year, 2 years. **Initial states:** several, at different ENSO phases.

**Scores against the truth, by lead:** pattern correlation and relative RMS error over the 38 slow variables, plus the components for Tₑ and h, and cost.

Code: `PredictableOcean/src/sensitivity.jl`, `drivers/toy_sensitivity_comparison.jl`, `plots/sensitivity_comparison_pluto.jl`.

### 4.1 First results (2026-11-16, `run/toy-sensitivity/`)
4 initial states; truth from 200 paired members with δ = 0.5 s; ensemble-mean adjoint from the same 200 members; regression from 400 members; LIM fit to 50 years.

Pattern correlation with the truth (J = Tₑ), pooled over states, at leads 2 wk / 2 mo / 6 mo / 1 yr / 2 yr:

| estimator | 2 wk | 2 mo | 6 mo | 1 yr | 2 yr |
|---|---|---|---|---|---|
| single-trajectory adjoint | 0.98 | 0.10 | −0.22 | −0.11 | −0.21 |
| ensemble-mean adjoint | 1.00 | −0.21 | −0.11 | −0.04 | −0.13 |
| regression | 0.41 | 0.12 | 0.11 | −0.05 | 0.01 |
| LIM | 0.99 | 0.89 | 0.66 | 0.64 | 0.41 |

| truth signal/noise, rms(truth)/rms(s.e.) | 2 wk | 2 mo | 6 mo | 1 yr | 2 yr |
|---|---|---|---|---|---|
| J = Tₑ | 49 | 2.3 | 1.6 | 1.3 | 1.2 |
| J = T₁ | 24 | 1.1 | 1.1 | 0.9 | 1.0 |

What it shows:
1. **At 2 weeks, the adjoints are excellent.** The ensemble-mean adjoint has a relative error of 0.04. The LIM is almost as good (0.13).
2. **Beyond 2 weeks, both adjoints fail completely.** Their relative errors are 1e6 at 2 months and 1e50 or more at 1 year. The ensemble-mean adjoint does **not** converge, as Lea et al. (2000) predict: its variance grows like e^(2λT).
3. **Regression is poor at every lead.** The initial slow-variable spread is small (0.05 s), so the regression signal is buried in atmosphere-driven variance of J. This is an intrinsic weakness of ensemble sensitivity when analysis errors in the slow variables are small.
4. **The LIM is the only estimator that keeps skill at long leads** (Tₑ: 0.89 at 2 months, 0.64 at 1 year).
5. **Caveat: the truth is noise-dominated beyond about 2 months** (signal/noise ≈ 1–2, and ≈ 1 for T₁). The long-lead scores therefore under-state every estimator, and the T₁ results beyond 2 weeks can't be interpreted. The truth needs a redesign (§4.2).

### 4.2 Next: a resolvable truth (proposed)
Estimate the expected-forecast sensitivity along a few **directions** instead of 38 individual variables, with many more members:
- **Directions:** Tₑ, h, and the leading slow-variable patterns (LIM/EOF modes).
- **Members:** about 2000.

Score the estimators by projecting them onto the same directions. Optionally add a hybrid estimator: an adjoint over the short TLM-valid window (about a week), then the LIM beyond it.

## 5. Filtering fast, unpredictable modes (2026-11-17)
Beyond the atmospheric predictability limit, forecasts are only useful through their slow, predictable part. Initialized ensembles isolate it by averaging over members. The methods below do it **without ensembles**, from a single long run or a single realization, or **use ensembles once** to build filters that are then applied everywhere. Methods 1–4 and 6 are the ones to use first: they are cheap and work directly on archived institutional data (control runs, monthly means).

1. **Time averaging.** Fit and forecast on monthly or seasonal means, the standard LIM practice (Penland & Sardeshmukh 1995). Averaging over longer than the atmospheric decorrelation time (days to weeks) leaves mainly the slow, ocean-forced part. The averaging window must be shorter than half the period of the slowest oscillation of interest, and fitting lags ℓ are chosen accordingly (τ-test).
2. **Subspace projection.** Restrict to slow variables, or truncate to leading EOFs of SST and upper-ocean heat content.
   - In the toy model, a LIM fitted to the full state (with the chaotic Lorenz-96 atmosphere) failed: it gave negative "variances" and failed the τ-test.
   - On the 38 slow variables it passed the τ-test, and its leading predictable patterns matched the ensemble's with |correlation| 0.96–1.00 (`drivers/toy_predictable_components.jl`).
3. **Projection onto least-damped modes.** Keep only the POP/LIM eigenmodes of L with long decay times, i.e. eigenvalues with real part closest to zero. Fast modes are, by construction, the strongly damped ones.
   - Caveat: G is non-normal, so damped modes can interfere constructively. Keep the optimal-growth structures (singular vectors of G(τ)) alongside the least-damped modes.
4. **Predictable-component filters.** The filters q from Σ(τ) q = γ C(0) q (Schneider & Griffies 1999) are linear projections that extract the predictable part of *any* single state: the component amplitude is qᵀx.
   - Estimate them once from initialized ensembles; then apply them to the nature run, to single forecasts or to observations. This is the main remaining job of the initialized ensembles.
   - Only components with γ clearly below the sampling-noise floor, about (1 − √(d/ν))² for dimension d and ν degrees of freedom (Marchenko–Pastur), are real. In the toy at 2 years only one component (the ENSO pair's leading phase) passes.
   - Related: signal-to-noise-maximizing patterns and optimal fingerprints (e.g. Venzke et al. 1999).
5. *(Secondary.)* Wiener-type optimal filtering of the slow signal.
6. **Conditional-mean estimation from one long run.** Learn the expected forecast E[x(t+τ) | x(t)] by regression on a single long run, with no ensembles:
   - **linear:** the LIM propagator G(τ) = C(τ) C(0)⁻¹;
   - **nonlinear:** analog or kernel-analog forecasting (e.g. Zhao & Giannakis), or a neural network trained with mean-squared error.
   - **Link to the emulators:** an ML emulator trained deterministically (MSE) *is* a conditional-mean estimator. Its under-dispersion, a flaw for ensemble forecasting, makes it a built-in filter of chaos that gives the expected forecast. The same emulator serves two roles: the conditional mean (deterministic), and the full forecast distribution once calibrated noise is added (phase 6).
   - Check against initialized ensembles: the conditional mean should match the ensemble mean x̄(t+τ) at every lead.

Other options, for later:
7. Spectral splitting of the dynamics: covariant Lyapunov vectors (unstable, neutral, stable), or Koopman and Perron–Frobenius eigenfunctions from DMD (Demaeyer, Penny & Vannitsem 2022).
8. Stochastic mode reduction (Majda–Timofeyev–Vanden-Eijnden): eliminate the fast modes analytically, replacing them with noise plus deterministic corrections.
9. Slow-manifold projection, the coupled analogue of normal-mode or digital-filter initialization in NWP.
10. Nudging or synchronizing the atmosphere toward a reference (Köhl et al. 2026). Useful for estimation and adjoints, not for forecasts.

### 5.1 The hybrid model: a linear stochastic atmosphere plus slow ocean dynamics
Split the state into slow variables s (ocean) and fast variables f (atmosphere):
- fast: df/dt = A_ff f + A_fs s + ξ, a damped linear atmosphere forced by the ocean, plus noise ξ;
- slow: ds/dt = L_ss s + L_sf f.

The atmosphere block can come from three sources:
- dynamically, from the tendency Jacobian A at the mean state with eddy damping (the one place derivatives fit);
- empirically, from a LIM fitted to (s, f) at short fitting lags;
- from ensemble propagators over days.

Because the atmosphere adjusts quickly, f ≈ −A_ff⁻¹ A_fs s + noise. Substituting gives a LIM for the ocean alone:
- **effective dynamics:** L_eff = L_ss − L_sf A_ff⁻¹ A_fs, the coupled feedback (Bjerknes-like) inside the ocean operator;
- **effective noise:** the atmosphere's stochastic forcing, filtered through L_sf.

This is Hasselmann's stochastic climate model, made formal (Mori–Zwanzig, or method 8). It is mainly for **interpretation**: it separates the predictability created by coupled feedback (in L_eff) from the noise that limits it, and identifies which atmospheric patterns matter.

## 6. Forecast error covariance, sensitivity and targeting, and what an ML emulator must supply (2026-11-22)
This section derives how forecast errors grow (linearly, then to saturation), how sensitive a forecast metric is to the state at an earlier time, and how much one observation reduces the metric's error variance. Then it says what an emulator must supply to do the same. Symbols are those of the notation table; the ones new here are defined in §6.1.

### 6.1 Symbols
| Symbol | Meaning |
|---|---|
| t, τ | start time and lead; x(τ) is short for x(t + τ) |
| τ_o, τ_v | lead of a candidate observation, and verification lead of the metric; τ_o < τ_v |
| Δτ | emulator step, 6 h, a whole number of physics steps Δt |
| d, p | state dimension; number of observations |
| K | ensemble size; members x_k(τ), k = 1 … K |
| x̄(τ), x′_k(τ) | ensemble mean; member deviation x′_k = x_k − x̄ |
| X(τ) | d × K matrix of scaled deviations, [x′_1 … x′_K] / √(K − 1) |
| Σ(τ), Σ(τ, 0) | forecast covariance X(τ) X(τ)ᵀ; lagged ensemble covariance X(τ) X(0)ᵀ |
| M(Δτ) | physics model run for one step Δτ (nonlinear) |
| M′(Δτ; x) | Jacobian of M(Δτ) about state x. It depends on x and on the forcing at that time (date, insolation, SST) |
| M′(τ_v, τ_o) | product of one-step Jacobians along x̄ from τ_o to τ_v (tangent-linear propagator) |
| J, J′_k | scalar metric of the state at τ_v; member deviation J′_k = J(x_k(τ_v)) − mean over k |
| j | K-vector [J′_1 … J′_K]ᵀ / √(K − 1) |
| z(τ) | adjoint vector (sensitivity) ∂J/∂x(τ) |
| y, H, R | observations at τ_o, y = H x(τ_o) + ε, with observation error ε of covariance R |
| Y | p × K matrix H X(τ_o): observed deviations |
| M̂(Δτ) | ML emulator of M(Δτ): a nonlinear network for one step, rolled out step by step |
| M̂′(Δτ; x) | Jacobian of M̂(Δτ) about x |
| x̂(τ) | single emulator forecast from the ensemble-mean start, x̂(0) = x̄(0) |
| T(x) | tangent network, trained so that T(x) ≈ M′(Δτ; x) |

### 6.2 Linear propagation of forecast errors
Each member obeys x_k(τ + Δτ) = M(Δτ)(x_k(τ)), with M written on anomalies (notation table): the model steps the full state x_clim + x_k. Expanding about the ensemble mean,

x_k(τ + Δτ) = M(Δτ)(x̄(τ)) + M′(Δτ; x̄(τ)) x′_k(τ) + O(|x′|²).

Averaging over k removes the linear term (the deviations sum to zero), so x̄(τ + Δτ) = M(Δτ)(x̄(τ)) + O(|x′|²). Subtracting,

(1) x′_k(τ + Δτ) = M′(Δτ; x̄(τ)) x′_k(τ),  i.e.  X(τ + Δτ) = M′ X(τ),

(2) Σ(τ + Δτ) = M′ Σ(τ) M′ᵀ,  and over many steps X(τ) = M′(τ, 0) X(0).

- Valid while the O(|x′|²) terms are negligible: about 1–2 weeks for the atmosphere.
- There is no added-noise term. The model is perfect and deterministic, so all spread comes from Σ(0).

**Saturation.** Beyond the linear window the members decorrelate, still each a realistic state on the model's attractor. The ensemble mean relaxes to the climatological mean, x̄(τ) → 0 (x is an anomaly), and the spread saturates at the climate covariance, Σ(τ) → C(0). This comes from the nonlinear M; the linear propagation (2) has no saturation and grows without bound. Predictability at long leads is the part of x̄(τ) that has not yet relaxed, and the part of C(0) − Σ(τ) that has not yet closed (the signal S(τ)).

### 6.3 Three propagators
1. **M′(τ, 0):** the exact Jacobian along x̄. Flow-dependent; acts on any direction.
2. **Σ(τ, 0) Σ(0)⁺** (⁺: pseudo-inverse): least-squares regression, across the K members, of the deviations at τ on the deviations at 0. By (1), Σ(τ, 0) Σ(0)⁺ = M′(τ, 0) X(0) X(0)ᵀ (X(0) X(0)ᵀ)⁺ = M′(τ, 0) Π(0), where Π(0) projects onto the K − 1 directions the ensemble spans. So the ensemble regression recovers the tangent-linear propagator inside the ensemble subspace.
3. **G(τ) = C(τ) C(0)⁻¹:** least-squares regression over the times of the nature run (Penland & Sardeshmukh 1995). It is state-independent, so Σ(τ) = G Σ(0) Gᵀ + N(τ) needs the noise term N(τ) for what a fixed linear operator misses. Equation (2) has none because M′ is the exact local linearization.

### 6.4 Forecast variance and sensitivity of a metric
Linearizing J about x̄(τ_v),

(3) J′_k = z(τ_v)ᵀ x′_k(τ_v),  with z(τ_v) = ∂J/∂x at x̄(τ_v),

(4) j = X(τ_v)ᵀ z(τ_v),  Var J = jᵀ j = z(τ_v)ᵀ Σ(τ_v) z(τ_v).

Substituting X(τ_v) = M′(τ_v, τ_o) X(τ_o),

(5) j = X(τ_o)ᵀ z(τ_o),  with z(τ_o) = M′(τ_v, τ_o)ᵀ z(τ_v).

- z(τ_o) is the **adjoint sensitivity**: the change in J per unit change of the state at τ_o. It is computed backwards one step at a time, z(τ − Δτ) = M′(Δτ; x̄(τ − Δτ))ᵀ z(τ).
- j can be found two ways: evaluate J on every member at τ_v (no adjoint), or use (5) (adjoint plus the error patterns at τ_o).
- **Ensemble sensitivity** (Ancell & Hakim 2007; Torn & Hakim 2008), the covariance across members of x(τ_o) with J, is X(τ_o) j = Σ(τ_o) z(τ_o): the adjoint sensitivity weighted by the error covariance at τ_o.

### 6.5 Variance reduction from an observation
Observe y = H x(τ_o) + ε at τ_o. In the linear regime, the least-squares (Kalman) update of the state at τ_o, propagated to τ_v by (1), reduces Var J by the variance of J explained by regression on y:

ΔVar J = cov(J, y) var(y)⁻¹ cov(y, J),  with cov(y, J) = Y j and var(y) = Y Yᵀ + R,

(6) ΔVar J = jᵀ Yᵀ (Y Yᵀ + R)⁻¹ Y j.

- For one observation of a variable hᵀx: ΔVar J = (hᵀ X(τ_o) j)² / (hᵀ Σ(τ_o) h + R). Observe where the error covariance with J is large compared with the observed variable's total variance.
- **Targeting** chooses H (where and what) and τ_o (when) to maximize ΔVar J (Bishop et al. 2001).
- **Beyond the linear window.** (6) needs only X(τ_o) and j, the member deviations, not M′. From a nonlinear ensemble it is the least-squares regression of J on y across members, and stays meaningful past the linear window while that regression is close to linear. The adjoint form (5) does not, because it needs M′.

### 6.6 What an emulator must supply
An emulator replaces the physics ensemble with K runs of M̂ from the same K initial states. Each run is a nonlinear trajectory. The ensemble is right if:
1. **early growth is right**, M̂′ ≈ M′, so that (1)–(2) hold with M̂′ in the linear window;
2. **saturation is right**, Σ(τ) → C(0), so the emulator's climate matches the physics.

Then X(τ), j and (6) come from the emulator ensemble at all leads. The adjoint z(τ) of (5), in the linear window, needs M′ᵀ along a single trajectory, x̂ ≈ x̄ (from x̄(τ) = M(τ)(x̄(0)) + O(|x′|²)).

- **Known failure:** M̂′ is too contractive (Selz & Craig 2023; Bonavita 2024). Then X, Σ, j and ΔVar are too small and the patterns wrong. A single deterministic forecast of such an emulator, run long, blurs toward climatology; a good emulator ensemble keeps realistic members and saturates. Stochastic emulators (GenCast, AIFS-CRPS) are built to do that.
- **Why it happens:** a deterministic model's next state is a deterministic function of its full current state, so a one-step MSE fit to full model states can in principle learn M. Damping comes from targets that the inputs do not determine:
  - training data that lack small scales (analyses);
  - multi-step losses at leads beyond the predictability limit, where the MSE optimum is the smooth conditional mean;
  - networks' bias toward large scales.

**Training M̂ on values and derivatives.** Add Jacobian products to the one-step loss: inputs a state x_k(τ), its forcing, and a direction v (columns of X(τ), the directions that make up Σ); targets M(Δτ)(x_k), M′ v and M′ᵀ v, computed with Enzyme. This constrains M̂′ ≈ M′ along the attractor, which is what sets error growth. Then M̂′ supplies the adjoint directly.

**Tangent network T ≈ M′** (fallback): a separate network trained on the same Jacobian products, used if M̂ itself cannot be made to match M′. Use: X(τ + Δτ) = T(x̂(τ)) X(τ), z(τ − Δτ) = T(x̂(τ − Δτ))ᵀ z(τ). Caveat: x̂ may be smoother than physics states, so add linearizations about x̂ to its training data.

**Perfect-model tests** for any emulator, against the physics:
- one-step error, and Jacobian error ‖M̂′ v − M′ v‖;
- leading singular values and vectors over 1–5 days, and finite-time Lyapunov exponents;
- ensemble spread Σ(τ) in both phases, and the climate C(0);
- z against Enzyme adjoints of the physics; ΔVar J against OSSEs.

### 6.7 First implementation
- **Network:** M̂ (and T, if needed) is a spherical Fourier neural operator (SFNO; Bonev et al. 2023), whose convolutions act on spherical-harmonic coefficients, like SpeedyWeather.
- **Configuration:** atmosphere only, SST prescribed from the nature run, T31 with 8 levels, Δτ = 6 h.
  - Variables: vorticity, divergence, temperature, humidity, log surface pressure.
  - Training data: about 10⁵ six-hourly states from decades of nature run.
- **Software:** Julia (Lux.jl, SpeedyTransforms, Enzyme, Reactant).
- **Timing (open):** after the nature run (phase 2), alongside the ensembles (phase 3).
- **Later:** the emulated atmosphere coupled to the physical Oceananigans ocean.

### 6.8 Experiment: which emulator deficiencies bias predictability and targeting
In this perfect-model setting M is known, with its derivatives. So:
1. **Reference emulator:** train M̂ one step at a time on full states, values plus Jacobian products (§6.6). Check that it reproduces the physics statistics: one-step and Jacobian errors, Lyapunov exponents, Σ(τ) in both phases, and C(0). Trajectories still diverge after the predictability time; the target is statistics, not trajectories.
2. **Degraded emulators,** each mimicking a known cause:
   - no derivative loss (one-step values only);
   - multi-step MSE loss out to lead L, for L below and beyond the predictability time;
   - withheld state: truncated spectrum (T21 from T31, like analyses without small scales) or dropped variables or levels. The withheld part becomes hidden dynamics, so a deterministic M̂ cannot be exact: it needs noise or memory (§5, method 8);
   - noisy training inputs (analysis error);
   - less data or a smaller network.
3. **Measure, for each emulator against the physics ensembles:** the diagnostics of §6.6; the predictability estimates (predictable components, average predictability time, §5); and the targeting quantities z, ensemble sensitivity and ΔVar J.

Result: a map from each deficiency to its bias in spread, in estimated predictability, and in targeting. In particular, whether under-dispersion makes an emulator overstate predictability (Selz & Craig 2023).

Costs: each training sample stores a state plus a few Jacobian products, several times the storage of states alone. Start with the atmosphere under prescribed SST.

References: `notes/reading_list.md`.

## 7. Emulator experiment: the 6 h map, its derivatives, and the data pipeline (proposed 2026-11-22)
Decisions (2026-11-22): SpeedyWeather 0.25.0, with NumericalEarth pinned to its main branch (commit 83e1c68, which adds 0.25 compat; return to a release ≥ 0.9.5 when out); PrimitiveWet with reduced physics, as in `drivers/reactant_atmosphere_check.jl` (no convection, constant longwave transmissivity), whose Reactant gradient is verified; aquaplanet (`AquaPlanetMask`, `AquaPlanet` ocean: prescribed, zonally symmetric, time-independent SST, (302 − 273 K) cos²φ + 273 K; `PrescribedSeaIce`; default `LandModel` kept, since `DryLandModel` lacks the snow depth the wet surface fluxes need, and the land fraction is 0); perpetual equinox with no daily cycle (`Earth(; seasonal_cycle = false, daily_cycle = false)`), so M(Δτ) is autonomous and depends on the state alone; truncation 32 (classic T31), 8 layers, Float64 for the physics; Lux.jl for the network.

### 7.1 The 6 h map M(Δτ)
- SpeedyWeather steps with leapfrog: x(t + Δt) depends on two time levels, x(t) and x(t − Δt). The emulator sees one snapshot every Δτ = 6 h = 9 steps of Δt = 40 min.
- **Definition:** M(Δτ)(x) = start from the single level x with a forward-Euler first step, then leapfrog to 6 h. This is an exact function of one snapshot, and the Euler start also removes the leapfrog computational mode.
- **Restart in-process** (SpeedyWeather 0.24.1 source; recheck in 0.25): copy prognostic time level 2 (current) into level 1, then reinitialize the time stepping (`initialize!(sim; steps = 10)`, which recomputes no model constants). The next 10 steps are an Euler half step Δt/2, a leapfrog step Δt, then filtered leapfrog steps 2Δt; together they advance 9Δt = 6 h. The date carries across restarts. Caveat (found 2026-11-22): a freshly initialized model holds its initial conditions in level 1 and zeros in level 2, so the first segment must run without the 2 → 1 copy.
- **Nature run:** one continuous process. Every 6 h it writes the snapshot (when due) and restarts the leapfrog from that single level with an Euler step, without stopping. So the nature run, the training targets and the Jacobian products all come from the same map M(Δτ).

### 7.2 Jacobian products
- **Adjoint products (VJPs), M′ᵀ w:** Reactant reverse mode, as verified in `drivers/reactant_atmosphere_check.jl`.
- **Tangent-linear products (JVPs), M′ v:** centred finite differences of M(Δτ) in Float64, [M(x + εv) − M(x − εv)] / 2ε, with ε chosen from a convergence scan. Forward mode under Reactant is untested for SpeedyWeather, and over 6 h the map is close to linear.
- **What is differentiated:** x is placed in time level 1; the differentiated function runs the grid refresh `transform!(vars, model; initialize = true)` (which copies level 1 → 2 and computes the grid fields from x) and then the 10 time steps. So derivatives flow through the grid fields of the first step, and the gradient with respect to x is the gradient with respect to level 1. The clock counters are reset outside, by `initialize!(sim; steps = 10)`.
- **Check:** the dot-product test wᵀ(M′ v) = (M′ᵀ w)ᵀ v for random v, w, to near the finite-difference truncation error.

### 7.3 Pipeline (drivers write NetCDF; separate jobs can run concurrently)
1. **Derivative check** (`drivers/speedy_map_check.jl`; written 2026-11-22; a CPU smoke test gave 0.28 s per 6 h map on the Mac, so 40 years of nature run is about 4.5 h): build M(Δτ); time one 6 h map on CPU; ε scan for the JVP; dot-product test against the Reactant VJP; Reactant compile time and memory.
2. **Nature run** (`drivers/speedy_nature_run.jl`): spin-up, then decades of 6-hourly spectral prognostic snapshots (vorticity, divergence, temperature, humidity, log surface pressure), one NetCDF file per year, with the date for the forcing.
3. **Jacobian samples** (`drivers/speedy_jacobian_samples.jl`): reads finished years; at sampled states, directions v from short perturbed ensembles plus random vectors; writes (x, v, M′ v) and (x, w, M′ᵀ w). Runs alongside step 2.
4. **SFNO and training** (`src/emulator/`, `drivers/train_emulator.jl`): GPU job, after enough years exist; held-out years for testing.
