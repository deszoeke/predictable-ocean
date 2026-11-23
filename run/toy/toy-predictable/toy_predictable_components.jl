# Predictable components of the toy model: eigenvectors of Σ(τ) C(0)⁻¹ (Schneider & Griffies),
# measured from ensembles (ground truth) and predicted by a LIM fitted to the nature run.
# Notation: docs/technical.md (Notation).
#
#   mkdir -p run/toy-predictable && cp drivers/toy_predictable_components.jl run/toy-predictable/
#   cd run/toy-predictable && julia -t auto --project=../.. toy_predictable_components.jl
#
# Plot with plots/predictable_components_pluto.jl.

#####
##### Settings
#####

leads_days   = [14, 60, 182, 365, 730]    # lead τ (days)
nstarts      = 20                         # start times t (spaced 2.5 years apart in the nature run)
nmembers     = 50                         # members per start
atm_err      = 0.1                        # Σ(0): atmosphere std, as a fraction of s
slow_err     = 0.05                       # Σ(0): ocean, Tₑ, h std, as a fraction of s
nature_years = 50                         # nature run for C(0) and the LIM
lim_lags     = [5, 30]                    # LIM fitting lags ℓ (days): the τ-test
lim_space    = :slow                      # LIM state: :slow (ocean, Tₑ, h) — a LIM cannot fit the chaotic atmosphere
nkeep        = 10                         # components saved
seed         = 2026
outfile      = "toy_predictable_components.nc"

#####
##### Setup
#####

using PredictableOcean
using LinearAlgebra, Random, Statistics, NCDatasets

m = ToyCoupled()
N = state_length(m)
spd = 4                                   # steps per day
rng = Xoshiro(seed)

# nature run (daily): C(0), s, and the LIMs
x = propagate(m, initial_state(m; rng), 10 * 365 * spd)
Y = trajectory(m, x, nature_years * 365 * spd; every = spd)
Ya = Y .- mean(Y; dims = 2)
C = Ya * Ya' ./ (size(Ya, 2) - 1)
s = sqrt.(diag(C))
iL = lim_space === :slow ? collect(slow_indices(m)) : collect(1:N)   # LIM subspace
Ls = [fit_lim(Y[iL, :], ℓ) for ℓ in lim_lags]    # per day, in the LIM subspace
CL = C[iL, iL]

# ensembles: starts spaced through the nature run; Σ(0) diagonal
Σ0 = Diagonal((vcat(fill(atm_err, m.params.K), fill(slow_err, N - m.params.K)) .* s) .^ 2)
starts = [Y[:, 1 + (k - 1) * round(Int, 2.5 * 365)] for k in 1:nstarts]
nl = length(leads_days)
ens = [Vector{Matrix{Float64}}(undef, nstarts) for _ in 1:nl]
for (is, x0) in enumerate(starts)
    E = zeros(N, nmembers, nl)
    Threads.@threads for k in 1:nmembers
        xk = x0 .+ sqrt.(diag(Σ0)) .* randn(Xoshiro(seed + 1000is + k), N)
        t = 0
        for (il, d) in enumerate(leads_days)
            xk = propagate(m, xk, (d - t) * spd); t = d
            E[:, k, il] .= xk
        end
    end
    for il in 1:nl
        ens[il][is] = E[:, :, il]
    end
end

γ_ens = zeros(N, nl); p_ens = zeros(N, nkeep, nl); q_ens = zeros(N, nkeep, nl); info_ens = zeros(nl)
nL = length(iL)
γ_sub = zeros(nL, nl); p_sub = zeros(nL, nkeep, nl); info_sub = zeros(nl)   # ensemble, in the LIM subspace
γ_lim = zeros(nL, nl, length(lim_lags)); p_lim = zeros(nL, nkeep, nl, length(lim_lags)); info_lim = zeros(nl, length(lim_lags))
for il in 1:nl
    Σ = pooled_forecast_covariance(ens[il])
    pc = predictable_components(Σ, C)
    γ_ens[:, il] .= pc.γ; p_ens[:, :, il] .= pc.p[:, 1:nkeep]; q_ens[:, :, il] .= pc.q[:, 1:nkeep]
    info_ens[il] = predictive_information(pc.γ)
    ps = predictable_components(Σ[iL, iL], CL)            # ensemble truth, same subspace as the LIM
    γ_sub[:, il] .= ps.γ; p_sub[:, :, il] .= ps.p[:, 1:nkeep]; info_sub[il] = predictive_information(ps.γ)
    for (j, L) in enumerate(Ls)
        ΣL = Symmetric(lim_forecast_covariance(L, CL, Matrix(Σ0)[iL, iL], leads_days[il]))
        pl = predictable_components(ΣL, CL)
        γ_lim[:, il, j] .= pl.γ; p_lim[:, :, il, j] .= pl.p[:, 1:nkeep]
        info_lim[il, j] = predictive_information(pl.γ)
    end
    println("lead $(leads_days[il]) d: ensemble γ₁..₃ = ", round.(γ_ens[1:3, il], digits = 3),
            ", predictive info = ", round(info_ens[il], digits = 2),
            " | $(lim_space) subspace: ensemble γ₁..₃ = ", round.(γ_sub[1:3, il], digits = 3),
            "; LIM(ℓ=$(lim_lags[1]) d) γ₁..₃ = ", round.(γ_lim[1:3, il, 1], digits = 3),
            ", LIM(ℓ=$(lim_lags[2]) d) γ₁..₃ = ", round.(γ_lim[1:3, il, 2], digits = 3))
end

# leading-pattern agreement: |correlation| of ensemble and LIM leading patterns (sign is arbitrary)
patcor = [abs(cor(p_sub[:, 1, il], p_lim[:, 1, il, j])) for il in 1:nl, j in eachindex(lim_lags)]
println("|corr| of leading patterns in the $(lim_space) subspace, ensemble vs LIM(ℓ = $(lim_lags)): ", round.(patcor, digits = 2))
for il in 1:nl
    println("lead $(leads_days[il]) d: predictive info, $(lim_space) subspace: ensemble ", round(info_sub[il], digits = 2), ", LIM ", round.(info_lim[il, :], digits = 2))
end

# sampling noise floor for γ (Marchenko–Pastur lower edge): pooled Σ has ν = nstarts (nmembers − 1)
# degrees of freedom in dimension d; pure noise gives γ down to about (1 − √(d/ν))²
ν = nstarts * (nmembers - 1)
println("noise floor for γ: full state ", round((1 - sqrt(N / ν))^2, digits = 2), ", slow subspace ", round((1 - sqrt(nL / ν))^2, digits = 2))

# where each leading component lives: fraction of its s-normalized pattern energy in each part
parts = (atmosphere = state_slices(m).atmosphere, ocean = state_slices(m).ocean, enso = vcat(collect(state_slices(m).Te), collect(state_slices(m).h)))
for il in 1:nl, c in 1:3
    e = (p_ens[:, c, il] ./ s) .^ 2
    f = map(ix -> round(sum(e[ix]) / sum(e), digits = 2), parts)
    println("lead $(leads_days[il]) d, component $c (γ = $(round(γ_ens[c, il], digits = 3))): energy fraction ", f)
end

attrs = merge(provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                         title = "Toy predictable components: eigenvectors of Sigma(tau) C(0)^-1, ensemble vs LIM"),
              parameter_attributes(m))
isfile(outfile) && rm(outfile)
NCDataset(outfile, "c") do ds
    for (k, v) in attrs
        ds.attrib[k] = v
    end
    defVar(ds, "lead_days", leads_days, ("lead",); attrib = Dict("units" => "days", "long_name" => "lead tau"))
    defVar(ds, "lim_lag_days", lim_lags, ("lim_lag",); attrib = Dict("units" => "days", "long_name" => "LIM fitting lag l"))
    defVar(ds, "state", collect(1:N), ("state",); attrib = Dict("long_name" => "state index: X 1:36, T 37:72, Te 73, h 74"))
    defVar(ds, "s", s, ("state",); attrib = Dict("long_name" => "nature-run standard deviation sqrt(diag C(0))"))
    defVar(ds, "gamma_ens", γ_ens, ("component", "lead"); attrib = Dict("long_name" => "eigenvalues of Sigma(tau) C(0)^-1 from ensembles: unpredictable fraction"))
    defVar(ds, "lim_state", iL, ("lim_component_state",); attrib = Dict("long_name" => "state indices of the LIM subspace"))
    defVar(ds, "gamma_sub", γ_sub, ("lim_component", "lead"); attrib = Dict("long_name" => "ensemble eigenvalues of Sigma C(0)^-1 restricted to the LIM subspace"))
    defVar(ds, "pattern_sub", p_sub, ("lim_component_state", "kept", "lead"))
    defVar(ds, "predictive_info_sub", info_sub, ("lead",))
    defVar(ds, "gamma_lim", γ_lim, ("lim_component", "lead", "lim_lag"); attrib = Dict("long_name" => "eigenvalues from the LIM-predicted Sigma, LIM subspace"))
    defVar(ds, "pattern_ens", p_ens, ("state", "kept", "lead"); attrib = Dict("long_name" => "predictable-component patterns p = C(0) q (ensemble)"))
    defVar(ds, "filter_ens", q_ens, ("state", "kept", "lead"); attrib = Dict("long_name" => "filters q, C(0)-normalized"))
    defVar(ds, "pattern_lim", p_lim, ("lim_component_state", "kept", "lead", "lim_lag"))
    defVar(ds, "predictive_info_ens", info_ens, ("lead",); attrib = Dict("long_name" => "-1/2 sum log gamma"))
    defVar(ds, "predictive_info_lim", info_lim, ("lead", "lim_lag"))
    defVar(ds, "leading_pattern_abs_corr", patcor, ("lead", "lim_lag"))
end
@info "wrote" outfile
