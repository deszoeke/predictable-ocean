# Enzyme/Reactant tutorial on the toy model: sensitivities of ENSO (Tₑ) and of one slab-ocean
# sector (T₁, a marine-heat-wave-like metric) to the initial state, at leads of 2 days to 1 year;
# correctness checks (dot-product, finite differences, Reactant vs plain Enzyme); and how long
# the tangent-linear model stays valid. Explanations: docs/tutorial_enzyme_reactant.md.
#
#   mkdir -p run/tutorial-ad && cp drivers/tutorial_ad.jl run/tutorial-ad/
#   cd run/tutorial-ad && julia --project=../.. tutorial_ad.jl
#
# Plot with plots/tutorial_ad_pluto.jl.

#####
##### Settings
#####

leads        = [8, 56, 240, 1460]     # steps of 6 h: 2 days, 2 weeks, 2 months, 1 year
nstarts      = 20                     # initial states (one per year of a free run)
amplitudes   = [0.01, 0.1, 0.3]       # TLM test: perturbation size / climatological std
nperturb     = 10                     # random perturbations per start and amplitude
use_reactant = true                   # also compile with Reactant and compare
seed         = 2026
outfile      = "tutorial_ad.nc"

#####
##### Setup
#####

using PredictableOcean
using Enzyme, LinearAlgebra, Random, Statistics, NCDatasets
use_reactant && using Reactant

m = ToyCoupled()
N = state_length(m)
s = state_slices(m)
steps_per_year = 1460
metrics = [Component(s.Te), Component(s.ocean[1:1])]
metric_names = ["Te", "T1"]

# initial states: one per year of a free run after spin-up; climatological std per variable
rng = Xoshiro(seed)
x = propagate(m, initial_state(m; rng), 10 * steps_per_year)
clim = trajectory(m, x, 20 * steps_per_year; every = 4)
σclim = vec(std(clim; dims = 2))
starts = [clim[:, 1 + (k - 1) * 365] for k in 1:nstarts]

nl, nm = length(leads), length(metrics)
grad = zeros(N, nstarts, nl, nm)            # ∂J(M(x₀))/∂x₀
tlm_relerr = zeros(length(amplitudes), nl, nm)
dot_test = zeros(nl); fd_relerr = zeros(nl); reactant_relerr = fill(NaN, nl)
t_enzyme = zeros(nl); t_reactant_compile = fill(NaN, nl); t_reactant_run = fill(NaN, nl)

for (il, n) in enumerate(leads)
    @info "lead" n days = n / 4
    # 1. sensitivities of each metric to the initial state (plain Enzyme, reverse mode)
    for (im, J) in enumerate(metrics), (is, x0) in enumerate(starts)
        t = @elapsed g = gradient_J(J, m, x0, n)
        grad[:, is, il, im] .= g
        (im == 1 && is == 2) && (t_enzyme[il] = t)       # a timing after the first compile
    end

    # 2. checks at the first start: dot-product test and finite differences
    x0 = starts[1]
    v = randn(rng, N); λ = randn(rng, N)
    Lv = jvp(m, x0, v, n)
    dot_test[il] = abs(dot(Lv, λ) - dot(v, vjp(m, x0, λ, n))) / abs(dot(Lv, λ))
    ε = 1e-7
    fd = (propagate(m, x0 .+ ε .* v, n) .- propagate(m, x0 .- ε .* v, n)) ./ (2ε)
    fd_relerr[il] = norm(fd - Lv) / norm(Lv)

    # 3. Reactant: compile the same gradient (n is baked into the compiled code) and compare
    if use_reactant
        J = metrics[1]
        mr, xr = Reactant.to_rarray(m), Reactant.to_rarray(x0)
        t_reactant_compile[il] = @elapsed (g_c = Reactant.@compile gradient_J(J, mr, xr, n, true))
        g_r = Array(g_c(J, mr, xr, n, true))
        t_reactant_run[il] = @elapsed g_c(J, mr, xr, n, true)
        reactant_relerr[il] = norm(g_r - grad[:, 1, il, 1]) / norm(grad[:, 1, il, 1])
    end

    # 4. TLM validity: nonlinear change of J against the linear prediction g ⋅ δx
    for (ia, a) in enumerate(amplitudes), (im, J) in enumerate(metrics)
        nl_, lin = Float64[], Float64[]
        for (is, x0) in enumerate(starts), _ in 1:nperturb
            δ = a .* σclim .* randn(rng, N)
            push!(nl_, J(propagate(m, x0 .+ δ, n)) - J(propagate(m, x0, n)))
            push!(lin, dot(grad[:, is, il, im], δ))
        end
        tlm_relerr[ia, il, im] = sqrt(mean(abs2, nl_ .- lin)) / sqrt(mean(abs2, nl_))
    end
end

#####
##### Output
#####

attrs = merge(provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                         title = "AD tutorial: toy-model sensitivities and checks"), parameter_attributes(m))
isfile(outfile) && rm(outfile)
NCDataset(outfile, "c") do ds
    for (k, v) in attrs
        ds.attrib[k] = v
    end
    defVar(ds, "lead", leads, ("lead",); attrib = Dict("units" => "steps of 6 h", "long_name" => "lead time"))
    defVar(ds, "lead_days", leads ./ 4, ("lead",); attrib = Dict("units" => "days"))
    defVar(ds, "amplitude", amplitudes, ("amplitude",); attrib = Dict("long_name" => "perturbation std / climatological std"))
    defVar(ds, "state", collect(1:N), ("state",); attrib = Dict("long_name" => "state index: X 1:36, T 37:72, Te 73, h 74"))
    defVar(ds, "start", collect(1:nstarts), ("start",))
    defVar(ds, "metric", metric_names, ("metric",))
    defVar(ds, "sigma_clim", σclim, ("state",); attrib = Dict("long_name" => "climatological std of each state variable"))
    defVar(ds, "grad", grad, ("state", "start", "lead", "metric");
           attrib = Dict("long_name" => "dJ(M(x0))/dx0, sensitivity of the forecast metric to the initial state"))
    defVar(ds, "tlm_relerr", tlm_relerr, ("amplitude", "lead", "metric");
           attrib = Dict("long_name" => "rms(nonlinear dJ - linear dJ) / rms(nonlinear dJ)"))
    defVar(ds, "dot_test", dot_test, ("lead",); attrib = Dict("long_name" => "relative error of <Lv,λ> = <v,L'λ>"))
    defVar(ds, "fd_relerr", fd_relerr, ("lead",); attrib = Dict("long_name" => "relative error of JVP vs centered finite difference, eps=1e-7"))
    defVar(ds, "reactant_relerr", reactant_relerr, ("lead",); attrib = Dict("long_name" => "relative difference of Reactant vs plain-Enzyme gradient of Te"))
    defVar(ds, "t_enzyme", t_enzyme, ("lead",); attrib = Dict("units" => "s", "long_name" => "plain Enzyme gradient time (after compile)"))
    defVar(ds, "t_reactant_compile", t_reactant_compile, ("lead",); attrib = Dict("units" => "s"))
    defVar(ds, "t_reactant_run", t_reactant_run, ("lead",); attrib = Dict("units" => "s"))
end
@info "wrote" outfile
for (il, n) in enumerate(leads)
    println("lead $(n ÷ 4) d: dot $(round(dot_test[il], sigdigits = 2)), FD $(round(fd_relerr[il], sigdigits = 2)), ",
            "Reactant $(round(reactant_relerr[il], sigdigits = 2)); TLM rel err (Te) by amplitude: ",
            round.(tlm_relerr[:, il, 1], sigdigits = 2), "; Enzyme $(round(t_enzyme[il], sigdigits = 2)) s, ",
            "Reactant compile $(round(t_reactant_compile[il], sigdigits = 2)) s, run $(round(t_reactant_run[il], sigdigits = 2)) s")
end
