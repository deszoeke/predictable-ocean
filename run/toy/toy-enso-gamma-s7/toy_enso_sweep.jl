# Sweep the toy ENSO parameters (wind-burst forcing σ and Bjerknes growth R) and record the
# ENSO period, amplitude and irregularity (coefficient of variation of cycle lengths) of each run.
# Used to tune ToyParams defaults.
#
#   mkdir -p run/toy-enso-sweep && cp drivers/toy_enso_sweep.jl run/toy-enso-sweep/
#   cd run/toy-enso-sweep && julia --project=../.. toy_enso_sweep.jl

#####
##### Settings
#####

σs           = [0.03]
Rs           = [-0.006]
years        = 300                               # per run, after spin-up
spinup_years = 10
seed         = 7
outfile      = "toy_enso_sweep.nc"
other_params = (γ = 0.0156,)                               # other ToyParams overrides held fixed

#####
##### Setup
#####

using PredictableOcean
using PredictableOcean: cycle_stats, monthly_means
using Random, NCDatasets

day = 0.2
m0 = ToyCoupled(ToyParams(; other_params...))
spd = round(Int, day / timestep(m0))           # steps per day
spm = 30 * spd                                 # steps per (30-day) month
spy = 365 * spd

res = Dict(k => fill(NaN, length(σs), length(Rs)) for k in ("cycle_mean", "cycle_std", "cycle_cv", "Te_std", "ncycles"))
for (i, σ) in enumerate(σs), (j, R) in enumerate(Rs)
    m = ToyCoupled(ToyParams(; other_params..., σ, R))
    xs = propagate(m, initial_state(m; rng = Xoshiro(seed)), spinup_years * spy)
    X = trajectory(m, xs, years * spy; every = spd)            # daily
    Te = X[state_slices(m).Te[1], :]
    Tm = monthly_means(Te, 30)
    c = cycle_stats(Tm, 30 / 365)
    res["cycle_mean"][i, j] = c.mean
    res["cycle_std"][i, j] = c.std
    res["cycle_cv"][i, j] = c.cv
    res["Te_std"][i, j] = sqrt(sum(abs2, Te .- sum(Te) / length(Te)) / (length(Te) - 1))
    res["ncycles"][i, j] = length(c.lengths)
    println("σ = $σ, R = $R: cycle = $(round(c.mean, digits = 2)) ± $(round(c.std, digits = 2)) yr, ",
            "cv = $(round(c.cv, digits = 3)), std Tₑ = $(round(res["Te_std"][i, j], digits = 3))")
end

attrs = merge(provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                         title = "ToyCoupled ENSO parameter sweep"), parameter_attributes(m0))
isfile(outfile) && rm(outfile)
NCDataset(outfile, "c") do ds
    for (k, v) in attrs
        ds.attrib[k] = v
    end
    defVar(ds, "sigma", σs, ("sigma",); attrib = Dict("long_name" => "wind-burst forcing σ"))
    defVar(ds, "R", Rs, ("R",); attrib = Dict("long_name" => "Bjerknes growth R"))
    units = Dict("cycle_mean" => "years", "cycle_std" => "years", "cycle_cv" => "1", "Te_std" => "K", "ncycles" => "1")
    for (k, v) in res
        defVar(ds, k, v, ("sigma", "R"); attrib = Dict("units" => units[k]))
    end
end
@info "wrote" outfile
