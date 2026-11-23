# Toy-model nature run: a long free run of ToyCoupled (Lorenz-96 + slab ocean + recharge
# oscillator), saved daily to NetCDF with provenance. This is the "truth" for the toy OSSEs and
# the run in which the toy's ENSO period and amplitude are checked and tuned.
#
#   mkdir -p run/toy-nature && cp drivers/toy_nature_run.jl run/toy-nature/   # then edit Settings
#   cd run/toy-nature && julia --project=../.. toy_nature_run.jl
#
# Plot it with plots/toy_nature_run.ipynb.

#####
##### Settings
#####

years        = 50          # length of the saved run (model years of 365 days)
spinup_years = 10          # discarded spin-up
save_every   = 4           # steps between saved states (4 × 6 h = daily)
seed         = 2026        # random seed for the initial state
outfile      = "toy_nature_run.nc"

# Model parameters: any field of ToyParams, e.g. (R = 0.002, γ = 0.02, a = 0.022); empty for defaults
param_overrides = (;)

#####
##### Setup
#####

using PredictableOcean
using Random

params = ToyParams(; param_overrides...)
m = ToyCoupled(params)
day = 0.2                                  # model time units per day (1 unit ≈ 5 days)
steps_per_day = round(Int, day / timestep(m))
steps_per_year = 365 * steps_per_day

x0 = initial_state(m; rng = Xoshiro(seed))
@info "spin-up" spinup_years
xs = propagate(m, x0, spinup_years * steps_per_year)

nsteps = years * steps_per_year
@info "nature run" years nsteps
X = trajectory(m, xs, nsteps; every = save_every)
t = (0:(size(X, 2) - 1)) .* (save_every / steps_per_day)   # days since start

attrs = merge(provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                         title = "ToyCoupled nature run"),
              parameter_attributes(m),
              Dict{String, Any}("spinup_years" => spinup_years, "seed" => seed))
isfile(outfile) && rm(outfile)
write_trajectory(outfile, m, t, X; attributes = attrs)
@info "wrote" outfile size(X)
