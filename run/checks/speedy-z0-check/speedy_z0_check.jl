# How much of the 6 h map does the carried-over surface roughness decide? (docs/technical.md §7), SpeedyWeather 0.25.
# The nature run's 6 h map is M(x, z₀): SpeedyWeather's LearnedWindSlowdown
# (src/parameterizations/surface_fluxes/surface_condition.jl:190) reads vars.parameterizations.momentum_roughness
# from the previous step, so the first step of each map (the Euler reset) sees the z₀ left by the previous map.
# If that z₀ is 0, it falls back to a fixed z₀ (1e-4 m over ocean; land fraction is 0 on the aquaplanet).
# For nstates snapshots x_i of one nature-run year, with z₀_i the roughness stored with x_i, this driver computes
#   A = M(x_i, z₀_i)        as in the nature run;
#   B = M(x_i, z₀*(x_i))    z₀* from x_i itself: one Euler step from x_i with z₀ = 0, then read the roughness;
#   C = M(x_i, 0)           fixed z₀ in the first step: a map of x alone;
# and, per field, with Δ = A − x_i the 6 h change, r_AB = ‖A − B‖/‖Δ‖ and r_AC = ‖A − C‖/‖Δ‖, the part δ of the
# output that z₀ adds and x does not determine. For reference also ‖Δ‖/‖x_i‖ and ‖z₀_i − z₀*‖/‖z₀_i‖.
# Norms are over all spectral coefficients (all layers) of one field.
#
# Logs: speedy_z0_check.log holds the driver's own messages (one line per state, one summary line per field)
# and, on failure, a short error summary. Everything else goes to speedy_z0_check_full.log through the redirect.
#
#   mkdir -p run/speedy-z0-check && cp drivers/speedy_z0_check.jl drivers/speedy_aquaplanet.jl run/speedy-z0-check/
#   cd run/speedy-z0-check && caffeinate -i julia --project=../.. speedy_z0_check.jl > speedy_z0_check_full.log 2>&1
#   (in another terminal: tail -f run/speedy-z0-check/speedy_z0_check.log)

#####
##### Settings
#####

year        = 11           # nature-run year to sample
nstates     = 20           # snapshots, spread evenly over that year
truncation  = 32           # SpeedyWeather's 32 = classic T31; Δt = 40 min
nlayers     = 8
steps_map   = 10           # Euler Δt/2 + leapfrog Δt + 8 × 2Δt leapfrog: advances 9Δt = 6 h
config      = "T31L8-aqua"
natdir      = "../speedy-nature"    # nature-run directory, relative to this run directory
outfile     = "speedy_z0_check.nc"
logfile     = "speedy_z0_check.log"
fulllog     = "speedy_z0_check_full.log"   # set by the shell redirect

#####
##### Setup
#####

using SpeedyWeather
using PredictableOcean: provenance, settings_block
using Dates, LinearAlgebra, NCDatasets, Printf

# zenith-kernel patch, FIELDS, create_model, level, field, snapshot, set_level1!, map6h!, note, error_summary
include(joinpath(@__DIR__, "speedy_aquaplanet.jl"))

const NCNAMES = ("vor", "div", "T", "q", "lnps")   # nature-run variable names, in the order of FIELDS

roughness(sim) = sim.variables.parameterizations.momentum_roughness

"Snapshot `it` of the nature-run file as Float64 complex arrays (in the order of FIELDS), and its roughness z₀."
function read_state(ds, it)
    x = map(NCNAMES) do name
        re, im = ds["$(name)_re"], ds["$(name)_im"]
        ndims(re) == 3 ? complex.(Float64.(re[:, :, it]), Float64.(im[:, :, it])) :
                         complex.(Float64.(re[:, it]), Float64.(im[:, it]))
    end
    return x, Float64.(ds["momentum_roughness"][:, it])
end

"Reset the variables to the reference (nothing carries over), insert x in level 1 and z₀ as the roughness."
function reset_vars!(sim, vars_ref, x, z0)
    SpeedyWeather.copy!(sim.variables, vars_ref)    # dest, src: variables only, not the model
    set_level1!(sim.variables, x)
    copyto!(roughness(sim), z0)
    return sim
end

"M(x, z₀): the 6 h map from x with z₀ present before the first (Euler) step."
function apply_M(sim, vars_ref, x, z0)
    reset_vars!(sim, vars_ref, x, z0)
    map6h!(sim, steps_map)
    return snapshot(sim.variables)
end

"z₀*(x): the roughness after a single Euler step from x, started with z₀ = 0 (fixed-z₀ fallback)."
function z0_star(sim, vars_ref, x)
    reset_vars!(sim, vars_ref, x, zeros(length(roughness(sim))))
    map6h!(sim, 1)
    return Float64.(vec(Array(roughness(sim))))
end

relnorm(a, b) = norm(a) / norm(b)
median_of(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

#####
##### Main
#####

function main(log)
    natfile = joinpath(natdir, @sprintf("nature_%s_y%04d.nc", config, year))
    sim = initialize!(create_model(SpeedyWeather.CPU(); truncation, nlayers))
    map6h!(sim, steps_map)                 # compile and fill every variable once
    vars_ref = deepcopy(sim.variables)     # reference: every map starts from these variables
    nfield = length(FIELDS)

    ds = NCDataset(natfile)
    nmaps = size(ds["vor_re"], 3)
    idx = round.(Int, range(1, nmaps; length = nstates))   # time indices of the sampled states
    note(log, "model"; config, truncation, nlayers, steps_map, threads = Threads.nthreads())
    note(log, "nature file"; natfile, nmaps, idx = idx)

    r_AB = zeros(nfield, nstates)       # ‖A − B‖/‖Δ‖
    r_AC = zeros(nfield, nstates)       # ‖A − C‖/‖Δ‖
    r_dx = zeros(nfield, nstates)       # ‖Δ‖/‖x_i‖
    r_z0 = zeros(nstates)               # ‖z₀_i − z₀*‖/‖z₀_i‖
    z0s = zeros(length(roughness(sim)), nstates)       # z₀_i
    z0stars = zeros(length(roughness(sim)), nstates)   # z₀*(x_i)

    for (i, it) in enumerate(idx)
        x, z0 = read_state(ds, it)
        @assert all(map((a, b) -> size(a) == size(b), x, snapshot(sim.variables)))
        A = apply_M(sim, vars_ref, x, z0)
        if i == 1   # numerical determinism: the same map twice must give bit-identical output
            A2 = apply_M(sim, vars_ref, x, z0)
            same = all(map(==, A, A2))
            note(log, "determinism, A twice"; identical = same)
            same || error("A = M(x, z₀) is not bit-identical when run twice")
        end
        zs = z0_star(sim, vars_ref, x)
        B = apply_M(sim, vars_ref, x, zs)
        C = apply_M(sim, vars_ref, x, zeros(length(zs)))
        all(y -> all(isfinite, y), (A..., B..., C...)) || error("non-finite output at state $i (map $it)")

        for f in 1:nfield
            Δ = A[f] .- x[f]
            r_AB[f, i] = relnorm(A[f] .- B[f], Δ)
            r_AC[f, i] = relnorm(A[f] .- C[f], Δ)
            r_dx[f, i] = relnorm(Δ, x[f])
        end
        r_z0[i] = relnorm(z0 .- zs, z0)
        z0s[:, i] .= z0; z0stars[:, i] .= zs

        note(log, @sprintf("state %2d (map %4d): r_AB %s | r_AC %s | |z0-z0*|/|z0| %.2e", i, it,
            join((@sprintf("%s %.2e", n, r_AB[f, i]) for (f, n) in enumerate(NCNAMES)), " "),
            join((@sprintf("%s %.2e", n, r_AC[f, i]) for (f, n) in enumerate(NCNAMES)), " "), r_z0[i]))
    end
    close(ds)

    for (f, n) in enumerate(NCNAMES)
        note(log, @sprintf("summary %-4s: r_AB median %.2e max %.2e | r_AC median %.2e max %.2e | |Δ|/|x| median %.2e",
            n, median_of(r_AB[f, :]), maximum(r_AB[f, :]), median_of(r_AC[f, :]), maximum(r_AC[f, :]), median_of(r_dx[f, :])))
    end

    # Output
    isfile(outfile) && rm(outfile)
    NCDataset(outfile, "c") do out
        for (k, val) in provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                                   title = "Surface-roughness dependence of the 6 h map M(x, z0), SpeedyWeather aquaplanet $config")
            out.attrib[k] = val
        end
        out.attrib["nature_file"] = natfile
        out.attrib["fields"] = join(NCNAMES, " ")   # order of the field dimension
        out.attrib["maps"] = "A = M(x, z0) as in the nature run; B = M(x, z0*(x)), z0* from one Euler step from x with z0 = 0; " *
                             "C = M(x, 0), fixed z0 (1e-4 m over ocean) in the first step's wind slowdown. Each map: Euler reset from x, then 9 dt = 6 h"
        defVar(out, "time_index", Int32.(idx), ("state",);
               attrib = Dict("long_name" => "time index of the state in the nature-run year file (1-based)", "units" => "1"))
        defVar(out, "time", 6.0 .* ((year - 1) * nmaps .+ idx), ("state",);
               attrib = Dict("long_name" => "nature-run time of the state", "units" => "hours since 0001-01-01 00:00:00", "calendar" => "noleap"))
        fattr(long) = Dict("long_name" => string(long, "; field dimension order: ", join(NCNAMES, ", ")), "units" => "1")
        defVar(out, "r_AB", r_AB, ("field", "state");
               attrib = fattr("||A - B|| / ||A - x||, norm over all spectral coefficients of the field"))
        defVar(out, "r_AC", r_AC, ("field", "state");
               attrib = fattr("||A - C|| / ||A - x||, norm over all spectral coefficients of the field"))
        defVar(out, "r_dx", r_dx, ("field", "state");
               attrib = fattr("||A - x|| / ||x||, relative 6 h change of the field"))
        defVar(out, "r_z0", r_z0, ("state",);
               attrib = Dict("long_name" => "||z0 - z0*(x)|| / ||z0||, over model grid points", "units" => "1"))
        defVar(out, "momentum_roughness", z0s, ("grid", "state");
               attrib = Dict("long_name" => "surface momentum roughness length stored with x in the nature run (model grid, flattened)", "units" => "m"))
        defVar(out, "momentum_roughness_star", z0stars, ("grid", "state");
               attrib = Dict("long_name" => "surface momentum roughness length after one Euler step from x with z0 = 0 (model grid, flattened)", "units" => "m"))
    end
    note(log, "wrote $outfile")
end

open(logfile, "w") do log
    note(log, "---- $(now()) on $(gethostname())")
    try
        main(log)
    catch e
        error_summary(log, e, catch_backtrace(), fulllog)
        rethrow()      # full message and trace go to stderr, i.e. the full log
    end
end
