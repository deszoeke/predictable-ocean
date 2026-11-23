# Nature run for the emulator experiment (docs/technical.md §7.3, step 2), SpeedyWeather 0.25:
#   - atmosphere-only aquaplanet, the same model and 6 h map M(Δτ) as drivers/speedy_map_check.jl
#     (shared code in drivers/speedy_aquaplanet.jl): every 6 h an Euler step starts the leapfrog from the single
#     current snapshot, in one continuous process;
#   - spin-up (not written), then one NetCDF file per model year, nature_T31L8-aqua_yNNNN.nc, holding the
#     6-hourly spectral prognostic snapshots (vorticity, divergence, temperature, humidity, log surface
#     pressure). Each file is written under a .tmp name and renamed when complete, so a concurrent reader
#     (the Jacobian-sample job) only ever sees complete files;
#   - after each year a restart file (Float64 snapshot, years completed, clock time, and the surface
#     momentum roughness, the one diagnostic field that carries over between maps; see below); if it exists at
#     startup, the run skips the spin-up and resumes with the next year. Resubmitting the job continues it.
# Configuration: PrimitiveWet, reduced physics (no convection, constant longwave transmissivity), aquaplanet
# with prescribed zonal SST, perpetual equinox, T31 (truncation 32), 8 layers, Float64 physics, CPU.
# Smoke test: NATURE_YEARS, NATURE_SPINUP_DAYS and NATURE_DAYS_PER_YEAR override the settings below.
#
# Logs: speedy_nature_run.log holds the driver's own messages (one line per year) and, on failure, a short
# error summary. Everything else goes to speedy_nature_run_full.log through the redirect below.
#
#   mkdir -p run/speedy-nature && cp drivers/speedy_nature_run.jl drivers/speedy_aquaplanet.jl run/speedy-nature/
#   cd run/speedy-nature && caffeinate -i julia --project=../.. speedy_nature_run.jl > speedy_nature_run_full.log 2>&1
#   (in another terminal: tail -f run/speedy-nature/speedy_nature_run.log)
#   On the HPC: drivers/speedy_nature_run.sbatch.

#####
##### Settings
#####

truncation    = 32         # SpeedyWeather's 32 = classic T31; Δt = 40 min
nlayers       = 8
steps_map     = 10         # Euler Δt/2 + leapfrog Δt + 8 × 2Δt leapfrog: advances 9Δt = 6 h
dt_minutes    = 40
spinup_days   = parse(Int, get(ENV, "NATURE_SPINUP_DAYS", "730"))    # not written
nyears        = parse(Int, get(ENV, "NATURE_YEARS", "40"))
days_per_year = parse(Int, get(ENV, "NATURE_DAYS_PER_YEAR", "365"))  # 365 days = 1460 maps per file
store_type    = Float32    # precision of the stored snapshots
config        = "T31L8-aqua"
logfile       = "speedy_nature_run.log"
fulllog       = "speedy_nature_run_full.log"    # set by the shell redirect
restartfile   = "restart_nature_T31L8-aqua.nc"

#####
##### Setup
#####

using SpeedyWeather
using PredictableOcean: provenance, settings_block
using Dates, NCDatasets, Printf

# zenith-kernel patch, FIELDS, create_model, level, field, snapshot, set_level1!, map6h!, note, error_summary
include(joinpath(@__DIR__, "speedy_aquaplanet.jl"))

# NetCDF names and CF attributes of the five fields, in the order of FIELDS
const NCNAMES = ("vor", "div", "T", "q", "lnps")
const NCUNITS = ("s-1", "s-1", "K", "kg kg-1", "1")
const NCLONG = ("relative vorticity", "divergence", "temperature", "specific humidity",
                "natural logarithm of surface pressure in Pa")
const SH_NOTE = ": complex spherical-harmonic coefficients (SpeedyWeather orthonormal convention: " *
                "the l = 0, m = 0 coefficient is 2√π times the global mean), "

"Spectral coefficient degree l and order m (0-based, Int32) in LowerTriangularArray storage order, and σ."
function coordinates(sim)
    lta = getfield(sim.variables.prognostic, :vorticity)
    lmax, mmax = lta.spectrum.lmax, lta.spectrum.mmax     # 1-based sizes, lmax = mmax + 1
    l = Int32[]; m = Int32[]
    for mi in 1:mmax, li in mi:lmax          # column-major lower triangle: lm2i(l, m) = l + (m - 1) lmax - m (m - 1) ÷ 2
        push!(l, li - 1); push!(m, mi - 1)
    end
    @assert length(l) == size(field(sim.variables, :vorticity), 1)
    truncation == 32 && @assert length(l) == 560
    sigma = Array(sim.model.geometry.σ_levels_full)
    return (; l, m, sigma, lmax, mmax)
end

"Year buffers: one complex `T` array per field, (coef, layer, time) or (coef, time)."
year_buffers(::Type{T}, x, nmaps) where {T} = map(xi -> zeros(Complex{T}, size(xi)..., nmaps), x)

store!(buf::AbstractArray{<:Any, 3}, xi, i) = (buf[:, :, i] .= xi; buf)
store!(buf::AbstractArray{<:Any, 2}, xi, i) = (buf[:, i] .= xi; buf)

"Run one model year of 6 h maps from the snapshot x, storing each new snapshot in `bufs`; returns the last."
function run_year!(sim, x, bufs, z0buf, year, steps)
    nmaps = size(bufs[1], ndims(bufs[1]))
    for i in 1:nmaps
        set_level1!(sim.variables, x)    # Euler step from the current snapshot
        map6h!(sim, steps)
        x = snapshot(sim.variables)
        all(xi -> all(isfinite, xi), x) || error("non-finite state in year $year, map $i")
        foreach((b, xi) -> store!(b, xi, i), bufs, x)
        z0buf[:, i] .= vec(Array(roughness(sim)))   # carried into the next map: part of the true state (M(x, z₀))
    end
    return x
end

dims_of(x::AbstractArray, extra...) = ndims(x) == (2 + length(extra)) ? ("coef", "layer", extra...) : ("coef", extra...)

function write_attributes!(ds, attrs)
    for (k, v) in attrs
        ds.attrib[k] = v
    end
end

"Write one year of snapshots to `path` (through `path.tmp`, renamed when complete)."
function write_year(path, bufs, z0buf, coords, time_h, attrs)
    tmp = path * ".tmp"
    isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do ds
        write_attributes!(ds, attrs)
        defVar(ds, "l", coords.l, ("coef",); attrib = Dict("long_name" => "spherical-harmonic degree (0-based)", "units" => "1"))
        defVar(ds, "m", coords.m, ("coef",); attrib = Dict("long_name" => "spherical-harmonic order (0-based)", "units" => "1"))
        defVar(ds, "sigma", coords.sigma, ("layer",);
               attrib = Dict("long_name" => "sigma at full levels (p / p_surface), layer 1 at the top",
                             "units" => "1", "standard_name" => "atmosphere_sigma_coordinate", "positive" => "down"))
        defVar(ds, "time", time_h, ("time",);
               attrib = Dict("long_name" => "time since the start of the written record (perpetual equinox: the date has no physical effect)",
                             "units" => "hours since 0001-01-01 00:00:00", "calendar" => "noleap"))
        for (name, units, long, b) in zip(NCNAMES, NCUNITS, NCLONG, bufs)
            for (part, f) in (("re", real), ("im", imag))
                defVar(ds, "$(name)_$(part)", f.(b), dims_of(b, "time");
                       attrib = Dict("units" => units, "long_name" => string(long, SH_NOTE, part == "re" ? "real part" : "imaginary part")))
            end
        end
        defVar(ds, "momentum_roughness", z0buf, ("grid", "time");
               attrib = Dict("units" => "m", "long_name" => string("surface momentum roughness length after each map, on the model grid ",
                   "(flattened, SpeedyWeather ring order). Carried into the next map, so the 6 h map is M(x, z0), not M(x) alone")))
    end
    mv(tmp, path; force = true)
    return path
end

# The 6 h map is not exactly a function of the snapshot alone: the surface momentum roughness
# (vars.parameterizations.momentum_roughness, grid space) is carried over from the previous step into the
# first step of the next map (found 2026-11-22 by copying every non-prognostic variable between a continued
# and a fresh simulation; this was the only one needed for bit-identical output). It goes in the restart
# file so that resuming is bit-identical to running on.
roughness(sim) = sim.variables.parameterizations.momentum_roughness

"Write the restart file (Float64 snapshot, years completed, clock time, roughness) through a .tmp name."
function write_restart(path, x, z0, years_done, clock_time, coords, attrs)
    tmp = path * ".tmp"
    isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do ds
        write_attributes!(ds, attrs)
        ds.attrib["years_completed"] = years_done
        ds.attrib["clock_time"] = string(clock_time)
        defVar(ds, "l", coords.l, ("coef",))
        defVar(ds, "m", coords.m, ("coef",))
        for (name, xi) in zip(NCNAMES, x)
            defVar(ds, "$(name)_re", real.(xi), dims_of(xi))
            defVar(ds, "$(name)_im", imag.(xi), dims_of(xi))
        end
        defVar(ds, "momentum_roughness", Array{Float64}(Array(z0)), ("grid",);
               attrib = Dict("units" => "m", "long_name" => "surface momentum roughness length, model grid points"))
    end
    mv(tmp, path; force = true)
    return path
end

read_field(v) = ndims(v) == 2 ? Array{Float64}(v[:, :]) : Array{Float64}(v[:])

"Read the restart file: snapshot (Float64, complex), roughness, years completed, clock time."
function read_restart(path)
    NCDataset(path) do ds
        x = map(name -> complex.(read_field(ds["$(name)_re"]), read_field(ds["$(name)_im"])), NCNAMES)
        return x, Array{Float64}(ds["momentum_roughness"][:]), Int(ds.attrib["years_completed"]), DateTime(ds.attrib["clock_time"])
    end
end

#####
##### Main
#####

function main(log)
    maps_per_year = 4 * days_per_year
    sim = initialize!(create_model(SpeedyWeather.CPU(); truncation, nlayers))
    clock = sim.variables.prognostic.clock
    coords = coordinates(sim)
    norm_sphere = sim.model.spectral_transform.norm_sphere
    note(log, "model"; config, truncation, nlayers, ncoef = length(coords.l), lmax = coords.lmax, mmax = coords.mmax,
         sigma = round.(coords.sigma, digits = 3), threads = Threads.nthreads())

    if isfile(restartfile)
        x, z0, years_done, clock_time = read_restart(restartfile)
        copyto!(roughness(sim), z0)
        clock.time = convert(typeof(clock.time), clock_time)   # irrelevant under perpetual equinox, kept for the record
        note(log, "resumed from $restartfile"; years_completed = years_done, clock_time)
    else
        map6h!(sim, steps_map)   # first segment from the initial conditions in level 1: a fresh model has level 2 = 0
        x = snapshot(sim.variables)
        t_spinup = @elapsed for _ in 1:(spinup_days * 4)
            set_level1!(sim.variables, x)   # Euler step from the current snapshot
            map6h!(sim, steps_map)
            x = snapshot(sim.variables)
        end
        all(xi -> all(isfinite, xi), x) || error("non-finite state at the end of the spin-up")
        years_done = 0
        note(log, "started fresh, spin-up done"; days = spinup_days, t_spinup_s = round(t_spinup, digits = 2),
             clock_time = clock.time)
    end
    years_done >= nyears && note(log, "nothing to do: $years_done of $nyears years already completed")

    bufs = year_buffers(store_type, x, maps_per_year)
    z0buf = zeros(store_type, length(roughness(sim)), maps_per_year)
    title = "SpeedyWeather aquaplanet nature run, 6-hourly spectral prognostic snapshots, $config"
    base = provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__), title)
    merge!(base, Dict{String, Any}("nyears" => nyears, "maps_per_year" => maps_per_year, "days_per_year" => days_per_year,
        "spinup_days" => spinup_days, "steps_map" => steps_map, "dt_minutes" => dt_minutes, "store_type" => string(store_type),
        "config" => config, "truncation" => truncation, "nlayers" => nlayers,
        "map" => "M(6 h): an Euler step from the single snapshot starts the leapfrog, then run steps_map steps (9 dt = 6 h)"))
    for year in (years_done + 1):nyears
        t_year = @elapsed x = run_year!(sim, x, bufs, z0buf, year, steps_map)
        scale = sim.variables.prognostic.scale[]
        attrs = merge(base, Dict{String, Any}("year" => year,
            "radius_scaling" => "vor and div are unscaled (true s-1): prognostic.scale[] = $scale after run!, which unscales in finalize!"))
        time_h = 6.0 .* (((year - 1) * maps_per_year + 1):(year * maps_per_year))
        path = @sprintf("nature_%s_y%04d.nc", config, year)
        t_write = @elapsed begin
            write_year(path, bufs, z0buf, coords, collect(time_h), attrs)
            write_restart(restartfile, x, roughness(sim), year, clock.time, coords, attrs)
        end
        Ttop, Tbot = real(x[3][1, 1]) / norm_sphere, real(x[3][1, end]) / norm_sphere
        note(log, @sprintf("year %4d: %.1f s (%.3f s per 6 h map), write %.1f s; global-mean T top %.2f K, bottom %.2f K; max |vor coef| %.3e; scale %g; wrote %s",
                           year, t_year, t_year / maps_per_year, t_write, Ttop, Tbot, maximum(abs, x[1]), scale, path))
    end
    note(log, "done"; years_completed = max(years_done, nyears))
end

open(logfile, "a") do log
    note(log, "---- $(now()) on $(gethostname())")
    try
        main(log)
    catch e
        error_summary(log, e, catch_backtrace(), fulllog)
        rethrow()      # full message and trace go to stderr, i.e. the full log
    end
end
