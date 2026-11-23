# R3, physics test ensembles for the emulator experiment (docs/technical.md §7.3, step 4), SpeedyWeather 0.25.
# For each start k = 1, 2, ... at nature day (k − 1)·45 from the start of year 46 (40 starts span years 46–50,
# the test years), with x and z₀ the nature snapshot and roughness at that time:
#   - member 0, the control: x itself (Float32 nature snapshot, stepped in Float64);
#   - members 1..K: x + δ, δ a random draw with each field's climatological spectrum, rescaled by one scalar
#     to σ-RMS pert_amp (1e-3 climatological standard deviations);
#   - every member runs the 6 h map M(x, z₀) repeatedly, an Euler step every 6 h as in the nature run, with
#     z₀ from the nature file before the first map and carried on by the model afterwards;
#   - X(τ) saved at lead 0, every 6 h to day 5, then daily to ens_days.
# Units: the σ-RMS of an increment d is sqrt(Σ_f ‖d_f‖²/σ_f² / Σ_f N_f), σ_f the climatological RMS anomaly of
# field f over its N_f stored coefficients (from every 10th day of years 11–40), as in R2.
# Output: one NetCDF file per start, ens_T31L8-aqua_sNNNN.nc, Float32, written through a .tmp name; starts
# whose file exists are skipped, so resubmitting continues the job.
#
# Environment overrides (smoke tests, job arrays): ENS_STARTS ("1:40"), ENS_MEMBERS (40, perturbed; the control
# is always added), ENS_DAYS (30), ENS_NATDIR ("../speedy-nature"), ENS_CLIM_YEARS ("11:40"). Overriding
# ENS_MEMBERS or ENS_DAYS writes ..._smoke.nc files.
#
# Logs: speedy_ensembles_sAAAA-BBBB.log holds the driver's own messages (one line per start) and, on failure, a
# short error summary. Everything else goes to the _full.log through the redirect.
#
#   mkdir -p run/speedy-ensembles && cp drivers/speedy_ensembles.jl drivers/speedy_aquaplanet.jl drivers/speedy_ensembles.sbatch run/speedy-ensembles/
#   cd run/speedy-ensembles && sbatch --array=0-7 speedy_ensembles.sbatch       (HPC; see the .sbatch)
#   Mac smoke test:
#   cd run/speedy-ensembles && ENS_STARTS=1:1 ENS_MEMBERS=3 ENS_DAYS=2 ENS_CLIM_YEARS=11:12 julia --project=../.. -t 1 speedy_ensembles.jl > speedy_ensembles_s0001-0001_full.log 2>&1

#####
##### Settings
#####

parse_range(s) = (r = split(s, ':'); parse(Int, r[1]):parse(Int, r[end]))
starts        = parse_range(get(ENV, "ENS_STARTS", "1:40"))
clim_years    = parse_range(get(ENV, "ENS_CLIM_YEARS", "11:40"))
natdir        = get(ENV, "ENS_NATDIR", "../speedy-nature")    # nature-run directory, relative to this run directory
nmembers      = parse(Int, get(ENV, "ENS_MEMBERS", "40"))     # perturbed members; member 0 (control) is added
ens_days      = parse(Int, get(ENV, "ENS_DAYS", "30"))
smoke         = haskey(ENV, "ENS_MEMBERS") || haskey(ENV, "ENS_DAYS")
first_year    = 46         # test years 46–50
last_year     = 50
start_spacing = 45         # days between starts
days_per_year = 365        # nature-run year files: 1460 maps
pert_amp      = 1e-3       # initial perturbation, σ-RMS (climatological standard deviations)
six_hourly_to = 5          # days with 6-hourly output; daily after
clim_stride   = 40         # maps between climatology samples (10 days), as in R2
truncation    = 32         # SpeedyWeather's 32 = classic T31; Δt = 40 min
nlayers       = 8
steps_map     = 10         # Euler Δt/2 + leapfrog Δt + 8 × 2Δt leapfrog: advances 9Δt = 6 h
seed          = 2027
config        = "T31L8-aqua"
tag           = string("s", lpad(first(starts), 4, '0'), "-", lpad(last(starts), 4, '0'))
logfile       = "speedy_ensembles_$tag.log"
fulllog       = "speedy_ensembles_$(tag)_full.log"   # set by the shell redirect

#####
##### Setup
#####

using SpeedyWeather
using PredictableOcean: provenance, settings_block
using Dates, LinearAlgebra, NCDatasets, Printf, Random

# zenith-kernel patch, FIELDS, create_model, level, field, snapshot, set_level1!, map6h!, note, error_summary
include(joinpath(@__DIR__, "speedy_aquaplanet.jl"))

const NCNAMES = ("vor", "div", "T", "q", "lnps")   # nature-run variable names, in the order of FIELDS
const NCUNITS = ("s-1", "s-1", "K", "kg kg-1", "1")

natfile(year) = joinpath(natdir, @sprintf("nature_%s_y%04d.nc", config, year))
roughness(sim) = sim.variables.parameterizations.momentum_roughness

#####
##### Helpers copied from drivers/speedy_jacobian_samples.jl (R2): state algebra, σ-RMS, climatology,
##### random climatological draws, variable reset
#####

scaled(a::Tuple, s) = map(ai -> s .* ai, a)
plus(a::Tuple, b::Tuple) = map(.+, a, b)
minus(a::Tuple, b::Tuple) = map(.-, a, b)
"σ-RMS of a state increment: sqrt(Σ_f ‖d_f‖²/σ_f² / Σ_f N_f)."
sigma_rms(d::Tuple, σ) = sqrt(sum(sum(abs2, di) / σi^2 for (di, σi) in zip(d, σ)) / sum(length, d))
"Rescale by one scalar to σ-RMS `amp`; returns the direction and the scalar applied."
function rescale(d::Tuple, σ, amp = 1.0)
    s = amp / sigma_rms(d, σ)
    return scaled(d, s), s
end

"""
Climatology from every `clim_stride`-th map of the nature years `clim_years`: per field the anomaly variance
of each coefficient, averaged over the coefficients of each degree l (per layer), expanded back to the
coefficient array as an amplitude `amp`; σ_f = RMS anomaly over all coefficients and layers.
"""
function climatology(clim_years, lidx)
    S1 = nothing; S2 = nothing; n = 0
    for y in clim_years
        NCDataset(natfile(y)) do ds
            nt = size(ds["vor_re"], ndims(ds["vor_re"]))
            it = 1:clim_stride:nt
            xs = map(NCNAMES) do name
                re, im = ds["$(name)_re"], ds["$(name)_im"]
                ndims(re) == 3 ? complex.(Float64.(re[:, :, it]), Float64.(im[:, :, it])) :
                                 complex.(Float64.(re[:, it]), Float64.(im[:, it]))
            end
            s1 = map(x -> dropdims(sum(x; dims = ndims(x)); dims = ndims(x)), xs)
            s2 = map(x -> dropdims(sum(abs2, x; dims = ndims(x)); dims = ndims(x)), xs)
            S1 = S1 === nothing ? s1 : map(.+, S1, s1)
            S2 = S2 === nothing ? s2 : map(.+, S2, s2)
            n += length(it)
        end
    end
    var = map((a, b) -> max.(b ./ n .- abs2.(a ./ n), 0.0), S1, S2)       # per coefficient (and layer)
    σ = map(v -> sqrt(sum(v) / length(v)), var)
    nl = maximum(lidx) + 1
    var_l = map(var) do v                                               # (degree, layer) or (degree,)
        out = zeros(nl, size(v, 2))
        cnt = zeros(nl)
        for (k, l) in enumerate(lidx)
            out[l + 1, :] .+= v[k, :]; cnt[l + 1] += 1
        end
        out ./ cnt
    end
    amp = map((v, vl) -> reshape(sqrt.(vl[lidx .+ 1, :]), size(v)), var, var_l)
    return (; σ, var_l, amp, nsamples = n)
end

"Random increment with the climatological spectrum: m = 0 real, m > 0 complex with variance split evenly."
function randclim(rng, amp, midx)
    map(amp) do a
        r = randn(rng, size(a)) .+ im .* randn(rng, size(a))
        mask = midx .> 0          # along the first (coefficient) dimension
        @. ifelse(mask, a * r / sqrt(2.0), a * real(r) + 0im)
    end
end

"Reset the variables to the reference (nothing carries over), insert x in level 1 and z₀ as the roughness."
function reset_vars!(sim, vars_ref, x, z0)
    SpeedyWeather.copy!(sim.variables, vars_ref)    # dest, src: variables only, not the model
    set_level1!(sim.variables, x)
    copyto!(roughness(sim), z0)
    return sim
end

"Nature snapshot `it` of an open year file as Float64 complex arrays (in the order of FIELDS), and its z₀."
function read_state(ds, it)
    x = map(NCNAMES) do name
        re, im = ds["$(name)_re"], ds["$(name)_im"]
        ndims(re) == 3 ? complex.(Float64.(re[:, :, it]), Float64.(im[:, :, it])) :
                         complex.(Float64.(re[:, it]), Float64.(im[:, it]))
    end
    return x, Float64.(ds["momentum_roughness"][:, it])
end

#####
##### Ensemble
#####

"Year and in-year time index (1-based) of start k: nature day (k − 1)·start_spacing from the start of first_year."
function start_time(k)
    d = (k - 1) * start_spacing
    year = first_year + d ÷ days_per_year
    it = (d % days_per_year) * 4 + 1
    year <= last_year || error("start $k (day $d) falls after year $last_year")
    return year, it
end

"Maps after which X is saved: 6-hourly to day six_hourly_to, then daily, to ens_days."
save_maps(ndays) = [n for n in 1:(4 * ndays) if n <= 4 * six_hourly_to || n % 4 == 0]

"Run one member from (x0, z0) for nmaps maps, storing the snapshot in `X` at lead 1 and after each map in `saves`."
function run_member!(Xbufs, j, sim, vars_ref, x0, z0, nmaps, saveidx)
    foreach((b, xi) -> at(b, j, 1) .= xi, Xbufs, x0)
    reset_vars!(sim, vars_ref, x0, z0)
    map6h!(sim, steps_map)
    for n in 1:nmaps
        n > 1 && (set_level1!(sim.variables, snapshot(sim.variables)); map6h!(sim, steps_map))   # Euler step; z₀ carries on
        il = saveidx[n]
        if il > 0
            x = snapshot(sim.variables)
            all(xi -> all(isfinite, xi), x) || error("non-finite state, member $(j - 1), map $n")
            foreach((b, xi) -> at(b, j, il) .= xi, Xbufs, x)
        end
    end
    return Xbufs
end

#####
##### Output
#####

"View of `buf` at trailing indices `idx` (all leading dimensions kept)."
at(buf, idx...) = view(buf, ntuple(_ -> Colon(), ndims(buf) - length(idx))..., idx...)

fielddims(xi, extra...) = ndims(xi) == 2 ? ("coef", "layer", extra...) : ("coef", extra...)

function write_start(path, X, lead_h, time_h, coords, clim, pscale, attrs)
    tmp = path * ".tmp"
    isfile(tmp) && rm(tmp)
    K1 = size(X[1], ndims(X[1]) - 1)
    NCDataset(tmp, "c") do ds
        for (k, v) in attrs
            ds.attrib[k] = v
        end
        defVar(ds, "l", coords.l, ("coef",); attrib = Dict("long_name" => "spherical-harmonic degree (0-based)", "units" => "1"))
        defVar(ds, "m", coords.m, ("coef",); attrib = Dict("long_name" => "spherical-harmonic order (0-based)", "units" => "1"))
        defVar(ds, "sigma", coords.sigma, ("layer",); attrib = Dict("long_name" => "sigma at full levels, layer 1 at the top",
               "units" => "1", "standard_name" => "atmosphere_sigma_coordinate", "positive" => "down"))
        defVar(ds, "member", Int32.(0:(K1 - 1)), ("member",); attrib = Dict("long_name" => "ensemble member; 0 is the unperturbed control", "units" => "1"))
        defVar(ds, "lead_hours", lead_h, ("lead",); attrib = Dict("long_name" => "forecast lead", "units" => "hours"))
        defVar(ds, "time", time_h, ("lead",); attrib = Dict("long_name" => "nature-run time at each lead",
               "units" => "hours since 0001-01-01 00:00:00", "calendar" => "noleap"))
        defVar(ds, "perturbation_scale", pscale, ("member",); attrib = Dict("long_name" =>
               "scalar applied to the unit-amplitude climatological draw to reach sigma-RMS pert_amp (0 for the control)"))
        defVar(ds, "sigma_clim", collect(clim.σ), ("field",); attrib = Dict("long_name" =>
               "climatological RMS anomaly of each field over all stored coefficients (field order: $(join(NCNAMES, ", ")))"))
        for (name, units, vl) in zip(NCNAMES, NCUNITS, clim.var_l)
            defVar(ds, "clim_var_l_$name", size(vl, 2) > 1 ? vl : vec(vl), size(vl, 2) > 1 ? ("degree", "layer") : ("degree",);
                   attrib = Dict("long_name" => "climatological anomaly variance per coefficient, averaged over the orders m of each degree l", "units" => "($units)2"))
        end
        for (name, units, b) in zip(NCNAMES, NCUNITS, X)
            dims = (fielddims(at(b, 1, 1), "member", "lead")...,)
            for (part, op) in (("re", real), ("im", imag))
                defVar(ds, "$(name)_$(part)", Float32.(op.(b)), dims;
                       attrib = Dict("units" => units, "long_name" => "ensemble state X(tau), $name, $(part == "re" ? "real" : "imaginary") part of spectral coefficients"))
            end
        end
    end
    mv(tmp, path; force = true)
    return path
end

"σ-RMS spread of the perturbed members about the control at lead index il: sqrt(mean_j sigma_rms(X_j − X_0)²)."
function spread(X, il, σ)
    K1 = size(X[1], ndims(X[1]) - 1)
    ctrl = map(b -> at(b, 1, il), X)
    sqrt(sum(sigma_rms(minus(map(b -> at(b, j, il), X), ctrl), σ)^2 for j in 2:K1) / (K1 - 1))
end

#####
##### Main
#####

function main(log)
    coords = NCDataset(natfile(first_year)) do ds
        (; l = Int32.(ds["l"][:]), m = Int32.(ds["m"][:]), sigma = Float64.(ds["sigma"][:]))
    end
    nmaps = 4 * ens_days
    smaps = save_maps(ens_days)
    saveidx = zeros(Int, nmaps)                   # lead index (≥ 2) of each map's output, 0 if not saved
    for (i, n) in enumerate(smaps)
        saveidx[n] = i + 1
    end
    lead_h = 6.0 .* [0; smaps]
    nlead = length(lead_h)
    note(log, "settings"; starts, clim_years, natdir, nmembers, ens_days, nlead, pert_amp, smoke, threads = Threads.nthreads())

    t_clim = @elapsed clim = climatology(clim_years, coords.l)
    note(log, "climatology"; nsamples = clim.nsamples, t_s = round(t_clim, digits = 1),
         sigma = join((@sprintf("%s %.3e", n, s) for (n, s) in zip(NCNAMES, clim.σ)), ", "))

    # CPU simulation and reference variables (every member starts from these)
    sim = initialize!(create_model(SpeedyWeather.CPU(); truncation, nlayers))
    map6h!(sim, steps_map)
    vars_ref = deepcopy(sim.variables)

    for k in starts
        year, it = start_time(k)
        outpath = @sprintf("ens_%s_s%04d%s.nc", config, k, smoke ? "_smoke" : "")
        if isfile(outpath)
            note(log, "start $k: $outpath exists, skipped")
            continue
        end
        t0 = time()
        x, z0 = NCDataset(ds -> read_state(ds, it), natfile(year))
        rng = Xoshiro(UInt64(seed) * 10^4 + UInt64(k))   # per start: independent of how starts are split
        X = map(xi -> zeros(ComplexF64, size(xi)..., nmembers + 1, nlead), x)
        pscale = zeros(nmembers + 1)
        run_member!(X, 1, sim, vars_ref, x, z0, nmaps, saveidx)                 # member 0: control
        for j in 1:nmembers
            δ, s = rescale(randclim(rng, clim.amp, coords.m), clim.σ, pert_amp)
            pscale[j + 1] = s
            run_member!(X, j + 1, sim, vars_ref, plus(x, δ), z0, nmaps, saveidx)
        end
        t_run = time() - t0

        il(day) = findfirst(==(6.0 * 4 * min(day, ens_days)), lead_h)
        sp = map(d -> spread(X, il(d), clim.σ), (1, 5, 30))
        time_h = 6.0 .* ((year - 1) * 4 * days_per_year + it) .+ lead_h
        attrs = provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                           title = "Physics test ensemble (R3), SpeedyWeather aquaplanet $config, start $k")
        merge!(attrs, Dict{String, Any}("start" => k, "start_year" => year, "start_index" => it,
            "start_index_note" => "time index of the start state in the nature-run year file (1-based)",
            "nature_file" => natfile(year), "config" => config, "fields" => join(NCNAMES, " "),
            "nmembers_perturbed" => nmembers, "ens_days" => ens_days, "pert_amp" => pert_amp,
            "clim_years" => string(clim_years), "clim_stride" => clim_stride, "clim_nsamples" => clim.nsamples,
            "sigma_rms" => "sigma-RMS of an increment d: sqrt(sum_f ||d_f||^2 / sigma_f^2 / sum_f N_f), sigma_f = sigma_clim, N_f stored coefficients",
            "map" => "M(x, z0): Euler step from the single snapshot, then 9 dt = 6 h; z0 from the nature file before the first map, carried by the model after",
            "smoke" => string(smoke)))
        t_write = @elapsed write_start(outpath, X, lead_h, time_h, coords, clim, pscale, attrs)
        note(log, @sprintf("start %2d (year %d, index %4d): run %.1f s, write %.1f s; sigma-RMS spread about the control: day 1 %.2e, day 5 %.2e, day %d %.2e; wrote %s",
                           k, year, it, t_run, t_write, sp[1], sp[2], min(30, ens_days), sp[3], outpath))
    end
    note(log, "done")
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
