# R2, Jacobian samples for the emulator experiment (docs/technical.md §7.3, step 3), SpeedyWeather 0.25.
# For states x_k of the nature run (every 4 days), with z₀_k the roughness stored with x_k:
#   - M(x_k):  the 6 h map M(x_k, z₀_k), as in the nature run (z₀ is a fixed input, not differentiated);
#   - v (4):   2 grown directions (differences of members of a 4-member ensemble started 2 days earlier from
#              small random perturbations) and 2 random directions with each field's climatological spectrum;
#   - w (4):   random cotangent directions with each field's climatological spectrum;
#   - M′v:     centred finite differences [M(x + εv) − M(x − εv)] / 2ε on CPU, Float64;
#   - M′ᵀw:    Reactant reverse pass, compiled once;
#   - the 4 × 4 dot-product test wⱼᵀ(M′vᵢ) = (M′ᵀwⱼ)ᵀvᵢ at every state (free), and a linearity check of the
#     finite differences (ε against 10ε and ε/10) every `check_every` states.
# Units: every direction is rescaled by one scalar (field ratios kept) to unit σ-RMS, where the σ-RMS of a
# state increment d is sqrt(Σ_f ‖d_f‖²/σ_f² / Σ_f N_f), σ_f the climatological RMS anomaly of field f over its
# N_f stored coefficients. So ε = eps_fd is a perturbation of eps_fd climatological standard deviations.
# Output: one NetCDF file per nature year, jacobian_T31L8-aqua_yNNNN.nc, Float32, written through a .tmp name;
# years whose file exists are skipped, so resubmitting continues the job.
#
# Environment overrides (smoke tests, job arrays): JAC_YEARS ("11:40"), JAC_NATDIR ("../speedy-nature"),
# JAC_STATES_PER_YEAR (all; a smaller number writes ..._smoke.nc files), JAC_CLIM_YEARS ("11:40").
#
# Logs: speedy_jacobian_samples_yAAAA-BBBB.log holds the driver's own messages and, on failure, a short error
# summary. Everything else (Reactant MLIR dumps, full traces) goes to the _full.log through the redirect.
#
#   mkdir -p run/speedy-jacobian && cp drivers/speedy_jacobian_samples.jl drivers/speedy_aquaplanet.jl drivers/speedy_jacobian_samples.sbatch run/speedy-jacobian/
#   cd run/speedy-jacobian && sbatch speedy_jacobian_samples.sbatch          (HPC; see the .sbatch for a job array)
#   Mac smoke test:
#   cd run/speedy-jacobian && JAC_YEARS=11:11 JAC_STATES_PER_YEAR=3 JAC_CLIM_YEARS=11:12 julia --project=../.. -t 1 speedy_jacobian_samples.jl > speedy_jacobian_samples_y0011-0011_full.log 2>&1

#####
##### Settings
#####

years         = let r = split(get(ENV, "JAC_YEARS", "11:40"), ':'); parse(Int, r[1]):parse(Int, r[end]) end
clim_years    = let r = split(get(ENV, "JAC_CLIM_YEARS", "11:40"), ':'); parse(Int, r[1]):parse(Int, r[end]) end
natdir        = get(ENV, "JAC_NATDIR", "../speedy-nature")    # nature-run directory, relative to this run directory
states_limit  = parse(Int, get(ENV, "JAC_STATES_PER_YEAR", "0"))   # 0: all states
state_first   = 9          # first state's time index in a year file: the ensemble starts 8 maps (2 days) earlier
state_stride  = 16         # maps between states: 4 days, 91 states per year
grow_maps     = 8          # ensemble length before the state: 2 days
nmembers      = 4          # members 1−2 and 3−4 give the 2 grown directions
pert_amp      = 1e-3       # initial ensemble perturbation, σ-RMS
nv_grown      = 2
nv_random     = 2
nw            = 4
eps_fd        = 1e-8       # finite-difference step, σ-RMS units
check_every   = 10         # linearity check (ε vs 10ε and ε/10) every check_every states
clim_stride   = 40         # maps between climatology samples (10 days)
log_every     = 10         # per-state log line every log_every states
truncation    = 32         # SpeedyWeather's 32 = classic T31; Δt = 40 min
nlayers       = 8
steps_map     = 10         # Euler Δt/2 + leapfrog Δt + 8 × 2Δt leapfrog: advances 9Δt = 6 h
checkpoint    = 3          # Reactant.Binomial budget for the reverse pass
seed          = 2026
config        = "T31L8-aqua"
tag           = string("y", lpad(first(years), 4, '0'), "-", lpad(last(years), 4, '0'))
logfile       = "speedy_jacobian_samples_$tag.log"
fulllog       = "speedy_jacobian_samples_$(tag)_full.log"   # set by the shell redirect

#####
##### Setup
#####

using SpeedyWeather, Reactant, Enzyme
using CUDA   # Reactant compiles KernelAbstractions kernels through its CUDA extension, even on CPU
using PredictableOcean: provenance, settings_block
using Dates, LinearAlgebra, NCDatasets, Printf, Random
Reactant.set_default_backend("cpu")

# zenith-kernel patch, FIELDS, create_model, level, field, snapshot, set_level1!, map6h!, note, error_summary
include(joinpath(@__DIR__, "speedy_aquaplanet.jl"))

const NCNAMES = ("vor", "div", "T", "q", "lnps")   # nature-run variable names, in the order of FIELDS
const NCUNITS = ("s-1", "s-1", "K", "kg kg-1", "1")

natfile(year) = joinpath(natdir, @sprintf("nature_%s_y%04d.nc", config, year))
roughness(sim) = sim.variables.parameterizations.momentum_roughness

# state algebra on tuples of complex spectral arrays (one per field)
inner(a::Tuple, b::Tuple) = sum(real(sum(conj.(ai) .* bi)) for (ai, bi) in zip(a, b))
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

#####
##### Climatology: per-field σ and the spectrum by degree and layer
#####

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

#####
##### The map on CPU (as in drivers/speedy_z0_check.jl)
#####

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

"Diagnostic: wall time of 3 bare 6 h maps (map6h! only) and 3 full apply_M (copy! of the variables + map)."
function time_maps(log, label, sim, vars_ref, x, z0)
    reset_vars!(sim, vars_ref, x, z0)
    t_bare = @elapsed for _ in 1:3
        map6h!(sim, steps_map)
    end
    t_full = @elapsed for _ in 1:3
        apply_M(sim, vars_ref, x, z0)
    end
    note(log, label; t_bare_map_s = round(t_bare / 3, digits = 3), t_apply_M_s = round(t_full / 3, digits = 3),
         julia_threads = Threads.nthreads(), cpu_threads = Sys.CPU_THREADS)
end

"JVP M′v by centred differences at step ε (σ-RMS units, v has unit σ-RMS)."
jvp_fd(sim, vars_ref, x, z0, v, ε) =
    scaled(minus(apply_M(sim, vars_ref, plus(x, scaled(v, ε)), z0),
                 apply_M(sim, vars_ref, minus(x, scaled(v, ε)), z0)), 1 / 2ε)

"Grown directions: member differences (1−2, 3−4, ...) of an ensemble run `grow_maps` maps from (x0, z0), unit σ-RMS."
function grown_dirs(sim, vars_ref, x0, z0, rng, clim, midx)
    members = map(1:nmembers) do _
        δ, _ = rescale(randclim(rng, clim.amp, midx), clim.σ, pert_amp)
        reset_vars!(sim, vars_ref, plus(x0, δ), z0)
        map6h!(sim, steps_map)
        for _ in 2:grow_maps
            set_level1!(sim.variables, snapshot(sim.variables))   # Euler step; roughness carries on, as in the nature run
            map6h!(sim, steps_map)
        end
        snapshot(sim.variables)
    end
    return [rescale(minus(members[2k - 1], members[2k]), clim.σ) for k in 1:nv_grown]
end

#####
##### Reactant VJP (loss_map, compile_vjp and vjp_rx copied from drivers/speedy_map_check.jl)
#####

# Differentiated map: the grid refresh from level 1 is inside, so derivatives flow through it
function loss_map(vars, model, w, n, options)
    SpeedyWeather.transform!(vars, model; initialize = true)
    SpeedyWeather.time_steps!(Simulation(vars, model), n; options...)
    s = zero(real(eltype(field(vars, :vorticity))))
    for (name, wi) in zip(FIELDS, w)
        s += real(sum(conj.(wi) .* level(field(vars, name), 2)))
    end
    return s
end

"Compile the reverse pass once (Reactant `@compile`); returns the compiled function."
function compile_vjp(vars, model, w, n, options, opts)
    dvars = Enzyme.make_zero(vars)
    wr = map(Reactant.to_rarray, w)
    return @compile compile_options = opts Enzyme.autodiff(Reverse, loss_map, Active,
        Duplicated(vars, dvars), Const(model), Const(wr), Const(n), Const(options))
end

"""
VJP M′ᵀw with the compiled reverse pass, on a fresh copy of the variables, in physical units.
`initialize!` scales vorticity and divergence by the radius R (all time levels), so the reverse pass sees
x̃ = S x and ỹ = S y, S = R for vor and div, 1 otherwise, and returns S⁻¹ M′ᵀ S w̃. Passing w̃ = S⁻¹ w and
multiplying the result by S gives M′ᵀw, matching the finite-difference JVP (which `run!` unscales).
"""
function vjp_rx(compiled, vars, model, w, n, options, R)
    S = map(name -> name in (:vorticity, :divergence) ? R : one(R), FIELDS)
    v = deepcopy(vars)
    dvars = Enzyme.make_zero(v)
    compiled(Reverse, loss_map, Active, Duplicated(v, dvars), Const(model), Const(map((wi, s) -> Reactant.to_rarray(wi ./ s), w, S)),
             Const(n), Const(options))
    return map((name, s) -> s .* level(Array(field(dvars, name)), 1), FIELDS, S)   # gradient w.r.t. the input level
end

"Load (x, z₀) into the Reactant simulation: set up on CPU (reset, level 1, roughness, initialize!), then copy!."
function load_rx!(sim_rx, sim, vars_ref, x, z0)
    reset_vars!(sim, vars_ref, x, z0)
    initialize!(sim; steps = steps_map)                 # clock reset, radius scaling of vor/div, level 1 → 2, grid refresh, on CPU
    sim.variables.prognostic.scale[] == sim.model.planet.radius ||
        error("expected vor/div scaled by the radius after initialize!, got scale = $(sim.variables.prognostic.scale[])")
    SpeedyWeather.copy!(sim_rx.variables, sim.variables)   # dest, src
    return sim_rx
end

#####
##### Output
#####

"View of `buf` at trailing indices `idx` (all leading dimensions kept)."
at(buf, idx...) = view(buf, ntuple(_ -> Colon(), ndims(buf) - length(idx))..., idx...)

fielddims(xi, extra...) = ndims(xi) == 2 ? ("coef", "layer", extra...) : ("coef", extra...)

function write_year(path, B, coords, clim, attrs)
    tmp = path * ".tmp"
    isfile(tmp) && rm(tmp)
    NCDataset(tmp, "c") do ds
        for (k, v) in attrs
            ds.attrib[k] = v
        end
        defVar(ds, "l", coords.l, ("coef",); attrib = Dict("long_name" => "spherical-harmonic degree (0-based)", "units" => "1"))
        defVar(ds, "m", coords.m, ("coef",); attrib = Dict("long_name" => "spherical-harmonic order (0-based)", "units" => "1"))
        defVar(ds, "sigma", coords.sigma, ("layer",); attrib = Dict("long_name" => "sigma at full levels, layer 1 at the top",
               "units" => "1", "standard_name" => "atmosphere_sigma_coordinate", "positive" => "down"))
        defVar(ds, "time", B.time, ("state",); attrib = Dict("long_name" => "nature-run time of x",
               "units" => "hours since 0001-01-01 00:00:00", "calendar" => "noleap"))
        defVar(ds, "time_index", Int32.(B.it), ("state",); attrib = Dict("long_name" => "time index of x in the nature-run year file (1-based)", "units" => "1"))
        defVar(ds, "v_kind", Int8[fill(1, nv_grown); fill(2, nv_random)], ("vdir",);
               attrib = Dict("long_name" => "kind of tangent direction", "flag_values" => Int8[1, 2],
                             "flag_meanings" => "grown_ensemble_difference random_climatological_spectrum"))
        defVar(ds, "sigma_clim", collect(clim.σ), ("field",); attrib = Dict("long_name" =>
               "climatological RMS anomaly of each field over all stored coefficients (field order: $(join(NCNAMES, ", ")))"))
        for (name, units, vl) in zip(NCNAMES, NCUNITS, clim.var_l)
            defVar(ds, "clim_var_l_$name", size(vl, 2) > 1 ? vl : vec(vl), size(vl, 2) > 1 ? ("degree", "layer") : ("degree",);
                   attrib = Dict("long_name" => "climatological anomaly variance per coefficient, averaged over the orders m of each degree l", "units" => "($units)2"))
        end
        defVar(ds, "momentum_roughness", B.z0, ("grid", "state");
               attrib = Dict("units" => "m", "long_name" => "surface momentum roughness stored with x in the nature run; a fixed input of M(x, z0)"))
        for (f, (name, units)) in enumerate(zip(NCNAMES, NCUNITS))
            vars = (("x", B.x, (), units, "state x from the nature run"),
                    ("Mx", B.Mx, (), units, "M(x, z0), the 6 h map"),
                    ("v", B.v, ("vdir",), units, "tangent direction v, unit sigma-RMS"),
                    ("Mv", B.Mv, ("vdir",), units, "JVP M'v by centred finite differences, step eps_fd"),
                    ("w", B.w, ("wdir",), "1", "cotangent direction w, unit sigma-RMS"),
                    ("MTw", B.MTw, ("wdir",), "1", "VJP M'^T w, Reactant reverse pass (gradient of w^T M(x) with respect to x)"))
            for (vname, buf, extra, u, long) in vars
                b = buf[f]
                dims = (fielddims(at(b, ntuple(_ -> 1, length(extra) + 1)...), extra...)..., "state")
                for (part, op) in (("re", real), ("im", imag))
                    defVar(ds, "$(vname)_$(name)_$(part)", Float32.(op.(b)), dims;
                           attrib = Dict("units" => u, "long_name" => "$long, $name, $(part == "re" ? "real" : "imaginary") part of spectral coefficients"))
                end
            end
        end
        defVar(ds, "v_scale", B.vscale, ("vdir", "state"); attrib = Dict("long_name" =>
               "scalar applied to the raw direction to reach unit sigma-RMS (grown: member difference; random: unit-amplitude climatological draw)"))
        defVar(ds, "w_scale", B.wscale, ("wdir", "state"); attrib = Dict("long_name" => "scalar applied to the raw cotangent draw"))
        defVar(ds, "tl", B.tl, ("vdir", "wdir", "state"); attrib = Dict("long_name" => "w_j^T (M'v_i), finite differences"))
        defVar(ds, "adj", B.adj, ("vdir", "wdir", "state"); attrib = Dict("long_name" => "(M'^T w_j)^T v_i, Reactant"))
        defVar(ds, "lin_hi", B.lin_hi, ("vdir", "state"); attrib = Dict("long_name" =>
               "sigma-RMS |M'v(10 eps) - M'v(eps)| / |M'v(eps)| at check states (NaN elsewhere)"))
        defVar(ds, "lin_lo", B.lin_lo, ("vdir", "state"); attrib = Dict("long_name" =>
               "sigma-RMS |M'v(eps/10) - M'v(eps)| / |M'v(eps)| at check states (NaN elsewhere)"))
        defVar(ds, "nature_consistency", B.natcon, ("state",); attrib = Dict("long_name" =>
               "sigma-RMS |M(x, z0) - x_next| / |M(x, z0) - x|, x_next the next nature-run snapshot (Float32 storage error)"))
        defVar(ds, "t_state", B.tstate, ("state",); attrib = Dict("long_name" => "wall time per state", "units" => "s"))
    end
    mv(tmp, path; force = true)
    return path
end

median_of(v) = (s = sort(v); n = length(s); n == 0 ? NaN : isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

#####
##### Main
#####

function main(log)
    coords = NCDataset(natfile(first(years))) do ds
        (; l = Int32.(ds["l"][:]), m = Int32.(ds["m"][:]), sigma = Float64.(ds["sigma"][:]))
    end
    nv = nv_grown + nv_random
    note(log, "settings"; years, clim_years, natdir, states_limit, nv, nw, eps_fd, threads = Threads.nthreads())

    t_clim = @elapsed clim = climatology(clim_years, coords.l)
    note(log, "climatology"; nsamples = clim.nsamples, t_s = round(t_clim, digits = 1),
         sigma = join((@sprintf("%s %.3e", n, s) for (n, s) in zip(NCNAMES, clim.σ)), ", "))

    # CPU simulation and reference variables (every map starts from these)
    sim = initialize!(create_model(SpeedyWeather.CPU(); truncation, nlayers))
    map6h!(sim, steps_map)
    vars_ref = deepcopy(sim.variables)
    x_t, z0_t = snapshot(sim.variables), Float64.(vec(Array(roughness(sim))))
    time_maps(log, "CPU map timing before Reactant", sim, vars_ref, x_t, z0_t)

    # Reactant simulation; compile the reverse pass once, at the first state of the first year
    sim_rx = initialize!(create_model(SpeedyWeather.ReactantDevice(); truncation, nlayers))
    time_maps(log, "CPU map timing after the Reactant simulation is set up", sim, vars_ref, x_t, z0_t)
    opts = SpeedyWeather.reactant_compile_options(; raise_first = true)
    options = (; checkpointing = Reactant.Binomial(checkpoint))
    compiled = nothing

    for year in years
        smoke = states_limit > 0
        outpath = @sprintf("jacobian_%s_y%04d%s.nc", config, year, smoke ? "_smoke" : "")
        if isfile(outpath)
            note(log, "year $year: $outpath exists, skipped")
            continue
        end
        # whole year in memory (Float32 complex, about 0.2 GB)
        X, Z0, nt = NCDataset(natfile(year)) do ds
            X = map(NCNAMES) do name
                re, im = ds["$(name)_re"], ds["$(name)_im"]
                ndims(re) == 3 ? complex.(re[:, :, :], im[:, :, :]) : complex.(re[:, :], im[:, :])
            end
            X, Array(ds["momentum_roughness"][:, :]), size(ds["vor_re"], ndims(ds["vor_re"]))
        end
        state = it -> map(Xf -> Complex{Float64}.(at(Xf, it)), X)
        its = collect(state_first:state_stride:(nt - 1))
        smoke && (its = its[1:min(states_limit, length(its))])
        ns = length(its)

        x1 = state(its[1])
        buf = (extra...) -> map(xi -> zeros(ComplexF64, size(xi)..., extra..., ns), x1)
        B = (; it = its, time = 6.0 .* ((year - 1) * nt .+ its), z0 = zeros(Float32, size(Z0, 1), ns),
             x = buf(), Mx = buf(), v = buf(nv), Mv = buf(nv), w = buf(nw), MTw = buf(nw),
             vscale = zeros(nv, ns), wscale = zeros(nw, ns), tl = zeros(nv, nw, ns), adj = zeros(nv, nw, ns),
             lin_hi = fill(NaN, nv, ns), lin_lo = fill(NaN, nv, ns), natcon = zeros(ns), tstate = zeros(ns))
        t_ens = 0.0; t_jvp = 0.0; t_vjp = 0.0

        for (s, it) in enumerate(its)
            t0 = time()
            rng = Xoshiro(UInt64(seed) * 10^7 + UInt64(year) * 10^4 + UInt64(it))   # per state: independent of how years are split
            x = state(it); z0 = Float64.(Z0[:, it])

            # tangent directions: 2 grown, 2 random; cotangent directions: random
            t_ens += @elapsed grown = grown_dirs(sim, vars_ref, state(it - grow_maps), Float64.(Z0[:, it - grow_maps]), rng, clim, coords.m)
            rand_v = [rescale(randclim(rng, clim.amp, coords.m), clim.σ) for _ in 1:nv_random]
            ws = [rescale(randclim(rng, clim.amp, coords.m), clim.σ) for _ in 1:nw]
            vs = vcat(grown, rand_v)

            # M(x, z₀) and the JVPs
            t_jvp += @elapsed begin
                Mx = apply_M(sim, vars_ref, x, z0)
                Mvs = [jvp_fd(sim, vars_ref, x, z0, v, eps_fd) for (v, _) in vs]
                if (s - 1) % check_every == 0
                    for (i, (v, _)) in enumerate(vs)
                        B.lin_hi[i, s] = sigma_rms(minus(jvp_fd(sim, vars_ref, x, z0, v, 10eps_fd), Mvs[i]), clim.σ) / sigma_rms(Mvs[i], clim.σ)
                        B.lin_lo[i, s] = sigma_rms(minus(jvp_fd(sim, vars_ref, x, z0, v, eps_fd / 10), Mvs[i]), clim.σ) / sigma_rms(Mvs[i], clim.σ)
                    end
                end
            end
            B.natcon[s] = sigma_rms(minus(Mx, state(it + 1)), clim.σ) / sigma_rms(minus(Mx, x), clim.σ)

            # the VJPs (compile on first use)
            load_rx!(sim_rx, sim, vars_ref, x, z0)
            vars_rx, model_rx = SpeedyWeather.unpack(sim_rx)
            if compiled === nothing
                note(log, "compiling the Reactant reverse pass")
                t_compile = @elapsed compiled = compile_vjp(deepcopy(vars_rx), model_rx, ws[1][1], steps_map, options, opts)
                note(log, "compiled"; t_compile_s = round(t_compile, digits = 1))
                time_maps(log, "CPU map timing after the compile", sim, vars_ref, x_t, z0_t)
            end
            t_vjp += @elapsed MTws = [vjp_rx(compiled, vars_rx, model_rx, w, steps_map, options, sim.model.planet.radius) for (w, _) in ws]

            # 4 × 4 dot-product test
            for i in 1:nv, j in 1:nw
                B.tl[i, j, s] = inner(ws[j][1], Mvs[i])
                B.adj[i, j, s] = inner(MTws[j], vs[i][1])
            end

            # store
            B.z0[:, s] .= z0
            foreach((b, xi) -> at(b, s) .= xi, B.x, x)
            foreach((b, xi) -> at(b, s) .= xi, B.Mx, Mx)
            for i in 1:nv
                foreach((b, xi) -> at(b, i, s) .= xi, B.v, vs[i][1])
                foreach((b, xi) -> at(b, i, s) .= xi, B.Mv, Mvs[i])
                B.vscale[i, s] = vs[i][2]
            end
            for j in 1:nw
                foreach((b, xi) -> at(b, j, s) .= xi, B.w, ws[j][1])
                foreach((b, xi) -> at(b, j, s) .= xi, B.MTw, MTws[j])
                B.wscale[j, s] = ws[j][2]
            end
            B.tstate[s] = time() - t0
            all(b -> all(isfinite, b), (B.Mv..., B.MTw...)) || error("non-finite derivative at year $year, state $s (map $it)")

            if s == 1 || s % log_every == 0 || s == ns
                tl = B.tl[:, :, s]; adj = B.adj[:, :, s]
                note(log, @sprintf("year %d state %3d/%d (map %4d): %.2f s; dot test max|tl-adj|/rms(tl) %.2e; nature consistency %.2e%s",
                    year, s, ns, it, B.tstate[s], maximum(abs.(tl .- adj)) / sqrt(sum(abs2, tl) / length(tl)), B.natcon[s],
                    isnan(B.lin_hi[1, s]) ? "" : @sprintf("; linearity 10ε %.2e, ε/10 %.2e (max over v)", maximum(B.lin_hi[:, s]), maximum(B.lin_lo[:, s]))))
            end
        end

        # per-year summary
        dot = [maximum(abs.(B.tl[:, :, s] .- B.adj[:, :, s])) / sqrt(sum(abs2, B.tl[:, :, s]) / (nv * nw)) for s in 1:ns]
        lh = filter(!isnan, vec(B.lin_hi)); ll = filter(!isnan, vec(B.lin_lo))
        note(log, @sprintf("year %d summary: %d states; dot test median %.2e max %.2e; linearity 10ε median %.2e max %.2e, ε/10 median %.2e max %.2e; nature consistency median %.2e; per state %.2f s (ensemble %.2f, M and JVPs %.2f, VJPs %.2f)",
            year, ns, median_of(dot), maximum(dot), median_of(lh), isempty(lh) ? NaN : maximum(lh), median_of(ll), isempty(ll) ? NaN : maximum(ll),
            median_of(B.natcon), sum(B.tstate) / ns, t_ens / ns, t_jvp / ns, t_vjp / ns))

        attrs = provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                           title = "Jacobian samples of the 6 h map M(x, z0): JVPs (finite differences) and VJPs (Reactant), SpeedyWeather aquaplanet $config")
        merge!(attrs, Dict{String, Any}("year" => year, "nature_file" => natfile(year), "config" => config,
            "fields" => join(NCNAMES, " "), "eps_fd" => eps_fd, "pert_amp" => pert_amp, "grow_maps" => grow_maps,
            "clim_years" => string(clim_years), "clim_stride" => clim_stride, "clim_nsamples" => clim.nsamples,
            "sigma_rms" => "sigma-RMS of an increment d: sqrt(sum_f ||d_f||^2 / sigma_f^2 / sum_f N_f), sigma_f = sigma_clim, N_f stored coefficients",
            "map" => "M(x, z0): Euler step from x with roughness z0 (stored with x in the nature run, not differentiated), then 9 dt = 6 h",
            "smoke" => string(smoke)))
        t_write = @elapsed write_year(outpath, B, coords, clim, attrs)
        note(log, "year $year: wrote $outpath"; t_write_s = round(t_write, digits = 1))
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
