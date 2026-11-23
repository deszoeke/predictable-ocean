# Check of the 6 h map M(Δτ) of the emulator experiment (docs/technical.md §7), SpeedyWeather 0.25:
#   - M(Δτ)(x): restart the leapfrog from the single snapshot x (Euler start), run 9Δt = 6 h;
#   - JVPs M′v by centred finite differences on CPU (Float64), scanned over ε;
#   - VJPs M′ᵀw by Reactant reverse mode;
#   - dot-product test wᵀ(M′v) = (M′ᵀw)ᵀv for a few random (v, w);
#   - timing of one 6 h map on CPU, and the Reactant compile + run time.
# Configuration: PrimitiveWet, reduced physics (no convection, constant longwave transmissivity, as in
# drivers/reactant_atmosphere_check.jl), aquaplanet with prescribed zonal SST, perpetual equinox.
#
# Logs: speedy_map_check.log holds the driver's own messages, the results table and, on failure, a short
# error summary (message head and the driver/SpeedyWeather frames). Everything else (Reactant MLIR dumps,
# full stack traces with type parameters) goes to speedy_map_check_full.log through the redirect below.
#
#   mkdir -p run/speedy-map-check && cp drivers/speedy_map_check.jl drivers/speedy_aquaplanet.jl run/speedy-map-check/
#   cd run/speedy-map-check && caffeinate -i julia --project=../.. speedy_map_check.jl > speedy_map_check_full.log 2>&1
#   (in another terminal: tail -f run/speedy-map-check/speedy_map_check.log)

#####
##### Settings
#####

truncation  = 32           # SpeedyWeather's 32 = classic T31; Δt = 40 min
nlayers     = 8
steps_map   = 10           # Euler Δt/2 + leapfrog Δt + 8 × 2Δt leapfrog: advances 9Δt = 6 h
spinup_days = 20           # from the default initial conditions, restarting every 6 h
npairs      = 3            # random (v, w) pairs for the dot-product test
epsilons    = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)   # relative to the state norm, per field
eps_fine    = 10 .^ range(-9, -3, length = 25)   # linearity scan, pair 1
checkpoint  = 3           # Reactant.Binomial budget for the reverse pass
seed        = 1
outfile     = "speedy_map_check.nc"
logfile     = "speedy_map_check.log"
fulllog     = "speedy_map_check_full.log"   # set by the shell redirect

#####
##### Setup
#####

using SpeedyWeather, Reactant, Enzyme
using CUDA   # Reactant compiles KernelAbstractions kernels through its CUDA extension, even on CPU
using PredictableOcean: provenance, settings_block
using LinearAlgebra, NCDatasets, Printf, Random
Reactant.set_default_backend("cpu")

# zenith-kernel patch, FIELDS, create_model, level, field, snapshot, set_level1!, map6h!, note, error_summary
include(joinpath(@__DIR__, "speedy_aquaplanet.jl"))

"M(Δτ)(x) on CPU: reset the work simulation's variables to the reference (same clock), insert x, run."
function apply_M(sim_work, sim_ref, x)
    SpeedyWeather.copy!(sim_work.variables, sim_ref.variables)    # dest, src: variables only, not the model
    set_level1!(sim_work.variables, x)
    map6h!(sim_work, steps_map)
    return snapshot(sim_work.variables)
end

# real inner product of complex spectral coefficients, summed over the fields
inner(a::Tuple, b::Tuple) = sum(real(sum(conj.(ai) .* bi)) for (ai, bi) in zip(a, b))
scaled(a::Tuple, s) = map(ai -> s .* ai, a)
plus(a::Tuple, b::Tuple) = map(.+, a, b)
minus(a::Tuple, b::Tuple) = map(.-, a, b)
"Random direction with each field's norm equal to that field's norm in x."
randdir(rng, x) = map(xi -> (d = randn(rng, eltype(xi), size(xi)); d .* (norm(xi) / norm(d))), x)

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

"VJP M′ᵀw with the compiled reverse pass, on a fresh copy of the variables."
function vjp_rx(compiled, vars, model, w, n, options)
    v = deepcopy(vars)
    dvars = Enzyme.make_zero(v)
    compiled(Reverse, loss_map, Active, Duplicated(v, dvars), Const(model), Const(map(Reactant.to_rarray, w)),
             Const(n), Const(options))
    return map(name -> copy(level(Array(field(dvars, name)), 1)), FIELDS)   # gradient w.r.t. the input level
end

#####
##### Main
#####

function main(log)
    # Spin-up on CPU, restarting every 6 h (one process)
    sim_cpu = initialize!(create_model(SpeedyWeather.CPU(); truncation, nlayers))
    map6h!(sim_cpu, steps_map)   # first segment from the initial conditions in level 1: a fresh model has level 2 = 0
    t_spinup = @elapsed for _ in 1:(spinup_days * 4)
        set_level1!(sim_cpu.variables, snapshot(sim_cpu.variables))   # restart from the current level
        map6h!(sim_cpu, steps_map)
    end
    x0 = snapshot(sim_cpu.variables)
    set_level1!(sim_cpu.variables, x0)
    note(log, "spin-up"; days = spinup_days, t_spinup_s = round(t_spinup, digits = 2),
         finite = all(xi -> all(isfinite, xi), x0))

    # one 6 h map on CPU (after compilation by the spin-up); the work simulation is reused
    sim_work = deepcopy(sim_cpu)
    apply_M(sim_work, sim_cpu, x0)
    t_map = @elapsed y0 = apply_M(sim_work, sim_cpu, x0)
    note(log, "one 6 h map on CPU"; t_map_s = round(t_map, digits = 3))

    # numerical determinism: the same input twice must give bit-identical output
    y0b = apply_M(sim_work, sim_cpu, x0)
    note(log, "determinism, M(x0) twice"; identical = all(map(==, y0, y0b)),
         max_rel_diff = maximum(norm.(minus(y0, y0b)) ./ norm.(y0)), threads = Threads.nthreads())

    # Reactant: VJP of M(Δτ) at x0
    sim_rx = initialize!(create_model(SpeedyWeather.ReactantDevice(); truncation, nlayers))
    SpeedyWeather.copy!(sim_rx.variables, sim_cpu.variables)    # dest, src
    let xr = snapshot(sim_rx.variables)
        note(log, "CPU state copied to Reactant"; max_rel_diff = maximum(norm.(minus(xr, x0)) ./ norm.(x0)))
    end
    initialize!(sim_rx; steps = steps_map)      # level 1 already holds x0 (set on CPU before the copy); clock reset, level 1 → 2, grid refresh
    vars_rx, model_rx = SpeedyWeather.unpack(sim_rx)
    opts = SpeedyWeather.reactant_compile_options(; raise_first = true)

    rng = Xoshiro(seed)
    rows = NamedTuple[]
    t_rx = Float64[]
    options = (; checkpointing = Reactant.Binomial(checkpoint))
    note(log, "compiling the Reactant reverse pass")
    t_compile = @elapsed compiled = compile_vjp(deepcopy(vars_rx), model_rx, randdir(rng, y0), steps_map, options, opts)
    note(log, "compiled"; t_compile_s = round(t_compile, digits = 1))
    v1 = w1 = nothing
    for ipair in 1:npairs
        v = randdir(rng, x0)
        w = randdir(rng, y0)
        ipair == 1 && (v1 = v; w1 = w)
        push!(t_rx, @elapsed g = vjp_rx(compiled, vars_rx, model_rx, w, steps_map, options))
        note(log, "pair $ipair: VJP done"; t_s = round(t_rx[end], digits = 2))
        adj = inner(g, v)                                   # (M′ᵀw)ᵀ v
        for ε in epsilons
            Mv = scaled(minus(apply_M(sim_work, sim_cpu, plus(x0, scaled(v, ε))),
                              apply_M(sim_work, sim_cpu, minus(x0, scaled(v, ε)))), 1 / 2ε)
            tl = inner(w, Mv)                               # wᵀ (M′v)
            relerr = abs(tl - adj) / abs(tl)
            push!(rows, (; ipair, ε, tl, adj, relerr))
            note(log, @sprintf("pair %d  ε = %.0e:  wᵀ(M′v) = % .10e  (M′ᵀw)ᵀv = % .10e  rel. diff %.2e",
                               ipair, ε, tl, adj, relerr))
        end
    end

    # linearity: finite differences of wᵀM along v for pair 1 on a fine ε grid. A smooth map gives a
    # plateau (truncation error ∝ ε² above it, roundoff ∝ 1/ε below); kinks from physics switches give scatter.
    tl_fine = map(eps_fine) do ε
        inner(w1, scaled(minus(apply_M(sim_work, sim_cpu, plus(x0, scaled(v1, ε))),
                               apply_M(sim_work, sim_cpu, minus(x0, scaled(v1, ε)))), 1 / 2ε))
    end
    for (ε, tl) in zip(eps_fine, tl_fine)
        note(log, @sprintf("linearity, pair 1  ε = %.2e:  wᵀ(M′v) = % .10e", ε, tl))
    end

    # Output
    isfile(outfile) && rm(outfile)
    NCDataset(outfile, "c") do ds
        for (k, val) in provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                                   title = "6 h map check: finite-difference JVP vs Reactant VJP, SpeedyWeather aquaplanet")
            ds.attrib[k] = val
        end
        ds.attrib["t_spinup_s"] = t_spinup
        ds.attrib["t_map_cpu_s"] = t_map
        defVar(ds, "t_vjp_reactant_s", t_rx, ("pair",); attrib = Dict("long_name" => "Reactant VJP wall time (first includes compilation)"))
        defVar(ds, "epsilon", collect(epsilons), ("eps",))
        rel = [r.relerr for r in rows]; tl = [r.tl for r in rows]; adj = [r.adj for r in rows]
        defVar(ds, "relerr", reshape(rel, length(epsilons), npairs), ("eps", "pair");
               attrib = Dict("long_name" => "|wᵀ(M′v) − (M′ᵀw)ᵀv| / |wᵀ(M′v)|, M′v by finite differences"))
        defVar(ds, "tl", reshape(tl, length(epsilons), npairs), ("eps", "pair"); attrib = Dict("long_name" => "wᵀ(M′v)"))
        defVar(ds, "adj", reshape(adj, length(epsilons), npairs), ("eps", "pair"); attrib = Dict("long_name" => "(M′ᵀw)ᵀv"))
        ds.attrib["t_compile_reactant_s"] = t_compile
        defVar(ds, "epsilon_fine", collect(eps_fine), ("eps_fine",))
        defVar(ds, "tl_fine", tl_fine, ("eps_fine",); attrib = Dict("long_name" => "wᵀ(M′v) by finite differences, pair 1"))
    end
    note(log, "wrote $outfile")
end

open(logfile, "w") do log
    try
        main(log)
    catch e
        error_summary(log, e, catch_backtrace(), fulllog)
        rethrow()      # full message and trace go to stderr, i.e. the full log
    end
end
