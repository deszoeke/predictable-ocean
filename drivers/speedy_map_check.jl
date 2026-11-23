# Check of the 6 h map M(Δτ) of the emulator experiment (docs/technical.md §7), SpeedyWeather 0.25:
#   - M(Δτ)(x): restart the leapfrog from the single snapshot x (Euler start), run 9Δt = 6 h;
#   - JVPs M′v by centred finite differences on CPU (Float64), scanned over ε;
#   - VJPs M′ᵀw by Reactant reverse mode;
#   - dot-product test wᵀ(M′v) = (M′ᵀw)ᵀv for a few random (v, w);
#   - timing of one 6 h map on CPU, and the Reactant compile + run time.
# Configuration: PrimitiveWet, reduced physics (no convection, constant longwave transmissivity, as in
# drivers/reactant_atmosphere_check.jl), aquaplanet with prescribed zonal SST, perpetual equinox.
#
#   mkdir -p run/speedy-map-check && cp drivers/speedy_map_check.jl run/speedy-map-check/
#   cd run/speedy-map-check && caffeinate -i julia --project=../.. speedy_map_check.jl

#####
##### Settings
#####

truncation  = 32           # SpeedyWeather's 32 = classic T31; Δt = 40 min
nlayers     = 8
steps_map   = 10           # Euler Δt/2 + leapfrog Δt + 8 × 2Δt leapfrog: advances 9Δt = 6 h
spinup_days = 20           # from the default initial conditions, restarting every 6 h
npairs      = 3            # random (v, w) pairs for the dot-product test
epsilons    = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)   # relative to the state norm, per field
checkpoint  = 3            # Reactant.Binomial budget for the reverse pass
seed        = 1
outfile     = "speedy_map_check.nc"

#####
##### Setup
#####

using SpeedyWeather, Reactant, Enzyme
using CUDA   # Reactant compiles KernelAbstractions kernels through its CUDA extension, even on CPU
using PredictableOcean: provenance, settings_block
using LinearAlgebra, NCDatasets, Printf, Random
Reactant.set_default_backend("cpu")

const FIELDS = (:vorticity, :divergence, :temperature, :humidity, :pressure)

function create_model(arch)
    sg = SpectralGrid(; NF = Float64, architecture = arch, nlayers, truncation)
    transmissivity = ConstantLongwaveTransmissivity(sg)
    radiation = Radiation(sg, longwave = OneBandLongwave(sg; transmissivity))
    planet = Earth(sg; seasonal_cycle = false, daily_cycle = false)
    PrimitiveWetModel(sg; spectral_transform = SpectralTransform(sg), feedback = nothing,
        initial_conditions = InitialConditions(sg, PrimitiveWetModel), radiation, convection = nothing,
        land_sea_mask = AquaPlanetMask(sg), ocean = AquaPlanet(sg), sea_ice = PrescribedSeaIce(sg),
        planet)   # default LandModel kept: DryLandModel lacks the snow depth the wet surface fluxes need; land fraction is 0
end

# time levels: index 2 of the last dimension is the current state, index 1 the previous one
level(A::AbstractArray{T, 3}, i) where {T} = view(A, :, :, i)
level(A::AbstractArray{T, 2}, i) where {T} = view(A, :, i)
field(vars, name) = getfield(vars.prognostic, name).data

"Current snapshot x (level 2) of the five spectral prognostic fields, as plain arrays."
snapshot(vars) = map(name -> copy(level(Array(field(vars, name)), 2)), FIELDS)   # Array first: no scalar indexing of device arrays

"Put snapshot x into level 1 (the level `initialize!` and `transform!(; initialize = true)` copy forward)."
function set_level1!(vars, x)
    for (name, xi) in zip(FIELDS, x)
        level(field(vars, name), 1) .= xi
    end
    return vars
end

"One 6 h map on a CPU simulation, in place: restart from the snapshot in level 1, then run."
function map6h!(sim)
    run!(sim; steps = steps_map)      # initialize! (clock reset, level 1 → 2, grid refresh) + time stepping
    return sim
end

"M(Δτ)(x) on CPU from a fresh copy of the reference simulation (time restored, so evaluations agree)."
function apply_M(sim_ref, x)
    sim = deepcopy(sim_ref)
    set_level1!(sim.variables, x)
    map6h!(sim)
    return snapshot(sim.variables)
end

# real inner product of complex spectral coefficients, summed over the fields
inner(a::Tuple, b::Tuple) = sum(real(sum(conj.(ai) .* bi)) for (ai, bi) in zip(a, b))
scaled(a::Tuple, s) = map(ai -> s .* ai, a)
plus(a::Tuple, b::Tuple) = map(.+, a, b)
minus(a::Tuple, b::Tuple) = map(.-, a, b)
"Random direction with each field's norm equal to that field's norm in x."
randdir(rng, x) = map(xi -> (d = randn(rng, eltype(xi), size(xi)); d .* (norm(xi) / norm(d))), x)

#####
##### Spin-up on CPU, restarting every 6 h (one process)
#####

sim_cpu = initialize!(create_model(SpeedyWeather.CPU()))
map6h!(sim_cpu)   # first segment from the initial conditions in level 1: a fresh model has level 2 = 0
nseg_spinup = spinup_days * 4
t_spinup = @elapsed for _ in 1:nseg_spinup
    set_level1!(sim_cpu.variables, snapshot(sim_cpu.variables))   # restart from the current level
    map6h!(sim_cpu)
end
x0 = snapshot(sim_cpu.variables)
set_level1!(sim_cpu.variables, x0)
@info "spin-up" days = spinup_days t_spinup finite = all(xi -> all(isfinite, xi), x0)

# timing of one 6 h map on CPU (after compilation by the spin-up)
t_map = @elapsed y0 = apply_M(sim_cpu, x0)
@info "one 6 h map on CPU" t_map

#####
##### Reactant: VJP of M(Δτ) at x0
#####

sim_rx = initialize!(create_model(SpeedyWeather.ReactantDevice()))
SpeedyWeather.copy!(sim_rx.variables, sim_cpu.variables)    # dest, src
let xr = snapshot(sim_rx.variables), d = maximum(norm.(minus(xr, x0)) ./ norm.(x0))
    @info "CPU state copied to Reactant" max_rel_diff = d
end
initialize!(sim_rx; steps = steps_map)      # level 1 already holds x0 (set on CPU before the copy); clock reset, level 1 → 2, grid refresh
vars_rx, model_rx = SpeedyWeather.unpack(sim_rx)

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

opts = SpeedyWeather.reactant_compile_options(; raise_first = true)
function vjp_rx(vars, model, w, n, options)
    dvars = Enzyme.make_zero(vars)
    wr = map(Reactant.to_rarray, w)
    @jit compile_options = opts Enzyme.autodiff(Reverse, loss_map, Active,
        Duplicated(vars, dvars), Const(model), Const(wr), Const(n), Const(options))
    return map(name -> copy(level(Array(field(dvars, name)), 1)), FIELDS)   # gradient w.r.t. the input level
end

rng = Xoshiro(seed)
rows = NamedTuple[]
t_rx = Float64[]
for ipair in 1:npairs
    v = randdir(rng, x0)
    w = randdir(rng, y0)
    push!(t_rx, @elapsed g = vjp_rx(deepcopy(vars_rx), model_rx, w, steps_map,
                                    (; checkpointing = Reactant.Binomial(checkpoint))))
    adj = inner(g, v)                                   # (M′ᵀw)ᵀ v
    for ε in epsilons
        Mv = scaled(minus(apply_M(sim_cpu, plus(x0, scaled(v, ε))),
                          apply_M(sim_cpu, minus(x0, scaled(v, ε)))), 1 / 2ε)
        tl = inner(w, Mv)                               # wᵀ (M′v)
        relerr = abs(tl - adj) / abs(tl)
        push!(rows, (; ipair, ε, tl, adj, relerr))
        @printf("pair %d  ε = %.0e:  wᵀ(M′v) = % .10e  (M′ᵀw)ᵀv = % .10e  rel. diff %.2e\n", ipair, ε, tl, adj, relerr)
    end
end
@info "Reactant VJP times (first includes compilation)" t_rx

#####
##### Output
#####

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
end
@info "wrote $outfile"
