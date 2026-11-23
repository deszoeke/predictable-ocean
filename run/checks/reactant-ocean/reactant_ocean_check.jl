# Reactant check, ocean alone: Oceananigans HydrostaticFreeSurfaceModel gradient under Reactant on a
# LatitudeLongitudeGrid (not gradient-tested upstream, which uses Rectilinear and Tripolar grids),
# verified against centered finite differences of the compiled forward model along a random
# direction (as in Oceananigans' test/reactant/tripolar.jl). ExplicitFreeSurface (the FFT implicit
# free surface does not work with Reactant); no CATKE (>10 min to compile with Reactant).
# Loss: mean squared T after n steps; gradient w.r.t. the initial T.
#
#   mkdir -p run/reactant-ocean && cp drivers/reactant_ocean_check.jl run/reactant-ocean/
#   cd run/reactant-ocean && caffeinate -i julia --project=../.. reactant_ocean_check.jl

#####
##### Settings
#####

ocean_size = (36, 16, 4)        # 10° × 10°, 4 levels: small, for a fast compile
latitude   = (-80, 80)
depth      = 1000               # m
Δt         = 600.0              # s
nsteps     = 10
dynamics   = :buoyant           # :passive (T only advected) | :buoyant (linear EOS, Coriolis, momentum)
outfile    = "reactant_ocean_check.nc"

#####
##### Setup
#####

using Oceananigans, Reactant, Enzyme
using Oceananigans.Architectures: ReactantState
using CUDA   # Reactant compiles KernelAbstractions kernels through its CUDA extension, even on CPU (deferred: CUDA.jl not yet in the environment)
using Reactant: @trace
using PredictableOcean: provenance, settings_block
using LinearAlgebra, Random, Statistics, NCDatasets, Printf
Reactant.set_default_backend("cpu")

grid = LatitudeLongitudeGrid(ReactantState(); size = ocean_size, halo = (4, 4, 4), longitude = (0, 360),
                             latitude, z = (-depth, 0))

model_kwargs(::Val{:passive}) = (; buoyancy = nothing, tracers = :T)
model_kwargs(::Val{:buoyant}) = (; buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(), constant_salinity = 35),
                                  tracers = :T, coriolis = HydrostaticSphericalCoriolis(),
                                  momentum_advection = VectorInvariant())
# A compiled call advances the model it is given (velocities, η, clock), so every evaluation below
# uses a freshly built model: otherwise each starts from the previous call's end state.
new_model() = HydrostaticFreeSurfaceModel(grid; free_surface = ExplicitFreeSurface(), timestepper = :QuasiAdamsBashforth2,
                                          model_kwargs(Val(dynamics))...)
model = new_model()
dmodel = Enzyme.make_zero(model)

T_init = CenterField(grid)
set!(T_init, (λ, φ, z) -> 2 + 26 * cosd(φ)^2 * exp(z / 300))
dT_init = CenterField(grid)
set!(dT_init, 0.0)

function loss(model, T_init, Δt, nsteps)
    set!(model, T = T_init)
    @trace mincut = true checkpointing = true track_numbers = false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.tracers.T) .^ 2)
end

function grad_loss(model, dmodel, T_init, dT_init, Δt, nsteps)
    parent(dT_init) .= 0
    _, loss_value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss, Enzyme.Active,
                                    Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(T_init, dT_init),
                                    Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return dT_init, loss_value
end

t_cg = @elapsed compiled_grad = Reactant.@compile raise = true raise_first = true sync = true grad_loss(model, dmodel, T_init, dT_init, Δt, nsteps)
t_cl = @elapsed compiled_loss = Reactant.@compile raise = true raise_first = true sync = true loss(model, T_init, Δt, nsteps)
@info "compiled" t_grad_compile = t_cg t_loss_compile = t_cl

m = new_model()
dT, J0 = compiled_grad(m, Enzyme.make_zero(m), T_init, dT_init, Δt, nsteps)
G = Array(interior(dT))
@info "gradient" loss = Float64(J0) finite = all(isfinite, G) maxabs = maximum(abs, G)

# finite differences along a random direction, using the compiled forward model
δ = randn(Xoshiro(1), size(G))
T0 = Array(interior(T_init))
function J_of(Tarray)
    Tp = CenterField(grid)
    set!(Tp, Tarray)
    return Float64(compiled_loss(new_model(), Tp, Δt, nsteps))
end
rows = Tuple{Float64, Float64, Float64}[]
for ε in (1e-2, 1e-3, 1e-4, 1e-5)
    fd = (J_of(T0 .+ ε .* δ) - J_of(T0 .- ε .* δ)) / (2ε)
    ad = sum(G .* δ)
    push!(rows, (ε, fd, abs(fd - ad) / abs(fd)))
    @printf("ε = %.0e: finite difference %.10e, AD %.10e, relative error %.2e\n", ε, fd, ad, abs(fd - ad) / abs(fd))
end

isfile(outfile) && rm(outfile)
NCDataset(outfile, "c") do ds
    for (k, v) in provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                             title = "Reactant gradient check, Oceananigans HydrostaticFreeSurfaceModel on LatitudeLongitudeGrid")
        ds.attrib[k] = v
    end
    ds.attrib["t_grad_compile_s"] = t_cg
    defVar(ds, "epsilon", first.(rows), ("eps",))
    defVar(ds, "fd", getindex.(rows, 2), ("eps",))
    defVar(ds, "relerr", last.(rows), ("eps",); attrib = Dict("long_name" => "|FD - AD| / |FD| along a random direction"))
    defVar(ds, "grad_T", G, ("lon", "lat", "z"); attrib = Dict("long_name" => "d loss / d initial T"))
end
