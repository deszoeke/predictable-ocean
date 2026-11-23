# Reactant check, atmosphere alone: SpeedyWeather PrimitiveWetModel gradient under Reactant, verified
# against centered finite differences of the plain CPU model (no AD) along a random direction.
# Setup follows SpeedyWeather's test/reactant/differentation.jl. The plain-Enzyme CPU comparison used
# there segfaulted after 26 min on Julia 1.11 (2026-11-16; SpeedyWeather's full_diff_CI.jl notes
# "set_runtime_activity(Reverse) gives a segfault in 1.11"); set compare_plain_enzyme = true to retry.
# Float64 throughout so finite differences are accurate.
# Loss: sum of squared vorticity after n steps; gradient w.r.t. the initial prognostic vorticity.
#
#   mkdir -p run/reactant-atmosphere && cp drivers/reactant_atmosphere_check.jl run/reactant-atmosphere/
#   cd run/reactant-atmosphere && caffeinate -i julia --project=../.. reactant_atmosphere_check.jl

#####
##### Settings
#####

truncation   = 32          # as in SpeedyWeather's tests (T32L8)
nsteps       = 6           # > 4 so Reactant keeps the traced while loop
spinup_steps = 5
checkpoint   = 2           # Reactant.Binomial budget
compare_plain_enzyme = false    # plain Enzyme on CPU: slow (>25 min) and segfaults on Julia 1.11
outfile      = "reactant_atmosphere_check.nc"

#####
##### Setup
#####

using SpeedyWeather, Reactant, Enzyme
using CUDA   # Reactant compiles KernelAbstractions kernels through its CUDA extension, even on CPU (deferred: CUDA.jl not yet in the environment)
using PredictableOcean: provenance, settings_block
using LinearAlgebra, NCDatasets, Printf, Random
Reactant.set_default_backend("cpu")

# PrimitiveWet as in SpeedyWeather's tests: no convection, constant longwave transmissivity
# (Frierson radiation without convection amplifies ULP-level CPU/Reactant differences chaotically)
function create_model(arch)
    sg = SpectralGrid(; NF = Float64, architecture = arch, nlayers = 8, truncation)
    transmissivity = ConstantLongwaveTransmissivity(sg)
    radiation = Radiation(sg, longwave = OneBandLongwave(sg; transmissivity))
    PrimitiveWetModel(sg; spectral_transform = SpectralTransform(sg), feedback = nothing,
                      initial_conditions = InitialConditions(sg, PrimitiveWetModel), radiation, convection = nothing)
end

sync!(sim_cpu, sim_rx) = SpeedyWeather.copy!(first(SpeedyWeather.unpack(sim_cpu)), first(SpeedyWeather.unpack(sim_rx)))

t_init = @elapsed begin
    sim_cpu = initialize!(create_model(SpeedyWeather.CPU()))
    sim_rx = initialize!(create_model(SpeedyWeather.ReactantDevice()))
    run!(sim_cpu, steps = spinup_steps); run!(sim_rx, steps = spinup_steps)
    sync!(sim_cpu, sim_rx)
    initialize!(sim_cpu, steps = 1); initialize!(sim_rx, steps = 1)
    sync!(sim_cpu, sim_rx)
end
@info "initialized CPU and Reactant simulations" t_init

vars_rx, model_rx = SpeedyWeather.unpack(sim_rx)
vars_cpu, model_cpu = SpeedyWeather.unpack(sim_cpu)

function loss_n_steps(vars, model, n, options = (;))
    SpeedyWeather.time_steps!(Simulation(vars, model), n; options...)
    return sum(abs2, vars.prognostic.vorticity.data)
end

opts = SpeedyWeather.reactant_compile_options(; raise_first = true)
function grad_rx(vars, model, n, options)
    dvars = Enzyme.make_zero(vars)
    @jit compile_options = opts Enzyme.autodiff(Reverse, loss_n_steps, Active,
        Duplicated(vars, dvars), Const(model), Const(n), Const(options))
    return Array(dvars.prognostic.vorticity.data)
end
function grad_cpu(vars, model, n)
    dvars = Enzyme.make_zero(vars)
    Enzyme.autodiff(set_runtime_activity(Reverse), loss_n_steps, Active,
        Duplicated(vars, dvars), Const(model), Const(n))
    return Array(dvars.prognostic.vorticity.data)
end

t_rx = @elapsed g_rx = grad_rx(deepcopy(vars_rx), model_rx, nsteps, (; checkpointing = Reactant.Binomial(checkpoint)))
@info "Reactant gradient (compile + run)" t_rx finite = all(isfinite, g_rx)
# finite differences of the plain CPU model (no AD) along a random direction δ in vorticity
v0 = Array(vars_cpu.prognostic.vorticity.data)
δ = randn(Xoshiro(1), eltype(v0), size(v0)); δ .*= norm(v0) / norm(δ)
function loss_cpu(vort)
    v = deepcopy(vars_cpu)                     # fresh state every evaluation
    v.prognostic.vorticity.data .= vort
    return loss_n_steps(v, model_cpu, nsteps)
end
# directional derivative from a complex gradient: Re(g)·Re(δ) + Im(g)·Im(δ)
ad = real(sum(conj.(g_rx) .* δ))
rows = Tuple{Float64, Float64, Float64}[]
for ε in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
    fd = (loss_cpu(v0 .+ ε .* δ) - loss_cpu(v0 .- ε .* δ)) / (2ε)
    push!(rows, (ε, fd, abs(fd - ad) / abs(fd)))
    @printf("ε = %.0e: finite difference %.10e, Reactant AD %.10e, relative error %.2e\n", ε, fd, ad, abs(fd - ad) / abs(fd))
end

t_cpu = NaN; relerr_enzyme = NaN
if compare_plain_enzyme
    t_cpu = @elapsed g_cpu = grad_cpu(deepcopy(vars_cpu), model_cpu, nsteps)
    relerr_enzyme = norm(g_rx - g_cpu) / norm(g_cpu)
    @printf("|g_reactant - g_enzyme| / |g_enzyme| = %.2e\n", relerr_enzyme)
end

isfile(outfile) && rm(outfile)
NCDataset(outfile, "c") do ds
    for (k, v) in provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                             title = "Reactant gradient check (finite differences), SpeedyWeather PrimitiveWetModel")
        ds.attrib[k] = v
    end
    ds.attrib["relerr_reactant_vs_enzyme"] = relerr_enzyme
    ds.attrib["t_reactant_s"] = t_rx
    ds.attrib["t_enzyme_cpu_s"] = t_cpu
    defVar(ds, "epsilon", first.(rows), ("eps",))
    defVar(ds, "fd", getindex.(rows, 2), ("eps",))
    defVar(ds, "relerr_fd", last.(rows), ("eps",); attrib = Dict("long_name" => "|FD - AD| / |FD| along a random direction"))
    defVar(ds, "grad_vorticity_reactant_re", vec(real.(g_rx)), ("coef",); attrib = Dict("long_name" => "Re d loss / d initial vorticity (spectral)"))
    defVar(ds, "grad_vorticity_reactant_im", vec(imag.(g_rx)), ("coef",))
end
