# Shared helpers for the SpeedyWeather aquaplanet drivers (docs/technical.md §7): the zenith-kernel patch,
# the model setup, the 6 h map M(Δτ) and the logging helpers. Included by drivers/speedy_map_check.jl and
# drivers/speedy_nature_run.jl; copy it into run/<name>/ together with the driver. The including driver
# loads SpeedyWeather (and Printf etc.) before including this file.

# Runtime patch of SpeedyWeather 0.25.0, src/parameterizations/radiation/zenith.jl:321-342. The original
# `ifelse(ϕ * δ > 0, π, 0)` mixes an Irrational and an Int, so the result is boxed and Reactant cannot
# raise the kernel ("failed to raise func ... solar_zenith_season_kernel"). Same arithmetic, typed as NF.
# Remove when SpeedyWeather fixes it. Julia warns that the method is overwritten.
@eval SpeedyWeather begin
    @kernel inbounds = true function solar_zenith_season_kernel!(
            cos_zenith, δ, sinδ, cosδ, sinlat, coslat, lat, whichring
        )
        ij = @index(Global, Linear)
        j = whichring[ij]
        NF = eltype(cos_zenith)
        ϕ = lat[j]
        h₀ = ifelse(
            2 * (abs(δ) + abs(ϕ)) < π,                                  # polar day/night?
            NF(acos(clamp(-tan(ϕ) * tan(δ), -one(NF), one(NF)))),        # length of day
            ifelse(ϕ * δ > 0, NF(π), zero(NF)),                          # polar day if signs equal, else night
        )
        sinϕ, cosϕ = sinlat[j], coslat[j]
        cos_zenith_j = h₀ * sinδ * sinϕ + cosδ * cosϕ * sin(h₀)
        cos_zenith[ij] = cos_zenith_j / NF(π)
    end
end

const FIELDS = (:vorticity, :divergence, :temperature, :humidity, :pressure)

"""
    create_model(arch; truncation, nlayers)

PrimitiveWet aquaplanet with reduced physics (no convection, constant longwave transmissivity),
perpetual equinox and no daily cycle, Float64.
"""
function create_model(arch; truncation, nlayers)
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

"One 6 h map on a CPU simulation, in place: an Euler step from the snapshot in level 1 starts the leapfrog, then run `steps` steps."
function map6h!(sim, steps = 10)
    run!(sim; steps)      # initialize! (clock reset, level 1 → 2, grid refresh) + time stepping
    return sim
end

"Driver message: one line to the readable log (flushed) and to stdout."
function note(log, msg; kw...)
    s = isempty(kw) ? msg : string(msg, ": ", join(("$k = $v" for (k, v) in kw), ", "))
    println(log, s); flush(log)
    println(stdout, s)
end

"Short error summary for the readable log: message head, and frames in the drivers or SpeedyWeather."
function error_summary(log, e, bt, fulllog)
    msg = sprint(showerror, e)
    println(log, "ERROR: ", typeof(e).name.name, ": ", first(first(split(msg, '\n')), 300))
    for fr in stacktrace(bt)
        file = string(fr.file)
        occursin(r"speedy_|SpeedyWeather", file) || continue
        println(log, "  at ", fr.func, "  ", replace(file, homedir() => "~"), ":", fr.line)
    end
    println(log, "full message and trace: ", fulllog)
    flush(log)
end
