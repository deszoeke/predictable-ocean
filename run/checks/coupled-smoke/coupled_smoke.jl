# Coupled smoke test: SpeedyWeather (PrimitiveWetModel) + Oceananigans through NumericalEarth's
# EarthSystemModel, on CPU at coarse resolution, idealized initial ocean (no ocean data download;
# SpeedyWeather downloads its orography/land/vegetation assets on first use). Checks that the
# coupled model builds, runs, stays finite, and how long it takes.
#
#   mkdir -p run/coupled-smoke && cp drivers/coupled_smoke.jl run/coupled-smoke/
#   cd run/coupled-smoke && julia --project=../.. coupled_smoke.jl

#####
##### Settings
#####

ocean_size     = (180, 80, 20)      # 2° × 2°, 20 levels
ocean_latitude = (-80, 80)
ocean_depth    = 4000               # m, flat bottom
truncation     = 31                 # SpeedyWeather spectral truncation
nlayers        = 8
stop_days      = 5
outfile        = "coupled_smoke.nc"

#####
##### Setup
#####

using Oceananigans, SpeedyWeather, ConservativeRegridding, NumericalEarth
using Oceananigans.Units
using PredictableOcean: provenance, settings_block
using NCDatasets, Printf

t_build = @elapsed begin
    grid = LatitudeLongitudeGrid(Oceananigans.CPU(); size = ocean_size, halo = (7, 7, 4), longitude = (0, 360),
                                 latitude = ocean_latitude, z = (-ocean_depth, 0))
    ocean = NumericalEarth.Oceans.ocean_simulation(grid)          # defaults: CATKE, WENO, split-explicit free surface
    # idealized stratified, meridionally varying ocean: warm surface tropics, cold deep
    Tᵢ(λ, φ, z) = 2 + 26 * cosd(φ)^2 * exp(z / 500)
    Oceananigans.set!(ocean.model, T = Tᵢ, S = 35)

    spectral_grid = SpeedyWeather.SpectralGrid(; NF = Float64, truncation, nlayers, Grid = FullClenshawGrid)
    atmosphere = NumericalEarth.atmosphere_simulation(spectral_grid; stop_time = stop_days * days)

    earth_model = EarthSystemModel(; atmosphere, sea_ice = default_sea_ice(), ocean)
    Δt = convert(eltype(grid), atmosphere.model.time_stepping.Δt)
    earth = Oceananigans.Simulation(earth_model; Δt, stop_time = stop_days * days)
end
@info "built coupled model" t_build Δt

T = ocean.model.tracers.T
sst() = Array(interior(T, :, :, size(grid, 3)))
sst0 = sst()
t_run = @elapsed Oceananigans.run!(earth)
sst1 = sst()

ok = all(isfinite, sst1)
@info "ran" t_run stop_days ok
@printf("SST range: %.2f–%.2f °C → %.2f–%.2f °C; mean change %.3f K\n",
        extrema(sst0)..., extrema(sst1)..., sum(sst1 .- sst0) / length(sst1))

isfile(outfile) && rm(outfile)
NCDataset(outfile, "c") do ds
    for (k, v) in provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                             title = "Coupled SpeedyWeather + Oceananigans smoke test")
        ds.attrib[k] = v
    end
    ds.attrib["t_build_s"] = t_build
    ds.attrib["t_run_s"] = t_run
    λ, φ = Array(λnodes(grid, Center())), Array(φnodes(grid, Center()))
    defVar(ds, "lon", λ, ("lon",); attrib = Dict("units" => "degrees_east"))
    defVar(ds, "lat", φ, ("lat",); attrib = Dict("units" => "degrees_north"))
    defVar(ds, "sst_initial", sst0, ("lon", "lat"); attrib = Dict("units" => "degC"))
    defVar(ds, "sst_final", sst1, ("lon", "lat"); attrib = Dict("units" => "degC"))
end
@info "wrote" outfile
