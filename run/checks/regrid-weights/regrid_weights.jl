# Extract the conservative regridding weights (SpeedyWeather grid <-> NumericalEarth exchange
# grid) from the coupled model, check them, and save them to NetCDF. Saved weights become
# constant sparse operators in a Reactant-compiled coupled step (docs/plan.md phase 4).
#
#   mkdir -p run/regrid-weights && cp drivers/regrid_weights.jl run/regrid-weights/
#   cd run/regrid-weights && caffeinate -i julia --project=../.. regrid_weights.jl

#####
##### Settings
#####

ocean_size     = (180, 80, 20)
ocean_latitude = (-80, 80)
ocean_depth    = 4000
truncation     = 31
nlayers        = 8

#####
##### Setup
#####

using NumericalEarth, SpeedyWeather, Oceananigans, ConservativeRegridding
using PredictableOcean
using LinearAlgebra, SparseArrays, Random, Printf

t = @elapsed c = build_coupled_model(; ocean_size, ocean_latitude, ocean_depth, truncation, nlayers)
@info "built coupled model" seconds = round(t)

w = regridding_weights(c.earth_model)
fa, ta = w.from_atmosphere, w.to_atmosphere
println("from atmosphere: ", fa.src_grid, " → ", fa.dst_grid, ", ", size(fa.intersections), ", nnz ", nnz(fa.intersections))
println("to atmosphere:   ", ta.src_grid, " → ", ta.dst_grid, ", ", size(ta.intersections), ", nnz ", nnz(ta.intersections))

for (name, a, b) in (("from_atmosphere", fa, ta), ("to_atmosphere", ta, fa))
    ch = regridding_checks(a, b)
    @printf("%s: transpose err %.1e; dst coverage %.4f–%.4f; src coverage %.4f–%.4f; regrid(1) on %d fully covered cells %.12f–%.12f\n",
            name, ch.transpose, ch.dst_coverage..., ch.src_coverage..., ch.n_full_dst, ch.constant...)
end

# our formula against ConservativeRegridding's own regrid! on a random field
reg = c.earth_model.interfaces.exchanger.atmosphere.regridder
for (name, ours, theirs) in (("from_atmosphere", fa, reg.from_atmosphere), ("to_atmosphere", ta, reg.to_atmosphere))
    f = randn(Xoshiro(1), length(ours.src_areas))
    dst = zeros(length(ours.dst_areas))
    ConservativeRegridding.regrid!(dst, theirs, f)
    @printf("%s: |ours - ConservativeRegridding.regrid!| / |.| = %.1e\n", name, norm(regrid(ours, f) - dst) / norm(dst))
end

# save, reload, compare
attrs = provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                   title = "Conservative regridding weights for the coupled model")
for (file, x) in (("regrid_atmosphere_to_exchange.nc", fa), ("regrid_exchange_to_atmosphere.nc", ta))
    isfile(file) && rm(file)
    save_regridding_weights(file, x; attributes = attrs)
    y = load_regridding_weights(file)
    @printf("%s: reload identical = %s\n", file, y.intersections == x.intersections && y.dst_areas == x.dst_areas)
end
