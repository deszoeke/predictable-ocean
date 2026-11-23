# Compare estimators of the sensitivity of the EXPECTED forecast of J (Tₑ, and slab-ocean T₁) to
# the slow initial state (ocean, Tₑ, h) against the exact ensemble answer, by lead time.
# Design: docs/technical.md §4. Estimators: single-trajectory adjoint, ensemble-mean adjoint,
# ensemble regression, LIM.
#
#   mkdir -p run/toy-sensitivity && cp drivers/toy_sensitivity_comparison.jl run/toy-sensitivity/
#   cd run/toy-sensitivity && julia -t auto --project=../.. toy_sensitivity_comparison.jl
#
# Plot with plots/sensitivity_comparison_pluto.jl.

#####
##### Settings
#####

leads        = [56, 240, 730, 1460, 2920]   # steps of 6 h: 2 wk, 2 mo, 6 mo, 1 yr, 2 yr
nstarts      = 4                           # initial states (spaced 5 years apart in a free run)
nmembers     = 200                         # ensemble size for the truth and the ensemble-mean adjoint
nregression  = 400                         # ensemble size for the regression estimator
atm_err      = 0.1                         # analysis error, atmosphere (fraction of s_clim)
slow_err     = 0.05                        # analysis error, ocean/Tₑ/h (fraction of s_clim)
fd_step      = 0.5                         # truth finite-difference step (fraction of s_clim)
lim_lag_days = 30                          # LIM fitting lag
seed         = 2026
outfile      = "toy_sensitivity_comparison.nc"

#####
##### Setup
#####

using PredictableOcean
using LinearAlgebra, Random, Statistics, NCDatasets

@info "threads" Threads.nthreads()
m = ToyCoupled()
N = state_length(m)
sl = state_slices(m)
slow = slow_indices(m)
nslow = length(slow)
steps_per_day = 4
metrics = [Component(sl.Te), Component(sl.ocean[1:1])]
metric_names = ["Te", "T1"]
rng = Xoshiro(seed)

# climatology from a long free run (daily): s_clim and the LIM
x = propagate(m, initial_state(m; rng), 10 * 1460)
clim = trajectory(m, x, 50 * 1460; every = steps_per_day)
s_clim = vec(std(clim; dims = 2))
L = fit_lim(clim[slow, :], lim_lag_days)                # per day
starts = [clim[:, 1 + (k - 1) * 5 * 365] for k in 1:nstarts]

B_std = vcat(fill(atm_err, sl.atmosphere.stop), fill(slow_err, nslow)) .* s_clim   # diagonal analysis error
δ = fd_step .* s_clim[slow]

nl, nm = length(leads), length(metrics)
truth = zeros(nslow, nl, nm, nstarts); truth_se = similar(truth)
single = similar(truth); ensgrad = similar(truth); ensgrad_se = similar(truth)
regress = similar(truth); lim = zeros(nslow, nl, nm)
for (im, J) in enumerate(metrics), (il, n) in enumerate(leads)
    row = J.indices[1] - first(slow) + 1
    lim[:, il, im] .= lim_sensitivity(L, n / steps_per_day, row)
end

for (is, x0) in enumerate(starts)
    @info "start" is Te0 = x0[sl.Te[1]]
    noise = B_std .* randn(rng, N, nmembers)
    for (im, J) in enumerate(metrics)
        t = @elapsed tr = ensemble_fd_sensitivity(J, m, x0, slow, δ, noise, leads)
        truth[:, :, im, is] .= tr.s
        truth_se[:, :, im, is] .= tr.se
        @info "truth" metric_names[im] seconds = round(t)

        for (il, n) in enumerate(leads)
            single[:, il, im, is] .= gradient_J(J, m, x0, n)[slow]
            eg = ensemble_mean_gradient(J, m, x0, noise, n)
            ensgrad[:, il, im, is] .= eg.g[slow]
            ensgrad_se[:, il, im, is] .= eg.se[slow]
        end

        # regression: an independent ensemble drawn from B; regress J on the slow perturbations
        Dn = B_std .* randn(rng, N, nregression)
        Jv = zeros(nregression, nl)
        Threads.@threads for k in 1:nregression
            Jv[k, :] .= forecast_metric_series(J, m, x0 .+ Dn[:, k], leads)
        end
        for il in 1:nl
            regress[:, il, im, is] .= regression_sensitivity(Jv[:, il], Dn[slow, :])
        end
    end
end

# scores against the truth, pooled over starts: pattern correlation and relative rms error
score(est, tr) = (cor(vec(est), vec(tr)), sqrt(mean(abs2, est .- tr)) / sqrt(mean(abs2, tr)))
names = ["single_adjoint", "ensemble_mean_adjoint", "regression", "lim"]
corr = zeros(length(names), nl, nm); relerr = similar(corr)
for im in 1:nm, il in 1:nl
    tr = truth[:, il, im, :]
    ests = (single[:, il, im, :], ensgrad[:, il, im, :], regress[:, il, im, :], repeat(lim[:, il, im], 1, nstarts))
    for (ie, e) in enumerate(ests)
        corr[ie, il, im], relerr[ie, il, im] = score(e, tr)
    end
end

#####
##### Output
#####

attrs = merge(provenance(; driver = abspath(@__FILE__), settings = settings_block(@__FILE__),
                         title = "Toy: estimators of the sensitivity of the expected forecast vs ensemble truth"),
              parameter_attributes(m))
isfile(outfile) && rm(outfile)
NCDataset(outfile, "c") do ds
    for (k, v) in attrs
        ds.attrib[k] = v
    end
    defVar(ds, "lead_days", leads ./ steps_per_day, ("lead",); attrib = Dict("units" => "days"))
    defVar(ds, "slow", collect(slow), ("slow",); attrib = Dict("long_name" => "state index of slow variable: T 37:72, Te 73, h 74"))
    defVar(ds, "metric", metric_names, ("metric",))
    defVar(ds, "estimator", names, ("estimator",))
    defVar(ds, "s_clim", s_clim[slow], ("slow",); attrib = Dict("long_name" => "climatological std of slow variables"))
    da = ("slow", "lead", "metric", "start")
    sens = "d E[J(T)] / d x0 (per unit of the slow variable)"
    defVar(ds, "truth", truth, da; attrib = Dict("long_name" => "ensemble finite-difference truth: $sens"))
    defVar(ds, "truth_se", truth_se, da; attrib = Dict("long_name" => "standard error of truth"))
    defVar(ds, "single_adjoint", single, da)
    defVar(ds, "ensemble_mean_adjoint", ensgrad, da)
    defVar(ds, "ensemble_mean_adjoint_se", ensgrad_se, da)
    defVar(ds, "regression", regress, da)
    defVar(ds, "lim", lim, ("slow", "lead", "metric"))
    defVar(ds, "corr", corr, ("estimator", "lead", "metric"); attrib = Dict("long_name" => "pattern correlation with truth (pooled over starts)"))
    defVar(ds, "relerr", relerr, ("estimator", "lead", "metric"); attrib = Dict("long_name" => "rms(est - truth) / rms(truth)"))
end
@info "wrote" outfile
for im in 1:nm, (ie, nme) in enumerate(names)
    println(metric_names[im], " ", rpad(nme, 22), " corr ", round.(corr[ie, :, im], digits = 2),
            "  relerr ", round.(relerr[ie, :, im], sigdigits = 2))
end
