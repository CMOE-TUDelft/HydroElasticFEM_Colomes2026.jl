#=
L2-2 — Woodbury equivalence: monolithic FE and Foldy with the FE transfer compliance.
Pre- and post-processing of the test defined in src/L2_2_woodbury.jl.

Study: the L2-1 channel (P1 plate, L_p = 20 m, H = 10 m, open water of 6
evanescent decay lengths on each side), 8 elements per wavelength, p = 3; resonators at the 20 regular points;
point load at x = 7.25 m; sets :identical, :random, :graded;
ω ∈ {2, 5, 8, 8.5, 9, 9.5, 9.8, 10, 10.2, 10.5, 11, 12} rad/s.

Acceptance (test matrix §L2.2), every case: type I,
‖η_mono − η_Foldy‖/‖η_mono‖ ≤ 1e-8 and the same for the resonator
displacements q. The difference is round-off of the sparse direct solver: on
the same systems (up to 3.5e5 unknowns at ω = 12) the reciprocity residual of
G_{B,h}, also a round-off measure, reaches 5e-10, so the matrix tolerance 1e-10
is below the attainable level; both are reported.

Outputs:
  data/sims/L2-2/      one DrWatson record per case
  data/exp_pro/L2-2/   errors.csv, checks.csv
  plots/L2-2/          woodbury.pdf/png (differences against ω; |η| along the plate at ω = 10)

Run:
  julia --project -O1 scripts/L2-configB-vertical/L2-2_woodbury.jl [--smoke] [--force]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L2_2

const TEST_ID = "L2-2"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

const STUDY_W = (2.0, 5.0, 8.0, 8.5, 9.0, 9.5, 9.8, 10.0, 10.2, 10.5, 11.0, 12.0)
const SETS = (:identical, :random, :graded)

function cases(; smoke::Bool=false)
    smoke && return [L2_2_Params(ω=10.0, set=:random, elements_per_wavelength=4)]
    return [L2_2_Params(ω=ω, set=s, elements_per_wavelength=8) for s in SETS for ω in STUDY_W]
end

record_name(p::L2_2_Params) = savename(p; accesses=VARIABLE, sigdigits=4) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

function simulate(p::L2_2_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L2_2_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L2_2_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

errors(p::L2_2_Params, d) = (set=String(p.set), ω=p.ω, ndofs=d["ndofs"], diff_eta=d["diff_eta"], diff_q=d["diff_q"],
                             reciprocity=d["reciprocity"], max_abs_eta=maximum(abs, d["eta_mono"]),
                             wall_time_s=d["wall_time_s"])

function checks(df::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    for r in eachrow(df)
        tag = @sprintf("set=%s, ω=%g", r.set, r.ω)
        add!("η mono vs Foldy(G_B,h), $(tag)", "I", r.diff_eta, 1e-8)
        add!("q mono vs Foldy(G_B,h), $(tag)", "I", r.diff_q, 1e-8)
    end
    return DataFrame(rows)
end

function plot_woodbury(df::DataFrame, cs, records, dir::AbstractString)
    nrow(df) == 0 && return
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        fig = Figure(size=(950, 340))
        ax1 = Axis(fig[1, 1]; yscale=log10, xlabel=L"\omega\ \mathrm{(rad/s)}", ylabel="relative difference",
                   title="monolithic vs Foldy with G_B,h")
        ax2 = Axis(fig[1, 2]; xlabel=L"x\ \mathrm{(m)}", ylabel=L"|\eta(x_i)|", title="ω = 10 rad/s")
        for (i, s) in enumerate(SETS)
            c = Makie.wong_colors()[i]
            g = sort(filter(r -> r.set == String(s), df), :ω)
            scatterlines!(ax1, g.ω, max.(g.diff_eta, 1e-16); color=c, label=String(s))
            k = findfirst(p -> p.set == s && p.ω == 10.0, cs)
            k === nothing && continue
            d = records[k]
            lines!(ax2, d["points"], abs.(d["eta_foldy"]); color=c)
            scatter!(ax2, d["points"], abs.(d["eta_mono"]); color=c, markersize=6)
        end
        hlines!(ax1, [1e-8]; color=:gray50, linestyle=:dash)
        Legend(fig[1, 3], ax1; framevisible=false, labelsize=10)
        save(joinpath(dir, "woodbury.pdf"), fig)
        save(joinpath(dir, "woodbury.png"), fig; px_per_unit=300 / 96)
    end
end

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO)
    cs = cases(; smoke=smoke)
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    ck = checks(df)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "checks.csv"), ck)
    plot_woodbury(df, cs, records, joinpath(root, "plots", TEST_ID))
    show(stdout, MIME"text/plain"(), df; allrows=true)
    println()
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-70s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (errors=df, checks=ck)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; smoke="--smoke" in ARGS, force="--force" in ARGS)
end
