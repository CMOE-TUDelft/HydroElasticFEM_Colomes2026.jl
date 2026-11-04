#=
L1-2 — Flux normalisation.
Pre- and post-processing of the test defined in src/L1_2_flux_normalisation.jl
(physics, relation to the paper and observables are documented there). The
FE problem and its parameters are those of L1-1 (`L1_1_Params`).

Study: P1 host (Table 1 of the paper), H = 10 m, T_ρ ∈ {0, 9.8},
ω ∈ {2, 5, 10} rad/s; p = 2 with 20, 40, 80 and p = 3 with 10, 20, 40
elements per plate wavelength (the ladders of the L1-1 sweep). Observables:
the radiative width γ_r from the far field and from the source patch, and the
work–flux balance.

Acceptance (test matrix §0.3, type D), per T_ρ, ω and order:
  - p = 2: observed rate of each error on the two finest meshes ≥ 2(p − 1) − 0.3;
  - all orders: error on the finest mesh ≤ 1e-3.
As in L1-1, the p = 3 rate is reported but not checked: on the finest p = 3
meshes the errors reach the floor ≈ 1e-7–1e-6 (L1-1, open point).

Outputs (relative to the repository root):
  data/sims/L1-2/      one DrWatson record per case (git commit, patch, HydroElasticFEM pin)
  data/exp_pro/L1-2/   errors.csv, rates.csv, checks.csv
  plots/L1-2/          one convergence figure per order (.pdf, .png; one panel per
                       measure, dashed: theoretical rate 2(p − 1)) and table (.tex)

Run:
  julia --project -O1 scripts/L1-configA-vertical/L1-2_flux_normalisation.jl                 # full study
  julia --project -O1 scripts/L1-configA-vertical/L1-2_flux_normalisation.jl --cases 1:18    # compute
        cases 1–18 only (no post-processing), to split the study over several runs
  julia --project -O1 scripts/L1-configA-vertical/L1-2_flux_normalisation.jl --smoke         # one coarse case
  julia --project -O1 scripts/L1-configA-vertical/L1-2_flux_normalisation.jl --force         # recompute records
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L1_1: L1_1_Params, VARIABLE, fixed_hash
using HydroElasticFEM_xxx2026.L1_2

const TEST_ID = "L1-2"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ─── 1. Cases ───────────────────────────────────────────────────────────────

const STUDY_ω = (2.0, 5.0, 10.0)
const STUDY_T = (0.0, 9.8)
const LADDER = (2 => (20, 40, 80), 3 => (10, 20, 40))

"Cases of the study (`smoke = true`: one coarse case, used by the test suite)."
function cases(; smoke::Bool=false)
    smoke && return [L1_1_Params(elements_per_wavelength=20, order=2, half_window=3.0)]
    return [L1_1_Params(ω=ω, T_ρ=T, elements_per_wavelength=n, order=p)
            for T in STUDY_T for ω in STUDY_ω for (p, ladder) in LADDER for n in ladder]
end

# ─── 2. Simulation records (DrWatson) ───────────────────────────────────────

"Record name: the variable parameters, then a hash of the fixed ones."
record_name(p::L1_1_Params) = savename(p; accesses=VARIABLE, sigdigits=4) * "_fixed=" * fixed_hash(p)

"HydroElasticFEM as pinned in the Manifest (URL, revision, tree hash)."
function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

"""
Load the record of case `p`, or compute it with `L1_2.run_case` and save it
with all parameters, the git commit and patch of this repository, the
HydroElasticFEM pin, Julia version, wall time and allocations.
"""
function simulate(p::L1_1_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L1_1_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    # guard against two cases sharing a record name (values rounded in the name)
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L1_1_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── 3. Post-processing ─────────────────────────────────────────────────────

"Errors of one case against the references."
function errors(p::L1_1_Params, d)
    γr = d["gamma_r"]
    return (T_ρ=p.T_ρ, ω=p.ω, order=p.order, elements_per_wavelength=p.elements_per_wavelength,
            ndofs=d["ndofs"], ny=d["ny"], h_over_lambda=d["element_size"] / d["lambda_p"],
            gamma_r=γr, gamma_far=real(d["gamma_far_R"]), gamma_patch=d["gamma_patch"],
            err_far=max(abs(d["gamma_far_R"] - γr), abs(d["gamma_far_L"] - γr)) / γr,
            err_patch=abs(d["gamma_patch"] - γr) / γr,
            err_power=abs(d["P_work"] / d["P_flux"] - 1),
            sponge_reflection=d["sponge_reflection"], fit_residual=d["fit_residual"],
            wall_time_s=d["wall_time_s"])
end

const ERROR_COLUMNS = (:err_far => "far field", :err_patch => "source patch", :err_power => "work–flux balance")

"Observed order log(eᵢ/eᵢ₊₁)/log(hᵢ/hᵢ₊₁) between the two finest meshes (NaN with one mesh)."
last_rate(h, e) = length(e) < 2 ? NaN : log(e[end - 1] / e[end]) / log(h[end - 1] / h[end])

"Per T_ρ, ω and order: the error on the finest mesh and the observed rate on the two finest meshes."
function rates(df::DataFrame)
    rows = NamedTuple[]
    for g in groupby(sort(df, [:T_ρ, :ω, :order, :elements_per_wavelength]), [:T_ρ, :ω, :order])
        r = (T_ρ=first(g.T_ρ), ω=first(g.ω), order=first(g.order), finest=g[end, :elements_per_wavelength])
        for (col, _) in ERROR_COLUMNS
            r = merge(r, NamedTuple{(Symbol(col, :_finest), Symbol(:rate_, col))}(
                (g[end, col], last_rate(g.h_over_lambda, g[!, col]))))
        end
        push!(rows, r)
    end
    return DataFrame(rows)
end

"Acceptance checks (one row per criterion)."
function checks(rt::DataFrame)
    rows = NamedTuple[]
    for r in eachrow(rt), (col, _) in ERROR_COLUMNS
        tag = @sprintf("T=%g, ω=%g, p=%d", r.T_ρ, r.ω, r.order)
        e = r[Symbol(col, :_finest)]
        push!(rows, (name="$(col), finest mesh, $(tag)", kind="D", value=e, tolerance=1e-3, passed=e <= 1e-3))
        if r.order == 2 && !isnan(r[Symbol(:rate_, col)])
            rate, theory = r[Symbol(:rate_, col)], 2(r.order - 1)
            push!(rows, (name="rate $(col), $(tag)", kind="D", value=rate, tolerance=theory - 0.3,
                         passed=rate >= theory - 0.3))
        end
    end
    return DataFrame(rows)
end

# ─── 4. Figures and tables ──────────────────────────────────────────────────

"""
One figure per polynomial order, one panel per measure of γ_r: error against
the mesh size for every T_ρ (marker) and ω (colour), with the theoretical rate
2(p − 1) (dashed) from the largest error on the coarsest mesh.
"""
function plot_convergence(df::DataFrame, dir::AbstractString)
    colors = Dict(2.0 => :black, 5.0 => :royalblue, 10.0 => :firebrick)
    markers = Dict(0.0 => :circle, 9.8 => :rect)
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        for p in sort(unique(df.order))
            dp = filter(r -> r.order == p, df)
            n = sort(unique(dp.elements_per_wavelength))
            fig = Figure(size=(1060, 330))
            local ax
            for (j, (col, title)) in enumerate(ERROR_COLUMNS)
                ax = Axis(fig[1, j]; xscale=log10, yscale=log10, title=title,
                          xlabel="elements per wavelength", ylabel=j == 1 ? "relative error" : "",
                          xticks=(n, string.(n)))
                for T in STUDY_T, ω in STUDY_ω
                    g = sort(filter(r -> r.T_ρ == T && r.ω == ω, dp), :elements_per_wavelength)
                    ωs, Ts = @sprintf("%g", ω), @sprintf("%g", T)
                    scatterlines!(ax, g.elements_per_wavelength, g[!, col]; color=colors[ω], marker=markers[T],
                                  label=L"\omega = %$ωs,\; T_\rho = %$Ts")
                end
                rate = 2(p - 1)
                e0 = maximum(filter(r -> r.elements_per_wavelength == n[1], dp)[!, col])
                lines!(ax, n, e0 .* (n[1] ./ n) .^ rate; color=:gray40, linestyle=:dash,
                       label=L"\mathcal{O}(h^{%$rate})")
            end
            Legend(fig[1, length(ERROR_COLUMNS) + 1], ax; framevisible=false, labelsize=10)
            Label(fig[0, :], "p = $(p)"; fontsize=14, font=:bold)
            save(joinpath(dir, "convergence_p$(p).pdf"), fig)
            save(joinpath(dir, "convergence_p$(p).png"), fig; px_per_unit=300 / 96)
        end
    end
end

function write_table(df::DataFrame, dir::AbstractString)
    mkpath(dir)
    e(x) = @sprintf("%.1e", x)
    finest = combine(groupby(df, [:T_ρ, :ω, :order]), g -> g[argmax(g.elements_per_wavelength), :])
    open(joinpath(dir, "gamma_table.tex"), "w") do io
        println(io, "\\begin{table}[htbp]\n\\centering")
        println(io, "\\caption{L1-2: relative errors of the radiative width \$\\gamma_r = 1/\\mathcal{D}_0'(k_0)\$ from the far field, from the source patch and from the work--flux balance, on the finest mesh (P1, \$H = 10\$ m).}")
        println(io, "\\label{tab:L1-2}\n\\begin{tabular}{rrrrcccc}\n\\toprule")
        println(io, "\$T_\\rho\$ & \$\\omega\$ & \$p\$ & el./\$\\lambda_p\$ & \$\\gamma_r\$ & far field & patch & power \\\\\n\\midrule")
        for r in eachrow(sort(finest, [:T_ρ, :ω, :order]))
            println(io, join([r.T_ρ, r.ω, r.order, r.elements_per_wavelength, @sprintf("%.4f", r.gamma_r),
                              e(r.err_far), e(r.err_patch), e(r.err_power)], " & "), " \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}\n\\end{table}")
    end
end

# ─── 5. Driver ──────────────────────────────────────────────────────────────

"""
    main(; smoke=false, force=false, root=REPO, only=nothing)

Run the study: compute or load every case, write the errors, rates and checks
to `<root>/data/exp_pro/L1-2/`, the figures and table to `<root>/plots/L1-2/`,
and print a summary. Returns `(errors, checks)`. With `only` (case indices),
compute those records and return without post-processing.
"""
function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO,
              only::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cs = cases(; smoke=smoke)
    if only !== nothing
        for (i, p) in zip(only, cs[only])
            @info "case $(i)/$(length(cs))" p.ω p.T_ρ p.order p.elements_per_wavelength
            simulate(p; root=root, force=force)
        end
        return nothing
    end
    df = DataFrame([errors(p, simulate(p; root=root, force=force)) for p in cs])
    rt = rates(df)
    ck = checks(rt)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "rates.csv"), rt)
    CSV.write(joinpath(out, "checks.csv"), ck)
    figdir = joinpath(root, "plots", TEST_ID)
    plot_convergence(df, figdir)
    write_table(df, figdir)

    show(stdout, MIME"text/plain"(), select(df, :T_ρ, :ω, :order, :elements_per_wavelength, :ndofs,
                                            :err_far, :err_patch, :err_power); allrows=true)
    println()
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-50s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (errors=df, checks=ck)
end

"Case indices following `--cases` (comma-separated indices and ranges `a:b`), or `nothing`."
function cases_arg(args)
    i = findfirst(==("--cases"), args)
    i === nothing && return nothing
    part(s) = occursin(':', s) ? range(parse.(Int, split(s, ":"))...) : [parse(Int, s)]
    return reduce(vcat, collect.(part.(split(args[i + 1], ","))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; smoke="--smoke" in ARGS, force="--force" in ARGS, only=cases_arg(ARGS))
end
