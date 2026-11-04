#=
L1-6 — Complete stop band against the lattice spacing.
Pre- and post-processing of the test defined in src/L1_6_spacing_sweep.jl
(physics, relation to the paper and observables are documented there).

Study: P1 host, H = 10 m, ω_r = 10 rad/s, fixed loading n₀m_ρ,r = 0.01 m,
lossless; a ∈ {0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.1} m (the lattice-sum table of
the test matrix §0.2b), T_ρ ∈ {0, 9.8}; p = 3 with 4, 8, 16 and p = 2 with
16, 32, 64 elements per cell.

Acceptance (test matrix §L1.6, §0.3):
  - type D, every a, T_ρ and order, finest mesh: ω_Γ and ω_X to 1e-4
    relative, and the gap ω_Γ − ω_X to 1e-2 relative (the gap is 7e-5 ω_r at
    a = 0.25, so this needs edges accurate to ~1e-7);
  - type A, T_ρ = 0, p = 3: fitted slope d ln Δω/d ln a of the FE gaps for
    a ≤ 0.5 within 4 ± 0.3 (eq:gap_a4); FE gap within 2 % of eq:gap_asymptotic
    for ak_c/π ≤ 0.75; ratio FE gap / truncated-local gap within 5 % of 1.89
    for a ≤ 0.7 (test matrix). The T_ρ = 9.8 values are reported.
  - type I: Bloch projection leakage ≤ 1e-6.

Outputs:
  data/sims/L1-6/      one DrWatson record per case
  data/exp_pro/L1-6/   errors.csv, gaps.csv (finest p = 3 per a, T_ρ), checks.csv
  plots/L1-6/          gap.pdf/png (FE, lattice sums and the gap laws against a), gap_table.tex

Run:
  julia --project -O1 scripts/L1-configA-vertical/L1-6_spacing_sweep.jl [--smoke] [--force]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_Colomes2026.L1_6

const TEST_ID = "L1-6"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ─── 1. Cases ───────────────────────────────────────────────────────────────

const STUDY_A = (0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.1)
const STUDY_T = (0.0, 9.8)
const LADDER = (3 => (4, 8, 16), 2 => (16, 32, 64))

function cases(; smoke::Bool=false)
    smoke && return [L1_6_Params(a=0.5, elements_per_cell=4, order=3)]
    return [L1_6_Params(a=a, T_ρ=T, elements_per_cell=n, order=p)
            for a in STUDY_A for T in STUDY_T for (p, l) in LADDER for n in l]
end

# ─── 2. Simulation records (DrWatson) ───────────────────────────────────────

record_name(p::L1_6_Params) = savename(p; accesses=VARIABLE, sigdigits=4) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

function simulate(p::L1_6_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L1_6_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L1_6_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── 3. Post-processing ─────────────────────────────────────────────────────

rel(h, r) = abs(h / r - 1)

function errors(p::L1_6_Params, d)
    gap_h = (d["omega_gamma_h"] - d["omega_X_h"]) / p.ω_r
    gap_ref = (d["omega_gamma_ref"] - d["omega_X_ref"]) / p.ω_r
    return (a=p.a, T_ρ=p.T_ρ, order=p.order, elements_per_cell=p.elements_per_cell, ndofs_cell=d["ndofs_cell"],
            akc_over_pi=d["akc_over_pi"], omega_X_h=d["omega_X_h"], omega_X_ref=d["omega_X_ref"],
            omega_gamma_h=d["omega_gamma_h"], omega_gamma_ref=d["omega_gamma_ref"],
            err_X=rel(d["omega_X_h"], d["omega_X_ref"]), err_gamma=rel(d["omega_gamma_h"], d["omega_gamma_ref"]),
            gap_h=gap_h, gap_ref=gap_ref, err_gap=rel(gap_h, gap_ref),
            gap_asymptotic=d["gap_asymptotic"], gap_a4=d["gap_a4"], gap_truncated_local=d["gap_truncated_local"],
            Pi_L=d["Pi_L"], leak=d["leak"], fe_solves=d["fe_solves"], wall_time_s=d["wall_time_s"])
end

"Finest mesh per a, T_ρ and order."
finest(df) = combine(groupby(df, [:a, :T_ρ, :order]), g -> g[argmax(g.elements_per_cell), :])

"Least-squares slope of log y against log x."
function loglog_slope(x, y)
    X, Y = log.(x), log.(y)
    return sum((X .- sum(X) / length(X)) .* (Y .- sum(Y) / length(Y))) / sum((X .- sum(X) / length(X)) .^ 2)
end

function checks(df::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol; ge=false) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol,
                                                          passed=ge ? value >= tol : value <= tol))
    fn = finest(df)
    for r in eachrow(sort(fn, [:T_ρ, :order, :a]))
        tag = @sprintf("a=%g, T=%g, p=%d", r.a, r.T_ρ, r.order)
        add!("ω_X, finest mesh, $(tag)", "D", r.err_X, 1e-4)
        add!("ω_Γ, finest mesh, $(tag)", "D", r.err_gamma, 1e-4)
        add!("gap, finest mesh, $(tag)", "D", r.err_gap, 1e-2)
    end
    add!("Bloch projection leakage, all cases", "I", maximum(df.leak), 1e-6)
    g = sort(filter(r -> r.T_ρ == 0 && r.order == 3, fn), :a)
    if nrow(g) >= 3
        s = loglog_slope(g.a[g.a .<= 0.5], g.gap_h[g.a .<= 0.5])
        add!("slope d ln Δω/d ln a, a ≤ 0.5, T=0, p=3 (|s − 4|)", "A", abs(s - 4), 0.3)
        for r in eachrow(filter(r -> r.akc_over_pi <= 0.75, g))
            add!(@sprintf("gap vs eq:gap_asymptotic, a=%g, T=0", r.a), "A", rel(r.gap_h, r.gap_asymptotic), 0.02)
        end
        for r in eachrow(filter(r -> r.a <= 0.7, g))
            add!(@sprintf("gap / truncated-local gap vs 1.89, a=%g, T=0", r.a), "A",
                 rel(r.gap_h / r.gap_truncated_local, 1.89), 0.05)
        end
    end
    return DataFrame(rows)
end

# ─── 4. Figures and tables ──────────────────────────────────────────────────

"Gap against a, one panel per T_ρ: FE (p = 3, finest), lattice sums, eq:gap_asymptotic, eq:gap_a4, truncated local."
function plot_gap(gaps::DataFrame, dir::AbstractString)
    nrow(gaps) == 0 && return
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        Ts = sort(unique(gaps.T_ρ))
        fig = Figure(size=(360 * length(Ts) + 200, 360))
        local ax
        for (j, T) in enumerate(Ts)
            g = sort(filter(r -> r.T_ρ == T, gaps), :a)
            Tstr = @sprintf("%g", T)
            ax = Axis(fig[1, j]; xscale=log10, yscale=log10, xlabel=L"a\ \mathrm{(m)}",
                      ylabel=j == 1 ? L"(\omega_\Gamma - \omega_X)/\omega_r" : "", title=L"T_\rho = %$Tstr",
                      xticks=([0.25, 0.5, 1.0], ["0.25", "0.5", "1"]))
            lines!(ax, g.a, g.gap_ref; color=:black, label="lattice sums")
            ok = g.akc_over_pi .<= 0.75
            lines!(ax, g.a[ok], g.gap_asymptotic[ok]; color=:royalblue, label="eq. gap_asymptotic")
            lines!(ax, g.a, g.gap_a4; color=:gray50, linestyle=:dash, label=L"\propto a^4\ \mathrm{(eq.\ gap\_a4)}")
            lines!(ax, g.a, g.gap_truncated_local; color=:royalblue, linestyle=:dot, label="local, truncated at π/a")
            scatter!(ax, g.a, g.gap_h; color=:firebrick, markersize=9, label="FE (p = 3)")
        end
        Legend(fig[1, length(Ts) + 1], ax; framevisible=false, labelsize=10)
        save(joinpath(dir, "gap.pdf"), fig)
        save(joinpath(dir, "gap.png"), fig; px_per_unit=300 / 96)
    end
end

function write_table(gaps::DataFrame, dir::AbstractString)
    mkpath(dir)
    open(joinpath(dir, "gap_table.tex"), "w") do io
        println(io, "\\begin{table}[htbp]\n\\centering")
        println(io, "\\caption{L1-6: complete stop band of a line array against the spacing at fixed loading \$n_0m_{\\rho,r} = 0.01\$ m (P1, \$H = 10\$ m, \$\\omega_r = 10\$ rad/s): lattice sums, finite element (\$p = 3\$, finest mesh) and the gap laws.}")
        println(io, "\\label{tab:L1-6}\n\\begin{tabular}{rrcccccc}\n\\toprule")
        println(io, "\$T_\\rho\$ & \$a\$ & \$ak_c/\\pi\$ & \$\\Delta\\omega/\\omega_r\$ (lattice) & err. FE & eq.~gap\\_asymptotic & eq.~gap\\_a4 & truncated local \\\\\n\\midrule")
        for r in eachrow(sort(gaps, [:T_ρ, :a]))
            println(io, join([r.T_ρ, r.a, @sprintf("%.2f", r.akc_over_pi), @sprintf("%.3e", r.gap_ref),
                              @sprintf("%.1e", r.err_gap), @sprintf("%.3e", r.gap_asymptotic),
                              @sprintf("%.3e", r.gap_a4), @sprintf("%.3e", r.gap_truncated_local)], " & "), " \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}\n\\end{table}")
    end
end

# ─── 5. Driver ──────────────────────────────────────────────────────────────

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO)
    cs = cases(; smoke=smoke)
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    ck = checks(df)
    gaps = filter(r -> r.order == 3, finest(df))
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "gaps.csv"), gaps)
    CSV.write(joinpath(out, "checks.csv"), ck)
    figdir = joinpath(root, "plots", TEST_ID)
    plot_gap(gaps, figdir)
    write_table(gaps, figdir)

    show(stdout, MIME"text/plain"(), select(finest(df), :a, :T_ρ, :order, :elements_per_cell, :gap_ref, :gap_h,
                                            :err_X, :err_gamma, :err_gap, :gap_asymptotic, :gap_truncated_local);
         allrows=true)
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
