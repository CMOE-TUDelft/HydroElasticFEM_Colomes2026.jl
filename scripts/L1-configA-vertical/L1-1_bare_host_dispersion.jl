#=
L1-1 — Bare host: dispersion and Green function.
Pre- and post-processing of the test defined in src/L1_1_bare_host_dispersion.jl
(physics, relation to the paper and observables are documented there).

Study, step 1 (`mesh`): one frequency, mesh convergence.
  P1 host (Table 1 of the paper), T_ρ = 0, H = 10 m, ω = 10 rad/s;
  p = 2 with 5, 10, 20, 40, 80 and p = 3 with 5, 10, 20, 40 elements per
  plate wavelength (p = 3 at 80 is ≈ 2.2M DOFs with the refined vertical
  grading; p = 3 reaches its floor ≈ 1e-6 at 40). All four observables.

Vertical mesh: the grading ratio is refined with the horizontal mesh
(`L1_1.vertical_mesh`); with a fixed ratio the P2 rate dropped to 1.25 at
ω ≈ 2, where the wave reaches the bottom.

Study, step 2 (`sweep`): frequency sweep, far field only.
  P1 host, T_ρ ∈ {0, 9.8}, H = 10 m, ω: 30 values log-spaced in [0.3, 15]
  (4 significant digits); p = 2 with 20, 40, 80 and p = 3 with 10, 20, 40
  elements per plate wavelength. Observables k_h and γ_h only: at low ω the
  element size (λ_p/20 ≈ 10 m at ω = 0.3) is far above the near-field decay
  lengths (flexural ≈ 0.7 m, fluid evanescent ≈ 3 m), so the near field is
  checked at ω = 10 only (step 1).

Acceptance (test matrix §0.3, type D), per order (and per T_ρ, ω in step 2):
  - p = 2: observed rate of each error on the two finest meshes ≥ 2(p − 1) − 0.3
    (C/DG plate: energy-norm rate p − 1, smooth functionals 2(p − 1));
  - all orders: error on the finest mesh ≤ 1e-3.

Outputs (relative to the repository root):
  data/sims/L1-1/      one DrWatson record per case (git commit, patch, HydroElasticFEM pin)
  data/exp_pro/L1-1/   step 1: errors.csv, checks.csv;
                       step 2: errors_sweep.csv, rates_sweep.csv, checks_sweep.csv
  plots/L1-1/          step 1: one convergence figure per order (.pdf, .png; dashed:
                       theoretical rate 2(p − 1)) and table (.tex);
                       step 2: sweep_err_k, sweep_err_gamma (.pdf, .png)

Run:
  julia --project -O1 scripts/L1-configA-vertical/L1-1_bare_host_dispersion.jl                     # step 1
  julia --project -O1 scripts/L1-configA-vertical/L1-1_bare_host_dispersion.jl --sweep             # step 2
  julia --project -O1 scripts/L1-configA-vertical/L1-1_bare_host_dispersion.jl --sweep --cases 1:60,75  # compute
        cases 1–60 and 75 of step 2 only (no post-processing), to split the sweep over several runs
  julia --project -O1 scripts/L1-configA-vertical/L1-1_bare_host_dispersion.jl --smoke             # one coarse case
  julia --project -O1 scripts/L1-configA-vertical/L1-1_bare_host_dispersion.jl --force             # recompute records
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L1_1

const TEST_ID = "L1-1"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ─── 1. Cases ───────────────────────────────────────────────────────────────

"Frequencies of the sweep (rounded so that the record names carry them exactly)."
const SWEEP_ω = round.(exp.(range(log(0.3), log(15.0); length=30)); sigdigits=4)
const SWEEP_T = (0.0, 9.8)
const SWEEP_LADDER = (2 => (20, 40, 80), 3 => (10, 20, 40))

"""
Cases of `study` (`:mesh`, step 1, or `:sweep`, step 2); `smoke = true`: one
coarse case, used by the test suite.
"""
function cases(study::Symbol=:mesh; smoke::Bool=false)
    smoke && return [L1_1_Params(elements_per_wavelength=20, order=2, half_window=3.0)]
    study == :mesh &&
        return [[L1_1_Params(elements_per_wavelength=n, order=2) for n in (5, 10, 20, 40, 80)];
                [L1_1_Params(elements_per_wavelength=n, order=3) for n in (5, 10, 20, 40)]]
    study == :sweep &&
        return [L1_1_Params(ω=ω, T_ρ=T, elements_per_wavelength=n, order=p)
                for T in SWEEP_T for ω in SWEEP_ω for (p, ladder) in SWEEP_LADDER for n in ladder]
    throw(ArgumentError("unknown study $(study)"))
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
Load the record of case `p`, or compute it with `L1_1.run_case` and save it
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
    return (order=p.order, elements_per_wavelength=p.elements_per_wavelength, ω=p.ω, T_ρ=p.T_ρ,
            ndofs=d["ndofs"], ny=d["ny"], h_over_lambda=d["element_size"] / d["lambda_p"],
            err_k=max(abs(d["k_h_R"] / d["k0"] - 1), abs(d["k_h_L"] / d["k0"] - 1)),
            err_gamma=max(abs(d["gamma_far_R"] - γr), abs(d["gamma_far_L"] - γr)) / γr,
            err_near_re=d["err_near_re"], err_near_im=d["err_near_im"],
            sponge_reflection=d["sponge_reflection"], fit_residual=d["fit_residual"],
            buffer_over_decay=d["buffer"] * d["kappa1"], wall_time_s=d["wall_time_s"])
end

const ERROR_COLUMNS = (:err_k => L"k_h", :err_gamma => L"\gamma_h",
                       :err_near_re => L"\mathrm{Re}\,\eta_h", :err_near_im => L"\mathrm{Im}\,\eta_h")
"Observables of the frequency sweep (far field)."
const SWEEP_COLUMNS = ERROR_COLUMNS[1:2]

"Observed order log(eᵢ/eᵢ₊₁)/log(hᵢ/hᵢ₊₁) between the two finest meshes."
last_rate(h, e) = log(e[end - 1] / e[end]) / log(h[end - 1] / h[end])

"Acceptance checks of the study (one row per criterion)."
function checks(df::DataFrame)
    rows = NamedTuple[]
    for g in groupby(sort(df, [:order, :elements_per_wavelength]), :order)
        p = first(g.order)
        for (col, _) in ERROR_COLUMNS
            e = g[end, col]
            push!(rows, (name="$(col), finest mesh, p=$(p)", kind="D", value=e, tolerance=1e-3, passed=e <= 1e-3))
            if p == 2 && nrow(g) >= 2
                rate, theory = last_rate(g.h_over_lambda, g[!, col]), 2(p - 1)
                push!(rows, (name="rate $(col), p=$(p)", kind="D", value=rate, tolerance=theory - 0.3,
                             passed=rate >= theory - 0.3))
            end
        end
    end
    return DataFrame(rows)
end

"""
Per T_ρ, ω and order: the error on the finest mesh and the observed rate on
the two finest meshes, for each far-field observable.
"""
function sweep_rates(df::DataFrame)
    rows = NamedTuple[]
    for g in groupby(sort(df, [:T_ρ, :ω, :order, :elements_per_wavelength]), [:T_ρ, :ω, :order])
        r = (T_ρ=first(g.T_ρ), ω=first(g.ω), order=first(g.order), finest=g[end, :elements_per_wavelength])
        for (col, _) in SWEEP_COLUMNS
            r = merge(r, NamedTuple{(Symbol(col, :_finest), Symbol(:rate_, col))}(
                (g[end, col], last_rate(g.h_over_lambda, g[!, col]))))
        end
        push!(rows, r)
    end
    return DataFrame(rows)
end

"Acceptance checks of the sweep: the step-1 criteria at every T_ρ and ω."
function sweep_checks(rates::DataFrame)
    rows = NamedTuple[]
    for r in eachrow(rates), (col, _) in SWEEP_COLUMNS
        tag = @sprintf("T=%g, ω=%g, p=%d", r.T_ρ, r.ω, r.order)
        e = r[Symbol(col, :_finest)]
        push!(rows, (name="$(col), finest mesh, $(tag)", kind="D", value=e, tolerance=1e-3, passed=e <= 1e-3))
        if r.order == 2
            rate, theory = r[Symbol(:rate_, col)], 2(r.order - 1)
            push!(rows, (name="rate $(col), $(tag)", kind="D", value=rate, tolerance=theory - 0.3,
                         passed=rate >= theory - 0.3))
        end
    end
    return DataFrame(rows)
end

# ─── 4. Figures and tables ──────────────────────────────────────────────────

"""
One figure per polynomial order: the errors of all quantities against the
mesh size, with the theoretical rate 2(p − 1) (dashed) from the largest
error on the coarsest mesh.
"""
function plot_convergence(df::DataFrame, dir::AbstractString)
    theme = Theme(fontsize=12, palette=(color=[:black, :royalblue, :firebrick, :forestgreen],),
                  Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        for p in sort(unique(df.order))
            g = sort(filter(r -> r.order == p, df), :elements_per_wavelength)
            n = g.elements_per_wavelength
            fig = Figure(size=(420, 340))
            ax = Axis(fig[1, 1]; xscale=log10, yscale=log10, xlabel="elements per wavelength",
                      ylabel="relative error", title="p = $(p)", xticks=(n, string.(n)))
            for (col, label) in ERROR_COLUMNS
                scatterlines!(ax, n, g[!, col]; label=label)
            end
            rate = 2(p - 1)
            e0 = maximum(g[1, col] for (col, _) in ERROR_COLUMNS)
            lines!(ax, n, e0 .* (n[1] ./ n) .^ rate; color=:gray40, linestyle=:dash,
                   label=L"\mathcal{O}(h^{%$rate})")
            axislegend(ax; position=:lb, framevisible=false)
            save(joinpath(dir, "convergence_single_frequency_p$(p).pdf"), fig)
            save(joinpath(dir, "convergence_single_frequency_p$(p).png"), fig; px_per_unit=300 / 96)
        end
    end
end

"""
One figure per far-field observable, one column per T_ρ. Top: error against ω
for every order and mesh (colour: order; line style: mesh, solid = finest).
Bottom: observed rate on the two finest meshes, dashed at 2(p − 1).
"""
function plot_sweep(df::DataFrame, rates::DataFrame, dir::AbstractString)
    colors = Dict(2 => :black, 3 => :royalblue)
    styles = (:dot, :dash, :solid)
    ωticks = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
    xticks = (ωticks, [@sprintf("%.1f", ω) for ω in ωticks])
    yticks = (10.0 .^ (-12:2:0), [L"10^{%$e}" for e in -12:2:0])   # whole decades
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        for (col, label) in SWEEP_COLUMNS
            Ts = sort(unique(df.T_ρ))
            fig = Figure(size=(380 * length(Ts) + 180, 600))
            for (j, T) in enumerate(Ts)
                ax = Axis(fig[1, j]; xscale=log10, yscale=log10, xticks=xticks, yticks=yticks, title=L"T_\rho = %$T",
                          ylabel=j == 1 ? "relative error" : "")
                axr = Axis(fig[2, j]; xscale=log10, xticks=xticks, xlabel=L"\omega\;\mathrm{(rad/s)}",
                           ylabel=j == 1 ? "observed rate" : "")
                linkxaxes!(ax, axr)
                for (p, ladder) in SWEEP_LADDER
                    for (n, ls) in zip(ladder, styles)
                        g = sort(filter(r -> r.T_ρ == T && r.order == p && r.elements_per_wavelength == n, df), :ω)
                        lines!(ax, g.ω, g[!, col]; color=colors[p], linestyle=ls, label="p = $(p), $(n) el./λ")
                    end
                    g = sort(filter(r -> r.T_ρ == T && r.order == p, rates), :ω)
                    scatterlines!(axr, g.ω, g[!, Symbol(:rate_, col)]; color=colors[p], markersize=5,
                                  label="p = $(p)")
                    hlines!(axr, 2(p - 1); color=colors[p], linestyle=:dash, linewidth=1)
                end
                if j == length(Ts)
                    Legend(fig[1, j + 1], ax; framevisible=false, labelsize=10)
                    Legend(fig[2, j + 1], axr; framevisible=false, labelsize=10)
                end
            end
            Label(fig[0, :], label; fontsize=16)
            save(joinpath(dir, "sweep_$(col).pdf"), fig)
            save(joinpath(dir, "sweep_$(col).png"), fig; px_per_unit=300 / 96)
        end
    end
end

function write_table(df::DataFrame, dir::AbstractString)
    mkpath(dir)
    e(x) = @sprintf("%.1e", x)
    open(joinpath(dir, "convergence_single_frequency.tex"), "w") do io
        println(io, "\\begin{table}[htbp]\n\\centering")
        println(io, "\\caption{L1-1: relative errors of the bare plate at \$\\omega = 10\$ rad/s (P1, \$T_\\rho = 0\$, \$H = 10\$ m).}")
        println(io, "\\label{tab:L1-1}\n\\begin{tabular}{rrrcccc}\n\\toprule")
        println(io, "\$p\$ & el./\$\\lambda_p\$ & DOFs & \$k_h\$ & \$\\gamma_h\$ & Re near field & Im near field \\\\\n\\midrule")
        for r in eachrow(sort(df, [:order, :elements_per_wavelength]))
            println(io, join([r.order, r.elements_per_wavelength, r.ndofs, e(r.err_k), e(r.err_gamma),
                              e(r.err_near_re), e(r.err_near_im)], " & "), " \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}\n\\end{table}")
    end
end

# ─── 5. Driver ──────────────────────────────────────────────────────────────

"""
    main(study=:mesh; smoke=false, force=false, root=REPO, only=nothing)

Run `study` (`:mesh` or `:sweep`): compute or load every case, write the
errors and checks to `<root>/data/exp_pro/L1-1/`, the figures (and, for
`:mesh`, the table) to `<root>/plots/L1-1/`, and print a summary. Returns
`(errors, checks)`. With `only` (case indices), compute those
records and return without post-processing.
"""
function main(study::Symbol=:mesh; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO,
              only::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cs = cases(study; smoke=smoke)
    if only !== nothing
        for (i, p) in zip(only, cs[only])
            @info "case $(i)/$(length(cs))" p.ω p.T_ρ p.order p.elements_per_wavelength
            simulate(p; root=root, force=force)
        end
        return nothing
    end
    df = DataFrame([errors(p, simulate(p; root=root, force=force)) for p in cs])
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    figdir = joinpath(root, "plots", TEST_ID)
    mkpath(out)
    if study == :sweep
        rates = sweep_rates(df)
        ck = sweep_checks(rates)
        CSV.write(joinpath(out, "errors_sweep.csv"), df)
        CSV.write(joinpath(out, "rates_sweep.csv"), rates)
        CSV.write(joinpath(out, "checks_sweep.csv"), ck)
        plot_sweep(df, rates, figdir)
        show(stdout, MIME"text/plain"(), rates; allrows=true)
        println()
        @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
        for c in eachrow(filter(r -> !r.passed, ck))
            @printf("FAIL  %-50s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
        end
        return (errors=df, checks=ck)
    end
    ck = checks(df)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "checks.csv"), ck)
    plot_convergence(df, figdir)
    write_table(df, figdir)

    show(stdout, MIME"text/plain"(), select(df, :order, :elements_per_wavelength, :ndofs, :err_k, :err_gamma,
                                            :err_near_re, :err_near_im); allrows=true)
    println()
    for c in eachrow(ck)
        @printf("%s  %-40s value = %.3g (tolerance %.3g)\n", c.passed ? "PASS" : "FAIL", c.name, c.value, c.tolerance)
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
    main("--sweep" in ARGS ? :sweep : :mesh; smoke="--smoke" in ARGS, force="--force" in ARGS,
         only=cases_arg(ARGS))
end
