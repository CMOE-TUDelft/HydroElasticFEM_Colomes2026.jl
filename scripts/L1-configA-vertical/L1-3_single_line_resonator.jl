#=
L1-3 — Single line resonator.
Pre- and post-processing of the test defined in src/L1_3_single_line_resonator.jl
(physics, relation to the paper and observables are documented there).

Study: P1 host (Table 1 of the paper), H = 10 m, resonator m_ρ,r = 0.01,
ω_r = 10 rad/s.
  - convergence ladder: T_ρ ∈ {0, 9.8}, c ∈ {0, c_crit} with the resonator on
    a mesh node, and c = 0 at an element midpoint; p = 2 with 20, 40, 80 and
    p = 3 with 10, 20, 40 elements per λ_p(ω_r) (the ladders of L1-1, L1-2);
  - response curves: T_ρ ∈ {0, 9.8}, c/c_crit ∈ {0, ½, 1, 2}, p = 3 with 10
    elements per λ_p, 41 frequencies in ω_r(1 ± 5/Q_rad).

Acceptance (test matrix §0.3), per T_ρ, c, attachment and order:
  - type D: relative error of ω̃ and of the half-power width on the finest
    mesh ≤ 1e-3; for p = 2 their observed rate on the two finest meshes
    ≥ 2(p − 1) − 0.3 (as in L1-1, the p = 3 rate is reported, not checked);
    on the finest mesh |R(ω̃) + 1| ≤ 1e-3 for c = 0 (T = 1 + R for a line
    resonator, so T(ω̃) = 0 is the same check), and |𝒜(ω̃) − ½| ≤ 1e-3 at
    critical coupling;
  - type I: resonator response q/η = κ/(κ − ω²m) at ω̃ to 1e-10.
The width is also compared with (1 + c/c_crit)ω̃/Q_rad (reported).

Outputs (relative to the repository root):
  data/sims/L1-3/      one DrWatson record per case (git commit, patch, HydroElasticFEM pin)
  data/exp_pro/L1-3/   errors.csv, rates.csv, checks.csv, curves.csv
  plots/L1-3/          one convergence figure per order (.pdf, .png; dashed:
                       theoretical rate 2(p − 1)), response curves, table (.tex)

Run:
  julia --project -O1 scripts/L1-configA-vertical/L1-3_single_line_resonator.jl                 # full study
  julia --project -O1 scripts/L1-configA-vertical/L1-3_single_line_resonator.jl --cases 1:12    # compute
        cases 1–12 only (no post-processing), to split the study over several runs
  julia --project -O1 scripts/L1-configA-vertical/L1-3_single_line_resonator.jl --smoke         # one coarse case
  julia --project -O1 scripts/L1-configA-vertical/L1-3_single_line_resonator.jl --force         # recompute records
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_Colomes2026.L1_3

const TEST_ID = "L1-3"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ─── 1. Cases ───────────────────────────────────────────────────────────────

const STUDY_T = (0.0, 9.8)
const LADDER = (2 => (20, 40, 80), 3 => (10, 20, 40))
const CURVE_C = (0.0, 0.5, 1.0, 2.0)

"Cases of the study (`smoke = true`: one coarse case, used by the test suite)."
function cases(; smoke::Bool=false)
    smoke && return [L1_3_Params(elements_per_wavelength=10, order=3)]
    ladder = [L1_3_Params(T_ρ=T, c_factor=c, snap=s, elements_per_wavelength=n, order=p)
              for T in STUDY_T for (c, s) in ((0.0, :node), (1.0, :node), (0.0, :midelement))
              for (p, l) in LADDER for n in l]
    curves = [L1_3_Params(T_ρ=T, c_factor=c, elements_per_wavelength=10, order=3, n_omega=41)
              for T in STUDY_T for c in CURVE_C]
    return [ladder; curves]
end

# ─── 2. Simulation records (DrWatson) ───────────────────────────────────────

"Record name: the variable parameters, then a hash of the fixed ones."
record_name(p::L1_3_Params) = savename(p; accesses=VARIABLE, sigdigits=4) * "_fixed=" * fixed_hash(p)

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
Load the record of case `p`, or compute it with `L1_3.run_case` and save it
with all parameters, the git commit and patch of this repository, the
HydroElasticFEM pin, Julia version, wall time and allocations.
"""
function simulate(p::L1_3_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L1_3_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    # guard against two cases sharing a record name (values rounded in the name)
    stored = data["params"]
    all(f -> get(stored, string(f), getfield(L1_3_Params(), f)) == getfield(p, f), fieldnames(L1_3_Params)) ||   # late fields: default
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── 3. Post-processing ─────────────────────────────────────────────────────

"Errors of one case against the references."
function errors(p::L1_3_Params, d)
    width_h = -(d["omega_half_h"]...)
    width_ref = -(d["omega_half_ref"]...)
    ω̃ = d["omega_peak_ref"]
    peak = p.c_factor == 0 ? abs(d["R_peak_h"] + 1) :          # R(ω̃) = −1
           p.c_factor == 1 ? abs(d["A_peak_h"] - 0.5) : NaN    # 𝒜(ω̃) = ½
    return (T_ρ=p.T_ρ, c_factor=p.c_factor, snap=String(p.snap), order=p.order,
            elements_per_wavelength=p.elements_per_wavelength, n_omega=p.n_omega,
            ndofs=d["ndofs"], ny=d["ny"], h_over_lambda=d["element_size"] / d["lambda_p"],
            omega_peak_h=d["omega_peak_h"], omega_peak_ref=ω̃,
            err_omega=abs(d["omega_peak_h"] / ω̃ - 1),
            width_h=abs(width_h), width_ref=abs(width_ref), err_width=abs(width_h / width_ref - 1),
            width_Qrad=(1 + p.c_factor) * ω̃ / d["Q_rad"],
            err_peak=peak, abs_T_peak=abs(d["T_peak_h"]), A_peak_h=d["A_peak_h"],
            err_q=abs(d["q_over_eta_peak_h"] / d["q_over_eta_peak_ref"] - 1),
            Q_rad=d["Q_rad"], dispersion_error=d["dispersion_error"],
            fit_residual=d["fit_residual_peak"], asymmetry=d["asymmetry_peak"], fe_solves=d["fe_solves"],
            wall_time_s=d["wall_time_s"])
end

const ERROR_COLUMNS = (:err_omega => L"\tilde\omega_r", :err_width => "half-power width",
                       :err_peak => L"|R + 1|\ (c = 0),\ |\mathcal{A} - \frac{1}{2}|\ (c = c_\mathrm{crit})")
const RATE_COLUMNS = (:err_omega, :err_width)

"Observed order log(eᵢ/eᵢ₊₁)/log(hᵢ/hᵢ₊₁) between the two finest meshes (NaN with one mesh)."
last_rate(h, e) = length(e) < 2 ? NaN : log(e[end - 1] / e[end]) / log(h[end - 1] / h[end])

"Ladder cases grouped by T_ρ, c, attachment and order: finest-mesh values and rates."
function rates(df::DataFrame)
    rows = NamedTuple[]
    ladder = filter(r -> r.n_omega == 0, df)
    for g in groupby(sort(ladder, [:T_ρ, :c_factor, :snap, :order, :elements_per_wavelength]),
                     [:T_ρ, :c_factor, :snap, :order])
        f = g[end, :]
        push!(rows, (T_ρ=f.T_ρ, c_factor=f.c_factor, snap=f.snap, order=f.order, finest=f.elements_per_wavelength,
                     err_omega_finest=f.err_omega, rate_err_omega=last_rate(g.h_over_lambda, g.err_omega),
                     err_width_finest=f.err_width, rate_err_width=last_rate(g.h_over_lambda, g.err_width),
                     err_peak_finest=f.err_peak, abs_T_peak_finest=f.abs_T_peak, err_q_finest=f.err_q))
    end
    return DataFrame(rows)
end

"Acceptance checks (one row per criterion)."
function checks(rt::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol,
                                                passed=value <= tol))
    for r in eachrow(rt)
        tag = @sprintf("T=%g, c/c_crit=%g, %s, p=%d", r.T_ρ, r.c_factor, r.snap, r.order)
        add!("ω̃, finest mesh, $(tag)", "D", r.err_omega_finest, 1e-3)
        add!("width, finest mesh, $(tag)", "D", r.err_width_finest, 1e-3)
        if r.order == 2
            for col in RATE_COLUMNS
                rate = r[Symbol(:rate_, col)]
                isnan(rate) && continue
                push!(rows, (name="rate $(col), $(tag)", kind="D", value=rate, tolerance=2(r.order - 1) - 0.3,
                             passed=rate >= 2(r.order - 1) - 0.3))
            end
        end
        if r.c_factor == 0
            add!("|R(ω̃) + 1|, finest mesh, $(tag)", "D", r.err_peak_finest, 1e-3)
        elseif r.c_factor == 1
            add!("|𝒜(ω̃) − ½|, finest mesh, $(tag)", "D", r.err_peak_finest, 1e-3)
        end
        add!("q/η at ω̃, $(tag)", "I", r.err_q_finest, 1e-10)
    end
    return DataFrame(rows)
end

"Response curves of the curve cases, one row per frequency."
function curves(cs, records)
    rows = NamedTuple[]
    for (p, d) in zip(cs, records)
        p.n_omega > 0 || continue
        for i in eachindex(d["omega"])
            push!(rows, (T_ρ=p.T_ρ, c_factor=p.c_factor, omega=d["omega"][i],
                         abs_R_h=abs(d["R_h"][i]), abs_T_h=abs(d["T_h"][i]), A_h=d["A_h"][i],
                         abs_R_ref=abs(d["R_ref"][i]), abs_T_ref=abs(d["T_ref"][i]), A_ref=d["A_ref"][i]))
        end
    end
    return DataFrame(rows)
end

# ─── 4. Figures and tables ──────────────────────────────────────────────────

const VARIANTS = ((0.0, "node") => (:black, "c = 0, node"), (1.0, "node") => (:firebrick, "critical c, node"),
                  (0.0, "midelement") => (:royalblue, "c = 0, mid-element"))

"""
One figure per polynomial order, one panel per observable: error against the
mesh size for every T_ρ (marker) and resonator variant (colour), with the
theoretical rate 2(p − 1) (dashed) from the largest error on the coarsest mesh.
"""
function plot_convergence(df::DataFrame, dir::AbstractString)
    markers = Dict(0.0 => :circle, 9.8 => :rect)
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        for p in sort(unique(df.order))
            dp = filter(r -> r.order == p && r.n_omega == 0, df)
            nrow(dp) == 0 && continue
            n = sort(unique(dp.elements_per_wavelength))
            fig = Figure(size=(1100, 340))
            local ax
            for (j, (col, title)) in enumerate(ERROR_COLUMNS)
                ax = Axis(fig[1, j]; xscale=log10, yscale=log10, title=title,
                          xlabel="elements per wavelength", ylabel=j == 1 ? "relative error" : "",
                          xticks=(n, string.(n)))
                for T in STUDY_T, ((c, s), (color, label)) in VARIANTS
                    g = sort(filter(r -> r.T_ρ == T && r.c_factor == c && r.snap == s, dp), :elements_per_wavelength)
                    (nrow(g) == 0 || all(isnan, g[!, col])) && continue
                    Ts = @sprintf("%g", T)
                    scatterlines!(ax, g.elements_per_wavelength, g[!, col]; color=color, marker=markers[T],
                                  label=label * ", Tρ = $(Ts)")
                end
                rate = 2(p - 1)
                e0 = maximum(filter(!isnan, filter(r -> r.elements_per_wavelength == n[1], dp)[!, col]))
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

"|R| and 𝒜 against ω, exact (lines) and FE (markers), one column per T_ρ."
function plot_curves(cv::DataFrame, dir::AbstractString)
    nrow(cv) == 0 && return
    colors = Dict(0.0 => :black, 0.5 => :royalblue, 1.0 => :firebrick, 2.0 => :forestgreen)
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        Ts = sort(unique(cv.T_ρ))
        fig = Figure(size=(380 * length(Ts) + 160, 540))
        local axR
        for (j, T) in enumerate(Ts)
            Tstr = @sprintf("%g", T)
            axR = Axis(fig[1, j]; ylabel=j == 1 ? L"|R|" : "", title=L"T_\rho = %$Tstr")
            axA = Axis(fig[2, j]; xlabel=L"\omega\;\mathrm{(rad/s)}", ylabel=j == 1 ? L"\mathcal{A}" : "")
            linkxaxes!(axR, axA)
            hlines!(axA, [0.5]; color=:gray60, linestyle=:dash, linewidth=1)
            for c in CURVE_C
                g = sort(filter(r -> r.T_ρ == T && r.c_factor == c, cv), :omega)
                cs = @sprintf("%g", c)
                lines!(axR, g.omega, g.abs_R_ref; color=colors[c], label=L"c/c_\mathrm{crit} = %$cs")
                scatter!(axR, g.omega, g.abs_R_h; color=colors[c], markersize=6)
                lines!(axA, g.omega, g.A_ref; color=colors[c])
                scatter!(axA, g.omega, g.A_h; color=colors[c], markersize=6)
            end
        end
        Legend(fig[1:2, length(Ts) + 1], axR; framevisible=false, labelsize=10)
        save(joinpath(dir, "response_curves.pdf"), fig)
        save(joinpath(dir, "response_curves.png"), fig; px_per_unit=300 / 96)
    end
end

function write_table(df::DataFrame, dir::AbstractString)
    mkpath(dir)
    e(x) = isnan(x) ? "--" : @sprintf("%.1e", x)
    ladder = filter(r -> r.n_omega == 0, df)
    finest = combine(groupby(ladder, [:T_ρ, :c_factor, :snap, :order]), g -> g[argmax(g.elements_per_wavelength), :])
    open(joinpath(dir, "resonator_table.tex"), "w") do io
        println(io, "\\begin{table}[htbp]\n\\centering")
        println(io, "\\caption{L1-3: single line resonator (P1, \$H = 10\$ m, \$m_{\\rho,r} = 0.01\$, \$\\omega_r = 10\$ rad/s), finest mesh: renormalised resonance, half-power width and peak response against the exact single-resonator solution.}")
        println(io, "\\label{tab:L1-3}\n\\begin{tabular}{rrlrrcccc}\n\\toprule")
        println(io, "\$T_\\rho\$ & \$c/c_{\\rm crit}\$ & attachment & \$p\$ & el./\$\\lambda_p\$ & \$\\tilde\\omega_r\$ & err. \$\\tilde\\omega_r\$ & err. width & peak \\\\\n\\midrule")
        for r in eachrow(sort(finest, [:T_ρ, :c_factor, :snap, :order]))
            println(io, join([r.T_ρ, r.c_factor, r.snap == "node" ? "node" : "mid-element", r.order,
                              r.elements_per_wavelength, @sprintf("%.6f", r.omega_peak_ref),
                              e(r.err_omega), e(r.err_width), e(r.err_peak)], " & "), " \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}\n\\end{table}")
    end
end

# ─── 5. Driver ──────────────────────────────────────────────────────────────

"""
    main(; smoke=false, force=false, root=REPO, only=nothing)

Run the study: compute or load every case, write the errors, rates, checks and
curves to `<root>/data/exp_pro/L1-3/`, the figures and table to
`<root>/plots/L1-3/`, and print a summary. Returns `(errors, checks)`. With
`only` (case indices), compute those records and return without post-processing.
"""
function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO,
              only::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cs = cases(; smoke=smoke)
    if only !== nothing
        for (i, p) in zip(only, cs[only])
            @info "case $(i)/$(length(cs))" p.T_ρ p.c_factor p.snap p.order p.elements_per_wavelength p.n_omega
            simulate(p; root=root, force=force)
        end
        return nothing
    end
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    rt = rates(df)
    ck = checks(rt)
    cv = curves(cs, records)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "rates.csv"), rt)
    CSV.write(joinpath(out, "checks.csv"), ck)
    CSV.write(joinpath(out, "curves.csv"), cv)
    figdir = joinpath(root, "plots", TEST_ID)
    plot_convergence(df, figdir)
    plot_curves(cv, figdir)
    write_table(df, figdir)

    show(stdout, MIME"text/plain"(), select(filter(r -> r.n_omega == 0, df), :T_ρ, :c_factor, :snap, :order,
                                            :elements_per_wavelength, :ndofs, :err_omega, :err_width, :err_peak,
                                            :abs_T_peak, :fe_solves); allrows=true)
    println()
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-60s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
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
