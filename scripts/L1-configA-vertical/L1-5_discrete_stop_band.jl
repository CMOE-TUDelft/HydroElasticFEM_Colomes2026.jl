#=
L1-5 — Discrete stop band of a line array.
Pre- and post-processing of the test defined in src/L1_5_discrete_stop_band.jl
(physics, relation to the paper and observables are documented there).

Study: P1 host (Table 1 of the paper), H = 10 m, line resonators
m_ρ,r = 0.01 m², ω_r = 10 rad/s, spacing a = 1 m, lossless.
  - convergence ladder: T_ρ ∈ {0, 9.8}; p = 2 with 8, 16, 32, 64 and p = 3
    with 4, 8, 16, 32 elements per cell (λ_p(ω_r)/a = 2.29 and 3.26).

Acceptance (test matrix §L1.5, §0.3), per T_ρ and order, on the finest mesh:
  - type D: relative error of ω_Γ, ω_X, of the interior band samples and of
    the bare-host frequency at q = π/a ≤ 1e-4; of the gap ω_Γ − ω_X ≤ 1e-3;
    for p = 2 the observed rate of ω_X and of the bare-host frequency on the
    two finest meshes ≥ 2(p − 1) − 0.3 (skipped when an error is below 1e-11,
    the round-off floor of the pole search);
  - type I: leakage |Im(1/g)| of the Bloch projection ≤ 1e-6 relative;
  - type M (must not match), p = 3: the FE edges are closer to the lattice-sum
    edges than to the truncated local edges [ω₋(π/a), ω_r] by a factor ≥ 100,
    and the FE gap is below half the homogenised anticrossing width.

Outputs (relative to the repository root):
  data/sims/L1-5/      one DrWatson record per case (git commit, patch, HydroElasticFEM pin)
  data/exp_pro/L1-5/   errors.csv, rates.csv, checks.csv, band_markers.csv (FE points for fig:lat_bands)
  plots/L1-5/          bands.pdf/png (lattice-sum bands, truncated local branches, FE markers),
                       convergence.pdf/png, edges_table.tex

Run:
  julia --project -O1 scripts/L1-configA-vertical/L1-5_discrete_stop_band.jl            # full study
  julia --project -O1 scripts/L1-configA-vertical/L1-5_discrete_stop_band.jl --smoke    # one coarse case
  julia --project -O1 scripts/L1-configA-vertical/L1-5_discrete_stop_band.jl --force    # recompute records
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_Colomes2026.L1_5
using HydroElasticFEM_Colomes2026.References: bloch_qa_1d, local_branches, resonator_from_frequency, HostSpec
import HydroElasticFEM_Colomes2026.L1_1

const TEST_ID = "L1-5"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ─── 1. Cases ───────────────────────────────────────────────────────────────

const STUDY_T = (0.0, 9.8)
const LADDER = (2 => (8, 16, 32, 64), 3 => (4, 8, 16, 32))

"Cases of the study (`smoke = true`: one coarse case, used by the test suite)."
function cases(; smoke::Bool=false)
    smoke && return [L1_5_Params(elements_per_cell=8, order=3)]
    return [L1_5_Params(T_ρ=T, elements_per_cell=n, order=p) for T in STUDY_T for (p, l) in LADDER for n in l]
end

# ─── 2. Simulation records (DrWatson) ───────────────────────────────────────

record_name(p::L1_5_Params) = savename(p; accesses=VARIABLE, sigdigits=4) * "_fixed=" * fixed_hash(p)

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

"Load the record of case `p`, or compute it with `L1_5.run_case` and save it with its provenance."
function simulate(p::L1_5_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L1_5_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L1_5_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── 3. Post-processing ─────────────────────────────────────────────────────

rel(h, r) = abs(h / r - 1)

"Errors of one case against the references, and the must-not-match ratios."
function errors(p::L1_5_Params, d)
    gap_h = d["omega_gamma_h"] - d["omega_X_h"]
    gap_ref = d["omega_gamma_ref"] - d["omega_X_ref"]
    band_err = isempty(d["band_omega_h"]) ? NaN : maximum(rel.(d["band_omega_h"], d["band_omega_ref"]))
    return (T_ρ=p.T_ρ, order=p.order, elements_per_cell=p.elements_per_cell, ndofs_cell=d["ndofs_cell"], ny=d["ny"],
            h=d["element_size"], h_over_lambda=d["element_size"] / d["lambda_p"],
            omega_X_h=d["omega_X_h"], omega_X_ref=d["omega_X_ref"],
            omega_gamma_h=d["omega_gamma_h"], omega_gamma_ref=d["omega_gamma_ref"],
            err_X=rel(d["omega_X_h"], d["omega_X_ref"]), err_gamma=rel(d["omega_gamma_h"], d["omega_gamma_ref"]),
            err_gap=rel(gap_h, gap_ref), err_band=band_err, n_band=length(d["band_omega_h"]),
            err_bare=rel(d["bare_omega_h"], d["bare_omega_ref"]),
            gap_over_omega_r=gap_h / p.ω_r, anticrossing=d["anticrossing_width"],
            liu_lower=d["liu_lower"], liu_ratio_X=abs(d["omega_X_h"] - d["omega_X_ref"]) / abs(d["omega_X_h"] - d["liu_lower"]),
            liu_ratio_gamma=abs(d["omega_gamma_h"] - d["omega_gamma_ref"]) / abs(d["omega_gamma_h"] - p.ω_r),
            Pi_L=d["Pi_L"], omega_gamma_PiL=d["omega_gamma_PiL"],
            leak=max(d["leak"], maximum(d["band_leak"]; init=0.0)), fe_solves=d["fe_solves"],
            wall_time_s=d["wall_time_s"])
end

"Observed order log(eᵢ/eᵢ₊₁)/log(hᵢ/hᵢ₊₁) between the two finest meshes (NaN below the round-off floor)."
last_rate(h, e; floor=1e-11) = (length(e) < 2 || min(e[end - 1], e[end]) < floor) ? NaN :
                               log(e[end - 1] / e[end]) / log(h[end - 1] / h[end])

const ERROR_COLUMNS = (:err_X => L"\omega_X", :err_gamma => L"\omega_\Gamma", :err_gap => L"\omega_\Gamma - \omega_X",
                       :err_band => "band samples", :err_bare => L"\omega_0(\pi/a)\ \mathrm{(bare)}")
const RATE_COLUMNS = (:err_X, :err_gamma, :err_bare)

"Finest-mesh values and rates per T_ρ and order."
function rates(df::DataFrame)
    rows = NamedTuple[]
    for g in groupby(sort(df, [:T_ρ, :order, :elements_per_cell]), [:T_ρ, :order])
        f = g[end, :]
        push!(rows, (T_ρ=f.T_ρ, order=f.order, finest=f.elements_per_cell,
                     err_X_finest=f.err_X, rate_err_X=last_rate(g.h, g.err_X),
                     err_gamma_finest=f.err_gamma, rate_err_gamma=last_rate(g.h, g.err_gamma),
                     err_gap_finest=f.err_gap, err_band_finest=f.err_band,
                     err_bare_finest=f.err_bare, rate_err_bare=last_rate(g.h, g.err_bare),
                     liu_ratio_X=f.liu_ratio_X, liu_ratio_gamma=f.liu_ratio_gamma,
                     gap_over_omega_r=f.gap_over_omega_r, anticrossing=f.anticrossing, leak=maximum(g.leak)))
    end
    return DataFrame(rows)
end

"Acceptance checks (one row per criterion)."
function checks(rt::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    for r in eachrow(rt)
        tag = @sprintf("T=%g, p=%d", r.T_ρ, r.order)
        add!("ω_X, finest mesh, $(tag)", "D", r.err_X_finest, 1e-4)
        add!("ω_Γ, finest mesh, $(tag)", "D", r.err_gamma_finest, 1e-4)
        add!("gap ω_Γ − ω_X, finest mesh, $(tag)", "D", r.err_gap_finest, 1e-3)
        add!("band samples, finest mesh, $(tag)", "D", r.err_band_finest, 1e-4)
        add!("bare ω₀(π/a), finest mesh, $(tag)", "D", r.err_bare_finest, 1e-4)
        if r.order == 2
            for col in RATE_COLUMNS
                rate = r[Symbol(:rate_, col)]
                isnan(rate) && continue
                push!(rows, (name="rate $(col), $(tag)", kind="D", value=rate, tolerance=2(r.order - 1) - 0.3,
                             passed=rate >= 2(r.order - 1) - 0.3))
            end
        end
        add!("Bloch projection leakage, $(tag)", "I", r.leak, 1e-6)
        if r.order == 3
            add!("ω_X closer to lattice sum than to ω₋(π/a) (ratio), $(tag)", "M", r.liu_ratio_X, 1e-2)
            add!("ω_Γ closer to lattice sum than to ω_r (ratio), $(tag)", "M", r.liu_ratio_gamma, 1e-2)
            add!("gap / anticrossing width, $(tag)", "M", r.gap_over_omega_r / r.anticrossing, 0.5)
        end
    end
    return DataFrame(rows)
end

"FE band points of the finest p = 3 case per T_ρ (Γ, X and interior samples), for the paper figure."
function band_markers(cs, records)
    rows = NamedTuple[]
    for T in STUDY_T
        idx = [i for (i, p) in enumerate(cs) if p.T_ρ == T && p.order == 3]
        isempty(idx) && continue
        i = idx[argmax([cs[k].elements_per_cell for k in idx])]
        p, d = cs[i], records[i]
        push!(rows, (T_ρ=T, point="Gamma", qa=0.0, omega_h=d["omega_gamma_h"], omega_ref=d["omega_gamma_ref"]))
        push!(rows, (T_ρ=T, point="X", qa=π, omega_h=d["omega_X_h"], omega_ref=d["omega_X_ref"]))
        for (qa, ωh, ωr) in zip(d["band_qa"], d["band_omega_h"], d["band_omega_ref"])
            push!(rows, (T_ρ=T, point="interior", qa=qa, omega_h=ωh, omega_ref=ωr))
        end
    end
    return DataFrame(rows)
end

# ─── 4. Figures and tables ──────────────────────────────────────────────────

"""
Bands near ω_r, one panel per T_ρ: lattice-sum bands (lines, from
`bloch_qa_1d`), truncated local branches of the homogenised model (dashed),
FE points (markers), the FE stop band (shaded) and the truncated local gap
[ω₋(π/a), ω_r] (dotted).
"""
function plot_bands(markers::DataFrame, p0::L1_5_Params, dir::AbstractString; ωlim=(9.2, 10.4))
    nrow(markers) == 0 && return
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        Ts = sort(unique(markers.T_ρ))
        fig = Figure(size=(360 * length(Ts) + 170, 380))
        local ax
        for (j, T) in enumerate(Ts)
            p = L1_5_Params(p0; T_ρ=T)
            h = L1_1.host(p)
            r = resonator_from_frequency(p.m_ρr, p.ω_r)
            Tstr = @sprintf("%g", T)
            ax = Axis(fig[1, j]; xlabel=L"qa", ylabel=j == 1 ? L"\omega\ \mathrm{(rad/s)}" : "",
                      title=L"T_\rho = %$Tstr", xticks=([0, π / 2, π], ["0", "π/2", "π"]), limits=(0, π, ωlim...))
            m = filter(r -> r.T_ρ == T, markers)
            ωX, ωΓ = only(m[m.point .== "X", :omega_h]), only(m[m.point .== "Gamma", :omega_h])
            band!(ax, [0, π], [ωX, ωX], [ωΓ, ωΓ]; color=(:gray, 0.2), label="stop band (FE)")
            ωs = range(ωlim...; length=1500)
            pts = [(qa, ω) for ω in ωs for qa in bloch_qa_1d(h, r, p.a, ω)]
            scatter!(ax, first.(pts), last.(pts); color=:black, markersize=1.5, label="lattice sums")
            ks = range(1e-3, π / p.a; length=400)
            br = local_branches.(Ref(h), Ref(r), 1 / p.a, ks)
            lines!(ax, ks .* p.a, getfield.(br, :lower); color=:royalblue, linestyle=:dash, label="local, truncated at π/a")
            lines!(ax, ks .* p.a, getfield.(br, :upper); color=:royalblue, linestyle=:dash)
            hlines!(ax, [local_branches(h, r, 1 / p.a, π / p.a).lower, p.ω_r]; color=:royalblue, linestyle=:dot,
                    linewidth=1)
            scatter!(ax, m.qa, m.omega_h; color=:firebrick, marker=:circle, markersize=8, label="FE (p = 3)")
        end
        Legend(fig[1, length(Ts) + 1], ax; framevisible=false, labelsize=10, unique=true)
        save(joinpath(dir, "bands.pdf"), fig)
        save(joinpath(dir, "bands.png"), fig; px_per_unit=300 / 96)
    end
end

"Error against elements per cell, one panel per observable; colour: order, marker: T_ρ; dashed: 2(p − 1)."
function plot_convergence(df::DataFrame, dir::AbstractString)
    markers = Dict(0.0 => :circle, 9.8 => :rect)
    colors = Dict(2 => :black, 3 => :royalblue)
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        fig = Figure(size=(1500, 340))
        local ax
        for (j, (col, title)) in enumerate(ERROR_COLUMNS)
            ax = Axis(fig[1, j]; xscale=log10, yscale=log10, title=title, xlabel="elements per cell",
                      ylabel=j == 1 ? "relative error" : "")
            for T in STUDY_T, pp in (2, 3)
                g = sort(filter(r -> r.T_ρ == T && r.order == pp, df), :elements_per_cell)
                v = max.(g[!, col], 1e-16)
                (nrow(g) == 0 || all(isnan, v)) && continue
                scatterlines!(ax, g.elements_per_cell, v; color=colors[pp], marker=markers[T],
                              label=@sprintf("p = %d, Tρ = %g", pp, T))
            end
            g = sort(filter(r -> r.order == 2, df), :elements_per_cell)
            if nrow(g) > 0
                n = sort(unique(g.elements_per_cell))
                e0 = maximum(filter(isfinite, g[g.elements_per_cell .== n[1], col]); init=1e-3)
                lines!(ax, n, e0 .* (n[1] ./ n) .^ 2; color=:gray40, linestyle=:dash, label=L"\mathcal{O}(h^2)")
            end
        end
        Legend(fig[1, length(ERROR_COLUMNS) + 1], ax; framevisible=false, labelsize=10)
        save(joinpath(dir, "convergence.pdf"), fig)
        save(joinpath(dir, "convergence.png"), fig; px_per_unit=300 / 96)
    end
end

function write_table(df::DataFrame, dir::AbstractString)
    mkpath(dir)
    e(x) = isnan(x) ? "--" : @sprintf("%.1e", x)
    finest = combine(groupby(df, [:T_ρ, :order]), g -> g[argmax(g.elements_per_cell), :])
    open(joinpath(dir, "edges_table.tex"), "w") do io
        println(io, "\\begin{table}[htbp]\n\\centering")
        println(io, "\\caption{L1-5: band edges of a line array (P1, \$H = 10\$ m, \$a = 1\$ m, \$m_{\\rho,r} = 0.01\$ m\$^2\$, \$\\omega_r = 10\$ rad/s): lattice sums, finite element (finest mesh) and the truncated local edges \$[\\omega_-(\\pi/a), \\omega_r]\$ that must not be matched.}")
        println(io, "\\label{tab:L1-5}\n\\begin{tabular}{rrccccccc}\n\\toprule")
        println(io, "\$T_\\rho\$ & \$p\$ & \$\\omega_X\$ & err. & \$\\omega_\\Gamma\$ & err. & err. gap & \$\\omega_-(\\pi/a)\$ & \$\\Delta\\omega/\\omega_r\$ \\\\\n\\midrule")
        for r in eachrow(sort(finest, [:T_ρ, :order]))
            println(io, join([r.T_ρ, r.order, @sprintf("%.6f", r.omega_X_ref), e(r.err_X),
                              @sprintf("%.6f", r.omega_gamma_ref), e(r.err_gamma), e(r.err_gap),
                              @sprintf("%.3f", r.liu_lower), @sprintf("%.4f", r.gap_over_omega_r)], " & "), " \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}\n\\end{table}")
    end
end

# ─── 5. Driver ──────────────────────────────────────────────────────────────

"""
    main(; smoke=false, force=false, root=REPO)

Run the study: compute or load every case, write errors, rates, checks and the
FE band markers to `<root>/data/exp_pro/L1-5/`, figures and table to
`<root>/plots/L1-5/`, and print a summary. Returns `(errors, checks)`.
"""
function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO)
    cs = cases(; smoke=smoke)
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    rt = rates(df)
    ck = checks(rt)
    bm = band_markers(cs, records)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "rates.csv"), rt)
    CSV.write(joinpath(out, "checks.csv"), ck)
    CSV.write(joinpath(out, "band_markers.csv"), bm)
    figdir = joinpath(root, "plots", TEST_ID)
    plot_bands(bm, first(cs), figdir)
    plot_convergence(df, figdir)
    write_table(df, figdir)

    show(stdout, MIME"text/plain"(), select(df, :T_ρ, :order, :elements_per_cell, :ndofs_cell, :err_X, :err_gamma,
                                            :err_gap, :err_band, :err_bare, :leak, :fe_solves); allrows=true)
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
