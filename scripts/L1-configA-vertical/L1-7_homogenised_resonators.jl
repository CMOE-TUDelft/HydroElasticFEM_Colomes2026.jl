#=
L1-7 — Homogenised resonator field and the lattice-compliance number.
Pre- and post-processing of the test defined in src/L1_7_homogenised_resonators.jl
(physics, relation to the paper and observables are documented there).

ENVIRONMENT. The homogenised field uses the upstream `ResonatorField`, which is
on the HydroElasticFEM.jl branch `feat/resonator-field` (not merged yet). L1-7
therefore runs in its own environment, envs/resonator_field, which develops
that branch's worktree and this repository; the root environment is not
touched. After the PR merges, L1-7 moves to the root environment: include
src/L1_7_homogenised_resonators.jl in the package module (see its header),
replace the `include` below by `using HydroElasticFEM_xxx2026.L1_7`, and run
with `julia --project -O1`.

Study: P1 host (Table 1 of the paper), H = 10 m, ω_r = 10 rad/s, lossless,
T_ρ ∈ {0, 9.8}.
  (i), (i-b) homogenised field, n₀m_ρ,r = 0.01 m (n₀ = 1 /m, m_ρ,r = 0.01 m²):
    - sweep: 400 frequencies, ω ∈ [8, 12] on a 401-point grid without ω_r,
      p = 3, 16 elements per wavelength 2π/k;
    - convergence ladder on 10 frequencies (8, 9, 9.5, 9.9, 9.99, 10.01, 10.1,
      10.5, 11, 12): p = 2 with 8, 16, 32 and p = 3 with 4, 8, 16 elements
      per wavelength.
  (ii) point lattice, L1-5 cell (one cell, uniform load), p = 3, fixed loading
    n₀m_ρ,r = 0.01 m (m_ρ,r = 0.01a, k_ρ,r = a): a ∈ {0.5, 1, 2, 3} m
    (Π_L ≈ a⁴/352.8 for T_ρ = 0), 16 and 32 elements per metre.

Acceptance (frozen in verification_test_matrix_2026-10-22.md §L1.7):
  - (i) type I, per T_ρ: on the sweep, max |k_h/k − 1| ≤ 1e-4 against the root
    of eq:omega_k_x_closed_form, and max |ω±(k_h)/ω − 1| ≤ 1e-4; ladder,
    finest mesh ≤ 1e-4 per order; p = 2 observed rate on the two finest
    meshes ≥ 2(p − 1) − 0.3; Bloch projection leakage ≤ 1e-6; field/plate
    amplitude ratio q/η = K/(K − ω²M) to 1e-8 (q and η share the FE space);
  - (i-b) type I, per T_ρ: number of sampled ω ≠ ω_r without a converged
    real FE root on the branch of the closed form (ω₋ below ω_r, ω₊ above) = 0;
  - (ii) type D, per T_ρ and a, finest mesh: FE ω_Γ against the lattice sums
    ≤ 1e-4 (relative); FE resolution of δ = ω_Γ − ω_r/√(1 + Π_L):
    |δ_h − δ_ref| ≤ 1e-2 |δ_ref|;
  - (ii) type A, per T_ρ: |δ|/ω_r ≤ Π_L² at every a, and the fitted slope
    d ln|δ|/d ln Π_L over the four spacings ≥ 2 − 0.3.

Outputs (relative to the repository root):
  data/sims/L1-7/      one DrWatson record per case (git commit, patch, HydroElasticFEM worktree commit)
  data/exp_pro/L1-7/   sweep.csv, ladder.csv, rates.csv, gamma.csv, checks.csv
  plots/L1-7/          bands_overlay.pdf/png (FE overlay on the discrete and homogenised bands,
                       δ against Π_L), k_error.pdf/png (k_h error against ω)

Run (one heavy run at a time; split with --cases):
  julia --project=envs/resonator_field -O1 scripts/L1-configA-vertical/L1-7_homogenised_resonators.jl [--cases G] [--smoke] [--force]
    G ∈ {ladder, full0, full9.8, gamma, gamma0, gamma9.8}: compute only that group and stop (no post-processing).
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_xxx2026
using HydroElasticFEM_xxx2026.References: bloch_qa_1d, resonator_from_frequency, HostSpec
import HydroElasticFEM_xxx2026.L1_1

const TEST_ID = "L1-7"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const ENV = joinpath(REPO, "envs", "resonator_field")

include(joinpath(REPO, "src", "L1_7_homogenised_resonators.jl"))
using .L1_7

# ─── 1. Cases ───────────────────────────────────────────────────────────────

const STUDY_T = (0.0, 9.8)
const LADDER = (2 => (8, 16, 32), 3 => (4, 8, 16))
const SWEEP = (order=3, elements_per_wavelength=16)
const SPACINGS = (0.5, 1.0, 2.0, 3.0)
const GAMMA_MESH = (16, 32)

"Cases of the study, by group (`smoke = true`: one coarse case per group)."
function cases(; smoke::Bool=false)
    if smoke
        return Dict("ladder" => [L1_7_Params(set=:ladder, order=3, elements_per_wavelength=4)],
                    "gamma" => [L1_7_Gamma_Params(a=1.0, elements_per_metre=16)])
    end
    gamma(Ts) = [L1_7_Gamma_Params(a=a, T_ρ=T, elements_per_metre=n) for T in Ts for a in SPACINGS for n in GAMMA_MESH]
    return Dict(
        "gamma0" => gamma((0.0,)), "gamma9.8" => gamma((9.8,)),
        "ladder" => [L1_7_Params(T_ρ=T, set=:ladder, order=p, elements_per_wavelength=n)
                     for T in STUDY_T for (p, l) in LADDER for n in l],
        "full0" => [L1_7_Params(; T_ρ=0.0, set=:full, SWEEP...)],
        "full9.8" => [L1_7_Params(; T_ρ=9.8, set=:full, SWEEP...)],
        "gamma" => gamma(STUDY_T),
    )
end

# ─── 2. Simulation records (DrWatson) ───────────────────────────────────────

record_name(p) = savename(p; accesses=L1_7._variable(p), sigdigits=4) * "_fixed=" * fixed_hash(p)

"HydroElasticFEM as developed in envs/resonator_field (path and the worktree commit)."
function hefem_pin()
    manifest = joinpath(ENV, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    path = get(e, "path", "")
    rev = isdir(path) ? readchomp(`git -C $path rev-parse HEAD`) : "unknown"
    branch = isdir(path) ? readchomp(`git -C $path rev-parse --abbrev-ref HEAD`) : "unknown"
    return Dict("hefem_source" => get(e, "repo-url", path), "hefem_rev" => rev, "hefem_branch" => branch,
                "hefem_version" => string(pkgversion(HydroElasticFEM)), "environment" => "envs/resonator_field")
end

"Load the record of case `p`, or compute it with `L1_7.run_case` and save it with its provenance."
function simulate(p; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(typeof(p)))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(typeof(p))) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── 3. Post-processing ─────────────────────────────────────────────────────

rel(h, r) = abs(h / r - 1)

"One row per sampled frequency of a homogenised-field case."
function field_rows(p::L1_7_Params, d)
    n = length(d["omega"])
    return DataFrame(T_ρ=fill(p.T_ρ, n), order=fill(p.order, n), elements_per_wavelength=fill(p.elements_per_wavelength, n),
                     omega=d["omega"], branch=d["branch"], k_ref=d["k_ref"], k_h=d["k_h"],
                     err_k=rel.(d["k_h"], d["k_ref"]), err_omega=rel.(d["omega_closed_at_k_h"], d["omega"]),
                     residual=d["residual"], leak=d["leak"], q_ratio=d["q_ratio"], solves=d["solves"],
                     ndofs=d["ndofs"], ny=d["ny"])
end

"Observed order between the two finest meshes (NaN below the round-off floor)."
last_rate(n, e; floor=1e-11) = (length(e) < 2 || min(e[end - 1], e[end]) < floor) ? NaN :
                               log(e[end - 1] / e[end]) / log(n[end] / n[end - 1])

"Ladder: max error over the 10 frequencies per mesh, and the rate on the two finest meshes."
function ladder_rates(ld::DataFrame)
    per_mesh = combine(groupby(ld, [:T_ρ, :order, :elements_per_wavelength]),
                       :err_k => (x -> maximum(replace(x, NaN => Inf))) => :err_k_max,
                       :leak => maximum => :leak_max)
    rows = NamedTuple[]
    for g in groupby(sort(per_mesh, [:T_ρ, :order, :elements_per_wavelength]), [:T_ρ, :order])
        push!(rows, (T_ρ=g.T_ρ[1], order=g.order[1], finest=g.elements_per_wavelength[end], err_k_finest=g.err_k_max[end],
                     rate=last_rate(g.elements_per_wavelength, g.err_k_max), leak=maximum(g.leak_max)))
    end
    return per_mesh, DataFrame(rows)
end

"(ii): one row per (T_ρ, a) on the finest mesh, with δ = ω_Γ − ω_r/√(1 + Π_L) from the FE and the lattice sums."
function gamma_rows(cs, recs)
    rows = NamedTuple[]
    for (p, d) in zip(cs, recs)
        ωr = p.ω_r
        δh = d["omega_gamma_h"] - d["omega_gamma_PiL"]
        δr = d["omega_gamma_ref"] - d["omega_gamma_PiL"]
        push!(rows, (T_ρ=p.T_ρ, a=p.a, elements_per_metre=p.elements_per_metre, elements_per_cell=d["elements_per_cell"],
                     ndofs_cell=d["ndofs_cell"], Pi_L=d["Pi_L"], Pi_L_estimate=d["Pi_L_estimate"],
                     omega_gamma_h=d["omega_gamma_h"], omega_gamma_ref=d["omega_gamma_ref"],
                     omega_gamma_PiL=d["omega_gamma_PiL"], err_gamma=rel(d["omega_gamma_h"], d["omega_gamma_ref"]),
                     delta_h=δh / ωr, delta_ref=δr / ωr, delta_over_PiL2=abs(δh) / ωr / d["Pi_L"]^2,
                     delta_resolution=abs(δh - δr) / abs(δr), leak=d["leak"], k_c_a_over_2pi=d["k_c"] / d["two_pi_over_a"]))
    end
    return sort(DataFrame(rows), [:T_ρ, :a, :elements_per_metre])
end

"Least-squares slope of log|δ| against log Π_L."
function loglog_slope(x, y)
    lx, ly = log.(x), log.(abs.(y))
    mx, my = sum(lx) / length(lx), sum(ly) / length(ly)
    return sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
end

"Acceptance checks (one row per criterion)."
function checks(sw::DataFrame, rt::DataFrame, ld::DataFrame, gm::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    for T in STUDY_T
        s = filter(r -> r.T_ρ == T, sw)
        nrow(s) == 0 && continue
        tag = @sprintf("T=%g", T)
        ok = .!isnan.(s.k_h)
        add!("(i) sweep, max |k_h/k − 1| over $(nrow(s)) ω, $(tag)", "I", maximum(replace(s.err_k, NaN => Inf)), 1e-4)
        add!("(i) sweep, max |ω±(k_h)/ω − 1|, $(tag)", "I", maximum(replace(s.err_omega, NaN => Inf)), 1e-4)
        add!("(i) sweep, Bloch projection leakage, $(tag)", "I", maximum(s.leak), 1e-6)
        add!("(i) sweep, q/η against K/(K − ω²M), $(tag)", "I", maximum(s.q_ratio), 1e-8)
        add!("(i-b) sampled ω ≠ ω_r without a real FE root on one branch (count), $(tag)", "I", count(!, ok), 0)
        add!("(i-b) ω < ω_r on ω₋ and ω > ω_r on ω₊ (misassigned count), $(tag)", "I",
             count(r -> (r.omega < 10.0) != (r.branch == "lower"), eachrow(s)), 0)
    end
    for r in eachrow(rt)
        tag = @sprintf("T=%g, p=%d", r.T_ρ, r.order)
        add!("(i) ladder, finest mesh ($(r.finest)/λ), max |k_h/k − 1|, $(tag)", "I", r.err_k_finest, 1e-4)
        r.order == 2 && !isnan(r.rate) &&
            push!(rows, (name="(i) ladder, rate of max |k_h/k − 1|, $(tag)", kind="I", value=r.rate,
                         tolerance=2(r.order - 1) - 0.3, passed=r.rate >= 2(r.order - 1) - 0.3))
        add!("(i) ladder, Bloch projection leakage, $(tag)", "I", r.leak, 1e-6)
    end
    for T in STUDY_T
        g = filter(r -> r.T_ρ == T && r.elements_per_metre == maximum(GAMMA_MESH), gm)
        nrow(g) == 0 && continue
        for r in eachrow(g)
            tag = @sprintf("T=%g, a=%g", T, r.a)
            add!("(ii) ω_Γ against lattice sums, $(tag)", "D", r.err_gamma, 1e-4)
            add!("(ii) FE resolution of δ = ω_Γ − ω_r/√(1+Π_L), |δ_h − δ_ref|/|δ_ref|, $(tag)", "D", r.delta_resolution, 1e-2)
            add!("(ii) |δ|/(ω_r Π_L²), $(tag)", "A", r.delta_over_PiL2, 1.0)
        end
        # slope over the near-field regime 2π/a > k_c (Π_L > 0); for 2π/a < k_c (a = 3 m) the folded
        # bare branch lies below ω_r, Π_L < 0 and ω_Γ > ω_r (reported, not fitted)
        nf = filter(r -> r.k_c_a_over_2pi < 1 && r.Pi_L > 0, g)
        slope = loglog_slope(nf.Pi_L, nf.delta_h)
        push!(rows, (name=@sprintf("(ii) slope d ln|δ|/d ln Π_L over a ∈ {%s} (2π/a > k_c), T=%g",
                                   join((@sprintf("%g", a) for a in nf.a), ", "), T), kind="A",
                     value=slope, tolerance=1.7, passed=slope >= 1.7))
    end
    return DataFrame(rows)
end

# ─── 4. Figures ─────────────────────────────────────────────────────────────

"""
FE overlay on the bands of `discrete_vs_homogenised_bands` (Section 4), one
panel per T_ρ: lattice-sum bands (dots), local branches (dashed), the
lattice-sum stop band (shaded), ω_r (dotted), the homogenised FE wavenumbers
k_h(ω) (open circles, every `stride`-th sweep frequency with k_h ≤ π/a) and
the discrete FE band points of L1-5 and ω_Γ of L1-7 (ii) at a = 1 (squares).
Third panel: |ω_Γ − ω_r/√(1 + Π_L)|/ω_r against Π_L (FE), with Π_L².
"""
function plot_overlay(sw::DataFrame, gm::DataFrame, dir::AbstractString; a=1.0, stride=8, ωlim=(6.0, 13.0))
    mkpath(dir)
    l15 = joinpath(REPO, "data", "exp_pro", "L1-5", "band_markers.csv")
    disc = isfile(l15) ? CSV.read(l15, DataFrame) : DataFrame(T_ρ=Float64[], qa=Float64[], omega_h=Float64[])
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    with_theme(theme) do
        fig = Figure(size=(1180, 380))
        local ax
        for (j, T) in enumerate(STUDY_T)
            p = L1_7_Params(T_ρ=T)
            h = L1_1.host(p)
            r = resonator_from_frequency(p.m_ρr * a, p.ω_r)     # point resonator of the lattice, n₀ = 1/a
            Tstr = @sprintf("%g", T)
            ax = Axis(fig[1, j]; xlabel=L"qa/\pi", ylabel=j == 1 ? L"\omega\ \mathrm{(rad/s)}" : "",
                      title=L"T_\rho = %$Tstr", limits=(0, 1, ωlim...))
            g = filter(x -> x.T_ρ == T && x.a == a && x.elements_per_metre == maximum(GAMMA_MESH), gm)
            ωs = range(ωlim...; length=1400)
            pts = [(qa / π, ω) for ω in ωs for qa in bloch_qa_1d(h, r, a, ω)]
            if nrow(g) == 1
                ωΓ = g.omega_gamma_ref[1]
                below = [w for (q, w) in pts if w < ωΓ - 1e-6 && w > 9]
                isempty(below) || band!(ax, [0, 1], fill(maximum(below), 2), fill(ωΓ, 2); color=(:orange, 0.3),
                                        label="stop band (lattice sums)")
            end
            scatter!(ax, first.(pts), last.(pts); color=:black, markersize=2, label="lattice sums")
            ks = range(2e-3, π / a; length=800)
            br = L1_7.closed_form_branches.(Ref(h), Ref(r), 1 / a, ks)
            lines!(ax, ks .* a ./ π, getfield.(br, :lower); color=:royalblue, linestyle=:dash, label="local branches")
            lines!(ax, ks .* a ./ π, getfield.(br, :upper); color=:royalblue, linestyle=:dash)
            hlines!(ax, [p.ω_r]; color=:gray, linestyle=:dot, linewidth=1)
            s = filter(x -> x.T_ρ == T && !isnan(x.k_h) && x.k_h * a <= π, sw)
            s = s[1:stride:end, :]
            scatter!(ax, s.k_h .* a ./ π, s.omega; color=:transparent, strokecolor=:firebrick, strokewidth=1.2,
                     marker=:circle, markersize=7, label="FE, homogenised field")
            dm = filter(x -> x.T_ρ == T, disc)
            dq = vcat(dm.qa ./ π, nrow(g) == 1 ? [0.0] : Float64[])
            dω = vcat(dm.omega_h, nrow(g) == 1 ? g.omega_gamma_h : Float64[])
            scatter!(ax, dq, dω; color=:seagreen, marker=:rect, markersize=8, label="FE, point resonators")
        end
        Legend(fig[1, 3], ax; framevisible=false, labelsize=10, unique=true)
        ax3 = Axis(fig[1, 4]; xscale=log10, yscale=log10, xlabel=L"\Pi_L",
                   ylabel=L"|\omega_\Gamma - \omega_r/\sqrt{1+\Pi_L}|/\omega_r", title="zone-centre edge")
        for (T, mk) in zip(STUDY_T, (:circle, :rect))
            g = filter(x -> x.T_ρ == T && x.elements_per_metre == maximum(GAMMA_MESH), gm)
            nrow(g) == 0 && continue
            nf = g.Pi_L .> 0
            scatterlines!(ax3, g.Pi_L[nf], abs.(g.delta_h[nf]); marker=mk, color=:black, label=@sprintf("FE, Tρ = %g", T))
            any(.!nf) && scatter!(ax3, abs.(g.Pi_L[.!nf]), abs.(g.delta_h[.!nf]); marker=mk, color=:white,
                                  strokecolor=:black, strokewidth=1, label="FE, Π_L < 0 (|Π_L|), a = 3 m")
        end
        if nrow(gm) > 0
            Π = sort(unique(abs.(gm.Pi_L)))
            lines!(ax3, Π, Π .^ 2; color=:gray40, linestyle=:dash, label=L"\Pi_L^2")
        end
        axislegend(ax3; position=:lt, framevisible=false, labelsize=9, unique=true)
        colsize!(fig.layout, 3, Auto(0.45))
        save(joinpath(dir, "bands_overlay.pdf"), fig)
        save(joinpath(dir, "bands_overlay.png"), fig; px_per_unit=300 / 96)
    end
end

"|k_h/k − 1| against ω on the sweep, one line per T_ρ."
function plot_k_error(sw::DataFrame, dir::AbstractString)
    nrow(sw) == 0 && return
    mkpath(dir)
    with_theme(Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))) do
        fig = Figure(size=(560, 340))
        ax = Axis(fig[1, 1]; yscale=log10, xlabel=L"\omega\ \mathrm{(rad/s)}", ylabel=L"|k_h/k - 1|",
                  title="homogenised field, p = 3, 16 el./λ")
        for (T, c) in zip(STUDY_T, (:black, :royalblue))
            s = filter(x -> x.T_ρ == T && !isnan(x.k_h), sw)
            nrow(s) == 0 && continue
            scatter!(ax, s.omega, max.(s.err_k, 1e-16); color=c, markersize=3, label=@sprintf("Tρ = %g", T))
        end
        vlines!(ax, [10.0]; color=:gray, linestyle=:dot)
        axislegend(ax; position=:rb, framevisible=false)
        save(joinpath(dir, "k_error.pdf"), fig)
        save(joinpath(dir, "k_error.png"), fig; px_per_unit=300 / 96)
    end
end

# ─── 5. Driver ──────────────────────────────────────────────────────────────

"""
    main(; smoke=false, force=false, groups=nothing, root=REPO)

Compute or load every case (only the cases of `groups` when given, then
stop), write the tables and checks to `<root>/data/exp_pro/L1-7/` and the
figures to `<root>/plots/L1-7/`, and print a summary.
"""
function main(; smoke::Bool=false, force::Bool=false, groups=nothing, root::AbstractString=REPO)
    cs = cases(; smoke=smoke)
    if groups !== nothing
        for g in groups, p in cs[g]
            t = @elapsed simulate(p; root=root, force=force)
            @printf("[%s] %s  %.1f s\n", g, record_name(p), t)
        end
        return nothing
    end
    field_cases = vcat((get(cs, g, L1_7_Params[]) for g in ("ladder", "full0", "full9.8"))...)
    field_recs = [simulate(p; root=root, force=force) for p in field_cases]
    rows = [field_rows(p, d) for (p, d) in zip(field_cases, field_recs)]
    stack(v) = isempty(v) ? first(rows)[1:0, :] : vcat(v...; cols=:union)   # (empty: no case of the set)
    sw = stack([r for (p, r) in zip(field_cases, rows) if p.set === :full])
    ld = stack([r for (p, r) in zip(field_cases, rows) if p.set === :ladder])
    per_mesh, rt = ladder_rates(ld)
    gcs = cs["gamma"]
    gm = gamma_rows(gcs, [simulate(p; root=root, force=force) for p in gcs])
    ck = checks(sw, rt, ld, gm)

    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "sweep.csv"), sw)
    CSV.write(joinpath(out, "ladder.csv"), ld)
    CSV.write(joinpath(out, "rates.csv"), leftjoin(per_mesh, rt; on=[:T_ρ, :order], makeunique=true))
    CSV.write(joinpath(out, "gamma.csv"), gm)
    CSV.write(joinpath(out, "checks.csv"), ck)
    figdir = joinpath(root, "plots", TEST_ID)
    plot_overlay(sw, gm, figdir)
    plot_k_error(sw, figdir)

    show(stdout, MIME"text/plain"(), per_mesh; allrows=true); println()
    show(stdout, MIME"text/plain"(), rt; allrows=true); println()
    show(stdout, MIME"text/plain"(), select(gm, :T_ρ, :a, :elements_per_metre, :Pi_L, :Pi_L_estimate, :err_gamma,
                                            :delta_h, :delta_ref, :delta_over_PiL2, :delta_resolution); allrows=true)
    println()
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-80s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (sweep=sw, ladder=ld, rates=rt, gamma=gm, checks=ck)
end

if abspath(PROGRAM_FILE) == @__FILE__
    i = findfirst(==("--cases"), ARGS)
    groups = i === nothing ? nothing : split(ARGS[i + 1], ",")
    main(; smoke="--smoke" in ARGS, force="--force" in ARGS, groups=groups)
end
