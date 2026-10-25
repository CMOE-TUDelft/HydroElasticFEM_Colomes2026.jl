#=
L2-5 — Finite-size cut-off of the fluid dilution.
Pre- and post-processing of the test defined in src/L2_5_finite_size_dilution.jl.

Study: P1 plate (T_ρ = 0), free edges, uniform lossless resonators every ≈ 1 m
at n₀m_ρ,r = 0.01 m tuned to the first elastic wet mode; L ∈ {2.3, 4.6, 11.5,
23, 46} m (≈ {1, 2, 5, 10, 20} λ_p at ω = 10), H ∈ {3, 10, 30} m; p = 3 with 16
elements per wavelength, and 32 at H = 10 m for L ≤ 11.5 m (mesh check).

Acceptance (test matrix §L2.5):
  - type I: the hybridised modes have a projection ≥ 0.2 on the bare mode (the
    pair is identified);
  - type A: FE splitting against the single-mode Galerkin prediction (bare
    wet mode, frequency-dependent modal added mass), to 1 % (5.5e-4 on the smoke case); the constant-mass
    two-degree-of-freedom value √μ_mode is reported (it misses the dispersion
    of the added mass between ω₋ and ω₊, ≈ 6 % on the smoke case);
  - type A: deep water (H = 30 m, L ≤ 23 m, L < πH/4): fitted exponent of the
    splitting against L within −½ ± 0.1 (paper: μ_B ∝ 1/L for L ≪ πH);
  - type D: splitting on 16 and 32 elements per wavelength (H = 10 m,
    L ≤ 11.5 m; at larger L the 32-element case exceeds the per-run budget), to 1e-2;
  - type I: the one-solve modal mass m_φ(ω₁) against φᵀ(M_η + Re A)φ from the
    full added-mass matrix, to 1e-6.
  Reported: the exponent for shallow water (H = 3 m, L ≥ 11.5 m; expected → −1),
  and the paper's √μ_B and the local √μ(β₁/L) against the FE.

Outputs:
  data/sims/L2-5/      one DrWatson record per case
  data/exp_pro/L2-5/   errors.csv, checks.csv
  plots/L2-5/          splitting.pdf/png (splitting against L for each H)

Run:
  julia --project -O1 scripts/L2-configB-vertical/L2-5_finite_size_dilution.jl [--smoke] [--force] [--cases 1:5]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L2_5

const TEST_ID = "L2-5"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

const STUDY_L = (2.3, 4.6, 11.5, 23.0, 46.0)
const STUDY_H = (3.0, 10.0, 30.0)
const MESH_L_MAX = 11.5               # 32-element check up to L = 11.5 m (≈ 16× the cost of 16 elements at larger L)
const DEEP_L_MAX = 23.0               # H = 30: L = 46 m exceeds the 60 CPU-min budget; the deep-water exponent uses L ≤ 23 m

function cases(; smoke::Bool=false)
    smoke && return [L2_5_Params(Lp=4.6, H=10.0, elements_per_wavelength=8)]
    main = [L2_5_Params(Lp=L, H=H, elements_per_wavelength=16) for H in STUDY_H for L in STUDY_L
            if !(H == 30.0 && L > DEEP_L_MAX)]
    fine = [L2_5_Params(Lp=L, H=10.0, elements_per_wavelength=32) for L in STUDY_L if L <= MESH_L_MAX]
    return [main; fine]
end

record_name(p::L2_5_Params) = savename(p; accesses=VARIABLE, sigdigits=6) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

function simulate(p::L2_5_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L2_5_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L2_5_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

errors(p::L2_5_Params, d) = (Lp=p.Lp, H=p.H, elements_per_wavelength=p.elements_per_wavelength, N=d["N"],
                             omega_1=d["omega_1"], split=d["split"], sqrt_mu_mode=d["sqrt_mu_mode"],
                             sqrt_mu_B=d["sqrt_mu_B"], sqrt_mu_local=d["sqrt_mu_local"],
                             ratio_mode=d["split"] / d["sqrt_mu_mode"],
                             split_galerkin=d["split_galerkin"], ratio_galerkin=d["split"] / d["split_galerkin"],
                             modal_mass_ratio=d["modal_mass_ratio"], modal_mass_check=d["modal_mass_check"], ratio_B=d["split"] / d["sqrt_mu_B"],
                             proj=min(d["proj_minus"], d["proj_plus"]), ndofs=d["ndofs"], wall_time_s=d["wall_time_s"])

function loglog_slope(x, y)
    X, Y = log.(x), log.(y)
    return sum((X .- sum(X) / length(X)) .* (Y .- sum(Y) / length(Y))) / sum((X .- sum(X) / length(X)) .^ 2)
end

function checks(df::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    main = filter(r -> r.elements_per_wavelength == 16, df)
    for r in eachrow(main)
        tag = @sprintf("L=%g, H=%g", r.Lp, r.H)
        push!(rows, (name="hybrid pair identified (projection ≥ 0.2), $(tag)", kind="I", value=r.proj, tolerance=0.2,
                     passed=r.proj >= 0.2))
        add!("one-solve modal mass vs full added-mass matrix, $(tag)", "I", r.modal_mass_check, 1e-6)
        add!("splitting vs single-mode Galerkin (dispersive added mass), $(tag)", "A", abs(r.ratio_galerkin - 1), 1e-2)
    end
    # N ≥ 5: with N = 2 (L = 2.3 m) the resonators sit near the nodes of the free–free mode
    # (0.22L, 0.78L), so the splitting is set by the sampling of φ, not by the size cut-off
    # (√μ_mode, which contains the sampling, still matches the FE there).
    deep = sort(filter(r -> r.H == 30.0 && r.Lp <= DEEP_L_MAX && r.N >= 5, main), :Lp)
    nrow(deep) >= 3 && add!("deep-water exponent (H=30, N ≥ 5, L ≤ 23), |s + ½|", "A", abs(loglog_slope(deep.Lp, deep.split) + 0.5), 0.1)
    for L in STUDY_L
        a = filter(r -> r.Lp == L && r.H == 10.0, df)
        nrow(a) == 2 || continue
        s = sort(a, :elements_per_wavelength)
        add!(@sprintf("splitting, mesh 16 vs 32, L=%g, H=10", L), "D", abs(s.split[1] / s.split[2] - 1), 1e-2)
    end
    return isempty(rows) ? DataFrame(name=String[], kind=String[], value=Float64[], tolerance=Float64[], passed=Bool[]) :
           DataFrame(rows)
end

function plot_split(df::DataFrame, dir::AbstractString)
    nrow(df) == 0 && return
    main = filter(r -> r.elements_per_wavelength == 16, df)
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        fig = Figure(size=(620, 380))
        ax = Axis(fig[1, 1]; xscale=log10, yscale=log10, xlabel=L"L\ \mathrm{(m)}", ylabel=L"(\omega_+ - \omega_-)/\omega_1")
        for (i, H) in enumerate(STUDY_H)
            g = sort(filter(r -> r.H == H, main), :Lp)
            c = Makie.wong_colors()[i]
            scatter!(ax, g.Lp, g.split; color=c, markersize=9, label=@sprintf("FE, H = %g m", H))
            lines!(ax, g.Lp, g.sqrt_mu_mode; color=c, label=@sprintf("√μ_mode, H = %g m", H))
            lines!(ax, g.Lp, g.sqrt_mu_B; color=c, linestyle=:dash, label=@sprintf("√μ_B (paper), H = %g m", H))
        end
        Legend(fig[1, 2], ax; framevisible=false, labelsize=9)
        save(joinpath(dir, "splitting.pdf"), fig)
        save(joinpath(dir, "splitting.png"), fig; px_per_unit=300 / 96)
    end
end

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO,
              only::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cs = cases(; smoke=smoke)
    if only !== nothing
        for (i, p) in zip(only, cs[only])
            @info "case $(i)/$(length(cs))" p.Lp p.H p.elements_per_wavelength
            simulate(p; root=root, force=force)
        end
        return nothing
    end
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    ck = checks(df)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "checks.csv"), ck)
    plot_split(df, joinpath(root, "plots", TEST_ID))
    show(stdout, MIME"text/plain"(), select(df, :Lp, :H, :elements_per_wavelength, :N, :omega_1, :split, :sqrt_mu_mode,
                                            :sqrt_mu_B, :ratio_mode, :ratio_galerkin, :modal_mass_ratio, :proj, :wall_time_s); allrows=true)
    println()
    deep3 = sort(filter(r -> r.H == 3.0 && r.Lp >= 11.5 && r.elements_per_wavelength == 16, df), :Lp)
    nrow(deep3) >= 2 && @printf("shallow-water exponent (H=3, L ≥ 11.5): %.3f\n", loglog_slope(deep3.Lp, deep3.split))
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-70s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (errors=df, checks=ck)
end

function cases_arg(args)
    i = findfirst(==("--cases"), args)
    i === nothing && return nothing
    part(s) = occursin(':', s) ? range(parse.(Int, split(s, ":"))...) : [parse(Int, s)]
    return reduce(vcat, collect.(part.(split(args[i + 1], ","))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; smoke="--smoke" in ARGS, force="--force" in ARGS, only=cases_arg(ARGS))
end
