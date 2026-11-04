#=
L2-4 — Absorption bound of a single resonator on a finite structure.
Pre- and post-processing of the test defined in src/L2_4_asymmetric_absorption.jl.

Study: P1 plate (T_ρ = 0), free edges, H = 10 m, lossless resonator
m_ρ,r = 0.01 m², ω_r = 10 rad/s; L_p ∈ {5, 20} m; x₁ ∈ {L_p/2, L_p/4, L_p/8,
0.25 m} and the mirror positions L_p − x₁ (incidence from the right); p = 3 with
8 elements per wavelength, and 16 for L_p = 5 and for x₁ ∈ {L_p/2, 0.25} at
L_p = 20 (mesh check). Damping ratio swept over 41 values in [1e-3, 1e-1].
Profile (figure line): the same case on 8 elements per wavelength every 0.05 m
(L_p = 5) and 0.1 m (L_p = 20) up to the centre; beyond it the bound follows from
bound(x₁) + bound(L_p − x₁) = 1. The radiation pattern repeats about every
1.14 m at ω ≈ 10, so the study positions alone alias it.

Acceptance (test matrix §L2.4):
  - type D: radiated power ½ω Im G_B(x₁, x₁) against the open-water flux of the
    fitted a^±, to 1e-3;
  - type D: max over ζ of 𝒜 and 𝒜 at critical coupling against the bound
    |a⁻|²/(|a⁻|² + |a⁺|²), to 1e-3 (relative);
  - type D: bound = ½ at the plate centre, and bound(x₁) + bound(L_p − x₁) = 1,
    to 1e-3;
  - type D: monolithic 𝒜 at critical coupling (dashpot power) against the
    Woodbury value, to 1e-5 (the two use separately fitted incident amplitudes,
    so the comparison is limited by the open-water fit, not by round-off);
  - type D: bound on the two meshes, to 1e-3, where both exist.

Outputs:
  data/sims/L2-4/      one DrWatson record per case (profile cases in profile/)
  data/exp_pro/L2-4/   errors.csv, absorption.csv (𝒜 against ζ), checks.csv; profile.csv (bound along the
                       plate), profile_errors.csv, profile_checks.csv (kept apart: the verification
                       tables read errors.csv and checks.csv)
  plots/L2-4/          bound.pdf/png (bound and max 𝒜 against x₁/L_p), absorption.pdf/png

Run:
  julia --project -O1 scripts/L2-configB-vertical/L2-4_asymmetric_absorption.jl [--smoke] [--force] [--cases 1:4] [--profile 1:60]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L2_4

const TEST_ID = "L2-4"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const PROFILE_DIR = "profile"

positions(L) = sort(unique([L / 2, L / 4, L / 8, 0.25, L - L / 4, L - L / 8, L - 0.25]))

function cases(; smoke::Bool=false)
    smoke && return [L2_4_Params(Lp=5.0, x1=1.25, elements_per_wavelength=6)]
    main = [L2_4_Params(Lp=L, x1=x, elements_per_wavelength=8) for L in (5.0, 20.0) for x in positions(L)]
    fine = [L2_4_Params(Lp=5.0, x1=x, elements_per_wavelength=16) for x in positions(5.0)]
    fine20 = [L2_4_Params(Lp=20.0, x1=x, elements_per_wavelength=16) for x in (10.0, 0.25)]
    return [main; fine; fine20]
end

"Profile positions up to the centre, without the study positions (their records are reused)."
function profile_positions(L)
    dx = L == 5.0 ? 0.05 : 0.1
    xs = [round(i * dx; digits=6) for i in 1:round(Int, L / 2 / dx)]
    return filter(x -> !any(y -> isapprox(x, y; atol=1e-9), positions(L)), xs)
end

profile_cases(; smoke::Bool=false) =
    smoke ? L2_4_Params[] : [L2_4_Params(Lp=L, x1=x, elements_per_wavelength=8) for L in (5.0, 20.0) for x in profile_positions(L)]

record_name(p::L2_4_Params) = savename(p; accesses=VARIABLE, sigdigits=4) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

function simulate(p::L2_4_Params; root::AbstractString=REPO, force::Bool=false, sub::AbstractString="")
    path = joinpath(root, "data", "sims", TEST_ID, sub)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L2_4_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L2_4_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

function errors(p::L2_4_Params, d)
    return (Lp=p.Lp, x1=p.x1, x1_rel=p.x1 / p.Lp, elements_per_wavelength=p.elements_per_wavelength, ndofs=d["ndofs"],
            omega_tilde=d["omega_tilde"], gamma_r=imag(d["G11"]), bound=d["bound"],
            max_absorbed=maximum(d["absorbed"]), absorbed_crit=d["absorbed_crit"], absorbed_mono=d["absorbed_mono"],
            zeta_crit=d["zeta_crit"], power_err=abs(d["power_G"] / d["power_flux"] - 1),
            radiation_in=d["radiation_in"] / (abs(d["a_left"]) + abs(d["a_right"])),
            fit_residual=d["fit_residual"], wall_time_s=d["wall_time_s"])
end

function checks(df::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    for r in eachrow(df)
        tag = @sprintf("Lp=%g, x1=%g, n=%d", r.Lp, r.x1, r.elements_per_wavelength)
        add!("radiated power ½ω Im G vs open-water flux, $(tag)", "D", r.power_err, 1e-3)
        add!("max_ζ 𝒜 vs bound, $(tag)", "D", abs(r.max_absorbed / r.bound - 1), 1e-3)
        add!("𝒜 at critical coupling vs bound, $(tag)", "D", abs(r.absorbed_crit / r.bound - 1), 1e-3)
        add!("monolithic 𝒜 vs Woodbury at critical coupling, $(tag)", "D", abs(r.absorbed_mono / r.absorbed_crit - 1), 1e-5)
        r.x1 == r.Lp / 2 && add!("bound = ½ at the centre, $(tag)", "D", abs(r.bound - 0.5), 1e-3)
    end
    for g in groupby(df, [:Lp, :elements_per_wavelength])
        for r in eachrow(g)
            r.x1 < r.Lp / 2 || continue
            m = findfirst(x -> isapprox(x, r.Lp - r.x1), g.x1)
            m === nothing && continue
            add!(@sprintf("bound(x1) + bound(Lp − x1) = 1, Lp=%g, x1=%g, n=%d", r.Lp, r.x1, r.elements_per_wavelength),
                 "D", abs(r.bound + g.bound[m] - 1), 1e-3)
        end
    end
    for g in groupby(df, [:Lp, :x1])
        nrow(g) >= 2 || continue
        s = sort(g, :elements_per_wavelength)
        add!(@sprintf("bound, mesh 8 vs 16, Lp=%g, x1=%g", s.Lp[1], s.x1[1]), "D", abs(s.bound[1] - s.bound[end]), 1e-3)
    end
    return DataFrame(rows)
end

function absorption_table(cs, records)
    rows = NamedTuple[]
    for (p, d) in zip(cs, records)
        p.elements_per_wavelength == 8 || continue
        for (ζ, a) in zip(d["zetas"], d["absorbed"])
            push!(rows, (Lp=p.Lp, x1=p.x1, zeta=ζ, absorbed=a, bound=d["bound"], zeta_crit=d["zeta_crit"]))
        end
    end
    return DataFrame(rows)
end

"""
Bound along the plate on 8 elements per wavelength: study and profile positions,
and their mirror images through bound(x₁) + bound(L_p − x₁) = 1 where not computed.
"""
function bound_profile(df::DataFrame, pe::DataFrame)
    rows = NamedTuple[]
    cols = [:Lp, :x1, :bound, :omega_tilde, :gamma_r]
    g = vcat(select(filter(r -> r.elements_per_wavelength == 8, df), cols), select(pe, cols))
    for r in eachrow(g)
        push!(rows, (Lp=r.Lp, x1=r.x1, x1_rel=r.x1 / r.Lp, bound=r.bound, omega_tilde=r.omega_tilde, gamma_r=r.gamma_r, mirrored=false))
        xm = r.Lp - r.x1
        any(s -> s.Lp == r.Lp && isapprox(s.x1, xm; atol=1e-9), eachrow(g)) && continue
        push!(rows, (Lp=r.Lp, x1=xm, x1_rel=xm / r.Lp, bound=1 - r.bound, omega_tilde=r.omega_tilde, gamma_r=r.gamma_r, mirrored=true))
    end
    return sort(DataFrame(rows), [:Lp, :x1])
end

function plots(df::DataFrame, ab::DataFrame, bp::DataFrame, dir::AbstractString)
    nrow(df) == 0 && return
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        fig = Figure(size=(760, 310))
        for (j, (L, c)) in enumerate(((5.0, :royalblue), (20.0, :firebrick)))
            Lstr = @sprintf("%g", L)
            ax = Axis(fig[1, j]; xlabel=L"x_1\ \mathrm{(m)}", ylabel=j == 1 ? L"\mathcal{A}" : "", title=L"L_p = %$Lstr\ \mathrm{m}",
                      limits=(0, L, 0, 1.05))
            gb = filter(r -> r.Lp == L, bp)
            g = sort(filter(r -> r.Lp == L && r.elements_per_wavelength == 8, df), :x1)
            lines!(ax, gb.x1, gb.bound; color=c)
            scatter!(ax, g.x1, g.max_absorbed; color=:black, marker=:circle, markersize=7)
            hlines!(ax, [0.5]; color=:gray60, linestyle=:dot)
        end
        Legend(fig[2, 1:2], [LineElement(color=:gray30), MarkerElement(color=:black, marker=:circle, markersize=7)],
               ["bound |a⁻|²/(|a⁻|²+|a⁺|²)", "max 𝒜 over ζ (FE)"]; orientation=:horizontal, framevisible=false, labelsize=10)
        save(joinpath(dir, "bound.pdf"), fig)
        save(joinpath(dir, "bound.png"), fig; px_per_unit=300 / 96)
        nrow(ab) == 0 && return
        fig = Figure(size=(560, 340))
        ax = Axis(fig[1, 1]; xscale=log10, xlabel=L"\zeta", ylabel=L"\mathcal{A}(\tilde\omega_r)")
        g20 = filter(r -> r.Lp == 20.0, ab)
        for (i, x) in enumerate(sort(unique(g20.x1)))
            g = sort(filter(r -> r.x1 == x, g20), :zeta)
            c = Makie.wong_colors()[mod1(i, 7)]
            lines!(ax, g.zeta, g.absorbed; color=c, label=@sprintf("x₁ = %g m", x))
            vlines!(ax, [g.zeta_crit[1]]; color=c, linestyle=:dot, linewidth=1)
        end
        axislegend(ax; position=:lt, framevisible=false, labelsize=9)
        save(joinpath(dir, "absorption.pdf"), fig)
        save(joinpath(dir, "absorption.png"), fig; px_per_unit=300 / 96)
    end
end

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO,
              only::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              only_profile::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cs = cases(; smoke=smoke)
    ps = profile_cases(; smoke=smoke)
    if only_profile !== nothing
        for (i, p) in zip(only_profile, ps[only_profile])
            @info "profile case $(i)/$(length(ps))" p.Lp p.x1
            simulate(p; root=root, force=force, sub=PROFILE_DIR)
        end
        return nothing
    end
    if only !== nothing
        for (i, p) in zip(only, cs[only])
            @info "case $(i)/$(length(cs))" p.Lp p.x1 p.elements_per_wavelength
            simulate(p; root=root, force=force)
        end
        return nothing
    end
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    ab = absorption_table(cs, records)
    ck = checks(df)
    precords = [simulate(p; root=root, force=force, sub=PROFILE_DIR) for p in ps]
    pe = isempty(ps) ? filter(r -> false, df) : DataFrame([errors(p, d) for (p, d) in zip(ps, precords)])
    pck = nrow(pe) == 0 ? DataFrame() : checks(pe)
    bp = bound_profile(df, pe)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "absorption.csv"), ab)
    CSV.write(joinpath(out, "checks.csv"), ck)
    CSV.write(joinpath(out, "profile.csv"), bp)
    CSV.write(joinpath(out, "profile_errors.csv"), pe)
    CSV.write(joinpath(out, "profile_checks.csv"), pck)
    plots(df, ab, bp, joinpath(root, "plots", TEST_ID))
    show(stdout, MIME"text/plain"(), select(df, :Lp, :x1, :elements_per_wavelength, :omega_tilde, :bound, :max_absorbed,
                                            :absorbed_crit, :absorbed_mono, :power_err, :radiation_in, :wall_time_s); allrows=true)
    println()
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    nrow(pck) > 0 && @printf("%d/%d profile checks pass\n", count(pck.passed), nrow(pck))
    for c in eachrow(filter(r -> !r.passed, nrow(pck) > 0 ? [ck; pck] : ck))
        @printf("FAIL  %-75s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (errors=df, checks=ck)
end

function cases_arg(args, flag="--cases")
    i = findfirst(==(flag), args)
    i === nothing && return nothing
    part(s) = occursin(':', s) ? range(parse.(Int, split(s, ":"))...) : [parse(Int, s)]
    return reduce(vcat, collect.(part.(split(args[i + 1], ","))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; smoke="--smoke" in ARGS, force="--force" in ARGS, only=cases_arg(ARGS), only_profile=cases_arg(ARGS, "--profile"))
end
