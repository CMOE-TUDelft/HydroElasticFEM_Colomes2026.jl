#=
L2-3 — Renormalised wet resonance of a resonator on a finite floating membrane.
Pre- and post-processing of the test defined in src/L2_3_wet_resonance.jl.

Study: set P2 (membrane L_m = 50 m on a 150 m channel, H = 25 m, ρ = 1025,
resonator ω_r = 0.5 rad/s, lossless), attachment x₁ ∈ {0 (edge), 12.5, 25
(centre)} m from the left edge; p = 3 with 70, 140, 280 elements per
wavelength at ω = 0.5, i.e. 20, 40, 80 elements on the membrane (element size
2.5, 1.25, 0.625 m; one mesh for all frequencies of a case). On a membrane
G_B(x, x) has a kink at the load, so an attachment inside an element converges
only at first order: this ladder puts all three positions on nodes at every level.
Profile (figure line): ω̃ alone at every mesh node of [0, 25] m of the finest mesh.

Acceptance (test matrix §L2.3), finest mesh unless stated:
  - type I: the fixed-point iteration of eq:eigWet converges (< 30 iterations);
  - type D: mesh error estimate of ω̃ (Green function) and of ω_wet (modal),
    |ω₁₄₀ − ω₂₈₀|/ω₂₈₀ ≤ 1e-4;
  - type D: modal ω_wet and Green-function ω̃ agree to 1e-3 (relative); the
    difference is reported also in units of the resonance half-width;
  - type D: forced-response peak and ω̃ agree to 1e-3;
  - reported: the complex wet frequency (full added mass) against the decay
    rate predicted from γ_r = Im G_B, and the previously reported
    0.4997 − 1.6e-5 i at x₁ = 25 m.

Outputs:
  data/sims/L2-3/      one DrWatson record per case (profile cases in profile/)
  data/exp_pro/L2-3/   errors.csv, checks.csv; profile.csv (kept apart: the verification tables
                       read errors.csv and checks.csv)
  plots/L2-3/          resonance.pdf/png (ω̃ along the membrane, and ω_wet, ω̃, forced peak at the study positions)

Run:
  julia --project -O1 scripts/L2-configB-vertical/L2-3_wet_resonance.jl [--smoke] [--force] [--profile 1:10]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_Colomes2026.L2_3
import HydroElasticFEM_Colomes2026.L2_3: green_resonance
import HydroElasticFEM_Colomes2026.L2_1: mesh_size

const TEST_ID = "L2-3"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

const STUDY_X = (0.0, 12.5, 25.0)
const LADDER = (70, 140, 280)

function cases(; smoke::Bool=false)
    smoke && return [L2_3_Params(x1=25.0, elements_per_wavelength=64)]
    return [L2_3_Params(x1=x, elements_per_wavelength=n) for x in STUDY_X for n in LADDER]
end

"Profile of ω̃: every node of [0, 25] m of the finest mesh (plate edges on nodes, see `L2_1.build`)."
function profile_cases(; smoke::Bool=false)
    smoke && return L2_3_Params[]
    p0 = L2_3_Params(elements_per_wavelength=last(LADDER))
    n = ceil(Int, p0.Lp / mesh_size(p0, p0.ω_mesh).he)
    he = p0.Lp / n
    return [L2_3_Params(p0; x1=k * he) for k in 0:floor(Int, p0.Lp / 2 / he)]
end

record_name(p::L2_3_Params) = savename(p; accesses=VARIABLE, sigdigits=4) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

function simulate(p::L2_3_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L2_3_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L2_3_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

"ω̃ and γ_r at one position; only the Green-function root, not the modal problem of `run_case`."
function simulate_profile(p::L2_3_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID, "profile")
    data, _ = produce_or_load(p, path; filename=q -> @sprintf("x1=%.6f_n=%d", q.x1, q.elements_per_wavelength),
                              tag=true, gitpath=REPO, storepatch=true, force=force) do p
        stats = @timed green_resonance(p)
        g = stats.value
        d = Dict{String,Any}("omega_green" => g.omega, "gamma_r" => g.gamma_r, "G11" => g.G, "green_solves" => g.solves)
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L2_3_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        return d
    end
    return data
end

function errors(p::L2_3_Params, d)
    ωw, ωg, ωf = d["omega_wet"], d["omega_green"], d["omega_forced"]
    ωc = d["omega_wet_complex"]
    return (x1=p.x1, elements_per_wavelength=p.elements_per_wavelength, ndofs=d["ndofs"], element_size=d["element_size"],
            omega_wet=ωw, omega_green=ωg, omega_forced=ωf, re_omega_complex=real(ωc), im_omega_complex=imag(ωc),
            gamma_r=d["gamma_r"], iterations=length(d["iterations"]), participation=d["participation"],
            diff_wet_green=abs(ωw / ωg - 1), diff_forced_green=abs(ωf / ωg - 1),
            diff_wet_green_halfwidths=abs(ωw - ωg) / abs(imag(ωc)), wall_time_s=d["wall_time_s"])
end

function checks(df::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    for g in groupby(sort(df, :elements_per_wavelength), :x1)
        tag = @sprintf("x₁=%g", g.x1[1])
        add!("fixed-point iterations (max over meshes), $(tag)", "I", maximum(g.iterations), 29)
        if nrow(g) >= 2
            add!("mesh error ω̃ (Green), $(tag)", "D", abs(g.omega_green[end - 1] / g.omega_green[end] - 1), 1e-4)
            add!("mesh error ω_wet (modal), $(tag)", "D", abs(g.omega_wet[end - 1] / g.omega_wet[end] - 1), 1e-4)
        end
        f = g[end, :]
        add!("modal ω_wet vs Green ω̃, finest, $(tag)", "D", f.diff_wet_green, 1e-3)
        add!("forced peak vs Green ω̃, finest, $(tag)", "D", f.diff_forced_green, 1e-3)
    end
    return DataFrame(rows)
end

function plot_resonance(df::DataFrame, pf::DataFrame, dir::AbstractString)
    nrow(df) == 0 && return
    fn = combine(groupby(df, :x1), g -> g[argmax(g.elements_per_wavelength), :])
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        fig = Figure(size=(560, 340))
        ax = Axis(fig[1, 1]; xlabel=L"x_1\ \mathrm{(m,\ from\ the\ membrane\ edge)}", ylabel=L"\omega\ \mathrm{(rad/s)}")
        if nrow(pf) > 0
            lines!(ax, pf.x1, pf.omega_green; color=:black, label="Green function, Re(1/α − G_B) = 0")
            scatter!(ax, fn.x1, fn.omega_green; color=:black, markersize=7)
        else
            scatterlines!(ax, fn.x1, fn.omega_green; color=:black, label="Green function, Re(1/α − G_B) = 0")
        end
        scatter!(ax, fn.x1, fn.omega_wet; color=:firebrick, marker=:utriangle, markersize=11, label="modal, eq. eigWet")
        scatter!(ax, fn.x1, fn.omega_forced; color=:royalblue, marker=:x, markersize=11, label="forced-response peak")
        hlines!(ax, [0.5]; color=:gray60, linestyle=:dot, label=L"\omega_r")
        axislegend(ax; position=:rb, framevisible=false, labelsize=10)
        save(joinpath(dir, "resonance.pdf"), fig)
        save(joinpath(dir, "resonance.png"), fig; px_per_unit=300 / 96)
    end
end

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO,
              only_profile::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    ps = profile_cases(; smoke=smoke)
    if only_profile !== nothing
        for (i, p) in zip(only_profile, ps[only_profile])
            @info "profile case $(i)/$(length(ps))" p.x1
            simulate_profile(p; root=root, force=force)
        end
        return nothing
    end
    cs = cases(; smoke=smoke)
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    ck = checks(df)
    pf = DataFrame([(x1=p.x1, omega_green=d["omega_green"], gamma_r=d["gamma_r"], green_solves=d["green_solves"], wall_time_s=d["wall_time_s"])
                    for (p, d) in ((p, simulate_profile(p; root=root, force=force)) for p in ps)])
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "checks.csv"), ck)
    CSV.write(joinpath(out, "profile.csv"), pf)
    plot_resonance(df, pf, joinpath(root, "plots", TEST_ID))
    show(stdout, MIME"text/plain"(), select(df, :x1, :elements_per_wavelength, :ndofs, :omega_wet, :omega_green,
                                            :omega_forced, :im_omega_complex, :gamma_r, :iterations, :diff_wet_green,
                                            :diff_forced_green); allrows=true)
    println()
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-70s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (errors=df, checks=ck)
end

function profile_arg(args)
    i = findfirst(==("--profile"), args)
    i === nothing && return nothing
    part(s) = occursin(':', s) ? range(parse.(Int, split(s, ":"))...) : [parse(Int, s)]
    return reduce(vcat, collect.(part.(split(args[i + 1], ","))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; smoke="--smoke" in ARGS, force="--force" in ARGS, only_profile=profile_arg(ARGS))
end
