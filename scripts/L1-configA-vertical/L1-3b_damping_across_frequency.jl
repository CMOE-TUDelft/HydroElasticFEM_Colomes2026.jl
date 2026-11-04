#=
L1-3b — Single resonator across frequency: optimal damping and its collapse at low frequency.
The physics, the FE problem and the observables of one case are those of L1-3
(src/L1_3_single_line_resonator.jl, `L1_3.run_case`); this script varies the
resonator frequency and the dashpot.

Relation to the paper: critical coupling γ_d = γ_r gives the optimal damping
ratio 2ζ_opt/(1 + 4ζ_opt²) = m_ρ,r ω_r²/𝒟₀′(k₀) (eq:zeta_opt), bounded by
m_ρ,r k₀²/2 (eq:zeta_bound): at ω_r = 1 the fluid makes up > 98 % of 𝒟₀′ and
ζ_opt = 4.3e-5 (`fig:lattice_design`(b)). At critical coupling the absorbed
fraction at resonance is ½, and 𝒜(ω̃) = 4x/(1 + x)², x = γ_d/γ_r, otherwise.

Study: P1 host, H = 10 m, m_ρ,r = 0.01 m², T_ρ ∈ {0, 9.8},
ω_r ∈ {1, 2, 5, 10} rad/s, c/c_crit ∈ {¼, ½, 1, 2, 4} (resonator on a node),
p = 3 with 10 and 20 elements per λ_p(ω_r).

Acceptance (test matrix §L1.3b), per T_ρ and ω_r, finest mesh:
  - type D: ω̃ (relative) and 𝒜(ω̃) (absolute) against the exact
    single-resonator solution with the same resonator to 1e-3, every c;
    |𝒜(ω̃) − ½| ≤ 1e-3 at c = c_crit;
  - type D: the maximum of the FE 𝒜(ω̃) over c, located by a parabola in
    log c through c/c_crit ∈ {½, 1, 2}, at c_crit within 1e-2 (relative);
    the corresponding ζ against ζ_opt of eq:zeta_opt is reported (c_crit is
    defined at the renormalised resonance, ζ_opt at ω_r, so they differ by the
    renormalisation shift, O(1e-3)).

Outputs:
  data/sims/L1-3b/     one DrWatson record per case
  data/exp_pro/L1-3b/  errors.csv, optimum.csv, checks.csv
  plots/L1-3b/         absorption.pdf/png (𝒜(ω̃) against c/c_crit, FE and exact),
                       zeta_opt.pdf/png (ζ at the FE maximum against ω_r, eq:zeta_opt and the bound)

Run:
  julia --project -O1 scripts/L1-configA-vertical/L1-3b_damping_across_frequency.jl [--smoke] [--force] [--cases 1:10]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L1_3
import HydroElasticFEM_xxx2026.L1_3
import HydroElasticFEM_xxx2026.L1_1
using HydroElasticFEM_xxx2026.References: k0, dD0, natural_frequency, resonator_from_frequency

const TEST_ID = "L1-3b"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ─── 1. Cases ───────────────────────────────────────────────────────────────

const STUDY_T = (0.0, 9.8)
const STUDY_W = (1.0, 2.0, 5.0, 10.0)
const STUDY_C = (0.25, 0.5, 1.0, 2.0, 4.0)
const STUDY_N = (10, 20)
const VISCOUS_SPONGE = 1.0      # loss factor of the viscous sponges at ω_r = 1 (`L1_1.viscous_sponges`)

"Record fields of L1-3b: the L1-3 variable fields plus the resonator frequency."
const VARIABLE_3b = (:ω_r, VARIABLE...)

function cases(; smoke::Bool=false)
    smoke && return [L1_3_Params(ω_r=2.0, c_factor=1.0, elements_per_wavelength=10, order=3)]
    # at ω_r = 1 the bending sponges are transparent (bending is 1e-5 of gravity): viscous sponges
    sv(ω) = ω <= 1.0 ? VISCOUS_SPONGE : 0.0
    return [L1_3_Params(T_ρ=T, ω_r=ω, c_factor=c, elements_per_wavelength=n, order=3, sponge_viscous=sv(ω))
            for T in STUDY_T for ω in STUDY_W for c in STUDY_C for n in STUDY_N]
end

# ─── 2. Simulation records (DrWatson) ───────────────────────────────────────

"Fixed-parameter hash without ω_r (a variable of this study)."
function fixed_hash_3b(p::L1_3_Params)
    fixed = L1_3.hashed_fields(p, VARIABLE_3b)
    return string(hash(fixed); base=16, pad=16)[1:8]
end

record_name(p::L1_3_Params) = savename(p; accesses=VARIABLE_3b, sigdigits=4) * "_fixed=" * fixed_hash_3b(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

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
    stored = data["params"]
    all(f -> get(stored, string(f), getfield(L1_3_Params(), f)) == getfield(p, f), fieldnames(L1_3_Params)) ||   # late fields: default
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── 3. Post-processing ─────────────────────────────────────────────────────

"ζ_opt of eq:zeta_opt (smaller root of 2ζ/(1 + 4ζ²) = m_ρ,r ω_r²/𝒟₀′(k₀)) and the bound of eq:zeta_bound."
function zeta_opt(p::L1_3_Params)
    h = L1_1.host(p)
    kγ = p.m_ρr * p.ω_r^2 / dD0(h, k0(h, p.ω_r), p.ω_r)
    return (opt=(1 - sqrt(1 - 4kγ^2)) / (4kγ), bound=p.m_ρr * k0(h, p.ω_r)^2 / 2)
end

function errors(p::L1_3_Params, d)
    r = resonator_from_frequency(p.m_ρr, p.ω_r)
    ζ_crit = d["c_crit"] / (2sqrt(r.k_ρ * r.m_ρ))
    z = zeta_opt(p)
    return (T_ρ=p.T_ρ, ω_r=p.ω_r, c_factor=p.c_factor, elements_per_wavelength=p.elements_per_wavelength,
            ndofs=d["ndofs"], omega_peak_h=d["omega_peak_h"], omega_peak_ref=d["omega_peak_ref"],
            err_omega=abs(d["omega_peak_h"] / d["omega_peak_ref"] - 1),
            err_omega_over_width=abs(d["omega_peak_h"] - d["omega_peak_ref"]) / abs(-(d["omega_half_ref"]...)),
            A_peak_h=d["A_peak_h"], A_peak_ref=d["A_peak_ref"], err_A=abs(d["A_peak_h"] - d["A_peak_ref"]),
            zeta=p.c_factor * ζ_crit, zeta_crit=ζ_crit, zeta_opt=z.opt, zeta_bound=z.bound, Q_rad=d["Q_rad"],
            lambda_p=d["lambda_p"], L=d["L"], fe_solves=d["fe_solves"], wall_time_s=d["wall_time_s"])
end

finest(df) = combine(groupby(df, [:T_ρ, :ω_r, :c_factor]), g -> g[argmax(g.elements_per_wavelength), :])

"Vertex of the parabola through (log xᵢ, yᵢ), i = 1:3 (x of the maximum)."
function parabola_vertex(x, y)
    t = log.(x)
    A = [t .^ 2 t ones(3)]
    c = A \ y
    return exp(-c[2] / (2c[1]))
end

"FE maximum of 𝒜(ω̃) over c per T_ρ and ω_r (finest mesh)."
function optimum(df::DataFrame)
    rows = NamedTuple[]
    for g in groupby(sort(finest(df), :c_factor), [:T_ρ, :ω_r])
        mid = filter(r -> r.c_factor in (0.5, 1.0, 2.0), g)
        nrow(mid) == 3 || continue
        x = parabola_vertex(mid.c_factor, mid.A_peak_h)
        f = g[1, :]
        push!(rows, (T_ρ=f.T_ρ, ω_r=f.ω_r, c_star_over_crit=x, zeta_star=x * f.zeta_crit, zeta_crit=f.zeta_crit,
                     zeta_opt=f.zeta_opt, zeta_bound=f.zeta_bound,
                     A_at_crit=only(mid[mid.c_factor .== 1.0, :A_peak_h])))
    end
    return DataFrame(rows)
end

function checks(df::DataFrame, opt::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    for r in eachrow(sort(finest(df), [:T_ρ, :ω_r, :c_factor]))
        tag = @sprintf("T=%g, ωr=%g, c/c_crit=%g", r.T_ρ, r.ω_r, r.c_factor)
        add!("ω̃, finest mesh, $(tag)", "D", r.err_omega, 1e-3)
        add!("𝒜(ω̃), finest mesh, $(tag)", "D", r.err_A, 1e-3)
    end
    for r in eachrow(opt)
        tag = @sprintf("T=%g, ωr=%g", r.T_ρ, r.ω_r)
        add!("|𝒜(ω̃) − ½| at c_crit, $(tag)", "D", abs(r.A_at_crit - 0.5), 1e-3)
        add!("FE maximum of 𝒜 at c_crit (relative), $(tag)", "D", abs(r.c_star_over_crit - 1), 1e-2)
    end
    return DataFrame(rows)
end

# ─── 4. Figures ─────────────────────────────────────────────────────────────

function plot_absorption(df::DataFrame, dir::AbstractString)
    nrow(df) == 0 && return
    fn = finest(df)
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        Ts = sort(unique(fn.T_ρ))
        fig = Figure(size=(360 * length(Ts) + 150, 340))
        colors = Dict(1.0 => :black, 2.0 => :royalblue, 5.0 => :forestgreen, 10.0 => :firebrick)
        local ax
        for (j, T) in enumerate(Ts)
            Tstr = @sprintf("%g", T)
            ax = Axis(fig[1, j]; xscale=log10, xlabel=L"c/c_\mathrm{crit}", ylabel=j == 1 ? L"\mathcal{A}(\tilde\omega_r)" : "",
                      title=L"T_\rho = %$Tstr", xticks=([0.25, 1, 4], ["1/4", "1", "4"]))
            x = 10 .^ range(log10(0.2), log10(5); length=200)
            lines!(ax, x, 4x ./ (1 .+ x) .^ 2; color=:gray60, label=L"4x/(1+x)^2")
            for ω in sort(unique(fn.ω_r))
                g = sort(filter(r -> r.T_ρ == T && r.ω_r == ω, fn), :c_factor)
                scatter!(ax, g.c_factor, g.A_peak_h; color=colors[ω], label=@sprintf("FE, ωr = %g", ω))
            end
        end
        Legend(fig[1, length(Ts) + 1], ax; framevisible=false, labelsize=10, unique=true)
        save(joinpath(dir, "absorption.pdf"), fig)
        save(joinpath(dir, "absorption.png"), fig; px_per_unit=300 / 96)
    end
end

function plot_zeta(opt::DataFrame, p0::L1_3_Params, dir::AbstractString)
    nrow(opt) == 0 && return
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        fig = Figure(size=(520, 360))
        ax = Axis(fig[1, 1]; xscale=log10, yscale=log10, xlabel=L"\omega_r\ \mathrm{(rad/s)}", ylabel=L"\zeta")
        markers = Dict(0.0 => :circle, 9.8 => :rect)
        for T in sort(unique(opt.T_ρ))
            ωs = 10 .^ range(log10(0.5), log10(12); length=100)
            z = [zeta_opt(L1_3_Params(p0; T_ρ=T, ω_r=ω)) for ω in ωs]
            lines!(ax, ωs, getfield.(z, :opt); color=:black, linestyle=T == 0 ? :solid : :dash,
                   label=@sprintf("eq. zeta_opt, Tρ = %g", T))
            T == 0 && lines!(ax, ωs, getfield.(z, :bound); color=:gray60, linestyle=:dot, label="bound (eq. zeta_bound)")
            g = filter(r -> r.T_ρ == T, opt)
            scatter!(ax, g.ω_r, g.zeta_star; color=:firebrick, marker=markers[T], label=@sprintf("FE maximum, Tρ = %g", T))
        end
        axislegend(ax; position=:lt, framevisible=false, labelsize=10)
        save(joinpath(dir, "zeta_opt.pdf"), fig)
        save(joinpath(dir, "zeta_opt.png"), fig; px_per_unit=300 / 96)
    end
end

# ─── 5. Driver ──────────────────────────────────────────────────────────────

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO,
              only::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cs = cases(; smoke=smoke)
    if only !== nothing
        for (i, p) in zip(only, cs[only])
            @info "case $(i)/$(length(cs))" p.T_ρ p.ω_r p.c_factor p.elements_per_wavelength
            simulate(p; root=root, force=force)
        end
        return nothing
    end
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    opt = optimum(df)
    ck = checks(df, opt)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "optimum.csv"), opt)
    CSV.write(joinpath(out, "checks.csv"), ck)
    figdir = joinpath(root, "plots", TEST_ID)
    plot_absorption(df, figdir)
    plot_zeta(opt, first(cs), figdir)

    show(stdout, MIME"text/plain"(), select(finest(df), :T_ρ, :ω_r, :c_factor, :ndofs, :err_omega, :err_omega_over_width, :A_peak_h,
                                            :err_A, :zeta, :zeta_opt, :fe_solves); allrows=true)
    println()
    show(stdout, MIME"text/plain"(), opt; allrows=true)
    println()
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
