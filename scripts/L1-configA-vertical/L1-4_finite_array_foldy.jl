#=
L1-4 — Finite periodic array: Foldy, energy balance, decay in the stop band.
Pre- and post-processing of the test defined in src/L1_4_finite_array_foldy.jl
(physics, relation to the paper and observables are documented there).

Study: P1 host, H = 10 m, m_ρ,r = 0.01 m², ω_r = 10 rad/s, a = 1 m.
  - curves (p = 3, 4 elements per cell ≈ 9 per λ_p at ω = 10): T_ρ ∈ {0, 9.8},
    N ∈ {1, 2, 5, 10, 20, 40}, ζ ∈ {0, 0.05}; and ζ = 0.01 for N ∈ {10, 40},
    T_ρ = 0; 41 frequencies in [8, 12] and 61 in [9.55, 10.05];
  - convergence (5 frequencies): N = 10, T_ρ ∈ {0, 9.8}, ζ ∈ {0, 0.05}, p = 3
    with 4, 8, 16 elements per cell.

Acceptance (test matrix §L1.4):
  - type D, curves: max over ω of |R_h − R|, |T_h − T| and |𝒜_h − 𝒜| against
    the Foldy solution ≤ 1e-3 (𝒜 of Foldy is the dashpot power, which
    satisfies the energy identity exactly, so 𝒜_h = 1 − |R_h|² − |T_h|² against
    it checks the energy balance of the FE R, T);
  - type D, convergence: on the finest mesh max |T_h − T| ≤ 1e-4, and the
    error decreases with the mesh; FE power balance of the part driven by the
    propagating waves (net incoming flux against dashpot power, relative to the
    incoming flux, src/L1_4 observable 2) ≤ 1e-3;
  - type I: Foldy energy identity |R|² + |T|² + 𝒜 − 1 ≤ 1e-10;
  - type D (lossless, T_ρ = 0 and 9.8): inside the stop band, the FE decay
    rate (ln|T₁₀| − ln|T₂₀|)/(10a) against the same rate of the Foldy solution,
    to 0.01/a, wherever |T₂₀| ≥ 1e-4.
  Reported, not checked: the Bloch attenuations of the lattice sums
  (`bloch_decay_1d`). |T_N| is a sum over the decaying Bloch modes; besides the
  continuation of the propagating branch (Re q = π/a or 0), the continuation
  of the slowest fluid evanescent mode (Im q ≈ κ₁ = 0.158/m at H = 10 m) is
  excited, and it dominates long arrays wherever the propagating attenuation
  exceeds it, so the rate between two finite N is not a single Im q.

Outputs:
  data/sims/L1-4/      one DrWatson record per case
  data/exp_pro/L1-4/   errors.csv, curves.csv, decay.csv, checks.csv
  plots/L1-4/          transmission.pdf/png (|T_N| FE vs Foldy), decay.pdf/png

Run:
  julia --project -O1 scripts/L1-configA-vertical/L1-4_finite_array_foldy.jl [--smoke] [--force] [--cases 1:10]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_Colomes2026.L1_4
import HydroElasticFEM_Colomes2026.L1_1
using HydroElasticFEM_Colomes2026.References: bloch_decay_1d, resonator_from_frequency, omega_X_1d, omega_gamma_1d, kappa1, foldy_1d

const TEST_ID = "L1-4"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ─── 1. Cases ───────────────────────────────────────────────────────────────

const STUDY_T = (0.0, 9.8)
const STUDY_N = (1, 2, 5, 10, 20, 40)

function cases(; smoke::Bool=false)
    smoke && return [L1_4_Params(N=3, ζ=0.05, elements_per_cell=8, sweep=:check)]
    curves = [L1_4_Params(T_ρ=T, N=N, ζ=ζ) for T in STUDY_T for ζ in (0.0, 0.05) for N in STUDY_N]
    extra = [L1_4_Params(N=N, ζ=0.01) for N in (10, 40)]
    conv = [L1_4_Params(T_ρ=T, N=10, ζ=ζ, elements_per_cell=n, sweep=:check)
            for T in STUDY_T for ζ in (0.0, 0.05) for n in (4, 8, 16)]
    return [conv; curves; extra]
end

# ─── 2. Simulation records (DrWatson) ───────────────────────────────────────

record_name(p::L1_4_Params) = savename(p; accesses=VARIABLE, sigdigits=4) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

function simulate(p::L1_4_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L1_4_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L1_4_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── 3. Post-processing ─────────────────────────────────────────────────────

function errors(p::L1_4_Params, d)
    return (T_ρ=p.T_ρ, N=p.N, ζ=p.ζ, elements_per_cell=p.elements_per_cell, sweep=String(p.sweep), ndofs=d["ndofs"],
            err_R=maximum(abs.(d["R_h"] .- d["R_ref"])), err_T=maximum(abs.(d["T_h"] .- d["T_ref"])),
            err_A=maximum(abs.(d["A_h"] .- d["A_ref"])),
            power_residual=haskey(d, "power_residual_corrected_h") ? maximum(d["power_residual_corrected_h"]) : NaN,
            energy_residual_ref=d["energy_residual_ref"], fit_residual=d["fit_residual"],
            dispersion_error=d["dispersion_error"], n_omega=length(d["omega"]), wall_time_s=d["wall_time_s"])
end

function curves(cs, records)
    rows = NamedTuple[]
    for (p, d) in zip(cs, records)
        p.sweep == :curve || continue
        for i in eachindex(d["omega"])
            push!(rows, (T_ρ=p.T_ρ, N=p.N, ζ=p.ζ, omega=d["omega"][i],
                         abs_R_h=abs(d["R_h"][i]), abs_T_h=abs(d["T_h"][i]), A_h=d["A_h"][i],
                         abs_R_ref=abs(d["R_ref"][i]), abs_T_ref=abs(d["T_ref"][i]), A_ref=d["A_ref"][i]))
        end
    end
    return DataFrame(rows)
end

"Decay rate between N = 10 and N = 20 (lossless) against Im q of the lattice sums, at every curve frequency."
function decay(cv::DataFrame, p0::L1_4_Params)
    rows = NamedTuple[]
    nrow(cv) == 0 && return DataFrame()
    for T in STUDY_T
        c10 =sort(filter(r -> r.T_ρ == T && r.ζ == 0 && r.N == 10, cv), :omega)
        c20 = sort(filter(r -> r.T_ρ == T && r.ζ == 0 && r.N == 20, cv), :omega)
        (nrow(c10) == 0 || nrow(c20) != nrow(c10)) && continue
        p = L1_4_Params(p0; T_ρ=T)
        h, r = L1_1.host(p), resonator_from_frequency(p.m_ρr, p.ω_r)
        for i in 1:nrow(c10)
            ω = c10.omega[i]
            att = bloch_decay_1d(h, r, p.a, ω)
            zb = filter(<(0), att)                       # Re q = π/a
            push!(rows, (T_ρ=T, omega=ω, abs_T20_h=c20.abs_T_h[i], abs_T20_ref=c20.abs_T_ref[i],
                         rate_h=log(c10.abs_T_h[i] / c20.abs_T_h[i]) / (10p.a),
                         rate_foldy=log(c10.abs_T_ref[i] / c20.abs_T_ref[i]) / (10p.a),
                         Im_q_min=isempty(att) ? NaN : abs(att[1]) / p.a,
                         Im_q_zone_boundary=isempty(zb) ? NaN : abs(zb[1]) / p.a))
        end
    end
    return DataFrame(rows)
end

function checks(df::DataFrame, dc::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    for r in eachrow(filter(r -> r.sweep == "curve", df))
        tag = @sprintf("T=%g, N=%d, ζ=%g", r.T_ρ, r.N, r.ζ)
        add!("max |R_h − R|, $(tag)", "D", r.err_R, 1e-3)
        add!("max |T_h − T|, $(tag)", "D", r.err_T, 1e-3)
        add!("max |𝒜_h − 𝒜|, $(tag)", "D", r.err_A, 1e-3)
        add!("Foldy energy identity, $(tag)", "I", r.energy_residual_ref, 1e-10)
    end
    for g in groupby(sort(filter(r -> r.sweep == "check", df), :elements_per_cell), [:T_ρ, :ζ])
        tag = @sprintf("T=%g, ζ=%g, N=10", g.T_ρ[1], g.ζ[1])
        add!("max |T_h − T|, finest mesh, $(tag)", "D", g.err_T[end], 1e-4)
        add!("FE power balance (propagating part), finest mesh, $(tag)", "D", g.power_residual[end], 1e-3)
        dec = all(diff(g.err_T) .< 0)
        push!(rows, (name="error decreases with the mesh, $(tag)", kind="D", value=Float64(!dec), tolerance=0.0, passed=dec))
    end
    for T in STUDY_T
        g = filter(r -> r.T_ρ == T && isfinite(r.Im_q_min) && r.abs_T20_ref >= 1e-4, dc)
        nrow(g) == 0 && continue
        add!(@sprintf("decay rate FE vs Foldy in the stop band (max abs., 1/m), T=%g, %d frequencies", T, nrow(g)), "D",
             maximum(abs.(g.rate_h .- g.rate_foldy)), 0.01)
    end
    return DataFrame(rows)
end

# ─── 4. Figures ─────────────────────────────────────────────────────────────

function plot_transmission(cv::DataFrame, dir::AbstractString)
    nrow(cv) == 0 && return
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    colors = Dict(1 => :gray50, 2 => :purple, 5 => :royalblue, 10 => :forestgreen, 20 => :orange, 40 => :firebrick)
    mkpath(dir)
    with_theme(theme) do
        Ts = sort(unique(cv.T_ρ))
        ζs = sort(unique(filter(r -> r.N == 40 || r.N == 1, cv).ζ))
        ζs = filter(z -> z in (0.0, 0.05), ζs)
        fig = Figure(size=(380 * length(Ts) + 140, 300 * length(ζs)))
        local ax
        for (i, ζ) in enumerate(ζs), (j, T) in enumerate(Ts)
            Tstr, ζstr = @sprintf("%g", T), @sprintf("%g", ζ)
            ax = Axis(fig[i, j]; yscale=log10, xlabel=i == length(ζs) ? L"\omega\ \mathrm{(rad/s)}" : "",
                      ylabel=j == 1 ? L"|T_N|" : "", title=L"T_\rho = %$Tstr,\ \zeta = %$ζstr", limits=(nothing, (1e-7, 2)))
            for N in STUDY_N
                g = sort(filter(r -> r.T_ρ == T && r.ζ == ζ && r.N == N, cv), :omega)
                nrow(g) == 0 && continue
                lines!(ax, g.omega, max.(g.abs_T_ref, 1e-12); color=colors[N], label="N = $(N)")
                scatter!(ax, g.omega[1:3:end], max.(g.abs_T_h[1:3:end], 1e-12); color=colors[N], markersize=4)
            end
        end
        Legend(fig[:, length(Ts) + 1], ax; framevisible=false, labelsize=10)
        save(joinpath(dir, "transmission.pdf"), fig)
        save(joinpath(dir, "transmission.png"), fig; px_per_unit=300 / 96)
    end
end

"""
    decay_dense(p, ωs) -> (rate_foldy, Im_q_zone_boundary)

Foldy decay rate ln(|T₁₀|/|T₂₀|)/(10a) of the lossless array and Im q of the
zone-boundary Bloch mode at the frequencies `ωs` (|T| does not depend on where
the array is centred, so the resonators sit at (n − 1)a).
"""
function decay_dense(p::L1_4_Params, ωs)
    h, r = L1_1.host(p), resonator_from_frequency(p.m_ρr, p.ω_r)
    absT(N, ω) = abs(foldy_1d(h, ω, [(n - 1) * p.a for n in 1:N], fill(r, N)).T)
    rate = [log(absT(10, ω) / absT(20, ω)) / (10p.a) for ω in ωs]
    zb = map(ωs) do ω
        att = filter(<(0), bloch_decay_1d(h, r, p.a, ω))
        isempty(att) ? NaN : abs(att[1]) / p.a
    end
    return rate, zb
end

"""
Decay rate between N = 10 and 20 inside the stop band [ω_X, ω_Γ] of each T_ρ:
Foldy (line), FE (markers), Im q of the zone-boundary Bloch mode (dashed) and
the decay rate κ₁ of the slowest fluid evanescent mode (dotted). The continuous
curves are evaluated on `n_dense` frequencies, the FE only at the computed ones.
"""
function plot_decay(dc::DataFrame, p0::L1_4_Params, dir::AbstractString; n_dense::Int=401)
    nrow(dc) == 0 && return
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        fig = Figure(size=(380 * length(STUDY_T) + 220, 340))
        local ax
        for (j, T) in enumerate(STUDY_T)
            p = L1_4_Params(p0; T_ρ=T)
            h, r = L1_1.host(p), resonator_from_frequency(p.m_ρr, p.ω_r)
            ωX, ωΓ = omega_X_1d(h, r, p.a; ωmin=9.0, npts=4001), omega_gamma_1d(h, r, p.a)
            g = sort(filter(row -> row.T_ρ == T && ωX <= row.omega <= ωΓ, dc), :omega)
            nrow(g) == 0 && continue
            Tstr = @sprintf("%g", T)
            ax = Axis(fig[1, j]; xlabel=L"\omega\ \mathrm{(rad/s)}", ylabel=j == 1 ? L"\mathrm{decay\ rate}\ (1/\mathrm{m})" : "",
                      title=L"T_\rho = %$Tstr", limits=((ωX, ωΓ), (0, 1.0)))
            ωd = range(ωX, ωΓ; length=n_dense)
            rate_d, zb_d = decay_dense(p, ωd)
            lines!(ax, ωd, rate_d; color=:black, label=L"\mathrm{Foldy},\ \ln(|T_{10}|/|T_{20}|)/10a")
            scatter!(ax, g.omega, g.rate_h; color=:firebrick, markersize=6, label="FE")
            lines!(ax, ωd, zb_d; color=:royalblue, linestyle=:dash, label=L"\mathrm{Im}\,q,\ \mathrm{Re}\,q = \pi/a")
            hlines!(ax, [kappa1(h, p.ω_r)]; color=:gray50, linestyle=:dot, label=L"\kappa_1\ \mathrm{(fluid\ evanescent)}")
        end
        Legend(fig[1, length(STUDY_T) + 1], ax; framevisible=false, labelsize=10)
        save(joinpath(dir, "decay.pdf"), fig)
        save(joinpath(dir, "decay.png"), fig; px_per_unit=300 / 96)
    end
end

# ─── 5. Driver ──────────────────────────────────────────────────────────────

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO,
              only::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cs = cases(; smoke=smoke)
    if only !== nothing
        for (i, p) in zip(only, cs[only])
            @info "case $(i)/$(length(cs))" p.T_ρ p.N p.ζ p.elements_per_cell p.sweep
            simulate(p; root=root, force=force)
        end
        return nothing
    end
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    cv = curves(cs, records)
    dc = decay(cv, first(cs))
    ck = checks(df, dc)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "curves.csv"), cv)
    CSV.write(joinpath(out, "decay.csv"), dc)
    CSV.write(joinpath(out, "checks.csv"), ck)
    figdir = joinpath(root, "plots", TEST_ID)
    plot_transmission(cv, figdir)
    plot_decay(dc, first(cs), figdir)

    show(stdout, MIME"text/plain"(), select(df, :T_ρ, :N, :ζ, :elements_per_cell, :sweep, :ndofs, :err_R, :err_T, :err_A,
                                            :power_residual, :wall_time_s); allrows=true)
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
