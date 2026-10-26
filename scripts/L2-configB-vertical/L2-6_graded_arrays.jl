#=
L2-6 — Graded resonator arrays on a long floating plate.
Pre- and post-processing of the test defined in src/L2_6_graded_arrays.jl.

Study: P1 plate (T_ρ = 0), H = 10 m, open water on both sides; incident plate
wave from a point source on a bare lead-in, viscous sponges at both plate
ends (src/L2_6_graded_arrays.jl); p = 3, 8 elements per min(λ_open, λ_p, 4ℓ_f).
  - frequency set: ω ∈ {2, 3, 5, 7} rad/s; linear and exponential grading
    (ω_r0 = 10 rad/s, L_g = 500 m, down to ω_r = 1), a ∈ {1, 0.25} m,
    ζ ∈ {0.01, 0.02, 0.05} with uniform loading n₀m_ρ,r = 0.01 m, and the
    mass-graded loading (√μ* = 0.163) at ζ = 0.02;
  - density set: 41 frequencies in [9.5, 10.5] rad/s; ω_r = 10 rad/s, a = 1 m,
    n = n₀e^{−x/L_g} on [0, L_g], L_g ∈ {1, 5, 20, 100} λ (λ = 2.32 m, bare
    plate at ω = 9.8), ζ ∈ {0, 0.02}, and abrupt arrays with the same number of
    resonators;
  - mesh check: 12 elements per wavelength at ω ∈ {5, 7} (frequency set) and
    ω = 10 (density set).

Acceptance (test matrix §L2.6; set before the runs from the local theory):
  - type I, every case: G_B,h (evaluation matrix) against spring-based columns
    ≤ 1e-8; reciprocity ≤ 1e-8; energy balance |R|² + |T|² + ΣP − 1 ≤ 1e-3 for
    every array; bare plate: |A⁺(W2)/A⁺(W1)| − 1 ≤ 1e-3 (lossless lead);
  - type D, every case: sponge reflection |A⁻/A⁺| in W2 ≤ 1e-2;
  - type A, frequency set, a = 0.25 m (ε_s ≤ 0.1; at a = 1 m, ε_s reaches
    0.36 at ω = 7 and the local law is not expected to hold — reported):
    peak of the dissipated power per unit length within Δx/2 of x*
    (eq:x_star_linear, eq:x_star_exp), and FWHM/Δx ∈ [0.5, 2]
    (eq:footprint_general);
  - type A: exponential law, mass-graded, a = 0.25 m, ζ = 0.02: footprint flat
    in ω, max/min FWHM ≤ 1.3 (reported for the uniform loading);
  - type A, density set: band-averaged |R|² (41 frequencies) of the taper
    below that of the abrupt array for L_g ≥ 5λ (reported for L_g = λ, where
    λ/L_g = 1 and the taper is not adiabatic);
  - type D: 8 vs 12 elements per wavelength: peak position within 0.05Δx,
    FWHM within 5 %, absorbed fraction within 1e-2 (frequency set, a = 0.25 m);
    |R| within 1e-2 (density set, ω = 10).

Outputs:
  data/sims/L2-6/      one DrWatson record per case (set, ω)
  data/exp_pro/L2-6/   footprint.csv, density.csv, checks.csv
  plots/L2-6/          profiles.pdf/png, footprint.pdf/png, density.pdf/png

Run:
  julia --project -O1 scripts/L2-configB-vertical/L2-6_graded_arrays.jl [--smoke] [--force] [--cases 1:5]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
using Statistics: mean
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L2_6

const TEST_ID = "L2-6"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

const FREQ_W = (2.0, 3.0, 5.0, 7.0)
const DENS_W = Tuple(range(9.5, 10.5; length=41))
const N_MAIN, N_FINE = 8, 12
const FREQ_W_FINE = (5.0, 7.0)
const DENS_W_FINE = 10.0

function cases(; smoke::Bool=false)
    smoke && return [L2_6_Params(set=:density, ω=10.0, elements_per_wavelength=4, Lg_factors=(1.0, 5.0))]
    freq = [L2_6_Params(set=:frequency, ω=ω, elements_per_wavelength=N_MAIN) for ω in FREQ_W]
    dens = [L2_6_Params(set=:density, ω=ω, elements_per_wavelength=N_MAIN) for ω in DENS_W]
    fine = [[L2_6_Params(set=:frequency, ω=ω, elements_per_wavelength=N_FINE) for ω in FREQ_W_FINE];
            L2_6_Params(set=:density, ω=DENS_W_FINE, elements_per_wavelength=N_FINE)]
    return [freq; fine; dens]
end

record_name(p::L2_6_Params) = savename(p; accesses=VARIABLE, sigdigits=6) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

function simulate(p::L2_6_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L2_6_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L2_6_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── Post-processing ────────────────────────────────────────────────────────

"Peak position and full width at half maximum of the dissipated power per unit length."
function footprint(x::AbstractVector, y::AbstractVector)
    m = argmax(y)
    half = y[m] / 2
    l = findlast(i -> y[i] < half, 1:m)
    r = findfirst(i -> y[i] < half, m:length(y))
    cross(i, j) = x[i] + (half - y[i]) * (x[j] - x[i]) / (y[j] - y[i])
    xl = l === nothing ? NaN : cross(l, l + 1)
    xr = r === nothing ? NaN : cross(m + r - 2, m + r - 1)
    return (x_peak=x[m], fwhm=xr - xl)
end

function footprint_table(cs, records)
    rows = NamedTuple[]
    for (p, d) in zip(cs, records)
        p.set === :frequency || continue
        for (id, r) in d["arrays"]
            x, y = r["x"], r["absorbed"] ./ r["a"]
            fp = footprint(x, y)
            pr = predictions(p, p.ω, Symbol(r["family"]), r["zeta"], Symbol(r["mass"]))
            push!(rows, (ω=p.ω, n=p.elements_per_wavelength, id=id, family=r["family"], a=r["a"], zeta=r["zeta"],
                         mass=r["mass"], x_peak=fp.x_peak, fwhm=fp.fwhm, x_star=pr.x_star, dx_pred=pr.Δx,
                         sqrt_mu=pr.sqrt_mu, absorbed=sum(r["absorbed"]), R=abs(r["R"]), T=abs(r["T"]),
                         # width in units of the damping-only footprint 2ζ ω|dx*/dω|
                         fwhm_2zeta=fp.fwhm / (pr.Δx / max(2r["zeta"], pr.sqrt_mu) * 2r["zeta"]),
                         balance=r["balance"]))
        end
    end
    return DataFrame(rows)
end

function density_table(cs, records)
    rows = NamedTuple[]
    for (p, d) in zip(cs, records)
        p.set === :density || continue
        λ = d["lambda_ref"]
        for (id, r) in d["arrays"]
            push!(rows, (ω=p.ω, n=p.elements_per_wavelength, id=id, family=r["family"], Lg_lambda=round(r["Lg"] / λ),
                         zeta=r["zeta"], N=length(r["x"]), R=abs(r["R"]), T=abs(r["T"]), absorbed=sum(r["absorbed"]),
                         balance=r["balance"], balance_RT=get(r, "balance_RT", NaN),
                         incident_change=get(r, "incident_change", NaN)))
        end
    end
    return DataFrame(rows)
end

function checks(cs, records, fp::DataFrame, dn::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    for (p, d) in zip(cs, records)
        tag = @sprintf("%s, ω=%.4g, n=%d", p.set, p.ω, p.elements_per_wavelength)
        add!("G_B,h evaluation matrix vs spring columns, $(tag)", "I", d["check_G"], 1e-8)
        add!("reciprocity of G_B,h, $(tag)", "I", d["reciprocity"], 1e-8)
        add!("energy balance (max over arrays), $(tag)", "I", maximum(abs(r["balance"]) for r in values(d["arrays"])), 1e-3)
        add!("bare transmission W1 → W2, |T₀| − 1, $(tag)", "I", abs(d["bare_transmission"] - 1), 1e-3)
        add!("sponge reflection, $(tag)", "D", d["sponge_reflection"], 1e-2)
    end
    main = filter(r -> r.n == N_MAIN && r.a == 0.25, fp)
    for r in eachrow(main)
        tag = @sprintf("%s, %s, ζ=%g, ω=%g", r.family, r.mass, r.zeta, r.ω)
        add!("peak within Δx/2 of x*, a=0.25, $(tag)", "A", abs(r.x_peak - r.x_star) / r.dx_pred, 0.5)
        ratio = r.fwhm / r.dx_pred
        push!(rows, (name="FWHM/Δx ∈ [0.5, 2], a=0.25, $(tag)", kind="A", value=ratio, tolerance=2.0,
                     passed=0.5 <= ratio <= 2.0))
    end
    g = filter(r -> r.family == "exp" && r.mass == "graded" && r.zeta == 0.02, main)
    nrow(g) >= 2 && add!("mass-graded exponential: max/min FWHM over ω", "A", maximum(g.fwhm) / minimum(g.fwhm), 1.3)
    for Lgl in sort(unique(dn.Lg_lambda)), ζ in sort(unique(dn.zeta))
        Lgl >= 5 || continue
        s = filter(r -> r.n == N_MAIN && r.Lg_lambda == Lgl && r.zeta == ζ, dn)
        t, a = filter(r -> r.family == "taper", s), filter(r -> r.family == "abrupt", s)
        (nrow(t) > 0 && nrow(a) > 0) || continue
        add!(@sprintf("taper vs abrupt, band-averaged |R|² ratio, L_g=%gλ, ζ=%g", Lgl, ζ), "A",
             mean(t.R .^ 2) / mean(a.R .^ 2), 1.0)
    end
    for ω in FREQ_W_FINE, r in eachrow(filter(r -> r.ω == ω && r.n == N_MAIN && r.a == 0.25, fp))
        f = filter(s -> s.ω == ω && s.n == N_FINE && s.id == r.id, fp)
        nrow(f) == 1 || continue
        tag = @sprintf("%s, ω=%g", r.id, ω)
        add!("mesh 8 vs 12: peak shift/Δx, $(tag)", "D", abs(r.x_peak - f.x_peak[1]) / r.dx_pred, 0.05)
        add!("mesh 8 vs 12: FWHM, $(tag)", "D", abs(r.fwhm / f.fwhm[1] - 1), 0.05)
        add!("mesh 8 vs 12: absorbed fraction, $(tag)", "D", abs(r.absorbed - f.absorbed[1]), 1e-2)
    end
    for r in eachrow(filter(r -> r.ω == DENS_W_FINE && r.n == N_MAIN, dn))
        f = filter(s -> s.ω == DENS_W_FINE && s.n == N_FINE && s.id == r.id, dn)
        nrow(f) == 1 || continue
        add!("mesh 8 vs 12: |R|, $(r.id), ω=10", "D", abs(r.R - f.R[1]), 1e-2)
    end
    return isempty(rows) ? DataFrame(name=String[], kind=String[], value=Float64[], tolerance=Float64[], passed=Bool[]) :
           DataFrame(rows)
end

# ─── Figures ────────────────────────────────────────────────────────────────

function plot_profiles(cs, records, dir::AbstractString)
    k = [i for (i, p) in enumerate(cs) if p.set === :frequency && p.elements_per_wavelength == N_MAIN]
    isempty(k) && return
    theme = Theme(fontsize=11, Axis=(xgridvisible=false, ygridvisible=false))
    with_theme(theme) do
        fig = Figure(size=(900, 520))
        for (row, fam) in enumerate(("linear", "exp")), (col, mass) in enumerate(("uniform", "graded"))
            ax = Axis(fig[row, col]; xlabel=L"x\ \mathrm{(m)}", ylabel="absorbed fraction per m",
                      title="$(fam), $(mass) loading, a = 0.25 m, ζ = 0.02")
            for (j, i) in enumerate(k)
                p, d = cs[i], records[i]
                id = "$(fam)_a=0.25_zeta=0.02_$(mass)"
                haskey(d["arrays"], id) || continue
                r = d["arrays"][id]
                c = Makie.wong_colors()[j]
                lines!(ax, r["x"], r["absorbed"] ./ r["a"]; color=c, label=@sprintf("ω = %g", p.ω))
                pr = predictions(p, p.ω, Symbol(fam), 0.02, Symbol(mass))
                vspan!(ax, [pr.x_star - pr.Δx / 2], [pr.x_star + pr.Δx / 2]; color=(c, 0.15))
            end
            row == 1 && col == 2 && axislegend(ax; position=:rt, framevisible=false, labelsize=9)
        end
        mkpath(dir)
        save(joinpath(dir, "profiles.pdf"), fig)
        save(joinpath(dir, "profiles.png"), fig; px_per_unit=300 / 96)
    end
end

function plot_footprint(fp::DataFrame, dir::AbstractString)
    nrow(fp) == 0 && return
    theme = Theme(fontsize=11, Axis=(xgridvisible=false, ygridvisible=false))
    with_theme(theme) do
        fig = Figure(size=(820, 330))
        for (col, fam) in enumerate(("linear", "exp"))
            ax = Axis(fig[1, col]; xlabel=L"\omega\ \mathrm{(rad/s)}", ylabel="FWHM (m)", title="$(fam) grading, a = 0.25 m")
            for (j, (mass, ζ)) in enumerate((("uniform", 0.01), ("uniform", 0.02), ("uniform", 0.05), ("graded", 0.02)))
                g = sort(filter(r -> r.n == N_MAIN && r.a == 0.25 && r.family == fam && r.mass == mass && r.zeta == ζ, fp), :ω)
                nrow(g) == 0 && continue
                c = Makie.wong_colors()[j]
                scatter!(ax, g.ω, g.fwhm; color=c, label="$(mass), ζ = $(ζ)")
                lines!(ax, g.ω, g.dx_pred; color=c, linestyle=:dash)
            end
            col == 2 && axislegend(ax; position=:lt, framevisible=false, labelsize=9)
        end
        mkpath(dir)
        save(joinpath(dir, "footprint.pdf"), fig)
        save(joinpath(dir, "footprint.png"), fig; px_per_unit=300 / 96)
    end
end

function plot_density(dn::DataFrame, dir::AbstractString)
    nrow(dn) == 0 && return
    nm = N_MAIN in dn.n ? N_MAIN : minimum(dn.n)          # smoke runs use a coarser mesh
    theme = Theme(fontsize=11, Axis=(xgridvisible=false, ygridvisible=false))
    with_theme(theme) do
        fig = Figure(size=(900, 330))
        for (col, ζ) in enumerate(sort(unique(dn.zeta)))
            ax = Axis(fig[1, col]; xlabel=L"\omega\ \mathrm{(rad/s)}", ylabel=L"|R|", title="ζ = $(ζ)")
            for (j, Lgl) in enumerate(sort(unique(dn.Lg_lambda)))
                c = Makie.wong_colors()[j]
                for (fam, ls) in (("taper", :solid), ("abrupt", :dash))
                    g = sort(filter(r -> r.n == nm && r.zeta == ζ && r.Lg_lambda == Lgl && r.family == fam, dn), :ω)
                    nrow(g) == 0 && continue
                    if fam == "taper"
                        lines!(ax, g.ω, g.R; color=c, linestyle=ls, label=@sprintf("L_g = %gλ", Lgl))
                    else
                        lines!(ax, g.ω, g.R; color=c, linestyle=ls)
                    end
                end
            end
            col == 1 && axislegend(ax; position=:lt, framevisible=false, labelsize=9)
        end
        mkpath(dir)
        save(joinpath(dir, "density.pdf"), fig)
        save(joinpath(dir, "density.png"), fig; px_per_unit=300 / 96)
    end
end

# ─── Driver ─────────────────────────────────────────────────────────────────

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO,
              only::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cs = cases(; smoke=smoke)
    if only !== nothing
        for (i, p) in zip(only, cs[only])
            @info "case $(i)/$(length(cs))" p.set p.ω p.elements_per_wavelength
            simulate(p; root=root, force=force)
        end
        return nothing
    end
    records = [simulate(p; root=root, force=force) for p in cs]
    fp = footprint_table(cs, records)
    dn = density_table(cs, records)
    ck = checks(cs, records, fp, dn)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "footprint.csv"), fp)
    CSV.write(joinpath(out, "density.csv"), dn)
    CSV.write(joinpath(out, "checks.csv"), ck)
    plots = joinpath(root, "plots", TEST_ID)
    plot_profiles(cs, records, plots)
    plot_footprint(fp, plots)
    plot_density(dn, plots)
    nrow(fp) > 0 && show(stdout, MIME"text/plain"(),
                         select(filter(r -> r.a == 0.25, fp), :ω, :n, :family, :mass, :zeta, :x_peak, :x_star, :fwhm,
                                :dx_pred, :absorbed, :R, :balance); allrows=true)
    println()
    if nrow(dn) > 0
        nm = N_MAIN in dn.n ? N_MAIN : minimum(dn.n)
        s = combine(groupby(filter(r -> r.n == nm, dn), [:family, :Lg_lambda, :zeta]),
                    :R => (r -> mean(r .^ 2)) => :mean_R2, :absorbed => mean => :mean_absorbed, :N => first => :N)
        show(stdout, MIME"text/plain"(), sort(s, [:zeta, :Lg_lambda, :family]); allrows=true)
        println()
    end
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-80s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (footprint=fp, density=dn, checks=ck)
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
