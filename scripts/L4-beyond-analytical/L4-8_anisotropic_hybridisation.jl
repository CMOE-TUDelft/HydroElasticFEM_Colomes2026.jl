#=
L4-8 — anisotropic hybridisation of a finite rectangular plate (eq:mode_dilution).
Pre- and post-processing of src/L4_8_anisotropic_hybridisation.jl (problem,
Ritz reduction and observables are documented there).

Binding configuration (frozen in row L4.8 of verification_test_matrix_2026-10-22.md
before the binding run): P1 plate (T = 0, ν = 0), free edges, L_x = 20 m,
L_y ∈ {20, 40, 80} m, H ∈ {10, 50} m, uniform homogenised population
n₀m_ρ,r = 0.01 m tuned to each bare wet mode. Quarter domain, p = 2, 8 cells per
λ_open (grid top), plate ≥ 16 cells per side, nz = 8; open water λ_open, frame
1.5λ_open (grid bottom); 7 Chebyshev frequencies in ω_ref(1 ± 0.12); Ritz basis
X_i(x)X_j(y), i, j ∈ {0, 1, 3, 5, 7}.

Cases (one Julia process each, rss_guard + run_limited, heavy-run lock):
   1–12   main: H ∈ {10, 50} × (L_y = 20: sym, anti; 40: m10, m01; 80: m10, m01)
   13–14  buffer variant (gap 1.5λ, frame 2λ), L_y = 40, H = 10, m10 and m01
   15–16  mesh variant (11 cells per λ, plate 24, nz = 10), same targets
   --validate2d   the reduced procedure on the L2-5 channel (L = 23, 46 m, H = 10 m)

Outputs:
  data/sims/L4-8/       one record per case (not committed)
  data/exp_pro/L4-8/    modes.csv, validation_2d.csv, checks.csv
  plots/L4-8/           anisotropic_hybridisation.{pdf,png}

Run:
  julia --project -O1 scripts/L4-beyond-analytical/L4-8_anisotropic_hybridisation.jl --list
  julia --project -O1 scripts/L4-beyond-analytical/L4-8_anisotropic_hybridisation.jl --case 1
  julia --project -O1 scripts/L4-beyond-analytical/L4-8_anisotropic_hybridisation.jl --smoke 3
  julia --project -O1 scripts/L4-beyond-analytical/L4-8_anisotropic_hybridisation.jl --validate2d
  julia --project scripts/L4-beyond-analytical/L4-8_anisotropic_hybridisation.jl       # tables, checks, figure
=#

using DrWatson
using DataFrames
using CSV
using Printf
using TOML
using LinearAlgebra
import HydroElasticFEM
using HydroElasticFEM_xxx2026
using HydroElasticFEM_xxx2026.References: added_mass
include(joinpath(@__DIR__, "..", "..", "src", "L4_8_anisotropic_hybridisation.jl"))
using .L4_8

const TEST_ID = "L4-8"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const LX = 20.0
const LOADING = 0.01

"""Grid centres ω_ref (rad/s) per (L_y, H, target): ω_mn of the coarse smoke pass (4 cells/λ, plate 8, nz = 4,
±30 % grid, 5 nodes). m10, L_y = 40, H = 10: smoke 2026-10-29 (2.073); the others smoke 2026-10-30 (cases 1, 2, 4–12)."""
const OMEGA_REF = Dict(
    (20.0, 10.0, :sym) => 2.119, (20.0, 10.0, :anti) => 2.008,
    (40.0, 10.0, :m10) => 2.07, (40.0, 10.0, :m01) => 1.501,
    (80.0, 10.0, :m10) => 2.020, (80.0, 10.0, :m01) => 1.110,
    (20.0, 50.0, :sym) => 2.117, (20.0, 50.0, :anti) => 2.013,
    (40.0, 50.0, :m10) => 2.067, (40.0, 50.0, :m01) => 1.504,
    (80.0, 50.0, :m10) => 2.068, (80.0, 50.0, :m01) => 1.097)

const MAIN = [(Ly, H, t) for H in (10.0, 50.0) for (Ly, t) in ((20.0, :sym), (20.0, :anti), (40.0, :m10), (40.0, :m01),
                                                                  (80.0, :m10), (80.0, :m01))]

function cases()
    c = [L4_8_Params(Ly=Ly, H=H, target=t, omega_ref=OMEGA_REF[(Ly, H, t)]) for (Ly, H, t) in MAIN]
    for t in (:m10, :m01)
        push!(c, L4_8_Params(Ly=40.0, H=10.0, target=t, omega_ref=OMEGA_REF[(40.0, 10.0, t)], gap_over_λ=1.5, frame_over_λ=2.0))
    end
    for t in (:m10, :m01)
        push!(c, L4_8_Params(Ly=40.0, H=10.0, target=t, omega_ref=OMEGA_REF[(40.0, 10.0, t)], epw=11.0, plate_cells=24, nz=10))
    end
    return c
end

"Coarse smoke version of case p: wide grid to locate ω_mn."
smoke(p) = L4_8_Params(p; epw=4.0, plate_cells=8, nz=4, grid_halfwidth=0.3, n_grid=5)

record_name(p::L4_8_Params) = savename(p; accesses=VARIABLE, sigdigits=6) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")), "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"))
end

function simulate(p::L4_8_Params; force::Bool=false, sub::String="")
    path = joinpath(REPO, "data", "sims", TEST_ID, sub)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO, storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L4_8_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        return d
    end
    return data
end

"Reduced analysis of one record (full basis and the {0,1,3,5} sub-basis)."
function analyse(p::L4_8_Params, d)
    t = L4_8.target_vector(p)
    sh = L4_8.shapes(p)
    full = L4_8.reduce_case(d["K_B"], d["M_B"], d["N_B"], d["A_re"], d["A_im"], d["omegas"], t; loading=LOADING)
    sel = [a for (a, (i, j)) in enumerate(sh) if i <= 5 && j <= 5]
    sub = L4_8.reduce_case(d["K_B"], d["M_B"], d["N_B"], d["A_re"], d["A_im"], d["omegas"], t; loading=LOADING, sel=sel)
    return full, sub
end

"Wavenumber of the target mode in eq:mode_dilution."
kmn(p) = p.target == :m01 ? BETA[1] / p.Ly : BETA[1] / LX

function row(p::L4_8_Params, d)
    full, sub = analyse(p, d)
    N, M, K = d["N_B"], d["M_B"], d["K_B"]
    sh = L4_8.shapes(p)
    i00 = findfirst(==((0, 0)), sh)
    area = p.Lx * p.Ly / 4
    est = local_estimate(kmn(p), p.H; loading=LOADING)
    # forced-response truncation check at ω_ref (grid centre): ∫f_t η_h (FE) vs reduced
    ωr = p.omega_ref
    jc = findmin(abs.(d["omegas"] .- ωr))[2]
    Z = K - ωr^2 * (M + d["A_re"][jc] + im * d["A_im"][jc])
    fB = d["forced_load_B"]
    Wred = transpose(fB) * (Z \ fB)
    Asym = maximum(norm(a - transpose(a)) / norm(a) for a in d["A_re"])
    # square: invariance under x ↔ y (permutation of the basis)
    swap = 0.0
    if p.Ly == p.Lx
        π_ = [findfirst(==((j, i)), sh) for (i, j) in sh]
        swap = maximum(norm(X[π_, π_] - X) / norm(X) for X in (K, M, d["A_re"][jc]))
    end
    res = maximum(d["w$(n)_rel_residual"] for n in 1:length(d["omegas"]))
    return (Ly=p.Ly, H=p.H, target=String(p.target), variant=variant(p), omega_ref=p.omega_ref,
            omega=full.omega, split=full.split, sqrt_mu_mode=sqrt(full.mu_mode), mu_mode=full.mu_mode,
            omega_minus=full.omega_minus, omega_plus=full.omega_plus, zeta_rad=full.zeta_rad,
            resolvable=full.split / (2 * full.zeta_rad), mac_target=full.mac_target,
            proj=min(full.proj_minus, full.proj_plus), inside=full.inside,
            omega_sub=sub.omega, split_sub=sub.split, k_mn=kmn(p), mu_est=est.mu, omega_est=est.omega,
            ratio_mu=full.split^2 / est.mu,
            mass_check=norm(M - p.m_ρ * N) / norm(M), heave_check=abs(K[i00, i00] / (p.g * area) - 1),
            linearity=d["Sss_linearity"], A_sym=Asym, swap=swap, residual=max(res, d["forced_rel_residual"]),
            forced_trunc=abs(Wred / d["forced_eta_t"] - 1), ordering=d["ordering"], N=d["N"],
            nx=d["nx"], ny=d["ny"], nz=d["nz"], wall_time_s=d["wall_time_s"])
end

variant(p) = p.gap_over_λ != 1.0 ? "buffer" : (p.epw != 8.0 ? "mesh" : "main")

function loglog_slope(x, y)
    X, Y = log.(x), log.(y)
    return sum((X .- sum(X) / length(X)) .* (Y .- sum(Y) / length(Y))) / sum((X .- sum(X) / length(X)) .^ 2)
end

"μ₁₀/μ₀₁ (FE, from Δ²) and the estimates, per depth and aspect ratio."
function ratios(df)
    out = NamedTuple[]
    main = filter(r -> r.variant == "main", df)
    for H in (10.0, 50.0)
        sq = filter(r -> r.H == H && r.Ly == LX, main)
        nrow(sq) == 2 && push!(out, (H=H, aspect=1.0, ratio_FE=sq.split[1]^2 / sq.split[2]^2, ratio_est=1.0,
                                     ratio_deep=1.0, ratio_mode=sq.mu_mode[1] / sq.mu_mode[2]))
        for Ly in (40.0, 80.0)
            a = filter(r -> r.H == H && r.Ly == Ly && r.target == "m10", main)
            b = filter(r -> r.H == H && r.Ly == Ly && r.target == "m01", main)
            (nrow(a) == 1 && nrow(b) == 1) || continue
            push!(out, (H=H, aspect=Ly / LX, ratio_FE=a.split[1]^2 / b.split[1]^2, ratio_est=a.mu_est[1] / b.mu_est[1],
                        ratio_deep=Ly / LX, ratio_mode=a.mu_mode[1] / b.mu_mode[1]))
        end
    end
    return DataFrame(out)
end

"""
Near-degenerate (1,0)/(0,3) pair at L_y = 40 (k = 0.237 vs 0.275 1/m): both candidate
wet modes, tracked from X₁(x) and from X₃(y), with their MACs against both dry shapes
(Governor/PI request 2026-10-30). Reported, not a frozen check.
"""
function pair_L40(cs, recs)
    out = NamedTuple[]
    for (p, d) in zip(cs, recs)
        (p.Ly == 40.0 && p.target == :m10) || continue
        sh = L4_8.shapes(p)
        e10 = zeros(length(sh)); e10[findfirst(==((1, 0)), sh)] = 1
        e03 = zeros(length(sh)); e03[findfirst(==((0, 3)), sh)] = 1
        for (name, t) in (("from X1(x)", e10), ("from X3(y)", e03))
            r = L4_8.reduce_case(d["K_B"], d["M_B"], d["N_B"], d["A_re"], d["A_im"], d["omegas"], t; loading=LOADING)
            push!(out, (H=p.H, variant=variant(p), candidate=name, omega=r.omega, split=r.split, mu_mode=r.mu_mode,
                        zeta_rad=r.zeta_rad, mac_10=L4_8.macN(r.psi, e10, r.N), mac_03=L4_8.macN(r.psi, e03, r.N),
                        mac_01=L4_8.macN(r.psi, (s = zeros(length(sh)); s[findfirst(==((0, 1)), sh)] = 1; s), r.N),
                        inside=r.inside))
        end
    end
    return DataFrame(out)
end

"""
Oblique absorption of the frame: buffer variant (gap 1.5λ, frame 2λ) against the main
layout at L_y = 40, H = 10, same ω grid. Relative change of Re/Im A_B (Frobenius, max
over ω), of the target diagonal entry, and of ω_mn, Δ, ζ_rad.
"""
function buffer_absorption(cs, recs, df)
    out = NamedTuple[]
    for t in (:m10, :m01)
        jb = findfirst(p -> variant(p) == "buffer" && p.target == t, cs)
        jm = findfirst(p -> variant(p) == "main" && p.target == t && p.Ly == 40.0 && p.H == 10.0, cs)
        (jb === nothing || jm === nothing) && continue
        db, dm = recs[jb], recs[jm]
        maximum(abs.(db["omegas"] .- dm["omegas"])) < 1e-12 || continue
        tv = L4_8.target_vector(cs[jm]); a = findfirst(!=(0), tv)
        rre = maximum(norm(x - y) / norm(y) for (x, y) in zip(db["A_re"], dm["A_re"]))
        rim = maximum(norm(x - y) / norm(y) for (x, y) in zip(db["A_im"], dm["A_im"]))
        dre = maximum(abs(x[a, a] / y[a, a] - 1) for (x, y) in zip(db["A_re"], dm["A_re"]))
        dim = maximum(abs(x[a, a] / y[a, a] - 1) for (x, y) in zip(db["A_im"], dm["A_im"]))
        rb = only(eachrow(filter(r -> r.variant == "buffer" && r.target == String(t), df)))
        rm = only(eachrow(filter(r -> r.variant == "main" && r.target == String(t) && r.Ly == 40.0 && r.H == 10.0, df)))
        push!(out, (target=String(t), dRe_AB=rre, dIm_AB=rim, dRe_Att=dre, dIm_Att=dim, d_omega=rb.omega / rm.omega - 1,
                    d_split=rb.split / rm.split - 1, d_zeta=rb.zeta_rad / rm.zeta_rad - 1))
    end
    return DataFrame(out)
end

const L25 = Dict(23.0 => (omega=1.9416727438753165, split=0.06274357175359825),
                 46.0 => (omega=1.3694701198206192, split=0.04334768452250789))   # L2-5 records, H = 10, 16 el./λ

function checks(df, rt, v2)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    ge!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value >= tol))
    for r in eachrow(df)
        tag = @sprintf("L_y=%g, H=%g, %s, %s", r.Ly, r.H, r.target, r.variant)
        add!("residual (S_ff, every RHS, and forced full solve), $(tag)", "I", r.residual, 1e-8)
        add!("S_ss linear in ω² (third grid ω), $(tag)", "I", r.linearity, 1e-8)
        add!("dry mass M_B = m_ρ N, $(tag)", "I", r.mass_check, 1e-8)
        add!("heave stiffness K_B,00 = g·area, $(tag)", "I", r.heave_check, 1e-8)
        add!("symmetry of Re A_B (reciprocity), $(tag)", "I", r.A_sym, 1e-8)
        add!("METIS ordering used (0 = yes), $(tag)", "I", r.ordering == "metis" ? 0.0 : 1.0, 0.0)
        r.Ly == LX && add!("square: K_B, M_B, A_B invariant under x↔y, $(tag)", "I", r.swap, 1e-8)
        ge!("wet mode identified: MAC(ψ, dry target) ≥ 0.3, $(tag)", "I", r.mac_target, 0.3)
        ge!("hybrid pair identified: projection ≥ 0.2, $(tag)", "I", r.proj, 0.2)
        add!("ω_mn, ω± inside the interpolation grid (0 = yes), $(tag)", "I", r.inside ? 0.0 : 1.0, 0.0)
        add!("basis truncation {0,1,3,5} vs {0,1,3,5,7}: splitting, $(tag)", "D", abs(r.split_sub / r.split - 1), 1e-2)
        add!("basis truncation {0,1,3,5} vs {0,1,3,5,7}: ω_mn, $(tag)", "D", abs(r.omega_sub / r.omega - 1), 1e-3)
        add!("forced response to the target load at ω_ref: reduced vs FE, $(tag)", "D", r.forced_trunc, 2e-2)
        r.variant == "main" && add!("splitting vs √μ_mode (L2-5: 1–6.5 %), $(tag)", "A", abs(r.split / r.sqrt_mu_mode - 1), 0.1)
    end
    for v in ("buffer", "mesh"), t in ("m10", "m01")
        a = filter(r -> r.variant == v && r.target == t, df)
        b = filter(r -> r.variant == "main" && r.target == t && r.Ly == 40.0 && r.H == 10.0, df)
        (nrow(a) == 1 && nrow(b) == 1) || continue
        add!("$(v) variant: splitting, L_y=40, H=10, $(t)", "D", abs(a.split[1] / b.split[1] - 1), 2e-2)
        add!("$(v) variant: ω_mn, L_y=40, H=10, $(t)", "D", abs(a.omega[1] / b.omega[1] - 1), 2e-3)
    end
    for r in eachrow(v2)
        add!(@sprintf("2D Ritz vs L2-5 eigWet: ω₁, L=%g, H=10", r.L), "A", abs(r.omega / r.omega_L25 - 1), 2e-3)
        add!(@sprintf("2D Ritz vs L2-5 eigWet: splitting, L=%g, H=10", r.L), "A", abs(r.split / r.split_L25 - 1), 2e-2)
    end
    for H in (10.0, 50.0)
        g = sort(filter(r -> r.H == H, rt), :aspect)
        nrow(g) == 3 || continue
        ge!(@sprintf("trend: μ₁₀/μ₀₁ increases with L_y/L_x (min step ratio), H=%g", H), "A",
            minimum(g.ratio_FE[2:end] ./ g.ratio_FE[1:end-1]), 1.0 + 1e-6)
        for r in eachrow(g)
            r.aspect == 1 && continue
            add!(@sprintf("μ₁₀/μ₀₁ vs eq:mode_dilution (finite H), L_y/L_x=%g, H=%g", r.aspect, H), "A",
                 abs(r.ratio_FE / r.ratio_est - 1), 0.25)
        end
        H == 50.0 && add!("deep water (H=50): exponent of μ₁₀/μ₀₁ vs L_y/L_x, |s − 1|", "A",
                          abs(loglog_slope(g.aspect, g.ratio_FE) - 1), 0.3)
    end
    return DataFrame(rows)
end

function validate2d(; force=false)
    path = joinpath(REPO, "data", "sims", TEST_ID, "validation_2d.csv")
    isfile(path) && !force && return CSV.read(path, DataFrame)
    rows = NamedTuple[]
    for L in (23.0, 46.0)
        st = @timed L4_8.ritz_2d(L, 10.0; omega_ref=L25[L].omega)
        r = st.value
        push!(rows, (L=L, omega=r.red.omega, split=r.red.split, omega_L25=L25[L].omega, split_L25=L25[L].split,
                     zeta_rad=r.red.zeta_rad, mac_target=r.red.mac_target, linearity=r.linearity, inside=r.red.inside,
                     wall_time_s=st.time))
        @info "2D" rows[end]
    end
    df = DataFrame(rows)
    mkpath(dirname(path)); CSV.write(path, df)
    return df
end

function plot_all(df, rt, cs, records, dir)
    @eval using CairoMakie
    mkpath(dir)
    Base.invokelatest() do
        fig = Figure(size=(900, 330), fontsize=11)
        # (a) μ₁₀/μ₀₁ vs aspect ratio
        ax = Axis(fig[1, 1]; xscale=log2, yscale=log2, xlabel="L_y / L_x", ylabel="μ₁₀ / μ₀₁", title="(a)")
        for (i, H) in enumerate((10.0, 50.0))
            g = sort(filter(r -> r.H == H, rt), :aspect)
            c = Makie.wong_colors()[i]
            scatter!(ax, g.aspect, g.ratio_FE; color=c, label="FE (Δ²), H = $(Int(H)) m")
            lines!(ax, g.aspect, g.ratio_est; color=c, linestyle=:dash, label="eq. mode_dilution, H = $(Int(H)) m")
        end
        lines!(ax, [1, 4], [1, 4]; color=:black, linestyle=:dot, label="L_y/L_x (deep)")
        axislegend(ax; position=:lt, framevisible=false, labelsize=8)
        # (b) splitting vs √μ estimate
        ax2 = Axis(fig[1, 2]; xlabel="k_mn (1/m)", ylabel="Δ = (ω₊ − ω₋)/ω_mn", title="(b)")
        m = filter(r -> r.variant == "main", df)
        for (i, H) in enumerate((10.0, 50.0))
            g = sort(filter(r -> r.H == H, m), :k_mn)
            c = Makie.wong_colors()[i]
            scatter!(ax2, g.k_mn, g.split; color=c, label="FE, H = $(Int(H)) m")
            lines!(ax2, g.k_mn, sqrt.(g.mu_est); color=c, linestyle=:dash, label="√μ_mn, eq. mode_dilution")
            scatter!(ax2, g.k_mn, 2 .* g.zeta_rad; color=c, marker=:xcross, label="2ζ_rad")
        end
        axislegend(ax2; position=:lt, framevisible=false, labelsize=8)
        # (c) forced response of the reduced model, L_y = 80, H = 10, mode (1,0)
        ax3 = Axis(fig[1, 3]; xlabel="ω / ω_mn", ylabel="|response| (normalised)", title="(c)")
        j = findfirst(p -> p.Ly == 80.0 && p.H == 10.0 && p.target == :m10 && variant(p) == "main", cs)
        if j !== nothing
            p = cs[j]; d = records[j]
            full, _ = analyse(p, d)
            ωq = range(minimum(d["omegas"]), maximum(d["omegas"]); length=301)
            bare, res = L4_8.forced_curves(full, L4_8.target_vector(p), ωq; loading=LOADING)
            s = maximum(bare)
            lines!(ax3, ωq ./ full.omega, bare ./ s; color=:black, label="bare")
            lines!(ax3, ωq ./ full.omega, res ./ s; color=Makie.wong_colors()[3], label="with resonators")
            vlines!(ax3, [full.omega_minus, full.omega_plus] ./ full.omega; color=:gray, linestyle=:dash, label="ω±")
            axislegend(ax3; position=:rt, framevisible=false, labelsize=8)
        end
        save(joinpath(dir, "anisotropic_hybridisation.pdf"), fig)
        save(joinpath(dir, "anisotropic_hybridisation.png"), fig; px_per_unit=300 / 96)
    end
end

function main()
    cs = cases()
    if "--list" in ARGS
        for (i, p) in enumerate(cs)
            lay = layout(p)
            n = (2lay.nx + 1) * (2lay.ny + 1) * (2lay.nz + 1) + (2lay.nx + 1) * (2lay.ny + 1)
            @printf("%2d  L_y=%-4g H=%-4g %-5s ω_ref=%.3f %-6s  %3d×%3d×%2d  N≈%7d  gap %.1f m frame %.1f m h_open %.2f\n",
                    i, p.Ly, p.H, p.target, p.omega_ref, variant(p), lay.nx, lay.ny, lay.nz, n, lay.gap, lay.frame, lay.hopen)
        end
        return
    end
    i = findfirst(==("--case"), ARGS)
    if i !== nothing
        for k in parse.(Int, split(ARGS[i + 1], ","))
            @info "case $(k)" cs[k]
            d = simulate(cs[k]; force="--force" in ARGS)
            @printf("case %d done: N = %d, wall %.0f s, maxrss %.2f GB\n", k, d["N"], d["wall_time_s"], d["maxrss_final_gb"])
        end
        return
    end
    i = findfirst(==("--smoke"), ARGS)
    if i !== nothing
        for k in parse.(Int, split(ARGS[i + 1], ","))
            p = smoke(cs[k])
            d = simulate(p; force="--force" in ARGS, sub="smoke")
            r = row(p, d)
            @printf("smoke %d (L_y=%g H=%g %s): ω_mn=%.4f Δ=%.4f √μ_mode=%.4f ζ_rad=%.3f MAC=%.3f inside=%s N=%d wall=%.0f s rss=%.2f GB\n",
                    k, p.Ly, p.H, p.target, r.omega, r.split, r.sqrt_mu_mode, r.zeta_rad, r.mac_target, r.inside, r.N,
                    r.wall_time_s, d["maxrss_final_gb"])
            println("   checks: mass ", r.mass_check, " heave ", r.heave_check, " lin ", r.linearity, " Asym ", r.A_sym,
                    " swap ", r.swap, " res ", r.residual, " forced ", r.forced_trunc, " sub ", r.split_sub / r.split - 1)
        end
        return
    end
    if "--validate2d" in ARGS
        show(stdout, MIME"text/plain"(), validate2d(; force="--force" in ARGS)); println()
        return
    end
    # post-processing of the stored records
    path = joinpath(REPO, "data", "sims", TEST_ID)
    have = [isfile(joinpath(path, record_name(p) * ".jld2")) for p in cs]
    recs = [have[k] ? simulate(cs[k]) : nothing for k in eachindex(cs)]
    csk = cs[have]; rk = recs[have]
    df = DataFrame([row(p, d) for (p, d) in zip(csk, rk)])
    rt = ratios(df)
    v2 = validate2d()
    ck = checks(df, rt, v2)
    out = joinpath(REPO, "data", "exp_pro", TEST_ID); mkpath(out)
    CSV.write(joinpath(out, "modes.csv"), df); CSV.write(joinpath(out, "ratios.csv"), rt)
    CSV.write(joinpath(out, "validation_2d.csv"), v2); CSV.write(joinpath(out, "checks.csv"), ck)
    pl = pair_L40(csk, rk); ba = buffer_absorption(csk, rk, df)
    CSV.write(joinpath(out, "pair_L40.csv"), pl); CSV.write(joinpath(out, "buffer_absorption.csv"), ba)
    println(); show(stdout, MIME"text/plain"(), pl; allrows=true); println()
    println(); show(stdout, MIME"text/plain"(), ba; allrows=true); println()
    plot_all(df, rt, csk, rk, joinpath(REPO, "plots", TEST_ID))
    show(stdout, MIME"text/plain"(), select(df, :Ly, :H, :target, :variant, :omega, :split, :sqrt_mu_mode, :mu_est,
                                            :zeta_rad, :mac_target, :proj, :forced_trunc, :N, :wall_time_s); allrows=true)
    println(); show(stdout, MIME"text/plain"(), rt); println()
    @printf("%d/%d checks pass (%d of %d cases present)\n", count(ck.passed), nrow(ck), count(have), length(cs))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-80s value = %.4g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
