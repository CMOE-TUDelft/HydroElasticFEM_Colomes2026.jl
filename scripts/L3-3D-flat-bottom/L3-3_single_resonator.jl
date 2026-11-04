#=
L3-3 — single resonator on a large 3D plate, quarter symmetry.
Pre- and post-processing of src/L3_3_single_resonator.jl (problem, mesh and
solves are documented there).

Binding configuration (frozen in row L3.3 of verification_test_matrix_2026-10-22.md
before the binding runs): set P1, ω_r = 10, m_r = 0.01 m³; p = 3, h = λ_p/5 frozen, run at
h = λ_p/4.5 (post-freeze deviation: guard footprint 18.2–18.6 GiB at λ_p/5; R_s = 22 cells = 4.89λ_p,
frame 14 cells = 3.11λ_p, L_q = 8λ_p)
(λ_p at ω = 10), nz = 8; quarter domain L_q = 8λ_p: undamped region
max(x, y) < 5λ_p, Kelvin–Voigt frame 3λ_p (ωτ = 10, ramp ξ²). One METIS
factorisation per ω (unit point force); the ζ sweep is Sherman–Morrison on
that factor.

Cases (one Julia process each, under rss_guard + run_limited, one heavy-run lock each):
   1        P, ω = ω̃ (renormalised resonance of the theory); also the smoke case of the peak memory
   2–5      P, ω̃ ∓ 0.13, ω̃ ∓ 0.26
   6        B (buffer variant, frame at 4λ_p, L_q = 7λ_p), ω̃
   7        M (mesh variant, h = λ_p/4), ω̃
   8–13     P, ω̃ ∓ 0.06, ω̃ ∓ 0.40, ω̃ ∓ 0.56
   14, 15   identity checks on a small mesh (h = λ_p/3, L_q = 2.5λ_p): quarter and full
            domain, each with the monolithic resonator (ζ = 0.0129)

Outputs (relative to the repository root):
  data/sims/L3-3/      one DrWatson record per case (not committed)
  data/exp_pro/L3-3/   frequencies.csv, zeta_sweep.csv, sensitivity.csv, checks.csv
  plots/L3-3/          single_resonator.{pdf,png}

Run:
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-3_single_resonator.jl --list
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-3_single_resonator.jl --case 1
  julia --project scripts/L3-3D-flat-bottom/L3-3_single_resonator.jl            # tables, checks, figure
=#

using DrWatson
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_Colomes2026
using HydroElasticFEM_Colomes2026.References
include(joinpath(@__DIR__, "..", "..", "src", "L3_3_single_resonator.jl"))
using .L3_3

const TEST_ID = "L3-3"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const P1 = HostSpec(g=9.8, ρ=1000.0, H=10.0, m_ρ=0.01, C_ρ=0.49)
const R0 = resonator_from_frequency(0.01, 10.0)
const ΩT = round(renormalised_frequency(P1, R0; dim=3), digits=4)          # 9.9506
const ZC = critical_damping(P1, R0, ΩT; dim=3) / (2sqrt(R0.k_ρ * R0.m_ρ))   # ζ_crit at ω̃ (0.0129)
const OFFS = (0.0, -0.13, 0.13, -0.26, 0.26)
const OFFS2 = (-0.06, 0.06, -0.40, 0.40, -0.56, 0.56)

"Cases in run order (essential first)."
function cases()
    c = L3_3_Params[]
    for d in OFFS
        push!(c, L3_3_Params(ω=round(ΩT + d, digits=4)))
    end
    push!(c, L3_3_Params(ω=ΩT, r_sponge=4.0))                     # B: buffer variant
    push!(c, L3_3_Params(ω=ΩT, epw=4.0))                          # M: mesh variant
    for d in OFFS2
        push!(c, L3_3_Params(ω=round(ΩT + d, digits=4)))
    end
    for full in (false, true)                                    # identity checks, small mesh
        push!(c, L3_3_Params(ω=ΩT, epw=3.0, r_sponge=1.5, frame=1.0, full=full, monolithic=true,
                             ring_rmin=0.3, ring_rmax_margin=0.2, n_r=5, n_θ=4, n_line=11))
    end
    return c
end

variant(p::L3_3_Params) = p.epw == 3.0 ? (p.full ? "I-full" : "I-quarter") :
                          p.r_sponge == 4.0 ? "B" : p.epw == 4.0 ? "M" : "P"

record_name(p) = savename(p; accesses=VARIABLE) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    e = only(TOML.parsefile(joinpath(REPO, "Manifest.toml"))["deps"]["HydroElasticFEM"])
    return Dict("hefem_rev" => get(e, "repo-rev", "unknown"), "hefem_tree" => get(e, "git-tree-sha1", "unknown"))
end

function simulate(i::Int)
    p = cases()[i]
    path = joinpath(REPO, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO, storepatch=false) do p
        t0 = time()
        d = run_case(p)
        d["wall_s"] = time() - t0
        d["maxrss_final_gb"] = Sys.maxrss() / 1024^3
        d["mem_limit_gb"] = get(ENV, "LRH_MEM_LIMIT_GB", "unset")
        merge!(d, hefem_pin())
        d
    end
    @printf("[L3-3] case %d (%s, ω = %.4f) done: wall %.0f s, peak RSS %.2f GB, G0 = %s\n", i, variant(p), p.ω,
            data["wall_s"], data["maxrss_final_gb"], string(data["G0"]))
    return data
end

# ─── Post-processing ────────────────────────────────────────────────────────

function records()
    path = joinpath(REPO, "data", "sims", TEST_ID)
    out = Tuple{Int,L3_3_Params,Dict}[]
    for (i, p) in enumerate(cases())
        f = joinpath(path, record_name(p) * ".jld2")
        isfile(f) && push!(out, (i, p, wload(f)))
    end
    return out
end

"Trapezoid weights on [0, π/2] (the azimuthal mean over the quarter)."
trapw(n) = (w = fill(1.0, n); w[1] = w[end] = 0.5; w ./ sum(w))

"Far-field analysis of one record: fits of the azimuthal mean and of each angle."
function farfield(p, d)
    rs, θs, E = d["ring_r"], d["ring_theta"], d["ring_eta"]
    m = E * trapw(length(θs))
    fm = fit_ring(rs, m, P1, p.ω)
    fθ = [fit_ring(rs, E[:, j], P1, p.ω) for j in eachindex(θs)]
    return (mean=fm, perθ=fθ)
end

"Single-resonator quantities from G_h(0) and the far-field amplitude C⁺ (theory: G(0), iγ_r)."
function scattering(ω, G0, Cplus, ζ)
    r = resonator_from_frequency(R0.m_ρ, natural_frequency(R0); ζ=ζ)
    kp = k0(P1, ω); Dp = dD0(P1, kp, ω)
    FA = 1 / (inv_alpha(r, ω) - G0)                 # F₁/A, eq. (single_force)
    γd = gamma_d(r, ω)
    σabs = 2γd * abs2(FA) / Dp                      # P_abs / I, I = (ω/4)𝒟₀′|A|²
    σsc = 4abs2(FA * Cplus) / kp                    # ring flux of F₁G_h / I
    σext = -4 / kp * real(FA * Cplus)               # optical theorem, forward amplitude f = F₁C⁺/A
    return (; σabs, σsc, σext, γd, FA)
end

zgrid() = sort(unique(vcat([0.0, ZC, 3ZC], exp10.(range(-3, -1; length=81)))))

"Polynomial (degree `deg`) least-squares interpolant of y(ω)."
function polyfit(x, y, deg)
    x0 = sum(x) / length(x)
    V = [(xi - x0)^j for xi in x, j in 0:deg]
    c = V \ y
    return t -> sum(c[j + 1] * (t - x0)^j for j in 0:deg)
end

function analyse(recs)
    P = [(p, d) for (_, p, d) in recs if variant(p) == "P"]
    sort!(P; by=x -> x[1].ω)
    freq = NamedTuple[]
    sweep = NamedTuple[]
    for (p, d) in P
        ω = p.ω
        ff = farfield(p, d)
        Cp, Cm = ff.mean.Cplus, ff.mean.Cminus
        kp = k0(P1, ω); Dp = dD0(P1, kp, ω); γr = gamma_r(P1, ω; dim=3); G0t = green_2d_integral(P1, 0.0, ω)
        G0 = d["G0"]
        Pwork = 0.5ω * imag(G0)
        Pflux = ω * Dp * (abs2(Cp) - abs2(Cm)) / kp
        iso = maximum(abs(f.Cplus / Cp - 1) for f in ff.perθ)
        push!(freq, (ω=ω, N=d["bare_N"], ReG0_h=real(G0), ImG0_h=imag(G0), ReG0=real(G0t), gamma_r=γr,
                     err_ImG=imag(G0) / γr - 1, err_ReG=real(G0) / real(G0t) - 1,
                     Cplus_re=real(Cp), Cplus_im=imag(Cp), err_Cplus=abs(Cp / (im * γr) - 1),
                     phase_Cplus_deg=rad2deg(angle(Cp / im)), reflection=abs(Cm / Cp), k_h_over_k0=ff.mean.k / kp,
                     fit_residual=ff.mean.residual, isotropy=iso, power_balance=Pflux / Pwork - 1,
                     residual=d["bare_rel_residual"], peak_rss_gb=d["maxrss_final_gb"], wall_s=d["wall_s"],
                     factorize_s=d["bare_factorize_wall_s"], lunz=d["bare_lunz"]))
        for ζ in zgrid()
            s = scattering(ω, G0, Cp, ζ)
            t = cross_sections_3d(P1, resonator_from_frequency(R0.m_ρ, 10.0; ζ=ζ), ω)
            push!(sweep, (ω=ω, ζ=ζ, sigma_abs_k0=s.σabs * kp, sigma_sc_k0=s.σsc * kp, sigma_ext_k0=s.σext * kp,
                          optical_residual=ζ == 0 && s.σext == 0 ? 0.0 : (s.σext - s.σsc - s.σabs) / s.σext,
                          sigma_abs_k0_theory=t.σ_abs * kp, sigma_sc_k0_theory=t.σ_sc * kp,
                          gamma_d=s.γd, ImG0_h=imag(G0)))
        end
    end
    return DataFrame(freq), DataFrame(sweep), P
end

"Renormalised resonance ω̃_h and ζ_opt,h from the FE G_h(0; ω) (cubic interpolant of the FE frequencies)."
function renormalised_h(P)
    ωs = [p.ω for (p, _) in P]
    length(ωs) >= 3 || return nothing
    deg = min(3, length(ωs) - 1)
    reG = polyfit(ωs, [real(d["G0"]) for (_, d) in P], deg)
    imG = polyfit(ωs, [imag(d["G0"]) for (_, d) in P], deg)
    f(ω) = 1 / (ω^2 * R0.m_ρ) - 1 / R0.k_ρ - reG(ω)
    a, b = minimum(ωs), maximum(ωs)
    f(a) * f(b) < 0 || return nothing
    for _ in 1:100
        c = (a + b) / 2
        f(a) * f(c) <= 0 ? (b = c) : (a = c)
    end
    ωh = (a + b) / 2
    γ = imG(ωh)                     # imG interpolates Im G_h(0; ω) (real values)
    # damping with γ_d = Im G_h(ω̃_h) (smaller root, as `critical_damping`)
    disc = 1 - 4γ^2 * R0.k_ρ^2
    c = (1 - sqrt(disc)) / (2γ * ωh)
    return (ω=ωh, ImG=γ, ReG=reG(ωh), ζopt=c / (2sqrt(R0.k_ρ * R0.m_ρ)), reG=reG, imG=imG)
end

function checks(recs, ft, sw, P, rh)
    ck = NamedTuple[]
    add!(name, kind, value, tol, passed=abs(value) <= tol) =
        push!(ck, (name=name, kind=kind, value=Float64(value), tolerance=Float64(tol), passed=passed))
    # I: every solve
    for (i, p, d) in recs
        v = variant(p)
        add!(@sprintf("residual, case %d (%s, ω = %.4f)", i, v, p.ω), "I", d["bare_rel_residual"], 1e-8)
        add!(@sprintf("METIS ordering used, case %d", i), "I", d["bare_ordering"] == "metis" ? 0 : 1, 0)
        haskey(d, "mono_rel_residual") && add!(@sprintf("residual monolithic, case %d", i), "I", d["mono_rel_residual"], 1e-8)
        if v in ("P", "B", "M")
            add!(@sprintf("peak RSS < 20 GiB, case %d (%s, ω = %.4f)", i, v, p.ω), "I", d["maxrss_final_gb"], 20.0)
        end
    end
    # I: identities on the small mesh
    q = [(p, d) for (_, p, d) in recs if variant(p) == "I-quarter"]
    f = [(p, d) for (_, p, d) in recs if variant(p) == "I-full"]
    if !isempty(q) && !isempty(f)
        dq, df = q[1][2], f[1][2]
        add!("quarter (Nitsche symmetry) vs full domain: |ΔG_h(0)|/|G_h(0)|", "I", abs(dq["G0"] / df["G0"] - 1), 1e-8)
        mq = dq["mirror_eta"][1]
        add!("quarter vs full: max |Δη|/|η| at 3 interior points", "I", maximum(abs.(mq ./ df["mirror_eta"][1] .- 1)), 1e-8)
        add!("full domain: mirror symmetry of η (4 quadrants)", "I",
             maximum(maximum(abs.(m ./ df["mirror_eta"][1] .- 1)) for m in df["mirror_eta"]), 1e-8)
        for (lbl, d) in (("quarter", dq), ("full", df))
            add!("Sherman–Morrison vs monolithic resonator η(0), $lbl", "I", abs(d["mono_eta0"] / d["mono_eta0_SM"] - 1), 1e-8)
        end
        add!("monolithic: quarter vs full η(0)", "I", abs(dq["mono_eta0"] / df["mono_eta0"] - 1), 1e-8)
    end
    tol = 0.02
    for r in eachrow(ft)
        w = @sprintf("ω = %.4f", r.ω)
        add!("|Im G_h(0)/γ_r − 1|, $w", "D", r.err_ImG, tol)
        add!("|Re G_h(0)/Re G(0) − 1|, $w", "D", r.err_ReG, 0.05)
        add!("|C⁺_h/(iγ_r) − 1| (far-field amplitude), $w", "D", r.err_Cplus, tol)
        add!("power balance |P_flux/P_work − 1| (radiation optical theorem), $w", "D", r.power_balance, tol)
        add!("frame reflection |C⁻/C⁺|, $w", "D", r.reflection, tol)
        add!("isotropy max_θ |C⁺(θ)/⟨C⁺⟩ − 1|, $w", "D", r.isotropy, tol)
        add!("|k_h/k₀ − 1|, $w", "D", r.k_h_over_k0 - 1, 1e-2)
    end
    # bound and optical theorem over the ζ sweep
    if nrow(sw) > 0
        add!("max over (ω, ζ) of σ_abs k₀ ≤ 1 + 0.02 (bound eq:absorption_bound)", "D", maximum(sw.sigma_abs_k0) - 1, tol,
             maximum(sw.sigma_abs_k0) <= 1 + tol)
        for r in eachrow(sw)
            (r.ζ == ZC || r.ζ == 3ZC || r.ζ == 0.0) || continue
            add!(@sprintf("optical theorem |σ_ext − σ_sc − σ_abs|/σ_ext, ω = %.4f, ζ = %.4g", r.ω, r.ζ), "D",
                 r.optical_residual, tol)
            r.ζ == 0.0 && continue
            add!(@sprintf("σ_abs k₀ FE vs theory, ω = %.4f, ζ = %.4g", r.ω, r.ζ), "D",
                 r.sigma_abs_k0 - r.sigma_abs_k0_theory, tol)
        end
        for r in eachrow(sw)
            r.ζ == 0.0 || continue
            add!(@sprintf("σ_sc k₀ FE vs theory, ζ = 0, ω = %.4f", r.ω), "D", r.sigma_sc_k0 / r.sigma_sc_k0_theory - 1, tol)
        end
    end
    if rh !== nothing
        add!("ω̃_h/ω̃ − 1 (renormalised resonance, eq:renorm_single)", "D", rh.ω / renormalised_frequency(P1, R0; dim=3) - 1, 1e-3)
        add!("ζ_opt,h/ζ_crit − 1 at ω̃_h (critical coupling, eq:critical_coupling)", "D", rh.ζopt / ZC - 1, tol)
        # bound attained at ω̃_h, γ_d = Im G_h: σ_abs k₀ = 2γ_d/(𝒟₀′ (2γ_d)²) k₀ = k₀/(2𝒟₀′ Im G_h) = γ_r/Im G_h
        att = gamma_r(P1, rh.ω; dim=3) / rh.ImG
        add!("bound attained at critical coupling: σ_abs k₀ at (ω̃_h, ζ_opt,h) − 1", "D", att - 1, tol)
        # equipartition at critical coupling (P_abs = P_sc), ring flux at the FE frequency nearest ω̃_h
        j = argmin(abs.(ft.ω .- rh.ω))
        r = ft[j, :]
        add!(@sprintf("critical coupling P_abs/P_sc − 1 (ring flux), ω = %.4f", r.ω), "D", -(r.power_balance / (1 + r.power_balance)), tol)
    end
    # sensitivity at ω̃
    ref = [(p, d) for (_, p, d) in recs if variant(p) == "P" && p.ω == ΩT]
    if !isempty(ref)
        G0 = ref[1][2]["G0"]
        for v in ("B", "M")
            s = [(p, d) for (_, p, d) in recs if variant(p) == v]
            isempty(s) && continue
            Gv = s[1][2]["G0"]
            lbl = v == "B" ? "buffer (frame at 4λ_p vs 4.89λ_p)" : "mesh (h = λ_p/4 vs λ_p/4.5)"
            add!("$lbl: |ΔG_h(0)|/|G_h(0)| at ω̃", "D", abs(Gv - G0) / abs(G0), v == "B" ? 5e-3 : tol)
            add!("$lbl: |ΔIm G_h(0)|/γ_r at ω̃", "D", abs(imag(Gv - G0)) / gamma_r(P1, ΩT; dim=3), v == "B" ? 5e-3 : tol)
        end
    end
    return DataFrame(ck)
end

function sensitivity_table(recs)
    rows = NamedTuple[]
    for (i, p, d) in recs
        variant(p) in ("P", "B", "M") && p.ω == ΩT || continue
        ff = farfield(p, d)
        push!(rows, (case=i, variant=variant(p), epw=p.epw, r_sponge=p.r_sponge, Lq_over_λp=d["Lq"] / d["λ"], N=d["bare_N"],
                     ReG0_h=real(d["G0"]), ImG0_h=imag(d["G0"]), err_ImG=imag(d["G0"]) / gamma_r(P1, p.ω; dim=3) - 1,
                     reflection=abs(ff.mean.Cminus / ff.mean.Cplus), peak_rss_gb=d["maxrss_final_gb"], wall_s=d["wall_s"]))
    end
    return DataFrame(rows)
end

"""
Frozen-mesh (h = λ_p/5) records of the P frequencies run before the post-freeze coarsening
(cases 1–4), compared with the run mesh h = λ_p/4.5 (informational, not a frozen check).
"""
function fine_mesh_table(recs)
    rows = NamedTuple[]
    path = joinpath(REPO, "data", "sims", TEST_ID)
    for (i, p, d) in recs
        variant(p) == "P" || continue
        q = L3_3_Params(p; epw=5.0)
        f = joinpath(path, record_name(q) * ".jld2")
        isfile(f) || continue
        e = wload(f)
        γr = gamma_r(P1, p.ω; dim=3)
        fe, fd = farfield(q, e), farfield(p, d)
        Ce, Cd = fe.mean.Cplus, fd.mean.Cplus
        iso(ff) = maximum(abs(f.Cplus / ff.mean.Cplus - 1) for f in ff.perθ)
        push!(rows, (case=i, ω=p.ω, N_5=e["bare_N"], N_45=d["bare_N"], ImG_5_err=imag(e["G0"]) / γr - 1,
                     ImG_45_err=imag(d["G0"]) / γr - 1, dG_rel=abs(d["G0"] - e["G0"]) / abs(e["G0"]),
                     dImG_over_γr=abs(imag(d["G0"] - e["G0"])) / γr, dCplus_rel=abs(Cd / Ce - 1), isotropy_5=iso(fe), isotropy_45=iso(fd),
                     peak_rss_5=e["maxrss_final_gb"], wall_5=e["wall_s"]))
    end
    return DataFrame(rows)
end

function figure(ft, sw, P, rh)
    dir = joinpath(REPO, "plots", TEST_ID); mkpath(dir)
    with_theme(Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))) do
        fig = Figure(size=(1180, 400))
        # (a) σ_abs k₀ against ω at ζ_crit and 3ζ_crit; σ_sc k₀/4 at ζ = 0
        ax1 = Axis(fig[1, 1], xlabel="ω (rad/s)", ylabel="σ k₀", title="(a) cross-sections against ω")
        ωf = range(minimum(ft.ω), maximum(ft.ω); length=301)
        for (ζ, col, lbl) in ((ZC, :black, "σ_abs, ζ_crit"), (3ZC, :steelblue, "σ_abs, 3ζ_crit"))
            t = [cross_sections_3d(P1, resonator_from_frequency(R0.m_ρ, 10.0; ζ=ζ), ω).σ_abs * k0(P1, ω) for ω in ωf]
            lines!(ax1, ωf, t; color=col, label=lbl * " (theory)")
            s = filter(r -> r.ζ == ζ, sw)
            scatter!(ax1, s.ω, s.sigma_abs_k0; color=col, marker=:circle, label=lbl * " (FE)")
        end
        t0 = [cross_sections_3d(P1, R0, ω).σ_sc * k0(P1, ω) / 4 for ω in ωf]
        lines!(ax1, ωf, t0; color=:tomato, linestyle=:dash, label="σ_sc/4, ζ = 0 (theory)")
        s0 = filter(r -> r.ζ == 0.0, sw)
        scatter!(ax1, s0.ω, s0.sigma_sc_k0 ./ 4; color=:tomato, marker=:utriangle, label="σ_sc/4, ζ = 0 (FE)")
        hlines!(ax1, [1.0]; color=:gray, linestyle=:dot)
        axislegend(ax1; position=:rt, framevisible=false, labelsize=8)
        # (b) σ_abs k₀ against ζ at ω̃ (FE frequency) and the bound
        ax2 = Axis(fig[1, 2], xscale=log10, xlabel="ζ", ylabel="k₀σ", title=@sprintf("(b) ζ sweep at ω = %.4f", ΩT))
        s = filter(r -> r.ω == ΩT && r.ζ > 0, sw)
        if nrow(s) > 0
            lines!(ax2, s.ζ, s.sigma_abs_k0_theory; color=:black, label="σ_abs (theory)")
            scatter!(ax2, s.ζ, s.sigma_abs_k0; color=:black, markersize=5, label="σ_abs (FE, Sherman–Morrison)")
            lines!(ax2, s.ζ, s.sigma_ext_k0 .- s.sigma_sc_k0; color=:orange, linestyle=:dash, label="σ_ext − σ_sc (FE)")
            hlines!(ax2, [1.0]; color=:gray, linestyle=:dot, label="bound 1/k₀")
            vlines!(ax2, [ZC]; color=:gray, linestyle=:dash, label="ζ_crit")
            axislegend(ax2; position=:lt, framevisible=false, labelsize=8)
        end
        # (c) plate response along the x axis at ω̃ and the far-field amplitude against θ
        ax3 = Axis(fig[1, 3], xlabel="x / λ_p", ylabel="|G_h(x, 0)| / γ_r", yscale=log10,
                   title="(c) point-load field along y = 0, ω̃")
        ref = [(p, d) for (p, d) in P if p.ω == ΩT]
        if !isempty(ref)
            p, d = ref[1]
            λ = d["λ"]; γr = gamma_r(P1, p.ω; dim=3)
            x = d["line_x"]
            lines!(ax3, x ./ λ, abs.(d["line_eta"]) ./ γr; color=:black, label="FE, y = 0")
            lines!(ax3, x ./ λ, abs.(d["diag_eta"]) ./ γr; color=:steelblue, label="FE, diagonal")
            xr = x[x .> 0.2λ]
            lines!(ax3, xr ./ λ, abs.(green_2d.(Ref(P1), xr, p.ω)) ./ γr; color=:tomato, linestyle=:dash,
                   label="infinite plate")
            vspan!(ax3, d["Rs"] / λ, d["Lq"] / λ; color=(:gray, 0.15))
            axislegend(ax3; position=:lb, framevisible=false, labelsize=8)
        end
        save(joinpath(dir, "single_resonator.pdf"), fig)
        save(joinpath(dir, "single_resonator.png"), fig; px_per_unit=300 / 96)
    end
end

function post()
    recs = records()
    ft, sw, P = analyse(recs)
    rh = renormalised_h(P)
    ck = checks(recs, ft, sw, P, rh)
    st = sensitivity_table(recs)
    out = joinpath(REPO, "data", "exp_pro", TEST_ID); mkpath(out)
    CSV.write(joinpath(out, "frequencies.csv"), ft)
    CSV.write(joinpath(out, "zeta_sweep.csv"), sw)
    CSV.write(joinpath(out, "sensitivity.csv"), st)
    CSV.write(joinpath(out, "checks.csv"), ck)
    fm = fine_mesh_table(recs)
    nrow(fm) > 0 && CSV.write(joinpath(out, "fine_mesh_reference.csv"), fm)
    if rh !== nothing
        CSV.write(joinpath(out, "renormalised.csv"),
                  DataFrame(omega_h=[rh.ω], omega_theory=[renormalised_frequency(P1, R0; dim=3)], ImG_h=[rh.ImG],
                            ReG_h=[rh.ReG], zeta_opt_h=[rh.ζopt], zeta_crit=[ZC]))
    end
    if nrow(ft) > 0
        @eval Main using CairoMakie
        Base.invokelatest(figure, ft, sw, P, rh)
    end
    show(stdout, MIME("text/plain"), ft; allcols=true); println()
    show(stdout, MIME("text/plain"), st; allcols=true); println()
    nrow(fm) > 0 && (show(stdout, MIME("text/plain"), fm; allcols=true); println())
    rh === nothing || @printf("ω̃_h = %.5f (theory %.5f), ζ_opt,h = %.5f (ζ_crit %.5f), Im G_h = %.6g\n", rh.ω,
                              renormalised_frequency(P1, R0; dim=3), rh.ζopt, ZC, rh.ImG)
    for r in eachrow(ck)
        r.passed || @printf("FAIL  %-90s %-2s %.4g (tol %.3g)\n", r.name, r.kind, r.value, r.tolerance)
    end
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--list" in ARGS
        for (i, p) in enumerate(cases())
            println(i, "  ", variant(p), "  ω = ", p.ω, "  ", size_counts(p))
        end
    elseif (i = findfirst(==("--case"), ARGS)) !== nothing
        simulate(parse(Int, ARGS[i + 1]))
    else
        post()
    end
end
