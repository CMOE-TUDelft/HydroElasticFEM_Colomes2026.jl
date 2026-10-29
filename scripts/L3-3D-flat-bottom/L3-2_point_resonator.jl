#=
L3-2 — point resonator on the 3D plate (port of the legacy VT3D_B, set P3).
Pre- and post-processing of src/L3_2_point_resonator.jl (problem and solves
are documented there).

Cases (one Julia process per (C_ρ, level, part), each with one assembly and
one factorisation, inside run_limited at -O0 (compile memory), METIS solver;
parts: bare (S1, S2), res_node (S3 at x_r), res_edge, res_interior, bare09
(S1_09, S2_09), S4, PG (L2); 32 processes in all):
   1–3  P3 (C_ρ = 0.05g), levels L1, L2, L3: S1, S2 and S3 at the node, edge and
        interior points, S1_09, S2_09, S4, and PG at L2;
   4–6  C_ρ = 0.005g, L1–L3: S1, S2, S3 at the node (W ∝ λ_p);
   7–9  C_ρ = 5e-4g, L1–L3: idem;
   10   D-W static Winkler limit (P3 plate alone), h_min = l_c/{8, 16, 32}.

Checks (frozen in the L3.2 row of verification_test_matrix_2026-10-22.md
before the binding run): identities (residual, incidence in the right-hand
side only, load sign, B-P1, B-P2 at every attachment point), convergence of
the driving-point compliance G_h and q_h (B-P3), the near field (B-P4), the
r² ln r coefficient c = f/(8πC_ρ) (B-P5), the attachment location, the
penalty control (P-γ), the C_ρ scaling of |G_h| and the Hertz/Wyman limit
(B-W1, B-W2).

Outputs (relative to the repository root):
  data/sims/L3-2/      one DrWatson record per case
  data/exp_pro/L3-2/   points.csv, convergence.csv, fits.csv, winkler.csv, checks.csv
  plots/L3-2/          point_resonator.{pdf,png}

Run:
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl --list
  julia --project -O0 --heap-size-hint=2500M scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl --case 1
  julia --project scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl            # tables, checks, figure
=#

using DrWatson                 # CairoMakie is loaded in `post` only (≈ 0.7 GB less per compute process)
using DataFrames
using CSV
using Printf
using TOML
using LinearAlgebra: cond
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L3_2

const TEST_ID = "L3-2"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
# Far field: `--graded` = the first run (P3 legacy grading, far cells change with the level; 4 GB cap);
# default = the refinement L3-2r of 2026-10-28 (nested fixed far field Δx = λ_p/16, Δy = W/14, levels
# h_min = h_far/{1, 2, 4, 8}; 3D batch rules, 20 GB cap).
const FAR = "--graded" in ARGS ? :graded : :nested
const CAP_GB = FAR == :graded ? 4.0 : 20.0
const LEVELS = FAR == :graded ? ("L1", "L2", "L3") : ("L1", "L2", "L3", "L4")
const LADDER = ("L1", "L2", "L3")                  # the frozen ladder of every rate check
const CS = (0.05, 0.005, 5e-4)
const W_LEVELS = (8.0, 16.0, 32.0)
const OUT_SUB = FAR == :graded ? "graded_far_field" : ""

params(l, c) = FAR == :graded ? L3_2_Params(level=l, C_over_g=c) :
               L3_2_Params(level=l, C_over_g=c, far_mesh=:nested, dx_plate_over_λp=16.0, ny_base=14)

parts(c, l) = c == 0.05 ? (l == "L2" ? (:bare, :res_node, :res_edge, :res_interior, :bare09, :S4, :PG) :
                                         (:bare, :res_node, :res_edge, :res_interior, :bare09, :S4)) : (:bare, :res_node)

"Cases (params, mode): one process per (C_ρ, level, part) and one for D-W."
cases() = [[(params(l, c), m) for c in CS for l in LEVELS for m in parts(c, l)];
           [(L3_2_Params(), :winkler)]]

record_name(p, mode) = mode == :winkler ? "winkler_fixed=" * fixed_hash(p) :
                       savename(p; accesses=VARIABLE) * "_part=$(mode)_fixed=" * fixed_hash(p)

function hefem_pin()
    e = only(TOML.parsefile(joinpath(REPO, "Manifest.toml"))["deps"]["HydroElasticFEM"])
    return Dict("hefem_rev" => get(e, "repo-rev", "unknown"), "hefem_tree" => get(e, "git-tree-sha1", "unknown"))
end

function simulate(i::Int)
    p, mode = cases()[i]
    path = joinpath(REPO, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=q -> record_name(q, mode), tag=true, gitpath=REPO,
                              storepatch=false) do p
        t0 = time()
        d = if mode == :winkler
            Dict{String,Any}("levels" => [winkler_level(p, n) for n in W_LEVELS])
        else
            run_case(p; part=mode)
        end
        d["wall_s"] = time() - t0
        d["maxrss_final_gb"] = Sys.maxrss() / 1024^3
        d["mem_limit_gb"] = get(ENV, "LRH_MEM_LIMIT_GB", "unset")
        merge!(d, hefem_pin())
        d
    end
    @printf("[L3-2] case %d done: wall %.0f s, peak RSS %.2f GB\n", i, data["wall_s"], data["maxrss_final_gb"])
    return data
end

# ─── Post-processing ────────────────────────────────────────────────────────

"Records (params, mode, record), parts of one (C_ρ, level) merged into mode :fe (only when all parts exist)."
function records()
    path = joinpath(REPO, "data", "sims", TEST_ID)
    out = Tuple{L3_2_Params,Symbol,Dict}[]
    merged = Dict{Tuple{Float64,String},Vector{Dict}}()
    for (p, mode) in cases()
        f = joinpath(path, record_name(p, mode) * ".jld2")
        isfile(f) || continue
        d = wload(f)
        if mode == :winkler
            push!(out, (p, mode, d))
        else
            push!(get!(merged, (p.C_over_g, p.level), Dict[]), d)
        end
    end
    for c in CS, l in LEVELS
        ds = get(merged, (c, l), Dict[])
        length(ds) == length(parts(c, l)) || continue
        d = merge(ds...)
        d["maxrss_final_gb"] = maximum(x["maxrss_final_gb"] for x in ds)
        d["wall_s"] = sum(x["wall_s"] for x in ds)
        push!(out, (params(l, c), :fe, d))
    end
    return out
end

cauchy(v1, v2, v3) = (d12=abs(v1 - v2), d23=abs(v2 - v3), p=log2(abs(v1 - v2) / abs(v2 - v3)), rel23=abs(v2 - v3) / abs(v3))

"LSQ fit ⟨η⟩(r) = a0 + c r² ln r + a2 r² + a4 r⁴ (complex, a0 free)."
function fit_rlnr(r, y)
    M = hcat(ones(length(r)), r .^ 2 .* log.(r), r .^ 2, r .^ 4)
    coef = M \ y
    return (a0=coef[1], c=coef[2], a2=coef[3], a4=coef[4], cond=cond(M))
end

l2w(w, a) = sqrt(sum(w .* abs2.(a)))

G(d, s=:node) = d["bare_S2_eta_δ_$(s)"] / 1.0           # f = 1
q(d, s=:node) = d["S3_$(s)_q"]

"Sherman–Morrison prediction of q at ω_r (legacy B-P2)."
q_pred(d, s=:node) = -d["bare_S1_eta_δ_$(s)"] / (d["K"] / d["rho"] * G(d, s))

function eta09_pred(d)
    ω, C, K, M, ρ = d["S4_omega"], d["S4_C"], d["K"], d["M"], d["rho"]
    Zr = (K - im * ω * C) * ω^2 * (M / ρ) / (K - im * ω * C - ω^2 * M)
    return d["bare09_S1_eta_δ_node"] / (1 - d["bare09_S2_eta_δ_node"] * Zr)
end

function points_table(recs)
    rows = NamedTuple[]
    for (p, mode, d) in recs
        mode == :fe || continue
        full = p.C_over_g == 0.05
        g(k) = get(d, k, NaN)
        push!(rows, (C_over_g=p.C_over_g, level=p.level, hmin=d["hmin"], lambda_p=d["lambda_p"], nx=d["nx"], ny=d["ny"],
                     N=d["bare_N"], N_res=d["S3_node_N"], lunz=d["bare_lunz"], umf_peak_gb=d["bare_umf_peak_gb"],
                     peak_rss_gb=d["maxrss_final_gb"], wall_s=d["wall_s"],
                     factorize_s=d["bare_factorize_wall_s"], assembly_s=d["bare_assembly_wall_s"],
                     re_G=real(G(d)), im_G=imag(G(d)), abs_G=abs(G(d)),
                     abs_G_edge=full ? abs(G(d, :edge)) : NaN, abs_G_interior=full ? abs(G(d, :interior)) : NaN,
                     re_q=real(q(d)), im_q=imag(q(d)), abs_q=abs(q(d)),
                     abs_q_edge=full ? abs(q(d, :edge)) : NaN, abs_q_interior=full ? abs(q(d, :interior)) : NaN,
                     abs_eta_S1=abs(d["bare_S1_eta_δ_node"]), abs_eta_S3=abs(d["S3_node_eta"]),
                     abs_G_inf=abs(d["G_inf"]), re_G_PG=real(g("PG_S2_eta_δ_node")), im_G_PG=imag(g("PG_S2_eta_δ_node"))))
    end
    return DataFrame(rows)
end

"Acceptance checks (frozen rows of the L3.2 matrix entry)."
function checks(recs)
    ck = NamedTuple[]
    add!(name, kind, value, tol, passed) = push!(ck, (name=name, kind=kind, value=Float64(value), tolerance=Float64(tol),
                                                       passed=passed))
    byC = Dict{Float64,Dict{String,Dict}}()
    for (p, mode, d) in recs
        mode == :fe || continue
        get!(byC, p.C_over_g, Dict{String,Dict}())[p.level] = d
        full = p.C_over_g == 0.05
        tag = @sprintf("C=%gg, %s", p.C_over_g, p.level)
        pts = full ? (:node, :edge, :interior) : (:node,)
        res = maximum(filter(isfinite, [d["bare_rel_residual"]; [d["S3_$(s)_rel_residual"] for s in pts];
                                        full ? [d["bare09_rel_residual"], d["S4_rel_residual"]] : Float64[];
                                        haskey(d, "PG_rel_residual") ? [d["PG_rel_residual"]] : Float64[]]))
        add!("residual, max over solves, $tag", "I", res, 1e-8, res <= 1e-8)
        if haskey(d, "bare_incidence_matrix_diff")   # checked at L1
            add!("incident wave in the RHS only: |A(η₀) − A(0)|₁/|A|₁, $tag", "I", d["bare_incidence_matrix_diff"], 1e-14,
                 d["bare_incidence_matrix_diff"] <= 1e-14)
            add!("incident wave in the RHS only: |b(η₀ = 0)|, $tag", "I", d["bare_incidence_free_rhs_norm"], 1e-14,
                 d["bare_incidence_free_rhs_norm"] <= 1e-14)
        end
        e = abs(d["bare_load_sum_eta"] - 1) + d["bare_load_sum_abs_other"]
        add!("load sign: |Σ b_η − f| + Σ|b_other|, $tag", "I", e, 1e-10, e <= 1e-10)
        add!("load sign: |A_ηη·1/((g − ω²m_ρ)|Γη|) − 1|, $tag", "I", d["bare_A_etaeta_one_rel"], 1e-8,
             d["bare_A_etaeta_one_rel"] <= 1e-8 && d["bare_A_etaeta_one_re"] > 0)
        add!("METIS ordering used by UMFPACK, $tag", "I", Float64(d["bare_ordering"] == "metis"), 1.0,
             d["bare_ordering"] == "metis")
        add!("peak RSS < $(Int(CAP_GB)) GiB, max over the processes of the case (GiB), $tag", "R", d["maxrss_final_gb"], CAP_GB, d["maxrss_final_gb"] < CAP_GB)
        for s in pts
            b1 = abs(d["S3_$(s)_eta"]) / abs(d["bare_S1_eta_δ_$(s)"])
            add!("B-P1 |η_S3(x)|/|η_S1(x)| at $(s), $tag", "I", b1, 1e-8, b1 <= 1e-8)
            b2 = abs(q(d, s) - q_pred(d, s)) / abs(q(d, s))
            add!("B-P2 Sherman–Morrison q at ω_r, $(s), $tag", "I", b2, 1e-8, b2 <= 1e-8)
        end
        if full
            b2 = abs(d["S4_eta"] - eta09_pred(d)) / abs(d["S4_eta"])
            add!("B-P2 Sherman–Morrison η(x_r) at 0.9ω_r, ζ = 0.02, $tag", "I", b2, 1e-8, b2 <= 1e-8)
        end
    end
    fits = NamedTuple[]
    for c in CS
        haskey(byC, c) || continue
        L = byC[c]
        tag = @sprintf("C=%gg", c)
        if !haskey(L, "L3") && haskey(L, "L1") && haskey(L, "L2")   # L3 not run (over the 4 GiB cap)
            add!("L3 not run: the P3 L3 processes peak at 4.1–4.8 GiB, $tag", "R", NaN, NaN, true)
            for (lbl, f) in [("G_h(x_r, x_r)", d -> G(d)), ("|q_h|", d -> abs(q(d)))]
                r = abs(f(L["L2"]) - f(L["L1"])) / abs(f(L["L2"]))
                add!("$lbl |L2 − L1|/|L2| (reported), $tag", "R", r, Inf, true)
            end
            for (lbl, rk, mk) in [("S3", "S3_polar_r", "S3_polar_m"), ("S2", "S2_polar_r", "S2_polar_m")]
                isempty(L["L2"][rk]) && continue
                ft = fit_rlnr(L["L2"][rk], L["L2"][mk])
                fhl = lbl == "S3" ? L["L2"]["S3_node_f"] : 1.0 + 0im
                ratio = ft.c * 8π * L["L2"]["C_rho"] / fhl
                push!(fits, (C_over_g=c, level="L2", solve=lbl, re_ratio=real(ratio), im_ratio=imag(ratio),
                             err=abs(ratio - 1), cond=ft.cond, re_c=real(ft.c), im_c=imag(ft.c)))
                add!("B-P5 |c_h·8πC_ρ/f_h − 1| ($(lbl)) at L2 (reported, window ratio 2), $tag", "R", abs(ratio - 1), Inf, true)
            end
            continue
        end
        all(haskey(L, l) for l in LADDER) || continue
        full = c == 0.05
        trips = haskey(L, "L4") ? [("L1", "L2", "L3"), ("L2", "L3", "L4")] : [("L1", "L2", "L3")]
        for (a, b, f3) in trips
            tl = f3 == "L3" ? "" : " [$(a)–$(f3)]"
            for (lbl, f) in [("G_h(x_r, x_r)", d -> G(d)), ("|q_h|", d -> abs(q(d)))]
                cc = cauchy(f(L[a]), f(L[b]), f(L[f3]))
                add!("B-P3 $lbl rate p_obs$tl, $tag", "D", cc.p, 1.0, cc.p >= 1.0)
                add!("B-P3 $lbl d₂₃/|finest|$tl, $tag", "D", cc.rel23, 0.02, cc.rel23 <= 0.02)
            end
            full || continue
            for s in (:edge, :interior)
                cc = cauchy(G(L[a], s), G(L[b], s), G(L[f3], s))
                add!("attachment: G_h at $(s), rate p_obs$tl, $tag", "D", cc.p, 1.0, cc.p >= 1.0)
                add!("attachment: G_h at $(s), d₂₃/|finest|$tl, $tag", "D", cc.rel23, 0.02, cc.rel23 <= 0.02)
            end
        end
        if full
            for s in (:edge, :interior)
                for l in LEVELS
                    r = abs(G(L[l], s) / G(L[l]) - 1)
                    l == "L3" ? add!("attachment: |G_h($(s))/G_h(node) − 1| at L3, $tag", "D", r, 0.02, r <= 0.02) :
                                add!("attachment: |G_h($(s))/G_h(node) − 1| at $l (reported), $tag", "R", r, Inf, true)
                end
            end
            pg = abs(L["L2"]["PG_S2_eta_δ_node"] - G(L["L2"])) / abs(G(L["L2"]))
            add!("P-γ |G_h(γ h_min/h_far) − G_h|/|G_h| at L2, $tag", "D", pg, 0.02, pg <= 0.02)
        end
        # B-P4: S3 near field on the fixed L3 reference set
        w = L["L3"]["ref_weights"]
        nf(l, k) = L[l]["S3_nf_$(k)"]
        nL2 = l2w(w, nf("L3", "eta")); nH1 = sqrt(l2w(w, nf("L3", "gx"))^2 + l2w(w, nf("L3", "gy"))^2)
        dl2(a, b) = l2w(w, nf(a, "eta") .- nf(b, "eta")) / nL2
        dh1(a, b) = sqrt(l2w(w, nf(a, "gx") .- nf(b, "gx"))^2 + l2w(w, nf(a, "gy") .- nf(b, "gy"))^2) / nH1
        for (a, b, f3) in trips
            tl = f3 == "L3" ? "" : " [$(a)–$(f3)]"
            pL2 = log2(dl2(a, b) / dl2(b, f3)); pH1 = log2(dh1(a, b) / dh1(b, f3))
            add!("B-P4 S3 near field L²(B_R) rate p_obs$tl, $tag", "D", pL2, 1.0, pL2 >= 1.0)
            add!("B-P4 S3 near field L²(B_R) d₂₃$tl, $tag", "D", dl2(b, f3), 0.02, dl2(b, f3) <= 0.02)
            add!("B-P4 S3 near field H¹(B_R) rate p_obs$tl, $tag", "D", pH1, 0.5, pH1 >= 0.5)
            add!("B-P4 S3 near field H¹(B_R) d₂₃$tl, $tag", "D", dh1(b, f3), 0.05, dh1(b, f3) <= 0.05)
        end
        # B-P5: r² ln r coefficient at L3 (S3 with f_h from the resonator, S2 with f)
        Cρ = L["L3"]["C_rho"]
        for (lbl, rk, mk, fh) in [("S3", "S3_polar_r", "S3_polar_m", L["L3"]["S3_node_f"]),
                                  ("S2", "S2_polar_r", "S2_polar_m", 1.0 + 0im)]
            for l in ("L2", "L3")
                isempty(L[l][rk]) && continue
                ft = fit_rlnr(L[l][rk], L[l][mk])
                fhl = lbl == "S3" ? L[l]["S3_node_f"] : fh
                ratio = ft.c * 8π * Cρ / fhl
                push!(fits, (C_over_g=c, level=l, solve=lbl, re_ratio=real(ratio), im_ratio=imag(ratio),
                             err=abs(ratio - 1), cond=ft.cond, re_c=real(ft.c), im_c=imag(ft.c)))
                l == "L3" && add!("B-P5 |c_h·8πC_ρ/f_h − 1| ($(lbl), free a₀) at L3, $tag", "A", abs(ratio - 1), 0.10,
                                  abs(ratio - 1) <= 0.10)
            end
        end
    end
    # C_ρ scaling of |G_h| at L3 against the infinite plate on the fluid
    # (L2 when the variant L3 cases are not run)
    lvC = all(haskey(byC, c) && haskey(byC[c], "L3") for c in CS) ? "L3" : "L2"
    if all(haskey(byC, c) && haskey(byC[c], lvC) for c in CS)
        lc = log.(collect(CS)); lg = [log(abs(G(byC[c][lvC]))) for c in CS]; li = [log(abs(byC[c][lvC]["G_inf"])) for c in CS]
        sl(y) = (X = hcat(ones(3), lc); (X \ y)[2])
        s_h, s_inf = sl(lg), sl(li)
        add!("C_ρ scaling: slope d ln|G_h|/d ln C_ρ at $lvC vs infinite plate ($(round(s_inf; digits=3)))", "A",
             s_h - s_inf, 0.15, abs(s_h - s_inf) <= 0.15)
        add!("C_ρ scaling: slope d ln|G_h|/d ln C_ρ at $lvC (reported; Hertz −1/2)", "R", s_h, Inf, true)
    end
    # D-W
    wk = [d for (p, mode, d) in recs if mode == :winkler]
    wrows = NamedTuple[]
    if !isempty(wk)
        lv = wk[1]["levels"]
        w0x = lv[end]["w0_exact"]
        for v in lv
            ft = fit_rlnr(v["polar_r"], ComplexF64.(v["polar_m"]))
            push!(wrows, (n_lc=v["n_lc"], hmin=v["hmin"], ndofs=v["ndofs"], w0=v["w0"], w0_exact=w0x,
                          err_w0=v["w0"] / w0x - 1, c_h=real(ft.c), c_pred=v["c_pred"], err_c=real(ft.c) / v["c_pred"] - 1,
                          rel_residual=v["rel_residual"]))
            e = abs(v["load_sum"] - 1)
            add!(@sprintf("D-W load sign |Σb − 1|, h_min = l_c/%d", v["n_lc"]), "I", e, 1e-10, e <= 1e-10)
            add!(@sprintf("D-W load sign |A·1/(g area) − 1|, h_min = l_c/%d", v["n_lc"]), "I", v["A_one_rel"], 1e-8,
                 v["A_one_rel"] <= 1e-8 && v["A_one_re"] > 0)
        end
        cw = cauchy(lv[1]["w0"], lv[2]["w0"], lv[3]["w0"])
        e1 = abs(lv[3]["w0"] / w0x - 1)
        add!("B-W1 |w_h(0)/w(0) − 1| at l_c/32", "A", e1, 0.01, e1 <= 0.01)
        add!("B-W1 rate p_obs of w_h(0)", "D", cw.p, 1.0, cw.p >= 1.0)
        e2 = wrows[end].err_c
        add!("B-W2 |c_h/c_pred − 1| (same sign) at l_c/32", "A", abs(e2), 0.10, abs(e2) <= 0.10 && wrows[end].c_h > 0)
    end
    return DataFrame(ck), DataFrame(fits), DataFrame(wrows), byC
end

function convergence_table(byC)
    rows = NamedTuple[]
    for c in CS
        haskey(byC, c) && all(haskey(byC[c], l) for l in LADDER) || continue
        L = byC[c]
        qs = [("G_node", d -> G(d)), ("abs_q", d -> abs(q(d)))]
        c == 0.05 && append!(qs, [("G_edge", d -> G(d, :edge)), ("G_interior", d -> G(d, :interior))])
        trips = haskey(L, "L4") ? [("L1", "L2", "L3"), ("L2", "L3", "L4")] : [("L1", "L2", "L3")]
        for (a, b, f3) in trips, (lbl, f) in qs
            cc = cauchy(f(L[a]), f(L[b]), f(L[f3]))
            push!(rows, (C_over_g=c, triplet="$(a)-$(f3)", quantity=lbl, d12=cc.d12, d23=cc.d23, p_obs=cc.p,
                         rel23=cc.rel23))
        end
    end
    return DataFrame(rows)
end

function figure(byC, fits, wk)
    dir = joinpath(REPO, "plots", TEST_ID, OUT_SUB); mkpath(dir)
    with_theme(Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))) do
        fig = Figure(size=(1180, 430))
        # (a) azimuthal mean of η about x_r, S3 at L3 (P3), with the free-a₀ fit and c_pred
        ax1 = Axis(fig[1, 1], xlabel="r / λ_p", ylabel="Re⟨η⟩ − a₀  (m)", title="(a) near field, S3, L3, C_ρ = 0.05g")
        if haskey(byC, 0.05) && haskey(byC[0.05], "L3")
            d = byC[0.05]["L3"]; r = d["S3_polar_r"]; m = d["S3_polar_m"]; ft = fit_rlnr(r, m)
            λp = d["lambda_p"]; cp = d["S3_node_f"] / (8π * d["C_rho"])
            scatter!(ax1, r ./ λp, real.(m .- ft.a0); color=:black, markersize=5, label="FE")
            rr = range(minimum(r), maximum(r); length=200)
            lines!(ax1, rr ./ λp, real.(cp .* rr .^ 2 .* log.(rr) .+ ft.a2 .* rr .^ 2 .+ ft.a4 .* rr .^ 4);
                   color=:tomato, label="c_pred r² ln r + a₂r² + a₄r⁴")
            axislegend(ax1; position=:lb, framevisible=false)
        end
        # (b) Cauchy differences of G_h and q_h against h_min
        ax2 = Axis(fig[1, 2], xscale=log10, yscale=log10, xlabel="h_min / λ_p (finer level of the pair)",
                   ylabel="Cauchy difference / |finest value|", title="(b) driving point under refinement")
        mk = Dict(0.05 => :circle, 0.005 => :rect, 5e-4 => :utriangle)
        ax2.xreversed = true
        for c in CS
            haskey(byC, c) && all(haskey(byC[c], l) for l in LADDER) || continue
            L = byC[c]; lv = [l for l in LEVELS if haskey(L, l)]
            h = [L[l]["hmin"] / L[l]["lambda_p"] for l in lv[2:end]]
            for (f, col, lbl) in [(d -> G(d), :black, "G_h"), (d -> abs(q(d)), :steelblue, "|q_h|")]
                v = f.(getindex.(Ref(L), lv))
                dd = [abs(v[i] - v[i + 1]) for i in 1:(length(v) - 1)] ./ abs(v[end])
                scatterlines!(ax2, h, dd; color=col, marker=mk[c], label=@sprintf("%s, C_ρ = %gg", lbl, c))
            end
        end
        h0 = FAR == :graded ? [1 / 24, 1 / 48] : [1 / 32, 1 / 128]
        lines!(ax2, h0, 0.02 .* (h0 ./ h0[1]); color=:gray, linestyle=:dash, label="slope 1")
        Legend(fig[2, 2], ax2; framevisible=false, labelsize=9, nbanks=3, orientation=:horizontal)
        # (c) |G_h| against C_ρ
        ax3 = Axis(fig[1, 3], xscale=log10, yscale=log10, xlabel="C_ρ / g", ylabel="|G(x_r, x_r)|  (m/(m⁴s⁻²))",
                   title="(c) driving-point compliance against C_ρ")
        lv = all(haskey(byC, c) && haskey(byC[c], "L3") for c in CS) ? "L3" : "L2"
        cs = [c for c in CS if haskey(byC, c) && haskey(byC[c], lv)]
        if !isempty(cs)
            scatter!(ax3, cs, [abs(G(byC[c][lv])) for c in cs]; color=:black, label="FE, $lv")
            scatter!(ax3, cs, [abs(byC[c][lv]["G_inf"]) for c in cs]; color=:tomato, marker=:xcross,
                     label="infinite plate on the fluid")
            c1 = cs[1]; g1 = abs(G(byC[c1][lv]))
            lines!(ax3, [5e-4, 0.05], g1 .* ([5e-4, 0.05] ./ c1) .^ -0.5; color=:gray, linestyle=:dash, label="∝ C_ρ^(−1/2)")
            axislegend(ax3; position=:rt, framevisible=false)
        end
        save(joinpath(dir, "point_resonator.pdf"), fig)
        save(joinpath(dir, "point_resonator.png"), fig; px_per_unit=300 / 96)
    end
end

function post()
    recs = records()
    pt = points_table(recs)
    ck, fits, wk, byC = checks(recs)
    cv = convergence_table(byC)
    out = joinpath(REPO, "data", "exp_pro", TEST_ID, OUT_SUB); mkpath(out)
    CSV.write(joinpath(out, "points.csv"), pt)
    CSV.write(joinpath(out, "convergence.csv"), cv)
    CSV.write(joinpath(out, "fits.csv"), fits)
    CSV.write(joinpath(out, "winkler.csv"), wk)
    CSV.write(joinpath(out, "checks.csv"), ck)
    @eval Main using CairoMakie
    Base.invokelatest(figure, byC, fits, wk)
    show(stdout, MIME("text/plain"), pt; allcols=true); println()
    show(stdout, MIME("text/plain"), cv; allcols=true); println()
    show(stdout, MIME("text/plain"), fits; allcols=true); println()
    show(stdout, MIME("text/plain"), wk; allcols=true); println()
    for r in eachrow(ck)
        r.passed || @printf("FAIL  %-80s %-2s %.4g (tol %.3g)\n", r.name, r.kind, r.value, r.tolerance)
    end
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--list" in ARGS
        for (i, (p, mode)) in enumerate(cases())
            s = mode == :winkler ? nothing : setup_counts(p)
            println(i, "  ", mode, "  ", mode == :winkler ? "D-W, h_min = l_c/{8,16,32}" : (p.C_over_g, p.level, s))
        end
    elseif "--case" in ARGS
        for (j, a) in enumerate(ARGS)
            a == "--case" && simulate(parse(Int, ARGS[j + 1]))
        end
    else
        post()
    end
end
