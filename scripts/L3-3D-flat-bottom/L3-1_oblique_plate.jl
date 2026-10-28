#=
L3-1 — bare plate in 3D, oblique plane wave, periodic in y (port of the legacy VT3D_A,
part A-D, extended to oblique incidence: matrix row L3.1(a)). Problem and observables
are documented in src/L3_1_oblique_plate.jl.

Cases (one Julia process per case or small group; METIS solver; nz = 8, p = 2):
   1     full layout (L3-0 target), θ = 30°, ω = 10, h = λ_p/8      ← binding target
   2–3   full layout, ω = 10, θ = 0°, 60°
   4–7   full layout, ω ∈ {5, 2}, θ ∈ {30°, 0°}
   8–13  short layout (mesh study), ω = 10, θ ∈ {30°, 0°}, h = λ_p/{6, 8, 12}
   14    short layout, ω = 10, θ = 0°, h = λ_p/16 (reported)

Checks: frozen in the L3.1 row of verification_test_matrix_2026-10-22.md before the
binding run.

Outputs (relative to the repository root):
  data/sims/L3-1/      one DrWatson record per case
  data/exp_pro/L3-1/   results.csv, ladder.csv, checks.csv
  plots/L3-1/          oblique_plate.{pdf,png}

Run:
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-1_oblique_plate.jl --list
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-1_oblique_plate.jl --case 1 [--case 2 ...]
  julia --project scripts/L3-3D-flat-bottom/L3-1_oblique_plate.jl            # tables, checks, figure
=#

using DrWatson                 # CairoMakie only in `post`
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L3_1

const TEST_ID = "L3-1"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const MEM_GB = 17.0                    # METIS-projected launch bound of the 3D batch rules

cases() = [
    L3_1_Params(θ_deg=30.0, ω=10.0),
    L3_1_Params(θ_deg=0.0, ω=10.0),
    L3_1_Params(θ_deg=60.0, ω=10.0),
    L3_1_Params(θ_deg=30.0, ω=5.0),
    L3_1_Params(θ_deg=0.0, ω=5.0),
    L3_1_Params(θ_deg=30.0, ω=2.0),
    L3_1_Params(θ_deg=0.0, ω=2.0),
    [L3_1_Params(θ_deg=θ, ω=10.0, epw=e, layout_kind=:short) for θ in (30.0, 0.0) for e in (6, 8, 12)]...,
    L3_1_Params(θ_deg=0.0, ω=10.0, epw=16, layout_kind=:short),
]

record_name(p) = @sprintf("theta=%g_omega=%g_epw=%d_nz=%d_layout=%s_fixed=%s", p.θ_deg, p.ω, p.epw, p.nz,
                          p.layout_kind, fixed_hash(p))

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
        merge!(d, hefem_pin())
        d
    end
    @printf("[L3-1] case %d done: N = %d, wall %.0f s, peak RSS %.2f GB\n", i, data["N"], data["wall_s"],
            data["maxrss_final_gb"])
    return data
end

# ─── Post-processing ────────────────────────────────────────────────────────

function records()
    path = joinpath(REPO, "data", "sims", TEST_ID)
    out = Tuple{L3_1_Params,Dict}[]
    for p in cases()
        f = joinpath(path, record_name(p) * ".jld2")
        isfile(f) && push!(out, (p, wload(f)))
    end
    return out
end

kh(d) = sqrt(((d["kx_h_R"] + d["kx_h_L"]) / 2)^2 + d["ky"]^2)
err_k(d) = kh(d) / d["kp"] - 1
γh(d) = (abs(d["gamma_h_R"]) + abs(d["gamma_h_L"])) / 2
err_γ(d, θ) = γh(d) * cosd(θ) / d["gamma_r"] - 1
err_E(d) = d["flux_net"] / d["P_in"] - 1

function results_table(recs)
    rows = NamedTuple[]
    for (p, d) in recs
        push!(rows, (layout=String(p.layout_kind), theta=p.θ_deg, omega=p.ω, epw=p.epw, nz=p.nz, nx=d["nx"], ny=d["ny"],
                     N=d["N"], lunz=d["lunz"], W_over_lp=d["W"] / d["λ"], Lx_over_lp=d["Lx"] / d["λ"],
                     kp=d["kp"], ky=d["ky"], kx=d["kx"], kx_h_R=d["kx_h_R"], kx_h_L=d["kx_h_L"], err_k=err_k(d),
                     gamma_h=γh(d), gamma_theta=d["gamma_theta"], err_gamma=err_γ(d, p.θ_deg),
                     P_in=d["P_in"], flux_net=d["flux_net"], err_energy=err_E(d),
                     recip=abs(d["G_ab"] - d["G_ba"]) / abs(d["G_ab"]), rel_residual=d["rel_residual"],
                     fit_residual=d["fit_residual"], transverse_residual=d["transverse_residual"],
                     sponge_reflection=d["sponge_reflection"], asym_A=d["sym_rel"],
                     peak_rss_gb=d["maxrss_final_gb"], wall_s=d["wall_s"], factorize_s=d["factorize_wall_s"],
                     assembly_s=d["assembly_wall_s"]))
    end
    return DataFrame(rows)
end

rate(e1, e2, h1, h2) = log(abs(e1) / abs(e2)) / log(h1 / h2)

"Acceptance checks (frozen L3.1 row)."
function checks(recs)
    ck = NamedTuple[]
    add!(name, kind, value, tol, passed) = push!(ck, (name=name, kind=kind, value=Float64(value), tolerance=Float64(tol),
                                                       passed=passed))
    get_(lk, θ, ω, e) = (i = findfirst(((p, d),) -> p.layout_kind == lk && p.θ_deg == θ && p.ω == ω && p.epw == e, recs);
                         i === nothing ? nothing : recs[i][2])
    for (p, d) in recs
        tag = @sprintf("%s, θ=%g°, ω=%g, h=λp/%d", p.layout_kind, p.θ_deg, p.ω, p.epw)
        add!("residual, max over the 3 RHS, $tag", "I", d["rel_residual"], 1e-8, d["rel_residual"] <= 1e-8)
        r = abs(d["G_ab"] - d["G_ba"]) / abs(d["G_ab"])
        add!("reciprocity |G(a,b) − G(b,a)|/|G(a,b)|, point loads, $tag", "I", r, 1e-8, r <= 1e-8)
        add!("METIS ordering used, $tag", "I", Float64(d["ordering"] == "metis"), 1.0, d["ordering"] == "metis")
        add!("wave fit residual at the cell vertices (reported), $tag", "R", d["fit_residual"], Inf, true)
        add!("transverse residual, energy outside e^{ik_y y} (reported), $tag", "R", d["transverse_residual"], Inf, true)
        add!("sponge reflection |A_in/A_out| (reported), $tag", "R", d["sponge_reflection"], Inf, true)
        add!("peak RSS of the process (GiB, reported; launch bound $(MEM_GB) GB projected), $tag", "R",
             d["maxrss_final_gb"], Inf, true)
        if p.layout_kind == :full
            e = abs(err_k(d))
            add!("|k_h/k₀ − 1|, $tag", "D", e, 1.5e-2, e <= 1.5e-2)
            sk = abs(d["kx_h_R"] - d["kx_h_L"]) / d["kx"]
            add!("left/right symmetry |k_x,h(R) − k_x,h(L)|/k_x, $tag", "I", sk, 1e-6, sk <= 1e-6)
            eg = abs(err_γ(d, p.θ_deg))
            add!("radiated amplitude |γ_h cos θ/γ_r − 1|, $tag", "D", eg, 5e-2, eg <= 5e-2)
            ee = abs(err_E(d))
            add!("energy |net flux/P_in − 1|, $tag", "D", ee, 5e-2, ee <= 5e-2)
            if p.θ_deg != 0
                d0 = get_(:full, 0.0, p.ω, p.epw)
                if d0 !== nothing
                    iso = abs(kh(d) / kh(d0) - 1)
                    add!("isotropy |k_h(θ)/k_h(0°) − 1|, $tag", "D", iso, 1e-2, iso <= 1e-2)
                    rg = abs(γh(d) * cosd(p.θ_deg) / γh(d0) - 1)
                    add!("γ_h(θ) cos θ/γ_h(0°) − 1 (reported), $tag", "R", rg, Inf, true)
                end
            end
        end
    end
    # mesh study (short layout, ω = 10): anisotropy and θ = 0 error against h
    lad = NamedTuple[]
    for e in (6, 8, 12, 16)
        d30, d0 = get_(:short, 30.0, 10.0, e), get_(:short, 0.0, 10.0, e)
        d0 === nothing && continue
        push!(lad, (epw=e, err_k_0=err_k(d0), err_k_30=d30 === nothing ? NaN : err_k(d30),
                    aniso=d30 === nothing ? NaN : kh(d30) / kh(d0) - 1,
                    err_gamma_0=err_γ(d0, 0.0), err_gamma_30=d30 === nothing ? NaN : err_γ(d30, 30.0),
                    err_E_0=err_E(d0), err_E_30=d30 === nothing ? NaN : err_E(d30)))
    end
    ladder = DataFrame(lad)
    if nrow(ladder) >= 3 && all(e -> e in ladder.epw, (6, 8, 12))
        L = Dict(r.epw => r for r in eachrow(ladder))
        pa = rate(L[8].aniso, L[12].aniso, 1 / 8, 1 / 12)
        add!("mesh study: anisotropy rate p_obs (h = λp/8 → λp/12)", "D", pa, 1.7, pa >= 1.7)
        add!("mesh study: anisotropy rate p_obs (h = λp/6 → λp/8, reported)", "R", rate(L[6].aniso, L[8].aniso, 1 / 6, 1 / 8), Inf, true)
        for (lbl, f) in (("|k_h/k₀ − 1|, θ = 0", r -> r.err_k_0), ("|k_h/k₀ − 1|, θ = 30°", r -> r.err_k_30),
                         ("|γ_h cosθ/γ_r − 1|, θ = 0", r -> r.err_gamma_0), ("|γ_h cosθ/γ_r − 1|, θ = 30°", r -> r.err_gamma_30),
                         ("energy, θ = 0", r -> r.err_E_0), ("energy, θ = 30°", r -> r.err_E_30))
            add!("mesh study: rate of $lbl (h = λp/8 → λp/12, reported; nz = 8 fixed)", "R",
                 rate(f(L[8]), f(L[12]), 1 / 8, 1 / 12), Inf, true)
        end
        haskey(L, 16) && add!("mesh study: rate of |k_h/k₀ − 1|, θ = 0 (λp/12 → λp/16, reported)", "R",
                              rate(L[12].err_k_0, L[16].err_k_0, 1 / 12, 1 / 16), Inf, true)
        e12 = abs(L[12].aniso)
        add!("mesh study: anisotropy at h = λp/12", "D", e12, 3e-3, e12 <= 3e-3)
    end
    return DataFrame(ck), ladder
end

function figure(recs, ladder)
    dir = joinpath(REPO, "plots", TEST_ID); mkpath(dir)
    with_theme(Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))) do
        fig = Figure(size=(1180, 400))
        i = findfirst(((p, d),) -> p.layout_kind == :full && p.θ_deg == 30 && p.ω == 10 && p.epw == 8, recs)
        ax1 = Axis(fig[1, 1], xlabel="(x − x_s) / λ_p", ylabel="η(x, W/3) / (F γ_θ)",
                   title="(a) θ = 30°, ω = 10, h = λ_p/8 (N = $(i === nothing ? "—" : recs[i][2]["N"]))")
        if i !== nothing
            d = recs[i][2]; x = (d["fig_x"] .- d["x_s"]) ./ d["λ"]; s = d["gamma_theta"]
            lines!(ax1, x, real.(d["fig_eta"]) ./ s; color=:black, linewidth=0.8, label="Re η (FE)")
            lines!(ax1, x, abs.(d["fig_eta"]) ./ s; color=:tomato, label="|η| (FE)")
            for w in (d["win_R"], d["win_L"])
                vspan!(ax1, (w[1] - d["x_s"]) / d["λ"], (w[2] - d["x_s"]) / d["λ"]; color=(:steelblue, 0.12))
            end
            for xx in (d["Ls"], d["Lx"] - d["Ls"])
                vlines!(ax1, [(xx - d["x_s"]) / d["λ"]]; color=:gray, linestyle=:dash)
            end
            ylims!(ax1, -1.05, 1.45)
            axislegend(ax1; position=:ct, framevisible=false, orientation=:horizontal)
        end
        ax2 = Axis(fig[1, 2], xscale=log10, yscale=log10, xlabel="h / λ_p", ylabel="relative error",
                   title="(b) mesh study (short channel, nz = 8)")
        if nrow(ladder) > 0
            h = 1 ./ ladder.epw
            scatterlines!(ax2, h, abs.(ladder.err_k_0); color=:black, marker=:circle, label="|k_h/k₀ − 1|, θ = 0")
            ok = .!isnan.(ladder.err_k_30)
            scatterlines!(ax2, h[ok], abs.(ladder.err_k_30[ok]); color=:steelblue, marker=:rect, label="|k_h/k₀ − 1|, θ = 30°")
            scatterlines!(ax2, h[ok], abs.(ladder.aniso[ok]); color=:tomato, marker=:utriangle, label="|k_h(30°)/k_h(0°) − 1|")
            hh = [1 / 6, 1 / 12]
            lines!(ax2, hh, 2e-2 .* (hh ./ hh[1]) .^ 2; color=:gray, linestyle=:dash, label="slope 2")
            axislegend(ax2; position=:rb, framevisible=false, labelsize=9)
        end
        ax3 = Axis(fig[1, 3], xlabel="θ (deg)", ylabel="value", title="(c) h = λ_p/8: amplitude and energy")
        fr = [(p, d) for (p, d) in recs if p.layout_kind == :full]
        mk = Dict(10.0 => :circle, 5.0 => :rect, 2.0 => :utriangle)
        for ω in (10.0, 5.0, 2.0)
            s = sort([(p.θ_deg, d) for (p, d) in fr if p.ω == ω]; by=first)
            isempty(s) && continue
            scatter!(ax3, first.(s), [γh(d) / d["gamma_theta"] for (_, d) in s]; color=:black, marker=mk[ω],
                     label="γ_h/γ_θ, ω = $(Int(ω))")
            scatter!(ax3, first.(s), [d["flux_net"] / d["P_in"] for (_, d) in s]; color=:tomato, marker=mk[ω],
                     label="net flux/P_in, ω = $(Int(ω))")
        end
        hlines!(ax3, [1.0]; color=:gray, linestyle=:dash)
        axislegend(ax3; position=:rb, framevisible=false, labelsize=9)
        save(joinpath(dir, "oblique_plate.pdf"), fig)
        save(joinpath(dir, "oblique_plate.png"), fig; px_per_unit=300 / 96)
    end
end

function post()
    recs = records()
    rt = results_table(recs)
    ck, ladder = checks(recs)
    out = joinpath(REPO, "data", "exp_pro", TEST_ID); mkpath(out)
    CSV.write(joinpath(out, "results.csv"), rt)
    CSV.write(joinpath(out, "ladder.csv"), ladder)
    CSV.write(joinpath(out, "checks.csv"), ck)
    @eval Main using CairoMakie
    Base.invokelatest(figure, recs, ladder)
    show(stdout, MIME("text/plain"), rt[:, [:layout, :theta, :omega, :epw, :N, :err_k, :err_gamma, :err_energy, :recip,
                                            :fit_residual, :sponge_reflection, :peak_rss_gb, :wall_s]]; allcols=true); println()
    show(stdout, MIME("text/plain"), ladder; allcols=true); println()
    for r in eachrow(ck)
        r.passed || @printf("FAIL  %-90s %-2s %.4g (tol %.3g)\n", r.name, r.kind, r.value, r.tolerance)
    end
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--list" in ARGS
        for (i, p) in enumerate(cases())
            println(i, "  ", record_name(p), "  ", setup_counts(p))
        end
    elseif "--case" in ARGS
        for (j, a) in enumerate(ARGS)
            a == "--case" && simulate(parse(Int, ARGS[j + 1]))
        end
    else
        post()
    end
end
