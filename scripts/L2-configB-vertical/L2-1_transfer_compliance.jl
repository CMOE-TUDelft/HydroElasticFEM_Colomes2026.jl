#=
L2-1 — Transfer compliance G_B of a finite plate in open water.
Pre- and post-processing of the test defined in src/L2_1_transfer_compliance.jl
(physics, relation to the paper and observables are documented there).

Study: P1 plate (T_ρ = 0), L_p = 20 m, free edges, H = 10 m; 25 points (20
regular, a = 1 m, and 5 random); ω ∈ {2, 5, 9.8, 10, 10.2} rad/s; p = 3.
  - mesh ladder: 4, 8, 16 elements per min(λ_open, λ_p, 4ℓ_f), open water of 6
    evanescent decay lengths 1/κ₁ on each side (6.3–25 m),
    and 32 at ω ∈ {2, 5} (coarse meshes; the observed order at ω = 2 is ≈ 2.5
    from 4/8/16, so the 8-vs-16 change alone is ≈ 1e-4);
  - truncation: 4 and 8 decay lengths, 16 elements per wavelength (on 8
    elements the discrete radiation condition, posed with the exact k, leaves
    a spurious reflection of relative size ~1e-6 whose phase depends on the
    domain length; it masks the e^{-2κ₁Δ} decay of the truncation change at
    ω ≥ 5 and disappears on 16 elements);
  - derivative: ω(1 ± 1e-6) on 16 elements per wavelength (for the frequency-equivalent error);
  - profile: G_B(x, x) at 200 regular points (spacing 0.1 m, no random points)
    on 16 elements per wavelength, 6 decay lengths, for the diagonal figure.
    The points are not mesh nodes; these cases are stored in
    data/sims/L2-1/profile/ (their record names do not carry n_regular).

Acceptance (test matrix §L2.1; reciprocity is an identity of the discrete
problem, not a truncation diagnostic: the Sommerfeld condition keeps the
operator complex symmetric):
  - type I, every case (profile cases: same checks, in profile_checks.csv): ‖G − Gᵀ‖/‖G‖ ≤ 1e-8; mirror symmetry of the regular
    block about the plate centre ≤ 1e-8; spring residual (point-load
    construction) ≤ 1e-12;
  - type D, every ω: the changes of G between meshes and between truncations,
    expressed as a frequency shift δω = ‖ΔG‖/‖∂G/∂ω‖: near the sharp wet
    resonances of the free plate (high Q: plate waves are almost totally
    reflected at the free edges when λ_open ≪ λ_p) a small shift of a resonance
    changes G by a large fraction, so the relative change of G is reported but
    the check is on δω/ω: mesh (two finest levels) ≤ 1e-4, truncation (6 vs 8 decay
    lengths) ≤ 1e-5 and smaller than for 4 vs 6.

Outputs:
  data/sims/L2-1/      one DrWatson record per case (profile cases in profile/)
  data/exp_pro/L2-1/   errors.csv, convergence.csv, diagonal.csv (G_B(x, x) at the 20 regular points),
                       checks.csv; profile.csv (G_B(x, x) at the 200 profile points),
                       profile_errors.csv, profile_checks.csv (kept apart: the verification tables read
                       errors.csv and checks.csv)
  plots/L2-1/          diagonal.pdf/png (Re and Im G_B(x, x) along the plate: lines at the profile
                       points, markers at the 20 regular points)

Run:
  julia --project -O1 scripts/L2-configB-vertical/L2-1_transfer_compliance.jl [--smoke] [--force]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
using LinearAlgebra: norm
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L2_1

const TEST_ID = "L2-1"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ─── 1. Cases ───────────────────────────────────────────────────────────────

const STUDY_W = (2.0, 5.0, 9.8, 10.0, 10.2)
const LADDER = (4, 8, 16)
const GAPS = (4.0, 8.0)
const GAP_REF = 6.0
const N_TRUNC = 16                    # truncation on the finest ladder mesh (see header)
const FINE_W = (2.0, 5.0)              # frequencies with a 32-element level (coarse meshes, cheap)
const DW = 1e-6                       # relative frequency step of the derivative
const N_PROFILE = 200                 # regular points of the profile cases (0.1 m spacing)
const PROFILE_DIR = "profile"

function cases(; smoke::Bool=false)
    smoke && return [L2_1_Params(ω=2.0, elements_per_wavelength=4)]
    ladder = [L2_1_Params(ω=ω, elements_per_wavelength=n, gap_decays=GAP_REF) for ω in STUDY_W for n in LADDER]
    fine = [L2_1_Params(ω=ω, elements_per_wavelength=32, gap_decays=GAP_REF) for ω in FINE_W]
    trunc = [L2_1_Params(ω=ω, elements_per_wavelength=N_TRUNC, gap_decays=g) for ω in STUDY_W for g in GAPS]
    deriv = [L2_1_Params(ω=ω * (1 + s * DW), elements_per_wavelength=16, gap_decays=GAP_REF)
             for ω in STUDY_W for s in (-1, 1)]
    return [ladder; fine; trunc; deriv]
end

"Dense profile of G_B(x, x): finest mesh, reference truncation, `N_PROFILE` regular points."
profile_cases(; smoke::Bool=false) =
    smoke ? L2_1_Params[] : [L2_1_Params(ω=ω, elements_per_wavelength=16, gap_decays=GAP_REF, n_regular=N_PROFILE, n_random=0)
                             for ω in STUDY_W]

# ─── 2. Simulation records (DrWatson) ───────────────────────────────────────

record_name(p::L2_1_Params) = savename(p; accesses=VARIABLE, sigdigits=8)   # ω(1 ± 1e-6) cases * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

function simulate(p::L2_1_Params; root::AbstractString=REPO, force::Bool=false, sub::AbstractString="")
    path = joinpath(root, "data", "sims", TEST_ID, sub)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L2_1_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L2_1_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── 3. Post-processing ─────────────────────────────────────────────────────

"Mirror-symmetry residual of the regular block: G[i, j] = G[N+1−i, N+1−j]."
function mirror_residual(Gm, n)
    B = Gm[1:n, 1:n]
    return norm(B - B[end:-1:1, end:-1:1]) / norm(B)
end

function errors(p::L2_1_Params, d)
    return (ω=p.ω, elements_per_wavelength=p.elements_per_wavelength, gap_decays=p.gap_decays, gap=d["gap"], ndofs=d["ndofs"],
            element_size=d["element_size"], lambda_open=d["lambda_open"], lambda_p=d["lambda_p"], ell_f=d["ell_f"],
            reciprocity=d["reciprocity"], mirror=mirror_residual(d["G"], p.n_regular),
            spring_residual=d["spring_residual"], wall_time_s=d["wall_time_s"])
end

relchange(A, B) = norm(A - B) / norm(B)

"""
Mesh and truncation convergence of G per frequency: relative changes of G and
the same changes as frequency shifts δω/ω = ‖ΔG‖/(ω‖∂G/∂ω‖), with ∂G/∂ω by
central differences on the finest mesh.
"""
function convergence(cs, records)
    rows = NamedTuple[]
    find(ω, n, g) = (i = findfirst(p -> p.ω == ω && p.elements_per_wavelength == n && p.gap_decays == g, cs);
                     i === nothing ? nothing : records[i]["G"])
    for ω in STUDY_W
        G4, G8, G16, G32 = find(ω, 4, GAP_REF), find(ω, 8, GAP_REF), find(ω, 16, GAP_REF), find(ω, 32, GAP_REF)
        Gt4, Gt6, Gt8 = find(ω, N_TRUNC, 4.0), find(ω, N_TRUNC, GAP_REF), find(ω, N_TRUNC, 8.0)
        Gm, Gp = find(ω * (1 - DW), 16, GAP_REF), find(ω * (1 + DW), 16, GAP_REF)
        any(isnothing, (G4, G8, G16, Gt4, Gt6, Gt8, Gm, Gp)) && continue
        dG = norm(Gp - Gm) / (2DW * ω)                     # ‖∂G/∂ω‖
        shift(A, B) = norm(A - B) / (dG * ω)
        push!(rows, (ω=ω, change_4_8=relchange(G4, G8), change_8_16=relchange(G8, G16),
                     shift_4_8=shift(G4, G8), shift_8_16=shift(G8, G16),
                     shift_16_32=G32 === nothing ? NaN : shift(G16, G32),
                     finest_pair=G32 === nothing ? "8 vs 16" : "16 vs 32",
                     shift_finest=G32 === nothing ? shift(G8, G16) : shift(G16, G32),
                     observed_order=log2(relchange(G4, G8) / relchange(G8, G16)),
                     trunc_4_6=relchange(Gt4, Gt6), trunc_8_6=relchange(Gt8, Gt6),
                     shift_trunc_4_6=shift(Gt4, Gt6), shift_trunc_8_6=shift(Gt8, Gt6),
                     resonance_sharpness=dG * ω / norm(G16)))      # ω‖∂G/∂ω‖/‖G‖ ~ Q near a resonance
    end
    return DataFrame(rows)
end

"G_B(x, x) along the plate on the finest mesh, regular points (the 20 of the study, or the profile points)."
function diagonal(cs, records)
    rows = NamedTuple[]
    for (p, d) in zip(cs, records)
        (p.gap_decays == GAP_REF && p.elements_per_wavelength == 16 && p.ω in STUDY_W) || continue
        for i in 1:p.n_regular
            push!(rows, (ω=p.ω, x=d["points"][i], re_G=real(d["G"][i, i]), im_G=imag(d["G"][i, i])))
        end
    end
    return DataFrame(rows)
end

function checks(df::DataFrame, cv::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    for r in eachrow(df)
        tag = @sprintf("ω=%.7g, n=%d, gap=%g/κ₁", r.ω, r.elements_per_wavelength, r.gap_decays)
        add!("reciprocity ‖G − Gᵀ‖/‖G‖, $(tag)", "I", r.reciprocity, 1e-8)
        add!("mirror symmetry, $(tag)", "I", r.mirror, 1e-8)
        add!("spring residual, $(tag)", "I", r.spring_residual, 1e-12)
    end
    for r in eachrow(cv)
        tag = @sprintf("ω=%g", r.ω)
        add!("mesh: frequency-equivalent change δω/ω ($(r.finest_pair)), $(tag)", "D", r.shift_finest, 1e-4)
        add!("truncation: δω/ω (6 vs 8 decay lengths), $(tag)", "D", r.shift_trunc_8_6, 1e-5)
        push!(rows, (name="truncation decreases with the gap, $(tag)", kind="D", value=r.trunc_8_6 / r.trunc_4_6,
                     tolerance=1.0, passed=r.trunc_8_6 < r.trunc_4_6))
    end
    return DataFrame(rows)
end

function plot_diagonal(dg::DataFrame, pf::DataFrame, dir::AbstractString)
    nrow(dg) == 0 && return
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    mkpath(dir)
    with_theme(theme) do
        fig = Figure(size=(900, 340))
        axr = Axis(fig[1, 1]; xlabel=L"x\ \mathrm{(m)}", ylabel=L"\mathrm{Re}\,G_B(x,x)", title="point compliance")
        axi = Axis(fig[1, 2]; xlabel=L"x\ \mathrm{(m)}", ylabel=L"\mathrm{Im}\,G_B(x,x) = \gamma_r(x)", title="radiative width")
        for (i, ω) in enumerate(sort(unique(dg.ω)))
            g = sort(filter(r -> r.ω == ω, dg), :x)
            c = Makie.wong_colors()[i]
            lab = @sprintf("ω = %g", ω)
            gp = sort(filter(r -> r.ω == ω, pf), :x)
            if nrow(gp) > 0
                lines!(axr, gp.x, gp.re_G; color=c, label=lab)
                lines!(axi, gp.x, gp.im_G; color=c)
                scatter!(axr, g.x, g.re_G; color=c, markersize=5)
                scatter!(axi, g.x, g.im_G; color=c, markersize=5)
            else
                scatterlines!(axr, g.x, g.re_G; color=c, markersize=5, label=lab)
                scatterlines!(axi, g.x, g.im_G; color=c, markersize=5)
            end
        end
        Legend(fig[1, 3], axr; framevisible=false, labelsize=10)
        save(joinpath(dir, "diagonal.pdf"), fig)
        save(joinpath(dir, "diagonal.png"), fig; px_per_unit=300 / 96)
    end
end

# ─── 4. Driver ──────────────────────────────────────────────────────────────

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO)
    cs = cases(; smoke=smoke)
    records = [simulate(p; root=root, force=force) for p in cs]
    ps = profile_cases(; smoke=smoke)
    precords = [simulate(p; root=root, force=force, sub=PROFILE_DIR) for p in ps]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    cv = convergence(cs, records)
    dg = diagonal(cs, records)
    pf = diagonal(ps, precords)
    ck = checks(df, cv)
    pe = DataFrame([errors(p, d) for (p, d) in zip(ps, precords)])
    pck = nrow(pe) == 0 ? DataFrame() : checks(pe, DataFrame())
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "convergence.csv"), cv)
    CSV.write(joinpath(out, "diagonal.csv"), dg)
    CSV.write(joinpath(out, "profile.csv"), pf)
    CSV.write(joinpath(out, "profile_errors.csv"), pe)
    CSV.write(joinpath(out, "profile_checks.csv"), pck)
    CSV.write(joinpath(out, "checks.csv"), ck)
    plot_diagonal(dg, pf, joinpath(root, "plots", TEST_ID))

    show(stdout, MIME"text/plain"(), select(df, :ω, :elements_per_wavelength, :gap_decays, :gap, :ndofs, :reciprocity, :mirror,
                                            :spring_residual, :wall_time_s); allrows=true)
    println()
    show(stdout, MIME"text/plain"(), cv; allrows=true)
    println()
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    nrow(pck) > 0 && @printf("%d/%d profile checks pass\n", count(pck.passed), nrow(pck))
    for c in eachrow(filter(r -> !r.passed, nrow(pck) > 0 ? [ck; pck] : ck))
        @printf("FAIL  %-70s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (errors=df, checks=ck)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; smoke="--smoke" in ARGS, force="--force" in ARGS)
end
