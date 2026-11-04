#=
L4-4 — anisotropic resonator island on a 3D plate (homogenised `ResonatorField`).
Pre- and post-processing of src/L4_4_aniso_island.jl (problem, layout and
references are documented there). Runs in envs/resonator_field (upstream
`ResonatorField`, branch feat/resonator-field, commit 8314700).

Binding configuration (RE-FREEZE 2026-10-31, PI-approved, row L4.4 of
verification_test_matrix_2026-10-22.md, frozen 2026-11-02 before the binding runs): P1 plate,
H = 10, half domain y ≥ 0, p = 3, h = λ_p/4, nz = 7 (8 projected 15.2–15.7 GiB > 14 for
A8/A12/A10/A8w), Kelvin–Voigt frame 2λ_p
(s = 10, σ²) downstream and at the side, 0.5λ_p upstream with a 0.25λ_p gap
(smoke-tested: `L4-4_upstream_frame_test.jl`), downstream gap 0.5λ_p, WIDE layout
in y (undamped region to y_end = 12.5–16 m), Gaussian islands tapered on ξ ∈ [2, 2.5].

Cases (one Julia process each, rss_guard + run_limited, heavy-run lock):
   1  A8    (L_x, L_y) = (3, 2),    ω = 8,  n₀m = 0.03           converging lens
   2  B8    (2, 3),                 ω = 8,  0.03                incidence along the long axis
   3  A12   (2.25, 1.5),            ω = 12, 0.03                diverging lens (above ω_r)
   4  A10   (3, 2),                 ω = 10, 0.01, ζ = 0.05      optical depth
   5  B10   (2, 3),                 ω = 10, 0.01, ζ = 0.05      optical depth
   6  A8w   (3, 2),                 ω = 8,  0.01                weak loading (thin lens)
   7  M4    case 1, narrow layout (y_end = default), h = λ_p/4  mesh variant pair
   8  M5    case 1, narrow layout, h = λ_p/5
   9  M4z8  case 7 with nz = 8                                 vertical-mesh variant
The first freeze (2026-10-30, h = λ_p/3, frame 1.5λ_p, larger islands) is kept in
`cases_freeze1()` for the record (its records and lenses are in data/sims/L4-4/ and
the first as-run block of the matrix row).

Processing: T = η/e^{ik₀x} on three lines x = ξ_cL_x + {0.5, 1.0, 1.5} m, y ∈ [0, y_end],
cosine window of T − 1 over the last TAPER m, back-propagation to x = 0 with the bare
propagator, f = −k₀/Φ''(0) (fit on |y| ≤ L_y/2), 𝒜 = −ln|T(0)|; mean over the three lines.
The PE reference is processed identically (same lines, same y_end, same window).

Outputs:
  data/sims/L4-4/       one record per case (not committed)
  data/exp_pro/L4-4/    sizing.csv, references.csv, y_sensitivity.csv, lenses.csv, checks.csv
  plots/L4-4/           aniso_island.{pdf,png}

Run:
  julia --project=envs/resonator_field -O1 scripts/L4-beyond-analytical/L4-4_aniso_island.jl --list
  julia --project=envs/resonator_field -O1 scripts/L4-beyond-analytical/L4-4_aniso_island.jl --smoke    # identities, small mesh
  julia --project=envs/resonator_field -O1 scripts/L4-beyond-analytical/L4-4_aniso_island.jl --size     # pattern fill, projection
  julia --project=envs/resonator_field -O1 scripts/L4-beyond-analytical/L4-4_aniso_island.jl --refs     # ray, thin lens, PE
  julia --project=envs/resonator_field -O1 scripts/L4-beyond-analytical/L4-4_aniso_island.jl --case 1   # under rss_guard
  julia --project=envs/resonator_field scripts/L4-beyond-analytical/L4-4_aniso_island.jl                # checks, figure
=#

using DrWatson
using DataFrames
using CSV
using Printf
using TOML
using LinearAlgebra
using HydroElasticFEM_Colomes2026
using HydroElasticFEM_Colomes2026.References
include(joinpath(@__DIR__, "..", "..", "src", "L4_4_aniso_island.jl"))
using .L4_4

const TEST_ID = "L4-4"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const ENV = joinpath(REPO, "envs", "resonator_field")
const TAPER = 8.0              # cosine window of T − 1 at the end of the line data (m)
const SPEC = (Ny=8192, Yp=200.0)
const PE_DX = 0.025
const Y_SENS_MAX = 30.0        # reference insensitivity: y' ∈ [y_end, 30] m

const NAMES = ["A8", "B8", "A12", "A10", "B10", "A8w", "M4", "M5", "M4z8"]

"Re-freeze 2026-10-31, frozen 2026-11-02 with nz = 7 and case 9 (binding)."
function cases()
    A8 = L4_4_Params(ω=8.0, Lx=3.0, Ly=2.0, load=0.03, epw=4.0, frame=2.0, frame_up=0.5, gap_up=0.25, y_end=15.4, nz=7)
    return [A8,
            L4_4_Params(A8; Lx=2.0, Ly=3.0, y_end=16.0),
            L4_4_Params(A8; ω=12.0, Lx=2.25, Ly=1.5, y_end=13.0),
            L4_4_Params(A8; ω=10.0, load=0.01, ζ=0.05, y_end=12.5),
            L4_4_Params(A8; ω=10.0, Lx=2.0, Ly=3.0, load=0.01, ζ=0.05, y_end=12.5),
            L4_4_Params(A8; load=0.01),
            L4_4_Params(A8; y_end=0.0),
            L4_4_Params(A8; y_end=0.0, epw=5.0),
            L4_4_Params(A8; y_end=0.0, nz=8)]
end

"First freeze 2026-10-30 18:47 (history; names A8, B8, A12, A10, B10, A8w, M3, M4)."
function cases_freeze1()
    A8 = L4_4_Params(ω=8.0, Lx=4.5, Ly=3.0, load=0.03, epw=3.0, y_end=21.9)
    return [A8,
            L4_4_Params(A8; Lx=3.0, Ly=4.5, y_end=24.9),
            L4_4_Params(A8; ω=12.0, Lx=3.0, Ly=2.0, y_end=19.5),
            L4_4_Params(A8; ω=10.0, load=0.01, ζ=0.05, y_end=17.5),
            L4_4_Params(A8; ω=10.0, Lx=3.0, Ly=4.5, load=0.01, ζ=0.05, y_end=17.5),
            L4_4_Params(A8; load=0.01),
            L4_4_Params(A8; y_end=0.0),
            L4_4_Params(A8; y_end=0.0, epw=4.0)]
end

"Small-mesh version (identities)."
smoke_params(; kw...) = L4_4_Params(; Lx=2.0, Ly=1.5, epw=2.0, frame=1.0, gap=0.3, nz=4, ζ=0.05,
                                    line_dx=(0.3, 0.6, 0.9), n_map=(11, 5), n_axis=21, kw...)

record_name(p::L4_4_Params) = savename(p; accesses=VARIABLE, sigdigits=6) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(ENV, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")), "hefem_rev" => "8314700")
end

function simulate(p::L4_4_Params; force::Bool=false, sub::String="")
    path = joinpath(REPO, "data", "sims", TEST_ID, sub)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO, storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L4_4_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        return d
    end
    return data
end

# ─── Sizing: METIS fill of the pattern, calibrated on the measured L3-3 runs ─────────

"""L3-3 calibration (same machine, same solver): CHOLMOD METIS nnz(L) of the L3-3 pattern
(36×36×8 and 40×40×8 cells, p = 3) against the largest rss_guard peak footprint measured
for that mesh (15.84 GiB, 18.63 GiB). Linear law peak = a + b·lnz."""
const CAL = (lnz=(2.1412e8, 2.8558e8), peak=(15.84, 18.63))
projected_peak(lnz) = CAL.peak[1] + (lnz - CAL.lnz[1]) * (CAL.peak[2] - CAL.peak[1]) / (CAL.lnz[2] - CAL.lnz[1])

"""L4-4 calibration (re-freeze 2026-10-31): least-squares line through the largest guard peaks
measured for six L4-4 patterns of the first freeze (lnz 0.79–2.07e8 → 7.92–14.04 GiB: M3, the
frame tests 2/2.5/3λ, M4, A8), shifted up by the largest positive residual (0.48 GiB), i.e. an
upper envelope of the measured L4-4 peaks. Used for the re-freeze (the L3-3 law above reads
1.1–2.8 GiB high on the L4-4 runs)."""
projected_peak_l44(lnz) = 4.99 + 4.70 * lnz / 1e8

function sizing(cs)
    rows = NamedTuple[]
    for (i, p) in enumerate(cs)
        l = layout(p); s = size_counts(p)
        f = pattern_fill(l.nx, l.ny, p.nz; order=p.order)
        push!(rows, (case=i, name=NAMES[i], ω=p.ω, Lx=p.Lx, Ly=p.Ly, epw=p.epw, nx=l.nx, ny=l.ny, nz=p.nz,
                     box_x=l.x1 - l.x0, box_y=l.Y, yend=l.yend, N=s.N, lnz_metis=f.chol_lnz_metis,
                     flops_metis=f.chol_fl_metis, projected_peak_gib=projected_peak_l44(f.chol_lnz_metis),
                     projected_peak_l33cal_gib=projected_peak(f.chol_lnz_metis)))
        @printf("%d %-4s %3d×%3d×%d  box %.1f×%.1f m  y_end %.2f  N=%7d  lnz %.3e  projected peak %.1f GiB (L3-3 law %.1f)\n", i, NAMES[i],
                l.nx, l.ny, p.nz, l.x1 - l.x0, l.Y, l.yend, s.N, f.chol_lnz_metis, projected_peak_l44(f.chol_lnz_metis),
                projected_peak(f.chol_lnz_metis))
        flush(stdout)
    end
    return DataFrame(rows)
end

# ─── Processing of line data (FE and PE identically) ──────────────────────────

"f, 𝒜 of one half-line record T(y) at x = xm (window, back-propagation, curvature fit)."
function lens_of_line(ys, T, xm, k, Ly; yend=maximum(ys), taper=TAPER)
    keep = findall(y -> 0 <= y <= yend, ys)
    yb, Tb = backprop(ys[keep], T[keep], xm, k; SPEC..., taper=taper)
    c = curvature_fit(yb, Tb, k, 0.5Ly)
    return (f=c.f, A=c.A0, resid=c.resid, yb=yb, Tb=Tb)
end

"Mean over the lines, spread (max − min)/|mean|."
function lens_mean(ls)
    fs = [l.f for l in ls]; As = [l.A for l in ls]
    return (f=sum(fs) / length(fs), f_spread=(maximum(fs) - minimum(fs)) / abs(sum(fs) / length(fs)),
            A=sum(As) / length(As), fs=fs, As=As)
end

hostp(p) = HostSpec(g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.m_ρ, C_ρ=p.C_ρ)

"References of case p: ray, thin lens, eikonal, PE (untruncated and truncated as the FE data)."
function references(p::L4_4_Params)
    h = hostp(p); l = layout(p)
    χ = chi(p.ω, p.ω_r, p.ζ)
    kφ, kb = local_k_table(h, p.ω, p.load, χ)
    fray = p.ζ == 0 ? ray_focus(kφ, kb, p.Lx, p.Ly, p.ξ_t, p.ξ_c) : NaN
    J = thin_lens_J(kφ, kb, p.ξ_t, p.ξ_c)
    fthin = kb * p.Ly^2 / (p.Lx * J)
    Φ0 = eikonal_phase(kφ, kb, p.Lx, p.Ly, [0.0], p.ξ_t, p.ξ_c)[1]
    ys, ψs = pe_march(kφ, kb, p.Lx, p.Ly, p.ξ_t, p.ξ_c, l.lines; SPEC..., dx=PE_DX)
    half = findall(y -> y >= 0, ys)
    yh = ys[half]
    unt = lens_mean([lens_of_line(yh, ψ[half], xm, kb, p.Ly; taper=0.0) for (xm, ψ) in zip(l.lines, ψs)])
    tr = lens_mean([lens_of_line(yh, ψ[half], xm, kb, p.Ly; yend=l.yend) for (xm, ψ) in zip(l.lines, ψs)])
    sens = NamedTuple[]
    for ym in vcat(l.yend, collect(ceil(l.yend):1.0:Y_SENS_MAX))
        s = lens_mean([lens_of_line(yh, ψ[half], xm, kb, p.Ly; yend=ym) for (xm, ψ) in zip(l.lines, ψs)])
        push!(sens, (y_end=ym, f=s.f, A=s.A, f_rel=s.f / unt.f - 1, A_rel=s.A / unt.A - 1))
    end
    return (; kb, fray, fthin, fgauss=thin_lens_gaussian(h, p.ω, p.load, χ, p.Lx, p.Ly), Φ0, unt, tr, sens,
            pe_lines=[ψ[half] for ψ in ψs], pe_y=yh)
end

# ─── Analysis of the records ───────────────────────────────────────────────────

"Frame reflection estimate on the axis behind the island: T = a₀ + a₁x + b e^{−2ikx}, |b|/mean|T − 1|."
function axis_reflection(d, p)
    xa, Ta, k = d["axis_x"], d["axis_T"], d["k"]
    xs = p.ξ_c * p.Lx
    idx = findall(x -> xs + 0.25 <= x, xa)
    x = xa[idx]; T = Ta[idx]
    M = hcat(ones(length(x)), x, cis.(-2k .* x))
    c = M \ T
    return abs(c[3]) / (sum(abs.(T .- 1)) / length(T))
end

function fe_lens(d, p)
    k = d["k"]; ys = d["line_y"]
    ls = [lens_of_line(ys, T, xm, k, p.Ly; yend=d["yend"]) for (xm, T) in zip(d["lines"], d["line_T"])]
    return lens_mean(ls), ls
end

function row(i, p, d, r)
    fe, _ = fe_lens(d, p)
    return (case=i, name=NAMES[i], ω=p.ω, Lx=p.Lx, Ly=p.Ly, load=p.load, ζ=p.ζ, epw=p.epw, y_end=d["yend"],
            N=d["N"], f_fe=fe.f, f_fe_spread=fe.f_spread, A_fe=fe.A, f_pe=r.tr.f, f_pe_spread=r.tr.f_spread, A_pe=r.tr.A,
            f_pe_untrunc=r.unt.f, A_pe_untrunc=r.unt.A, f_ray=r.fray, f_thin=r.fthin, f_gauss=r.fgauss,
            A_eik=imag(r.Φ0), frame_reflection=axis_reflection(d, p), residual=d["rel_residual"],
            ordering=d["ordering"], maxrss_gb=d["maxrss_final_gb"], wall_s=d["wall_time_s"],
            factorize_s=d["factorize_wall_s"], lunz=d["lunz"])
end

function checks(df, sens, smoke)
    rows = NamedTuple[]
    le!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=abs(value) <= tol))
    if smoke !== nothing
        le!("load identity b = −(A(2n) − A(n))U_inc (small mesh, ζ = 0.05)", "I", smoke.rhs_identity, 1e-10)
        le!("half vs full domain η at 3 points (small mesh)", "I", smoke.half_full, 1e-8)
        le!("full-domain mirror symmetry (small mesh)", "I", smoke.mirror, 1e-8)
        le!("residual, small mesh half/full", "I", max(smoke.res_half, smoke.res_full), 1e-8)
    end
    get(n) = (j = findfirst(==(n), df.name); j === nothing ? nothing : df[j, :])
    for r in eachrow(df)
        le!("residual ‖Ax − b‖/‖b‖, $(r.name)", "I", r.residual, 1e-8)
        le!("METIS ordering used (0 = yes), $(r.name)", "I", r.ordering == "metis" ? 0.0 : 1.0, 0.0)
        le!("peak RSS (Julia) < 20 GiB, $(r.name)", "I", r.maxrss_gb, 20.0)
    end
    for r in eachrow(df)
        le!("frame reflection on the axis behind the island |b|/mean|T − 1|, $(r.name)", "D", r.frame_reflection, 0.02)
    end
    for r in eachrow(df)
        r.name in ("M4", "M5", "M4z8") && continue
        s = filter(x -> x.name == r.name, sens)
        if r.ζ == 0
            # B8 (f ≈ 35 m): the truncation error of Φ''(0) is the same in rad/m² as for A8, so 3 % in f (re-freeze)
            le!("reference insensitive to the y-extent: max_{y' ∈ [y_end, 30]} |f_PE(y')/f_PE(∞) − 1|, $(r.name)", "D",
                maximum(abs.(s.f_rel)), r.name == "B8" ? 0.03 : 0.02)
            le!("FE vs PE focal length |f_FE/f_PE − 1|, $(r.name)", "D", r.f_fe / r.f_pe - 1, 0.05)
            le!("line consistency (max − min)/|mean| of f_FE over 3 lines, $(r.name)", "D", r.f_fe_spread, 0.05)
        else
            le!("reference insensitive to the y-extent: max_{y' ∈ [y_end, 30]} |𝒜_PE(y')/𝒜_PE(∞) − 1|, $(r.name)", "D",
                maximum(abs.(s.A_rel)), 0.01)
            le!("FE vs PE optical depth |𝒜_FE/𝒜_PE − 1|, $(r.name)", "D", r.A_fe / r.A_pe - 1, 0.05)
        end
    end
    m4, m5 = get("M4"), get("M5")
    if m4 !== nothing && m5 !== nothing
        le!("mesh variant λ_p/4 vs λ_p/5 (narrow layout): |Δf|/|f|", "D", (m4.f_fe - m5.f_fe) / m5.f_fe, 0.02)
        le!("mesh variant λ_p/4 vs λ_p/5 (narrow layout): |Δ𝒜| (absolute)", "D", m4.A_fe - m5.A_fe, 0.005)
    end
    m8 = get("M4z8")
    if m4 !== nothing && m8 !== nothing
        le!("vertical mesh nz = 7 vs 8 (narrow layout, λ_p/4): |Δf|/|f|", "D", (m4.f_fe - m8.f_fe) / m8.f_fe, 0.02)
        le!("vertical mesh nz = 7 vs 8 (narrow layout, λ_p/4): |Δ𝒜| (absolute)", "D", m4.A_fe - m8.A_fe, 0.005)
    end
    # re-freeze: ray / thin-lens tolerances widened for the smaller islands (PE vs ray 1.1 / 11.8 / 14.1 %, PE / thin 1.21)
    for n in ("A8", "B8", "A8w")
        r = get(n); r === nothing && continue
        le!("FE vs ray focus |f_FE/f_ray − 1| (ω = 8), $(n)", "A", r.f_fe / r.f_ray - 1, 0.20)
        le!("converging below ω_r: f_FE > 0 (0 = yes), $(n)", "A", r.f_fe > 0 ? 0.0 : 1.0, 0.0)
    end
    r = get("A12")
    r !== nothing && le!("diverging above ω_r: f_FE < 0 (0 = yes), A12", "A", r.f_fe < 0 ? 0.0 : 1.0, 0.0)
    r = get("A8w")
    r !== nothing && le!("weak loading: FE vs thin lens eq:focal_general |f_FE/f_thin − 1|, A8w", "A", r.f_fe / r.f_thin - 1, 0.25)
    a, b = get("A8"), get("B8")
    if a !== nothing && b !== nothing
        le!("focal ratio f_x/f_y: FE vs ray (A8/B8)", "A", (a.f_fe / b.f_fe) / (a.f_ray / b.f_ray) - 1, 0.15)
    end
    a, b = get("A10"), get("B10")
    if a !== nothing && b !== nothing
        le!("optical depth ratio 𝒜_x/𝒜_y = L_x/L_y (eq:aspect_ratio_law), FE (A10/B10)", "A",
            (a.A_fe / b.A_fe) / (a.Lx / a.Ly) - 1, 0.05)
    end
    return DataFrame(rows)
end

function run_smoke()
    base = smoke_params()
    println(size_counts(base))
    r = rhs_identity(base)
    @printf("rhs identity %.3e  N=%d\n", r["rhs_identity"], r["N"])
    h = run_case(base); f = run_case(L4_4_Params(base; full=true))
    mh = h["mirror_T"][1]; mf1, mf2 = f["mirror_T"]
    out = (rhs_identity=r["rhs_identity"], half_full=maximum(abs.(mh .- mf1)) / maximum(abs.(mf1)),
           mirror=maximum(abs.(mf1 .- mf2)) / maximum(abs.(mf1)), res_half=h["rel_residual"], res_full=f["rel_residual"],
           ordering=h["ordering"], N_half=h["N"], N_full=f["N"])
    println(out)
    dir = joinpath(REPO, "data", "exp_pro", TEST_ID); mkpath(dir)
    CSV.write(joinpath(dir, "smoke_identities.csv"), DataFrame([out]))
    return out
end

function plot_all(df, cs, recs, refs, dir)
    @eval using CairoMakie
    mkpath(dir)
    Base.invokelatest() do
        fig = Figure(size=(720, 620), fontsize=10)
        # (a) map of |T| for A8 with the lines
        i1 = findfirst(==("A8"), df.name)
        if i1 !== nothing
            k = df.case[i1]; d = recs[k]; p = cs[k]
            gl = fig[1, 1:3] = GridLayout()
            ax = Axis(gl[1, 1]; xlabel="x (m)", ylabel="y (m)", title="(a) |η/η_inc|, A8 (ω = 8, L_x = 3, L_y = 2 m)")
            colsize!(gl, 1, Aspect(1, -(-(extrema(d["map_x"])...)) / -(-(extrema(d["map_y"])...))))
            hm = heatmap!(ax, d["map_x"], d["map_y"], permutedims(abs.(d["map_T"])); colormap=:viridis)
            Colorbar(gl[1, 2], hm; width=8)
            θ = range(0, π / 2; length=100)
            lines!(ax, p.ξ_c * p.Lx .* cos.(θ) .* [1], p.ξ_c * p.Ly .* sin.(θ); color=:white, linestyle=:dash)
            lines!(ax, -p.ξ_c * p.Lx .* cos.(θ), p.ξ_c * p.Ly .* sin.(θ); color=:white, linestyle=:dash)
            for xm in d["lines"]
                lines!(ax, [xm, xm], [0, d["yend"]]; color=:white, linewidth=0.8)
            end
        end
        # (b) back-propagated phase at x = 0, FE vs PE
        ax2 = Axis(fig[2, 1]; xlabel="y/L_y", ylabel="Φ(y) − Φ(0) (rad)", title="(b) phase at x = 0 (line 1)")
        cols = Makie.wong_colors()
        for (j, n) in enumerate(("A8", "B8", "A12", "A8w"))
            i = findfirst(==(n), df.name); i === nothing && continue
            k = df.case[i]; d = recs[k]; p = cs[k]; r = refs[k]
            _, ls = fe_lens(d, p)
            sel = findall(y -> 0 <= y <= p.Ly, ls[1].yb)
            i0 = argmin(abs.(ls[1].yb))
            scatter!(ax2, ls[1].yb[sel][1:8:end] ./ p.Ly, angle.(ls[1].Tb[sel][1:8:end] ./ ls[1].Tb[i0]); color=cols[j],
                     markersize=5, label="FE $(n)")
            lp = lens_of_line(r.pe_y, r.pe_lines[1], d["lines"][1], r.kb, p.Ly; yend=d["yend"])
            lines!(ax2, lp.yb[sel] ./ p.Ly, angle.(lp.Tb[sel] ./ lp.Tb[i0]); color=cols[j])
        end
        Legend(fig[3, 1], ax2; orientation=:horizontal, nbanks=2, framevisible=false, labelsize=8, tellheight=true)
        # (c) focal lengths
        ax3 = Axis(fig[2, 2:3]; ylabel="f (m)", title="(c) focal length from the island centre",
                   xticks=(1:4, ["A8", "B8", "A12", "A8w"]))
        for (j, n) in enumerate(("A8", "B8", "A12", "A8w"))
            i = findfirst(==(n), df.name); i === nothing && continue
            r = df[i, :]
            scatter!(ax3, [j - 0.15], [r.f_fe]; color=cols[1], marker=:circle, label=j == 1 ? "FE" : nothing)
            scatter!(ax3, [j - 0.05], [r.f_pe]; color=cols[2], marker=:rect, label=j == 1 ? "PE (same window)" : nothing)
            scatter!(ax3, [j + 0.05], [r.f_ray]; color=cols[3], marker=:diamond, label=j == 1 ? "ray" : nothing)
            scatter!(ax3, [j + 0.15], [r.f_thin]; color=cols[4], marker=:utriangle, label=j == 1 ? "thin lens" : nothing)
        end
        hlines!(ax3, [0.0]; color=:gray, linewidth=0.5)
        Legend(fig[3, 2:3], ax3; orientation=:horizontal, nbanks=2, framevisible=false, labelsize=8, tellheight=true)
        rowsize!(fig.layout, 1, Relative(0.45))
        save(joinpath(dir, "aniso_island.pdf"), fig)
        save(joinpath(dir, "aniso_island.png"), fig; px_per_unit=300 / 96)
    end
end

function main()
    cs = cases()
    out = joinpath(REPO, "data", "exp_pro", TEST_ID)
    if "--list" in ARGS
        for (i, p) in enumerate(cs)
            l = layout(p); s = size_counts(p)
            @printf("%d %-4s ω=%-4g (%g,%g) load %g ζ %g h=λ/%g  %3d×%3d×%d  box %.1f×%.1f m  y_end %.2f  lines %s  N=%d\n",
                    i, NAMES[i], p.ω, p.Lx, p.Ly, p.load, p.ζ, p.epw, l.nx, l.ny, p.nz, l.x1 - l.x0, l.Y, l.yend,
                    string(round.(l.lines; digits=2)), s.N)
        end
        return
    end
    if "--smoke" in ARGS
        run_smoke(); return
    end
    if "--size" in ARGS
        mkpath(out); CSV.write(joinpath(out, "sizing.csv"), sizing(cs)); return
    end
    if "--refs" in ARGS
        mkpath(out)
        rr = NamedTuple[]; ss = NamedTuple[]
        for (i, p) in enumerate(cs)
            i >= 7 && continue
            t = @elapsed r = references(p)
            push!(rr, (case=i, name=NAMES[i], k0=r.kb, f_ray=r.fray, f_thin=r.fthin, f_gauss=r.fgauss, A_eik=imag(r.Φ0),
                       f_pe_untrunc=r.unt.f, A_pe_untrunc=r.unt.A, f_pe=r.tr.f, A_pe=r.tr.A, f_pe_spread=r.tr.f_spread,
                       f_pe_lines=join(round.(r.tr.fs; digits=3), ";")))
            for s in r.sens
                push!(ss, (; name=NAMES[i], s...))
            end
            @printf("%-4s ray %.2f thin %.2f PE(∞) %.2f PE(y_end) %.2f [%s]  𝒜 PE %.4f eik %.4f  max|Δf| %.2f%% max|Δ𝒜| %.2f%% (%.0f s)\n",
                    NAMES[i], r.fray, r.fthin, r.unt.f, r.tr.f, join(round.(r.tr.fs; digits=2), ", "), r.tr.A, imag(r.Φ0),
                    100maximum(abs(s.f_rel) for s in r.sens), 100maximum(abs(s.A_rel) for s in r.sens), t)
            flush(stdout)
        end
        CSV.write(joinpath(out, "references.csv"), DataFrame(rr)); CSV.write(joinpath(out, "y_sensitivity.csv"), DataFrame(ss))
        return
    end
    i = findfirst(==("--case"), ARGS)
    if i !== nothing
        for k in parse.(Int, split(ARGS[i + 1], ","))
            @info "case $(k) $(NAMES[k])" cs[k]
            d = simulate(cs[k]; force="--force" in ARGS)
            @printf("case %d done: N = %d, residual %.2e, wall %.0f s, maxrss %.2f GB\n", k, d["N"], d["rel_residual"],
                    d["wall_time_s"], d["maxrss_final_gb"])
        end
        return
    end
    # post-processing of the stored records
    path = joinpath(REPO, "data", "sims", TEST_ID)
    have = [isfile(joinpath(path, record_name(p) * ".jld2")) for p in cs]
    recs = [have[k] ? simulate(cs[k]) : nothing for k in eachindex(cs)]
    refs = Any[nothing for _ in cs]
    for k in eachindex(cs)
        have[k] || continue
        refs[k] = references(k >= 7 ? L4_4_Params(cs[k]) : cs[k])
    end
    df = DataFrame([row(k, cs[k], recs[k], refs[k]) for k in eachindex(cs) if have[k]])
    sens = CSV.read(joinpath(out, "y_sensitivity.csv"), DataFrame)
    sm = isfile(joinpath(out, "smoke_identities.csv")) ? first(CSV.read(joinpath(out, "smoke_identities.csv"), DataFrame)) : nothing
    ck = checks(df, sens, sm)
    CSV.write(joinpath(out, "lenses.csv"), df); CSV.write(joinpath(out, "checks.csv"), ck)
    plot_all(df, cs, recs, refs, joinpath(REPO, "plots", TEST_ID))
    show(stdout, MIME"text/plain"(), select(df, :name, :N, :f_fe, :f_pe, :f_pe_untrunc, :f_ray, :f_thin, :f_fe_spread, :A_fe, :A_pe,
                                            :A_eik, :frame_reflection, :residual, :wall_s); allrows=true)
    println()
    @printf("%d/%d checks pass (%d of %d cases present)\n", count(ck.passed), nrow(ck), count(have), length(cs))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-90s value = %.4g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
