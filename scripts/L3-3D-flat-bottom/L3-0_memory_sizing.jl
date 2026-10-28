#=
L3-0 — 3D memory sizing (first Phase-4 card).
Pre- and post-processing of src/L3_0_memory_sizing.jl (problem and
measurements are documented there).

Study: plate-covered fluid box, set P1, ω = 10, h = λ_p/8 horizontally, nz
graded cells; one case per Julia process (peak RSS is a process high-water
mark), each after a warm-up case that pays the compilation.
  - full cases (assemble, UMFPACK factorise, solve), as far as the 4 GB
    `run_limited` cap allows: square ladders nx = ny for p = 2, nz = 8;
    p = 3, nz = 8; p = 2, nz = 16; one channel case (nx ≫ ny, the shape of
    L3-1/L3-2); a few cases with UMFPACK's METIS ordering;
  - analysis cases (assemble + exact symbolic Cholesky fill and flops with AMD
    and METIS, no factorisation) on larger meshes, up to what the assembly
    alone allows under the cap, including some target meshes.

Model of the cost (calibrated on the full cases, see `calibration`):
  lunz_UMF ≈ κ_L (2 nnz(L_chol) − N)       (UMFPACK symmetric strategy ≈ Cholesky fill of A + Aᵀ)
  M_LU     ≈ β · lunz_UMF                  (UMFPACK peak, bytes per factor entry)
  peak RSS ≈ R₀ + r·N + M_LU               (R₀: Julia + compiled code after warm-up)
  t_fact+solve ≈ flops_chol / rate
and nnz(L_chol), flops_chol as power laws of N per (p, nz) on squares
(full + analysis cases), or ∝ N·w, N·w² (w = front width ∝ p²·ny·nz) for
channels. Extrapolation: the domains of L3-1, L3-2, L3-3, L4-4 and L4-8
(`targets`), judged against 4 GB (run_limited default) and 20 GB (usable on
the 24 GB machine).

Checks (sanity, not acceptance): DOFs equal the tensor-product formula; each
full run completed under the cap (peak RSS < 4 GB); solution finite and LU
residual ≤ 1e-10; on the channel case (the only one with a ≥ 1.5 λ_p window),
the y-uniform centreline wavenumber |k_h/k₀ − 1| ≤ 1e-2 (h = λ_p/8) and |η|
uniform across the width; calibration spread (κ_L, β, rate) ≤ 30 %.

Outputs (relative to the repository root):
  data/sims/L3-0/      one DrWatson record per case (git commit, HydroElasticFEM pin)
  data/exp_pro/L3-0/   sizing.csv (full cases), fill.csv (all cases), fits.csv,
                       calibration.csv, targets.csv, checks.csv
  plots/L3-0/          memory, fill and time against DOFs, with fits and caps (.pdf, .png)

Run (each case inside run_limited, one at a time):
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl --list      # case table
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl --case 3
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl             # post-processing
=#

using DrWatson
using DataFrames
using CSV
using Printf
using Statistics: median
using TOML
import HydroElasticFEM
using HydroElasticFEM_xxx2026.L3_0
using HydroElasticFEM_xxx2026.References: HostSpec, k0, kappa1

const TEST_ID = "L3-0"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CAP_GB = 4.0          # run_limited default
const MACHINE_GB = 20.0     # usable on the 24 GB machine
const CPU_CAP_S = 3600.0    # run_limited default CPU budget per process

# ─── 1. Cases ───────────────────────────────────────────────────────────────

"Full cases (factorised). Larger ones exceed 4 GB: p=2 24×24×16 (N = 81 634) was killed at 4.3 GB."
const LADDER = [
    # (order, nx, ny, nz)
    (2, 16, 16, 8), (2, 20, 20, 8), (2, 24, 24, 8), (2, 28, 28, 8), (2, 32, 32, 8),
    (3, 8, 8, 8), (3, 10, 10, 8), (3, 12, 12, 8), (3, 14, 14, 8), (3, 16, 16, 8),
    (2, 12, 12, 16), (2, 16, 16, 16), (2, 20, 20, 16),
    (2, 128, 16, 8),    # channel
]

"Full cases with UMFPACK's METIS ordering instead of the default (what a custom `solver` could buy)."
const LADDER_METIS = [(2, 24, 24, 8), (2, 32, 32, 8), (3, 12, 12, 8), (3, 16, 16, 8)]

"Analysis-only cases (assembly + symbolic fill): larger squares, channels and some target meshes (L3-1, L4-8)."
const LADDER_ANALYSE = [
    (2, 48, 48, 8), (2, 64, 64, 8), (2, 80, 80, 8), (2, 96, 96, 8),
    (3, 24, 24, 8), (3, 32, 32, 8), (3, 40, 40, 8), (3, 48, 48, 8),
    (2, 32, 32, 16), (2, 48, 48, 16), (2, 64, 64, 16),
    (2, 256, 16, 8), (2, 333, 16, 8),                      # channels (333×16×8 = L3-1 target, p=2, nz=8)
    (2, 76, 35, 8), (2, 91, 38, 8), (2, 108, 51, 8),       # L4-8 targets, nz = 8
    (2, 76, 35, 16), (2, 91, 38, 16),                      # L4-8 targets, nz = 16
]

"All cases as (params, mode), mode ∈ (:full, :analyse)."
cases() = [[(L3_0_Params(order=o, nx=nx, ny=ny, nz=nz), :full) for (o, nx, ny, nz) in LADDER];
           [(L3_0_Params(order=o, nx=nx, ny=ny, nz=nz, ordering=:metis), :full) for (o, nx, ny, nz) in LADDER_METIS];
           [(L3_0_Params(order=o, nx=nx, ny=ny, nz=nz), :analyse) for (o, nx, ny, nz) in LADDER_ANALYSE]]

shape(p) = p.nx == p.ny ? "square" : (min(p.nx, p.ny) <= 16 ? "channel" : "rectangle")

# ─── 2. Records ─────────────────────────────────────────────────────────────

record_name(p::L3_0_Params, mode::Symbol=:full) =
    savename(p; accesses=VARIABLE) * (p.ordering == :default ? "" : "_ordering=$(p.ordering)") *
    (mode == :full ? "" : "_mode=$(mode)") * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

"Compute (after a warm-up case) and save the record of case `p` in `mode`."
function simulate(p::L3_0_Params, mode::Symbol=:full; force::Bool=false)
    path = joinpath(REPO, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=q -> record_name(q, mode), tag=true, gitpath=REPO, storepatch=false,
                              force=force) do p
        t0 = time()
        warm = L3_0_Params(p; nx=48, ny=1, nz=2)         # compile everything (incl. the fit) on a tiny channel
        mode == :full ? run_case(warm; verbose=false) : analyse_case(warm; verbose=false)
        GC.gc()
        warm_s = time() - t0
        base_max, base_rss = L3_0.maxrss_gb(), L3_0.rss_gb()
        @printf("[warm-up] %.1f s  maxrss %.3f GB  rss %.3f GB\n", warm_s, base_max, base_rss)
        d = mode == :full ? run_case(p) : analyse_case(p)
        d["mode"] = String(mode)
        d["warmup_s"] = warm_s
        d["baseline_maxrss_gb"] = base_max
        d["baseline_rss_gb"] = base_rss
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L3_0_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["threads_blas"] = get(ENV, "OPENBLAS_NUM_THREADS", "unset")
        d["mem_limit_gb"] = get(ENV, "LRH_MEM_LIMIT_GB", "unset")
        return d
    end
    return data
end

# ─── 3. Tables, fits and calibration ────────────────────────────────────────

"RSS after warm-up (GB); the high-water mark when `ps` was unavailable."
base_rss(d) = isfinite(d["baseline_rss_gb"]) ? d["baseline_rss_gb"] : d["baseline_maxrss_gb"]

"Saved records as (params, mode, record); cases not yet computed are skipped."
function records()
    path = joinpath(REPO, "data", "sims", TEST_ID)
    out = Tuple{L3_0_Params,Symbol,Dict}[]
    for (p, mode) in cases()
        f = joinpath(path, record_name(p, mode) * ".jld2")
        isfile(f) && push!(out, (p, mode, wload(f)))
    end
    return out
end

"Full (factorised) cases."
function sizing_table(recs)
    rows = NamedTuple[]
    for (p, mode, d) in recs
        mode == :full || continue
        push!(rows, (order=p.order, nx=p.nx, ny=p.ny, nz=p.nz, shape=shape(p), ordering=String(p.ordering),
                     h_over_lambda=1 / p.elements_per_wavelength,
                     Lx_m=d["Lx"], Ly_m=d["Ly"], ndofs=d["ndofs"], ndofs_formula=d["ndofs_formula"],
                     ndofs_phi=d["ndofs_fields"][1], ndofs_eta=d["ndofs_fields"][2], nnzA=d["nnzA"],
                     lunz=d["lunz"], fill_ratio=d["lunz"] / d["nnzA"],
                     chol_lnz_amd=get(d, "chol_lnz_amd", NaN), chol_lnz_metis=get(d, "chol_lnz_metis", NaN),
                     chol_fl_amd=get(d, "chol_fl_amd", NaN), chol_fl_metis=get(d, "chol_fl_metis", NaN),
                     matrix_gb=d["matrix_gb"], umf_numeric_gb=d["umf_numeric_actual_gb"],
                     umf_peak_gb=d["umf_peak_actual_gb"], umf_peak_estimate_gb=d["umf_peak_estimate_gb"],
                     baseline_rss_gb=base_rss(d), baseline_maxrss_gb=d["baseline_maxrss_gb"],
                     peak_rss_gb=d["maxrss_final_gb"], case_mem_gb=d["maxrss_final_gb"] - base_rss(d),
                     rss_after_assembly_gb=d["assembly_rss_gb"], maxrss_after_numeric_gb=d["numeric_maxrss_gb"],
                     warmup_s=d["warmup_s"], assembly_s=d["assembly_wall_s"], symbolic_s=d["symbolic_wall_s"],
                     numeric_s=d["numeric_wall_s"], solve_s=d["solve_wall_s"],
                     factor_solve_s=d["symbolic_wall_s"] + d["numeric_wall_s"] + d["solve_wall_s"],
                     umf_flops=d["umf_flops"], umf_ordering_used=d["umf_ordering_used"],
                     rel_residual=d["rel_residual"], solution_finite=d["solution_finite"],
                     dispersion_error=d["dispersion_error"], fit_residual=d["fit_residual"],
                     reflection=d["reflection"], y_nonuniformity=d["y_nonuniformity"],
                     window_wavelengths=d["window_wavelengths"]))
    end
    return sort!(DataFrame(rows), [:ordering, :shape, :order, :nz, :ndofs])
end

"Symbolic fill of all default-ordering cases (full and analysis)."
function fill_table(recs)
    rows = NamedTuple[]
    for (p, mode, d) in recs
        p.ordering == :default && haskey(d, "chol_lnz_amd") || continue
        push!(rows, (order=p.order, nx=p.nx, ny=p.ny, nz=p.nz, shape=shape(p), mode=String(mode), ndofs=d["ndofs"],
                     ndofs_formula=d["ndofs_formula"], nnzA=d["nnzA"],
                     chol_lnz_amd=d["chol_lnz_amd"], chol_lnz_metis=d["chol_lnz_metis"],
                     chol_fl_amd=d["chol_fl_amd"], chol_fl_metis=d["chol_fl_metis"],
                     assembly_s=d["assembly_wall_s"], analyse_s=d["analyse_wall_s"],
                     baseline_rss_gb=base_rss(d), peak_rss_gb=d["maxrss_final_gb"],
                     analyse_peak_gb=d["analyse_maxrss_gb"], assembly_mem_gb=d["assembly_maxrss_gb"] - base_rss(d)))
    end
    return sort!(DataFrame(rows), [:shape, :order, :nz, :ndofs])
end

"Least-squares fit log y = log C + α log N (returns C, α)."
function powerfit(N, y)
    X = [ones(length(N)) log.(N)]
    c = X \ log.(y)
    return (C=exp(c[1]), α=c[2])
end

"""
Power laws C·N^α:
- measured, per square full ladder (ordering, p, nz), all points: case memory
  (peak RSS − RSS after warm-up), UMFPACK peak, lunz, assembly and
  factorisation+solve times;
- fill, per square ladder (p, nz) of all default-ordering cases with N ≥ `nmin`:
  nnz(L) and flops with AMD and METIS; assembly time.
"""
function fits(df, fl; nmin::Int=30_000)
    rows = NamedTuple[]
    push_fit!(kind, ordering, o, nz, q, s, col) = begin
        f = powerfit(s.ndofs, s[!, col])
        push!(rows, (kind=kind, ordering=ordering, order=o, nz=nz, quantity=q, C=f.C, exponent=f.α,
                     n_points=nrow(s), N_min=minimum(s.ndofs), N_max=maximum(s.ndofs)))
    end
    for g in groupby(filter(r -> r.shape == "square", df), [:ordering, :order, :nz])
        nrow(g) >= 2 || continue
        for q in (:case_mem_gb, :umf_peak_gb, :lunz, :assembly_s, :factor_solve_s)
            push_fit!("measured", first(g.ordering), first(g.order), first(g.nz), String(q), g, q)
        end
    end
    for g in groupby(filter(r -> r.shape == "square" && r.ndofs >= nmin, fl), [:order, :nz])
        nrow(g) >= 2 || continue
        for q in (:chol_lnz_amd, :chol_fl_amd, :chol_lnz_metis, :chol_fl_metis, :assembly_s)
            push_fit!("fill", "-", first(g.order), first(g.nz), String(q), g, q)
        end
    end
    return DataFrame(rows)
end

"""
Calibration of the cost model, per ordering, on the square full cases with
N ≥ `nmin` (asymptotic regime; AMD fill for the default ordering, METIS fill for
METIS): κ_L = lunz/(2 nnz(L) − N), β = UMFPACK peak bytes per factor entry,
rate = flops_chol/(fact.+solve time); median, range and relative spread
(max/min − 1). r (bytes per DOF besides the LU: matrix, FE structures,
assembly transients) is the assembly memory per DOF (peak RSS at the end of the
assembly − RSS after warm-up) of the analysis cases, per p: conservative, since
part of it is freed before the factorisation. (The process peak RSS minus the
UMFPACK peak is too noisy for r: GC timing and RSS reuse make it even negative.)
R₀ = RSS after warm-up (max over all cases).
"""
function calibration(df, fl; nmin::Int=30_000)
    rows = NamedTuple[]
    stat!(o, q, v) = (v = filter(isfinite, v); isempty(v) ||
        push!(rows, (ordering=o, quantity=q, value=median(v), min=minimum(v), max=maximum(v),
                     spread=maximum(v) / minimum(v) - 1, n_points=length(v))))
    for g in groupby(filter(r -> r.shape == "square" && r.ndofs >= nmin, df), :ordering)
        o = first(g.ordering)
        lnz = o == "metis" ? g.chol_lnz_metis : g.chol_lnz_amd
        fl_ = o == "metis" ? g.chol_fl_metis : g.chol_fl_amd
        stat!(o, "kappa_L", g.lunz ./ (2 .* lnz .- g.ndofs))
        stat!(o, "beta_bytes_per_entry", g.umf_peak_gb .* 1024^3 ./ g.lunz)
        stat!(o, "rate_flops_per_s", fl_ ./ g.factor_solve_s)
    end
    an = filter(r -> r.mode == "analyse", fl)
    for g in groupby(an, :order)
        stat!("p=$(first(g.order))", "r_bytes_per_dof", g.assembly_mem_gb .* 1024^3 ./ g.ndofs)
    end
    stat!("all", "R0_gb", vcat(df.baseline_rss_gb, fl.baseline_rss_gb))
    return DataFrame(rows)
end

calval(cal, ordering, q) = (r = filter(r -> r.ordering == ordering && r.quantity == q, cal); nrow(r) == 1 ? only(r.value) : NaN)

function fitted(ft, kind, order, nz, quantity; ordering="-")
    r = filter(r -> r.kind == kind && r.ordering == ordering && r.order == order && r.nz == nz && r.quantity == quantity, ft)
    nrow(r) == 1 || return nothing
    return first(eachrow(r))
end
predict(ft, kind, order, nz, quantity, N; ordering="-") =
    (f = fitted(ft, kind, order, nz, quantity; ordering=ordering); f === nothing ? NaN : f.C * N^f.exponent)

"""
Peak RSS (GB) and factorisation+solve time (s) of a case with N DOFs and
Cholesky fill `lnz`, flops `fl`, from the calibration of `ordering`.
"""
function cost(cal, N, lnz, fl, order; ordering="default")
    κ, β = calval(cal, ordering, "kappa_L"), calval(cal, ordering, "beta_bytes_per_entry")
    r = calval(cal, "p=$order", "r_bytes_per_dof")
    R0, rate = calval(cal, "all", "R0_gb"), calval(cal, ordering, "rate_flops_per_s")
    lunz = κ * (2lnz - N)
    mlu = β * lunz / 1024^3
    return (lunz=lunz, lu_gb=mlu, peak=R0 + r * N / 1024^3 + mlu, t=fl / rate)
end

# ─── 4. Target domains of the 3D tests ──────────────────────────────────────

const P1 = HostSpec(g=9.8, ρ=1000.0, H=10.0, m_ρ=0.01, C_ρ=0.49)
const OPEN = HostSpec(g=9.8, ρ=1000.0, H=10.0, m_ρ=0.0, C_ρ=0.0)

"Cells on [0, len] with sizes from `hmin` (at one end) growing by `ratio` to `hfar` (VT3D-B `graded_sizes`)."
function graded_cells(len, hmin, hfar, ratio)
    n, s, h = 0, 0.0, hmin
    while h < hfar && s + h < len
        n += 1; s += h; h *= ratio
    end
    return n + ceil(Int, max(len - s, 0.0) / hfar - 1e-9)
end
cells(len, h) = max(1, ceil(Int, len / h - 1e-9))

"""
Target configurations: horizontal cells (tensor-product mesh), nz, p, symmetry
and number of factorisations of each test. Lengths follow the test matrix
(§0.3: absorbing zones ≥ 4λ, gap ≥ max(λ/2, 5/κ₁)) unless the variant says
otherwise; `matrix` marks the domain written in the matrix row.
"""
function targets()
    T = NamedTuple[]
    add!(test, variant, nx, ny; nz=8, order=2, symmetry="none", solves=1, shape="square", note="") =
        push!(T, (test=test, variant=variant, nx=nx, ny=ny, nz=nz, order=order, symmetry=symmetry,
                  solves=solves, shape=shape, note=note))
    ω = 10.0
    λ = 2π / k0(P1, ω)                   # 2.29 m
    gap = 5 / kappa1(P1, ω)              # 31.6 m = 13.8 λ_p
    for (epw, o) in ((8, 2), (16, 2), (8, 3), (12, 3)), nz in (8, 16)
        o == 3 && nz == 16 && continue
        h = λ / epw
        res = "p=$o, h=λp/$epw, nz=$nz"
        # L3-1(a): oblique plane wave, periodic in y with W = λ_p/sinθ (θ = 30°), sponges 4λ_p, gaps 5/κ₁, window 6λ_p
        Lx = 2 * 4λ + 2gap + 6λ
        add!("L3-1", "θ=30°, W=2λp, $res", cells(Lx, h), cells(2λ, h); nz=nz, order=o, solves=3, shape="channel",
             note="3 ω ∈ {2,5,10} (ω=10 largest); θ=0 is one cell wide; θ=60° narrower")
        # L3-3: plate-covered square around one resonator, absorbing frame 4λ_p; quarter symmetry (radiation problem)
        rref = 4                         # extra graded cells near the resonator (h_min = λp/12 … λp/48 in R = λp/4)
        add!("L3-3", "matrix 16λp square + 4λp frame, quarter, $res", cells(12λ, h) + rref, cells(12λ, h) + rref;
             nz=nz, order=o, symmetry="quarter", solves=41,
             note="41 ω; one bare-plate point-load solve per ω gives G_h(0) for all ζ")
        add!("L3-3", "gap 5/κ₁ + 4λp frame, quarter, $res", cells(gap + 4λ, h) + rref, cells(gap + 4λ, h) + rref;
             nz=nz, order=o, symmetry="quarter", solves=41, note="§0.3-compliant gap")
        # L4-4: island (min side 6λ_p, aspect 2: 12λ_p × 6λ_p) + gap + 4λ_p frame; half symmetry (incidence +x)
        for (gname, gg) in (("gap 5/κ₁", gap), ("gap 2λp", 2λ))
            add!("L4-4", "island 12×6λp, $gname, half, $res", cells(12λ + 2gg + 8λ, h), cells(3λ + gg + 4λ, h);
                 nz=nz, order=o, symmetry="half (y)", solves=24,
                 note="aspect {½,1,2} × ω {9.5,10} × ζ {0.02,0.05} × size {3,6}; +y incidence = rotated aspect")
        end
    end
    # L3-2: VT3D-B channel (set P3, frozen mesh rules): open water λ0/6.4, plate λp/8 graded to h_min, W = 2 m, nz = 8
    g3 = 9.80665
    P3 = HostSpec(g=g3, ρ=1000.0, H=10.0, m_ρ=0.01, C_ρ=0.05g3)
    O3 = HostSpec(g=g3, ρ=1000.0, H=10.0, m_ρ=0.0, C_ρ=0.0)
    λ0, λp = 2π / k0(O3, 10.0), 2π / k0(P3, 10.0)
    for (lev, r) in (("L1", 12), ("L2", 24), ("L3", 48))
        hmin = λp / r
        nx = 2 * (cells(4λ0, λ0 / 6.4) + cells(0.5λ0, λ0 / 6.4)) + 2 * graded_cells(2λp, hmin, λp / 8, 1.5)
        ny = 2 * graded_cells(1.0, hmin, 2 / 7, 1.5)
        add!("L3-2", "VT3D-B level $lev (h_min=λp/$r), p=2, nz=8", nx, ny; solves=6, shape="channel",
             note="S1–S4, S1_09, S2_09 per level (+S5, PG at L2)")
    end
    # L4-8: finite plate 11.5 m × 4.6 m in open water (L2-5 sizes, H = 10), forced response near the two
    # first modes (ω₁ ≈ 2.74 along x, 4.74 along y); open water λ0/6.4, plate λp/8, zones 4λ0 on all sides,
    # half symmetry for incidence along x (and along y: other half-plane)
    for ωf in (2.74, 4.74), (gname, gfun) in (("gap 5/κ₁", (λ0f, κ) -> max(λ0f / 2, 5 / κ)), ("gap λ0", (λ0f, κ) -> λ0f))
        λ0f = 2π / k0(OPEN, ωf)
        λpf = 2π / k0(P1, ωf)
        κf = kappa1(OPEN, ωf)
        gg = gfun(λ0f, κf)
        h0 = λ0f / 6.4
        hp = min(λpf / 8, h0)
        nx = cells(11.5, hp) + 2 * cells(gg + 4λ0f, h0)
        ny = cells(2.3, hp) + cells(gg + 4λ0f, h0)
        add!("L4-8", @sprintf("plate 11.5×4.6 m, ω=%.2f, %s, half, p=2, nz=8", ωf, gname), nx, ny;
             symmetry="half", solves=40, note="2 incidences × ~20 ω around the split peaks")
        add!("L4-8", @sprintf("plate 11.5×4.6 m, ω=%.2f, %s, half, p=2, nz=16", ωf, gname), nx, ny; nz=16,
             symmetry="half", solves=40, note="2 incidences × ~20 ω around the split peaks")
    end
    return DataFrame(T)
end

verdict(peak) = isnan(peak) ? "no fit" : peak <= CAP_GB ? "fits 4 GB cap" : peak <= MACHINE_GB ? "needs 24 GB machine" :
                "needs iterative solver / symmetry reduction / cluster"

"Cholesky fill and flops (AMD, METIS) of target `t` with N DOFs, and how they were obtained."
function target_fill(ft, fl, t, N)
    m = filter(r -> r.order == t.order && r.nx == t.nx && r.ny == t.ny && r.nz == t.nz, fl)
    if nrow(m) > 0
        r = first(eachrow(m))
        return (r.chol_lnz_amd, r.chol_fl_amd, r.chol_lnz_metis, r.chol_fl_metis, r.assembly_s, "analysed mesh")
    end
    ch = filter(r -> r.shape == "channel", fl)
    if t.shape == "channel" && nrow(ch) > 0
        # banded factors: fill ∝ N·w, flops ∝ N·w², w ∝ p²·ny·nz (front width in DOFs)
        c = last(sort(ch, :ndofs))
        w = (t.order * t.ny) * (t.order * t.nz) / ((c.order * c.ny) * (c.order * c.nz))
        s = N / c.ndofs
        return (c.chol_lnz_amd * s * w, c.chol_fl_amd * s * w^2, c.chol_lnz_metis * s * w, c.chol_fl_metis * s * w^2,
                c.assembly_s * s, "channel $(c.nx)×$(c.ny)×$(c.nz) scaled ∝ N·w, N·w²")
    end
    pr(q) = predict(ft, "fill", t.order, t.nz, q, N)
    return (pr("chol_lnz_amd"), pr("chol_fl_amd"), pr("chol_lnz_metis"), pr("chol_fl_metis"), pr("assembly_s"),
            "square fill law (p=$(t.order), nz=$(t.nz))")
end

"Target table with predicted DOFs, peak memory, time per factorisation and verdict."
function target_table(ft, cal, fl)
    rows = NamedTuple[]
    for t in eachrow(targets())
        N = dof_formula(t.order, t.nx, t.ny, t.nz)
        la, fa, lm, fm, ta, method = target_fill(ft, fl, t, N)
        ca = cost(cal, N, la, fa, t.order; ordering="default")
        cm = cost(cal, N, lm, fm, t.order; ordering="metis")
        push!(rows, (test=t.test, variant=t.variant, nx=t.nx, ny=t.ny, nz=t.nz, order=t.order, symmetry=t.symmetry,
                     ndofs=N, chol_lnz_amd=la, lu_gb=ca.lu_gb, peak_rss_gb=ca.peak, assembly_s=ta,
                     factor_solve_s=ca.t, solves=t.solves, total_h=t.solves * (ta + ca.t) / 3600,
                     chol_lnz_metis=lm, peak_rss_metis_gb=cm.peak, factor_solve_metis_s=cm.t,
                     total_metis_h=t.solves * (ta + cm.t) / 3600,
                     verdict=verdict(ca.peak), verdict_metis=verdict(cm.peak),
                     one_solve_within_60_cpu_min=(ta + ca.t) <= CPU_CAP_S, method=method, note=t.note))
    end
    return DataFrame(rows)
end

# ─── 5. Checks ──────────────────────────────────────────────────────────────

function checks(df, fl, ft, cal)
    rows = NamedTuple[]
    add!(name, value, tol, passed) = push!(rows, (name=name, value=Float64(value), tolerance=Float64(tol), passed=passed))
    for r in eachrow(df)
        tag = @sprintf("p=%d %d×%d×%d %s", r.order, r.nx, r.ny, r.nz, r.ordering)
        add!("DOFs = (p·nx+1)(p·ny+1)(p·nz+1) + (p·nx+1)(p·ny+1), $tag", abs(r.ndofs - r.ndofs_formula), 0, r.ndofs == r.ndofs_formula)
        add!("peak RSS under the 4 GB cap, $tag", r.peak_rss_gb, CAP_GB, r.peak_rss_gb < CAP_GB)
        add!("solution finite and LU residual, $tag", r.rel_residual, 1e-10, r.solution_finite && r.rel_residual <= 1e-10)
    end
    for r in eachrow(filter(r -> r.mode == "analyse", fl))
        tag = @sprintf("p=%d %d×%d×%d (analysis)", r.order, r.nx, r.ny, r.nz)
        add!("DOFs = tensor-product formula, $tag", abs(r.ndofs - r.ndofs_formula), 0, r.ndofs == r.ndofs_formula)
        add!("peak RSS (Sys.maxrss) under the 4 GB cap, $tag", r.peak_rss_gb, CAP_GB, r.peak_rss_gb < CAP_GB)
    end
    # the cost model against the measured peak RSS and time of every full case (model should not be optimistic)
    for r in eachrow(df)
        isfinite(r.chol_lnz_amd) || continue
        o = r.ordering
        c = cost(cal, r.ndofs, o == "metis" ? r.chol_lnz_metis : r.chol_lnz_amd,
                 o == "metis" ? r.chol_fl_metis : r.chol_fl_amd, r.order; ordering=o)
        tag = @sprintf("p=%d %d×%d×%d %s", r.order, r.nx, r.ny, r.nz, o)
        add!("model peak RSS / measured ∈ [0.9, 1.6], $tag", c.peak / r.peak_rss_gb, 1.6,
             0.9 <= c.peak / r.peak_rss_gb <= 1.6)
        add!("model fact.+solve time / measured ∈ [0.7, 1.4], $tag", c.t / r.factor_solve_s, 1.4,
             0.7 <= c.t / r.factor_solve_s <= 1.4)
    end
    for r in eachrow(filter(r -> !isnan(r.dispersion_error), df))
        tag = @sprintf("p=%d %d×%d×%d %s (h=λp/8, window %.1f λp)", r.order, r.nx, r.ny, r.nz, r.ordering, r.window_wavelengths)
        add!("|k_h/k₀ − 1| on the centreline, $tag", abs(r.dispersion_error), 1e-2, abs(r.dispersion_error) <= 1e-2)
        add!("y-uniformity of η (ν = 0, free edges), $tag", r.y_nonuniformity, 1e-6, r.y_nonuniformity <= 1e-6)
    end
    for c in eachrow(filter(r -> r.ordering == "default" && r.quantity in ("kappa_L", "beta_bytes_per_entry",
                                                                          "rate_flops_per_s"), cal))
        add!("calibration spread of $(c.quantity) (default ordering)", c.spread, 0.3, c.spread <= 0.3)
    end
    # the square fill law applied to the analysed rectangles (L4-8 meshes): is the aspect ratio harmless?
    for r in eachrow(filter(r -> r.shape == "rectangle", fl))
        pred = predict(ft, "fill", r.order, r.nz, "chol_lnz_amd", r.ndofs)
        isnan(pred) && continue
        add!(@sprintf("square fill law on rectangle p=%d %d×%d×%d: |pred/actual − 1| of nnz(L)", r.order, r.nx, r.ny, r.nz),
             abs(pred / r.chol_lnz_amd - 1), 0.3, abs(pred / r.chol_lnz_amd - 1) <= 0.3)
    end
    return DataFrame(rows)
end

# ─── 6. Figure ──────────────────────────────────────────────────────────────

function plot_sizing(df, fl, ft, cal, tt, figdir)
    @eval using CairoMakie
    mkpath(figdir)
    Base.invokelatest() do
        fig = Figure(; size=(1250, 400))
        groups = [(2, 8, :black, "p=2, nz=8"), (3, 8, :firebrick, "p=3, nz=8"), (2, 16, :royalblue, "p=2, nz=16")]
        axm = Axis(fig[1, 1]; xscale=log10, yscale=log10, xlabel="DOFs N", ylabel="process peak RSS (GB)",
                   title="peak memory (dashed: model, AMD; dotted: METIS)")
        axl = Axis(fig[1, 2]; xscale=log10, yscale=log10, xlabel="DOFs N", ylabel="nnz(L) of A + Aᵀ",
                   title="symbolic Cholesky fill (o AMD, △ METIS)")
        axt = Axis(fig[1, 3]; xscale=log10, yscale=log10, xlabel="DOFs N", ylabel="factorisation + solve (s)",
                   title="time, 1 thread (dashed: model)")
        Nl = 10 .^ range(4, 7.6; length=60)
        for (o, nz, c, lab) in groups
            s = filter(r -> r.shape == "square" && r.ordering == "default" && r.order == o && r.nz == nz, df)
            nrow(s) > 0 && scatter!(axm, s.ndofs, s.peak_rss_gb; color=c, label=lab)
            nrow(s) > 0 && scatter!(axt, s.ndofs, s.factor_solve_s; color=c, label=lab)
            f = filter(r -> r.shape == "square" && r.order == o && r.nz == nz, fl)
            nrow(f) > 0 && scatter!(axl, f.ndofs, f.chol_lnz_amd; color=c, label=lab)
            nrow(f) > 0 && scatter!(axl, f.ndofs, f.chol_lnz_metis; color=c, marker=:utriangle)
            la = [predict(ft, "fill", o, nz, "chol_lnz_amd", N) for N in Nl]
            lm = [predict(ft, "fill", o, nz, "chol_lnz_metis", N) for N in Nl]
            fa = [predict(ft, "fill", o, nz, "chol_fl_amd", N) for N in Nl]
            all(isnan, la) && continue
            lines!(axl, Nl, la; color=c, linestyle=:dash)
            lines!(axl, Nl, lm; color=c, linestyle=:dot)
            lines!(axm, Nl, [cost(cal, N, l, 1.0, o).peak for (N, l) in zip(Nl, la)]; color=c, linestyle=:dash)
            lines!(axm, Nl, [cost(cal, N, l, 1.0, o; ordering="metis").peak for (N, l) in zip(Nl, lm)]; color=c, linestyle=:dot)
            lines!(axt, Nl, [cost(cal, N, l, f, o).t for (N, l, f) in zip(Nl, la, fa)]; color=c, linestyle=:dash)
        end
        sm = filter(r -> r.ordering == "metis", df)
        nrow(sm) > 0 && scatter!(axm, sm.ndofs, sm.peak_rss_gb; color=[o == 2 ? :black : :firebrick for o in sm.order],
                                 marker=:utriangle, label="METIS ordering")
        ch = filter(r -> r.shape == "channel", df)
        nrow(ch) > 0 && scatter!(axm, ch.ndofs, ch.peak_rss_gb; color=:gray, marker=:diamond, label="channel")
        scatter!(axm, tt.ndofs, clamp.(tt.peak_rss_gb, 0.5, 1e4); color=:transparent, strokecolor=:gray, strokewidth=1,
                 marker=:rect, markersize=7, label="3D targets (model)")
        hlines!(axm, [CAP_GB, MACHINE_GB]; color=[:orange, :red], linestyle=:solid, linewidth=1)
        text!(axm, 1.2e4, CAP_GB; text="4 GB cap", align=(:left, :bottom), fontsize=11)
        text!(axm, 1.2e4, MACHINE_GB; text="20 GB (24 GB machine)", align=(:left, :bottom), fontsize=11)
        hlines!(axt, [CPU_CAP_S]; color=:orange, linewidth=1)
        text!(axt, 1.2e4, CPU_CAP_S; text="60 CPU-min", align=(:left, :bottom), fontsize=11)
        ylims!(axm, 1, 1e4)
        axislegend(axm; position=:lt, framevisible=false, labelsize=10)
        axislegend(axl; position=:lt, framevisible=false, labelsize=10)
        save(joinpath(figdir, "sizing.pdf"), fig)
        save(joinpath(figdir, "sizing.png"), fig; px_per_unit=2)
    end
    return nothing
end

# ─── 7. Driver ──────────────────────────────────────────────────────────────

function main()
    recs = records()
    isempty(recs) && error("no L3-0 records: run the cases first (--case i)")
    df = sizing_table(recs)
    fl = fill_table(recs)
    ft = fits(df, fl)
    cal = calibration(df, fl)
    tt = target_table(ft, cal, fl)
    ck = checks(df, fl, ft, cal)
    out = joinpath(REPO, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "sizing.csv"), df)
    CSV.write(joinpath(out, "fill.csv"), fl)
    CSV.write(joinpath(out, "fits.csv"), ft)
    CSV.write(joinpath(out, "calibration.csv"), cal)
    CSV.write(joinpath(out, "targets.csv"), tt)
    CSV.write(joinpath(out, "checks.csv"), ck)
    plot_sizing(df, fl, ft, cal, tt, joinpath(REPO, "plots", TEST_ID))
    show(stdout, MIME"text/plain"(), select(df, :ordering, :order, :nx, :ny, :nz, :ndofs, :nnzA, :lunz, :fill_ratio,
                                            :umf_peak_gb, :peak_rss_gb, :case_mem_gb, :assembly_s, :factor_solve_s);
         allrows=true, allcols=true)
    println()
    show(stdout, MIME"text/plain"(), select(fl, :mode, :order, :nx, :ny, :nz, :ndofs, :chol_lnz_amd, :chol_lnz_metis,
                                            :chol_fl_amd, :chol_fl_metis, :assembly_s, :peak_rss_gb);
         allrows=true, allcols=true)
    println()
    show(stdout, MIME"text/plain"(), ft; allrows=true, allcols=true)
    println()
    show(stdout, MIME"text/plain"(), cal; allrows=true, allcols=true)
    println()
    show(stdout, MIME"text/plain"(), select(tt, :test, :variant, :ndofs, :peak_rss_gb, :peak_rss_metis_gb, :factor_solve_s,
                                            :factor_solve_metis_s, :total_h, :total_metis_h, :verdict, :verdict_metis, :method);
         allrows=true, allcols=true, truncate=0)
    println()
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-70s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (sizing=df, fill=fl, fits=ft, calibration=cal, targets=tt, checks=ck)
end

if abspath(PROGRAM_FILE) == @__FILE__
    cs = cases()
    i = findfirst(==("--case"), ARGS)
    if "--list" in ARGS
        for (j, (p, mode)) in enumerate(cs)
            @printf("%2d  %-7s %-7s p=%d  %3d×%3d×%2d  N=%9d\n", j, mode, p.ordering, p.order, p.nx, p.ny, p.nz,
                    dof_formula(p.order, p.nx, p.ny, p.nz))
        end
    elseif i !== nothing
        j = parse(Int, ARGS[i + 1])
        simulate(cs[j]...; force="--force" in ARGS)
    else
        main()
    end
end
