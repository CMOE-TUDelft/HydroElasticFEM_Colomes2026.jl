#=
L3-0b — cost of `Solvers.MetisLUSolver` (UMFPACK LU, METIS ordering) on the
L3-0 ladder, run end to end through HydroElasticFEM's `config.solver` hook.

Cases (L3-0 problem, set P1, ω = 10, h = λ_p/8): p = 2 24×24×8, p = 2 32×32×8,
p = 3 16×16×8 (all three have a METIS row and a measured CHOLMOD fill in
L3-0), and p = 2 24×24×16 (N = 81.6 k; killed above 4 GiB with the default
solver in L3-0, no fill measured). One case per Julia process, after a warm-up
case that pays the compilation. Per case:
  - `S.simulate(problem)` with `FreqDomainConfig(solver=MetisLUSolver())`:
    wall time and the process peak RSS (`Sys.maxrss`) afterwards;
  - a second factorisation with `factorize` (UMFPACK statistics: fill lunz,
    UMFPACK peak, ordering used), then solves with the stored factors
    (one and four right-hand sides) and the residual.
The L3-0 METIS projections are recomputed from the L3-0 outputs:
  lunz = κ_L (2 nnz(L) − N), M_LU = β·lunz, peak = R₀ + r N + M_LU, t = flops/rate
with the METIS calibration (κ_L, β, rate of `calibration.csv`), nnz(L) and
flops from `fill.csv` (same mesh) or from the METIS fill law of `fits.csv`.

Run (each case inside run_limited, under the heavy-run lock):
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-0b_metis_solver.jl --case 1
  julia --project -O1 scripts/L3-3D-flat-bottom/L3-0b_metis_solver.jl            # table
Outputs: data/sims/L3-0b/ (records), data/exp_pro/L3-0b/metis_solver.csv.
=#

using DrWatson
using DataFrames
using CSV
using Printf
using LinearAlgebra: norm
using Gridap: get_matrix, get_vector, get_free_dof_values
import HydroElasticFEM.Simulation as S
using HydroElasticFEM_xxx2026.Solvers
using HydroElasticFEM_xxx2026.L3_0

const TEST_ID = "L3-0b"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CASES = [(2, 24, 24, 8), (2, 32, 32, 8), (3, 16, 16, 8), (2, 24, 24, 16)]

params(i) = ((o, nx, ny, nz) = CASES[i]; L3_0_Params(order=o, nx=nx, ny=ny, nz=nz, ordering=:metis))
record_name(p) = savename(p; accesses=VARIABLE) * "_solver=MetisLUSolver_fixed=" * fixed_hash(p)

function measure(p::L3_0_Params)
    rec = Dict{String,Any}()
    solver = MetisLUSolver()
    t0 = time()
    warm = L3_0_Params(p; nx=48, ny=1, nz=2)
    S.simulate(first(L3_0.build(warm; solver=solver)))
    GC.gc()
    rec["warmup_s"] = time() - t0
    rec["baseline_maxrss_gb"] = L3_0.maxrss_gb()
    problem = L3_0.phase!(() -> first(L3_0.build(p; solver=solver)), rec, "assembly")
    res = L3_0.phase!(() -> S.simulate(problem), rec, "simulate")      # factorise + solve through config.solver
    rec["peak_rss_gb"] = L3_0.maxrss_gb()
    A, b = get_matrix(res.op), get_vector(res.op)
    x = get_free_dof_values(res.solution)
    rec["ndofs"] = size(A, 1)
    rec["ndofs_formula"] = dof_formula(p.order, p.nx, p.ny, p.nz)
    rec["nnzA"] = length(A.nzval)
    rec["rel_residual"] = norm(A * x - b) / norm(b)
    res = nothing; x = nothing; problem = nothing
    GC.gc()
    ns = L3_0.phase!(() -> factorize(solver, A), rec, "factorize")
    st = factor_stats(ns)
    rec["lunz"], rec["umf_peak_gb"], rec["umf_numeric_gb"] = st.lunz, st.peak_gb, st.numeric_gb
    rec["umf_flops"], rec["ordering_used"] = st.flops, String(st.ordering)
    x1 = L3_0.phase!(() -> solve_rhs(ns, b), rec, "solve1")
    B = hcat(b, im .* b, conj.(b), ones(ComplexF64, length(b)))
    X = L3_0.phase!(() -> solve_rhs(ns, B), rec, "solve4")
    rec["rel_residual_multi"] = maximum(norm(A * X[:, j] - B[:, j]) / norm(B[:, j]) for j in axes(B, 2))
    rec["rel_diff_resolve"] = norm(x1 - X[:, 1]) / norm(x1)
    rec["maxrss_final_gb"] = L3_0.maxrss_gb()
    rec["mem_limit_gb"] = get(ENV, "LRH_MEM_LIMIT_GB", "unset")
    return rec
end

function run_one(i::Int)
    p = params(i)
    path = joinpath(REPO, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO, storepatch=false) do p
        measure(p)
    end
    @printf("[L3-0b] case %d: N = %d  simulate %.1f s  peak RSS %.2f GB  lunz %.3g  ordering %s\n", i,
            data["ndofs"], data["simulate_wall_s"], data["peak_rss_gb"], data["lunz"], data["ordering_used"])
end

"L3-0 METIS projection for a case (order, nx, ny, nz, N)."
function projection(o, nx, ny, nz, N)
    d0 = joinpath(REPO, "data", "exp_pro", "L3-0")
    cal = CSV.read(joinpath(d0, "calibration.csv"), DataFrame)
    getc(ord, q) = only(cal[(cal.ordering .== ord) .& (cal.quantity .== q), :value])
    κ, β, rate = getc("metis", "kappa_L"), getc("metis", "beta_bytes_per_entry"), getc("metis", "rate_flops_per_s")
    r = getc("p=$o", "r_bytes_per_dof"); R0 = getc("all", "R0_gb")
    fill = CSV.read(joinpath(d0, "fill.csv"), DataFrame)
    row = fill[(fill.order .== o) .& (fill.nx .== nx) .& (fill.ny .== ny) .& (fill.nz .== nz), :]
    if nrow(row) > 0
        lnz, fl, src = row.chol_lnz_metis[1], row.chol_fl_metis[1], "fill.csv"
    else
        fits = CSV.read(joinpath(d0, "fits.csv"), DataFrame)
        law(q) = (f = fits[(fits.kind .== "fill") .& (fits.order .== o) .& (fits.nz .== nz) .& (fits.quantity .== q), :];
                  f.C[1] * N^f.exponent[1])
        lnz, fl, src = law("chol_lnz_metis"), law("chol_fl_metis"), "fits.csv law"
    end
    lunz = κ * (2lnz - N)
    M = β * lunz / 1024^3
    return (lunz_proj=lunz, umf_peak_proj_gb=M, peak_proj_gb=R0 + r * N / 1024^3 + M, fact_solve_proj_s=fl / rate,
            proj_source=src)
end

function table()
    path = joinpath(REPO, "data", "sims", TEST_ID)
    sz = CSV.read(joinpath(REPO, "data", "exp_pro", "L3-0", "sizing.csv"), DataFrame)
    rows = NamedTuple[]
    for i in eachindex(CASES)
        p = params(i)
        f = joinpath(path, record_name(p) * ".jld2")
        isfile(f) || (println("[L3-0b] case $i missing"); continue)
        d = wload(f)
        (o, nx, ny, nz) = CASES[i]
        pr = projection(o, nx, ny, nz, d["ndofs"])
        l30(ord, col) = (r = sz[(sz.order .== o) .& (sz.nx .== nx) .& (sz.ny .== ny) .& (sz.nz .== nz) .&
                               (sz.ordering .== ord), :]; nrow(r) > 0 ? r[1, col] : NaN)
        push!(rows, (order=o, nx=nx, ny=ny, nz=nz, ndofs=d["ndofs"], nnzA=d["nnzA"], lunz=d["lunz"],
                     lunz_proj=pr.lunz_proj, umf_peak_gb=d["umf_peak_gb"], umf_peak_proj_gb=pr.umf_peak_proj_gb,
                     peak_rss_gb=d["peak_rss_gb"], peak_proj_gb=pr.peak_proj_gb,
                     baseline_maxrss_gb=d["baseline_maxrss_gb"],
                     simulate_s=d["simulate_wall_s"], factorize_s=d["factorize_wall_s"],
                     fact_solve_proj_s=pr.fact_solve_proj_s, solve1_s=d["solve1_wall_s"], solve4_s=d["solve4_wall_s"],
                     assembly_s=d["assembly_wall_s"], rel_residual=d["rel_residual"],
                     rel_residual_multi=d["rel_residual_multi"], ordering_used=d["ordering_used"],
                     L30_default_peak_rss_gb=l30("default", :peak_rss_gb), L30_default_fact_solve_s=l30("default", :factor_solve_s),
                     L30_metis_peak_rss_gb=l30("metis", :peak_rss_gb), L30_metis_fact_solve_s=l30("metis", :factor_solve_s),
                     proj_source=pr.proj_source))
    end
    df = DataFrame(rows)
    out = joinpath(REPO, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "metis_solver.csv"), df)
    show(stdout, MIME("text/plain"), df; allcols=true)
    println()
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    i = findfirst(==("--case"), ARGS)
    isnothing(i) ? table() : run_one(parse(Int, ARGS[i + 1]))
end
