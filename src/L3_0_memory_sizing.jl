"""
# L3-0 — 3D memory sizing (fluid box with a floating plate)

## Purpose

First card of Phase 4 (3D). Not a verification test of the paper: it measures
the cost of the 3D monolithic problem (DOFs, peak memory, assembly and solve
time with the library's default direct solver) on a ladder of meshes, so that
the domains of L3-1, L3-2, L3-3, L4-4 and L4-8 can be sized against the 4 GB
`run_limited` cap and the 24 GB machine before they are pre-registered.

## Problem

Configuration A in 3D: a box [0, Lx] × [0, Ly] × [0, H] of fluid (potential
φ, Q_p) fully covered by a Kirchhoff–Love plate (η, Q_p, C/DG with γ = 6),
set P1 (m_ρ = 0.01, C_ρ = 0.49 via E = 12C_ρρ/h_b³, ν = 0), ω = 10. The plate
is driven by a line load across the width at x_s; two Kelvin–Voigt
`StructuralDampingZone`s (τ = s/ω, σ = ξ²) absorb the waves at both x ends.
With ν = 0 and free lateral edges the field is uniform in y, so the 3D
centreline equals the 2D (L1-1) field: the discrete wavenumber k_h on the
window between source and sponge is a sanity check of the solution.

Mesh: uniform horizontal cells h = λ_p/`elements_per_wavelength` (λ_p at ω),
`nx` × `ny` cells, `nz` vertical cells graded geometrically towards the surface
with top cell k₀h_top = `top_k` (matrix §0.3).

## Measurements (one case per Julia process)

- DOFs per field and in total (checked against the tensor-product formula
  (p·nx+1)(p·ny+1)(p·nz+1) + (p·nx+1)(p·ny+1)), nnz(A);
- wall and CPU time of `build_problem` (model, spaces, assembly), of the
  UMFPACK symbolic and numeric factorisations (the factorisation behind
  Gridap's `LUSolver`, the library default) and of the solve;
- UMFPACK's own memory accounting (peak and numeric size, actual), lnz + unz;
- process peak RSS (`Sys.maxrss`) and current RSS (`ps`) at phase ends, after
  a warm-up case that pays the compilation;
- after the solve (outside the measured peak), the exact symbolic Cholesky fill
  nnz(L) and flop count of the pattern of A + Aᵀ with AMD (UMFPACK's default
  ordering for symmetric patterns) and with METIS nested dissection
  (CHOLMOD analysis, `cholmod_fill`).

`analyse_case` builds the matrix and runs only the CHOLMOD analysis (no
factorisation). Its memory is that of the assembly, so it reaches meshes whose
LU does not fit under the cap; the fill and flops calibrated on the full cases
then give the LU memory and time of the larger meshes.

## Reproduce

    julia --project -O1 scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl --case 3   # one ladder case
    julia --project -O1 scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl            # tables, fits, figure
"""
module L3_0

using Gridap
using Parameters
using Printf
using LinearAlgebra: norm, Symmetric
import SparseArrays
import SparseArrays.UMFPACK as UMF
import SparseArrays.LibSuiteSparse as LSS
import SparseArrays.CHOLMOD as CM
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

using ..References: HostSpec, k0, kappa1, host_roots
import ..L1_1

export L3_0_Params, VARIABLE, fixed_hash, run_case, analyse_case, cholmod_fill, dof_formula

"Parameters of one L3-0 sizing case."
@with_kw struct L3_0_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    order::Int = 2                        # polynomial order of φ and η
    nx::Int = 16                          # horizontal cells along x
    ny::Int = 16                          # horizontal cells along y
    nz::Int = 8                           # vertical cells (graded)
    elements_per_wavelength::Int = 8      # h = λ_p/epw
    ordering::Symbol = :default           # UMFPACK fill-reducing ordering: :default (as LUSolver) or :metis

    # ── Fixed: host, set P1 ───────────────────────────────────────────────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49
    hb::Float64 = 0.01                    # KL thickness: E = 12C_ρρ/hb³, ρb = m_ρρ/hb
    ν::Float64 = 0.0
    ω::Float64 = 10.0

    # ── Fixed: forcing, sponges, discretisation ──────────────────────────────
    F::Float64 = 1.0
    source_elements::Int = 2
    sponge_wavelengths::Float64 = 2.0     # each end (capped at Lx/5)
    sponge_strength::Float64 = 3.0        # ωτ at the wall
    top_k::Float64 = 0.2                  # k₀·(top cell)
    γ_penalty::Float64 = 6.0

    # ── Fixed: post-processing ────────────────────────────────────────────────
    n_modes::Int = 6
    n_samples::Int = 300
end

const VARIABLE = (:order, :nx, :ny, :nz, :elements_per_wavelength)

"Fields that name a record only when they differ from their default (added after the first records)."
const LATE_FIELDS = (:ordering => :default,)

function fixed_hash(p::L3_0_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L3_0_Params)
                  if !(f in VARIABLE) && !any(l -> l.first == f, LATE_FIELDS))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

host(p) = HostSpec(g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.m_ρ, C_ρ=p.C_ρ)

"DOFs of continuous Q_p φ on the nx×ny×nz box plus Q_p η on the full top surface."
dof_formula(order, nx, ny, nz) = (order * nx + 1) * (order * ny + 1) * (order * nz + 1) +
                                 (order * nx + 1) * (order * ny + 1)

"Geometric vertical map of [0, H] with `nz` cells, top cell `top` (base b = (H/top)^{1/(nz−1)})."
function zmap(H, nz, top)
    nz == 1 && return z -> z
    b = (H / top)^(1 / (nz - 1))
    return z -> (z >= H - 1e-12 ? H : H - H / b^(z / (H / nz)))
end

function layout(p::L3_0_Params)
    h = host(p)
    k = k0(h, p.ω)
    λ = 2π / k
    he = λ / p.elements_per_wavelength
    Lx, Ly = p.nx * he, p.ny * he
    Ls = min(p.sponge_wavelengths * λ, Lx / 5)
    x_s = round((Ls + 0.5λ) / he) * he
    window = (x_s + λ, Lx - Ls - 0.25λ)
    return (; λ, k, he, Lx, Ly, Ls, x_s, b=p.source_elements * he, window, top=p.top_k / k)
end

# ─── Instrumentation ────────────────────────────────────────────────────────

maxrss_gb() = Sys.maxrss() / 1024^3

"Current resident set size (GB) from `ps`; NaN if unavailable."
function rss_gb()
    try
        return parse(Float64, strip(read(`ps -o rss= -p $(getpid())`, String))) / 1024^2
    catch
        return NaN
    end
end

function cpu_s()
    buf = zeros(UInt8, 256)   # struct rusage: ru_utime, ru_stime (timeval: Int64 sec, Int32 usec + pad)
    ccall(:getrusage, Cint, (Cint, Ptr{UInt8}), 0, buf)
    v = reinterpret(Int64, buf[1:32])
    return v[1] + (v[2] & 0xffffffff) / 1e6 + v[3] + (v[4] & 0xffffffff) / 1e6
end

"Run `f()`, record wall/CPU time, peak and current RSS under `name` in `rec`."
function phase!(f, rec::Dict, name::String; verbose::Bool=true)
    t0, c0 = time_ns(), cpu_s()
    val = f()
    rec[name * "_wall_s"] = (time_ns() - t0) / 1e9
    rec[name * "_cpu_s"] = cpu_s() - c0
    rec[name * "_maxrss_gb"] = maxrss_gb()
    rec[name * "_rss_gb"] = rss_gb()
    verbose && @printf("[phase] %-18s wall %8.2f s  cpu %8.2f s  maxrss %6.3f GB  rss %6.3f GB\n", name,
                       rec[name * "_wall_s"], rec[name * "_cpu_s"], rec[name * "_maxrss_gb"], rec[name * "_rss_gb"])
    flush(stdout)
    return val
end

umf_info(F, key) = F.info[key + 1]
function umf_unit(F)
    u = umf_info(F, LSS.UMFPACK_SIZE_OF_UNIT)
    return isfinite(u) && u > 0 ? u : 8.0
end

# ─── Symbolic fill (CHOLMOD analysis) ───────────────────────────────────────

"Store `val` in field `name` of CHOLMOD's global `cholmod_common` (the Julia wrapper exposes it read-only)."
function set_cholmod_common!(name::Symbol, val)
    T = LSS.cholmod_common_struct
    c = CM.getcommon()
    i = findfirst(==(name), fieldnames(T))
    GC.@preserve c begin
        ptr = Base.unsafe_convert(Ptr{T}, c)
        unsafe_store!(Ptr{fieldtype(T, i)}(ptr + fieldoffset(T, i)), convert(fieldtype(T, i), val))
    end
    return nothing
end

"""
    cholmod_fill(A)

Exact symbolic Cholesky statistics of the pattern of A (values ignored; the FE
pattern is structurally symmetric, checked on the full cases): nnz(L)
(diagonal included) and flop count with CHOLMOD's AMD and METIS orderings
(`cholmod_analyze` with `nmethods = 3`: given (skipped), AMD, METIS).
"""
cholmod_fill(A::SparseArrays.SparseMatrixCSC) = cholmod_fill_lower(lower_pattern(A))

"Lower triangle (diagonal forced) of the pattern of A, as a Float64 CSC matrix of ones."
function lower_pattern(A::SparseArrays.SparseMatrixCSC)
    N = size(A, 1)
    colptr = Vector{Int}(undef, N + 1)
    cnt = 0
    for j in 1:N, k in A.colptr[j]:(A.colptr[j + 1] - 1)
        cnt += A.rowval[k] > j
    end
    rowval = Vector{Int}(undef, cnt + N)
    q = 1
    for j in 1:N
        colptr[j] = q
        rowval[q] = j; q += 1
        for k in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[k]
            i > j && (rowval[q] = i; q += 1)
        end
    end
    colptr[N + 1] = q
    return SparseArrays.SparseMatrixCSC(N, N, colptr, rowval, ones(Float64, cnt + N))
end

"Structural symmetry of the pattern of A."
pattern_symmetric(A::SparseArrays.SparseMatrixCSC) =
    (P = SparseArrays.SparseMatrixCSC(size(A)..., A.colptr, A.rowval, ones(Int8, SparseArrays.nnz(A))); P == copy(P'))

function cholmod_fill_lower(L::SparseArrays.SparseMatrixCSC{Float64})
    S = CM.Sparse(Symmetric(L, :L))
    set_cholmod_common!(:nmethods, 3)
    try
        F = CM.symbolic(S; postorder=true)
        c = CM.getcommon()[]
        m = c.method
        # method[2]: AMD (ordering 2), method[3]: METIS (ordering 3)
        @assert m[2].ordering == LSS.CHOLMOD_AMD && m[3].ordering == LSS.CHOLMOD_METIS
        out = (chol_lnz_amd=m[2].lnz, chol_fl_amd=m[2].fl, chol_lnz_metis=m[3].lnz, chol_fl_metis=m[3].fl,
               chol_anz=c.anz)
        finalize(F)
        return out
    finally
        set_cholmod_common!(:nmethods, 0)
    end
end

# ─── Problem ────────────────────────────────────────────────────────────────

"Build the problem of case `p` (returns the HydroElasticFEM problem and the layout)."
function build(p::L3_0_Params; lay=layout(p))
    tank = G.TankDomain(L=lay.Lx, W=lay.Ly, H=p.H, nx=p.nx, ny=p.ny, nz=p.nz,
                        map=let fz = zmap(p.H, p.nz, lay.top); x -> VectorValue(x[1], x[2], fz(x[3])) end,
                        structure_domains=[G.StructureDomain(L=lay.Lx, W=lay.Ly, x₀=[0.0, 0.0, p.H],
                                                             domain_symbol=:Γs)])
    fe(; kw...) = PH.FESpaceConfig(; order=p.order, vector_type=Vector{ComplexF64}, kw...)
    zones = [P.StructuralDampingZone(profile=P.ramp_profile(x_start=lay.Ls, x_end=0.0), τ=p.sponge_strength / p.ω),
             P.StructuralDampingZone(profile=P.ramp_profile(x_start=lay.Lx - lay.Ls, x_end=lay.Lx),
                                     τ=p.sponge_strength / p.ω)]
    fluid = P.PotentialFlow(g=p.g, ρw=p.ρ, dim=3, fe=fe(), space_domain_symbol=:Ω)
    plate = P.KirchhoffLovePlate(E=12 * p.C_ρ * p.ρ / p.hb^3, ν=p.ν, hb=p.hb, ρ=p.ρ, ρb=p.m_ρ * p.ρ / p.hb,
                                 g=p.g, ambient_dim=3, manifold_dim=2, symbol=:η, fe=fe(γ=p.γ_penalty),
                                 space_domain_symbol=:Γη, damping_zones=zones)
    load(x) = abs(x[1] - lay.x_s) < lay.b / 2 ? complex(p.F / lay.b) : zero(ComplexF64)
    field_order(fmap) = [s for (s, _) in sort!(collect(fmap._map); by=last)]
    rhs_fn = (ctx, fmap) -> [sym == :η ? load : 0.0 for sym in field_order(fmap)]
    problem = S.build_problem(tank, P.PhysicsParameters[fluid, plate], PH.FreqDomainConfig(ω=p.ω); rhs_fn=rhs_fn)
    return problem, lay
end

"Evaluate a surface CellField at 3D points (direct, else by an Interpolable search)."
function eval_on_plate(fh, pts)
    try
        return fh(pts)
    catch
        fi = Gridap.CellData.Interpolable(fh; searchmethod=Gridap.CellData.KDTreeSearch(num_nearest_vertices=4))
        return [fi(q) for q in pts]
    end
end

"""
    run_case(p; verbose=true)

Assemble, factorise (UMFPACK, as `LUSolver`) and solve case `p`, recording
sizes, times and memory (see the module docstring). Returns a Dict.
"""
function run_case(p::L3_0_Params; verbose::Bool=true)
    rec = Dict{String,Any}()
    lay = layout(p)
    problem, _ = phase!(() -> build(p; lay=lay), rec, "assembly"; verbose=verbose)
    op = S.get_fe_operator(problem)
    A, b = get_matrix(op), get_vector(op)
    X = S.get_trial_fe_space(problem)
    fmap = S.get_field_map(problem)
    N = size(A, 1)
    rec["ndofs"] = N
    rec["ndofs_fields"] = [num_free_dofs(Xi) for Xi in X.spaces]
    rec["ndofs_formula"] = dof_formula(p.order, p.nx, p.ny, p.nz)
    rec["nnzA"] = SparseArrays.nnz(A)
    rec["matrix_gb"] = (SparseArrays.nnz(A) * 24 + (N + 1) * 8) / 1024^3
    verbose && @printf("[info] N = %d (formula %d)  nnz(A) = %d  fields = %s\n", N, rec["ndofs_formula"],
                       rec["nnzA"], string(rec["ndofs_fields"]))
    F = UMF.UmfpackLU(A)
    p.ordering == :metis && (F.control[LSS.UMFPACK_ORDERING + 1] = LSS.UMFPACK_ORDERING_METIS)
    p.ordering in (:default, :metis) || throw(ArgumentError("ordering must be :default or :metis"))
    phase!(() -> UMF.umfpack_symbolic!(F, nothing), rec, "symbolic"; verbose=verbose)
    u = umf_unit(F)
    rec["umf_peak_estimate_gb"] = umf_info(F, LSS.UMFPACK_PEAK_MEMORY_ESTIMATE) * u / 1024^3
    rec["umf_flops_estimate"] = umf_info(F, LSS.UMFPACK_FLOPS_ESTIMATE)
    rec["umf_ordering_used"] = umf_info(F, LSS.UMFPACK_ORDERING_USED)
    phase!(() -> UMF.umfpack_numeric!(F), rec, "numeric"; verbose=verbose)
    rec["umf_peak_actual_gb"] = umf_info(F, LSS.UMFPACK_PEAK_MEMORY) * u / 1024^3
    rec["umf_numeric_actual_gb"] = umf_info(F, LSS.UMFPACK_NUMERIC_SIZE) * u / 1024^3
    rec["umf_flops"] = umf_info(F, LSS.UMFPACK_FLOPS)
    l = UMF.umf_lunz(F)
    rec["lunz"] = Int(l[1]) + Int(l[2])
    x = phase!(() -> F \ b, rec, "solve"; verbose=verbose)
    rec["rel_residual"] = norm(A * x - b) / norm(b)
    rec["solution_finite"] = all(isfinite, x)

    # Sanity check: discrete wavenumber on the centreline window (y-uniform field)
    phase!(rec, "post"; verbose=verbose) do
        uh = FEFunction(X, x)
        ηh = uh[fmap[:η]]
        xa, xb = lay.window
        rec["window_wavelengths"] = (xb - xa) / lay.λ
        y_c = lay.Ly / 2 + 0.013 * lay.he
        if xb - xa >= 1.5lay.λ
            xs = collect(range(xa, xb; length=p.n_samples))
            ηs = ComplexF64.(eval_on_plate(ηh, [Point(xx, y_c, p.H) for xx in xs]))
            roots = host_roots(host(p), p.ω)
            fit = L1_1.fit_waves(xs, ηs, roots; x_ref=lay.x_s, n_modes=p.n_modes)
            rec["dispersion_error"] = fit.k / real(roots[1]) - 1
            rec["fit_residual"] = fit.residual
            rec["reflection"] = abs(fit.Aminus / fit.Aplus)
            # y-uniformity: the edge line against the centreline
            ηe = ComplexF64.(eval_on_plate(ηh, [Point(xx, 0.07 * lay.he, p.H) for xx in xs]))
            rec["y_nonuniformity"] = norm(ηe - ηs) / norm(ηs)
        else
            rec["dispersion_error"] = NaN
            rec["fit_residual"] = NaN
            rec["reflection"] = NaN
            rec["y_nonuniformity"] = NaN
        end
        nothing
    end
    rec["lambda_p"] = lay.λ
    rec["element_size"] = lay.he
    rec["Lx"], rec["Ly"] = lay.Lx, lay.Ly
    rec["top_cell"] = lay.top
    rec["kappa1"] = kappa1(host(p), p.ω)
    rec["maxrss_final_gb"] = maxrss_gb()

    # Symbolic fill of the same matrix, outside the measured peak (calibrates `analyse_case`)
    F = nothing; x = nothing
    GC.gc()
    phase!(rec, "analyse"; verbose=verbose) do
        rec["pattern_symmetric"] = pattern_symmetric(A)
        merge!(rec, Dict(string(k) => v for (k, v) in pairs(cholmod_fill(A))))
        nothing
    end
    return rec
end

"""
    analyse_case(p; verbose=true)

Assemble case `p` and run only the CHOLMOD analysis (`cholmod_fill`) on its
pattern: DOFs, nnz(A), assembly time and memory, nnz(L) and flops with AMD
and METIS. No factorisation, so it reaches meshes whose LU exceeds the cap.
"""
function analyse_case(p::L3_0_Params; verbose::Bool=true)
    rec = Dict{String,Any}()
    lay = layout(p)
    B = phase!(rec, "assembly"; verbose=verbose) do
        problem, _ = build(p; lay=lay)
        A = get_matrix(S.get_fe_operator(problem))
        rec["ndofs_fields"] = [num_free_dofs(Xi) for Xi in S.get_trial_fe_space(problem).spaces]
        rec["ndofs"] = size(A, 1)
        rec["nnzA"] = SparseArrays.nnz(A)
        lower_pattern(A)
    end
    GC.gc()
    N = rec["ndofs"]
    rec["ndofs_formula"] = dof_formula(p.order, p.nx, p.ny, p.nz)
    rec["matrix_gb"] = (rec["nnzA"] * 24 + (N + 1) * 8) / 1024^3
    verbose && @printf("[info] N = %d (formula %d)  nnz(A) = %d  fields = %s\n", N, rec["ndofs_formula"],
                       rec["nnzA"], string(rec["ndofs_fields"]))
    phase!(rec, "analyse"; verbose=verbose) do
        merge!(rec, Dict(string(k) => v for (k, v) in pairs(cholmod_fill_lower(B))))
        nothing
    end
    verbose && @printf("[info] nnz(L): AMD %.4g, METIS %.4g;  flops: AMD %.4g, METIS %.4g\n", rec["chol_lnz_amd"],
                       rec["chol_lnz_metis"], rec["chol_fl_amd"], rec["chol_fl_metis"])
    rec["Lx"], rec["Ly"] = lay.Lx, lay.Ly
    rec["maxrss_final_gb"] = maxrss_gb()
    return rec
end

end
