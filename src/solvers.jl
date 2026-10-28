"""
# Solvers — sparse direct solvers for the 3D runs

`MetisLUSolver` is a Gridap `LinearSolver`: sparse LU by UMFPACK (SuiteSparse,
shipped with Julia's SparseArrays) with the METIS nested-dissection
fill-reducing ordering instead of UMFPACK's default (AMD on A + Aᵀ for the
structurally symmetric FE pattern). METIS is part of Julia's SuiteSparse
build (`libsuitesparse_jll`), so no new dependency is needed.

L3-0 measured the gain on the 3D configuration-A matrices: the LU fill drops
1.8× at N = 76 k and the projected fill 2.4× at N = 470 k, the flops 7.6×
(docs/L3_implementation_report.md, L3-0).

## Use

Through HydroElasticFEM's solver hook (no library change):

    config = PH.FreqDomainConfig(ω=ω, solver=MetisLUSolver())
    result = S.simulate(S.build_problem(domain, entities, config; rhs_fn))

Factorise once, solve many right-hand sides at one ω:

    ns = factorize(MetisLUSolver(), A)      # = numerical_setup(symbolic_setup(s, A), A)
    x1 = solve_rhs(ns, b1); x2 = solve_rhs(ns, b2)
    X  = solve_rhs(ns, B)                  # B with one column per right-hand side

`numerical_setup!(ns, A2)` refactorises a matrix with the same pattern (e.g.
the next ω) and reuses the symbolic analysis, hence the METIS ordering.

`factor_stats(ns)` returns the UMFPACK statistics of the factorisation (fill
lunz = nnz(L) + nnz(U), UMFPACK peak and numeric memory, flops, ordering used).
"""
module Solvers

using LinearAlgebra: ldiv!, lu, lu!
import SparseArrays
import SparseArrays: SparseMatrixCSC
import SparseArrays.UMFPACK as UMF
import SparseArrays.LibSuiteSparse as LSS
using Gridap.Algebra: LinearSolver, SymbolicSetup, NumericalSetup
import Gridap.Algebra: symbolic_setup, numerical_setup, numerical_setup!, solve!

export MetisLUSolver, factorize, solve_rhs, factor_stats

const ORDERINGS = Dict(:metis => LSS.UMFPACK_ORDERING_METIS, :amd => LSS.UMFPACK_ORDERING_AMD,
                       :cholmod => LSS.UMFPACK_ORDERING_CHOLMOD, :best => LSS.UMFPACK_ORDERING_BEST,
                       :default => nothing)

const ORDERING_NAMES = Dict(Int(LSS.UMFPACK_ORDERING_CHOLMOD) => :cholmod, Int(LSS.UMFPACK_ORDERING_AMD) => :amd,
                            Int(LSS.UMFPACK_ORDERING_GIVEN) => :given, Int(LSS.UMFPACK_ORDERING_METIS) => :metis,
                            Int(LSS.UMFPACK_ORDERING_BEST) => :best, Int(LSS.UMFPACK_ORDERING_NONE) => :none,
                            Int(LSS.UMFPACK_ORDERING_USER) => :user)

"""
    MetisLUSolver(; ordering=:metis)

Sparse LU (UMFPACK) with a chosen fill-reducing ordering: `:metis` (default),
`:amd`, `:cholmod`, `:best`, or `:default` (UMFPACK's own choice, the same
factorisation as Gridap's `LUSolver`). Every other UMFPACK control is Julia's
default (no iterative refinement).
"""
struct MetisLUSolver <: LinearSolver
    ordering::Symbol
    function MetisLUSolver(; ordering::Symbol=:metis)
        haskey(ORDERINGS, ordering) || throw(ArgumentError("unknown ordering $ordering; use one of $(keys(ORDERINGS))"))
        return new(ordering)
    end
end

"UMFPACK control vector of `s` for matrices with entries `Tv`."
function umfpack_control(s::MetisLUSolver, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti}
    c = UMF.get_umfpack_control(Tv, Ti)
    o = ORDERINGS[s.ordering]
    isnothing(o) || (c[LSS.UMFPACK_ORDERING + 1] = o)
    return c
end

struct MetisLUSymbolicSetup <: SymbolicSetup
    solver::MetisLUSolver
end

mutable struct MetisLUNumericalSetup{F} <: NumericalSetup
    factors::F
    solver::MetisLUSolver
end

symbolic_setup(s::MetisLUSolver, ::AbstractMatrix) = MetisLUSymbolicSetup(s)

function numerical_setup(ss::MetisLUSymbolicSetup, A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    F = lu(A; control=umfpack_control(ss.solver, Tv, Ti))
    return MetisLUNumericalSetup(F, ss.solver)
end

numerical_setup(ss::MetisLUSymbolicSetup, A::AbstractMatrix) = numerical_setup(ss, SparseMatrixCSC(A))

"Refactorise `A` (same sparsity pattern) reusing the symbolic analysis and its ordering."
function numerical_setup!(ns::MetisLUNumericalSetup, A::SparseMatrixCSC)
    lu!(ns.factors, A; reuse_symbolic=true)
    return ns
end

function solve!(x::AbstractVector, ns::MetisLUNumericalSetup, b::AbstractVector)
    ldiv!(x, ns.factors, b)
    return x
end

function solve!(X::AbstractMatrix, ns::MetisLUNumericalSetup, B::AbstractMatrix)
    ldiv!(X, ns.factors, B)
    return X
end

"Factorise `A` once with solver `s` (returns the numerical setup, reusable for many right-hand sides)."
factorize(s::MetisLUSolver, A::AbstractMatrix) = numerical_setup(symbolic_setup(s, A), A)

"Solve with an existing factorisation; `b` is a vector or a matrix with one right-hand side per column."
solve_rhs(ns::MetisLUNumericalSetup, b::AbstractVecOrMat) = solve!(similar(b, eltype(ns.factors)), ns, b)

"""
    factor_stats(ns)

UMFPACK statistics of the factorisation in `ns`: `lunz` (nnz(L) + nnz(U)),
`peak_gb` and `numeric_gb` (UMFPACK's actual memory accounting), `flops`,
and `ordering` (the ordering UMFPACK used, e.g. `:metis`).
"""
function factor_stats(ns::MetisLUNumericalSetup)
    F = ns.factors
    info(key) = F.info[key + 1]
    u = info(LSS.UMFPACK_SIZE_OF_UNIT)
    u = isfinite(u) && u > 0 ? u : 8.0
    l = UMF.umf_lunz(F)
    used = info(LSS.UMFPACK_ORDERING_USED)
    return (lunz=Int(l[1]) + Int(l[2]), peak_gb=info(LSS.UMFPACK_PEAK_MEMORY) * u / 1024^3,
            numeric_gb=info(LSS.UMFPACK_NUMERIC_SIZE) * u / 1024^3, flops=info(LSS.UMFPACK_FLOPS),
            ordering=get(ORDERING_NAMES, isfinite(used) ? Int(used) : -1, :unknown))
end

end
