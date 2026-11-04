"""
Tests of `Solvers.MetisLUSolver` (src/solvers.jl): through HydroElasticFEM's
`config.solver` hook it gives the solution of the default `LUSolver` to
≤ 1e-10 relative on a small 2D case (L1-1 channel with a line resonator) and a
small 3D case (L3-0 box); UMFPACK reports the METIS ordering; one
factorisation solves several right-hand sides; a refactorisation with the same
pattern reuses the symbolic analysis.
"""

using Test
using LinearAlgebra: norm, lu
import SparseArrays
using Gridap: get_free_dof_values, get_matrix, get_vector
import Gridap.Algebra as GA
import HydroElasticFEM.Simulation as S
using HydroElasticFEM_Colomes2026.Solvers
using HydroElasticFEM_Colomes2026.Solvers: factorize
using HydroElasticFEM_Colomes2026.References: ResonatorSpec
using HydroElasticFEM_Colomes2026.L1_1
using HydroElasticFEM_Colomes2026: L3_0
using .L3_0: L3_0_Params

relerr(x, y) = norm(x - y) / norm(y)

@testset "Solvers: MetisLUSolver vs LUSolver, 2D (L1-1 + resonator)" begin
    p = L1_1_Params(order=2, elements_per_wavelength=10)
    res = [ResonatorSpec(m_ρ=0.01, k_ρ=0.01 * 10.0^2, c_ρ=0.002)]
    lay = L1_1.layout(p)
    xr = [lay.x_s + 1.3lay.λ]
    ref = L1_1.solve(p; resonators=res, x_res=xr)
    met = L1_1.solve(p; resonators=res, x_res=xr, solver=MetisLUSolver())
    @test met.ndofs == ref.ndofs
    @test relerr(get_free_dof_values(met.ηh), get_free_dof_values(ref.ηh)) <= 1e-10
    @test relerr(met.q, ref.q) <= 1e-10
end

@testset "Solvers: MetisLUSolver vs LUSolver, 3D (L3-0 box)" begin
    p = L3_0_Params(order=2, nx=8, ny=3, nz=3)
    ref = S.simulate(first(L3_0.build(p)))
    met = S.simulate(first(L3_0.build(p; solver=MetisLUSolver())))
    x_ref, x_met = get_free_dof_values(ref.solution), get_free_dof_values(met.solution)
    @test length(x_met) == L3_0.dof_formula(p.order, p.nx, p.ny, p.nz)
    @test relerr(x_met, x_ref) <= 1e-10

    # Factorise once, solve many (one ω), and the reported ordering
    op = met.op
    A, b = get_matrix(op), get_vector(op)
    ns = factorize(MetisLUSolver(), A)
    st = factor_stats(ns)
    @test st.ordering == :metis
    @test st.lunz > SparseArrays.nnz(A) ÷ 2 && st.peak_gb > 0
    @test factor_stats(factorize(MetisLUSolver(ordering=:amd), A)).ordering == :amd
    B = hcat(b, 2im .* b .+ 1.0, ComplexF64.(1:length(b)))
    X = solve_rhs(ns, B)
    Xref = lu(A) \ B
    @test all(relerr(X[:, j], Xref[:, j]) <= 1e-10 for j in axes(B, 2))
    @test relerr(solve_rhs(ns, b), x_ref) <= 1e-10
    x = similar(b)
    GA.solve!(x, ns, B[:, 2])
    @test relerr(x, Xref[:, 2]) <= 1e-10

    # Refactorisation with the same pattern (as for the next ω): symbolic analysis reused
    A2 = copy(A)
    nz2 = SparseArrays.nonzeros(A2)
    nz2 .*= 1 .+ 0.05im .* cos.(1:length(nz2))
    GA.numerical_setup!(ns, A2)
    @test relerr(solve_rhs(ns, b), lu(A2) \ b) <= 1e-10
    @test factor_stats(ns).ordering == :metis
end
