"""
Tests of L1-2 (src/L1_2_flux_normalisation.jl and its script): the patch
quadrature, and a smoke run of the script (one coarse case) into a temporary
folder.
"""

using Test
import HydroElasticFEM_xxx2026.L1_2
using Gridap: Point

const L1_2_SCRIPT = joinpath(@__DIR__, "..", "scripts", "L1-configA-vertical", "L1-2_flux_normalisation.jl")

@testset "L1-2: patch quadrature" begin
    t, w = L1_2.gauss_legendre(4)                  # exact up to degree 7
    @test sum(w) ≈ 2
    @test sum(w .* t .^ 6) ≈ 2 / 7
    # average of e^{ikx} over three elements: sinc
    k, xa, xb = 2.7, 1.0, 1.6
    η(pt) = exp(im * k * pt[1])
    avg = L1_2.patch_average(η, xa, xb, 3; n=8)
    @test avg ≈ (exp(im * k * xb) - exp(im * k * xa)) / (im * k * (xb - xa)) rtol = 1e-12
end

module L1_2_ScriptSandbox
include(Main.L1_2_SCRIPT)
end

@testset "L1-2: script smoke run" begin
    mktempdir() do tmp
        out = L1_2_ScriptSandbox.main(; smoke=true, root=tmp)
        @test size(out.errors, 1) == 1
        @test out.errors.err_far[1] < 2e-2                # P2, 20 el./λ_p: ≈ 6e-3
        @test out.errors.err_power[1] < 2e-2
        @test isfile(joinpath(tmp, "data", "exp_pro", "L1-2", "checks.csv"))
        @test isfile(joinpath(tmp, "plots", "L1-2", "convergence_p2.pdf"))
    end
end
