"""
Tests of L1-3 (src/L1_3_single_line_resonator.jl and its script): the
scattering relations and root finder on synthetic data, the layout rules, and
a smoke run of the script (one coarse case) into a temporary folder.
"""

using Test
import HydroElasticFEM_xxx2026.L1_3
using HydroElasticFEM_xxx2026.References: kappa1

const L1_3_SCRIPT = joinpath(@__DIR__, "..", "scripts", "L1-configA-vertical", "L1-3_single_line_resonator.jl")

@testset "L1-3: point scattering and secant root" begin
    R = -0.3 + 0.4im
    a, b, E = 0.9 - 0.1im, 0.2 + 0.05im, 0.01 - 0.02im   # incident, entering from downstream, near-field drive
    S = R * (a + b + E)                                   # radiated to both sides
    s = L1_3.point_scattering((Aplus=a, Aminus=b + S), (Aplus=a + S, Aminus=b), E)
    @test s.R ≈ R rtol = 1e-12
    @test s.T ≈ 1 + R rtol = 1e-12
    @test s.asymmetry < 1e-12
    root = L1_3.secant_root(ω -> (ω - 10.02)^3 + 4(ω - 10.02), 10.0, 10.01; xtol=1e-12)
    @test root.converged
    @test root.root ≈ 10.02 rtol = 1e-12
end

@testset "L1-3: layout" begin
    for snap in (:node, :midelement)
        p = L1_3.L1_3_Params(elements_per_wavelength=20, snap=snap)
        lay = L1_3.layout(p)
        @test lay.window[1] <= lay.x_s - lay.b / 2 && lay.down[2] <= lay.window[2]
        @test lay.x_s + lay.b / 2 < lay.up[1] < lay.up[2] < lay.x_r < lay.down[1] < lay.down[2]
        @test min(lay.x_s, lay.L - lay.x_r) - lay.Ls >= p.gap_decays / kappa1(L1_3.L1_1.host(p), p.ω_r) - lay.he
        n = lay.L / lay.λ - p.length_fraction
        @test isapprox(n, round(n); atol=1e-9)
        @test isapprox(mod(lay.x_r / lay.he, 1.0), snap == :node ? 0.0 : 0.5; atol=1e-9)
    end
    @test L1_3.fixed_hash(L1_3.L1_3_Params()) == L1_3.fixed_hash(L1_3.L1_3_Params(T_ρ=9.8, c_factor=1.0, order=3))
end

module L1_3_ScriptSandbox
include(Main.L1_3_SCRIPT)
end

@testset "L1-3: script smoke run" begin
    mktempdir() do tmp
        out = L1_3_ScriptSandbox.main(; smoke=true, root=tmp)
        @test size(out.errors, 1) == 1
        @test out.errors.err_omega[1] < 1e-6            # P3, 10 el./λ_p: ≈ 6e-9
        @test out.errors.err_peak[1] < 1e-6             # |R(ω̃) + 1| ≈ 2e-10
        @test isfile(joinpath(tmp, "data", "exp_pro", "L1-3", "checks.csv"))
        @test isfile(joinpath(tmp, "plots", "L1-3", "convergence_p3.pdf"))
    end
end
