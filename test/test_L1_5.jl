"""
Tests of the periodic-supercell tests L1-5, L1-6, L1-8 (src/L1_5_discrete_stop_band.jl,
src/L1_6_spacing_sweep.jl, src/L1_8_fluid_dilution.jl and their scripts): the
new references (Bloch phase at given ω, bare branch, gap laws), the derived
geometry of L1-8, and a smoke run of each script into a temporary folder.
"""

using Test
import HydroElasticFEM_Colomes2026.L1_5
import HydroElasticFEM_Colomes2026.L1_6
import HydroElasticFEM_Colomes2026.L1_8
using HydroElasticFEM_Colomes2026.References

const L1_5_SCRIPT = joinpath(@__DIR__, "..", "scripts", "L1-configA-vertical", "L1-5_discrete_stop_band.jl")
const L1_6_SCRIPT = joinpath(@__DIR__, "..", "scripts", "L1-configA-vertical", "L1-6_spacing_sweep.jl")
const L1_8_SCRIPT = joinpath(@__DIR__, "..", "scripts", "L1-configA-vertical", "L1-8_fluid_dilution.jl")

@testset "L1-5/6/8: references" begin
    h = HostSpec(m_ρ=0.01, C_ρ=0.49)
    r = resonator_from_frequency(0.01, 10.0)
    # Bloch phase at given ω is the inverse of the band at given q
    for qa in (0.4, 1.3, 2.5)
        for ω in bloch_roots_1d(h, r, qa, 1.0, range(8.0, 12.0; length=801))
            @test any(x -> isapprox(x, qa; atol=1e-6), bloch_qa_1d(h, r, 1.0, ω))
        end
    end
    @test isempty(bloch_qa_1d(h, r, 1.0, 9.8))                           # inside [ω_X, ω_Γ] = [9.646, 9.986]
    @test D0(h, π, bare_frequency(h, π)) ≈ 0 atol = 1e-9
    # gap laws against the lattice sums (test matrix §0.2b: within 2 % for ak_c/π ≲ 0.75)
    for a in (0.25, 0.5)
        ra = resonator_from_frequency(0.01a, 10.0)
        exact = (omega_gamma_1d(h, ra, a) - omega_X_1d(h, ra, a; ωmin=9.0, npts=4001)) / 10
        @test gap_asymptotic_1d(h, ra, a) ≈ exact rtol = 0.02
    end
    @test gap_a4_1d(h, resonator_from_frequency(0.0025, 10.0), 0.25) ≈ 7.6e-5 rtol = 0.05
end

@testset "L1-8: geometry at the crossing" begin
    p = L1_8.L1_8_Params(M=8, elements_per_wavelength=32)
    d = L1_8.derived(p)
    @test 2π * p.j / (p.M * d.a) ≈ d.kc rtol = 1e-14
    @test d.cell.m_ρr ≈ p.loading * d.a
    @test d.cell.elements_per_cell == 4
    @test_throws ArgumentError L1_8.derived(L1_8.L1_8_Params(M=16, elements_per_wavelength=24))
    @test L1_6.fixed_hash(L1_6.L1_6_Params(a=0.5)) == L1_6.fixed_hash(L1_6.L1_6_Params(a=1.0))
end

module L1_5_ScriptSandbox
include(Main.L1_5_SCRIPT)
end
module L1_6_ScriptSandbox
include(Main.L1_6_SCRIPT)
end
module L1_8_ScriptSandbox
include(Main.L1_8_SCRIPT)
end

@testset "L1-5/6/8: script smoke runs" begin
    mktempdir() do tmp
        out = L1_5_ScriptSandbox.main(; smoke=true, root=tmp)
        @test out.errors.err_X[1] < 1e-7                 # P3, 8 el./cell: ≈ 5e-9
        @test out.errors.err_gamma[1] < 1e-9
        @test isfile(joinpath(tmp, "plots", "L1-5", "bands.pdf"))
        out = L1_6_ScriptSandbox.main(; smoke=true, root=tmp)
        @test out.errors.err_gap[1] < 1e-3
        out = L1_8_ScriptSandbox.main(; smoke=true, root=tmp)
        @test out.errors.err_split[1] < 1e-4
        @test isfile(joinpath(tmp, "data", "exp_pro", "L1-8", "checks.csv"))
    end
end
