"""
Tests of L1-1 (src/L1_1_bare_host_dispersion.jl and its script): the wave fit
on synthetic fields, the layout and mesh rules, record naming, and a smoke run
of the script (one coarse case) into a temporary folder.
"""

using Test
using HydroElasticFEM_xxx2026.References
using HydroElasticFEM_xxx2026.L1_1

const L1_1_SCRIPT = joinpath(@__DIR__, "..", "scripts", "L1-configA-vertical", "L1-1_bare_host_dispersion.jl")

@testset "L1-1: wave fit on a synthetic field" begin
    p = L1_1_Params()
    h, ω = L1_1.host(p), 10.0
    roots = host_roots(h, ω; n_evanescent=20)
    kp = real(roots[1])
    xa, xb, xr = 3.0, 12.0, 1.5
    Ap, Am = 0.7 - 0.2im, 0.1 + 0.3im
    field(x; k=kp) = Ap * exp(im * k * (x - xr)) + Am * exp(-im * k * (x - xr)) +
                     0.02exp(im * roots[4] * (x - xa)) + 0.01im * exp(im * roots[4] * (xb - x)) +
                     0.05exp(im * roots[2] * (x - xa)) + 0.03exp(im * roots[3] * (xb - x))
    xs = range(xa, xb; length=300)
    fit = L1_1.fit_waves(xs, field.(xs), roots; x_ref=xr, n_modes=6, k=kp)
    @test fit.Aplus ≈ Ap rtol = 1e-10
    @test fit.Aminus ≈ Am rtol = 1e-10
    kh = 0.997kp                                   # a discrete wavenumber is recovered
    fitk = L1_1.fit_waves(xs, field.(xs; k=kh), roots; x_ref=xr, n_modes=6)
    @test fitk.k ≈ kh rtol = 1e-9
    @test fitk.Aplus ≈ Ap rtol = 1e-7
end

@testset "L1-1: layout, mesh and record naming" begin
    p = L1_1_Params(elements_per_wavelength=40)
    lay = L1_1.layout(p)
    h = L1_1.host(p)
    @test lay.x_s - lay.Ls >= p.gap_decays / kappa1(h, p.ω) - 1e-9      # evanescent buffer
    @test lay.buffer >= lay.λ - 1e-12
    n = lay.L / lay.λ - p.length_fraction                                  # off the cavity resonances
    @test isapprox(n, round(n); atol=1e-9)
    @test isapprox(lay.x_s / lay.he, round(lay.x_s / lay.he); atol=1e-9)   # source on a node
    vm = L1_1.vertical_mesh(p, lay.he)
    r = vm.ratio
    @test 1 < r <= 1 + (p.max_ratio - 1) * p.ratio_elements / p.elements_per_wavelength
    @test p.H * (r - 1) / (r^vm.ny - 1) ≈ p.top_fraction * lay.he          # top cell
    # variable parameters name the record, fixed ones change the hash
    @test fixed_hash(p) == fixed_hash(L1_1_Params(elements_per_wavelength=80, ω=5.0, order=3, T_ρ=9.8))
    @test fixed_hash(p) != fixed_hash(L1_1_Params(elements_per_wavelength=40, H=20.0))
end

module L1_1_ScriptSandbox
include(Main.L1_1_SCRIPT)
end

@testset "L1-1: script smoke run" begin
    mktempdir() do tmp
        out = L1_1_ScriptSandbox.main(; smoke=true, root=tmp)
        @test size(out.errors, 1) == 1
        @test out.errors.err_k[1] < 1e-2                  # P2, 20 el./λ_p: ≈ 1.5e-3
        @test isfile(joinpath(tmp, "data", "exp_pro", "L1-1", "checks.csv"))
        @test isfile(joinpath(tmp, "plots", "L1-1", "convergence_single_frequency_p2.pdf"))
        @test length(readdir(joinpath(tmp, "data", "sims", "L1-1"))) == 1
    end
end
