"""
Tests of the analytical reference library shared by all tests (src/references.jl).

Two kinds of tests: internal consistency (two independent evaluations of the
same quantity must agree) and pins against published values
(test/reference/paper_values.toml).
"""

using Test
using TOML
using LinearAlgebra
using HydroElasticFEM_xxx2026
using HydroElasticFEM_xxx2026.References

# Baseline sets of the test matrix (§0.1) used by these tests. In the
# verification tests themselves each test carries its own parameter struct.
"P1: Table 1 of the paper; ω_r = 10 rad/s, spacing a = 1 m (n₀ = 1 m⁻¹)."
P1(; T_ρ=0.0, H=10.0) = (host=HostSpec(g=9.8, ρ=1000.0, H=H, m_ρ=0.01, C_ρ=0.49, T_ρ=T_ρ),
                         resonator=resonator_from_frequency(0.01, 10.0), a=1.0, n0=1.0)
"P1-LF: low-frequency variant; ω_r = 1 rad/s, a = 5 m (same mass loading n₀m_ρ,r as P1)."
P1LF(; H=10.0) = (host=HostSpec(g=9.8, ρ=1000.0, H=H, m_ρ=0.01, C_ρ=0.49),
                  resonator=resonator_from_frequency(0.05, 1.0), a=5.0, n0=0.2)
"P2: 50 m LDPE membrane of the Results draft (ρ = 1025, T = 10 kN/m, H = 25 m)."
P2() = (host=HostSpec(g=9.81, ρ=1025.0, H=25.0, m_ρ=910.0 * 0.05 / 1025.0, C_ρ=0.0, T_ρ=1.0e4 / 1025.0),
        resonator=ResonatorSpec(m_ρ=113.75 / 1025.0, k_ρ=28.4375 / 1025.0))

const PINS = TOML.parsefile(joinpath(@__DIR__, "reference", "paper_values.toml"))

"Agreement to half a unit in the last of `digits` significant digits."
function pinned(value, pin::AbstractDict)
    ref = pin["value"]
    return abs(value - ref) <= 0.5 * 10.0^(floor(log10(abs(ref))) - pin["digits"] + 1)
end
pinned(value, ref::Real, digits::Int) = pinned(value, Dict("value" => ref, "digits" => digits))

@testset "Resonator parameters" begin
    r = resonator_from_frequency(0.01, 10.0; ζ=0.05)
    @test natural_frequency(r) ≈ 10.0
    @test damping_ratio(r) ≈ 0.05
    @test damping_ratio(with_damping_ratio(r, 0.2)) ≈ 0.2
    @test P1LF().n0 * P1LF().resonator.m_ρ ≈ P1().n0 * P1().resonator.m_ρ
    @test P2().host.m_ρ ≈ 0.0444 atol = 1e-4                # values printed in the Results draft
    @test P2().host.T_ρ ≈ 9.7561 atol = 1e-4
    @test natural_frequency(P2().resonator) ≈ 0.5
end

@testset "Host symbol and roots" begin
    for T in (0.0, 9.8), ω in (1.0, 5.0, 10.0)
        h = P1(T_ρ=T).host
        roots = host_roots(h, ω; n_evanescent=50)
        @test all(imag.(roots) .>= 0)
        @test abs(D0(h, roots[1], ω)) < 1e-10 * abs(dD0(h, roots[1], ω) * roots[1])
        # all roots are zeros of the entire form 𝒟₀ k tanh kH
        @test maximum(k -> abs(D0(h, k, ω)) / abs(dD0(h, k, ω) * k), roots) < 1e-10
        # flexural roots come as (k_c, −conj k_c); with pre-tension at low ω the
        # pair can collapse onto the imaginary axis (two extra evanescent roots)
        flex = filter(k -> abs(real(k)) > 1e-8 && imag(k) > 1e-8, roots)
        @test length(flex) in (0, 2)
        length(flex) == 2 && @test flex[2] ≈ -conj(flex[1])
        # evanescent roots: increasing, at least one per branch of cot(κH)
        κ = imag.(filter(k -> abs(real(k)) < 1e-12, roots))
        @test length(κ) == 50 && issorted(κ)
        @test all(n -> any(x -> (n - 1) * π / h.H < x < n * π / h.H, κ), 1:45)
    end
    # derivative against finite differences, also with damping
    for h in (P1().host, HostSpec(P1(T_ρ=9.8).host; τ=1e-3, τ_b=2e-3))
        k, ω, δ = 1.3 + 0.2im, 7.0, 1e-6
        @test dD0(h, k, ω) ≈ (D0(h, k + δ, ω) - D0(h, k - δ, ω)) / 2δ rtol = 1e-7
    end
    @test_throws DomainError host_roots(HostSpec(P1().host; H=Inf), 10.0)

    dl = PINS["matrix"]["decay_lengths"]
    @test 1 / imag(host_roots(P1().host, 10.0)[2]) ≈ dl["flexural_T0"] rtol = 2e-3
    @test 1 / imag(host_roots(P1(T_ρ=9.8).host, 10.0)[2]) ≈ dl["flexural_T98"] rtol = 2e-3
    @test 1 / imag(host_roots(P1().host, 10.0)[4]) ≈ dl["evanescent_first"] rtol = 2e-3
    @test kappa1(P1().host, 10.0) ≈ imag(host_roots(P1().host, 10.0)[4])

    # matrix §0.2 table (independent SciPy computation)
    for row in PINS["matrix"]["host"]
        h, r = P1(T_ρ=row["T_rho"]).host, P1().resonator
        ω = row["omega"]
        @test k0(h, ω) ≈ row["k0"] rtol = 1e-3
        @test dD0(h, k0(h, ω), ω) ≈ row["D0p"] rtol = 1e-3
        @test gamma_r(h, ω; dim=2) ≈ row["gamma_r_2"] rtol = 1e-2
        @test gamma_r(h, ω; dim=3) ≈ row["gamma_r_3"] rtol = 1e-2
        @test Q_rad(h, r, ω) ≈ row["Q_rad"] rtol = 1e-2
    end
end

@testset "Green functions: residue series vs contour integral" begin
    # (T_ρ = 9.8, ω = 1): no complex pair, the flexural roots are evanescent
    cases = [(P1().host, 10.0), (P1(T_ρ=9.8).host, 10.0), (P1().host, 2.0),
             (P1(T_ρ=9.8).host, 1.0), (P1LF(H=3.0).host, 1.0)]
    for (h, ω) in cases
        roots = host_roots(h, ω)
        for x in (0.0, 0.3, 1.5)
            @test green_1d(h, x, ω; roots=roots) ≈ green_1d_integral(h, x, ω) rtol = 1e-8
        end
        @test imag(green_1d(h, 0.0, ω; roots=roots)) ≈ gamma_r(h, ω; dim=2) rtol = 1e-10
        for r in (0.3, 1.5)
            @test green_2d(h, r, ω; roots=roots) ≈ green_2d_integral(h, r, ω) rtol = 1e-8
        end
        @test imag(green_2d_integral(h, 0.0, ω)) ≈ gamma_r(h, ω; dim=3) rtol = 1e-8
    end
    # membrane: slower (κ⁻⁴) convergence of the evanescent series
    h, ω = P2().host, 0.5
    @test green_1d(h, 0.0, ω; roots=host_roots(h, ω; n_evanescent=400)) ≈
          green_1d_integral(h, 0.0, ω) rtol = 1e-6
    @test_throws DomainError green_2d_integral(h, 0.0, ω)
    # damped host: continued roots
    hd = HostSpec(P1().host; τ_b=1e-3)
    @test green_1d(hd, 0.4, 10.0) ≈ green_1d_integral(hd, 0.4, 10.0) rtol = 1e-7
    @test imag(host_roots(hd, 10.0)[1]) > 0
end

@testset "Line-array lattice sums and band edges" begin
    for T in (0.0, 9.8)
        h, a = P1(T_ρ=T).host, 1.0
        for (q, ω) in ((0.0, 9.0), (1.3, 9.7), (π, 9.95), (0.7, 5.0))
            @test lattice_sum_1d(h, q, ω, a) ≈ real(lattice_sum_1d_modal(h, q, ω, a)) rtol = 1e-7
        end
    end

    for (T, key) in ((0.0, "T0"), (9.8, "T98"))
        ps = P1(T_ρ=T)
        h, r, a = ps.host, ps.resonator, ps.a
        edges = PINS["paper"]["band_edges"][key]
        ωΓ = omega_gamma_1d(h, r, a)
        ωX = omega_X_1d(h, r, a; ωmin=9.0)
        @test pinned(ωX, edges["omega_X"], edges["digits"])
        @test pinned(ωΓ, edges["omega_Gamma"], edges["digits"])
        @test pinned((ωΓ - ωX) / natural_frequency(r), PINS["paper"]["gap_width"][key])
        # the lower-band maximum is attained at the zone boundary
        @test lower_band_maximum_1d(h, r, a; nq=11, ωmin=9.5, npts=201) ≈ ωX rtol = 1e-9
        # eq. lat_gamma: ω_Γ ≈ ω_r/√(1 + Π_L)
        ΠL = lattice_compliance_number_1d(h, r, a)
        @test pinned(ΠL, PINS["paper"]["Pi_L"][key])
        @test ωΓ ≈ natural_frequency(r) / sqrt(1 + ΠL) rtol = 1e-5
        # band-edge roots satisfy the Bloch condition
        @test abs(bloch_residual_1d(h, r, π / a, ωX, a)) < 1e-10 * abs(inv_alpha(r, ωX))
    end
    p = P1()
    @test pinned(Pi_L_estimate_plate(p.resonator, p.host, p.a), PINS["paper"]["Pi_L"]["plate_estimate_T0"])
    unit = ResonatorSpec(m_ρ=1.0, k_ρ=1.0)
    @test pinned(Pi_L_estimate_square(unit, HostSpec(m_ρ=0.0, C_ρ=1.0), 1.0),
                 PINS["paper"]["Pi_L"]["square_coefficient"])
end

@testset "Local (homogenised) branches" begin
    lm = PINS["paper"]["local_model"]
    for (T, key) in ((0.0, "T0"), (9.8, "T98"))
        ps = P1(T_ρ=T)
        h, r, n0 = ps.host, ps.resonator, ps.n0
        @test pinned(anticrossing_width(h, r, n0), lm["anticrossing_$(key)"])
        @test pinned(crossing_wavenumber(h, r) * ps.a / π, lm["kc_a_over_pi_$(key)"])
        @test pinned(local_branches(h, r, n0, π / ps.a).lower, lm["lower_at_zone_edge_$(key)"])
        for k in (0.05, 0.5, 2.0, 5.0)
            b = local_branches(h, r, n0, k)
            @test b.lower < natural_frequency(r) < b.upper
            # both branches are roots of the lossless homogenised law
            for ω in (b.lower, b.upper)
                meff = h.m_ρ + n0 * r.m_ρ * r.k_ρ / (r.k_ρ - ω^2 * r.m_ρ)
                @test abs(D0(h, k, ω) - ω^2 * (meff - h.m_ρ)) < 1e-9 * ω^2 * added_mass(k, h.H)
            end
        end
    end
end

@testset "Single resonator" begin
    ps = P1()
    h, r = ps.host, ps.resonator
    ω̃ = renormalised_frequency(h, r; dim=2)
    s = single_resonator_1d(h, r, ω̃)
    @test abs(s.x) < 1e-10
    @test s.R ≈ -1 atol = 1e-10                    # total reflection, lossless
    @test abs(s.T) < 1e-10
    # critical coupling: absorbed fraction reaches 1/2 and never exceeds it
    c = critical_damping(h, r, ω̃)
    rd = ResonatorSpec(r; c_ρ=c)
    @test gamma_d(rd, ω̃) ≈ gamma_r(h, ω̃) rtol = 1e-10
    ωs = range(0.97ω̃, 1.03ω̃; length=601)
    𝒜 = [single_resonator_1d(h, rd, ω).absorbed for ω in ωs]
    @test maximum(𝒜) <= 0.5 + 1e-12
    @test maximum(𝒜) ≈ 0.5 rtol = 1e-3
    @test all(ω -> let s = single_resonator_1d(h, rd, ω)
                       s.absorbed ≈ 2s.γ_d * s.γ_r / (s.x^2 + (s.γ_d + s.γ_r)^2)
                   end, ωs[1:50:end])
    # 3D cross-sections: bounds and attainment
    ω̃3 = renormalised_frequency(h, r; dim=3)
    cs = cross_sections_3d(h, r, ω̃3)
    @test cs.σ_sc ≈ cs.σ_sc_max rtol = 1e-8
    rd3 = ResonatorSpec(r; c_ρ=critical_damping(h, r, ω̃3; dim=3))
    cs3 = cross_sections_3d(h, rd3, ω̃3)
    @test cs3.σ_abs <= cs3.σ_abs_max * (1 + 1e-12)
    @test cs3.σ_abs ≈ cs3.σ_abs_max rtol = 1e-3
end

@testset "Multiple scattering" begin
    ps = P1()
    h, r = ps.host, ps.resonator
    ω = 9.8
    roots = host_roots(h, ω)
    # N = 1 reduces to the single-resonator closed form
    f1 = foldy_1d(h, ω, [0.37], [r]; roots=roots)
    s = single_resonator_1d(h, r, ω)
    @test f1.R ≈ s.R * exp(2im * k0(h, ω) * 0.37) rtol = 1e-12
    @test f1.T ≈ s.T rtol = 1e-12
    # energy identity, damped and detuned array
    xs = collect(0.0:1.0:9.0) .+ 0.1 .* sin.(1:10)
    rs = [resonator_from_frequency(0.01, 10.0 + 0.2sin(3i); ζ=0.01 + 0.01cos(i)) for i in 1:10]
    fN = foldy_1d(h, ω, xs, rs; roots=roots)
    @test abs(fN.energy_residual) < 1e-12
    @test 0 < fN.absorbed < 1
    # single-mode Foldy coincides with the single-mode Fabry–Pérot law
    for ω in (9.0, 9.8, 10.3), N in (1, 5, 12)
        fs = foldy_1d(h, ω, collect(0.0:(N - 1)), fill(r, N); roots=host_roots(h, ω)[1:1])
        @test abs2(fs.T) ≈ fabry_perot_transmission(h, r, ω, 1.0, N) rtol = 1e-10
    end
    # 3D: N = 1 reproduces the closed-form cross-sections; optical theorem
    rd = with_damping_ratio(r, 0.02)
    f3 = foldy_2d(h, ω, [(0.0, 0.0)], [rd]; roots=roots)
    cs = cross_sections_3d(h, rd, ω)
    @test f3.σ_sc ≈ cs.σ_sc rtol = 1e-10
    @test f3.σ_abs ≈ cs.σ_abs rtol = 1e-10
    cluster = [(0.0, 0.0), (1.0, 0.2), (-0.4, 1.1), (0.6, -0.9), (2.0, 1.5)]
    fc = foldy_2d(h, ω, cluster, fill(rd, 5); θ_inc=0.3, roots=roots)
    @test abs(fc.optical_residual) < 1e-9 * (fc.σ_sc + fc.σ_abs)
end
