using HydroElasticFEM_Colomes2026
using HydroElasticFEM_Colomes2026.References
include(joinpath(@__DIR__, "..", "..", "src", "L4_4_aniso_island.jl"))
using .L4_4
using Printf
h = HostSpec(g=9.8, ρ=1000.0, H=10.0, m_ρ=0.01, C_ρ=0.49)
# port vs python (plain Gaussian)
for (ω, load, Lx, Ly, ref) in [(8,0.03,12,8,46.7),(8,0.03,8,12,141.1),(12,0.03,12,8,-35.0),(8,0.01,12,8,122.7),(8,0.01,8,12,396.4)]
    χ = chi(ω, 10.0, 0.0); kφ, kb = local_k_table(h, ω, load, χ)
    f = ray_focus(kφ, kb, Lx, Ly, 2.0, 2.5; taper=false)
    @printf("port ω=%g load=%g (%g,%g): ray %.2f (py %.1f) thin %.2f\n", ω, load, Lx, Ly, f, ref, thin_lens_gaussian(h, ω, load, χ, Lx, Ly))
end
# FFT check
a = rand(ComplexF64, 16); b = copy(a); fft!(b); ifft!(b); @printf("fft roundtrip %.2e\n", maximum(abs.(a-b)))
for (ω, load, ζ, Lx, Ly) in [(8,0.03,0,6,4),(8,0.03,0,4,6),(12,0.03,0,6,4),(8,0.01,0,6,4),(8,0.01,0,4,6),(10,0.01,0.05,6,4),(10,0.01,0.05,4,6),(10,0.003,0.02,6,4),(10,0.003,0.02,4,6),(10,0.005,0.05,6,4)]
    χ = chi(ω, 10.0, ζ); kφ, kb = local_k_table(h, ω, load, χ)
    fr = ζ == 0 ? ray_focus(kφ, kb, Lx, Ly, 2.0, 2.5) : NaN
    J = thin_lens_J(kφ, kb, 2.0, 2.5); fthin = kb*Ly^2/(Lx*J)
    Φ = eikonal_phase(kφ, kb, Lx, Ly, [0.0], 2.0, 2.5)[1]
    xs = [2.5Lx + d for d in (0.5, 1.5, 2.5)]
    t = @elapsed ys, ψs = pe_march(kφ, kb, Lx, Ly, 2.0, 2.5, xs; Ny=4096, Yp=100.0, dx=0.05)
    res = String[]
    for (xm, ψ) in zip(xs, ψs)
        keep = findall(y -> 0 <= y <= 2.5Ly + 2.0, ys)   # truncated half line as in FE (y_end ≈ 2.5Ly + gap)
        yb, Tb = backprop(ys[keep], ψ[keep], xm, kb; Ny=4096, Yp=100.0)
        c = curvature_fit(yb, Tb, kb, 0.5Ly)
        yb2, Tb2 = backprop(ys[ys .>= 0], ψ[ys .>= 0], xm, kb; Ny=4096, Yp=100.0)
        c2 = curvature_fit(yb2, Tb2, kb, 0.5Ly)
        push!(res, @sprintf("[x=%.1f fPE %.2f (untrunc %.2f) A %.3f]", xm, c.f, c2.f, c.A0))
    end
    @printf("ω=%g load=%g ζ=%g (%g,%g): k0=%.4f ray %.2f thin(J) %.2f thinG %.2f | Φx(0)=%.3f+%.3fi | %s  (%.1fs)\n", ω, load, ζ, Lx, Ly, kb, fr, fthin, thin_lens_gaussian(h, ω, load, χ, Lx, Ly), real(Φ), imag(Φ), join(res, " "), t)
end
