"""
Analytical and semi-analytical references of the paper, shared by all tests.
This is the only shared code of the repository: it contains no FE, only the
formulas of Section 3 (`sec:dispersion_relation`, `dispersion_theory.tex`) and
Section 4 (`sec:discrete_lattice`, `discrete_lattices.tex`) that the FE results
are compared with. Each function cites its equation label.

Conventions: density-normalised quantities (subscript ρ: m_ρ = m/ρ,
T_ρ = T/ρ, C_ρ = D/ρ), time factor exp(−iωt), configuration A (the plate
covers the whole free surface, flat bottom of depth H). `dim = 2` is the
vertical section, `dim = 3` the 3D fluid with a 2D isotropic plate.

Tested in `test/test_references.jl` (internal consistency and pins against the
values published in the paper, `test/reference/paper_values.toml`).
"""
module References

using LinearAlgebra
using Parameters
using SpecialFunctions: besselj, hankelh1
using QuadGK: quadgk
using Roots: find_zero, Bisection

export HostSpec, ResonatorSpec
export resonator_from_frequency, natural_frequency, damping_ratio, with_damping_ratio

# ─── Model parameters ───────────────────────────────────────────────────────

"""
Bare host: pre-tensioned Kirchhoff–Love plate on a fluid of depth `H`
(`H = Inf` for deep water), eq. (lat_symbol). `τ`, `τ_b` are the tension and
bending damping times; `ν` is only used by 3D plates.
"""
@with_kw struct HostSpec
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64
    C_ρ::Float64 = 0.0
    T_ρ::Float64 = 0.0
    τ::Float64 = 0.0
    τ_b::Float64 = 0.0
    ν::Float64 = 0.0
end

"Single resonator: mass, stiffness and damping per unit fluid density."
@with_kw struct ResonatorSpec
    m_ρ::Float64
    k_ρ::Float64
    c_ρ::Float64 = 0.0
end

"Resonator with natural frequency `ω_r` and damping ratio `ζ`, eq. (resonator_natural_freq_damping)."
function resonator_from_frequency(m_ρ::Real, ω_r::Real; ζ::Real=0.0)
    k_ρ = m_ρ * ω_r^2
    return ResonatorSpec(m_ρ=m_ρ, k_ρ=k_ρ, c_ρ=2ζ * sqrt(k_ρ * m_ρ))
end

natural_frequency(r::ResonatorSpec) = sqrt(r.k_ρ / r.m_ρ)
damping_ratio(r::ResonatorSpec) = r.c_ρ / (2sqrt(r.k_ρ * r.m_ρ))
with_damping_ratio(r::ResonatorSpec, ζ::Real) = resonator_from_frequency(r.m_ρ, natural_frequency(r); ζ=ζ)


export added_mass, D0, dD0, k_open, k0, host_roots, kappa1
export green_1d, green_1d_integral, green_1d_patch, green_2d, green_2d_integral
export inv_alpha, alpha, gamma_r, gamma_d, energy_flux
export lattice_sum_1d, lattice_sum_1d_modal, bloch_residual_1d, bloch_roots_1d
export omega_gamma_1d, omega_X_1d, lower_band_maximum_1d, lattice_compliance_number_1d, bloch_qa_1d, bare_frequency
export gap_asymptotic_1d, gap_a4_1d, bloch_decay_1d
export Pi_L_estimate_plate, Pi_L_estimate_membrane, Pi_L_estimate_square
export local_branches, crossing_wavenumber, anticrossing_width
export single_resonator_1d, cross_sections_3d, renormalised_frequency, critical_damping, Q_rad
export foldy_1d, foldy_2d, point_source_field_1d, single_mode_cos_qa, fabry_perot_transmission

# ─── Bare host symbol, eq. (lat_symbol) ─────────────────────────────────────

_kt(k, H) = isinf(H) ? k : k * tanh(k * H)

"Fluid added mass per unit area M_f,ρ = 1/(k tanh kH), eq. (plane_wave_added_mass)."
added_mass(k, H) = 1 / _kt(k, H)

"Polynomial (structural) part g + T̃k² + C̃k⁴ − ω²m_ρ of the symbol."
_A(h::HostSpec, k, ω) = h.g + h.T_ρ * (1 - im * ω * h.τ) * k^2 +
                        h.C_ρ * (1 - im * ω * h.τ_b) * k^4 - ω^2 * h.m_ρ
_dA(h::HostSpec, k, ω) = 2h.T_ρ * (1 - im * ω * h.τ) * k + 4h.C_ρ * (1 - im * ω * h.τ_b) * k^3

_lossless(h::HostSpec) = iszero(h.τ) && iszero(h.τ_b)
_maybe_real(h::HostSpec, k, z) = (_lossless(h) && k isa Real) ? real(z) : z

"""
    D0(h, k, ω)

Bare-host symbol 𝒟₀(k, ω) = 𝒵_f + g + T̃k² + C̃k⁴ − ω²m_ρ, with
𝒵_f = −ω²/(k tanh kH). Even in k; 𝒟₀ → −∞ as k → 0.
"""
function D0(h::HostSpec, k, ω)
    iszero(k) && return -Inf
    return _maybe_real(h, k, -ω^2 / _kt(k, h.H) + _A(h, k, ω))
end

"∂𝒟₀/∂k."
function dD0(h::HostSpec, k, ω)
    dZ = isinf(h.H) ? ω^2 / k^2 :
         ω^2 * (tanh(k * h.H) + k * h.H * sech(k * h.H)^2) / (k * tanh(k * h.H))^2
    return _maybe_real(h, k, dZ + _dA(h, k, ω))
end

"Entire form P(k) = 𝒟₀(k) k tanh(kH): same non-zero roots, no poles."
_P(h::HostSpec, k, ω) = _A(h, k, ω) * _kt(k, h.H) - ω^2
_dP(h::HostSpec, k, ω) = _dA(h, k, ω) * _kt(k, h.H) +
                         _A(h, k, ω) * (isinf(h.H) ? one(k) : tanh(k * h.H) + k * h.H * sech(k * h.H)^2)

# ─── Roots of the symbol ────────────────────────────────────────────────────

function _bracket_root(f, a, b)
    return find_zero(f, (a, b), Bisection(); xatol=0.0, xrtol=4eps())
end

"Open-water wavenumber: k tanh kH = ω²/g."
function k_open(ω::Real, H::Real, g::Real)
    f(k) = _kt(k, H) - ω^2 / g
    kmax = max(1.0, 2ω^2 / g)
    while f(kmax) < 0
        kmax *= 2
    end
    return _bracket_root(f, 1e-12, kmax)
end

"""
    k0(h, ω)

Real propagating root of the lossless bare host (𝒟₀ is monotone increasing in k > 0,
so the root is unique).
"""
function k0(h::HostSpec, ω::Real)
    hl = _lossless(h) ? h : HostSpec(h; τ=0.0, τ_b=0.0)
    f(k) = D0(hl, k, ω)
    kmin, kmax = 1e-10, 1.0
    while f(kmax) < 0
        kmax *= 2
    end
    return _bracket_root(f, kmin, kmax)
end

# Evanescent fluid roots k = iκ of 𝒟₀(iκ) = ω² cot(κH)/κ + g − Tκ² + Cκ⁴ − ω²m.
# Searched in the pole-free form E(κ) = ω² cos(κH) + κ A(κ) sin(κH) (same roots;
# E = ±ω² where sin κH = 0): for large κ the roots crowd to within ω²/(κAH) of
# the poles of cot(κH) and would be missed by a search in the cot form.
function _imaginary_roots(h::HostSpec, ω::Real, n::Int)
    isinf(h.H) && return ComplexF64[]
    E(κ) = ω^2 * cos(κ * h.H) +
           κ * (h.g - h.T_ρ * κ^2 + h.C_ρ * κ^4 - ω^2 * h.m_ρ) * sin(κ * h.H)
    roots = Float64[]
    j = 0
    while length(roots) < n
        j <= 4n + 10 || error("Evanescent root search did not find $(n) roots")
        lo, hi = j * π / h.H, (j + 1) * π / h.H
        npts = j < 5 ? 2001 : 101
        κs = range(j == 0 ? 1e-12 * hi : lo, hi; length=npts)
        Es = E.(κs)
        for i in 1:(npts - 1)
            if Es[i] == 0
                push!(roots, κs[i])
            elseif sign(Es[i]) != sign(Es[i + 1])
                push!(roots, _bracket_root(E, κs[i], κs[i + 1]))
            end
        end
        j += 1
    end
    return im .* roots[1:n]
end

"""
    kappa1(h, ω)

Decay rate κ₁ of the slowest fluid evanescent mode, e^{−κ₁|x|} (first purely
imaginary root of the lossless host). 1/κ₁ approaches 2H/π for ω²H/g ≫ 1 and
sets how far sources and scatterers must be from the domain ends.
"""
function kappa1(h::HostSpec, ω::Real)
    hl = _lossless(h) ? h : HostSpec(h; τ=0.0, τ_b=0.0)
    return imag(only(_imaginary_roots(hl, ω, 1)))
end

function _newton(F, dF, z; tol=1e-14, maxiter=100)
    for _ in 1:maxiter
        δ = F(z) / dF(z)
        z -= δ
        abs(δ) <= tol * max(abs(z), 1.0) && return z, true
    end
    return z, false
end

# Complex (flexural) roots off the axes, first quadrant, seeded by the
# deep-water polynomial C k⁵ + T k³ + (g − ω²m) k − ω² = 0 (eq. deep_water_poly).
function _complex_roots(h::HostSpec, ω::Real)
    (h.C_ρ > 0 || h.T_ρ > 0) || return ComplexF64[]
    c = [-ω^2, h.g - ω^2 * h.m_ρ, 0.0, h.T_ρ, 0.0, h.C_ρ]   # ascending powers
    while iszero(c[end])
        pop!(c)
    end
    nd = length(c) - 1
    companion = zeros(nd, nd)
    companion[2:end, 1:(end - 1)] .= I(nd - 1)
    companion[:, end] .= -c[1:(end - 1)] ./ c[end]
    seeds = filter(z -> real(z) > 0 && imag(z) > 0, eigvals(companion))
    found = ComplexF64[]
    for s in seeds
        z, ok = _newton(k -> _P(h, k, ω), k -> _dP(h, k, ω), complex(s))
        ok || continue
        z = complex(abs(real(z)), abs(imag(z)))
        abs(real(z)) < 1e-8abs(z) && continue        # converged onto an evanescent root
        abs(imag(z)) < 1e-8abs(z) && continue        # converged onto k₀
        any(w -> abs(w - z) < 1e-8abs(z), found) || push!(found, z)
    end
    return found
end

"""
    host_roots(h, ω; n_evanescent=100)

Roots of 𝒟₀(k, ω) = 0 with Im k ≥ 0 used in the residue series
(`eq:lat_green`): the propagating root k₀, the complex flexural roots k_c and
−conj(k_c), and `n_evanescent` evanescent fluid roots iκₙ, ordered by κₙ.
For large n, κₙ → nπ/H, but for ω²H/g ≫ 1 the first roots lie near
(n − ½)π/H, so the slowest evanescent decay length 1/κ₁ can approach 2H/π.
With material damping the lossless roots are continued by Newton iteration.

Finite depth only: for H = ∞ the symbol has a branch point at k = 0 and the
residue series is incomplete.
"""
function host_roots(h::HostSpec, ω::Real; n_evanescent::Int=100)
    isfinite(h.H) || throw(DomainError(h.H, "residue series requires finite depth"))
    hl = HostSpec(h; τ=0.0, τ_b=0.0)
    kc = _complex_roots(hl, ω)
    roots = ComplexF64[k0(hl, ω); kc; -conj.(kc); _imaginary_roots(hl, ω, n_evanescent)]
    _lossless(h) && return roots
    damped = map(roots) do z
        zd, ok = _newton(k -> _P(h, k, ω), k -> _dP(h, k, ω), z)
        ok || error("Root continuation to the damped host failed from k = $(z)")
        imag(zd) < 0 ? -zd : zd
    end
    return damped
end

# ─── Green functions, eq. (lat_green) ───────────────────────────────────────

"""
    green_1d(h, x, ω; roots=host_roots(h, ω))

Vertical-section Green function, residue form: G(x) = i Σₙ e^{ikₙ|x|}/𝒟₀′(kₙ).
"""
function green_1d(h::HostSpec, x::Real, ω::Real; roots=host_roots(h, ω))
    return im * sum(k -> exp(im * k * abs(x)) / dD0(h, k, ω), roots)
end

# Contour indented below k₀: k(t) = t − iβ sin(πt/(2k₀)) on [0, 2k₀], then the
# real axis. `x` is the oscillation frequency of the integrand on the real axis.
# For x > 0 the real-axis part is integrated panel by panel (one period each) up
# to K, and ∫_K^∞ f(k) cos(kx − φ) dk is closed by two integration-by-parts terms
# of the smooth asymptotic amplitude `tail_amplitude` f.
function _indented_integral(h::HostSpec, ω::Real, integrand; x::Real=0.0,
                            tail_amplitude=nothing, tail_phase::Real=0.0)
    kp = k0(h, ω)
    β = 0.2kp
    path(t) = t - im * β * sin(π * t / (2kp))
    dpath(t) = 1 - im * β * π / (2kp) * cos(π * t / (2kp))
    I1, _ = quadgk(t -> integrand(path(t)) * dpath(t), 0.0, kp, 2kp; rtol=1e-12)
    if x == 0
        I2, _ = quadgk(integrand, 2kp, Inf; rtol=1e-12)
        return I1 + I2
    end
    K = max(2000.0, 100kp)
    edges = range(2kp, K; length=max(2, ceil(Int, (K - 2kp) * x / π)) + 1)
    # absolute tolerance per panel: tiny far-tail panels must not chase round-off
    atol = 1e-14 * abs(I1) / length(edges)
    I2 = sum(quadgk(integrand, edges[i], edges[i + 1]; rtol=1e-12, atol=atol)[1]
             for i in 1:(length(edges) - 1))
    f = tail_amplitude
    δ = 1e-4K
    df = (f(K + δ) - f(K - δ)) / 2δ
    tail = -f(K) * sin(K * x - tail_phase) / x - df * cos(K * x - tail_phase) / x^2
    return I1 + I2 + tail
end

"G(x) = (1/π)∫₀^∞ cos(kx)/𝒟₀ dk on the outgoing contour (independent check of `green_1d`)."
function green_1d_integral(h::HostSpec, x::Real, ω::Real)
    x = abs(x)
    return _indented_integral(h, ω, k -> cos(k * x) / D0(h, k, ω);
                              x=x, tail_amplitude=k -> 1 / D0(h, k, ω)) / π
end

"""
    green_2d(h, r, ω; roots=host_roots(h, ω))

3D isotropic Green function, residue form (r > 0):
G(r) = (i/2) Σₙ kₙ/𝒟₀′(kₙ) H₀⁽¹⁾(kₙr). Use `green_2d_integral` at r = 0.
"""
function green_2d(h::HostSpec, r::Real, ω::Real; roots=host_roots(h, ω))
    r > 0 || throw(DomainError(r, "residue form is singular term by term at r = 0; use green_2d_integral"))
    return im / 2 * sum(k -> k / dD0(h, k, ω) * hankelh1(0, k * r), roots)
end

"G(r) = (1/2π)∫₀^∞ J₀(kr) k/𝒟₀ dk on the outgoing contour; finite at r = 0 iff C_ρ > 0."
function green_2d_integral(h::HostSpec, r::Real, ω::Real)
    r == 0 && h.C_ρ == 0 && throw(DomainError(r, "G(0) diverges for a membrane in 3D"))
    # tail: J₀(kr) ≈ √(2/(πkr)) cos(kr − π/4)
    return _indented_integral(h, ω, k -> besselj(0, k * r) * k / D0(h, k, ω);
                              x=r, tail_phase=π / 4,
                              tail_amplitude=k -> sqrt(2 / (π * k * r)) * k / D0(h, k, ω)) / (2π)
end

# ─── Attachment strength and single resonator ───────────────────────────────

"1/α = 1/(ω²m_ρ,r) − 1/κ, κ = k_ρ,r − iωc_ρ,r, eq. (lat_alpha)."
inv_alpha(r::ResonatorSpec, ω) = 1 / (ω^2 * r.m_ρ) - 1 / (r.k_ρ - im * ω * r.c_ρ)
alpha(r::ResonatorSpec, ω) = 1 / inv_alpha(r, ω)

"Radiative width γ_r = Im G(0), eq. (lat_gamma_r)."
function gamma_r(h::HostSpec, ω::Real; dim::Int=2)
    kp = k0(h, ω)
    d = dD0(h, kp, ω)
    return dim == 2 ? 1 / d : kp / (2d)
end

"Dissipative width γ_d = ωc_ρ,r/|κ|², eq. (lat_detuning)."
gamma_d(r::ResonatorSpec, ω) = ω * r.c_ρ / abs2(r.k_ρ - im * ω * r.c_ρ)

"Time-averaged energy flux of a plane wave of amplitude A, eq. (lat_flux)."
energy_flux(h::HostSpec, ω::Real, A) = h.ρ * ω / 4 * dD0(h, k0(h, ω), ω) * abs2(A)

"G(0) for the given dimension (residue series in 2D, contour integral in 3D)."
_G0(h, ω, dim) = dim == 2 ? green_1d(h, 0.0, ω) : green_2d_integral(h, 0.0, ω)

"""
    single_resonator_1d(h, r, ω)

Line resonator in the vertical section: detuning x, widths, R = iγ_r/(x − iΓ),
T = 1 + R and absorbed fraction 𝒜 = 1 − |R|² − |T|² (eq. lat_absorbed_1d).
"""
function single_resonator_1d(h::HostSpec, r::ResonatorSpec, ω::Real)
    G0 = green_1d(h, 0.0, ω)
    x = real(inv_alpha(r, ω) - G0)
    γr, γd = gamma_r(h, ω; dim=2), gamma_d(r, ω)
    R = im * γr / (x - im * (γd + γr))
    T = 1 + R
    return (x=x, γ_r=γr, γ_d=γd, G0=G0, R=R, T=T, absorbed=1 - abs2(R) - abs2(T))
end

"Scattering and absorption cross-sections of a point resonator in 3D, eq. (lat_bound)."
function cross_sections_3d(h::HostSpec, r::ResonatorSpec, ω::Real)
    G0 = green_2d_integral(h, 0.0, ω)
    x = real(inv_alpha(r, ω) - G0)
    γr, γd = gamma_r(h, ω; dim=3), gamma_d(r, ω)
    kp = k0(h, ω)
    Γ2 = x^2 + (γd + γr)^2
    return (x=x, γ_r=γr, γ_d=γd, G0=G0, σ_sc=4 / kp * γr^2 / Γ2, σ_abs=4 / kp * γd * γr / Γ2,
            σ_sc_max=4 / kp, σ_abs_max=1 / kp)
end

"""
    renormalised_frequency(h, r; dim=2, window=(0.5, 1.5))

Root ω̃ of x(ω) = 1/(ω²m_ρ,r) − 1/k_ρ,r − Re G(0; ω) = 0 (eq. lat_renorm_single),
searched in `window .* ω_r`.
"""
function renormalised_frequency(h::HostSpec, r::ResonatorSpec; dim::Int=2, window=(0.5, 1.5))
    ωr = natural_frequency(r)
    f(ω) = 1 / (ω^2 * r.m_ρ) - 1 / r.k_ρ - real(_G0(h, ω, dim))
    ωs = range(window[1] * ωr, window[2] * ωr; length=201)
    fs = f.(ωs)
    i = findfirst(j -> sign(fs[j]) != sign(fs[j + 1]), 1:(length(ωs) - 1))
    i === nothing && error("No renormalised resonance in the window $(window) × ω_r")
    return _bracket_root(f, ωs[i], ωs[i + 1])
end

"Dashpot c_ρ,r giving critical coupling γ_d = γ_r at frequency ω (smaller root)."
function critical_damping(h::HostSpec, r::ResonatorSpec, ω::Real; dim::Int=2)
    γ = gamma_r(h, ω; dim=dim)
    disc = 1 - 4γ^2 * r.k_ρ^2
    disc >= 0 || error("Critical coupling unattainable: 2γ_r k_ρ,r = $(2γ * r.k_ρ) > 1")
    return (1 - sqrt(disc)) / (2γ * ω)
end

"Radiative quality factor Q_rad ≈ 𝒟₀′(k₀)/(ω²m_ρ,r) of a line resonator, eq. (lat_Qrad)."
Q_rad(h::HostSpec, r::ResonatorSpec, ω::Real) = dD0(h, k0(h, ω), ω) / (ω^2 * r.m_ρ)

# ─── Line-array lattice sums and band edges ─────────────────────────────────

"""
    lattice_sum_1d(h, q, ω, a; N=4000, exclude_zero=false)

𝒮(q, ω) = (1/a) Σₙ 1/𝒟₀(q + 2πn/a) (eq. lat_bloch), |n| ≤ N plus the
asymptotic bending tail. `exclude_zero=true` drops n = 0 (σ′/a, eq. lat_exact_em).
"""
function lattice_sum_1d(h::HostSpec, q::Real, ω::Real, a::Real; N::Int=4000, exclude_zero::Bool=false)
    s = zero(D0(h, 1.0, ω))
    for n in (-N):N
        (exclude_zero && n == 0) && continue
        d = D0(h, q + 2π * n / a, ω)
        s += isinf(d) ? 0 : 1 / d
    end
    if h.C_ρ > 0
        s += 2 / (h.C_ρ * (2π / a)^4) / (3N^3)
    elseif h.T_ρ > 0
        s += 2 / (h.T_ρ * (2π / a)^2) / N
    end
    return s / a
end

"""
    lattice_sum_1d_modal(h, q, ω, a; roots=host_roots(h, ω))

Multimode form of 𝒮 (eq. lat_1d_exact): −Σₘ sin(kₘa)/(𝒟₀′(kₘ)(cos qa − cos kₘa)).
"""
function lattice_sum_1d_modal(h::HostSpec, q::Real, ω::Real, a::Real; roots=host_roots(h, ω))
    return -sum(k -> sin(k * a) / (dD0(h, k, ω) * (cos(q * a) - cos(k * a))), roots)
end

"Bloch residual 1/α(ω) − 𝒮(q, ω); zero on the Bloch bands (eq. lat_bloch)."
bloch_residual_1d(h::HostSpec, r::ResonatorSpec, q, ω, a; N::Int=4000) =
    inv_alpha(r, ω) - lattice_sum_1d(h, q, ω, a; N=N)

"""
    bloch_roots_1d(h, r, q, a, ωs; N=4000)

Real band frequencies at Bloch wavenumber q in the grid `ωs` (lossless).
Sign changes across the empty-lattice poles of 𝒮 are rejected.
"""
function bloch_roots_1d(h::HostSpec, r::ResonatorSpec, q::Real, a::Real, ωs; N::Int=4000)
    f(ω) = real(bloch_residual_1d(h, r, q, ω, a; N=N))
    fs = f.(ωs)
    roots = Float64[]
    for i in 1:(length(ωs) - 1)
        (isfinite(fs[i] * fs[i + 1]) && fs[i] * fs[i + 1] < 0) || continue
        ω = _bracket_root(f, ωs[i], ωs[i + 1])
        abs(f(ω)) < 1e-6 * (1 + abs(fs[i]) + abs(fs[i + 1])) && push!(roots, ω)
    end
    return roots
end

"""
    omega_gamma_1d(h, r, a; N=4000)

Upper band edge at the zone centre: 1/(ω²m_ρ,r) = 1/k_ρ,r + n₀σ′(0, ω) (eq. lat_gamma).
"""
function omega_gamma_1d(h::HostSpec, r::ResonatorSpec, a::Real; N::Int=4000)
    ωr = natural_frequency(r)
    f(ω) = 1 / (ω^2 * r.m_ρ) - 1 / r.k_ρ - lattice_sum_1d(h, 0.0, ω, a; N=N, exclude_zero=true)
    return _bracket_root(f, 0.5ωr, ωr * (1 - 1e-12))
end

"Lower band edge at the zone boundary q = π/a: largest Bloch root below ω_Γ."
function omega_X_1d(h::HostSpec, r::ResonatorSpec, a::Real; N::Int=4000, ωmin=nothing, npts::Int=2001)
    ωΓ = omega_gamma_1d(h, r, a; N=N)
    lo = ωmin === nothing ? 0.5natural_frequency(r) : ωmin
    roots = bloch_roots_1d(h, r, π / a, a, range(lo, ωΓ * (1 - 1e-9); length=npts); N=N)
    isempty(roots) && error("No lower-band root at the zone boundary below ω_Γ = $(ωΓ)")
    return maximum(roots)
end

"Maximum of the lower band over q ∈ (0, π/a] (checks that the edge is at the zone boundary)."
function lower_band_maximum_1d(h::HostSpec, r::ResonatorSpec, a::Real;
                               nq::Int=81, ωmin=nothing, npts::Int=1001, N::Int=4000)
    ωΓ = omega_gamma_1d(h, r, a; N=N)
    lo = ωmin === nothing ? 0.9natural_frequency(r) : ωmin
    ωs = range(lo, ωΓ * (1 - 1e-9); length=npts)
    best = -Inf
    for q in range(π / a / nq, π / a; length=nq)
        roots = bloch_roots_1d(h, r, q, a, ωs; N=N)
        isempty(roots) || (best = max(best, maximum(roots)))
    end
    return best
end

"""
    bloch_qa_1d(h, r, a, ω; nc=2001, roots=host_roots(h, ω))

Real Bloch phases qa ∈ [0, π] of a lossless line array at frequency ω: the
roots in c = cos qa of 1/α(ω) = 𝒮(q, ω), with 𝒮 in the multimode form
(`lattice_sum_1d_modal`), which is a sum of simple poles in c. Scanned on `nc`
points of [−1, 1]; sign changes across the pole c = cos k₀a are rejected. An
empty result means that ω lies in a stop band.
"""
function bloch_qa_1d(h::HostSpec, r::ResonatorSpec, a::Real, ω::Real; nc::Int=2001, roots=host_roots(h, ω))
    s = [sin(k * a) / dD0(h, k, ω) for k in roots]
    cm = [cos(k * a) for k in roots]
    F(c) = real(inv_alpha(r, ω) + sum(s ./ (c .- cm)))
    cs = range(-1.0, 1.0; length=nc)
    Fs = F.(cs)
    qa = Float64[]
    for i in 1:(nc - 1)
        (isfinite(Fs[i] * Fs[i + 1]) && Fs[i] * Fs[i + 1] < 0) || continue
        c = _bracket_root(F, cs[i], cs[i + 1])
        abs(F(c)) < 1e-6 * (abs(Fs[i]) + abs(Fs[i + 1])) && push!(qa, acos(clamp(c, -1.0, 1.0)))
    end
    return sort!(qa)
end

"""
    bloch_decay_1d(h, r, a, ω; cmax=1e4, nc=4001, roots=host_roots(h, ω))

Attenuations Im(q)a of the Bloch modes of a lossless line array with real
c = cos qa, |c| > 1, at frequency ω: the real roots of 1/α(ω) = 𝒮(q, ω)
(multimode form, as `bloch_qa_1d`), Im(q)a = acosh|c| (eq. lat_decay), sorted by
magnitude and signed: negative for c < −1 (Re q = π/a), positive for c > 1 (Re q = 0). Besides the mode that continues the
propagating branch into the stop band, the list contains the modes that
continue the fluid evanescent roots kₙ = iκₙ (c ≈ cosh κₙa); the slowest of
them, Im q ≈ κ₁, bounds the attenuation of long arrays. Complex roots
(Re q ∉ {0, π/a}) are not searched.
"""
function bloch_decay_1d(h::HostSpec, r::ResonatorSpec, a::Real, ω::Real; cmax::Real=1e4, nc::Int=4001,
                        roots=host_roots(h, ω))
    s = [sin(k * a) / dD0(h, k, ω) for k in roots]
    cm = [cos(k * a) for k in roots]
    F(c) = real(inv_alpha(r, ω) + sum(s ./ (c .- cm)))
    found = Float64[]
    for sgn in (1.0, -1.0)
        cs = sgn .* exp.(range(log(1 + 1e-12), log(cmax); length=nc))
        Fs = F.(cs)
        for i in 1:(nc - 1)
            (isfinite(Fs[i] * Fs[i + 1]) && Fs[i] * Fs[i + 1] < 0) || continue
            c = _bracket_root(F, minmax(cs[i], cs[i + 1])...)
            abs(F(c)) < 1e-6 * (abs(Fs[i]) + abs(Fs[i + 1])) && push!(found, sgn * acosh(abs(c)))
        end
    end
    return sort!(found; by=abs)
end

"Frequency of the bare host at real wavenumber k (lossless): 𝒟₀(k, ω) = 0, i.e. ω₀(k) of eq. (normalised_quadratic)."
bare_frequency(h::HostSpec, k::Real) = sqrt((h.g + h.T_ρ * k^2 + h.C_ρ * k^4) / (h.m_ρ + added_mass(k, h.H)))

"""
    gap_asymptotic_1d(h, r, a)

Asymptotic width of the complete stop band of a line array, eq. (gap_asymptotic):
(ω_Γ − ω_X)/ω_r ≃ μ(π/a)ω_r²/(ω₀²(π/a) − ω_r²) − Π_L/2, with
μ(k) = n₀m_ρ,r/(m_ρ + M_f,ρ(k, H)) (eq. lat_mu), n₀ = 1/a. Valid for ak_c/π ≲ 0.75.
"""
function gap_asymptotic_1d(h::HostSpec, r::ResonatorSpec, a::Real)
    ωr = natural_frequency(r)
    μ = (r.m_ρ / a) / (h.m_ρ + added_mass(π / a, h.H))
    return μ * ωr^2 / (bare_frequency(h, π / a)^2 - ωr^2) - lattice_compliance_number_1d(h, r, a) / 2
end

"Bending-dominated limit of the gap, eq. (gap_a4): (1/π⁴ − 1/1440) n₀m_ρ,r ω_r² a⁴/C_ρ."
gap_a4_1d(h::HostSpec, r::ResonatorSpec, a::Real) = (1 / π^4 - 1 / 1440) * (r.m_ρ / a) * natural_frequency(r)^2 * a^4 / h.C_ρ

"Lattice-compliance number Π_L = n₀ k_ρ,r σ′(0, ω_r), eq. (lat_PiL), line array."
lattice_compliance_number_1d(h::HostSpec, r::ResonatorSpec, a::Real; N::Int=4000) =
    r.k_ρ * lattice_sum_1d(h, 0.0, natural_frequency(r), a; N=N, exclude_zero=true)

Pi_L_estimate_plate(r::ResonatorSpec, h::HostSpec, a) = r.k_ρ * a^3 / (720h.C_ρ)
Pi_L_estimate_membrane(r::ResonatorSpec, h::HostSpec, a) = r.k_ρ * a / (12h.T_ρ)
"Square lattice: 𝒵_Λ(2)/(16π⁴) k_ρ,r a²/C_ρ, 𝒵_Λ(2) = 4ζ(2)β(2)."
function Pi_L_estimate_square(r::ResonatorSpec, h::HostSpec, a)
    catalan = 0.915965594177219015054603514932384110774
    return 4 * (π^2 / 6) * catalan / (16π^4) * r.k_ρ * a^2 / h.C_ρ
end

# ─── Local (homogenised) branches, eq. (omega_k_x_closed_form) ──────────────

"""
    local_branches(h, r, n0, k)

Lossless local branches ω₋(k) ≤ ω_r ≤ ω₊(k) of the homogenised law with
uniform density n₀ (eqs. graded_quadratic, omega_k_x_closed_form).
"""
function local_branches(h::HostSpec, r::ResonatorSpec, n0::Real, k::Real)
    K = h.g + h.T_ρ * k^2 + h.C_ρ * k^4
    M = h.m_ρ + added_mass(k, h.H)
    ωr2 = natural_frequency(r)^2
    S = K + ωr2 * (M + n0 * r.m_ρ)
    d = sqrt(S^2 - 4ωr2 * K * M)
    return (lower=sqrt((S - d) / (2M)), upper=sqrt((S + d) / (2M)))
end

"Wavenumber k_c where the bare branch crosses ω_r."
function crossing_wavenumber(h::HostSpec, r::ResonatorSpec)
    return k0(h, natural_frequency(r))
end

"Anticrossing separation (ω₊ − ω₋)/ω_r at k_c."
function anticrossing_width(h::HostSpec, r::ResonatorSpec, n0::Real)
    b = local_branches(h, r, n0, crossing_wavenumber(h, r))
    return (b.upper - b.lower) / natural_frequency(r)
end

# ─── Multiple scattering, eq. (lat_foldy) ───────────────────────────────────

"""
    foldy_1d(h, ω, xs, resonators; A=1.0, roots=host_roots(h, ω))

Line resonators at `xs` in the vertical section, incident wave A e^{ik₀x}.
Returns attachment displacements, R, T (eq. lat_RT), the absorbed fraction and
the residual of the energy identity (eq. lat_energy_1d).
"""
function foldy_1d(h::HostSpec, ω::Real, xs::AbstractVector, resonators::AbstractVector{ResonatorSpec};
                  A=1.0, roots=host_roots(h, ω))
    N = length(xs)
    length(resonators) == N || throw(DimensionMismatch("one resonator per position"))
    G0 = green_1d(h, 0.0, ω; roots=roots)
    G = [i == j ? G0 : green_1d(h, xs[i] - xs[j], ω; roots=roots) for i in 1:N, j in 1:N]
    α = alpha.(resonators, ω)
    kp = real(roots[1])
    d = dD0(h, kp, ω)
    η = (I - G * Diagonal(α)) \ (A .* exp.(im * kp .* xs))
    R = im / (d * A) * sum(α .* η .* exp.(im * kp .* xs))
    T = 1 + im / (d * A) * sum(α .* η .* exp.(-im * kp .* xs))
    absorbed = 2 / (d * abs2(A)) * sum(imag.(α) .* abs2.(η))
    return (η=η, R=R, T=T, absorbed=absorbed, energy_residual=abs2(R) + abs2(T) + absorbed - 1)
end

"""
    green_1d_patch(h, x, ω, b; roots=host_roots(h, ω))

Response to a unit force spread uniformly over a patch of width `b` centred at
0 (pressure 1/b), for |x| ≥ b/2:
G_b(x) = i Σₙ sinc(kₙb/2) e^{ikₙ|x|}/𝒟₀′(kₙ), sinc(s) = sin(s)/s.
Reduces to `green_1d` for b → 0.
"""
function green_1d_patch(h::HostSpec, x::Real, ω::Real, b::Real; roots=host_roots(h, ω))
    b == 0 && return green_1d(h, x, ω; roots=roots)
    abs(x) >= b / 2 || throw(DomainError(x, "residue form of the patch response requires |x| ≥ b/2"))
    s(k) = sin(k * b / 2) / (k * b / 2)
    return im * sum(k -> s(k) * exp(im * k * abs(x)) / dD0(h, k, ω), roots)
end

"""
    point_source_field_1d(h, ω, x_s, F, xs, resonators; b=0, roots=host_roots(h, ω))

Vertical-section response to a density-normalised force `F` at `x_s` (spread
over a patch of width `b`) with line resonators at `xs` (eq. lat_coupled):
η(x) = F G_b(x − x_s) + Σⱼ αⱼ ηⱼ G(x − xⱼ), with ηⱼ from the Foldy system
driven by ηⱼ^inc = F G_b(xⱼ − x_s). Returns the attachment displacements and
the field as a function of x (valid for |x − x_s| ≥ b/2).
"""
function point_source_field_1d(h::HostSpec, ω::Real, x_s::Real, F::Number,
                               xs::AbstractVector, resonators::AbstractVector{ResonatorSpec};
                               b::Real=0.0, roots=host_roots(h, ω))
    N = length(xs)
    length(resonators) == N || throw(DimensionMismatch("one resonator per position"))
    G(x) = green_1d(h, x, ω; roots=roots)
    Gs(x) = green_1d_patch(h, x, ω, b; roots=roots)
    α = alpha.(resonators, ω)
    Gm = [G(xs[i] - xs[j]) for i in 1:N, j in 1:N]
    η = N == 0 ? ComplexF64[] : (I - Gm * Diagonal(α)) \ [F * Gs(x - x_s) for x in xs]
    field(x) = F * Gs(x - x_s) + sum((α[j] * η[j] * G(x - xs[j]) for j in 1:N); init=zero(ComplexF64))
    return (η=η, field=field)
end

"""
    foldy_2d(h, ω, xs, resonators; θ_inc=0.0, A=1.0, nθ=720)

Point resonators at `xs` (vector of (x, y)) on a 3D configuration-A plate.
Returns displacements, far-field amplitude f(θ) on `θs` (eq. lat_farfield),
cross-sections and the optical-theorem residual (eq. lat_optical).
"""
function foldy_2d(h::HostSpec, ω::Real, xs::AbstractVector, resonators::AbstractVector{ResonatorSpec};
                  θ_inc::Real=0.0, A=1.0, nθ::Int=720, roots=host_roots(h, ω))
    N = length(xs)
    G0 = green_2d_integral(h, 0.0, ω)
    dist(i, j) = hypot(xs[i][1] - xs[j][1], xs[i][2] - xs[j][2])
    G = [i == j ? G0 : green_2d(h, dist(i, j), ω; roots=roots) for i in 1:N, j in 1:N]
    α = alpha.(resonators, ω)
    kp = real(roots[1])
    d = dD0(h, kp, ω)
    phase(θ, x) = kp * (cos(θ) * x[1] + sin(θ) * x[2])
    η = (I - G * Diagonal(α)) \ [A * exp(im * phase(θ_inc, x)) for x in xs]
    f(θ) = im / 2 * kp / d * sum(α[j] * η[j] * exp(-im * phase(θ, xs[j])) for j in 1:N)
    θs = range(0, 2π; length=nθ + 1)[1:nθ]
    fθ = f.(θs)
    σ_sc = 2 / (π * kp * abs2(A)) * sum(abs2, fθ) * (2π / nθ)
    σ_abs = 2 / (d * abs2(A)) * sum(imag.(α) .* abs2.(η))
    optical = -4 / kp * real(f(θ_inc) / A)
    return (η=η, θs=θs, f=fθ, σ_sc=σ_sc, σ_abs=σ_abs, optical_residual=σ_sc + σ_abs - optical)
end

"cos qa of the single-mode (transfer-matrix) law, eq. (lat_1d_single)."
function single_mode_cos_qa(h::HostSpec, r::ResonatorSpec, ω::Real, a::Real)
    kp = k0(h, ω)
    return cos(kp * a) - alpha(r, ω) / dD0(h, kp, ω) * sin(kp * a)
end

# U_{N−1}(X) = sin(Nqa)/sin(qa), valid for complex X = cos qa.
function _chebyshev_U(n::Int, X)
    n == 0 && return one(X)
    U0, U1 = one(X), 2X
    for _ in 2:n
        U0, U1 = U1, 2X * U1 - U0
    end
    return U1
end

"Single-mode Fabry–Pérot |T_N|², eq. (lat_fabry_perot), identical lossless resonators."
function fabry_perot_transmission(h::HostSpec, r::ResonatorSpec, ω::Real, a::Real, N::Int)
    X = single_mode_cos_qa(h, r, ω, a)
    ratio = alpha(r, ω) / dD0(h, k0(h, ω), ω)
    return 1 / (1 + abs2(ratio * _chebyshev_U(N - 1, X)))
end

end # module References
