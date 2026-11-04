"""
# L2-4 — Absorption bound of a single resonator on a finite structure (configuration B)

## Relation to the paper

On a finite structure in open water the radiation of a resonator at x₁ is not
symmetric. With a^± the open-water amplitudes radiated towards x → ±∞ by a
unit point force at x₁, the radiated power is ½ω Im G_B(x₁, x₁) =
Σ± ½g c_g|a^±|² (per unit density), and by reciprocity the deflection at x₁ due
to a wave incident from x → −∞ is proportional to a⁻ (a Haskind-type relation).
The absorbed fraction of a single resonator is then bounded by

    𝒜 ≤ |a⁻|²/(|a⁻|² + |a⁺|²)                                            (eq:lat_absorbed_B)

with equality at critical coupling γ_d = γ_r = Im G_B(x₁, x₁) at the renormalised
resonance. It reduces to ½ for a symmetric radiator (resonator at the centre of
a symmetric structure) and can approach one near an edge, which is the basis of
the claim that a resonator on a finite structure can absorb more than half of
the incident energy (introduction, PROVISIONAL clause).

Verification test matrix: §L2.4.

## Problem

The configuration-B channel of L2-1 (`L2_1.build`), P1 plate of length `Lp`
with free edges, one attachment point x₁ (plate coordinate). Incidence from the
left only; incidence from the right is the mirror case x₁ → Lp − x₁.

## Observables (at the renormalised resonance ω̃ of the lossless resonator)

1. Radiation: massless spring at x₁ loaded by a unit force; G₁₁ = G_B(x₁, x₁);
   the free surface in each open-water region is fitted (`L1_1.fit_waves`) by
   the propagating waves and the slowest evanescent modes from both window
   ends, giving the outgoing a⁻ (left) and a⁺ (right). Check: ½ω Im G₁₁ against
   ½g c_g(|a⁻|² + |a⁺|²).
2. ω̃: root of Re(1/α₀ − G₁₁) for the lossless resonator (secant, one FE
   factorisation per iterate, one mesh sized at ω_r).
3. Incidence: the bare plate (unloaded spring at x₁) under a wave from the left;
   the incident amplitude A is fitted in the left open water, η_inc = η(x₁).
4. Absorption for damping ratios ζ (Woodbury, exact for the discrete problem,
   L2-2): η₁ = η_inc/(1 − α G₁₁), 𝒜 = ½ω Im α |η₁|²/(½g c_g|A|²); the maximum
   over ζ against the bound, and the critical damping γ_d = Im G₁₁.
5. One monolithic solve with the resonator at critical coupling under the same
   incident wave: 𝒜 from the dashpot power, ½ω²c|q − η₁|², against item 4.

## Reproduce

    julia --project -O1 scripts/L2-configB-vertical/L2-4_asymmetric_absorption.jl
"""
module L2_4

using Parameters
using LinearAlgebra: lu
using Roots: find_zero, Brent
using ..References: ResonatorSpec, resonator_from_frequency, inv_alpha, alpha
import ..L1_1
import ..L2_1
import ..L1_3: secant_root

export L2_4_Params, VARIABLE, fixed_hash, run_case

"Parameters of one L2-4 case; channel field names as `L2_1_Params`."
@with_kw struct L2_4_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    Lp::Float64 = 20.0                    # plate length (m)
    x1::Float64 = 10.0                    # attachment, plate coordinate
    elements_per_wavelength::Int = 8
    order::Int = 3
    gap_decays::Float64 = 6.0
    T_ρ::Float64 = 0.0

    # ── Fixed: host (P1) and resonator ───────────────────────────────────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49
    m_ρr::Float64 = 0.01
    ω_r::Float64 = 10.0
    k_spring::Float64 = 1.0
    amplitude::Float64 = 0.01             # incident wave amplitude (m); linear problem
    ζs::Tuple{Vararg{Float64}} = Tuple(10 .^ range(-3, -1; length=41))   # damping sweep

    # ── Fixed: discretisation and post-processing ────────────────────────────
    n_regular::Int = 0
    n_random::Int = 0
    seed::Int = 0
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0
    βₕ::Float64 = 0.5
    n_modes::Int = 3                      # evanescent open-water modes in the fits
    n_samples::Int = 400
    window_decays::Float64 = 1.0          # fit windows start this many 1/κ₁ from walls and plate edges
    xtol::Float64 = 1e-10
end

"Fields of `L2_4_Params` varied by the L2-4 study (they name the records)."
const VARIABLE = (:Lp, :x1, :elements_per_wavelength, :order, :gap_decays, :T_ρ)

fixed_hash(p::L2_4_Params) = L2_1.fixed_hash_fields(p, VARIABLE)

"""
    open_water_waves(p, sys, x, ω)

Fit of the free surface in the left and right open water of solution `x`:
amplitudes referred to the plate edges, (A_in, A_out) on each side
(left: in = right-going; right: in = left-going), and the fit residual.
"""
function open_water_waves(p, sys, x::AbstractVector, ω::Real)
    roots = L2_1.open_roots(ω, p.H, p.g; n=p.n_modes)
    d = p.window_decays / imag(roots[2])
    xl, xr = sys.x0, sys.x0 + p.Lp                           # plate edges (absolute)
    fit(a, b, xref) = (xs = range(a, b; length=p.n_samples);
                       L1_1.fit_waves(xs, L2_1.surface(sys, x, xs), roots; x_ref=xref, n_modes=p.n_modes,
                                      k=real(roots[1])))
    left = fit(d, xl - d, xl)
    right = fit(xr + d, sys.L - d, xr)
    return (left_in=left.Aplus, left_out=left.Aminus, right_in=right.Aminus, right_out=right.Aplus,
            residual=max(left.residual, right.residual))
end

"G₁₁ and the solution for a unit point force at x₁ (bare plate, massless spring)."
function radiation(p, ω)
    sys = L2_1.build(p; ω=ω, ω_mesh=p.ω_r, xs=[p.x1], resonators=[ResonatorSpec(m_ρ=0.0, k_ρ=p.k_spring)])
    x = lu(sys.A) \ L2_1.point_load_vector(sys, 1, 1.0, p.k_spring)
    return (G11=only(L2_1.deflection(sys, x, [p.x1])), sys=sys, x=x)
end

"""
    run_case(p) -> Dict

Radiation amplitudes, renormalised resonance, incident drive, absorption over
the damping sweep (Woodbury), the bound, and the monolithic check at critical coupling.
"""
function run_case(p::L2_4_Params)
    r0 = resonator_from_frequency(p.m_ρr, p.ω_r)
    cache = Dict{Float64,ComplexF64}()
    G11(ω) = get!(() -> radiation(p, ω).G11, cache, ω)
    s = secant_root(ω -> real(inv_alpha(r0, ω) - G11(ω)), p.ω_r, p.ω_r * (1 - 1e-3); xtol=p.xtol)
    ω̃ = s.root

    # radiation at ω̃: amplitudes and power identity
    rad = radiation(p, ω̃)
    w = open_water_waves(p, rad.sys, rad.x, ω̃)
    cg = L2_1.c_group(ω̃, p.H, p.g)
    P_G = ω̃ / 2 * imag(rad.G11)
    P_flux = p.g * cg / 2 * (abs2(w.left_out) + abs2(w.right_out))
    bound = abs2(w.left_out) / (abs2(w.left_out) + abs2(w.right_out))

    # incidence from the left on the bare plate
    spring = ResonatorSpec(m_ρ=0.0, k_ρ=p.k_spring)
    sysi = L2_1.build(p; ω=ω̃, ω_mesh=p.ω_r, xs=[p.x1], resonators=[spring], incident=p.amplitude)
    xi = lu(sysi.A) \ sysi.b
    η_inc = only(L2_1.deflection(sysi, xi, [p.x1]))
    wi = open_water_waves(p, sysi, xi, ω̃)
    A = wi.left_in
    flux_in = p.g * cg / 2 * abs2(A)

    # absorption over ζ (Woodbury) and at critical coupling γ_d = γ_r
    absorbed(r) = (a = alpha(r, ω̃); η1 = η_inc / (1 - a * rad.G11); ω̃ / 2 * imag(a) * abs2(η1) / flux_in)
    𝒜 = [absorbed(resonator_from_frequency(p.m_ρr, p.ω_r; ζ=ζ)) for ζ in p.ζs]
    γr = imag(rad.G11)
    γd(ζ) = (r = resonator_from_frequency(p.m_ρr, p.ω_r; ζ=ζ); imag(alpha(r, ω̃)) / abs2(alpha(r, ω̃)))
    ζc = find_zero(ζ -> γd(ζ) - γr, (1e-6, 0.5), Brent())
    rc = resonator_from_frequency(p.m_ρr, p.ω_r; ζ=ζc)
    𝒜_crit = absorbed(rc)

    # monolithic check at critical coupling
    sysm = L2_1.build(p; ω=ω̃, ω_mesh=p.ω_r, xs=[p.x1], resonators=[rc], incident=p.amplitude)
    xm = lu(sysm.A) \ sysm.b
    η1 = only(L2_1.deflection(sysm, xm, [p.x1]))
    q1 = L2_1.displacements(sysm, xm)[1]
    Am = open_water_waves(p, sysm, xm, ω̃).left_in
    𝒜_mono = ω̃^2 / 2 * rc.c_ρ * abs2(q1 - η1) / (p.g * cg / 2 * abs2(Am))

    return Dict{String,Any}(
        "omega_tilde" => ω̃, "secant_converged" => s.converged, "G11" => rad.G11,
        "a_left" => w.left_out, "a_right" => w.right_out, "radiation_in" => abs(w.left_in) + abs(w.right_in),
        "power_G" => P_G, "power_flux" => P_flux, "bound" => bound,
        "eta_inc" => η_inc, "A_inc" => A, "zetas" => collect(p.ζs), "absorbed" => 𝒜,
        "zeta_crit" => ζc, "absorbed_crit" => 𝒜_crit, "absorbed_mono" => 𝒜_mono,
        "fit_residual" => max(w.residual, wi.residual), "ndofs" => size(rad.sys.A, 1),
        "element_size" => rad.sys.he, "gap" => rad.sys.gap,
    )
end

end # module L2_4
