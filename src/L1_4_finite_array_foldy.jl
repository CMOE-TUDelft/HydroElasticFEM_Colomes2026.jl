"""
# L1-4 — Finite periodic array: Foldy, energy balance (configuration A, vertical section)

## Relation to the paper

N identical line resonators at x_n on the bare host scatter an incident wave
through the Foldy system (eq:lat_foldy)

    ηᵢ = ηᵢ^inc + Σⱼ G(xᵢ − xⱼ) αⱼ ηⱼ,

with R, T from the forces αⱼηⱼ (eq:lat_RT) and |R|² + |T|² + 𝒜 = 1, 𝒜 the
power dissipated in the dashpots over the incident flux (eq:lat_energy_1d).
Inside the stop band of the lattice (L1-5) |T_N| decays exponentially with N
at the rate Im q of the complex Bloch wavenumber at real ω (eq:lat_decay).
`References.foldy_1d` evaluates the Foldy solution with the residue Green
function; this test compares it with the monolithic FE.

Verification test matrix: §L1.4.

## Problem

The L1-1 channel (`L1_1.solve`) with N resonators, spacing a, centred at x_c:

    wall ── sponge ── buffer ── │ source · probe ↑ · x₁ … x_N · probe ↓ │ ── buffer ── sponge ── wall

The element size is a/`elements_per_cell`, so every resonator sits on a node
and the mesh is symmetric about x_c. Buffers, sponges and channel length follow
the L1-1 rules, with λ_p at `ω_ref`.

## Observables

1. At each frequency, the probe samples minus the exact near fields of the
   source and of every resonator (force fₙ = ω²m qₙ from the FE) are fitted by
   two propagating waves (discrete wavenumber, δ measured once on the bare
   plate at `ω_ref`, as L1-3). With a_L, a_R the incoming and b_L, b_R the
   outgoing amplitudes, referenced at x_c, the symmetric array is a two-port,

       b_L = R a_L + T a_R + c_L,   b_R = T a_L + R a_R + c_R,

   exact under any reflection from the ends. c_L, c_R are the waves radiated
   by the array driven by the source near field e(xₙ) (O(2e-3)), from the
   Foldy response to e; their error is second order.
2. 𝒜 = 1 − |R|² − |T|², and independently the power balance of the FE
   solution: net incoming flux ¼ω𝒟₀′(|a_L|² + |a_R|² − |b_L|² − |b_R|²)
   against the dashpot power Σ ½ω²c|qₙ − ηₙ|².
3. References: `foldy_1d` (R, T, 𝒜) with the same resonators.

## Reproduce

    julia --project -O1 scripts/L1-configA-vertical/L1-4_finite_array_foldy.jl
"""
module L1_4

using Parameters
using LinearAlgebra: I, Diagonal
using ..References: HostSpec, ResonatorSpec, k0, kappa1, host_roots, dD0, resonator_from_frequency,
                    foldy_1d, green_1d, alpha
import ..L1_1
import ..L1_3

export L1_4_Params, VARIABLE, fixed_hash, run_case

"""
Parameters of one L1-4 case. Lengths of the layout are in plate wavelengths
λ_p at `ω_ref`. The host, forcing, sponge and discretisation field names are
those of `L1_1_Params`, so that `L1_1.solve` takes this struct.
"""
@with_kw struct L1_4_Params
    # ── Variable: what the study sweeps (names the record) ───────────────────
    T_ρ::Float64 = 0.0
    N::Int = 10                           # number of resonators
    ζ::Float64 = 0.0                      # damping ratio of every resonator
    elements_per_cell::Int = 8            # horizontal elements per spacing a
    order::Int = 3
    sweep::Symbol = :curve                # :curve (n_wide + n_fine frequencies) or :check (`check_omegas`)

    # ── Fixed: host, set P1 of the test matrix = Table 1 of the paper ────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49

    # ── Fixed: array ──────────────────────────────────────────────────────────
    a::Float64 = 1.0
    m_ρr::Float64 = 0.01
    ω_r::Float64 = 10.0

    # ── Fixed: frequencies ────────────────────────────────────────────────────
    ω_ref::Float64 = 10.0                 # sizes the layout and δ
    wide::Tuple{Float64,Float64} = (8.0, 12.0)
    n_wide::Int = 81
    fine::Tuple{Float64,Float64} = (9.55, 10.05)   # around the stop band [ω_X, ω_Γ]
    n_fine::Int = 101
    check_omegas::Tuple{Vararg{Float64}} = (8.5, 9.5, 9.8, 10.02, 11.0)

    # ── Fixed: forcing ────────────────────────────────────────────────────────
    F::Float64 = 1.0
    source_elements::Int = 2

    # ── Fixed: layout (as L1-3) ──────────────────────────────────────────────
    source_offset::Float64 = 0.25
    source_gap::Float64 = 0.5
    probe::Float64 = 3.0
    offset::Float64 = 1.0
    end_margin::Float64 = 0.25
    gap_decays::Float64 = 5.0
    length_fraction::Float64 = 0.25
    sponge_wavelengths::Float64 = 6.0
    sponge_strength::Float64 = 3.0
    sponge_power::Int = 2

    # ── Fixed: discretisation (as L1-1) ──────────────────────────────────────
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0

    # ── Fixed: post-processing ────────────────────────────────────────────────
    n_modes::Int = 6
    n_samples::Int = 400
end

"Fields of `L1_4_Params` varied by the L1-4 study (they name the records)."
const VARIABLE = (:T_ρ, :N, :ζ, :elements_per_cell, :order, :sweep)

"Short hash of the fixed parameters, appended to the record name."
function fixed_hash(p::L1_4_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L1_4_Params) if !(f in VARIABLE))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

"Elements per λ_p at `ω_ref` implied by `elements_per_cell` (used by the vertical-mesh rule of L1-1)."
elements_per_wavelength(p) = p.elements_per_cell * 2π / (k0(L1_1.host(p), p.ω_ref) * p.a)

"Resonator of the case."
resonator(p) = resonator_from_frequency(p.m_ρr, p.ω_r; ζ=p.ζ)

"""
    layout(p)

Absolute geometry: the fields of `L1_1.layout` (channel, sponge, buffer,
window, source, element size) plus the resonator positions `xs`, the centre
`x_c` and the probe windows `up`, `down`. Every resonator, the source and x_c
lie on mesh nodes.
"""
function layout(p::L1_4_Params)
    h = L1_1.host(p)
    λ = 2π / k0(h, p.ω_ref)
    he = p.a / p.elements_per_cell
    snap(x) = round(x / he) * he
    Ls = p.sponge_wavelengths * λ
    x1_w = (p.source_offset + p.source_gap + p.probe + p.offset) * λ        # first resonator, from the window start
    span = (p.N - 1) * p.a
    W = x1_w + span + (p.offset + p.probe + p.end_margin) * λ
    d = min(p.source_offset * λ, (p.offset + p.probe + p.end_margin) * λ)
    buffer = max(λ, p.gap_decays / kappa1(h, p.ω_ref) - d)
    L = λ * (ceil((2 * (Ls + buffer) + W) / λ - p.length_fraction) + p.length_fraction)
    nx = round(Int, L / he)
    L = nx * he
    buffer = (L - W) / 2 - Ls
    xw = Ls + buffer
    x_s = snap(xw + p.source_offset * λ)
    x1 = snap(xw + x1_w)
    xs = [x1 + (n - 1) * p.a for n in 1:p.N]
    x_N = xs[end]
    return (; λ, L, Ls, buffer, window=(xw, xw + W), x_s, b=p.source_elements * he, nx, he, xs,
            x_c=(x1 + x_N) / 2, up=(x1 - (p.offset + p.probe) * λ, x1 - p.offset * λ),
            down=(x_N + p.offset * λ, x_N + (p.offset + p.probe) * λ))
end

"`L1_1.solve` reads `elements_per_wavelength` for the vertical mesh; this view supplies it."
solver_params(p::L1_4_Params) = merge(NamedTuple{fieldnames(L1_4_Params)}(Tuple(getfield(p, f) for f in fieldnames(L1_4_Params))),
                                      (elements_per_wavelength=elements_per_wavelength(p),))

"""
    dispersion_error(p, lay)

δ = k_h/k₀ − 1 at `ω_ref` on the mesh of the case (bare plate, one-way field
past the source, as `L1_3.dispersion_error`).
"""
function dispersion_error(p::L1_4_Params, lay)
    sp = solver_params(p)
    sol = L1_1.solve(sp; ω=p.ω_ref, lay=lay)
    roots = host_roots(L1_1.host(p), p.ω_ref)
    xs = range(lay.x_s + lay.λ, lay.window[2]; length=p.n_samples)
    fit = L1_1.fit_waves(xs, L1_1.sample(sol.ηh, xs), roots; x_ref=lay.x_s, n_modes=p.n_modes)
    return fit.k / real(roots[1]) - 1
end

"""
    two_port(aL, aR, bL, bR)

R and T of a symmetric two-port from incoming (aL from the left, aR from the
right) and outgoing (bL to the left, bR to the right) amplitudes:
bL = R aL + T aR, bR = T aL + R aR.
"""
function two_port(aL, aR, bL, bR)
    M = [aL aR; aR aL]
    R, T = M \ [bL, bR]
    return (R=R, T=T)
end

"""
    scattering(p, lay, r, ω, δ)

FE scattering by the array at frequency ω: R, T (phase reference at x_c), 𝒜,
the power-balance residual of the FE solution and the fit residual.
"""
function scattering(p::L1_4_Params, lay, r::ResonatorSpec, ω::Real, δ::Real)
    sp = solver_params(p)
    sol = L1_1.solve(sp; ω=ω, lay=lay, resonators=fill(r, p.N), x_res=lay.xs)
    h = L1_1.host(p)
    roots = host_roots(h, ω)
    kp = real(roots[1])
    kh = kp * (1 + δ)
    η = L1_1.sample(sol.ηh, lay.xs)
    f = ω^2 * r.m_ρ .* sol.q                                  # resonator forces on the plate
    src_near(x) = p.F * L1_3.near_field(h, roots, x - lay.x_s, ω; b=lay.b)
    near(x) = src_near(x) + sum(f[n] * L1_3.near_field(h, roots, x - lay.xs[n], ω) for n in 1:p.N)
    fit(w) = (xs = range(w...; length=p.n_samples);
              L1_1.fit_waves(xs, L1_1.sample(sol.ηh, xs) .- near.(xs), roots; x_ref=lay.x_c, n_modes=0, k=kh))
    up, down = fit(lay.up), fit(lay.down)
    aL, bL, bR, aR = up.Aplus, up.Aminus, down.Aplus, down.Aminus

    # waves radiated by the array driven by the source near field, from the Foldy response
    d = dD0(h, kp, ω)
    α = alpha(r, ω)
    xr = lay.xs .- lay.x_c
    G = [green_1d(h, xr[i] - xr[j], ω; roots=roots) for i in 1:p.N, j in 1:p.N]
    e = src_near.(lay.xs)
    ηe = (I - α * G) \ e
    cL = im / d * sum(α .* ηe .* exp.(im * kp .* xr))
    cR = im / d * sum(α .* ηe .* exp.(-im * kp .* xr))
    s = two_port(aL, aR, bL - cL, bR - cR)

    # power balance of the FE solution (density-normalised, per unit width)
    flux_in = ω / 4 * d * (abs2(aL) + abs2(aR) - abs2(bL) - abs2(bR))
    dashpot = ω^2 / 2 * r.c_ρ * sum(abs2, sol.q .- η)
    return (R=s.R, T=s.T, A=1 - abs2(s.R) - abs2(s.T), power_residual=abs(flux_in - dashpot) / (ω / 4 * d * (abs2(aL) + abs2(aR))),
            fit_residual=max(up.residual, down.residual), ndofs=sol.ndofs, ny=sol.vm.ny)
end

"Frequencies of the case."
omegas(p::L1_4_Params) = p.sweep == :check ? collect(p.check_omegas) :
                         sort!(unique!([collect(range(p.wide...; length=p.n_wide)); collect(range(p.fine...; length=p.n_fine))]))

"""
    run_case(p) -> Dict

FE R, T, 𝒜 and power-balance residual at every frequency of the case, with the
Foldy references for the same resonators.
"""
function run_case(p::L1_4_Params)
    h = L1_1.host(p)
    r = resonator(p)
    lay = layout(p)
    δ = dispersion_error(p, lay)
    ωs = omegas(p)
    fe = [scattering(p, lay, r, ω, δ) for ω in ωs]
    xr = lay.xs .- lay.x_c
    ref = [foldy_1d(h, ω, xr, fill(r, p.N)) for ω in ωs]
    return Dict{String,Any}(
        "omega" => ωs,
        "R_h" => getfield.(fe, :R), "T_h" => getfield.(fe, :T), "A_h" => getfield.(fe, :A),
        "R_ref" => getfield.(ref, :R), "T_ref" => getfield.(ref, :T), "A_ref" => getfield.(ref, :absorbed),
        "energy_residual_ref" => maximum(abs, getfield.(ref, :energy_residual)),
        "power_residual_h" => getfield.(fe, :power_residual), "fit_residual" => maximum(getfield.(fe, :fit_residual)),
        "dispersion_error" => δ, "ndofs" => fe[1].ndofs, "ny" => fe[1].ny, "lambda_p" => lay.λ,
        "element_size" => lay.he, "L" => lay.L, "x_c" => lay.x_c, "fe_solves" => length(ωs) + 1,
    )
end

end # module L1_4
