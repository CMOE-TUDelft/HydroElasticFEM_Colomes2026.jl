"""
# L2-2 — Woodbury equivalence: monolithic FE and Foldy with the FE transfer compliance (configuration B)

## Relation to the paper

With the bare-structure transfer compliance G_B, the response of the structure
with N resonators to a forcing that gives ηᵢⁱⁿᶜ at the attachments is the
Foldy system with G = G_B (`eq:lat_foldy`, `eq:lat_woodbury`):

    (diag(1/αᵢ) − G_B) f = ηⁱⁿᶜ,   η = ηⁱⁿᶜ + G_B f,

with fᵢ = αᵢηᵢ the resonator forces. Equation `eq:lat_woodbury` is the
Sherman–Morrison–Woodbury identity applied to the monolithic system: adding N
point resonators is a rank-N update of the bare operator. For the discrete
problem it is therefore exact: the monolithic FE solution with resonators and
the Foldy solution built from the FE columns G_{B,h} must agree to round-off.
This is the identity that allows parameter sweeps over resonator populations
(L2-4 … L2-6) with one factorisation of the bare structure per frequency.

Verification test matrix: §L2.2.

## Problem

The configuration-B channel of L2-1 (`L2_1.build`). Resonators at the
`n_regular` regular points of L2-1; forcing: a point load F = 1 at `x_source`
(massless spring, as in L2-1). Three resonator sets:

- `:identical` — m_ρ,r = 0.01 m², ω_r = 10 rad/s, ζ = 0.02;
- `:random` — ω_r ∈ [8, 12], ζ ∈ [0, 0.1], m_ρ,r ∈ [0.005, 0.02], uniform, seed fixed;
- `:graded` — ω_r from 8 to 12 rad/s linearly along the plate, m_ρ,r = 0.01, ζ = 0.02.

## Observables

Monolithic: η at the attachments and resonator displacements q from one solve
with the resonators. Foldy: G_{B,h} among the attachments and the source column
from one factorisation of the bare structure (massless springs), then
f = (diag(1/α) − G_B)⁻¹ g_s, η = g_s + G_B f, q = f/(ω²m). Relative differences
‖η_mono − η_Foldy‖/‖η_mono‖ and the same for q.

## Reproduce

    julia --project -O1 scripts/L2-configB-vertical/L2-2_woodbury.jl
"""
module L2_2

using Parameters
using LinearAlgebra: lu, norm, Diagonal
using Random: MersenneTwister
using ..References: ResonatorSpec, resonator_from_frequency, inv_alpha
import ..L2_1

export L2_2_Params, VARIABLE, fixed_hash, run_case

"Parameters of one L2-2 case; the field names of the channel are those of `L2_1_Params`."
@with_kw struct L2_2_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    ω::Float64 = 10.0
    set::Symbol = :identical              # :identical, :random, :graded
    elements_per_wavelength::Int = 12
    gap_decays::Float64 = 6.0
    order::Int = 3
    T_ρ::Float64 = 0.0

    # ── Fixed: host and channel (as L2-1) ────────────────────────────────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49
    Lp::Float64 = 20.0
    n_regular::Int = 20
    n_random::Int = 0                     # resonators only at the regular points
    seed::Int = 20261023
    k_spring::Float64 = 1.0
    x_source::Float64 = 7.25              # point load (plate coordinate), between attachments
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0
    βₕ::Float64 = 0.5
end

"Fields of `L2_2_Params` varied by the L2-2 study (they name the records)."
const VARIABLE = (:ω, :set, :elements_per_wavelength, :gap_decays, :order, :T_ρ)

fixed_hash(p::L2_2_Params) = L2_1.fixed_hash_fields(p, VARIABLE)

"Resonators of the set (one per regular point)."
function resonator_set(p::L2_2_Params)
    N = p.n_regular
    if p.set == :identical
        return [resonator_from_frequency(0.01, 10.0; ζ=0.02) for _ in 1:N]
    elseif p.set == :random
        rng = MersenneTwister(p.seed + 1)
        return [resonator_from_frequency(0.005 + 0.015rand(rng), 8 + 4rand(rng); ζ=0.1rand(rng)) for _ in 1:N]
    elseif p.set == :graded
        return [resonator_from_frequency(0.01, 8 + 4(i - 1) / (N - 1); ζ=0.02) for i in 1:N]
    end
    throw(ArgumentError("set must be :identical, :random or :graded"))
end

"""
    run_case(p) -> Dict

Monolithic and Foldy responses of resonator set `p.set` to a point load at
`x_source`, and their relative differences.
"""
function run_case(p::L2_2_Params)
    xr = L2_1.points(p)                      # n_random = 0: the regular points
    N = length(xr)
    rs = resonator_set(p)
    spring = ResonatorSpec(m_ρ=0.0, k_ρ=p.k_spring)

    # Foldy with the FE transfer compliance: bare structure, massless springs at the attachments and the source
    gc = L2_1.green_columns(p; xs=[xr; p.x_source])
    Grr = gc.G[1:N, 1:N]
    gs = gc.G[1:N, N + 1]
    ia = inv_alpha.(rs, p.ω)
    f = (Diagonal(ia) - Grr) \ gs
    η_f = gs + Grr * f
    q_f = f ./ (p.ω^2 .* getfield.(rs, :m_ρ))

    # monolithic: resonators at the attachments, massless spring at the source
    sys = L2_1.build(p; xs=[xr; p.x_source], resonators=[rs; spring])
    x = lu(sys.A) \ L2_1.point_load_vector(sys, N + 1, 1.0, p.k_spring)
    η_m = L2_1.deflection(sys, x, xr)
    q_m = L2_1.displacements(sys, x)[1:N]

    return Dict{String,Any}(
        "points" => xr, "eta_mono" => η_m, "eta_foldy" => η_f, "q_mono" => q_m, "q_foldy" => q_f,
        "diff_eta" => norm(η_m - η_f) / norm(η_m), "diff_q" => norm(q_m - q_f) / norm(q_m),
        "reciprocity" => norm(gc.G - transpose(gc.G)) / norm(gc.G),
        "omega_r" => [sqrt(r.k_ρ / r.m_ρ) for r in rs], "ndofs" => size(sys.A, 1), "element_size" => sys.he,
    )
end

end # module L2_2
