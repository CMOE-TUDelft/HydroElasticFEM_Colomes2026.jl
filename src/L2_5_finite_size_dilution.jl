"""
# L2-5 — Finite-size cut-off of the fluid dilution (configuration B, vertical section)

## Relation to the paper

On an infinite floating plate the hybridisation of a resonator population with
the plate is set by the fluid-diluted mass ratio μ(k) = n₀m_ρ,r/(m_ρ + M_f,ρ(k, H)),
and M_f,ρ → ∞ as k → 0 (`subsubsec:fluid_dilution`, `eq:anticrossing`). A
structure of length L supports no wavenumber below ~π/L, so the divergence is
cut off and, to leading order (`subsubsec:lat_long_wave`, red, pending FE),

    μ_B ~ n₀m_ρ,r π tanh(πH/L)/L,   splitting ~ √μ_B,

i.e. the attainable hybridisation decreases with the size of the structure, as
L^{−1/2} in deep water (L ≪ πH) and as L^{−1} in shallow water (L ≫ πH). The
same mechanism underlies the anisotropic estimate `eq:mode_dilution` (L4-8).

Verification test matrix: §L2.5.

## Problem

The configuration-B channel of L2-1 with a P1 plate (T_ρ = 0) of length L and
free edges. The first elastic (two-node) wet mode of the bare plate is computed
by the fixed-point iteration of eq:eigWet (`L2_3`); its frequency ω₁ is the
tuning of a uniform population of lossless resonators at x = (i − ½)a, a = L/N,
N = round(L/`spacing`), with fixed loading n₀m_ρ,r = `loading`. The two
hybridised wet modes ω₋ < ω₁ < ω₊ are then computed by the same iteration,
started from the two modes of largest projection on the bare mode.

## Observables

- Splitting Δ = (ω₊ − ω₋)/ω₁ (FE, eq:eigWet).
- Two-degree-of-freedom prediction from the FE: √μ_mode, μ_mode =
  m_ρ,r Σᵢ φ(xᵢ)²/φᵀ(M_η + Re A(ω₁))φ, with φ the bare wet mode (its
  generalised mass includes the fluid added mass).
- Single-mode Galerkin prediction (bare wet mode φ, frequency-dependent
  modal added mass): roots ω of
      φᵀKφ − ω² m_φ(ω) − ω² m_eff ω₁²/(ω₁² − ω²) = 0,
  m_φ(ω) = φᵀ(M_η + Re A(ω))φ, m_eff = m_ρ,r Σᵢ φ(xᵢ)²; it differs from √μ_mode
  (constant m_φ(ω₁)) by the dispersion of the added mass over ω₋ … ω₊, and
  from the FE by the change of the mode shape.
- Estimates: √μ_B of the paper, and √μ(k₁) with the exact local added mass at
  the modal wavenumber k₁ = β₁/L, β₁ = 4.730 (free–free first elastic mode).

## Reproduce

    julia --project -O1 scripts/L2-configB-vertical/L2-5_finite_size_dilution.jl
"""
module L2_5

using Gridap
using Parameters
using LinearAlgebra: eigen, Symmetric, dot, norm
using ..References: ResonatorSpec, added_mass
import ..L2_1
import ..L2_3
import ..L1_3: secant_root

export L2_5_Params, VARIABLE, fixed_hash, run_case

"Parameters of one L2-5 case; channel field names as `L2_1_Params`."
@with_kw struct L2_5_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    Lp::Float64 = 23.0                    # plate length (m)
    H::Float64 = 10.0                     # depth (m)
    elements_per_wavelength::Int = 16
    order::Int = 3
    gap_decays::Float64 = 6.0
    T_ρ::Float64 = 0.0

    # ── Fixed: host (P1) and population ──────────────────────────────────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49
    spacing::Float64 = 1.0                # target resonator spacing a (m)
    loading::Float64 = 0.01               # n₀m_ρ,r (m)

    # ── Fixed: discretisation and solvers ────────────────────────────────────
    k_spring::Float64 = 1.0
    n_regular::Int = 0
    n_random::Int = 0
    seed::Int = 0
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0
    βₕ::Float64 = 0.5
    tol::Float64 = 1e-10
    n_shape::Int = 81                     # samples along the plate to classify mode shapes
end

"Fields of `L2_5_Params` varied by the L2-5 study (they name the records)."
const VARIABLE = (:Lp, :H, :elements_per_wavelength, :order, :gap_decays, :T_ρ)

fixed_hash(p::L2_5_Params) = L2_1.fixed_hash_fields(p, VARIABLE)

const β1 = 4.730040744862704              # first elastic free–free beam constant

"The parameter set seen by `L2_1`/`L2_3` at mesh frequency ω (the mesh is sized at ω₁ estimate)."
withmesh(p::L2_5_Params, ωm) = merge(NamedTuple{fieldnames(L2_5_Params)}(Tuple(getfield(p, f) for f in fieldnames(L2_5_Params))),
                                     (ω_mesh=ωm, ω=ωm, Lp=p.Lp))

"Local estimate of the first elastic wet frequency (start of the iteration and mesh frequency)."
function omega_estimate(p)
    k = β1 / p.Lp
    return sqrt((p.g + p.T_ρ * k^2 + p.C_ρ * k^4) / (p.m_ρ + added_mass(k, p.H)))
end

"Plate deflection of the η part of a structural vector at `n_shape` points along the plate."
function shape(q, W, v)
    Vη = W.sys.X.spaces[W.sys.fmap[:η]]
    ηh = FEFunction(Vη, ComplexF64.(v[1:W.nη]))
    xs = range(0, q.Lp; length=q.n_shape)
    return real.([ηh(Point(W.sys.x0 + x, 0.0)) for x in xs])
end

"Number of sign changes of a sampled shape (ignoring values below 1e-6 of the maximum)."
function sign_changes(s)
    t = filter(x -> abs(x) > 1e-6 * maximum(abs, s), s)
    return count(i -> sign(t[i]) != sign(t[i + 1]), 1:(length(t) - 1))
end

"Frozen-ω generalised eigenproblem of eq:eigWet (Re A) for attachment points `xs`."
function frozen(q, ω; xs=Float64[], rs=ResonatorSpec[])
    W = L2_3.wet_matrices(q, ω; xs=xs, resonators=rs)
    W.asym < 1e-10 || error("wet matrices not symmetric ($(W.asym))")
    e = eigen(Symmetric(real(W.K)), Symmetric(real(W.M + real(W.Aadd))))
    return (W=W, λ=e.values, V=e.vectors)
end

mac(u, v) = abs2(dot(u, v)) / (dot(u, u) * dot(v, v))

"""
    bare_mode(p, q)

First elastic wet mode of the bare plate: fixed point of eq:eigWet, at each
step the lowest mode with two sign changes (heave and pitch have zero and one).
"""
function bare_mode(q; maxiter=40)
    ω = q.ω_mesh
    v = nothing
    local F, j
    for _ in 1:maxiter
        F = frozen(q, ω)
        if v === nothing
            j = findfirst(i -> F.λ[i] > 0 && sign_changes(shape(q, F.W, F.V[:, i])) == 2, eachindex(F.λ))
        else
            j = argmax([mac(F.V[:, i], v) for i in eachindex(F.λ)])
        end
        v = F.V[:, j]
        ωn = sqrt(F.λ[j])
        conv = abs(ωn - ω) <= q.tol * ω
        ω = ωn
        conv && break
    end
    return (omega=ω, v=v, W=F.W)
end

"""
    hybrid_modes(q, ω1, φ, xs, rs)

The two hybridised wet modes: at ω₁ the two eigenvectors of largest projection
on the bare mode φ (one below, one above ω₁), then the fixed point of each with
MAC tracking.
"""
function hybrid_modes(q, ω1, φ, xs, rs; maxiter=40)
    F = frozen(q, ω1; xs=xs, rs=rs)
    nη = F.W.nη
    Mη = real(F.W.M + real(F.W.Aadd))[1:nη, 1:nη]
    proj = [abs2(dot(φ, Mη * F.V[1:nη, i])) / (dot(φ, Mη * φ) * dot(F.V[1:nη, i], Mη * F.V[1:nη, i]) + eps()) for i in eachindex(F.λ)]
    below = [i for i in eachindex(F.λ) if 0 < F.λ[i] < ω1^2]
    above = [i for i in eachindex(F.λ) if F.λ[i] > ω1^2]
    starts = (below[argmax(proj[below])], above[argmax(proj[above])])
    out = Float64[]
    for j0 in starts
        v, ω = F.V[:, j0], sqrt(F.λ[j0])
        for _ in 1:maxiter
            G = frozen(q, ω; xs=xs, rs=rs)
            j = argmax([mac(G.V[:, i], v) for i in eachindex(G.λ)])
            v = G.V[:, j]
            ωn = sqrt(G.λ[j])
            conv = abs(ωn - ω) <= q.tol * ω
            ω = ωn
            conv && break
        end
        push!(out, ω)
    end
    return (minus=out[1], plus=out[2], proj_minus=proj[starts[1]], proj_plus=proj[starts[2]])
end

"""
    run_case(p) -> Dict

Bare first elastic wet mode, tuned population, hybridised pair, splitting and predictions.
"""
function run_case(p::L2_5_Params)
    q = withmesh(p, omega_estimate(p))
    bm = bare_mode(q)
    ω1 = bm.omega
    N = max(1, round(Int, p.Lp / p.spacing))
    a = p.Lp / N
    xs = [(i - 0.5) * a for i in 1:N]
    mr = p.loading * a
    rs = fill(ResonatorSpec(m_ρ=mr, k_ρ=mr * ω1^2), N)
    hm = hybrid_modes(q, ω1, bm.v[1:bm.W.nη], xs, rs)

    # two-degree-of-freedom prediction from the bare wet mode
    nη = bm.W.nη
    φ = bm.v[1:nη]
    Mφ = dot(φ, real(bm.W.M + real(bm.W.Aadd))[1:nη, 1:nη] * φ)
    Vη = bm.W.sys.X.spaces[bm.W.sys.fmap[:η]]
    ηh = FEFunction(Vη, ComplexF64.(φ))
    φx = real.([ηh(Point(bm.W.sys.x0 + x, 0.0)) for x in xs])
    μ_mode = mr * sum(abs2, φx) / Mφ
    Kφ = dot(φ, real(bm.W.K)[1:nη, 1:nη] * φ)
    meff = mr * sum(abs2, φx)
    mφ(ω) = (W = L2_3.wet_matrices(q, ω; xs=Float64[], resonators=ResonatorSpec[]); dot(φ, real(W.M + real(W.Aadd))[1:nη, 1:nη] * φ))
    galerkin(ω) = Kφ - ω^2 * mφ(ω) - ω^2 * meff * ω1^2 / (ω1^2 - ω^2)
    gm = secant_root(galerkin, hm.minus, hm.minus * (1 + 1e-3); xtol=1e-9)
    gp = secant_root(galerkin, hm.plus, hm.plus * (1 - 1e-3); xtol=1e-9)
    k1 = β1 / p.Lp
    μ_local = p.loading / (p.m_ρ + added_mass(k1, p.H))
    μ_B = p.loading * π * tanh(π * p.H / p.Lp) / p.Lp

    return Dict{String,Any}(
        "omega_1" => ω1, "omega_minus" => hm.minus, "omega_plus" => hm.plus,
        "split" => (hm.plus - hm.minus) / ω1, "sqrt_mu_mode" => sqrt(μ_mode),
        "split_galerkin" => (gp.root - gm.root) / ω1, "galerkin_converged" => gm.converged && gp.converged,
        "modal_mass_ratio" => (mφ(hm.minus) - mφ(hm.plus)) / Mφ, "sqrt_mu_local" => sqrt(μ_local),
        "sqrt_mu_B" => sqrt(μ_B), "N" => N, "a" => a, "proj_minus" => hm.proj_minus, "proj_plus" => hm.proj_plus,
        "omega_estimate" => q.ω_mesh, "ndofs" => bm.W.ndofs, "element_size" => bm.W.he,
    )
end

end # module L2_5
