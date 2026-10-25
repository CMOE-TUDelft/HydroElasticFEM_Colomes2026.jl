"""
# L2-3 — Renormalised wet resonance of a resonator on a finite floating membrane (configuration B)

## Relation to the paper

A resonator attached at x₁ to a finite structure in open water resonates where
the detuning vanishes,

    Re(1/α(ω̃) − G_B(x₁, x₁; ω̃)) = 0,                                  (eq:renorm_single with G_B)

with radiative width γ_r = Im G_B(x₁, x₁) (`sec:finite_domain`): the resonance
depends on the attachment position through the point compliance of the wet
structure. Section `sec:modalAnalysis` defines the wet natural frequencies
instead by the modal problem (eq:eigWet)

    (K_wet − ω²(M_dry + Re A(ω))) [η; q] = 0,

solved by the fixed-point iteration of `fig:modal_iteration`, with A(ω) the
hydrodynamic added-mass matrix. This test computes the resonator-dominated wet
mode of the membrane of the existing results (set P2) both ways from the same
FE discretisation, and the peak of the forced response.

Verification test matrix: §L2.3.

## Problem

The configuration-B channel of L2-1 (`L2_1.build`) with the P2 membrane
(L_m = 50 m, C_ρ = 0, T_ρ = 9.7561 m³/s², m_ρ = 0.0444 m, H = 25 m, ρ = 1025 kg/m³;
open water of `gap_decays` evanescent decay lengths on each side, as L2-1) and one resonator, m_ρ,r = 113.75/1025 m²,
ω_r = 0.5 rad/s, lossless, at x₁ (plate coordinate).

## Observables

1. Modal: the assembled matrix at ω is split into structural (η, q) and fluid
   (ϕ, κ) degrees of freedom. The structural block equals K_wet − ω²M_dry
   (recovered from two frequencies); the fluid Schur complement
   −A_sf A_ff⁻¹ A_fs equals −ω²A(ω). The fixed-point iteration solves the
   generalised eigenproblem with Re A(ω_k) frozen and follows the mode with the
   largest resonator participation by the modal assurance criterion,
   ω_{k+1} = √λ, to a relative step of `tol`. With the full complex A at the
   converged ω, the complex eigenvalue gives the radiative decay.
2. Green function: G_{B,h}(x₁, x₁; ω) of the bare membrane (L2-1 columns) and
   ω̃ from the root of Re(1/α − G_{B,h}) by secant iteration; γ_r = Im G_{B,h}.
3. Forced response: monolithic solve with the resonator and a point load at
   `x_source`; the frequency of the maximum of |q/G_B(x₁, x_s)| (resonator
   amplitude over the bare drive) by golden-section search.

## Reproduce

    julia --project -O1 scripts/L2-configB-vertical/L2-3_wet_resonance.jl
"""
module L2_3

using Parameters
using LinearAlgebra: lu, eigen, norm, dot, Symmetric
using SparseArrays: SparseMatrixCSC
using ..References: ResonatorSpec, inv_alpha
import ..L2_1
import ..L1_3: secant_root

export L2_3_Params, VARIABLE, fixed_hash, run_case

"Parameters of one L2-3 case (set P2 of the test matrix); channel field names as `L2_1_Params`."
@with_kw struct L2_3_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    x1::Float64 = 25.0                    # attachment, plate coordinate (membrane on [0, 50])
    elements_per_wavelength::Int = 16
    order::Int = 3
    gap_decays::Float64 = 6.0
    T_ρ::Float64 = 10000.0 / 1025.0       # 9.7561 m³/s²

    # ── Fixed: P2 membrane and fluid ─────────────────────────────────────────
    g::Float64 = 9.81
    ρ::Float64 = 1025.0
    H::Float64 = 25.0
    m_ρ::Float64 = 45.5 / 1025.0          # 0.0444 m
    C_ρ::Float64 = 0.0
    Lp::Float64 = 50.0

    # ── Fixed: resonator and forcing ─────────────────────────────────────────
    m_ρr::Float64 = 113.75 / 1025.0
    ω_r::Float64 = 0.5
    x_source::Float64 = 10.3              # point load for the forced response
    k_spring::Float64 = 1.0

    # ── Fixed: discretisation and solvers ────────────────────────────────────
    n_regular::Int = 0                    # (L2-1 point set unused)
    n_random::Int = 0
    seed::Int = 0
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0
    βₕ::Float64 = 0.5
    ω_mesh::Float64 = 0.5                 # frequency that sizes the mesh (one mesh for all ω)
    tol::Float64 = 1e-11                  # fixed-point / secant tolerance (relative)
    golden_window::Float64 = 2e-3         # forced-peak search in ω̃(1 ± window)
end

"Fields of `L2_3_Params` varied by the L2-3 study (they name the records)."
const VARIABLE = (:x1, :elements_per_wavelength, :order, :gap_decays, :T_ρ)

fixed_hash(p::L2_3_Params) = L2_1.fixed_hash_fields(p, VARIABLE)

resonator(p) = ResonatorSpec(m_ρ=p.m_ρr, k_ρ=p.m_ρr * p.ω_r^2)

"""
    blocks(p, ω; xs, resonators)

Assembled matrix at ω (mesh sized at `ω_mesh`) and the free indices of the
structural (η, q) and fluid (ϕ, κ) degrees of freedom.
"""
function blocks(p, ω::Real; xs, resonators)
    sys = L2_1.build(p; ω=ω, ω_mesh=p.ω_mesh, xs=xs, resonators=resonators)
    offs = cumsum([0; [L2_1.num_free(V) for V in sys.X.spaces]])
    range_of(sym) = (i = sys.fmap[sym]; (offs[i] + 1):offs[i + 1])
    s = [collect(range_of(:η)); sys.qdof]
    f = [collect(range_of(:ϕ)); collect(range_of(:κ))]
    return (; A=sys.A, s, f, sys)
end

"""
    wet_matrices(p, ω; xs=[p.x1], resonators=[resonator(p)])

K_wet, M_dry (from the structural blocks at ω and 2ω) and the added-mass
matrix A(ω) (from the fluid Schur complement at ω), on the structural degrees
of freedom (η, then q), for resonators at the plate coordinates `xs` (none for
the bare structure). `sys` is the assembled system at ω (for mode shapes).
"""
function wet_matrices(p, ω::Real; xs=[p.x1], resonators=[resonator(p)])
    rs = resonators
    b1 = blocks(p, ω; xs=xs, resonators=rs)
    b2 = blocks(p, 2ω; xs=xs, resonators=rs)
    S1 = Matrix(b1.A[b1.s, b1.s])
    S2 = Matrix(b2.A[b2.s, b2.s])
    M = (S1 - S2) ./ (3ω^2)                        # S(ω) = K − ω²M
    K = S1 + ω^2 .* M
    Aff = lu(b1.A[b1.f, b1.f])
    Asf, Afs = b1.A[b1.s, b1.f], b1.A[b1.f, b1.s]
    Fs = zeros(eltype(b1.A), length(b1.s), length(b1.s))
    for c in Iterators.partition(1:length(b1.s), 64)   # column blocks: no dense n_f × n_s array
        Fs[:, c] = Asf * (Aff \ Matrix(Afs[:, c]))
    end
    Aadd = Fs ./ ω^2                               # −A_sf A_ff⁻¹ A_fs = −ω² A
    # HydroElasticFEM assembles the resonator rows dimensionally (×ρ) and the plate rows per unit ρ:
    # scale each resonator row so that its coupling to η equals the transposed column (symmetric
    # K, M; a row scaling of K and M together leaves the eigenproblem unchanged)
    nη = length(b1.s) - length(xs)
    for j in (nη + 1):length(b1.s)
        c = sum(K[1:nη, j]) / sum(K[j, 1:nη])
        K[j, :] .*= c; M[j, :] .*= c; Aadd[j, :] .*= c
    end
    asym = norm(K - transpose(K)) / norm(K) + norm(M - transpose(M)) / norm(M)
    return (; K, M, Aadd, nq=length(xs), ndofs=size(b1.A, 1), he=b1.sys.he, asym, sys=b1.sys, nη)
end

"Modal assurance criterion between two vectors."
mac(u, v) = abs2(dot(u, v)) / (real(dot(u, u)) * real(dot(v, v)))

"""
    wet_mode(p)

Fixed-point iteration of eq:eigWet (Re A frozen) for the resonator-dominated
mode, started at ω_r with the mode of largest resonator participation. Returns
the wet frequency, the iterations, the complex frequency with the full A at
that ω, the resonator participation and the matrices of the last iterate.
"""
function wet_mode(p::L2_3_Params; maxiter::Int=30)
    ω = p.ω_r
    v = nothing
    hist = Float64[]
    local W
    for k in 1:maxiter
        W = wet_matrices(p, ω)
        W.asym < 1e-10 || error("wet matrices not symmetric after row scaling ($(W.asym))")
        e = eigen(Symmetric(real(W.K)), Symmetric(real(W.M + real(W.Aadd))))
        λs, V = e.values, e.vectors
        if v === nothing
            part = [abs2(V[end, j]) / sum(abs2, V[:, j]) for j in axes(V, 2)]   # resonator participation
            j = argmax(part .* (λs .> 0) ./ (1 .+ abs.(sqrt.(abs.(λs)) .- p.ω_r)))
        else
            j = argmax([mac(V[:, i], v) for i in axes(V, 2)])
        end
        v = V[:, j]
        ωn = sqrt(λs[j])
        push!(hist, ωn)
        conv = abs(ωn - ω) <= p.tol * ω
        ω = ωn
        conv && break
    end
    # complex eigenvalue with the full added mass at the converged ω: closest to ω²
    ec = eigen(W.K, W.M + W.Aadd)
    jc = argmin(abs.(ec.values .- ω^2))
    ωc = sqrt(complex(ec.values[jc]))
    return (omega=ω, iterations=hist, omega_complex=ωc, participation=abs2(v[end]) / sum(abs2, v),
            ndofs=W.ndofs, he=W.he)
end

"G_{B,h}(x₁, x₁; ω) of the bare membrane (massless spring at x₁), mesh sized at ω_mesh."
G11(p, ω) = only(L2_1.green_columns(p; ω=ω, ω_mesh=p.ω_mesh, xs=[p.x1]).G)

"""
    green_resonance(p)

ω̃ from Re(1/α − G_{B,h}(x₁, x₁)) = 0 (secant from ω_r), and γ_r = Im G_{B,h} there.
"""
function green_resonance(p::L2_3_Params)
    r = resonator(p)
    cache = Dict{Float64,ComplexF64}()
    g(ω) = get!(() -> G11(p, ω), cache, ω)
    f(ω) = real(inv_alpha(r, ω) - g(ω))
    s = secant_root(f, p.ω_r, p.ω_r * (1 - 1e-3); xtol=p.tol)
    ω̃ = s.root
    return (omega=s.converged ? ω̃ : NaN, gamma_r=imag(g(ω̃)), G=g(ω̃), solves=length(cache))
end

"""
    forced_peak(p, ω̃)

Frequency of the maximum of |q/G_B(x₁, x_s)| for a point load at `x_source`,
golden-section search in ω̃(1 ± `golden_window`).
"""
function forced_peak(p::L2_3_Params, ω̃::Real)
    r = resonator(p)
    spring = ResonatorSpec(m_ρ=0.0, k_ρ=p.k_spring)
    function amp(ω)
        sys = L2_1.build(p; ω=ω, ω_mesh=p.ω_mesh, xs=[p.x1, p.x_source], resonators=[r, spring])
        x = lu(sys.A) \ L2_1.point_load_vector(sys, 2, 1.0, p.k_spring)
        q = L2_1.displacements(sys, x)[1]
        gb = only(L2_1.green_columns(p; ω=ω, ω_mesh=p.ω_mesh, xs=[p.x1, p.x_source]).G[1, 2])
        return abs(q / gb)
    end
    a, b = ω̃ * (1 - p.golden_window), ω̃ * (1 + p.golden_window)
    φ = (sqrt(5) - 1) / 2
    c1, c2 = b - φ * (b - a), a + φ * (b - a)
    f1, f2 = amp(c1), amp(c2)
    n = 2
    while b - a > 1e-9 * ω̃ && n < 60
        if f1 > f2
            b, c2, f2 = c2, c1, f1
            c1 = b - φ * (b - a); f1 = amp(c1)
        else
            a, c1, f1 = c1, c2, f2
            c2 = a + φ * (b - a); f2 = amp(c2)
        end
        n += 1
    end
    return (omega=(a + b) / 2, solves=n)
end

"""
    run_case(p) -> Dict

Wet resonator mode by eq:eigWet, renormalised resonance from G_{B,h}, and the forced-response peak.
"""
function run_case(p::L2_3_Params)
    wm = wet_mode(p)
    gr = green_resonance(p)
    fp = forced_peak(p, isnan(gr.omega) ? wm.omega : gr.omega)
    Γ = gr.gamma_r
    return Dict{String,Any}(
        "omega_wet" => wm.omega, "omega_wet_complex" => wm.omega_complex, "iterations" => wm.iterations,
        "participation" => wm.participation, "omega_green" => gr.omega, "gamma_r" => Γ, "G11" => gr.G,
        "omega_forced" => fp.omega, "forced_solves" => fp.solves, "green_solves" => gr.solves,
        "ndofs" => wm.ndofs, "element_size" => wm.he,
    )
end

end # module L2_3
