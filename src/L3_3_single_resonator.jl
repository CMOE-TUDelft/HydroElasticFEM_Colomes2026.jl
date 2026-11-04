"""
# L3-3 — single resonator on a large 3D plate (quarter symmetry)

## Purpose

Matrix row L3.3 (Phase 4, §6.5). A point resonator at the centre of a
plate that covers the whole free surface (configuration A, set P1, H = 10 m),
in a 3D fluid. The checks are the 3D absorption bound σ_abs ≤ 1/k₀
(`eq:absorption_bound`), the optical theorem (extinction = absorption +
scattering) and the ζ sweep through critical coupling, against the
single-resonator theory of `subsubsec:energy_flux_single`
(`eq:green_function`, `eq:ImG`, `eq:single_force`, `eq:single_sigma`).

## Problem

Quarter domain [0, L_q]² × [0, H] with the resonator at the corner (0, 0, H)
(radiation problem only: the resonator and its point load are axisymmetric).
The plate is P1 (m_ρ = 0.01, C_ρ = 0.49, ν = 0), Kirchhoff–Love C/DG with
penalty γ. The symmetry planes x = 0 and y = 0 are fluid walls (natural) and,
for the plate, ∂η/∂n = 0 imposed by Nitsche on the plate edges
(`symmetry_matrix`): consistency + symmetry + penalty 2γD/h, which is the
restriction to the quarter of the interior C/DG facet terms of the
mirrored full plate (checked against a full-domain solve, `full = true`).
Twisting moment and Kirchhoff shear vanish naturally. An absorbing frame of
width `frame` λ_p: Kelvin–Voigt `StructuralDampingZone` with τ = s/ω and the
profile σ = clamp((max(x, y) − R_s)/frame)², R_s = `r_sponge` λ_p.

Mesh: uniform horizontal cells h = λ_p(ω_ref)/`epw` (fixed across ω),
`nz` vertical cells graded with top cell 0.2/k₀(ω_ref). φ and η of order
`order`.

## Solves (one factorisation per ω)

The bare plate at ω with a unit point force at the origin (f/4 on the
quarter): G_h(0) = η_h(0) and the field G_h(x) on a polar grid in the
undamped region (fit of the outgoing amplitude, `fit_ring`). Every damping
ratio then follows from the bare factor by Sherman–Morrison (scalar for one
resonator): F₁/A = 1/(1/α − G_h(0)), scattered field F₁G_h(x). `monolithic`
solves the resonator in the FE system (library `resonator_array`, M/4, K/4,
C/4 on the quarter) for the identity check.

Included by `scripts/L3-3D-flat-bottom/L3-3_single_resonator.jl` (not part
of the package module list, which other cards edit in parallel).
"""
module L3_3

using Gridap
using Parameters
using Printf
using LinearAlgebra: norm
using SpecialFunctions: hankelh1, hankelh2
import SparseArrays
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

using HydroElasticFEM_Colomes2026.References
using HydroElasticFEM_Colomes2026.Solvers: MetisLUSolver, factorize, solve_rhs, factor_stats
import HydroElasticFEM_Colomes2026.Solvers as Solvers
import SparseArrays.LibSuiteSparse as LSS
using LinearAlgebra: lu
import HydroElasticFEM_Colomes2026.L3_0

export L3_3_Params, VARIABLE, fixed_hash, layout, run_case, fit_ring, host, resonator, size_counts

"Parameters of one L3-3 case (one ω, one mesh)."
@with_kw struct L3_3_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    ω::Float64 = 9.95                     # frequency (rad/s)
    order::Int = 3                        # φ and η order
    epw::Float64 = 4.5                    # horizontal cells per λ_p(ω_ref) (frozen 5.0; 4.5 post-freeze, 20 GiB footprint)
    r_sponge::Float64 = 5.0               # frame starts at max(x, y) = r_sponge λ_p
    frame::Float64 = 3.0                  # frame width (λ_p)
    full::Bool = false                    # full domain (identity check, small meshes only)
    monolithic::Bool = false              # also solve the resonator in the FE system (ζ = ζ_mono)

    # ── Fixed: host (set P1) and resonator ───────────────────────────────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49
    hb::Float64 = 0.01
    ν::Float64 = 0.0
    m_r::Float64 = 0.01                   # resonator mass / ρ (m³)
    ω_r::Float64 = 10.0
    ζ_mono::Float64 = 0.0129              # damping ratio of the monolithic check
    f::Float64 = 1.0                      # point force / ρ (full plane)

    # ── Fixed: discretisation ─────────────────────────────────────────────────
    ω_ref::Float64 = 10.0                 # mesh sized with λ_p(ω_ref)
    nz::Int = 8
    top_k::Float64 = 0.2
    γ::Float64 = 12.0                     # C/DG penalty (p(p+1) for p = 3)
    sponge_strength::Float64 = 10.0       # ωτ at full strength
    sponge_power::Int = 2

    # ── Fixed: post-processing ────────────────────────────────────────────────
    ring_rmin::Float64 = 1.5              # λ_p(ω_ref)
    ring_rmax_margin::Float64 = 0.5       # ring ends this far (λ_p) inside the frame
    n_r::Int = 37
    n_θ::Int = 19                         # θ ∈ [0, π/2], 5° steps
    n_line::Int = 241
end

const VARIABLE = (:ω, :order, :epw, :r_sponge, :frame, :full, :monolithic)

function fixed_hash(p::L3_3_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L3_3_Params) if !(f in VARIABLE))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

host(p) = HostSpec(g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.m_ρ, C_ρ=p.C_ρ)
resonator(p; ζ=0.0) = resonator_from_frequency(p.m_r, p.ω_r; ζ=ζ)

"Mesh layout: cell size, inner and frame cells, R_s, L_q; x0 = −L_q for the full domain."
function layout(p::L3_3_Params)
    λ = 2π / k0(host(p), p.ω_ref)
    he = λ / p.epw
    ni = round(Int, p.r_sponge * p.epw)
    nf = round(Int, p.frame * p.epw)
    n = ni + nf
    Rs, Lq = ni * he, n * he
    rmax = Rs - p.ring_rmax_margin * λ
    return (; λ, he, ni, nf, n, Rs, Lq, Lf=Lq - Rs, x0=p.full ? -Lq : 0.0, parts=p.full ? 2n : n,
            top=p.top_k / k0(host(p), p.ω_ref), ring=(p.ring_rmin * λ, rmax))
end

"DOF count (tensor product: φ in the box, η on the top surface; one extra per resonator)."
function size_counts(p::L3_3_Params)
    l = layout(p); o = p.order; m = l.parts
    N = (o * m + 1)^2 * (o * p.nz + 1) + (o * m + 1)^2
    return (; N, cells=m, l.Lq, l.Rs)
end

"Frame profile σ(x) ∈ [0, 1] (square frame, power ramp)."
sponge_profile(p, l) = x -> clamp((max(abs(x[1]), abs(x[2])) - l.Rs) / l.Lf, 0.0, 1.0)^p.sponge_power

function tank(p, l)
    fz = L3_0.zmap(p.H, p.nz, l.top)
    cart = G.CartesianDomain(mins=(l.x0, l.x0, 0.0), maxs=(l.Lq, l.Lq, p.H), parts=(l.parts, l.parts, p.nz),
                             map=x -> VectorValue(x[1], x[2], fz(x[3])))
    W = l.Lq - l.x0
    return G.TankDomain(cart;
        structure_domains=[G.StructureDomain(L=W, W=W, x₀=[l.x0, l.x0, p.H], domain_symbol=:Γ_plate)],
        resonator_domains=[G.ResonatorDomain(location=[0.0, 0.0, p.H], delta_symbol=:δ1)])
end

"Fluid, plate (with the frame) and, for `res`, the resonator (quarter: M/4, K/4, C/4)."
function physics(p, l, ω; res=nothing)
    CT = Vector{ComplexF64}
    fe(; kw...) = PH.FESpaceConfig(; order=p.order, vector_type=CT, kw...)
    fluid = P.PotentialFlow(g=p.g, ρw=p.ρ, dim=3, fe=fe(), space_domain_symbol=:Ω)
    zone = P.StructuralDampingZone(profile=sponge_profile(p, l), τ=p.sponge_strength / ω)
    plate = P.KirchhoffLovePlate(E=12 * p.C_ρ * p.ρ / p.hb^3, ν=p.ν, hb=p.hb, ρ=p.ρ, ρb=p.m_ρ * p.ρ / p.hb,
                                 g=p.g, ambient_dim=3, manifold_dim=2, symbol=:η, fe=fe(γ=p.γ),
                                 space_domain_symbol=:Γη, damping_zones=[zone])
    ents = P.PhysicsParameters[fluid, plate]
    if !isnothing(res)
        s = p.full ? 1.0 : 0.25
        ra = P.resonator_array(1, s * res.m_ρ * p.ρ, s * res.k_ρ * p.ρ, s * res.c_ρ * p.ρ;
                               host_domain_symbol=:Γη, delta_domain_symbols=[:δ1], ρw=p.ρ,
                               fe=PH.FESpaceConfig(space_type=VectorValue{1,Float64}, vector_type=CT))
        length(P.attached_resonators(ra, plate)) == 1 || error("resonator not attached to the plate")
        push!(ents, ra)
    end
    return ents, plate
end

"""
    symmetry_matrix(problem, plate, p, l, ω)

Nitsche terms for ∂η/∂n = 0 on the plate edges x = 0 and y = 0 (quarter only):
∫ w [−∂ₙv M_nn(η) − M_nn(v) ∂ₙη + 2γD/h ∂ₙv ∂ₙη] dΛ, with M = C ⊙ ∇∇η and
the Kelvin–Voigt weight w = 1 − i s σ(x) of the frame; h is the library's
plate h (`:h_η`). Assembled on the problem's multi-field spaces.
"""
function symmetry_matrix(problem, plate, p, l, ω)
    dom = S.get_integration_domains(problem)
    U, V = S.get_trial_fe_space(problem), S.get_test_fe_space(problem)
    iη = S.get_field_map(problem)[:η]
    dΛ, n, h = dom[:dΛ∂η], dom[:n_Λ∂η], dom[:h_η]
    C = plate.C
    Dρ = C[1, 1, 1, 1]
    σ = sponge_profile(p, l)
    tol = 1e-8 * l.Lq
    w(x) = (abs(x[1]) < tol || abs(x[2]) < tol) ? (1.0 - im * p.sponge_strength * σ(x)) : zero(ComplexF64)
    Mnn(u) = n ⋅ ((C ⊙ ∇∇(u)) ⋅ n)
    dn(u) = ∇(u) ⋅ n
    a(u, v) = ∫(w * (-(dn(v[iη]) * Mnn(u[iη])) - Mnn(v[iη]) * dn(u[iη]) + (2 * p.γ * Dρ / h) * dn(v[iη]) * dn(u[iη])))dΛ
    return assemble_matrix(a, U, V)
end

"Point-load vector f·δ(v_η) at the origin (quarter: f/4)."
function point_load(problem, p)
    Y = S.get_test_fe_space(problem)
    iη = S.get_field_map(problem)[:η]
    δ = S.get_integration_domains(problem)[:δ1]
    return (p.full ? p.f : p.f / 4) .* assemble_vector(y -> δ(y[iη]), Y)
end

eval_on_plate(fh, pts) = ComplexF64.(L3_0.eval_on_plate(fh, pts))

"""
    factorize_prealloc(solver, A)

`factorize(solver, A)` with UMFPACK's initial numeric allocation set to its full
symbolic estimate (Control[ALLOC_INIT] = 1 instead of 0.7): same ordering, pivoting
and factors; avoids the realloc growth of the LU memory, whose copies raised the
process footprint (the rss_guard measure) well above UMFPACK's own peak.
"""
function factorize_prealloc(solver::MetisLUSolver, A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    c = Solvers.umfpack_control(solver, Tv, Ti)
    c[LSS.UMFPACK_ALLOC_INIT + 1] = 1.0
    return Solvers.MetisLUNumericalSetup(lu(A; control=c), solver)
end

"Return freed malloc pages to the OS (macOS: they otherwise stay in the process footprint that rss_guard limits)."
release_freed() = Sys.isapple() ? ccall(:malloc_zone_pressure_relief, Csize_t, (Ptr{Cvoid}, Csize_t), C_NULL, 0) : 0

"Assemble (+ symmetry terms), factorise, solve one RHS. Returns (η field, point value, stats Dict)."
function solve_one(p, l, ω; res=nothing, rec, tag)
    tk = tank(p, l)
    ents, plate = physics(p, l, ω; res=res)
    solver = MetisLUSolver()
    problem = L3_0.phase!(() -> S.build_problem(tk, ents, PH.FreqDomainConfig(ω=ω, solver=solver)), rec,
                          "$(tag)_assembly")
    op = S.get_fe_operator(problem)
    A = get_matrix(op)
    if !p.full
        As = L3_0.phase!(() -> symmetry_matrix(problem, plate, p, l, ω), rec, "$(tag)_symmetry")
        rec["$(tag)_sym_norm_rel"] = norm(As, 1) / norm(A, 1)
        A = A + As
        As = nothing
    end
    b = point_load(problem, p)
    rec["$(tag)_load_sum"] = sum(b)
    X = S.get_trial_fe_space(problem)
    fmap = S.get_field_map(problem)
    rec["$(tag)_N"] = size(A, 1); rec["$(tag)_nnzA"] = SparseArrays.nnz(A)
    problem = nothing; op = nothing; GC.gc(true); release_freed()
    ns = L3_0.phase!(() -> factorize_prealloc(solver, A), rec, "$(tag)_factorize")
    st = factor_stats(ns)
    for (k, j) in (("alloc_init_used", LSS.UMFPACK_ALLOC_INIT_USED), ("realloc", LSS.UMFPACK_NUMERIC_REALLOC),
                   ("costly_realloc", LSS.UMFPACK_NUMERIC_COSTLY_REALLOC), ("defrag", LSS.UMFPACK_NUMERIC_DEFRAG))
        rec["$(tag)_umf_$(k)"] = ns.factors.info[j + 1]
    end
    rec["$(tag)_lunz"] = st.lunz; rec["$(tag)_umf_peak_gb"] = st.peak_gb; rec["$(tag)_ordering"] = String(st.ordering)
    x = L3_0.phase!(() -> solve_rhs(ns, b), rec, "$(tag)_solve")
    rec["$(tag)_rel_residual"] = norm(A * x - b) / norm(b)
    finalize(ns.factors); ns = nothing; A = nothing; GC.gc(true); release_freed()
    uh = FEFunction(X, x)
    ηh = uh[fmap[:η]]
    η0 = eval_on_plate(ηh, [Point(0.0, 0.0, p.H)])[1]
    q = isnothing(res) ? NaN + 0im : ComplexF64(get_free_dof_values(uh[fmap[:q1]])[1])
    return ηh, η0, q
end

"""
    run_case(p) -> Dict

Bare plate at p.ω (G_h(0), polar grid, x-axis line and diagonal); with
`monolithic`, also the resonator (ζ_mono) driven by the same point force.
"""
function run_case(p::L3_3_Params)
    l = layout(p)
    rec = Dict{String,Any}()
    for k in (:λ, :he, :ni, :nf, :n, :Rs, :Lq, :Lf, :parts)
        rec[string(k)] = getfield(l, k)
    end
    rec["ring_r"] = collect(range(l.ring[1], l.ring[2]; length=p.n_r))
    rec["ring_theta"] = collect(range(0.0, π / 2; length=p.n_θ))
    ηh, G0, _ = solve_one(p, l, p.ω; rec=rec, tag="bare")
    rec["G0"] = G0 / p.f
    L3_0.phase!(rec, "post") do
        pts = [Point(r * cos(θ), r * sin(θ), p.H) for r in rec["ring_r"] for θ in rec["ring_theta"]]
        rec["ring_eta"] = permutedims(reshape(eval_on_plate(ηh, pts), p.n_θ, p.n_r)) ./ p.f   # n_r × n_θ
        xs = collect(range(0.0, l.Lq; length=p.n_line))
        rec["line_x"] = xs
        rec["line_eta"] = eval_on_plate(ηh, [Point(x, 0.0, p.H) for x in xs]) ./ p.f
        rec["diag_eta"] = eval_on_plate(ηh, [Point(x / sqrt(2), x / sqrt(2), p.H) for x in xs]) ./ p.f
        q = [(0.23l.Rs, 0.1l.Rs), (0.6l.Rs, 0.35l.Rs), (0.9l.Rs, 0.0)]
        rec["mirror_pts"] = q
        if p.full    # mirror images of a few first-quadrant points (symmetry of the full solution)
            rec["mirror_eta"] = [eval_on_plate(ηh, [Point(sx * a, sy * b, p.H) for (a, b) in q])
                                 for (sx, sy) in ((1, 1), (-1, 1), (1, -1), (-1, -1))]
        else
            rec["mirror_eta"] = [eval_on_plate(ηh, [Point(a, b, p.H) for (a, b) in q])]
        end
        nothing
    end
    ηh = nothing; GC.gc(true)
    if p.monolithic
        r = resonator(p; ζ=p.ζ_mono)
        _, η0, q = solve_one(p, l, p.ω; res=r, rec=rec, tag="mono")
        rec["mono_eta0"], rec["mono_q"] = η0, q
        α = alpha(r, p.ω)
        rec["mono_eta0_SM"] = rec["G0"] * p.f / (1 - α * rec["G0"])
        rec["mono_zeta"] = p.ζ_mono
    end
    rec["maxrss_final_gb"] = L3_0.maxrss_gb()
    return rec
end

# ─── Far-field fit ──────────────────────────────────────────────────────────

"""
    fit_ring(rs, ηs, h, ω; n_extra=6, kscan=0.03)

Least-squares fit of a radial profile of the source-free field outside the
near field:

    η(r) = C⁺H₀⁽¹⁾(k r) + C⁻H₀⁽²⁾(k r) + Σₙ cₙ H₀⁽¹⁾(kₙ r),

kₙ the next `n_extra` roots of the bare dispersion relation (complex pair,
first four fluid-evanescent modes; each ≤ 0.2 % of the propagating term at r = 1.5λ_p). The propagating k is fitted (scan within ±`kscan` of k₀,
then golden section). Returns (Cplus, Cminus, k, residual).
"""
function fit_ring(rs::AbstractVector, ηs::AbstractVector, h::HostSpec, ω::Real; n_extra::Int=6, kscan::Real=0.03)
    roots = host_roots(h, ω; n_evanescent=10)
    k0_ = real(roots[1])
    extra = roots[2:(1 + n_extra)]
    function lsq(k)
        M = hcat(hankelh1.(0, k .* rs), hankelh2.(0, k .* rs), [hankelh1.(0, kn .* rs) for kn in extra]...)
        c = M \ ηs
        return c, norm(M * c - ηs) / norm(ηs)
    end
    ks = range((1 - kscan) * k0_, (1 + kscan) * k0_; length=121)
    res = [lsq(k)[2] for k in ks]
    i = argmin(res)
    a, b = ks[max(i - 1, 1)], ks[min(i + 1, length(ks))]
    φ = (sqrt(5) - 1) / 2
    for _ in 1:80
        c1, c2 = b - φ * (b - a), a + φ * (b - a)
        lsq(c1)[2] < lsq(c2)[2] ? (b = c2) : (a = c1)
    end
    k = (a + b) / 2
    c, r = lsq(k)
    return (Cplus=c[1], Cminus=c[2], k=k, residual=r, extra=c[3:end])
end

end
