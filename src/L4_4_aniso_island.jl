"""
# L4-4 — anisotropic resonator island on a 3D plate (homogenised `ResonatorField`)

## Relation to the paper

Section 3.3, `subsubsec:aniso_rays`. An isotropic plate (set P1, T_ρ = 0,
H = 10 m) carries an island of resonators with the elliptic density
n(x) = n₀φ(ξ), ξ = √((x/L_x)² + (y/L_y)²) (`eq:elliptic_density`). The local
wavenumber depends on position through ξ only (`eq:k_of_position_general`).
A plane wave crossing the island along x accumulates the complex phase
Φ_x(y) = ∫(k − k₀)dx (`eq:eikonal_phase`). Predictions tested here:

- optical depth ratio 𝒜_x/𝒜_y = L_x/L_y (`eq:aspect_ratio_law`, any loading);
- focal length f_x = k₀L_y²/(L_xJ) and f_x/f_y = (L_y/L_x)³
  (`eq:focal_general`, `eq:focal_ratio_law`; Gaussian weak-loading value
  `eq:focal_gaussian`); converging below ω_r, diverging above;
- the reference of the matrix row is the ray script
  `R-SEC34-05_sec33_aniso/aniso_island_rays.py` (ported below: `ray_focus`).

The focal points (20–400 m) lie outside any affordable 3D domain, so the
lens is measured by the phase of the transmitted field: the FE field on
lines behind the island is propagated back to the island centre x = 0 with
the exact bare-plate propagator (angular spectrum), which gives the
equivalent thin transmission T(y) = |T|e^{iΦ(y)}. Then f = −k₀/Φ''(0)
(the paraxial ray focus measured from the centre, thick lens included) and
𝒜 = −ln|T(0)|. A wave reference that keeps diffraction, the wide-angle
split-step parabolic equation through the exact local wavenumber field
(`pe_march`), is evaluated on the same lines and processed identically.

## Problem

Half domain y ≥ 0 (incidence along +x, the island is symmetric in y):
box [x₀, x₁] × [0, Y] × [0, H] covered by the Kirchhoff–Love plate (C/DG,
γ = 12 for p = 3). Symmetry plane y = 0: fluid wall (natural) and, for the
plate, ∂η/∂n = 0 by Nitsche (`symmetry_matrix`, as in L3-3). Absorbing frame
of width `frame` λ_p on the three outer sides: Kelvin–Voigt
`StructuralDampingZone`, τ = s/ω, ramp σ². The island is the upstream
`ResonatorField` (branch `feat/resonator-field`): q on the whole plate,
n(x) = n₀[φ(ξ) + n_floor], m_r = 0.01 m³ per resonator, ω_r = 10,
C = 2ζ√(KM). Profile: Gaussian e^{−ξ²/2} with a cosine taper to zero on
ξ ∈ [ξ_t, ξ_c] = [2, 2.5] (the laws hold for any profile; the references use
the same φ). n_floor = 1e-6 keeps the q rows regular outside the island.

Scattered-field formulation: u = u_inc + u_s with u_inc the bare plate wave
(η_inc = e^{ik₀x}, q_inc = 0). Since the bare operator annihilates u_inc,
A u_s = −A_island u_inc, i.e. the load
b(v, ξ) = ∫ n(K − iωC)/ρ η_inc (ξ − v) dΓ, which lives on the island only;
the frame therefore absorbs the scattered field only. The identity of this
load with −(A(2n) − A(n))U_inc is checked on a small mesh.

## Mesh

Uniform horizontal cells h = λ_p(ω)/`epw`, `nz` vertical cells graded with
top cell 0.2/k₀ (L3-0 rule), φ, η, q of order p = 3.

Included by `scripts/L4-beyond-analytical/L4-4_aniso_island.jl`, run in
`envs/resonator_field` (not part of the package module).
"""
module L4_4

using Gridap
using Parameters
using Printf
using LinearAlgebra: norm, dot
import SparseArrays
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

using HydroElasticFEM_xxx2026.References: HostSpec, k0, dD0
using HydroElasticFEM_xxx2026.Solvers: MetisLUSolver, factorize, solve_rhs, factor_stats
import HydroElasticFEM_xxx2026.L3_0

export L4_4_Params, VARIABLE, fixed_hash, layout, size_counts, run_case, rhs_identity,
       host, chi, profile, local_k_table, eikonal_phase, ray_focus, thin_lens_gaussian, thin_lens_J,
       pe_march, backprop, curvature_fit, fft!, ifft!

"Parameters of one L4-4 case (one ω, one island, one mesh)."
@with_kw struct L4_4_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    ω::Float64 = 8.0
    Lx::Float64 = 6.0                     # island semi-axes (m)
    Ly::Float64 = 4.0
    load::Float64 = 0.03                  # peak loading n₀m_ρ,r (m)
    ζ::Float64 = 0.0
    epw::Float64 = 4.0                    # horizontal cells per λ_p(ω)
    frame::Float64 = 1.5                  # frame width (λ_p)
    gap::Float64 = 0.5                    # island support / last line → frame (λ_p)
    full::Bool = false                    # full domain in y (identity check, small meshes only)
    scale_n::Float64 = 1.0                # multiplies n (identity check of the load)

    # ── Fixed: host P1, resonators ────────────────────────────────────────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49
    hb::Float64 = 0.01
    ν::Float64 = 0.0
    m_r::Float64 = 0.01                   # resonator mass / ρ (m³); n₀ = load/m_r
    ω_r::Float64 = 10.0
    ξ_t::Float64 = 2.0                    # taper start
    ξ_c::Float64 = 2.5                    # support end
    n_floor::Float64 = 1e-6

    # ── Fixed: discretisation ─────────────────────────────────────────────────
    order::Int = 3
    nz::Int = 8
    top_k::Float64 = 0.2
    γ::Float64 = 12.0
    sponge_strength::Float64 = 10.0
    sponge_power::Int = 2

    # ── Fixed: post-processing ────────────────────────────────────────────────
    line_dx::Tuple{Float64,Float64,Float64} = (0.5, 1.5, 2.5)   # lines at x = ξ_c L_x + d (m)
    dy_line::Float64 = 0.05
    n_axis::Int = 401
    n_map::Tuple{Int,Int} = (161, 61)
end

const VARIABLE = (:ω, :Lx, :Ly, :load, :ζ, :epw, :frame, :gap, :full, :scale_n)

function fixed_hash(p::L4_4_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L4_4_Params) if !(f in VARIABLE))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

host(p) = HostSpec(g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.m_ρ, C_ρ=p.C_ρ)

"χ(ω) of `eq:aniso_density_dispersion` (time factor e^{−iωt})."
chi(ω, ω_r, ζ) = (1 - 2im * ζ * ω / ω_r) / (1 - (ω / ω_r)^2 - 2im * ζ * ω / ω_r)

"Island profile φ(ξ): Gaussian with a cosine taper to 0 on [ξ_t, ξ_c] (`taper=false`: plain Gaussian)."
function profile(ξ::Real, ξ_t::Real, ξ_c::Real; taper::Bool=true)
    g = exp(-ξ^2 / 2)
    taper || return g
    ξ <= ξ_t && return g
    ξ >= ξ_c && return 0.0
    return g * 0.5 * (1 + cos(π * (ξ - ξ_t) / (ξ_c - ξ_t)))
end
profile(p::L4_4_Params, x, y) = profile(sqrt((x / p.Lx)^2 + (y / p.Ly)^2), p.ξ_t, p.ξ_c)

"Layout: λ, cell size, cells, box, frame, lines."
function layout(p::L4_4_Params)
    λ = 2π / k0(host(p), p.ω)
    he = λ / p.epw
    F = p.frame * λ
    xs, ys = p.ξ_c * p.Lx, p.ξ_c * p.Ly
    x0 = -(xs + (p.gap + p.frame) * λ)
    nx = ceil(Int, (xs + maximum(p.line_dx) + (p.gap + p.frame) * λ - x0) / he - 1e-9)
    ny = ceil(Int, (ys + (p.gap + p.frame) * λ) / he - 1e-9)
    x1 = x0 + nx * he
    Y = ny * he
    lines = [xs + d for d in p.line_dx]
    yend = Y - F                          # undamped region |y| < yend
    return (; λ, k=2π / λ, he, F, x0, x1, Y, nx, ny, y0=p.full ? -Y : 0.0, nyt=p.full ? 2ny : ny,
            lines, yend, xin=(x0 + F, x1 - F), top=p.top_k * λ / (2π))
end

"DOF count: φ in the box, η and q on the top surface."
function size_counts(p::L4_4_Params)
    l = layout(p); o = p.order
    Ns = (o * l.nx + 1) * (o * l.nyt + 1)
    N = Ns * (o * p.nz + 1) + 2Ns
    return (; N, nx=l.nx, ny=l.nyt, nz=p.nz, Lx=l.x1 - l.x0, Ly=l.Y - l.y0)
end

"Frame profile σ ∈ [0, 1]: distance into the frame / F, squared."
function sponge_profile(p, l)
    return x -> begin
        dx = max(l.xin[1] - x[1], x[1] - l.xin[2], 0.0)
        dy = max(abs(x[2]) - l.yend, 0.0)
        clamp(max(dx, dy) / l.F, 0.0, 1.0)^p.sponge_power
    end
end

function tank(p, l)
    fz = L3_0.zmap(p.H, p.nz, l.top)
    cart = G.CartesianDomain(mins=(l.x0, l.y0, 0.0), maxs=(l.x1, l.Y, p.H), parts=(l.nx, l.nyt, p.nz),
                             map=x -> VectorValue(x[1], x[2], fz(x[3])))
    return G.TankDomain(cart;
        structure_domains=[G.StructureDomain(L=l.x1 - l.x0, W=l.Y - l.y0, x₀=[l.x0, l.y0, p.H],
                                             domain_symbol=:Γ_plate)])
end

"Number density n(x) (per m²) of the island, floor included."
density(p) = x -> p.scale_n * (p.load / p.m_r) * (profile(p, x[1], x[2]) + p.n_floor)

"Resonator constants per resonator / ρ: (M, K, C)."
function resonator_mkc(p)
    M = p.m_r; K = M * p.ω_r^2
    return (M=M, K=K, C=2p.ζ * sqrt(K * M))
end

function physics(p, l, ω)
    CT = Vector{ComplexF64}
    fe(; kw...) = PH.FESpaceConfig(; order=p.order, vector_type=CT, kw...)
    fluid = P.PotentialFlow(g=p.g, ρw=p.ρ, dim=3, fe=fe(), space_domain_symbol=:Ω)
    zone = P.StructuralDampingZone(profile=sponge_profile(p, l), τ=p.sponge_strength / ω)
    plate = P.KirchhoffLovePlate(E=12 * p.C_ρ * p.ρ / p.hb^3, ν=p.ν, hb=p.hb, ρ=p.ρ, ρb=p.m_ρ * p.ρ / p.hb,
                                 g=p.g, ambient_dim=3, manifold_dim=2, symbol=:η, fe=fe(γ=p.γ),
                                 space_domain_symbol=:Γη, damping_zones=[zone])
    r = resonator_mkc(p)
    field = P.ResonatorField(M=r.M * p.ρ, K=r.K * p.ρ, C=r.C * p.ρ, n=density(p), ρw=p.ρ, symbol=:q,
                             space_domain_symbol=:Γη, fe=fe())
    return P.PhysicsParameters[fluid, plate, field], plate
end

"""
    symmetry_matrix(problem, plate, p, l)

Nitsche terms for ∂η/∂n = 0 on the plate edge y = 0 (half domain):
∫ w [−∂ₙv M_nn(η) − M_nn(v) ∂ₙη + 2γD/h ∂ₙv ∂ₙη] dΛ, w = 1 − i s σ(x) (the
Kelvin–Voigt weight of the frame), as `L3_3.symmetry_matrix` restricted to
one edge.
"""
function symmetry_matrix(problem, plate, p, l)
    dom = S.get_integration_domains(problem)
    U, V = S.get_trial_fe_space(problem), S.get_test_fe_space(problem)
    iη = S.get_field_map(problem)[:η]
    dΛ, n, h = dom[:dΛ∂η], dom[:n_Λ∂η], dom[:h_η]
    C = plate.C
    Dρ = C[1, 1, 1, 1]
    σ = sponge_profile(p, l)
    tol = 1e-8 * (l.x1 - l.x0)
    w(x) = abs(x[2]) < tol ? (1.0 - im * p.sponge_strength * σ(x)) : zero(ComplexF64)
    Mnn(u) = n ⋅ ((C ⊙ ∇∇(u)) ⋅ n)
    dn(u) = ∇(u) ⋅ n
    a(u, v) = ∫(w * (-(dn(v[iη]) * Mnn(u[iη])) - Mnn(v[iη]) * dn(u[iη]) + (2 * p.γ * Dρ / h) * dn(v[iη]) * dn(u[iη])))dΛ
    return assemble_matrix(a, U, V)
end

"Plate measure and the η / q test indices of a problem."
function plate_measure(problem, p)
    V = S.get_test_fe_space(problem)
    fm = S.get_field_map(problem)
    trian = get_triangulation(V.spaces[fm[:η]])
    return Measure(trian, 2p.order + 4), fm[:η], fm[:q], trian
end

"""
    island_load(problem, p, ω)

Scattered-field load b(v, ξ) = ∫ n(K − iωC)/ρ η_inc (ξ − v) dΓ, η_inc = e^{ik₀x}.
"""
function island_load(problem, p, ω)
    dΓ, iη, iq, trian = plate_measure(problem, p)
    r = resonator_mkc(p)
    kk = k0(host(p), ω)
    nf = density(p)
    c = CellField(x -> nf(x) * (r.K - im * ω * r.C) * cis(kk * x[1]), trian)
    V = S.get_test_fe_space(problem)
    return assemble_vector(y -> ∫(c * (y[iq] - y[iη]))dΓ, V)
end

eval_on_plate(fh, pts) = ComplexF64.(L3_0.eval_on_plate(fh, pts))

"Assembled (library + symmetry) matrix, load, spaces."
function assemble_case(p, l, ω; rec, tag)
    tk = tank(p, l)
    ents, plate = physics(p, l, ω)
    problem = L3_0.phase!(() -> S.build_problem(tk, ents, PH.FreqDomainConfig(ω=ω, solver=MetisLUSolver())), rec,
                          "$(tag)_assembly")
    A = get_matrix(S.get_fe_operator(problem))
    if !p.full
        As = L3_0.phase!(() -> symmetry_matrix(problem, plate, p, l), rec, "$(tag)_symmetry")
        rec["$(tag)_sym_norm_rel"] = norm(As, 1) / norm(A, 1)
        A = A + As
    end
    b = island_load(problem, p, ω)
    return problem, A, b
end

"""
    rhs_identity(p) -> Dict

Small-mesh identity: the scattered-field load equals −(A(2n) − A(n))U_inc,
U_inc = (0, interpolated η_inc, 0), to round-off.
"""
function rhs_identity(p::L4_4_Params)
    l = layout(p)
    rec = Dict{String,Any}()
    pr1, A1, b = assemble_case(p, l, p.ω; rec=rec, tag="n1")
    p2 = L4_4_Params(p; scale_n=2p.scale_n)
    _, A2, _ = assemble_case(p2, l, p.ω; rec=rec, tag="n2")
    X = S.get_trial_fe_space(pr1)
    fm = S.get_field_map(pr1)
    kk = k0(host(p), p.ω)
    objs = Any[0.0 for _ in 1:length(X.spaces)]
    objs[fm[:η]] = x -> cis(kk * x[1])
    U = get_free_dof_values(interpolate_everywhere(objs, X))
    bref = -((A2 - A1) * U)
    rec["rhs_identity"] = norm(b - bref) / norm(bref)
    rec["N"] = size(A1, 1)
    return rec
end

"""
    run_case(p) -> Dict

Assemble, factorise (METIS), solve the scattered field, evaluate
T = (η_inc + η_s)/η_inc on the lines, the axis and a map.
"""
function run_case(p::L4_4_Params)
    l = layout(p)
    rec = Dict{String,Any}()
    for k in (:λ, :k, :he, :F, :x0, :x1, :Y, :nx, :ny, :lines, :yend)
        rec[string(k)] = getfield(l, k)
    end
    problem, A, b = assemble_case(p, l, p.ω; rec=rec, tag="fe")
    X = S.get_trial_fe_space(problem)
    fm = S.get_field_map(problem)
    rec["N"] = size(A, 1); rec["nnzA"] = SparseArrays.nnz(A)
    rec["ndofs_fields"] = [num_free_dofs(Xi) for Xi in X.spaces]
    problem = nothing; GC.gc(true)
    solver = MetisLUSolver()
    ns = L3_0.phase!(() -> factorize(solver, A), rec, "factorize")
    st = factor_stats(ns)
    rec["lunz"] = st.lunz; rec["umf_peak_gb"] = st.peak_gb; rec["ordering"] = String(st.ordering)
    x = L3_0.phase!(() -> solve_rhs(ns, b), rec, "solve")
    rec["rel_residual"] = norm(A * x - b) / norm(b)
    finalize(ns.factors); ns = nothing; A = nothing; GC.gc(true)
    uh = FEFunction(X, x)
    ηh = uh[fm[:η]]
    qh = uh[fm[:q]]
    kk = l.k
    L3_0.phase!(rec, "post") do
        ys = collect(0.0:p.dy_line:(l.yend - 1e-9))
        rec["line_y"] = ys
        rec["line_T"] = [1 .+ eval_on_plate(ηh, [Point(xm, y, p.H) for y in ys]) ./ cis(kk * xm) for xm in l.lines]
        xa = collect(range(l.xin[1], l.xin[2]; length=p.n_axis))
        rec["axis_x"] = xa
        rec["axis_T"] = 1 .+ eval_on_plate(ηh, [Point(x, 0.0, p.H) for x in xa]) ./ cis.(kk .* xa)
        rec["axis_q"] = eval_on_plate(qh, [Point(x, 0.0, p.H) for x in xa])
        mx = collect(range(l.xin[1], l.xin[2]; length=p.n_map[1]))
        my = collect(range(0.0, l.yend; length=p.n_map[2]))
        rec["map_x"], rec["map_y"] = mx, my
        rec["map_T"] = reshape(1 .+ eval_on_plate(ηh, [Point(x, y, p.H) for x in mx for y in my]) ./
                               repeat(cis.(kk .* mx); inner=length(my)), length(my), length(mx))
        if p.full   # mirror points of the full solution (symmetry identity)
            q = [(l.lines[1], 0.3l.yend), (0.0, 0.5p.Ly), (-0.5p.Lx, 0.2l.yend)]
            rec["mirror_pts"] = q
            rec["mirror_T"] = [eval_on_plate(ηh, [Point(a, s * b, p.H) for (a, b) in q]) for s in (1, -1)]
        else
            q = [(l.lines[1], 0.3l.yend), (0.0, 0.5p.Ly), (-0.5p.Lx, 0.2l.yend)]
            rec["mirror_pts"] = q
            rec["mirror_T"] = [eval_on_plate(ηh, [Point(a, b, p.H) for (a, b) in q])]
        end
        nothing
    end
    rec["maxrss_final_gb"] = L3_0.maxrss_gb()
    return rec
end

# ─── References: local wavenumber field ─────────────────────────────────────

"""
    local_k_table(h, ω, load, χ; n=4001)

Exact local root k(M) of 𝒟₀(k, ω) − ω²Mχ = 0 on the branch continued from
the bare root k₀ (Newton in complex k, continuation in M), on M ∈ [0, load].
Returns a function φ ↦ k(load·φ) (linear interpolation in φ ∈ [0, 1]).
"""
function local_k_table(h::HostSpec, ω::Real, load::Real, χ::Number; n::Int=4001)
    kb = k0(h, ω)
    D(k, M) = h.g + h.C_ρ * k^4 - ω^2 * (h.m_ρ + 1 / (k * tanh(k * h.H)) + M * χ)
    dD(k) = 4h.C_ρ * k^3 + ω^2 * (tanh(k * h.H) + k * h.H / cosh(k * h.H)^2) / (k * tanh(k * h.H))^2
    φs = range(0.0, 1.0; length=n)
    ks = Vector{ComplexF64}(undef, n)
    k = complex(kb)
    for (i, φ) in enumerate(φs)
        M = load * φ
        for _ in 1:60
            δ = D(k, M) / dD(k)
            k -= δ
            abs(δ) < 1e-14 * abs(k) && break
        end
        ks[i] = k
    end
    function kφ(φ)
        t = clamp(φ, 0.0, 1.0) * (n - 1)
        i = min(floor(Int, t), n - 2)
        s = t - i
        return (1 - s) * ks[i + 1] + s * ks[i + 2]
    end
    return kφ, kb
end

"""
    eikonal_phase(kφ, k0, Lx, Ly, ys, ξ_t, ξ_c; taper=true, nx=4001)

Straight-path phase Φ_x(y) = ∫(k − k₀)dx (`eq:eikonal_phase`) at each y.
"""
function eikonal_phase(kφ, kb, Lx, Ly, ys, ξ_t, ξ_c; taper::Bool=true, nx::Int=4001, xmax=nothing)
    xm = isnothing(xmax) ? (taper ? ξ_c * Lx : 8Lx) : xmax
    xs = range(-xm, xm; length=nx)
    w = fill(step(xs), nx); w[1] = w[end] = step(xs) / 2
    return [sum(w[i] * (kφ(profile(sqrt((xs[i] / Lx)^2 + (y / Ly)^2), ξ_t, ξ_c; taper=taper)) - kb) for i in 1:nx)
            for y in ys]
end

"J of `eq:focal_general` from F(ξ) = Re k − k₀: J = −∫F'(|u|)/|u| du, numerically."
function thin_lens_J(kφ, kb, ξ_t, ξ_c; taper::Bool=true, n::Int=20001)
    um = taper ? ξ_c : 10.0
    us = range(1e-6, um; length=n)
    F(u) = real(kφ(profile(u, ξ_t, ξ_c; taper=taper))) - kb
    d = 1e-5
    vals = [-(F(u + d) - F(max(u - d, 0.0))) / ((u + d) - max(u - d, 0.0)) / u for u in us]
    w = fill(step(us), n); w[1] = w[end] = step(us) / 2
    return 2sum(w .* vals)
end

"Thin-lens focal length of the Gaussian island, weak loading (`eq:focal_gaussian`)."
thin_lens_gaussian(h::HostSpec, ω, load, χ, Lx, Ly) =
    k0(h, ω) * Ly^2 * dD0(h, k0(h, ω), ω) / (sqrt(2π) * Lx * ω^2 * load * real(χ))

"""
    ray_focus(kφ, kb, Lx, Ly, ξ_t, ξ_c; taper=true, y0=0.02, ds=Lx/400)

Port of `aniso_island_rays.py`: a paraxial ray launched at y = y0·L_y from
x = −6L_x (−ξ_cL_x − L_x with the taper) through n = Re k(x, y); RK4 in arc
length. Converging lens: crossing of y = 0; diverging: virtual focus from
the exit ray at x = 6L_x. Distance from the island centre.
"""
function ray_focus(kφ, kb, Lx, Ly, ξ_t, ξ_c; taper::Bool=true, y0::Real=0.02, ds=Lx / 400)
    dφ = 1e-6
    function rhs(z)
        x, y, px, py = z
        ξ = sqrt((x / Lx)^2 + (y / Ly)^2)
        φ = profile(ξ, ξ_t, ξ_c; taper=taper)
        # ∇ Re k = (dRe k/dφ)(dφ/dξ)∇ξ
        dkdφ = real(kφ(min(φ + dφ, 1.0)) - kφ(max(φ - dφ, 0.0))) / (min(φ + dφ, 1.0) - max(φ - dφ, 0.0))
        dφdξ = (profile(ξ + 1e-7, ξ_t, ξ_c; taper=taper) - profile(max(ξ - 1e-7, 0.0), ξ_t, ξ_c; taper=taper)) /
               (ξ + 1e-7 - max(ξ - 1e-7, 0.0))
        gξ = ξ > 0 ? (x / (Lx^2 * ξ), y / (Ly^2 * ξ)) : (0.0, 0.0)
        g = dkdφ * dφdξ
        pn = hypot(px, py)
        return (px / pn, py / pn, g * gξ[1], g * gξ[2])
    end
    xstart = taper ? -(ξ_c * Lx + Lx) : -6Lx
    z = (xstart, y0 * Ly, kb, 0.0)
    add(a, b, s) = a .+ s .* b
    xexit = taper ? ξ_c * Lx + Lx : 6Lx
    for _ in 1:10_000_000
        k1 = rhs(z); k2 = rhs(add(z, k1, ds / 2)); k3 = rhs(add(z, k2, ds / 2)); k4 = rhs(add(z, k3, ds))
        zn = z .+ (ds / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        if zn[2] < 0 && z[2] >= 0      # converging: crossing
            return z[1] - z[2] * (zn[1] - z[1]) / (zn[2] - z[2])
        end
        if zn[1] >= xexit && zn[4] > 0   # diverging: exit ray going outwards, virtual focus
            return zn[1] - zn[2] * zn[3] / zn[4]
        end
        zn[1] > 1e5 && return NaN
        z = zn
    end
    return NaN
end

# ─── Spectral tools: FFT, parabolic equation, back-propagation ──────────────

"In-place radix-2 FFT (length a power of 2), sign −1 (forward)."
function fft!(a::Vector{ComplexF64}; sign::Int=-1)
    n = length(a)
    j = 1
    for i in 1:n
        if i < j
            a[i], a[j] = a[j], a[i]
        end
        m = n >> 1
        while m >= 1 && j > m
            j -= m; m >>= 1
        end
        j += m
    end
    len = 2
    while len <= n
        w = cis(sign * 2π / len)
        for s in 1:len:n
            wk = one(ComplexF64)
            for k in 0:(len ÷ 2 - 1)
                u = a[s + k]; v = a[s + k + len ÷ 2] * wk
                a[s + k] = u + v; a[s + k + len ÷ 2] = u - v
                wk *= w
            end
        end
        len <<= 1
    end
    return a
end
ifft!(a::Vector{ComplexF64}) = (fft!(a; sign=1); a ./= length(a); a)

"Transverse grid of the spectral tools: y ∈ [−Yp, Yp), Ny points, wavenumbers."
function ygrid(Yp::Real, Ny::Int)
    dy = 2Yp / Ny
    ys = [-Yp + (j - 1) * dy for j in 1:Ny]
    kys = [2π / (2Yp) * (j - 1 <= Ny ÷ 2 - 1 ? j - 1 : j - 1 - Ny) for j in 1:Ny]
    return ys, kys
end

"""
    pe_march(kφ, kb, Lx, Ly, ξ_t, ξ_c, xs_out; Yp=200, Ny=8192, dx=0.025)

Wide-angle split-step (Strang) parabolic equation for ψ = η/e^{ik₀x}:
free step e^{i(√(k₀² − k_y²) − k₀)dx} (evanescent k_y decay), medium step
e^{i(k(x, y) − k₀)dx} with the exact local root. Starts at x = −ξ_cL_x with
ψ = 1; returns (ys, [ψ(x, ys) for x in xs_out]) on the full y grid.
"""
function pe_march(kφ, kb, Lx, Ly, ξ_t, ξ_c, xs_out; Yp=200.0, Ny::Int=8192, dx=0.025)
    ys, kys = ygrid(Yp, Ny)
    prop = [cis((sqrt(complex(kb^2 - ky^2)) - kb) * dx / 2) for ky in kys]
    ψ = ones(ComplexF64, Ny)
    x = -ξ_c * Lx
    out = Vector{Vector{ComplexF64}}()
    for xo in sort(collect(xs_out))
        nstep = max(0, round(Int, (xo - x) / dx))
        for _ in 1:nstep
            fft!(ψ); ψ .*= prop; ifft!(ψ)
            xm = x + dx / 2
            for j in 1:Ny
                ξ = sqrt((xm / Lx)^2 + (ys[j] / Ly)^2)
                ξ < ξ_c && (ψ[j] *= cis((kφ(profile(ξ, ξ_t, ξ_c)) - kb) * dx))
            end
            fft!(ψ); ψ .*= prop; ifft!(ψ)
            x += dx
        end
        # remaining fraction: exact free propagation (outside the island)
        r = xo - x
        if abs(r) > 1e-12
            fft!(ψ); ψ .*= [cis((sqrt(complex(kb^2 - ky^2)) - kb) * r) for ky in kys]; ifft!(ψ); x = xo
        end
        push!(out, copy(ψ))
    end
    return ys, out
end

"""
    backprop(yh, Th, xm, kb; Yp=200, Ny=8192)

Back-propagation of a half-line record T(y), y = yh ≥ 0 (mirrored, T − 1
set to 0 beyond max(yh)), from x = xm to x = 0 with the bare propagator
e^{−i(√(k² − k_y²) − k)xm} for |k_y| < k (evanescent part dropped).
Returns (ys, T₀(ys)) on the full grid.
"""
function backprop(yh::AbstractVector, Th::AbstractVector, xm::Real, kb::Real; Yp=200.0, Ny::Int=8192)
    ys, kys = ygrid(Yp, Ny)
    s = zeros(ComplexF64, Ny)
    ymax = maximum(yh)
    for j in 1:Ny
        a = abs(ys[j])
        a <= ymax || continue
        i = clamp(searchsortedlast(yh, a), 1, length(yh) - 1)
        t = (a - yh[i]) / (yh[i + 1] - yh[i])
        s[j] = (1 - t) * (Th[i] - 1) + t * (Th[i + 1] - 1)
    end
    fft!(s)
    for j in 1:Ny
        ky = kys[j]
        s[j] = abs(ky) < kb ? s[j] * cis(-(sqrt(kb^2 - ky^2) - kb) * xm) : 0.0im
    end
    ifft!(s)
    return ys, 1 .+ s
end

"""
    curvature_fit(ys, T, k, ywin) -> (f, Φ2, A0, resid)

Fit of the unwrapped phase of T on |y| ≤ ywin by a + b y² + c y⁴:
Φ''(0) = 2b, f = −k/Φ''(0); A0 = −ln|T(0)|.
"""
function curvature_fit(ys::AbstractVector, T::AbstractVector, k::Real, ywin::Real)
    idx = findall(y -> abs(y) <= ywin, ys)
    y = ys[idx]
    i0 = argmin(abs.(ys))
    ph = angle.(T[idx] ./ T[i0])
    M = hcat(ones(length(y)), y .^ 2, y .^ 4)
    c = M \ ph
    Φ2 = 2c[2]
    return (f=-k / Φ2, Φ2=Φ2, A0=-log(abs(T[i0])), resid=norm(M * c - ph) / max(norm(ph), eps()))
end

end
