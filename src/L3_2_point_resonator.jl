"""
# L3-2 — point resonator on the 3D plate (port of the legacy VT3D_B, set P3)

## Purpose

Matrix row L3.2 (Phase 4, §6.5). A single point resonator attached to the
C_ρ > 0 Kirchhoff–Love plate in a 3D channel. The plate Green function is
finite at the source (η − η(x_r) ~ c r² ln r, c = f/(8πC_ρ)), so the
driving-point response and the local field converge under local refinement
without a finite footprint. Port of `code/scripts/VT3D_B_point.jl`,
`VT3D_B_winkler.jl` and `VT3D_B_nearfield.jl` (LRH folder, prereg
VT-3D-CONV-B v2 + A1), with the frozen parameter set P3
(`deliverables/06_Numerical_Verification/data/VT3D_B/params_VT3D_B.toml`).

## Problem (set P3)

Channel of width W = 2 m and depth H = 10 m, walls without flux. Along x: a
damping zone of 4λ₀, a gap of λ₀/2, the plate (4λ_p long, the full width,
free lateral edges), a gap and a zone. The plate is LRH Table 1 with
C_ρ = 0.05g, m_ρ = 0.01, ν = 0; the open water carries a free surface κ.
Incident wave (S1, S3, S4): velocity inlet and the damping-zone forcing of
VT-3D-CONV-A (η₀ = 0.01, μ₀ = 2.5, βₕ = 0.5). One resonator at the plate centre
x_r, M_ρ = 0.01 m³, K = Mω_r², ω_r = 10. φ, κ, η Q2; plate penalty γ = 6.

Mesh: the far field is fixed (Δx ≤ λ₀/6.4 in open water, λ_p/8 on the plate,
Δy ≤ W/7 across the channel, nz = 8 graded with top cell 0.2/k_p). Towards
x_r (and y = 0) the cells grow from h_min by a ratio 1.5. Level L1, L2, L3:
h_min = λ_p/12, λ_p/24, λ_p/48. x_r and the plate and zone edges are nodes.

C_ρ variant (C_ρ = 0.005g, 5e-4g): every length of the plate and the mesh
rules scale with λ_p, and so does the channel width (W = 2 m·λ_p/λ_p,P3,
`scale_W`), so that the meshes have the DOFs of P3 at every level.

## Solves of one case (C_ρ, level)

Bare plate at ω_r (one factorisation, several right-hand sides): S1 incident
wave; S2 unit point load f δ(x − x′) at x′ = x_r and, for P3, at the edge
point x_r + (h_min/2, 0) and the interior point x_r + (h_min/3, h_min/3).
Resonator at ω_r, ζ = 0, incident wave: S3 at x_r and, for P3, at the edge
and interior points. For P3 only: S1_09, S2_09 (bare plate at 0.9ω_r, one
factorisation) and S4 (resonator, ζ = 0.02, 0.9ω_r); at L2, PG (S2 with the
penalty γ h_min/h_far). Static Winkler limit (D-W): the same plate alone,
ω = 0, square [−10l_c, 10l_c]², point load at the centre (`winkler_level`).

The point load is assembled from the package-built `DiracDelta` of a
`ResonatorDomain` and added to the right-hand side; HydroElasticFEM's `rhs_fn`
only gives distributed loads (legacy A1 item 3).

Each case is run in parts (`run_case(p; part)`), one process per part, so
that each process stays under the 4 GB cap.

## Reproduce

    julia --project -O1 scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl --list
"""
module L3_2

using Gridap
using Parameters
using Printf
using LinearAlgebra: norm
import SparseArrays
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

using ..References: HostSpec, k_open, k0, green_2d_integral
using ..Solvers: MetisLUSolver, factorize, solve_rhs, factor_stats
import ..L3_0

export L3_2_Params, VARIABLE, fixed_hash, run_case, winkler_level, layout, case_constants, ref_points, setup_counts

"Parameters of one L3-2 case. Fixed fields are set P3 (params_VT3D_B.toml) unless marked."
@with_kw struct L3_2_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    level::String = "L1"                  # L1, L2, L3: h_min = λ_p/12, /24, /48
    C_over_g::Float64 = 0.05              # C_ρ/g: 0.05 (P3), 0.005, 5e-4 (matrix variant)

    # ── Fixed: plate and fluid (P3) ──────────────────────────────────────────
    g::Float64 = 9.80665
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    W::Float64 = 2.0
    hb::Float64 = 0.01
    ρb::Float64 = 1000.0                  # m_ρ = ρb hb/ρ = 0.01
    ν::Float64 = 0.0

    # ── Fixed: layout and incident wave (P3) ─────────────────────────────────
    zone_over_λ0::Float64 = 4.0
    gap_over_λ0::Float64 = 0.5
    plate_over_λp::Float64 = 4.0
    η₀::Float64 = 0.01
    μ₀::Float64 = 2.5
    βₕ::Float64 = 0.5
    order::Int = 2
    γ::Float64 = 6.0

    # ── Fixed: mesh (P3) ──────────────────────────────────────────────────────
    dx_open_over_λ0::Float64 = 6.4
    dx_plate_over_λp::Float64 = 8.0
    ny_base::Int = 7
    scale_W::Bool = true                  # added: W ∝ λ_p for the C_ρ variants (W/λ_p as in P3; inactive for P3)
    nz::Int = 8
    top_k::Float64 = 0.2
    ratio::Float64 = 1.5

    # ── Fixed: resonator and load (P3) ────────────────────────────────────────
    M_ρ::Float64 = 0.01
    ω_r::Float64 = 10.0
    ζ4::Float64 = 0.02
    ω4_over_ωr::Float64 = 0.9
    f::Float64 = 1.0

    # ── Fixed: near field (P3) ────────────────────────────────────────────────
    R_over_λp::Float64 = 0.25
    n_angles::Int = 24
    n_radii::Int = 40
    rmin_over_hmin::Float64 = 3.0
end

const VARIABLE = (:level, :C_over_g)
const HMIN = Dict("L1" => 12.0, "L2" => 24.0, "L3" => 48.0)

function fixed_hash(p::L3_2_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L3_2_Params) if !(f in VARIABLE))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

is_P3(p::L3_2_Params) = p.C_over_g == 0.05

host(p) = HostSpec(g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.ρb * p.hb / p.ρ, C_ρ=p.C_over_g * p.g)

"Plate wavenumber of set P3 (C_ρ = 0.05g) at ω_r."
kp_P3(p) = k0(HostSpec(g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.ρb * p.hb / p.ρ, C_ρ=0.05 * p.g), p.ω_r)

"Derived constants of case `p` (W: channel width, = P3 W·λ_p/λ_p,P3 when `scale_W`)."
function case_constants(p::L3_2_Params)
    C_ρ = p.C_over_g * p.g
    m_ρ = p.ρb * p.hb / p.ρ
    kk = k_open(p.ω_r, p.H, p.g)
    kp = k0(host(p), p.ω_r)
    M = p.M_ρ * p.ρ
    W = p.scale_W ? p.W * kp_P3(p) / kp : p.W
    return (; W, C_ρ, m_ρ, E=12 * C_ρ * p.ρ / p.hb^3, k0=kk, kp, λ0=2π / kk, λp=2π / kp, M, K=M * p.ω_r^2,
            ω4=p.ω4_over_ωr * p.ω_r, R=p.R_over_λp * 2π / kp)
end

# ─── Mesh ───────────────────────────────────────────────────────────────────

"Cell sizes on [0, len] from 0: h_min·ratio^j up to h_far, then uniform ≤ h_far (legacy `graded_sizes`)."
function graded_sizes(len, hmin, hfar, ratio)
    sizes = Float64[]; s = 0.0; h = hmin
    while h < hfar && s + h < len - 1e-12
        push!(sizes, h); s += h; h *= ratio
    end
    rem = len - s
    if rem > 1e-12
        n = max(1, ceil(Int, rem / hfar - 1e-9))
        if n == 1 && !isempty(sizes) && rem < 0.5 * sizes[end]
            sizes[end] += rem
        else
            append!(sizes, fill(rem / n, n))
        end
    end
    return sizes
end

uniform_sizes(len, hmax) = (n = max(1, ceil(Int, len / hmax - 1e-9)); fill(len / n, n))

"Node vectors and layout of case `p` at `level` (legacy `b_layout`)."
function layout(p::L3_2_Params; level::String=p.level)
    c = case_constants(p)
    hmin = c.λp / HMIN[level]
    Ld, gap, Lpl = p.zone_over_λ0 * c.λ0, p.gap_over_λ0 * c.λ0, p.plate_over_λp * c.λp
    dx0, dxp = c.λ0 / p.dx_open_over_λ0, c.λp / p.dx_plate_over_λp
    W = c.W
    dyf = W / p.ny_base
    half = graded_sizes(Lpl / 2, hmin, dxp, p.ratio)
    sx = vcat(uniform_sizes(Ld, dx0), uniform_sizes(gap, dx0), reverse(half), half,
              uniform_sizes(gap, dx0), uniform_sizes(Ld, dx0))
    xs = cumsum([0.0; sx])
    hy = graded_sizes(W / 2, hmin, dyf, p.ratio)
    ys = cumsum([-W / 2; reverse(hy); hy]); ys[end] = W / 2
    xp0 = Ld + gap; xp1 = xp0 + Lpl; xr = xp0 + Lpl / 2; LΩ = xs[end]
    for e in (Ld, xp0, xr, xp1, LΩ - Ld)
        minimum(abs.(xs .- e)) <= 1e-10 * LΩ || error("layout point x = $e is not a node")
    end
    minimum(abs.(ys)) <= 1e-12 || error("y = 0 is not a node")
    return (; W, level, hmin, Ld, gap, Lpl, xp0, xp1, xr, LΩ, xs, ys, dx0, dxp, dyf, hfar=dxp,
            nx=length(xs) - 1, ny=length(ys) - 1, nz=p.nz, top=p.top_k / c.kp)
end

"Piecewise-linear map from the uniform partition to `nodes`."
function node_map(nodes)
    a = nodes[1]; n = length(nodes) - 1; Lr = nodes[end] - a
    return function (s)
        t = clamp((s - a) / Lr * n, 0.0, Float64(n))
        i = clamp(floor(Int, t), 0, n - 1)
        return nodes[i + 1] + (t - i) * (nodes[i + 2] - nodes[i + 1])
    end
end

"Attachment points: node x_r, edge x_r + (h_min/2, 0), interior x_r + (h_min/3, h_min/3)."
attach_points(lay) = (node=(lay.xr, 0.0), edge=(lay.xr + lay.hmin / 2, 0.0),
                      interior=(lay.xr + lay.hmin / 3, lay.hmin / 3))

"Tank with optional resonator point `xres` (δ1) and load points `loads` (symbol => (x, y))."
function tank(p, lay; xres=nothing, loads=Pair{Symbol,NTuple{2,Float64}}[])
    fx, fy = node_map(lay.xs), node_map(lay.ys)
    fz = L3_0.zmap(p.H, lay.nz, lay.top)
    cart = G.CartesianDomain(mins=(0.0, -lay.W / 2, 0.0), maxs=(lay.LΩ, lay.W / 2, p.H), parts=(lay.nx, lay.ny, lay.nz),
                             map=x -> VectorValue(fx(x[1]), fy(x[2]), fz(x[3])))
    rds = G.ResonatorDomain[]
    isnothing(xres) || push!(rds, G.ResonatorDomain(location=[xres[1], xres[2], p.H], delta_symbol=:δ1))
    for (s, xy) in loads
        push!(rds, G.ResonatorDomain(location=[xy[1], xy[2], p.H], delta_symbol=s))
    end
    return G.TankDomain(cart;
        structure_domains=[G.StructureDomain(L=lay.Lpl, W=lay.W, x₀=[lay.xp0, -lay.W / 2, p.H], domain_symbol=:Γ_plate)],
        damping_zones=[G.DampingZone(L=lay.Ld, x₀=[0.0, -lay.W / 2, p.H], domain_symbol=:Γ_d_in),
                       G.DampingZone(L=lay.Ld, x₀=[lay.LΩ - lay.Ld, -lay.W / 2, p.H], domain_symbol=:Γ_d_out)],
        resonator_domains=rds)
end

"HydroElasticFEM entities (legacy `b_physics`): fluid with inlet and zones, free surface, plate, resonator."
function physics(p, lay, ω; incident::Bool, resonator=nothing, γ=p.γ)
    c = case_constants(p)
    k = k_open(ω, p.H, p.g); η₀ = incident ? p.η₀ : 0.0; H = p.H
    Ld = lay.Ld; xd_out = lay.LΩ - Ld
    μ1p_in(x) = p.μ₀ * (1 - cos(π / 2 * clamp((Ld - x[1]) / Ld, 0.0, 1.0)))
    μ1p_out(x) = p.μ₀ * (1 - cos(π / 2 * clamp((x[1] - xd_out) / Ld, 0.0, 1.0)))
    μ1_in(x) = μ1p_in(x) / p.βₕ;  μ1_out(x) = μ1p_out(x) / p.βₕ
    μ2_in(x) = μ1p_in(x) * k;     μ2_out(x) = μ1p_out(x) * k
    η_in(x) = η₀ * exp(im * k * x[1])
    vz_in(x) = -im * ω * η₀ * exp(im * k * x[1])
    u_in(x) = η₀ * ω * cosh(k * x[3]) / sinh(k * H) * exp(im * k * x[1])
    w_in(x) = -im * η₀ * ω * sinh(k * x[3]) / sinh(k * H) * exp(im * k * x[1])
    v_in(x) = VectorValue(u_in(x), 0.0 + 0.0im, w_in(x))
    zero_c(x) = 0.0 + 0.0im
    CT = Vector{ComplexF64}
    fluid = P.PotentialFlow(ρw=p.ρ, g=p.g, dim=3, fe=PH.FESpaceConfig(order=p.order, vector_type=CT),
        space_domain_symbol=:Ω,
        boundary_conditions=P.AbstractPotentialFlowBC[
            P.PrescribedInletPotentialBC(domain=:dΓin, forcing=v_in, quantity=:velocity),
            P.DampingZoneBC(domain=:dΓd_1, μ₁=μ1_in, μ₂=μ2_in, η_in=η_in, vz_in=vz_in, βₕ=p.βₕ),
            P.DampingZoneBC(domain=:dΓd_2, μ₁=μ1_out, μ₂=μ2_out, η_in=zero_c, vz_in=zero_c, βₕ=p.βₕ)])
    fs = P.FreeSurface(ρw=p.ρ, g=p.g, βₕ=p.βₕ, dim=3, fe=PH.FESpaceConfig(order=p.order, vector_type=CT),
                       space_domain_symbol=:Γκ)
    plate = P.KirchhoffLovePlate(E=c.E, ν=p.ν, hb=p.hb, ρ=p.ρ, ρb=p.ρb, g=p.g, ambient_dim=3, manifold_dim=2,
                                 fe=PH.FESpaceConfig(order=p.order, vector_type=CT, γ=γ), space_domain_symbol=:Γη)
    ents = P.PhysicsParameters[fluid, fs, plate]
    if !isnothing(resonator)
        ra = P.resonator_array(1, resonator.M, resonator.K, resonator.C; host_domain_symbol=plate.space_domain_symbol,
                               delta_domain_symbols=[:δ1], ρw=p.ρ,
                               fe=PH.FESpaceConfig(space_type=VectorValue{1,Float64}, vector_type=CT))
        na = length(P.attached_resonators(ra, plate))
        na == 1 || error("resonator not attached to the plate ($na attached)")
        push!(ents, ra)
    end
    return ents
end

eval_on_plate(fh, pts) = L3_0.eval_on_plate(fh, pts)

"Fixed reference points: 4×4 Gauss points of the L3 plate cells within r ≤ R of x_r (offsets, weights)."
function ref_points(p::L3_2_Params)
    c = case_constants(p); lay3 = layout(p; level="L3")
    gx = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
    gw = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
    pts = NTuple{2,Float64}[]; wts = Float64[]
    for i in 1:lay3.nx, j in 1:lay3.ny
        x0, x1 = lay3.xs[i], lay3.xs[i + 1]; y0, y1 = lay3.ys[j], lay3.ys[j + 1]
        (x1 < lay3.xr - c.R || x0 > lay3.xr + c.R || y1 < -c.R || y0 > c.R) && continue
        for a in 1:4, b in 1:4
            x = x0 + (gx[a] + 1) * (x1 - x0) / 2; y = y0 + (gx[b] + 1) * (y1 - y0) / 2
            hypot(x - lay3.xr, y) <= c.R || continue
            push!(pts, (x - lay3.xr, y)); push!(wts, gw[a] * gw[b] * (x1 - x0) * (y1 - y0) / 4)
        end
    end
    return pts, wts
end

"Cells and DOF estimate per level (no assembly)."
function setup_counts(p::L3_2_Params)
    lay = layout(p)
    o = p.order
    nφ = (o * lay.nx + 1) * (o * lay.ny + 1) * (o * lay.nz + 1)
    nsurf = (o * lay.nx + 1) * (o * lay.ny + 1)
    return (; lay.nx, lay.ny, lay.nz, N_estimate=nφ + nsurf, lay.hmin)
end

# ─── Solves ─────────────────────────────────────────────────────────────────

"Azimuthal means of η about `xy` at log-spaced radii [r_min, R]."
function polar_mean(p, lay, ηh, xy)
    c = case_constants(p)
    rmin = p.rmin_over_hmin * lay.hmin
    rmin < c.R || return (r=Float64[], m=ComplexF64[])
    rs = exp.(range(log(rmin), log(c.R); length=p.n_radii))
    θs = [2π * (m - 1) / p.n_angles for m in 1:p.n_angles]
    pts = [Point(xy[1] + r * cos(θ), xy[2] + r * sin(θ), p.H) for r in rs for θ in θs]
    v = ComplexF64.(eval_on_plate(ηh, pts))
    return (r=collect(rs), m=[sum(v[(i - 1) * p.n_angles + 1:i * p.n_angles]) / p.n_angles for i in eachindex(rs)])
end

"η and its surface gradient at the fixed reference set (offsets from x_r)."
function near_field(p, lay, ηh, ref)
    pts = [Point(lay.xr + q[1], q[2], p.H) for q in ref]
    η = ComplexF64.(eval_on_plate(ηh, pts))
    method = "gridap_gradient"
    g = try
        gg = eval_on_plate(∇(ηh), pts)
        (ComplexF64[v[1] for v in gg], ComplexF64[v[2] for v in gg])
    catch
        method = "central_difference_1e-6"
        δ = 1e-6
        sh(dx, dy) = ComplexF64.(eval_on_plate(ηh, [Point(q[1] + dx, q[2] + dy, q[3]) for q in pts]))
        ((sh(δ, 0.0) .- sh(-δ, 0.0)) ./ 2δ, (sh(0.0, δ) .- sh(0.0, -δ)) ./ 2δ)
    end
    return (; η, gx=g[1], gy=g[2], method)
end

"Field index helpers for the free DOF vector of `problem`."
function blocks(problem)
    X = S.get_trial_fe_space(problem)
    fmap = S.get_field_map(problem)
    offs = cumsum([0; [num_free_dofs(Xi) for Xi in X.spaces]])
    rng(s) = (offs[fmap[s]] + 1):offs[fmap[s] + 1]
    return (; X, fmap, rng)
end

"Assemble the point-load vector f·δ_s(v_η) from the package DiracDelta `s`."
function point_load(problem, s::Symbol, f)
    Y = S.get_test_fe_space(problem)
    iη = S.get_field_map(problem)[:η]
    δ = S.get_integration_domains(problem)[s]
    return f .* assemble_vector(y -> δ(y[iη]), Y)
end

release!(ns) = (finalize(ns.factors); GC.gc(); nothing)

"""
Bare plate at `ω`: one METIS factorisation; S1 (incident) and S2 at each load
point. Returns point values, the S1/S2 fields needed downstream, and checks.
"""
function bare_solves(p, lay, ω, loadpts::Vector{Pair{Symbol,NTuple{2,Float64}}}; rec, tag, γ=p.γ,
                     check_incidence::Bool=false, keep_fields::Bool=false)
    c = case_constants(p)
    solver = MetisLUSolver()
    tk = tank(p, lay; loads=loadpts)
    problem = L3_0.phase!(() -> S.build_problem(tk, physics(p, lay, ω; incident=true, γ=γ),
                                                PH.FreqDomainConfig(ω=ω, solver=solver)), rec, "$(tag)_assembly")
    op = S.get_fe_operator(problem)
    A, b_inc = get_matrix(op), copy(get_vector(op))
    bl = blocks(problem)
    rη = bl.rng(:η)
    out = Dict{String,Any}()
    out["N"] = size(A, 1); out["nnzA"] = SparseArrays.nnz(A)
    if check_incidence   # the incident wave enters the right-hand side only
        pr0 = S.build_problem(tk, physics(p, lay, ω; incident=false, γ=γ), PH.FreqDomainConfig(ω=ω))
        op0 = S.get_fe_operator(pr0)
        out["incidence_matrix_diff"] = norm(get_matrix(op0) - A, 1) / norm(A, 1)
        out["incidence_free_rhs_norm"] = norm(get_vector(op0))
        pr0 = nothing; op0 = nothing
    end
    bpts = Dict(s => point_load(problem, s, 1.0) for (s, _) in loadpts)
    # load-sign checks (legacy A1 item 4)
    s1 = first(loadpts).first
    out["load_sum_eta"] = sum(bpts[s1][rη])
    out["load_sum_abs_other"] = sum(abs, bpts[s1]) - sum(abs, bpts[s1][rη])
    Aηη1 = sum(A[rη, rη] * ones(length(rη)))
    out["A_etaeta_one_rel"] = abs(Aηη1 / ((p.g - ω^2 * c.m_ρ) * lay.Lpl * lay.W) - 1)
    out["A_etaeta_one_re"] = real(Aηη1)
    problem = nothing; op = nothing; GC.gc(true)   # free the assembly before the factorisation
    ns = L3_0.phase!(() -> factorize(solver, A), rec, "$(tag)_factorize")
    st = factor_stats(ns)
    out["lunz"] = st.lunz; out["umf_peak_gb"] = st.peak_gb; out["ordering"] = String(st.ordering)
    B = hcat(b_inc, (p.f .* bpts[s] for (s, _) in loadpts)...)
    Xs = L3_0.phase!(() -> solve_rhs(ns, B), rec, "$(tag)_solve")
    out["rel_residual"] = maximum(norm(A * Xs[:, j] - B[:, j]) / norm(B[:, j]) for j in axes(B, 2))
    release!(ns)
    fields = Dict{String,Any}()
    allpts = [xy for (_, xy) in loadpts]
    η1 = FEFunction(bl.X, Xs[:, 1])[bl.fmap[:η]]
    v1 = ComplexF64.(eval_on_plate(η1, [Point(xy[1], xy[2], p.H) for xy in allpts]))
    for (i, (s, xy)) in enumerate(loadpts)
        out["S1_eta_$(s)"] = v1[i]
        η2 = FEFunction(bl.X, Xs[:, 1 + i])[bl.fmap[:η]]
        out["S2_eta_$(s)"] = ComplexF64(eval_on_plate(η2, [Point(xy[1], xy[2], p.H)])[1])
        keep_fields && i == 1 && (fields["S2"] = η2)
    end
    out["S1_eta_max_plate"] = maximum(abs, ComplexF64.(eval_on_plate(η1,
        [Point(x, 0.0, p.H) for x in range(lay.xp0 + 1e-6, lay.xp1 - 1e-6; length=101)])))
    return out, fields
end

"Resonator at `xy` with damping C at `ω`, incident wave. Returns η(x_att), q, f_h and the η field."
function resonator_solve(p, lay, ω, xy, C; rec, tag)
    c = case_constants(p)
    solver = MetisLUSolver()
    tk = tank(p, lay; xres=xy)
    problem = L3_0.phase!(() -> S.build_problem(tk, physics(p, lay, ω; incident=true, resonator=(M=c.M, K=c.K, C=C)),
                                                PH.FreqDomainConfig(ω=ω, solver=solver)), rec, "$(tag)_assembly")
    op = S.get_fe_operator(problem)
    A, b = get_matrix(op), get_vector(op)
    bl = blocks(problem)
    problem = nothing; op = nothing; GC.gc(true)   # free the assembly before the factorisation
    ns = L3_0.phase!(() -> factorize(solver, A), rec, "$(tag)_factorize")
    x = solve_rhs(ns, b)
    release!(ns)
    uh = FEFunction(bl.X, x)
    ηh = uh[bl.fmap[:η]]
    ηx = ComplexF64(eval_on_plate(ηh, [Point(xy[1], xy[2], p.H)])[1])
    q = ComplexF64(get_free_dof_values(uh[bl.fmap[:q1]])[1])
    out = Dict{String,Any}("eta" => ηx, "q" => q, "f" => (c.K / p.ρ - im * ω * C / p.ρ) * (q - ηx),
                           "rel_residual" => norm(A * x - b) / norm(b), "N" => size(A, 1))
    return out, ηh
end

"""
    run_case(p; part=:all)

Solves of case `p` (see the module docstring), split in parts so that each
part fits in one process under the 4 GB cap (one assembly and one
factorisation each): `:bare` (S1, S2 at ω_r, near field of S2), `:res_node`
(S3 at x_r, near field), `:res_edge`, `:res_interior` (S3 at the edge and
interior points), `:bare09` (S1_09, S2_09), `:S4`, `:PG` (L2), or `:all`. Returns a Dict record.
"""
function run_case(p::L3_2_Params; part::Symbol=:all)
    c = case_constants(p)
    lay = layout(p)
    rec = Dict{String,Any}()
    rec["nx"], rec["ny"], rec["nz"], rec["hmin"] = lay.nx, lay.ny, lay.nz, lay.hmin
    rec["lambda_p"], rec["lambda0"], rec["kp"], rec["R"], rec["C_rho"] = c.λp, c.λ0, c.kp, c.R, c.C_ρ
    rec["K"], rec["M"], rec["rho"], rec["W"] = c.K, c.M, p.ρ, lay.W
    rec["G_inf"] = green_2d_integral(host(p), 0.0, p.ω_r)     # infinite plate on the fluid, reported
    ap = attach_points(lay)
    full = is_P3(p)
    do_(x) = part == :all || part == x
    variants = full ? [:node, :edge, :interior] : [:node]
    loadpts = Pair{Symbol,NTuple{2,Float64}}[Symbol("δ_", v) => getfield(ap, v) for v in variants]

    if do_(:bare)   # bare plate at ω_r: S1, S2 at the attachment points (one factorisation)
        ref, wts = ref_points(p)
        rec["ref_weights"] = wts
        o, flds = bare_solves(p, lay, p.ω_r, loadpts; rec=rec, tag="bare", check_incidence=(p.level == "L1"),
                              keep_fields=true)
        merge!(rec, Dict("bare_" * k => v for (k, v) in o))
        pm = polar_mean(p, lay, flds["S2"], ap.node)
        rec["S2_polar_r"], rec["S2_polar_m"] = pm.r, pm.m
        nf = near_field(p, lay, flds["S2"], ref)
        rec["S2_nf_eta"], rec["S2_nf_gx"], rec["S2_nf_gy"] = nf.η, nf.gx, nf.gy
        flds = nothing; GC.gc()
    end
    # resonator at ω_r, ζ = 0, incident wave: S3
    vs = vcat(do_(:res_node) ? [:node] : Symbol[], (do_(:res_edge) && full) ? [:edge] : Symbol[],
              (do_(:res_interior) && full) ? [:interior] : Symbol[])
    for v in vs
        o, ηh = resonator_solve(p, lay, p.ω_r, getfield(ap, v), 0.0; rec=rec, tag="S3_$(v)")
        merge!(rec, Dict("S3_$(v)_" * k => val for (k, val) in o))
        if v == :node
            ref, wts = ref_points(p)
            rec["ref_weights"] = wts
            pm = polar_mean(p, lay, ηh, ap.node)
            rec["S3_polar_r"], rec["S3_polar_m"] = pm.r, pm.m
            nf = near_field(p, lay, ηh, ref)
            rec["S3_nf_eta"], rec["S3_nf_gx"], rec["S3_nf_gy"], rec["grad_method"] = nf.η, nf.gx, nf.gy, nf.method
        end
        ηh = nothing; GC.gc()
    end
    if do_(:bare09) && full   # bare plate at 0.9ω_r: S1_09, S2_09
        o, _ = bare_solves(p, lay, c.ω4, loadpts[1:1]; rec=rec, tag="bare09")
        merge!(rec, Dict("bare09_" * k => v for (k, v) in o))
        GC.gc()
    end
    if do_(:S4) && full       # resonator, ζ = 0.02, 0.9ω_r
        C4 = 2p.ζ4 * sqrt(c.K * c.M)
        o, _ = resonator_solve(p, lay, c.ω4, ap.node, C4; rec=rec, tag="S4")
        merge!(rec, Dict("S4_" * k => v for (k, v) in o))
        rec["S4_C"], rec["S4_omega"] = C4, c.ω4
        GC.gc()
    end
    if do_(:PG) && full       # penalty control at L2
        if p.level == "L2"
            γ = p.γ * lay.hmin / lay.hfar
            o, _ = bare_solves(p, lay, p.ω_r, loadpts[1:1]; rec=rec, tag="PG", γ=γ)
            merge!(rec, Dict("PG_" * k => v for (k, v) in o))
            rec["PG_gamma"] = γ
        end
    end
    rec["maxrss_final_gb"] = L3_0.maxrss_gb()
    return rec
end

# ─── D-W: static Winkler limit (plate only) ─────────────────────────────────

"""
    winkler_level(p, n_lc; half_width=10, hfar=1)

C_ρ∇⁴w + g w = f δ(x) on [−10l_c, 10l_c]², free edges, ω = 0, with
h_min = l_c/`n_lc` graded (ratio 1.5) to h_far = l_c (legacy D-W, package
route: the same `KirchhoffLovePlate` alone on Γη of a one-layer tank, a
field-free ring of width l_c around it). Returns w(0), polar means on
[3h_min, 2l_c], residual and the load-sign check.
"""
function winkler_level(p::L3_2_Params, n_lc::Real; half_width=10.0, hfar_over_lc=1.0, rfit_max_over_lc=2.0)
    C = p.C_over_g * p.g; g = p.g; lc = (C / g)^(1 / 4)
    hw = half_width * lc; hmin = lc / n_lc; hfar = hfar_over_lc * lc
    hs = graded_sizes(hw, hmin, hfar, p.ratio)
    nodes = cumsum([-hw; reverse(hs); hs]); nodes[end] = hw
    n = length(nodes) - 1
    ring = lc; Hd = 1.0
    fm = node_map([-hw - ring; nodes; hw + ring])
    cart = G.CartesianDomain(mins=(-hw - ring, -hw - ring, 0.0), maxs=(hw + ring, hw + ring, Hd), parts=(n + 2, n + 2, 1),
                             map=x -> VectorValue(fm(x[1]), fm(x[2]), x[3]))
    tk = G.TankDomain(cart;
        structure_domains=[G.StructureDomain(L=2hw, W=2hw, x₀=[-hw, -hw, Hd], domain_symbol=:Γ_plate)],
        resonator_domains=[G.ResonatorDomain(location=[0.0, 0.0, Hd], delta_symbol=:δload)])
    plate = P.KirchhoffLovePlate(E=12 * C * p.ρ / p.hb^3, ν=0.0, hb=p.hb, ρ=p.ρ, ρb=p.ρb, g=g, ambient_dim=3,
                                 manifold_dim=2, fe=PH.FESpaceConfig(order=2, vector_type=Vector{ComplexF64}, γ=p.γ),
                                 space_domain_symbol=:Γη)
    problem = S.build_problem(tk, P.PhysicsParameters[plate], PH.FreqDomainConfig(ω=0.0))
    A = get_matrix(S.get_fe_operator(problem))
    U = S.get_trial_fe_space(problem); V = S.get_test_fe_space(problem)
    δ = S.get_integration_domains(problem)[:δload]
    bpt = assemble_vector(y -> δ(y[1]), V)
    A1 = sum(A * ones(size(A, 2))); area = (2hw)^2
    b = p.f .* bpt
    ns = factorize(MetisLUSolver(), A)
    x = solve_rhs(ns, b)
    release!(ns)
    wh = FEFunction(U, x)[1]
    w0 = real(eval_on_plate(wh, [Point(0.0, 0.0, Hd)])[1])
    rs = exp.(range(log(p.rmin_over_hmin * hmin), log(rfit_max_over_lc * lc); length=p.n_radii))
    θs = [2π * (m - 1) / p.n_angles for m in 1:p.n_angles]
    wp = real.(eval_on_plate(wh, [Point(r * cos(θ), r * sin(θ), Hd) for r in rs for θ in θs]))
    m = [sum(wp[(i - 1) * p.n_angles + 1:i * p.n_angles]) / p.n_angles for i in eachindex(rs)]
    return Dict{String,Any}("n_lc" => n_lc, "lc" => lc, "hmin" => hmin, "ncells" => n, "ndofs" => num_free_dofs(U),
                            "w0" => w0, "w0_exact" => p.f / (8sqrt(g * C)), "c_pred" => p.f / (8π * C),
                            "polar_r" => collect(rs), "polar_m" => m, "rel_residual" => norm(A * x - b) / norm(b),
                            "load_sum" => sum(bpt), "A_one_rel" => abs(A1 / (g * area) - 1), "A_one_re" => real(A1))
end

end
