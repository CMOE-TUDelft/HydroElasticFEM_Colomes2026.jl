"""
# L3-1 — bare plate in 3D: oblique plane wave (port of the legacy VT3D_A, part A-D)

## Relation to the paper

The host symbol 𝒟₀(k, ω) (eq:host_symbol) depends on |k| only, so a plate wave
with wave vector (k_x, k_y) exists iff k_x² + k_y² = k₀². A line load
F e^{ik_y y} across the plate radiates two oblique waves; the residue
calculation in k_x gives

    η(x, y) = e^{ik_y y} F G_θ(x),   G_θ(x) = i Σₘ (kₘ/k_{x,m}) e^{ik_{x,m}|x|}/𝒟₀′(kₘ),
    k_{x,m} = √(kₘ² − k_y²),  Im k_{x,m} ≥ 0,

so the radiated amplitude is γ_θ = γ_r/cos θ (γ_r = 1/𝒟₀′(k₀)), the x-flux of
each wave is ¼ω𝒟₀′(k₀)|A|² cos θ, and the input power ½ω Im(F̄η) equals the
net outgoing flux. With k_y ≠ 0 every fluid evanescent mode decays at least
as fast as √(κ₁² + k_y²), much faster than κ₁ alone.

The legacy VT3D_A (code/scripts/VT3D_A_{dispersion,analyse}.jl, prereg
VT-3D-CONV-A v2 + A1) had normal incidence only (an m = 0 wave in a walled
channel). This port is the matrix row L3.1(a): periodic in y.

## Problem

Configuration A in 3D, set P1 (C_ρ = 0.49, m_ρ = 0.01, T_ρ = 0, H = 10, ν = 0):
a box [0, Lx] × [0, W] × [0, H] fully covered by the Kirchhoff–Love plate,
**periodic in y** with W = m λ_p/sin θ (m = 1), so k_y = k₀ sin θ is a periodic
mode. Load: F e^{ik_y y}/b on the patch |x − x_s| < b/2 (b = 2 cells), x_s a
node. Sponges at both x ends: `StructuralDampingZone`s with Kelvin–Voigt
τ = s/ω and a viscous dashpot (absorb all wavelengths), ramp ξ²; rigid walls
at x = 0, Lx. φ and η Q_p (p = 2), C/DG plate penalty γ = 6. Horizontal cells
h = λ_p/`epw` (in y: W/round(W/h)); vertical: `nz` cells graded towards the
surface with top cell k₀h_top = `top_k` (as L3-0).

Layouts (`layout_kind`):
- `:full` — the L3-0 target: sponges 4λ_p, gaps 5/κ₁, half-window 3λ_p
  (Lx = 8λ_p + 2·5/κ₁ + 6λ_p; 333×16×8 cells at ω = 10, θ = 30°, h = λ_p/8);
- `:short` — for the mesh study: sponges 2λ_p, gaps 5/√(κ₁² + k_y²), half-window 2λ_p.

## Observables (one factorisation, several right-hand sides)

- y-mode projection ηₘ(x) = W⁻¹∫ η e^{−ik_y y} dy (Gauss-5 per y cell) and the
  transverse residual (energy outside the mode);
- k_{x,h} from a wave fit (L1_1.fit_waves with roots k_{x,m}) on windows
  either side of the source; k_h = √(k_{x,h}² + k_y²);
- radiated amplitude γ_h = (A_out − A_in,other)/(iF sinc(k_x b/2)) against γ_θ;
- energy: P_in = ½ω Im(b̄·x)/W (exact discrete input power) against the net
  x-flux ¼ω𝒟₀′ cos θ Σ(|A_out|² − |A_in|²);
- reciprocity: two point loads (package `DiracDelta`), G(a,b) against G(b,a);
- residual of every solve.

## Reproduce

    julia --project -O1 scripts/L3-3D-flat-bottom/L3-1_oblique_plate.jl --list
"""
module L3_1

using Gridap
using Parameters
using Printf
using LinearAlgebra: norm, dot
import SparseArrays
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

using ..References: HostSpec, k0, kappa1, host_roots, dD0, gamma_r, added_mass, bare_frequency
using ..Solvers: MetisLUSolver, factorize, solve_rhs, factor_stats
import ..L1_1
import ..L3_0

export L3_1_Params, VARIABLE, fixed_hash, run_case, layout, setup_counts

"Parameters of one L3-1 case (set P1)."
@with_kw struct L3_1_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    θ_deg::Float64 = 30.0                 # incidence angle from the x axis
    ω::Float64 = 10.0
    epw::Int = 8                          # horizontal cells per λ_p
    nz::Int = 8
    layout_kind::Symbol = :full           # :full (L3-0 target) or :short (mesh study)

    # ── Fixed: host P1 ────────────────────────────────────────────────────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49
    hb::Float64 = 0.01
    ν::Float64 = 0.0

    # ── Fixed: forcing, sponges, discretisation ──────────────────────────────
    F::Float64 = 1.0
    source_elements::Int = 2
    m_periods::Int = 1                    # W = m λ_p / sin θ
    ny_normal::Int = 3                    # y cells at θ = 0 (W = ny_normal·h; Gridap needs ≥ 3 periodic cells)
    sponge_tau::Float64 = 3.0             # ωτ at the wall (Kelvin–Voigt)
    sponge_viscous::Float64 = 1.0         # dashpot loss factor (L1_1.viscous_sponges)
    sponge_power::Int = 2
    top_k::Float64 = 0.2
    order::Int = 2
    γ_penalty::Float64 = 6.0

    # ── Fixed: post-processing ────────────────────────────────────────────────
    n_modes::Int = 6
    x_sampling::Symbol = :nodes           # :nodes (cell vertices in x: a discrete Bloch wave is exactly e^{ik_h x} there) or :uniform
    n_xsamples::Int = 240
    probe_offset::Float64 = 1.0           # windows start this many λ_p from x_s
end

const VARIABLE = (:θ_deg, :ω, :epw, :nz, :layout_kind)

function fixed_hash(p::L3_1_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L3_1_Params) if !(f in VARIABLE))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

host(p) = HostSpec(g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.m_ρ, C_ρ=p.C_ρ)

cells(len, h) = max(1, ceil(Int, len / h - 1e-9))

"Geometry of case `p`: wavenumbers, W, cells, x_s, sponges, windows."
function layout(p::L3_1_Params)
    hs = host(p)
    kp = k0(hs, p.ω); λ = 2π / kp
    θ = deg2rad(p.θ_deg)
    ky = kp * sin(θ)
    he = λ / p.epw
    if p.θ_deg == 0
        ny = p.ny_normal; W = ny * he
    else
        W = p.m_periods * 2π / ky
        ny = max(2, round(Int, W / he))
    end
    hy = W / ny
    κ1 = kappa1(hs, p.ω)
    κeff = sqrt(κ1^2 + ky^2)
    if p.layout_kind == :full
        Ls, gap, half = 4λ, 5 / κ1, 3λ
    elseif p.layout_kind == :short
        Ls, gap, half = 2λ, 5 / κeff, 2λ
    else
        throw(ArgumentError("layout_kind must be :full or :short"))
    end
    nx = cells(2 * (Ls + gap + half), he)
    Lx = nx * he
    x_s = round((Lx / 2) / he) * he
    win_R = (x_s + p.probe_offset * λ, Lx - Ls - 0.25λ)
    win_L = (Ls + 0.25λ, x_s - p.probe_offset * λ)
    return (; kp, λ, θ, ky, kx=sqrt(kp^2 - ky^2), he, W, ny, hy, nx, Lx, x_s, b=p.source_elements * he, Ls, gap, half,
            κ1, κeff, win_R, win_L, nz=p.nz, top=p.top_k / kp)
end

"DOFs with y periodic: φ (p nx+1)(p ny)(p nz+1) + η (p nx+1)(p ny)."
function setup_counts(p::L3_1_Params)
    lay = layout(p); o = p.order
    N = (o * lay.nx + 1) * (o * lay.ny) * (o * p.nz + 1) + (o * lay.nx + 1) * (o * lay.ny)
    return (; lay.nx, lay.ny, nz=p.nz, N_estimate=N, W_over_λ=lay.W / lay.λ, Lx_over_λ=lay.Lx / lay.λ)
end

"Reciprocity load points (on the plate, away from the source and the sponges)."
recip_points(lay) = (a=(lay.x_s - 0.37lay.λ, 0.31lay.W), b=(lay.x_s + 0.71lay.λ, 0.58lay.W))

function tank(p, lay)
    fz = L3_0.zmap(p.H, p.nz, lay.top)
    ra, rb = recip_points(lay)
    G.TankDomain(L=lay.Lx, W=lay.W, H=p.H, nx=lay.nx, ny=lay.ny, nz=p.nz, is_periodic=(false, true, false),
                 map=x -> VectorValue(x[1], x[2], fz(x[3])),
                 structure_domains=[G.StructureDomain(L=lay.Lx, W=lay.W, x₀=[0.0, 0.0, p.H], domain_symbol=:Γs)],
                 resonator_domains=[G.ResonatorDomain(location=[ra[1], ra[2], p.H], delta_symbol=:δa),
                                    G.ResonatorDomain(location=[rb[1], rb[2], p.H], delta_symbol=:δb)])
end

function sponges(p, lay)
    k = lay.kp
    c = p.sponge_viscous * bare_frequency(host(p), k) * (p.m_ρ + added_mass(k, p.H))
    τ = p.sponge_tau / p.ω
    return [P.StructuralDampingZone(profile=P.ramp_profile(x_start=lay.Ls, x_end=0.0, power=p.sponge_power), cᵨ=c, τ=τ),
            P.StructuralDampingZone(profile=P.ramp_profile(x_start=lay.Lx - lay.Ls, x_end=lay.Lx, power=p.sponge_power),
                                    cᵨ=c, τ=τ)]
end

"Build the problem (oblique patch load in the RHS); returns problem and layout."
function build(p::L3_1_Params; lay=layout(p), solver=MetisLUSolver())
    fe(; kw...) = PH.FESpaceConfig(; order=p.order, vector_type=Vector{ComplexF64}, kw...)
    fluid = P.PotentialFlow(g=p.g, ρw=p.ρ, dim=3, fe=fe(), space_domain_symbol=:Ω)
    plate = P.KirchhoffLovePlate(E=12 * p.C_ρ * p.ρ / p.hb^3, ν=p.ν, hb=p.hb, ρ=p.ρ, ρb=p.m_ρ * p.ρ / p.hb,
                                 g=p.g, ambient_dim=3, manifold_dim=2, symbol=:η, fe=fe(γ=p.γ_penalty),
                                 space_domain_symbol=:Γη, damping_zones=sponges(p, lay))
    ky, xs, b, F = lay.ky, lay.x_s, lay.b, p.F
    load(x) = abs(x[1] - xs) < b / 2 ? complex(F / b) * exp(im * ky * x[2]) : zero(ComplexF64)
    field_order(fmap) = [s for (s, _) in sort!(collect(fmap._map); by=last)]
    rhs_fn = (ctx, fmap) -> [sym == :η ? load : 0.0 for sym in field_order(fmap)]
    problem = S.build_problem(tank(p, lay), P.PhysicsParameters[fluid, plate],
                              PH.FreqDomainConfig(ω=p.ω, solver=solver); rhs_fn=rhs_fn)
    return problem, lay
end

const GX5 = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
const GW5 = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]

"y-mode projection of η on x samples: ηₘ(x) and the transverse residual."
function project(ηh, xs, lay, H)
    yq = Float64[]; wq = Float64[]
    for j in 1:lay.ny
        y0 = (j - 1) * lay.hy
        append!(yq, y0 .+ (GX5 .+ 1) .* (lay.hy / 2)); append!(wq, GW5 .* (lay.hy / 2))
    end
    pts = [Point(x, y, H) for y in yq for x in xs]           # x fastest
    v = reshape(ComplexF64.(L3_0.eval_on_plate(ηh, pts)), length(xs), length(yq))
    e = exp.(-im * lay.ky .* yq)
    ηm = (v * (wq .* e)) ./ lay.W
    resid = v .- ηm .* transpose(conj.(e))
    tr = sqrt(sum(abs2.(resid) * wq) / sum(abs2.(v) * wq))
    return ηm, tr
end

"Roots in x: k_{x,m} = √(kₘ² − k_y²) with Im ≥ 0 (propagating root first, real)."
function roots_x(p, lay)
    r = host_roots(host(p), p.ω)
    rx = map(r) do k
        z = sqrt(complex(k^2 - lay.ky^2))
        imag(z) < 0 ? -z : z
    end
    rx[1] = complex(sqrt(real(r[1])^2 - lay.ky^2))
    return r, rx
end

"""
    run_case(p) -> Dict

Assemble, factorise (METIS), solve the oblique load and two point loads, and
return the observables (raw; checks are formed by the script).
"""
function run_case(p::L3_1_Params)
    rec = Dict{String,Any}()
    lay = layout(p)
    solver = MetisLUSolver()
    problem, _ = L3_0.phase!(() -> build(p; lay=lay, solver=solver), rec, "assembly")
    op = S.get_fe_operator(problem)
    A, b = get_matrix(op), copy(get_vector(op))
    X = S.get_trial_fe_space(problem)
    fmap = S.get_field_map(problem)
    Y = S.get_test_fe_space(problem)
    iη = fmap[:η]
    doms = S.get_integration_domains(problem)
    ba = assemble_vector(y -> doms[:δa](y[iη]), Y)
    bb = assemble_vector(y -> doms[:δb](y[iη]), Y)
    rec["N"] = size(A, 1); rec["nnzA"] = SparseArrays.nnz(A)
    rec["ndofs_fields"] = [num_free_dofs(Xi) for Xi in X.spaces]
    rec["sym_rel"] = norm(A - transpose(A), 1) / norm(A, 1)
    problem = nothing; op = nothing; GC.gc(true)
    ns = L3_0.phase!(() -> factorize(solver, A), rec, "factorize")
    st = factor_stats(ns)
    rec["lunz"] = st.lunz; rec["umf_peak_gb"] = st.peak_gb; rec["ordering"] = String(st.ordering)
    B = hcat(b, ba, bb)
    Xs = L3_0.phase!(() -> solve_rhs(ns, B), rec, "solve")
    rec["rel_residual"] = maximum(norm(A * Xs[:, j] - B[:, j]) / norm(B[:, j]) for j in 1:3)
    finalize(ns.factors); ns = nothing; A = nothing; GC.gc()
    x = Xs[:, 1]
    # exact discrete input power per unit width (density-normalised)
    rec["P_in"] = 0.5 * p.ω * imag(dot(b, x)) / lay.W
    # reciprocity: G(a ← b) vs G(b ← a)
    rec["G_ab"] = transpose(ba) * Xs[:, 3]
    rec["G_ba"] = transpose(bb) * Xs[:, 2]
    L3_0.phase!(rec, "post") do
        ηh = FEFunction(X, x)[iη]
        r, rx = roots_x(p, lay)
        out = Dict{Symbol,Any}()
        for (side, w) in ((:R, lay.win_R), (:L, lay.win_L))
            xs = p.x_sampling == :nodes ?
                 [i * lay.he for i in ceil(Int, w[1] / lay.he):floor(Int, w[2] / lay.he)] :
                 collect(range(w[1], w[2]; length=p.n_xsamples))
            ηm, tr = project(ηh, xs, lay, p.H)
            fit = L1_1.fit_waves(xs, ηm, rx; x_ref=lay.x_s, n_modes=p.n_modes)
            out[side] = (fit=fit, tr=tr)
        end
        R, Lf = out[:R].fit, out[:L].fit
        rec["kx_h_R"], rec["kx_h_L"] = R.k, Lf.k
        rec["fit_residual"] = max(R.residual, Lf.residual)
        rec["transverse_residual"] = max(out[:R].tr, out[:L].tr)
        out_R, in_R = R.Aplus, R.Aminus
        out_L, in_L = Lf.Aminus, Lf.Aplus
        rec["A_out_R"], rec["A_in_R"], rec["A_out_L"], rec["A_in_L"] = out_R, in_R, out_L, in_L
        kx = lay.kx
        s = sin(kx * lay.b / 2) / (kx * lay.b / 2)
        rec["gamma_h_R"] = (out_R - in_L) / (im * s * p.F)
        rec["gamma_h_L"] = (out_L - in_R) / (im * s * p.F)
        d1 = dD0(host(p), lay.kp, p.ω)
        rec["dD0"] = d1
        rec["gamma_theta"] = lay.kp / (kx * d1)
        rec["gamma_r"] = gamma_r(host(p), p.ω; dim=2)
        c = p.ω / 4 * d1 * kx / lay.kp
        rec["flux_net"] = c * (abs2(out_R) - abs2(in_R) + abs2(out_L) - abs2(in_L))
        rec["sponge_reflection"] = max(abs(in_R / out_R), abs(in_L / out_L))
        # field sample along y = W/3 for the figure
        xf = collect(range(lay.he / 7, lay.Lx - lay.he / 7; length=600))
        rec["fig_x"] = xf
        rec["fig_eta"] = ComplexF64.(L3_0.eval_on_plate(ηh, [Point(xx, lay.W / 3, p.H) for xx in xf]))
        nothing
    end
    for k in (:kp, :λ, :ky, :kx, :he, :W, :ny, :hy, :nx, :Lx, :x_s, :b, :Ls, :gap, :κ1, :κeff, :top)
        rec[string(k)] = getfield(lay, k)
    end
    rec["win_R"], rec["win_L"] = collect(lay.win_R), collect(lay.win_L)
    rec["maxrss_final_gb"] = L3_0.maxrss_gb()
    return rec
end

end
