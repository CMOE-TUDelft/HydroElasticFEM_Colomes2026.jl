"""
# L4-8 — anisotropic hybridisation of a finite rectangular plate (configuration B, 3D)

## Relation to the paper

`subsubsec:aniso_finite` (red, estimate): on a free rectangular plate L_x × L_y
in open water, a wet mode of dominant wavenumber k_mn sees the fluid-diluted
mass ratio (`eq:mode_dilution`)

    μ_mn ≃ n₀m_ρ,r / (m_ρ + M_f,ρ(k_mn, H)),   k_mn ≃ √((β_m/L_x)² + (β_n/L_y)²),

so that in deep water μ₁₀/μ₀₁ ≃ L_y/L_x: a uniform resonator population couples
anisotropically to the wet modes of an elongated plate. L2-5 confirmed the 2D
size law and found the absolute value low by a factor ≈ 1.5 (prefactor).

Verification test matrix: row L4.8 (2026-10-15 base; frozen row in the 2026-10-22 file).

## Why not the peak splitting of a plane-incidence forced response

The planned route (forced response, plane incidence along x and along y, peak
splitting) cannot resolve the splitting: the first elastic wet modes of these
plates radiate strongly. A 2D probe with the L2-5 model (complex eigenvalue with
the full added mass) gives ζ_rad = 0.13 (L = 20 m, H = 10 and 50 m) and 0.18
(L = 80 m), against hybridisation splittings Δ ≈ 0.035–0.07: the two peaks merge
into one (Δ < 2ζ_rad). The test therefore uses the forced response in a
different way, below. `forced_curves` reproduces the merged single peak.

## Method: forced radiation responses projected on a Ritz basis

Quarter domain (symmetry planes x = 0, y = 0: walls for the fluid), plate
[0, L_x/2] × [0, L_y/2] at the free surface (P1: m_ρ = 0.01, C_ρ = 0.49, ν = 0,
free edges), open water with a free surface κ, and a damping frame (library
`DampingZoneBC`, zones along the outer x and y strips with one common profile
μ₀(1 − cos(π/2 · s)), s = max(s_x, s_y) the normalised depth into the frame).

Ritz basis: the doubly symmetric dry free–free beam products
Φ_ij = X_i(x/L_x) X_j(y/L_y), i, j ∈ {0, 1, 3, 5, 7} (X₀ = 1, X₁ … X₇ the
symmetric elastic modes, β = 4.730, 10.996, 17.279, 23.562). With ν = 0, X_i(x)·1 is an
exact dry mode of the free plate on the hydrostatic foundation. Symmetric trial
functions satisfy ∂η/∂n = 0 on the symmetry lines, so the quarter energy needs no
symmetry condition.

At each ω, one METIS factorisation of the fluid block S_ff of the assembled bare
system gives the projected added-mass matrix (a forced radiation response per
basis shape, all shapes as right-hand sides of one factorisation):

    A_B(ω) = Φᵀ S_sf S_ff⁻¹ S_fs Φ / ω²,   Z_B(ω) = K_B − ω²(M_B + A_B(ω)),

K_B, M_B from the structural block at two frequencies (S_ss = K − ω²M). The wet
modes follow from the reduced `eq:eigWet` (Re A_B frozen, fixed point with MAC
tracking, A_B interpolated on a Chebyshev grid of `n_grid` frequencies around
the mode), and the hybridised pair from the same reduced problem with the
homogenised resonator field. For a uniform lossless field q = K/(K − ω²M) η
pointwise, so q lies in the span of the basis and the field adds the blocks
[k N, −k N; −k N, k N] and diag(0, m N), m = n₀m_ρ,r, k = mω_r², N the Gram
matrix of the basis on the plate (no truncation from the field itself).

Observables per target mode: ω_mn, splitting Δ = (ω₊ − ω₋)/ω_mn with ω_r = ω_mn,
μ_mode = m ψᵀNψ/ψᵀ(M_B + Re A_B)ψ, ζ_rad (complex eigenvalue with the full A_B),
and the estimates of `eq:mode_dilution`.

The same procedure in 2D (`ritz_2d`) reproduces the L2-5 eigWet results
(validation of the truncation of the Ritz basis).

## Reproduce

    julia --project -O1 scripts/L4-beyond-analytical/L4-8_anisotropic_hybridisation.jl --list
"""
module L4_8

using Gridap
using Parameters
using LinearAlgebra
import SparseArrays
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

using HydroElasticFEM_Colomes2026.References: k_open, added_mass, ResonatorSpec
using HydroElasticFEM_Colomes2026.Solvers: MetisLUSolver, factorize, solve_rhs, factor_stats
import HydroElasticFEM_Colomes2026.L3_0
import HydroElasticFEM_Colomes2026.L2_3
import HydroElasticFEM_Colomes2026.L2_5

export L4_8_Params, VARIABLE, fixed_hash, layout, run_case, reduce_case, BETA, beam, local_estimate

# ─── Beam shapes ───────────────────────────────────────────────────────────

"Symmetric free–free beam constants: index 0 (rigid heave), 1, 3, 5, 7 (tan(β/2) + tanh(β/2) = 0)."
const BETA = Dict(0 => 0.0, 1 => 4.730040744862704, 3 => 10.995607838001671, 5 => 17.27875953208824,
                  7 => 23.56194490204046)

"Symmetric dry mode X_i(s), s ∈ [−½, ½] (centred, unit length); X₀ = 1."
function beam(i::Int, s::Real)
    i == 0 && return 1.0
    β = BETA[i]
    return cosh(β * s) / cosh(β / 2) + cos(β * s) / cos(β / 2)
end

"Local estimate ω² = (g + C k⁴)/(m_ρ + M_f(k, H)) and μ of eq:mode_dilution for wavenumber k."
function local_estimate(k, H; g=9.8, C=0.49, m=0.01, loading=0.01)
    Mf = added_mass(k, H)
    return (omega=sqrt((g + C * k^4) / (m + Mf)), mu=loading / (m + Mf), Mf=Mf)
end

# ─── 3D problem ────────────────────────────────────────────────────────────

"Parameters of one L4-8 3D case (one plate, one depth, one mesh/frequency family)."
@with_kw struct L4_8_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    Ly::Float64 = 40.0                    # plate length along y (m)
    H::Float64 = 10.0                     # depth (m)
    target::Symbol = :m10                 # :m10, :m01 (L_y > L_x); :sym, :anti (square: X₁(x) ± X₁(y))
    omega_ref::Float64 = 2.4              # centre of the frequency grid (rad/s), from the smoke pass
    gap_over_λ::Float64 = 1.0             # open water between plate and frame (λ_open at the grid bottom)
    frame_over_λ::Float64 = 1.5           # frame width
    epw::Float64 = 8.0                    # open-water cells per λ_open (grid top)
    plate_cells::Int = 16                 # plate cells per full side length (at least)
    nz::Int = 8

    # ── Fixed: host (P1) and population ──────────────────────────────────────
    Lx::Float64 = 20.0
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49
    hb::Float64 = 0.01
    ν::Float64 = 0.0
    loading::Float64 = 0.01               # n₀m_ρ,r (m)

    # ── Fixed: discretisation ─────────────────────────────────────────────────
    order::Int = 2
    γ::Float64 = 6.0
    top_k::Float64 = 0.2                  # top cell = top_k / k_open(grid top)
    ratio::Float64 = 1.15                 # max growth of the open-water cells
    μ₀::Float64 = 2.5
    βₕ::Float64 = 0.5
    grid_halfwidth::Float64 = 0.12        # ω ∈ ω_ref (1 ± w)
    n_grid::Int = 7
    basis::Vector{Int} = [0, 1, 3, 5, 7]
end

const VARIABLE = (:Ly, :H, :target, :omega_ref, :gap_over_λ, :frame_over_λ, :epw, :plate_cells, :nz)

function fixed_hash(p::L4_8_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L4_8_Params) if !(f in VARIABLE))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

"Chebyshev nodes of the frequency grid."
grid(p) = [p.omega_ref * (1 + p.grid_halfwidth * cos(π * (2j - 1) / (2p.n_grid))) for j in p.n_grid:-1:1]

"Geometric cell sizes on a segment of length `len` growing from `h0` to at most `hmax` (ratio ≤ r)."
function grown_sizes(len, h0, hmax, r)
    sizes = Float64[]; s = 0.0; h = h0
    while s + h < len - 1e-9
        h = min(h * r, hmax); push!(sizes, h); s += h
    end
    isempty(sizes) && return [len]
    sizes .*= len / sum(sizes)
    return sizes
end

"""
    layout(p)

Tensor node vectors of the quarter domain. Plate: uniform cells (≥ `plate_cells`
per full side, ≤ λ_open/epw); open water: cells growing by `ratio` up to
λ_open(ω_hi)/epw; open water `gap_over_λ`·λ_open(ω_lo), frame `frame_over_λ`·λ_open(ω_lo).
"""
function layout(p::L4_8_Params)
    ωs = grid(p); ωlo, ωhi = p.omega_ref * (1 - p.grid_halfwidth), p.omega_ref * (1 + p.grid_halfwidth)
    λlo = 2π / k_open(ωlo, p.H, p.g); khi = k_open(ωhi, p.H, p.g); λhi = 2π / khi
    hopen = λhi / p.epw
    gap, frame = p.gap_over_λ * λlo, p.frame_over_λ * λlo
    function axis(half)
        n = max(ceil(Int, p.plate_cells / 2), ceil(Int, half / hopen))
        hp = half / n
        hp_over = min(hp, hopen)
        outer = grown_sizes(gap + frame, hp_over, hopen, p.ratio)
        nodes = cumsum([0.0; fill(hp, n); outer])
        return nodes
    end
    xs, ys = axis(p.Lx / 2), axis(p.Ly / 2)
    LΩx, LΩy = xs[end], ys[end]
    return (; xs, ys, LΩx, LΩy, nx=length(xs) - 1, ny=length(ys) - 1, nz=p.nz, gap, frame, hopen, λlo, λhi,
            top=p.top_k / khi, ωs, ωlo, ωhi, Ld=frame)
end

node_map(nodes) = (a = nodes[1]; n = length(nodes) - 1; Lr = nodes[end] - a;
                   s -> (t = clamp((s - a) / Lr * n, 0.0, Float64(n)); i = clamp(floor(Int, t), 0, n - 1);
                         nodes[i + 1] + (t - i) * (nodes[i + 2] - nodes[i + 1])))

function tank(p, lay)
    fx, fy = node_map(lay.xs), node_map(lay.ys)
    fz = L3_0.zmap(p.H, lay.nz, lay.top)
    cart = G.CartesianDomain(mins=(0.0, 0.0, 0.0), maxs=(lay.LΩx, lay.LΩy, p.H), parts=(lay.nx, lay.ny, lay.nz),
                             map=x -> VectorValue(fx(x[1]), fy(x[2]), fz(x[3])))
    Ld = lay.Ld
    return G.TankDomain(cart;
        structure_domains=[G.StructureDomain(L=p.Lx / 2, W=p.Ly / 2, x₀=[0.0, 0.0, p.H], domain_symbol=:Γ_plate)],
        damping_zones=[G.DampingZone(L=Ld, W=lay.LΩy, x₀=[lay.LΩx - Ld, 0.0, p.H], domain_symbol=:Γ_dx),
                       G.DampingZone(L=lay.LΩx - Ld, W=Ld, x₀=[0.0, lay.LΩy - Ld, p.H], domain_symbol=:Γ_dy)])
end

function physics(p, lay, ω)
    k = k_open(ω, p.H, p.g)
    Ld = lay.Ld; x0, y0 = lay.LΩx - Ld, lay.LΩy - Ld
    s(x) = clamp(max((x[1] - x0) / Ld, (x[2] - y0) / Ld), 0.0, 1.0)
    μ1p(x) = p.μ₀ * (1 - cos(π / 2 * s(x)))
    μ1(x) = μ1p(x) / p.βₕ
    μ2(x) = μ1p(x) * k
    zero_c(x) = 0.0 + 0.0im
    CT = Vector{ComplexF64}
    fluid = P.PotentialFlow(ρw=p.ρ, g=p.g, dim=3, fe=PH.FESpaceConfig(order=p.order, vector_type=CT),
        space_domain_symbol=:Ω,
        boundary_conditions=P.AbstractPotentialFlowBC[
            P.DampingZoneBC(domain=:dΓd_1, μ₁=μ1, μ₂=μ2, η_in=zero_c, vz_in=zero_c, βₕ=p.βₕ),
            P.DampingZoneBC(domain=:dΓd_2, μ₁=μ1, μ₂=μ2, η_in=zero_c, vz_in=zero_c, βₕ=p.βₕ)])
    fs = P.FreeSurface(ρw=p.ρ, g=p.g, βₕ=p.βₕ, dim=3, fe=PH.FESpaceConfig(order=p.order, vector_type=CT),
                       space_domain_symbol=:Γκ)
    ρb = p.m_ρ * p.ρ / p.hb
    plate = P.KirchhoffLovePlate(E=12 * p.C_ρ * p.ρ / p.hb^3, ν=p.ν, hb=p.hb, ρ=p.ρ, ρb=ρb, g=p.g, ambient_dim=3,
                                 manifold_dim=2, fe=PH.FESpaceConfig(order=p.order, vector_type=CT, γ=p.γ),
                                 space_domain_symbol=:Γη)
    return P.PhysicsParameters[fluid, fs, plate]
end

"Ritz shapes (i, j) of the basis, row-major in (i, j)."
shapes(p) = [(i, j) for i in p.basis for j in p.basis]

"Target coefficient vector on the basis (unit dry shape of the target mode)."
function target_vector(p)
    sh = shapes(p); t = zeros(length(sh))
    idx(i, j) = findfirst(==((i, j)), sh)
    if p.target == :m10
        t[idx(1, 0)] = 1
    elseif p.target == :m01
        t[idx(0, 1)] = 1
    elseif p.target == :sym
        t[idx(1, 0)] = 1; t[idx(0, 1)] = 1
    elseif p.target == :anti
        t[idx(1, 0)] = 1; t[idx(0, 1)] = -1
    end
    return t
end

"Assemble the bare system at ω; return A, index sets, and the η space/plate measure."
function assemble(p, lay, ω; rec, tag)
    tk = tank(p, lay)
    problem = L3_0.phase!(() -> S.build_problem(tk, physics(p, lay, ω), PH.FreqDomainConfig(ω=ω, solver=MetisLUSolver())),
                          rec, "$(tag)_assembly")
    op = S.get_fe_operator(problem)
    A = get_matrix(op)
    X = S.get_trial_fe_space(problem)
    fmap = S.get_field_map(problem)
    offs = cumsum([0; [num_free_dofs(Xi) for Xi in X.spaces]])
    rng(sym) = (offs[fmap[sym]] + 1):offs[fmap[sym] + 1]
    s = collect(rng(:η))
    f = setdiff(1:size(A, 1), s)
    dΓ = S.get_integration_domains(problem)[:dΓη]
    Vη = X.spaces[fmap[:η]]
    return (; A, s, f, Vη, dΓ, X, fmap, problem)
end

"Basis matrix Φ (η free DOFs × shapes) and Gram matrix N = ∫ φᵢφⱼ on the quarter plate."
function basis_matrix(p, sys)
    sh = shapes(p)
    fns = [x -> beam(i, x[1] / p.Lx) * beam(j, x[2] / p.Ly) for (i, j) in sh]   # x, y measured from the centre
    hs = [interpolate(f, sys.Vη) for f in fns]
    Φ = hcat([real.(get_free_dof_values(h)) for h in hs]...)
    n = length(sh)
    N = [real(sum(∫(hs[a] * hs[b])sys.dΓ)) for a in 1:n, b in 1:n]
    return Φ, N
end

"""
    run_case(p) -> Dict

Assemble and factorise S_ff at every grid frequency (and at ω_ref for the
structural matrices), store Re/Im of A_B(ω), K_B, M_B, N, and the FE forced
response to the target load at ω_ref (full system) for the truncation check.
"""
function run_case(p::L4_8_Params)
    lay = layout(p)
    rec = Dict{String,Any}("nx" => lay.nx, "ny" => lay.ny, "nz" => lay.nz, "LOx" => lay.LΩx, "LOy" => lay.LΩy,
                           "gap" => lay.gap, "frame" => lay.frame, "hopen" => lay.hopen, "lambda_lo" => lay.λlo,
                           "lambda_hi" => lay.λhi, "omegas" => lay.ωs)
    Aω = Matrix{ComplexF64}[]
    Sss = Dict{Float64,Matrix{Float64}}()
    Φ = nothing; N = nothing
    for (n, ω) in enumerate(lay.ωs)
        sys = assemble(p, lay, ω; rec=rec, tag="w$(n)")
        if Φ === nothing
            Φ, N = basis_matrix(p, sys)
            rec["N_eta"] = length(sys.s); rec["N"] = size(sys.A, 1); rec["nnzA"] = SparseArrays.nnz(sys.A)
        end
        n in (1, length(lay.ωs) ÷ 2 + 1, length(lay.ωs)) && (Sss[ω] = real(transpose(Φ) * (sys.A[sys.s, sys.s] * Φ)))
        Sff = sys.A[sys.f, sys.f]; Sfs = sys.A[sys.f, sys.s] * Φ; Ssf = sys.A[sys.s, sys.f]
        sys = nothing; GC.gc(true)
        ns = L3_0.phase!(() -> factorize(MetisLUSolver(), Sff), rec, "w$(n)_factorize")
        n == 1 && (st = factor_stats(ns); rec["lunz"] = st.lunz; rec["umf_peak_gb"] = st.peak_gb;
                   rec["ordering"] = String(st.ordering))
        Y = solve_rhs(ns, Matrix(Sfs))
        rec["w$(n)_rel_residual"] = maximum(norm(Sff * Y[:, j] - Sfs[:, j]) / norm(Sfs[:, j]) for j in axes(Y, 2))
        finalize(ns.factors); ns = nothing
        push!(Aω, transpose(Φ) * (Ssf * Y) ./ ω^2)
        Sff = nothing; Y = nothing; GC.gc(true)
    end
    # structural matrices from S_ss = K − ω²M at the first and last grid ω; linearity checked at the middle one
    ωa, ωm, ωb = lay.ωs[1], lay.ωs[length(lay.ωs) ÷ 2 + 1], lay.ωs[end]
    M = (Sss[ωa] - Sss[ωb]) ./ (ωb^2 - ωa^2)
    K = Sss[ωa] + ωa^2 .* M
    rec["Sss_linearity"] = norm(Sss[ωm] - (K - ωm^2 .* M)) / norm(Sss[ωm])
    rec["K_B"], rec["M_B"], rec["N_B"] = K, M, N
    rec["A_re"] = [real(a) for a in Aω]; rec["A_im"] = [imag(a) for a in Aω]
    # forced response (full monolithic system) to the target load at ω_ref: truncation check of the basis
    sys = assemble(p, lay, p.omega_ref; rec=rec, tag="forced")
    t = target_vector(p)
    # load vector ∫ f_t v_η dΓ with the target pressure f_t = Σ t_a φ_a
    ft = x -> sum(t[a] * beam(ij[1], x[1] / p.Lx) * beam(ij[2], x[2] / p.Ly) for (a, ij) in enumerate(shapes(p)))
    Yη = S.get_test_fe_space(sys.problem).spaces[sys.fmap[:η]]
    bη = assemble_vector(v -> ∫(ft * v)sys.dΓ, Yη)
    b = zeros(ComplexF64, size(sys.A, 1)); b[sys.s] = bη
    rec["forced_load_sum"] = real(sum(bη))
    A = sys.A; s = sys.s
    sys = nothing; GC.gc(true)
    ns = L3_0.phase!(() -> factorize(MetisLUSolver(), A), rec, "forced_factorize")
    x = solve_rhs(ns, b)
    rec["forced_rel_residual"] = norm(A * x - b) / norm(b)
    finalize(ns.factors); ns = nothing
    rec["forced_load_B"] = real(transpose(Φ) * bη)          # reduced load Φᵀb_η
    rec["forced_eta_t"] = transpose(bη) * x[s]                # ∫ f_t η_h dΓ (work-conjugate response)
    rec["maxrss_final_gb"] = L3_0.maxrss_gb()
    return rec
end

# ─── Reduced eigWet (shared by 2D and 3D) ──────────────────────────────────

"Barycentric Chebyshev interpolation weights for nodes xs (first-kind nodes)."
function bary_weights(xs)
    n = length(xs)
    return [prod(j == k ? 1.0 : (xs[j] - xs[k]) for k in 1:n) for j in 1:n] .|> inv
end

"Interpolant ω ↦ A(ω) (entrywise, barycentric polynomial through the grid)."
function interpolant(ωs, As)
    w = bary_weights(ωs)
    return function (ω)
        j = findfirst(x -> abs(ω - x) < 1e-14 * x, ωs)
        j === nothing || return As[j]
        c = w ./ (ω .- ωs)
        return sum(c[j] * As[j] for j in eachindex(ωs)) / sum(c)
    end
end

symm(A) = (A + transpose(A)) / 2
macN(u, v, N) = abs2(dot(u, N * v)) / (dot(u, N * u) * dot(v, N * v))

"Secant-accelerated fixed point ω = F(ω) (as `L2_5.fixed_point`)."
function fixed_point(step, ω0; tol=1e-10, maxiter=40)
    ωa = ω0; ga = step(ωa) - ωa; ωb = ωa + ga
    for _ in 1:maxiter
        gb = step(ωb) - ωb
        abs(gb) <= tol * ωb && return ωb + gb
        ωn = abs(gb) < abs(ga) ? ωb - gb * (ωb - ωa) / (gb - ga) : ωb + gb
        (ωa, ga, ωb) = (ωb, gb, ωn)
    end
    return NaN
end

"""
    reduce_case(K, M, N, Are, Aim, ωs, t; loading, sel=:)

Reduced eigWet on the Ritz basis: bare wet mode tracked from the target vector
t, the hybridised pair with the homogenised field (ω_r = ω_mn), μ_mode, ζ_rad.
`sel` selects a sub-basis (truncation check). Returns a NamedTuple.
"""
function reduce_case(K, M, N, Are, Aim, ωs, t; loading, sel=Colon())
    K, M, N = symm(K[sel, sel]), symm(M[sel, sel]), symm(N[sel, sel])
    t = t[sel]
    Ar = interpolant(ωs, [symm(a[sel, sel]) for a in Are])
    Ai = interpolant(ωs, [symm(a[sel, sel]) for a in Aim])
    n = size(K, 1)
    # bare mode: start from the frozen mode closest (MAC) to t at the grid centre
    v = t
    function step(ω)
        e = eigen(Symmetric(K), Symmetric(M + Ar(ω)))
        j = argmax([e.values[i] > 0 ? macN(e.vectors[:, i], v, N) : -1.0 for i in 1:n])
        v = e.vectors[:, j]
        return sqrt(e.values[j])
    end
    ω0 = ωs[(length(ωs) + 1) ÷ 2]
    step(ω0); ωmn = fixed_point(step, step(ω0))
    ψ = v
    mac_target = macN(ψ, t, N)
    Mt = M + Ar(ωmn)
    μ_mode = loading * dot(ψ, N * ψ) / dot(ψ, Mt * ψ)
    # radiation damping: complex eigenvalue (full A) closest to ω_mn²
    ec = eigen(K, M + Ar(ωmn) + im * Ai(ωmn))
    jc = argmin(abs.(ec.values .- ωmn^2))
    ωc = sqrt(complex(ec.values[jc]))
    # hybridised pair with the homogenised field tuned to ω_mn
    m = loading; k = m * ωmn^2
    Kaug = [K + k * N -k * N; -k * N k * N]
    Maug(ω) = [M + Ar(ω) zeros(n, n); zeros(n, n) m * N]
    F = eigen(Symmetric(Kaug), Symmetric(Maug(ωmn)))
    proj = [abs2(dot(ψ, Mt * F.vectors[1:n, i])) / (dot(ψ, Mt * ψ) * dot(F.vectors[1:n, i], Mt * F.vectors[1:n, i]) + eps())
            for i in 1:2n]
    below = [i for i in 1:2n if 0 < F.values[i] < ωmn^2]
    above = [i for i in 1:2n if F.values[i] > ωmn^2]
    out = map((below[argmax(proj[below])], above[argmax(proj[above])])) do j0
        u = F.vectors[:, j0]
        Naug = [N zeros(n, n); zeros(n, n) N]
        function stp(ω)
            G = eigen(Symmetric(Kaug), Symmetric(Maug(ω)))
            jj = argmax([G.values[i] > 0 ? macN(G.vectors[:, i], u, Naug) : -1.0 for i in 1:2n])
            u = G.vectors[:, jj]
            return sqrt(G.values[jj])
        end
        (fixed_point(stp, sqrt(F.values[j0])), proj[j0])
    end
    ωm, ωp = out[1][1], out[2][1]
    inside = minimum(ωs) <= ωm && ωp <= maximum(ωs) && minimum(ωs) <= ωmn <= maximum(ωs)
    return (omega=ωmn, psi=ψ, mac_target=mac_target, mu_mode=μ_mode, omega_minus=ωm, omega_plus=ωp,
            split=(ωp - ωm) / ωmn, proj_minus=out[1][2], proj_plus=out[2][2], omega_complex=ωc,
            zeta_rad=-imag(ωc) / abs(ωc), inside=inside, Ar=Ar, Ai=Ai, K=K, M=M, N=N)
end

"""
    forced_curves(red, t, ωq; loading)

|response| of the reduced complex model (full A) to the target load, bare and
with the field tuned to ω_mn: R(ω) = tᵀN a(ω). Shows the merged peak.
"""
function forced_curves(red, t, ωq; loading)
    K, M, N = red.K, red.M, red.N
    n = size(K, 1); m = loading; k = m * red.omega^2
    f = N * t
    bare = [abs(dot(t, N * ((K - ω^2 * (M + red.Ar(ω) + im * red.Ai(ω))) \ f))) for ω in ωq]
    res = map(ωq) do ω
        Z = [K + k * N - ω^2 * (M + red.Ar(ω) + im * red.Ai(ω)) -k * N; -k * N k * N - ω^2 * m * N]
        a = Z \ [f; zeros(n)]
        abs(dot(t, N * a[1:n]))
    end
    return bare, res
end

# ─── 2D validation against L2-5 ─────────────────────────────────────────────

"""
    ritz_2d(Lp, H; epw=16, order=3, basis=[0,1,3,5], n_grid=7, halfwidth=0.12, omega_ref)

The reduced procedure on the L2-5 channel (same mesh rules): Φ from the 2D beam
shapes, A_B on a Chebyshev grid, K_B, M_B from S_ss. Discrete resonators every
`spacing` (as L2-5): N_res = m_r Σ φ(x_k)φ(x_k)ᵀ replaces loading·N. Returns
ω₁ and the splitting for comparison with the L2-5 records.
"""
function ritz_2d(Lp, H; epw=16, order=3, basis=[0, 1, 3, 5, 7], n_grid=7, halfwidth=0.12, omega_ref, loading=0.01,
                 spacing=1.0)
    p = L2_5.L2_5_Params(Lp=Lp, H=H, elements_per_wavelength=epw, order=order)
    q = L2_5.withmesh(p, L2_5.omega_estimate(p))
    ωs = [omega_ref * (1 + halfwidth * cos(π * (2j - 1) / (2n_grid))) for j in n_grid:-1:1]
    Aω = Matrix{ComplexF64}[]; Sss = Matrix{Float64}[]
    Φ = nothing; Pk = nothing; xk = nothing
    for ω in ωs
        b = L2_3.blocks(q, ω; xs=Float64[], resonators=ResonatorSpec[])
        if Φ === nothing
            Vη = b.sys.X.spaces[b.sys.fmap[:η]]
            x0 = b.sys.x0
            hs = [interpolate(x -> beam(i, (x[1] - x0) / Lp - 0.5), Vη) for i in basis]
            Φ = hcat([real.(get_free_dof_values(h)) for h in hs]...)
            Nr = max(1, round(Int, Lp / spacing)); a = Lp / Nr
            xk = [(i - 0.5) * a for i in 1:Nr]
            Pk = hcat([real.([h(Point(x0 + x, 0.0)) for x in xk]) for h in hs]...)   # Nr × nb
            Pk = (Pk, loading * a)
        end
        push!(Sss, real(transpose(Φ) * (b.A[b.s, b.s] * Φ)))
        Y = lu(b.A[b.f, b.f]) \ Matrix(b.A[b.f, b.s] * Φ)
        push!(Aω, transpose(Φ) * (b.A[b.s, b.f] * Y) ./ ω^2)
    end
    M = (Sss[1] - Sss[end]) ./ (ωs[end]^2 - ωs[1]^2)
    K = Sss[1] + ωs[1]^2 .* M
    lin = norm(Sss[(n_grid + 1) ÷ 2] - (K - ωs[(n_grid + 1) ÷ 2]^2 .* M)) / norm(Sss[(n_grid + 1) ÷ 2])
    # discrete resonators: N_res/m = Pᵀ P a (so that loading·N ↦ m_r Σ φφᵀ)
    Pm, mr = Pk
    Nres = transpose(Pm) * Pm .* (mr / loading)
    t = zeros(length(basis)); t[findfirst(==(1), basis)] = 1
    red = reduce_case(K, M, Nres, [real(a) for a in Aω], [imag(a) for a in Aω], ωs, t; loading=loading)
    return (red=red, linearity=lin, omegas=ωs, Nres=Nres, M=M, K=K)
end

end # module L4_8
