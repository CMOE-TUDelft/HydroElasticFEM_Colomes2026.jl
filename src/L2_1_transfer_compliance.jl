"""
# L2-1 — Transfer compliance G_B of a finite plate in open water (configuration B, vertical section)

## Relation to the paper

For a structure of finite extent floating in open water, the Green function of
Section 4 is replaced by the transfer compliance of the bare structure,
G_B(xᵢ, xⱼ; ω): the deflection at xᵢ due to a unit point force at xⱼ, with the
radiation condition imposed in the open water (`sec:finite_domain`). Every
configuration-B result of the paper is written in terms of G_B: the Foldy
system with G = G_B (`eq:lat_woodbury`), the renormalised resonance and
radiative width γ_r = Im G_B(x₁, x₁) of a single resonator, the asymmetric
absorption bound (`eq:lat_absorbed_B`). By reciprocity G_B is symmetric,
G_B(xᵢ, xⱼ) = G_B(xⱼ, xᵢ).

Verification test matrix: §L2.1.

## Problem

Configuration B: a plate of length `Lp` with free edges, floating in a channel
of depth H, with open water (`FreeSurface`) on both sides and Sommerfeld
radiation conditions (`RadiationBC`, open-water wavenumber) on the vertical end
walls. The Sommerfeld condition is exact for the propagating wave only; the
open water on each side, max(λ_open, `gap_decays`/κ₁), with κ₁ the decay rate of
the slowest open-water evanescent mode, keeps its reflection from the walls
below e^{−2·gap_decays}. The element size is h = min(λ_open, λ_p, 4ℓ_f)/n,
n = `elements_per_wavelength`, with ℓ_f the decay length of the flexural roots,
so that the open-water wave, the plate wave and the flexural near field are all
resolved; the vertical mesh follows the rule of L1-1.

Point loads: a massless spring (`ResonatorArray` entry with M = 0, stiffness
k_ρ) is attached at every load point. Its equation is k_ρ(q − η(xⱼ)) = F, so a
load F on its degree of freedom applies exactly the point force F to the plate
through the Dirac coupling, and an unloaded spring is inert (q = η). The
assembled matrix is factorised once per frequency; the load vectors are built
directly on the spring degrees of freedom, b[qⱼ] = A[qⱼ, qⱼ] F/k_ρ, which is
independent of the internal scaling of the resonator equations. (HydroElasticFEM
does not assemble right-hand sides on resonator degrees of freedom:
`has_rhs_form(::ResonatorArray) = false`.)

## Observables

G_B at the `points` (columns from one factorisation per frequency), its
reciprocity residual ‖G − Gᵀ‖/‖G‖ (identity of the discrete problem), and
the columns themselves for the mesh and truncation studies of the script.

## Reproduce

    julia --project -O1 scripts/L2-configB-vertical/L2-1_transfer_compliance.jl
"""
module L2_1

using Gridap
using Parameters
using LinearAlgebra: lu, norm
using Roots: find_zero, Bisection
using Random: MersenneTwister
import WaveSpec
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH
using HydroElasticFEM: map_vertical_GP_for_const_dep

using ..References: HostSpec, ResonatorSpec, k0, k_open, host_roots
import ..L1_1

export L2_1_Params, VARIABLE, fixed_hash, run_case

"""
Parameters of one L2-1 case. Positions are measured from the left edge of the plate.
"""
@with_kw struct L2_1_Params
    # ── Variable: what the study sweeps (names the record) ───────────────────
    ω::Float64 = 10.0
    elements_per_wavelength::Int = 8      # h = min(λ_open, λ_p, 8ℓ_f)/n
    gap_decays::Float64 = 6.0             # open water on each side: max(λ_open, gap_decays/κ₁,open)
    order::Int = 3
    T_ρ::Float64 = 0.0

    # ── Fixed: host, set P1 of the test matrix = Table 1 of the paper ────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49
    Lp::Float64 = 20.0                    # plate length, free edges

    # ── Fixed: load / attachment points ──────────────────────────────────────
    n_regular::Int = 20                   # x = 0.5, 1.5, …, 19.5 (a = 1)
    n_random::Int = 5                     # uniform in [0.2, Lp − 0.2], fixed seed
    seed::Int = 20261023
    k_spring::Float64 = 1.0               # stiffness / ρ of the massless springs

    # ── Fixed: discretisation (vertical rule of L1-1) ────────────────────────
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0
    βₕ::Float64 = 0.5                     # free-surface stabilisation
end

"Fields of `L2_1_Params` varied by the L2-1 study (they name the records)."
const VARIABLE = (:ω, :elements_per_wavelength, :gap_decays, :order, :T_ρ)

"Short hash of the fields of `p` not in `variable` (shared by the L2 tests)."
function fixed_hash_fields(p, variable)
    fixed = Tuple(getfield(p, f) for f in fieldnames(typeof(p)) if !(f in variable))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

"Short hash of the fixed parameters, appended to the record name."
fixed_hash(p::L2_1_Params) = fixed_hash_fields(p, VARIABLE)

"Load / attachment points (from the left plate edge): regular, then random."
function points(p)
    regular = [(i - 0.5) * p.Lp / p.n_regular for i in 1:p.n_regular]
    rng = MersenneTwister(p.seed)
    random = 0.2 .+ (p.Lp - 0.4) .* rand(rng, p.n_random)
    return [regular; random]
end

"""
    mesh_size(p, ω)

Element size h = min(λ_open, λ_p, 4ℓ_f)/n, the open-water length on each side, and their ingredients.
"""
function mesh_size(p, ω::Real)
    h = L1_1.host(p)
    λo = 2π / k_open(ω, p.H, p.g)
    λp = 2π / k0(h, ω)
    ℓf = p.C_ρ > 0 ? 1 / imag(host_roots(h, ω; n_evanescent=2)[2]) : Inf   # flexural pair (none for a membrane)
    κ1 = kappa1_open(ω, p.H, p.g)
    return (he=min(λo, λp, 4ℓf) / p.elements_per_wavelength, gap=max(λo, p.gap_decays / κ1),
            λ_open=λo, λ_p=λp, ℓ_f=ℓf, κ1_open=κ1)
end

"""
    kappa1_open(ω, H, g)

Decay rate κ₁ of the slowest evanescent mode of open water: the smallest root of
ω²/g = −κ tan κH, which lies in (π/2, π)/H.
"""
function kappa1_open(ω::Real, H::Real, g::Real)
    f(κ) = ω^2 / g + κ * tan(κ * H)
    return find_zero(f, ((π / 2 + 1e-12) / H, (π - 1e-12) / H), Bisection())
end

"Number of free degrees of freedom of a (single-field) FE space."
num_free(V) = num_free_dofs(V)

"""
    open_roots(ω, H, g; n=4)

Open-water roots: the propagating k (k tanh kH = ω²/g) and iκₙ, n = 1…`n`,
with κₙ the roots of ω²/g = −κ tan κH in ((n − ½)π, nπ)/H, as complex
wavenumbers with positive imaginary part (`L1_1.fit_waves` convention).
"""
function open_roots(ω::Real, H::Real, g::Real; n::Int=4)
    f(κ) = ω^2 / g + κ * tan(κ * H)
    κs = [find_zero(f, (((j - 0.5) * π + 1e-12) / H, (j * π - 1e-12) / H), Bisection()) for j in 1:n]
    return ComplexF64[k_open(ω, H, g); im .* κs]
end

"Group velocity of open-water waves."
function c_group(ω::Real, H::Real, g::Real)
    k = k_open(ω, H, g)
    return ω / k / 2 * (1 + 2k * H / sinh(2k * H))
end

"""
    radiation_state(ω, H, g; amplitude=1e-3)

Single-frequency sea state carrying the open-water wavenumber of the radiation
condition and, when used for an incident wave, its amplitude.
"""
function radiation_state(ω, H, g; amplitude::Real=1e-3)
    spec = WaveSpec.ContinuousSpectrums.RegularWave(2amplitude, 2π / ω)
    ds = WaveSpec.SpectralSpreading.DiscreteSpectralSpreading(spec; mess=false)
    spread = WaveSpec.AngularSpreading.DiscreteAngularSpreading(0.0)
    return WaveSpec.AiryWaves.AiryState(ds, spread, 1, 1, [ω], [k_open(ω, H, g)], [0.0], H, 1)
end

"""
    build(p; ω=p.ω, ω_mesh=ω, xs, resonators, incident=0.0)

Assemble the configuration-B problem with one `ResonatorArray` entry per point
`xs` (plate coordinates): `resonators[i]` (a `ResonatorSpec`; mass 0 for a
massless spring). `p` is any parameter struct with the fields of `L2_1_Params`.
The mesh is sized at `ω_mesh` (default ω), so that a frequency sweep can use one mesh.
With `incident` > 0 a regular wave of that amplitude enters through the left
wall (Sommerfeld condition with incident traction, as HydroElasticFEM's
floating-beam example); its load vector is returned as `b`.
Returns the matrix, the free index of every resonator DOF, the trial space,
the field map, the mesh data and the plate offset.
"""
function build(p; ω::Real=p.ω, ω_mesh::Real=ω, xs::AbstractVector{<:Real}, resonators::AbstractVector{ResonatorSpec},
               incident::Real=0.0)
    length(xs) == length(resonators) || throw(DimensionMismatch("one resonator per point"))
    ms = mesh_size(p, ω_mesh)
    # plate edges on mesh nodes: an integer number of elements on the plate and in each open-water
    # region (otherwise the structure domain snaps its edges to element boundaries and the plate
    # length changes with the mesh and with the open-water length)
    n_plate = ceil(Int, p.Lp / ms.he)
    he = p.Lp / n_plate
    n_gap = ceil(Int, ms.gap / he)
    gap = n_gap * he
    nx = n_plate + 2n_gap
    L = nx * he
    q = (H=p.H, top_fraction=p.top_fraction, max_ratio=p.max_ratio, ratio_elements=p.ratio_elements,
         elements_per_wavelength=min(ms.λ_open, ms.λ_p) / he)
    vm = L1_1.vertical_mesh(q, he)
    H = p.H
    vertical_map(x) = VectorValue(x[1], map_vertical_GP_for_const_dep(x[2] - H, vm.ratio, vm.ny, H; dbgmsg=false))
    N = length(xs)
    x0 = gap
    tank = G.TankDomain(L=L, H=H, nx=nx, ny=vm.ny, map=vertical_map,
                        structure_domains=[G.StructureDomain(L=p.Lp, x₀=[x0, 0.0], domain_symbol=:Γs)],
                        resonator_domains=G.ResonatorDomain[G.ResonatorDomain(location=[x0 + x, 0.0], delta_symbol=Symbol("δ$i"))
                                                            for (i, x) in enumerate(xs)])
    fe(; kw...) = PH.FESpaceConfig(; order=p.order, vector_type=Vector{ComplexF64}, kw...)
    state = radiation_state(ω, H, p.g; amplitude=incident > 0 ? incident : 1e-3)
    bcs = P.AbstractPotentialFlowBC[P.RadiationBC(domain=:dΓin), P.RadiationBC(domain=:dΓout)]
    if incident > 0
        wave(x) = WaveSpec.AiryWaves.generate_sea(state, [x[1]], [0.0], [x[2]], [0.0], vars=[:ϕ, :u, :w])
        k = state.k[1]
        f_in(x) = (w = wave(x); VectorValue(w[:u][1], w[:w][1]) ⋅ VectorValue(-1.0, 0.0) - im * k * w[:ϕ][1])
        push!(bcs, P.PrescribedInletPotentialBC(domain=:dΓin, forcing=f_in, quantity=:traction))
    end
    fluid = P.PotentialFlow(g=p.g, ρw=p.ρ, sea_state=state, boundary_conditions=bcs, fe=fe(), space_domain_symbol=:Ω)
    fs = P.FreeSurface(ρw=p.ρ, g=p.g, βₕ=p.βₕ, fe=fe(), space_domain_symbol=:Γκ)
    plate = P.TensionedEulerBernoulliBeam(L=p.Lp, mᵨ=p.m_ρ, EIᵨ=p.C_ρ, Tᵨ=p.T_ρ, g=p.g, symbol=:η,
                                          fe=fe(γ=p.γ_penalty), space_domain_symbol=:Γs)
    ents = P.PhysicsParameters[fluid, fs, plate]
    N > 0 && push!(ents, P.resonator_array(N, [r.m_ρ * p.ρ for r in resonators], [r.k_ρ * p.ρ for r in resonators],
                                           [r.c_ρ * p.ρ for r in resonators];
                                           delta_domain_symbols=[Symbol("δ$i") for i in 1:N], ρw=p.ρ,
                                           host_domain_symbol=:Γs,
                                           fe=PH.FESpaceConfig(space_type=VectorValue{1,Float64},
                                                               vector_type=Vector{ComplexF64})))
    problem = S.build_problem(tank, ents, PH.FreqDomainConfig(ω=ω))
    A = get_matrix(problem.fe_operator)
    b = get_vector(problem.fe_operator)
    X = problem.trial_fe_space
    fmap = problem.field_map
    offsets = cumsum([0; [num_free(V) for V in X.spaces]])
    qdof = [offsets[fmap[Symbol("q$i")]] + 1 for i in 1:N]
    all(i -> num_free_dofs(X.spaces[fmap[Symbol("q$i")]]) == 1, 1:N) || error("resonator spaces must have one DOF")
    return (; A, b, qdof, X, fmap, x0, he, ny=vm.ny, nx, ms, L, gap)
end

"""
    point_load_vector(sys, j, F, k)

Load vector applying the point force F at the attachment of resonator j, whose
spring stiffness / ρ is k: b[qⱼ] = A[qⱼ, qⱼ] F/k (massless spring:
k(qⱼ − η(xⱼ)) = F). For a resonator with mass, use the spring only through
`green_columns`, which loads massless springs.
"""
function point_load_vector(sys, j::Int, F::Number, k::Real)
    b = zeros(ComplexF64, size(sys.A, 1))
    b[sys.qdof[j]] = sys.A[sys.qdof[j], sys.qdof[j]] * F / k
    return b
end

"Plate deflection of the solution vector `x` at plate coordinates `xs`."
function deflection(sys, x::AbstractVector, xs::AbstractVector{<:Real})
    uh = FEFunction(sys.X, x)
    ηh = uh[sys.fmap[:η]]
    return ComplexF64[ηh(Point(sys.x0 + xi, 0.0)) for xi in xs]
end

"Free-surface elevation of the solution vector `x` at absolute coordinates `xs` (open water)."
function surface(sys, x::AbstractVector, xs::AbstractVector{<:Real})
    κh = FEFunction(sys.X, x)[sys.fmap[:κ]]
    return ComplexF64[κh(Point(xi, 0.0)) for xi in xs]
end

"Resonator displacements of the solution vector `x`."
displacements(sys, x::AbstractVector) = x[sys.qdof]

"""
    green_columns(p; ω=p.ω, ω_mesh=ω, xs=points(p))

G_B(xᵢ, xⱼ) at the points `xs`, one factorisation and length(xs) back-solves:
massless springs at every point, unit load on spring j for column j. Also
returns the spring-equation residual max|k(qⱼ − ηⱼ) − F| over the loaded
springs (checks the point-load construction) and the mesh data.
"""
function green_columns(p; ω::Real=p.ω, ω_mesh::Real=ω, xs::AbstractVector{<:Real}=points(p))
    N = length(xs)
    springs = fill(ResonatorSpec(m_ρ=0.0, k_ρ=p.k_spring), N)
    sys = build(p; ω=ω, ω_mesh=ω_mesh, xs=xs, resonators=springs)
    F = lu(sys.A)
    Gm = zeros(ComplexF64, N, N)
    spring_residual = 0.0
    for j in 1:N
        x = F \ point_load_vector(sys, j, 1.0, p.k_spring)
        η = deflection(sys, x, xs)
        Gm[:, j] = η
        spring_residual = max(spring_residual, abs(p.k_spring * (displacements(sys, x)[j] - η[j]) - 1))
    end
    return (G=Gm, spring_residual=spring_residual, sys=sys)
end

"""
    run_case(p) -> Dict

G_B at the regular and random points, its reciprocity residual and the mesh data.
"""
function run_case(p::L2_1_Params)
    xs = points(p)
    gc = green_columns(p; xs=xs)
    Gm = gc.G
    return Dict{String,Any}(
        "points" => xs, "G" => Gm,
        "reciprocity" => norm(Gm - transpose(Gm)) / norm(Gm),
        "spring_residual" => gc.spring_residual,
        "ndofs" => size(gc.sys.A, 1), "element_size" => gc.sys.he, "nx" => gc.sys.nx, "ny" => gc.sys.ny,
        "lambda_open" => gc.sys.ms.λ_open, "lambda_p" => gc.sys.ms.λ_p, "ell_f" => gc.sys.ms.ℓ_f,
        "gap" => gc.sys.gap, "kappa1_open" => gc.sys.ms.κ1_open, "L" => gc.sys.L,
    )
end

end # module L2_1
