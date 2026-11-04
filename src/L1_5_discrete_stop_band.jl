"""
# L1-5 — Discrete stop band of a line array (configuration A, vertical section)

## Relation to the paper

Identical lossless line resonators with spacing a on the bare host have Bloch
bands given by

    1/α(ω) = 𝒮(q, ω),   𝒮(q, ω) = (1/a) Σₙ 1/𝒟₀(q + 2πn/a, ω)          (eq:bloch_condition)

Because 𝒟₀ → −∞ as k → 0 (fluid incompressibility), the band above the lower
band starts at the zone centre at

    1/(ω_Γ²m_r) = 1/k_r + 𝒮′(0, ω_Γ)/a,   ω_Γ ≃ ω_r/√(1 + Π_L)          (eq:lat_gamma)

and the lower band ends at the zone boundary q = π/a at ω_X, the root of
1/α = 𝒮(π/a, ω). The complete stop band [ω_X, ω_Γ] is set by the discrete
geometry (`subsubsec:lat_long_wave`, `fig:lat_bands`): for the parameters of
Table 1 and a = 1 m, [9.646, 9.986] (T_ρ = 0) and [9.916, 9.990]
(T_ρ = 9.8). The homogenised model has no complete stop band; its anticrossing
width (0.163, 0.138) and the band edges obtained by truncating the local
branches at qa = π (Liu et al. 2025: [ω₋(π/a), ω_r]) must *not* be reproduced.

Verification test matrix: §L1.5.

## Problem

Configuration A without radiation: a periodic channel (`is_periodic`) whose
whole free surface is covered by the plate, rigid bottom, no sponges. A
supercell of M cells of length a carries M resonators at x_n = (n − ½)a.
Bloch BCs are not needed: the supercell admits the Bloch wavenumbers
q = 2πj/(Ma), j = 0, …, M − 1, and the forcing selects one of them. Each
resonator position carries a patch load (width two elements, centred on the
node of the attachment) with phase e^{iqx_n}; in the lossless closed cell this
load excites only the Bloch-q modes. The band frequencies at q are the poles of
the projected response

    g(ω) = (1/M) Σₙ e^{−iqx_n} q_n(ω),

found as the roots of 1/g, a smooth function of ω near each pole, by secant
iteration started at the lattice-sum reference. Γ is M = 1, j = 0 and X is
M = 2, j = 1. The bare host is checked the same way (no resonators): the pole
at q = π/a is the frequency ω₀(π/a) of the bare branch.

## Observables

- ω_Γ, ω_X and the gap ω_Γ − ω_X, against `omega_gamma_1d`, `omega_X_1d`.
- Band frequencies at the supercell wavenumbers q = 2πj/(M_band a),
  j = 1, …, M_band/2 − 1, inside `band_window` ω_r, against `bloch_roots_1d`.
- Bare host at q = π/a against `bare_frequency`.
- Values that must *not* match: the truncated local edges [ω₋(π/a), ω_r]
  (`local_branches` with n₀ = 1/a) and the anticrossing width.

## Reproduce

    julia --project -O1 scripts/L1-configA-vertical/L1-5_discrete_stop_band.jl
"""
module L1_5

using Gridap
using Parameters
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH
using HydroElasticFEM: map_vertical_GP_for_const_dep

using ..References: HostSpec, ResonatorSpec, k0, resonator_from_frequency, omega_gamma_1d, omega_X_1d,
                    bloch_roots_1d, bare_frequency, local_branches, anticrossing_width,
                    lattice_compliance_number_1d
import ..L1_1
import ..L1_3: secant_root

export L1_5_Params, VARIABLE, fixed_hash, run_case

"""
Parameters of one L1-5 case. The host, discretisation and vertical-mesh field
names are those of `L1_1_Params`; `elements_per_cell` replaces
`elements_per_wavelength` (the mesh is uniform in x, with an integer number of
elements per cell so that every resonator sits on a node).
"""
@with_kw struct L1_5_Params
    # ── Variable: what the study sweeps (names the record) ───────────────────
    T_ρ::Float64 = 0.0                    # pre-tension / ρ (m³/s²)
    elements_per_cell::Int = 8            # horizontal elements per cell of length a
    order::Int = 2                        # polynomial order of ϕ and η

    # ── Fixed: host, set P1 of the test matrix = Table 1 of the paper ────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49

    # ── Fixed: lattice ────────────────────────────────────────────────────────
    a::Float64 = 1.0                      # spacing (m)
    m_ρr::Float64 = 0.01                  # resonator mass / ρ (m²); n₀m_ρr = m_ρr/a
    ω_r::Float64 = 10.0                   # natural frequency (rad/s)

    # ── Fixed: forcing ────────────────────────────────────────────────────────
    F::Float64 = 1.0                      # patch force / ρ per resonator
    source_elements::Int = 2              # patch width in elements

    # ── Fixed: band sampling ──────────────────────────────────────────────────
    M_band::Int = 8                       # supercell for the interior band samples
    band_window::Tuple{Float64,Float64} = (0.8, 1.2)   # band samples in this range of ω/ω_r

    # ── Fixed: discretisation (vertical rule of L1-1) ────────────────────────
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0

    # ── Fixed: root finding ──────────────────────────────────────────────────
    start::Float64 = 1e-4                 # secant started at ω_ref(1 ± start)
    xtol::Float64 = 1e-13                 # secant tolerance (relative)
    accept::Float64 = 1e-2                # roots further than this (relative) from ω_ref are rejected
end

"Fields of `L1_5_Params` varied by the L1-5 study (they name the records)."
const VARIABLE = (:T_ρ, :elements_per_cell, :order)

"Short hash of the fixed parameters, appended to the record name."
function fixed_hash(p::L1_5_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L1_5_Params) if !(f in VARIABLE))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

"Relative residual |Re(1/g)|/|ω ∂_ω Re(1/g)| below which a pole is accepted although the secant steps still jitter (round-off)."
const ROOT_TOL = 1e-9

"Lossless resonator of the case."
resonator(p) = resonator_from_frequency(p.m_ρr, p.ω_r)

"""
    vertical_mesh(p, he)

The vertical rule of L1-1 (`L1_1.vertical_mesh`) with the elements per plate
wavelength at ω_r implied by `elements_per_cell`.
"""
function vertical_mesh(p, he::Real)
    λ = 2π / k0(L1_1.host(p), p.ω_r)
    q = (H=p.H, top_fraction=p.top_fraction, max_ratio=p.max_ratio, ratio_elements=p.ratio_elements,
         elements_per_wavelength=λ / he)
    return L1_1.vertical_mesh(q, he)
end

# ─── FE problem (HydroElasticFEM.jl) ────────────────────────────────────────

"""
    solve_cell(p, ω; L, nx, x_load, F_load, x_res=[], resonators=[])

Periodic channel of length `L` (`nx` elements), plate over the whole surface,
patch loads of complex amplitude `F_load` centred at `x_load`, line
resonators at `x_res`. `p` is any parameter struct with the host,
discretisation and vertical-mesh fields of `L1_5_Params`. Returns the plate
deflection, the resonator displacements, the number of DOFs and the vertical mesh.
"""
function solve_cell(p, ω::Real; L::Real, nx::Int, x_load::AbstractVector{<:Real}, F_load::AbstractVector,
                    x_res::AbstractVector{<:Real}=Float64[], resonators::AbstractVector{ResonatorSpec}=ResonatorSpec[])
    length(resonators) == length(x_res) || throw(DimensionMismatch("one position per resonator"))
    he = L / nx
    vm = vertical_mesh(p, he)
    H = p.H
    vertical_map(x) = VectorValue(x[1], map_vertical_GP_for_const_dep(x[2] - H, vm.ratio, vm.ny, H; dbgmsg=false))
    N = length(resonators)
    tank = G.TankDomain(L=L, H=H, nx=nx, ny=vm.ny, map=vertical_map, is_periodic=(true, false),
                        structure_domains=[G.StructureDomain(L=L, x₀=[0.0, 0.0], domain_symbol=:Γs)],
                        resonator_domains=G.ResonatorDomain[G.ResonatorDomain(location=[x, 0.0], delta_symbol=Symbol("δ$i"))
                                                            for (i, x) in enumerate(x_res)])

    fe(; kw...) = PH.FESpaceConfig(; order=p.order, vector_type=Vector{ComplexF64}, kw...)
    fluid = P.PotentialFlow(g=p.g, ρw=p.ρ, fe=fe(), space_domain_symbol=:Ω)
    plate = P.TensionedEulerBernoulliBeam(L=L, mᵨ=p.m_ρ, EIᵨ=p.C_ρ, Tᵨ=p.T_ρ, g=p.g, symbol=:η,
                                          fe=fe(γ=p.γ_penalty), space_domain_symbol=:Γs)
    entities = P.PhysicsParameters[fluid, plate]
    if N > 0   # HydroElasticFEM takes dimensional mass, stiffness and damping
        push!(entities, P.resonator_array(N, [r.m_ρ * p.ρ for r in resonators], [r.k_ρ * p.ρ for r in resonators],
                                          [r.c_ρ * p.ρ for r in resonators];
                                          delta_domain_symbols=[Symbol("δ$i") for i in 1:N], ρw=p.ρ,
                                          host_domain_symbol=:Γs,
                                          fe=PH.FESpaceConfig(space_type=VectorValue{1,Float64},
                                                              vector_type=Vector{ComplexF64})))
    end

    b = p.source_elements * he
    function load(x)
        s = zero(ComplexF64)
        for (xl, Fl) in zip(x_load, F_load)
            abs(x[1] - xl) < b / 2 && (s += Fl / b)
        end
        return s
    end
    # `rhs_fn` returns one load per field, in the field order of the FieldMap (as L1-1)
    field_order(fmap) = [s for (s, _) in sort!(collect(fmap._map); by=last)]
    rhs_fn = (ctx, fmap) -> [sym == :η ? load : 0.0 for sym in field_order(fmap)]

    problem = S.build_problem(tank, entities, PH.FreqDomainConfig(ω=ω); rhs_fn=rhs_fn)
    result = S.simulate(problem)
    ηh = result.solution[result.fmap[:η]]
    q = ComplexF64[first(get_free_dof_values(result.solution[result.fmap[Symbol("q$i")]])) for i in 1:N]
    return (; ηh, q, ndofs=length(get_free_dof_values(result.solution)), vm)
end

"""
    bloch_response(p, ω; M, j, res=resonator(p))

Projected response g(ω) = (1/M) Σₙ e^{−iqx_n} uₙ of a supercell of `M` cells
of length `p.a` loaded with phase e^{iqx_n}, q = 2πj/(Ma); uₙ is the resonator
displacement, or the plate deflection at x_n when `res === nothing`
(bare host). Also returns the number of DOFs and vertical elements.
"""
function bloch_response(p, ω::Real; M::Int, j::Int, res::Union{Nothing,ResonatorSpec}=resonator(p))
    a = p.a
    q = 2π * j / (M * a)
    xs = [(n - 0.5) * a for n in 1:M]
    phase = cis.(q .* xs)
    rs = res === nothing ? ResonatorSpec[] : fill(res, M)
    sol = solve_cell(p, ω; L=M * a, nx=M * p.elements_per_cell, x_load=xs, F_load=p.F .* phase,
                     x_res=res === nothing ? Float64[] : xs, resonators=rs)
    u = res === nothing ? L1_1.sample(sol.ηh, xs) : sol.q
    return (g=sum(conj.(phase) .* u) / (M * p.F), ndofs=sol.ndofs, ny=sol.vm.ny)
end

"""
    band_frequency(p, ω_ref; M, j, res=resonator(p))

Pole of `bloch_response` next to the reference frequency `ω_ref`: root of
Re(1/g) by secant iteration from ω_ref(1 ± `start`). Returns the root (NaN if
neither the step criterion `xtol` nor the residual criterion
|Re(1/g)|/|ω ∂_ω Re(1/g)| ≤ `ROOT_TOL` holds, or if it lies further than
`accept` from ω_ref), the relative residual, the number of FE solves,
|Im(1/g)/Re(1/g)′| at the root (≈ 0 for the lossless problem, a check of the
projection), and the DOFs.
"""
function band_frequency(p, ω_ref::Real; M::Int, j::Int, res::Union{Nothing,ResonatorSpec}=resonator(p))
    cache = Dict{Float64,Any}()
    resp(ω) = get!(() -> bloch_response(p, ω; M=M, j=j, res=res), cache, ω)
    f(ω) = real(1 / resp(ω).g)
    s = secant_root(f, ω_ref * (1 - p.start), ω_ref * (1 + p.start); xtol=p.xtol, maxiter=40)
    ω = s.root
    slope = (f(ω * (1 + 1e-7)) - f(ω * (1 - 1e-7))) / (2e-7 * ω)
    # at round-off level the secant steps jitter above xtol; the root is accepted when its residual
    # fixes ω to `ROOT_TOL` (relative), whether or not the step criterion was met
    residual = abs(f(ω)) / abs(slope * ω)
    ok = (s.converged || residual <= ROOT_TOL) && abs(ω / ω_ref - 1) <= p.accept
    leak = abs(imag(1 / resp(ω).g)) / abs(slope * ω)
    r = resp(ω)
    return (omega=ok ? ω : NaN, solves=length(cache), leak=leak, residual=residual, ndofs=r.ndofs, ny=r.ny)
end

"""
    run_case(p) -> Dict

Band edges, interior band samples and the bare-host check of case `p`, with
their lattice-sum references and the values that must not be matched.
"""
function run_case(p::L1_5_Params)
    h = L1_1.host(p)
    r = resonator(p)
    a = p.a

    ωΓ_ref = omega_gamma_1d(h, r, a)
    ωX_ref = omega_X_1d(h, r, a; ωmin=p.band_window[1] * p.ω_r)
    Γ = band_frequency(p, ωΓ_ref; M=1, j=0)
    X = band_frequency(p, ωX_ref; M=2, j=1)

    # interior band samples q = 2πj/(M_band a), every band inside the window
    ωs = range(p.band_window[1] * p.ω_r, p.band_window[2] * p.ω_r; length=801)
    qa, ω_ref, ω_h, leak = Float64[], Float64[], Float64[], Float64[]
    for j in 1:(p.M_band ÷ 2 - 1)
        q = 2π * j / (p.M_band * a)
        for ω0 in bloch_roots_1d(h, r, q, a, ωs)
            s = band_frequency(p, ω0; M=p.M_band, j=j)
            push!(qa, q * a); push!(ω_ref, ω0); push!(ω_h, s.omega); push!(leak, s.leak)
        end
    end

    # bare host at the zone boundary
    bare_ref = bare_frequency(h, π / a)
    bare = band_frequency(p, bare_ref; M=2, j=1, res=nothing)

    # values that must not be matched (homogenised model, n₀ = 1/a)
    liu_lower = local_branches(h, r, 1 / a, π / a).lower

    return Dict{String,Any}(
        "omega_gamma_h" => Γ.omega, "omega_gamma_ref" => ωΓ_ref,
        "omega_X_h" => X.omega, "omega_X_ref" => ωX_ref,
        "band_qa" => qa, "band_omega_h" => ω_h, "band_omega_ref" => ω_ref, "band_leak" => leak,
        "bare_omega_h" => bare.omega, "bare_omega_ref" => bare_ref,
        "liu_lower" => liu_lower, "liu_upper" => p.ω_r,
        "anticrossing_width" => anticrossing_width(h, r, 1 / a),
        "Pi_L" => lattice_compliance_number_1d(h, r, a),
        "omega_gamma_PiL" => p.ω_r / sqrt(1 + lattice_compliance_number_1d(h, r, a)),
        "leak" => max(Γ.leak, X.leak), "fe_solves" => Γ.solves + X.solves + bare.solves,
        "ndofs_cell" => Γ.ndofs, "ny" => Γ.ny, "element_size" => a / p.elements_per_cell,
        "lambda_p" => 2π / k0(h, p.ω_r),
    )
end

end # module L1_5
