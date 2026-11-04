"""
# L1-7 — Homogenised resonator field and the lattice-compliance number (configuration A, vertical section)

## Relation to the paper

(i) A uniform population of n₀ resonators per unit length, smeared into a
field q(x) on the plate (homogenised model), gives the local dispersion law
of Section 3, whose lossless real branches are (`eq:omega_k_x_closed_form`)

    ω±(k) = √s±,   s± = (Σ ± √(Σ² − 4ω_r²K(k)(m_ρ + M_f(k)))) / (2(m_ρ + M_f(k))),
    Σ = K(k) + ω_r²(m_ρ + M_f(k) + M_r),   K = g + T_ρk² + C_ρk⁴,
    M_f = 1/(k tanh kH),   M_r = n₀m_ρ,r.

(i-b) Because M_f → ∞ as k → 0 (fluid dilution, `subsubsec:fluid_dilution`),
ω₊(k) → ω_r⁺ as k → 0 while ω₋(k) → ω_r⁻ as k → ∞: every ω ≠ ω_r has one
real wavenumber, on ω₋ below ω_r and on ω₊ above. The homogenised floating
plate has no complete stop band.

(ii) The discrete lattice (spacing a, point resonators) has its upper band
edge at the zone centre, ω_Γ ≃ ω_r/√(1 + Π_L) (`eq:lat_gamma`), with the
lattice-compliance number Π_L = n₀k_ρ,r σ′(0, ω_r) (`eq:PiL`). The estimate
evaluates σ′ at ω_r instead of ω_Γ, so ω_Γ − ω_r/√(1 + Π_L) = O(Π_L²).

Verification test matrix: §L1.7 (frozen in `verification_test_matrix_2026-10-22.md`).

## Problem

(i) Configuration A without radiation, as L1-5: a periodic channel
(`is_periodic`) covered by the plate, rigid bottom, no sponges, plus the
upstream `ResonatorField` (field q on the plate, density n₀, mass, spring;
lossless). At a sampled frequency ω the cell is one wavelength long,
L = 2π/k, with `elements_per_wavelength` elements; the plate carries the
distributed load F e^{ikx}, which in the lossless periodic cell excites
only the Bloch-k mode. The FE wavenumber k_h(ω) is the root in k of
Re(1/g), g(k) = ⟨e^{−ikx} η⟩/F the projected response (a pole of g is a
zero of the FE dispersion function), found by secant iteration started at
the closed-form root. The vertical mesh is fixed at the closed-form root, so
1/g varies smoothly with k (only the horizontal element size changes).

(ii) The L1-5 periodic cells with point resonators (`L1_5.band_frequency`,
one cell, uniform load): ω_Γ for a ∈ {0.5, 1, 2, 3} m at fixed n₀m_ρ,r = 0.01 m,
against `omega_gamma_1d` (lattice sums) and against ω_r/√(1 + Π_L).

## Environment

The upstream `ResonatorField` is on branch `feat/resonator-field` of
HydroElasticFEM.jl (not merged yet). This file is therefore not included in
the package module: the L1-7 script includes it and runs in
`envs/resonator_field` (`julia --project=envs/resonator_field -O1`). After the
PR merges, add `include("L1_7_homogenised_resonators.jl")` to
`src/HydroElasticFEM_xxx2026.jl`, replace `HydroElasticFEM_xxx2026.` by `..`
in the imports below, and run L1-7 in the root environment.

## Reproduce

    julia --project=envs/resonator_field -O1 scripts/L1-configA-vertical/L1-7_homogenised_resonators.jl
"""
module L1_7

using Gridap
using Parameters
using Roots: find_zero, Bisection
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH
using HydroElasticFEM: map_vertical_GP_for_const_dep

using HydroElasticFEM_xxx2026.References: HostSpec, ResonatorSpec, resonator_from_frequency, local_branches,
                                          omega_gamma_1d, lattice_compliance_number_1d, Pi_L_estimate_plate,
                                          D0, k0, lattice_sum_1d, bare_frequency
import HydroElasticFEM_xxx2026.L1_1
import HydroElasticFEM_xxx2026.L1_5
import HydroElasticFEM_xxx2026.L1_3: secant_root

export L1_7_Params, L1_7_Gamma_Params, VARIABLE, VARIABLE_GAMMA, fixed_hash, run_case, homogenised_k

# ─── (i) Homogenised field ──────────────────────────────────────────────────

"""
Parameters of one L1-7 (i) case: the homogenised FE wavenumber at every
sampled frequency of `omegas` for one host and one mesh.
"""
@with_kw struct L1_7_Params
    # ── Variable: what the study sweeps (names the record) ───────────────────
    T_ρ::Float64 = 0.0                    # pre-tension / ρ (m³/s²)
    elements_per_wavelength::Int = 16     # horizontal elements per wavelength 2π/k
    order::Int = 3                        # polynomial order of ϕ, η and q
    set::Symbol = :full                   # frequency set: :full (sweep) or :ladder (convergence subset)

    # ── Fixed: host, set P1 of the test matrix = Table 1 of the paper ────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49

    # ── Fixed: homogenised population ────────────────────────────────────────
    n0::Float64 = 1.0                     # resonators per metre (n₀m_ρ,r = 0.01 m)
    m_ρr::Float64 = 0.01                  # resonator mass / ρ (m²)
    ω_r::Float64 = 10.0                   # natural frequency (rad/s)

    # ── Fixed: frequency samples ─────────────────────────────────────────────
    omega_range::Tuple{Float64,Float64} = (8.0, 12.0)
    n_full::Int = 401                     # :full, ω = ω_r excluded
    omega_ladder::Tuple{Vararg{Float64}} = (8.0, 9.0, 9.5, 9.9, 9.99, 10.01, 10.1, 10.5, 11.0, 12.0)

    # ── Fixed: forcing and projection ────────────────────────────────────────
    F::Float64 = 1.0                      # load amplitude / ρ (per unit length)
    samples_per_element::Int = 2          # deflection samples for the projection

    # ── Fixed: discretisation (vertical rule of L1-1) ────────────────────────
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0

    # ── Fixed: root finding ──────────────────────────────────────────────────
    start::Float64 = 1e-4                 # secant started at k_ref(1 ± start)
    xtol::Float64 = 1e-13                 # secant tolerance (relative)
    accept::Float64 = 1e-2                # roots further than this (relative) from k_ref are rejected
end

"Fields of `L1_7_Params` varied by the study (they name the records)."
const VARIABLE = (:T_ρ, :elements_per_wavelength, :order, :set)

"Short hash of the fixed parameters, appended to the record name."
function fixed_hash(p)
    fixed = Tuple(getfield(p, f) for f in fieldnames(typeof(p)) if !(f in _variable(p)))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

"Relative residual below which a root is accepted although the secant steps still jitter (round-off)."
const ROOT_TOL = 1e-9

"Lossless resonator of the case."
resonator(p) = resonator_from_frequency(p.m_ρr, p.ω_r)

"Sampled frequencies of the case (ω_r excluded)."
function omegas(p::L1_7_Params)
    p.set === :ladder && return collect(p.omega_ladder)
    ωs = collect(range(p.omega_range...; length=p.n_full))
    return filter(ω -> abs(ω / p.ω_r - 1) > 1e-12, ωs)
end

"""
    homogenised_k(h, r, n0, ω)

Real wavenumber of the homogenised law at ω ≠ ω_r: root of ω₋(k) = ω below
ω_r, of ω₊(k) = ω above (`eq:omega_k_x_closed_form`, `local_branches`). Both
branches are monotone in k, so the root is unique. Returns `(k, branch)`.
"""
function homogenised_k(h::HostSpec, r::ResonatorSpec, n0::Real, ω::Real)
    lower = ω < sqrt(r.k_ρ / r.m_ρ)
    br(k) = (b = closed_form_branches(h, r, n0, k); lower ? b.lower : b.upper)
    k = find_zero(k -> br(k) - ω, (1e-6, 1e3), Bisection(); xtol=0.0, xrtol=1e-15)
    return (k=k, branch=lower ? :lower : :upper)
end

"""
    closed_form_branches(h, r, n0, k)

ω±(k) of `eq:omega_k_x_closed_form`, with s₋ written as 2ω_r²K/(Σ + √(Σ² − 4ω_r²KM))
(the same root, without the cancellation of Σ − √(…) at large k). Equal to
`References.local_branches` where that is well conditioned.
"""
function closed_form_branches(h::HostSpec, r::ResonatorSpec, n0::Real, k::Real)
    K = h.g + h.T_ρ * k^2 + h.C_ρ * k^4
    M = h.m_ρ + 1 / (k * tanh(k * h.H))
    ωr2 = r.k_ρ / r.m_ρ
    Σ = K + ωr2 * (M + n0 * r.m_ρ)
    d = sqrt(Σ^2 - 4ωr2 * K * M)
    return (lower=sqrt(2ωr2 * K / (Σ + d)), upper=sqrt((Σ + d) / (2M)))
end

"Vertical mesh of the L1-1 rule for horizontal element size `he` and wavelength `λ`."
function vertical_mesh(p, he::Real, λ::Real)
    q = (H=p.H, top_fraction=p.top_fraction, max_ratio=p.max_ratio, ratio_elements=p.ratio_elements,
         elements_per_wavelength=λ / he)
    return L1_1.vertical_mesh(q, he)
end

"""
    solve_field(p, ω, k; vm)

Periodic cell of length 2π/k (`elements_per_wavelength` elements, vertical
mesh `vm`) covered by the plate with a uniform `ResonatorField`, loaded by
F e^{ikx}. Returns g = ⟨e^{−ikx}η⟩/F (mean over equispaced samples), the
same projection of q, and the number of DOFs.
"""
function solve_field(p::L1_7_Params, ω::Real, k::Real; vm)
    L = 2π / k
    nx = p.elements_per_wavelength
    H = p.H
    vertical_map(x) = VectorValue(x[1], map_vertical_GP_for_const_dep(x[2] - H, vm.ratio, vm.ny, H; dbgmsg=false))
    tank = G.TankDomain(L=L, H=H, nx=nx, ny=vm.ny, map=vertical_map, is_periodic=(true, false),
                        structure_domains=[G.StructureDomain(L=L, x₀=[0.0, 0.0], domain_symbol=:Γs)])
    fe(; kw...) = PH.FESpaceConfig(; order=p.order, vector_type=Vector{ComplexF64}, kw...)
    fluid = P.PotentialFlow(g=p.g, ρw=p.ρ, fe=fe(), space_domain_symbol=:Ω)
    plate = P.TensionedEulerBernoulliBeam(L=L, mᵨ=p.m_ρ, EIᵨ=p.C_ρ, Tᵨ=p.T_ρ, g=p.g, symbol=:η,
                                          fe=fe(γ=p.γ_penalty), space_domain_symbol=:Γs)
    r = resonator(p)   # HydroElasticFEM takes dimensional mass and stiffness per resonator
    field = P.ResonatorField(M=r.m_ρ * p.ρ, K=r.k_ρ * p.ρ, n=p.n0, ρw=p.ρ, symbol=:q,
                             space_domain_symbol=:Γs, fe=fe())
    entities = P.PhysicsParameters[fluid, plate, field]

    load(x) = p.F * cis(k * x[1])
    field_order(fmap) = [s for (s, _) in sort!(collect(fmap._map); by=last)]
    rhs_fn = (ctx, fmap) -> [sym == :η ? load : 0.0 for sym in field_order(fmap)]

    result = S.simulate(S.build_problem(tank, entities, PH.FreqDomainConfig(ω=ω); rhs_fn=rhs_fn))
    ηh = result.solution[result.fmap[:η]]
    qh = result.solution[result.fmap[:q]]
    ns = p.samples_per_element * nx
    xs = [(j - 0.5) * L / ns for j in 1:ns]
    w = cis.(-k .* xs)
    gη = sum(w .* L1_1.sample(ηh, xs)) / (ns * p.F)
    gq = sum(w .* L1_1.sample(qh, xs)) / (ns * p.F)
    return (g=gη, gq=gq, ndofs=length(get_free_dof_values(result.solution)))
end

"""
    field_wavenumber(p, ω)

FE wavenumber of the homogenised plate at ω: root of Re(1/g(k)) by secant
iteration from k_ref(1 ± `start`), with the vertical mesh fixed at k_ref.
Returns the root (NaN if neither the step criterion `xtol` nor the residual
criterion |Re(1/g)|/|k ∂_k Re(1/g)| ≤ `ROOT_TOL` holds, or if it lies further
than `accept` from k_ref), the reference root and branch, the residual, the
leakage |Im(1/g)|/|k ∂_k Re(1/g)|, the ratio q/η of the Bloch amplitudes
against K/(K − ω²M), the number of solves and the DOFs.
"""
function field_wavenumber(p::L1_7_Params, ω::Real)
    h = L1_1.host(p)
    r = resonator(p)
    ref = homogenised_k(h, r, p.n0, ω)
    kr = ref.k
    λ = 2π / kr
    vm = vertical_mesh(p, λ / p.elements_per_wavelength, λ)
    cache = Dict{Float64,Any}()
    resp(k) = get!(() -> solve_field(p, ω, k; vm=vm), cache, k)
    f(k) = real(1 / resp(k).g)
    s = secant_root(f, kr * (1 - p.start), kr * (1 + p.start); xtol=p.xtol, maxiter=16)
    k = s.root
    # slope from the two closest evaluations (no extra solves)
    ks = sort(collect(keys(cache)); by=x -> abs(x - k))
    k1, k2 = ks[1] == k ? (ks[2], ks[3]) : (ks[1], ks[2])
    slope = (f(k1) - f(k2)) / (k1 - k2)
    residual = abs(f(k)) / abs(slope * k)
    ok = (s.converged || residual <= ROOT_TOL) && abs(k / kr - 1) <= p.accept && isfinite(k)
    res = resp(k)
    leak = abs(imag(1 / res.g)) / abs(slope * k)
    q_ratio = abs((res.gq / res.g) / (r.k_ρ / (r.k_ρ - ω^2 * r.m_ρ)) - 1)
    return (k=ok ? k : NaN, k_ref=kr, branch=ref.branch, residual=residual, leak=leak, q_ratio=q_ratio,
            solves=length(cache), ndofs=res.ndofs, ny=vm.ny)
end

"""
    run_case(p::L1_7_Params) -> Dict

FE wavenumber k_h(ω) at every sampled ω, with the closed-form root, the
branch, the residual, leakage and q/η check per frequency.
"""
function run_case(p::L1_7_Params)
    ωs = omegas(p)
    rows = [field_wavenumber(p, ω) for ω in ωs]
    h = L1_1.host(p)
    r = resonator(p)
    # closed-form frequency at the FE wavenumber (the branch of the reference)
    ω_at_kh = [isnan(x.k) ? NaN :
               (b = closed_form_branches(h, r, p.n0, x.k); x.branch === :lower ? b.lower : b.upper) for x in rows]
    return Dict{String,Any}(
        "omega" => ωs, "k_h" => [x.k for x in rows], "k_ref" => [x.k_ref for x in rows],
        "branch" => [string(x.branch) for x in rows], "omega_closed_at_k_h" => ω_at_kh,
        "residual" => [x.residual for x in rows], "leak" => [x.leak for x in rows],
        "q_ratio" => [x.q_ratio for x in rows], "solves" => [x.solves for x in rows],
        "ndofs" => [x.ndofs for x in rows], "ny" => [x.ny for x in rows],
        "k_bare_at_omega_r" => k0(h, p.ω_r),
    )
end

# ─── (ii) Zone-centre edge of the point lattice against Π_L ─────────────────

"""
Parameters of one L1-7 (ii) case: ω_Γ of the discrete line array of spacing
`a` at fixed loading n₀m_ρ,r = `loading` (m_ρ,r = loading·a, k_ρ,r = m_ρ,r ω_r²),
with the L1-5 periodic cell.
"""
@with_kw struct L1_7_Gamma_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    a::Float64 = 1.0                      # spacing (m)
    T_ρ::Float64 = 0.0
    elements_per_metre::Int = 32          # horizontal elements per metre (a·elements_per_metre per cell)
    order::Int = 3

    # ── Fixed ─────────────────────────────────────────────────────────────────
    loading::Float64 = 0.01               # n₀m_ρ,r (m)
    ω_r::Float64 = 10.0
end

"Fields of `L1_7_Gamma_Params` varied by the study."
const VARIABLE_GAMMA = (:a, :T_ρ, :elements_per_metre, :order)

_variable(::L1_7_Params) = VARIABLE
_variable(::L1_7_Gamma_Params) = VARIABLE_GAMMA

"The L1-5 parameters of a (ii) case (host and discretisation fields of L1-5)."
function l15_params(p::L1_7_Gamma_Params)
    n_c = round(Int, p.a * p.elements_per_metre)
    n_c ≈ p.a * p.elements_per_metre || throw(ArgumentError("a·elements_per_metre must be an integer"))
    return L1_5.L1_5_Params(T_ρ=p.T_ρ, elements_per_cell=n_c, order=p.order, a=p.a, m_ρr=p.loading * p.a,
                            ω_r=p.ω_r)
end

"""
    gamma_reference(h, r, a; npts=8001)

Lattice-sum ω_Γ: the root of 1/(ω²m_ρ,r) − 1/k_ρ,r − σ′(0, ω)/a (`eq:lat_gamma`,
exact form) closest to the estimate ω_r/√(1 + Π_L), on a scan of
[0.6ω_r, 1.4ω_r]; sign changes across the poles at the folded bare
frequencies ω₀(2πn/a) are rejected. Equal to `omega_gamma_1d` for
2π/a > k_c (Π_L > 0, root below ω_r). For 2π/a < k_c (a = 3 m here) the
folded bare branch ω₀(2π/a) lies below ω_r, Π_L < 0 and the root near ω_r
lies above it, outside the bracket of `omega_gamma_1d`.
"""
function gamma_reference(h::HostSpec, r::ResonatorSpec, a::Real; npts::Int=8001)
    ωr = sqrt(r.k_ρ / r.m_ρ)
    ωe = ωr / sqrt(1 + lattice_compliance_number_1d(h, r, a))
    f(ω) = 1 / (ω^2 * r.m_ρ) - 1 / r.k_ρ - lattice_sum_1d(h, 0.0, ω, a; exclude_zero=true)
    ωs = range(0.6ωr, 1.4ωr; length=npts)
    fs = f.(ωs)
    best = NaN
    for i in 1:(npts - 1)
        (isfinite(fs[i] * fs[i + 1]) && fs[i] * fs[i + 1] < 0) || continue
        ω = find_zero(f, (ωs[i], ωs[i + 1]), Bisection())
        abs(f(ω)) < 1e-6 * (abs(fs[i]) + abs(fs[i + 1])) || continue
        (isnan(best) || abs(ω - ωe) < abs(best - ωe)) && (best = ω)
    end
    isnan(best) && error("no zone-centre root near ω_r for a = $(a)")
    return best
end

"""
    run_case(p::L1_7_Gamma_Params) -> Dict

FE ω_Γ (one L1-5 cell, uniform load), the lattice-sum ω_Γ, Π_L (lattice sums
and the bending-only estimate k_ρ,r a³/(720 C_ρ)) and ω_r/√(1 + Π_L).
"""
function run_case(p::L1_7_Gamma_Params)
    q = l15_params(p)
    h = L1_1.host(q)
    r = L1_5.resonator(q)
    ωΓ_ref = gamma_reference(h, r, p.a)
    Γ = L1_5.band_frequency(q, ωΓ_ref; M=1, j=0)
    Π = lattice_compliance_number_1d(h, r, p.a)
    return Dict{String,Any}(
        "omega_gamma_h" => Γ.omega, "omega_gamma_ref" => ωΓ_ref, "Pi_L" => Π,
        "Pi_L_estimate" => Pi_L_estimate_plate(r, h, p.a), "omega_gamma_PiL" => p.ω_r / sqrt(1 + Π),
        "residual" => Γ.residual, "leak" => Γ.leak, "fe_solves" => Γ.solves, "ndofs_cell" => Γ.ndofs,
        "ny" => Γ.ny, "elements_per_cell" => q.elements_per_cell,
        "two_pi_over_a" => 2π / p.a, "k_c" => k0(h, p.ω_r), "omega_bare_2pi_over_a" => bare_frequency(h, 2π / p.a),
    )
end

end # module L1_7
