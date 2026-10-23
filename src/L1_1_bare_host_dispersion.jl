"""
# L1-1 — Bare host: dispersion and Green function (configuration A, vertical section)

## Relation to the paper

The bare host is the floating plate without resonators. Its symbol

    D₀(k, ω) = ρg + (1 − iωτ)Tk² + (1 − iωτ_b)Dk⁴ − ω²[m + ρ/(k tanh kH)]     (eq:host_symbol)

defines the propagating root k₀ (D₀(k₀) = 0, the flexural–gravity wave), and in
the vertical section the response to a unit point force is the Green function

    G(x) = i Σₘ e^{ikₘ|x|}/D₀′(kₘ),   Im G(x) = cos(k₀x)/D₀′(k₀)             (eq:lat_green_1d)

whose propagating term radiates two waves of amplitude 1/D₀′(k₀), carrying the
power ½ω Im G(0) (eq:point_power, eq:energy_flux). G, k₀ and γ_r = Im G(0) are
the building blocks of every resonator result of Section 4
(`sec:discrete_lattice`): a resonator enters only through 1/α − G(0). This test
checks that the monolithic FE formulation (eq:discreteform_freq, Section
`sec:numerical_formulation`) reproduces them. The code uses density-normalised
quantities (D₀/ρ, F/ρ), so G here is ρ times the G of the paper.

Verification test matrix: §L1.1 (with the γ_r part of §L1.2).

## Problem

Configuration A, vertical section: the plate covers the whole free surface of
a channel of depth H, with rigid walls at both ends.

    wall ── sponge ── buffer ── │ probe ··· source ··· probe │ ── buffer ── sponge ── wall
                                └──────── window ─────────┘

- Source: a density-normalised force F spread uniformly over a patch of width
  b = `source_elements` elements, centred on a mesh node. HydroElasticFEM's
  `rhs_fn` takes distributed loads only; the exact patch response is
  G_b(x) = i Σₘ sinc(kₘb/2) e^{ikₘ|x|}/D₀′(kₘ) for |x| ≥ b/2.
- Sponges: viscoelastic bending EI(x) = C_ρ(1 − i s(x)) over
  `sponge_wavelengths` plate wavelengths, s ramping to `sponge_strength`.
  They act on bending only, so they absorb poorly where tension or gravity
  dominate (long waves); the reflected waves are removed in post-processing.
- Buffer: the slowest fluid evanescent mode, e^{−κ₁|x|} with 1/κ₁ ≈ 2H/π, is
  not damped by the sponges and is reflected by them and by the walls. The
  buffer keeps the source `gap_decays`/κ₁ away from the sponges (with ±4 λ_p
  and a one-wavelength buffer, the reflection left a mesh-independent error of
  9e-6 in Re G).
- Channel length: at low ω the bending-only sponges barely absorb, and the
  walled channel is a lossless cavity, resonant at L = nλ_p/2 (with
  L = 22λ_p the response was 10⁴ times too large at ω = 0.3). The buffers
  are lengthened so that L = (integer + `length_fraction`)·λ_p, midway
  between two resonances for `length_fraction` = 1/4.

## Observables (one frequency, one mesh)

1. Discrete wavenumber k_h: least-squares fit, on probe windows on either
   side of the source, of A⁺e^{ikx} + A⁻e^{−ikx} plus the flexural pair and a
   few evanescent modes decaying from the window ends (`fit_waves`), with k
   fitted. Error |k_h/k₀ − 1|.
2. Radiative width from the far field: the source is transparent to waves
   coming back from the sponges, so the radiated amplitude is
   A_out − A_in(other side), and γ_h = A/(iF sinc(k₀b/2)). Error against
   γ_r = 1/D₀′(k₀).
3. Near field: plate deflection between the patch edge and the probes, minus
   the propagating waves coming back from the sponges, against F·G_b(x − x_s);
   real and imaginary parts separately (Re G sets the renormalised resonance
   of a resonator, L1-3).

## Parameters

`L1_1_Params` holds every parameter of a case. The fields listed in
`VARIABLE` are the ones a study varies; they name the simulation record. All
other fields are fixed for the study; they are stored in the record and a hash
of them is appended to the record name, so changing a fixed value never
reuses an old record.

## Reproduce

    julia --project -O1 scripts/L1-configA-vertical/L1-1_bare_host_dispersion.jl
"""
module L1_1

using Gridap
using Parameters
using LinearAlgebra: norm
using Roots: find_zero, Bisection
import HydroElasticFEM.Geometry as G
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH
using HydroElasticFEM: map_vertical_GP_for_const_dep

using ..References: HostSpec, ResonatorSpec, k0, kappa1, host_roots, gamma_r, green_1d_patch, bare_frequency,
                    added_mass

export L1_1_Params, VARIABLE, fixed_hash, run_case

"""
Parameters of one L1-1 case. Lengths of the layout are in plate wavelengths
λ_p = 2π/k₀(ω) unless stated otherwise.
"""
@with_kw struct L1_1_Params
    # ── Variable: what a study sweeps (names the record) ──────────────────────
    ω::Float64 = 10.0                     # frequency (rad/s)
    elements_per_wavelength::Int = 20     # horizontal elements per λ_p
    order::Int = 2                        # polynomial order of ϕ and η
    T_ρ::Float64 = 0.0                    # pre-tension / ρ (m³/s²)

    # ── Fixed: host, set P1 of the test matrix = Table 1 of the paper ────────
    g::Float64 = 9.8                      # gravity (m/s²)
    ρ::Float64 = 1000.0                   # fluid density (kg/m³)
    H::Float64 = 10.0                     # depth (m)
    m_ρ::Float64 = 0.01                   # plate mass / ρ (m)
    C_ρ::Float64 = 0.49                   # bending stiffness D / ρ (m⁵/s²)

    # ── Fixed: forcing ────────────────────────────────────────────────────────
    F::Float64 = 1.0                      # force / ρ (m⁴/s² per unit width)
    source_elements::Int = 2              # patch width b in elements

    # ── Fixed: layout ─────────────────────────────────────────────────────────
    half_window::Float64 = 4.0            # window = source ± half_window
    probe_offset::Float64 = 1.0           # probes start this far from the source
    gap_decays::Float64 = 5.0             # source ≥ gap_decays/κ₁ from the sponges
    length_fraction::Float64 = 0.25       # L/λ_p ∈ ℤ + length_fraction (off the cavity resonances)
    sponge_wavelengths::Float64 = 6.0     # sponge length
    sponge_strength::Float64 = 3.0        # maximum loss factor s
    sponge_power::Int = 2                 # ramp s ∝ ξ^power

    # ── Fixed: discretisation ─────────────────────────────────────────────────
    top_fraction::Float64 = 0.5           # top vertical cell / horizontal size
    max_ratio::Float64 = 1.2              # maximum vertical growth ratio at ratio_elements el./λ_p
    ratio_elements::Int = 20              # max ratio − 1 scales as ratio_elements/elements_per_wavelength
    γ_penalty::Float64 = 6.0              # C/DG interior penalty of the plate

    # ── Fixed: post-processing ────────────────────────────────────────────────
    n_modes::Int = 6                      # non-propagating modes in the wave fit
    n_samples::Int = 400                  # samples per probe / near-field window
end

"Fields of `L1_1_Params` varied by the L1-1 study (they name the records)."
const VARIABLE = (:ω, :T_ρ, :elements_per_wavelength, :order)

"Short hash of the fixed parameters, appended to the record name."
function fixed_hash(p::L1_1_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L1_1_Params) if !(f in VARIABLE))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

"Bare host of the case, in the notation of `References` (any parameter struct with the host fields)."
host(p) = HostSpec(g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.m_ρ, C_ρ=p.C_ρ, T_ρ=p.T_ρ)

# ─── Geometry and mesh ──────────────────────────────────────────────────────

"""
    layout(p)

Absolute geometry: channel length `L`, sponge length `Ls`, buffer, window,
source centre `x_s` (on a mesh node), patch width `b`, number of elements `nx`
and element size `he`. The buffer is the larger of λ_p and the evanescent
gap, lengthened until L/λ_p − `length_fraction` is an integer.
"""
function layout(p::L1_1_Params)
    h = host(p)
    λ = 2π / k0(h, p.ω)
    Ls = p.sponge_wavelengths * λ
    half = p.half_window * λ
    buffer = max(λ, p.gap_decays / kappa1(h, p.ω) - half)   # source ≥ gap_decays/κ₁ from the sponges
    L = λ * (ceil(2 * (Ls + buffer + half) / λ - p.length_fraction) + p.length_fraction)
    buffer = L / 2 - Ls - half
    nx = ceil(Int, L * p.elements_per_wavelength / λ)
    he = L / nx
    x_s = round((L / 2) / he) * he
    return (; λ, L, Ls, buffer, window=(Ls + buffer, L - Ls - buffer), x_s,
            b=p.source_elements * he, nx, he)
end

"""
    vertical_mesh(p, he)

Number of vertical elements and growth ratio of a geometric progression
(finest at the surface) whose top cell is `top_fraction·he` and whose ratio
does not exceed r_max = 1 + (`max_ratio` − 1)·`ratio_elements`/`elements_per_wavelength`.
Shrinking r_max − 1 with he makes every cell, down to the bottom, refine with
the horizontal mesh: with a fixed ratio the deep cells keep their size
(≈ H(r − 1)/r), which left a non-converging error where the wave reaches the
bottom (P2 rate 1.25 at ω ≈ 2, kH ≈ 4).
"""
function vertical_mesh(p, he::Real)
    h_top = p.top_fraction * he
    rmax = 1 + (p.max_ratio - 1) * p.ratio_elements / p.elements_per_wavelength
    p.H <= h_top && return (ny=1, ratio=1.0)
    ny = ceil(Int, log(1 + p.H * (rmax - 1) / h_top) / log(rmax))
    p.H / ny <= h_top && return (ny=ny, ratio=1.0)
    ratio = find_zero(r -> p.H * (r - 1) / (r^ny - 1) - h_top, (1.0 + 1e-12, rmax + 1e-9), Bisection())
    return (ny=ny, ratio=ratio)
end

"Loss factor s(x) of the plate sponges (0 outside them)."
function sponge_loss(p, x::Real, L::Real, Ls::Real)
    ξ = max(Ls - x, x - (L - Ls), 0.0) / Ls
    return p.sponge_strength * ξ^p.sponge_power
end

"""
    viscous_sponges(p, lay)

Optional viscous dashpot sponges (`StructuralDampingZone`) over the same
`Ls` as the bending sponges, for parameter structs with a field
`sponge_viscous` > 0 (none otherwise). The dashpot cᵨσ(x)ηₜ absorbs waves
of every length, unlike the bending sponge, which is transparent to long
waves. Its strength is the loss factor s_v = `sponge_viscous` relative to the
host inertia at the wavenumber k = 2π/λ of the layout:
cᵨ = s_v ω₀(k)(m_ρ + M_f,ρ(k, H)); σ ramps as ξ^`sponge_power`.
"""
function viscous_sponges(p, lay)
    sv = hasproperty(p, :sponge_viscous) ? p.sponge_viscous : 0.0
    sv > 0 || return P.StructuralDampingZone[]
    h = host(p)
    k = 2π / lay.λ
    c = sv * bare_frequency(h, k) * (p.m_ρ + added_mass(k, p.H))
    return [P.StructuralDampingZone(profile=P.ramp_profile(x_start=lay.Ls, x_end=0.0, power=p.sponge_power), cᵨ=c),
            P.StructuralDampingZone(profile=P.ramp_profile(x_start=lay.L - lay.Ls, x_end=lay.L, power=p.sponge_power), cᵨ=c)]
end

# ─── FE problem (HydroElasticFEM.jl) ────────────────────────────────────────

"""
    solve(p; ω=p.ω, lay=layout(p), resonators=ResonatorSpec[], x_res=Float64[])

Build and solve the frequency-domain problem with HydroElasticFEM: potential
flow in the channel (`PotentialFlow`), the plate as a tensioned
Euler–Bernoulli beam with C/DG bending (`TensionedEulerBernoulliBeam`, sponge
through a complex EI(x)), the patch load through `rhs_fn`, and optionally
line resonators (`resonator_array`, mass–spring–dashpot attached to the plate
at `x_res`). `p` is any parameter struct with the host, forcing, sponge and
discretisation fields of `L1_1_Params`; `lay` any layout with the fields of
`layout` (L1-3 passes its own, sized at the resonator frequency, and sweeps
`ω` on that mesh). Returns the plate deflection `ηh`, the resonator
displacements `q`, the layout, the vertical mesh and the number of DOFs.
"""
function solve(p; ω::Real=p.ω, lay=layout(p), resonators::AbstractVector{ResonatorSpec}=ResonatorSpec[],
               x_res::AbstractVector{<:Real}=Float64[])
    length(resonators) == length(x_res) || throw(DimensionMismatch("one position per resonator"))
    vm = vertical_mesh(p, lay.he)
    H, L = p.H, lay.L
    vertical_map(x) = VectorValue(x[1], map_vertical_GP_for_const_dep(x[2] - H, vm.ratio, vm.ny, H; dbgmsg=false))
    N = length(resonators)
    tank = G.TankDomain(L=L, H=H, nx=lay.nx, ny=vm.ny, map=vertical_map, is_periodic=(false, false),
                        structure_domains=[G.StructureDomain(L=L, x₀=[0.0, 0.0], domain_symbol=:Γs)],
                        resonator_domains=G.ResonatorDomain[G.ResonatorDomain(location=[x, 0.0], delta_symbol=Symbol("δ$i"))
                                           for (i, x) in enumerate(x_res)])

    fe(; kw...) = PH.FESpaceConfig(; order=p.order, vector_type=Vector{ComplexF64}, kw...)
    fluid = P.PotentialFlow(g=p.g, ρw=p.ρ, fe=fe(), space_domain_symbol=:Ω)
    EI(x) = p.C_ρ * (1 - im * sponge_loss(p, x[1], L, lay.Ls))
    plate = P.TensionedEulerBernoulliBeam(L=L, mᵨ=p.m_ρ, EIᵨ=EI, Tᵨ=p.T_ρ, g=p.g, symbol=:η,
                                          fe=fe(γ=p.γ_penalty), space_domain_symbol=:Γs,
                                          damping_zones=viscous_sponges(p, lay))

    entities = P.PhysicsParameters[fluid, plate]
    if N > 0   # HydroElasticFEM takes dimensional mass, stiffness and damping
        push!(entities, P.resonator_array(N, [r.m_ρ * p.ρ for r in resonators], [r.k_ρ * p.ρ for r in resonators],
                                          [r.c_ρ * p.ρ for r in resonators];
                                          delta_domain_symbols=[Symbol("δ$i") for i in 1:N], ρw=p.ρ,
                                          host_domain_symbol=:Γs,   # must match the plate's space domain
                                          fe=PH.FESpaceConfig(space_type=VectorValue{1,Float64},
                                                              vector_type=Vector{ComplexF64})))
    end

    load(x) = abs(x[1] - lay.x_s) < lay.b / 2 ? complex(p.F / lay.b) : zero(ComplexF64)
    # `rhs_fn` returns one load per field, in the field order of the FieldMap
    # it receives (HydroElasticFEM's internal `_map`; no public accessor)
    field_order(fmap) = [s for (s, _) in sort!(collect(fmap._map); by=last)]
    rhs_fn = (ctx, fmap) -> [sym == :η ? load : 0.0 for sym in field_order(fmap)]

    problem = S.build_problem(tank, entities, PH.FreqDomainConfig(ω=ω); rhs_fn=rhs_fn)
    result = S.simulate(problem)
    ηh = result.solution[result.fmap[:η]]
    q = ComplexF64[first(get_free_dof_values(result.solution[result.fmap[Symbol("q$i")]])) for i in 1:N]
    return (; ηh, q, lay, vm, ndofs=length(get_free_dof_values(result.solution)))
end

# ─── Post-processing ────────────────────────────────────────────────────────

"Plate deflection sampled at `xs` on the surface z = 0."
sample(ηh, xs) = ComplexF64[ηh(Point(x, 0.0)) for x in xs]

"""
    fit_waves(xs, ηs, roots; x_ref, n_modes, k=nothing)

Least-squares fit of a source-free plate field on a window [xa, xb]:

    η(x) = A⁺e^{ik(x−x_ref)} + A⁻e^{−ik(x−x_ref)} + Σₙ cₙ⁺e^{ikₙ(x−xa)} + cₙ⁻e^{ikₙ(xb−x)},

with kₙ = `roots[2:n_modes+1]` (flexural pair, then evanescent; Im kₙ > 0).
With `k = nothing` the propagating wavenumber is fitted: the residual is
scanned on a grid within ±5 % of k₀ = `roots[1]`, then minimised by
golden-section search. Keep `n_modes` small: deep evanescent modes are
nearly zero across the window and make the fit ill-conditioned.
Returns `(Aplus, Aminus, k, residual)`.
"""
function fit_waves(xs::AbstractVector{<:Real}, ηs::AbstractVector, roots::AbstractVector;
                   x_ref::Real, n_modes::Int, k::Union{Nothing,Real}=nothing)
    xa, xb = extrema(xs)
    ks = roots[2:(n_modes + 1)]
    # decaying mode e^{ikₙd}, d ≥ 0; flushed to zero before it turns subnormal
    # (on long windows at low ω; subnormals make the BLAS return NaN)
    decay(kn, d) = imag(kn) * d > 700 ? zero(ComplexF64) : exp(im * kn * d)
    function lsq(kk)
        cols = [exp.(im * kk .* (xs .- x_ref)), exp.(-im * kk .* (xs .- x_ref))]
        for kn in ks
            push!(cols, decay.(kn, xs .- xa), decay.(kn, xb .- xs))
        end
        M = reduce(hcat, cols)
        c = M \ ηs
        return c, norm(M * c - ηs) / norm(ηs)
    end
    if k === nothing
        kp = real(roots[1])
        grid = range(0.95kp, 1.05kp; length=201)
        i = argmin([last(lsq(kk)) for kk in grid])
        a, b = grid[max(i - 1, 1)], grid[min(i + 1, end)]
        φ = (sqrt(5) - 1) / 2
        while b - a > 1e-13kp
            c1, c2 = b - φ * (b - a), a + φ * (b - a)
            last(lsq(c1)) < last(lsq(c2)) ? (b = c2) : (a = c1)
        end
        k = (a + b) / 2
    end
    c, res = lsq(k)
    return (Aplus=c[1], Aminus=c[2], k=k, residual=res)
end

"""
    run_case(p) -> Dict

Solve one L1-1 case and return the FE observables, their references and the
case metadata (raw values; errors and checks are formed by the script).
"""
function run_case(p::L1_1_Params)
    h = host(p)
    sol = solve(p)
    lay = sol.lay
    x_s, b, λ = lay.x_s, lay.b, lay.λ
    roots = host_roots(h, p.ω)
    kp = real(roots[1])

    # 1–2. far field on both sides: wavenumber and amplitudes
    probe(xa, xb) = (xs = range(xa, xb; length=p.n_samples);
                     fit_waves(xs, sample(sol.ηh, xs), roots; x_ref=x_s, n_modes=p.n_modes))
    right = probe(x_s + p.probe_offset * λ, lay.window[2])
    left = probe(lay.window[1], x_s - p.probe_offset * λ)
    out_R, in_R = right.Aplus, right.Aminus
    out_L, in_L = left.Aminus, left.Aplus
    s = sin(kp * b / 2) / (kp * b / 2)
    γ_R = (out_R - in_L) / (im * s * p.F)
    γ_L = (out_L - in_R) / (im * s * p.F)

    # 3. near field right of the patch, propagating waves from the sponges removed
    xn = collect(range(x_s + b / 2, x_s + p.probe_offset * λ; length=p.n_samples))
    incoming(x) = in_L * exp(im * right.k * (x - x_s)) + in_R * exp(-im * right.k * (x - x_s))
    ηn = sample(sol.ηh, xn) .- incoming.(xn)
    # max(·, b/2): x − x_s falls a round-off below b/2 at the patch edge
    ηref = [p.F * green_1d_patch(h, max(x - x_s, b / 2), p.ω, b; roots=roots) for x in xn]
    rel(a, r) = norm(a - r) / norm(r)

    return Dict{String,Any}(
        "k0" => kp, "k_h_R" => right.k, "k_h_L" => left.k,
        "gamma_r" => gamma_r(h, p.ω; dim=2), "gamma_far_R" => γ_R, "gamma_far_L" => γ_L,
        "err_near_re" => rel(real(ηn), real(ηref)), "err_near_im" => rel(imag(ηn), imag(ηref)),
        "x_near" => xn .- x_s, "eta_near_h" => ηn, "eta_near_ref" => ηref,
        "sponge_reflection" => max(abs(in_R / out_R), abs(in_L / out_L)),
        "fit_residual" => max(right.residual, left.residual),
        "lambda_p" => λ, "kappa1" => kappa1(h, p.ω), "L" => lay.L, "buffer" => lay.buffer,
        "b" => b, "element_size" => lay.he, "nx" => lay.nx, "ny" => sol.vm.ny,
        "vertical_ratio" => sol.vm.ratio, "ndofs" => sol.ndofs,
    )
end

end # module L1_1
