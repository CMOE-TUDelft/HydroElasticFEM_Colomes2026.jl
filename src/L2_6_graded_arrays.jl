"""
# L2-6 — Graded resonator arrays on a long floating plate (configuration B, vertical section)

## Relation to the paper

Frequency grading ω_r(x) moves the local resonance along the structure: an
incident frequency ω is absorbed around the turning point x*(ω), ω_r(x*) = ω
(`eq:x_star_linear`, `eq:x_star_exp`), over a footprint
Δx ≈ ω|dx*/dω| max{2ζ, √μ(k_c(x*))} (`eq:footprint_general`,
`eq:footprint_lin_exp`): with uniform resonator mass √μ grows with ω_r, so
grading the mass by `eq:mass_grading_rule`, n m_ρ,r = μ*(m_ρ + M_f,ρ(k_c, H)),
makes the footprint of the exponential law independent of ω. Density grading
n = n₀e^{−x/L_g} at fixed ω_r (`eq:density_grade`) is an apodisation of the
resonator impedance, adiabatic for λ/L_g ≪ 1
(`eq:homogenization_scale_separation`); it should reflect less than an abrupt
array with the same number of resonators.

Verification test matrix: §L2.6.

## Problem

The configuration-B channel of L2-1 (`L2_1.build`, P1 plate, free edges, open
water on both sides) with the plate long enough for the arrays, and the
incident plate wave generated on the plate: a unit point force at x_s on a
bare lead-in section, with viscous sponges (`StructuralDampingZone`, as
L1-1) over `sponge_wavelengths` λ_p at both plate ends, so that the waves
reflected and transmitted by an array leave the measurement region. (A wave
incident from the open water is not usable here: at ω ≈ 10 rad/s,
λ_open = 0.62 m ≪ λ_p = 2.3 m and the free plate edge reflects |R| > 0.99 of
it for any array; found on the first trial of this test.)

Plate layout (plate coordinate s): sponge | buffer | source x_s | window W1 |
array (x = s − s_a ∈ [0, x_end]) | window W2 | buffer | sponge; every
separation is `decays`/κ₁ (κ₁ the slowest evanescent mode under the plate)
and the windows are `window_wavelengths` λ_p long.

Thousands of resonators are not represented as FE fields:
the bare plate is assembled and factorised once per frequency and the arrays
enter through the Woodbury (Foldy) system with the FE transfer compliance
(exact for the discrete problem, L2-2),

    (diag(1/αᵢ) − G_B,h) f = η_inc,     G_B,h = Pᵀ A⁻¹ P,

where P (n_dofs × N) evaluates the plate shape functions at the attachment
points: column j of P is the load vector of a unit point force at xⱼ and
Pᵀu the plate deflection at the points. The total field of an array is
u = A⁻¹(b_s + P f) (one more back-solve). The plate deflection in W1 and W2
is fitted (`L1_1.fit_waves`, host roots) by the propagating plate waves and
the slowest evanescent modes: R = (A⁻ − A⁻_bare)/A⁺_bare in W1 (the bare
left-going part is the sponge reflection), T = A⁺/A⁺_bare in W2.

Sets (`set`):
- `:frequency` — points xᵢ = i a_fine up to the end of the exponential array
  (L_g ln(ω_r0/1) = 1151 m); the a = 1 m arrays use every fourth point.
  Linear law ω_r = ω_r0(1 − x/L_g) for ω_r ≥ 1 rad/s (x ≤ 450 m), exponential
  law ω_r = ω_r0 e^{−x/L_g}; ζ ∈ `ζs`; uniform loading n₀m_ρ,r and the
  mass-graded loading at ζ = 0.02.
- `:density` — points xᵢ = i (a = 1 m) up to 100λ; ω_r = 10 rad/s,
  m_ρ,r,i = m_ρ,r a e^{−xᵢ/L_g} on x ≤ L_g, L_g ∈ `Lg_factors` × λ (λ the
  bare-plate wavelength at ω = 9.8); and the abrupt array of uniform
  resonators with the same number of resonators N_eff = Σᵢ e^{−xᵢ/L_g}
  (rounded); ζ ∈ `ζs_density`.

## Observables (per frequency case, for every array)

Dissipated power per resonator Pᵢ = ½ω Im αᵢ |ηᵢ|² normalised by the incident
plate-wave flux (ω/4)𝒟₀′(k₀)|A⁺_bare|² (per unit ρ; `References.energy_flux`,
verified in L1-2); R, T; energy balance |R|² + |T|² + ΣPᵢ = 1. Checks of the
construction: G_B,h against spring-based columns (`L2_1.green_columns`) at a
few points; sponge reflection |A⁻_bare/A⁺_bare| in W2.

## Reproduce

    julia --project -O1 scripts/L2-configB-vertical/L2-6_graded_arrays.jl
"""
module L2_6

using Gridap
using Parameters
using SparseArrays: sparse, SparseMatrixCSC
using LinearAlgebra: lu, norm, Diagonal
using ..References: HostSpec, ResonatorSpec, resonator_from_frequency, inv_alpha, k0, kappa1, added_mass,
                    host_roots, bare_frequency, energy_flux
import HydroElasticFEM.Physics as P
import ..L1_1
import ..L2_1

export L2_6_Params, VARIABLE, fixed_hash, run_case, arrays, predictions, wavelength_ref

"Parameters of one L2-6 case (one set at one frequency); channel field names as `L2_1_Params`."
@with_kw struct L2_6_Params
    # ── Variable ──────────────────────────────────────────────────────────────
    set::Symbol = :frequency              # :frequency or :density
    ω::Float64 = 5.0
    elements_per_wavelength::Int = 8
    order::Int = 3
    gap_decays::Float64 = 6.0
    T_ρ::Float64 = 0.0

    # ── Fixed: host (P1) ─────────────────────────────────────────────────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49

    # ── Fixed: populations ───────────────────────────────────────────────────
    loading::Float64 = 0.01               # n₀m_ρ,r (m)
    ω_r0::Float64 = 10.0
    L_g::Float64 = 500.0                  # frequency grading length (m)
    ω_r_min::Float64 = 1.0                # arrays end where ω_r = ω_r_min
    a_fine::Float64 = 0.25
    a_coarse::Float64 = 1.0
    ζs::Tuple{Vararg{Float64}} = (0.01, 0.02, 0.05)
    ζ_graded::Float64 = 0.02
    sqrt_mu_star::Float64 = 0.163         # mass-graded target √μ*
    Lg_factors::Tuple{Vararg{Float64}} = (1.0, 5.0, 20.0, 100.0)
    ω_lambda::Float64 = 9.8               # frequency of the reference wavelength λ (density set)
    ζs_density::Tuple{Vararg{Float64}} = (0.0, 0.02)

    # ── Fixed: plate source, sponges and windows ─────────────────────────────
    sponge_wavelengths::Float64 = 6.0     # sponge length (λ_p)
    sponge_viscous::Float64 = 1.0         # loss factor s_v of the viscous sponge (as L1-1)
    sponge_power::Int = 2
    decays::Float64 = 5.0                 # separations source/windows/array/sponges (1/κ₁)
    window_wavelengths::Float64 = 4.0

    # ── Fixed: discretisation and post-processing ────────────────────────────
    k_spring::Float64 = 1.0
    n_check::Int = 5                      # spring-based columns for the G_B,h check
    block::Int = 32                       # right-hand sides per block of back-solves
    n_regular::Int = 0
    n_random::Int = 0
    seed::Int = 0
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0
    βₕ::Float64 = 0.5
    n_modes::Int = 3
    n_samples::Int = 200                  # samples per fit window
end

"Fields of `L2_6_Params` varied by the L2-6 study (they name the records)."
const VARIABLE = (:set, :ω, :elements_per_wavelength, :order, :gap_decays, :T_ρ)

fixed_hash(p::L2_6_Params) = L2_1.fixed_hash_fields(p, VARIABLE)

host(p) = HostSpec(g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.m_ρ, C_ρ=p.C_ρ, T_ρ=p.T_ρ)

"Bare-plate wavelength at `p.ω_lambda` (unit of the density grading lengths)."
wavelength_ref(p) = 2π / k0(host(p), p.ω_lambda)

"Last attachment point of the set (m from the left plate edge)."
x_end(p) = p.set === :frequency ? p.L_g * log(p.ω_r0 / p.ω_r_min) : maximum(p.Lg_factors) * wavelength_ref(p)

"Attachment points of the set (array coordinate x): xᵢ = i a, a = a_fine (frequency) or a_coarse (density)."
function points(p)
    a = p.set === :frequency ? p.a_fine : p.a_coarse
    return [i * a for i in 1:floor(Int, x_end(p) / a + 1e-9)]
end

"""
    layout(p)

Plate coordinates of the sponges (length Ls), source x_s, windows W1, W2,
array origin s_a, and the plate length, at frequency p.ω.
"""
function layout(p)
    h = host(p)
    λ = 2π / k0(h, p.ω)
    Ls = p.sponge_wavelengths * λ
    d = p.decays / kappa1(h, p.ω)
    W = p.window_wavelengths * λ
    x_s = Ls + d
    W1 = (x_s + d, x_s + d + W)
    s_a = W1[2] + d
    W2 = (s_a + x_end(p) + d, s_a + x_end(p) + d + W)
    return (; λ, Ls, d, x_s, W1, s_a, W2, Lp=W2[2] + d + Ls)
end

"Viscous sponges over both plate ends (absolute coordinates for plate offset x0), as `L1_1.viscous_sponges`."
function sponges(p, lay, x0)
    h = host(p)
    k = 2π / lay.λ
    c = p.sponge_viscous * bare_frequency(h, k) * (p.m_ρ + added_mass(k, p.H))
    return [P.StructuralDampingZone(profile=P.ramp_profile(x_start=x0 + lay.Ls, x_end=x0, power=p.sponge_power), cᵨ=c),
            P.StructuralDampingZone(profile=P.ramp_profile(x_start=x0 + lay.Lp - lay.Ls, x_end=x0 + lay.Lp,
                                                            power=p.sponge_power), cᵨ=c)]
end

"Mass-graded loading n m_ρ,r = μ*(m_ρ + M_f,ρ(k_c, H)) at resonance frequency ω_r (`eq:mass_grading_rule`)."
graded_loading(p, ω_r) = p.sqrt_mu_star^2 * (p.m_ρ + added_mass(k0(host(p), ω_r), p.H))

"""
    arrays(p, xs)

The arrays of the set, each a NamedTuple (id, family, a, ζ, mass, Lg, idx, res)
with `idx` the indices into the points `xs` and `res` the resonators.
"""
function arrays(p, xs)
    out = NamedTuple[]
    if p.set === :frequency
        laws = (linear=x -> p.ω_r0 * (1 - x / p.L_g), exp=x -> p.ω_r0 * exp(-x / p.L_g))
        for (law, ωr) in pairs(laws), a in (p.a_coarse, p.a_fine)
            idx = [i for (i, x) in enumerate(xs) if abs(x / a - round(x / a)) < 1e-9 && ωr(x) >= p.ω_r_min - 1e-12]
            for (mass, ζs) in ((:uniform, p.ζs), (:graded, (p.ζ_graded,)))
                for ζ in ζs
                    load(x) = mass === :uniform ? p.loading : graded_loading(p, ωr(x))
                    res = [resonator_from_frequency(load(xs[i]) * a, ωr(xs[i]); ζ=ζ) for i in idx]
                    push!(out, (id="$(law)_a=$(a)_zeta=$(ζ)_$(mass)", family=law, a=a, ζ=ζ, mass=mass, Lg=p.L_g,
                                idx=idx, res=res))
                end
            end
        end
    else
        λ = wavelength_ref(p)
        a = p.a_coarse
        for f in p.Lg_factors, ζ in p.ζs_density
            Lg = f * λ
            taper = [i for (i, x) in enumerate(xs) if x <= Lg + 1e-9]
            w = [exp(-xs[i] / Lg) for i in taper]
            res = [resonator_from_frequency(p.loading * a * wi, p.ω_r0; ζ=ζ) for wi in w]
            push!(out, (id="taper_Lg=$(f)lambda_zeta=$(ζ)", family=:taper, a=a, ζ=ζ, mass=:density, Lg=Lg,
                        idx=taper, res=res))
            n_eff = max(1, round(Int, sum(w)))
            abrupt = collect(1:n_eff)
            push!(out, (id="abrupt_Lg=$(f)lambda_zeta=$(ζ)", family=:abrupt, a=a, ζ=ζ, mass=:uniform, Lg=Lg,
                        idx=abrupt, res=fill(resonator_from_frequency(p.loading * a, p.ω_r0; ζ=ζ), n_eff)))
        end
    end
    return out
end

"""
    predictions(p, ω, family, ζ, mass)

Local-theory turning point x* and footprint Δx (`eq:x_star_linear`,
`eq:x_star_exp`, `eq:footprint_general`) for the frequency-graded arrays;
√μ = √μ* for the mass-graded loading, else from the uniform loading at k_c(x*).
"""
function predictions(p, ω, family, ζ, mass)
    if family === :linear
        xs, dxdω = p.L_g * (1 - ω / p.ω_r0), p.L_g / p.ω_r0
    else
        xs, dxdω = p.L_g * log(p.ω_r0 / ω), p.L_g / ω
    end
    sμ = mass === :graded ? p.sqrt_mu_star : sqrt(p.loading / (p.m_ρ + added_mass(k0(host(p), ω), p.H)))
    return (x_star=xs, Δx=ω * dxdω * max(2ζ, sμ), sqrt_mu=sμ)
end

"""
    evaluation_matrix(sys, xs)

P (n_dofs × N): column j holds the values at the plate point xⱼ of the plate
shape functions (global numbering); a unit point force at xⱼ has load vector
P[:, j] and the plate deflection at the points of a solution u is Pᵀu.
"""
function evaluation_matrix(sys, xs::AbstractVector{<:Real})
    Vη = sys.X.spaces[sys.fmap[:η]]
    cc = collect(get_cell_coordinates(get_triangulation(Vη)))
    cd = collect(get_cell_dof_ids(Vη))
    basis = Gridap.CellData.get_data(get_fe_basis(Vη))
    DomainStyle(get_fe_basis(Vη)) isa ReferenceDomain || error("plate basis not in the reference domain")
    offs = cumsum([0; [L2_1.num_free(V) for V in sys.X.spaces]])
    o = offs[sys.fmap[:η]]
    first_x = [c[1][1] for c in cc]
    last_x = [c[2][1] for c in cc]
    lo, hi = min.(first_x, last_x), max.(first_x, last_x)
    order = sortperm(lo)
    n = size(sys.A, 1)
    I, J, V = Int[], Int[], Float64[]
    for (j, x) in enumerate(xs)
        X = sys.x0 + x
        k = searchsortedlast(lo[order], X + 1e-12)
        c = order[clamp(k, 1, length(order))]
        lo[c] - 1e-9 <= X <= hi[c] + 1e-9 || error("point x = $(x) outside the plate")
        ξ = (X - first_x[c]) / (last_x[c] - first_x[c])
        vals = vec(evaluate(basis[c], [Point(ξ)]))
        for (l, d) in enumerate(cd[c])
            d > 0 || continue
            push!(I, o + d); push!(J, j); push!(V, vals[l])
        end
    end
    return sparse(I, J, ComplexF64.(V), n, length(xs))
end

"""
    green_matrix(F, P; block)

G_B,h = Pᵀ A⁻¹ P from the factorisation F of A, in blocks of right-hand sides.
"""
function green_matrix(F, P; block::Int)
    N = size(P, 2)
    G = zeros(ComplexF64, N, N)
    Pt = transpose(P)
    for c in Iterators.partition(1:N, block)
        G[:, c] = Pt * (F \ Matrix(P[:, c]))
    end
    return G
end

"Woodbury solve for one array: resonator forces fᵢ and dissipated powers ½ω Im αᵢ|ηᵢ|² (per unit ρ)."
function solve_array(ω, G, η_inc, arr)
    ia = [inv_alpha(r, ω) for r in arr.res]
    Gs = G[arr.idx, arr.idx]
    f = (Diagonal(ia) - Gs) \ η_inc[arr.idx]
    power = [-ω / 2 * imag(ia[i]) * abs2(f[i]) for i in eachindex(f)]     # ½ω Im α |η|² = −½ω Im(1/α)|f|²
    return f, power
end

"G_B at the points `ss` from massless springs (as `L2_1.green_columns`, with the sponges)."
function spring_columns(q, ω, ss, zones)
    springs = fill(ResonatorSpec(m_ρ=0.0, k_ρ=q.k_spring), length(ss))
    sys = L2_1.build(q; ω=ω, xs=ss, resonators=springs, zones=zones)
    F = lu(sys.A)
    return reduce(hcat, [L2_1.deflection(sys, F \ L2_1.point_load_vector(sys, j, 1.0, q.k_spring), ss)
                         for j in eachindex(ss)])
end

"Fit of the plate deflection of `u` over a window: (A⁺, A⁻) referred to the window start, residual."
function plate_waves(p, sys, u, win; k=nothing)
    xs = collect(range(win[1], win[2]; length=p.n_samples))
    η = transpose(evaluation_matrix(sys, xs)) * u
    return L1_1.fit_waves(xs, η, host_roots(host(p), p.ω; n_evanescent=p.n_modes + 2); x_ref=win[1],
                          n_modes=p.n_modes, k=k)
end

"""
    run_case(p) -> Dict

One factorisation of the bare plate (with sponges) at ω, G_B,h at the points
of the set, the plate-source drive, and for every array of the set:
per-resonator dissipated power (normalised by the incident plate-wave flux),
R, T and the energy balance.
"""
function run_case(p::L2_6_Params)
    ω = p.ω
    lay = layout(p)
    xs = points(p)
    ss = lay.s_a .+ xs                                    # plate coordinates of the points
    q = merge(NamedTuple{fieldnames(L2_6_Params)}(Tuple(getfield(p, f) for f in fieldnames(L2_6_Params))),
              (Lp=lay.Lp,))
    zones(x0) = sponges(p, lay, x0)
    sys = L2_1.build(q; ω=ω, xs=Float64[], resonators=ResonatorSpec[], zones=zones)
    F = lu(sys.A)
    Pm = evaluation_matrix(sys, ss)
    t_G = @elapsed G = green_matrix(F, Pm; block=p.block)
    b_s = Vector(evaluation_matrix(sys, [lay.x_s])[:, 1])  # unit point force at the source
    u_bare = F \ b_s
    η_inc = transpose(Pm) * u_bare
    w1 = plate_waves(p, sys, u_bare, lay.W1)
    w2 = plate_waves(p, sys, u_bare, lay.W2; k=w1.k)
    flux = energy_flux(host(p), ω, w1.Aplus) / p.ρ
    reciprocity = norm(G - transpose(G)) / norm(G)

    # check of the evaluation-matrix construction against spring-based columns (separate system)
    ck = round.(Int, range(1, length(ss); length=p.n_check))
    Gs = spring_columns(q, ω, ss[ck], zones)
    check_G = norm(G[ck, ck] - Gs) / norm(Gs)

    results = Dict{String,Any}()
    for arr in arrays(p, xs)
        f, power = solve_array(ω, G, η_inc, arr)
        u = F \ (b_s + Pm[:, arr.idx] * f)
        t1 = plate_waves(p, sys, u, lay.W1; k=w1.k)
        t2 = plate_waves(p, sys, u, lay.W2; k=w1.k)
        R = (t1.Aminus - w1.Aminus) / w1.Aplus
        T = t2.Aplus / w2.Aplus
        absorbed = power ./ flux
        results[arr.id] = Dict{String,Any}(
            "family" => String(arr.family), "a" => arr.a, "zeta" => arr.ζ, "mass" => String(arr.mass), "Lg" => arr.Lg,
            "x" => xs[arr.idx], "absorbed" => absorbed, "R" => R, "T" => T,
            "balance" => abs2(R) + abs2(T) + sum(absorbed) - 1, "fit_residual" => max(t1.residual, t2.residual),
        )
    end
    return Dict{String,Any}(
        "arrays" => results, "n_points" => length(xs), "plate_length" => lay.Lp, "layout" => Dict(pairs(lay)),
        "ndofs" => size(sys.A, 1), "element_size" => sys.he, "flux" => flux, "k_fit" => w1.k, "k0" => k0(host(p), ω),
        "sponge_reflection" => abs(w2.Aminus / w2.Aplus), "lead_reflection" => abs(w1.Aminus / w1.Aplus),
        "bare_transmission" => abs(w2.Aplus / w1.Aplus), "reciprocity" => reciprocity, "check_G" => check_G,
        "check_points" => ss[ck], "green_time_s" => t_G, "lambda_ref" => wavelength_ref(p),
        "fit_residual_bare" => max(w1.residual, w2.residual),
    )
end

end # module L2_6
