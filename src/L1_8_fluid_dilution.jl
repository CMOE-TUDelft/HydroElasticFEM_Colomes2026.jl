"""
# L1-8 — Fluid dilution of the hybridisation (configuration A, vertical section)

## Relation to the paper

In the local model the resonators couple to the floating plate through the
wavenumber-dependent mass ratio μ(k) = n₀m_ρ,r/(m_ρ + M_f,ρ(k, H))
(eq:lat_mu), and at the crossing wavenumber k_c, ω₀(k_c) = ω_r, the two
branches are separated by exactly

    (ω₊(k_c) − ω₋(k_c))/ω_r = √μ(k_c)                                       (eq:anticrossing)

with the fluid added mass in the denominator: for the parameters of Table 1,
√μ(k_c) = 0.163 (T_ρ = 0), 0.138 (T_ρ = 9.8), against √(n₀m_ρ,r/m_ρ) = 1 in
vacuo (`subsubsec:fluid_dilution`). Depth enters only through k_c
(eq:gravity_bound). This test checks that the discrete lattice, computed with
the FE, has this splitting at q = k_c and that the discreteness correction
vanishes as the spacing decreases at fixed loading.

Verification test matrix: §L1.8.

## Problem

The periodic supercells of L1-5 (`L1_5.band_frequency`). The Bloch wavenumber
q = 2πj/(Ma) of a supercell of M cells equals k_c exactly when the spacing is
a = 2πj/(Mk_c); j/M sets ε_s = k_c a/(2π) = j/M. For each case the two band
frequencies at q = k_c on either side of ω_r are found as poles of the Bloch
response, started at the lattice-sum roots (`bloch_roots_1d`). The resonator
mass is m_ρ,r = `loading`·a.

## Observables

ω₋(k_c), ω₊(k_c) and the splitting (ω₊ − ω₋)/ω_r: FE against the lattice
sums (discretisation), and against √μ(k_c) (`anticrossing_width`, the local
model; the difference is the discreteness correction, O(ε_s^≈4.5) from the
lattice sums).

## Reproduce

    julia --project -O1 scripts/L1-configA-vertical/L1-8_fluid_dilution.jl
"""
module L1_8

using Parameters
using ..References: k0, bloch_roots_1d, anticrossing_width, resonator_from_frequency, added_mass
import ..L1_1
import ..L1_5

export L1_8_Params, VARIABLE, fixed_hash, run_case

"""
Parameters of one L1-8 case; the field names are those of `L1_5_Params`, so
that the L1-5 supercell functions take this struct. `a` and `m_ρr` are derived
from k_c and j/M by `derived(p)`.
"""
@with_kw struct L1_8_Params
    # ── Variable: what the study sweeps (names the record) ───────────────────
    T_ρ::Float64 = 0.0                    # pre-tension / ρ (m³/s²)
    C_ρ::Float64 = 0.49                   # bending stiffness / ρ (m⁵/s²)
    H::Float64 = 10.0                     # depth (m)
    ω_r::Float64 = 10.0                   # resonator frequency (rad/s)
    M::Int = 8                            # cells per supercell; ε_s = j/M
    elements_per_wavelength::Int = 32     # elements per λ_c = 2π/k_c (elements per cell = this·j/M)
    order::Int = 3

    # ── Fixed: host and lattice ──────────────────────────────────────────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    m_ρ::Float64 = 0.01
    loading::Float64 = 0.01               # n₀m_ρ,r (m)
    j::Int = 1                            # Bloch index: q = 2πj/(Ma) = k_c

    # ── Fixed: forcing, discretisation and root finding (as L1-5) ────────────
    F::Float64 = 1.0
    source_elements::Int = 2
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0
    start::Float64 = 1e-5
    xtol::Float64 = 1e-14
    accept::Float64 = 1e-2
    window::Tuple{Float64,Float64} = (0.6, 1.4)   # band search in this range of ω/ω_r
end

"Fields of `L1_8_Params` varied by the L1-8 study (they name the records)."
const VARIABLE = (:T_ρ, :C_ρ, :H, :ω_r, :M, :elements_per_wavelength, :order)

"Short hash of the fixed parameters, appended to the record name."
function fixed_hash(p::L1_8_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L1_8_Params) if !(f in VARIABLE))
    return string(hash(fixed); base=16, pad=16)[1:8]
end

"""
    derived(p)

Crossing wavenumber k_c = k₀(ω_r), spacing a = 2πj/(Mk_c), and the parameter
set in the field names of `L1_5_Params` (a, m_ρr = loading·a, elements per
cell) taken by the L1-5 supercell functions.
"""
function derived(p::L1_8_Params)
    kc = k0(L1_1.host(p), p.ω_r)
    a = 2π * p.j / (p.M * kc)
    n_cell = p.elements_per_wavelength * p.j / p.M
    isinteger(n_cell) && iseven(Int(n_cell)) ||
        throw(ArgumentError("elements_per_wavelength·j/M must be an even integer (resonators on nodes)"))
    q = (g=p.g, ρ=p.ρ, H=p.H, m_ρ=p.m_ρ, C_ρ=p.C_ρ, T_ρ=p.T_ρ, a=a, m_ρr=p.loading * a, ω_r=p.ω_r,
         F=p.F, source_elements=p.source_elements, elements_per_cell=Int(n_cell), order=p.order,
         top_fraction=p.top_fraction, max_ratio=p.max_ratio, ratio_elements=p.ratio_elements,
         γ_penalty=p.γ_penalty, start=p.start, xtol=p.xtol, accept=p.accept)
    return (; kc, a, cell=q)
end

"""
    run_case(p) -> Dict

FE band frequencies at q = k_c below and above ω_r, the lattice-sum references
and the local splitting √μ(k_c).
"""
function run_case(p::L1_8_Params)
    h = L1_1.host(p)
    d = derived(p)
    c = d.cell
    r = L1_5.resonator(c)
    roots = bloch_roots_1d(h, r, d.kc, d.a, range(p.window[1] * p.ω_r, p.window[2] * p.ω_r; length=2001))
    lo_ref, hi_ref = maximum(filter(<(p.ω_r), roots)), minimum(filter(>(p.ω_r), roots))
    lo = L1_5.band_frequency(c, lo_ref; M=p.M, j=p.j)
    hi = L1_5.band_frequency(c, hi_ref; M=p.M, j=p.j)
    μ_c = (p.loading) / (p.m_ρ + added_mass(d.kc, p.H))
    return Dict{String,Any}(
        "omega_minus_h" => lo.omega, "omega_minus_ref" => lo_ref,
        "omega_plus_h" => hi.omega, "omega_plus_ref" => hi_ref,
        "split_h" => (hi.omega - lo.omega) / p.ω_r, "split_ref" => (hi_ref - lo_ref) / p.ω_r,
        "split_local" => anticrossing_width(h, r, 1 / d.a), "sqrt_mu" => sqrt(μ_c),
        "split_vacuo" => sqrt(p.loading / p.m_ρ), "kc" => d.kc, "a" => d.a, "eps_s" => p.j / p.M,
        "kcH" => d.kc * p.H, "elements_per_cell" => c.elements_per_cell,
        "leak" => max(lo.leak, hi.leak), "fe_solves" => lo.solves + hi.solves,
        "ndofs" => lo.ndofs, "ny" => lo.ny,
    )
end

end # module L1_8
