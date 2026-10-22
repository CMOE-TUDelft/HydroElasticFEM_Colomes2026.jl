"""
# L1-6 — Complete stop band against the lattice spacing (configuration A, vertical section)

## Relation to the paper

At fixed loading n₀m_ρ,r = m_ρ,r/a, the complete stop band [ω_X, ω_Γ] of a
floating line array (L1-5) is a discreteness effect: away from the Bragg
regime its width is

    (ω_Γ − ω_X)/ω_r ≃ μ(π/a)ω_r²/(ω₀²(π/a) − ω_r²) − Π_L/2               (eq:gap_asymptotic)

(within 2 % of the lattice sums for ak_c/π ≲ 0.75), and when bending dominates
the host at the lattice scale

    (ω_Γ − ω_X)/ω_r ≃ (1/π⁴ − 1/1440) n₀m_ρ,r ω_r² a⁴/C_ρ                 (eq:gap_a4)

so the gap vanishes as a⁴ in the homogenisation limit, whereas a metaplate in
vacuo keeps the finite gap ω_r(√(1+μ) − 1) (`subsubsec:lat_long_wave`,
`fig:lattice_design`(a)). At a = 1 m the width is about 1.8 times the gap of
the local branches truncated at qa = π.

Verification test matrix: §L1.6.

## Problem

The periodic supercells of L1-5 (`L1_5.band_frequency`): ω_Γ from one cell
(M = 1, j = 0) and ω_X from two cells with alternating load (M = 2, j = 1),
for each spacing a with m_ρ,r = `loading`·a.

## Observables

ω_Γ, ω_X and the gap against the lattice sums (`omega_gamma_1d`,
`omega_X_1d`); the gap laws `gap_asymptotic_1d`, `gap_a4_1d`; the gap of the
truncated local branches ω_r − ω₋(π/a). For a ≤ 0.35 the gap is below 3e-4
ω_r, so the edges must be accurate to ≪ 1e-5: p = 3 is the reference order.

## Reproduce

    julia --project -O1 scripts/L1-configA-vertical/L1-6_spacing_sweep.jl
"""
module L1_6

using Parameters
using ..References: k0, omega_gamma_1d, omega_X_1d, local_branches, gap_asymptotic_1d, gap_a4_1d,
                    lattice_compliance_number_1d, crossing_wavenumber
import ..L1_1
import ..L1_5

export L1_6_Params, VARIABLE, fixed_hash, run_case

"""
Parameters of one L1-6 case; the field names are those of `L1_5_Params`, so
that the L1-5 supercell functions take this struct. `m_ρr` follows from the
spacing and the fixed loading.
"""
@with_kw struct L1_6_Params
    # ── Variable: what the study sweeps (names the record) ───────────────────
    a::Float64 = 1.0                      # spacing (m)
    T_ρ::Float64 = 0.0                    # pre-tension / ρ (m³/s²)
    elements_per_cell::Int = 8            # horizontal elements per cell
    order::Int = 3                        # polynomial order of ϕ and η

    # ── Fixed: host, set P1 of the test matrix = Table 1 of the paper ────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49

    # ── Fixed: lattice ────────────────────────────────────────────────────────
    loading::Float64 = 0.01               # n₀m_ρ,r (m)
    m_ρr::Float64 = loading * a           # resonator mass / ρ (m²), derived
    ω_r::Float64 = 10.0

    # ── Fixed: forcing, discretisation and root finding (as L1-5) ────────────
    F::Float64 = 1.0
    source_elements::Int = 2
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0
    start::Float64 = 1e-6
    xtol::Float64 = 1e-14
    accept::Float64 = 1e-2
end

"Fields of `L1_6_Params` varied by the L1-6 study (they name the records)."
const VARIABLE = (:a, :T_ρ, :elements_per_cell, :order)

"Short hash of the fixed parameters (`m_ρr` is derived from `a` and excluded)."
function fixed_hash(p::L1_6_Params)
    fixed = Tuple(getfield(p, f) for f in fieldnames(L1_6_Params) if !(f in VARIABLE) && f != :m_ρr)
    return string(hash(fixed); base=16, pad=16)[1:8]
end

"""
    run_case(p) -> Dict

FE band edges of spacing `p.a`, with the lattice-sum references and the gap laws.
"""
function run_case(p::L1_6_Params)
    p.m_ρr ≈ p.loading * p.a || throw(ArgumentError("m_ρr must equal loading·a"))
    h = L1_1.host(p)
    r = L1_5.resonator(p)
    a = p.a
    ωΓ_ref = omega_gamma_1d(h, r, a)
    ωX_ref = omega_X_1d(h, r, a; ωmin=0.8p.ω_r, npts=4001)
    Γ = L1_5.band_frequency(p, ωΓ_ref; M=1, j=0)
    X = L1_5.band_frequency(p, ωX_ref; M=2, j=1)
    return Dict{String,Any}(
        "omega_gamma_h" => Γ.omega, "omega_gamma_ref" => ωΓ_ref,
        "omega_X_h" => X.omega, "omega_X_ref" => ωX_ref,
        "gap_asymptotic" => gap_asymptotic_1d(h, r, a), "gap_a4" => gap_a4_1d(h, r, a),
        "gap_truncated_local" => (p.ω_r - local_branches(h, r, 1 / a, π / a).lower) / p.ω_r,
        "Pi_L" => lattice_compliance_number_1d(h, r, a),
        "akc_over_pi" => a * crossing_wavenumber(h, r) / π,
        "leak" => max(Γ.leak, X.leak), "fe_solves" => Γ.solves + X.solves,
        "ndofs_cell" => Γ.ndofs, "ny" => Γ.ny, "element_size" => a / p.elements_per_cell,
        "lambda_p" => 2π / k0(h, p.ω_r),
    )
end

end # module L1_6
