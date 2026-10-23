"""
# L1-3 — Single line resonator (configuration A, vertical section)

## Relation to the paper

A line resonator (mass m_ρ,r, spring k_ρ,r, dashpot c_ρ,r per unit fluid
density) attached to the bare host scatters an incident wave with

    R = iγ_r/(x − iΓ),   T = 1 + R,   Γ = γ_r + γ_d,
    x(ω) = Re(1/α(ω) − G(0; ω))                                 (eq:lat_detuning)

(`References.single_resonator_1d`). The renormalised resonance ω̃ is the root
of x (eq:lat_renorm_single): there R = −1 and T = 0 for a lossless resonator,
and the absorbed fraction 𝒜 = 1 − |R|² − |T|² reaches ½ at critical coupling
γ_d = γ_r (eq:lat_absorbed_1d). The half-power points |x| = Γ set the width,
≈ (1 + c/c_crit)ω̃/Q_rad (2ω̃/Q_rad at critical coupling).

Verification test matrix: §L1.3.

## Problem

The L1-1 problem (`L1_1.solve`) with one resonator, on a layout sized with
the plate wavelength λ_p at ω_r (one mesh for every frequency of a case):

    wall ── sponge ── buffer ── │ source · probe ↑ · resonator · probe ↓ │ ── buffer ── sponge ── wall

The source (patch, as L1-1) is `source_offset` λ_p from the window start; the
upstream probe ends `source_gap` λ_p past it; the probes have length
`probe` λ_p and stop `offset` λ_p from the resonator. The resonator sits on a
mesh node or at an element midpoint (`snap`). Buffer and channel length follow
the L1-1 rules (≥ `gap_decays`/κ₁ between source/resonator and the sponges;
L/λ_p ∈ ℤ + `length_fraction` at ω_r).

## Observables

1. R at each frequency from the waves on both probes, phase reference at the
   resonator (`point_scattering`). The exact near fields of the source and of
   the resonator (force f = ω²m_ρ,r q from the FE) are subtracted, so that
   the probes hold only the two propagating waves. The resonator is driven by
   the incident wave, the wave entering from the downstream sponge, and the
   tail of the source near field (its slowest evanescent mode, 1/κ₁ ≈ 6 m,
   is still 2e-3 of the incident wave at the resonator); R is the scattered
   amplitude over this full drive, and T = 1 + R (line resonator). Ignoring
   the near-field drive biased the phase of R by 2e-3 rad, shifting ω̃ by
   −1.8e-5 (T_ρ = 0) on every mesh.
   The propagating wavenumber of these fits is the discrete one,
   k_h = k₀(ω)(1 + δ), with δ measured once per case on the bare plate at ω_r
   (one-way field over the window past the source): fitting k on the short
   probes is ill-posed when the field is a standing wave.
2. For the FE with its own G_h, 1/R = −(Γ + ix)/γ_r: ω̃_h is the root of
   Im(1/R) and the half-power points are the roots of Im(1/R) ∓ Re(1/R)
   (x decreases with ω: x = Γ below ω̃, x = −Γ above). They are found by
   secant iteration on FE solves and compared with the same roots of the exact
   solution with the same (damped) resonator.
3. At ω̃_h: R, T, 𝒜 and the resonator response q/η = κ/(κ − ω²m_ρ,r),
   κ = k_ρ,r − iωc_ρ,r.
4. Optionally (`n_omega > 0`) |R|, |T| and 𝒜 on `n_omega` frequencies in
   ω_r(1 ± `span`/Q_rad).

## Reproduce

    julia --project -O1 scripts/L1-configA-vertical/L1-3_single_line_resonator.jl
"""
module L1_3

using Parameters
using Roots: find_zero
using ..References: HostSpec, ResonatorSpec, k0, kappa1, host_roots, dD0, resonator_from_frequency,
                    single_resonator_1d, renormalised_frequency, critical_damping, Q_rad, gamma_r
import ..L1_1

export L1_3_Params, VARIABLE, fixed_hash, run_case

"""
Parameters of one L1-3 case. Lengths of the layout are in plate wavelengths
λ_p at ω_r. The field names of the host, forcing, sponge and discretisation
are those of `L1_1_Params`, so that `L1_1.solve` takes this struct.
"""
@with_kw struct L1_3_Params
    # ── Variable: what the study sweeps (names the record) ───────────────────
    T_ρ::Float64 = 0.0                    # pre-tension / ρ (m³/s²)
    c_factor::Float64 = 0.0               # dashpot / critical dashpot
    snap::Symbol = :node                  # resonator on a mesh :node or at the :midelement
    elements_per_wavelength::Int = 20     # horizontal elements per λ_p(ω_r)
    order::Int = 2                        # polynomial order of ϕ and η
    n_omega::Int = 0                      # response-curve frequencies (0: resonance only)

    # ── Fixed: host, set P1 of the test matrix = Table 1 of the paper ────────
    g::Float64 = 9.8
    ρ::Float64 = 1000.0
    H::Float64 = 10.0
    m_ρ::Float64 = 0.01
    C_ρ::Float64 = 0.49

    # ── Fixed: resonator ──────────────────────────────────────────────────────
    m_ρr::Float64 = 0.01                  # resonator mass / ρ (m²)
    ω_r::Float64 = 10.0                   # natural frequency √(k/m) (rad/s)

    # ── Fixed: forcing ────────────────────────────────────────────────────────
    F::Float64 = 1.0
    source_elements::Int = 2

    # ── Fixed: layout ─────────────────────────────────────────────────────────
    source_offset::Float64 = 0.25         # source from the window start
    source_gap::Float64 = 0.5             # source to the upstream probe
    probe::Float64 = 3.0                  # probe length
    offset::Float64 = 1.0                 # probe end to the resonator
    end_margin::Float64 = 0.25            # downstream probe to the window end
    gap_decays::Float64 = 5.0
    length_fraction::Float64 = 0.25
    sponge_wavelengths::Float64 = 6.0
    sponge_strength::Float64 = 3.0
    sponge_power::Int = 2
    sponge_viscous::Float64 = 0.0         # viscous sponge loss factor (`L1_1.viscous_sponges`); 0: none

    # ── Fixed: discretisation (as L1-1) ──────────────────────────────────────
    top_fraction::Float64 = 0.5
    max_ratio::Float64 = 1.2
    ratio_elements::Int = 20
    γ_penalty::Float64 = 6.0

    # ── Fixed: post-processing ────────────────────────────────────────────────
    n_modes::Int = 6                      # non-propagating modes fitted for δ (bare plate)
    n_samples::Int = 400
    span::Float64 = 5.0                   # curves and admissible roots in ω_r(1 ± span/Q_rad)
    xtol::Float64 = 1e-8                  # secant tolerance (fit noise ≈ 1e-7 in 1/R)
end

"Fields of `L1_3_Params` varied by the L1-3 study (they name the records)."
const VARIABLE = (:T_ρ, :c_factor, :snap, :elements_per_wavelength, :order, :n_omega)

"Fields added after the first L1-3 records; left out of the hash at their default, so those records keep their names."
const LATE_FIELDS = (:sponge_viscous => 0.0,)

"Fixed fields of `p` that enter the record hash (all but `exclude`, and late fields at their default)."
hashed_fields(p::L1_3_Params, exclude) =
    Tuple(getfield(p, f) for f in fieldnames(L1_3_Params)
          if !(f in exclude) && !any(l -> l.first == f && getfield(p, f) == l.second, LATE_FIELDS))

"Short hash of the fixed parameters, appended to the record name."
function fixed_hash(p::L1_3_Params)
    fixed = hashed_fields(p, VARIABLE)
    return string(hash(fixed); base=16, pad=16)[1:8]
end

"""
    resonator(p)

The resonator of the case, ω̃ of the undamped one (which defines c_crit) and
c_crit (γ_d = γ_r at that ω̃).
"""
function resonator(p::L1_3_Params)
    h = L1_1.host(p)
    r0 = resonator_from_frequency(p.m_ρr, p.ω_r)
    ω̃0 = renormalised_frequency(h, r0)
    c_crit = critical_damping(h, r0, ω̃0)
    return (r=ResonatorSpec(m_ρ=r0.m_ρ, k_ρ=r0.k_ρ, c_ρ=p.c_factor * c_crit), ω̃0=ω̃0, c_crit=c_crit)
end

"""
    layout(p)

Absolute geometry (λ_p at ω_r): the fields of `L1_1.layout` (channel, sponge,
buffer, window, source, element size) plus the resonator position `x_r` and the
probe windows `up`, `down`.
"""
function layout(p::L1_3_Params)
    h = L1_1.host(p)
    λ = 2π / k0(h, p.ω_r)
    Ls = p.sponge_wavelengths * λ
    xr_w = (p.source_offset + p.source_gap + p.probe + p.offset) * λ        # resonator, from the window start
    W = xr_w + (p.offset + p.probe + p.end_margin) * λ
    d = min(p.source_offset * λ, W - xr_w)                                  # source/resonator to the window ends
    buffer = max(λ, p.gap_decays / kappa1(h, p.ω_r) - d)
    L = λ * (ceil((2 * (Ls + buffer) + W) / λ - p.length_fraction) + p.length_fraction)
    buffer = (L - W) / 2 - Ls
    nx = ceil(Int, L * p.elements_per_wavelength / λ)
    he = L / nx
    xw = Ls + buffer
    x_s = round((xw + p.source_offset * λ) / he) * he
    x_r = p.snap == :node ? round((xw + xr_w) / he) * he :
          p.snap == :midelement ? (floor((xw + xr_w) / he) + 0.5) * he :
          throw(ArgumentError("snap must be :node or :midelement"))
    return (; λ, L, Ls, buffer, window=(xw, xw + W), x_s, b=p.source_elements * he, nx, he, x_r,
            up=(x_r - (p.offset + p.probe) * λ, x_r - p.offset * λ),
            down=(x_r + p.offset * λ, x_r + (p.offset + p.probe) * λ))
end

"""
    point_scattering(up, down, E)

R of a point (line) scatterer from the propagating amplitudes fitted upstream
and downstream, both referenced to its position, and the non-propagating part
`E` of the incident field there. The scatterer radiates the same amplitude S
to both sides, driven by D = up.Aplus + down.Aminus + E:

    down.Aplus = up.Aplus + S,   up.Aminus = down.Aminus + S,   R = S/D,   T = 1 + R.

`asymmetry` = |S_down − S_up|/|S| measures how well the two sides agree.
"""
function point_scattering(up, down, E::Number)
    S_down, S_up = down.Aplus - up.Aplus, up.Aminus - down.Aminus
    R = (S_down + S_up) / 2 / (up.Aplus + down.Aminus + E)
    return (R=R, T=1 + R, asymmetry=abs(S_down - S_up) / abs(S_down + S_up) * 2)
end

"Secant iteration from x0, x1; `converged` when the step is below `xtol` relative."
function secant_root(f, x0::Real, x1::Real; xtol::Real, maxiter::Int=30)
    f0, f1 = f(x0), f(x1)
    n = 2
    while abs(x1 - x0) > xtol * abs(x1) && n < maxiter && f1 != f0
        x0, x1 = x1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f0, f1 = f1, f(x1)
        n += 1
    end
    return (root=x1, evaluations=n, converged=abs(x1 - x0) <= xtol * abs(x1))
end

"""
    dispersion_error(p, lay)

δ = k_h/k₀ − 1 at ω_r on the mesh of the case: the bare plate driven by the
same source, k fitted on the one-way field from one wavelength past the source
to the window end.
"""
function dispersion_error(p::L1_3_Params, lay)
    sol = L1_1.solve(p; ω=p.ω_r, lay=lay)
    roots = host_roots(L1_1.host(p), p.ω_r)
    xs = range(lay.x_s + lay.λ, lay.window[2]; length=p.n_samples)
    fit = L1_1.fit_waves(xs, L1_1.sample(sol.ηh, xs), roots; x_ref=lay.x_s, n_modes=p.n_modes)
    return fit.k / real(roots[1]) - 1
end

"""
    near_field(roots, x, ω, h; b=0)

Non-propagating part of the (patch, width `b`) Green function at distance x:
i Σₘ₍≥₂₎ sinc(kₘb/2) e^{ikₘ|x|}/𝒟₀′(kₘ), the residue series without k₀.
"""
function near_field(h::HostSpec, roots, x::Real, ω::Real; b::Real=0.0)
    sinc_(z) = b == 0 ? one(z) : sin(z * b / 2) / (z * b / 2)
    return im * sum(k -> sinc_(k) * exp(im * k * abs(x)) / dD0(h, k, ω), @view roots[2:end])
end

"""
    scattering(p, lay, r, ω, δ)

FE scattering at frequency ω: R (phase reference at the resonator), T, 𝒜,
the resonator displacement q and the plate deflection η at the attachment.
"""
function scattering(p::L1_3_Params, lay, r::ResonatorSpec, ω::Real, δ::Real)
    sol = L1_1.solve(p; ω=ω, lay=lay, resonators=[r], x_res=[lay.x_r])
    h = L1_1.host(p)
    roots = host_roots(h, ω)
    kh = real(roots[1]) * (1 + δ)
    q = only(sol.q)
    f_r = ω^2 * r.m_ρ * q                         # resonator force on the plate, κ(q − η) = ω²m q
    src_near(x) = p.F * near_field(h, roots, x - lay.x_s, ω; b=lay.b)
    near(x) = src_near(x) + f_r * near_field(h, roots, x - lay.x_r, ω)
    fit(w) = (xs = range(w...; length=p.n_samples);
              L1_1.fit_waves(xs, L1_1.sample(sol.ηh, xs) .- near.(xs), roots; x_ref=lay.x_r, n_modes=0, k=kh))
    up, down = fit(lay.up), fit(lay.down)
    s = point_scattering(up, down, src_near(lay.x_r))
    return (R=s.R, T=s.T, A=1 - abs2(s.R) - abs2(s.T), q=q,
            η1=only(L1_1.sample(sol.ηh, [lay.x_r])), asymmetry=s.asymmetry,
            fit_residual=max(up.residual, down.residual), ndofs=sol.ndofs, ny=sol.vm.ny)
end

"""
    run_case(p) -> Dict

Resonance, half-power points and resonator response of case `p` from the FE
and from the exact single-resonator solution with the same resonator; with
`n_omega > 0` also the response curves.
"""
function run_case(p::L1_3_Params)
    h = L1_1.host(p)
    res = resonator(p)
    r = res.r
    Q = Q_rad(h, r, p.ω_r)
    lay = layout(p)
    δ = dispersion_error(p, lay)
    cache = Dict{Float64,Any}()
    fe(ω) = get!(() -> scattering(p, lay, r, ω, δ), cache, ω)
    invR(ω) = 1 / fe(ω).R
    invR_ref(ω) = 1 / single_resonator_1d(h, r, ω).R
    lower(f) = ω -> imag(f(ω)) - real(f(ω))       # x = Γ, below ω̃
    upper(f) = ω -> imag(f(ω)) + real(f(ω))       # x = −Γ, above ω̃

    # references: roots of the exact solution with the (damped) resonator
    Δ = 4p.ω_r / Q
    ω̃_ref = find_zero(ω -> imag(invR_ref(ω)), (res.ω̃0 - Δ, res.ω̃0 + Δ))
    lo_ref = find_zero(lower(invR_ref), (ω̃_ref - Δ, ω̃_ref))
    hi_ref = find_zero(upper(invR_ref), (ω̃_ref, ω̃_ref + Δ))

    # FE roots, started at the references; NaN if not converged within ω̃₀(1 ± span/Q). The window is
    # centred on the renormalised resonance: at low ω_r the shift ω̃₀ − ω_r is itself ≈ ω_r·5/Q
    # (−4.2e-4 at ω_r = 1), so a window centred on ω_r rejects the FE root.
    band = p.ω_r .* (1 - p.span / Q, 1 + p.span / Q)              # response curves
    admissible = res.ω̃0 .* (1 - p.span / Q, 1 + p.span / Q)
    function fe_root(f, x0)
        s = secant_root(f, x0, x0 * (1 + 1e-3); xtol=p.xtol)
        return s.converged && admissible[1] <= s.root <= admissible[2] ? s.root : NaN
    end
    ω̃_h = fe_root(ω -> imag(invR(ω)), ω̃_ref)
    lo_h = fe_root(lower(invR), lo_ref)
    hi_h = fe_root(upper(invR), hi_ref)

    peak = fe(isnan(ω̃_h) ? ω̃_ref : ω̃_h)
    ref = single_resonator_1d(h, r, ω̃_ref)
    ωq = isnan(ω̃_h) ? ω̃_ref : ω̃_h
    κ = r.k_ρ - im * ωq * r.c_ρ
    data = Dict{String,Any}(
        "omega_peak_h" => ω̃_h, "omega_peak_ref" => ω̃_ref, "omega_peak_undamped" => res.ω̃0,
        "omega_half_h" => [lo_h, hi_h], "omega_half_ref" => [lo_ref, hi_ref],
        "R_peak_h" => peak.R, "T_peak_h" => peak.T, "A_peak_h" => peak.A,
        "R_peak_ref" => ref.R, "T_peak_ref" => ref.T, "A_peak_ref" => ref.absorbed,
        "q_over_eta_peak_h" => peak.q / peak.η1, "q_over_eta_peak_ref" => κ / (κ - ωq^2 * r.m_ρ),
        "Q_rad" => Q, "c_crit" => res.c_crit, "c_ρr" => r.c_ρ, "gamma_r" => gamma_r(h, p.ω_r; dim=2),
        "dispersion_error" => δ, "fe_solves" => length(cache) + 1,
        "fit_residual_peak" => peak.fit_residual, "asymmetry_peak" => peak.asymmetry, "ndofs" => peak.ndofs, "ny" => peak.ny,
        "lambda_p" => lay.λ, "element_size" => lay.he, "L" => lay.L, "buffer" => lay.buffer,
        "kappa1" => kappa1(h, p.ω_r), "x_r" => lay.x_r, "offset_in_element" => mod(lay.x_r / lay.he, 1.0),
    )
    if p.n_omega > 0
        ωs = collect(range(band...; length=p.n_omega))
        sweep = fe.(ωs)
        refs = single_resonator_1d.(Ref(h), Ref(r), ωs)
        merge!(data, Dict{String,Any}(
            "omega" => ωs,
            "R_h" => getfield.(sweep, :R), "T_h" => getfield.(sweep, :T), "A_h" => getfield.(sweep, :A),
            "R_ref" => getfield.(refs, :R), "T_ref" => getfield.(refs, :T), "A_ref" => getfield.(refs, :absorbed),
            "fit_residual_sweep" => maximum(getfield.(sweep, :fit_residual)),
        ))
        data["fe_solves"] = length(cache) + 1
    end
    return data
end

end # module L1_3
