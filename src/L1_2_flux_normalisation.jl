"""
# L1-2 — Flux normalisation (configuration A, vertical section)

## Relation to the paper

A force F on the bare host radiates two waves of amplitude A = iF/𝒟₀′(k₀)
(eq:lat_green_1d), each carrying the flux ¼ρω𝒟₀′(k₀)|A|² (eq:energy_flux),
and does the work ½ρωγ_r|F|² with γ_r = Im G(0) = 1/𝒟₀′(k₀) (eq:ImG,
eq:point_power). γ_r is the radiative width of every resonator result of
Section 4: it sets the critical damping, the absorption maximum and the
bandwidth. This test checks three independent FE measures of it, so that the
normalisation of the flux in the FE post-processing is the one of the paper.
The code uses density-normalised quantities (𝒟₀/ρ, F/ρ), as in L1-1.

Verification test matrix: §L1.2.

## Problem

The L1-1 problem, unchanged (`L1_1.solve`: plate-covered channel, sponges,
evanescent buffer, channel length off the cavity resonances, uniform patch
load of width b = `source_elements` elements); the cases are `L1_1_Params`.

## Observables (one frequency, one mesh)

1. Far field: γ_h = A/(iF sinc(k₀b/2)) from the waves fitted on either side
   (`L1_1.fit_waves`, k fitted), with the waves coming back from the sponges
   removed (the source is transparent). Same as L1-1, observable 2.
2. Patch: for the lossless host Im G(x) = cos(k₀x)/𝒟₀′(k₀) exactly (the
   evanescent and flexural terms of the residue series are real), so the
   average of the patch response over the patch is
   Im⟨G_b⟩ = sinc²(k₀b/2)/𝒟₀′(k₀). The FE average ⟨η⟩ (Gauss–Legendre on the
   patch elements), minus the incoming waves averaged over the patch, gives
   γ_patch = Im⟨η − η_in⟩/(F sinc²(k₀b/2)).
3. Power: the work of the load, ½ρω Im(F̄⟨η⟩), against the net flux
   ¼ρω𝒟₀′(k₀) Σ(|A_out|² − |A_in|²) through both probe windows; error
   |P_work/P_flux − 1|.

## Reproduce

    julia --project -O1 scripts/L1-configA-vertical/L1-2_flux_normalisation.jl
"""
module L1_2

using LinearAlgebra: SymTridiagonal, eigen
using ..References: dD0, gamma_r, host_roots, kappa1
import ..L1_1
using ..L1_1: L1_1_Params

export run_case

"Gauss–Legendre nodes and weights on [−1, 1] with `n` points (Golub–Welsch)."
function gauss_legendre(n::Int)
    β = [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

"""
    patch_average(ηh, xa, xb, ne; n)

Average of the plate deflection over [xa, xb], which spans `ne` elements, by
Gauss–Legendre with `n` points per element (exact for the FE polynomial when
n ≥ (order + 1)/2).
"""
function patch_average(ηh, xa::Real, xb::Real, ne::Int; n::Int)
    t, w = gauss_legendre(n)
    he = (xb - xa) / ne
    xs = [xa + (e + (ti + 1) / 2) * he for e in 0:(ne - 1) for ti in t]
    ws = repeat(w ./ 2 .* he, ne)
    return sum(ws .* L1_1.sample(ηh, xs)) / (xb - xa)
end

"""
    run_case(p::L1_1_Params) -> Dict

Solve the L1-1 problem for case `p` and return the three FE measures of the
radiative width, their reference and the case metadata (raw values; errors and
checks are formed by the script).
"""
function run_case(p::L1_1_Params)
    h = L1_1.host(p)
    sol = L1_1.solve(p)
    lay = sol.lay
    x_s, b, λ = lay.x_s, lay.b, lay.λ
    roots = host_roots(h, p.ω)
    kp = real(roots[1])

    # 1. far field (as L1-1)
    probe(xa, xb) = (xs = range(xa, xb; length=p.n_samples);
                     L1_1.fit_waves(xs, L1_1.sample(sol.ηh, xs), roots; x_ref=x_s, n_modes=p.n_modes))
    right = probe(x_s + p.probe_offset * λ, lay.window[2])
    left = probe(lay.window[1], x_s - p.probe_offset * λ)
    out_R, in_R = right.Aplus, right.Aminus
    out_L, in_L = left.Aminus, left.Aplus
    s = sin(kp * b / 2) / (kp * b / 2)
    γ_R = (out_R - in_L) / (im * s * p.F)
    γ_L = (out_L - in_R) / (im * s * p.F)

    # 2. patch average; the incoming waves average to s·(in_L + in_R) over the patch
    ηavg = patch_average(sol.ηh, x_s - b / 2, x_s + b / 2, p.source_elements; n=p.order + 2)
    γ_patch = imag((ηavg - s * (in_L + in_R)) / p.F) / s^2

    # 3. work of the load against the net flux through both probe windows
    P_work = p.ρ * p.ω / 2 * imag(conj(p.F) * ηavg)
    P_flux = p.ρ * p.ω / 4 * dD0(h, kp, p.ω) * (abs2(out_R) - abs2(in_R) + abs2(out_L) - abs2(in_L))

    return Dict{String,Any}(
        "gamma_r" => gamma_r(h, p.ω; dim=2),
        "gamma_far_R" => γ_R, "gamma_far_L" => γ_L, "gamma_patch" => γ_patch,
        "eta_patch" => ηavg, "P_work" => P_work, "P_flux" => P_flux,
        "k0" => kp, "k_h_R" => right.k, "k_h_L" => left.k,
        "sponge_reflection" => max(abs(in_R / out_R), abs(in_L / out_L)),
        "fit_residual" => max(right.residual, left.residual),
        "lambda_p" => λ, "kappa1" => kappa1(h, p.ω), "L" => lay.L, "buffer" => lay.buffer,
        "b" => b, "element_size" => lay.he, "nx" => lay.nx, "ny" => sol.vm.ny,
        "vertical_ratio" => sol.vm.ratio, "ndofs" => sol.ndofs,
    )
end

end # module L1_2
