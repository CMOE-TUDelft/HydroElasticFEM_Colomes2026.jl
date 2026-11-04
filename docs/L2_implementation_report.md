# Level 2 verification, configuration B: implementation report

Tests L2-1 (transfer compliance), L2-2 (Woodbury equivalence), L2-3 (wet
resonance), L2-4 (asymmetric absorption bound), L2-5 (finite-size dilution)
and L2-6 (graded arrays) of the verification test matrix, as implemented in
this repository (state of 2026-10-26, commit `046889b`). The report follows
[L1_implementation_report.md](L1_implementation_report.md). For each test it
gives the module and script, the method, the parameters, the checks, the key
numbers, the deviations from the matrix and the findings.

**Sources referred to**

- Paper J2025-LRMM, staged in [paper/numerical_results.tex](../paper/numerical_results.tex),
  subsection `subsec:num_configB` (L2-1 … L2-6). Labels: `sec:finite_domain`,
  `eq:lat_woodbury`, `eq:lat_absorbed_B`, `eq:renorm_single`, `eq:eigWet`,
  `subsubsec:lat_long_wave`, `eq:mode_dilution`, `eq:x_star_linear`,
  `eq:x_star_exp`, `eq:footprint_general`, `eq:footprint_lin_exp`,
  `eq:mass_grading_rule`, `eq:density_grade`.
- Test matrix: `00_Roadmap/verification_test_matrix_2026-10-22.md` (LRH Dropbox
  folder; frozen rows for Phase 3), which revises the 2026-10-15 matrix, §L2.1–§L2.6.
- Code: `src/L2_1_transfer_compliance.jl` … `src/L2_6_graded_arrays.jl`, and
  the scripts in [scripts/L2-configB-vertical/](../scripts/L2-configB-vertical/).
- Outputs: `data/exp_pro/L2-*/checks.csv` (plus tables), `plots/L2-*/`,
  `data/sims/L2-*/` (DrWatson records, not committed).

---

## 1. What Level 2 verifies

In configuration B the structure is finite and floats in open water. The
Green function of the host is replaced by the transfer compliance
G_B(xᵢ, xⱼ; ω) of the bare structure (`sec:finite_domain`). G_B has no closed
form, so the FE model is the only way to obtain it. Level 2 therefore
verifies (i) the identities that relate the monolithic model to G_B
(reciprocity, `eq:lat_woodbury`, `eq:lat_absorbed_B`), (ii) the wet modal
analysis `eq:eigWet` against the resonance condition written with G_B, and
(iii) two local-theory predictions on finite structures: the size law μ_B
(`subsubsec:lat_long_wave`, `eq:mode_dilution`) and the footprint and taper of
graded arrays.

| Test | Quantity verified | Type | Checks |
|---|---|---|---|
| L2-1 | G_{B,h}: reciprocity, mirror symmetry, mesh and truncation convergence | I + D | 126/126 |
| L2-2 | monolithic solution = Woodbury/Foldy with G_{B,h} | I | 72/72 |
| L2-3 | wet resonance: eq:eigWet vs Re(1/α − G_B) = 0 vs forced peak | I + D | 15/15 |
| L2-4 | absorption bound with asymmetric radiation, `eq:lat_absorbed_B` | D | 114/114 |
| L2-5 | finite-size cut-off of the fluid dilution, μ_B | I + A + D | 46/46 |
| L2-6 | graded arrays: x*, footprint, mass grading, density taper | I + D + A | 359/375 (16 type-A findings) |

### 1.1 Shared machinery

The plan (§3) hosts the shared machinery of a family in its first test. Here
`L2_1` holds the configuration-B channel and the G-column assembly, and later
tests reuse it:

- `L2_1.build`: plate of length L_p with free edges, open water on both sides,
  Sommerfeld condition (`RadiationBC`) with the open-water wavenumber on the
  vertical ends. Open water of max(λ_open, `gap_decays`/κ₁) per side, with κ₁
  the slowest open-water evanescent decay rate; `gap_decays` = 6 by default
  (evanescent reflection ≤ e⁻¹²). Element size h = min(λ_open, λ_p, 4ℓ_f)/n,
  with ℓ_f the flexural decay length. Plate edges and all attachment points
  on nodes; p = 3; vertical grading as L1-1. From commit `6465248` the build
  also accepts structural damping zones (used by L2-6).
- `L2_1.green_columns`: point loads through a massless spring (`ResonatorArray`
  entry with M = 0). Its equation k_ρ(q − η(xⱼ)) = F transmits exactly F and an
  unloaded spring is inert. One factorisation per frequency. The load vector
  is set directly on the spring DOF, b[qⱼ] = A[qⱼ,qⱼ]F/k_ρ, because
  HydroElasticFEM does not assemble right-hand sides on resonator DOFs
  (`has_rhs_form(::ResonatorArray) = false`). The upstream branch
  `fix/resonator-rhs-forcing` adds an opt-in `forced` flag; its PR is still to
  be opened by the PI.
- `L1_1.fit_waves` (from L1) fits the open-water or plate waves. The
  near-field subtraction of L1-3/L1-4 is not needed in open water.
- `L2_3`: modal machinery for `eq:eigWet` (structural/fluid split, fluid Schur
  complement = −ω²A(ω), fixed-point iteration with MAC tracking). L2-5 reuses it.

### 1.2 Metric for mesh and truncation changes

The wet resonances of a free plate are sharp: plate waves are almost totally
reflected at free edges when λ_open ≪ λ_p. A small shift of a resonance then
changes G_B by a large fraction. Changes of G_{B,h} are therefore reported as
an equivalent frequency shift, δω = ‖ΔG‖/‖∂_ωG‖ (`eq:num_freq_equivalent`),
with ∂_ωG from central differences at ω(1 ± 10⁻⁶). The relative change of G
is reported but is not a check.

---

## 2. L2-1: transfer compliance

**Module / script.** `src/L2_1_transfer_compliance.jl` (module `L2_1`),
`scripts/L2-configB-vertical/L2-1_transfer_compliance.jl`.

**Method.** G_{B,h} at 25 points from one factorisation per frequency
(§1.1). Reciprocity and the mirror symmetry of the regular block are checked
in every case. The mesh ladder and the open-water length are varied.

**Parameters.** P1 plate, T_ρ = 0, L_p = 20 m, H = 10 m. 20 regular points
(x = 0.5 … 19.5 m) and 5 random points. ω ∈ {2, 5, 9.8, 10, 10.2} rad/s. Mesh
n ∈ {4, 8, 16}, plus n = 32 at ω ∈ {2, 5}. Open water 6/κ₁ (6.3–25 m).
Truncation: 4, 6 and 8 decay lengths at n = 16. 37 records.

**Checks (126/126).**

| Check | Type | n | Max value | Tolerance |
|---|---|---|---|---|
| ‖G − Gᵀ‖/‖G‖ | I | 37 | 1.7e-9 | 1e-8 |
| mirror symmetry of the regular block | I | 37 | 3.9e-9 | 1e-8 |
| spring residual (point-load construction) | I | 37 | 4.4e-16 | 1e-12 |
| mesh δω/ω, two finest levels (8 vs 16; 16 vs 32 at ω = 2, 5) | D | 5 | 1.4e-5 | 1e-4 |
| truncation δω/ω, 6 vs 8 decay lengths | D | 5 | 1.8e-8 | 1e-5 |
| truncation change decreases with the gap (ratio 6→8 / 4→6) | D | 5 | 0.10 | 1 |

**Key numbers.** Reciprocity 1e-14 … 1.7e-9, growing with the number of
unknowns as round-off does. Observed mesh order of δω 2.0–2.7. The truncation
change 6→8 is 2–10 % of the change 4→6, consistent with e^{−2κ₁Δ}. Im G_B(x,x)
varies by a factor of up to 6 between nodes and antinodes of the wet modes at
ω ≈ 10 (figure `plots/L2-1/diagonal.png`).

**Deviations from the matrix (§L2.1).**
- Reciprocity is a type-I identity (≤ 1e-8), not a truncation diagnostic. The
  Sommerfeld condition keeps the discrete operator complex symmetric.
- The mesh and truncation checks use δω/ω, not ‖ΔG‖/‖G‖ (§1.2).
- Zone lengths {1, 2, 4, 8}λ_open and gaps {¼, ½, 1}λ_open are replaced by
  open water of {4, 6, 8}/κ₁ with a Sommerfeld end. There are no absorbing zones.
- The truncation study runs on n = 16. On n = 8 the discrete radiation
  condition, posed with the exact k, leaves a spurious reflection of about 1e-6
  whose phase depends on the domain length. It masks the e^{−2κ₁Δ} decay at ω ≥ 5.

---

## 3. L2-2: Woodbury equivalence

**Module / script.** `src/L2_2_woodbury.jl` (module `L2_2`),
`scripts/L2-configB-vertical/L2-2_woodbury.jl`.

**Method.** Resonators at the 20 regular points of L2-1 and a point load at
x = 7.25 m. The monolithic solution (resonators as unknowns) is compared with
`eq:lat_woodbury`, (diag(1/αᵢ) − G_{B,h}) f = η_inc, using G_{B,h} and η_inc
from the bare model on the same mesh.

**Parameters.** L2-1 channel, n = 8, p = 3. Sets: identical; random (ω_r, ζ,
m_ρ,r, fixed seed); linearly graded. 12 frequencies,
ω ∈ {2, 5, 8, 8.5, 9, 9.5, 9.8, 10, 10.2, 10.5, 11, 12}. 36 cases.

**Checks (72/72).** ‖η_mono − η_Foldy‖/‖η_mono‖ ≤ 1e-8 (max 2.6e-10) and the
same for q (max 2.5e-10), in all 36 cases.

**Key numbers.** The differences are 9e-13 … 2.6e-10. This is the level of the
reciprocity residual on the same systems (≤ 8.9e-11, up to 3.5e5 unknowns at
ω = 12), which is pure round-off. So `eq:lat_woodbury` is an identity of the
discrete model.

**Deviations.**
- Tolerance 1e-8 instead of 1e-10. The round-off of the sparse direct solver
  reaches 5e-10 on these systems, so 1e-10 is below the attainable level.
- 12 frequencies in [2, 12] (refined near ω_r) instead of 21 in [8, 12].

---

## 4. L2-3: wet resonance of a single resonator

**Module / script.** `src/L2_3_wet_resonance.jl` (module `L2_3`),
`scripts/L2-configB-vertical/L2-3_wet_resonance.jl`.

**Method.** The resonator-dominated wet mode of set P2 is computed in three ways:
1. *Green function*: ω̃ is the root of Re(1/α − G_{B,h}(x₁,x₁)), found by secant iteration.
2. *Modal*: `eq:eigWet` with Re A(ω), by the fixed-point iteration of
   `fig:modal_iteration`. The assembled matrix is split into structural and
   fluid blocks, and the fluid Schur complement gives −ω²A(ω) (in blocks since
   `6465248`). The resonator rows are rescaled so that the wet matrices are
   symmetric. The mode is tracked by MAC on the resonator participation.
3. *Forced*: peak of |q/G_B(x₁, x_s)|, found by golden-section search.

The complex wet frequency with the full A(ω) is also reported.

**Parameters.** P2: membrane L_m = 50 m on a 150 m channel, H = 25 m,
ρ = 1025, T_ρ = 9.7561, m_ρ = 0.0444. Resonator ω_r = 0.5 rad/s, lossless.
x₁ ∈ {0, 12.5, 25} m. p = 3 with 64, 128 and 256 elements per wavelength at ω_r.

**Checks (15/15).**

| Check | Type | Max | Tolerance |
|---|---|---|---|
| fixed-point iterations | I | 3 | < 30 |
| mesh error of ω̃, 128 vs 256 | D | 5.5e-5 | 1e-4 |
| mesh error of ω_wet, 128 vs 256 | D | 5.5e-5 | 1e-4 |
| ω_wet vs ω̃ | D | 4.1e-5 | 1e-3 |
| forced peak vs ω̃ | D | 3.6e-9 | 1e-3 |

**Key numbers.** The resonance is lowered by 1.4e-3 ω_r at the edge and by
6.7e-4 ω_r at x₁ = 12.5 and 25 m. The complex wet frequency at the centre is
0.499663 − 1.56e-5 i (the matrix quotes 0.4997 − 1.6e-5 i).

**Finding.** ω_wet lies below ω̃ by 2.0–4.1e-5, i.e. 0.7–1.3 half-widths. The
difference does not decrease with the mesh. It comes from the two definitions
of an undamped resonance: `eq:eigWet` drops radiation damping before the
condensation onto the resonator, while ω̃ drops it after, through Re G_B. The
two agree to within the physical resolution of the resonance.

**Deviations.**
- Mesh ladder in elements per wavelength (64/128/256 at ω_r, h ≈ 2.4/1.2/0.6 m)
  instead of dx = {0.8, 0.4, 0.2} m, ny = {8, 15, 30}.
- p = 3 instead of the P2 membrane order of the old results.
- "Agree to D tolerance" is set to 1e-3, with the mesh error at 1e-4.

---

## 5. L2-4: absorption by a single resonator on a finite structure

**Module / script.** `src/L2_4_asymmetric_absorption.jl` (module `L2_4`),
`scripts/L2-configB-vertical/L2-4_asymmetric_absorption.jl`.

**Method.** At the renormalised resonance ω̃ of the lossless resonator:
1. A unit force on a massless spring at x₁ gives G₁₁ and the radiated
   amplitudes a^±, fitted in both open-water regions. The power ½ω Im G₁₁ is
   checked against ½g c_g(|a⁻|² + |a⁺|²).
2. The bare plate is solved under a wave from the left, which gives η_inc(x₁).
3. 𝒜(ζ) from the one-resonator Woodbury formula, for 41 ζ ∈ [1e-3, 1e-1]
   and at critical coupling γ_d = Im G₁₁.
4. One monolithic solve at critical coupling, with 𝒜 from the dashpot power.

Incidence from the right is the mirror position L_p − x₁.

**Parameters.** P1 plate, T_ρ = 0, H = 10 m. m_ρ,r = 0.01 m², ω_r = 10 rad/s.
L_p ∈ {5, 20} m. x₁ ∈ {L_p/2, L_p/4, L_p/8, 0.25 m} and the mirror positions.
n = 8; n = 16 for L_p = 5 and for x₁ ∈ {L_p/2, 0.25} at L_p = 20. 23 cases.

**Checks (114/114).**

| Check | Type | n | Max | Tolerance |
|---|---|---|---|---|
| ½ω Im G vs open-water flux | D | 23 | 1.2e-5 | 1e-3 |
| max_ζ 𝒜 vs bound | D | 23 | 8.2e-4 | 1e-3 |
| 𝒜 at critical coupling vs bound | D | 23 | 1.8e-4 | 1e-3 |
| monolithic vs Woodbury 𝒜 | D | 23 | 5.5e-6 | 1e-5 |
| bound = ½ at the centre | D | 4 | 5.6e-10 | 1e-3 |
| bound(x₁) + bound(L_p − x₁) = 1 | D | 9 | 7.5e-10 | 1e-3 |
| bound on two meshes | D | 9 | 6.1e-5 | 1e-3 |

**Key numbers.** The power identity holds to 1.2e-5 (n = 8) and 2e-7 (n = 16).
The bound ranges over 0.088–0.912 (L_p = 5) and 0.10–0.90 (L_p = 20). At
x₁ = L_p/8 on the 5 m plate one critically damped resonator absorbs 91 % of the
incident power. ζ_crit ranges over 0.003–0.013 and is proportional to γ_r(x₁).
The bound follows the phase of the wet mode at x₁, not the distance to the
edge. This settles the PROVISIONAL "> ½" clause of the introduction.

**Deviations.**
- The ζ sweep has 41 points instead of 40, and is evaluated by Woodbury, which
  is exact for the discrete problem (L2-2), plus one monolithic check per case.
- Incidence from +∞ is realised as the mirror position.
- Only ω̃ (the renormalised resonance), not ω_r.

---

## 6. L2-5: finite-size cut-off of the fluid dilution

**Module / script.** `src/L2_5_finite_size_dilution.jl` (module `L2_5`),
`scripts/L2-configB-vertical/L2-5_finite_size_dilution.jl`.

**Method.**
1. The first elastic wet mode (ω₁, φ) of the bare free plate, from
   `eq:eigWet` (L2-3 machinery; secant fixed point, blocked Schur complement).
2. N = round(L/a) lossless resonators at x_i = (i − ½)a, a ≈ 1 m, tuned to ω₁,
   with n₀m_ρ,r = 0.01 m.
3. The two hybridised wet modes ω∓, started from the two modes of largest
   projection on φ. Splitting Δ = (ω₊ − ω₋)/ω₁.

Δ is compared with:
- √μ_mode, with μ_mode = m_ρ,r Σφ(xᵢ)²/φᵀ(M_η + Re A(ω₁))φ;
- the single-mode Galerkin solution with the dispersive modal added mass
  m_φ(ω) = φᵀ(M_η + Re A(ω))φ, using a one-solve modal mass;
- √μ_B of the paper, and √μ(k₁) with k₁ = β₁/L, β₁ = 4.730.

**Parameters.** P1 plate, T_ρ = 0. L ∈ {2.3, 4.6, 11.5, 23, 46} m.
H ∈ {3, 10, 30} m, but L = 46 m at H = 30 is not run (`DEEP_L_MAX` = 23).
n = 16, plus n = 32 at H = 10 for L ≤ 11.5 (`MESH_L_MAX`). 14 main cases and 3
mesh-check cases.

**Checks (46/46).**

| Check | Type | n | Max | Tolerance |
|---|---|---|---|---|
| hybrid pair identified (projection on φ ≥ 0.2) | I | 14 | min 0.981 | ≥ 0.2 |
| one-solve modal mass vs full added-mass matrix | I | 14 | 1.5e-13 | 1e-6 |
| Δ vs single-mode Galerkin | A | 14 | 2.2e-3 | 1e-2 |
| deep-water exponent \|s + ½\| (H = 30, N ≥ 5, L ≤ 23) | A | 1 | 0.037 | 0.1 |
| Δ, mesh 16 vs 32 (H = 10, L ≤ 11.5) | D | 3 | 7.8e-4 | 1e-2 |

**Key numbers.** The deep-water exponent is −0.54 (predicted −½). The shallow
slope (H = 3) steepens from −0.59 (L = 4.6–23) to −0.79 (23–46), towards −1.
Galerkin agrees to 2.2e-3, and to 6e-4 for N ≥ 5. √μ_mode agrees to 1–6.5 %,
the difference being the dispersion of the added mass between ω₋ and ω₊.

**Findings** (handoff §4.1–4.2).
- The exponents of μ_B are confirmed. The prefactor underestimates Δ by a
  factor of 1.7–1.8 (deep and intermediate water; 1.8–2.8 at H = 3). The local
  √μ(β₁/L) underestimates it by 1.4–1.5 (H ≥ 10).
- At L = 2.3 m (N = 2) the resonators sit near the nodes of the free–free mode
  (0.22L, 0.78L), so Δ is small. √μ_mode captures this sampling effect; no
  continuum estimate can.
- For `eq:mode_dilution` (L4-8) this supports the cut-off and the scaling, with
  a prefactor of O(1.5) that the wet modal mass gives exactly.

**Deviations.**
- **DEEP_L_MAX.** L = 46 m at H = 30 m is dropped: it exceeds the 60 CPU-min
  `run_limited` budget. The deep-water exponent uses L ≤ 23 m
  (L < πH/4 = 23.6 m).
- **N ≥ 5 fit.** The deep-water exponent is a least-squares log–log slope
  over the cases with N ≥ 5 only: L ∈ {4.6, 11.5, 23} m (N = 5, 12, 23) at
  H = 30 m. The N = 2 case (L = 2.3 m) samples the mode near its nodes.
- The mesh check (16 vs 32) is limited to L ≤ 11.5 m at H = 10, for the same
  budget reason.
- ω_r is tuned to the bare first elastic wet mode, and Δ comes from
  `eq:eigWet`, not from a 401-point forced sweep in [8, 12].
- The main type-A check is FE vs single-mode Galerkin (to 1 %), not FE vs √μ_B.
  √μ_B is reported (its prefactor is a finding, not a pass/fail item).

---

## 7. L2-6: graded arrays

**Module / script.** `src/L2_6_graded_arrays.jl` (module `L2_6`),
`scripts/L2-configB-vertical/L2-6_graded_arrays.jl`.

**Method.**
- *Plate source with sponges.* Incidence from open water is not usable. At the
  free edge |R| > 0.99 for ω ≳ 2 (λ_open ≪ λ_p), so almost no power enters the
  plate (found on the first trial). Instead a unit point force on a bare
  lead-in generates the plate wave. Viscous `StructuralDampingZone` sponges
  (6λ_p, loss factor 1, quadratic ramp) absorb it at both plate ends.
  Layout: sponge | buffer | source | W1 | array | W2 | buffer | sponge, with
  every separation 5/κ₁ and windows of 4λ_p.
- *Woodbury.* The bare plate is factorised once per frequency, and every array
  enters through `eq:lat_woodbury` with G_{B,h} = PᵀA⁻¹P. P evaluates the plate
  shape functions at the attachment points (an evaluation matrix, checked
  against spring-based columns).
- R and T come from fits in W1 and W2.
- *Exact net-flux energy balance* (commit `046889b`). The left sponge
  re-reflects part of R towards the array, so |R|² + |T|² + ΣP = 1 is not
  exact. The check uses the net plate-wave flux into the array,
  (|A⁺₁|² − |A⁻₁|² − |A⁺₂|² + |A⁻₂|²)/|A⁺_{1,0}|², against ΣP. This balance is
  exact. The R/T balance is still stored (`balance_RT`).
- Observables: dissipated power per unit length, its peak and FWHM vs x* and
  Δx; band-averaged |R|² of taper vs abrupt array.

**Parameters.** P1 plate, T_ρ = 0, H = 10 m, p = 3, n = 8.
- *Frequency set*: ω ∈ {2, 3, 5, 7}. Linear and exponential grading from
  ω_r0 = 10 with L_g = 500 m down to ω_r = 1 (x ≤ 450 m linear, 1151 m
  exponential). a ∈ {0.25, 1} m. ζ ∈ {0.01, 0.02, 0.05} with uniform loading
  n₀m_ρ,r = 0.01 m. Mass-graded loading (`eq:mass_grading_rule`, √μ* = 0.163)
  at ζ = 0.02.
- *Density set*: 41 ω in [9.5, 10.5]. ω_r = 10, a = 1 m, n = n₀e^{−x/L_g} on
  [0, L_g], L_g ∈ {1, 5, 20, 100}λ (λ = 2.32 m). ζ ∈ {0, 0.02}. Abrupt arrays
  with N_eff = round(Σe^{−xᵢ/L_g}) resonators.
- Mesh check n = 12 at ω ∈ {5, 7} and at ω = 10 (density).
- 48 per-solve records (4 + 2 frequency-set solves, 41 + 1 density-set solves).

**Checks (359/375).**

| Check | Type | n | Passing | Max / range | Tolerance |
|---|---|---|---|---|---|
| G_{B,h} evaluation matrix vs spring columns | I | 48 | 48 | 7.5e-10 | 1e-8 |
| reciprocity of G_{B,h} | I | 48 | 48 | 1.7e-9 | 1e-8 |
| net-flux energy balance (max over arrays) | I | 48 | 48 | 9.3e-7 | 1e-3 |
| bare transmission W1 → W2 | I | 48 | 48 | 2.1e-7 | 1e-3 |
| sponge reflection | D | 48 | 48 | 1.4e-3 | 1e-2 |
| mesh 8 vs 12: peak shift/Δx, FWHM, 𝒜, \|R\| | D | 64 | 64 | 4.8e-3 | 0.05 / 0.05 / 1e-2 / 1e-2 |
| peak within Δx/2 of x* (a = 0.25) | A | 32 | 26 | up to 1.01 | 0.5 |
| FWHM/Δx ∈ [0.5, 2] (a = 0.25) | A | 32 | 23 | 0.26–1.99 | [0.5, 2] |
| mass-graded exponential: max/min FWHM over ω | A | 1 | 0 | 1.80 | 1.3 |
| taper vs abrupt, band-averaged \|R\|² (L_g ≥ 5λ) | A | 6 | 6 | 0.77–0.99 | < 1 |

All type-I and type-D checks pass. All 16 failures are type A (local-theory
acceptance, set before the runs). They are reported as findings, not bugs.

**The 16 failing type-A checks.**

| # | Check | Case (a = 0.25 m) | Value |
|---|---|---|---|
| 1 | FWHM/Δx | linear, mass-graded, ζ = 0.02, ω = 2 | 0.29 |
| 2 | FWHM/Δx | exp, mass-graded, ζ = 0.02, ω = 2 | 0.498 |
| 3 | FWHM/Δx | exp, uniform, ζ = 0.01, ω = 2 | 0.34 |
| 4 | FWHM/Δx | linear, uniform, ζ = 0.01, ω = 2 | 0.32 |
| 5 | FWHM/Δx | linear, mass-graded, ζ = 0.02, ω = 3 | 0.46 |
| 6 | FWHM/Δx | exp, uniform, ζ = 0.01, ω = 3 | 0.45 |
| 7 | FWHM/Δx | linear, uniform, ζ = 0.01, ω = 3 | 0.26 |
| 8 | FWHM/Δx | linear, uniform, ζ = 0.01, ω = 5 | 0.31 |
| 9 | FWHM/Δx | linear, uniform, ζ = 0.01, ω = 7 | 0.41 |
| 10 | peak/Δx | exp, mass-graded, ζ = 0.02, ω = 3 | 0.515 (marginal) |
| 11 | peak/Δx | exp, uniform, ζ = 0.05, ω = 3 | 0.81 |
| 12 | peak/Δx | linear, uniform, ζ = 0.05, ω = 5 | 0.61 |
| 13 | peak/Δx | exp, uniform, ζ = 0.05, ω = 5 | 1.01 |
| 14 | peak/Δx | linear, uniform, ζ = 0.05, ω = 7 | 0.87 |
| 15 | peak/Δx | exp, uniform, ζ = 0.05, ω = 7 | 0.99 |
| 16 | mass-graded max/min FWHM | exp, ζ = 0.02, ω ∈ {2, 3, 5, 7} | 1.80 |

**Findings** (handoff §2, §4.3–4.4; paper §L2-6).
1. **Footprint width.** FWHM ≈ c·2ζ·ω|dx*/dω|. The width is controlled by
   damping, also where √μ > 2ζ, so FWHM/Δx falls to 0.26–0.45 at ζ = 0.01
   (checks 1–9). The factor c rises with array opacity: ≈ 1 at ω = 2 (linear
   array absorbs 40 %), 1.2 at ω = 3, 2 at ω = 5, 3 at ω = 7 (absorption
   > 99.99 %). √μ bounds the frequency window of the branch, not the spatial
   width. It enters through the depletion of the wave upstream of x*, which
   broadens the footprint. The paragraph after `eq:footprint_lin_exp` needs
   rewording (theory card).
2. **Peak position.** x* is confirmed for ζ ≤ 0.02 (one marginal case, 0.52Δx,
   check 10). At ζ = 0.05 the peak moves upstream by 0.6–1.0Δx (checks 11–15),
   because the wave is absorbed before it reaches x*.
3. **Mass-graded exponential law.** The FWHM is 0.8–0.9Δx and flat to 13 % for
   ω ∈ [3, 7], against a factor of 3.2 with uniform loading. Including ω = 2,
   where the footprint is half as wide, max/min = 1.8 (check 16).
4. **Density taper** (passes). It reflects less than the abrupt array for
   L_g ≥ 5λ. The ratio of band-averaged |R|² is 0.77/0.86/0.92 (L_g = 5/20/100λ,
   ζ = 0) and 0.79/0.94/0.99 (ζ = 0.02). This is a modest gain that shrinks
   with L_g. 𝒜 rises from 0.26 (L_g = λ) to 0.88 (L_g ≥ 20λ) at ζ = 0.02.
5. **Open-water incidence on a free plate** gives |R| > 0.99 at the edge for
   ω ≳ 2 (handoff §4.6). This is a Discussion item.

**Deviations.**
- **Plate source with sponges** instead of incidence from open water (see
  Method). The test matrix row assumed incidence from −∞.
- **Exact net-flux energy balance** instead of |R|² + |T|² + ΣP = 1, which is
  not exact with a re-reflecting upstream sponge.
- Type-A tolerances (peak within Δx/2, FWHM/Δx ∈ [0.5, 2], mass-graded
  max/min ≤ 1.3, taper < abrupt for L_g ≥ 5λ) are applied to a = 0.25 m only.
  At a = 1 m, ε_s reaches 0.36 and the result is reported only.
- L_g = λ in the density set is reported, not checked (λ/L_g = 1, not adiabatic).
- The predicted Δx follows `eq:footprint_general` (2026-10-15 matrix values).
  The finding above supersedes it for the width.

---

## 8. Deviations from the test matrix (summary)

| Test | Matrix item | Implementation | Reason |
|---|---|---|---|
| L2-1 | reciprocity as a truncation diagnostic, ≤ 1e-6 at finest | type-I identity ≤ 1e-8, every case | Sommerfeld end keeps the operator complex symmetric |
| L2-1 | ‖ΔG‖/‖G‖ | δω/ω (`eq:num_freq_equivalent`) | sharp wet resonances |
| L2-1 | zones {1, 2, 4, 8}λ_open, gaps {¼, ½, 1}λ_open | Sommerfeld end, open water {4, 6, 8}/κ₁; truncation on n = 16 | evanescent field sets truncation; 8-element radiation condition leaves ~1e-6 reflection |
| L2-2 | ≤ 1e-10, 21 ω in [8, 12] | ≤ 1e-8, 12 ω in [2, 12] | solver round-off reaches 5e-10 |
| L2-3 | dx = {0.8, 0.4, 0.2}, ny = {8, 15, 30} | 64/128/256 el./λ, p = 3 | uniform mesh rule of the suite |
| L2-4 | 40 ζ, incidence from ±∞ | 41 ζ by Woodbury + monolithic check; mirror positions | Woodbury exact (L2-2); symmetry |
| L2-5 | all (L, H); 401 ω forced sweep | L = 46 m at H = 30 dropped (DEEP_L_MAX); eq:eigWet | 60 CPU-min budget |
| L2-5 | exponent over all L | N ≥ 5 fit | N = 2 samples the mode near its nodes |
| L2-6 | incidence from −∞ | plate source on a bare lead-in, viscous sponges 6λ_p | free-edge \|R\| > 0.99 |
| L2-6 | energy identity \|R\|² + \|T\|² + ΣP = 1 | exact net-flux balance | sponge re-reflection |
| L2-6 | type-A at a ∈ {1, 0.25} | type-A at a = 0.25 only | ε_s up to 0.36 at a = 1 |

The L1 conventions are carried forward where they apply: finest-mesh
tolerances instead of Richardson extrapolation, and observed rates reported
for p = 3 but not used as criteria.

---

## 9. Records, figures and paper

- Records: `data/sims/L2-*/` (DrWatson, git commit and HydroElasticFEM pin;
  not committed).
- Tables: `data/exp_pro/L2-1/{errors,convergence,diagonal,checks}.csv`,
  `L2-2/{errors,checks}.csv`, `L2-3/{errors,checks}.csv`,
  `L2-4/{errors,absorption,checks}.csv`, `L2-5/{errors,checks}.csv`,
  `L2-6/{footprint,density,checks}.csv`.
- Figures → paper (`scripts/export_paper_assets.jl`): `plots/L2-1/diagonal.png`
  → `L2_1_diagonal.png`; `L2-2/woodbury.png` → `L2_2_woodbury.png` (appendix);
  `L2-3/resonance.png` → `L2_3_resonance.png`; `L2-4/bound.png`,
  `absorption.png` → `L2_4_bound.png`, `L2_4_absorption.png`;
  `L2-5/splitting.png` → `L2_5_splitting.png`; `L2-6/profiles.png`,
  `footprint.png`, `density.png` → `L2_6_*.png`.
- Paper: `subsec:num_configB` in `paper/numerical_results.tex` (red wrapper).

## 10. Open points

1. Count mismatches between the paper text and `checks.csv`, to reconcile in
   the consistency pass. The L2-1 text says "all 42 cases" but there are 37
   records. The L2-6 text says "45 solves" but there are 48 records
   (6 frequency-set + 42 density-set, including mesh checks).
2. L2-5: L = 46 m at H = 30 m (deep-water exponent over four lengths) would
   need more than the per-run budget, or a cluster.
3. Theory feedback (handoff §4): μ_B prefactor, `eq:mode_dilution`
   prefactor O(1.5), the footprint paragraph after `eq:footprint_lin_exp`,
   and the density taper stated as a modest gain.
4. Upstream: the PR for `fix/resonator-rhs-forcing` (forced flag on
   `ResonatorArray` DOFs) would replace the massless-spring load workaround.
