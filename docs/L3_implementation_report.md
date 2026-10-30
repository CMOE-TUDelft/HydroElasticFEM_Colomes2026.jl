# Level 3 (3D): implementation report

Phase 4 of the numerical programme (3D fluid with a plate): L3-1, L3-2, L3-3,
L4-4 and L4-8. This report starts with the sizing card L3-0 (state of
2026-10-28), which sets how each 3D test can be run. Sections for L3-1 … L4-8
will be added as they land. Section L3-0b documents the METIS-ordered
direct solver used for the 3D runs.

**Sources**

- Plan: `00_Roadmap/numerical_tests_plan_2026-10-22.md`, "Phase 4", and
  `00_Roadmap/agent_handoff_2026-10-26.md` §5 item 4 (LRH Dropbox folder).
- Code: `src/L3_0_memory_sizing.jl` (module `L3_0`) and
  [scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl](../scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl).
- Outputs: `data/exp_pro/L3-0/` (`sizing.csv`, `fill.csv`, `fits.csv`,
  `calibration.csv`, `targets.csv`, `checks.csv`), `plots/L3-0/sizing.{pdf,png}`,
  and `data/sims/L3-0/` (DrWatson records; not committed).

---

## L3-0: 3D memory sizing

### Test description

Configuration A in 3D: a box of fluid (φ, continuous Q_p) fully covered by a
Kirchhoff–Love plate (η, Q_p, C/DG with γ = 6). Set P1, ω = 10, a line load,
and Kelvin–Voigt sponges at both x ends. The cells are uniform horizontally
(h = λ_p/8). In the vertical there are `nz` cells, graded so that the top
cell is k₀h_top = 0.2. The linear solve is the library default (`LUSolver`,
UMFPACK with its default ordering, which is AMD on A + Aᵀ for this
symmetric pattern). Every case is one Julia process. Each run goes through
`run_limited.sh` (4 GB, 1 thread) under the shared heavy-run lock, with an
RSS watchdog that samples every second and kills the run above 4 GiB. A
warm-up case pays the compilation first.

There are two kinds of case:

1. **Full cases** (assemble, factorise, solve). They give N, nnz(A), the LU
   fill (lunz), the UMFPACK peak, the process peak RSS, the assembly time and
   the factorisation+solve time. Square ladders: p = 2, nz = 8; p = 3, nz = 8;
   p = 2, nz = 16. There is also one channel case (128×16×8) and four cases
   with UMFPACK's METIS ordering.
2. **Analysis cases** (assemble, then a CHOLMOD symbolic analysis only). They
   give the exact Cholesky fill nnz(L) and the flop count of the pattern,
   with AMD and with METIS. Their memory is that of the assembly, so they
   reach N ≈ 480 k, which is 6× beyond the largest factorisable case. They
   include some target meshes themselves (the L3-1 channel and four L4-8
   meshes).

The cost model is calibrated on the full cases. It is then applied to the
fill of the larger meshes:

- lunz_UMF = κ_L (2 nnz(L) − N);
- M_LU = β · lunz_UMF;
- peak RSS = R₀ + r·N + M_LU;
- t_fact+solve = flops / rate.

For targets that were not analysed directly, nnz(L) and flops come from
power laws in N per (p, nz), fitted on the squares with N ≥ 30 k. Channels
(L3-1, L3-2) instead scale as N·w and N·w², where w ∝ p²·ny·nz is the front
width.

### Results

**Measured ladder (full cases, 1 thread).** All cases with N ≤ 76 k fit
under the cap. The largest are p=2 32×32×8 (3.5 GB), p=3 16×16×8 (3.9 GB)
and p=2 20×20×16 (3.9 GB). p=2 24×24×16 (N = 81.6 k) is killed above 4 GiB.
After the warm-up, the Julia process alone holds R₀ = 1.5–2.0 GB, so a case
has only about 2 GB of headroom under the cap.

| ordering | p | nx×ny×nz | N | nnz(A) | lunz (LU fill) | lunz/nnz | UMFPACK peak GB | peak RSS GB | assembly s | fact.+solve s |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 2 | 16×16×8 | 19602 | 1.23M | 16.0M | 13.0 | 0.34 | 2.21 | 0.9 | 1.9 |
| default | 2 | 20×20×8 | 30258 | 1.92M | 28.7M | 15.0 | 0.58 | 2.23 | 1.2 | 4.2 |
| default | 2 | 24×24×8 | 43218 | 2.75M | 51.2M | 18.6 | 0.99 | 2.96 | 1.7 | 10.1 |
| default | 2 | 28×28×8 | 58482 | 3.74M | 79.5M | 21.2 | 1.46 | 2.70 | 2.4 | 18.0 |
| default | 2 | 32×32×8 | 76050 | 4.88M | 107.0M | 21.9 | 2.19 | 3.47 | 2.9 | 26.5 |
| default | 2 | 12×12×16 | 21250 | 1.30M | 16.8M | 12.9 | 0.34 | 2.38 | 0.9 | 2.0 |
| default | 2 | 16×16×16 | 37026 | 2.29M | 39.2M | 17.1 | 0.80 | 2.70 | 1.4 | 7.0 |
| default | 2 | 20×20×16 | 57154 | 3.57M | 76.9M | 21.5 | 1.41 | 3.88 | 2.1 | 18.4 |
| default | 3 | 8×8×8 | 16250 | 1.93M | 10.2M | 5.3 | 0.23 | 2.52 | 1.6 | 0.9 |
| default | 3 | 10×10×8 | 24986 | 3.01M | 21.1M | 7.0 | 0.49 | 2.89 | 2.4 | 2.8 |
| default | 3 | 12×12×8 | 35594 | 4.33M | 35.4M | 8.2 | 0.76 | 2.89 | 3.4 | 5.7 |
| default | 3 | 14×14×8 | 48074 | 5.89M | 52.5M | 8.9 | 0.99 | 3.19 | 4.5 | 9.3 |
| default | 3 | 16×16×8 | 62426 | 7.68M | 80.0M | 10.4 | 1.65 | 3.89 | 5.7 | 17.9 |
| default | 2 | 128×16×8 (channel) | 152658 | 9.77M | 176.5M | 18.1 | 2.82 | 3.56 | 5.8 | 30.7 |
| metis | 2 | 24×24×8 | 43218 | 2.75M | 28.7M | 10.4 | 0.48 | 2.99 | 1.8 | 3.2 |
| metis | 2 | 32×32×8 | 76050 | 4.88M | 59.0M | 12.1 | 0.96 | 3.81 | 2.9 | 7.7 |
| metis | 3 | 12×12×8 | 35594 | 4.33M | 24.9M | 5.8 | 0.48 | 3.19 | 3.4 | 2.7 |
| metis | 3 | 16×16×8 | 62426 | 7.68M | 52.4M | 6.8 | 0.91 | 3.47 | 5.8 | 6.8 |

**Analysis cases (symbolic Cholesky fill).** The table `fill.csv` lists all
of them. At N = 467 k (p=2, 80×80×8), nnz(L) is 6.7e8 with AMD and 2.8e8
with METIS, and the flop counts are 4.2e12 and 5.6e11. METIS thus cuts the
fill by 2.4× and the flops by 7.6×, and both gaps grow with N.

**Fit exponents** (C·N^α; `fits.csv`):

| quantity | p=2, nz=8 | p=3, nz=8 | p=2, nz=16 | data |
|---|---|---|---|---|
| nnz(L), AMD | 1.41 | 1.44 | 1.52 | squares, N = 30 k–470 k |
| flops, AMD | 2.00 | 2.07 | 2.26 | idem |
| nnz(L), METIS | 1.25 | 1.29 | 1.33 | idem |
| flops, METIS | 1.68 | 1.77 | 1.86 | idem |
| UMFPACK peak (measured) | 1.38 | 1.40 | 1.46 | full cases, N ≤ 76 k |
| fact.+solve time (measured) | 2.00 | 2.13 | 2.25 | idem |
| assembly time | 0.98 | 1.00 | 0.97 | all cases |

nz is fixed (8 or 16), so the meshes are thick 2D slabs. With nested
dissection, the exponents tend towards the 2D values (N log N and N^1.5)
only once the horizontal extent is much larger than the depth. AMD is far
from that limit.

**Calibration** (`calibration.csv`, squares with N ≥ 30 k):

- κ_L = 1.0009 (spread 0.03 %): UMFPACK's fill equals the Cholesky fill of
  A + Aᵀ, because there is no pivoting fill.
- β = 21.6 bytes per factor entry with the default ordering (spread 17 %),
  and 18.2 with METIS.
- rate = 4.35 GFlop/s with the default ordering (spread 8 %), and 3.6 with
  METIS.
- r = 3.9 kB/DOF for p = 2 and 5.0 kB/DOF for p = 3, taken from the
  assembly memory of the analysis cases.
- R₀ = 1.8 GB.

Against the measured cases, the model gives the time within 0.7–1.4× in
every case. The peak RSS is within 0.75–1.6×. The model is up to 25 % low
only on the smallest cases and on the METIS cases, where Julia's heap slack
dominates. At target sizes the LU term is more than 80 % of the peak.

**Per-test projections.** The rows below are the resolution of each matrix
row (p = 2, h = λ_p/8, nz = 8), plus every L3-2 and L4-8 variant.
`targets.csv` also holds the refined variants (h = λ_p/16, nz = 16 and
p = 3). All of them need more than 40 GB even with METIS, except L3-1 with
nz = 16 (24 GB with METIS). Each verdict uses the default solver; the column
before it gives the peak with a METIS ordering. Times are 1 thread.

| test | variant | N | peak GB, default LU | peak GB, METIS | fact.+solve per solve (default / METIS) | solves | total (default / METIS) | verdict (default LU) |
|---|---|---|---|---|---|---|---|---|
| L3-1 | θ=30°, W=2λp, channel (mesh analysed) | 0.40 M | 13.2 | 8.5 | 91 s / 41 s | 3 | 0.1 h / 0.05 h | needs the 24 GB machine |
| L3-1 | same, nz = 16 | 0.75 M | 42 | 24 | 12 min / 5 min | 3 | 0.6 h / 0.3 h | iterative / symmetry / cluster |
| L3-2 | VT3D-B level L1, channel | 0.058 M | 2.8 | 2.4 | 3 s / 1.5 s | 6 | < 1 min | fits the 4 GB cap |
| L3-2 | VT3D-B level L2 | 0.073 M | 3.2 | 2.7 | 7 s / 3 s | 6 | 1 min | fits the 4 GB cap (marginal) |
| L3-2 | VT3D-B level L3 | 0.10 M | 4.5 | 3.4 | 18 s / 8 s | 6 | 2 min | needs the 24 GB machine (4 GB with METIS) |
| L3-3 | matrix domain 16λp square + 4λp frame, quarter | 0.73 M | 58 | 21 | 45 min / 6 min | 41 | 31 h / 4.2 h | iterative / symmetry / cluster |
| L3-3 | gap 5/κ₁ + 4λp frame, quarter | 1.57 M | 167 | 51 | 3.5 h / 21 min | 41 | 143 h / 15 h | iterative / symmetry / cluster |
| L4-4 | island 12×6λp, gap 2λp, half | 1.00 M | 91 | 30 | 86 min / 10 min | 24 | 35 h / 4.2 h | iterative / symmetry / cluster |
| L4-4 | island 12×6λp, gap 5/κ₁, half | 4.60 M | 747 | 185 | 30 h / 2.1 h | 24 | 725 h / 51 h | iterative / symmetry / cluster |
| L4-8 | ω=2.74, gap λ0, half (analysed) | 0.20 M | 9.2 | 5.7 | 104 s / 34 s | 40 | 1.2 h / 0.5 h | needs the 24 GB machine |
| L4-8 | ω=2.74, gap λ0, half, nz=16 (analysed) | 0.37 M | 24 | 11.7 | 12 min / 3 min | 40 | 7.8 h / 2.1 h | iterative / symmetry / cluster (24 GB with METIS) |
| L4-8 | ω=2.74, gap 5/κ₁, half (analysed) | 0.40 M | 20.9 | 10.8 | 6.6 min / 1.9 min | 40 | 4.6 h / 1.4 h | iterative / symmetry / cluster (24 GB with METIS) |
| L4-8 | ω=2.74, gap 5/κ₁, half, nz=16 | 0.76 M | 82 | 28 | 108 min / 13 min | 40 | 72 h / 8.7 h | iterative / symmetry / cluster |
| L4-8 | ω=4.74, gap λ0, half (analysed) | 0.25 M | 13.4 | 7.0 | 3.5 min / 51 s | 40 | 2.4 h / 0.7 h | needs the 24 GB machine |
| L4-8 | ω=4.74, gap λ0, half, nz=16 (analysed) | 0.48 M | 38 | 15.2 | 24 min / 4.3 min | 40 | 16 h / 3.0 h | iterative / symmetry / cluster (24 GB with METIS) |
| L4-8 | ω=4.74, gap 5/κ₁, half | 1.65 M | 179 | 54 | 3.9 h / 22 min | 40 | 155 h / 16 h | iterative / symmetry / cluster |
| L4-8 | ω=4.74, gap 5/κ₁, half, nz=16 | 3.12 M | 672 | 166 | 44 h / 2.9 h | 40 | 1754 h / 118 h | iterative / symmetry / cluster |

Verdict thresholds: 4 GB (the `run_limited` default) and 20 GB (usable on
the 24 GB machine). Above that, the test needs an iterative solver, a further
symmetry or domain reduction, or a cluster. The uncertainty of the projections
is about ±25 % where the mesh was analysed or the law is interpolated. It
grows for extrapolations far beyond 0.5 M DOFs. The square law applied to the
analysed L4-8 rectangles is off by 3–33 %; it overestimates elongated domains.

**Checks.** 118 of 128 pass (`checks.csv`):

- DOF counts match the tensor-product formula in every case.
- Every full run stays under the 4 GB cap.
- LU residuals are ≤ 1e-10.
- The FE pattern is structurally symmetric.
- On the channel, |k_h/k₀ − 1| = 9.3e-3 (tolerance 1e-2) and the field is
  uniform in y to 3e-12.
- The calibration spreads of κ_L, β and the rate are below 30 %.

The 10 failures are reported as findings, not hidden:

1. Three analysis cases (80×80×8, 91×38×16 and the L3-1 mesh 333×16×8) reach
   a `Sys.maxrss` of 4.12–4.27 GiB. That is a transient overshoot of at most
   0.27 GiB during the CHOLMOD/METIS analysis, which the 1 s RSS watchdog
   did not sample (its sampled peaks were ≤ 4.0 GiB). The cap is the
   watchdog's. The data are kept, but **future analysis runs should stay at
   N ≤ 400 k**.
2. In six full cases, the model's peak RSS is 0.76–0.87× the measured
   value, so the model is low. These are the small p=3 cases and the four
   METIS cases, where Julia's GC slack dominates. At target sizes this term
   is small, but treat projections near a threshold (L3-2 L2) as marginal.
3. The square fill law misses the analysed 108×51×8 rectangle by 33 %. It
   overestimates it, so the law is conservative for the elongated L4-4
   domains.

Other notes:

- The channel case 128×16×8 was re-run with the analysis step. It was
  killed above 4 GiB during the solve/post-processing. Its 2026-10-26
  record (peak 3.56 GB, without the fill fields) is used instead.
- The UMFPACK symbolic peak estimate overshoots the actual peak by 7–23×,
  so it is useless for sizing. This is why the CHOLMOD analysis is used.
- The library does accept a custom solver: `config.solver` is read in
  `HydroElasticFEM.Simulation.simulate`.

### Recommendation: how to run each 3D test

1. **Change the solver before any large 3D run.** A Gridap `LinearSolver`
   that calls UMFPACK with `UMFPACK_ORDERING_METIS` takes about 30 lines and
   goes in through `config.solver`, so no library change is needed. It cuts
   the projected peak memory by 1.5–4× and the time by 2–15× on the targets,
   and the gain grows with N. MUMPS or Pardiso with nested dissection would do better still,
   but would need a new dependency, which is a PI decision. For the
   multi-RHS tests, factorise once per frequency and reuse the factors: the
   ζ sweeps in L3-3 (G_h(0) from one bare-plate solve) and the two
   incidences in L4-8.
2. **L3-2 (VT3D-B port): run now, under the 4 GB cap.** Levels L1 and L2 fit
   under the cap (2.8 and 3.2 GB), each in about 1 min for 6 solves. Level L3
   (4.5 GB) needs either the METIS solver (3.4 GB, which fits) or a cap of
   8 GB.
3. **L3-1 (VT3D-A port): needs the 24 GB machine at p=2, h=λp/8, nz=8.**
   This is 13 GB, or 8.5 GB with METIS, so the PI would raise `run_limited`
   to 16 GB. It needs 3 solves of 1.5 min each. θ = 0 is one cell wide and
   trivial. Do not attempt refinement in 3D (h/16, nz = 16 or p = 3 all need
   more than 24 GB). Show discretisation convergence on a shorter channel
   (smaller gaps), or rely on the y-uniform 2D equivalence that L3-0 already
   checks.
4. **L3-3 (single resonator): does not fit the 24 GB machine with a direct
   solver**, even in a quarter domain. The matrix domain needs 58 GB, or
   21 GB with METIS, which is just over 20 GB. The §0.3-compliant gap
   5/κ₁ needs 51 GB even with METIS. Options, in order:
   1. METIS solver plus a slightly smaller frame (3λp) or a coarser far
      field (graded h). This might bring the matrix domain under 20 GB (not yet
      sized: run an analysis case of the reduced mesh first). 41 ω would
      then take about 4 h on the 24 GB machine. Justify the reduced buffer
      by a buffer-sensitivity run at 2–3 frequencies.
   2. Fewer frequencies (about 10 instead of 41) near the resonance.
   3. A cluster account for the 5/κ₁ gap. Ask the PI, as the hand-off
      advises.
5. **L4-4 (anisotropic island): cluster or iterative solver only.** The
   smallest variant (gap 2λp, half domain) needs 30 GB with METIS (91 GB by
   default), for 24 solves of 10 min each. With the gap 5/κ₁ it needs
   185 GB. Recommend a cluster account (MUMPS, distributed memory), or a
   smaller island (3λp, 6λp × 3λp) with gap 2λp and METIS. That could come
   under 20 GB, but it would need a new analysis run to confirm. An iterative
   solver for this indefinite, damped Helmholtz-type coupled system is a
   research task in itself; do not count on it for this paper.
6. **L4-8 (forced response):**
   - Near ω₁ = 2.74, use nz = 8 with the gap λ0 on the 24 GB machine: 9 GB,
     1.2 h for 40 solves.
   - With METIS, the §0.3 gap 5/κ₁ (10.8 GB, 1.4 h) and nz = 16 with the
     gap λ0 (12 GB) also fit on the 24 GB machine.
   - Near ω₂ = 4.74, the gap λ0 at nz = 8 needs 13 GB (7 GB with METIS). The
     gap 5/κ₁ (1.65 M DOFs, 54 GB with METIS) needs a cluster or a reduced
     buffer.
7. **CPU budget.** Every 24 GB-machine variant above takes ≤ 7 min per
   factorisation with METIS, so one frequency per `run_limited` call fits
   the 60 CPU-min budget. With the default solver, L3-3 at the matrix
   domain takes 45 min per frequency, which is close to that budget.

### Reproduce

    julia --project -O1 scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl --list      # 36 cases (18 full, 18 analysis)
    bash Loop_Framework/tools/run_limited.sh -- julia --project=<repo> -O1 scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl --case <i>
    julia --project -O1 scripts/L3-3D-flat-bottom/L3-0_memory_sizing.jl             # tables, fits, targets, checks, figure

Cases 26, 29 and 22 (analysis, N = 547 k, 566 k and 670 k) exceed the cap
during the assembly or analysis, and have no record.

---

## L3-0b: METIS-ordered direct solver

### Implementation

`src/solvers.jl` (module `Solvers`) defines `MetisLUSolver`, a Gridap
`LinearSolver`. It is UMFPACK sparse LU with `UMFPACK_ORDERING_METIS`. METIS
ships with Julia's SuiteSparse build (`libsuitesparse_jll`), so no package is
added. CHOLMOD is not an option: the system is complex symmetric, not
Hermitian. The solver goes in through the existing hook, with no change to
HydroElasticFEM.jl:

    S.build_problem(domain, entities, PH.FreqDomainConfig(ω=ω, solver=MetisLUSolver()); rhs_fn)

Factorise once, solve many: `ns = factorize(MetisLUSolver(), A)`, then
`solve_rhs(ns, b)` or `solve_rhs(ns, B)` (one column per right-hand side).
`numerical_setup!(ns, A2)` refactorises a matrix with the same pattern and
reuses the symbolic analysis, and thus the METIS ordering. `factor_stats(ns)`
returns lunz, the UMFPACK peak and numeric memory, the flops and the ordering
used. `ordering=:amd`, `:best` or `:default` select other orderings.
`L1_1.solve` and `L3_0.build` gained a `solver` keyword (default `nothing`,
the library `LUSolver`).

### Validation (`test/test_solvers.jl`, 13/13 pass)

- 2D: the L1-1 channel with one line resonator (p = 2, 10 el./λ_p), through
  `simulate`. η and q equal the `LUSolver` values to ≤ 1e-10 relative.
- 3D: the L3-0 box, p = 2, 8×3×3 cells. The full solution equals the
  `LUSolver` solution to ≤ 1e-10 relative. UMFPACK reports the METIS
  ordering (and AMD when asked).
- Multi-RHS: three right-hand sides from one factorisation equal `lu(A) \ B`
  to ≤ 1e-10. A refactorisation with the same pattern and new values (the
  next ω) matches `lu(A2) \ b` to ≤ 1e-10.

### Cost on the L3-0 ladder (`scripts/L3-3D-flat-bottom/L3-0b_metis_solver.jl`)

One case per process, under `run_limited` (4 GB, 1 thread), after a warm-up.
The peak RSS is `Sys.maxrss` right after `simulate`. "simulate" is the
factorisation plus one solve through the hook. The projection is the L3-0
METIS model (κ_L, β = 18.2 B/entry, rate = 3.56 GFlop/s, R₀, r) applied to the
measured CHOLMOD fill (`fill.csv`), or to the METIS fill law for 24×24×16,
which has no fill record. Output: `data/exp_pro/L3-0b/metis_solver.csv`.

| p | nx×ny×nz | N | lunz meas. / proj. | UMFPACK peak GB meas. / proj. | peak RSS GB meas. / proj. | fact.+solve s meas. / proj. | re-solve, 1 / 4 RHS (s) | L3-0 default: peak GB, fact.+solve s |
|---|---|---|---|---|---|---|---|---|
| 2 | 24×24×8 | 43218 | 2.87e7 / 2.87e7 | 0.48 / 0.49 | 2.46 / 2.47 | 3.7 / 2.9 | 0.03 / 0.08 | 2.96, 10.1 |
| 2 | 32×32×8 | 76050 | 5.90e7 / 5.90e7 | 0.96 / 1.00 | 2.93 / 3.10 | 8.2 / 7.8 | 0.05 / 0.15 | 3.47, 26.5 |
| 3 | 16×16×8 | 62426 | 5.24e7 / 5.24e7 | 0.91 / 0.89 | 3.27 / 3.00 | 7.3 / 7.6 | 0.04 / 0.13 | 3.89, 17.9 |
| 2 | 24×24×16 | 81634 | 7.24e7 / 7.11e7 | 1.21 / 1.21 | **4.09** / 3.32 | 11.5 / 11.9 | 0.09 / 0.18 | killed > 4 GiB |

- The fill and the UMFPACK peak match the projection to 2 % and 4 %. The
  factorisation time (`factorize`, without the solve) is 0.94–1.04× the
  projection.
- The peak RSS is 0.95–1.09× the projection on the three `nz = 8` cases. On
  24×24×16 the model is 19 % low, so the case does not fit: 4.09 GiB against
  the 4 GiB cap. Later in that process (a second factorisation for the
  statistics, with the first one not yet collected), the peak reached
  5.1 GiB. `run_limited` caps Julia only by a GC hint, so the run was not
  killed. Treat projections above about 3.3 GB as not fitting.
- Against the default ordering, the time drops 2.4–3.2× and the peak 0.5–0.6 GB
  on these meshes. A re-solve with the stored factors costs 1 % of the
  factorisation.

### Reproduce

    julia --project -O1 -e 'using Test, HydroElasticFEM_xxx2026; include("test/test_solvers.jl")'
    bash Loop_Framework/tools/run_limited.sh -- julia --project=<repo> -O1 scripts/L3-3D-flat-bottom/L3-0b_metis_solver.jl --case <1..4>
    julia --project scripts/L3-3D-flat-bottom/L3-0b_metis_solver.jl      # table

---

## L3-2: point resonator on the 3D plate (port of VT3D-B, set P3)

**Sources.** Code: `src/L3_2_point_resonator.jl` (module `L3_2`),
[scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl](../scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl).
Legacy: `code/scripts/VT3D_B_{point,winkler,nearfield}.jl` and prereg VT-3D-CONV-B v2 (+ A1 draft)
in the LRH folder. Matrix row L3.2 of `verification_test_matrix_2026-10-22.md`, frozen
before the binding run. Outputs: `data/exp_pro/L3-2/` (`points.csv`, `convergence.csv`,
`fits.csv`, `winkler.csv`, `checks.csv`), `plots/L3-2/point_resonator.{pdf,png}`.

### Test description

Set P3 unchanged. A channel 2 m wide and 10 m deep holds a damping zone (4λ₀), a gap
(λ₀/2), a Kirchhoff–Love plate (4λ_p long, the full width, free lateral edges, ν = 0,
C_ρ = 0.05g, m_ρ = 0.01), a gap and a zone. The incident wave comes from a velocity inlet and
the zone forcing. One resonator sits at the plate centre x_r (M_ρ = 0.01 m³, ω_r = 10). φ, κ
and η are Q2, with plate penalty γ = 6. The far field has Δx ≤ λ₀/6.4 in open water, λ_p/8 on
the plate, Δy ≤ W/7 and nz = 8. Towards x_r the cells grow from h_min by a ratio of 1.5, with
h_min = λ_p/12, λ_p/24 and λ_p/48 (L1, L2, L3; N = 57.9 k, 73.0 k and 102.9 k).

Solves per level. The bare plate at ω_r is factorised once and gives S1 (incident wave) and S2
(unit point load) at the node x_r, at the edge point x_r + (h_min/2, 0) and at the interior
point x_r + (h_min/3, h_min/3). S3 (resonator, ζ = 0, incident wave) is solved at the same three
points. S1_09 and S2_09 are the bare plate at 0.9ω_r, and S4 the resonator with ζ = 0.02 there.
PG (at L2) is S2 with the penalty γh_min/h_far. The C_ρ variant (0.005g, 5e-4g) runs S1, S2 and
S3 at x_r. All plate lengths and the channel width scale with λ_p there (W = 1.31 m and 0.88 m),
so the meshes keep the P3 DOF counts. D-W is the static Winkler limit: the plate alone,
[−10l_c, 10l_c]², h_min = l_c/{8, 16, 32}. The point load uses the package `DiracDelta`. The
solver is `MetisLUSolver`.

Compute: one process per (C_ρ, level, solve group), each with one assembly and one
factorisation, inside `run_limited` (4 GB, 1 thread), `-O0`, `--heap-size-hint=2500M`, 32
processes. The first attempt ran several solves per process and loaded CairoMakie (about
0.7 GB) at `-O1`. Its processes reached 3.8–4.8 GiB, so they were stopped and redone (same
values). The table and checks come from the lean runs only.

### Results (113/136 checks; 23 failures, all convergence rates)

| C_ρ | level | N | \|G_h(x_r,x_r)\| | \|q_h\| | peak RSS GiB (max over processes) |
|---|---|---|---|---|---|
| 0.05g | L1 / L2 / L3 | 57.9 k / 73.0 k / 102.9 k | 0.0805 / 0.0834 / 0.0880 | 0.0269 / 0.0261 / 0.0249 | 3.70 / 3.96 / 3.91 |
| 0.005g | L1 / L2 / L3 | same | 0.2213 / 0.2270 / 0.2349 | 0.01449 / 0.01419 / 0.01379 | 3.63 / 3.81 / 3.79 |
| 5e-4g | L1 / L2 / L3 | same | 0.9250 / 0.9187 / 0.8977 | 0.00504 / 0.00511 / 0.00525 | 3.62 / 3.87 / 3.56 |

Infinite plate on the fluid (`green_2d_integral`): \|G\| = 0.0274, 0.1135 and 0.4817.

- **Identities (72 rows, all pass).** The residual is ≤ 1.3e-9. The incident wave enters the
  right-hand side only (matrix difference 0, incidence-free RHS 0). The load sign holds
  (Σb_η = f to 9e-16; A_ηη·1 matches (g − ω²m_ρ)|Γη| to 3e-11). UMFPACK uses METIS. B-P1
  holds: η_h vanishes at the attachment point of a resonator at resonance, to ≤ 2.5e-17 of
  η_bare, at the node, edge and interior points. B-P2 (Sherman–Morrison) holds to ≤ 2.4e-10 at
  ω_r and at 0.9ω_r. The point coupling is therefore exact at any location in a cell.
- **Memory.** Every process stays below 4 GiB (3.36–3.96 GiB), L3 included, with METIS.
- **Local singularity (B-P5, pass).** The r² ln r coefficient of the azimuthal mean at L3 gives
  c_h·8πC_ρ/f_h = 1 within 1.3 % (S3) and 2.1 % (S2) for P3, 1.9 % for 0.005g and 3.1–3.5 % for
  5e-4g (tolerance 10 %). The plate field near a point resonator is the biharmonic Green function.
- **Static limit (pass).** w_h(0)/w(0) − 1 = −1.15 %, −0.83 % and −0.67 % (tolerance 1 % at
  l_c/32), with p_obs = 1.03. The r² ln r coefficient is within 2.7 % of f/(8πC_ρ), with the
  same sign.
- **Attachment location (pass).** G_h at the edge and interior points differs from G_h at the
  node by 2.1 % and 3.1 % (L1), 0.5 % and 0.8 % (L2), and 0.13 % and 0.20 % (L3; tolerance 2 %).
  The difference falls 4× per halving of h_min.
- **Penalty (P-γ, pass).** γ → γh_min/h_far changes G_h by 0.75 % at L2 (tolerance 2 %).
- **C_ρ scaling (pass).** d ln|G_h|/d ln C_ρ = −0.504 at L3. The infinite plate gives −0.623
  (tolerance 0.15) and Hertz −1/2. The driving-point response stays bounded and grows as
  C_ρ^{-1/2}.
- **Convergence of the point values (B-P3 and the G_h ladders at the edge and interior
  points, 16 failures) and of the near field (B-P4, 7 failures).** The L2–L3 differences of G_h are 5.2 %, 3.6 % and 4.8 % of the L3 value (tolerance
  2 %). They are larger than the L1–L2 differences, so p_obs < 0 (−0.4 to −0.8). |q_h| behaves the
  same way (2.7–4.8 %). The near field passes its d₂₃ tolerances (L² 1.3–2.4 %, H¹ 2.1–2.5 %),
  except L² at 5e-4g (2.4 % > 2 %), but its rates are below 1 (and 0.5). These are findings:
  1. They are not the ln(1/h) growth of a membrane. |G_h| increases with refinement for 0.05g
     and 0.005g but decreases for 5e-4g. The local structure (B-P5) and the static limit
     converge.
  2. The "fixed far field" premise of the legacy prereg does not hold. The grading from h_min
     with ratio 1.5 reaches the far-field size further out at each level, so ny = 8, 10 and 14
     and the cells next to the graded zone change with the level. The plate strip is narrower
     than λ_p (W/λ_p = 0.87), so G_h is a global quantity of the strip (3.2× the infinite-plate
     value). Its far-field discretisation error (Q2 at λ_p/8 across the width) changes by a few
     per cent between levels.
  3. The legacy stop rule (a finite footprint) therefore does not apply. A conclusive ladder
     needs the far field refined with h_min (e.g. h_far = λ_p/16, ny_base = 14) or a nested
     grading that keeps every far cell. That is N ≈ 2× P3 L3, beyond the 4 GB cap: it needs the
     24 GB machine.

### Deviations from the frozen row (post-run, compute only)

- One process per solve group, `-O0`, heap hint 2.5 GB (lower than the default) and CairoMakie
  loaded only for post-processing. The first attempt exceeded 4 GiB at L3 and was redone.
- The incidence check (matrix independent of η₀) is run at L1 for each C_ρ, as frozen.

### Reproduce

    julia --project -O0 scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl --list
    bash Loop_Framework/tools/run_limited.sh -- julia --project=<repo> -O0 --heap-size-hint=2500M scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl --case <1..32>
    julia --project scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl     # tables, checks, figure

---

## L3-1: bare plate under oblique incidence (port of VT3D-A, matrix row L3.1(a))

**Sources.** Code: `src/L3_1_oblique_plate.jl` (module `L3_1`),
[scripts/L3-3D-flat-bottom/L3-1_oblique_plate.jl](../scripts/L3-3D-flat-bottom/L3-1_oblique_plate.jl).
Legacy: `code/scripts/VT3D_A_{dispersion,analyse,yago}.jl` (LRH folder, prereg VT-3D-CONV-A v2 + A1).
The legacy A-D driver covers normal incidence only (an m = 0 wave in a walled channel), and A-Y is the
Yago VLFS benchmark (set P4, outside the paper scope, not ported). The oblique case of row L3.1(a) is new.
Matrix row L3.1 of `verification_test_matrix_2026-10-22.md`, frozen before the binding run. Outputs:
`data/exp_pro/L3-1/` (`results.csv`, `ladder.csv`, `checks.csv`), `plots/L3-1/oblique_plate.{pdf,png}`,
paper text `paper/snippets/L3-1.tex`.

### Test description

Set P1 (C_ρ = 0.49, m_ρ = 0.01, T_ρ = 0, H = 10, ν = 0). The plate covers the whole box, which is
**periodic in y** (`TankDomain(…; is_periodic=(false, true, false))`) with W = λ_p/sin θ. A line load
F e^{ik_y y}/b acts on two cells at x_s, with k_y = k₀ sin θ. At both ends there are `StructuralDampingZone`
sponges (Kelvin–Voigt ωτ = 3 plus a viscous dashpot, ramp ξ²). φ and η are Q2, γ = 6. The mesh has
h = λ_p/8 and nz = 8, graded with a top cell of 0.2/k₀. The full layout is the L3-0 target: sponges 4λ_p,
gaps 5/κ₁ and 3λ_p on each side of the source (333×16×8 cells at ω = 10, θ = 30°). The reference is the
oblique Green function, from the residues in k_x:

    η = F e^{ik_y y} i Σ (k_m/k_{x,m}) e^{ik_{x,m}|x|}/𝒟₀′(k_m),   k_{x,m} = √(k_m² − k_y²).

It gives k_h = √(k_x,h² + k_y²) → k₀, γ_θ = γ_r/cos θ, and an x-flux of ¼ω𝒟₀′ cos θ |A|² per wave.
Each process does one METIS factorisation with 3 RHS: the oblique load and two point loads (`DiracDelta`)
for the reciprocity check. The observables come from the y-projection onto e^{ik_y y} (Gauss-5 per
y cell) and a wave fit at the **cell vertices**, where a discrete Bloch wave is exactly exponential. A
uniform sampling gave a 2 % left/right bias in a smoke run. P_in = ½ω Im(b̄·x)/W is the exact discrete
input power.

Mesh study (deviation from the §0.3 ladder, as L3-0 recommends): the 333×16×8 channel cannot be refined
on this machine. A short channel (sponges 2λ_p, gaps 5/√(κ₁² + k_y²), half-window 2λ_p) is used instead at
ω = 10, θ ∈ {0°, 30°}, h = λ_p/{6, 8, 12}, plus θ = 0 at λ_p/16. With k_y ≠ 0 the fluid evanescent modes
decay at ≥ k_y, so the shorter gaps are §0.3-equivalent. The vertical convergence is that of the vertical
section (L1-1).

### Results (144/144 checks)

| layout | θ | ω | h | N | \|k_h/k₀ − 1\| | γ_h cos θ/γ_r − 1 | flux/P_in − 1 | reciprocity | wall s |
|---|---|---|---|---|---|---|---|---|---|
| full | 30° | 10 | λp/8 | 384 192 | 4.19e-3 | −2.2 % | −2.3 % | 1.3e-12 | 96 |
| full | 0° | 10 | λp/8 | 72 036 | 9.25e-3 | −3.5 % | −3.8 % | 2.8e-14 | 6 |
| full | 60° | 10 | λp/8 | 216 108 | 4.54e-3 | +2.0 % | +1.9 % | 1.6e-11 | 23 |
| full | 30° / 0° | 5 | λp/8 | 288 576 / 54 108 | 3.24e-3 / 7.13e-3 | −1.9 / −3.3 % | −2.1 / −3.6 % | ≤ 5.3e-10 | 81 / 4 |
| full | 30° / 0° | 2 | λp/8 | 159 552 / 29 916 | 1.7e-4 / 3.5e-4 | −0.1 / −0.2 % | −0.2 / −0.3 % | ≤ 5e-14 | 25 / 2 |

- Identities: residual ≤ 4.7e-9, reciprocity ≤ 5.3e-10, METIS ordering in every case, left/right
  symmetry of k_x,h ≤ 1e-6. The vertex fit residual is ≤ 5.7e-9 for oblique waves and ≤ 1.9e-6 at θ = 0
  (model mismatch of the slow fluid evanescent modes). The transverse residual (field outside the
  e^{ik_y y} mode) is ≤ 2e-3, and the sponge reflection is ≤ 1.5e-3 in the full layout.
- Isotropy: \|k_h(θ)/k_h(0°) − 1\| = 5.1e-3 (30°), 4.8e-3 (60°) at ω = 10; 3.9e-3 and 1.8e-4 at ω = 5 and 2.
  The oblique error is *smaller* than at normal incidence on the same mesh. This is the anisotropy of the
  Q2 dispersion.
- Mesh study: the anisotropy is 8.7e-3, 5.1e-3 and 2.3e-3 at λ_p/{6, 8, 12}, a rate of 1.84 and then 1.92.
  At θ = 0, \|k_h/k₀ − 1\| converges at rate 1.92 (λ_p/8 → 12) and 1.93 (12 → 16). With nz = 8 the
  vertical floor is not yet reached. γ and the energy balance converge at the same rate. The short and
  full channels give the same k_h to 1e-8.
- Compute (rss_guard, physical footprint): 11.8 GiB for cases 1–3 in one process (process maxrss 7.8 GB;
  target factorisation 46 s), 9.3 GiB for cases 4–14. The L3-0 METIS projection for the target was 8.5 GB.
- Exploratory runs before the freeze (disclosed in the row): smoke runs at λ_p/{4, 6, 8} with
  nz ∈ {2, 4}. They checked the periodic seam (the plate skeleton has the periodic edge count, and 1 and 2
  periods give identical k) and the sampling choice.

### Reproduce

    julia --project -O1 scripts/L3-3D-flat-bottom/L3-1_oblique_plate.jl --list
    python3 -I Loop_Framework/tools/rss_guard.py --mem-gb 22 -- bash Loop_Framework/tools/run_limited.sh --mem-gb 20 --cpu-min 120 --wall-min 120 -- julia --project=<repo> -O1 scripts/L3-3D-flat-bottom/L3-1_oblique_plate.jl --case 1 --case 2 --case 3
    julia --project scripts/L3-3D-flat-bottom/L3-1_oblique_plate.jl          # tables, checks, figure

---

## L3-2r: far-field refinement of the point-resonator ladder (2026-10-28)

**Why.** The first L3-2 run (commit 89ab3ca, 113/136) failed 23 convergence-rate checks (16 B-P3/attachment
rows, 7 B-P4 rows). Its graded far field changed with the level (ny = 8, 10, 14), so the "fixed far field"
assumption of the ladder did not hold. Those results are kept in `data/exp_pro/L3-2/graded_far_field/` and
`plots/L3-2/graded_far_field/` (regenerate them with `--graded`; the records are unchanged, and the legacy
hash 1bc7c3a0 is preserved through `LATE_FIELDS`).

**What changed (frozen in the L3.2 row before the run).** `L3_2_Params.far_mesh = :nested`:
- the plate far field is uniform and identical at every level, Δx = λ_p/16 and Δy = W/14;
- only the 2 far cells next to x_r (and next to y = 0) are replaced by a geometric grading from
  h_min = h_far/2^m (`zone_sizes`, `nested_sizes`, ratio ≤ 1.5): h_min = λ_p/{16, 32, 64, 128} for L1–L4;
- N = 130 k, 150 k, 171 k, 215 k;
- open water, zones, nz = 8, the top cell, all solves and all C_ρ variants are unchanged;
- the script default is the nested ladder, and `--graded` gives the first run.

Every check is re-evaluated on the frozen ladder L1–L2–L3. The same rate and d₂₃ checks on L2–L3–L4 are
added. The memory row is < 20 GiB. D-W reuses its record.

**Compute** (3D batch rules: rss_guard 22 GiB, run_limited 20 GB, lock, `-O1`, several parts per process):

| batch | cases | guard peak footprint | wall |
|---|---|---|---|
| P3 L1+L2 | 1–13 | 6.30 GiB | 5 min |
| P3 L3+L4 | 14–25 | 9.31 GiB | 7 min |
| C_ρ variants L1–L4 | 26–41 | 8.85 GiB | 7 min |
| diagnostic (bare plate, L2–L4, top_k = 0.1 and nz = 12) | — | 12.92 GiB | 5 min |

Process maxrss ≤ 9.13 GB. The largest factorisation is 25 s (L4, N = 215 k, METIS).

### Results: frozen ladder 136/136 (first run 113/136); with the added L2–L4 rows 178/192

| C_ρ | \|G_h\| L1 / L2 / L3 / L4 | p_obs G (L1–L3) | d₂₃/\|L3\| G | p_obs \|q\| | d₂₃ \|q\| | B-P4 p_obs L² / H¹ | L3→L4 change of G |
|---|---|---|---|---|---|---|---|
| 0.05g | 0.08858 / 0.08905 / 0.08917 / 0.08944 | 1.90 | 0.15 % | 2.09 | 0.10 % | 1.50 / 1.74 | 0.31 % |
| 0.005g | 0.2305 / 0.2317 / 0.2321 / 0.2327 | 1.69 | 0.21 % | 1.83 | 0.12 % | 1.48 / 1.66 | 0.30 % |
| 5e-4g | 0.8613 / 0.8612 / 0.8612 / 0.8611 | 1.62 | 0.24 % | 1.83 | 0.03 % | 1.53 / 1.66 | 0.31 % |

- **All 23 former failures pass.** G_h at the edge and interior points converges at rates 2.09 and 1.97
  (d₂₃ 0.21 % and 0.47 %). B-P4 d₂₃ is 0.15–0.35 % (L²) and 0.18–0.35 % (H¹).
- Identities: residual ≤ 7.7e-9, B-P1 ≤ 1.4e-15, B-P2 ≤ 1.2e-9 (ω_r) and 3.6e-9 (0.9ω_r), incidence in the
  RHS only (0), load sign ≤ 1.8e-15, METIS everywhere.
- B-P5 (L3): 2.3 % (S3) and 3.7 % (S2) for P3, ≤ 3.9 % for the softer plates. Attachment at L3: 0.06 % (edge)
  and 0.11 % (interior), falling 4× per level. P-γ: 0.15 %. C_ρ slope: −0.492 (infinite plate −0.623).
  B-W1/B-W2 are unchanged (0.67 %, p_obs 1.03; 2.7 %).
- The first run's finest |G_h| (0.0880 for P3) lies 1.3 % below the new L3 value. Its 5 % uncertainty came
  from the far field.

### The 14 failing checks (all added L2–L4 rate rows): finding, not a bug

The 14 failures are the p_obs rows on L2–L3–L4:
- P3: G_h, |q_h|, G_h at the edge and interior points, and B-P4 L² and H¹;
- each softer plate: G_h, |q_h|, and B-P4 L² and H¹.

All their d₂₃ rows pass (≤ 0.39 %). The L3→L4 change (≈ 0.3 % of G_h for every C_ρ, 0.15–0.35 % of the
near field) is larger than the L2→L3 change (0.12–0.24 %), so p_obs < 1. The ladder reaches a plateau at
about 3e-3:

1. **Not the vertical mesh.** The diagnostic re-ran the P3 bare plate at L2–L4 with a top cell half as
   thick (top_k = 0.1) and with nz = 12 (`data/exp_pro/L3-2/diag_vertical.csv`). The level increments of
   |G_h| are identical to 1 % in the three vertical meshes: 1.15e-4 (L2→L3) and 2.73e-4 (L3→L4). The
   vertical mesh shifts G_h only by a constant (−0.3 % and +0.3 %).
2. **Not the far field.** The far-field nodes are identical at every level by construction.
3. **Not the post-processing.** The Cauchy formula and the reference set are those of the frozen ladder,
   and the identities hold to round-off at L4.
4. **Most likely cause:** the tensor-product grading creates elongated cells along the lines y = 0 and
   x = x_r. Their aspect ratio h_far/h_min is 1, 2, 4 and 8 at L1–L4, and the C/DG plate penalty and the Q2
   fluid cells under them lose accuracy as the aspect ratio grows. This is not confirmed: the confirming
   run (a refinement local in both directions, or a γ scaled with the cell aspect) was not done, because
   the compute window had closed.

So the driving-point compliance is resolved to ≈ 0.3 %. The frozen L1–L3 ladder converges at rates 1.6–2.1.

### Reproduce

    julia --project -O1 scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl --list            # nested ladder, 42 cases
    python3 -I Loop_Framework/tools/rss_guard.py --mem-gb 22 -- bash Loop_Framework/tools/run_limited.sh --mem-gb 20 --cpu-min 120 --wall-min 120 -- julia --project=<repo> -O1 scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl --case 1 … --case 13
    julia --project scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl                        # tables, checks, figure
    julia --project scripts/L3-3D-flat-bottom/L3-2_point_resonator.jl --graded               # first run (graded far field)

---

## L4-4: anisotropic resonator island (incomplete, usage stop 2026-10-29)

No binding run was made and the matrix row is not frozen (draft in `verification_test_matrix_2026-10-22.md`).
Code: `src/L4_4_aniso_island.jl` (module `L4_4`, run in `envs/resonator_field`). It contains the half-domain 3D
model with the upstream `ResonatorField` (n(x) Gaussian with a taper, floor 1e-6), a Nitsche symmetry on y = 0,
the scattered-field load, line, axis and map outputs, and the references: exact local complex root, eikonal
phase, thin-lens J, the port of `aniso_island_rays.py`, a split-step PE, back-propagation and curvature fit.

- Upstream: `ResonatorField` assembles and solves in 3D (KirchhoffLovePlate host). No blocker.
- Smoke (N = 12 k): residual 2.7e-13, METIS used, half vs full domain 9.1e-13. The load identity gives 8.6e-3
  because the reference used the interpolated η_inc and the load used the analytic e^{ik₀x}. Use the interpolant
  in the load and re-check before the binding run.
- References (`scripts/L4-beyond-analytical/L4-4_references_check.jl`): the ray port reproduces the Python
  values to 0.1 m. With untruncated data, the PE back-propagated focal length is within 2.4 % / 0.2 % / 7.9 %
  of the ray focus (A at ω = 8, B at ω = 8, A at ω = 12).
- Open issue before freezing: truncating the line data at y = 2.5L_y + 2 m shifts f by 10–60 % and makes it
  depend on the line. The y-width of the FE domain must grow, or the FE must be compared with PE data truncated
  the same way. Sizing (CHOLMOD analysis of the chosen layout) has not been run yet.

---

## L3-3: single resonator on a large 3D plate (quarter symmetry), 2026-10-29/30

**Sources.** Code: `src/L3_3_single_resonator.jl` (module `L3_3`, included by the script),
[scripts/L3-3D-flat-bottom/L3-3_single_resonator.jl](../scripts/L3-3D-flat-bottom/L3-3_single_resonator.jl).
Matrix row L3.3 of `verification_test_matrix_2026-10-22.md` (frozen 2026-10-29 before the binding run).
Outputs: `data/exp_pro/L3-3/` (`frequencies.csv`, `zeta_sweep.csv`, `sensitivity.csv`, `renormalised.csv`,
`checks.csv`, `fine_mesh_reference.csv`), `plots/L3-3/single_resonator.{pdf,png}`, paper snippet
`paper/snippets/L3-3.tex` (figure staged as `paper/figures/L3_3_single_resonator.png`).

### Test description

Set P1 (C_ρ = 0.49, m_ρ = 0.01, ν = 0, H = 10), plate over the whole free surface, one resonator
(m_r = 0.01 m³, ω_r = 10) at the centre. Quarter domain [0, L_q]² × [0, H], L_q = 8λ_p, resonator at the
corner. Symmetry planes: fluid natural, plate ∂η/∂n = 0 by Nitsche (`symmetry_matrix`, the restriction of the
C/DG facet terms of the mirrored plate, incl. the frame weight). Kelvin–Voigt frame (ωτ = 10, ramp ξ²).
φ, η Q3, γ = 12, nz = 8 graded. One METIS factorisation per ω of the bare plate under a unit point load gives
G_h(0) and G_h(x); every ζ follows by scalar Sherman–Morrison, F₁/A = 1/(1/α − G_h(0)). Far field: radial fit
of η on r ∈ [1.5λ_p, R_s − 0.5λ_p] × 19 angles (H₀⁽¹⁾, H₀⁽²⁾, 6 evanescent terms, k fitted).

**Run mesh (post-freeze deviation): h = λ_p/4.5** (frozen λ_p/5). At λ_p/5 (N = 380 666) the guard peak
footprint was 17.4 / 18.39 / 18.63 / 18.20 GiB (cases 1, 3, 2, 4) against the 20 GiB guard; case 2 exceeded
the ~18.5 GiB threshold, so all cases were re-run at λ_p/4.5 as authorised (PI 2026-10-29: "20GB max, if not
possible, reduce size"). Layout at λ_p/4.5: 36 × 36 × 8 cells, N = 308 906; R_s = 22 cells = 4.89λ_p, frame
14 cells = 3.11λ_p, L_q = 8λ_p; B: R_s = 4λ_p, L_q = 7.11λ_p; M: h = λ_p/4, R_s = 5λ_p. Memory mitigations
tried first (no change to the factors): freeing the operator + GC + `malloc_zone_pressure_relief` before the
factorisation, and UMFPACK Control[ALLOC_INIT] = 1 (`factorize_prealloc`, bitwise-identical solutions). Neither
changed the peak materially (18.2 GiB vs 18.4–18.6); the footprint grows steadily during the numeric
factorisation (≈ 3 GiB before it, ≈ 15 GiB during), roughly twice UMFPACK's own peak estimate (8.6 GB).
The four λ_p/5 records are kept and compared in `fine_mesh_reference.csv` (informational).

### Results (203/203 checks)

| Group | n | worst value | tolerance |
|---|---|---|---|
| residual (every solve) | 15 | 4.3e-10 | 1e-8 |
| quarter vs full G_h(0) / η at 3 points / mirror symmetry | 3 | 4.5e-13 / 8.4e-13 / 6.7e-13 | 1e-8 |
| Sherman–Morrison vs monolithic (quarter, full); monolithic quarter vs full | 3 | 7.0e-14; 2.1e-13 | 1e-8 |
| Im G_h(0)/γ_r − 1 | 11 | 3.8e-3 | 2 % |
| Re G_h(0)/Re G(0) − 1 | 11 | −3.9 % (ω = 10.51; −2.8 to −3.9 %, growing with ω) | 5 % |
| \|C⁺/(iγ_r) − 1\| | 11 | 1.0e-3 | 2 % |
| power balance P_flux/P_work − 1 | 11 | −2.9e-3 | 2 % |
| frame reflection \|C⁻/C⁺\| | 11 | 9.3e-4 | 2 % |
| isotropy max_θ \|C⁺(θ)/⟨C⁺⟩ − 1\| | 11 | 1.67 % (ω = 9.69) | 2 % |
| k_h/k₀ − 1 | 11 | −3.9e-4 | 1e-2 |
| max σ_abs k₀ over (ω, ζ) | 1 | 0.9961 | ≤ 1.02 |
| optical theorem \|σ_ext − σ_sc − σ_abs\|/σ_ext (ζ = 0, ζ_crit, 3ζ_crit) | 33 | 3.2e-3 | 2 % |
| k₀σ_abs FE − theory (ζ_crit, 3ζ_crit) | 22 | −5.6e-3 | 0.02 |
| σ_sc/σ_sc,theory − 1 (ζ = 0) | 11 | −1.37 % | 2 % |
| ω̃_h/ω̃ − 1 (ω̃_h = 9.95218, theory 9.95063) | 1 | 1.6e-4 | 1e-3 |
| ζ_opt,h/ζ_crit − 1 (0.012934 vs 0.012894) | 1 | 3.2e-3 | 2 % |
| bound attained at (ω̃_h, ζ_opt,h): k₀σ_abs − 1 | 1 | −3.4e-3 | 2 % |
| P_abs/P_sc − 1 at critical coupling | 1 | 2.6e-3 | 2 % |
| B: \|ΔG_h\|/\|G_h\|, \|ΔIm G_h\|/γ_r | 2 | 2.0e-3, 2.1e-4 | 5e-3 |
| M: \|ΔG_h\|/\|G_h\|, \|ΔIm G_h\|/γ_r | 2 | 4.6e-3, 1.8e-3 | 2 % |
| peak RSS (Julia) < 20 GiB | 13 | 10.5 GB | 20 |

Notes. (i) The isotropy error rises below ω̃ (0.55 % at 10.08, 1.67 % at 9.69, 0.95 % at 9.39) at both meshes
(λ_p/5: 1.38 % at 9.69), so it is not the coarsening; it is the closest margin (×1.2). (ii) Re G_h(0) is the
near-field quantity and is the least converged (−2.7 % at λ_p/5, −3.1 % at λ_p/4.5 at ω̃). (iii) λ_p/5 vs
λ_p/4.5 at ω ∈ {9.69, 9.82, 9.95, 10.08}: |ΔG_h(0)|/|G_h(0)| ≤ 2.3e-3, |ΔIm G_h|/γ_r ≤ 8.3e-4,
|ΔC⁺|/|C⁺| ≤ 5.2e-4.

Compute (rss_guard 20 GiB, run_limited 18 GB, heavy-run lock, `-O1`, one case per process): at λ_p/4.5
every P case took 164–172 s wall (factorisation 97–104 s, lunz 4.07–4.20e8), B and M 130 / 126 s; guard peak
footprints 14.08 (1), 15.61 (2), 15.84 (3), 15.24 (4), 14.19 (5), 12.68 (B), 12.29 (M), 15.34 (8), 15.63 (9),
15.27 (10), 15.30 (11), 14.73 (12), 15.52 (13) GiB. Every guarded run ended with a shell segfault (exit 139)
of the guard process after it logged exit 0; all records were saved.

Post-run fix in the post-processing (no effect on the solves): `renormalised_h` took `imag` of the
interpolant of Im G_h (a real function), which gave ζ_opt,h = NaN; fixed before the final checks.

### Reproduce

```
julia --project -O1 scripts/L3-3D-flat-bottom/L3-3_single_resonator.jl --list
julia --project -O1 scripts/L3-3D-flat-bottom/L3-3_single_resonator.jl --case <i>   # i = 1…15, under rss_guard
julia --project scripts/L3-3D-flat-bottom/L3-3_single_resonator.jl                 # tables, checks, figure
julia scripts/export_paper_assets.jl L3-3
```
