# Level 3 (3D): implementation report

Phase 4 of the numerical programme (3D fluid with a plate): L3-1, L3-2, L3-3,
L4-4 and L4-8. This report starts with the sizing card L3-0 (state of
2026-10-28), which sets how each 3D test can be run. Sections for L3-1 … L4-8
will be added as they land.

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
