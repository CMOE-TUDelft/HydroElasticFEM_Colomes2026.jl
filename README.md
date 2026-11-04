# HydroElasticFEM_xxx2026.jl

Numerical verification of the locally resonant hydroelastic (LRH)
formulation of the paper J2025-LRMM, built on
[HydroElasticFEM.jl](https://github.com/CMOE-TUDelft/HydroElasticFEM.jl).
The tests and their acceptance criteria are specified in
[`docs/verification_test_matrix.md`](docs/verification_test_matrix.md).

## Structure

Every test is self-contained and can be read, run and reproduced on its own.

| Location | Content |
| --- | --- |
| `src/Lx_y_<name>.jl` | One module per test: relation to the paper, the problem, one parameter struct (fixed and variable parameters), the FE problem written directly with the HydroElasticFEM.jl API, and the observables of one case (`run_case`). |
| `scripts/<level>/Lx-y_<name>.jl` | One script per test: the cases of the study, DrWatson records, post-processing, acceptance checks, figures and tables. |
| `src/references.jl` | The only shared code: analytical references of the paper (Sections 3–4), no FE. |
| `test/` | Tests of the references, of each test's helpers, and a smoke run of each script. |

Each test module defines

- `Lx_y_Params` — every parameter of a case, grouped by comments into
  *variable* (swept by the study) and *fixed* (host, forcing, layout,
  discretisation, post-processing);
- `VARIABLE` — the names of the variable fields; they name the simulation
  record, and a hash of the fixed fields (`fixed_hash`) is appended, so a
  changed fixed value never reuses an old record;
- `run_case(p) -> Dict` — the raw observables and references of one case.

## Tests

| Test | Paper | Status |
| --- | --- | --- |
| [L1-1](src/L1_1_bare_host_dispersion.jl) bare host: dispersion and Green function (configuration A, vertical section) | `eq:host_symbol`, `eq:lat_green_1d`, `eq:point_power`, `eq:energy_flux`; FE `eq:discreteform_freq` | step 1 (one frequency, mesh convergence) |

## Reproduce a test

```bash
julia --project -O1 scripts/L1-configA-vertical/L1-1_bare_host_dispersion.jl           # full study
julia --project -O1 scripts/L1-configA-vertical/L1-1_bare_host_dispersion.jl --smoke   # one coarse case
julia --project -O1 scripts/L1-configA-vertical/L1-1_bare_host_dispersion.jl --force   # recompute
```

Outputs, relative to the repository root:

| Location | Content | Committed |
| --- | --- | --- |
| `data/sims/<test-id>/` | one DrWatson record per case: all parameters, git commit and patch, HydroElasticFEM pin (URL, revision, tree hash), Julia version, wall time | no |
| `data/exp_pro/<test-id>/` | processed results and `checks.csv` (pass/fail per acceptance criterion) | yes |
| `plots/<test-id>/` | figures (`.pdf`, `.png`) and tables (`.tex`) | yes |

HydroElasticFEM.jl is tracked from its `main` branch; the Manifest pin is
stored in every record. Use `-O1`: compilation dominates the run time at the
default optimisation level.

## Tests of the code

```bash
julia --project -O1 -e 'using Pkg; Pkg.test(julia_args=["-O1"])'
```
