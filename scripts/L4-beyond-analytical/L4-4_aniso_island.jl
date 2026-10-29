#= L4-4 smoke/identity driver (small mesh; not the binding run). Run in envs/resonator_field:
   julia --project=envs/resonator_field -O1 scripts/L4-beyond-analytical/L4-4_aniso_island.jl
Binding cases, post-processing and checks.csv: still to write (usage stop 2026-10-29). =#
using HydroElasticFEM_xxx2026
using HydroElasticFEM_xxx2026.References
include(joinpath(@__DIR__, "..", "..", "src", "L4_4_aniso_island.jl"))
using .L4_4, Printf
base = L4_4_Params(Lx=2.0, Ly=1.5, epw=2.0, frame=1.0, gap=0.3, nz=4, line_dx=(0.3, 0.6, 0.9), n_map=(11, 5), n_axis=21)
println(size_counts(base))
r = rhs_identity(base); @printf("rhs identity %.3e  N=%d\n", r["rhs_identity"], r["N"])
h = run_case(base); f = run_case(L4_4_Params(base; full=true))
@printf("residual half %.2e full %.2e ordering %s\n", h["rel_residual"], f["rel_residual"], h["ordering"])
mh = h["mirror_T"][1]; mf1, mf2 = f["mirror_T"]
@printf("half vs full %.3e, full mirror %.3e, |eta_s| %.3e\n", maximum(abs.(mh .- mf1)) / maximum(abs.(mf1)), maximum(abs.(mf1 .- mf2)) / maximum(abs.(mf1)), maximum(abs.(mf1)))
println("T line1 y=0: ", h["line_T"][1][1], "  maxrss ", h["maxrss_final_gb"])
