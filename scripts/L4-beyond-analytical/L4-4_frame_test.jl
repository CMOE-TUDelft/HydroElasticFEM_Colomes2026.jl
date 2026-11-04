# L4-4 frame-reflection test (narrow layout of case 1, h = λ_p/3): julia --project=envs/resonator_field -O1 scripts/L4-beyond-analytical/L4-4_frame_test.jl <frame λ> <strength> <power>, under rss_guard. Results: data/exp_pro/L4-4/frame_tests.csv.
include(joinpath(@__DIR__, "L4-4_aniso_island.jl"))

fr, s, pw = parse(Float64, ARGS[1]), parse(Float64, ARGS[2]), parse(Int, ARGS[3])
p = L4_4_Params(cases()[7]; frame=fr, sponge_strength=s, sponge_power=pw)
l = layout(p); println("layout ", l.nx, "×", l.ny, " N=", size_counts(p).N)
d = run_case(p)
refl = axis_reflection(d, p)
@printf("FRAME %g s %g pow %d: residual %.2e axis reflection %.4f maxrss %.2f\n", fr, s, pw, d["rel_residual"], refl, d["maxrss_final_gb"])
wsave(joinpath(REPO, "data", "sims", "L4-4", "frame_tests", "frame_$(fr)_$(s)_$(pw).jld2"), Dict("line_y"=>d["line_y"], "line_T"=>d["line_T"], "axis_x"=>d["axis_x"], "axis_T"=>d["axis_T"], "lines"=>d["lines"], "refl"=>refl))  #line_y=d["line_y"], line_T=d["line_T"], axis_x=d["axis_x"], axis_T=d["axis_T"], lines=d["lines"], refl=refl)
