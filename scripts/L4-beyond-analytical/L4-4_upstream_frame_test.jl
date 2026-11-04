# L4-4 upstream-frame smoke test (re-freeze 2026-10-31, before the freeze): case M4 (narrow layout of
# case 1, h = λ_p/4, frame 2λ_p downstream and at the side) with the upstream frame/gap
# (frame_up, gap_up) given in λ_p; compares with the 2λ_p / 0.5λ_p upstream reference when both exist.
#   julia --project=envs/resonator_field -O1 scripts/L4-beyond-analytical/L4-4_upstream_frame_test.jl <frame_up> <gap_up>
#   julia --project=envs/resonator_field scripts/L4-beyond-analytical/L4-4_upstream_frame_test.jl --compare
# Under rss_guard. Records: data/sims/L4-4/upstream_frame/; result: data/exp_pro/L4-4/upstream_frame_test.csv.
include(joinpath(@__DIR__, "L4-4_aniso_island.jl"))

const UDIR = joinpath(REPO, "data", "sims", "L4-4", "upstream_frame")
ufile(fu, gu) = joinpath(UDIR, "up_$(fu)_$(gu).jld2")

if "--compare" in ARGS
    p0 = cases()[7]
    ref = wload(ufile(2.0, 0.5)); thin = wload(ufile(0.5, 0.25))
    pr = L4_4_Params(p0; frame_up=2.0, gap_up=0.5); pt = L4_4_Params(p0; frame_up=0.5, gap_up=0.25)
    lr, _ = fe_lens(ref, pr); lt, _ = fe_lens(thin, pt)
    dT = maximum(maximum(abs.(a .- b)) for (a, b) in zip(ref["line_T"], thin["line_T"]))
    sT = maximum(maximum(abs.(a .- 1)) for a in ref["line_T"])
    out = DataFrame([(case="M4", frame_up_ref=2.0, gap_up_ref=0.5, frame_up=0.5, gap_up=0.25, N_ref=ref["N"], N=thin["N"],
                      f_ref=lr.f, f=lt.f, df_rel=lt.f / lr.f - 1, A_ref=lr.A, A=lt.A, dA=lt.A - lr.A,
                      max_dT_over_max_T_minus_1=dT / sT, refl_ref=axis_reflection(ref, pr), refl=axis_reflection(thin, pt))])
    show(stdout, MIME"text/plain"(), out); println()
    CSV.write(joinpath(REPO, "data", "exp_pro", "L4-4", "upstream_frame_test.csv"), out)
else
    fu, gu = parse(Float64, ARGS[1]), parse(Float64, ARGS[2])
    p = L4_4_Params(cases()[7]; frame_up=fu, gap_up=gu)
    l = layout(p); println("layout ", l.nx, "×", l.ny, " x0 = ", l.x0, " N = ", size_counts(p).N)
    d = run_case(p)
    @printf("UPSTREAM %g/%g: residual %.2e axis reflection %.4f maxrss %.2f\n", fu, gu, d["rel_residual"], axis_reflection(d, p),
            d["maxrss_final_gb"])
    mkpath(UDIR)
    wsave(ufile(fu, gu), Dict(k => d[k] for k in ("N", "k", "yend", "lines", "line_y", "line_T", "axis_x", "axis_T", "rel_residual",
                                                     "maxrss_final_gb")))
end
