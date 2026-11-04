# L4-4 post-freeze mesh diagnostic (2026-11-02, after the binding re-run; NOT a binding check).
# The binding M4/M5 pair has h-dependent y_end (6.46 vs 6.72 m), which moves f_PE by 12 %; here the
# narrow layout has a FIXED y_end, so FE(λ_p/4) vs FE(λ_p/5) and FE vs PE are on the same truncation.
# Variants: 1 A8n4, 2 A8n5 (case 1, y_end = 7 m); 3 A12n4, 4 A12n5 (case 3, y_end = 6 m); nz = 7.
#   julia --project=envs/resonator_field -O1 scripts/L4-beyond-analytical/L4-4_mesh_diag.jl --size | --case j   (rss_guard)
#   julia --project=envs/resonator_field scripts/L4-beyond-analytical/L4-4_mesh_diag.jl                        (post)
# Records: data/sims/L4-4/mesh_diag/; result: data/exp_pro/L4-4/mesh_diag.csv.
include(joinpath(@__DIR__, "L4-4_aniso_island.jl"))

const DNAMES = ["A8n4", "A8n5", "A12n4", "A12n5"]
function diag_cases()
    c = cases()
    return [L4_4_Params(c[1]; y_end=7.0), L4_4_Params(c[1]; y_end=7.0, epw=5.0),
            L4_4_Params(c[3]; y_end=6.0), L4_4_Params(c[3]; y_end=6.0, epw=5.0)]
end

let ds = diag_cases()
    if "--size" in ARGS
        for (i, p) in enumerate(ds)
            l = layout(p); f = pattern_fill(l.nx, l.ny, p.nz; order=p.order)
            @printf("%-6s %d×%d×%d y_end %.2f N %d lnz %.3e projected %.1f GiB\n", DNAMES[i], l.nx, l.ny, p.nz, l.yend,
                    size_counts(p).N, f.chol_lnz_metis, projected_peak_l44(f.chol_lnz_metis))
        end
    elseif (j = findfirst(==("--case"), ARGS)) !== nothing
        k = parse(Int, ARGS[j + 1]); d = simulate(ds[k]; sub="mesh_diag")
        @printf("%s done: N = %d, residual %.2e, wall %.0f s\n", DNAMES[k], d["N"], d["rel_residual"], d["wall_time_s"])
    else
        rows = NamedTuple[]
        for (i, p) in enumerate(ds)
            d = simulate(p; sub="mesh_diag"); r = references(p); fe, _ = fe_lens(d, p)
            push!(rows, (name=DNAMES[i], ω=p.ω, epw=p.epw, y_end=d["yend"], N=d["N"], f_fe=fe.f, f_pe=r.tr.f,
                         fe_over_pe=fe.f / r.tr.f, f_fe_spread=fe.f_spread, f_pe_spread=r.tr.f_spread, A_fe=fe.A, A_pe=r.tr.A,
                         frame_reflection=axis_reflection(d, p), residual=d["rel_residual"]))
        end
        df = DataFrame(rows); show(stdout, MIME"text/plain"(), df); println()
        CSV.write(joinpath(REPO, "data", "exp_pro", "L4-4", "mesh_diag.csv"), df)
    end
end
