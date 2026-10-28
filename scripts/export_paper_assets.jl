#=
export_paper_assets.jl — stage the paper figures of the verification suite.

Maps test id → source figure under plots/<id>/ (or a table under
data/exp_pro/<id>/) → paper file name in paper/figures/ (the staging folder;
PNGs there are git-ignored). The Overleaf project is read-only for agents: this
script never writes there. It copies sources into paper/figures/ and prints the
`cp` commands the PI runs to copy paper/figures/<name> into the Overleaf
figures folder.

The mapping below was recovered on 2026-10-26 by scanning
paper/numerical_results.tex for \includegraphics{figures/...} and matching each
staged file in paper/figures/ byte for byte (SHA-1) against the candidates in
plots/. Every one of the 24 included figures has exactly one source. Since the
paper is being restructured (some figures move to appendix_verification.tex),
each run re-scans every staged paper/*.tex file and reports any included figure
that has no entry here (add it to ASSETS) and any entry that no staged file
includes any more.

Usage (no packages are loaded; Base Julia only):
  julia --project=. scripts/export_paper_assets.jl [ids...] [--dry-run] [--all]
    ids        test ids (L2-6) or level prefixes (L2); default: every test
    --dry-run  print what would be copied and the PI commands, change nothing
    --all      print PI commands also for files whose Overleaf copy is identical
=#

const REPO = normpath(joinpath(@__DIR__, ".."))
const STAGE = joinpath(REPO, "paper", "figures")
const PAPER = joinpath(REPO, "paper")
const OVERLEAF = "/Users/ocolomesgene/Library/CloudStorage/Dropbox/Aplicaciones/Overleaf/J2025-LRMM"
const OVERLEAF_FIGS = joinpath(OVERLEAF, "figures")

# (test id, source path relative to the repo, paper name in figures/)
const ASSETS = [
    ("L1-1", "plots/L1-1/convergence_single_frequency_p2.png", "L1_1_convergence_p2.png"),
    ("L1-1", "plots/L1-1/convergence_single_frequency_p3.png", "L1_1_convergence_p3.png"),
    ("L1-1", "plots/L1-1/sweep_err_k.png", "L1_1_sweep_k.png"),
    ("L1-1", "plots/L1-1/sweep_err_gamma.png", "L1_1_sweep_gamma.png"),
    ("L1-2", "plots/L1-2/convergence_p2.png", "L1_2_convergence_p2.png"),
    ("L1-2", "plots/L1-2/convergence_p3.png", "L1_2_convergence_p3.png"),
    ("L1-3", "plots/L1-3/convergence_p2.png", "L1_3_convergence_p2.png"),
    ("L1-3", "plots/L1-3/convergence_p3.png", "L1_3_convergence_p3.png"),
    ("L1-3", "plots/L1-3/response_curves.png", "L1_3_response_curves.png"),
    ("L1-3b", "plots/L1-3b/zeta_opt.png", "L1_3b_zeta_opt.png"),
    ("L1-4", "plots/L1-4/transmission.png", "L1_4_transmission.png"),
    ("L1-4", "plots/L1-4/decay.png", "L1_4_decay.png"),
    ("L1-5", "plots/L1-5/bands.png", "L1_5_bands.png"),
    ("L1-6", "plots/L1-6/gap.png", "L1_6_gap.png"),
    ("L1-7", "plots/L1-7/bands_overlay.png", "L1_7_bands_overlay.png"),
    ("L1-8", "plots/L1-8/dilution.png", "L1_8_dilution.png"),
    ("L2-1", "plots/L2-1/diagonal.png", "L2_1_diagonal.png"),
    ("L2-2", "plots/L2-2/woodbury.png", "L2_2_woodbury.png"),
    ("L2-3", "plots/L2-3/resonance.png", "L2_3_resonance.png"),
    ("L2-4", "plots/L2-4/bound.png", "L2_4_bound.png"),
    ("L2-4", "plots/L2-4/absorption.png", "L2_4_absorption.png"),
    ("L2-5", "plots/L2-5/splitting.png", "L2_5_splitting.png"),
    ("L2-6", "plots/L2-6/profiles.png", "L2_6_profiles.png"),
    ("L2-6", "plots/L2-6/footprint.png", "L2_6_footprint.png"),
    ("L2-6", "plots/L2-6/density.png", "L2_6_density.png"),
    ("L3-2", "plots/L3-2/point_resonator.png", "L3_2_point_resonator.png"),
    # Tables: the paper tables are typeset inline in numerical_results.tex, so
    # no file of plots/<id>/*.tex or data/exp_pro/<id>/ is included at present.
    # Add ("<id>", "data/exp_pro/<id>/<file>", "<paper name>") here when one is.
]

"Figure name => staged .tex file including it (\\includegraphics[...]{figures/<name>}), over paper/*.tex."
function included_figures(dir::AbstractString=PAPER)
    found = Dict{String,String}()
    isdir(dir) || return found
    for f in sort(filter(endswith(".tex"), readdir(dir)))
        src = join(filter(l -> !startswith(lstrip(l), "%"), readlines(joinpath(dir, f))), "\n")
        for m in eachmatch(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", src)
            get!(found, m.captures[1], f)
        end
    end
    return found
end

selected(id, wanted) = isempty(wanted) || any(w -> id == w || startswith(id, w * "-"), wanted)

same_file(a, b) = isfile(a) && isfile(b) && filesize(a) == filesize(b) && read(a) == read(b)

function main(args)
    dry = "--dry-run" in args
    all_cmds = "--all" in args
    wanted = filter(a -> !startswith(a, "--"), args)

    # Consistency of the mapping with the paper.
    inc = included_figures()
    mapped = Set(last.(ASSETS))
    missing_map = sort(filter(n -> !(n in mapped), collect(keys(inc))))
    unused = sort(filter(n -> !haskey(inc, n), collect(mapped)))
    isempty(missing_map) || println("WARNING: included by paper/*.tex but not in ASSETS: ",
                                    join(["$(n) ($(inc[n]))" for n in missing_map], ", "))
    isempty(unused) || println("NOTE: in ASSETS but not included by any paper/*.tex: ", join(unused, ", "))

    rows = filter(r -> selected(r[1], wanted), ASSETS)
    isempty(rows) && (println("no asset matches ", join(wanted, " ")); return 1)

    dry || mkpath(STAGE)
    cmds = String[]
    nfail = 0
    for (id, rel, name) in rows
        src = joinpath(REPO, rel)
        dst = joinpath(STAGE, name)
        # Guard: this script writes only into paper/figures/.
        startswith(normpath(dst), normpath(STAGE)) || error("refusing to write outside $(STAGE)")
        if !isfile(src)
            println("MISSING  ", rpad(id, 6), rel, " (run the test script first)")
            nfail += 1
            continue
        end
        status = same_file(src, dst) ? "staged" : (isfile(dst) ? "update" : "new")
        if status != "staged"
            dry || cp(src, dst; force=true)
        end
        on_overleaf = same_file(dry && status != "staged" ? src : dst, joinpath(OVERLEAF_FIGS, name))
        println(rpad(dry && status != "staged" ? "would-" * status : status, 13), rpad(id, 6),
                rpad(rel, 52), "→ paper/figures/", rpad(name, 26), " [", get(inc, name, "not included"), "]",
                on_overleaf ? "  (Overleaf identical)" : "")
        (all_cmds || !on_overleaf) && push!(cmds, "cp $(dst) $(OVERLEAF_FIGS)/")
    end

    println()
    if isempty(cmds)
        println("Overleaf figures are up to date for the selection; nothing for the PI to copy.")
    else
        println("# Commands for the PI (the Overleaf folder is read-only for agents):")
        foreach(println, cmds)
    end
    return nfail == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
