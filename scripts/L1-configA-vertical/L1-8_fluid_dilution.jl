#=
L1-8 — Fluid dilution of the hybridisation.
Pre- and post-processing of the test defined in src/L1_8_fluid_dilution.jl
(physics, relation to the paper and observables are documented there).

Study: P1 plate, fixed loading n₀m_ρ,r = 0.01 m, lossless; ε_s = j/M ∈
{1/4, 1/8, 1/16} (a = 2πε_s/k_c), p = 3 with 32 and 64 elements per λ_c, and
the variants (T_ρ, C_ρ, H, ω_r):
  (0, 0.49, 10, 10), (9.8, 0.49, 10, 10)       reference, pre-tension
  (0, 0.49, 10, 5), (0, 0.49, 10, 2)           resonator frequency (moves k_c)
  (0, 0.049, 10, 10)                            bending stiffness (moves k_c)
  (0, 0.49, 2, 10), (0, 0.49, 100, 10)          depth (k_cH = 5.5, 274: no dependence predicted)

Acceptance (test matrix §L1.8):
  - type D, finest mesh, every variant and ε_s: splitting and both band
    frequencies against the lattice sums to 1e-4 relative;
  - type A, every variant: the FE discreteness correction
    |split/√μ(k_c) − 1| decreases with ε_s and is ≤ 1e-3 at ε_s = 1/16;
    depth: the FE splittings at ε_s = 1/16 for H ∈ {2, 10, 100} agree to 1e-3;
  - type I: Bloch projection leakage ≤ 1e-6.
The splitting is also reported relative to the in-vacuo value √(n₀m_ρ,r/m_ρ) = 1.
(The dry-plate control of the matrix needs a plate-only problem, not available
in HydroElasticFEM; the in-vacuo value is a closed form.)

Outputs:
  data/sims/L1-8/      one DrWatson record per case
  data/exp_pro/L1-8/   errors.csv, checks.csv
  plots/L1-8/          dilution.pdf/png (correction vs ε_s; splitting vs √μ(k_c)), dilution_table.tex

Run:
  julia --project -O1 scripts/L1-configA-vertical/L1-8_fluid_dilution.jl [--smoke] [--force]
=#

using DrWatson
using CairoMakie
using DataFrames
using CSV
using Printf
using TOML
import HydroElasticFEM
using HydroElasticFEM_Colomes2026.L1_8

const TEST_ID = "L1-8"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ─── 1. Cases ───────────────────────────────────────────────────────────────

const VARIANTS = ((0.0, 0.49, 10.0, 10.0), (9.8, 0.49, 10.0, 10.0), (0.0, 0.49, 10.0, 5.0), (0.0, 0.49, 10.0, 2.0),
                  (0.0, 0.049, 10.0, 10.0), (0.0, 0.49, 2.0, 10.0), (0.0, 0.49, 100.0, 10.0))
const STUDY_M = (4, 8, 16)
const STUDY_N = (32, 64)

function cases(; smoke::Bool=false)
    smoke && return [L1_8_Params(M=4, elements_per_wavelength=32)]
    return [L1_8_Params(T_ρ=T, C_ρ=C, H=H, ω_r=ω, M=M, elements_per_wavelength=n)
            for (T, C, H, ω) in VARIANTS for M in STUDY_M for n in STUDY_N]
end

# ─── 2. Simulation records (DrWatson) ───────────────────────────────────────

record_name(p::L1_8_Params) = savename(p; accesses=VARIABLE, sigdigits=4) * "_fixed=" * fixed_hash(p)

function hefem_pin()
    manifest = joinpath(REPO, "Manifest.toml")
    isfile(manifest) || return Dict("hefem_source" => "unknown")
    e = only(TOML.parsefile(manifest)["deps"]["HydroElasticFEM"])
    return Dict("hefem_source" => get(e, "repo-url", get(e, "path", "registry")),
                "hefem_rev" => get(e, "repo-rev", "unknown"),
                "hefem_tree" => get(e, "git-tree-sha1", "unknown"),
                "hefem_version" => string(pkgversion(HydroElasticFEM)))
end

function simulate(p::L1_8_Params; root::AbstractString=REPO, force::Bool=false)
    path = joinpath(root, "data", "sims", TEST_ID)
    data, _ = produce_or_load(p, path; filename=record_name, tag=true, gitpath=REPO,
                              storepatch=true, force=force) do p
        stats = @timed run_case(p)
        d = stats.value
        d["params"] = Dict(string(f) => getfield(p, f) for f in fieldnames(L1_8_Params))
        merge!(d, hefem_pin())
        d["julia_version"] = string(VERSION)
        d["wall_time_s"] = stats.time
        d["allocated_bytes"] = stats.bytes
        return d
    end
    stored = data["params"]
    all(f -> stored[string(f)] == getfield(p, f), fieldnames(L1_8_Params)) ||
        error("record $(record_name(p)) holds other parameters than the case")
    return data
end

# ─── 3. Post-processing ─────────────────────────────────────────────────────

rel(h, r) = abs(h / r - 1)
variant_tag(r) = @sprintf("T=%g, C=%g, H=%g, ωr=%g", r.T_ρ, r.C_ρ, r.H, r.ω_r)

function errors(p::L1_8_Params, d)
    return (T_ρ=p.T_ρ, C_ρ=p.C_ρ, H=p.H, ω_r=p.ω_r, M=p.M, eps_s=d["eps_s"], a=d["a"], kc=d["kc"], kcH=d["kcH"],
            elements_per_wavelength=p.elements_per_wavelength, elements_per_cell=d["elements_per_cell"],
            ndofs=d["ndofs"], split_h=d["split_h"], split_ref=d["split_ref"], split_local=d["split_local"],
            sqrt_mu=d["sqrt_mu"], split_vacuo=d["split_vacuo"],
            err_split=rel(d["split_h"], d["split_ref"]),
            err_minus=rel(d["omega_minus_h"], d["omega_minus_ref"]), err_plus=rel(d["omega_plus_h"], d["omega_plus_ref"]),
            disc_h=rel(d["split_h"], d["split_local"]), disc_ref=rel(d["split_ref"], d["split_local"]),
            leak=d["leak"], fe_solves=d["fe_solves"], wall_time_s=d["wall_time_s"])
end

finest(df) = combine(groupby(df, [:T_ρ, :C_ρ, :H, :ω_r, :M]), g -> g[argmax(g.elements_per_wavelength), :])

function checks(df::DataFrame)
    rows = NamedTuple[]
    add!(name, kind, value, tol) = push!(rows, (name=name, kind=kind, value=value, tolerance=tol, passed=value <= tol))
    fn = finest(df)
    for r in eachrow(sort(fn, [:T_ρ, :C_ρ, :H, :ω_r, :M]))
        tag = variant_tag(r) * @sprintf(", ε=1/%d", r.M)
        add!("splitting vs lattice sums, $(tag)", "D", r.err_split, 1e-4)
        add!("ω₋(k_c), ω₊(k_c) vs lattice sums, $(tag)", "D", max(r.err_minus, r.err_plus), 1e-4)
    end
    for g in groupby(sort(fn, :eps_s; rev=true), [:T_ρ, :C_ρ, :H, :ω_r])
        tag = variant_tag(g[1, :])
        dec = all(diff(g.disc_h) .< 0)
        push!(rows, (name="discreteness correction decreases with ε, $(tag)", kind="A", value=Float64(!dec),
                     tolerance=0.0, passed=dec))
        add!("|split/√μ(k_c) − 1| at ε=1/16, $(tag)", "A", g.disc_h[end], 1e-3)
    end
    dep = filter(r -> r.T_ρ == 0 && r.C_ρ == 0.49 && r.ω_r == 10 && r.M == maximum(STUDY_M), fn)
    if nrow(dep) > 1
        add!("depth independence of the splitting (H = $(join(Int.(sort(dep.H)), ", "))), ε=1/16", "A",
             (maximum(dep.split_h) - minimum(dep.split_h)) / minimum(dep.split_h), 1e-3)
    end
    add!("Bloch projection leakage, all cases", "I", maximum(df.leak), 1e-6)
    return DataFrame(rows)
end

# ─── 4. Figures and tables ──────────────────────────────────────────────────

"(a) Discreteness correction against ε_s, FE (markers) and lattice sums (lines); (b) FE splitting at ε_s = 1/16 against √μ(k_c)."
function plot_dilution(df::DataFrame, dir::AbstractString)
    nrow(df) == 0 && return
    fn = finest(df)
    theme = Theme(fontsize=12, Axis=(xgridvisible=false, ygridvisible=false))
    palette = Makie.wong_colors()
    mkpath(dir)
    with_theme(theme) do
        fig = Figure(size=(1000, 380))
        axa = Axis(fig[1, 1]; xscale=log10, yscale=log10, xlabel=L"\varepsilon_s = k_c a/2\pi",
                   ylabel=L"|(\omega_+ - \omega_-)/(\omega_r\sqrt{\mu(k_c)}) - 1|", title="(a) discreteness correction",
                   xticks=([1 / 16, 1 / 8, 1 / 4], ["1/16", "1/8", "1/4"]))
        axb = Axis(fig[1, 2]; xlabel=L"\sqrt{\mu(k_c)}", ylabel=L"(\omega_+ - \omega_-)/\omega_r\ \mathrm{(FE)}",
                   title="(b) splitting at ε_s = 1/16")
        groups = collect(groupby(sort(fn, :eps_s), [:T_ρ, :C_ρ, :H, :ω_r]))
        for (i, g) in enumerate(groups)
            col = palette[mod1(i, length(palette))]
            lab = variant_tag(g[1, :])
            lines!(axa, g.eps_s, g.disc_ref; color=col)
            scatter!(axa, g.eps_s, max.(g.disc_h, 1e-12); color=col, label=lab)
            f = g[argmin(g.eps_s), :]
            scatter!(axb, [f.sqrt_mu], [f.split_h]; color=col)
        end
        s = range(0, 0.22; length=2)
        lines!(axb, s, s; color=:gray40, linestyle=:dash)
        Legend(fig[1, 3], axa; framevisible=false, labelsize=9)
        save(joinpath(dir, "dilution.pdf"), fig)
        save(joinpath(dir, "dilution.png"), fig; px_per_unit=300 / 96)
    end
end

function write_table(df::DataFrame, dir::AbstractString)
    mkpath(dir)
    fn = filter(r -> r.M == maximum(STUDY_M), finest(df))
    open(joinpath(dir, "dilution_table.tex"), "w") do io
        println(io, "\\begin{table}[htbp]\n\\centering")
        println(io, "\\caption{L1-8: splitting of the bands at the crossing wavenumber, \$\\varepsilon_s = 1/16\$, \$n_0m_{\\rho,r} = 0.01\$ m: finite element, lattice sums and the local value \$\\sqrt{\\mu(k_c)}\$; the in-vacuo value is \$1\$.}")
        println(io, "\\label{tab:L1-8}\n\\begin{tabular}{rrrrcccc}\n\\toprule")
        println(io, "\$T_\\rho\$ & \$C_\\rho\$ & \$H\$ & \$\\omega_r\$ & \$k_cH\$ & \$\\sqrt{\\mu(k_c)}\$ & FE & err. vs lattice \\\\\n\\midrule")
        for r in eachrow(sort(fn, [:T_ρ, :C_ρ, :H, :ω_r]))
            println(io, join([r.T_ρ, r.C_ρ, r.H, r.ω_r, @sprintf("%.1f", r.kcH), @sprintf("%.5f", r.sqrt_mu),
                              @sprintf("%.5f", r.split_h), @sprintf("%.1e", r.err_split)], " & "), " \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}\n\\end{table}")
    end
end

# ─── 5. Driver ──────────────────────────────────────────────────────────────

function main(; smoke::Bool=false, force::Bool=false, root::AbstractString=REPO)
    cs = cases(; smoke=smoke)
    records = [simulate(p; root=root, force=force) for p in cs]
    df = DataFrame([errors(p, d) for (p, d) in zip(cs, records)])
    ck = checks(df)
    out = joinpath(root, "data", "exp_pro", TEST_ID)
    mkpath(out)
    CSV.write(joinpath(out, "errors.csv"), df)
    CSV.write(joinpath(out, "checks.csv"), ck)
    figdir = joinpath(root, "plots", TEST_ID)
    plot_dilution(df, figdir)
    write_table(df, figdir)

    show(stdout, MIME"text/plain"(), select(finest(df), :T_ρ, :C_ρ, :H, :ω_r, :M, :a, :split_h, :sqrt_mu,
                                            :err_split, :disc_h, :disc_ref); allrows=true)
    println()
    @printf("%d/%d checks pass\n", count(ck.passed), nrow(ck))
    for c in eachrow(filter(r -> !r.passed, ck))
        @printf("FAIL  %-80s value = %.3g (tolerance %.3g)\n", c.name, c.value, c.tolerance)
    end
    return (errors=df, checks=ck)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; smoke="--smoke" in ARGS, force="--force" in ARGS)
end
