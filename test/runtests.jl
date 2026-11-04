using Test

include("test_references.jl")   # shared analytical references
include("test_solvers.jl")      # MetisLUSolver vs LUSolver (2D, 3D), multi-RHS
include("test_L1_1.jl")         # L1-1 bare host
include("test_L1_2.jl")         # L1-2 flux normalisation
include("test_L1_3.jl")         # L1-3 single line resonator
include("test_L1_5.jl")         # L1-5, L1-6, L1-8 periodic supercells
