__precompile__()
module RLEnvDiscrete
using Parameters, Reexport
@reexport using ReinforcementLearning
import ReinforcementLearning: interact!, getstate, reset!

include("cliffwalking.jl")
include("maze.jl")
include("pomdps.jl")

end # module
