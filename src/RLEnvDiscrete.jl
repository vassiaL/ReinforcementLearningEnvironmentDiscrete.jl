__precompile__()
module RLEnvDiscrete
using Parameters, Reexport
import StatsBase: sample
import GR: imshow
@reexport using ReinforcementLearning
import ReinforcementLearning: interact!, getstate, reset!,
getprobvecdeterministic, callback!, plotenv

include("cliffwalking.jl")
include("maze.jl")
include("pomdps.jl")

end # module
