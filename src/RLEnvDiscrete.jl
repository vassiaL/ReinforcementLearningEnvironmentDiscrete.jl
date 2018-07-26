__precompile__()
module RLEnvDiscrete
using Parameters, Reexport
import StatsBase: sample
import PlotlyJS
@reexport using ReinforcementLearning
import ReinforcementLearning: interact!, getstate, reset!,
getprobvecdeterministic, callback!

include("cliffwalking.jl")
include("maze.jl")
include("pomdps.jl")

end # module
