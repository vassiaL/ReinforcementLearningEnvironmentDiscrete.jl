__precompile__()
module ReinforcementLearningEnvironmentDiscrete
using Parameters, Reexport, Compat, Compat.SparseArrays
import StatsBase: sample
import GR: imshow
@reexport using ReinforcementLearning
import ReinforcementLearning: interact!, getstate, reset!,
getprobvecdeterministic, callback!, plotenv

include("cliffwalking.jl")
include("maze.jl")
# include("pomdps.jl") removed until ready for julia 0.7

end # module
