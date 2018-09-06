module ReinforcementLearningEnvironmentDiscrete
using SparseArrays, POMDPModels, Random, ReinforcementLearningBase, LinearAlgebra
import StatsBase: sample
import GR: imshow
import ReinforcementLearningBase: interact!, getstate, reset!, plotenv,
actionspace
export interact!, getstate, reset!, plotenv, MDP, POMDPEnvironment,
MDPEnvironment, DiscreteMaze, treeMDP, DetMDP, DetTreeMDP, DetTreeMDPwithinrew,
StochMDP, StochTreeMDP, AbsorbingDetMDP, CliffWalkingMDP

include("mdp.jl")
include("randommdp.jl")
include("cliffwalking.jl")
include("maze.jl")
include("pomdps.jl") 

end # module
