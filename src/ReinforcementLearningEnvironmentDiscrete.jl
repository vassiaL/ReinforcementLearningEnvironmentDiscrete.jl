module ReinforcementLearningEnvironmentDiscrete
using ReinforcementLearningBase, SparseArrays, POMDPs, POMDPModels, Random, LinearAlgebra
import StatsBase: sample, wsample
import GR: imshow
import ReinforcementLearningBase: interact!, getstate, reset!, plotenv, actionspace

const ENV_RNG = MersenneTwister(0)

include("mdp.jl")
include("randommdp.jl")
include("cliffwalking.jl")
include("maze.jl")
include("pomdps.jl")

export MDP, ChangeMDP, POMDPEnv, MDPEnv, DiscreteMaze, treeMDP, DetMDP, DetTreeMDP,
DetTreeMDPwithinrew, StochMDP, StochTreeMDP, AbsorbingDetMDP, CliffWalking

end # module
